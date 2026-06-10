//! Der Agent-Task: die agentic Loop.
//!
//! Pro Prompt läuft die klassische Schleife — Modell aufrufen → Tool-Calls
//! ausführen → Ergebnisse zurück ans Modell → wiederholen, bis das Modell ohne
//! Tool-Aufruf antwortet. Läuft als eigener tokio-Task und kommuniziert nur über
//! die Kanäle aus [`crate::event`], blockiert die UI also nie.
//!
//! Abbrechen: Während eines Turns hängt der Task in einem `await` (Streaming,
//! Tool-Ausführung oder beim Warten auf eine `ask_user`-Antwort) und pollt den
//! Command-Kanal nicht. Deshalb gibt es einen separaten `cancel`-Kanal, der an
//! jedem Wartepunkt per [`tokio::select!`] beobachtet wird.

use serde_json::Value;
use tokio::sync::mpsc::{UnboundedReceiver, UnboundedSender};

use crate::auth::{self, AuthInfo};
use crate::event::{AgentCommand, AgentEvent};
use crate::llm::{AssistantTurn, ChatMessage, LlmClient, ProviderKind, ToolCall, ToolResult, ToolSpec};
use crate::openai_subscription;
use crate::session::SessionWriter;
use crate::tokens::{self, TokenStats};
use crate::tools;

const SYSTEM_PROMPT: &str = "Du bist anvil, ein Coding-Agent, der in einem Terminal läuft. \
     Du hast Werkzeuge, um Dateien zu lesen, zu schreiben und zu bearbeiten sowie Shell-Befehle \
     auszuführen. Nutze sie eigenständig, um die Aufgabe des Nutzers zu erledigen, statt nur zu \
     beschreiben, was zu tun wäre. Wichtig: Quellcode- und Dateiänderungen dürfen ausschließlich \
     über write_file oder edit_file erfolgen. bash ist nur für read-only Inspektion, Builds und Tests; \
     nutze bash niemals, um Dateien per Python/Sed/Redirect/etc. zu verändern. Fasse dich kurz. \
     Antworte in der Sprache des Nutzers.";

const COMPACT_KEEP_TOKENS: usize = 8_000;
const TOOL_OUTPUT_MAX_CHARS: usize = 2_000;
const SUMMARY_TEMPLATE: &str = r#"Output exactly the Markdown structure shown inside <template> and keep the section order unchanged. Do not include the <template> tags in your response.
<template>
## Goal
- [single-sentence task summary]

## Constraints & Preferences
- [user constraints, preferences, specs, or "(none)"]

## Progress
### Done
- [completed work or "(none)"]

### In Progress
- [current work or "(none)"]

### Blocked
- [blockers or "(none)"]

## Key Decisions
- [decision and why, or "(none)"]

## Next Steps
- [ordered next actions or "(none)"]

## Critical Context
- [important technical facts, errors, open questions, or "(none)"]

## Relevant Files
- [file or directory path: why it matters, or "(none)"]
</template>

Rules:
- Keep every section, even when empty.
- Use terse bullets, not prose paragraphs.
- Preserve exact file paths, commands, error strings, and identifiers when known.
- Do not mention the summary process or that context was compacted."#;

/// Ergebnis eines Stream-Durchlaufs, nachdem er gegen den Abbruch gerennt ist.
enum Step {
    Turn(AssistantTurn),
    Failed(String),
    Cancelled,
}

/// Hauptschleife des Agent-Tasks. `client` ist `None`, wenn beim Start kein
/// API-Key gefunden wurde — dann wird jeder Prompt mit einem Hinweis beantwortet.
pub async fn run(
    client: Option<LlmClient>,
    mut commands: UnboundedReceiver<AgentCommand>,
    mut cancel: UnboundedReceiver<()>,
    events: UnboundedSender<AgentEvent>,
) {
    let tools = tools::specs();
    let mut tool_state = tools::ToolState::default();
    let mut history: Vec<ChatMessage> = Vec::new();
    let mut session: Option<SessionWriter> = None;
    let mut client = client;

    while let Some(command) = commands.recv().await {
        match command {
            AgentCommand::Prompt(prompt) => {
                let Some(client) = client.as_mut() else {
                    let _ = events.send(AgentEvent::Error(
                        "Kein Provider aktiv. Melde dich mit /login <provider> an \
                         (oder setze z. B. OPENAI_API_KEY) — ein Neustart ist nicht nötig."
                            .to_string(),
                    ));
                    continue;
                };

                handle_prompt(
                    client,
                    &tools,
                    &mut history,
                    &mut commands,
                    &mut cancel,
                    &events,
                    &mut session,
                    &mut tool_state,
                    prompt,
                )
                .await;
            }
            // Eine gespeicherte Sitzung fortsetzen.
            AgentCommand::SetContext {
                history: loaded,
                id,
            } => {
                history = loaded;
                tool_state.reset();
                session = match SessionWriter::open(&id) {
                    Ok(writer) => Some(writer),
                    Err(error) => {
                        let _ = events.send(AgentEvent::Error(format!(
                            "Sitzung konnte nicht zum Anhängen geöffnet werden: {error:#}"
                        )));
                        None
                    }
                };
            }
            // Provider/Modell wechseln, ohne den Kontext zu verlieren.
            AgentCommand::SetClient {
                kind,
                model,
                secret,
                auth_info,
            } => {
                client = Some(LlmClient::with_auth(kind, secret.0, Some(model), auth_info));
            }
            AgentCommand::LoginOauth { kind } => {
                if kind != ProviderKind::OpenAi {
                    let _ = events.send(AgentEvent::Error(format!(
                        "OAuth ist für {} nicht verfügbar.",
                        kind.display()
                    )));
                    continue;
                }
                let _ = events.send(AgentEvent::ToolStarted(
                    "OpenAI OAuth · Browser-Login".to_string(),
                ));
                match openai_subscription::login_browser().await {
                    Ok(tokens) => {
                        let auth_info = AuthInfo::Oauth {
                            access: tokens.access.clone(),
                            refresh: tokens.refresh,
                            expires: tokens.expires,
                            account_id: tokens.account_id,
                        };
                        // login_browser speichert bereits; der zweite Schreibvorgang ist nur
                        // defensiv, falls sich die Store-Logik dort später ändert.
                        let _ = auth::set("openai", auth_info.clone());
                        client = Some(LlmClient::with_auth(
                            ProviderKind::OpenAi,
                            tokens.access,
                            None,
                            Some(auth_info),
                        ));
                        let _ = events.send(AgentEvent::ToolFinished {
                            ok: true,
                            summary: "OpenAI Subscription angemeldet.".to_string(),
                        });
                        let _ = events.send(AgentEvent::Done);
                    }
                    Err(error) => {
                        let _ = events.send(AgentEvent::ToolFinished {
                            ok: false,
                            summary: format!("Login fehlgeschlagen: {error:#}"),
                        });
                        let _ = events.send(AgentEvent::Error(format!("OpenAI-OAuth: {error:#}")));
                    }
                }
            }
            AgentCommand::Compact => {
                let Some(client) = client.as_mut() else {
                    let _ = events.send(AgentEvent::Error(
                        "Kein Provider aktiv. Melde dich mit /login <provider> an.".to_string(),
                    ));
                    continue;
                };
                compact_history(client, &mut history, &events, &mut session).await;
            }
            // Frische Sitzung: Kontext und Datei-Handle fallenlassen.
            AgentCommand::Reset => {
                history.clear();
                tool_state.reset();
                session = None;
            }
        }
    }
}

/// Verarbeitet einen Prompt von Anfang bis Ende: streamt Antworten, führt
/// angeforderte Tools aus und füttert deren Ergebnisse zurück, bis das Modell
/// fertig ist. Bei Fehler oder Abbruch wird der gesamte Turn aus der Historie
/// entfernt, damit ein erneuter Versuch auf konsistentem Kontext aufsetzt.
#[allow(clippy::too_many_arguments)]
async fn handle_prompt(
    client: &mut LlmClient,
    tools: &[ToolSpec],
    history: &mut Vec<ChatMessage>,
    commands: &mut UnboundedReceiver<AgentCommand>,
    cancel: &mut UnboundedReceiver<()>,
    events: &UnboundedSender<AgentEvent>,
    session: &mut Option<SessionWriter>,
    tool_state: &mut tools::ToolState,
    prompt: String,
) {
    // Veraltete Abbruch-Signale verwerfen, bevor der Turn beginnt.
    while cancel.try_recv().is_ok() {}

    let restore_to = history.len();
    history.push(ChatMessage::User(prompt));
    emit_token_stats(events, client, history, 0);

    loop {
        // 1) Modell streamen — abbrechbar. Das Stream-Future leiht `history`,
        //    deshalb wird das Ergebnis erst nach dem select! ausgewertet.
        let step = {
            let mut on_text = |delta: String| {
                let _ = events.send(AgentEvent::Chunk(delta));
            };
            tokio::select! {
                result = client.stream(Some(SYSTEM_PROMPT), history, tools, &mut on_text) => match result {
                    Ok(turn) => Step::Turn(turn),
                    Err(error) => Step::Failed(format!("{error:#}")),
                },
                _ = cancel.recv() => Step::Cancelled,
            }
        };

        let turn = match step {
            Step::Turn(turn) => turn,
            Step::Failed(reason) => {
                return abort(history, restore_to, events, AgentEvent::Error(reason));
            }
            Step::Cancelled => return abort(history, restore_to, events, AgentEvent::Cancelled),
        };

        let tool_calls = turn.tool_calls.clone();
        let received = tokens::estimate_text(&turn.text)
            + turn
                .tool_calls
                .iter()
                .map(|call| 8 + tokens::estimate_text(&call.name) + tokens::estimate_text(&call.arguments.to_string()))
                .sum::<usize>();
        history.push(ChatMessage::Assistant {
            text: turn.text,
            tool_calls: turn.tool_calls,
        });

        // Keine Tool-Aufrufe → das war die finale Antwort. Turn persistieren.
        if tool_calls.is_empty() {
            emit_token_stats(events, client, history, received);
            persist_turn(session, &history[restore_to..]);
            let _ = events.send(AgentEvent::Done);
            return;
        }

        // 2) Tools ausführen — jeder Aufruf ist abbrechbar.
        let mut results: Vec<ToolResult> = Vec::with_capacity(tool_calls.len());
        for call in &tool_calls {
            // ask_user läuft nicht lokal, sondern wartet auf eine Eingabe.
            if call.name == tools::ASK_USER {
                tokio::select! {
                    answer = ask_user(call, events, commands) => results.push(answer),
                    _ = cancel.recv() => return abort(history, restore_to, events, AgentEvent::Cancelled),
                }
                continue;
            }

            let _ = events.send(AgentEvent::ToolStarted(describe_call(call)));
            let result = tokio::select! {
                result = tools::execute(call, tool_state) => result,
                _ = cancel.recv() => return abort(history, restore_to, events, AgentEvent::Cancelled),
            };
            let _ = events.send(AgentEvent::ToolFinished {
                ok: !result.is_error,
                summary: summarize(&call.name, &result.content),
            });
            results.push(result);
        }
        history.push(ChatMessage::ToolResults(results));
        emit_token_stats(events, client, history, received);
    }
}

fn emit_token_stats(
    events: &UnboundedSender<AgentEvent>,
    client: &LlmClient,
    history: &[ChatMessage],
    received_since_command: usize,
) {
    let stats = TokenStats {
        sent_since_prompt: tokens::estimate_messages(Some(SYSTEM_PROMPT), history),
        received_since_command,
        context: tokens::estimate_messages(Some(SYSTEM_PROMPT), history),
        context_limit: tokens::context_limit(client.model()),
    };
    let _ = events.send(AgentEvent::TokenStats(stats));
}

async fn compact_history(
    client: &mut LlmClient,
    history: &mut Vec<ChatMessage>,
    events: &UnboundedSender<AgentEvent>,
    session: &mut Option<SessionWriter>,
) {
    let selected = match select_compaction_context(history, COMPACT_KEEP_TOKENS) {
        Some(selected) if !selected.head.is_empty() => selected,
        _ => {
            let _ = events.send(AgentEvent::Error(
                "Nichts zum Verdichten: der Verlauf ist leer oder bereits kompakt.".to_string(),
            ));
            return;
        }
    };

    let previous_summary = history.iter().find_map(|message| match message {
        ChatMessage::User(text) => parse_compaction_summary(text),
        _ => None,
    });
    let prompt = build_compaction_prompt(previous_summary.as_deref(), &[selected.head.clone()]);
    let _ = events.send(AgentEvent::CompactStarted);

    let mut ignored = |_delta: String| {};
    let summary = match client.stream(None, &[ChatMessage::User(prompt)], &[], &mut ignored).await {
        Ok(turn) if !turn.text.trim().is_empty() => turn.text.trim().to_string(),
        Ok(_) => {
            let _ = events.send(AgentEvent::Error("Verdichtung lieferte keine Summary.".to_string()));
            return;
        }
        Err(error) => {
            let _ = events.send(AgentEvent::Error(format!("Verdichtung fehlgeschlagen: {error:#}")));
            return;
        }
    };

    let compacted = ChatMessage::User(compaction_context_message(&summary, &selected.recent));
    history.clear();
    history.push(compacted.clone());
    if let Some(writer) = session {
        let _ = writer.append(&compacted);
    }
    emit_token_stats(events, client, history, tokens::estimate_text(&summary));
    let _ = events.send(AgentEvent::Compacted {
        summary,
        recent: selected.recent,
    });
    let _ = events.send(AgentEvent::Done);
}

struct SelectedCompaction {
    head: String,
    recent: String,
}

fn select_compaction_context(messages: &[ChatMessage], keep_tokens: usize) -> Option<SelectedCompaction> {
    let conversation: Vec<String> = messages
        .iter()
        .filter(|message| !matches!(message, ChatMessage::User(text) if parse_compaction_summary(text).is_some()))
        .map(serialize_for_compaction)
        .filter(|text| !text.trim().is_empty())
        .collect();
    if conversation.is_empty() {
        return None;
    }

    let mut total = 0;
    let mut split = conversation.len();
    let mut split_prefix = String::new();
    let mut split_suffix = String::new();
    for index in (0..conversation.len()).rev() {
        let next = total + tokens::estimate_text(&conversation[index]);
        if next > keep_tokens {
            let remaining = keep_tokens.saturating_sub(total) * 4;
            if remaining > 0 {
                let split_at = floor_char_boundary(&conversation[index], conversation[index].len().saturating_sub(remaining));
                split_prefix = conversation[index][..split_at].to_string();
                split_suffix = conversation[index][split_at..].to_string();
                split = index + 1;
            }
            break;
        }
        total = next;
        split = index;
    }

    Some(SelectedCompaction {
        head: conversation[..split]
            .iter()
            .chain(std::iter::once(&split_prefix))
            .filter(|text| !text.is_empty())
            .cloned()
            .collect::<Vec<_>>()
            .join("\n\n"),
        recent: std::iter::once(&split_suffix)
            .chain(conversation[split..].iter())
            .filter(|text| !text.is_empty())
            .cloned()
            .collect::<Vec<_>>()
            .join("\n\n"),
    })
}

fn build_compaction_prompt(previous_summary: Option<&str>, context: &[String]) -> String {
    let instruction = match previous_summary {
        Some(summary) => format!(
            "Update the anchored summary below using the conversation history above.\n\
             Preserve still-true details, remove stale details, and merge in the new facts.\n\
             <previous-summary>\n{summary}\n</previous-summary>"
        ),
        None => "Create a new anchored summary from the conversation history.".to_string(),
    };
    std::iter::once(instruction)
        .chain(std::iter::once(SUMMARY_TEMPLATE.to_string()))
        .chain(context.iter().filter(|text| !text.is_empty()).cloned())
        .collect::<Vec<_>>()
        .join("\n\n")
}

fn compaction_context_message(summary: &str, recent: &str) -> String {
    format!(
        "[Compacted context]\nSummary:\n{summary}\n\nRecent conversation retained verbatim:\n{}",
        if recent.trim().is_empty() { "(none)" } else { recent.trim() }
    )
}

fn parse_compaction_summary(text: &str) -> Option<String> {
    let rest = text.strip_prefix("[Compacted context]\nSummary:\n")?;
    let (summary, _) = rest.split_once("\n\nRecent conversation retained verbatim:")?;
    Some(summary.trim().to_string())
}

fn serialize_for_compaction(message: &ChatMessage) -> String {
    match message {
        ChatMessage::User(text) => format!("[User]: {text}"),
        ChatMessage::Assistant { text, tool_calls } => {
            let mut parts = Vec::new();
            if !text.trim().is_empty() {
                parts.push(format!("[Assistant]: {text}"));
            }
            for call in tool_calls {
                parts.push(format!("[Assistant tool call]: {}({})", call.name, call.arguments));
            }
            parts.join("\n")
        }
        ChatMessage::ToolResults(results) => results
            .iter()
            .map(|result| {
                let content = if result.content.len() <= TOOL_OUTPUT_MAX_CHARS {
                    result.content.clone()
                } else {
                    let truncate_at = floor_char_boundary(&result.content, TOOL_OUTPUT_MAX_CHARS);
                    format!("{}\n[truncated]", &result.content[..truncate_at])
                };
                if result.is_error {
                    format!("[Tool error]: {content}")
                } else {
                    format!("[Tool result]: {content}")
                }
            })
            .collect::<Vec<_>>()
            .join("\n"),
    }
}

fn floor_char_boundary(text: &str, index: usize) -> usize {
    if index >= text.len() {
        return text.len();
    }
    let mut idx = index;
    while idx > 0 && !text.is_char_boundary(idx) {
        idx -= 1;
    }
    idx
}

/// Speichert die in diesem Turn entstandenen Nachrichten append-only. Legt beim
/// ersten Turn einer Sitzung die Datei an (Titel = erster Prompt). Best-effort:
/// Schreibfehler werden ignoriert, anvil läuft weiter.
fn persist_turn(session: &mut Option<SessionWriter>, turn: &[ChatMessage]) {
    if turn.is_empty() {
        return;
    }
    if session.is_none() {
        let title = turn
            .iter()
            .find_map(|message| match message {
                ChatMessage::User(text) => Some(text.as_str()),
                _ => None,
            })
            .unwrap_or("Sitzung");
        match SessionWriter::create(title) {
            Ok(writer) => *session = Some(writer),
            Err(_) => return, // best effort — ohne Persistenz weiterlaufen
        }
    }
    if let Some(writer) = session {
        for message in turn {
            let _ = writer.append(message);
        }
    }
}

/// Verwirft den aktuellen Turn aus der Historie und meldet das gegebene Event.
fn abort(
    history: &mut Vec<ChatMessage>,
    restore_to: usize,
    events: &UnboundedSender<AgentEvent>,
    event: AgentEvent,
) {
    history.truncate(restore_to);
    let _ = events.send(event);
}

/// Behandelt einen `ask_user`-Aufruf: stellt die Frage an die UI und „borgt" sich
/// die nächste Eingabe des Nutzers als Tool-Ergebnis (passend zur Call-ID).
async fn ask_user(
    call: &ToolCall,
    events: &UnboundedSender<AgentEvent>,
    commands: &mut UnboundedReceiver<AgentCommand>,
) -> ToolResult {
    let _ = events.send(AgentEvent::AskUser(arg(&call.arguments, "question")));

    match commands.recv().await {
        Some(AgentCommand::Prompt(answer)) => ToolResult {
            id: call.id.clone(),
            content: answer,
            is_error: false,
        },
        // Kanal geschlossen → die App wird beendet.
        None => ToolResult {
            id: call.id.clone(),
            content: "(Keine Antwort — Sitzung beendet.)".to_string(),
            is_error: true,
        },
        // Andere Commands kommen während einer Rückfrage normalerweise nicht vor.
        Some(_) => ToolResult {
            id: call.id.clone(),
            content: "(Keine Antwort.)".to_string(),
            is_error: true,
        },
    }
}

/// Kompakte Beschreibung eines Tool-Aufrufs für die UI, z. B. `"bash · ls -la"`.
///
/// `bash` bekommt eine eigene Form: der **vollständige** Befehl (mehrzeilig
/// möglich) steht ab Zeile 2 im Block, die Kopfzeile ist nur `bash`. So kann die
/// UI exakt zeigen, was ausgeführt wurde, und kürzt erst beim Rendern auf die
/// sichtbaren Zeilen. Die anderen Tools behalten die Kurzform `name · pfad`.
fn describe_call(call: &ToolCall) -> String {
    if call.name == "bash" {
        return format!("bash\n{}", arg(&call.arguments, "command"));
    }
    let detail = match call.name.as_str() {
        "read_file" | "write_file" | "edit_file" => arg(&call.arguments, "path"),
        _ => String::new(),
    };
    if detail.is_empty() {
        call.name.clone()
    } else {
        format!("{} · {}", call.name, shorten(&detail, 70))
    }
}

/// Bereitet das Tool-Ergebnis für die UI auf. `bash` und `read_file` reichen die
/// **vollständige** Ausgabe durch — die UI kürzt sie beim Rendern auf die
/// sichtbaren Zeilen, damit exakt sichtbar bleibt, was ausgeführt/gelesen wurde.
/// Andere Tools liefern nur die erste nicht-leere Zeile (bzw. den Diff).
fn summarize(name: &str, content: &str) -> String {
    if name == "bash" || name == "read_file" {
        return content.to_string();
    }
    if let Some((first, diff)) = content.split_once("\n```diff") {
        return format!("{}\n```diff{}", first.trim_end(), diff);
    }
    let first = content.lines().find(|l| !l.trim().is_empty()).unwrap_or("");
    shorten(first, 80)
}

fn arg(arguments: &Value, key: &str) -> String {
    arguments
        .get(key)
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_string()
}

fn shorten(text: &str, max: usize) -> String {
    let text = text.replace('\n', " ");
    if text.chars().count() <= max {
        text
    } else {
        let kept: String = text.chars().take(max.saturating_sub(1)).collect();
        format!("{kept}…")
    }
}
