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

use crate::event::{AgentCommand, AgentEvent};
use crate::llm::{AssistantTurn, ChatMessage, LlmClient, ToolCall, ToolResult, ToolSpec};
use crate::session::SessionWriter;
use crate::tools;

const SYSTEM_PROMPT: &str = "Du bist anvil, ein Coding-Agent, der in einem Terminal läuft. \
     Du hast Werkzeuge, um Dateien zu lesen, zu schreiben und zu bearbeiten sowie Shell-Befehle \
     auszuführen. Nutze sie eigenständig, um die Aufgabe des Nutzers zu erledigen, statt nur zu \
     beschreiben, was zu tun wäre. Wichtig: Quellcode- und Dateiänderungen dürfen ausschließlich \
     über write_file oder edit_file erfolgen. bash ist nur für read-only Inspektion, Builds und Tests; \
     nutze bash niemals, um Dateien per Python/Sed/Redirect/etc. zu verändern. Fasse dich kurz. \
     Antworte in der Sprache des Nutzers.";

/// Obergrenze für Tool-Runden pro Prompt — verhindert Endlosschleifen, falls das
/// Modell immer weiter Tools aufruft.
const MAX_STEPS: usize = 30;

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

    while let Some(command) = commands.recv().await {
        match command {
            AgentCommand::Prompt(prompt) => {
                let Some(client) = client.as_ref() else {
                    let _ = events.send(AgentEvent::Error(
                        "Kein Provider aktiv (API-Key fehlt). Setze ANTHROPIC_API_KEY \
                         oder OPENAI_API_KEY und starte anvil neu."
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
    client: &LlmClient,
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

    for _ in 0..MAX_STEPS {
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
        history.push(ChatMessage::Assistant {
            text: turn.text,
            tool_calls: turn.tool_calls,
        });

        // Keine Tool-Aufrufe → das war die finale Antwort. Turn persistieren.
        if tool_calls.is_empty() {
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
                summary: summarize(&result.content),
            });
            results.push(result);
        }
        history.push(ChatMessage::ToolResults(results));
    }

    // Step-Limit erreicht.
    abort(
        history,
        restore_to,
        events,
        AgentEvent::Error(format!("Abgebrochen nach {MAX_STEPS} Tool-Schritten.")),
    );
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
fn describe_call(call: &ToolCall) -> String {
    let detail = match call.name.as_str() {
        "bash" => arg(&call.arguments, "command"),
        "read_file" | "write_file" | "edit_file" => arg(&call.arguments, "path"),
        _ => String::new(),
    };
    if detail.is_empty() {
        call.name.clone()
    } else {
        format!("{} · {}", call.name, shorten(&detail, 70))
    }
}

/// Erste nicht-leere Zeile eines Tool-Ergebnisses, gekürzt — als kurze Statuszeile.
fn summarize(content: &str) -> String {
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
