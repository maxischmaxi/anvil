//! Der gesamte UI-Zustand und die Logik, die ihn verändert.
//!
//! `App` ist bewusst rein: Es zeichnet nichts und macht keine I/O. Es bekommt
//! Events (Tastatur, Agent) rein und aktualisiert seinen Zustand. Das Rendern
//! lebt in [`crate::ui`], das Senden an den Agent geht über einen Kanal.

use std::time::{Duration, SystemTime};

use ratatui::crossterm::event::{KeyCode, KeyEvent, KeyEventKind, KeyModifiers};
use tokio::sync::mpsc::UnboundedSender;

use crate::auth::{self, AuthInfo};
use crate::config;
use crate::event::{AgentCommand, AgentEvent};
use crate::llm::{AuthMode, ChatMessage, ProviderKind, Secret, ToolCall, ToolResult};
use crate::session::{self, SessionId, SessionMeta};
use crate::tokens::TokenStats;

/// Wer eine Nachricht im Verlauf geschrieben hat.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Role {
    User,
    Assistant,
    System,
    /// Eine Tool-Aktivität (Aufruf + Ergebnis-Status).
    Tool,
}

/// Eine einzelne Nachricht im Gesprächsverlauf.
#[derive(Debug, Clone)]
pub struct Message {
    pub role: Role,
    pub content: String,
}

impl Message {
    fn new(role: Role, content: impl Into<String>) -> Self {
        Self {
            role,
            content: content.into(),
        }
    }
}

/// Welcher Provider/Modell/Credential-Modus gerade aktiv ist.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ActiveModel {
    pub kind: ProviderKind,
    pub model: String,
    pub auth_mode: AuthMode,
}

impl ActiveModel {
    pub fn new(kind: ProviderKind, model: String, auth_mode: AuthMode) -> Self {
        Self {
            kind,
            model,
            auth_mode,
        }
    }

    pub fn label(&self) -> String {
        format!(
            "{} · {} · {}",
            self.kind.display(),
            self.model,
            self.auth_mode.indicator()
        )
    }
}

/// Ob der Agent gerade arbeitet — steuert die Statuszeile und blockiert das
/// Absenden eines zweiten Prompts, solange noch einer läuft.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Status {
    Idle,
    Thinking,
    /// Das Modell hat eine Rückfrage gestellt; die nächste Eingabe ist die
    /// Antwort darauf (Eingabe ist erlaubt, obwohl gerade ein Turn läuft).
    AwaitingAnswer,
}

/// Spinner-Frames (Braille) für den „anvil arbeitet"-Indikator.
const SPINNER: [&str; 10] = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];
const LARGE_PASTE_LINE_THRESHOLD: usize = 8;
const LARGE_PASTE_CHAR_THRESHOLD: usize = 1_200;

#[derive(Debug, Clone, Copy)]
pub struct CommandSuggestion {
    pub name: &'static str,
    pub hint: &'static str,
}

#[derive(Debug, Clone)]
struct ResumePicker {
    sessions: Vec<SessionMeta>,
    selected: usize,
}

const COMMANDS: &[CommandSuggestion] = &[
    CommandSuggestion { name: "login", hint: "Provider anmelden" },
    CommandSuggestion { name: "resume", hint: "Sitzung auswählen/fortsetzen" },
    CommandSuggestion { name: "new", hint: "neue Sitzung starten" },
    CommandSuggestion { name: "compact", hint: "Kontext verdichten" },
    CommandSuggestion { name: "clear", hint: "UI + Sitzung leeren" },
    CommandSuggestion { name: "models", hint: "Modelle anzeigen/wechseln" },
    CommandSuggestion { name: "logout", hint: "Provider abmelden" },
    CommandSuggestion { name: "help", hint: "Hilfe anzeigen" },
    CommandSuggestion { name: "open", hint: "Alias für resume" },
    CommandSuggestion { name: "model", hint: "Alias für models" },
    CommandSuggestion { name: "auth", hint: "Alias für login" },
    CommandSuggestion { name: "cls", hint: "Alias für clear" },
    CommandSuggestion { name: "?", hint: "Alias für help" },
];

/// Der komplette Anwendungszustand.
pub struct App {
    pub input: String,
    /// Cursor-Position als **Byte-Offset** in `input`. Liegt immer auf einer
    /// Char-Grenze, damit `String::insert`/`remove` nicht paniken.
    pub cursor: usize,
    pub status: Status,
    pub should_quit: bool,
    /// Frame-Zähler für die Spinner-Animation.
    spinner_frame: usize,
    /// Der gerade entstehende Block (gestreamte Antwort oder laufendes Tool),
    /// live im Viewport. Ist er fertig, wandert er in den Scrollback.
    pending: Option<Message>,
    /// Alle bereits committed Blöcke. Wird gehalten, damit bei Terminal-Resize
    /// die komplette History mit neuer Breite neu gerendert werden kann.
    transcript: Vec<Message>,
    /// Fertige Blöcke, die der Haupt-Loop per `insert_before` in den echten
    /// Terminal-Scrollback schiebt (und dann leert).
    scrollback: Vec<Message>,
    /// Aktuell geöffnete Resume-Auswahl. Pfeiltasten bewegen die Auswahl,
    /// Enter lädt die markierte Sitzung.
    resume_picker: Option<ResumePicker>,
    /// Aktiver Provider + Modell + Credential-Modus (für die Statuszeile und die
    /// `●`-Markierung in `/models`). `None`, solange kein Credential vorliegt.
    active: Option<ActiveModel>,
    /// Zuletzt per `/models` angezeigte Provider/Modell-Kombis (Nummerierung für
    /// `/models <n>`).
    model_list: Vec<(ProviderKind, String)>,
    /// Läuft gerade eine maskierte Key-Eingabe (`/login <provider>`)? Dann wird
    /// die nächste Eingabe als Secret behandelt, maskiert und nicht in den
    /// Scrollback gespiegelt.
    secret_entry: Option<ProviderKind>,
    /// Einmaliges Signal an den Haupt-Loop, Bildschirm + Scrollback zu löschen.
    clear_requested: bool,
    /// Einmaliges Signal an den Haupt-Loop, die gesamte History neu zu rendern.
    reflow_requested: bool,
    /// Index innerhalb der aktuell gefilterten Slash-Command-Vorschläge.
    suggestion_index: usize,
    /// Kanal zum Agent-Task (Prompts/Antworten).
    commands: UnboundedSender<AgentCommand>,
    /// Separater Kanal, um einen laufenden Turn abzubrechen.
    cancel: UnboundedSender<()>,
    token_stats: TokenStats,
    last_activity: Option<SystemTime>,
}

impl App {
    pub fn new(
        commands: UnboundedSender<AgentCommand>,
        cancel: UnboundedSender<()>,
        intro: String,
        active: Option<ActiveModel>,
    ) -> Self {
        let scrollback = if intro.is_empty() {
            Vec::new()
        } else {
            vec![Message::new(Role::System, intro)]
        };

        Self {
            input: String::new(),
            cursor: 0,
            status: Status::Idle,
            should_quit: false,
            spinner_frame: 0,
            pending: None,
            transcript: Vec::new(),
            scrollback,
            resume_picker: None,
            active,
            model_list: Vec::new(),
            secret_entry: None,
            clear_requested: false,
            reflow_requested: false,
            suggestion_index: 0,
            commands,
            cancel,
            token_stats: TokenStats::default(),
            last_activity: None,
        }
    }

    /// Ob gerade ein Secret (API-Key) maskiert eingegeben wird.
    pub fn masking(&self) -> bool {
        self.secret_entry.is_some()
    }

    pub fn command_suggestions(&self) -> Vec<CommandSuggestion> {
        command_suggestions_for(&self.input, self.cursor)
    }

    pub fn selected_suggestion_index(&self) -> usize {
        let count = self.command_suggestions().len();
        if count == 0 {
            0
        } else {
            self.suggestion_index.min(count - 1)
        }
    }

    pub fn resume_picker_open(&self) -> bool {
        self.resume_picker.is_some()
    }

    pub fn resume_picker_rows(&self) -> Option<Vec<(bool, String, String)>> {
        let picker = self.resume_picker.as_ref()?;
        Some(
            picker
                .sessions
                .iter()
                .enumerate()
                .map(|(index, meta)| {
                    (
                        index == picker.selected,
                        meta.title.clone(),
                        relative_time(meta.modified),
                    )
                })
                .collect(),
        )
    }

    fn reset_suggestion_index(&mut self) {
        let count = self.command_suggestions().len();
        if count == 0 {
            self.suggestion_index = 0;
        } else if self.suggestion_index >= count {
            self.suggestion_index = count - 1;
        }
    }

    fn move_suggestion(&mut self, delta: isize) -> bool {
        let count = self.command_suggestions().len();
        if count == 0 {
            self.suggestion_index = 0;
            return false;
        }
        self.suggestion_index = if delta < 0 {
            self.suggestion_index.saturating_sub(delta.unsigned_abs())
        } else {
            (self.suggestion_index + delta as usize).min(count - 1)
        };
        true
    }

    fn accept_suggestion(&mut self) -> bool {
        let suggestions = self.command_suggestions();
        let Some(suggestion) = suggestions.get(self.selected_suggestion_index()) else {
            return false;
        };
        let replacement = format!("/{} ", suggestion.name);
        if self.input.len() == self.cursor {
            self.input = replacement;
            self.cursor = self.input.len();
            return true;
        }
        false
    }

    /// Label des aktiven Modells für die Statuszeile, z. B.
    /// `"OpenAI · gpt-5.5 · ◉ Subscription"`.
    pub fn active_label(&self) -> Option<String> {
        self.active.as_ref().map(ActiveModel::label)
    }

    pub fn token_stats(&self) -> TokenStats {
        self.token_stats
    }

    pub fn stalled_hint(&self) -> bool {
        matches!(self.status, Status::Thinking)
            && self
                .last_activity
                .and_then(|t| t.elapsed().ok())
                .is_some_and(|elapsed| elapsed > Duration::from_secs(90))
    }

    // ---- Eingabe-Bearbeitung (Unicode-sicher über Byte-Offsets) ----

    fn insert_char(&mut self, c: char) {
        self.input.insert(self.cursor, c);
        self.cursor += c.len_utf8();
    }

    fn insert_str(&mut self, text: &str) {
        self.input.insert_str(self.cursor, text);
        self.cursor += text.len();
    }

    fn delete_backward(&mut self) {
        if self.cursor == 0 {
            return;
        }
        let prev_len = self.input[..self.cursor]
            .chars()
            .next_back()
            .map(char::len_utf8)
            .unwrap_or(0);
        self.cursor -= prev_len;
        self.input.remove(self.cursor);
    }

    fn move_left(&mut self) {
        if let Some(c) = self.input[..self.cursor].chars().next_back() {
            self.cursor -= c.len_utf8();
        }
    }

    fn move_right(&mut self) {
        if let Some(c) = self.input[self.cursor..].chars().next() {
            self.cursor += c.len_utf8();
        }
    }

    fn move_word_left(&mut self) {
        self.cursor = self.prev_boundary(is_word_char);
    }

    fn move_word_right(&mut self) {
        self.cursor = self.next_boundary(is_word_char);
    }

    /// Zeichen unter dem Cursor löschen (Ctrl+D / Entf).
    fn delete_forward(&mut self) {
        if self.cursor < self.input.len() {
            self.input.remove(self.cursor);
        }
    }

    /// Wort vor dem Cursor löschen (Alt+Backspace) — Wortgrenze = alphanumerisch/`_`.
    fn delete_word_backward(&mut self) {
        let start = self.prev_boundary(is_word_char);
        self.input.replace_range(start..self.cursor, "");
        self.cursor = start;
    }

    /// „WORD" vor dem Cursor löschen (Ctrl+W) — Grenze = Whitespace, wie in der Shell.
    fn delete_word_backward_whitespace(&mut self) {
        let start = self.prev_boundary(|c| !c.is_whitespace());
        self.input.replace_range(start..self.cursor, "");
        self.cursor = start;
    }

    /// Wort nach dem Cursor löschen (Alt+D).
    fn delete_word_forward(&mut self) {
        let end = self.next_boundary(is_word_char);
        self.input.replace_range(self.cursor..end, "");
    }

    /// Vom Cursor bis Zeilenanfang löschen (Ctrl+U).
    fn delete_to_start(&mut self) {
        self.input.replace_range(..self.cursor, "");
        self.cursor = 0;
    }

    /// Vom Cursor bis Zeilenende löschen (Ctrl+K).
    fn delete_to_end(&mut self) {
        self.input.truncate(self.cursor);
    }

    /// Die zwei Zeichen vor dem Cursor vertauschen (Ctrl+T).
    fn transpose(&mut self) {
        let mut boundaries = self.input[..self.cursor].char_indices().rev();
        let Some((last, _)) = boundaries.next() else {
            return;
        };
        let Some((prev, _)) = boundaries.next() else {
            return;
        };
        let left = self.input[prev..last].to_string();
        let right = self.input[last..self.cursor].to_string();
        self.input
            .replace_range(prev..self.cursor, &format!("{right}{left}"));
    }

    /// Byte-Offset des vorherigen Wortanfangs: erst Nicht-Wort-Zeichen, dann
    /// Wort-Zeichen rückwärts überspringen.
    fn prev_boundary(&self, is_word: impl Fn(char) -> bool) -> usize {
        let prefix = &self.input[..self.cursor];
        let mut pos = self.cursor;
        let mut iter = prefix.char_indices().rev().peekable();
        while let Some(&(idx, c)) = iter.peek() {
            if is_word(c) {
                break;
            }
            pos = idx;
            iter.next();
        }
        while let Some(&(idx, c)) = iter.peek() {
            if !is_word(c) {
                break;
            }
            pos = idx;
            iter.next();
        }
        pos
    }

    /// Byte-Offset des nächsten Wortendes: erst Nicht-Wort-Zeichen, dann
    /// Wort-Zeichen vorwärts überspringen.
    fn next_boundary(&self, is_word: impl Fn(char) -> bool) -> usize {
        let suffix = &self.input[self.cursor..];
        let mut pos = self.cursor;
        let mut iter = suffix.char_indices().peekable();
        while let Some(&(idx, c)) = iter.peek() {
            if is_word(c) {
                break;
            }
            pos = self.cursor + idx + c.len_utf8();
            iter.next();
        }
        while let Some(&(idx, c)) = iter.peek() {
            if !is_word(c) {
                break;
            }
            pos = self.cursor + idx + c.len_utf8();
            iter.next();
        }
        pos
    }

    fn submit(&mut self) {
        // Maskierte Key-Eingabe (/login): die Eingabe ist das Secret, nicht ein
        // Prompt oder Befehl — abfangen, bevor irgendetwas in den Scrollback geht.
        if let Some(kind) = self.secret_entry.take() {
            let key = self.input.trim().to_string();
            self.input.clear();
            self.cursor = 0;
            if key.is_empty() {
                self.note("Login abgebrochen (kein Key eingegeben).".to_string());
            } else {
                self.store_and_activate_key(kind, key);
            }
            return;
        }

        if self.resume_picker.is_some() {
            return;
        }
        let text = self.input.trim().to_string();
        if text.is_empty() {
            return;
        }
        // Slash-Befehle nur im Leerlauf (nicht als laufende Antwort/Rückfrage).
        if self.status == Status::Idle
            && let Some(command) = text.strip_prefix('/')
        {
            self.input.clear();
            self.cursor = 0;
            self.handle_command(command);
            return;
        }
        if self.status == Status::Thinking {
            return;
        }
        // Der Prompt geht sofort in den Scrollback; die Antwort entsteht im
        // Viewport und wird committed, wenn sie fertig ist.
        self.scrollback.push(Message::new(Role::User, text.clone()));
        self.status = Status::Thinking;
        self.last_activity = Some(SystemTime::now());
        self.input.clear();
        self.cursor = 0;
        let _ = self.commands.send(AgentCommand::Prompt(text));
    }

    // ---- Slash-Befehle ----

    fn handle_command(&mut self, command: &str) {
        let mut parts = command.split_whitespace();
        let name = parts.next().unwrap_or("");
        let arg = parts.next();
        // Rest der Zeile (für /login <provider> [key|oauth]).
        let rest = command[name.len()..].trim();
        match name {
            "resume" | "open" => self.cmd_resume(),
            "new" => self.cmd_new(),
            "compact" => self.cmd_compact(),
            "clear" | "cls" => self.cmd_clear(),
            "models" | "model" => self.cmd_models(arg),
            "login" | "auth" => self.cmd_login(rest),
            "logout" => self.cmd_logout(arg),
            "help" | "?" => self.cmd_help(),
            other => self.note(format!(
                "Unbekannter Befehl: /{other}. /help zeigt alle Befehle."
            )),
        }
    }

    fn cmd_resume(&mut self) {
        let sessions = session::list();
        if sessions.is_empty() {
            self.note("Noch keine gespeicherten Sitzungen.".to_string());
            return;
        }
        self.resume_picker = Some(ResumePicker {
            sessions,
            selected: 0,
        });
        self.input = "/resume".to_string();
        self.cursor = self.input.len();
    }

    fn load_session(&mut self, id: SessionId) {
        self.resume_picker = None;
        self.input.clear();
        self.cursor = 0;

        match session::load(&id) {
            Ok(history) => {
                self.scrollback
                    .push(Message::new(Role::System, "── Sitzung fortgesetzt ──"));
                for message in history_to_display(&history) {
                    self.scrollback.push(message);
                }
                let _ = self.commands.send(AgentCommand::SetContext { history, id });
            }
            Err(error) => self.note(format!("Konnte Sitzung nicht laden: {error:#}")),
        }
    }

    fn move_resume_selection(&mut self, delta: isize) -> bool {
        let Some(picker) = &mut self.resume_picker else {
            return false;
        };
        if picker.sessions.is_empty() {
            picker.selected = 0;
            return true;
        }
        picker.selected = if delta < 0 {
            picker.selected.saturating_sub(delta.unsigned_abs())
        } else {
            (picker.selected + delta as usize).min(picker.sessions.len() - 1)
        };
        true
    }

    fn accept_resume_selection(&mut self) -> bool {
        let Some(picker) = &self.resume_picker else {
            return false;
        };
        let Some(meta) = picker.sessions.get(picker.selected) else {
            return false;
        };
        self.load_session(meta.id.clone());
        true
    }

    fn cancel_resume_selection(&mut self) -> bool {
        if self.resume_picker.take().is_some() {
            self.input.clear();
            self.cursor = 0;
            return true;
        }
        false
    }

    fn cmd_new(&mut self) {
        let _ = self.commands.send(AgentCommand::Reset);
        self.input.clear();
        self.cursor = 0;
        self.status = Status::Idle;
        self.pending = None;
        self.transcript.clear();
        self.scrollback.clear();
        self.resume_picker = None;
        self.clear_requested = true;
        self.reflow_requested = false;
        self.note("Neue Sitzung gestartet.".to_string());
    }

    fn cmd_compact(&mut self) {
        self.status = Status::Thinking;
        self.last_activity = Some(SystemTime::now());
        self.note("Verdichte Kontext…".to_string());
        let _ = self.commands.send(AgentCommand::Compact);
    }

    fn cmd_clear(&mut self) {
        let _ = self.commands.send(AgentCommand::Reset);
        self.input.clear();
        self.cursor = 0;
        self.status = Status::Idle;
        self.pending = None;
        self.transcript.clear();
        self.scrollback.clear();
        self.resume_picker = None;
        self.clear_requested = true;
        self.reflow_requested = false;
    }

    // ---- /models ----

    fn cmd_models(&mut self, arg: Option<&str>) {
        if let Some(sel) = arg {
            self.cmd_models_select(sel);
            return;
        }
        self.model_list.clear();
        let mut text = String::from(
            "Modelle (● = aktiv, [key]/[sub] = Credential-Schiene) — /models <n> zum Wechseln:\n",
        );
        for kind in ProviderKind::ALL {
            if config::secret_for(kind).is_some() {
                for model in kind.models() {
                    self.model_list.push((kind, (*model).to_string()));
                    let n = self.model_list.len();
                    let active = self
                        .active
                        .as_ref()
                        .is_some_and(|active| active.kind == kind && active.model == *model);
                    let mark = if active { "●" } else { " " };
                    let mode = config::auth_mode_for_provider(kind)
                        .map(|mode| mode.short_indicator())
                        .unwrap_or("?");
                    text.push_str(&format!(
                        "  {n:>2}  {mark} [{mode}] {} · {model}\n",
                        kind.display()
                    ));
                }
            } else {
                text.push_str(&format!(
                    "       · {} — nicht angemeldet (/login {})\n",
                    kind.display(),
                    kind.id()
                ));
            }
        }
        if self.model_list.is_empty() {
            text.push_str("\nNoch kein Provider angemeldet. /login <provider> zum Start.");
        }
        self.note(text);
    }

    fn cmd_models_select(&mut self, sel: &str) {
        let Some(number) = sel.parse::<usize>().ok() else {
            self.note("Nutzung: /models <nummer> — erst /models ausführen.".to_string());
            return;
        };
        let Some((kind, model)) = self.model_list.get(number.wrapping_sub(1)).cloned() else {
            self.note("Keine Modellnummer. Erst /models ausführen.".to_string());
            return;
        };
        self.switch_to(kind, model);
    }

    /// Schaltet Provider+Modell zur Laufzeit um (Kontext bleibt erhalten).
    fn switch_to(&mut self, kind: ProviderKind, model: String) {
        let Some(secret) = config::secret_for(kind) else {
            self.note(format!(
                "Kein Credential für {}. Erst /login {}.",
                kind.display(),
                kind.id()
            ));
            return;
        };
        let auth_info = config::auth_info_for(kind);
        let auth_mode = config::auth_mode_for_provider(kind).unwrap_or(AuthMode::ApiKey);
        let _ = self.commands.send(AgentCommand::SetClient {
            kind,
            model: model.clone(),
            secret: Secret(secret),
            auth_info,
        });
        let active = ActiveModel::new(kind, model, auth_mode);
        self.note(format!("Aktiv: {}", active.label()));
        self.active = Some(active);
    }

    // ---- /login + /logout ----

    fn cmd_login(&mut self, rest: &str) {
        let mut parts = rest.split_whitespace();
        let Some(id) = parts.next() else {
            self.cmd_login_list();
            return;
        };
        let Some(kind) = provider_from_arg(id) else {
            self.note(format!(
                "Unbekannter Provider: {id}. Bekannt: openai, anthropic, google, openrouter."
            ));
            return;
        };
        match parts.next() {
            Some("oauth") => self.cmd_login_oauth(kind),
            Some(key) => {
                // Inline-Key: bequem, aber sichtbar.
                self.note(
                    "Hinweis: der Key stand sichtbar in der Eingabe. Für maskierte Eingabe \
                     nur '/login <provider>' (ohne Key) verwenden."
                        .to_string(),
                );
                self.store_and_activate_key(kind, key.to_string());
            }
            None => {
                self.secret_entry = Some(kind);
                self.note(format!(
                    "🔑 API-Key für {} eingeben — Enter speichert, Esc bricht ab. \
                     (Die Eingabe wird maskiert.)",
                    kind.display()
                ));
            }
        }
    }

    fn cmd_login_list(&mut self) {
        let mut text = String::from("Anmelden mit /login <provider>:\n");
        for kind in ProviderKind::ALL {
            let status = match config::source_label(kind) {
                Some(src) => format!("✓ angemeldet ({src})"),
                None => "— nicht angemeldet".to_string(),
            };
            text.push_str(&format!(
                "  {:<11}{:<16}{status}\n",
                kind.id(),
                kind.display()
            ));
        }
        text.push_str(
            "\n/login <provider>     Key maskiert eingeben\n\
             /login <provider> oauth   (Subscription-Login — siehe Hinweis)\n\
             /logout <provider>    gespeicherten Key entfernen",
        );
        self.note(text);
    }

    fn cmd_login_oauth(&mut self, kind: ProviderKind) {
        if kind.supports_oauth() {
            self.note(
                "OpenAI-Subscription-Login gestartet. Falls der Browser nicht aufgeht: \
                 öffne die angezeigte Auth-Seite manuell. Nach Erfolg ist OpenAI aktiv."
                    .to_string(),
            );
            self.status = Status::Thinking;
            self.active = Some(ActiveModel::new(
                kind,
                kind.default_model().to_string(),
                AuthMode::OpenAiSubscription,
            ));
            let _ = self.commands.send(AgentCommand::LoginOauth { kind });
        } else {
            self.note(format!(
                "Für {} ist kein OAuth-/Subscription-Login verfügbar. Nutze /login {} mit API-Key.",
                kind.display(),
                kind.id()
            ));
        }
    }

    fn store_and_activate_key(&mut self, kind: ProviderKind, key: String) {
        match auth::set(kind.id(), AuthInfo::Api { key }) {
            Ok(()) => {
                self.note(format!("✓ {} angemeldet — API-Key gespeichert.", kind.display()));
                // Direkt aktiv schalten — praktisch, wenn vorher kein Provider lief.
                self.switch_to(kind, kind.default_model().to_string());
            }
            Err(error) => self.note(format!("Konnte Key nicht speichern: {error:#}")),
        }
    }

    fn cmd_logout(&mut self, arg: Option<&str>) {
        let Some(id) = arg else {
            self.note("Nutzung: /logout <provider>.".to_string());
            return;
        };
        let Some(kind) = provider_from_arg(id) else {
            self.note(format!("Unbekannter Provider: {id}."));
            return;
        };
        match auth::remove(kind.id()) {
            Ok(true) => self.note(format!(
                "{} abgemeldet (gespeicherter Key entfernt).",
                kind.display()
            )),
            Ok(false) => self.note(format!("Für {} war kein Key gespeichert.", kind.display())),
            Err(error) => self.note(format!("Konnte Key nicht entfernen: {error:#}")),
        }
    }

    fn cmd_help(&mut self) {
        self.note(
            "Befehle:\n  \
             /resume       Sitzungen auswählen/fortsetzen (↑/↓, Enter)\n  \
             /new          neue Sitzung beginnen\n  \
             /compact      Kontext wie opencode verdichten\n  \
             /clear        neue blanke Sitzung + UI löschen\n  \
             /models [n]   Modelle anzeigen / Provider+Modell wechseln\n  \
             /login [prov] Provider anmelden (API-Key, maskiert)\n  \
             /logout <prov> gespeicherten Key entfernen\n  \
             /help         diese Hilfe"
                .to_string(),
        );
    }

    /// Eine System-Notiz in den Scrollback schreiben.
    fn note(&mut self, text: String) {
        self.scrollback.push(Message::new(Role::System, text));
    }

    /// Esc: läuft ein Turn, wird er abgebrochen; sonst wird die Eingabe geleert.
    /// (Beenden geht über Strg+C.)
    fn on_escape(&mut self) {
        if self.cancel_resume_selection() {
            return;
        }
        if self.secret_entry.take().is_some() {
            self.input.clear();
            self.cursor = 0;
            self.note("Login abgebrochen.".to_string());
            return;
        }
        match self.status {
            Status::Thinking | Status::AwaitingAnswer => {
                let _ = self.cancel.send(());
            }
            Status::Idle => {
                self.input.clear();
                self.cursor = 0;
            }
        }
    }

    // ---- Event-Handler ----

    /// Reagiert auf einen Tastendruck aus dem Terminal.
    pub fn on_key(&mut self, key: KeyEvent) {
        // Manche Terminals (Windows, Kitty-Protokoll) melden auch Key-Release;
        // wir reagieren nur auf das Drücken.
        if key.kind == KeyEventKind::Release {
            return;
        }
        let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);
        let alt = key.modifiers.contains(KeyModifiers::ALT);
        let shift = key.modifiers.contains(KeyModifiers::SHIFT);

        match key.code {
            // Steuerung
            KeyCode::Char('c') if ctrl => self.should_quit = true,
            KeyCode::Esc => self.on_escape(),
            KeyCode::Up if self.move_resume_selection(-1) => {}
            KeyCode::Down if self.move_resume_selection(1) => {}
            KeyCode::Enter if self.accept_resume_selection() => {}
            KeyCode::Tab if !ctrl && !alt => {
                self.accept_suggestion();
            }
            KeyCode::Up if self.move_suggestion(-1) => {}
            KeyCode::Down if self.move_suggestion(1) => {}
            // Alt+Enter / Shift+Enter: neue Zeile; Enter: absenden.
            KeyCode::Enter if alt || shift => self.insert_char('\n'),
            KeyCode::Enter => self.submit(),

            // Cursor wortweise (Strg/Alt + Pfeil, Alt+B/F)
            KeyCode::Left if ctrl || alt => self.move_word_left(),
            KeyCode::Right if ctrl || alt => self.move_word_right(),
            KeyCode::Char('b') if alt => self.move_word_left(),
            KeyCode::Char('f') if alt => self.move_word_right(),

            // Cursor zeichenweise (Pfeil, Ctrl+B/F)
            KeyCode::Left => self.move_left(),
            KeyCode::Right => self.move_right(),
            KeyCode::Char('b') if ctrl => self.move_left(),
            KeyCode::Char('f') if ctrl => self.move_right(),

            // Zeilenanfang/-ende (Home/End, Ctrl+A/E)
            KeyCode::Home => self.cursor = 0,
            KeyCode::End => self.cursor = self.input.len(),
            KeyCode::Char('a') if ctrl => self.cursor = 0,
            KeyCode::Char('e') if ctrl => self.cursor = self.input.len(),

            // Löschen
            KeyCode::Backspace if ctrl => self.delete_word_backward_whitespace(),
            KeyCode::Backspace if alt => self.delete_word_backward(),
            KeyCode::Backspace => self.delete_backward(),
            KeyCode::Delete => self.delete_forward(),
            KeyCode::Char('h') if ctrl => self.delete_backward(),
            KeyCode::Char('d') if ctrl => self.delete_forward(),
            KeyCode::Char('w') if ctrl => self.delete_word_backward_whitespace(),
            KeyCode::Char('d') if alt => self.delete_word_forward(),
            KeyCode::Char('u') if ctrl => self.delete_to_start(),
            KeyCode::Char('k') if ctrl => self.delete_to_end(),
            KeyCode::Char('t') if ctrl => self.transpose(),

            // Normales Zeichen einfügen (keine Modifier-Kombination)
            KeyCode::Char(c) if !ctrl && !alt => self.insert_char(c),
            _ => {}
        }
        self.reset_suggestion_index();
    }

    /// Reagiert auf eingefügten Text. Kleine Pastes landen unverändert in der
    /// Eingabe (inkl. Zeilenumbrüchen); sehr große Pastes werden kompakt als
    /// Platzhalter eingefügt, damit das Prompt-Fenster nicht explodiert.
    pub fn on_paste(&mut self, text: &str) {
        if text.is_empty() {
            return;
        }

        let normalized = text.replace("\r\n", "\n").replace('\r', "\n");
        let line_count = normalized.lines().count().max(1);
        let char_count = normalized.chars().count();
        if line_count > LARGE_PASTE_LINE_THRESHOLD || char_count > LARGE_PASTE_CHAR_THRESHOLD {
            self.insert_str(&format!("[{line_count} inserted lines]"));
        } else {
            self.insert_str(&normalized);
        }
    }

    /// Reagiert auf eine Meldung des Agent-Tasks.
    pub fn on_agent_event(&mut self, event: AgentEvent) {
        self.last_activity = Some(SystemTime::now());
        match event {
            AgentEvent::Chunk(text) => self.append_assistant(&text),
            AgentEvent::TokenStats(stats) => self.token_stats = stats,
            AgentEvent::CompactStarted => {
                self.commit_pending();
                self.pending = Some(Message::new(Role::System, "Verdichte Kontext…"));
            }
            AgentEvent::Compacted { summary, recent } => {
                self.commit_pending();
                self.note(format!(
                    "Kontext verdichtet.\n\n{summary}\n\nRecent erhalten: {} Tokens (geschätzt).",
                    crate::tokens::estimate_text(&recent)
                ));
            }
            AgentEvent::ToolStarted(title) => {
                // Vorherigen Block (z. B. Assistenten-Text) abschließen, dann das
                // laufende Tool als neuen pending-Block live im Viewport zeigen.
                self.commit_pending();
                self.pending = Some(Message::new(Role::Tool, title));
            }
            AgentEvent::ToolFinished { ok, summary } => {
                let mark = if ok { "→" } else { "✗" };
                let line = format!("\n  {mark} {summary}");
                match &mut self.pending {
                    Some(m) if m.role == Role::Tool => m.content.push_str(&line),
                    _ => self.pending = Some(Message::new(Role::Tool, line)),
                }
                // Tool fertig → in den Scrollback.
                self.commit_pending();
            }
            AgentEvent::AskUser(question) => {
                self.commit_pending();
                self.scrollback
                    .push(Message::new(Role::Assistant, question));
                self.status = Status::AwaitingAnswer;
            }
            AgentEvent::Done => {
                self.commit_pending();
                self.status = Status::Idle;
            }
            AgentEvent::Cancelled => {
                self.commit_pending();
                self.scrollback
                    .push(Message::new(Role::System, "⊘ Abgebrochen.".to_string()));
                self.status = Status::Idle;
            }
            AgentEvent::Error(reason) => {
                self.commit_pending();
                self.scrollback
                    .push(Message::new(Role::System, format!("Fehler: {reason}")));
                self.status = Status::Idle;
            }
        }
    }

    /// Hängt gestreamten Text an den laufenden Assistenten-Block an (oder legt
    /// einen an, falls gerade keiner offen ist).
    fn append_assistant(&mut self, text: &str) {
        match &mut self.pending {
            Some(m) if m.role == Role::Assistant => m.content.push_str(text),
            _ => {
                self.commit_pending();
                self.pending = Some(Message::new(Role::Assistant, text.to_string()));
            }
        }
    }

    /// Schiebt den fertigen pending-Block in die Scrollback-Queue.
    fn commit_pending(&mut self) {
        if let Some(message) = self.pending.take() {
            self.scrollback.push(message);
        }
    }

    /// Entnimmt die fertigen Blöcke, die in den Terminal-Scrollback sollen, und
    /// archiviert sie für späteres Reflow bei Terminal-Resize.
    pub fn drain_scrollback(&mut self) -> Vec<Message> {
        let messages = std::mem::take(&mut self.scrollback);
        self.transcript.extend(messages.iter().cloned());
        messages
    }

    /// Signalisiert, dass die gespeicherte History mit aktueller Terminalbreite
    /// neu in den Scrollback gerendert werden soll.
    pub fn request_reflow(&mut self) {
        self.reflow_requested = true;
    }

    /// Gibt die komplette History für ein einmaliges Reflow zurück. Noch nicht
    /// geflushte Blöcke werden vorher archiviert, damit nichts verloren geht.
    pub fn take_reflow_messages(&mut self) -> Option<Vec<Message>> {
        if !std::mem::take(&mut self.reflow_requested) {
            return None;
        }
        let queued = std::mem::take(&mut self.scrollback);
        self.transcript.extend(queued);
        Some(self.transcript.clone())
    }

    /// Gibt ein angefordertes Terminal-Clear einmalig an den Haupt-Loop weiter.
    pub fn take_clear_requested(&mut self) -> bool {
        std::mem::take(&mut self.clear_requested)
    }

    /// Schaltet den Spinner einen Frame weiter (vom Render-Tick aufgerufen).
    pub fn tick(&mut self) {
        self.spinner_frame = self.spinner_frame.wrapping_add(1);
    }

    /// Aktuelles Spinner-Zeichen.
    pub fn spinner(&self) -> &'static str {
        SPINNER[self.spinner_frame % SPINNER.len()]
    }
}

/// Ob ein Zeichen zu einem „Wort" gehört (für wortweise Bewegung/Löschung).
/// Unterstriche zählen mit, damit Bezeichner wie `foo_bar` als ein Wort gelten.
fn is_word_char(c: char) -> bool {
    c.is_alphanumeric() || c == '_'
}

fn command_suggestions_for(input: &str, cursor: usize) -> Vec<CommandSuggestion> {
    if cursor != input.len() || !input.starts_with('/') || input.contains('\n') {
        return Vec::new();
    }
    let typed = &input[1..];
    if typed.contains(char::is_whitespace) {
        return Vec::new();
    }
    COMMANDS
        .iter()
        .copied()
        .filter(|command| command.name.starts_with(typed))
        .collect()
}

/// Provider aus einem `/login`/`/logout`-Argument auflösen (inkl. Alias `gemini`).
fn provider_from_arg(id: &str) -> Option<ProviderKind> {
    ProviderKind::from_id(id).or(match id {
        "gemini" => Some(ProviderKind::Gemini),
        _ => None,
    })
}

/// Wandelt einen geladenen Verlauf in Anzeige-Nachrichten für den Scrollback um.
/// Tool-Ergebnisse werden an die zugehörigen Tool-Blöcke gehängt, damit
/// fortgesetzte Sessions nicht dauerhaft „läuft …" anzeigen.
fn history_to_display(history: &[ChatMessage]) -> Vec<Message> {
    let mut out = Vec::new();
    let mut pending_tools: Vec<Message> = Vec::new();
    for message in history {
        match message {
            ChatMessage::User(text) => {
                out.append(&mut pending_tools);
                out.push(Message::new(Role::User, text.clone()));
            }
            ChatMessage::Assistant { text, tool_calls } => {
                out.append(&mut pending_tools);
                if !text.is_empty() {
                    out.push(Message::new(Role::Assistant, text.clone()));
                }
                pending_tools = tool_calls
                    .iter()
                    .map(|call| Message::new(Role::Tool, tool_title(call)))
                    .collect();
            }
            ChatMessage::ToolResults(results) => {
                attach_tool_results(&mut pending_tools, results);
                out.append(&mut pending_tools);
            }
        }
    }
    out.append(&mut pending_tools);
    out
}

/// Kopf-/Befehlszeile eines Tool-Blocks. `bash` bekommt den **vollständigen**
/// Befehl in den Body (mehrzeilig möglich, Kopfzeile nur `bash`), passend zu
/// [`crate::agent`]; andere Tools die Kurzform `name · pfad`.
fn tool_title(call: &ToolCall) -> String {
    if call.name == "bash" {
        let command = call
            .arguments
            .get("command")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        return format!("bash\n{command}");
    }
    let detail = call
        .arguments
        .get("path")
        .and_then(|v| v.as_str())
        .unwrap_or("");
    if detail.is_empty() {
        call.name.clone()
    } else {
        format!("{} · {}", call.name, detail)
    }
}

/// Ob eine Tool-Nachricht ihre vollständige Ausgabe im UI-Block behalten soll.
fn keeps_full_tool_output(content: &str) -> bool {
    matches!(content.split('\n').next(), Some("bash") | Some("read_file"))
}

fn attach_tool_results(messages: &mut [Message], results: &[ToolResult]) {
    for (message, result) in messages.iter_mut().zip(results) {
        let mark = if result.is_error { "✗" } else { "→" };
        // bash/read_file zeigen die vollständige Ausgabe (gekürzt erst beim Rendern),
        // damit fortgesetzte Sessions denselben Block liefern wie der Live-Lauf;
        // andere Tools nur die kompakte Zusammenfassung.
        let body = if keeps_full_tool_output(&message.content) {
            result.content.clone()
        } else {
            summarize_tool_result(&result.content)
        };
        message.content.push_str(&format!("\n  {mark} {body}"));
    }
}

fn summarize_tool_result(content: &str) -> String {
    if let Some((first, diff)) = content.split_once("\n```diff") {
        return format!("{}\n```diff{}", first.trim_end(), diff);
    }
    content
        .lines()
        .find(|line| !line.trim().is_empty())
        .unwrap_or("")
        .to_string()
}

/// Grobe relative Zeitangabe für die Sitzungsliste.
fn relative_time(time: SystemTime) -> String {
    let seconds = SystemTime::now()
        .duration_since(time)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    match seconds {
        0..=59 => "gerade eben".to_string(),
        60..=3599 => format!("vor {} Min", seconds / 60),
        3600..=86399 => format!("vor {} Std", seconds / 3600),
        _ => format!("vor {} Tagen", seconds / 86400),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::sync::mpsc;

    fn app_with(input: &str, cursor: usize) -> App {
        let (commands, _command_rx) = mpsc::unbounded_channel();
        let (cancel, _cancel_rx) = mpsc::unbounded_channel();
        let mut app = App::new(commands, cancel, String::new(), None);
        app.input = input.to_string();
        app.cursor = cursor;
        app
    }

    #[test]
    fn login_enters_then_escapes_masked_entry() {
        let mut app = app_with("", 0);
        app.handle_command("login openai");
        assert!(app.masking(), "/login <provider> startet die maskierte Eingabe");
        app.on_escape();
        assert!(!app.masking(), "Esc bricht die Key-Eingabe ab");
    }

    #[test]
    fn login_unknown_provider_does_not_mask() {
        let mut app = app_with("", 0);
        app.handle_command("login does-not-exist");
        assert!(!app.masking());
    }

    #[test]
    fn move_word_left_jumps_to_word_start() {
        let mut app = app_with("foo bar", 7);
        app.move_word_left();
        assert_eq!(app.cursor, 4);
        app.move_word_left();
        assert_eq!(app.cursor, 0);
    }

    #[test]
    fn move_word_right_jumps_to_word_end() {
        let mut app = app_with("foo bar", 0);
        app.move_word_right();
        assert_eq!(app.cursor, 3);
        app.move_word_right();
        assert_eq!(app.cursor, 7);
    }

    #[test]
    fn ctrl_w_deletes_whitespace_delimited_word() {
        let mut app = app_with("foo bar baz", 11);
        app.delete_word_backward_whitespace();
        assert_eq!(app.input, "foo bar ");
        assert_eq!(app.cursor, 8);
    }

    #[test]
    fn ctrl_w_treats_punctuation_as_part_of_word() {
        let mut app = app_with("call foo.bar()", 14);
        app.delete_word_backward_whitespace();
        assert_eq!(app.input, "call ");
        assert_eq!(app.cursor, 5);
    }

    #[test]
    fn alt_backspace_stops_at_punctuation() {
        let mut app = app_with("foo.bar", 7);
        app.delete_word_backward();
        assert_eq!(app.input, "foo.");
        assert_eq!(app.cursor, 4);
    }

    #[test]
    fn alt_d_deletes_next_word() {
        let mut app = app_with("foo bar", 0);
        app.delete_word_forward();
        assert_eq!(app.input, " bar");
        assert_eq!(app.cursor, 0);
    }

    #[test]
    fn ctrl_u_deletes_to_start() {
        let mut app = app_with("hello world", 6);
        app.delete_to_start();
        assert_eq!(app.input, "world");
        assert_eq!(app.cursor, 0);
    }

    #[test]
    fn ctrl_k_deletes_to_end() {
        let mut app = app_with("hello world", 5);
        app.delete_to_end();
        assert_eq!(app.input, "hello");
        assert_eq!(app.cursor, 5);
    }

    #[test]
    fn ctrl_d_deletes_char_under_cursor() {
        let mut app = app_with("abc", 1);
        app.delete_forward();
        assert_eq!(app.input, "ac");
        assert_eq!(app.cursor, 1);
    }

    #[test]
    fn transpose_swaps_two_chars_before_cursor() {
        let mut app = app_with("ab", 2);
        app.transpose();
        assert_eq!(app.input, "ba");
    }

    #[test]
    fn slash_commands_are_suggested_and_accepted_with_tab() {
        let mut app = app_with("/lo", 3);
        let suggestions = app.command_suggestions();
        assert!(suggestions.iter().any(|s| s.name == "login"));
        assert!(suggestions.iter().any(|s| s.name == "logout"));
        app.accept_suggestion();
        assert_eq!(app.input, "/login ");
        assert_eq!(app.cursor, app.input.len());
    }

    #[test]
    fn suggestions_are_only_for_command_prefix_at_end() {
        assert!(app_with("hello /lo", 9).command_suggestions().is_empty());
        assert!(app_with("/login openai", 13).command_suggestions().is_empty());
        assert!(app_with("/login", 2).command_suggestions().is_empty());
    }

    #[test]
    fn arrow_keys_select_suggestions() {
        let mut app = app_with("/lo", 3);
        assert_eq!(app.selected_suggestion_index(), 0);
        assert!(app.move_suggestion(1));
        assert_eq!(app.selected_suggestion_index(), 1);
        app.accept_suggestion();
        assert_eq!(app.input, "/logout ");
    }

    #[test]
    fn editing_is_unicode_safe() {
        // Cursor hinter dem mehrbyte-Zeichen 'ä' (2 Bytes) + 'x'.
        let mut app = app_with("äx", "äx".len());
        app.delete_backward();
        assert_eq!(app.input, "ä");
        app.delete_backward();
        assert_eq!(app.input, "");
        assert_eq!(app.cursor, 0);
    }
}
