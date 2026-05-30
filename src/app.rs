//! Der gesamte UI-Zustand und die Logik, die ihn verändert.
//!
//! `App` ist bewusst rein: Es zeichnet nichts und macht keine I/O. Es bekommt
//! Events (Tastatur, Agent) rein und aktualisiert seinen Zustand. Das Rendern
//! lebt in [`crate::ui`], das Senden an den Agent geht über einen Kanal.

use std::time::SystemTime;

use ratatui::crossterm::event::{KeyCode, KeyEvent, KeyEventKind, KeyModifiers};
use tokio::sync::mpsc::UnboundedSender;

use crate::event::{AgentCommand, AgentEvent};
use crate::llm::ChatMessage;
use crate::session::{self, SessionId};

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
    /// Fertige Blöcke, die der Haupt-Loop per `insert_before` in den echten
    /// Terminal-Scrollback schiebt (und dann leert). Die App hält die History
    /// also nicht selbst — das Terminal tut es.
    scrollback: Vec<Message>,
    /// Zuletzt per `/sessions` angezeigte Sitzungen (Reihenfolge = Nummerierung),
    /// damit `/resume <n>` die richtige id auflösen kann.
    session_list: Vec<SessionId>,
    /// Kanal zum Agent-Task (Prompts/Antworten).
    commands: UnboundedSender<AgentCommand>,
    /// Separater Kanal, um einen laufenden Turn abzubrechen.
    cancel: UnboundedSender<()>,
}

impl App {
    pub fn new(
        commands: UnboundedSender<AgentCommand>,
        cancel: UnboundedSender<()>,
        intro: String,
    ) -> Self {
        Self {
            input: String::new(),
            cursor: 0,
            status: Status::Idle,
            should_quit: false,
            spinner_frame: 0,
            pending: None,
            scrollback: vec![Message::new(Role::System, intro)],
            session_list: Vec::new(),
            commands,
            cancel,
        }
    }

    // ---- Eingabe-Bearbeitung (Unicode-sicher über Byte-Offsets) ----

    fn insert_char(&mut self, c: char) {
        self.input.insert(self.cursor, c);
        self.cursor += c.len_utf8();
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
        self.input.replace_range(prev..self.cursor, &format!("{right}{left}"));
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
        self.input.clear();
        self.cursor = 0;
        let _ = self.commands.send(AgentCommand::Prompt(text));
    }

    // ---- Slash-Befehle ----

    fn handle_command(&mut self, command: &str) {
        let mut parts = command.split_whitespace();
        let name = parts.next().unwrap_or("");
        let arg = parts.next();
        match name {
            "sessions" | "ls" | "list" => self.cmd_sessions(),
            "resume" | "open" => self.cmd_resume(arg),
            "new" => self.cmd_new(),
            "help" | "?" => self.cmd_help(),
            other => self.note(format!(
                "Unbekannter Befehl: /{other}. /help zeigt alle Befehle."
            )),
        }
    }

    fn cmd_sessions(&mut self) {
        let sessions = session::list();
        self.session_list = sessions.iter().map(|s| s.id.clone()).collect();

        if sessions.is_empty() {
            self.note("Noch keine gespeicherten Sitzungen.".to_string());
            return;
        }

        let mut text = String::from("Sitzungen (neueste zuerst):\n");
        for (index, meta) in sessions.iter().enumerate() {
            text.push_str(&format!(
                "  {:>2}  {}  ·  {}\n",
                index + 1,
                meta.title,
                relative_time(meta.modified),
            ));
        }
        text.push_str("\n/resume <n> zum Fortsetzen · /new für eine neue");
        self.note(text);
    }

    fn cmd_resume(&mut self, arg: Option<&str>) {
        let Some(number) = arg.and_then(|s| s.parse::<usize>().ok()) else {
            self.note("Nutzung: /resume <nummer> — erst /sessions ausführen.".to_string());
            return;
        };
        let Some(id) = self.session_list.get(number.wrapping_sub(1)).cloned() else {
            self.note("Keine Sitzung mit dieser Nummer. Erst /sessions ausführen.".to_string());
            return;
        };

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

    fn cmd_new(&mut self) {
        let _ = self.commands.send(AgentCommand::Reset);
        self.note("Neue Sitzung gestartet.".to_string());
    }

    fn cmd_help(&mut self) {
        self.note(
            "Befehle:\n  \
             /sessions      gespeicherte Sitzungen auflisten\n  \
             /resume <n>    Sitzung n fortsetzen\n  \
             /new           neue Sitzung beginnen\n  \
             /help          diese Hilfe"
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
    }

    /// Reagiert auf eine Meldung des Agent-Tasks.
    pub fn on_agent_event(&mut self, event: AgentEvent) {
        match event {
            AgentEvent::Chunk(text) => self.append_assistant(&text),
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
                self.scrollback.push(Message::new(Role::Assistant, question));
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

    /// Entnimmt die fertigen Blöcke, die in den Terminal-Scrollback sollen.
    pub fn drain_scrollback(&mut self) -> Vec<Message> {
        std::mem::take(&mut self.scrollback)
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

/// Wandelt einen geladenen Verlauf in Anzeige-Nachrichten für den Scrollback um.
/// Tool-Ergebnisse bleiben im Kontext, werden aber nicht erneut ausgebreitet.
fn history_to_display(history: &[ChatMessage]) -> Vec<Message> {
    let mut out = Vec::new();
    for message in history {
        match message {
            ChatMessage::User(text) => out.push(Message::new(Role::User, text.clone())),
            ChatMessage::Assistant { text, tool_calls } => {
                if !text.is_empty() {
                    out.push(Message::new(Role::Assistant, text.clone()));
                }
                for call in tool_calls {
                    let detail = call
                        .arguments
                        .get("command")
                        .or_else(|| call.arguments.get("path"))
                        .and_then(|v| v.as_str())
                        .unwrap_or("");
                    let title = if detail.is_empty() {
                        call.name.clone()
                    } else {
                        format!("{} · {}", call.name, detail)
                    };
                    out.push(Message::new(Role::Tool, title));
                }
            }
            ChatMessage::ToolResults(_) => {}
        }
    }
    out
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
        let mut app = App::new(commands, cancel, String::new());
        app.input = input.to_string();
        app.cursor = cursor;
        app
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
