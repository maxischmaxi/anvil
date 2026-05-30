//! Rendering für den Inline-Modus.
//!
//! Zwei Aufgaben:
//! - [`message_lines`] rendert einen fertigen Block in Zeilen, die der Haupt-Loop
//!   per `insert_before` in den echten Terminal-Scrollback schiebt (dort scrollt
//!   und kopiert tmux wie gewohnt).
//! - [`render_viewport`] zeichnet das kleine Fenster unten: eine Trennlinie (die
//!   während eines Turns den Spinner trägt) und die (mehrzeilige) Eingabe. Das
//!   Fenster ist bewusst kompakt, damit im Leerlauf keine Leerzeilen-Lücke
//!   zwischen Scrollback und Eingabe entsteht.

use ratatui::{
    Frame,
    layout::Position,
    style::{Color, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Padding, Paragraph},
};

use crate::app::{App, Message, Role, Status};

/// Breite des Eingabe-Indikators `"❯ "` bzw. der Einrückung `"  "` in Spalten.
const PROMPT_WIDTH: u16 = 2;

/// Rendert einen fertigen Block als Zeilen für den Scrollback: farbige
/// Rollen-Kopfzeile, darunter der (umgebrochene) Inhalt.
pub fn message_lines(message: &Message, width: u16) -> Vec<Line<'static>> {
    let width = (width as usize).max(1);
    let (label, style) = role_style(message.role);

    let mut lines: Vec<Line<'static>> = vec![Line::from(Span::styled(label.to_string(), style))];
    for raw_line in message.content.split('\n') {
        for wrapped in wrap(raw_line, width) {
            lines.push(Line::from(wrapped));
        }
    }
    lines
}

fn role_style(role: Role) -> (&'static str, Style) {
    match role {
        Role::User => ("you", Style::new().fg(Color::Cyan).bold()),
        Role::Assistant => ("anvil", Style::new().fg(Color::Green).bold()),
        Role::System => ("sys", Style::new().fg(Color::DarkGray)),
        Role::Tool => ("⚙", Style::new().fg(Color::Yellow)),
    }
}

/// Zeichnet den Viewport (das kompakte Fenster unten). `frame.area()` ist hier
/// der Inline-Viewport, nicht der ganze Bildschirm.
pub fn render_viewport(frame: &mut Frame, app: &App) {
    let area = frame.area();

    // Spinner/Hinweis sitzt im Titel der oberen Linie — so braucht es keine
    // eigene Zeile (und keine Leerzeile im Leerlauf).
    let title = match app.status {
        Status::Thinking => Span::styled(
            format!(" {} anvil arbeitet …  (Esc bricht ab)", app.spinner()),
            Style::new().fg(Color::Yellow),
        ),
        Status::AwaitingAnswer => Span::styled(
            " ⌨ Rückfrage — gib deine Antwort ein".to_string(),
            Style::new().fg(Color::Magenta),
        ),
        Status::Idle => Span::raw(String::new()),
    };

    let block = Block::default()
        .borders(Borders::TOP)
        .title(title)
        .padding(Padding::right(2));
    let inner = block.inner(area);
    let input_visible = (inner.height as usize).max(1);

    // Eingabe in logische Zeilen + Cursor-Position (Zeile/Spalte) bestimmen.
    let logical: Vec<&str> = app.input.split('\n').collect();
    let before = &app.input[..app.cursor];
    let cursor_row = before.matches('\n').count();
    let line_start = before.rfind('\n').map(|i| i + 1).unwrap_or(0);
    let cursor_col = app.input[line_start..app.cursor].chars().count();

    // So scrollen, dass die Cursorzeile im sichtbaren Fenster liegt.
    let max_scroll = logical.len().saturating_sub(input_visible);
    let scroll = cursor_row
        .saturating_sub(input_visible.saturating_sub(1))
        .min(max_scroll);

    let chevron_color = if app.status == Status::AwaitingAnswer {
        Color::Magenta
    } else {
        Color::Cyan
    };

    let mut lines: Vec<Line<'static>> = Vec::new();
    for (index, text) in logical.iter().enumerate().skip(scroll).take(input_visible) {
        let prefix = if index == 0 { "❯ " } else { "  " };
        lines.push(Line::from(vec![
            Span::styled(prefix, Style::new().fg(chevron_color)),
            Span::raw(text.to_string()),
        ]));
    }
    frame.render_widget(Paragraph::new(lines).block(block), area);

    // Cursor positionieren (hinter Indikator + getippte Zeichen, am Rand geklemmt).
    let display_row = (cursor_row - scroll) as u16;
    let cursor_x = (inner.x + PROMPT_WIDTH + cursor_col as u16)
        .min(inner.x + inner.width.saturating_sub(1));
    frame.set_cursor_position(Position::new(cursor_x, inner.y + display_row));
}

/// Greedy-Wortumbruch auf `width` Zeichen. Übermäßig lange Wörter werden hart
/// getrennt. (Zählt Zeichen, nicht Display-Breite — für CJK/Emoji später feiner.)
fn wrap(text: &str, width: usize) -> Vec<String> {
    if text.is_empty() {
        return vec![String::new()];
    }

    let mut lines = Vec::new();
    let mut current = String::new();
    let mut len = 0usize;

    for word in text.split_inclusive(' ') {
        let word_len = word.chars().count();

        if len + word_len > width && len > 0 {
            lines.push(std::mem::take(&mut current));
            len = 0;
        }

        if word_len > width {
            for c in word.chars() {
                if len == width {
                    lines.push(std::mem::take(&mut current));
                    len = 0;
                }
                current.push(c);
                len += 1;
            }
        } else {
            current.push_str(word);
            len += word_len;
        }
    }

    lines.push(current);
    lines
}
