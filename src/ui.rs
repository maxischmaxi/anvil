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

    if message.role == Role::Tool {
        return tool_message_lines(&message.content, width);
    }

    let (label, style) = role_style(message.role);

    let mut lines: Vec<Line<'static>> = vec![Line::from(Span::styled(label.to_string(), style))];
    for raw_line in message.content.split('\n') {
        for wrapped in wrap(raw_line, width) {
            lines.push(Line::from(wrapped));
        }
    }
    lines
}

fn tool_message_lines(content: &str, width: usize) -> Vec<Line<'static>> {
    let mut raw = content.lines();
    let head = raw.next().unwrap_or("").trim();
    let (name, detail) = head.split_once(" · ").unwrap_or((head, ""));
    let rest: Vec<&str> = raw.collect();

    let border = Style::new().fg(Color::DarkGray);
    let tool = Style::new().fg(Color::Yellow).bold();
    let detail_style = Style::new().fg(Color::Gray);

    let mut lines = vec![Line::from(vec![
        Span::styled("╭─ ", border),
        Span::styled("⚙ ", tool),
        Span::styled(name.to_string(), tool),
    ])];

    if !detail.is_empty() {
        for wrapped in wrap(detail, width.saturating_sub(4).max(1)) {
            lines.push(Line::from(vec![
                Span::styled("│  ", border),
                Span::styled(wrapped, detail_style),
            ]));
        }
    }

    if rest.iter().all(|line| line.trim().is_empty()) {
        lines.push(Line::from(vec![
            Span::styled("╰─ ", border),
            Span::styled("läuft …", Style::new().fg(Color::DarkGray)),
        ]));
        return lines;
    }

    let mut index = 0usize;
    while index < rest.len() {
        let line = rest[index];
        if line.trim_start() == "```diff" {
            let mut diff = Vec::new();
            index += 1;
            while index < rest.len() && rest[index].trim_start() != "```" {
                diff.push(rest[index]);
                index += 1;
            }
            lines.extend(diff_lines(&diff, width, border));
        } else if !line.trim().is_empty() {
            let style = if line.trim_start().starts_with('✗') {
                Style::new().fg(Color::Red)
            } else {
                Style::new().fg(Color::Green)
            };
            lines.push(Line::from(vec![
                Span::styled("│  ", border),
                Span::styled(line.trim().to_string(), style),
            ]));
        }
        index += 1;
    }

    lines.push(Line::from(Span::styled("╰", border)));
    lines
}

/// Ab dieser Gesamtbreite zeigen wir entfernt/hinzugefügt nebeneinander statt
/// untereinander. Darunter wären die beiden Spalten zu schmal für echten Code.
const SIDE_BY_SIDE_MIN_WIDTH: usize = 100;

// Diff-Zeilen bleiben hintergrund-transparent (Terminal-Hintergrund scheint
// durch); erkennbar sind sie nur an den farbigen `-`/`+`-Markern.
const MARK_REMOVED: Color = Color::Rgb(248, 81, 73);
const MARK_ADDED: Color = Color::Rgb(63, 185, 80);

// Syntax-Highlight-Palette (an GitHub-Dark angelehnt).
const HL_COMMENT: Color = Color::Rgb(139, 148, 158);
const HL_STRING: Color = Color::Rgb(165, 214, 255);
const HL_NUMBER: Color = Color::Rgb(121, 192, 255);
const HL_KEYWORD: Color = Color::Rgb(255, 166, 87);
const HL_TYPE: Color = Color::Rgb(210, 168, 255);
const HL_IDENT: Color = Color::Rgb(214, 222, 230);
const HL_PUNCT: Color = Color::Rgb(160, 170, 180);

/// Eine geparste Diff-Zeile. Die Reihenfolge im `Vec<DiffRow>` entspricht exakt
/// der Quell-Reihenfolge — `Context` vor und nach einem `Change` bleibt also an
/// der richtigen Stelle (anders als beim früheren „alle Kontexte zuerst").
enum DiffRow<'a> {
    Context(&'a str),
    /// Ein zusammenhängender Änderungsblock: erst die entfernten, dann die
    /// hinzugefügten Zeilen — so wie GitHub aufeinanderfolgende `-`/`+`-Zeilen
    /// zu einem Hunk paart.
    Change(Vec<&'a str>, Vec<&'a str>),
    /// Hinweiszeilen ohne Diff-Präfix, z. B. `… 39 more lines`.
    Note(&'a str),
}

fn diff_lines(diff: &[&str], width: usize, border: Style) -> Vec<Line<'static>> {
    let rows = parse_diff(diff);
    if width >= SIDE_BY_SIDE_MIN_WIDTH {
        side_by_side_diff(&rows, width, border)
    } else {
        stacked_diff(&rows, width, border)
    }
}

fn parse_diff<'a>(diff: &[&'a str]) -> Vec<DiffRow<'a>> {
    /// Hängt einen offenen Änderungsblock an, sobald wieder Kontext o. Ä. kommt.
    fn flush<'a>(rows: &mut Vec<DiffRow<'a>>, removed: &mut Vec<&'a str>, added: &mut Vec<&'a str>) {
        if !removed.is_empty() || !added.is_empty() {
            rows.push(DiffRow::Change(
                std::mem::take(removed),
                std::mem::take(added),
            ));
        }
    }

    let mut rows = Vec::new();
    let mut removed = Vec::new();
    let mut added = Vec::new();

    for line in diff {
        if let Some(rest) = line.strip_prefix("- ") {
            removed.push(rest);
        } else if let Some(rest) = line.strip_prefix("+ ") {
            added.push(rest);
        } else if let Some(rest) = line.strip_prefix("  ") {
            flush(&mut rows, &mut removed, &mut added);
            rows.push(DiffRow::Context(rest));
        } else {
            flush(&mut rows, &mut removed, &mut added);
            rows.push(DiffRow::Note(line));
        }
    }
    flush(&mut rows, &mut removed, &mut added);
    rows
}

/// Entfernt links, hinzugefügt rechts — pro Änderungsblock werden die Zeilen
/// paarweise gegenübergestellt (überzählige Zeilen einer Seite bekommen einen
/// leeren Gegenpart).
fn side_by_side_diff(rows: &[DiffRow], width: usize, border: Style) -> Vec<Line<'static>> {
    // 3 Spalten linker Rand + 3 Spalten Mitteltrenner ` │ ` = 6 Spalten Gerüst.
    let col_width = (width.saturating_sub(6) / 2).max(16);
    let code_width = col_width.saturating_sub(2);
    let mut lines = Vec::new();

    for row in rows {
        match row {
            DiffRow::Context(text) => {
                let mut spans = vec![Span::styled("│  ", border)];
                spans.extend(cell("  ", HL_PUNCT, text, code_width, true));
                spans.push(Span::styled(" │ ", border));
                spans.extend(cell("  ", HL_PUNCT, text, code_width, false));
                lines.push(Line::from(spans));
            }
            DiffRow::Change(removed, added) => {
                for i in 0..removed.len().max(added.len()) {
                    let mut spans = vec![Span::styled("│  ", border)];
                    // Linke Spalte auf Breite auffüllen, damit Trenner und
                    // rechte Spalte bündig sitzen (transparent, kein Hintergrund).
                    match removed.get(i) {
                        Some(text) => spans.extend(cell("- ", MARK_REMOVED, text, code_width, true)),
                        None => spans.extend(cell("  ", HL_PUNCT, "", code_width, true)),
                    }
                    spans.push(Span::styled(" │ ", border));
                    if let Some(text) = added.get(i) {
                        spans.extend(cell("+ ", MARK_ADDED, text, code_width, false));
                    }
                    lines.push(Line::from(spans));
                }
            }
            DiffRow::Note(note) => lines.push(note_line(note, border)),
        }
    }
    lines
}

/// Entfernt und hinzugefügt untereinander (für schmale Terminals).
fn stacked_diff(rows: &[DiffRow], width: usize, border: Style) -> Vec<Line<'static>> {
    // 3 Spalten linker Rand + 2 Spalten Marker.
    let code_width = width.saturating_sub(5).max(16);
    let mut lines = Vec::new();

    for row in rows {
        match row {
            DiffRow::Context(text) => {
                let mut spans = vec![Span::styled("│  ", border)];
                spans.extend(cell("  ", HL_PUNCT, text, code_width, false));
                lines.push(Line::from(spans));
            }
            DiffRow::Change(removed, added) => {
                for text in removed {
                    let mut spans = vec![Span::styled("│  ", border)];
                    spans.extend(cell("- ", MARK_REMOVED, text, code_width, false));
                    lines.push(Line::from(spans));
                }
                for text in added {
                    let mut spans = vec![Span::styled("│  ", border)];
                    spans.extend(cell("+ ", MARK_ADDED, text, code_width, false));
                    lines.push(Line::from(spans));
                }
            }
            DiffRow::Note(note) => lines.push(note_line(note, border)),
        }
    }
    lines
}

fn note_line(note: &str, border: Style) -> Line<'static> {
    Line::from(vec![
        Span::styled("│  ", border),
        Span::styled(
            note.trim().to_string(),
            Style::new().fg(Color::DarkGray).italic(),
        ),
    ])
}

/// Baut eine Diff-Zelle: farbiger Marker (`- `/`+ `/`  `) plus syntaxgefärbter,
/// auf `code_width` getrimmter Code. Bei `pad` wird die Zelle mit transparenten
/// Leerzeichen bis zum Spaltenrand aufgefüllt (nur zur Ausrichtung).
fn cell(
    marker: &str,
    mark_color: Color,
    code: &str,
    code_width: usize,
    pad: bool,
) -> Vec<Span<'static>> {
    let mut spans = vec![Span::styled(marker.to_string(), Style::new().fg(mark_color))];
    spans.extend(highlight(code, code_width, pad));
    spans
}

/// Tokenisiert `code` und trimmt auf `width` Spalten. Bei `pad` wird mit
/// transparenten Leerzeichen bis zum Spaltenrand aufgefüllt.
fn highlight(code: &str, width: usize, pad: bool) -> Vec<Span<'static>> {
    let shown = truncate_cols(code, width);
    let used = shown.chars().count();
    let mut spans = tokenize(&shown);
    if pad && used < width {
        spans.push(Span::raw(" ".repeat(width - used)));
    }
    spans
}

/// Sehr leichter, sprach-agnostischer Tokenizer ohne tree-sitter: erkennt
/// Zeilenkommentare, String-Literale, Zahlen, Schlüsselwörter und (groß
/// beginnende) Typnamen. Für ein PR-artiges Highlighting reicht das.
fn tokenize(line: &str) -> Vec<Span<'static>> {
    let chars: Vec<char> = line.chars().collect();
    let n = chars.len();
    let mut spans = Vec::new();
    let mut i = 0;

    while i < n {
        let c = chars[i];

        // Zeilenkommentar `//` …
        if c == '/' && i + 1 < n && chars[i + 1] == '/' {
            spans.push(tok_span(chars[i..].iter().collect(), HL_COMMENT));
            break;
        }
        // `#`-Kommentar (Shell/Python/TOML), aber nicht Rust-Attribute `#[…]`/`#!`.
        if c == '#' && !(i + 1 < n && (chars[i + 1] == '[' || chars[i + 1] == '!')) {
            spans.push(tok_span(chars[i..].iter().collect(), HL_COMMENT));
            break;
        }
        // String-Literal, einfache oder doppelte Anführungszeichen, mit Escapes.
        if c == '"' || c == '\'' {
            let mut j = i + 1;
            let mut escaped = false;
            while j < n {
                let cj = chars[j];
                if escaped {
                    escaped = false;
                } else if cj == '\\' {
                    escaped = true;
                } else if cj == c {
                    j += 1;
                    break;
                }
                j += 1;
            }
            spans.push(tok_span(chars[i..j].iter().collect(), HL_STRING));
            i = j;
            continue;
        }
        // Zahl.
        if c.is_ascii_digit() {
            let mut j = i;
            while j < n && (chars[j].is_ascii_alphanumeric() || chars[j] == '.' || chars[j] == '_') {
                j += 1;
            }
            spans.push(tok_span(chars[i..j].iter().collect(), HL_NUMBER));
            i = j;
            continue;
        }
        // Bezeichner / Schlüsselwort / Typ.
        if c.is_alphabetic() || c == '_' {
            let mut j = i;
            while j < n && (chars[j].is_alphanumeric() || chars[j] == '_') {
                j += 1;
            }
            let word: String = chars[i..j].iter().collect();
            let color = if is_keyword(&word) {
                HL_KEYWORD
            } else if c.is_uppercase() {
                HL_TYPE
            } else {
                HL_IDENT
            };
            spans.push(tok_span(word, color));
            i = j;
            continue;
        }
        // Interpunktion / Whitespace bis zum nächsten Token-Anfang sammeln.
        let start = i;
        i += 1;
        while i < n {
            let cj = chars[i];
            if cj.is_alphanumeric() || cj == '_' || cj == '"' || cj == '\'' {
                break;
            }
            if cj == '/' && i + 1 < n && chars[i + 1] == '/' {
                break;
            }
            if cj == '#' && !(i + 1 < n && (chars[i + 1] == '[' || chars[i + 1] == '!')) {
                break;
            }
            i += 1;
        }
        spans.push(tok_span(chars[start..i].iter().collect(), HL_PUNCT));
    }

    spans
}

fn tok_span(text: String, color: Color) -> Span<'static> {
    Span::styled(text, Style::new().fg(color))
}

fn is_keyword(word: &str) -> bool {
    matches!(
        word,
        // Rust
        "fn" | "let" | "mut" | "const" | "static" | "pub" | "use" | "mod" | "crate"
            | "struct" | "enum" | "trait" | "impl" | "type" | "dyn" | "where" | "as" | "ref"
            | "if" | "else" | "match" | "for" | "while" | "loop" | "break" | "continue"
            | "return" | "move" | "async" | "await" | "in" | "self" | "Self" | "super"
            | "unsafe" | "extern" | "true" | "false"
            // Andere Sprachen
            | "function" | "var" | "def" | "class" | "import" | "from" | "export"
            | "default" | "public" | "private" | "protected" | "void" | "new" | "this"
            | "null" | "undefined" | "and" | "or" | "not" | "is" | "lambda" | "with"
            | "try" | "catch" | "finally" | "throw" | "yield" | "then"
    )
}

fn truncate_cols(text: &str, width: usize) -> String {
    if text.chars().count() <= width {
        text.to_string()
    } else {
        let kept: String = text.chars().take(width.saturating_sub(1)).collect();
        format!("{kept}…")
    }
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
    // eigene Zeile (und keine Leerzeile im Leerlauf). Die maskierte Key-Eingabe
    // hat Vorrang vor dem Status.
    let title = if app.masking() {
        Span::styled(
            " 🔑 API-Key eingeben — Enter speichert, Esc bricht ab".to_string(),
            Style::new().fg(Color::Magenta),
        )
    } else {
        match app.status {
            Status::Thinking => Span::styled(
                format!(" {} anvil arbeitet …  (Esc bricht ab)", app.spinner()),
                Style::new().fg(Color::Yellow),
            ),
            Status::AwaitingAnswer => Span::styled(
                " ⌨ Rückfrage — gib deine Antwort ein".to_string(),
                Style::new().fg(Color::Magenta),
            ),
            // Im Leerlauf das aktive Modell dezent anzeigen (wie opencodes Status).
            Status::Idle => match app.active_label() {
                Some(label) => Span::styled(format!(" {label}"), Style::new().fg(Color::DarkGray)),
                None => Span::raw(String::new()),
            },
        }
    };

    let block = Block::default()
        .borders(Borders::TOP)
        .title(title)
        .padding(Padding::right(2));
    let inner = block.inner(area);
    let input_visible = (inner.height as usize).max(1);

    // Eingabe visuell umbrechen + Cursor-Position (Zeile/Spalte) bestimmen. Bei
    // maskierter Key-Eingabe (/login) `*` statt der echten Zeichen zeigen —
    // API-Keys sind ASCII, daher bleiben die Byte-Offsets des Cursors gültig.
    let input_width = inner.width.saturating_sub(PROMPT_WIDTH).max(1) as usize;
    let masked = app.masking().then(|| "*".repeat(app.input.len()));
    let input_for_display = masked.as_deref().unwrap_or(&app.input);
    let (visual_lines, cursor_row, cursor_col) =
        wrap_input(input_for_display, app.cursor, input_width);

    // So scrollen, dass die Cursorzeile im sichtbaren Fenster liegt.
    let max_scroll = visual_lines.len().saturating_sub(input_visible);
    let scroll = cursor_row
        .saturating_sub(input_visible.saturating_sub(1))
        .min(max_scroll);

    let chevron_color = if app.status == Status::AwaitingAnswer {
        Color::Magenta
    } else {
        Color::Cyan
    };

    let mut lines: Vec<Line<'static>> = Vec::new();
    for (index, text) in visual_lines
        .iter()
        .enumerate()
        .skip(scroll)
        .take(input_visible)
    {
        let prefix = if index == 0 { "❯ " } else { "  " };
        lines.push(Line::from(vec![
            Span::styled(prefix, Style::new().fg(chevron_color)),
            Span::raw(text.clone()),
        ]));
    }
    frame.render_widget(Paragraph::new(lines).block(block), area);

    // Cursor positionieren (hinter Indikator + getippte Zeichen, am Rand geklemmt).
    let display_row = (cursor_row - scroll) as u16;
    let cursor_x =
        (inner.x + PROMPT_WIDTH + cursor_col as u16).min(inner.x + inner.width.saturating_sub(1));
    frame.set_cursor_position(Position::new(cursor_x, inner.y + display_row));
}

/// Bricht die Prompt-Eingabe hart auf verfügbare Spalten um und liefert dazu
/// die visuelle Cursor-Position. Zählt Zeichen statt Display-Breite (wie der
/// bestehende Scrollback-Wrapper); für normale Code-/Prompt-Texte reicht das.
fn wrap_input(input: &str, cursor: usize, width: usize) -> (Vec<String>, usize, usize) {
    let width = width.max(1);
    let mut lines = vec![String::new()];
    let mut row = 0usize;
    let mut col = 0usize;
    let mut cursor_pos = None;

    for (byte_index, c) in input.char_indices() {
        if byte_index == cursor {
            cursor_pos = Some((row, col));
        }

        if c == '\n' {
            lines.push(String::new());
            row += 1;
            col = 0;
            continue;
        }

        if col == width {
            lines.push(String::new());
            row += 1;
            col = 0;
        }

        lines[row].push(c);
        col += 1;
    }

    let (cursor_row, cursor_col) = cursor_pos.unwrap_or((row, col));
    (lines, cursor_row, cursor_col)
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

#[cfg(test)]
mod tests {
    use super::*;

    /// Klartext einer Zeile (alle Span-Inhalte aneinandergehängt).
    fn plain(line: &Line<'_>) -> String {
        line.spans.iter().map(|s| s.content.as_ref()).collect()
    }

    #[test]
    fn parse_diff_preserves_order_of_context_around_change() {
        let diff = ["  fn foo() {", "-     old();", "+     new();", "  }"];
        let rows = parse_diff(&diff);
        assert_eq!(rows.len(), 3, "Kontext-vorher, Change, Kontext-nachher");
        assert!(matches!(rows[0], DiffRow::Context("fn foo() {")));
        match &rows[1] {
            DiffRow::Change(removed, added) => {
                assert_eq!(removed, &["    old();"]);
                assert_eq!(added, &["    new();"]);
            }
            _ => panic!("erwartete einen Änderungsblock"),
        }
        assert!(matches!(rows[2], DiffRow::Context("}")));
    }

    #[test]
    fn stacked_diff_keeps_trailing_context_below_the_change() {
        // Regressionsschutz: früher landete *aller* Kontext oben, sodass die
        // schließende Klammer fälschlich über der Änderung stand.
        let diff = ["  fn foo() {", "-     old();", "+     new();", "  }"];
        let border = Style::new();
        let rendered: Vec<String> = stacked_diff(&parse_diff(&diff), 40, border)
            .iter()
            .map(plain)
            .collect();
        let order: Vec<&str> = rendered
            .iter()
            .map(|l| l.trim_start_matches('│').trim())
            .collect();
        assert_eq!(order[0], "fn foo() {");
        assert!(order[1].starts_with("- "));
        assert!(order[2].starts_with("+ "));
        assert!(order[3].ends_with('}'), "Kontext-nachher gehört nach unten");
    }

    #[test]
    fn note_lines_survive_parsing() {
        let diff = ["  ctx", "- gone", "… 12 more lines"];
        let rows = parse_diff(&diff);
        assert!(matches!(rows.last(), Some(DiffRow::Note("… 12 more lines"))));
    }

    #[test]
    fn highlight_pads_cell_to_full_width() {
        let spans = highlight("let x", 10, true);
        let text: String = spans.iter().map(|s| s.content.as_ref()).collect();
        assert_eq!(text.chars().count(), 10, "Zelle wird auf die Breite gefüllt");
        assert!(text.starts_with("let x"));
    }

    #[test]
    fn highlight_leaves_cell_transparent_without_padding() {
        let spans = highlight("let x", 10, false);
        let text: String = spans.iter().map(|s| s.content.as_ref()).collect();
        assert_eq!(text, "let x", "ohne pad keine angehängten Leerzeichen");
        assert!(
            spans.iter().all(|s| s.style.bg.is_none()),
            "Diff-Zellen bleiben hintergrund-transparent"
        );
    }

    #[test]
    fn tokenize_is_lossless() {
        let line = r#"let s = "hi"; // note"#;
        let spans = tokenize(line);
        let rebuilt: String = spans.iter().map(|s| s.content.as_ref()).collect();
        assert_eq!(rebuilt, line, "Tokenizer darf keine Zeichen verlieren");
    }

    #[test]
    fn rust_attribute_is_not_a_comment() {
        // `#[derive]` darf nicht als `#`-Kommentar geschluckt werden.
        let spans = tokenize("#[derive(Clone)]");
        let rebuilt: String = spans.iter().map(|s| s.content.as_ref()).collect();
        assert_eq!(rebuilt, "#[derive(Clone)]");
        assert!(
            spans.iter().all(|s| s.style.fg != Some(HL_COMMENT)),
            "Attribut sollte nicht als Kommentar gefärbt sein"
        );
    }
}
