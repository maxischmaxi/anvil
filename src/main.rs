//! anvil — ein Coding-Agent im Terminal.
//!
//! Einstiegspunkt und Haupt-Event-Loop. anvil rendert **inline**: nur ein kleines
//! Fenster unten gehört der App (Eingabe, Spinner, Live-Vorschau des laufenden
//! Turns). Fertige Blöcke wandern per `insert_before` in den echten
//! Terminal-Scrollback — dort scrollt und kopiert tmux wie gewohnt.

mod agent;
mod app;
mod auth;
mod config;
mod event;
mod llm;
mod oauth;
mod openai_subscription;
mod session;
mod tokens;
mod tools;
mod ui;

use std::io::Stdout;
use std::time::Duration;

use anyhow::Result;
use futures::StreamExt;
use ratatui::backend::{Backend, CrosstermBackend};
use ratatui::crossterm::cursor::MoveTo;
use ratatui::crossterm::event::{
    DisableBracketedPaste, EnableBracketedPaste, Event as TerminalEvent, EventStream,
};
use ratatui::crossterm::execute;
use ratatui::crossterm::style::Print;
use ratatui::crossterm::terminal::{Clear, ClearType, disable_raw_mode, enable_raw_mode, size};
use ratatui::widgets::{Paragraph, Widget};
use ratatui::{Terminal, TerminalOptions, Viewport, layout::Rect};
use tokio::sync::mpsc;

use crate::app::{ActiveModel, App};
use crate::event::{AgentCommand, AgentEvent};

type Tui = Terminal<CrosstermBackend<Stdout>>;

#[tokio::main]
async fn main() -> Result<()> {
    // Provider aus den Umgebungsvariablen bestimmen. Fehlt der Key, läuft die App
    // trotzdem und zeigt den Hinweis als erste Scrollback-Zeile.
    let (client, intro) = match config::load() {
        Ok(client) => (Some(client), String::new()),
        Err(reason) => (None, reason),
    };

    let mut terminal = setup_terminal()?;
    let result = run(&mut terminal, client, intro).await;
    restore_terminal(&mut terminal);
    result
}

/// Raw-Mode + Inline-Viewport einrichten, plus Panic-Hook fürs Aufräumen.
fn setup_terminal() -> Result<Tui> {
    let previous_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        let _ = disable_raw_mode();
        previous_hook(info);
    }));

    enable_raw_mode()?;
    let viewport_height = initial_viewport_height();
    execute!(
        std::io::stdout(),
        EnableBracketedPaste,
        Clear(ClearType::All),
        Clear(ClearType::Purge)
    )?;
    build_inline_terminal(viewport_height)
}

fn restore_terminal(terminal: &mut Tui) {
    let _ = terminal.clear();
    let _ = execute!(terminal.backend_mut(), DisableBracketedPaste);
    let _ = Backend::flush(terminal.backend_mut());
    let _ = disable_raw_mode();
}

fn initial_viewport_height() -> u16 {
    let rows = size().map(|(_, r)| r).unwrap_or(24);
    rows.min(2)
}

fn build_inline_terminal(viewport_height: u16) -> Result<Tui> {
    let rows = size().map(|(_, r)| r).unwrap_or(24);
    let viewport_height = rows.min(viewport_height.max(1));
    execute!(
        std::io::stdout(),
        MoveTo(0, rows.saturating_sub(viewport_height))
    )?;
    let backend = CrosstermBackend::new(std::io::stdout());
    Ok(Terminal::with_options(
        backend,
        TerminalOptions {
            viewport: Viewport::Inline(viewport_height),
        },
    )?)
}

fn desired_viewport_height(app: &App) -> u16 {
    let (cols, rows) = size().unwrap_or((80, 24));
    let input_width = cols.saturating_sub(2).max(1) as usize;
    let input_lines = visual_input_line_count(app, input_width) as u16;
    let suggestions = if app.resume_picker_open() {
        0
    } else {
        app.command_suggestions()
            .len()
            .min(5)
            .min(rows.saturating_sub(2) as usize) as u16
    };
    let min_height = 2 + suggestions;
    let max_height = rows.max(1);
    (1 + suggestions + input_lines).clamp(min_height, max_height)
}

fn visual_input_line_count(app: &App, width: usize) -> usize {
    let width = width.max(1);
    let mut lines = 1usize;
    let mut col = 0usize;

    if app.masking() {
        for _ in app.input.chars() {
            if col == width {
                lines += 1;
                col = 0;
            }
            col += 1;
        }
        return lines;
    }

    for c in app.input.chars() {
        if c == '\n' {
            lines += 1;
            col = 0;
        } else {
            if col == width {
                lines += 1;
                col = 0;
            }
            col += 1;
        }
    }
    lines
}

fn rebuild_terminal_for_viewport(terminal: &mut Tui, height: u16) -> Result<()> {
    execute!(
        terminal.backend_mut(),
        Clear(ClearType::All),
        Clear(ClearType::Purge)
    )?;
    *terminal = build_inline_terminal(height)?;
    Ok(())
}

async fn run(terminal: &mut Tui, client: Option<llm::LlmClient>, intro: String) -> Result<()> {
    // Kanäle: UI -> Agent (Commands), UI -> Agent (Abbruch) und Agent -> UI (Events).
    let (command_tx, command_rx) = mpsc::unbounded_channel::<AgentCommand>();
    let (cancel_tx, cancel_rx) = mpsc::unbounded_channel::<()>();
    let (event_tx, mut event_rx) = mpsc::unbounded_channel::<AgentEvent>();

    // Aktiven Provider/Modell festhalten, bevor der Client in den Task wandert,
    // damit /models das gerade laufende Modell markieren kann.
    let active = client.as_ref().map(|c| {
        ActiveModel::new(c.kind(), c.model().to_string(), c.auth_mode())
    });
    tokio::spawn(agent::run(client, command_rx, cancel_rx, event_tx));

    let mut app = App::new(command_tx, cancel_tx, intro, active);
    let mut picker_was_open = false;
    let mut viewport_height = initial_viewport_height();
    let mut terminal_events = EventStream::new();
    // Periodischer Tick, damit der Spinner animiert (und der Viewport frisch bleibt).
    let mut ticker = tokio::time::interval(Duration::from_millis(120));

    // Viewport einmal etablieren, bevor wir Zeilen davor einfügen.
    terminal.draw(|frame| ui::render_viewport(frame, &app))?;

    while !app.should_quit {
        let picker_open = app.resume_picker_open();
        if picker_open && !picker_was_open {
            clear_terminal(terminal)?;
        } else if !picker_open && picker_was_open {
            app.request_reflow();
        }
        picker_was_open = picker_open;

        if app.take_clear_requested() {
            clear_terminal(terminal)?;
        }
        if let Some(messages) = app.take_reflow_messages() {
            reflow_terminal(terminal, messages)?;
        }

        let wanted_height = desired_viewport_height(&app);
        if wanted_height != viewport_height {
            viewport_height = wanted_height;
            rebuild_terminal_for_viewport(terminal, viewport_height)?;
        }

        if app.resume_picker_open() {
            render_picker_screen(terminal, &app)?;
            terminal.draw(|frame| ui::render_viewport(frame, &app))?;
        } else {
            flush_scrollback(terminal, &mut app)?;
            terminal.draw(|frame| ui::render_viewport(frame, &app))?;
        }

        tokio::select! {
            _ = ticker.tick() => app.tick(),
            maybe_terminal = terminal_events.next() => match maybe_terminal {
                Some(Ok(TerminalEvent::Key(key))) => app.on_key(key),
                Some(Ok(TerminalEvent::Paste(text))) => app.on_paste(&text),
                Some(Ok(TerminalEvent::Resize(width, height))) => resize_terminal(terminal, &mut app, width, height)?,
                Some(Ok(_)) => {} // Maus/Fokus: nächste Schleife zeichnet neu
                Some(Err(_)) | None => app.should_quit = true,
            },
            maybe_event = event_rx.recv() => {
                if let Some(event) = maybe_event {
                    app.on_agent_event(event);
                }
            }
        }
    }

    // Verbleibende fertige Blöcke noch in den Scrollback schreiben.
    flush_scrollback(terminal, &mut app)?;
    Ok(())
}

fn render_picker_screen(terminal: &mut Tui, app: &App) -> Result<()> {
    let (width, height) = size()?;
    let lines = ui::picker_lines(app, width, height.saturating_sub(2));
    execute!(
        terminal.backend_mut(),
        MoveTo(0, 0),
        Clear(ClearType::All),
        Clear(ClearType::Purge)
    )?;
    terminal.clear()?;
    for (index, line) in lines.into_iter().enumerate() {
        execute!(
            terminal.backend_mut(),
            MoveTo(0, index as u16),
            Print(line)
        )?;
    }
    Ok(())
}

fn clear_terminal(terminal: &mut Tui) -> Result<()> {
    execute!(
        terminal.backend_mut(),
        Clear(ClearType::All),
        Clear(ClearType::Purge),
        MoveTo(0, 0)
    )?;
    terminal.clear()?;
    Ok(())
}

fn resize_terminal(terminal: &mut Tui, app: &mut App, width: u16, height: u16) -> Result<()> {
    terminal.resize(Rect::new(0, 0, width, height))?;
    app.request_reflow();
    Ok(())
}

fn reflow_terminal(terminal: &mut Tui, messages: Vec<crate::app::Message>) -> Result<()> {
    clear_terminal(terminal)?;
    insert_messages(terminal, messages)?;
    Ok(())
}

/// Schreibt alle fertigen Blöcke per `insert_before` in den Terminal-Scrollback.
fn flush_scrollback(terminal: &mut Tui, app: &mut App) -> Result<()> {
    let messages = app.drain_scrollback();
    insert_messages(terminal, messages)
}

fn insert_messages(terminal: &mut Tui, messages: Vec<crate::app::Message>) -> Result<()> {
    let width = terminal.backend().size()?.width;
    for message in messages {
        let mut lines = ui::message_lines(&message, width);
        lines.push(ratatui::text::Line::from(String::new())); // Leerzeile zwischen Blöcken
        let height = lines.len() as u16;
        terminal.insert_before(height, move |buf| {
            Paragraph::new(lines).render(buf.area, buf);
        })?;
    }
    Ok(())
}
