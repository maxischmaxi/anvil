//! anvil — ein Coding-Agent im Terminal.
//!
//! Einstiegspunkt und Haupt-Event-Loop. anvil rendert **inline**: nur ein kleines
//! Fenster unten gehört der App (Eingabe, Spinner, Live-Vorschau des laufenden
//! Turns). Fertige Blöcke wandern per `insert_before` in den echten
//! Terminal-Scrollback — dort scrollt und kopiert tmux wie gewohnt.

mod agent;
mod app;
mod config;
mod event;
mod llm;
mod session;
mod tools;
mod ui;

use std::io::Stdout;
use std::time::Duration;

use anyhow::Result;
use futures::StreamExt;
use ratatui::backend::CrosstermBackend;
use ratatui::crossterm::event::{Event as TerminalEvent, EventStream};
use ratatui::crossterm::terminal::{disable_raw_mode, enable_raw_mode, size};
use ratatui::widgets::{Paragraph, Widget};
use ratatui::{Terminal, TerminalOptions, Viewport};
use tokio::sync::mpsc;

use crate::app::App;
use crate::event::{AgentCommand, AgentEvent};

type Tui = Terminal<CrosstermBackend<Stdout>>;

#[tokio::main]
async fn main() -> Result<()> {
    // Provider aus den Umgebungsvariablen bestimmen. Fehlt der Key, läuft die App
    // trotzdem und zeigt den Hinweis als erste Scrollback-Zeile.
    let (client, intro) = match config::load() {
        Ok(client) => {
            let intro = format!(
                "anvil bereit · {} · Enter sendet, Alt+Enter neue Zeile, Strg+C beendet.",
                client.describe()
            );
            (Some(client), intro)
        }
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
    // Kompaktes Fenster: obere Linie + eine Eingabezeile. So gibt es im Leerlauf
    // keine Leerzeilen-Lücke; mehrzeilige Eingabe scrollt intern. (Die
    // Inline-Höhe ist nach init fix — größer = mehr sichtbare Eingabezeilen,
    // aber wieder Luft im Leerlauf.)
    let rows = size().map(|(_, r)| r).unwrap_or(24);
    let viewport_height = rows.clamp(1, 2);
    let backend = CrosstermBackend::new(std::io::stdout());
    let terminal = Terminal::with_options(
        backend,
        TerminalOptions {
            viewport: Viewport::Inline(viewport_height),
        },
    )?;
    Ok(terminal)
}

fn restore_terminal(terminal: &mut Tui) {
    let _ = terminal.clear();
    let _ = disable_raw_mode();
}

async fn run(terminal: &mut Tui, client: Option<llm::LlmClient>, intro: String) -> Result<()> {
    // Kanäle: UI -> Agent (Commands), UI -> Agent (Abbruch) und Agent -> UI (Events).
    let (command_tx, command_rx) = mpsc::unbounded_channel::<AgentCommand>();
    let (cancel_tx, cancel_rx) = mpsc::unbounded_channel::<()>();
    let (event_tx, mut event_rx) = mpsc::unbounded_channel::<AgentEvent>();

    tokio::spawn(agent::run(client, command_rx, cancel_rx, event_tx));

    let mut app = App::new(command_tx, cancel_tx, intro);
    let mut terminal_events = EventStream::new();
    // Periodischer Tick, damit der Spinner animiert (und der Viewport frisch bleibt).
    let mut ticker = tokio::time::interval(Duration::from_millis(120));

    // Viewport einmal etablieren, bevor wir Zeilen davor einfügen.
    terminal.draw(|frame| ui::render_viewport(frame, &app))?;

    while !app.should_quit {
        flush_scrollback(terminal, &mut app)?;
        terminal.draw(|frame| ui::render_viewport(frame, &app))?;

        tokio::select! {
            _ = ticker.tick() => app.tick(),
            maybe_terminal = terminal_events.next() => match maybe_terminal {
                Some(Ok(TerminalEvent::Key(key))) => app.on_key(key),
                Some(Ok(_)) => {} // Resize/Maus/Paste: nächste Schleife zeichnet neu
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

/// Schreibt alle fertigen Blöcke per `insert_before` in den Terminal-Scrollback.
fn flush_scrollback(terminal: &mut Tui, app: &mut App) -> Result<()> {
    let (width, _) = size().unwrap_or((80, 24));
    for message in app.drain_scrollback() {
        let mut lines = ui::message_lines(&message, width);
        lines.push(ratatui::text::Line::from(String::new())); // Leerzeile zwischen Blöcken
        let height = lines.len() as u16;
        terminal.insert_before(height, move |buf| {
            Paragraph::new(lines).render(buf.area, buf);
        })?;
    }
    Ok(())
}
