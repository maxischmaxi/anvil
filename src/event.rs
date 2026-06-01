//! Nachrichten-Typen für die Kommunikation zwischen UI-Loop und Agent-Task.
//!
//! Die einzige Schnittstelle zwischen den beiden Welten: Die UI schickt
//! [`AgentCommand`]s rein, der Agent meldet [`AgentEvent`]s zurück.

use crate::llm::{ChatMessage, ProviderKind, Secret};

/// Was die UI dem Agent-Task aufträgt.
#[derive(Debug, Clone)]
pub enum AgentCommand {
    /// Der Nutzer hat einen Prompt abgeschickt.
    Prompt(String),
    /// Eine gespeicherte Sitzung fortsetzen: Verlauf als Kontext übernehmen und
    /// künftige Nachrichten an diese Datei anhängen.
    SetContext {
        history: Vec<ChatMessage>,
        id: String,
    },
    /// Provider/Modell zur Laufzeit wechseln (über `/models` bzw. nach `/login`).
    /// Der Gesprächskontext bleibt erhalten — nur der Client wird getauscht.
    SetClient {
        kind: ProviderKind,
        model: String,
        secret: Secret,
    },
    /// Eine frische Sitzung beginnen (Kontext leeren).
    Reset,
}

/// Was der Agent-Task an die UI zurückmeldet, während er arbeitet.
#[derive(Debug, Clone)]
pub enum AgentEvent {
    /// Ein Stück gestreamter Assistenten-Text.
    Chunk(String),
    /// Ein Tool wird ausgeführt (z. B. `"bash · ls -la"`).
    ToolStarted(String),
    /// Das zuletzt gestartete Tool ist fertig.
    ToolFinished { ok: bool, summary: String },
    /// Das Modell stellt eine Rückfrage und wartet auf die Antwort des Nutzers.
    AskUser(String),
    /// Die Antwort ist vollständig (keine weiteren Tool-Aufrufe).
    Done,
    /// Der laufende Turn wurde vom Nutzer abgebrochen.
    Cancelled,
    /// Es ist ein Fehler aufgetreten.
    Error(String),
}
