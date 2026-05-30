//! Persistenz von Sitzungen als JSONL-Dateien.
//!
//! Eine Sitzung ist eine Datei `<id>.jsonl` unter `~/.config/anvil/sessions/`:
//! - Zeile 1: ein [`Header`] (`{v, id, title, created_ms}`), einmal beim Anlegen.
//! - danach: pro Zeile eine [`ChatMessage`] als JSON, append-only.
//!
//! Append-only heißt: jede abgeschlossene Runde wird sofort drangehängt
//! (crash-sicher), ohne die Datei je neu zu schreiben. Gespeichert wird pro
//! abgeschlossenem Turn — abgebrochene Turns landen nicht in der Datei.

use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

use crate::llm::ChatMessage;

const VERSION: u32 = 1;
const TITLE_MAX_CHARS: usize = 80;

/// Die Identität einer Sitzung — die Erstellungszeit in Millisekunden, als String
/// (gleichzeitig der Dateiname-Stamm). Sortierbar = chronologisch.
pub type SessionId = String;

/// Kopfzeile einer Sitzungsdatei.
#[derive(Debug, Serialize, Deserialize)]
struct Header {
    v: u32,
    id: SessionId,
    title: String,
    created_ms: u64,
}

/// Zusammenfassung einer Sitzung für die `/sessions`-Liste.
#[derive(Debug, Clone)]
pub struct SessionMeta {
    pub id: SessionId,
    pub title: String,
    pub modified: SystemTime,
}

/// Schreibt Nachrichten append-only an eine Sitzungsdatei.
pub struct SessionWriter {
    file: File,
}

impl SessionWriter {
    /// Legt eine neue Sitzung im Standardverzeichnis an.
    pub fn create(title: &str) -> Result<Self> {
        Self::create_in(&sessions_dir()?, title)
    }

    /// Öffnet eine bestehende Sitzung im Standardverzeichnis zum Anhängen.
    pub fn open(id: &str) -> Result<Self> {
        Self::open_in(&sessions_dir()?, id)
    }

    fn create_in(dir: &Path, title: &str) -> Result<Self> {
        std::fs::create_dir_all(dir).with_context(|| format!("Verzeichnis {dir:?} anlegen"))?;
        let id = new_id();
        let path = dir.join(format!("{id}.jsonl"));
        let mut file = OpenOptions::new()
            .create(true)
            .truncate(true)
            .write(true)
            .open(&path)
            .with_context(|| format!("Sitzungsdatei {path:?} anlegen"))?;
        let header = Header {
            v: VERSION,
            id: id.clone(),
            title: clean_title(title),
            created_ms: id.parse().unwrap_or(0),
        };
        writeln!(file, "{}", serde_json::to_string(&header)?)?;
        file.flush()?;
        Ok(Self { file })
    }

    fn open_in(dir: &Path, id: &str) -> Result<Self> {
        let path = dir.join(format!("{id}.jsonl"));
        let file = OpenOptions::new()
            .append(true)
            .open(&path)
            .with_context(|| format!("Sitzungsdatei {path:?} öffnen"))?;
        Ok(Self { file })
    }

    /// Hängt eine Nachricht als JSON-Zeile an und flusht sofort (crash-sicher).
    pub fn append(&mut self, message: &ChatMessage) -> Result<()> {
        let line = serde_json::to_string(message)?;
        writeln!(self.file, "{line}")?;
        self.file.flush()?;
        Ok(())
    }
}

/// Lädt den Nachrichtenverlauf einer Sitzung aus dem Standardverzeichnis.
pub fn load(id: &str) -> Result<Vec<ChatMessage>> {
    load_from(&sessions_dir()?, id)
}

/// Listet alle Sitzungen im Standardverzeichnis (neueste zuerst).
pub fn list() -> Vec<SessionMeta> {
    sessions_dir().map(|dir| list_in(&dir)).unwrap_or_default()
}

fn load_from(dir: &Path, id: &str) -> Result<Vec<ChatMessage>> {
    let path = dir.join(format!("{id}.jsonl"));
    let file = File::open(&path).with_context(|| format!("Sitzungsdatei {path:?} öffnen"))?;
    let reader = BufReader::new(file);

    let mut messages = Vec::new();
    for (index, line) in reader.lines().enumerate() {
        let line = line?;
        if index == 0 || line.trim().is_empty() {
            continue; // Zeile 0 ist der Header
        }
        // Defekte Zeilen (z. B. durch einen Crash abgeschnitten) überspringen.
        if let Ok(message) = serde_json::from_str::<ChatMessage>(&line) {
            messages.push(message);
        }
    }
    Ok(messages)
}

fn list_in(dir: &Path) -> Vec<SessionMeta> {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return Vec::new();
    };

    let mut sessions = Vec::new();
    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) != Some("jsonl") {
            continue;
        }
        let Some(id) = path.file_stem().and_then(|s| s.to_str()).map(str::to_string) else {
            continue;
        };
        let modified = entry
            .metadata()
            .and_then(|m| m.modified())
            .unwrap_or(UNIX_EPOCH);
        let title = read_header(&path)
            .map(|h| h.title)
            .filter(|t| !t.is_empty())
            .unwrap_or_else(|| id.clone());
        sessions.push(SessionMeta { id, title, modified });
    }

    sessions.sort_by(|a, b| b.modified.cmp(&a.modified));
    sessions
}

fn read_header(path: &Path) -> Option<Header> {
    let file = File::open(path).ok()?;
    let mut first_line = String::new();
    BufReader::new(file).read_line(&mut first_line).ok()?;
    serde_json::from_str(first_line.trim()).ok()
}

/// Verzeichnis für Sitzungen: `$ANVIL_HOME` bzw. `$XDG_CONFIG_HOME/anvil` bzw.
/// `~/.config/anvil`, jeweils plus `sessions/`.
pub fn sessions_dir() -> Result<PathBuf> {
    let base = base_dir().context("Konnte kein Konfigurationsverzeichnis bestimmen (HOME?)")?;
    let dir = base.join("sessions");
    std::fs::create_dir_all(&dir).with_context(|| format!("Verzeichnis {dir:?} anlegen"))?;
    Ok(dir)
}

fn base_dir() -> Option<PathBuf> {
    for var in ["ANVIL_HOME", "XDG_CONFIG_HOME"] {
        if let Ok(value) = std::env::var(var)
            && !value.is_empty()
        {
            let path = PathBuf::from(value);
            return Some(if var == "ANVIL_HOME" {
                path
            } else {
                path.join("anvil")
            });
        }
    }
    std::env::var("HOME")
        .ok()
        .filter(|h| !h.is_empty())
        .map(|home| PathBuf::from(home).join(".config").join("anvil"))
}

fn new_id() -> SessionId {
    let millis = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0);
    millis.to_string()
}

/// Kürzt einen Titel auf eine Zeile und [`TITLE_MAX_CHARS`] Zeichen.
fn clean_title(title: &str) -> String {
    let single_line = title.replace('\n', " ");
    let trimmed = single_line.trim();
    if trimmed.chars().count() <= TITLE_MAX_CHARS {
        trimmed.to_string()
    } else {
        let kept: String = trimmed.chars().take(TITLE_MAX_CHARS - 1).collect();
        format!("{kept}…")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm::ToolCall;
    use serde_json::json;

    fn temp_dir(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("anvil_session_test_{name}"));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn write_then_load_roundtrip() {
        let dir = temp_dir("roundtrip");
        let mut writer = SessionWriter::create_in(&dir, "Erste Frage").unwrap();
        writer.append(&ChatMessage::User("hallo".into())).unwrap();
        writer
            .append(&ChatMessage::Assistant {
                text: "hi".into(),
                tool_calls: vec![ToolCall {
                    id: "c1".into(),
                    name: "bash".into(),
                    arguments: json!({"command": "ls"}),
                }],
            })
            .unwrap();

        // Datei-Stamm aus der Liste holen (= id).
        let id = list_in(&dir)[0].id.clone();
        let loaded = load_from(&dir, &id).unwrap();

        assert_eq!(loaded.len(), 2);
        assert!(matches!(&loaded[0], ChatMessage::User(t) if t == "hallo"));
        assert!(matches!(&loaded[1], ChatMessage::Assistant { text, tool_calls }
            if text == "hi" && tool_calls.len() == 1 && tool_calls[0].name == "bash"));

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn list_returns_title_and_sorts_newest_first() {
        let dir = temp_dir("list");
        SessionWriter::create_in(&dir, "Titel A").unwrap();
        // Sicherstellen, dass die zweite Datei eine andere id (ms) bekommt.
        std::thread::sleep(std::time::Duration::from_millis(2));
        SessionWriter::create_in(&dir, "Titel B").unwrap();

        let list = list_in(&dir);
        assert_eq!(list.len(), 2);
        // Neueste zuerst.
        assert_eq!(list[0].title, "Titel B");
        assert_eq!(list[1].title, "Titel A");

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn load_skips_corrupt_trailing_line() {
        let dir = temp_dir("corrupt");
        let mut writer = SessionWriter::create_in(&dir, "x").unwrap();
        writer.append(&ChatMessage::User("eins".into())).unwrap();
        drop(writer);

        let id = list_in(&dir)[0].id.clone();
        // Eine abgeschnittene/defekte Zeile anhängen.
        let path = dir.join(format!("{id}.jsonl"));
        let mut file = OpenOptions::new().append(true).open(&path).unwrap();
        writeln!(file, "{{ kaputt").unwrap();
        drop(file);

        let loaded = load_from(&dir, &id).unwrap();
        assert_eq!(loaded.len(), 1); // defekte Zeile übersprungen

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn title_is_truncated_and_single_line() {
        let long = "a".repeat(200);
        let cleaned = clean_title(&format!("zeile1\n{long}"));
        assert!(!cleaned.contains('\n'));
        assert!(cleaned.chars().count() <= TITLE_MAX_CHARS);
    }
}
