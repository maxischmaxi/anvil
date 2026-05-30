//! Die Tools, die der Agent dem Modell anbietet, plus ihre Ausführung.
//!
//! Basic-Set: `read_file`, `write_file`, `edit_file`, `bash` — das, worauf pi
//! und Claude Code konvergieren. `bash` deckt `ls`/`grep`/`find`/git/Build/Test
//! ab, daher kommen wir mit vier Tools aus.
//!
//! Sicherheitshinweis: `bash` führt beliebige Befehle ungefragt aus. Für einen
//! echten Agent ist eine Bestätigungs-/Genehmigungsschicht der nächste Schritt.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::Duration;

use serde_json::{Value, json};

use crate::llm::{ToolCall, ToolResult, ToolSpec};

/// Maximale Zeichenzahl, die ein Tool-Ergebnis ans Modell zurückgibt — schützt
/// das Kontextfenster vor riesigen Dateien/Ausgaben.
const MAX_OUTPUT_CHARS: usize = 30_000;
/// Maximale Diff-Zeilen, die `edit_file` im Tool-Ergebnis ausgibt.
const MAX_DIFF_LINES: usize = 80;
/// Kontextzeilen vor/nach dem bearbeiteten Bereich im Diff.
const DIFF_CONTEXT_LINES: usize = 2;
/// Zeitlimit für einen `bash`-Befehl.
const BASH_TIMEOUT: Duration = Duration::from_secs(120);
const FORBIDDEN_BASH_WRITE_PATTERNS: &[&str] = &[
    ">",
    ">>",
    "tee ",
    "python ",
    "python3 ",
    "perl ",
    "ruby ",
    "node ",
    "sed -i",
    "truncate ",
    "rm ",
    "mv ",
    "cp ",
    "touch ",
    "chmod ",
    "chown ",
    "mkdir ",
    "rmdir ",
    "cat >",
    "cat <<",
    "git apply",
    "git checkout",
    "git restore",
    "git reset",
    "git clean",
    "git rm",
    "git mv",
    "git add",
    "git commit",
    "cargo fmt",
    "rustfmt",
    "prettier --write",
];

/// Flüchtiger Tool-Zustand pro Agent-Session. Dient aktuell als Safeguard:
/// `edit_file` darf nur Dateien bearbeiten, die vorher gelesen/geschrieben oder
/// durch ein voriges Edit in diesen Zustand übernommen wurden.
#[derive(Debug, Default)]
pub struct ToolState {
    read_files: HashMap<String, String>,
}

impl ToolState {
    pub fn reset(&mut self) {
        self.read_files.clear();
    }

    fn remember(&mut self, path: &str, content: String) {
        self.read_files.insert(path_key(path), content);
    }

    fn read_snapshot(&self, path: &str) -> Option<&str> {
        self.read_files.get(&path_key(path)).map(String::as_str)
    }
}

/// Die Tool-Definitionen, die mit jedem Request ans Modell gehen.
pub fn specs() -> Vec<ToolSpec> {
    vec![
        ToolSpec {
            name: "read_file",
            description: "Read the full contents of a file at the given path.",
            parameters: json!({
                "type": "object",
                "properties": {
                    "path": { "type": "string", "description": "File path, relative to the working directory or absolute." }
                },
                "required": ["path"]
            }),
        },
        ToolSpec {
            name: "write_file",
            description: "Create a new file or overwrite an existing one with the given content. Creates parent directories as needed.",
            parameters: json!({
                "type": "object",
                "properties": {
                    "path": { "type": "string", "description": "File path to write." },
                    "content": { "type": "string", "description": "Full content to write to the file." }
                },
                "required": ["path", "content"]
            }),
        },
        ToolSpec {
            name: "edit_file",
            description: "Replace an exact text snippet in a file. Safeguard: call read_file for the target file/range before edit_file. `old_string` must appear exactly once; include enough surrounding context to make it unique.",
            parameters: json!({
                "type": "object",
                "properties": {
                    "path": { "type": "string", "description": "File path to edit." },
                    "old_string": { "type": "string", "description": "Exact text to replace (must occur exactly once)." },
                    "new_string": { "type": "string", "description": "Text to replace it with." }
                },
                "required": ["path", "old_string", "new_string"]
            }),
        },
        ToolSpec {
            name: "bash",
            description: "Run a read-only shell command via `sh -c` in the working directory. Use this for inspection, builds and tests only. Do not modify files with bash; use write_file or edit_file for all file changes. Returns stdout, stderr and the exit code.",
            parameters: json!({
                "type": "object",
                "properties": {
                    "command": { "type": "string", "description": "Read-only shell command to run. Must not modify files." }
                },
                "required": ["command"]
            }),
        },
        ToolSpec {
            name: ASK_USER,
            description: "Ask the user a clarifying question and wait for their answer. Use when the request is ambiguous or you need a decision before continuing.",
            parameters: json!({
                "type": "object",
                "properties": {
                    "question": { "type": "string", "description": "The question to ask the user." }
                },
                "required": ["question"]
            }),
        },
    ]
}

/// Name des `ask_user`-Tools. Es wird nicht hier ausgeführt, sondern vom
/// [`crate::agent`] abgefangen, weil es auf eine Eingabe des Nutzers warten muss.
pub const ASK_USER: &str = "ask_user";

/// Führt einen Tool-Aufruf aus und verpackt das Ergebnis. Ein Fehler des Tools
/// wird nicht durchgereicht, sondern als `is_error`-Ergebnis ans Modell gegeben,
/// damit es selbst darauf reagieren kann.
pub async fn execute(call: &ToolCall, state: &mut ToolState) -> ToolResult {
    let outcome = match call.name.as_str() {
        "read_file" => read_file(&call.arguments, state).await,
        "write_file" => write_file(&call.arguments, state).await,
        "edit_file" => edit_file(&call.arguments, state).await,
        "bash" => bash(&call.arguments).await,
        // Sollte vom Agent abgefangen werden, bevor es hierher kommt.
        ASK_USER => Err("ask_user wird vom Agent behandelt, nicht in execute().".to_string()),
        other => Err(format!("Unbekanntes Tool: {other}")),
    };

    match outcome {
        Ok(content) => ToolResult {
            id: call.id.clone(),
            content: truncate(content),
            is_error: false,
        },
        Err(message) => ToolResult {
            id: call.id.clone(),
            content: message,
            is_error: true,
        },
    }
}

async fn read_file(args: &Value, state: &mut ToolState) -> Result<String, String> {
    let path = string_arg(args, "path")?;
    let content = tokio::fs::read_to_string(&path)
        .await
        .map_err(|e| format!("Konnte '{path}' nicht lesen: {e}"))?;
    state.remember(&path, content.clone());
    Ok(content)
}

async fn write_file(args: &Value, state: &mut ToolState) -> Result<String, String> {
    let path = string_arg(args, "path")?;
    let content = string_arg(args, "content")?;

    if let Some(parent) = Path::new(&path).parent()
        && !parent.as_os_str().is_empty()
    {
        tokio::fs::create_dir_all(parent)
            .await
            .map_err(|e| format!("Konnte Verzeichnis für '{path}' nicht anlegen: {e}"))?;
    }

    let bytes = content.len();
    tokio::fs::write(&path, &content)
        .await
        .map_err(|e| format!("Konnte '{path}' nicht schreiben: {e}"))?;
    state.remember(&path, content);
    Ok(format!("Datei geschrieben: {path} ({bytes} Bytes)."))
}

async fn edit_file(args: &Value, state: &mut ToolState) -> Result<String, String> {
    let path = string_arg(args, "path")?;
    let old = string_arg(args, "old_string")?;
    let new = string_arg(args, "new_string")?;

    let Some(snapshot) = state.read_snapshot(&path) else {
        return Err(format!(
            "Safeguard: '{path}' muss vor edit_file mit read_file gelesen werden."
        ));
    };
    if !snapshot.contains(&old) {
        return Err(format!(
            "Safeguard: old_string wurde im zuvor gelesenen Inhalt von '{path}' nicht gesehen. Bitte Datei/Bereich erneut lesen."
        ));
    }

    let content = tokio::fs::read_to_string(&path)
        .await
        .map_err(|e| format!("Konnte '{path}' nicht lesen: {e}"))?;

    match content.matches(&old).count() {
        0 => return Err(format!("old_string kommt in '{path}' nicht vor.")),
        1 => {}
        n => {
            return Err(format!(
                "old_string kommt {n}× in '{path}' vor — bitte mehr Kontext angeben, damit es eindeutig ist."
            ));
        }
    }

    let start = content
        .find(&old)
        .expect("match count above guaranteed one occurrence");
    let updated = content.replacen(&old, &new, 1);
    tokio::fs::write(&path, &updated)
        .await
        .map_err(|e| format!("Konnte '{path}' nicht schreiben: {e}"))?;
    state.remember(&path, updated);

    let diff = edit_diff(&content, start, &old, &new);
    Ok(format!(
        "Datei bearbeitet: {path}.
{diff}"
    ))
}

async fn bash(args: &Value) -> Result<String, String> {
    let command = string_arg(args, "command")?;
    reject_mutating_bash(&command)?;

    let run = tokio::process::Command::new("sh")
        .arg("-c")
        .arg(&command)
        // Wird der Befehl per Esc abgebrochen, soll der Prozess mitsterben.
        .kill_on_drop(true)
        .output();

    let output = match tokio::time::timeout(BASH_TIMEOUT, run).await {
        Err(_) => {
            return Err(format!(
                "Befehl hat das Zeitlimit ({BASH_TIMEOUT:?}) überschritten."
            ));
        }
        Ok(Err(e)) => return Err(format!("Befehl konnte nicht gestartet werden: {e}")),
        Ok(Ok(output)) => output,
    };

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    let code = output.status.code().unwrap_or(-1);

    let mut result = String::new();
    if !stdout.trim().is_empty() {
        result.push_str(stdout.trim_end());
        result.push('\n');
    }
    if !stderr.trim().is_empty() {
        result.push_str("[stderr]\n");
        result.push_str(stderr.trim_end());
        result.push('\n');
    }
    if result.is_empty() {
        result.push_str("(keine Ausgabe)\n");
    }
    result.push_str(&format!("[exit {code}]"));
    // Auch bei exit != 0 ist das ein gültiges Ergebnis (kein Tool-Fehler) — das
    // Modell soll den Exit-Code und stderr sehen und selbst reagieren.
    Ok(result)
}

/// Liest ein Pflicht-String-Argument aus dem JSON-Objekt.
fn reject_mutating_bash(command: &str) -> Result<(), String> {
    let compact = command.to_lowercase();
    if FORBIDDEN_BASH_WRITE_PATTERNS
        .iter()
        .any(|pattern| compact.contains(pattern))
    {
        return Err(
            "bash ist read-only: Dateiänderungen müssen über write_file oder edit_file erfolgen."
                .to_string(),
        );
    }
    Ok(())
}

/// Liest ein Pflicht-String-Argument aus dem JSON-Objekt.
fn path_key(path: &str) -> String {
    let path = PathBuf::from(path);
    let absolute = if path.is_absolute() {
        path
    } else {
        std::env::current_dir()
            .unwrap_or_else(|_| PathBuf::from("."))
            .join(path)
    };
    absolute.to_string_lossy().to_string()
}

fn edit_diff(before: &str, start_byte: usize, old: &str, new: &str) -> String {
    let before_lines: Vec<&str> = before.split('\n').collect();
    let start_line = before[..start_byte].bytes().filter(|b| *b == b'\n').count();
    let old_line_count = old.split('\n').count();
    let end_line = (start_line + old_line_count).min(before_lines.len());
    let context_start = start_line.saturating_sub(DIFF_CONTEXT_LINES);
    let context_end = (end_line + DIFF_CONTEXT_LINES).min(before_lines.len());

    let mut lines = Vec::new();
    lines.push("```diff".to_string());

    for line in &before_lines[context_start..start_line] {
        lines.push(format!("  {line}"));
    }
    for line in old.split('\n') {
        lines.push(format!("- {line}"));
    }
    for line in new.split('\n') {
        lines.push(format!("+ {line}"));
    }
    for line in &before_lines[end_line..context_end] {
        lines.push(format!("  {line}"));
    }

    let hidden = lines.len().saturating_sub(MAX_DIFF_LINES + 1); // + Fence bleibt erhalten.
    if hidden > 0 {
        let mut limited = lines
            .into_iter()
            .take(MAX_DIFF_LINES.saturating_sub(1))
            .collect::<Vec<_>>();
        limited.push(format!("… {hidden} more lines"));
        limited.push("```".to_string());
        limited.join("\n")
    } else {
        lines.push("```".to_string());
        lines.join("\n")
    }
}

/// Liest ein Pflicht-String-Argument aus dem JSON-Objekt.
fn string_arg(args: &Value, key: &str) -> Result<String, String> {
    args.get(key)
        .and_then(Value::as_str)
        .map(str::to_string)
        .ok_or_else(|| format!("Pflichtargument '{key}' fehlt oder ist kein String."))
}

/// Kürzt zu lange Ausgaben auf [`MAX_OUTPUT_CHARS`] Zeichen.
fn truncate(text: String) -> String {
    let total = text.chars().count();
    if total <= MAX_OUTPUT_CHARS {
        return text;
    }
    let kept: String = text.chars().take(MAX_OUTPUT_CHARS).collect();
    format!("{kept}\n… [Ausgabe gekürzt — {total} Zeichen insgesamt]")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn call(name: &str, arguments: Value) -> ToolCall {
        ToolCall {
            id: "test".to_string(),
            name: name.to_string(),
            arguments,
        }
    }

    fn temp_path(name: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!("anvil_test_{name}"))
    }

    #[tokio::test]
    async fn bash_runs_and_reports_exit() {
        let result = execute(
            &call("bash", json!({ "command": "echo hallo" })),
            &mut ToolState::default(),
        )
        .await;
        assert!(!result.is_error);
        assert!(result.content.contains("hallo"));
        assert!(result.content.contains("[exit 0]"));
    }

    #[tokio::test]
    async fn bash_nonzero_exit_is_not_a_tool_error() {
        let result = execute(
            &call("bash", json!({ "command": "exit 3" })),
            &mut ToolState::default(),
        )
        .await;
        assert!(!result.is_error); // Tool lief; der Exit-Code steht im Ergebnis
        assert!(result.content.contains("[exit 3]"));
    }

    #[tokio::test]
    async fn write_then_read_roundtrip() {
        let path = temp_path("rw.txt");
        let p = path.to_str().unwrap();

        let mut state = ToolState::default();
        let written = execute(
            &call("write_file", json!({ "path": p, "content": "inhalt" })),
            &mut state,
        )
        .await;
        assert!(!written.is_error, "{}", written.content);

        let read = execute(&call("read_file", json!({ "path": p })), &mut state).await;
        assert!(!read.is_error);
        assert_eq!(read.content, "inhalt");

        let _ = std::fs::remove_file(&path);
    }

    #[tokio::test]
    async fn edit_replaces_unique_snippet() {
        let path = temp_path("edit.txt");
        let p = path.to_str().unwrap();
        std::fs::write(&path, "foo bar baz").unwrap();

        let mut state = ToolState::default();
        let _ = execute(&call("read_file", json!({ "path": p })), &mut state).await;
        let edited = execute(
            &call(
                "edit_file",
                json!({ "path": p, "old_string": "bar", "new_string": "qux" }),
            ),
            &mut state,
        )
        .await;
        assert!(!edited.is_error, "{}", edited.content);
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "foo qux baz");

        let _ = std::fs::remove_file(&path);
    }

    #[tokio::test]
    async fn edit_rejects_ambiguous_match() {
        let path = temp_path("ambig.txt");
        let p = path.to_str().unwrap();
        std::fs::write(&path, "x x x").unwrap();

        let mut state = ToolState::default();
        let _ = execute(&call("read_file", json!({ "path": p })), &mut state).await;
        let edited = execute(
            &call(
                "edit_file",
                json!({ "path": p, "old_string": "x", "new_string": "y" }),
            ),
            &mut state,
        )
        .await;
        assert!(edited.is_error);

        let _ = std::fs::remove_file(&path);
    }

    #[tokio::test]
    async fn read_missing_file_is_error() {
        let result = execute(
            &call("read_file", json!({ "path": "/no/such/anvil/path" })),
            &mut ToolState::default(),
        )
        .await;
        assert!(result.is_error);
    }

    #[tokio::test]
    async fn missing_argument_is_error() {
        let result = execute(&call("bash", json!({})), &mut ToolState::default()).await;
        assert!(result.is_error);
    }

    #[tokio::test]
    async fn unknown_tool_is_error() {
        let result = execute(&call("frobnicate", json!({})), &mut ToolState::default()).await;
        assert!(result.is_error);
    }
}
