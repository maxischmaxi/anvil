//! Die Tools, die der Agent dem Modell anbietet, plus ihre Ausführung.
//!
//! Basic-Set: `read_file`, `write_file`, `edit_file`, `bash` — das, worauf pi
//! und Claude Code konvergieren. `bash` deckt `ls`/`grep`/`find`/git/Build/Test
//! ab, daher kommen wir mit vier Tools aus.
//!
//! Sicherheitshinweis: `bash` führt beliebige Befehle ungefragt aus. Für einen
//! echten Agent ist eine Bestätigungs-/Genehmigungsschicht der nächste Schritt.

use std::path::Path;
use std::time::Duration;

use serde_json::{Value, json};

use crate::llm::{ToolCall, ToolResult, ToolSpec};

/// Maximale Zeichenzahl, die ein Tool-Ergebnis ans Modell zurückgibt — schützt
/// das Kontextfenster vor riesigen Dateien/Ausgaben.
const MAX_OUTPUT_CHARS: usize = 30_000;
/// Zeitlimit für einen `bash`-Befehl.
const BASH_TIMEOUT: Duration = Duration::from_secs(120);

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
            description: "Replace an exact text snippet in a file. `old_string` must appear exactly once; include enough surrounding context to make it unique.",
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
            description: "Run a shell command via `sh -c` in the working directory. Returns stdout, stderr and the exit code.",
            parameters: json!({
                "type": "object",
                "properties": {
                    "command": { "type": "string", "description": "The shell command to run." }
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
pub async fn execute(call: &ToolCall) -> ToolResult {
    let outcome = match call.name.as_str() {
        "read_file" => read_file(&call.arguments).await,
        "write_file" => write_file(&call.arguments).await,
        "edit_file" => edit_file(&call.arguments).await,
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

async fn read_file(args: &Value) -> Result<String, String> {
    let path = string_arg(args, "path")?;
    tokio::fs::read_to_string(&path)
        .await
        .map_err(|e| format!("Konnte '{path}' nicht lesen: {e}"))
}

async fn write_file(args: &Value) -> Result<String, String> {
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
    tokio::fs::write(&path, content)
        .await
        .map_err(|e| format!("Konnte '{path}' nicht schreiben: {e}"))?;
    Ok(format!("Datei geschrieben: {path} ({bytes} Bytes)."))
}

async fn edit_file(args: &Value) -> Result<String, String> {
    let path = string_arg(args, "path")?;
    let old = string_arg(args, "old_string")?;
    let new = string_arg(args, "new_string")?;

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

    let updated = content.replacen(&old, &new, 1);
    tokio::fs::write(&path, updated)
        .await
        .map_err(|e| format!("Konnte '{path}' nicht schreiben: {e}"))?;
    Ok(format!("Datei bearbeitet: {path}."))
}

async fn bash(args: &Value) -> Result<String, String> {
    let command = string_arg(args, "command")?;

    let run = tokio::process::Command::new("sh")
        .arg("-c")
        .arg(&command)
        // Wird der Befehl per Esc abgebrochen, soll der Prozess mitsterben.
        .kill_on_drop(true)
        .output();

    let output = match tokio::time::timeout(BASH_TIMEOUT, run).await {
        Err(_) => return Err(format!("Befehl hat das Zeitlimit ({BASH_TIMEOUT:?}) überschritten.")),
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
        let result = execute(&call("bash", json!({ "command": "echo hallo" }))).await;
        assert!(!result.is_error);
        assert!(result.content.contains("hallo"));
        assert!(result.content.contains("[exit 0]"));
    }

    #[tokio::test]
    async fn bash_nonzero_exit_is_not_a_tool_error() {
        let result = execute(&call("bash", json!({ "command": "exit 3" }))).await;
        assert!(!result.is_error); // Tool lief; der Exit-Code steht im Ergebnis
        assert!(result.content.contains("[exit 3]"));
    }

    #[tokio::test]
    async fn write_then_read_roundtrip() {
        let path = temp_path("rw.txt");
        let p = path.to_str().unwrap();

        let written = execute(&call("write_file", json!({ "path": p, "content": "inhalt" }))).await;
        assert!(!written.is_error, "{}", written.content);

        let read = execute(&call("read_file", json!({ "path": p }))).await;
        assert!(!read.is_error);
        assert_eq!(read.content, "inhalt");

        let _ = std::fs::remove_file(&path);
    }

    #[tokio::test]
    async fn edit_replaces_unique_snippet() {
        let path = temp_path("edit.txt");
        let p = path.to_str().unwrap();
        std::fs::write(&path, "foo bar baz").unwrap();

        let edited = execute(&call(
            "edit_file",
            json!({ "path": p, "old_string": "bar", "new_string": "qux" }),
        ))
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

        let edited = execute(&call(
            "edit_file",
            json!({ "path": p, "old_string": "x", "new_string": "y" }),
        ))
        .await;
        assert!(edited.is_error);

        let _ = std::fs::remove_file(&path);
    }

    #[tokio::test]
    async fn read_missing_file_is_error() {
        let result = execute(&call("read_file", json!({ "path": "/no/such/anvil/path" }))).await;
        assert!(result.is_error);
    }

    #[tokio::test]
    async fn missing_argument_is_error() {
        let result = execute(&call("bash", json!({}))).await;
        assert!(result.is_error);
    }

    #[tokio::test]
    async fn unknown_tool_is_error() {
        let result = execute(&call("frobnicate", json!({}))).await;
        assert!(result.is_error);
    }
}
