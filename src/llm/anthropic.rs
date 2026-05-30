//! Anthropic-Adapter (Messages API).
//!
//! System-Prompt = eigenes Top-Level-Feld, `max_tokens` Pflicht. Tool-Calls
//! sind `tool_use`-Content-Blöcke: `content_block_start` eröffnet sie (id+name),
//! die Argumente kommen als `input_json_delta`-Fragmente. Tool-Ergebnisse gehen
//! als **eine** user-Message mit `tool_result`-Blöcken zurück.

use anyhow::{Context, Result, bail};
use eventsource_stream::Eventsource;
use futures::StreamExt;
use serde::Deserialize;
use serde_json::{Value, json};

use super::{AssistantTurn, ChatMessage, ToolCall, ToolSpec};

const API_URL: &str = "https://api.anthropic.com/v1/messages";
const API_VERSION: &str = "2023-06-01";
const MAX_TOKENS: u32 = 8192;

pub async fn stream(
    http: &reqwest::Client,
    api_key: &str,
    model: &str,
    system: Option<&str>,
    messages: &[ChatMessage],
    tools: &[ToolSpec],
    on_text: &mut (dyn FnMut(String) + Send),
) -> Result<AssistantTurn> {
    let mut body = json!({
        "model": model,
        "max_tokens": MAX_TOKENS,
        "messages": build_messages(messages),
        "stream": true,
    });
    if let Some(system) = system {
        body["system"] = json!(system);
    }
    if !tools.is_empty() {
        body["tools"] = json!(
            tools
                .iter()
                .map(|t| json!({
                    "name": t.name,
                    "description": t.description,
                    "input_schema": t.parameters,
                }))
                .collect::<Vec<_>>()
        );
    }

    let response = http
        .post(API_URL)
        .header("x-api-key", api_key)
        .header("anthropic-version", API_VERSION)
        .json(&body)
        .send()
        .await
        .context("Anthropic-Request fehlgeschlagen")?;

    if !response.status().is_success() {
        let status = response.status();
        let detail = response.text().await.unwrap_or_default();
        bail!("Anthropic API {status}: {detail}");
    }

    let mut text = String::new();
    let mut blocks: Vec<PartialBlock> = Vec::new();

    let mut events = response.bytes_stream().eventsource();
    while let Some(event) = events.next().await {
        let event = event.context("Anthropic-SSE-Stream abgebrochen")?;
        match event.event.as_str() {
            "content_block_start" => {
                let start: BlockStart = serde_json::from_str(&event.data)
                    .context("content_block_start konnte nicht geparst werden")?;
                while blocks.len() <= start.index {
                    blocks.push(PartialBlock::default());
                }
                if start.content_block.kind == "tool_use" {
                    let block = &mut blocks[start.index];
                    block.is_tool = true;
                    block.id = start.content_block.id.unwrap_or_default();
                    block.name = start.content_block.name.unwrap_or_default();
                }
            }
            "content_block_delta" => {
                let delta: BlockDelta = serde_json::from_str(&event.data)
                    .context("content_block_delta konnte nicht geparst werden")?;
                match delta.delta.kind.as_str() {
                    "text_delta" => {
                        if let Some(chunk) = delta.delta.text {
                            text.push_str(&chunk);
                            on_text(chunk);
                        }
                    }
                    "input_json_delta" => {
                        if let Some(partial_json) = delta.delta.partial_json {
                            while blocks.len() <= delta.index {
                                blocks.push(PartialBlock::default());
                            }
                            blocks[delta.index].arguments.push_str(&partial_json);
                        }
                    }
                    _ => {}
                }
            }
            "message_stop" => break,
            "error" => bail!("Anthropic-Stream meldete einen Fehler: {}", event.data),
            // message_start, content_block_stop, ping, message_delta: ignorieren
            _ => {}
        }
    }

    let tool_calls = blocks
        .into_iter()
        .filter(|b| b.is_tool)
        .map(PartialBlock::finish)
        .collect();

    Ok(AssistantTurn { text, tool_calls })
}

/// Übersetzt unser Modell in das Anthropic-`messages`-Array.
fn build_messages(messages: &[ChatMessage]) -> Vec<Value> {
    let mut out = Vec::new();
    for message in messages {
        match message {
            ChatMessage::User(text) => {
                out.push(json!({ "role": "user", "content": text }));
            }
            ChatMessage::Assistant { text, tool_calls } => {
                let mut content = Vec::new();
                if !text.is_empty() {
                    content.push(json!({ "type": "text", "text": text }));
                }
                for tc in tool_calls {
                    content.push(json!({
                        "type": "tool_use",
                        "id": tc.id,
                        "name": tc.name,
                        "input": tc.arguments,
                    }));
                }
                out.push(json!({ "role": "assistant", "content": content }));
            }
            ChatMessage::ToolResults(results) => {
                let content: Vec<Value> = results
                    .iter()
                    .map(|r| {
                        json!({
                            "type": "tool_result",
                            "tool_use_id": r.id,
                            "content": r.content,
                            "is_error": r.is_error,
                        })
                    })
                    .collect();
                out.push(json!({ "role": "user", "content": content }));
            }
        }
    }
    out
}

#[derive(Default)]
struct PartialBlock {
    is_tool: bool,
    id: String,
    name: String,
    arguments: String,
}

impl PartialBlock {
    fn finish(self) -> ToolCall {
        let arguments = if self.arguments.trim().is_empty() {
            json!({})
        } else {
            serde_json::from_str(&self.arguments).unwrap_or_else(|_| json!({}))
        };
        ToolCall {
            id: self.id,
            name: self.name,
            arguments,
        }
    }
}

#[derive(Deserialize)]
struct BlockStart {
    index: usize,
    content_block: BlockStartInner,
}

#[derive(Deserialize)]
struct BlockStartInner {
    #[serde(rename = "type")]
    kind: String,
    #[serde(default)]
    id: Option<String>,
    #[serde(default)]
    name: Option<String>,
}

#[derive(Deserialize)]
struct BlockDelta {
    index: usize,
    delta: DeltaInner,
}

#[derive(Deserialize)]
struct DeltaInner {
    #[serde(rename = "type")]
    kind: String,
    #[serde(default)]
    text: Option<String>,
    #[serde(default)]
    partial_json: Option<String>,
}
