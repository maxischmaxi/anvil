//! OpenAI-Adapter (Chat Completions API).
//!
//! System-Prompt = normale Message. Tool-Calls kommen im Stream über
//! `choices[].delta.tool_calls[]` — fragmentiert und **pro Index** akkumuliert
//! (id/name im ersten Frame, die Argumente als JSON-String-Stücke danach).
//! Tool-Ergebnisse gehen als eigene Messages mit `role: "tool"` zurück.

use anyhow::{Context, Result, bail};
use eventsource_stream::Eventsource;
use futures::StreamExt;
use serde::Deserialize;
use serde_json::{Value, json};

use super::{AssistantTurn, ChatMessage, ToolCall, ToolSpec};

const API_URL: &str = "https://api.openai.com/v1/chat/completions";

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
        "messages": build_messages(system, messages),
        "stream": true,
    });
    if !tools.is_empty() {
        body["tools"] = json!(
            tools
                .iter()
                .map(|t| json!({
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description,
                        "parameters": t.parameters,
                    }
                }))
                .collect::<Vec<_>>()
        );
    }

    let response = http
        .post(API_URL)
        .bearer_auth(api_key)
        .json(&body)
        .send()
        .await
        .context("OpenAI-Request fehlgeschlagen")?;

    if !response.status().is_success() {
        let status = response.status();
        let detail = response.text().await.unwrap_or_default();
        bail!("OpenAI API {status}: {detail}");
    }

    let mut text = String::new();
    let mut partials: Vec<PartialToolCall> = Vec::new();

    let mut events = response.bytes_stream().eventsource();
    while let Some(event) = events.next().await {
        let event = event.context("OpenAI-SSE-Stream abgebrochen")?;
        if event.data == "[DONE]" {
            break;
        }
        let Ok(chunk) = serde_json::from_str::<ChatChunk>(&event.data) else {
            continue; // Keepalives/Leerframes überspringen
        };
        for choice in chunk.choices {
            if let Some(content) = choice.delta.content
                && !content.is_empty()
            {
                text.push_str(&content);
                on_text(content);
            }
            for delta in choice.delta.tool_calls {
                while partials.len() <= delta.index {
                    partials.push(PartialToolCall::default());
                }
                let partial = &mut partials[delta.index];
                if let Some(id) = delta.id {
                    partial.id = id;
                }
                if let Some(function) = delta.function {
                    if let Some(name) = function.name {
                        partial.name.push_str(&name);
                    }
                    if let Some(arguments) = function.arguments {
                        partial.arguments.push_str(&arguments);
                    }
                }
            }
        }
    }

    let tool_calls = partials
        .into_iter()
        .filter(|p| !p.name.is_empty())
        .map(PartialToolCall::finish)
        .collect();

    Ok(AssistantTurn { text, tool_calls })
}

/// Übersetzt unser Modell in das OpenAI-`messages`-Array.
fn build_messages(system: Option<&str>, messages: &[ChatMessage]) -> Vec<Value> {
    let mut out = Vec::new();
    if let Some(system) = system {
        out.push(json!({ "role": "system", "content": system }));
    }
    for message in messages {
        match message {
            ChatMessage::User(text) => {
                out.push(json!({ "role": "user", "content": text }));
            }
            ChatMessage::Assistant { text, tool_calls } => {
                let mut msg = json!({ "role": "assistant" });
                msg["content"] = if text.is_empty() {
                    Value::Null
                } else {
                    json!(text)
                };
                if !tool_calls.is_empty() {
                    msg["tool_calls"] = json!(
                        tool_calls
                            .iter()
                            .map(|tc| json!({
                                "id": tc.id,
                                "type": "function",
                                "function": {
                                    "name": tc.name,
                                    "arguments": tc.arguments.to_string(),
                                }
                            }))
                            .collect::<Vec<_>>()
                    );
                }
                out.push(msg);
            }
            ChatMessage::ToolResults(results) => {
                for result in results {
                    out.push(json!({
                        "role": "tool",
                        "tool_call_id": result.id,
                        "content": result.content,
                    }));
                }
            }
        }
    }
    out
}

#[derive(Default)]
struct PartialToolCall {
    id: String,
    name: String,
    arguments: String,
}

impl PartialToolCall {
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
struct ChatChunk {
    #[serde(default)]
    choices: Vec<Choice>,
}

#[derive(Deserialize)]
struct Choice {
    #[serde(default)]
    delta: Delta,
}

#[derive(Deserialize, Default)]
struct Delta {
    #[serde(default)]
    content: Option<String>,
    #[serde(default)]
    tool_calls: Vec<ToolCallDelta>,
}

#[derive(Deserialize)]
struct ToolCallDelta {
    index: usize,
    #[serde(default)]
    id: Option<String>,
    #[serde(default)]
    function: Option<FunctionDelta>,
}

#[derive(Deserialize)]
struct FunctionDelta {
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    arguments: Option<String>,
}
