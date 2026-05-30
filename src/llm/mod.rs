//! Die provider-agnostische LLM-Schicht.
//!
//! Hier lebt das gemeinsame Domain-Modell und der [`LlmClient`], der je nach
//! [`ProviderKind`] an den passenden Adapter ([`anthropic`]/[`openai`])
//! delegiert. Tool-Calling ist genau die Stelle, an der OpenAI und Anthropic
//! strukturell auseinanderlaufen — deshalb abstrahiert dieses Modul beide hinter
//! einem gemeinsamen Typ, und jeder Adapter übersetzt in sein Wire-Format.

pub mod anthropic;
pub mod openai;

use anyhow::Result;
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Default-Modelle (Stand Mai 2026). Per `ANVIL_MODEL` überschreibbar.
pub const ANTHROPIC_MODEL: &str = "claude-sonnet-4-6";
pub const OPENAI_MODEL: &str = "gpt-5.5";

/// Eine Nachricht im Gesprächskontext, der an das Modell geht. Die Varianten
/// bilden beide Provider verlustfrei ab:
/// - OpenAI: `User`→user-Message, `Assistant`→assistant mit `tool_calls`,
///   `ToolResults`→je Ergebnis eine `tool`-Message.
/// - Anthropic: `User`→user, `Assistant`→assistant mit `tool_use`-Blöcken,
///   `ToolResults`→**eine** user-Message mit `tool_result`-Blöcken.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ChatMessage {
    /// Eine Texteingabe des Nutzers.
    User(String),
    /// Ein Assistenten-Turn: Text (ggf. leer) plus 0..n Tool-Aufrufe.
    Assistant { text: String, tool_calls: Vec<ToolCall> },
    /// Ergebnisse ausgeführter Tools, die zurück ans Modell gehen.
    ToolResults(Vec<ToolResult>),
}

/// Ein vom Modell angeforderter Tool-Aufruf.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    /// Argumente als JSON (bei OpenAI aus einem String geparst, bei Anthropic
    /// aus akkumulierten `input_json_delta`-Fragmenten).
    pub arguments: Value,
}

/// Das Ergebnis eines ausgeführten Tools.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolResult {
    pub id: String,
    pub content: String,
    pub is_error: bool,
}

/// Die Definition eines Tools, die dem Modell angeboten wird.
pub struct ToolSpec {
    pub name: &'static str,
    pub description: &'static str,
    /// JSON-Schema der Argumente.
    pub parameters: Value,
}

/// Das Ergebnis eines Stream-Durchlaufs: der gesammelte Text plus alle vom
/// Modell angeforderten Tool-Aufrufe.
#[derive(Debug, Default)]
pub struct AssistantTurn {
    pub text: String,
    pub tool_calls: Vec<ToolCall>,
}

/// Welcher Anbieter angesprochen wird.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProviderKind {
    Anthropic,
    OpenAi,
}

/// Ein für einen Provider konfigurierter Client.
pub struct LlmClient {
    kind: ProviderKind,
    http: reqwest::Client,
    api_key: String,
    model: String,
}

impl LlmClient {
    pub fn new(kind: ProviderKind, api_key: String, model_override: Option<String>) -> Self {
        let model = model_override.unwrap_or_else(|| {
            match kind {
                ProviderKind::Anthropic => ANTHROPIC_MODEL,
                ProviderKind::OpenAi => OPENAI_MODEL,
            }
            .to_string()
        });
        Self {
            kind,
            http: reqwest::Client::new(),
            api_key,
            model,
        }
    }

    /// Menschenlesbare Beschreibung für die Statuszeile, z. B.
    /// `"Anthropic (claude-sonnet-4-6)"`.
    pub fn describe(&self) -> String {
        let name = match self.kind {
            ProviderKind::Anthropic => "Anthropic",
            ProviderKind::OpenAi => "OpenAI",
        };
        format!("{name} ({})", self.model)
    }

    /// Streamt eine Antwort des Modells. Text-Deltas werden live über `on_text`
    /// gemeldet (für die UI); zurückgegeben wird der vollständige Turn inklusive
    /// aller Tool-Aufrufe, die der Aufrufer dann ausführt.
    pub async fn stream(
        &self,
        system: Option<&str>,
        messages: &[ChatMessage],
        tools: &[ToolSpec],
        on_text: &mut (dyn FnMut(String) + Send),
    ) -> Result<AssistantTurn> {
        match self.kind {
            ProviderKind::Anthropic => {
                anthropic::stream(
                    &self.http, &self.api_key, &self.model, system, messages, tools, on_text,
                )
                .await
            }
            ProviderKind::OpenAi => {
                openai::stream(
                    &self.http, &self.api_key, &self.model, system, messages, tools, on_text,
                )
                .await
            }
        }
    }
}
