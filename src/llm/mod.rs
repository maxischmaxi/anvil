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

use crate::auth::AuthInfo;
use crate::openai_subscription;

/// Default-Modelle (Stand Mai 2026). Per `/models` oder `ANVIL_MODEL` wählbar.
pub const ANTHROPIC_MODEL: &str = "claude-sonnet-4-6";
pub const OPENAI_MODEL: &str = "gpt-5.5";
pub const GEMINI_MODEL: &str = "gemini-2.5-pro";
pub const OPENROUTER_MODEL: &str = "openai/gpt-5.5";

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
    Assistant {
        text: String,
        tool_calls: Vec<ToolCall>,
    },
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

/// Das Draht-Format eines Providers. Mehrere Provider teilen sich denselben
/// Adapter: OpenAI, Google Gemini (über deren OpenAI-kompatiblen Endpunkt) und
/// OpenRouter sprechen alle Chat-Completions; nur Anthropic weicht ab.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Wire {
    OpenAiCompatible,
    Anthropic,
    OpenAiSubscription,
}

/// Welcher Anbieter angesprochen wird. Trägt zugleich den kleinen Provider-
/// Katalog (Endpunkt, Env-Variable, Modell-Liste) im opencode-Stil.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProviderKind {
    Anthropic,
    OpenAi,
    Gemini,
    OpenRouter,
}

impl ProviderKind {
    /// Alle bekannten Provider in der Reihenfolge, in der sie in `/login` und bei
    /// der Auto-Auswahl auftauchen.
    pub const ALL: [ProviderKind; 4] = [
        ProviderKind::OpenAi,
        ProviderKind::Anthropic,
        ProviderKind::Gemini,
        ProviderKind::OpenRouter,
    ];

    /// Stabile ID (Schlüssel in `auth.json`, Argument für `/login <id>`).
    pub fn id(self) -> &'static str {
        match self {
            ProviderKind::Anthropic => "anthropic",
            ProviderKind::OpenAi => "openai",
            ProviderKind::Gemini => "google",
            ProviderKind::OpenRouter => "openrouter",
        }
    }

    pub fn from_id(id: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|p| p.id() == id)
    }

    /// Anzeigename für die UI.
    pub fn display(self) -> &'static str {
        match self {
            ProviderKind::Anthropic => "Anthropic",
            ProviderKind::OpenAi => "OpenAI",
            ProviderKind::Gemini => "Google Gemini",
            ProviderKind::OpenRouter => "OpenRouter",
        }
    }

    /// Name der Umgebungsvariable, aus der ein Key alternativ gelesen wird.
    pub fn env_var(self) -> &'static str {
        match self {
            ProviderKind::Anthropic => "ANTHROPIC_API_KEY",
            ProviderKind::OpenAi => "OPENAI_API_KEY",
            ProviderKind::Gemini => "GEMINI_API_KEY",
            ProviderKind::OpenRouter => "OPENROUTER_API_KEY",
        }
    }

    pub fn default_model(self) -> &'static str {
        match self {
            ProviderKind::Anthropic => ANTHROPIC_MODEL,
            ProviderKind::OpenAi => OPENAI_MODEL,
            ProviderKind::Gemini => GEMINI_MODEL,
            ProviderKind::OpenRouter => OPENROUTER_MODEL,
        }
    }

    /// Kuratierte Modell-Liste für `/models` (bewusst kurz und leicht editierbar;
    /// die eigentliche Wahrheit lebt beim Provider).
    pub fn models(self) -> &'static [&'static str] {
        match self {
            ProviderKind::Anthropic => &["claude-sonnet-4-6", "claude-opus-4-1"],
            ProviderKind::OpenAi => &[
                "gpt-5.5",
                "gpt-5.5-mini",
                "gpt-5.1-codex",
                "gpt-5.1-codex-mini",
                "gpt-5.2",
            ],
            ProviderKind::Gemini => &["gemini-2.5-pro", "gemini-2.5-flash"],
            ProviderKind::OpenRouter => &[
                "openai/gpt-5.5",
                "anthropic/claude-sonnet-4-6",
                "google/gemini-2.5-pro",
            ],
        }
    }

    /// Vollständige Endpunkt-URL.
    pub fn base_url(self) -> &'static str {
        match self {
            ProviderKind::Anthropic => "https://api.anthropic.com/v1/messages",
            ProviderKind::OpenAi => "https://api.openai.com/v1/chat/completions",
            ProviderKind::Gemini => {
                "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
            }
            ProviderKind::OpenRouter => "https://openrouter.ai/api/v1/chat/completions",
        }
    }

    fn wire(self) -> Wire {
        match self {
            ProviderKind::Anthropic => Wire::Anthropic,
            ProviderKind::OpenAi | ProviderKind::Gemini | ProviderKind::OpenRouter => {
                Wire::OpenAiCompatible
            }
        }
    }

    /// Ob für diesen Provider ein OAuth-/Subscription-Flow in anvil verdrahtet ist.
    pub fn supports_oauth(self) -> bool {
        matches!(self, ProviderKind::OpenAi)
    }
}

/// Ein Secret (API-Key oder OAuth-Access-Token), dessen `Debug`-Ausgabe nichts
/// preisgibt — damit Tokens nicht versehentlich in Logs/Traces landen.
#[derive(Clone)]
pub struct Secret(pub String);

impl std::fmt::Debug for Secret {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("Secret(***)")
    }
}

/// Welche Credential-Schiene der aktive Client tatsächlich verwendet.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AuthMode {
    /// Provider-API-Key (aus Env oder gespeicherter `auth.json`).
    ApiKey,
    /// OpenAI OAuth-/ChatGPT-Subscription-Token.
    OpenAiSubscription,
}

impl AuthMode {
    pub fn indicator(self) -> &'static str {
        match self {
            AuthMode::ApiKey => "🔑 API-Key",
            AuthMode::OpenAiSubscription => "◉ Subscription",
        }
    }

    pub fn short_indicator(self) -> &'static str {
        match self {
            AuthMode::ApiKey => "key",
            AuthMode::OpenAiSubscription => "sub",
        }
    }
}

/// Ein für einen Provider konfigurierter Client.
pub struct LlmClient {
    kind: ProviderKind,
    http: reqwest::Client,
    /// API-Key oder OAuth-Access-Token — was der Adapter als Bearer bzw.
    /// `x-api-key` sendet.
    secret: String,
    model: String,
    base_url: String,
    auth_info: Option<AuthInfo>,
}
impl LlmClient {
    pub fn with_auth(
        kind: ProviderKind,
        secret: String,
        model_override: Option<String>,
        auth_info: Option<AuthInfo>,
    ) -> Self {
        let model = model_override.unwrap_or_else(|| kind.default_model().to_string());
        let subscription = kind == ProviderKind::OpenAi && matches!(auth_info, Some(AuthInfo::Oauth { .. }));
        let http = if subscription {
            // Der ChatGPT/Codex-SSE-Stream ist mit HTTP/2 gelegentlich anfällig
            // für Body-Decoding/Transport-Abbrüche. HTTP/1.1 ist für SSE stabiler.
            reqwest::Client::builder()
                .http1_only()
                .build()
                .unwrap_or_else(|_| reqwest::Client::new())
        } else {
            reqwest::Client::new()
        };
        Self {
            kind,
            http,
            secret,
            model,
            base_url: kind.base_url().to_string(),
            auth_info,
        }
    }

    pub fn kind(&self) -> ProviderKind {
        self.kind
    }

    pub fn model(&self) -> &str {
        &self.model
    }

    pub fn auth_mode(&self) -> AuthMode {
        auth_mode_for(self.kind, self.auth_info.as_ref())
    }

    /// Menschenlesbare Beschreibung für die Statuszeile, z. B.
    /// `"OpenAI (gpt-5.5, ◉ Subscription)"`.
    pub fn describe(&self) -> String {
        format!(
            "{} ({}, {})",
            self.kind.display(),
            self.model,
            self.auth_mode().indicator()
        )
    }

    fn wire(&self) -> Wire {
        if self.kind == ProviderKind::OpenAi && matches!(self.auth_info, Some(AuthInfo::Oauth { .. })) {
            Wire::OpenAiSubscription
        } else {
            self.kind.wire()
        }
    }

    async fn refresh_subscription_token_if_needed(&mut self) -> Result<()> {
        if self.kind != ProviderKind::OpenAi {
            return Ok(());
        }
        let Some(auth_info) = self.auth_info.take() else {
            return Ok(());
        };
        let refreshed = openai_subscription::refresh_if_needed(auth_info).await?;
        if let AuthInfo::Oauth { access, .. } = &refreshed {
            self.secret = access.clone();
        }
        self.auth_info = Some(refreshed);
        Ok(())
    }

    /// Streamt eine Antwort des Modells. Text-Deltas werden live über `on_text`
    /// gemeldet (für die UI); zurückgegeben wird der vollständige Turn inklusive
    /// aller Tool-Aufrufe, die der Aufrufer dann ausführt.
    pub async fn stream(
        &mut self,
        system: Option<&str>,
        messages: &[ChatMessage],
        tools: &[ToolSpec],
        on_text: &mut (dyn FnMut(String) + Send),
    ) -> Result<AssistantTurn> {
        self.refresh_subscription_token_if_needed().await?;
        match self.wire() {
            Wire::Anthropic => {
                anthropic::stream(
                    &self.http,
                    &self.secret,
                    &self.model,
                    system,
                    messages,
                    tools,
                    on_text,
                )
                .await
            }
            Wire::OpenAiCompatible => {
                openai::stream(
                    &self.http,
                    &self.base_url,
                    &self.secret,
                    &self.model,
                    system,
                    messages,
                    tools,
                    on_text,
                )
                .await
            }
            Wire::OpenAiSubscription => {
                let account_id = match &self.auth_info {
                    Some(AuthInfo::Oauth { account_id, .. }) => account_id.as_deref(),
                    _ => None,
                };
                openai_subscription::stream(
                    &self.http,
                    &self.secret,
                    account_id,
                    &self.model,
                    system,
                    messages,
                    tools,
                    on_text,
                )
                .await
            }
        }
    }
}

pub fn auth_mode_for(kind: ProviderKind, auth_info: Option<&AuthInfo>) -> AuthMode {
    if kind == ProviderKind::OpenAi && matches!(auth_info, Some(AuthInfo::Oauth { .. })) {
        AuthMode::OpenAiSubscription
    } else {
        AuthMode::ApiKey
    }
}
