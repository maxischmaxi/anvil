//! Konfiguration beim Start: welche API-Keys liegen in der Umgebung, und
//! welcher Provider wird daraus gewählt.
//!
//! Regeln:
//! - `ANVIL_PROVIDER=anthropic|openai` erzwingt einen Provider (Fehler, wenn der
//!   passende Key fehlt).
//! - Ohne Vorgabe: ist nur ein Key gesetzt, wird der genommen; sind beide
//!   gesetzt, gewinnt Anthropic.
//! - `ANVIL_MODEL` überschreibt das Default-Modell des gewählten Providers.

use crate::llm::{LlmClient, ProviderKind};

/// Liest die Umgebung und baut einen [`LlmClient`]. Bei `Err` enthält der String
/// eine menschenlesbare Erklärung, die direkt im UI angezeigt werden kann.
pub fn load() -> Result<LlmClient, String> {
    let anthropic_key = read_env("ANTHROPIC_API_KEY");
    let openai_key = read_env("OPENAI_API_KEY");
    let preferred = read_env("ANVIL_PROVIDER").map(|s| s.to_lowercase());

    let kind = match preferred.as_deref() {
        Some("anthropic") => {
            if anthropic_key.is_none() {
                return Err(
                    "ANVIL_PROVIDER=anthropic, aber ANTHROPIC_API_KEY ist nicht gesetzt.".into(),
                );
            }
            ProviderKind::Anthropic
        }
        Some("openai") => {
            if openai_key.is_none() {
                return Err("ANVIL_PROVIDER=openai, aber OPENAI_API_KEY ist nicht gesetzt.".into());
            }
            ProviderKind::OpenAi
        }
        Some(other) => {
            return Err(format!(
                "Unbekannter ANVIL_PROVIDER={other:?}. Erlaubt sind: anthropic, openai."
            ));
        }
        None => match (anthropic_key.is_some(), openai_key.is_some()) {
            (true, _) => ProviderKind::Anthropic, // bei beiden Keys: Anthropic als Default
            (false, true) => ProviderKind::OpenAi,
            (false, false) => {
                return Err(
                    "Kein API-Key gefunden. Setze ANTHROPIC_API_KEY oder OPENAI_API_KEY \
                     (oder beide) und starte neu."
                        .into(),
                );
            }
        },
    };

    let api_key = match kind {
        ProviderKind::Anthropic => anthropic_key.unwrap(),
        ProviderKind::OpenAi => openai_key.unwrap(),
    };

    Ok(LlmClient::new(kind, api_key, read_env("ANVIL_MODEL")))
}

/// Liest eine Env-Variable, trimmt Whitespace und behandelt Leerstrings wie
/// nicht gesetzt.
fn read_env(name: &str) -> Option<String> {
    std::env::var(name)
        .ok()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
}
