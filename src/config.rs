//! Provider-Auswahl beim Start — und die gemeinsame Stelle, an der ein Secret zu
//! einem Provider aufgelöst wird (Env **oder** gespeicherte `auth.json`).
//!
//! Regeln:
//! - `ANVIL_PROVIDER=openai|anthropic|google|gemini|openrouter` erzwingt einen
//!   Provider (Fehler, wenn dafür kein Credential vorliegt).
//! - Ohne Vorgabe: der erste Provider aus [`ProviderKind::ALL`], für den ein
//!   Credential existiert.
//! - `ANVIL_MODEL` überschreibt das Default-Modell.
//!
//! Ein Credential kommt entweder aus der provider-spezifischen Env-Variable
//! (Vorrang) oder aus `/login` (`auth.json`).

use crate::auth::{self, AuthInfo};
use crate::llm::{LlmClient, ProviderKind};

/// Liest die Umgebung + gespeicherte Auth und baut einen [`LlmClient`]. Bei `Err`
/// enthält der String eine menschenlesbare Erklärung fürs UI.
pub fn load() -> Result<LlmClient, String> {
    let model_override = read_env("ANVIL_MODEL");

    if let Some(pref) = read_env("ANVIL_PROVIDER") {
        let kind = resolve_provider(&pref).ok_or_else(|| {
            format!("Unbekannter ANVIL_PROVIDER={pref:?}. Erlaubt: openai, anthropic, google, openrouter.")
        })?;
        let secret = secret_for(kind).ok_or_else(|| {
            format!(
                "ANVIL_PROVIDER={pref}, aber kein Credential: weder {} gesetzt noch ein /login-Eintrag.",
                kind.env_var()
            )
        })?;
        return Ok(LlmClient::new(kind, secret, model_override));
    }

    for kind in ProviderKind::ALL {
        if let Some(secret) = secret_for(kind) {
            return Ok(LlmClient::new(kind, secret, model_override));
        }
    }

    Err("Kein Credential gefunden. Melde dich mit /login <provider> an \
         (oder setze z. B. OPENAI_API_KEY) und leg los."
        .into())
}

/// Löst einen Provider-Bezeichner inkl. gängiger Aliase auf.
fn resolve_provider(name: &str) -> Option<ProviderKind> {
    let name = name.to_lowercase();
    ProviderKind::from_id(&name).or(match name.as_str() {
        "gemini" => Some(ProviderKind::Gemini),
        "open-router" => Some(ProviderKind::OpenRouter),
        _ => None,
    })
}

/// Das Secret eines Providers: Env-Variable hat Vorrang, sonst die gespeicherte
/// Auth. Bei OAuth wird (vorerst) das Access-Token genommen — der Refresh läuft
/// über [`crate::oauth`], sobald ein konkreter Flow hinterlegt ist.
pub fn secret_for(kind: ProviderKind) -> Option<String> {
    if let Some(env) = read_env(kind.env_var()) {
        return Some(env);
    }
    match auth::get(kind.id())? {
        AuthInfo::Api { key } => Some(key),
        AuthInfo::Oauth { access, .. } => Some(access),
    }
}

/// Woher das Credential eines Providers stammt — für die Anzeige in `/login`.
pub fn source_label(kind: ProviderKind) -> Option<&'static str> {
    if read_env(kind.env_var()).is_some() {
        return Some("env");
    }
    match auth::get(kind.id()) {
        Some(AuthInfo::Api { .. }) => Some("gespeichert"),
        Some(AuthInfo::Oauth { .. }) => Some("OAuth"),
        None => None,
    }
}

/// Liest eine Env-Variable, trimmt Whitespace und behandelt Leerstrings wie
/// nicht gesetzt.
fn read_env(name: &str) -> Option<String> {
    std::env::var(name)
        .ok()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
}
