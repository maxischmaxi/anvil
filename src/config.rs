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
//! Ein Credential kommt entweder aus `/login` (`auth.json`) oder aus der
//! provider-spezifischen Env-Variable. Ausnahme: ein gespeicherter OpenAI-
//! OAuth-/Subscription-Login hat Vorrang vor `OPENAI_API_KEY`, damit nicht
//! versehentlich Platform-Kosten entstehen.

use crate::auth::{self, AuthInfo};
use crate::llm::{auth_mode_for, AuthMode, LlmClient, ProviderKind};

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
        return Ok(LlmClient::with_auth(
            kind,
            secret,
            model_override,
            auth_info_for(kind),
        ));
    }

    for kind in ProviderKind::ALL {
        if let Some(secret) = secret_for(kind) {
            return Ok(LlmClient::with_auth(
                kind,
                secret,
                model_override,
                auth_info_for(kind),
            ));
        }
    }

    Err(
        "Kein Credential gefunden. Melde dich mit /login <provider> an \
         (oder setze z. B. OPENAI_API_KEY) und leg los."
            .into(),
    )
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

/// Das Secret eines Providers. Für OpenAI hat ein gespeicherter OAuth-/
/// Subscription-Login Vorrang vor `OPENAI_API_KEY`; bei allen anderen Fällen
/// bleibt Env-vor-`auth.json`. Bei OAuth wird das gespeicherte Access-Token
/// genommen; Subscription-Tokens werden beim Start/Wechsel best-effort refreshed.
pub fn secret_for(kind: ProviderKind) -> Option<String> {
    if let Some(AuthInfo::Oauth { access, .. }) = stored_oauth_for(kind) {
        return Some(access);
    }
    if let Some(env) = read_env(kind.env_var()) {
        return Some(env);
    }
    match auth::get(kind.id())? {
        AuthInfo::Api { key } => Some(key),
        AuthInfo::Oauth { access, .. } => Some(access),
    }
}

/// Gespeicherte Auth-Metadaten zum Provider, sofern nicht eine Env-Variable
/// Vorrang hat. Wichtig für OAuth: Account-ID und Refresh-Token bleiben so am
/// Client hängen.
pub fn auth_info_for(kind: ProviderKind) -> Option<AuthInfo> {
    if let Some(info) = stored_oauth_for(kind) {
        Some(info)
    } else if read_env(kind.env_var()).is_some() {
        None
    } else {
        auth::get(kind.id())
    }
}

/// Welche Credential-Schiene ein Provider aktuell verwenden würde.
pub fn auth_mode_for_provider(kind: ProviderKind) -> Option<AuthMode> {
    if let Some(info) = stored_oauth_for(kind) {
        Some(auth_mode_for(kind, Some(&info)))
    } else if read_env(kind.env_var()).is_some() {
        Some(AuthMode::ApiKey)
    } else {
        let info = auth::get(kind.id())?;
        Some(auth_mode_for(kind, Some(&info)))
    }
}

/// Woher das Credential eines Providers stammt — für die Anzeige in `/login`.
pub fn source_label(kind: ProviderKind) -> Option<&'static str> {
    if stored_oauth_for(kind).is_some() {
        return Some("OpenAI Subscription");
    }
    if read_env(kind.env_var()).is_some() {
        return Some("Env API-Key");
    }
    match auth::get(kind.id()) {
        Some(AuthInfo::Api { .. }) => Some("gespeicherter API-Key"),
        Some(AuthInfo::Oauth { .. }) => Some("OpenAI Subscription"),
        None => None,
    }
}

fn stored_oauth_for(kind: ProviderKind) -> Option<AuthInfo> {
    if kind != ProviderKind::OpenAi {
        return None;
    }
    match auth::get(kind.id()) {
        Some(info @ AuthInfo::Oauth { .. }) => Some(info),
        _ => None,
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    use std::sync::{Mutex, MutexGuard};
    use std::time::{SystemTime, UNIX_EPOCH};

    static ENV_LOCK: Mutex<()> = Mutex::new(());

    struct TestEnv {
        _guard: MutexGuard<'static, ()>,
        saved: Vec<(&'static str, Option<String>)>,
    }

    impl TestEnv {
        fn clear(vars: &[&'static str]) -> Self {
            let guard = ENV_LOCK.lock().unwrap();
            let saved = vars
                .iter()
                .map(|name| (*name, std::env::var(name).ok()))
                .collect::<Vec<_>>();
            for name in vars {
                remove_env(name);
            }
            Self {
                _guard: guard,
                saved,
            }
        }
    }

    impl Drop for TestEnv {
        fn drop(&mut self) {
            for (name, value) in &self.saved {
                match value {
                    Some(value) => set_env(name, value),
                    None => remove_env(name),
                }
            }
        }
    }

    fn set_env(name: &str, value: &str) {
        // Tests serialize env mutation with ENV_LOCK.
        unsafe { std::env::set_var(name, value) }
    }

    fn remove_env(name: &str) {
        // Tests serialize env mutation with ENV_LOCK.
        unsafe { std::env::remove_var(name) }
    }

    fn temp_home(name: &str) -> PathBuf {
        let millis = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_millis())
            .unwrap_or(0);
        let dir = std::env::temp_dir().join(format!("anvil_config_test_{name}_{millis}"));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn openai_oauth_beats_platform_api_key() {
        let _env = TestEnv::clear(&[
            "ANVIL_HOME",
            "ANVIL_MODEL",
            "ANVIL_PROVIDER",
            "OPENAI_API_KEY",
            "ANTHROPIC_API_KEY",
            "GEMINI_API_KEY",
            "OPENROUTER_API_KEY",
        ]);
        let home = temp_home("openai_oauth_beats_platform_api_key");
        set_env("ANVIL_HOME", home.to_str().unwrap());
        set_env("OPENAI_API_KEY", "sk-platform");
        auth::set(
            "openai",
            AuthInfo::Oauth {
                access: "oauth-access".into(),
                refresh: "oauth-refresh".into(),
                expires: u64::MAX,
                account_id: Some("acct".into()),
            },
        )
        .unwrap();

        assert_eq!(
            secret_for(ProviderKind::OpenAi).as_deref(),
            Some("oauth-access")
        );
        assert!(matches!(
            auth_info_for(ProviderKind::OpenAi),
            Some(AuthInfo::Oauth { .. })
        ));
        assert_eq!(
            auth_mode_for_provider(ProviderKind::OpenAi),
            Some(AuthMode::OpenAiSubscription)
        );
        assert_eq!(
            source_label(ProviderKind::OpenAi),
            Some("OpenAI Subscription")
        );

        let client = load().unwrap();
        assert_eq!(client.kind(), ProviderKind::OpenAi);
        assert_eq!(client.auth_mode(), AuthMode::OpenAiSubscription);
    }
}
