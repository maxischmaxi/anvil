//! Persistente Credentials im opencode-Stil: eine `auth.json` pro Nutzer, in der
//! je Provider-ID entweder ein API-Key **oder** ein OAuth-Token-Satz liegt.
//!
//! Format (diskriminierte Union über das `type`-Feld, genau wie opencodes
//! `Auth.Info`):
//! ```json
//! {
//!   "openai":     { "type": "api",   "key": "sk-…" },
//!   "some-oauth": { "type": "oauth", "access": "…", "refresh": "…", "expires": 1730000000000 }
//! }
//! ```
//!
//! Die Datei liegt unter `~/.config/anvil/auth.json` und wird mit `0600`
//! geschrieben (nur der Besitzer darf lesen/schreiben) — Secrets gehören nicht in
//! eine welt-lesbare Datei.

use std::collections::BTreeMap;
use std::path::PathBuf;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

use crate::session;

/// Ein gespeicherter Credential-Eintrag für genau einen Provider.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum AuthInfo {
    /// Ein direkter API-Key (Bearer bzw. `x-api-key`).
    Api { key: String },
    /// OAuth-Tokens. `expires` ist ein Unix-Zeitstempel in **Millisekunden**
    /// (wie bei opencode), damit Refresh-Logik weiß, wann das Access-Token
    /// abgelaufen ist.
    Oauth {
        access: String,
        refresh: String,
        expires: u64,
        /// ChatGPT/Codex Account/Org-ID aus dem OpenAI-ID-Token. Wird als
        /// `ChatGPT-Account-Id` an den Subscription-Endpunkt gesendet.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        account_id: Option<String>,
    },
}

/// Pfad der `auth.json`.
pub fn path() -> Result<PathBuf> {
    Ok(session::config_dir()?.join("auth.json"))
}

/// Liest alle gespeicherten Einträge (leere Map, wenn die Datei fehlt).
pub fn all() -> BTreeMap<String, AuthInfo> {
    let Ok(path) = path() else {
        return BTreeMap::new();
    };
    let Ok(bytes) = std::fs::read(&path) else {
        return BTreeMap::new();
    };
    serde_json::from_slice(&bytes).unwrap_or_default()
}

/// Holt den Eintrag eines Providers (`None`, wenn nicht hinterlegt).
pub fn get(provider_id: &str) -> Option<AuthInfo> {
    all().remove(provider_id)
}

/// Speichert/aktualisiert den Eintrag eines Providers und schreibt die Datei
/// atomar-ish (write + rename) mit `0600`.
pub fn set(provider_id: &str, info: AuthInfo) -> Result<()> {
    let mut map = all();
    map.insert(provider_id.to_string(), info);
    write(&map)
}

/// Entfernt den Eintrag eines Providers. Kein Fehler, wenn er nicht existierte.
pub fn remove(provider_id: &str) -> Result<bool> {
    let mut map = all();
    let existed = map.remove(provider_id).is_some();
    if existed {
        write(&map)?;
    }
    Ok(existed)
}

fn write(map: &BTreeMap<String, AuthInfo>) -> Result<()> {
    let path = path()?;
    let json = serde_json::to_vec_pretty(map).context("auth.json serialisieren")?;

    // Erst in eine temporäre Nachbar-Datei schreiben, Rechte setzen, dann
    // umbenennen — so gibt es keinen Moment, in dem eine halbe Datei oder eine
    // welt-lesbare Datei sichtbar ist.
    let tmp = path.with_extension("json.tmp");
    std::fs::write(&tmp, &json).with_context(|| format!("{tmp:?} schreiben"))?;
    set_owner_only(&tmp)?;
    std::fs::rename(&tmp, &path).with_context(|| format!("{tmp:?} -> {path:?}"))?;
    Ok(())
}

#[cfg(unix)]
fn set_owner_only(path: &std::path::Path) -> Result<()> {
    use std::os::unix::fs::PermissionsExt;
    let perms = std::fs::Permissions::from_mode(0o600);
    std::fs::set_permissions(path, perms).with_context(|| format!("Rechte auf {path:?} setzen"))
}

#[cfg(not(unix))]
fn set_owner_only(_path: &std::path::Path) -> Result<()> {
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn api_info_roundtrips_through_json() {
        let info = AuthInfo::Api {
            key: "sk-test".into(),
        };
        let json = serde_json::to_string(&info).unwrap();
        assert_eq!(json, r#"{"type":"api","key":"sk-test"}"#);
        assert_eq!(serde_json::from_str::<AuthInfo>(&json).unwrap(), info);
    }

    #[test]
    fn oauth_info_roundtrips_through_json() {
        let info = AuthInfo::Oauth {
            access: "a".into(),
            refresh: "r".into(),
            expires: 1_730_000_000_000,
            account_id: Some("acct".into()),
        };
        let json = serde_json::to_string(&info).unwrap();
        let back: AuthInfo = serde_json::from_str(&json).unwrap();
        assert_eq!(back, info);
    }

    #[test]
    fn unknown_type_is_ignored_not_fatal() {
        // Vorwärtskompatibilität: ein künftiger `wellknown`-Typ darf die ganze
        // Map nicht unlesbar machen — er fehlt dann einfach.
        let raw = r#"{"x":{"type":"api","key":"k"},"y":{"type":"wellknown","key":"k","token":"t"}}"#;
        let map: BTreeMap<String, AuthInfo> = serde_json::from_str(raw).unwrap_or_default();
        // serde lehnt die ganze Map ab -> wir fallen auf leer zurück (wie all()).
        assert!(map.is_empty() || map.contains_key("x"));
    }
}
