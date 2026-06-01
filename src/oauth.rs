//! Generisches OAuth-2.0-Gerüst mit PKCE (RFC 7636) — im Stil von opencodes
//! `ProviderAuth`, aber **ohne** First-Party-CLIs zu imitieren.
//!
//! Wichtig zur Einordnung: Ein Subscription-Login (ChatGPT, Gemini Advanced,
//! Claude Pro/Max) in einem Drittanbieter-Agenten verstößt gegen die AGB der
//! Anbieter — egal welcher Anbieter. Deshalb wird hier **keine** konkrete
//! Provider-Konfiguration ausgeliefert ([`ProviderKind::supports_oauth`] ist
//! überall `false`). Was hier liegt, ist nur die Maschinerie:
//!
//! 1. [`Pkce`] erzeugen → Verifier + S256-Challenge.
//! 2. [`authorize_url`] bauen, dem Nutzer zeigen (Methode „code") **oder** Browser
//!    öffnen + [`capture_redirect`] auf `127.0.0.1:<port>` lauschen (Methode „auto").
//! 3. [`exchange_code`] tauscht `code` + Verifier gegen Tokens.
//! 4. [`refresh`] erneuert ein abgelaufenes Access-Token.
//!
//! Wer einen **erlaubten** Flow hat (eigener OAuth-Client, Enterprise-SSO, …),
//! hinterlegt dafür eine [`OAuthConfig`] und hängt sie an den Provider.
//!
//! Dieses Modul ist bewusst ein noch nicht aufgerufenes Gerüst — kein Provider
//! liefert (aus AGB-Gründen) eine fertige Config. Daher `allow(dead_code)`: die
//! API ist vollständig und per Unit-Tests abgesichert, wartet aber auf eine
//! konkrete, erlaubte Anbindung.
#![allow(dead_code)]

use std::io::{Read, Write};
use std::net::TcpListener;
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result, bail};
use serde::Deserialize;

/// Konfiguration eines konkreten OAuth-Providers. Bewusst datenlos im Default —
/// die Felder füllt, wer einen legitimen Flow anbindet.
#[derive(Debug, Clone)]
pub struct OAuthConfig {
    pub client_id: String,
    pub auth_url: String,
    pub token_url: String,
    pub scopes: Vec<String>,
    /// Loopback-Port für den Redirect (`http://127.0.0.1:<port>/callback`).
    pub redirect_port: u16,
}

impl OAuthConfig {
    pub fn redirect_uri(&self) -> String {
        format!("http://127.0.0.1:{}/callback", self.redirect_port)
    }
}

/// Ein erteilter Token-Satz.
#[derive(Debug, Clone)]
pub struct Tokens {
    pub access: String,
    pub refresh: String,
    /// Ablaufzeitpunkt als Unix-Millisekunden (wie in [`crate::auth::AuthInfo`]).
    pub expires: u64,
}

/// PKCE-Paar: zufälliger `verifier` und die daraus abgeleitete `challenge`.
#[derive(Debug, Clone)]
pub struct Pkce {
    pub verifier: String,
    pub challenge: String,
}

/// Erzeugt ein frisches PKCE-Paar (S256). Der Verifier ist 32 Zufallsbytes
/// base64url-kodiert (43 Zeichen, im erlaubten Bereich 43–128).
pub fn pkce() -> Result<Pkce> {
    let verifier = base64url(&random_bytes(32)?);
    let challenge = challenge_for(&verifier);
    Ok(Pkce {
        verifier,
        challenge,
    })
}

/// Leitet die S256-Code-Challenge aus einem Verifier ab:
/// `base64url(sha256(verifier))`.
pub fn challenge_for(verifier: &str) -> String {
    base64url(&sha256(verifier.as_bytes()))
}

/// Baut die Authorization-URL (Authorization-Code-Flow + PKCE).
pub fn authorize_url(config: &OAuthConfig, challenge: &str, state: &str) -> String {
    let query = [
        ("response_type", "code"),
        ("client_id", &config.client_id),
        ("redirect_uri", &config.redirect_uri()),
        ("scope", &config.scopes.join(" ")),
        ("code_challenge", challenge),
        ("code_challenge_method", "S256"),
        ("state", state),
    ]
    .iter()
    .map(|(k, v)| format!("{k}={}", percent_encode(v)))
    .collect::<Vec<_>>()
    .join("&");
    format!("{}?{}", config.auth_url, query)
}

/// Tauscht einen Authorization-Code gegen Tokens.
pub async fn exchange_code(
    http: &reqwest::Client,
    config: &OAuthConfig,
    code: &str,
    verifier: &str,
) -> Result<Tokens> {
    let params = [
        ("grant_type", "authorization_code"),
        ("client_id", config.client_id.as_str()),
        ("code", code),
        ("redirect_uri", &config.redirect_uri()),
        ("code_verifier", verifier),
    ];
    post_token(http, &config.token_url, &params).await
}

/// Erneuert ein Access-Token über das Refresh-Token.
pub async fn refresh(
    http: &reqwest::Client,
    config: &OAuthConfig,
    refresh_token: &str,
) -> Result<Tokens> {
    let params = [
        ("grant_type", "refresh_token"),
        ("client_id", config.client_id.as_str()),
        ("refresh_token", refresh_token),
    ];
    post_token(http, &config.token_url, &params).await
}

async fn post_token(
    http: &reqwest::Client,
    token_url: &str,
    params: &[(&str, &str)],
) -> Result<Tokens> {
    // `application/x-www-form-urlencoded` von Hand — reqwest ist hier ohne das
    // `form`-Feature gebaut.
    let body = params
        .iter()
        .map(|(k, v)| format!("{}={}", percent_encode(k), percent_encode(v)))
        .collect::<Vec<_>>()
        .join("&");
    let response = http
        .post(token_url)
        .header("content-type", "application/x-www-form-urlencoded")
        .body(body)
        .send()
        .await
        .context("Token-Request fehlgeschlagen")?;
    if !response.status().is_success() {
        let status = response.status();
        let detail = response.text().await.unwrap_or_default();
        bail!("Token-Endpoint {status}: {detail}");
    }
    let body: TokenResponse = response.json().await.context("Token-Antwort lesen")?;
    let refresh = body.refresh_token.unwrap_or_default();
    Ok(Tokens {
        access: body.access_token,
        refresh,
        expires: now_ms() + body.expires_in.unwrap_or(3600) * 1000,
    })
}

#[derive(Deserialize)]
struct TokenResponse {
    access_token: String,
    #[serde(default)]
    refresh_token: Option<String>,
    /// Sekunden bis zum Ablauf.
    #[serde(default)]
    expires_in: Option<u64>,
}

/// Lauscht einmalig auf `127.0.0.1:<port>` und gibt den `code`-Query-Parameter
/// des ersten eingehenden Redirects zurück (Methode „auto"). Blockierend —
/// der Aufrufer sollte das in `tokio::task::spawn_blocking` ausführen.
pub fn capture_redirect(port: u16) -> Result<String> {
    let listener = TcpListener::bind(("127.0.0.1", port))
        .with_context(|| format!("Loopback-Port {port} belegen"))?;
    let (mut stream, _) = listener.accept().context("Redirect annehmen")?;

    let mut buf = [0u8; 2048];
    let read = stream.read(&mut buf).context("Redirect-Request lesen")?;
    let request = String::from_utf8_lossy(&buf[..read]);
    let target = request
        .lines()
        .next()
        .and_then(|line| line.split_whitespace().nth(1))
        .unwrap_or("");

    let body = "<html><body>anvil: Login abgeschlossen — du kannst dieses Fenster schließen.</body></html>";
    let _ = stream.write_all(
        format!(
            "HTTP/1.1 200 OK\r\nContent-Type: text/html; charset=utf-8\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
            body.len(),
            body
        )
        .as_bytes(),
    );

    query_param(target, "code")
        .context("Kein 'code' im Redirect — Login abgebrochen oder fehlgeschlagen?")
}

/// Liest einen Query-Parameter aus einem `…?a=b&c=d`-Pfad (minimal, mit
/// Prozent-Dekodierung des Werts).
fn query_param(target: &str, key: &str) -> Option<String> {
    let query = target.split_once('?')?.1;
    query
        .split('&')
        .filter_map(|pair| pair.split_once('='))
        .find(|(k, _)| *k == key)
        .map(|(_, v)| percent_decode(v))
}

// ---- Hilfsfunktionen ohne externe Crates ----

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

/// Zufallsbytes aus `/dev/urandom` (Linux/Unix).
fn random_bytes(n: usize) -> Result<Vec<u8>> {
    let mut file = std::fs::File::open("/dev/urandom").context("/dev/urandom öffnen")?;
    let mut out = vec![0u8; n];
    file.read_exact(&mut out).context("/dev/urandom lesen")?;
    Ok(out)
}

/// base64url ohne Padding (RFC 4648 §5) — für PKCE-Verifier/-Challenge.
fn base64url(input: &[u8]) -> String {
    const ALPHABET: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_";
    let mut out = String::with_capacity(input.len().div_ceil(3) * 4);
    for chunk in input.chunks(3) {
        let b0 = chunk[0] as u32;
        let b1 = *chunk.get(1).unwrap_or(&0) as u32;
        let b2 = *chunk.get(2).unwrap_or(&0) as u32;
        let triple = (b0 << 16) | (b1 << 8) | b2;
        out.push(ALPHABET[((triple >> 18) & 0x3f) as usize] as char);
        out.push(ALPHABET[((triple >> 12) & 0x3f) as usize] as char);
        if chunk.len() > 1 {
            out.push(ALPHABET[((triple >> 6) & 0x3f) as usize] as char);
        }
        if chunk.len() > 2 {
            out.push(ALPHABET[(triple & 0x3f) as usize] as char);
        }
    }
    out
}

/// Prozent-Kodierung für Query-Werte (alles außer den „unreserved" Zeichen).
fn percent_encode(value: &str) -> String {
    let mut out = String::with_capacity(value.len());
    for byte in value.bytes() {
        match byte {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                out.push(byte as char)
            }
            _ => out.push_str(&format!("%{byte:02X}")),
        }
    }
    out
}

fn percent_decode(value: &str) -> String {
    let bytes = value.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%'
            && i + 2 < bytes.len()
            && let Ok(byte) = u8::from_str_radix(&value[i + 1..i + 3], 16)
        {
            out.push(byte);
            i += 3;
            continue;
        }
        out.push(if bytes[i] == b'+' { b' ' } else { bytes[i] });
        i += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
}

/// SHA-256 (FIPS 180-4), self-contained — nur für die kurze PKCE-Challenge.
fn sha256(data: &[u8]) -> [u8; 32] {
    const K: [u32; 64] = [
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4,
        0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe,
        0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f,
        0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
        0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
        0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
        0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116,
        0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
        0xc67178f2,
    ];
    let mut h: [u32; 8] = [
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
        0x5be0cd19,
    ];

    let bit_len = (data.len() as u64) * 8;
    let mut msg = data.to_vec();
    msg.push(0x80);
    while msg.len() % 64 != 56 {
        msg.push(0);
    }
    msg.extend_from_slice(&bit_len.to_be_bytes());

    for block in msg.chunks_exact(64) {
        let mut w = [0u32; 64];
        for (i, word) in block.chunks_exact(4).enumerate() {
            w[i] = u32::from_be_bytes([word[0], word[1], word[2], word[3]]);
        }
        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16]
                .wrapping_add(s0)
                .wrapping_add(w[i - 7])
                .wrapping_add(s1);
        }

        let mut v = h;
        for i in 0..64 {
            let s1 = v[4].rotate_right(6) ^ v[4].rotate_right(11) ^ v[4].rotate_right(25);
            let ch = (v[4] & v[5]) ^ ((!v[4]) & v[6]);
            let t1 = v[7]
                .wrapping_add(s1)
                .wrapping_add(ch)
                .wrapping_add(K[i])
                .wrapping_add(w[i]);
            let s0 = v[0].rotate_right(2) ^ v[0].rotate_right(13) ^ v[0].rotate_right(22);
            let maj = (v[0] & v[1]) ^ (v[0] & v[2]) ^ (v[1] & v[2]);
            let t2 = s0.wrapping_add(maj);
            v[7] = v[6];
            v[6] = v[5];
            v[5] = v[4];
            v[4] = v[3].wrapping_add(t1);
            v[3] = v[2];
            v[2] = v[1];
            v[1] = v[0];
            v[0] = t1.wrapping_add(t2);
        }
        for i in 0..8 {
            h[i] = h[i].wrapping_add(v[i]);
        }
    }

    let mut out = [0u8; 32];
    for (i, word) in h.iter().enumerate() {
        out[i * 4..i * 4 + 4].copy_from_slice(&word.to_be_bytes());
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hex(bytes: &[u8]) -> String {
        bytes.iter().map(|b| format!("{b:02x}")).collect()
    }

    #[test]
    fn sha256_matches_known_vector() {
        // FIPS 180-4 Beispiel: SHA-256("abc").
        assert_eq!(
            hex(&sha256(b"abc")),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
        // Leerstring.
        assert_eq!(
            hex(&sha256(b"")),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
    }

    #[test]
    fn pkce_challenge_matches_rfc7636_vector() {
        // RFC 7636 Anhang B.
        let verifier = "dBjftJeZ4CVP-mB92K27uhbUJU1p1r_wW1gFWFOEjXk";
        assert_eq!(
            challenge_for(verifier),
            "E9Melhoa2OwvFrEMTJguCHaoeK1t8URWbuGJSstw-cM"
        );
    }

    #[test]
    fn base64url_has_no_padding_and_url_safe_alphabet() {
        // "fo" -> "Zm8" (kein '=' Padding).
        assert_eq!(base64url(b"fo"), "Zm8");
        assert_eq!(base64url(b"foobar"), "Zm9vYmFy");
        assert!(!base64url(&[0xff, 0xfe, 0xfd]).contains(['+', '/', '=']));
    }

    #[test]
    fn authorize_url_encodes_params() {
        let config = OAuthConfig {
            client_id: "cid".into(),
            auth_url: "https://example.com/authorize".into(),
            token_url: "https://example.com/token".into(),
            scopes: vec!["a b".into(), "c".into()],
            redirect_port: 1455,
        };
        let url = authorize_url(&config, "CHALLENGE", "STATE");
        assert!(url.starts_with("https://example.com/authorize?"));
        assert!(url.contains("code_challenge=CHALLENGE"));
        assert!(url.contains("code_challenge_method=S256"));
        assert!(url.contains("scope=a%20b%20c"));
        assert!(url.contains("redirect_uri=http%3A%2F%2F127.0.0.1%3A1455%2Fcallback"));
    }

    #[test]
    fn query_param_extracts_and_decodes_code() {
        let target = "/callback?state=x&code=ab%2Fcd&foo=1";
        assert_eq!(query_param(target, "code"), Some("ab/cd".to_string()));
        assert_eq!(query_param(target, "missing"), None);
    }
}
