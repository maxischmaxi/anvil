//! OpenAI-Subscription-Login im selben Stil wie opencodes eingebauter
//! `CodexAuthPlugin`: PKCE-OAuth über `auth.openai.com`, Token-Speicherung in
//! `auth.json`, Requests an den ChatGPT/Codex-Backend-Endpunkt.

use std::io::{Read, Write};
use std::net::TcpListener;
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

use std::time::Duration;

use anyhow::{Context, Result, bail};
use eventsource_stream::Eventsource;
use futures::StreamExt;
use serde::Deserialize;
use serde_json::{Value, json};

use crate::auth::{self, AuthInfo};
use crate::llm::{AssistantTurn, ChatMessage, ToolCall, ToolSpec};
use crate::oauth;

pub const CLIENT_ID: &str = "app_EMoamEEZ73f0CkXaXp7hrann";
pub const ISSUER: &str = "https://auth.openai.com";
pub const CODEX_API_ENDPOINT: &str = "https://chatgpt.com/backend-api/codex/responses";
pub const OAUTH_PORT: u16 = 1455;
pub const REDIRECT_PATH: &str = "/auth/callback";
const USER_AGENT: &str = concat!("anvil/", env!("CARGO_PKG_VERSION"));
const SUBSCRIPTION_IDLE_TIMEOUT: Duration = Duration::from_secs(90);

#[derive(Debug, Clone)]
pub struct SubscriptionToken {
    pub access: String,
    pub refresh: String,
    pub expires: u64,
    pub account_id: Option<String>,
}

/// Browser-Flow: lokalen Callback-Server starten, Auth-URL öffnen/anzeigen,
/// Code empfangen, Tokens tauschen und speichern.
pub async fn login_browser() -> Result<SubscriptionToken> {
    let pkce = oauth::pkce()?;
    let state = random_state()?;
    let redirect_uri = redirect_uri();
    let url = authorize_url(&redirect_uri, &pkce.challenge, &state);

    let listener = TcpListener::bind(("127.0.0.1", OAUTH_PORT))
        .with_context(|| format!("OAuth-Port {OAUTH_PORT} belegen"))?;
    let _ = open_browser(&url);

    let expected_state = state.clone();
    let code = tokio::task::spawn_blocking(move || capture_code(listener, &expected_state))
        .await
        .context("OAuth-Callback-Task")??;

    let http = reqwest::Client::new();
    let tokens = exchange_code(&http, &code, &redirect_uri, &pkce.verifier).await?;
    store(&tokens)?;
    Ok(tokens)
}

pub async fn refresh_if_needed(info: AuthInfo) -> Result<AuthInfo> {
    let AuthInfo::Oauth {
        access,
        refresh,
        expires,
        account_id,
    } = info
    else {
        return Ok(info);
    };

    if !access.is_empty() && expires > now_ms() + 30_000 {
        return Ok(AuthInfo::Oauth {
            access,
            refresh,
            expires,
            account_id,
        });
    }

    let http = reqwest::Client::new();
    let tokens = refresh_access_token(&http, &refresh).await?;
    let merged = SubscriptionToken {
        account_id: tokens.account_id.or(account_id),
        ..tokens
    };
    store(&merged)?;
    Ok(AuthInfo::Oauth {
        access: merged.access,
        refresh: merged.refresh,
        expires: merged.expires,
        account_id: merged.account_id,
    })
}

/// Streamt über den ChatGPT/Codex-Subscription-Endpunkt. Der Endpunkt spricht
/// das Responses-SSE-Format statt Chat-Completions; Tool-Calls werden analog zum
/// OpenAI-Adapter akkumuliert.
#[allow(clippy::too_many_arguments)]
pub async fn stream(
    http: &reqwest::Client,
    access_token: &str,
    account_id: Option<&str>,
    model: &str,
    system: Option<&str>,
    messages: &[ChatMessage],
    tools: &[ToolSpec],
    on_text: &mut (dyn FnMut(String) + Send),
) -> Result<AssistantTurn> {
    let body = build_body(model, system, messages, tools);

    let mut request = http
        .post(CODEX_API_ENDPOINT)
        .bearer_auth(access_token)
        .header("user-agent", USER_AGENT)
        .header("accept", "text/event-stream")
        // Der ChatGPT/Codex-Backend-Stream kommt gelegentlich komprimiert bzw.
        // mit kaputtem Body-Decoding zurück; bei SSE ist Kompression ohnehin
        // unnötig und kann in reqwest als `error decoding response body` enden.
        .header("accept-encoding", "identity")
        .header("cache-control", "no-cache")
        .json(&body);
    if let Some(account_id) = account_id.filter(|id| !id.is_empty()) {
        request = request.header("ChatGPT-Account-Id", account_id);
    }

    let response = request
        .send()
        .await
        .context("OpenAI-Subscription-Request fehlgeschlagen")?;
    if !response.status().is_success() {
        let status = response.status();
        let detail = response.text().await.unwrap_or_default();
        bail!("OpenAI-Subscription-API {status}: {detail}");
    }

    let mut text = String::new();
    let mut partials: Vec<PartialToolCall> = Vec::new();
    let mut events = response.bytes_stream().eventsource();
    loop {
        let event = match tokio::time::timeout(SUBSCRIPTION_IDLE_TIMEOUT, events.next()).await {
            Ok(Some(event)) => event.context("OpenAI-Subscription-SSE-Stream abgebrochen")?,
            Ok(None) => break,
            Err(_) => bail!(
                "OpenAI-Subscription-SSE-Stream seit {}s ohne Daten (vermutlich hängender ChatGPT/Codex-Backend-Stream). Bitte Prompt erneut senden; Kontext wurde nicht übernommen.",
                SUBSCRIPTION_IDLE_TIMEOUT.as_secs()
            ),
        };
        if event.data == "[DONE]" {
            break;
        }
        let Ok(value) = serde_json::from_str::<Value>(&event.data) else {
            continue;
        };
        collect_text_delta(&value, &mut text, on_text);
        collect_tool_call_delta(&value, &mut partials);
    }

    let tool_calls = partials
        .into_iter()
        .filter(|p| !p.name.is_empty())
        .map(PartialToolCall::finish)
        .collect();
    Ok(AssistantTurn { text, tool_calls })
}

fn store(tokens: &SubscriptionToken) -> Result<()> {
    auth::set(
        "openai",
        AuthInfo::Oauth {
            access: tokens.access.clone(),
            refresh: tokens.refresh.clone(),
            expires: tokens.expires,
            account_id: tokens.account_id.clone(),
        },
    )
}

fn authorize_url(redirect_uri: &str, challenge: &str, state: &str) -> String {
    let query = [
        ("response_type", "code"),
        ("client_id", CLIENT_ID),
        ("redirect_uri", redirect_uri),
        ("scope", "openid profile email offline_access"),
        ("code_challenge", challenge),
        ("code_challenge_method", "S256"),
        ("id_token_add_organizations", "true"),
        ("codex_cli_simplified_flow", "true"),
        ("state", state),
        ("originator", "anvil"),
    ]
    .iter()
    .map(|(k, v)| format!("{k}={}", percent_encode(v)))
    .collect::<Vec<_>>()
    .join("&");
    format!("{ISSUER}/oauth/authorize?{query}")
}

fn redirect_uri() -> String {
    format!("http://localhost:{OAUTH_PORT}{REDIRECT_PATH}")
}

async fn exchange_code(
    http: &reqwest::Client,
    code: &str,
    redirect_uri: &str,
    verifier: &str,
) -> Result<SubscriptionToken> {
    let body = form(&[
        ("grant_type", "authorization_code"),
        ("code", code),
        ("redirect_uri", redirect_uri),
        ("client_id", CLIENT_ID),
        ("code_verifier", verifier),
    ]);
    post_token(http, body).await
}

pub async fn refresh_access_token(
    http: &reqwest::Client,
    refresh_token: &str,
) -> Result<SubscriptionToken> {
    let body = form(&[
        ("grant_type", "refresh_token"),
        ("refresh_token", refresh_token),
        ("client_id", CLIENT_ID),
    ]);
    post_token(http, body).await
}

async fn post_token(http: &reqwest::Client, body: String) -> Result<SubscriptionToken> {
    let response = http
        .post(format!("{ISSUER}/oauth/token"))
        .header("content-type", "application/x-www-form-urlencoded")
        .header("user-agent", USER_AGENT)
        .body(body)
        .send()
        .await
        .context("OpenAI-Token-Request fehlgeschlagen")?;
    if !response.status().is_success() {
        let status = response.status();
        let detail = response.text().await.unwrap_or_default();
        bail!("OpenAI-Token-Endpoint {status}: {detail}");
    }
    let body: TokenResponse = response
        .json()
        .await
        .context("OpenAI-Token-Antwort lesen")?;
    let account_id = extract_account_id(body.id_token.as_deref())
        // Manchmal stecken Claims auch im Access-Token.
        .or_else(|| extract_account_id_from_jwt(&body.access_token));
    Ok(SubscriptionToken {
        access: body.access_token,
        refresh: body.refresh_token,
        expires: now_ms() + body.expires_in.unwrap_or(3600) * 1000,
        account_id,
    })
}

#[derive(Deserialize)]
struct TokenResponse {
    #[serde(default)]
    id_token: Option<String>,
    access_token: String,
    refresh_token: String,
    #[serde(default)]
    expires_in: Option<u64>,
}

fn capture_code(listener: TcpListener, expected_state: &str) -> Result<String> {
    let (mut stream, _) = listener.accept().context("OAuth-Redirect annehmen")?;
    let mut buf = [0u8; 4096];
    let read = stream.read(&mut buf).context("OAuth-Request lesen")?;
    let request = String::from_utf8_lossy(&buf[..read]);
    let target = request
        .lines()
        .next()
        .and_then(|line| line.split_whitespace().nth(1))
        .unwrap_or("");

    let error = query_param(target, "error_description").or_else(|| query_param(target, "error"));
    if let Some(error) = error {
        write_html(&mut stream, &html_error(&error), 400);
        bail!("OpenAI-OAuth fehlgeschlagen: {error}");
    }

    let state = query_param(target, "state").unwrap_or_default();
    if state != expected_state {
        write_html(&mut stream, &html_error("Invalid state"), 400);
        bail!("OAuth-State stimmt nicht (CSRF-Schutz)");
    }

    let code = query_param(target, "code").context("Kein OAuth-Code im Redirect")?;
    write_html(&mut stream, HTML_SUCCESS, 200);
    Ok(code)
}

const HTML_SUCCESS: &str = "<!doctype html><html><body style='font-family:system-ui;background:#131010;color:#f1ecec;display:grid;place-items:center;height:100vh'><main><h1>Authorization Successful</h1><p>Du kannst dieses Fenster schließen und zu anvil zurückkehren.</p></main><script>setTimeout(()=>window.close(),2000)</script></body></html>";

fn html_error(error: &str) -> String {
    format!(
        "<!doctype html><html><body style='font-family:system-ui;background:#131010;color:#f1ecec;display:grid;place-items:center;height:100vh'><main><h1 style='color:#fc533a'>Authorization Failed</h1><pre>{}</pre></main></body></html>",
        escape_html(error)
    )
}

fn write_html(stream: &mut std::net::TcpStream, body: &str, status: u16) {
    let status_text = if status == 200 { "OK" } else { "Bad Request" };
    let _ = stream.write_all(
        format!(
            "HTTP/1.1 {status} {status_text}\r\nContent-Type: text/html; charset=utf-8\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
            body.len(), body
        )
        .as_bytes(),
    );
}

fn extract_account_id(id_token: Option<&str>) -> Option<String> {
    id_token.and_then(extract_account_id_from_jwt)
}

fn build_body(
    model: &str,
    system: Option<&str>,
    messages: &[ChatMessage],
    tools: &[ToolSpec],
) -> Value {
    let mut body = json!({
        "model": model,
        // Der ChatGPT/Codex-Endpunkt verlangt dieses Feld zwingend, sogar wenn
        // kein System-Prompt vorhanden ist.
        "instructions": system.unwrap_or(""),
        "input": build_input(messages),
        "stream": true,
        // Der Codex-Backend-Endpunkt persistiert Antworten selbst nicht über
        // das Responses-API-Flag und lehnt den Default (`true`) ab.
        "store": false,
    });
    if !tools.is_empty() {
        body["tools"] = json!(
            tools
                .iter()
                .map(|tool| json!({
                    "type": "function",
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                }))
                .collect::<Vec<_>>()
        );
    }
    body
}

fn build_input(messages: &[ChatMessage]) -> Vec<Value> {
    let mut out = Vec::new();
    for message in messages {
        match message {
            ChatMessage::User(text) => out.push(json!({ "role": "user", "content": text })),
            ChatMessage::Assistant { text, tool_calls } => {
                if !text.is_empty() {
                    out.push(json!({ "role": "assistant", "content": text }));
                }
                for call in tool_calls {
                    out.push(json!({
                        "type": "function_call",
                        "call_id": call.id,
                        "name": call.name,
                        "arguments": call.arguments.to_string(),
                    }));
                }
            }
            ChatMessage::ToolResults(results) => {
                for result in results {
                    out.push(json!({
                        "type": "function_call_output",
                        "call_id": result.id,
                        "output": result.content,
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

fn collect_text_delta(value: &Value, text: &mut String, on_text: &mut (dyn FnMut(String) + Send)) {
    let kind = value.get("type").and_then(Value::as_str).unwrap_or("");
    let delta = match kind {
        "response.output_text.delta" | "response.refusal.delta" => value.get("delta"),
        "response.output_item.done" => value
            .get("item")
            .filter(|item| item.get("type").and_then(Value::as_str) == Some("message"))
            .and_then(message_text),
        _ => None,
    }
    .and_then(Value::as_str);

    if let Some(delta) = delta.filter(|delta| !delta.is_empty()) {
        text.push_str(delta);
        on_text(delta.to_string());
    }
}

fn message_text(item: &Value) -> Option<&Value> {
    item.get("content")?
        .as_array()?
        .iter()
        .find_map(|part| part.get("text").or_else(|| part.get("content")))
}

fn collect_tool_call_delta(value: &Value, partials: &mut Vec<PartialToolCall>) {
    let kind = value.get("type").and_then(Value::as_str).unwrap_or("");
    match kind {
        "response.function_call_arguments.delta" => {
            let index = output_index(value).unwrap_or(0);
            let partial = partial_at(partials, index);
            if let Some(delta) = value.get("delta").and_then(Value::as_str) {
                partial.arguments.push_str(delta);
            }
        }
        "response.output_item.added" | "response.output_item.done" => {
            if let Some(item) = value.get("item") {
                collect_tool_call_item(item, partials, output_index(value));
            }
        }
        _ => {}
    }
}

fn collect_tool_call_item(item: &Value, partials: &mut Vec<PartialToolCall>, index: Option<usize>) {
    if item.get("type").and_then(Value::as_str) != Some("function_call") {
        return;
    }
    let index = index
        .or_else(|| {
            item.get("output_index")
                .and_then(Value::as_u64)
                .map(|n| n as usize)
        })
        .unwrap_or(partials.len());
    let partial = partial_at(partials, index);
    if let Some(id) = item
        .get("call_id")
        .or_else(|| item.get("id"))
        .and_then(Value::as_str)
    {
        partial.id = id.to_string();
    }
    if let Some(name) = item.get("name").and_then(Value::as_str) {
        partial.name = name.to_string();
    }
    if let Some(arguments) = item.get("arguments").and_then(Value::as_str)
        && partial.arguments.is_empty()
    {
        partial.arguments = arguments.to_string();
    }
}

fn partial_at(partials: &mut Vec<PartialToolCall>, index: usize) -> &mut PartialToolCall {
    while partials.len() <= index {
        partials.push(PartialToolCall::default());
    }
    &mut partials[index]
}

fn output_index(value: &Value) -> Option<usize> {
    value
        .get("output_index")
        .and_then(Value::as_u64)
        .map(|n| n as usize)
}

fn extract_account_id_from_jwt(token: &str) -> Option<String> {
    let payload = token.split('.').nth(1)?;
    let json = String::from_utf8(base64url_decode(payload)?).ok()?;
    let claims: Value = serde_json::from_str(&json).ok()?;
    claims
        .get("chatgpt_account_id")
        .and_then(Value::as_str)
        .or_else(|| {
            claims
                .get("https://api.openai.com/auth")?
                .get("chatgpt_account_id")?
                .as_str()
        })
        .or_else(|| claims.get("organizations")?.get(0)?.get("id")?.as_str())
        .map(str::to_string)
}

fn open_browser(url: &str) -> Result<()> {
    #[cfg(target_os = "macos")]
    let mut cmd = Command::new("open");
    #[cfg(target_os = "linux")]
    let mut cmd = Command::new("xdg-open");
    #[cfg(target_os = "windows")]
    let mut cmd = {
        let mut c = Command::new("cmd");
        c.arg("/C").arg("start");
        c
    };
    cmd.arg(url).spawn().context("Browser öffnen")?;
    Ok(())
}

fn random_state() -> Result<String> {
    Ok(base64url(&random_bytes(32)?))
}

fn random_bytes(n: usize) -> Result<Vec<u8>> {
    let mut file = std::fs::File::open("/dev/urandom").context("/dev/urandom öffnen")?;
    let mut out = vec![0u8; n];
    file.read_exact(&mut out).context("/dev/urandom lesen")?;
    Ok(out)
}

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

fn form(params: &[(&str, &str)]) -> String {
    params
        .iter()
        .map(|(k, v)| format!("{}={}", percent_encode(k), percent_encode(v)))
        .collect::<Vec<_>>()
        .join("&")
}

fn query_param(target: &str, key: &str) -> Option<String> {
    let query = target.split_once('?')?.1;
    query
        .split('&')
        .filter_map(|pair| pair.split_once('='))
        .find(|(k, _)| *k == key)
        .map(|(_, v)| percent_decode(v))
}

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

fn base64url_decode(input: &str) -> Option<Vec<u8>> {
    let mut out = Vec::with_capacity(input.len() * 3 / 4);
    let mut buf = 0u32;
    let mut bits = 0u8;
    for b in input.bytes() {
        let val = match b {
            b'A'..=b'Z' => b - b'A',
            b'a'..=b'z' => b - b'a' + 26,
            b'0'..=b'9' => b - b'0' + 52,
            b'-' => 62,
            b'_' => 63,
            b'=' => break,
            _ => return None,
        } as u32;
        buf = (buf << 6) | val;
        bits += 6;
        if bits >= 8 {
            bits -= 8;
            out.push(((buf >> bits) & 0xff) as u8);
        }
    }
    Some(out)
}

fn escape_html(input: &str) -> String {
    input
        .replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn subscription_body_sends_system_prompt_as_required_instructions() {
        let body = build_body(
            "gpt-5.5",
            Some("be brief"),
            &[ChatMessage::User("sys".into())],
            &[],
        );

        assert_eq!(body["instructions"], json!("be brief"));
        assert_eq!(body["store"], json!(false));
        assert_eq!(body["input"], json!([{ "role": "user", "content": "sys" }]));
        assert!(
            body["input"]
                .as_array()
                .unwrap()
                .iter()
                .all(|item| item.get("role").and_then(Value::as_str) != Some("system"))
        );
    }

    #[test]
    fn subscription_body_keeps_empty_instructions_field() {
        let body = build_body("gpt-5.5", None, &[], &[]);

        assert!(body.as_object().unwrap().contains_key("instructions"));
        assert_eq!(body["instructions"], json!(""));
    }
}
