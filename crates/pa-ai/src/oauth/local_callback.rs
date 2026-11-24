//! The localhost callback server behind the browser redirect of the
//! subscription logins (Anthropic and the Google AI Pro/Ultra flow). Only a
//! matching redirect settles the login; error responses, missing parameters,
//! and unknown routes answer their page and keep waiting. A bind failure is a
//! hard error.

use std::net::ToSocketAddrs;
use std::sync::Arc;
use std::time::Duration;

use tokio::io::{AsyncReadExt, AsyncWriteExt};
use url::Url;

use super::types::{OAuthLoginUi, OAuthPrompt};

pub(crate) const CALLBACK_HOST_ENV: &str = "PI_OAUTH_CALLBACK_HOST";

/// One loopback redirect registration: the port and route the provider is
/// registered on, and the service named on the callback pages.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CallbackRoute {
    pub port: u16,
    pub path: &'static str,
    pub service: &'static str,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CallbackCode {
    pub code: String,
    pub state: String,
}

/// Settled once per login; the first settle wins.
// Two states: outer None = unsettled, inner None = settled-empty (cancelled).
#[allow(clippy::option_option)]
#[derive(Default, Debug)]
struct CallbackShared {
    result: tokio::sync::Mutex<Option<Option<CallbackCode>>>,
    notify: tokio::sync::Notify,
}

impl CallbackShared {
    async fn settle(&self, result: Option<CallbackCode>) {
        let mut slot = self.result.lock().await;
        if slot.is_none() {
            *slot = Some(result);
            drop(slot);
            // `notify_one` stores a permit when no waiter is registered yet, so a wait that checked
            // the empty slot just before the settle still wakes on its first poll —
            // `notify_waiters` would miss that window and hang the wait forever.
            self.notify.notify_one();
        }
    }
}

/// A running callback server.
///
/// Dropping aborts the accept loop, releasing the port — a settled or cancelled
/// login never wedges the registered port.
#[derive(Debug)]
pub struct LocalCallbackServer {
    shared: Arc<CallbackShared>,
    #[cfg(test)]
    port: u16,
    task: Option<tokio::task::JoinHandle<()>>,
}

impl Drop for LocalCallbackServer {
    fn drop(&mut self) {
        if let Some(task) = self.task.take() {
            task.abort();
        }
    }
}

impl LocalCallbackServer {
    /// Bind the route's registered redirect port on the callback host; a bind
    /// failure is the login's hard error.
    ///
    /// # Errors
    ///
    /// Returns an error when the listener cannot be bound.
    pub fn start(route: CallbackRoute, state: &str) -> Result<Self, String> {
        let host = std::env::var(CALLBACK_HOST_ENV).unwrap_or_else(|_| "127.0.0.1".to_string());
        Self::bind(&host, route.port, route, state)
    }

    /// Bind one exact host and port; the caller owns the failure.
    ///
    /// # Errors
    ///
    /// Returns an error when the listener cannot be bound.
    pub fn bind(host: &str, port: u16, route: CallbackRoute, state: &str) -> Result<Self, String> {
        // `SO_REUSEADDR`: lingering TIME_WAIT connections on the
        // registered port must not fail the next login's bind. Unix
        // only: on Windows the flag instead lets a second socket bind
        // over a live listener (the port-sharing hijack class), so the
        // plain bind keeps the accurate in-use failure.
        let addr: std::net::SocketAddr = (host, port)
            .to_socket_addrs()
            .map_err(|error| format!("port {port}: {error}"))?
            .next()
            .ok_or_else(|| format!("port {port}: no addresses"))?;
        let socket = if addr.is_ipv4() {
            tokio::net::TcpSocket::new_v4()
        } else {
            tokio::net::TcpSocket::new_v6()
        }
        .map_err(|error| format!("port {port}: {error}"))?;
        #[cfg(unix)]
        socket
            .set_reuseaddr(true)
            .map_err(|error| format!("port {port}: {error}"))?;
        socket
            .bind(addr)
            .map_err(|error| format!("port {port}: {error}"))?;
        let listener = socket
            .listen(1024)
            .map_err(|error| format!("port {port}: {error}"))?;
        #[cfg(test)]
        let bound_port = listener.local_addr().map_or(port, |addr| addr.port());
        let shared = Arc::new(CallbackShared::default());
        let task_shared = Arc::clone(&shared);
        let state = state.to_string();
        let task = tokio::spawn(async move {
            loop {
                let Ok((stream, _)) = listener.accept().await else {
                    break;
                };
                let shared = Arc::clone(&task_shared);
                let state = state.clone();
                tokio::spawn(async move {
                    serve_callback(stream, &shared, route, &state).await;
                });
            }
        });
        Ok(LocalCallbackServer {
            shared,
            #[cfg(test)]
            port: bound_port,
            task: Some(task),
        })
    }

    #[cfg(test)]
    pub fn port(&self) -> u16 {
        self.port
    }

    /// Settle the waiter with `None` (the tests drive the cancellation
    /// path).
    #[cfg(test)]
    pub async fn cancel(&self) {
        self.shared.settle(None).await;
    }

    /// The code and state, or `None` when the wait was cancelled.
    pub async fn wait_for_code(&self) -> Option<CallbackCode> {
        loop {
            if let Some(result) = self.shared.result.lock().await.take() {
                return result;
            }
            self.shared.notify.notified().await;
        }
    }
}

/// One browser request: read it, answer it, settle the login only on
/// a matching redirect.
async fn serve_callback(
    mut stream: tokio::net::TcpStream,
    shared: &CallbackShared,
    route: CallbackRoute,
    state: &str,
) {
    let Some(request) = read_request_head(&mut stream).await else {
        let _ = write_response(
            &mut stream,
            "400 Bad Request",
            error_page("Callback route not found.", None),
        )
        .await;
        return;
    };
    let target = request
        .lines()
        .next()
        .unwrap_or_default()
        .split_whitespace()
        .nth(1)
        .unwrap_or_default();
    let path = target.split('?').next().unwrap_or_default();
    if path != route.path {
        let _ = write_response(
            &mut stream,
            "404 Not Found",
            error_page("Callback route not found.", None),
        )
        .await;
        return;
    }
    let query = target.split_once('?').map_or("", |(_, query)| query);
    // The OAuth error response is checked before the parameters: the
    // provider answered, the login did not complete.
    if let Some(error) = query_param(query, "error") {
        let _ = write_response(
            &mut stream,
            "400 Bad Request",
            error_page(
                &format!("{} authentication did not complete.", route.service),
                Some(&format!("Error: {error}")),
            ),
        )
        .await;
        return;
    }
    let (code, echoed) = match (query_param(query, "code"), query_param(query, "state")) {
        (Some(code), Some(echoed)) if !code.is_empty() => (code, echoed),
        _ => {
            let _ = write_response(
                &mut stream,
                "400 Bad Request",
                error_page("Missing code or state parameter.", None),
            )
            .await;
            return;
        }
    };
    if echoed != state {
        let _ = write_response(
            &mut stream,
            "400 Bad Request",
            error_page("State mismatch.", None),
        )
        .await;
        return;
    }
    shared
        .settle(Some(CallbackCode {
            code,
            state: echoed,
        }))
        .await;
    let _ = write_response(
        &mut stream,
        "200 OK",
        success_page(&format!(
            "{} authentication completed. You can close this window.",
            route.service
        )),
    )
    .await;
}

/// Read one request head, accumulating until the blank line (a
/// fragmented browser request parses the same as a whole one).
async fn read_request_head(stream: &mut tokio::net::TcpStream) -> Option<String> {
    let mut head = Vec::with_capacity(1024);
    let mut buffer = [0u8; 8192];
    loop {
        if head.windows(4).any(|window| window == b"\r\n\r\n") {
            return Some(String::from_utf8_lossy(&head).to_string());
        }
        if head.len() >= buffer.len() {
            // An oversized head: parse what arrived; the route checks
            // reject it.
            return Some(String::from_utf8_lossy(&head).to_string());
        }
        let read = stream.read(&mut buffer).await.ok()?;
        if read == 0 {
            return None;
        }
        head.extend_from_slice(&buffer[..read]);
    }
}

async fn write_response(
    stream: &mut tokio::net::TcpStream,
    status: &str,
    body: String,
) -> std::io::Result<()> {
    let head = format!(
        "HTTP/1.1 {status}\r\nContent-Type: text/html; charset=utf-8\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        body.len()
    );
    stream.write_all(head.as_bytes()).await?;
    stream.write_all(body.as_bytes()).await?;
    stream.flush().await
}

/// One decoded query parameter (`+` as space, percent-decoded).
fn query_param(query: &str, name: &str) -> Option<String> {
    for pair in query.split('&') {
        if pair.is_empty() {
            continue;
        }
        let (key, value) = pair.split_once('=').unwrap_or((pair, ""));
        if percent_decode(&key.replace('+', " ")) == name {
            return Some(percent_decode(&value.replace('+', " ")));
        }
    }
    None
}

fn percent_decode(value: &str) -> String {
    let bytes = value.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut index = 0;
    while index < bytes.len() {
        if bytes[index] == b'%' && index + 2 < bytes.len() {
            if let Ok(byte) = u8::from_str_radix(&value[index + 1..index + 3], 16) {
                out.push(byte);
                index += 3;
                continue;
            }
        }
        out.push(bytes[index]);
        index += 1;
    }
    String::from_utf8_lossy(&out).to_string()
}

fn escape_html(value: &str) -> String {
    let mut out = String::with_capacity(value.len());
    for character in value.chars() {
        match character {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '"' => out.push_str("&quot;"),
            '\'' => out.push_str("&#39;"),
            other => out.push(other),
        }
    }
    out
}

/// The callback pages: a dark minimal page carrying the login's
/// outcome; the TS flow's wording rides the messages and details.
fn render_page(title: &str, message: &str, details: Option<&str>) -> String {
    let details_html = details.map_or_default(|details| {
        format!(
            r#"    <div class="details">{}</div>
"#,
            escape_html(details)
        )
    });
    format!(
        r#"<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{}</title>
  <style>
    body {{ margin: 0; min-height: 100vh; display: flex; align-items: center;
           justify-content: center; padding: 24px; background: #000; color: #fff;
           font-family: ui-sans-serif, system-ui, sans-serif; text-align: center; }}
    main {{ width: 100%; max-width: 560px; }}
    h1 {{ margin: 0 0 10px; font-size: 28px; line-height: 1.15; font-weight: 650; }}
    p {{ margin: 0; line-height: 1.7; color: #a1a1aa; font-size: 15px; }}
    .details {{ margin-top: 16px; font-family: ui-monospace, monospace; font-size: 13px;
                color: #a1a1aa; white-space: pre-wrap; word-break: break-word; }}
  </style>
</head>
<body>
  <main>
    <h1>Authentication</h1>
    <p>{}</p>
{}
  </main>
</body>
</html>"#,
        escape_html(title),
        escape_html(message),
        details_html
    )
}

fn success_page(message: &str) -> String {
    render_page("Prime Agent authentication completed", message, None)
}

fn error_page(message: &str, details: Option<&str>) -> String {
    render_page("Prime Agent authentication failed", message, details)
}

const CANCEL_POLL_INTERVAL: Duration = Duration::from_millis(250);

/// Wait for a login's authorization code: the browser callback wins over a late
/// manual paste (the manual answer only cancels the server's wait), the cancel
/// flag is re-checked between poll steps, and a paste is accepted in the URL,
/// `code=&state=`, `code#state`, or bare-code form with the echoed state checked
/// against `state`. `prompt` asks for the paste when neither path produced a
/// code; `cancelled` is the error a cancelled login reports.
///
/// # Errors
///
/// Returns an error when the login is cancelled or the echoed state mismatches.
pub(crate) async fn wait_for_authorization_code(
    server: &LocalCallbackServer,
    ui: &dyn OAuthLoginUi,
    state: &str,
    prompt: &OAuthPrompt,
    cancelled: &str,
) -> Result<CallbackCode, String> {
    let manual = ui.on_manual_code_input();
    let manual_available = manual.is_some();
    let manual_answer = async {
        match manual {
            Some(future) => future.await,
            None => std::future::pending::<Option<String>>().await,
        }
    };
    tokio::pin!(manual_answer);
    // The browser redirect wins over a late manual paste: the manual
    // answer only cancels the server's wait (TS `server.cancelWait`).
    // The browser redirect wins over a late manual paste: the manual
    // answer only cancels the server's wait.
    let wait = server.wait_for_code();
    tokio::pin!(wait);
    let mut tick = tokio::time::interval(CANCEL_POLL_INTERVAL);
    let raced = loop {
        if ui.is_cancelled() {
            return Err(cancelled.to_string());
        }
        tokio::select! {
            _ = tick.tick() => {}
            code = &mut wait => break RaceOutcome::Callback(code),
            answer = &mut manual_answer => break RaceOutcome::Manual(answer),
        }
    };
    let code: Option<CallbackCode> = match raced {
        RaceOutcome::Callback(Some(code)) => Some(code),
        RaceOutcome::Callback(None) => {
            if manual_available {
                match manual_answer.await {
                    None => return Err(cancelled.to_string()),
                    Some(input) => parse_paste(&input, state)?,
                }
            } else {
                None
            }
        }
        RaceOutcome::Manual(None) => return Err(cancelled.to_string()),
        RaceOutcome::Manual(Some(input)) => parse_paste(&input, state)?,
    };
    if let Some(code) = code {
        return Ok(code);
    }
    let answer = ui.on_prompt(prompt).await;
    let input = answer.ok_or_else(|| cancelled.to_string())?;
    parse_paste(&input, state)?.ok_or_else(|| "Missing authorization code".to_string())
}

enum RaceOutcome {
    /// The code, or `None` on a settled-empty wait.
    Callback(Option<CallbackCode>),
    /// The paste input, or `None` when cancelled.
    Manual(Option<String>),
}

pub(crate) fn parse_paste(input: &str, verifier: &str) -> Result<Option<CallbackCode>, String> {
    let (code, echoed) = parse_authorization_input(input);
    if let Some(echoed) = &echoed {
        if echoed != verifier {
            return Err("OAuth state mismatch".to_string());
        }
    }
    Ok(code.map(|code| CallbackCode {
        code,
        state: echoed.unwrap_or_else(|| verifier.to_string()),
    }))
}

pub(crate) fn parse_authorization_input(input: &str) -> (Option<String>, Option<String>) {
    let value = input.trim();
    if value.is_empty() {
        return (None, None);
    }
    let non_empty = |value: Option<String>| value.filter(|value| !value.is_empty());
    if let Ok(url) = Url::parse(value) {
        let get = |name: &str| {
            non_empty(
                url.query_pairs()
                    .find(|(key, _)| key == name)
                    .map(|(_, value)| value.to_string()),
            )
        };
        return (get("code"), get("state"));
    }
    if value.contains('#') {
        let mut parts = value.splitn(2, '#');
        let code = parts.next().unwrap_or_default();
        let state = parts.next().unwrap_or_default();
        return (
            non_empty(Some(code.to_string())),
            non_empty(Some(state.to_string())),
        );
    }
    if value.contains("code=") {
        let get = |name: &str| {
            non_empty(
                url::form_urlencoded::parse(value.as_bytes())
                    .find(|(key, _)| key == name)
                    .map(|(_, value)| value.to_string()),
            )
        };
        return (get("code"), get("state"));
    }
    (non_empty(Some(value.to_string())), None)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    use crate::oauth::anthropic::{ANTHROPIC_ROUTE, CALLBACK_PORT};

    fn live(state: &str) -> (LocalCallbackServer, u16) {
        let server = LocalCallbackServer::bind("127.0.0.1", 0, ANTHROPIC_ROUTE, state)
            .expect("a free loopback port binds");
        let port = server.port();
        (server, port)
    }

    async fn request(port: u16, target: &str) -> String {
        use tokio::io::AsyncReadExt as _;
        let mut stream = tokio::net::TcpStream::connect(("127.0.0.1", port))
            .await
            .expect("the callback server accepts the connection");
        stream
            .write_all(format!("GET {target} HTTP/1.1\r\nHost: localhost\r\n\r\n").as_bytes())
            .await
            .expect("the request writes");
        let mut response = String::new();
        let mut buffer = [0u8; 4096];
        loop {
            let read = stream.read(&mut buffer).await.unwrap_or(0);
            if read == 0 {
                break;
            }
            response.push_str(&String::from_utf8_lossy(&buffer[..read]));
            if response.contains("</html>") {
                break;
            }
        }
        response
    }

    #[tokio::test]
    async fn a_matching_redirect_settles_the_code_and_state() {
        let (server, port) = live("the-state");
        let response = request(port, "/callback?code=the-code&state=the-state").await;
        assert!(response.starts_with("HTTP/1.1 200 OK"));
        assert!(response.contains("Anthropic authentication completed"));
        assert_eq!(
            server.wait_for_code().await,
            Some(CallbackCode {
                code: "the-code".to_string(),
                state: "the-state".to_string(),
            })
        );
    }

    #[tokio::test]
    async fn missing_parameters_and_state_mismatch_never_settle() {
        let (server, port) = live("the-state");
        let missing = request(port, "/callback?code=the-code").await;
        assert!(missing.starts_with("HTTP/1.1 400 Bad Request"));
        assert!(missing.contains("Missing code or state parameter."));
        let mismatch = request(port, "/callback?code=the-code&state=other").await;
        assert!(mismatch.starts_with("HTTP/1.1 400 Bad Request"));
        assert!(mismatch.contains("State mismatch."));
        let unknown = request(port, "/other?code=the-code&state=the-state").await;
        assert!(unknown.starts_with("HTTP/1.1 404 Not Found"));
        assert!(unknown.contains("Callback route not found."));
        server.cancel().await;
        assert!(server.wait_for_code().await.is_none());
    }

    /// The missed-notification regression: a settle landing between the wait's empty-slot check and
    /// its `notified` registration still wakes (the `notify_one` permit). A `notify_waiters` settle
    /// would hang here.
    #[tokio::test]
    async fn a_settle_in_the_registration_window_still_wakes() {
        let server = LocalCallbackServer::bind("127.0.0.1", 0, ANTHROPIC_ROUTE, "the-state")
            .expect("a free loopback port binds");
        let shared = &server.shared;
        // The settle lands before the wait registers.
        shared
            .settle(Some(CallbackCode {
                code: "the-code".to_string(),
                state: "the-state".to_string(),
            }))
            .await;
        let result = tokio::time::timeout(Duration::from_secs(2), server.wait_for_code())
            .await
            .expect("the stored permit wakes the first poll")
            .expect("the settle settled a code");
        assert_eq!(result.code, "the-code");
    }

    /// The Anthropic callback has its own decoder, separate from the one in `callback.rs`.
    #[test]
    fn query_and_percent_decoding() {
        assert_eq!(percent_decode("a%20b"), "a b");
        assert_eq!(
            query_param("code=a+b&state=2", "code"),
            Some("a b".to_string())
        );
        // TS `URLSearchParams` keeps the bare key with an empty value
        // (`""` is falsy in the flow's missing-params check).
        assert_eq!(query_param("code&state=2", "code"), Some(String::new()));
    }
}
