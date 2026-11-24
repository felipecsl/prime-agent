//! The Google AI Pro/Ultra (Antigravity) OAuth flow: the PKCE request against
//! Google's authorize endpoint, the localhost callback raced against the manual
//! paste, the form-encoded token exchange, and the refresh. Unlike the Anthropic
//! flow, Google's `state` is its own random value: a leaked callback URL must not
//! also disclose the `code_verifier` that mints the tokens. Google hands out a
//! refresh token only under `access_type=offline` with `prompt=consent`, so both
//! are always sent.

use url::Url;

use super::local_callback::{
    wait_for_authorization_code, CallbackCode, CallbackRoute, LocalCallbackServer,
};
use super::pkce::generate_pkce;
use super::provider_http::{ProviderHttp, ProviderHttpMethod, ProviderHttpRequest};
use super::types::{OAuthLoginUi, OAuthPrompt};

/// Google's public Antigravity desktop client: a public OAuth client whose secret ships in the
/// CLI, not a private app credential. `GOOGLE_ANTIGRAVITY_CLIENT_ID`/`_SECRET` override it for a
/// caller's own OAuth app (Google can retire the shipped client, and the ID is easy to mistype:
/// it is `1071006060591-tmhssin2h21lcre235vtolojh4g403ep` — `tmhs`, not `thms`).
pub fn google_antigravity_client_id() -> String {
    env_override("GOOGLE_ANTIGRAVITY_CLIENT_ID", DEFAULT_CLIENT_ID)
}

/// The shipped client's secret, or `GOOGLE_ANTIGRAVITY_CLIENT_SECRET`.
pub fn google_antigravity_client_secret() -> String {
    env_override("GOOGLE_ANTIGRAVITY_CLIENT_SECRET", DEFAULT_CLIENT_SECRET)
}

/// The shipped client pair, or the caller's own OAuth app from the environment.
fn google_antigravity_client() -> (String, String) {
    (
        google_antigravity_client_id(),
        google_antigravity_client_secret(),
    )
}

const DEFAULT_CLIENT_ID: &str =
    "1071006060591-tmhssin2h21lcre235vtolojh4g403ep.apps.googleusercontent.com";
const DEFAULT_CLIENT_SECRET: &str = "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";

/// The env value when it is set and non-empty.
fn env_override(name: &str, fallback: &str) -> String {
    std::env::var(name)
        .ok()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| fallback.to_string())
}
const AUTHORIZE_URL: &str = "https://accounts.google.com/o/oauth2/v2/auth";
const TOKEN_URL: &str = "https://oauth2.googleapis.com/token";
/// The Antigravity surfaces: the Code Assist backend, the companion project
/// surface, the experiment/config surface, and the account identity.
const SCOPES: &[&str] = &[
    "https://www.googleapis.com/auth/aicode",
    "https://www.googleapis.com/auth/cloud-platform",
    "https://www.googleapis.com/auth/userinfo.email",
    "https://www.googleapis.com/auth/userinfo.profile",
    "https://www.googleapis.com/auth/cclog",
    "https://www.googleapis.com/auth/experimentsandconfigs",
];
/// The loopback registration Antigravity is registered on, with its
/// `redirect_uri`.
pub const CALLBACK_PORT: u16 = 51_121;
const CALLBACK_PATH: &str = "/oauth-callback";
const REDIRECT_URI: &str = "http://localhost:51121/oauth-callback";
const ROUTE: CallbackRoute = CallbackRoute {
    port: CALLBACK_PORT,
    path: CALLBACK_PATH,
    service: "Google",
};
pub const DEFAULT_TOKEN_TIMEOUT_MS: u64 = 30_000;
/// The refresh runs under the auth store's file lock, which a peer declares stale after 10 seconds;
/// the request must fit inside that window.
pub const REFRESH_TIMEOUT_MS: u64 = 8_000;
/// The credential's expiry skew: the token is treated as expired this long
/// before its actual expiry.
const EXPIRY_SKEW_MS: i64 = 5 * 60 * 1000;
/// Google's terms do not sanction third-party sign-in to Antigravity, and accounts have been
/// suspended for it; the sign-in says so once rather than hiding it behind a browser window.
const AUTH_INSTRUCTIONS: &str = "Complete sign-in in your browser. If the browser is on another \
     machine, paste the final redirect URL here. Note: Google's terms do not allow third-party \
     sign-in to Antigravity, and accounts have been suspended for using it - prefer GEMINI_API_KEY \
     or Google Vertex if you depend on this Google account.";
const PROMPT_MESSAGE: &str = "Paste the authorization code or full redirect URL:";
/// The cancel error the driving surface maps to the silent cancelled outcome.
pub const LOGIN_CANCELLED: &str = "Login cancelled";

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GoogleAntigravityCredentials {
    pub access: String,
    pub refresh: String,
    /// Wall-clock epoch milliseconds.
    pub expires: i64,
}

/// The callback code and the code exchange.
///
/// # Errors
///
/// Returns an error when the port cannot be bound, the login is cancelled
/// ([`LOGIN_CANCELLED`]), the paste mismatches, or the exchange fails.
pub async fn login_google_antigravity(
    http: &dyn ProviderHttp,
    ui: &dyn OAuthLoginUi,
) -> Result<GoogleAntigravityCredentials, String> {
    if ui.is_cancelled() {
        return Err(LOGIN_CANCELLED.to_string());
    }
    let (verifier, challenge) = generate_pkce();
    // A fresh random value, independent of the verifier and its challenge.
    let state = generate_pkce().0;
    let server = LocalCallbackServer::start(ROUTE, &state)?;
    login_with_server(http, ui, &server, &state, &verifier, &challenge).await
}

/// The flow after the bind: present the URL, race the callback against the
/// paste, exchange the code. `server` listens with `state` as its state.
async fn login_with_server(
    http: &dyn ProviderHttp,
    ui: &dyn OAuthLoginUi,
    server: &LocalCallbackServer,
    state: &str,
    verifier: &str,
    challenge: &str,
) -> Result<GoogleAntigravityCredentials, String> {
    ui.on_auth(
        &authorization_url(challenge, state),
        Some(AUTH_INSTRUCTIONS),
    );
    let code = wait_for_authorization_code(
        server,
        ui,
        state,
        &OAuthPrompt {
            message: PROMPT_MESSAGE.to_string(),
            placeholder: Some(REDIRECT_URI.to_string()),
            allow_empty: false,
        },
        LOGIN_CANCELLED,
    )
    .await?;
    if ui.is_cancelled() {
        return Err(LOGIN_CANCELLED.to_string());
    }
    ui.on_progress("Exchanging authorization code for tokens...");
    exchange_authorization_code(http, &code, verifier).await
}

/// Refresh an expired credential. Google omits `refresh_token` on refresh, so
/// the prior refresh token carries forward.
///
/// # Errors
///
/// Returns an error when the token refresh fails.
pub async fn refresh_google_antigravity_token(
    http: &dyn ProviderHttp,
    refresh_token: &str,
) -> Result<GoogleAntigravityCredentials, String> {
    let (client_id, client_secret) = google_antigravity_client();
    let form = form_body(&[
        ("client_id", client_id.as_str()),
        ("client_secret", client_secret.as_str()),
        ("refresh_token", refresh_token),
        ("grant_type", "refresh_token"),
    ]);
    let token =
        form_token_request(http, &form, "Google token refresh", REFRESH_TIMEOUT_MS, "").await?;
    Ok(GoogleAntigravityCredentials {
        access: token.access,
        refresh: token.refresh.unwrap_or_else(|| refresh_token.to_string()),
        expires: token.expires,
    })
}

fn authorization_url(challenge: &str, state: &str) -> String {
    let mut url = Url::parse(AUTHORIZE_URL).expect("the authorize url parses");
    let scope = SCOPES.join(" ");
    let client_id = google_antigravity_client_id();
    for (name, value) in [
        ("client_id", client_id.as_str()),
        ("response_type", "code"),
        ("redirect_uri", REDIRECT_URI),
        ("scope", scope.as_str()),
        ("code_challenge", challenge),
        ("code_challenge_method", "S256"),
        ("state", state),
        // Both are required for the offline grant that mints the refresh token.
        ("access_type", "offline"),
        ("prompt", "consent"),
    ] {
        url.query_pairs_mut().append_pair(name, value);
    }
    url.to_string()
}

async fn exchange_authorization_code(
    http: &dyn ProviderHttp,
    code: &CallbackCode,
    verifier: &str,
) -> Result<GoogleAntigravityCredentials, String> {
    let (client_id, client_secret) = google_antigravity_client();
    let form = form_body(&[
        ("client_id", client_id.as_str()),
        ("client_secret", client_secret.as_str()),
        ("code", &code.code),
        ("grant_type", "authorization_code"),
        ("redirect_uri", REDIRECT_URI),
        ("code_verifier", verifier),
    ]);
    let token = form_token_request(
        http,
        &form,
        "Token exchange",
        DEFAULT_TOKEN_TIMEOUT_MS,
        &format!(" redirect_uri={REDIRECT_URI}; response_type=authorization_code;"),
    )
    .await?;
    // Consent was requested with the offline grant; a response without a
    // refresh token is an unusable credential.
    let refresh = token.refresh.ok_or_else(|| {
        "Token exchange returned no refresh token; allow offline access on the Google consent screen."
            .to_string()
    })?;
    Ok(GoogleAntigravityCredentials {
        access: token.access,
        refresh,
        expires: token.expires,
    })
}

struct TokenResponse {
    access: String,
    refresh: Option<String>,
    /// Wall-clock epoch milliseconds, already skewed.
    expires: i64,
}

fn form_body(pairs: &[(&str, &str)]) -> String {
    url::form_urlencoded::Serializer::new(String::new())
        .extend_pairs(pairs)
        .finish()
}

/// One form-encoded token POST through the same error wrappers as the JSON
/// flows (`Error: <message>` heads). A missing field is this port's explicit
/// error where a looser client would persist an unusable credential.
async fn form_token_request(
    http: &dyn ProviderHttp,
    form: &str,
    label: &str,
    timeout_ms: u64,
    wire_context: &str,
) -> Result<TokenResponse, String> {
    let request = ProviderHttpRequest {
        method: ProviderHttpMethod::Post,
        url: TOKEN_URL.to_string(),
        headers: vec![(
            "Content-Type".to_string(),
            "application/x-www-form-urlencoded".to_string(),
        )],
        body: Some(form.to_string()),
        follow_redirects: true,
    };
    let response = http.request(request, timeout_ms).await.map_err(|message| {
        format!("{label} request failed. url={TOKEN_URL};{wire_context} details={message}")
    })?;
    if !response.ok() {
        return Err(format!(
            "{label} request failed. url={TOKEN_URL};{wire_context} details=Error: HTTP request failed. status={}; url={TOKEN_URL}; body={}",
            response.status, response.body
        ));
    }
    let json: serde_json::Value = serde_json::from_str(&response.body).map_err(|error| {
        format!("{label} returned invalid JSON. url={TOKEN_URL}; details=Error: {error}")
    })?;
    // A malformed SUCCESS body can echo credentials back: the errors name
    // the missing field, never the response body.
    let access = json
        .get("access_token")
        .and_then(serde_json::Value::as_str)
        .filter(|token| !token.is_empty())
        .ok_or_else(|| format!("{label} response missing field: access_token"))?
        .to_string();
    // Google's `expires_in` is a whole-second count; 3600 fits i64 and u64
    // without a bounds question worth a cast attribute.
    let expires_in = json
        .get("expires_in")
        .and_then(serde_json::Value::as_i64)
        .filter(|seconds| *seconds > 0)
        .ok_or_else(|| format!("{label} response missing field: expires_in"))?;
    let now = i64::try_from(crate::utils_inner::diagnostics::now_ms()).unwrap_or(i64::MAX);
    Ok(TokenResponse {
        access,
        refresh: json
            .get("refresh_token")
            .and_then(serde_json::Value::as_str)
            .filter(|token| !token.is_empty())
            .map(str::to_string),
        expires: now
            .saturating_add(expires_in.saturating_mul(1000))
            .saturating_sub(EXPIRY_SKEW_MS),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::{HashMap, VecDeque};
    use std::future::Future;
    use std::pin::Pin;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::{Arc, Mutex};
    use std::time::Duration;

    type ScriptedResponse = super::super::provider_http::ProviderHttpResponse;

    /// Queued responses per url (popped in order), with the requests recorded.
    struct ScriptedHttp {
        queued: Mutex<HashMap<String, VecDeque<ScriptedResponse>>>,
        requests: Mutex<Vec<ProviderHttpRequest>>,
    }

    impl ScriptedHttp {
        fn new() -> Self {
            ScriptedHttp {
                queued: Mutex::new(HashMap::new()),
                requests: Mutex::new(Vec::new()),
            }
        }

        fn entry(status: u16, body: &str) -> ScriptedResponse {
            ScriptedResponse {
                status,
                body: body.to_string(),
            }
        }

        fn queue(self, url: &str, responses: Vec<ScriptedResponse>) -> Self {
            self.queued.lock().unwrap().insert(
                url.to_string(),
                responses.into_iter().collect::<VecDeque<_>>(),
            );
            self
        }

        fn bodies_for(&self, url: &str) -> Vec<ProviderHttpRequest> {
            self.requests
                .lock()
                .unwrap()
                .iter()
                .filter(|request| request.url == url)
                .cloned()
                .collect()
        }
    }

    impl ProviderHttp for ScriptedHttp {
        fn request(
            &self,
            request: ProviderHttpRequest,
            _timeout_ms: u64,
        ) -> Pin<Box<dyn Future<Output = Result<ScriptedResponse, String>> + Send + '_>> {
            self.requests.lock().unwrap().push(request.clone());
            let response = self
                .queued
                .lock()
                .unwrap()
                .get_mut(&request.url)
                .and_then(std::collections::VecDeque::pop_front);
            Box::pin(
                async move { response.ok_or_else(|| format!("{} was not scripted", request.url)) },
            )
        }
    }

    struct ScriptedUi {
        auth_url: tokio::sync::watch::Sender<Option<String>>,
        auth_seen: tokio::sync::watch::Receiver<Option<String>>,
        cancelled: Arc<AtomicBool>,
    }

    impl ScriptedUi {
        fn new() -> Self {
            let (auth_url, auth_seen) = tokio::sync::watch::channel(None);
            ScriptedUi {
                auth_url,
                auth_seen,
                cancelled: Arc::new(AtomicBool::new(false)),
            }
        }

        /// The presented authorize url, once the flow presents it.
        async fn captured_auth(&self) -> String {
            let mut seen = self.auth_seen.clone();
            loop {
                if let Some(url) = seen.borrow_and_update().clone() {
                    return url;
                }
                assert!(
                    seen.changed().await.is_ok(),
                    "the ui went away before presenting the authorize url"
                );
            }
        }
    }

    impl OAuthLoginUi for ScriptedUi {
        fn on_auth(&self, url: &str, _instructions: Option<&str>) {
            let _ = self.auth_url.send(Some(url.to_string()));
        }

        fn on_prompt(
            &self,
            _prompt: &super::super::types::OAuthPrompt,
        ) -> Pin<Box<dyn Future<Output = Option<String>> + Send + '_>> {
            Box::pin(std::future::pending())
        }

        fn on_progress(&self, _message: &str) {}

        fn on_manual_code_input(
            &self,
        ) -> Option<Pin<Box<dyn Future<Output = Option<String>> + Send + '_>>> {
            None
        }

        fn is_cancelled(&self) -> bool {
            self.cancelled.load(Ordering::Relaxed)
        }
    }

    fn token_body(access: &str, expires_in: i64, refresh: Option<&str>) -> String {
        let refresh = refresh.map_or_default(|refresh| format!(",\"refresh_token\":\"{refresh}\""));
        format!(r#"{{"access_token":"{access}","expires_in":{expires_in}{refresh}}}"#)
    }

    #[test]
    fn the_authorize_url_carries_the_offline_pkce_request() {
        let url = authorization_url("the-challenge", "the-state");
        let parsed = Url::parse(&url).expect("the authorize url parses");
        let params: HashMap<String, String> = parsed.query_pairs().into_owned().collect();
        assert_eq!(parsed.as_str().split_once('?').unwrap().0, AUTHORIZE_URL);
        assert_eq!(params["client_id"], google_antigravity_client_id());
        assert_eq!(params["response_type"], "code");
        assert_eq!(params["redirect_uri"], REDIRECT_URI);
        assert_eq!(params["code_challenge"], "the-challenge");
        assert_eq!(params["code_challenge_method"], "S256");
        assert_eq!(params["state"], "the-state");
        assert_eq!(params["access_type"], "offline");
        assert_eq!(params["prompt"], "consent");
        let scope: Vec<&str> = params["scope"].split(' ').collect();
        assert_eq!(scope, SCOPES);
    }

    #[tokio::test]
    async fn the_exchange_posts_the_form_encoded_pkce_grant() {
        let http = ScriptedHttp::new().queue(
            TOKEN_URL,
            vec![ScriptedHttp::entry(
                200,
                &token_body("the-access", 3600, Some("the-refresh")),
            )],
        );
        let code = CallbackCode {
            code: "the-code".to_string(),
            state: "the-state".to_string(),
        };
        let credentials = exchange_authorization_code(&http, &code, "the-verifier")
            .await
            .expect("the exchange succeeds");
        assert_eq!(credentials.access, "the-access");
        assert_eq!(credentials.refresh, "the-refresh");
        let request = &http.bodies_for(TOKEN_URL)[0];
        assert_eq!(
            request.headers,
            vec![(
                "Content-Type".to_string(),
                "application/x-www-form-urlencoded".to_string()
            )]
        );
        let form: HashMap<String, String> =
            url::form_urlencoded::parse(request.body.as_ref().unwrap().as_bytes())
                .into_owned()
                .collect();
        assert_eq!(form["grant_type"], "authorization_code");
        assert_eq!(form["code"], "the-code");
        assert_eq!(form["code_verifier"], "the-verifier");
        assert_eq!(form["redirect_uri"], REDIRECT_URI);
        assert_eq!(form["client_id"], google_antigravity_client_id());
        assert_eq!(form["client_secret"], google_antigravity_client_secret());
    }

    #[tokio::test]
    async fn an_exchange_without_a_refresh_token_is_rejected() {
        let http = ScriptedHttp::new().queue(
            TOKEN_URL,
            vec![ScriptedHttp::entry(
                200,
                &token_body("the-access", 3600, None),
            )],
        );
        let code = CallbackCode {
            code: "the-code".to_string(),
            state: "the-state".to_string(),
        };
        let error = exchange_authorization_code(&http, &code, "the-verifier")
            .await
            .expect_err("an online-only credential is unusable");
        assert!(error.contains("no refresh token"), "{error}");
    }

    #[tokio::test]
    async fn a_refresh_keeps_the_prior_refresh_token_when_the_endpoint_omits_one() {
        let http = ScriptedHttp::new().queue(
            TOKEN_URL,
            vec![ScriptedHttp::entry(
                200,
                &token_body("fresh-access", 3600, None),
            )],
        );
        let credentials = refresh_google_antigravity_token(&http, "the-refresh")
            .await
            .expect("the refresh succeeds");
        assert_eq!(credentials.access, "fresh-access");
        assert_eq!(credentials.refresh, "the-refresh");
        let form: HashMap<String, String> = url::form_urlencoded::parse(
            http.bodies_for(TOKEN_URL)[0]
                .body
                .as_ref()
                .unwrap()
                .as_bytes(),
        )
        .into_owned()
        .collect();
        assert_eq!(form["grant_type"], "refresh_token");
        assert_eq!(form["refresh_token"], "the-refresh");
    }

    #[tokio::test]
    async fn a_rotated_refresh_token_replaces_the_prior_one() {
        let http = ScriptedHttp::new().queue(
            TOKEN_URL,
            vec![ScriptedHttp::entry(
                200,
                &token_body("fresh-access", 3600, Some("rotated-refresh")),
            )],
        );
        let credentials = refresh_google_antigravity_token(&http, "the-refresh")
            .await
            .expect("the refresh succeeds");
        assert_eq!(credentials.refresh, "rotated-refresh");
    }

    #[tokio::test]
    async fn the_expiry_is_skewed_from_the_wire_lifetime() {
        let http = ScriptedHttp::new().queue(
            TOKEN_URL,
            vec![ScriptedHttp::entry(
                200,
                &token_body("the-access", 3600, Some("the-refresh")),
            )],
        );
        let credentials = refresh_google_antigravity_token(&http, "the-refresh")
            .await
            .expect("the refresh succeeds");
        let now = i64::try_from(crate::utils_inner::diagnostics::now_ms()).unwrap_or(i64::MAX);
        // One hour of lifetime, minus the skew, within the test's runtime.
        assert!((credentials.expires - (now + 3_600_000 - EXPIRY_SKEW_MS)).abs() < 5_000);
    }

    #[tokio::test]
    async fn a_failed_exchange_names_the_status_and_body() {
        let http = ScriptedHttp::new().queue(
            TOKEN_URL,
            vec![ScriptedHttp::entry(400, r#"{"error":"invalid_grant"}"#)],
        );
        let error = refresh_google_antigravity_token(&http, "stale-refresh")
            .await
            .expect_err("a rejected grant fails the refresh");
        assert!(
            error.contains("Google token refresh request failed"),
            "{error}"
        );
        assert!(error.contains("status=400"), "{error}");
        assert!(error.contains("invalid_grant"), "{error}");
    }

    #[tokio::test]
    async fn a_cancelled_login_never_reaches_the_token_endpoint() {
        let http = ScriptedHttp::new();
        let ui = ScriptedUi::new();
        ui.cancelled.store(true, Ordering::Relaxed);
        let error = login_google_antigravity(&http, &ui)
            .await
            .expect_err("the cancelled login stops");
        assert_eq!(error, LOGIN_CANCELLED);
        assert_eq!(http.bodies_for(TOKEN_URL).len(), 0);
    }

    /// The loopback redirect settles the login and the exchanged code carries
    /// the PKCE verifier and the `state` the authorize url presented.
    #[tokio::test]
    async fn the_loopback_redirect_settles_the_login() {
        use tokio::io::{AsyncReadExt as _, AsyncWriteExt as _};
        let http = Arc::new(ScriptedHttp::new().queue(
            TOKEN_URL,
            vec![ScriptedHttp::entry(
                200,
                &token_body("the-access", 3600, Some("the-refresh")),
            )],
        ));
        let ui = Arc::new(ScriptedUi::new());
        let (verifier, challenge) = generate_pkce();
        let state = generate_pkce().0;
        // The flow moves these into the task; the assertions keep copies.
        let expected_state = state.clone();
        let expected_verifier = verifier.clone();
        let server = LocalCallbackServer::bind("127.0.0.1", 0, ROUTE, &state)
            .expect("a free loopback port binds");
        let port = server.port();
        let flow = {
            let flow_http = Arc::clone(&http);
            let flow_ui = Arc::clone(&ui);
            tokio::spawn(async move {
                login_with_server(
                    flow_http.as_ref(),
                    flow_ui.as_ref(),
                    &server,
                    &state,
                    &verifier,
                    &challenge,
                )
                .await
            })
        };
        let authorize = Url::parse(&ui.captured_auth().await).expect("the authorize url parses");
        let presented = authorize
            .query_pairs()
            .find(|(key, _)| key == "state")
            .map(|(_, value)| value.to_string())
            .expect("the authorize url carries the state");
        assert_eq!(presented, expected_state);
        let mut stream = tokio::net::TcpStream::connect(("127.0.0.1", port))
            .await
            .expect("the callback server accepts the redirect");
        stream
            .write_all(
                format!("GET {CALLBACK_PATH}?code=live-code&state={presented} HTTP/1.1\r\nHost: localhost\r\n\r\n")
                    .as_bytes(),
            )
            .await
            .expect("the redirect writes");
        let mut response = String::new();
        let mut buffer = [0u8; 1024];
        loop {
            let read = stream.read(&mut buffer).await.unwrap_or(0);
            if read == 0 || response.contains("</html>") {
                break;
            }
            response.push_str(&String::from_utf8_lossy(&buffer[..read]));
        }
        assert!(response.contains("200 OK"), "{response}");
        let credentials = tokio::time::timeout(Duration::from_secs(10), flow)
            .await
            .expect("the flow settles once the redirect lands")
            .expect("the flow does not panic")
            .expect("the login mints credentials");
        assert_eq!(credentials.access, "the-access");
        let form: HashMap<String, String> = url::form_urlencoded::parse(
            http.bodies_for(TOKEN_URL)[0]
                .body
                .as_ref()
                .expect("the exchange posts a body")
                .as_bytes(),
        )
        .into_owned()
        .collect();
        assert_eq!(form["code"], "live-code");
        assert_eq!(form["code_verifier"], expected_verifier);
    }
}
