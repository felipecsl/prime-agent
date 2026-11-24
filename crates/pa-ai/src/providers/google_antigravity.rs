//! Google AI Pro/Ultra (Antigravity) streaming provider: the Cloud Code Assist
//! `v1internal` backend reached with a Google subscription login. The project id and the
//! advertised runtime model ids are discovered from the backend, thinking effort selects the
//! effort-suffixed runtime model id, and the response SSE is the shared Google stream processor
//! behind the Antigravity `{"response": …}` chunk wrapper.

use std::collections::{BTreeSet, HashMap};
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

use serde_json::{json, Map, Value};
use sha2::{Digest as _, Sha256};

use crate::event_stream::{
    create_assistant_message_event_stream, AssistantMessageEvent, AssistantMessageEventStream,
    AssistantMessageEventWriter,
};
use crate::providers::google::GoogleToolChoice;
use crate::providers::google_shared::{
    convert_messages, convert_tools, map_tool_choice, API_GOOGLE_ANTIGRAVITY,
};
use crate::providers::google_stream::GoogleStreamState;
use crate::providers::simple_options::build_base_options;
use crate::registry::Provider;
use crate::types::{
    done_reason, error_reason, AssistantMessage, Context, Message, Model, ModelExt,
    ModelThinkingLevel, SimpleStreamOptions, StopReason, StreamOptions, Usage,
};
use crate::utils_inner::diagnostics::now_ms;
use crate::utils_inner::http::{send, HttpResponse, RequestOptions};
use crate::utils_inner::json_parse::parse_json_with_repair;
use crate::utils_inner::sanitize_unicode::sanitize_surrogates;
use crate::utils_inner::sse::SseDecoder;
use crate::utils_inner::stream_failure::{
    format_stream_failure_message, record_stream_failure, stream_failure_from_stop_reason,
    ConnectionErrorProfile, ProviderError,
};

/// The primary Antigravity endpoint (the one the CLI and the IDE extension use).
pub const DEFAULT_BASE_URL: &str = "https://daily-cloudcode-pa.googleapis.com";
/// The production Cloud Code Assist endpoint, tried when the primary is unavailable.
pub const FALLBACK_BASE_URL: &str = "https://cloudcode-pa.googleapis.com";

const LOAD_CODE_ASSIST: &str = "v1internal:loadCodeAssist";
const LIST_PROJECTS: &str = "v1internal:listCloudAICompanionProjects";
const FETCH_MODELS: &str = "v1internal:fetchAvailableModels";
const STREAM_PATH: &str = "v1internal:streamGenerateContent?alt=sse";

/// Discovery calls are best-effort and short: a stalled catalog must not hold a turn.
const DISCOVERY_TIMEOUT_MS: u64 = 10_000;
/// Project and model discovery is cached per access token for half an hour.
const DISCOVERY_TTL: Duration = Duration::from_mins(30);

/// The envelope `userAgent` tag; the CLI sends the same constant.
const ENVELOPE_USER_AGENT: &str = "antigravity";
/// The request `User-Agent` fingerprint the Antigravity CLI sends.
const REQUEST_USER_AGENT: &str = "antigravity/cli/1.5.5 (aidev_client; os_type=linux; arch=amd64; cl=982146307; auth_method=consumer)";

/// The effort suffixes the roster's runtime ids carry; an id that already ends with one keeps it.
const TIER_SUFFIXES: [&str; 4] = ["-xhigh", "-high", "-medium", "-low"];

/// Provider-native options.
#[derive(Clone, Default)]
pub struct GoogleAntigravityOptions {
    pub base: StreamOptions,
    pub tool_choice: Option<GoogleToolChoice>,
    /// Thinking effort: selects the runtime model tier and the thinking budget.
    pub thinking_level: Option<ModelThinkingLevel>,
    /// Explicit Cloud AI companion project; otherwise discovered from the backend.
    pub project: Option<String>,
    /// Explicit runtime model id, bypassing tier resolution.
    pub runtime_model: Option<String>,
}

impl GoogleAntigravityOptions {
    pub fn from_base(base: StreamOptions) -> Self {
        Self {
            base,
            tool_choice: None,
            thinking_level: None,
            project: None,
            runtime_model: None,
        }
    }
}

fn env_var(name: &str) -> Option<String> {
    std::env::var(name)
        .ok()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
}

/// The Google OAuth access token pa-core minted at login (or an explicit env token).
fn resolve_access_token(options: &GoogleAntigravityOptions) -> Result<String, String> {
    options
        .base
        .api_key
        .clone()
        .map(|key| key.trim().to_string())
        .filter(|key| !key.is_empty())
        .or_else(|| env_var("GOOGLE_ANTIGRAVITY_API_KEY"))
        .ok_or_else(|| {
            "Google AI Pro/Ultra needs a Google login: sign in with google-antigravity, or set \
             GOOGLE_ANTIGRAVITY_API_KEY to a Google access token."
                .to_string()
        })
}

/// Endpoint candidates in request order: explicit env, the model entry, then the CLI defaults.
fn endpoint_candidates(model: &Model) -> Vec<String> {
    let mut candidates: Vec<String> = Vec::new();
    let mut push = |candidate: Option<String>| {
        if let Some(base) = candidate.map(|value| value.trim().trim_end_matches('/').to_string()) {
            if !base.is_empty() && !base.contains('{') && !candidates.contains(&base) {
                candidates.push(base);
            }
        }
    };
    push(env_var("GOOGLE_ANTIGRAVITY_BASE_URL"));
    push(Some(model.base_url.clone()));
    push(Some(DEFAULT_BASE_URL.to_string()));
    push(Some(FALLBACK_BASE_URL.to_string()));
    candidates
}

/// A UUID-shaped id from a seed, like the CLI's `stableProjectId`/`stableUuid`. pa-ai already
/// depends on `sha2`, so SHA-256 stands in for the CLI's SHA-1 digest.
fn stable_uuid(seed: &str) -> String {
    let digest = Sha256::digest(seed.as_bytes());
    let mut bytes = [0u8; 16];
    bytes.copy_from_slice(&digest[..16]);
    bytes[6] = (bytes[6] & 0x0f) | 0x50;
    bytes[8] = (bytes[8] & 0x3f) | 0x80;
    let hex = hex::encode(bytes);
    format!(
        "{}-{}-{}-{}-{}",
        &hex[..8],
        &hex[8..12],
        &hex[12..16],
        &hex[16..20],
        &hex[20..]
    )
}

/// The `int64` session id the backend groups a conversation's requests under.
fn session_id(seed: &str) -> String {
    let digest = Sha256::digest(format!("antigravity:session:{seed}").as_bytes());
    let mut bytes = [0u8; 8];
    bytes.copy_from_slice(&digest[..8]);
    i64::from_le_bytes(bytes).to_string()
}

/// The seed conversation, trajectory, and session ids derive from: pa-core's session id when the
/// caller supplies one, otherwise the first context message (so a resumed transcript keeps its id).
fn conversation_seed(model: &Model, context: &Context, session_id: Option<&str>) -> String {
    if let Some(id) = session_id.map(str::trim).filter(|id| !id.is_empty()) {
        return format!("session:{id}");
    }
    match context.messages.first() {
        Some(message) => {
            let serialized = serde_json::to_string(message).unwrap_or_default();
            let head: String = serialized.chars().take(64).collect();
            format!("{}:{head}", model.id)
        }
        None => String::new(),
    }
}

/// Per-token project and runtime-model discovery, cached for [`DISCOVERY_TTL`].
#[derive(Clone)]
struct Discovery {
    project: Option<String>,
    models: Vec<String>,
    fetched_at: Instant,
}

impl Discovery {
    fn fresh(project: String, models: Vec<String>) -> Self {
        Self {
            project: Some(project),
            models,
            fetched_at: Instant::now(),
        }
    }
}

fn discovery_cache() -> &'static Mutex<HashMap<String, Discovery>> {
    static CACHE: OnceLock<Mutex<HashMap<String, Discovery>>> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

fn cached_discovery(token: &str) -> Option<Discovery> {
    let cache = discovery_cache().lock().ok()?;
    let entry = cache.get(token)?;
    (entry.fetched_at.elapsed() < DISCOVERY_TTL).then(|| entry.clone())
}

fn store_discovery(token: &str, discovery: Discovery) {
    if let Ok(mut cache) = discovery_cache().lock() {
        cache.retain(|_, entry| entry.fetched_at.elapsed() < DISCOVERY_TTL);
        cache.insert(token.to_string(), discovery);
    }
}

/// POST one JSON body and parse the response; any failure is a soft discovery miss.
async fn post_json(endpoints: &[String], path: &str, token: &str, body: &Value) -> Option<Value> {
    let payload = body.to_string();
    for endpoint in endpoints {
        let request = send(RequestOptions {
            method: reqwest::Method::POST,
            url: format!("{endpoint}/{path}"),
            headers: auth_headers(token),
            body: Some(payload.clone()),
            signal: None,
            timeout_ms: Some(DISCOVERY_TIMEOUT_MS),
            connection: ConnectionErrorProfile::RawFetch,
            transport: crate::utils_inner::http::Transport::Http1,
        })
        .await;
        let Ok(mut response) = request else { continue };
        if response.status >= 400 {
            continue;
        }
        let text = response.read_all_text().await.ok()?;
        if let Ok(parsed) = parse_json_with_repair(&text) {
            return Some(parsed);
        }
    }
    None
}

fn auth_headers(token: &str) -> Vec<(String, String)> {
    vec![
        ("Authorization".into(), format!("Bearer {token}")),
        ("Content-Type".into(), "application/json".into()),
        ("User-Agent".into(), REQUEST_USER_AGENT.into()),
    ]
}

/// Project ids appear under several field names across endpoint versions.
fn extract_project_id(value: &Value) -> Option<String> {
    let object = value.as_object()?;
    for key in [
        "antigravityProjectId",
        "projectId",
        "backendProjectId",
        "userDefinedCloudaicompanionProject",
        "cloudaicompanionProject",
        "project",
    ] {
        if let Some(text) = object.get(key).and_then(Value::as_str) {
            if !text.trim().is_empty() {
                return Some(text.trim().to_string());
            }
        }
    }
    for key in [
        "userDefinedCloudaicompanionProject",
        "cloudaicompanionProject",
        "project",
    ] {
        if let Some(id) = object
            .get(key)
            .and_then(Value::as_object)
            .and_then(|nested| nested.get("id"))
            .and_then(Value::as_str)
        {
            if !id.trim().is_empty() {
                return Some(id.trim().to_string());
            }
        }
    }
    for key in ["projects", "projectIds", "cloudaicompanionProjects"] {
        if let Some(entries) = object.get(key).and_then(Value::as_array) {
            for entry in entries {
                if let Some(id) = extract_project_id(entry).or_else(|| {
                    entry
                        .as_str()
                        .map(str::trim)
                        .filter(|text| !text.is_empty())
                        .map(str::to_string)
                }) {
                    return Some(id);
                }
            }
        }
    }
    None
}

/// Runtime model ids are the keys of the `models` payload; the nested `model` field is a
/// `MODEL_PLACEHOLDER_*` enum that 404s on `streamGenerateContent`.
fn collect_model_ids(value: &Value, out: &mut BTreeSet<String>) {
    match value {
        Value::Object(object) => {
            if let Some(models) = object.get("models").and_then(Value::as_object) {
                for (id, info) in models {
                    if usable_runtime_model(id, info) {
                        out.insert(id.clone());
                    }
                }
            }
            for nested in object.values() {
                collect_model_ids(nested, out);
            }
        }
        Value::Array(entries) => {
            for entry in entries {
                collect_model_ids(entry, out);
            }
        }
        _ => {}
    }
}

fn usable_runtime_model(id: &str, info: &Value) -> bool {
    if id.trim().is_empty() || id.to_lowercase().ends_with("-disabled") {
        return false;
    }
    if let Some(object) = info.as_object() {
        if object.get("disabled").and_then(Value::as_bool) == Some(true)
            || object.get("isInternal").and_then(Value::as_bool) == Some(true)
            || object.get("apiProvider").and_then(Value::as_str) == Some("API_PROVIDER_INTERNAL")
        {
            return false;
        }
    }
    true
}

async fn discover_project(endpoints: &[String], token: &str) -> Option<String> {
    if let Some(payload) = post_json(
        endpoints,
        LOAD_CODE_ASSIST,
        token,
        &json!({ "metadata": { "ideType": "ANTIGRAVITY" } }),
    )
    .await
    {
        if let Some(project) = extract_project_id(&payload) {
            return Some(project);
        }
    }
    post_json(endpoints, LIST_PROJECTS, token, &json!({}))
        .await
        .and_then(|payload| extract_project_id(&payload))
}

async fn discover_model_ids(endpoints: &[String], token: &str, project: &str) -> Vec<String> {
    let payload = post_json(
        endpoints,
        FETCH_MODELS,
        token,
        &json!({ "project": project }),
    )
    .await;
    let Some(payload) = payload else {
        return Vec::new();
    };
    let mut ids = BTreeSet::new();
    collect_model_ids(&payload, &mut ids);
    ids.into_iter().collect()
}

/// The tier suffix for an effort level; `off` keeps the base model id.
fn suffix_for(effort: ModelThinkingLevel) -> Option<&'static str> {
    match effort {
        ModelThinkingLevel::Off => None,
        ModelThinkingLevel::Minimal | ModelThinkingLevel::Low => Some("-low"),
        ModelThinkingLevel::Medium => Some("-medium"),
        ModelThinkingLevel::High => Some("-high"),
        ModelThinkingLevel::Xhigh | ModelThinkingLevel::Max => Some("-xhigh"),
    }
}

/// Discovery search order for one effort level: the requested tier first, then the neighbours.
fn effort_suffix_order(effort: ModelThinkingLevel) -> Vec<&'static str> {
    let mut order: Vec<&'static str> = Vec::new();
    let mut push = |suffix: Option<&'static str>| {
        if let Some(suffix) = suffix {
            if !order.contains(&suffix) {
                order.push(suffix);
            }
        }
    };
    match effort {
        ModelThinkingLevel::Xhigh | ModelThinkingLevel::Max => {
            for level in [
                ModelThinkingLevel::Xhigh,
                ModelThinkingLevel::High,
                ModelThinkingLevel::Medium,
                ModelThinkingLevel::Low,
            ] {
                push(suffix_for(level));
            }
        }
        ModelThinkingLevel::High => {
            for level in [
                ModelThinkingLevel::High,
                ModelThinkingLevel::Xhigh,
                ModelThinkingLevel::Medium,
                ModelThinkingLevel::Low,
            ] {
                push(suffix_for(level));
            }
        }
        ModelThinkingLevel::Medium => {
            for level in [
                ModelThinkingLevel::Medium,
                ModelThinkingLevel::Low,
                ModelThinkingLevel::High,
                ModelThinkingLevel::Xhigh,
            ] {
                push(suffix_for(level));
            }
        }
        _ => {
            for level in [
                ModelThinkingLevel::Low,
                ModelThinkingLevel::Medium,
                ModelThinkingLevel::High,
                ModelThinkingLevel::Xhigh,
            ] {
                push(suffix_for(level));
            }
        }
    }
    order
}

/// An effort level the model entry does not advertise falls back one tier: a catalog without
/// `xhigh` (Gemini 4 Argon, for one) answers `gemini-4-argon-xhigh` with a 400.
fn clamp_effort(model: &Model, effort: ModelThinkingLevel) -> ModelThinkingLevel {
    if effort == ModelThinkingLevel::Xhigh
        && matches!(
            model.thinking_level_map_value(ModelThinkingLevel::Xhigh),
            Some(None)
        )
    {
        return ModelThinkingLevel::High;
    }
    effort
}

/// Resolve the public catalog id plus effort to a runtime model id the backend advertises. With
/// no discovery, the tier is appended to the catalog id (the grouped-model naming convention).
fn resolve_runtime_model(base: &str, effort: ModelThinkingLevel, discovered: &[String]) -> String {
    let carries_tier = TIER_SUFFIXES.iter().any(|suffix| base.ends_with(suffix));
    if carries_tier {
        return base.to_string();
    }
    let Some(suffix) = suffix_for(effort) else {
        return base.to_string();
    };
    if discovered.is_empty() {
        return format!("{base}{suffix}");
    }
    let lowered: HashMap<String, String> = discovered
        .iter()
        .map(|id| (id.to_lowercase(), id.clone()))
        .collect();
    let lower_base = base.to_lowercase();
    if let Some(exact) = lowered.get(&lower_base) {
        return exact.clone();
    }
    for candidate in effort_suffix_order(effort) {
        if let Some(found) = lowered.get(&format!("{lower_base}{candidate}")) {
            return found.clone();
        }
    }
    base.to_string()
}

/// `thinkingConfig` for the Antigravity tiers: `off` sends no thinking config at all, and the
/// effort selects a budget (`-1` means the model decides).
fn thinking_config(effort: ModelThinkingLevel) -> Option<Value> {
    let budget = match effort {
        ModelThinkingLevel::Off => return None,
        ModelThinkingLevel::Minimal | ModelThinkingLevel::Low => 1_000,
        ModelThinkingLevel::Medium => 4_000,
        ModelThinkingLevel::High | ModelThinkingLevel::Xhigh | ModelThinkingLevel::Max => -1,
    };
    Some(json!({ "includeThoughts": true, "thinkingBudget": budget }))
}

/// The `model_enum` label the CLI attaches when it knows the backend's model enum.
const MODEL_ENUMS: [(&str, &str); 14] = [
    ("gemini-4-neon", "MODEL_PLACEHOLDER_M534"),
    ("gemini-4-neon-low", "MODEL_PLACEHOLDER_M534"),
    ("gemini-4-neon-medium", "MODEL_PLACEHOLDER_M533"),
    ("gemini-4-neon-high", "MODEL_PLACEHOLDER_M532"),
    ("gemini-4-neon-xhigh", "MODEL_PLACEHOLDER_M535"),
    ("gemini-4-argon", "MODEL_PLACEHOLDER_M419"),
    ("gemini-4-argon-low", "MODEL_PLACEHOLDER_M419"),
    ("gemini-4-argon-medium", "MODEL_PLACEHOLDER_M418"),
    ("gemini-4-argon-high", "MODEL_PLACEHOLDER_M417"),
    ("gemini-4-hydrogen", "MODEL_PLACEHOLDER_M527"),
    ("gemini-4-hydrogen-low", "MODEL_PLACEHOLDER_M527"),
    ("gemini-4-hydrogen-medium", "MODEL_PLACEHOLDER_M526"),
    ("gemini-4-hydrogen-high", "MODEL_PLACEHOLDER_M525"),
    ("gemini-4-hydrogen-xhigh", "MODEL_PLACEHOLDER_M528"),
];

fn model_enum(runtime_model: &str) -> Option<&'static str> {
    MODEL_ENUMS
        .iter()
        .find(|(id, _)| *id == runtime_model)
        .map(|(_, value)| *value)
}

/// The request labels and ids Antigravity ties a trajectory of requests together with.
fn envelope_ids(
    runtime_model: &str,
    contents_len: usize,
    assistant_turns: usize,
    seed: &str,
    timestamp_ms: u64,
) -> (String, String, Map<String, Value>) {
    let step = contents_len.max(1);
    let seed = if seed.is_empty() {
        stable_uuid(&format!("anon:{timestamp_ms}"))
    } else {
        seed.to_string()
    };
    let conversation_id = stable_uuid(&format!("antigravity:conv:{seed}"));
    let trajectory_id = stable_uuid(&format!("antigravity:traj:{seed}"));
    let mut labels = Map::new();
    labels.insert("last_step_index".into(), json!((step - 1).to_string()));
    labels.insert(
        "request_id".into(),
        json!(format!("{trajectory_id}-{assistant_turns}")),
    );
    labels.insert("trajectory_id".into(), json!(&trajectory_id));
    labels.insert("used_claude".into(), json!("false"));
    labels.insert("used_claude_conservative".into(), json!("false"));
    labels.insert("used_non_gemini_model".into(), json!("false"));
    if let Some(enum_value) = model_enum(runtime_model) {
        labels.insert("model_enum".into(), json!(enum_value));
    }
    // The CLI labels the previous turn's execution id only after the first request.
    if step > 1 {
        labels.insert(
            "last_execution_id".into(),
            json!(stable_uuid(&format!(
                "antigravity:exec:{trajectory_id}:{}",
                step - 1
            ))),
        );
    }
    let request_id = format!("agent/{conversation_id}/{timestamp_ms}/{trajectory_id}/{step}");
    (request_id, session_id(&seed), labels)
}

/// The `v1internal:streamGenerateContent` envelope: the Gemini request body plus the project,
/// runtime model, and Antigravity request metadata.
fn build_params(
    model: &Model,
    context: &Context,
    options: &GoogleAntigravityOptions,
    runtime_model: &str,
    project: &str,
    effort: ModelThinkingLevel,
    seed: &str,
) -> Value {
    let contents = convert_messages(model, context);

    let mut generation_config = Map::new();
    if let Some(temperature) = options.base.temperature {
        generation_config.insert("temperature".into(), json!(temperature));
    }
    if let Some(max_tokens) = options.base.max_tokens {
        generation_config.insert("maxOutputTokens".into(), json!(max_tokens));
    }
    if let Some(thinking) = thinking_config(effort) {
        generation_config.insert("thinkingConfig".into(), thinking);
    }

    let mut request = Map::new();
    request.insert("contents".into(), json!(contents));
    if !generation_config.is_empty() {
        request.insert("generationConfig".into(), Value::Object(generation_config));
    }
    if let Some(system_prompt) = &context.system_prompt {
        request.insert(
            "systemInstruction".into(),
            json!({
                "role": "user",
                "parts": [{ "text": sanitize_surrogates(system_prompt) }],
            }),
        );
    }
    let has_tools = context
        .tools
        .as_ref()
        .is_some_and(|tools| !tools.is_empty());
    if has_tools {
        if let Some(tools) = context
            .tools
            .as_ref()
            .and_then(|tools| convert_tools(tools, false))
        {
            request.insert("tools".into(), json!(tools));
        }
        if let Some(tool_choice) = options.tool_choice {
            request.insert(
                "toolConfig".into(),
                json!({
                    "functionCallingConfig": {
                        "mode": map_tool_choice(tool_choice.as_str()),
                    },
                }),
            );
        }
    }

    let assistant_turns = context
        .messages
        .iter()
        .filter(|message| {
            matches!(
                message,
                Message::Assistant(assistant)
                    if !matches!(assistant.stop_reason, StopReason::Error | StopReason::Aborted)
            )
        })
        .count();
    let (request_id, session_id, labels) = envelope_ids(
        runtime_model,
        contents.len(),
        assistant_turns,
        seed,
        now_ms(),
    );
    request.insert("sessionId".into(), json!(session_id));
    request.insert("labels".into(), Value::Object(labels));

    json!({
        "project": project,
        "model": runtime_model,
        "request": Value::Object(request),
        "requestType": "agent",
        "userAgent": ENVELOPE_USER_AGENT,
        "requestId": request_id,
    })
}

/// Parse one SSE payload and unwrap the Antigravity envelope.
fn parse_sse_chunk(data: &str) -> Result<Value, ProviderError> {
    let parsed = parse_json_with_repair(data).map_err(|error| {
        ProviderError::Message(format!("Could not parse Antigravity SSE chunk: {error}"))
    })?;
    unwrap_response_chunk(parsed).map_err(ProviderError::Message)
}

/// The stream request's headers. The Antigravity `User-Agent` is not cosmetic: the backend
/// resolves the client's license tier from it, and a request without it is answered `403 You do
/// not have a valid license of this product` (the discovery calls carry it through
/// [`auth_headers`], which is why only the stream call failed).
fn stream_headers(
    model: &Model,
    options: &GoogleAntigravityOptions,
    token: &str,
) -> Vec<(String, String)> {
    let mut headers: Vec<(String, String)> = model
        .headers
        .iter()
        .flatten()
        .map(|(name, value)| (name.clone(), value.clone()))
        .collect();
    if let Some(extra) = &options.base.headers {
        for (name, value) in extra {
            headers.retain(|(existing, _)| !existing.eq_ignore_ascii_case(name));
            headers.push((name.clone(), value.clone()));
        }
    }
    if !headers
        .iter()
        .any(|(name, _)| name.eq_ignore_ascii_case("user-agent"))
    {
        headers.push(("User-Agent".into(), REQUEST_USER_AGENT.into()));
    }
    headers.push(("Authorization".into(), format!("Bearer {token}")));
    headers
}

/// Antigravity wraps each SSE payload as `{"response": …}` and reports backend errors inline as
/// `{"error": …}`.
fn unwrap_response_chunk(chunk: Value) -> Result<Value, String> {
    if let Some(error) = chunk.get("error") {
        let message = error
            .get("message")
            .and_then(Value::as_str)
            .filter(|text| !text.trim().is_empty())
            .unwrap_or("Antigravity reported an error");
        return Err(message.to_string());
    }
    match chunk.get("response") {
        Some(response) if response.is_object() => Ok(response.clone()),
        _ => Ok(chunk),
    }
}

pub fn stream_google_antigravity(
    model: &Model,
    context: &Context,
    options: Option<&GoogleAntigravityOptions>,
) -> AssistantMessageEventStream {
    let options = options.cloned();
    let model = model.clone();
    let context = context.clone();
    let (writer, reader) = create_assistant_message_event_stream();

    tokio::spawn(async move {
        let mut output = AssistantMessage {
            content: Vec::new(),
            api: API_GOOGLE_ANTIGRAVITY.to_string(),
            provider: model.provider.clone(),
            model: model.id.clone(),
            response_model: None,
            response_id: None,
            diagnostics: None,
            usage: Usage::default(),
            stop_reason: StopReason::Stop,
            stop_reason_raw: None,
            error_message: None,
            timestamp: now_ms(),
            rest: Map::default(),
        };

        let result = run_stream(&model, &context, options.as_ref(), &mut output, &writer).await;
        match result {
            Ok(()) => {
                writer.push(AssistantMessageEvent::Done {
                    reason: done_reason(output.stop_reason),
                    message: output,
                });
                writer.end(None);
            }
            Err(error) => {
                output.stop_reason = if error == ProviderError::Aborted {
                    StopReason::Aborted
                } else {
                    StopReason::Error
                };
                output.error_message = Some(format_stream_failure_message(&error));
                record_stream_failure(
                    (&model.provider, &model.id, &model.api),
                    &mut output,
                    &error,
                );
                writer.push(AssistantMessageEvent::Error {
                    reason: error_reason(output.stop_reason),
                    error: output.clone(),
                });
                writer.end(Some(output));
            }
        }
    });

    reader
}

// Long by design: the discovery, failover, and SSE stages each stay in one place.
#[allow(clippy::too_many_lines)]
async fn run_stream(
    model: &Model,
    context: &Context,
    options: Option<&GoogleAntigravityOptions>,
    output: &mut AssistantMessage,
    writer: &AssistantMessageEventWriter,
) -> Result<(), ProviderError> {
    let options = options.cloned().unwrap_or_default();

    if options
        .base
        .signal
        .as_ref()
        .is_some_and(tokio_util::sync::CancellationToken::is_cancelled)
    {
        return Err(ProviderError::Aborted);
    }

    let token = resolve_access_token(&options).map_err(ProviderError::Message)?;
    let endpoints = endpoint_candidates(model);

    // Project: the caller's or the environment's, else discovered (and cached) per token.
    let cached = cached_discovery(&token);
    let mut project = options
        .project
        .clone()
        .or_else(|| env_var("GOOGLE_ANTIGRAVITY_PROJECT_ID"))
        .or_else(|| cached.as_ref().and_then(|entry| entry.project.clone()));
    if project.is_none() {
        project = discover_project(&endpoints, &token).await;
    }
    let project = project
        .or_else(|| env_var("GOOGLE_CLOUD_PROJECT"))
        // The CLI falls back to a stable pseudo-project id when onboarding is unknown.
        .unwrap_or_else(|| stable_uuid("antigravity:project:prime-agent"));

    let mut models = cached.map_or_else(Vec::new, |entry| entry.models);
    if models.is_empty() {
        models = discover_model_ids(&endpoints, &token, &project).await;
    }
    store_discovery(&token, Discovery::fresh(project.clone(), models.clone()));

    let effort = options
        .thinking_level
        .map_or(ModelThinkingLevel::Off, |level| clamp_effort(model, level));
    let runtime_model = env_var("GOOGLE_ANTIGRAVITY_RUNTIME_MODEL")
        .or_else(|| options.runtime_model.clone())
        .unwrap_or_else(|| resolve_runtime_model(&model.id, effort, &models));

    let seed = conversation_seed(model, context, options.base.session_id.as_deref());
    let mut params = build_params(
        model,
        context,
        &options,
        &runtime_model,
        &project,
        effort,
        &seed,
    );
    if let Some(on_payload) = &options.base.on_payload {
        if let Some(next) = on_payload(params.clone(), model) {
            params = next;
        }
    }

    let headers = stream_headers(model, &options, &token);

    // Endpoint candidates are tried in order. Only a connection failure or a 5xx moves to the
    // next candidate: a 4xx (including a quota wall) fails at once rather than billing twice.
    let mut response: Option<HttpResponse> = None;
    let mut last_status: Option<(u16, String)> = None;
    let mut last_error: Option<ProviderError> = None;
    for endpoint in &endpoints {
        let attempt = send(RequestOptions {
            method: reqwest::Method::POST,
            url: format!("{endpoint}/{STREAM_PATH}"),
            headers: headers.clone(),
            body: Some(params.to_string()),
            signal: options.base.signal.clone(),
            timeout_ms: options.base.timeout_ms,
            connection: ConnectionErrorProfile::RawFetch,
            transport: crate::utils_inner::http::Transport::Http1,
        })
        .await;
        match attempt {
            Ok(attempt) if attempt.status < 400 => {
                response = Some(attempt);
                break;
            }
            Ok(mut attempt) if attempt.status >= 500 => {
                last_status = Some((
                    attempt.status,
                    attempt.read_all_text().await.unwrap_or_default(),
                ));
            }
            Ok(mut attempt) => {
                let body = attempt.read_all_text().await.unwrap_or_default();
                return Err(ProviderError::from_http_status_body(
                    attempt.status,
                    &body,
                    attempt.headers.clone(),
                ));
            }
            Err(error) => last_error = Some(error),
        }
    }

    let Some(mut response) = response else {
        if let Some((status, body)) = last_status {
            return Err(ProviderError::from_http_status_body(
                status,
                &body,
                std::collections::HashMap::new(),
            ));
        }
        return Err(last_error.unwrap_or(ProviderError::Message(
            "Antigravity request failed before a response".to_string(),
        )));
    };

    if let Some(on_response) = &options.base.on_response {
        on_response(
            crate::types::ProviderResponse {
                status: response.status,
                headers: response
                    .headers
                    .iter()
                    .map(|(name, value)| (name.clone(), value.clone()))
                    .collect(),
            },
            model,
        );
    }

    writer.push(AssistantMessageEvent::Start {
        partial: output.clone(),
    });

    let mut state = GoogleStreamState::new();
    let mut decoder = SseDecoder::new();
    loop {
        let Some(chunk) = response.next_text().await? else {
            break;
        };
        for sse in decoder.push_text(&chunk) {
            if sse.data.trim().is_empty() {
                continue;
            }
            let parsed = parse_sse_chunk(&sse.data)?;
            state
                .handle_chunk(&parsed, model, output, writer)
                .map_err(ProviderError::Message)?;
        }
    }
    for sse in decoder.finish() {
        if sse.data.trim().is_empty() {
            continue;
        }
        let parsed = parse_sse_chunk(&sse.data)?;
        state
            .handle_chunk(&parsed, model, output, writer)
            .map_err(ProviderError::Message)?;
    }
    state.finish(output, writer);

    if options
        .base
        .signal
        .as_ref()
        .is_some_and(tokio_util::sync::CancellationToken::is_cancelled)
    {
        return Err(ProviderError::Aborted);
    }
    if matches!(output.stop_reason, StopReason::Aborted | StopReason::Error) {
        return Err(ProviderError::StreamFailure(
            stream_failure_from_stop_reason(output.stop_reason_raw.as_deref(), None),
        ));
    }

    Ok(())
}

pub fn stream_simple_google_antigravity(
    model: &Model,
    context: &Context,
    options: Option<&SimpleStreamOptions>,
) -> AssistantMessageEventStream {
    let base = build_base_options(model, options, None);
    let stream_options = GoogleAntigravityOptions {
        thinking_level: options.and_then(|options| options.reasoning),
        base,
        ..GoogleAntigravityOptions::default()
    };
    stream_google_antigravity(model, context, Some(&stream_options))
}

/// Registry provider for the `google-antigravity` API.
pub struct GoogleAntigravityProvider;

impl Provider for GoogleAntigravityProvider {
    fn api(&self) -> &str {
        API_GOOGLE_ANTIGRAVITY
    }

    fn stream(
        &self,
        model: &Model,
        context: &Context,
        options: Option<&StreamOptions>,
    ) -> AssistantMessageEventStream {
        let options = options.map(|base| GoogleAntigravityOptions::from_base(base.clone()));
        stream_google_antigravity(model, context, options.as_ref())
    }

    fn stream_simple(
        &self,
        model: &Model,
        context: &Context,
        options: Option<&SimpleStreamOptions>,
    ) -> AssistantMessageEventStream {
        stream_simple_google_antigravity(model, context, options)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn neon_model() -> Model {
        serde_json::from_value::<Model>(json!({
            "id": "gemini-4-neon", "name": "Gemini 4 Neon (Antigravity)",
            "api": "google-antigravity", "provider": "google-antigravity",
            "baseUrl": "https://daily-cloudcode-pa.googleapis.com",
            "reasoning": true, "input": ["text"],
            "cost": { "input": 3, "output": 15, "cacheRead": 0.15, "cacheWrite": 0 },
            "contextWindow": 1_048_576, "maxTokens": 65_536,
            "thinkingLevelMap": { "low": "low", "medium": "medium", "high": "high", "xhigh": "xhigh" }
        }))
        .unwrap()
    }

    fn argon_model() -> Model {
        let mut model = neon_model();
        model.id = "gemini-4-argon".into();
        model.thinking_level_map = None;
        model
    }

    fn options(api_key: &str) -> GoogleAntigravityOptions {
        GoogleAntigravityOptions {
            base: StreamOptions {
                api_key: Some(api_key.to_string()),
                ..StreamOptions::default()
            },
            project: Some("project-123".into()),
            ..GoogleAntigravityOptions::default()
        }
    }

    #[test]
    fn endpoints_are_tried_in_cli_order() {
        let candidates = endpoint_candidates(&neon_model());
        assert_eq!(candidates[0], "https://daily-cloudcode-pa.googleapis.com");
        assert_eq!(candidates[1], FALLBACK_BASE_URL);
        assert_eq!(
            candidates
                .iter()
                .filter(|base| *base == DEFAULT_BASE_URL)
                .count(),
            1,
            "the model entry and the default collapse to one candidate"
        );
    }

    #[test]
    fn a_missing_login_names_the_sign_in() {
        let error = resolve_access_token(&GoogleAntigravityOptions::default())
            .expect_err("no token without a login");
        assert!(error.contains("google-antigravity"), "{error}");
    }

    #[test]
    fn thinking_config_follows_the_cli_budgets() {
        assert_eq!(thinking_config(ModelThinkingLevel::Off), None);
        assert_eq!(
            thinking_config(ModelThinkingLevel::Low),
            Some(json!({ "includeThoughts": true, "thinkingBudget": 1_000 }))
        );
        assert_eq!(
            thinking_config(ModelThinkingLevel::Medium),
            Some(json!({ "includeThoughts": true, "thinkingBudget": 4_000 }))
        );
        assert_eq!(
            thinking_config(ModelThinkingLevel::Xhigh),
            Some(json!({ "includeThoughts": true, "thinkingBudget": -1 }))
        );
    }

    #[test]
    fn runtime_ids_take_the_effort_tier_from_the_catalog() {
        let discovered = [
            "gemini-4-neon-low".to_string(),
            "gemini-4-neon-medium".to_string(),
            "gemini-4-neon-high".to_string(),
        ];
        // Without discovery the grouped naming convention still resolves the tier.
        assert_eq!(
            resolve_runtime_model("gemini-4-neon", ModelThinkingLevel::High, &[]),
            "gemini-4-neon-high"
        );
        // A missing top tier walks down to the nearest advertised one.
        assert_eq!(
            resolve_runtime_model("gemini-4-neon", ModelThinkingLevel::Xhigh, &discovered),
            "gemini-4-neon-high"
        );
        assert_eq!(
            resolve_runtime_model("gemini-4-neon", ModelThinkingLevel::Medium, &discovered),
            "gemini-4-neon-medium"
        );
        // `off` keeps the base id, and an already-tiered id is left alone.
        assert_eq!(
            resolve_runtime_model("gemini-4-neon", ModelThinkingLevel::Off, &discovered),
            "gemini-4-neon"
        );
        assert_eq!(
            resolve_runtime_model("gemini-4-neon-high", ModelThinkingLevel::Low, &discovered),
            "gemini-4-neon-high"
        );
    }

    #[test]
    fn an_unadvertised_xhigh_falls_back_to_high() {
        let mut model = neon_model();
        model.thinking_level_map = Some(
            [
                (ModelThinkingLevel::Low, Some("low".to_string())),
                (ModelThinkingLevel::Medium, Some("medium".to_string())),
                (ModelThinkingLevel::High, Some("high".to_string())),
            ]
            .into_iter()
            .collect(),
        );
        assert_eq!(
            clamp_effort(&model, ModelThinkingLevel::Xhigh),
            ModelThinkingLevel::High
        );
        assert_eq!(
            clamp_effort(&argon_model(), ModelThinkingLevel::Xhigh),
            ModelThinkingLevel::Xhigh,
            "no thinking map means no tier knowledge"
        );
    }

    #[test]
    fn the_envelope_carries_the_project_runtime_model_and_labels() {
        let model = neon_model();
        let context = Context {
            system_prompt: Some("Answer briefly.".into()),
            messages: vec![Message::User(
                serde_json::from_value(json!({
                    "content": [{ "type": "text", "text": "hi" }], "timestamp": 1
                }))
                .unwrap(),
            )],
            tools: None,
        };
        let params = build_params(
            &model,
            &context,
            &options("token"),
            "gemini-4-neon-high",
            "project-123",
            ModelThinkingLevel::High,
            "session:abc",
        );
        assert_eq!(params["project"], "project-123");
        assert_eq!(params["model"], "gemini-4-neon-high");
        assert_eq!(params["requestType"], "agent");
        assert_eq!(params["userAgent"], "antigravity");
        assert!(params["requestId"].as_str().unwrap().starts_with("agent/"));
        let labels = params["request"]["labels"].as_object().unwrap();
        assert_eq!(labels["used_non_gemini_model"], "false");
        assert_eq!(labels["last_step_index"], "0");
        assert_eq!(
            labels["request_id"],
            format!("{}-0", labels["trajectory_id"].as_str().unwrap())
        );
        assert_eq!(labels["model_enum"], "MODEL_PLACEHOLDER_M532");
        assert!(!labels.contains_key("last_execution_id"));
        let session = params["request"]["sessionId"].as_str().unwrap();
        assert!(session.parse::<i64>().is_ok(), "{session} is an int64");
        assert_eq!(
            params["request"]["systemInstruction"]["parts"][0]["text"],
            "Answer briefly."
        );
        assert_eq!(
            params["request"]["generationConfig"]["thinkingConfig"],
            json!({ "includeThoughts": true, "thinkingBudget": -1 })
        );
        // The same session seed keeps one trajectory across requests.
        let again = build_params(
            &model,
            &context,
            &options("token"),
            "gemini-4-neon-high",
            "project-123",
            ModelThinkingLevel::High,
            "session:abc",
        );
        assert_eq!(
            again["request"]["labels"]["trajectory_id"],
            labels["trajectory_id"]
        );
    }

    #[test]
    fn later_steps_label_the_previous_execution() {
        let (_, _, labels) = envelope_ids("gemini-4-neon", 3, 1, "session:abc", 1_700_000_000_000);
        assert_eq!(labels["last_step_index"], "2");
        assert!(labels.contains_key("last_execution_id"));
    }

    #[test]
    fn sse_chunks_unwrap_the_response_and_surface_errors() {
        let wrapped = unwrap_response_chunk(json!({
            "response": { "candidates": [{ "content": { "parts": [{ "text": "hi" }] } }] }
        }))
        .unwrap();
        assert_eq!(
            wrapped["candidates"][0]["content"]["parts"][0]["text"],
            "hi"
        );
        let bare = unwrap_response_chunk(json!({ "candidates": [] })).unwrap();
        assert!(bare["candidates"].is_array());
        let error = unwrap_response_chunk(json!({
            "error": { "code": 429, "message": "Individual quota reached. Resets in 4h." }
        }))
        .expect_err("an inline error ends the stream");
        assert!(error.contains("Individual quota reached"), "{error}");
    }

    #[test]
    fn project_ids_are_read_from_the_onboarding_shapes() {
        assert_eq!(
            extract_project_id(&json!({ "antigravityProjectId": "agy-project" })),
            Some("agy-project".to_string())
        );
        assert_eq!(
            extract_project_id(&json!({
                "userDefinedCloudaicompanionProject": { "id": "nested-project" }
            })),
            Some("nested-project".to_string())
        );
        assert_eq!(
            extract_project_id(&json!({ "cloudaicompanionProjects": ["first-project"] })),
            Some("first-project".to_string())
        );
        assert_eq!(
            extract_project_id(&json!({ "onboardingState": "ONBOARDING" })),
            None
        );
    }

    #[test]
    fn internal_and_disabled_runtimes_are_skipped() {
        let payload = json!({
            "models": {
                "gemini-4-neon-high": { "model": "MODEL_PLACEHOLDER_M532" },
                "gemini-4-neon-xhigh": { "disabled": true },
                "internal-thing": { "isInternal": true },
                "hidden": { "apiProvider": "API_PROVIDER_INTERNAL" },
                "gemini-4-carbon-disabled": {}
            }
        });
        let mut ids = BTreeSet::new();
        collect_model_ids(&payload, &mut ids);
        assert_eq!(
            ids,
            ["gemini-4-neon-high".to_string()].into_iter().collect()
        );
    }

    #[test]
    fn the_stream_request_carries_the_antigravity_user_agent() {
        let headers = stream_headers(&neon_model(), &options("token"), "token");
        let user_agent = headers
            .iter()
            .find(|(name, _)| name.eq_ignore_ascii_case("user-agent"))
            .map(|(_, value)| value.as_str());
        assert_eq!(user_agent, Some(REQUEST_USER_AGENT));
        assert!(headers
            .iter()
            .any(|(name, value)| name == "Authorization" && value == "Bearer token"));
    }

    #[test]
    fn a_caller_user_agent_wins() {
        let mut options = options("token");
        options.base.headers = Some(
            [("user-agent".to_string(), "custom-agent/1".to_string())]
                .into_iter()
                .collect(),
        );
        let headers = stream_headers(&neon_model(), &options, "token");
        let agents: Vec<&str> = headers
            .iter()
            .filter(|(name, _)| name.eq_ignore_ascii_case("user-agent"))
            .map(|(_, value)| value.as_str())
            .collect();
        assert_eq!(agents, ["custom-agent/1"]);
    }

    #[test]
    fn stable_ids_are_uuid_shaped_and_repeatable() {
        let first = stable_uuid("antigravity:project:prime-agent");
        assert_eq!(first, stable_uuid("antigravity:project:prime-agent"));
        assert_eq!(first.len(), 36);
        assert_eq!(&first[14..15], "5");
        assert_eq!(first.matches('-').count(), 4);
    }
}
