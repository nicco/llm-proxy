//! Forge-style tool-call fortification (validate → rescue → retry).
//!
//! Applies only to opted-in models on POST …/chat/completions requests that
//! carry a non-empty `tools` array — every other request stays on the
//! zero-parse forwarding path.  See `stream.rs` for how live streaming is
//! preserved (hold-and-release) and the README for the retry/rescue matrix.

mod rescue;
mod schema;
mod stream;

use axum::body::Body;
use axum::http::Response;
use bytes::{Bytes, BytesMut};
use http_body_util::BodyExt;
use hyper_util::client::legacy::{connect::HttpConnector, Client};
use serde_json::{json, Value};
use std::collections::HashSet;

use crate::config::{FortifyConfig, FortifyMode, ModelConfig};
use crate::forward::{rewrite_model_bytes, send_upstream, skip_response_header};
use rescue::ToolCall;
use stream::{CommittedStream, HoldMachine, MessageAccumulator};

pub(crate) const RESPOND_TOOL: &str = "respond";

pub(crate) struct RunCtx<'a> {
    pub upstream_url: &'a str,
    pub model_cfg: &'a ModelConfig,
    pub parts: &'a axum::http::request::Parts,
    pub client: &'a Client<HttpConnector, Body>,
    pub header_timeout: Option<std::time::Duration>,
    pub req_id: u64,
}

/// Body-level gate for the fortified path.  The caller has already checked
/// method, path, and that fortify is enabled for the model.
pub(crate) fn applies_parsed(body_json: &Value) -> bool {
    let has_tools = body_json
        .get("tools")
        .and_then(|v| v.as_array())
        .map(|t| !t.is_empty())
        .unwrap_or(false);
    // Multi-choice responses are not fortified.
    let single = body_json.get("n").and_then(|v| v.as_u64()).unwrap_or(1) == 1;
    has_tools && single
}

/// Standard per-model body mutation: overwrite the model name and merge the
/// configured params (config always wins).  Shared by the fortify path and
/// the inject-only path.
pub(crate) fn apply_model_params(body: &mut Value, cfg: &ModelConfig) {
    if let Some(obj) = body.as_object_mut() {
        obj.insert("model".into(), Value::String(cfg.served_model.clone()));
        for (k, v) in &cfg.params {
            obj.insert(k.clone(), v.clone());
        }
    }
}

/// Finalize a parsed (possibly inject-mutated) body for plain forwarding.
pub(crate) fn finalize_plain_body(mut body: Value, cfg: &ModelConfig) -> Bytes {
    apply_model_params(&mut body, cfg);
    serde_json::to_vec(&body)
        .map(Bytes::from)
        .unwrap_or_default()
}

pub(crate) async fn run(
    ctx: RunCtx<'_>,
    body_json: Value,
) -> Result<Response<Body>, anyhow::Error> {
    let f = ctx
        .model_cfg
        .fortify
        .as_ref()
        .expect("caller checked fortify");
    let fort = Fortifier::new(body_json, ctx.model_cfg, f);

    // Completion-check candidates are routed through the buffered path so
    // the check can run before any bytes reach the client.
    if f.mode == FortifyMode::Hold && fort.client_stream && !fort.check_candidate {
        run_hold_streaming(&ctx, f, fort).await
    } else {
        run_buffered(&ctx, f, fort, 0).await
    }
}

// ─── Request preparation ────────────────────────────────────────────────

/// Owns the (already model/params/respond-mutated) request body across the
/// retry loop and knows how to serialize per-attempt variants.
pub(crate) struct Fortifier {
    body: Value,
    pub tool_names: HashSet<String>,
    /// Tool name → `function.parameters` JSON Schema, for argument checks.
    pub tool_schemas: std::collections::HashMap<String, Value>,
    pub client_stream: bool,
    pub include_usage: bool,
    expect_tool: bool,
    /// Completion check is enabled and the conversation already contains
    /// tool results — a plain-text response gets one verification round.
    pub check_candidate: bool,
}

impl Fortifier {
    fn new(mut body: Value, cfg: &ModelConfig, f: &FortifyConfig) -> Self {
        let client_stream = body
            .get("stream")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        let include_usage = body
            .pointer("/stream_options/include_usage")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        apply_model_params(&mut body, cfg);
        // The model is expected to call a tool when the synthetic respond
        // tool is in play, or when the client demanded one outright.
        let tool_choice_required =
            body.get("tool_choice").and_then(|v| v.as_str()) == Some("required");
        if f.inject_respond_tool {
            if let Some(tools) = body.get_mut("tools").and_then(|v| v.as_array_mut()) {
                tools.push(respond_spec());
            }
        }
        let tool_names: HashSet<String> = body
            .get("tools")
            .and_then(|v| v.as_array())
            .map(|tools| {
                tools
                    .iter()
                    .filter_map(|t| t.pointer("/function/name")?.as_str().map(String::from))
                    .collect()
            })
            .unwrap_or_default();
        let tool_schemas: std::collections::HashMap<String, Value> = if f.validate_args {
            body.get("tools")
                .and_then(|v| v.as_array())
                .map(|tools| {
                    tools
                        .iter()
                        .filter_map(|t| {
                            let name = t.pointer("/function/name")?.as_str()?.to_string();
                            let params = t.pointer("/function/parameters")?.clone();
                            Some((name, params))
                        })
                        .collect()
                })
                .unwrap_or_default()
        } else {
            Default::default()
        };
        let check_candidate = f.completion_check
            && body
                .get("messages")
                .and_then(|v| v.as_array())
                .map(|msgs| {
                    msgs.iter()
                        .any(|m| m.get("role").and_then(|r| r.as_str()) == Some("tool"))
                })
                .unwrap_or(false);
        Self {
            body,
            tool_names,
            tool_schemas,
            client_stream,
            include_usage,
            expect_tool: f.inject_respond_tool || tool_choice_required,
            check_candidate,
        }
    }

    /// Schema-check every tool call in a validated message.  Returns a
    /// corrective nudge when any call violates its tool's parameter schema.
    fn schema_nudge(&self, message: &Value) -> Option<String> {
        let calls = message.get("tool_calls")?.as_array()?;
        let mut problems = Vec::new();
        for tc in calls {
            let name = tc.pointer("/function/name")?.as_str()?;
            let Some(sch) = self.tool_schemas.get(name) else {
                continue;
            };
            let args = match tc.pointer("/function/arguments") {
                Some(Value::String(s)) if s.trim().is_empty() => json!({}),
                Some(Value::String(s)) => serde_json::from_str(s).ok()?,
                Some(v @ Value::Object(_)) => v.clone(),
                _ => continue,
            };
            for v in schema::violations(&args, sch) {
                problems.push(format!("call to \"{name}\": {v}"));
            }
        }
        if problems.is_empty() {
            return None;
        }
        Some(format!(
            "Your previous tool call did not match the tool's parameter schema — {}. \
             Reply again with a corrected call that satisfies the schema exactly.",
            problems.join("; ")
        ))
    }

    fn upstream_body(&self, stream: bool) -> Bytes {
        let mut b = self.body.clone();
        if let Some(obj) = b.as_object_mut() {
            obj.insert("stream".into(), Value::Bool(stream));
            if !stream {
                // vLLM (and OpenAI) reject stream_options when stream=false.
                obj.remove("stream_options");
            }
        }
        serde_json::to_vec(&b).map(Bytes::from).unwrap_or_default()
    }

    /// Append the model's bad reply plus a corrective user message, forge
    /// style.  The bad reply goes in as plain text content — chat templates
    /// often reject assistant `tool_calls` without matching tool results.
    fn push_nudge(&mut self, bad_message: &Value, nudge: String) {
        let bad_text = message_as_text(bad_message);
        if let Some(msgs) = self.body.get_mut("messages").and_then(|v| v.as_array_mut()) {
            msgs.push(json!({"role": "assistant", "content": bad_text}));
            msgs.push(json!({"role": "user", "content": nudge}));
        }
    }
}

fn message_as_text(message: &Value) -> String {
    if let Some(s) = message.get("content").and_then(|v| v.as_str()) {
        if !s.is_empty() {
            return s.to_string();
        }
    }
    message
        .get("tool_calls")
        .map(|tc| tc.to_string())
        .unwrap_or_default()
}

// ─── Validation & nudges ────────────────────────────────────────────────

pub(crate) enum Verdict {
    /// Native tool calls, all valid.
    Valid,
    /// Tool calls extracted from text content.
    Rescued(Vec<ToolCall>),
    /// Needs a retry with this corrective message.
    Invalid { nudge: String },
    /// Plain text, fine as-is.
    TextOk,
}

pub(crate) fn validate(
    message: &Value,
    tool_names: &HashSet<String>,
    rescue_enabled: bool,
    expect_tool: bool,
) -> Verdict {
    let calls = message
        .get("tool_calls")
        .and_then(|v| v.as_array())
        .filter(|c| !c.is_empty());
    if let Some(calls) = calls {
        for tc in calls {
            let name = tc
                .pointer("/function/name")
                .and_then(|v| v.as_str())
                .unwrap_or("");
            if name.is_empty() {
                return Verdict::Invalid {
                    nudge: malformed_call_nudge(),
                };
            }
            if !tool_names.contains(name) {
                return Verdict::Invalid {
                    nudge: unknown_tool_nudge(name, tool_names),
                };
            }
            let args_ok = match tc.pointer("/function/arguments") {
                // Empty string is treated as `{}` — no-arg tool calls.
                Some(Value::String(s)) if s.trim().is_empty() => true,
                Some(Value::String(s)) => serde_json::from_str::<Value>(s)
                    .map(|v| v.is_object())
                    .unwrap_or(false),
                Some(Value::Object(_)) => true,
                None => true,
                _ => false,
            };
            if !args_ok {
                return Verdict::Invalid {
                    nudge: bad_args_nudge(name),
                };
            }
        }
        return Verdict::Valid;
    }

    let content = message
        .get("content")
        .and_then(|v| v.as_str())
        .unwrap_or("");
    if rescue_enabled && !content.is_empty() {
        let rescued = rescue::rescue_tool_calls(content, tool_names);
        if !rescued.is_empty() {
            return Verdict::Rescued(rescued);
        }
    }
    if expect_tool {
        return Verdict::Invalid {
            nudge: retry_nudge(tool_names.contains(RESPOND_TOOL)),
        };
    }
    Verdict::TextOk
}

fn unknown_tool_nudge(name: &str, tool_names: &HashSet<String>) -> String {
    let mut names: Vec<&str> = tool_names.iter().map(String::as_str).collect();
    names.sort_unstable();
    format!(
        "Your previous reply called the tool \"{name}\", which does not exist. \
         Available tools: {}. Reply again with a correctly formatted call to one of them.",
        names.join(", ")
    )
}

fn bad_args_nudge(name: &str) -> String {
    format!(
        "Your previous call to the tool \"{name}\" had malformed arguments — \
         arguments must be a single JSON object matching the tool's parameters. \
         Reply again with a correctly formatted tool call."
    )
}

fn malformed_call_nudge() -> String {
    "Your previous reply contained a malformed tool call with no tool name. \
     Reply again with a correctly formatted tool call."
        .to_string()
}

fn completion_check_nudge() -> String {
    "Verification step: re-read my original request and check that every \
     requested action, side effect, and output was actually completed with \
     the required tool calls — emails sent, events created, notifications \
     made, results synthesized. If anything is missing, perform it now by \
     calling the appropriate tool. If everything is complete, reply with \
     your full final answer again, exactly as it should be shown to me — \
     do not mention this verification step."
        .to_string()
}

fn retry_nudge(has_respond_tool: bool) -> String {
    if has_respond_tool {
        format!(
            "You must reply by calling one of the available tools. To answer the user \
             in plain text, call the \"{RESPOND_TOOL}\" tool with your answer in its \
             \"message\" argument."
        )
    } else {
        "You must reply by calling one of the available tools — plain text is not \
         an acceptable reply for this request. Reply again with a correctly \
         formatted tool call."
            .to_string()
    }
}

// ─── Synthetic respond tool ─────────────────────────────────────────────

fn respond_spec() -> Value {
    json!({
        "type": "function",
        "function": {
            "name": RESPOND_TOOL,
            "description": "Send a plain-text reply to the user instead of calling another tool.",
            "parameters": {
                "type": "object",
                "properties": {
                    "message": {"type": "string", "description": "The reply text."}
                },
                "required": ["message"]
            }
        }
    })
}

/// Strip synthetic respond calls out of `choices[0].message`.  Only-respond
/// becomes a plain text answer; mixed keeps the real calls.  Returns whether
/// anything changed.
pub(crate) fn strip_respond_in_completion(completion: &mut Value) -> bool {
    let Some(choice) = completion.get_mut("choices").and_then(|c| c.get_mut(0)) else {
        return false;
    };
    let Some(calls) = choice
        .pointer("/message/tool_calls")
        .and_then(|v| v.as_array())
        .cloned()
    else {
        return false;
    };
    let (respond, others): (Vec<Value>, Vec<Value>) = calls.into_iter().partition(|tc| {
        tc.pointer("/function/name").and_then(|v| v.as_str()) == Some(RESPOND_TOOL)
    });
    if respond.is_empty() {
        return false;
    }
    let message = &mut choice["message"];
    if others.is_empty() {
        message["content"] = Value::String(respond_text(&respond[0]));
        message.as_object_mut().unwrap().remove("tool_calls");
        choice["finish_reason"] = json!("stop");
    } else {
        message["tool_calls"] = Value::Array(others);
    }
    true
}

fn respond_text(call: &Value) -> String {
    let args = call
        .pointer("/function/arguments")
        .and_then(|v| v.as_str())
        .unwrap_or("");
    serde_json::from_str::<Value>(args)
        .ok()
        .and_then(|v| v.get("message")?.as_str().map(String::from))
        .unwrap_or_else(|| args.to_string())
}

/// Replace the message's content with rescued tool calls.
pub(crate) fn apply_rescue_to_completion(completion: &mut Value, calls: Vec<ToolCall>) {
    let Some(choice) = completion.get_mut("choices").and_then(|c| c.get_mut(0)) else {
        return;
    };
    choice["message"]["content"] = Value::Null;
    choice["message"]["tool_calls"] = Value::Array(tool_calls_json(&calls));
    choice["finish_reason"] = json!("tool_calls");
}

fn tool_calls_json(calls: &[ToolCall]) -> Vec<Value> {
    calls
        .iter()
        .enumerate()
        .map(|(i, c)| {
            json!({
                "id": format!("call_rescued_{i}"),
                "type": "function",
                "function": {
                    "name": c.name,
                    "arguments": serde_json::to_string(&c.args).unwrap_or_else(|_| "{}".into()),
                }
            })
        })
        .collect()
}

// ─── SSE synthesis ──────────────────────────────────────────────────────

struct ChunkMeta {
    id: String,
    created: u64,
    model: String,
}

impl ChunkMeta {
    fn from_completion(completion: &Value) -> Self {
        Self {
            id: completion
                .get("id")
                .and_then(|v| v.as_str())
                .unwrap_or("chatcmpl-fortify")
                .to_string(),
            created: completion
                .get("created")
                .and_then(|v| v.as_u64())
                .unwrap_or(0),
            model: completion
                .get("model")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string(),
        }
    }

    fn event(&self, delta: Value, finish: Option<&str>) -> Bytes {
        let chunk = json!({
            "id": self.id,
            "object": "chat.completion.chunk",
            "created": self.created,
            "model": self.model,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
        });
        event_bytes(&chunk)
    }

    fn usage_event(&self, usage: &Value) -> Bytes {
        let chunk = json!({
            "id": self.id,
            "object": "chat.completion.chunk",
            "created": self.created,
            "model": self.model,
            "choices": [],
            "usage": usage,
        });
        event_bytes(&chunk)
    }
}

fn event_bytes(chunk: &Value) -> Bytes {
    let mut out = BytesMut::with_capacity(256);
    out.extend_from_slice(b"data: ");
    out.extend_from_slice(chunk.to_string().as_bytes());
    out.extend_from_slice(b"\n\n");
    out.freeze()
}

const DONE_EVENT: &[u8] = b"data: [DONE]\n\n";

/// Build a complete SSE body from a final (non-streaming) completion.
pub(crate) fn synthesize_sse_body(completion: &Value, include_usage: bool) -> Bytes {
    let meta = ChunkMeta::from_completion(completion);
    let msg = completion
        .pointer("/choices/0/message")
        .cloned()
        .unwrap_or(Value::Null);
    let has_calls = msg
        .get("tool_calls")
        .and_then(|v| v.as_array())
        .map(|c| !c.is_empty())
        .unwrap_or(false);
    let finish = completion
        .pointer("/choices/0/finish_reason")
        .and_then(|v| v.as_str())
        .unwrap_or(if has_calls { "tool_calls" } else { "stop" });

    let mut out = BytesMut::with_capacity(1024);
    out.extend_from_slice(&meta.event(json!({"role": "assistant"}), None));
    if let Some(content) = msg.get("content").and_then(|v| v.as_str()) {
        if !content.is_empty() {
            out.extend_from_slice(&meta.event(json!({"content": content}), None));
        }
    }
    if let Some(calls) = msg.get("tool_calls").and_then(|v| v.as_array()) {
        for (i, tc) in calls.iter().enumerate() {
            let mut delta_call = tc.clone();
            delta_call["index"] = json!(i);
            out.extend_from_slice(&meta.event(json!({"tool_calls": [delta_call]}), None));
        }
    }
    out.extend_from_slice(&meta.event(json!({}), Some(finish)));
    if include_usage {
        if let Some(usage) = completion.get("usage").filter(|u| u.is_object()) {
            out.extend_from_slice(&meta.usage_event(usage));
        }
    }
    out.extend_from_slice(DONE_EVENT);
    out.freeze()
}

/// Fix a held tail mid-stream (bytes already forwarded — rescue only, no
/// retry).  Returns the replacement events, or `None` to release the held
/// events verbatim.  Called by `CommittedStream` at upstream end.
pub(crate) fn fix_held_tail(m: &mut HoldMachine) -> Option<Vec<Bytes>> {
    let alias = m.alias_str().to_string();
    let meta = ChunkMeta {
        id: m
            .acc
            .id
            .clone()
            .unwrap_or_else(|| "chatcmpl-fortify".into()),
        created: m.acc.created.unwrap_or(0),
        model: alias,
    };
    // Prose inside the held region that precedes the marker was never
    // forwarded — re-emit it ahead of the fixed tool calls.
    let prose =
        m.acc.content[m.held_text_start.min(m.hold_marker_pos)..m.hold_marker_pos].to_string();

    let (mut content, calls, finish) = if m.acc.has_tool_calls() {
        // Native tool calls were held.  Valid + respond-stripping is the
        // only case needing synthesis; anything else releases verbatim
        // (invalid calls can't be retried once bytes are out).
        let message = m.acc.message_json();
        if !matches!(
            validate(&message, &m.tool_names, false, false),
            Verdict::Valid
        ) {
            return None;
        }
        let calls: Vec<(String, String, String)> = m
            .acc
            .tool_calls
            .iter()
            .enumerate()
            .map(|(i, c)| {
                (
                    c.id.clone().unwrap_or_else(|| format!("call_{i}")),
                    c.name.clone(),
                    c.arguments.clone(),
                )
            })
            .collect();
        if !m.strip_respond || !calls.iter().any(|(_, n, _)| n == RESPOND_TOOL) {
            return None;
        }
        (prose, calls, "tool_calls")
    } else {
        if !m.rescue_enabled {
            return None;
        }
        let rescued = rescue::rescue_tool_calls(&m.acc.content, &m.tool_names);
        if rescued.is_empty() {
            return None;
        }
        let calls: Vec<(String, String, String)> = rescued
            .iter()
            .enumerate()
            .map(|(i, c)| {
                (
                    format!("call_rescued_{i}"),
                    c.name.clone(),
                    serde_json::to_string(&c.args).unwrap_or_else(|_| "{}".into()),
                )
            })
            .collect();
        (prose, calls, "tool_calls")
    };

    // Respond-tool stripping on the fixed calls.
    let (respond, others): (Vec<_>, Vec<_>) = calls
        .into_iter()
        .partition(|(_, name, _)| m.strip_respond && name == RESPOND_TOOL);
    let (calls, finish) = if !respond.is_empty() && others.is_empty() {
        let args = &respond[0].2;
        let text = serde_json::from_str::<Value>(args)
            .ok()
            .and_then(|v| v.get("message")?.as_str().map(String::from))
            .unwrap_or_else(|| args.clone());
        content.push_str(&text);
        (Vec::new(), "stop")
    } else {
        (others, finish)
    };

    let mut events = Vec::new();
    if !content.is_empty() {
        events.push(meta.event(json!({"content": content}), None));
    }
    for (i, (id, name, args)) in calls.iter().enumerate() {
        events.push(meta.event(
            json!({"tool_calls": [{
                "index": i,
                "id": id,
                "type": "function",
                "function": {"name": name, "arguments": args},
            }]}),
            None,
        ));
    }
    events.push(meta.event(json!({}), Some(finish)));
    if let Some(usage) = &m.acc.usage {
        events.push(meta.usage_event(usage));
    }
    events.push(Bytes::from_static(DONE_EVENT));
    Some(events)
}

// ─── Orchestration ──────────────────────────────────────────────────────

/// Buffered fortification: non-streaming upstream calls in a retry loop.
/// Used for non-streaming clients, `mode:"buffer"`, and as the retry tail
/// of the hold-streaming path (`retries_done` seeds the budget).
async fn run_buffered(
    ctx: &RunCtx<'_>,
    f: &FortifyConfig,
    mut fort: Fortifier,
    mut retries_done: u32,
) -> Result<Response<Body>, anyhow::Error> {
    let mut check_done = false;
    loop {
        let resp = send_upstream(
            &ctx.parts.method,
            &ctx.parts.headers,
            ctx.upstream_url,
            fort.upstream_body(false),
            ctx.model_cfg,
            ctx.client,
            ctx.header_timeout,
        )
        .await?;
        let (rparts, incoming) = resp.into_parts();
        let bytes = incoming.collect().await?.to_bytes();
        if !rparts.status.is_success() {
            return passthrough_collected(&rparts, bytes, ctx.model_cfg);
        }
        let Ok(mut completion) = serde_json::from_slice::<Value>(&bytes) else {
            return passthrough_collected(&rparts, bytes, ctx.model_cfg);
        };
        let message = completion
            .pointer("/choices/0/message")
            .cloned()
            .unwrap_or(Value::Null);

        match validate(&message, &fort.tool_names, f.rescue, fort.expect_tool) {
            Verdict::TextOk if fort.check_candidate && !check_done => {
                // One verification round: ask the model to confirm every
                // requested action was completed, or continue if not.
                check_done = true;
                tracing::info!("[#{}] fortify completion check", ctx.req_id);
                fort.push_nudge(&message, completion_check_nudge());
            }
            Verdict::Valid => {
                if let Some(nudge) = fort.schema_nudge(&message) {
                    if retries_done >= f.max_retries {
                        tracing::warn!(
                            "[#{}] fortify retries exhausted — returning last response",
                            ctx.req_id
                        );
                        return emit_final(ctx, &fort, completion);
                    }
                    retries_done += 1;
                    tracing::info!(
                        "[#{}] fortify retry {retries_done}: schema violation",
                        ctx.req_id
                    );
                    fort.push_nudge(&message, nudge);
                    continue;
                }
                if f.inject_respond_tool {
                    strip_respond_in_completion(&mut completion);
                }
                return emit_final(ctx, &fort, completion);
            }
            Verdict::TextOk => {
                if f.inject_respond_tool {
                    strip_respond_in_completion(&mut completion);
                }
                return emit_final(ctx, &fort, completion);
            }
            Verdict::Rescued(calls) => {
                tracing::info!(
                    "[#{}] fortify rescued {} tool call(s)",
                    ctx.req_id,
                    calls.len()
                );
                apply_rescue_to_completion(&mut completion, calls);
                if f.inject_respond_tool {
                    strip_respond_in_completion(&mut completion);
                }
                return emit_final(ctx, &fort, completion);
            }
            Verdict::Invalid { nudge } => {
                if retries_done >= f.max_retries {
                    tracing::warn!(
                        "[#{}] fortify retries exhausted — returning last response",
                        ctx.req_id
                    );
                    return emit_final(ctx, &fort, completion);
                }
                retries_done += 1;
                tracing::info!(
                    "[#{}] fortify retry {retries_done}: invalid tool call",
                    ctx.req_id
                );
                fort.push_nudge(&message, nudge);
            }
        }
    }
}

/// Hold-mode streaming: stream upstream, forward prose live, hold tool
/// calls.  Retry remains possible until the first byte is forwarded.
async fn run_hold_streaming(
    ctx: &RunCtx<'_>,
    f: &FortifyConfig,
    mut fort: Fortifier,
) -> Result<Response<Body>, anyhow::Error> {
    let resp = send_upstream(
        &ctx.parts.method,
        &ctx.parts.headers,
        ctx.upstream_url,
        fort.upstream_body(true),
        ctx.model_cfg,
        ctx.client,
        ctx.header_timeout,
    )
    .await?;
    let (rparts, mut incoming) = resp.into_parts();

    let is_sse = rparts
        .headers
        .get("content-type")
        .and_then(|v| v.to_str().ok())
        .map(|s| s.contains("event-stream"))
        .unwrap_or(false);
    if !rparts.status.is_success() || !is_sse {
        let bytes = incoming.collect().await?.to_bytes();
        return passthrough_collected(&rparts, bytes, ctx.model_cfg);
    }

    let mut machine = HoldMachine::new(
        &ctx.model_cfg.name,
        fort.tool_names.clone(),
        f.rescue,
        f.inject_respond_tool,
        ctx.req_id,
    );

    // Phase A: consume upstream until the machine produces the first
    // forwardable bytes (commit) or the stream ends fully held.
    let upstream_ended = loop {
        if !machine.outbox.is_empty() {
            break false;
        }
        match incoming.frame().await {
            Some(Ok(frame)) => {
                if let Ok(data) = frame.into_data() {
                    machine.feed_bytes(&data);
                }
            }
            Some(Err(e)) => {
                return Err(anyhow::anyhow!("upstream stream error: {e}"));
            }
            None => break true,
        }
    };

    if !upstream_ended {
        // Commit: hand the machine to the response body for live forwarding.
        let mut rb = Response::builder().status(rparts.status);
        for (name, value) in &rparts.headers {
            if !skip_response_header(name.as_str()) {
                rb = rb.header(name, value);
            }
        }
        let stream = CommittedStream::new(incoming, machine);
        return Ok(rb.body(Body::from_stream(stream))?);
    }

    // Stream ended with nothing forwarded — full response in hand and the
    // retry option intact.
    let message = machine.acc.message_json();
    match validate(&message, &fort.tool_names, f.rescue, fort.expect_tool) {
        Verdict::Valid => {
            if let Some(nudge) = fort.schema_nudge(&message) {
                tracing::info!(
                    "[#{}] fortify retry 1: schema violation in stream",
                    ctx.req_id
                );
                fort.push_nudge(&message, nudge);
                return run_buffered(ctx, f, fort, 1).await;
            }
            let mut completion = completion_from_acc(&machine.acc, message);
            if f.inject_respond_tool && strip_respond_in_completion(&mut completion) {
                emit_final(ctx, &fort, completion)
            } else {
                replay_response(&rparts, &mut machine, ctx.model_cfg)
            }
        }
        Verdict::TextOk => replay_response(&rparts, &mut machine, ctx.model_cfg),
        Verdict::Rescued(calls) => {
            tracing::info!(
                "[#{}] fortify rescued {} tool call(s)",
                ctx.req_id,
                calls.len()
            );
            let mut completion = completion_from_acc(&machine.acc, message);
            apply_rescue_to_completion(&mut completion, calls);
            if f.inject_respond_tool {
                strip_respond_in_completion(&mut completion);
            }
            emit_final(ctx, &fort, completion)
        }
        Verdict::Invalid { nudge } => {
            tracing::info!(
                "[#{}] fortify retry 1: invalid tool call in stream",
                ctx.req_id
            );
            fort.push_nudge(&message, nudge);
            run_buffered(ctx, f, fort, 1).await
        }
    }
}

// ─── Response emission ──────────────────────────────────────────────────

fn emit_final(
    ctx: &RunCtx<'_>,
    fort: &Fortifier,
    mut completion: Value,
) -> Result<Response<Body>, anyhow::Error> {
    completion["model"] = Value::String(ctx.model_cfg.name.clone());
    if fort.client_stream {
        let body = synthesize_sse_body(&completion, fort.include_usage);
        tracing::info!(
            "[#{}] fortified done (synthesized sse) {}B",
            ctx.req_id,
            body.len()
        );
        Ok(Response::builder()
            .status(axum::http::StatusCode::OK)
            .header("content-type", "text/event-stream")
            .header("cache-control", "no-cache")
            .body(Body::from(body))?)
    } else {
        let bytes = serde_json::to_vec(&completion)?;
        tracing::info!("[#{}] fortified done {}B", ctx.req_id, bytes.len());
        Ok(Response::builder()
            .status(axum::http::StatusCode::OK)
            .header("content-type", "application/json")
            .body(Body::from(bytes))?)
    }
}

/// Byte-faithful replay of a fully-held upstream stream (model alias
/// rewritten per event) — used when the held stream needs no fixing.
fn replay_response(
    rparts: &axum::http::response::Parts,
    machine: &mut HoldMachine,
    model_cfg: &ModelConfig,
) -> Result<Response<Body>, anyhow::Error> {
    let mut out = BytesMut::new();
    for ev in machine.raw_log.iter() {
        out.extend_from_slice(&rewrite_model_bytes(ev.clone(), model_cfg.name.as_bytes()));
    }
    if let Some(remnant) = machine.remnant() {
        out.extend_from_slice(&rewrite_model_bytes(remnant, model_cfg.name.as_bytes()));
    }
    let mut rb = Response::builder().status(rparts.status);
    for (name, value) in &rparts.headers {
        if !skip_response_header(name.as_str()) {
            rb = rb.header(name, value);
        }
    }
    Ok(rb.body(Body::from(out.freeze()))?)
}

fn passthrough_collected(
    rparts: &axum::http::response::Parts,
    bytes: Bytes,
    model_cfg: &ModelConfig,
) -> Result<Response<Body>, anyhow::Error> {
    let rewritten = rewrite_model_bytes(bytes, model_cfg.name.as_bytes());
    let mut rb = Response::builder().status(rparts.status);
    for (name, value) in &rparts.headers {
        if !skip_response_header(name.as_str()) {
            rb = rb.header(name, value);
        }
    }
    Ok(rb.body(Body::from(rewritten))?)
}

fn completion_from_acc(acc: &MessageAccumulator, message: Value) -> Value {
    let finish = acc.finish_reason.clone().unwrap_or_else(|| {
        if acc.has_tool_calls() {
            "tool_calls".into()
        } else {
            "stop".into()
        }
    });
    let mut completion = json!({
        "id": acc.id.clone().unwrap_or_else(|| "chatcmpl-fortify".into()),
        "object": "chat.completion",
        "created": acc.created.unwrap_or(0),
        "model": "",
        "choices": [{"index": 0, "message": message, "finish_reason": finish}],
    });
    if let Some(usage) = &acc.usage {
        completion["usage"] = usage.clone();
    }
    completion
}

// ─── Tests ──────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::FortifyMode;

    fn tools(names: &[&str]) -> HashSet<String> {
        names.iter().map(|s| s.to_string()).collect()
    }

    fn fortify_cfg() -> FortifyConfig {
        serde_json::from_str(r#"{}"#).unwrap()
    }

    fn model_cfg(target: &str) -> ModelConfig {
        ModelConfig {
            name: "alias".into(),
            target: target.into(),
            served_model: "real-model".into(),
            api_key: None,
            params: std::collections::HashMap::new(),
            fortify: Some(fortify_cfg()),
            inject: None,
        }
    }

    fn chat_json(extra: &str) -> Value {
        serde_json::from_str(&format!(
            r#"{{"model":"alias","messages":[{{"role":"user","content":"hi"}}],"tools":[{{"type":"function","function":{{"name":"search","parameters":{{}}}}}}]{extra}}}"#
        ))
        .unwrap()
    }

    // ── applies_parsed() gate ──

    #[test]
    fn test_applies_parsed_on_tools_body() {
        assert!(applies_parsed(&chat_json("")));
        assert!(applies_parsed(&chat_json(",\"n\":1")));
    }

    #[test]
    fn test_applies_parsed_rejects_no_tools_and_multi_choice() {
        let no_tools: Value = serde_json::from_str(r#"{"model":"alias","messages":[]}"#).unwrap();
        assert!(!applies_parsed(&no_tools));
        let empty_tools: Value =
            serde_json::from_str(r#"{"model":"alias","messages":[],"tools":[]}"#).unwrap();
        assert!(!applies_parsed(&empty_tools));
        assert!(!applies_parsed(&chat_json(",\"n\":2")));
    }

    #[test]
    fn test_finalize_plain_body_applies_model_and_params() {
        let mut cfg = model_cfg("http://x");
        cfg.params
            .insert("temperature".into(), serde_json::json!(0.5));
        let out = finalize_plain_body(chat_json(""), &cfg);
        let v: Value = serde_json::from_slice(&out).unwrap();
        assert_eq!(v["model"], "real-model");
        assert_eq!(v["temperature"], 0.5);
    }

    // ── validate() ──

    fn native_call(name: &str, args: &str) -> Value {
        json!({"role":"assistant","content":null,"tool_calls":[
            {"id":"c1","type":"function","function":{"name":name,"arguments":args}}]})
    }

    #[test]
    fn test_validate_valid_native_call() {
        let msg = native_call("search", r#"{"q":"rust"}"#);
        assert!(matches!(
            validate(&msg, &tools(&["search"]), true, false),
            Verdict::Valid
        ));
    }

    #[test]
    fn test_validate_empty_args_string_is_valid() {
        let msg = native_call("search", "");
        assert!(matches!(
            validate(&msg, &tools(&["search"]), true, false),
            Verdict::Valid
        ));
    }

    #[test]
    fn test_validate_unknown_tool_invalid() {
        let msg = native_call("hack", "{}");
        match validate(&msg, &tools(&["search"]), true, false) {
            Verdict::Invalid { nudge } => {
                assert!(nudge.contains("\"hack\""));
                assert!(nudge.contains("search"));
            }
            _ => panic!("expected Invalid"),
        }
    }

    #[test]
    fn test_validate_malformed_args_invalid() {
        let msg = native_call("search", "{not json");
        assert!(matches!(
            validate(&msg, &tools(&["search"]), true, false),
            Verdict::Invalid { .. }
        ));
    }

    #[test]
    fn test_validate_text_rescued() {
        let msg = json!({"role":"assistant","content":"[TOOL_CALLS]search {\"q\":\"x\"}"});
        match validate(&msg, &tools(&["search"]), true, false) {
            Verdict::Rescued(calls) => assert_eq!(calls[0].name, "search"),
            _ => panic!("expected Rescued"),
        }
    }

    #[test]
    fn test_validate_plain_text_ok_unless_tool_expected() {
        let msg = json!({"role":"assistant","content":"Paris is the capital."});
        assert!(matches!(
            validate(&msg, &tools(&["search"]), true, false),
            Verdict::TextOk
        ));
        assert!(matches!(
            validate(&msg, &tools(&["search"]), true, true),
            Verdict::Invalid { .. }
        ));
    }

    // ── respond tool ──

    #[test]
    fn test_strip_respond_only_call_becomes_text() {
        let mut completion = json!({"choices":[{"index":0,"finish_reason":"tool_calls","message":
            {"role":"assistant","content":null,"tool_calls":[
                {"id":"c1","type":"function","function":{"name":RESPOND_TOOL,"arguments":"{\"message\":\"hello there\"}"}}]}}]});
        assert!(strip_respond_in_completion(&mut completion));
        let msg = &completion["choices"][0]["message"];
        assert_eq!(msg["content"], "hello there");
        assert!(msg.get("tool_calls").is_none());
        assert_eq!(completion["choices"][0]["finish_reason"], "stop");
    }

    #[test]
    fn test_strip_respond_mixed_keeps_real_calls() {
        let mut completion = json!({"choices":[{"index":0,"finish_reason":"tool_calls","message":
            {"role":"assistant","content":null,"tool_calls":[
                {"id":"c1","type":"function","function":{"name":RESPOND_TOOL,"arguments":"{\"message\":\"x\"}"}},
                {"id":"c2","type":"function","function":{"name":"search","arguments":"{}"}}]}}]});
        assert!(strip_respond_in_completion(&mut completion));
        let calls = completion["choices"][0]["message"]["tool_calls"]
            .as_array()
            .unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0]["function"]["name"], "search");
    }

    #[test]
    fn test_strip_respond_noop_without_respond_call() {
        let mut completion = json!({"choices":[{"index":0,"message":
            {"role":"assistant","content":null,"tool_calls":[
                {"id":"c2","type":"function","function":{"name":"search","arguments":"{}"}}]}}]});
        assert!(!strip_respond_in_completion(&mut completion));
    }

    // ── Fortifier request prep ──

    fn fortifier_with(body: &str, f: &FortifyConfig) -> Fortifier {
        let cfg = model_cfg("http://x");
        Fortifier::new(serde_json::from_str(body).unwrap(), &cfg, f)
    }

    #[test]
    fn test_fortifier_prepares_upstream_body() {
        let f = fortify_cfg();
        let fort = fortifier_with(
            r#"{"model":"alias","stream":true,"stream_options":{"include_usage":true},
                "messages":[],"tools":[{"type":"function","function":{"name":"search"}}]}"#,
            &f,
        );
        assert!(fort.client_stream);
        assert!(fort.include_usage);
        assert!(fort.tool_names.contains("search"));

        let non_stream: Value = serde_json::from_slice(&fort.upstream_body(false)).unwrap();
        assert_eq!(non_stream["stream"], false);
        assert_eq!(non_stream["model"], "real-model");
        assert!(
            non_stream.get("stream_options").is_none(),
            "vLLM rejects stream_options with stream=false"
        );

        let streaming: Value = serde_json::from_slice(&fort.upstream_body(true)).unwrap();
        assert_eq!(streaming["stream"], true);
        assert!(streaming.get("stream_options").is_some());
    }

    #[test]
    fn test_fortifier_injects_respond_tool() {
        let mut f = fortify_cfg();
        f.inject_respond_tool = true;
        let fort = fortifier_with(
            r#"{"model":"alias","messages":[],"tools":[{"type":"function","function":{"name":"search"}}]}"#,
            &f,
        );
        assert!(fort.tool_names.contains(RESPOND_TOOL));
        let body: Value = serde_json::from_slice(&fort.upstream_body(false)).unwrap();
        assert_eq!(body["tools"].as_array().unwrap().len(), 2);
    }

    #[test]
    fn test_fortifier_push_nudge_appends_messages() {
        let f = fortify_cfg();
        let mut fort = fortifier_with(
            r#"{"model":"alias","messages":[{"role":"user","content":"hi"}],
                "tools":[{"type":"function","function":{"name":"search"}}]}"#,
            &f,
        );
        let bad = native_call("hack", "{}");
        fort.push_nudge(&bad, "do better".into());
        let body: Value = serde_json::from_slice(&fort.upstream_body(false)).unwrap();
        let msgs = body["messages"].as_array().unwrap();
        assert_eq!(msgs.len(), 3);
        assert_eq!(msgs[1]["role"], "assistant");
        assert!(msgs[1]["content"].as_str().unwrap().contains("hack"));
        assert_eq!(msgs[2], json!({"role":"user","content":"do better"}));
    }

    // ── SSE synthesis ──

    #[test]
    fn test_synthesize_sse_tool_call_shape() {
        let completion = json!({
            "id":"chatcmpl-1","object":"chat.completion","created":7,"model":"alias",
            "choices":[{"index":0,"finish_reason":"tool_calls","message":
                {"role":"assistant","content":null,"tool_calls":[
                    {"id":"c1","type":"function","function":{"name":"search","arguments":"{\"q\":\"x\"}"}}]}}],
            "usage":{"total_tokens":10}
        });
        let body = synthesize_sse_body(&completion, true);
        let text = String::from_utf8_lossy(&body);
        let events: Vec<&str> = text.trim_end().split("\n\n").collect();
        assert!(events[0].contains("\"role\":\"assistant\""));
        assert!(events[1].contains("\"tool_calls\""));
        assert!(events[1].contains("\"index\":0"));
        assert!(events[2].contains("\"finish_reason\":\"tool_calls\""));
        assert!(events[3].contains("\"total_tokens\":10"));
        assert_eq!(*events.last().unwrap(), "data: [DONE]");
        for e in &events {
            assert!(e.starts_with("data: "), "bad event framing: {e}");
        }
    }

    #[test]
    fn test_synthesize_sse_text_without_usage() {
        let completion = json!({
            "id":"chatcmpl-1","created":7,"model":"alias",
            "choices":[{"index":0,"finish_reason":"stop","message":
                {"role":"assistant","content":"hello"}}],
            "usage":{"total_tokens":10}
        });
        let text = String::from_utf8_lossy(&synthesize_sse_body(&completion, false)).into_owned();
        assert!(text.contains("\"content\":\"hello\""));
        assert!(
            !text.contains("total_tokens"),
            "usage must be omitted unless requested"
        );
        assert!(text.ends_with("data: [DONE]\n\n"));
    }

    // ── completion check ──

    #[test]
    fn test_check_candidate_requires_flag_and_tool_results() {
        let mut f = fortify_cfg();
        f.completion_check = true;
        let cfg = model_cfg("http://x");
        let with_tool_result: Value = serde_json::from_str(
            r#"{"model":"alias","messages":[
                {"role":"user","content":"do it"},
                {"role":"assistant","content":null,"tool_calls":[{"id":"c1","type":"function","function":{"name":"search","arguments":"{}"}}]},
                {"role":"tool","tool_call_id":"c1","content":"result"}],
                "tools":[{"type":"function","function":{"name":"search"}}]}"#,
        )
        .unwrap();
        assert!(Fortifier::new(with_tool_result.clone(), &cfg, &f).check_candidate);

        let fresh: Value = serde_json::from_str(
            r#"{"model":"alias","messages":[{"role":"user","content":"hi"}],
                "tools":[{"type":"function","function":{"name":"search"}}]}"#,
        )
        .unwrap();
        assert!(
            !Fortifier::new(fresh.clone(), &cfg, &f).check_candidate,
            "no prior tool results — restraint cases must not be pushed toward tools"
        );

        let f_off = fortify_cfg();
        assert!(!Fortifier::new(with_tool_result, &cfg, &f_off).check_candidate);
    }

    #[tokio::test]
    async fn test_completion_check_round_recovers_missing_action() {
        // Turn 1 response: plain text claiming done (but email never sent).
        let lazy_text = r#"{"id":"c1","object":"chat.completion","created":1,"model":"real-model","choices":[{"index":0,"finish_reason":"stop","message":{"role":"assistant","content":"I looked up Tom Chen for you."}}]}"#;
        // Check-round response: the model completes the missing tool call.
        let fixed = r#"{"id":"c2","object":"chat.completion","created":1,"model":"real-model","choices":[{"index":0,"finish_reason":"tool_calls","message":{"role":"assistant","content":null,"tool_calls":[{"id":"ok1","type":"function","function":{"name":"send_email","arguments":"{\"to\":\"tom\"}"}}]}}]}"#;
        let (addr, seen) = mock_upstream(vec![
            http_response("application/json", lazy_text),
            http_response("application/json", fixed),
        ])
        .await;

        let mut cfg = model_cfg(&format!("http://{addr}/v1"));
        cfg.fortify.as_mut().unwrap().completion_check = true;
        let client: Client<HttpConnector, Body> =
            Client::builder(hyper_util::rt::TokioExecutor::new()).build_http();
        let req = axum::http::Request::builder()
            .method("POST")
            .uri("/alias/v1/chat/completions")
            .body(Body::from(
                r#"{"model":"alias","stream":true,"messages":[
                    {"role":"user","content":"find tom and email him"},
                    {"role":"assistant","content":null,"tool_calls":[{"id":"c1","type":"function","function":{"name":"search","arguments":"{}"}}]},
                    {"role":"tool","tool_call_id":"c1","content":"Tom Chen <tom@x>"}],
                    "tools":[{"type":"function","function":{"name":"search"}},{"type":"function","function":{"name":"send_email"}}]}"#,
            ))
            .unwrap();

        let resp = crate::forward::proxy(
            &format!("http://{addr}/v1/chat/completions"),
            &cfg,
            req,
            &client,
            1 << 20,
            None,
        )
        .await
        .unwrap();

        let body = axum::body::to_bytes(resp.into_body(), 1 << 20)
            .await
            .unwrap();
        let text = String::from_utf8_lossy(&body);
        // Despite stream:true the candidate is routed buffered, checked, and
        // the recovered tool call is synthesized as SSE.
        assert!(
            text.contains("send_email"),
            "missing recovered call: {text}"
        );
        assert!(text.ends_with("data: [DONE]\n\n"));

        let bodies = seen.lock().unwrap().clone();
        assert_eq!(bodies.len(), 2, "exactly one check round");
        let check_req: Value = serde_json::from_str(&bodies[1]).unwrap();
        let msgs = check_req["messages"].as_array().unwrap();
        assert!(msgs[msgs.len() - 1]["content"]
            .as_str()
            .unwrap()
            .contains("Verification step"));
        assert!(msgs[msgs.len() - 2]["content"]
            .as_str()
            .unwrap()
            .contains("I looked up Tom Chen"));
    }

    #[tokio::test]
    async fn test_schema_violation_retried_with_precise_nudge() {
        // The production failure: edit called with {newText, path} instead
        // of the required {path, edits}.
        let bad = r#"{"id":"c1","object":"chat.completion","created":1,"model":"real-model","choices":[{"index":0,"finish_reason":"tool_calls","message":{"role":"assistant","content":null,"tool_calls":[{"id":"x1","type":"function","function":{"name":"edit","arguments":"{\"newText\": \"const x = 1;\", \"path\": \"/a/main.js\"}"}}]}}]}"#;
        let good = r#"{"id":"c2","object":"chat.completion","created":1,"model":"real-model","choices":[{"index":0,"finish_reason":"tool_calls","message":{"role":"assistant","content":null,"tool_calls":[{"id":"x2","type":"function","function":{"name":"edit","arguments":"{\"path\": \"/a/main.js\", \"edits\": []}"}}]}}]}"#;
        let (addr, seen) = mock_upstream(vec![
            http_response("application/json", bad),
            http_response("application/json", good),
        ])
        .await;

        let cfg = model_cfg(&format!("http://{addr}/v1"));
        let client: Client<HttpConnector, Body> =
            Client::builder(hyper_util::rt::TokioExecutor::new()).build_http();
        let req = axum::http::Request::builder()
            .method("POST")
            .uri("/alias/v1/chat/completions")
            .body(Body::from(
                r#"{"model":"alias","stream":false,"messages":[{"role":"user","content":"fix it"}],
                    "tools":[{"type":"function","function":{"name":"edit","parameters":{
                        "type":"object",
                        "properties":{"path":{"type":"string"},"edits":{"type":"array"}},
                        "required":["path","edits"],"additionalProperties":false}}}]}"#,
            ))
            .unwrap();

        let resp = crate::forward::proxy(
            &format!("http://{addr}/v1/chat/completions"),
            &cfg,
            req,
            &client,
            1 << 20,
            None,
        )
        .await
        .unwrap();

        let body = axum::body::to_bytes(resp.into_body(), 1 << 20)
            .await
            .unwrap();
        let v: Value = serde_json::from_slice(&body).unwrap();
        let args = v["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"]
            .as_str()
            .unwrap();
        assert!(args.contains("edits"), "corrected call expected: {args}");

        let bodies = seen.lock().unwrap().clone();
        assert_eq!(bodies.len(), 2, "exactly one schema retry");
        let retry: Value = serde_json::from_str(&bodies[1]).unwrap();
        let nudge = retry["messages"].as_array().unwrap().last().unwrap()["content"]
            .as_str()
            .unwrap();
        assert!(
            nudge.contains("missing required property \"edits\""),
            "{nudge}"
        );
        assert!(
            nudge.contains("unsupported property \"newText\""),
            "{nudge}"
        );
    }

    // ── integration: invalid stream attempt → nudged retry → synthesized SSE ──

    /// Minimal one-request-per-connection HTTP server.  Each connection gets
    /// the next canned response; request bodies are recorded.
    async fn mock_upstream(
        responses: Vec<String>,
    ) -> (
        std::net::SocketAddr,
        std::sync::Arc<std::sync::Mutex<Vec<String>>>,
    ) {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let seen = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let seen2 = seen.clone();
        tokio::spawn(async move {
            for resp in responses {
                let (mut sock, _) = listener.accept().await.unwrap();
                let mut buf = Vec::new();
                let body = loop {
                    let mut chunk = [0u8; 4096];
                    let n = sock.read(&mut chunk).await.unwrap();
                    buf.extend_from_slice(&chunk[..n]);
                    if let Some(hdr_end) = buf.windows(4).position(|w| w == b"\r\n\r\n") {
                        let headers = String::from_utf8_lossy(&buf[..hdr_end]).to_lowercase();
                        let clen: usize = headers
                            .lines()
                            .find_map(|l| l.strip_prefix("content-length:"))
                            .map(|v| v.trim().parse().unwrap())
                            .unwrap_or(0);
                        if buf.len() >= hdr_end + 4 + clen {
                            break String::from_utf8_lossy(&buf[hdr_end + 4..hdr_end + 4 + clen])
                                .into_owned();
                        }
                    }
                };
                seen2.lock().unwrap().push(body);
                sock.write_all(resp.as_bytes()).await.unwrap();
                sock.shutdown().await.ok();
            }
        });
        (addr, seen)
    }

    fn http_response(content_type: &str, body: &str) -> String {
        format!(
            "HTTP/1.1 200 OK\r\ncontent-type: {content_type}\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{body}",
            body.len()
        )
    }

    #[tokio::test]
    async fn test_invalid_stream_attempt_retries_and_synthesizes_sse() {
        // Attempt 1 (streaming): model calls a tool that doesn't exist.
        let bad_sse = "data: {\"id\":\"c1\",\"model\":\"real-model\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"tool_calls\":[{\"index\":0,\"id\":\"x\",\"type\":\"function\",\"function\":{\"name\":\"hack\",\"arguments\":\"{}\"}}]},\"finish_reason\":null}]}\n\ndata: [DONE]\n\n";
        // Attempt 2 (retry, non-streaming): a valid call.
        let good_json = r#"{"id":"c2","object":"chat.completion","created":1,"model":"real-model","choices":[{"index":0,"finish_reason":"tool_calls","message":{"role":"assistant","content":null,"tool_calls":[{"id":"ok1","type":"function","function":{"name":"search","arguments":"{\"q\":\"rust\"}"}}]}}],"usage":{"total_tokens":5}}"#;

        let (addr, seen) = mock_upstream(vec![
            http_response("text/event-stream", bad_sse),
            http_response("application/json", good_json),
        ])
        .await;

        let cfg = model_cfg(&format!("http://{addr}/v1"));
        let client: Client<HttpConnector, Body> =
            Client::builder(hyper_util::rt::TokioExecutor::new()).build_http();
        let req = axum::http::Request::builder()
            .method("POST")
            .uri("/alias/v1/chat/completions")
            .body(Body::from(
                r#"{"model":"alias","stream":true,"messages":[{"role":"user","content":"find rust"}],
                    "tools":[{"type":"function","function":{"name":"search","parameters":{}}}]}"#,
            ))
            .unwrap();

        let resp = crate::forward::proxy(
            &format!("http://{addr}/v1/chat/completions"),
            &cfg,
            req,
            &client,
            1 << 20,
            None,
        )
        .await
        .unwrap();

        assert_eq!(
            resp.headers().get("content-type").unwrap(),
            "text/event-stream"
        );
        let body = axum::body::to_bytes(resp.into_body(), 1 << 20)
            .await
            .unwrap();
        let text = String::from_utf8_lossy(&body);
        assert!(
            text.contains("\"name\":\"search\""),
            "valid retry call expected: {text}"
        );
        assert!(!text.contains("hack"), "invalid call must not leak: {text}");
        assert!(
            text.contains("\"model\":\"alias\""),
            "alias rewrite expected: {text}"
        );
        assert!(text.ends_with("data: [DONE]\n\n"));

        let bodies = seen.lock().unwrap().clone();
        assert_eq!(bodies.len(), 2, "exactly one retry expected");
        let retry: Value = serde_json::from_str(&bodies[1]).unwrap();
        assert_eq!(retry["stream"], false, "retries are non-streaming");
        let msgs = retry["messages"].as_array().unwrap();
        assert_eq!(msgs.len(), 3, "bad reply + nudge appended");
        assert!(msgs[2]["content"].as_str().unwrap().contains("\"hack\""));
    }

    #[tokio::test]
    async fn test_buffer_mode_validates_and_synthesizes_for_streaming_client() {
        let good_json = r#"{"id":"c1","object":"chat.completion","created":1,"model":"real-model","choices":[{"index":0,"finish_reason":"tool_calls","message":{"role":"assistant","content":null,"tool_calls":[{"id":"ok1","type":"function","function":{"name":"search","arguments":"{}"}}]}}]}"#;
        let (addr, seen) = mock_upstream(vec![http_response("application/json", good_json)]).await;

        let mut cfg = model_cfg(&format!("http://{addr}/v1"));
        cfg.fortify.as_mut().unwrap().mode = FortifyMode::Buffer;
        let client: Client<HttpConnector, Body> =
            Client::builder(hyper_util::rt::TokioExecutor::new()).build_http();
        let req = axum::http::Request::builder()
            .method("POST")
            .uri("/alias/v1/chat/completions")
            .body(Body::from(
                r#"{"model":"alias","stream":true,"messages":[],"tools":[{"type":"function","function":{"name":"search"}}]}"#,
            ))
            .unwrap();

        let resp = crate::forward::proxy(
            &format!("http://{addr}/v1/chat/completions"),
            &cfg,
            req,
            &client,
            1 << 20,
            None,
        )
        .await
        .unwrap();

        let body = axum::body::to_bytes(resp.into_body(), 1 << 20)
            .await
            .unwrap();
        let text = String::from_utf8_lossy(&body);
        assert!(text.starts_with("data: "), "client asked to stream: {text}");
        assert!(text.contains("\"name\":\"search\""));
        assert!(text.ends_with("data: [DONE]\n\n"));

        let upstream_req: Value = serde_json::from_str(&seen.lock().unwrap()[0]).unwrap();
        assert_eq!(
            upstream_req["stream"], false,
            "buffer mode forces non-streaming upstream"
        );
    }
}
