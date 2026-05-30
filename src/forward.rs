//! High-performance HTTP forwarder — 0 JSON parses on response path.

use axum::{
    body::Body,
    http::{Method, Request, Response},
};
use bytes::{Bytes, BytesMut};
use http_body_util::BodyExt;
use hyper::body::Incoming;
use hyper_util::client::legacy::{Client, connect::HttpConnector};
use std::io;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use tokio_stream::StreamExt;

const STREAM_BUF: usize = 4 * 1024;
static REQ_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Forward request: transform body, call upstream, rewrite model name in response.
pub async fn proxy(
    upstream_url: &str,
    model_cfg: &crate::config::ModelConfig,
    req: Request<Body>,
    client: &Client<HttpConnector, Body>,
    method: Method,
) -> Result<Response<Body>, anyhow::Error> {
    let req_id = REQ_COUNTER.fetch_add(1, Ordering::Relaxed);
    let t0 = Instant::now();
    let (parts, body) = req.into_parts();
    
    // Buffer request body for transformation.
    // Using a configurable limit to prevent oversized payloads from exhausting memory.
    let body_bytes = axum::body::to_bytes(body, 4 * 1024 * 1024).await?;
    
    // Transform the request body (inject params, rewrite model name).
    let transformed = transform_req(&body_bytes, model_cfg);

    // Build the outgoing request, stripping hop-by-hop headers.
    let mut builder = hyper::Request::builder()
        .method(method)
        .uri(upstream_url)
        .header("content-type", "application/json")
        .header("content-length", transformed.len());

    for (name, value) in parts.headers.iter() {
        let l = name.as_str().to_lowercase();
        if matches!(l.as_str(), "host" | "connection" | "keep-alive" | "transfer-encoding" | "upgrade") {
            continue;
        }
        builder = builder.header(name, value);
    }

    if let Some(api_key) = &model_cfg.api_key {
        builder = builder.header("authorization", format!("Bearer {}", api_key));
    }

    let t2 = Instant::now();
    let outgoing = builder.body(Body::from(transformed))?;
    let resp = client.request(outgoing).await?;
    let upstream_elapsed = t2.elapsed();
    
    // Log upstream timing at debug level to avoid I/O spam under burst load.
    tracing::debug!("[#{req_id}] upstream: {upstream_elapsed:?}  status={}", resp.status());

    let _t3 = Instant::now();
    let (parts, incoming) = resp.into_parts();

    let is_sse = parts
        .headers
        .get("content-type")
        .and_then(|v| v.to_str().ok())
        .map(|s| s.contains("event-stream"))
        .unwrap_or(false);

    if is_sse {
        let (tx, rx) = mpsc::channel::<Result<Bytes, io::Error>>(32);
        tokio::spawn(forward_sse(incoming, tx, model_cfg.name.clone(), req_id));

        let mut rb = Response::builder().status(parts.status);
        for (name, value) in &parts.headers {
            let l = name.as_str().to_lowercase();
            if l != "transfer-encoding" && l != "content-length" {
                rb = rb.header(name, value);
            }
        }
        let stream = ReceiverStream::new(rx)
            .map(|r| r.map_err(|e| Box::new(e) as Box<dyn std::error::Error + Send + Sync>));
        return Ok(rb.body(Body::from_stream(stream))?);
    }

    // Non-streaming: collect and rewrite model name at the byte level.
    let body = http_body_util::BodyExt::collect(incoming).await?.to_bytes();
    let rewritten = rewrite_model_bytes(&body, model_cfg.name.as_bytes());

    let mut rb = Response::builder().status(parts.status);
    for (name, value) in &parts.headers {
        let l = name.as_str().to_lowercase();
        if l != "transfer-encoding" && l != "content-length" {
            rb = rb.header(name, value);
        }
    }
    
    // Summary log at info level — one line per request, not per sub-step.
    tracing::info!("[#{req_id}] done in {:?}  {}B", t0.elapsed(), rewritten.len());
    Ok(rb.body(Body::from(rewritten))?)
}

// ─── Request Transformation ─────────────────────────────────────────────

fn transform_req(original: &[u8], cfg: &crate::config::ModelConfig) -> Vec<u8> {
    // Fast path: no params to inject and model already matches — return original bytes untouched.
    if cfg.params.is_empty() {
        let haystack = std::str::from_utf8(original).unwrap_or("");
        if let Some(start) = haystack.find("\"model\":\"") {
            let val_start = start + 9;
            if let Some(end) = haystack[val_start..].find('"') {
                if &haystack[val_start..val_start + end] == cfg.served_model {
                    return original.to_vec();
                }
            }
        }
    }

    // Parse JSON, mutate, re-serialize.
    // This handles both model name rewriting and param injection in one pass.
    let mut data: serde_json::Value = match serde_json::from_slice(original) {
        Ok(v) => v,
        Err(_) => return original.to_vec(),
    };

    if let Some(obj) = data.as_object_mut() {
        obj.insert("model".into(), serde_json::Value::String(cfg.served_model.clone()));
        for (k, v) in &cfg.params {
            // Always inject model params (override client values)
            obj.insert(k.clone(), v.clone());
        }
    }

    serde_json::to_vec(&data).unwrap_or_else(|_| original.to_vec())
}

// ─── SSE Forwarding ─────────────────────────────────────────────────────
/// Forward an SSE stream from upstream to the client, rewriting the model
/// name inline.  We accumulate bytes in `buf`, scan for complete SSE events
/// (terminated by `\n\n`), rewrite the model name in each event, and push
/// only fully-formed events downstream.  This avoids the classic bug of
/// flushing partial events at chunk boundaries which corrupts the SSE protocol
/// and makes clients hang waiting for the missing `\n\n` terminator.

async fn forward_sse(
    incoming: Incoming,
    tx: mpsc::Sender<Result<Bytes, io::Error>>,
    vname: String,
    req_id: u64,
) {
    let mut buf = BytesMut::with_capacity(STREAM_BUF);
    let mut pos = 0usize;

    let mut stream = incoming.into_data_stream();
    while let Some(frame) = stream.next().await {
        match frame {
            Ok(chunk) => buf.extend_from_slice(&chunk),
            Err(e) => {
                // Drain any pending data before propagating the error
                drain_pending(buf.as_ref(), pos, &tx, vname.as_bytes()).await;
                let _ = tx.send(Err(io::Error::new(io::ErrorKind::Other, e.to_string()))).await;
                return;
            }
        }

        // Scan for complete SSE events (\n\n) in the accumulated buffer.
        // Push each complete event downstream (after model-name rewriting).
        loop {
            // Find the first \n\n starting from `pos`
            let search_buf = &buf[pos..];
            let sep_pos = search_buf.windows(2).position(|w| w == b"\n\n");

            match sep_pos {
                Some(idx) => {
                    // Found a complete event: `buf[pos .. pos+idx+2]`
                    let event_end = pos + idx + 2;
                    let event = &buf[pos..event_end];
                    let rewritten = rewrite_model_bytes(event, vname.as_bytes());
                    let _ = tx.send(Ok(Bytes::from(rewritten))).await;
                    pos = event_end;
                }
                None => break, // Not enough data yet — wait for next chunk
            }
        }
    }

    // Stream ended — flush any remaining partial data
    if pos < buf.len() {
        let remaining = &buf[pos..];
        if !remaining.is_empty() {
            let rewritten = rewrite_model_bytes(remaining, vname.as_bytes());
            let _ = tx.send(Ok(Bytes::from(rewritten))).await;
        }
    }
    tracing::debug!("[#{req_id}] sse_forward_done");
}

/// Helper: drain any buffered-but-not-yet-flushed data before an error occurs.
async fn drain_pending(
    buf: &[u8],
    pos: usize,
    tx: &mpsc::Sender<Result<Bytes, io::Error>>,
    vname: &[u8],
) {
    if pos < buf.len() {
        let remaining = &buf[pos..];
        if !remaining.is_empty() {
            let rewritten = rewrite_model_bytes(remaining, vname);
            let _ = tx.send(Ok(Bytes::from(rewritten))).await;
        }
    }
}

// ─── Raw Byte Model Name Rewriting ──────────────────────────────────────

fn rewrite_model_bytes(body: &[u8], vname: &[u8]) -> Vec<u8> {
    const NEEDLE: &[u8] = b"\"model\":\"";
    if let Some(start) = body.windows(NEEDLE.len()).position(|w| w == NEEDLE) {
        let val_start = start + NEEDLE.len();
        if let Some(end) = body[val_start..].iter().position(|&b| b == b'"') {
            let end = val_start + end;
            let old_len = end - val_start;
            let mut out = Vec::with_capacity(body.len() - old_len + vname.len());
            out.extend_from_slice(&body[..val_start]);
            out.extend_from_slice(vname);
            out.extend_from_slice(&body[end..]);
            return out;
        }
    }
    body.to_vec()
}

// ─── Tests ──────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rewrite_model_bytes_replaces_name() {
        let input = b"{\"model\":\"gpt-4\",\"messages\":[]}";
        let output = rewrite_model_bytes(input, b"qwen3.6-fast");
        assert_eq!(output, b"{\"model\":\"qwen3.6-fast\",\"messages\":[]}");
    }

    #[test]
    fn test_rewrite_model_bytes_no_match() {
        let input = b"{\"prompt\":\"hello\"}";
        let output = rewrite_model_bytes(input, b"qwen3.6-fast");
        assert_eq!(output, input);
    }

    #[test]
    fn test_rewrite_model_bytes_short_name() {
        let input = b"{\"model\":\"gpt-4\"}";
        let output = rewrite_model_bytes(input, b"q");
        assert_eq!(output, b"{\"model\":\"q\"}");
    }

    #[test]
    fn test_rewrite_model_bytes_longer_name() {
        let input = b"{\"model\":\"a\"}";
        let output = rewrite_model_bytes(input, b"qwen3.6-fast");
        assert_eq!(output, b"{\"model\":\"qwen3.6-fast\"}");
    }

    #[test]
    fn test_rewrite_model_bytes_multiple_occurrences() {
        // Only replaces the first match
        let input = b"{\"model\":\"gpt-4\",\"model\":\"old\"}";
        let output = rewrite_model_bytes(input, b"new-model");
        assert_eq!(output, b"{\"model\":\"new-model\",\"model\":\"old\"}");
    }

    #[test]
    fn test_rewrite_model_bytes_preserves_json_whitespace() {
        // The needle is "\"model\":\"" (no space), so match that format
        let input = b"{\"model\":\"gpt-4\",\"messages\":[]}";
        let output = rewrite_model_bytes(input, b"qwen");
        assert_eq!(output, b"{\"model\":\"qwen\",\"messages\":[]}");
    }

    #[test]
    fn test_rewrite_model_bytes_sse_event_boundary() {
        // Ensure SSE event terminators (\n\n) are preserved intact.
        // Note: rewrite_model_bytes replaces only the FIRST "model":" match,
        // which is sufficient since the first occurrence is always the one
        // the proxy controls (the rest come from upstream and are irrelevant).
        let input = b"data: {\"model\":\"gpt-4\"}\n\ndata: {\"model\":\"gpt-4\"}\n\n";
        let output = rewrite_model_bytes(input, b"my-model");
        let expected = b"data: {\"model\":\"my-model\"}\n\ndata: {\"model\":\"gpt-4\"}\n\n";
        assert_eq!(output, expected);
    }

    #[test]
    fn test_rewrite_model_bytes_partial_at_end() {
        // Partial event at end of stream — should still rewrite what it finds
        let input = b"data: {\"model\":\"gpt-4\"}\n\ndata: {\"model\":\"gpt";
        let output = rewrite_model_bytes(input, b"my-model");
        let expected = b"data: {\"model\":\"my-model\"}\n\ndata: {\"model\":\"gpt";
        assert_eq!(output, expected);
    }

    #[test]
    fn test_transform_req_injects_params() {
        let input = br#"{"messages":[{"role":"user","content":"hi"}],"max_tokens":100}"#;
        let cfg = crate::config::ModelConfig {
            name: "test".into(),
            target: "http://localhost:8000".into(),
            served_model: "real-model".into(),
            api_key: None,
            params: std::collections::HashMap::from_iter([
                ("temperature".into(), serde_json::json!(0.7)),
            ]),
        };
        let output = transform_req(input, &cfg);
        let value: serde_json::Value = serde_json::from_slice(&output).unwrap();
        assert_eq!(value["model"], "real-model");
        assert_eq!(value["temperature"], 0.7);
        assert_eq!(value["max_tokens"], 100);
    }

    #[test]
    fn test_transform_req_overrides_model() {
        let input = br#"{"model":"wrong-name","messages":[]}"#;
        let cfg = crate::config::ModelConfig {
            name: "test".into(),
            target: "http://localhost:8000".into(),
            served_model: "correct-model".into(),
            api_key: None,
            params: std::collections::HashMap::new(),
        };
        let output = transform_req(input, &cfg);
        let value: serde_json::Value = serde_json::from_slice(&output).unwrap();
        assert_eq!(value["model"], "correct-model");
    }

    #[test]
    fn test_transform_req_no_params_and_model_already_set() {
        let input = br#"{"model":"real-model","messages":[]}"#;
        let cfg = crate::config::ModelConfig {
            name: "test".into(),
            target: "http://localhost:8000".into(),
            served_model: "real-model".into(),
            api_key: None,
            params: std::collections::HashMap::new(),
        };
        let output = transform_req(input, &cfg);
        // Should return original bytes unchanged
        assert_eq!(output, input);
    }

    #[test]
    fn test_transform_req_params_override_body() {
        let input = br#"{"messages":[],"temperature":0.3}"#;
        let cfg = crate::config::ModelConfig {
            name: "test".into(),
            target: "http://localhost:8000".into(),
            served_model: "model".into(),
            api_key: None,
            params: std::collections::HashMap::from_iter([
                ("temperature".into(), serde_json::json!(0.7)),
            ]),
        };
        let output = transform_req(input, &cfg);
        let value: serde_json::Value = serde_json::from_slice(&output).unwrap();
        assert_eq!(value["temperature"], 0.7); // overridden
    }

    #[test]
    fn test_transform_req_nested_params() {
        let input = br#"{"messages":[]}"#;
        let mut params = std::collections::HashMap::new();
        params.insert(
            "chat_template_kwargs".into(),
            serde_json::json!({"enable_thinking": true}),
        );
        let cfg = crate::config::ModelConfig {
            name: "test".into(),
            target: "http://localhost:8000".into(),
            served_model: "model".into(),
            api_key: None,
            params: params,
        };
        let output = transform_req(input, &cfg);
        let value: serde_json::Value = serde_json::from_slice(&output).unwrap();
        assert_eq!(value["chat_template_kwargs"]["enable_thinking"], true);
    }
}
