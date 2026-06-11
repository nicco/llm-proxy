//! High-performance HTTP forwarder — 0 JSON parses on response path.

use axum::{
    body::Body,
    http::{Request, Response},
};
use bytes::{Bytes, BytesMut};
use http_body_util::BodyExt;
use hyper_util::client::legacy::{connect::HttpConnector, Client};
use std::io;
use std::pin::Pin;
use std::sync::atomic::{AtomicU64, Ordering};
use std::task::{Context, Poll};
use std::time::Instant;
use tokio_stream::Stream;

const STREAM_BUF: usize = 4 * 1024;
static REQ_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Hop-by-hop headers plus the ones we recompute ourselves
/// (content-type / content-length are set explicitly on the outgoing request,
/// so copying the client's values would produce duplicates).
fn skip_request_header(name: &str) -> bool {
    matches!(
        name,
        "host"
            | "connection"
            | "keep-alive"
            | "proxy-connection"
            | "te"
            | "trailer"
            | "transfer-encoding"
            | "upgrade"
            | "content-length"
            | "content-type"
    )
}

fn skip_response_header(name: &str) -> bool {
    matches!(
        name,
        "connection" | "keep-alive" | "transfer-encoding" | "content-length"
    )
}

/// Forward request: transform body, call upstream, rewrite model name in response.
pub async fn proxy(
    upstream_url: &str,
    model_cfg: &crate::config::ModelConfig,
    req: Request<Body>,
    client: &Client<HttpConnector, Body>,
    max_body: usize,
    header_timeout: Option<std::time::Duration>,
) -> Result<Response<Body>, anyhow::Error> {
    let req_id = REQ_COUNTER.fetch_add(1, Ordering::Relaxed);
    let t0 = Instant::now();
    let (parts, body) = req.into_parts();

    // Buffer request body for transformation, bounded by the configured limit.
    let body_bytes = axum::body::to_bytes(body, max_body).await?;

    // Transform the request body (inject params, rewrite model name).
    let transformed = transform_req(&body_bytes, model_cfg);

    let mut builder = hyper::Request::builder()
        .method(parts.method.clone())
        .uri(upstream_url)
        .header("content-type", "application/json")
        .header("content-length", transformed.len());

    let inject_auth = model_cfg.api_key.is_some();
    for (name, value) in parts.headers.iter() {
        // HeaderName is guaranteed lowercase — no per-header allocation needed.
        let n = name.as_str();
        if skip_request_header(n) || (inject_auth && n == "authorization") {
            continue;
        }
        builder = builder.header(name, value);
    }

    if let Some(api_key) = &model_cfg.api_key {
        builder = builder.header("authorization", format!("Bearer {api_key}"));
    }

    let t1 = Instant::now();
    let outgoing = builder.body(Body::from(transformed))?;

    // Bound the wait for response *headers* only — a wedged upstream that
    // accepts connections but never replies would otherwise pin this request
    // (and its client connection) forever.  Body streaming is never timed.
    let resp = match header_timeout {
        Some(limit) => tokio::time::timeout(limit, client.request(outgoing))
            .await
            .map_err(|_| {
                anyhow::anyhow!("upstream sent no response headers within {limit:?} (header timeout)")
            })??,
        None => client.request(outgoing).await?,
    };

    // Log upstream timing at debug level to avoid I/O spam under burst load.
    tracing::debug!("[#{req_id}] upstream: {:?}  status={}", t1.elapsed(), resp.status());

    let (parts, incoming) = resp.into_parts();

    let is_sse = parts
        .headers
        .get("content-type")
        .and_then(|v| v.to_str().ok())
        .map(|s| s.contains("event-stream"))
        .unwrap_or(false);

    let mut rb = Response::builder().status(parts.status);
    for (name, value) in &parts.headers {
        if !skip_response_header(name.as_str()) {
            rb = rb.header(name, value);
        }
    }

    if is_sse {
        // The response body polls the rewriter directly — no relay task, no
        // channel. Client reads drive upstream reads, so backpressure is
        // end-to-end and a client disconnect drops the upstream connection
        // immediately instead of draining it in the background.
        let stream = SseRewrite::new(
            incoming,
            Bytes::copy_from_slice(model_cfg.name.as_bytes()),
            req_id,
        );
        return Ok(rb.body(Body::from_stream(stream))?);
    }

    // Non-streaming: collect and rewrite model name at the byte level.
    let body = incoming.collect().await?.to_bytes();
    let rewritten = rewrite_model_bytes(body, model_cfg.name.as_bytes());

    // Summary log at info level — one line per request, not per sub-step.
    tracing::info!("[#{req_id}] done in {:?}  {}B", t0.elapsed(), rewritten.len());
    Ok(rb.body(Body::from(rewritten))?)
}

// ─── Request Transformation ─────────────────────────────────────────────

fn transform_req(original: &Bytes, cfg: &crate::config::ModelConfig) -> Bytes {
    // Fast path: no params to inject and model already matches — reuse the
    // original buffer (refcounted clone, zero-copy).
    if cfg.params.is_empty() {
        let haystack = std::str::from_utf8(original).unwrap_or("");
        if let Some(start) = haystack.find("\"model\":\"") {
            let val_start = start + 9;
            if let Some(end) = haystack[val_start..].find('"') {
                if haystack[val_start..val_start + end] == cfg.served_model {
                    return original.clone();
                }
            }
        }
    }

    // Parse JSON, mutate, re-serialize.
    // This handles both model name rewriting and param injection in one pass.
    let mut data: serde_json::Value = match serde_json::from_slice(original) {
        Ok(v) => v,
        Err(_) => return original.clone(),
    };

    if let Some(obj) = data.as_object_mut() {
        obj.insert("model".into(), serde_json::Value::String(cfg.served_model.clone()));
        for (k, v) in &cfg.params {
            // Always inject model params (override client values)
            obj.insert(k.clone(), v.clone());
        }
    }

    serde_json::to_vec(&data)
        .map(Bytes::from)
        .unwrap_or_else(|_| original.clone())
}

// ─── SSE Forwarding ─────────────────────────────────────────────────────

/// Streaming SSE rewriter.
///
/// Accumulates upstream bytes, scans for complete SSE events (terminated by
/// `\n\n`), rewrites the model name in each event, and yields only
/// fully-formed events.  This avoids the classic bug of flushing partial
/// events at chunk boundaries, which corrupts the SSE protocol and makes
/// clients hang waiting for the missing `\n\n` terminator.
///
/// Implemented as a `Stream` polled by the response body rather than a
/// spawned relay task writing into a channel, so that under concurrent load:
///  * emitted events are released from the buffer immediately (`split_to`) —
///    memory per stream is bounded by one event, not the whole response;
///  * client backpressure propagates directly to the upstream read;
///  * dropping the response (client disconnect) drops the upstream
///    connection right away instead of draining it in the background.
struct SseRewrite<B> {
    upstream: B,
    buf: BytesMut,
    /// How far `buf` has been scanned without finding an event terminator,
    /// so each new chunk doesn't re-scan from the start of the buffer.
    scanned: usize,
    model: Bytes,
    req_id: u64,
    upstream_done: bool,
    pending_err: Option<io::Error>,
    finished: bool,
}

impl<B> SseRewrite<B> {
    fn new(upstream: B, model: Bytes, req_id: u64) -> Self {
        Self {
            upstream,
            buf: BytesMut::with_capacity(STREAM_BUF),
            scanned: 0,
            model,
            req_id,
            upstream_done: false,
            pending_err: None,
            finished: false,
        }
    }
}

impl<B> Stream for SseRewrite<B>
where
    B: hyper::body::Body<Data = Bytes> + Unpin,
    B::Error: std::fmt::Display,
{
    type Item = Result<Bytes, io::Error>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = self.get_mut();
        loop {
            if this.finished {
                return Poll::Ready(None);
            }

            // Yield the next complete event already sitting in the buffer.
            if let Some(end) = next_event_end(&this.buf, &mut this.scanned) {
                let event = this.buf.split_to(end).freeze();
                this.scanned = 0;
                return Poll::Ready(Some(Ok(rewrite_model_bytes(event, &this.model))));
            }

            if this.upstream_done {
                // Flush a trailing partial event, then a pending error, then end.
                if !this.buf.is_empty() {
                    let rest = this.buf.split().freeze();
                    return Poll::Ready(Some(Ok(rewrite_model_bytes(rest, &this.model))));
                }
                if let Some(e) = this.pending_err.take() {
                    this.finished = true;
                    return Poll::Ready(Some(Err(e)));
                }
                this.finished = true;
                tracing::debug!("[#{}] sse_forward_done", this.req_id);
                return Poll::Ready(None);
            }

            match Pin::new(&mut this.upstream).poll_frame(cx) {
                Poll::Ready(Some(Ok(frame))) => {
                    if let Ok(data) = frame.into_data() {
                        this.buf.extend_from_slice(&data);
                    }
                }
                Poll::Ready(Some(Err(e))) => {
                    this.upstream_done = true;
                    this.pending_err = Some(io::Error::other(e.to_string()));
                }
                Poll::Ready(None) => this.upstream_done = true,
                Poll::Pending => return Poll::Pending,
            }
        }
    }
}

/// Find the end (exclusive, including the `\n\n`) of the first complete SSE
/// event in `buf`, resuming from where the previous scan left off.
fn next_event_end(buf: &[u8], scanned: &mut usize) -> Option<usize> {
    // Back up one byte: the terminator may straddle the previous chunk boundary.
    let start = scanned.saturating_sub(1);
    if let Some(idx) = buf[start..].windows(2).position(|w| w == b"\n\n") {
        return Some(start + idx + 2);
    }
    *scanned = buf.len();
    None
}

// ─── Raw Byte Model Name Rewriting ──────────────────────────────────────

fn rewrite_model_bytes(body: Bytes, vname: &[u8]) -> Bytes {
    const NEEDLE: &[u8] = b"\"model\":\"";
    let Some(start) = body.windows(NEEDLE.len()).position(|w| w == NEEDLE) else {
        return body;
    };
    let val_start = start + NEEDLE.len();
    let Some(end) = body[val_start..].iter().position(|&b| b == b'"') else {
        return body;
    };
    let end = val_start + end;
    if &body[val_start..end] == vname {
        return body; // already the right name — nothing to allocate
    }
    let mut out = BytesMut::with_capacity(body.len() - (end - val_start) + vname.len());
    out.extend_from_slice(&body[..val_start]);
    out.extend_from_slice(vname);
    out.extend_from_slice(&body[end..]);
    out.freeze()
}

// ─── Tests ──────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn rewrite(input: &'static [u8], vname: &[u8]) -> Bytes {
        rewrite_model_bytes(Bytes::from_static(input), vname)
    }

    #[test]
    fn test_rewrite_model_bytes_replaces_name() {
        let output = rewrite(b"{\"model\":\"gpt-4\",\"messages\":[]}", b"qwen3.6-fast");
        assert_eq!(&output[..], b"{\"model\":\"qwen3.6-fast\",\"messages\":[]}");
    }

    #[test]
    fn test_rewrite_model_bytes_no_match() {
        let input: &[u8] = b"{\"prompt\":\"hello\"}";
        let output = rewrite(b"{\"prompt\":\"hello\"}", b"qwen3.6-fast");
        assert_eq!(&output[..], input);
    }

    #[test]
    fn test_rewrite_model_bytes_short_name() {
        let output = rewrite(b"{\"model\":\"gpt-4\"}", b"q");
        assert_eq!(&output[..], b"{\"model\":\"q\"}");
    }

    #[test]
    fn test_rewrite_model_bytes_longer_name() {
        let output = rewrite(b"{\"model\":\"a\"}", b"qwen3.6-fast");
        assert_eq!(&output[..], b"{\"model\":\"qwen3.6-fast\"}");
    }

    #[test]
    fn test_rewrite_model_bytes_multiple_occurrences() {
        // Only replaces the first match
        let output = rewrite(b"{\"model\":\"gpt-4\",\"model\":\"old\"}", b"new-model");
        assert_eq!(&output[..], b"{\"model\":\"new-model\",\"model\":\"old\"}");
    }

    #[test]
    fn test_rewrite_model_bytes_preserves_json_whitespace() {
        // The needle is "\"model\":\"" (no space), so match that format
        let output = rewrite(b"{\"model\":\"gpt-4\",\"messages\":[]}", b"qwen");
        assert_eq!(&output[..], b"{\"model\":\"qwen\",\"messages\":[]}");
    }

    #[test]
    fn test_rewrite_model_bytes_sse_event_boundary() {
        // Ensure SSE event terminators (\n\n) are preserved intact.
        // Note: rewrite_model_bytes replaces only the FIRST "model":" match,
        // which is sufficient since the SSE stream rewrites one event at a time.
        let output = rewrite(
            b"data: {\"model\":\"gpt-4\"}\n\ndata: {\"model\":\"gpt-4\"}\n\n",
            b"my-model",
        );
        let expected: &[u8] = b"data: {\"model\":\"my-model\"}\n\ndata: {\"model\":\"gpt-4\"}\n\n";
        assert_eq!(&output[..], expected);
    }

    #[test]
    fn test_rewrite_model_bytes_partial_at_end() {
        // Partial event at end of stream — should still rewrite what it finds
        let output = rewrite(b"data: {\"model\":\"gpt-4\"}\n\ndata: {\"model\":\"gpt", b"my-model");
        let expected: &[u8] = b"data: {\"model\":\"my-model\"}\n\ndata: {\"model\":\"gpt";
        assert_eq!(&output[..], expected);
    }

    #[test]
    fn test_rewrite_model_bytes_same_name_is_zero_copy() {
        let input = Bytes::from_static(b"{\"model\":\"alias\"}");
        let ptr = input.as_ptr();
        let output = rewrite_model_bytes(input, b"alias");
        assert_eq!(output.as_ptr(), ptr);
    }

    #[tokio::test]
    async fn test_sse_rewrite_stream_handles_chunk_boundaries() {
        use http_body_util::StreamBody;
        use hyper::body::Frame;
        use std::convert::Infallible;
        use tokio_stream::StreamExt;

        // Two events split across three chunks: the event terminator and the
        // "model" key both straddle chunk boundaries.
        let chunks: Vec<Result<Frame<Bytes>, Infallible>> = vec![
            Ok(Frame::data(Bytes::from_static(b"data: {\"model\":\"real\",\"x\":1}\n"))),
            Ok(Frame::data(Bytes::from_static(b"\ndata: {\"mod"))),
            Ok(Frame::data(Bytes::from_static(b"el\":\"real\",\"x\":2}\n\n"))),
        ];
        let body = StreamBody::new(tokio_stream::iter(chunks));
        let mut stream = SseRewrite::new(body, Bytes::from_static(b"alias"), 0);

        let mut events = Vec::new();
        while let Some(item) = stream.next().await {
            events.push(item.unwrap());
        }
        assert_eq!(events.len(), 2);
        assert_eq!(&events[0][..], b"data: {\"model\":\"alias\",\"x\":1}\n\n");
        assert_eq!(&events[1][..], b"data: {\"model\":\"alias\",\"x\":2}\n\n");
    }

    #[tokio::test]
    async fn test_sse_rewrite_stream_releases_consumed_events() {
        use http_body_util::StreamBody;
        use hyper::body::Frame;
        use std::convert::Infallible;
        use tokio_stream::StreamExt;

        // Many events in one chunk: after draining them all, the internal
        // buffer must be empty (consumed data is released, not retained).
        let mut payload = Vec::new();
        for i in 0..100 {
            payload.extend_from_slice(format!("data: {{\"model\":\"real\",\"i\":{i}}}\n\n").as_bytes());
        }
        let chunks: Vec<Result<Frame<Bytes>, Infallible>> =
            vec![Ok(Frame::data(Bytes::from(payload)))];
        let body = StreamBody::new(tokio_stream::iter(chunks));
        let mut stream = SseRewrite::new(body, Bytes::from_static(b"alias"), 0);

        let mut count = 0;
        while let Some(item) = stream.next().await {
            item.unwrap();
            count += 1;
            assert!(stream.buf.capacity() <= 64 * 1024, "buffer should not accumulate");
        }
        assert_eq!(count, 100);
        assert!(stream.buf.is_empty());
    }

    #[tokio::test]
    async fn test_header_timeout_fires_on_silent_upstream() {
        // Upstream accepts the connection but never sends response headers.
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            let (sock, _) = listener.accept().await.unwrap();
            tokio::time::sleep(std::time::Duration::from_secs(60)).await;
            drop(sock);
        });

        let client: Client<HttpConnector, Body> =
            Client::builder(hyper_util::rt::TokioExecutor::new()).build_http();
        let model_cfg = cfg("m", std::collections::HashMap::new());
        let req = Request::builder()
            .method("POST")
            .uri("/x")
            .body(Body::from("{}"))
            .unwrap();

        let err = proxy(
            &format!("http://{addr}/x"),
            &model_cfg,
            req,
            &client,
            4096,
            Some(std::time::Duration::from_millis(100)),
        )
        .await
        .unwrap_err();
        assert!(err.to_string().contains("header timeout"), "got: {err}");
    }

    fn cfg(
        served_model: &str,
        params: std::collections::HashMap<String, serde_json::Value>,
    ) -> crate::config::ModelConfig {
        crate::config::ModelConfig {
            name: "test".into(),
            target: "http://localhost:8000".into(),
            served_model: served_model.into(),
            api_key: None,
            params,
        }
    }

    #[test]
    fn test_transform_req_injects_params() {
        let input = Bytes::from_static(
            br#"{"messages":[{"role":"user","content":"hi"}],"max_tokens":100}"#,
        );
        let cfg = cfg(
            "real-model",
            std::collections::HashMap::from_iter([("temperature".into(), serde_json::json!(0.7))]),
        );
        let output = transform_req(&input, &cfg);
        let value: serde_json::Value = serde_json::from_slice(&output).unwrap();
        assert_eq!(value["model"], "real-model");
        assert_eq!(value["temperature"], 0.7);
        assert_eq!(value["max_tokens"], 100);
    }

    #[test]
    fn test_transform_req_overrides_model() {
        let input = Bytes::from_static(br#"{"model":"wrong-name","messages":[]}"#);
        let cfg = cfg("correct-model", std::collections::HashMap::new());
        let output = transform_req(&input, &cfg);
        let value: serde_json::Value = serde_json::from_slice(&output).unwrap();
        assert_eq!(value["model"], "correct-model");
    }

    #[test]
    fn test_transform_req_no_params_and_model_already_set() {
        let input = Bytes::from_static(br#"{"model":"real-model","messages":[]}"#);
        let cfg = cfg("real-model", std::collections::HashMap::new());
        let output = transform_req(&input, &cfg);
        // Should return original bytes unchanged (and zero-copy)
        assert_eq!(output, input);
        assert_eq!(output.as_ptr(), input.as_ptr());
    }

    #[test]
    fn test_transform_req_params_override_body() {
        let input = Bytes::from_static(br#"{"messages":[],"temperature":0.3}"#);
        let cfg = cfg(
            "model",
            std::collections::HashMap::from_iter([("temperature".into(), serde_json::json!(0.7))]),
        );
        let output = transform_req(&input, &cfg);
        let value: serde_json::Value = serde_json::from_slice(&output).unwrap();
        assert_eq!(value["temperature"], 0.7); // overridden
    }

    #[test]
    fn test_transform_req_nested_params() {
        let input = Bytes::from_static(br#"{"messages":[]}"#);
        let mut params = std::collections::HashMap::new();
        params.insert(
            "chat_template_kwargs".into(),
            serde_json::json!({"enable_thinking": true}),
        );
        let cfg = cfg("model", params);
        let output = transform_req(&input, &cfg);
        let value: serde_json::Value = serde_json::from_slice(&output).unwrap();
        assert_eq!(value["chat_template_kwargs"]["enable_thinking"], true);
    }
}
