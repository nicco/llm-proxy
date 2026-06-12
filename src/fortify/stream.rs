//! Hold-and-release machinery for fortified SSE streams.
//!
//! Prose is forwarded live (with a small guard lag so a tool-call marker
//! straddling delta boundaries is still caught).  The moment the response
//! turns into a tool call — native `tool_calls` deltas or a malformed-call
//! text marker — forwarding stops and the rest is buffered.  At end of
//! stream the held tail is validated/rescued and the fixed events emitted.
//!
//! Retry (re-asking the model) is only possible while nothing has been
//! forwarded; `HoldMachine` therefore stays outbox-empty through SNIFF and
//! hold-from-start, letting `fortify::run` keep the retry option until the
//! first forwardable byte exists.

use bytes::{Bytes, BytesMut};
use std::collections::{HashSet, VecDeque};
use std::io;
use std::pin::Pin;
use std::task::{Context, Poll};
use tokio_stream::Stream;

use crate::forward::{next_event_end, rewrite_model_bytes};

/// Markers that flip RELEASE into HOLD.  Conservative, high-precision set —
/// the full rescue parser runs over the complete text at end of stream, so a
/// missed exotic format only costs rescue for already-forwarded text.
const MARKERS: &[&str] = &["```", "[TOOL_CALLS]", "<function=", "<tool_call>", "[ARGS]"];
/// Longest marker is `[TOOL_CALLS]` (12 bytes); retaining this many bytes of
/// content behind the live edge guarantees a straddling marker is never
/// split across the forwarded/held boundary.
const GUARD: usize = 15;
/// Things SNIFF must wait out when content starts with a prefix of them.
const SNIFF_PREFIXES: &[&str] = &[
    "```",
    "[TOOL_CALLS]",
    "<function=",
    "<tool_call>",
    "<think>",
];
/// Give up sniffing and release once this much content looks like prose.
const SNIFF_CAP: usize = 64;
const THINK_CLOSE: &str = "</think>";

/// Largest byte index `<= i` that lies on a char boundary of `s`.  Cursor
/// arithmetic (`len - GUARD` etc.) can land inside a multi-byte codepoint
/// (emoji, em-dashes…), and slicing there panics.
fn floor_char_boundary(s: &str, mut i: usize) -> usize {
    if i >= s.len() {
        return s.len();
    }
    while !s.is_char_boundary(i) {
        i -= 1;
    }
    i
}

// ─── Delta accumulation ─────────────────────────────────────────────────

#[derive(Debug, Default, Clone)]
pub(crate) struct AccToolCall {
    pub index: u64,
    pub id: Option<String>,
    pub name: String,
    pub arguments: String,
}

/// Folds `chat.completion.chunk` deltas back into a complete message.
#[derive(Debug, Default)]
pub(crate) struct MessageAccumulator {
    pub id: Option<String>,
    pub created: Option<u64>,
    pub system_fingerprint: Option<serde_json::Value>,
    pub content: String,
    pub reasoning_len: usize,
    pub tool_calls: Vec<AccToolCall>,
    pub finish_reason: Option<String>,
    pub usage: Option<serde_json::Value>,
}

impl MessageAccumulator {
    /// Feed one complete SSE event (`data: {...}\n\n`, possibly with
    /// comment/other lines around it).
    pub fn feed_event(&mut self, event: &[u8]) {
        for line in event.split(|&b| b == b'\n') {
            let Some(payload) = line.strip_prefix(b"data:") else {
                continue;
            };
            let payload = payload.strip_prefix(b" ").unwrap_or(payload);
            if payload.starts_with(b"[DONE]") {
                continue;
            }
            if let Ok(chunk) = serde_json::from_slice::<serde_json::Value>(payload) {
                self.feed_chunk(&chunk);
            }
        }
    }

    fn feed_chunk(&mut self, chunk: &serde_json::Value) {
        if self.id.is_none() {
            self.id = chunk.get("id").and_then(|v| v.as_str()).map(String::from);
        }
        if self.created.is_none() {
            self.created = chunk.get("created").and_then(|v| v.as_u64());
        }
        if self.system_fingerprint.is_none() {
            self.system_fingerprint = chunk.get("system_fingerprint").cloned();
        }
        if let Some(usage) = chunk.get("usage").filter(|u| u.is_object()) {
            self.usage = Some(usage.clone());
        }
        let Some(choice) = chunk.get("choices").and_then(|c| c.get(0)) else {
            return;
        };
        if let Some(reason) = choice.get("finish_reason").and_then(|v| v.as_str()) {
            self.finish_reason = Some(reason.to_string());
        }
        let Some(delta) = choice.get("delta") else {
            return;
        };
        if let Some(text) = delta.get("content").and_then(|v| v.as_str()) {
            self.content.push_str(text);
        }
        // vLLM emits `reasoning_content`; some builds (e.g. eugr nightly)
        // and OpenRouter-style APIs use `reasoning`.
        for key in ["reasoning_content", "reasoning"] {
            if let Some(text) = delta.get(key).and_then(|v| v.as_str()) {
                self.reasoning_len += text.len();
            }
        }
        if let Some(calls) = delta.get("tool_calls").and_then(|v| v.as_array()) {
            for tc in calls {
                let index = tc.get("index").and_then(|v| v.as_u64()).unwrap_or(0);
                let slot = match self.tool_calls.iter_mut().find(|c| c.index == index) {
                    Some(slot) => slot,
                    None => {
                        self.tool_calls.push(AccToolCall {
                            index,
                            ..Default::default()
                        });
                        self.tool_calls.last_mut().unwrap()
                    }
                };
                if slot.id.is_none() {
                    slot.id = tc.get("id").and_then(|v| v.as_str()).map(String::from);
                }
                if let Some(f) = tc.get("function") {
                    if let Some(name) = f.get("name").and_then(|v| v.as_str()) {
                        slot.name.push_str(name);
                    }
                    if let Some(args) = f.get("arguments").and_then(|v| v.as_str()) {
                        slot.arguments.push_str(args);
                    }
                }
            }
        }
    }

    pub fn has_tool_calls(&self) -> bool {
        !self.tool_calls.is_empty()
    }

    /// Assemble the assistant message this stream described, in the same
    /// shape a non-streaming `choices[0].message` would have.
    pub fn message_json(&self) -> serde_json::Value {
        let mut msg = serde_json::json!({"role": "assistant"});
        msg["content"] = if self.content.is_empty() && self.has_tool_calls() {
            serde_json::Value::Null
        } else {
            serde_json::Value::String(self.content.clone())
        };
        if self.has_tool_calls() {
            let calls: Vec<serde_json::Value> = self
                .tool_calls
                .iter()
                .enumerate()
                .map(|(i, c)| {
                    serde_json::json!({
                        "id": c.id.clone().unwrap_or_else(|| format!("call_{i}")),
                        "type": "function",
                        "function": {"name": c.name, "arguments": c.arguments},
                    })
                })
                .collect();
            msg["tool_calls"] = serde_json::Value::Array(calls);
        }
        msg
    }
}

// ─── Hold machine ───────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Phase {
    Sniff,
    Release,
    Hold,
}

struct PendingEvent {
    raw: Bytes,
    /// Cumulative `acc.content` length after this event was fed.
    content_end: usize,
}

pub(crate) struct HoldMachine {
    pub acc: MessageAccumulator,
    alias: Bytes,
    req_id: u64,
    pub tool_names: HashSet<String>,
    pub rescue_enabled: bool,
    pub strip_respond: bool,

    // SSE event framing (same approach as forward::SseRewrite).
    buf: BytesMut,
    scanned: usize,

    phase: Phase,
    /// Content offset SNIFF decisions apply from (re-sniff after think).
    sniff_from: usize,
    /// Inside a `<think>` block streamed live — only watch for its close.
    in_think: bool,
    /// Content offset already scanned for markers in RELEASE.
    scan_cursor: usize,
    /// Reasoning bytes already flushed live during SNIFF.
    flushed_reasoning: usize,
    /// Cumulative content length covered by flushed (forwarded) events.
    flushed_content_end: usize,

    pending: VecDeque<PendingEvent>,
    pub outbox: VecDeque<Bytes>,
    /// Raw withheld events once HOLD is entered.
    pub held: Vec<Bytes>,
    /// Content offset where unforwarded (held) text begins.
    pub held_text_start: usize,
    /// Content offset of the marker that triggered HOLD (== held_text_start
    /// when holding from the start or on a native tool_calls delta).
    pub hold_marker_pos: usize,
    /// Every event in arrival order — for byte-faithful replay when nothing
    /// was forwarded.  `Bytes` clones are refcounted handles, not copies.
    pub raw_log: Vec<Bytes>,
}

impl HoldMachine {
    pub fn new(
        alias: &str,
        tool_names: HashSet<String>,
        rescue_enabled: bool,
        strip_respond: bool,
        req_id: u64,
    ) -> Self {
        Self {
            acc: MessageAccumulator::default(),
            alias: Bytes::copy_from_slice(alias.as_bytes()),
            req_id,
            tool_names,
            rescue_enabled,
            strip_respond,
            buf: BytesMut::with_capacity(4 * 1024),
            scanned: 0,
            phase: Phase::Sniff,
            sniff_from: 0,
            in_think: false,
            scan_cursor: 0,
            flushed_reasoning: 0,
            flushed_content_end: 0,
            pending: VecDeque::new(),
            outbox: VecDeque::new(),
            held: Vec::new(),
            held_text_start: 0,
            hold_marker_pos: 0,
            raw_log: Vec::new(),
        }
    }

    pub fn alias_str(&self) -> &str {
        std::str::from_utf8(&self.alias).unwrap_or("")
    }

    /// Trailing bytes that never formed a complete SSE event (no `\n\n`).
    pub fn remnant(&mut self) -> Option<Bytes> {
        (!self.buf.is_empty()).then(|| self.buf.split().freeze())
    }

    /// Feed raw upstream bytes; complete SSE events are processed, partial
    /// ones wait in the framing buffer.
    pub fn feed_bytes(&mut self, data: &[u8]) {
        self.buf.extend_from_slice(data);
        while let Some(end) = next_event_end(&self.buf, &mut self.scanned) {
            let event = self.buf.split_to(end).freeze();
            self.scanned = 0;
            self.process_event(event);
        }
    }

    fn process_event(&mut self, raw: Bytes) {
        self.raw_log.push(raw.clone());
        let had_tool_delta = self.acc.has_tool_calls();
        self.acc.feed_event(&raw);

        if self.phase == Phase::Hold {
            self.held.push(raw);
            return;
        }
        self.pending.push_back(PendingEvent {
            raw,
            content_end: self.acc.content.len(),
        });

        // A native tool_calls delta is an unambiguous hold trigger in any
        // phase — no prose can be hiding in it.
        if !had_tool_delta && self.acc.has_tool_calls() {
            // The event carrying the delta has content_end == current total,
            // which the flush-clean sweep would forward — pull it out first
            // so it is always held.
            let current = self.pending.pop_back().unwrap();
            let pos = self.acc.content.len();
            // In SNIFF nothing is flushed — keeping everything held
            // preserves the retry option.
            let flush_clean = self.phase == Phase::Release;
            self.enter_hold(pos, flush_clean);
            self.held.push(current.raw);
            return;
        }

        match self.phase {
            Phase::Sniff => self.drain_sniff(),
            Phase::Release => self.drain_release(),
            Phase::Hold => {}
        }
    }

    fn drain_sniff(&mut self) {
        let text = &self.acc.content[self.sniff_from..];
        let t = text.trim_start();
        if t.is_empty() {
            // No content yet.  Role chunks stay pending (holding them keeps
            // retry alive), but live reasoning deltas are forwarded — a
            // thinking model may reason for thousands of tokens before its
            // first content byte.
            if self.acc.reasoning_len > self.flushed_reasoning {
                self.flush_all_pending();
                self.flushed_reasoning = self.acc.reasoning_len;
            }
            return;
        }
        let lead_ws = text.len() - t.len();
        let t_start = self.sniff_from + lead_ws;

        if t.starts_with("<think>") {
            // Stream the think block live; re-sniff after it closes.
            self.phase = Phase::Release;
            self.in_think = true;
            self.scan_cursor = t_start;
            self.drain_release();
            return;
        }
        if t.starts_with('{') || MARKERS.iter().any(|m| t.starts_with(m)) {
            self.enter_hold(t_start, false);
            return;
        }
        let may_grow_into_marker = SNIFF_PREFIXES
            .iter()
            .any(|m| m.len() > t.len() && m.starts_with(t));
        if may_grow_into_marker && t.len() < SNIFF_CAP {
            return; // wait for more bytes
        }
        // Prose — go live.
        self.phase = Phase::Release;
        self.scan_cursor = self.sniff_from;
        self.drain_release();
    }

    fn drain_release(&mut self) {
        if self.in_think {
            match self.acc.content[self.scan_cursor..].find(THINK_CLOSE) {
                Some(rel) => {
                    let after = self.scan_cursor + rel + THINK_CLOSE.len();
                    self.in_think = false;
                    self.phase = Phase::Sniff;
                    self.sniff_from = after;
                    self.scan_cursor = after;
                    self.flush_pending_below(after);
                    self.drain_sniff();
                }
                None => {
                    // Only the think-close tag matters inside the block —
                    // fences etc. are common in reasoning and must stream.
                    self.scan_cursor = floor_char_boundary(
                        &self.acc.content,
                        self.acc.content.len().saturating_sub(THINK_CLOSE.len() - 1),
                    )
                    .max(self.scan_cursor);
                    self.flush_with_guard();
                }
            }
            return;
        }

        match self.find_marker(self.scan_cursor) {
            Some(pos) => self.enter_hold(pos, true),
            None => {
                self.scan_cursor = floor_char_boundary(
                    &self.acc.content,
                    self.acc.content.len().saturating_sub(GUARD),
                )
                .max(self.scan_cursor);
                self.flush_with_guard();
            }
        }
    }

    fn find_marker(&self, from: usize) -> Option<usize> {
        // `from` is kept on a char boundary by the scan-cursor updates, but
        // slicing a multi-byte codepoint would panic the whole request task —
        // snap down defensively.
        let from = floor_char_boundary(&self.acc.content, from);
        let hay = &self.acc.content[from..];
        MARKERS
            .iter()
            .filter_map(|m| hay.find(m))
            .min()
            .map(|rel| from + rel)
    }

    /// Enter HOLD at `marker_pos`.  In RELEASE (`flush_clean`), events whose
    /// content lies entirely before the marker are still forwarded; in SNIFF
    /// everything stays held so retry remains possible.
    fn enter_hold(&mut self, marker_pos: usize, flush_clean: bool) {
        if flush_clean {
            while let Some(front) = self.pending.front() {
                if front.content_end <= marker_pos {
                    let ev = self.pending.pop_front().unwrap();
                    self.flush_event(ev);
                } else {
                    break;
                }
            }
        }
        self.held_text_start = self.flushed_content_end;
        self.hold_marker_pos = marker_pos.max(self.held_text_start);
        self.held.extend(self.pending.drain(..).map(|e| e.raw));
        self.phase = Phase::Hold;
        tracing::debug!(
            "[#{}] fortify hold at content[{}..] (forwarded {}B)",
            self.req_id,
            self.hold_marker_pos,
            self.flushed_content_end
        );
    }

    /// Forward queued events, always retaining the newest event plus enough
    /// trailing content (GUARD bytes) that a marker straddling the live edge
    /// can never end up partially forwarded.
    fn flush_with_guard(&mut self) {
        let total = self.acc.content.len();
        while self.pending.len() > 1 {
            let front = self.pending.front().unwrap();
            if front.content_end + GUARD <= total {
                let ev = self.pending.pop_front().unwrap();
                self.flush_event(ev);
            } else {
                break;
            }
        }
    }

    fn flush_pending_below(&mut self, offset: usize) {
        while let Some(front) = self.pending.front() {
            if front.content_end <= offset {
                let ev = self.pending.pop_front().unwrap();
                self.flush_event(ev);
            } else {
                break;
            }
        }
    }

    fn flush_all_pending(&mut self) {
        while let Some(ev) = self.pending.pop_front() {
            self.flush_event(ev);
        }
    }

    fn flush_event(&mut self, ev: PendingEvent) {
        self.flushed_content_end = ev.content_end;
        self.outbox
            .push_back(rewrite_model_bytes(ev.raw, &self.alias));
    }

    /// Upstream ended while committed to a live response (bytes already
    /// forwarded).  Fix the held tail in place: rescued/validated tool calls
    /// become synthesized deltas; anything unfixable is released verbatim —
    /// never worse than plain passthrough.
    pub fn finish_committed(&mut self) {
        let remnant = (!self.buf.is_empty()).then(|| self.buf.split().freeze());

        if self.phase != Phase::Hold {
            self.flush_all_pending();
            if let Some(r) = remnant {
                self.outbox.push_back(rewrite_model_bytes(r, &self.alias));
            }
            return;
        }

        if let Some(r) = remnant {
            self.held.push(r);
        }

        match super::fix_held_tail(self) {
            Some(events) => self.outbox.extend(events),
            None => {
                // Release verbatim (model alias still rewritten per event).
                let held = std::mem::take(&mut self.held);
                for ev in held {
                    self.outbox.push_back(rewrite_model_bytes(ev, &self.alias));
                }
            }
        }
        tracing::debug!("[#{}] fortify stream done", self.req_id);
    }
}

// ─── Committed response body ────────────────────────────────────────────

/// Response body for a fortified stream once live forwarding has begun.
/// Polled directly by the client response (no relay task): backpressure
/// propagates to the upstream read and a client disconnect drops the
/// upstream connection immediately.
pub(crate) struct CommittedStream<B> {
    upstream: B,
    machine: HoldMachine,
    upstream_done: bool,
    pending_err: Option<io::Error>,
    finished: bool,
}

impl<B> CommittedStream<B> {
    pub fn new(upstream: B, machine: HoldMachine) -> Self {
        Self {
            upstream,
            machine,
            upstream_done: false,
            pending_err: None,
            finished: false,
        }
    }
}

impl<B> Stream for CommittedStream<B>
where
    B: hyper::body::Body<Data = Bytes> + Unpin,
    B::Error: std::fmt::Display,
{
    type Item = Result<Bytes, io::Error>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = self.get_mut();
        loop {
            if let Some(bytes) = this.machine.outbox.pop_front() {
                return Poll::Ready(Some(Ok(bytes)));
            }
            if this.finished {
                // Outbox is drained (checked above); surface a deferred
                // upstream error before ending the stream.
                if let Some(e) = this.pending_err.take() {
                    return Poll::Ready(Some(Err(e)));
                }
                return Poll::Ready(None);
            }
            if this.upstream_done {
                this.machine.finish_committed();
                this.finished = true;
                continue;
            }
            match Pin::new(&mut this.upstream).poll_frame(cx) {
                Poll::Ready(Some(Ok(frame))) => {
                    if let Ok(data) = frame.into_data() {
                        this.machine.feed_bytes(&data);
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

// ─── Tests ──────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use http_body_util::StreamBody;
    use hyper::body::Frame;
    use std::convert::Infallible;
    use tokio_stream::StreamExt;

    fn machine() -> HoldMachine {
        let tools: HashSet<String> = ["search".to_string()].into_iter().collect();
        HoldMachine::new("alias", tools, true, false, 0)
    }

    fn role_ev() -> String {
        chunk_ev(r#"{"role":"assistant"}"#)
    }

    fn content_ev(text: &str) -> String {
        chunk_ev(&format!(
            r#"{{"content":{}}}"#,
            serde_json::to_string(text).unwrap()
        ))
    }

    fn tool_ev(name: &str, args: &str) -> String {
        chunk_ev(&format!(
            r#"{{"tool_calls":[{{"index":0,"id":"call_1","type":"function","function":{{"name":"{name}","arguments":{}}}}}]}}"#,
            serde_json::to_string(args).unwrap()
        ))
    }

    fn chunk_ev(delta: &str) -> String {
        format!(
            "data: {{\"id\":\"c1\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"real\",\"choices\":[{{\"index\":0,\"delta\":{delta},\"finish_reason\":null}}]}}\n\n"
        )
    }

    fn finish_ev() -> String {
        "data: {\"id\":\"c1\",\"model\":\"real\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n".to_string()
    }

    fn done_ev() -> String {
        "data: [DONE]\n\n".to_string()
    }

    async fn drive(events: Vec<String>) -> (Vec<Bytes>, String) {
        let chunks: Vec<Result<Frame<Bytes>, Infallible>> = events
            .into_iter()
            .map(|e| Ok(Frame::data(Bytes::from(e))))
            .collect();
        let body = StreamBody::new(tokio_stream::iter(chunks));
        let mut stream = CommittedStream::new(body, machine());
        let mut out = Vec::new();
        while let Some(item) = stream.next().await {
            out.push(item.unwrap());
        }
        let all = out
            .iter()
            .map(|b| String::from_utf8_lossy(b).into_owned())
            .collect::<String>();
        (out, all)
    }

    #[test]
    fn test_accumulator_folds_deltas() {
        let mut acc = MessageAccumulator::default();
        acc.feed_event(role_ev().as_bytes());
        acc.feed_event(content_ev("hel").as_bytes());
        acc.feed_event(content_ev("lo").as_bytes());
        acc.feed_event(finish_ev().as_bytes());
        acc.feed_event(done_ev().as_bytes());
        assert_eq!(acc.content, "hello");
        assert_eq!(acc.id.as_deref(), Some("c1"));
        assert_eq!(acc.finish_reason.as_deref(), Some("stop"));
        assert!(!acc.has_tool_calls());
    }

    #[test]
    fn test_accumulator_merges_tool_call_fragments() {
        let mut acc = MessageAccumulator::default();
        acc.feed_event(tool_ev("search", "{\"q\":").as_bytes());
        acc.feed_event(tool_ev("", "\"rust\"}").as_bytes());
        assert_eq!(acc.tool_calls.len(), 1);
        assert_eq!(acc.tool_calls[0].name, "search");
        assert_eq!(acc.tool_calls[0].arguments, "{\"q\":\"rust\"}");
    }

    #[test]
    fn test_prose_releases_with_guard_lag() {
        let mut m = machine();
        m.feed_bytes(role_ev().as_bytes());
        m.feed_bytes(content_ev("The capital of France ").as_bytes());
        m.feed_bytes(content_ev("is Paris, and it has ").as_bytes());
        m.feed_bytes(content_ev("a lot of history.").as_bytes());
        // Role + first content events flush; the live edge stays held.
        assert!(!m.outbox.is_empty(), "prose should be forwarded live");
        let sent: String = m
            .outbox
            .iter()
            .map(|b| String::from_utf8_lossy(b).into_owned())
            .collect();
        assert!(sent.contains("The capital of France"));
        assert!(
            sent.contains("\"model\":\"alias\""),
            "model rewritten: {sent}"
        );
        assert!(
            !sent.contains("a lot of history"),
            "live edge must stay held"
        );
    }

    #[test]
    fn test_reasoning_deltas_flush_live_during_sniff() {
        let mut m = machine();
        m.feed_bytes(role_ev().as_bytes());
        // Both spellings seen in the wild.
        m.feed_bytes(chunk_ev(r#"{"reasoning":"thinking about it"}"#).as_bytes());
        assert!(
            !m.outbox.is_empty(),
            "reasoning must stream live while sniffing"
        );
        m.feed_bytes(chunk_ev(r#"{"reasoning_content":"more thoughts"}"#).as_bytes());
        // A tool call after streamed reasoning still holds (rescue-only).
        m.feed_bytes(tool_ev("search", "{}").as_bytes());
        m.outbox.clear();
        m.feed_bytes(finish_ev().as_bytes());
        assert!(m.outbox.is_empty(), "tool call after reasoning must hold");
    }

    #[tokio::test]
    async fn test_multibyte_content_does_not_panic() {
        // Production crash: scan_cursor advanced to len - GUARD landed inside
        // '✅' / '—' and slicing panicked, killing the request task.
        // "Launching " is 10 bytes, each 🚀 is 4: after this delta the guard
        // cursor (len - 15 = 11) lands inside the first 🚀 (bytes 10..14).
        let (_, all) = drive(vec![
            role_ev(),
            content_ev("Launching 🚀🚀🚀🚀"),
            content_ev("Server is running ✅ — everything looks good — ✅✅✅ "),
            content_ev("the deploy — succeeded ✅ and the tests — passed ✅."),
            content_ev("More ✅—✅—✅—✅—✅—✅—✅—✅ padding to move the guard around."),
            finish_ev(),
            done_ev(),
        ])
        .await;
        assert!(all.contains("Server is running ✅"));
        assert!(
            all.ends_with("data: [DONE]\n\n"),
            "stream must end cleanly: {all}"
        );
    }

    #[tokio::test]
    async fn test_multibyte_in_think_block_does_not_panic() {
        let (_, all) = drive(vec![
            role_ev(),
            content_ev("<think>weighing options — ✅ pros — ❌ cons — more — thought ✅"),
            content_ev(" still thinking ✅—✅—✅</think>The answer is 42."),
            finish_ev(),
            done_ev(),
        ])
        .await;
        assert!(all.contains("The answer is 42."));
        assert!(all.ends_with("data: [DONE]\n\n"));
    }

    #[test]
    fn test_prose_then_native_tool_call_holds_delta_event() {
        let mut m = machine();
        m.feed_bytes(role_ev().as_bytes());
        m.feed_bytes(
            content_ev("Let me check the weather for you right now, one moment. ").as_bytes(),
        );
        m.feed_bytes(
            content_ev("Querying the service for current conditions in the city. ").as_bytes(),
        );
        m.outbox.clear(); // prose already forwarded — only watch what follows
        m.feed_bytes(tool_ev("search", "{\"q\":\"x\"}").as_bytes());
        let leaked: String = m
            .outbox
            .iter()
            .map(|b| String::from_utf8_lossy(b).into_owned())
            .collect();
        assert!(
            !leaked.contains("tool_calls"),
            "the event carrying the tool_calls delta must be held, not forwarded: {leaked}"
        );
    }

    #[test]
    fn test_native_tool_call_holds_everything() {
        let mut m = machine();
        m.feed_bytes(role_ev().as_bytes());
        m.feed_bytes(tool_ev("search", "{\"q\":\"rust\"}").as_bytes());
        m.feed_bytes(finish_ev().as_bytes());
        assert!(
            m.outbox.is_empty(),
            "nothing may be forwarded for a tool call"
        );
        assert!(m.acc.has_tool_calls());
        assert_eq!(m.raw_log.len(), 3);
    }

    #[test]
    fn test_leading_marker_holds_from_start() {
        let mut m = machine();
        m.feed_bytes(role_ev().as_bytes());
        m.feed_bytes(content_ev("[TOOL_CALLS]search {\"q\"").as_bytes());
        m.feed_bytes(content_ev(": \"rust\"}").as_bytes());
        assert!(m.outbox.is_empty(), "marker at content start must hold all");
    }

    #[test]
    fn test_sniff_waits_on_marker_prefix() {
        let mut m = machine();
        m.feed_bytes(role_ev().as_bytes());
        m.feed_bytes(content_ev("[TOOL_").as_bytes());
        assert!(m.outbox.is_empty(), "prefix of a marker must keep sniffing");
        m.feed_bytes(content_ev("CALLS]search ").as_bytes());
        assert!(m.outbox.is_empty(), "completed marker must hold");
    }

    #[tokio::test]
    async fn test_prose_then_malformed_tool_call_is_rescued() {
        let (_, all) = drive(vec![
            role_ev(),
            content_ev("Let me look that up for you right away. "),
            content_ev("I will use the search tool to do it. "),
            content_ev("[TOOL_CALLS]search {\"q\": \"rust\"}"),
            finish_ev(),
            done_ev(),
        ])
        .await;
        assert!(all.contains("Let me look that up"));
        assert!(
            !all.contains("[TOOL_CALLS]"),
            "marker text must not reach the client: {all}"
        );
        assert!(
            all.contains("\"tool_calls\""),
            "rescued call missing: {all}"
        );
        assert!(all.contains("call_rescued_0"));
        assert!(all.contains("\"finish_reason\":\"tool_calls\""));
        assert!(all.ends_with("data: [DONE]\n\n"));
    }

    #[tokio::test]
    async fn test_false_positive_fence_released_verbatim() {
        let (_, all) = drive(vec![
            role_ev(),
            content_ev("Here is an example config you can copy: "),
            content_ev("```json\n{\"foo\": 1}\n``` "),
            content_ev("adjust it as needed."),
            finish_ev(),
            done_ev(),
        ])
        .await;
        assert!(
            all.contains("```json"),
            "non-tool fence must be released: {all}"
        );
        assert!(all.contains("adjust it as needed."));
        assert!(all.contains("\"finish_reason\":\"stop\""));
        assert!(all.ends_with("data: [DONE]\n\n"));
        assert!(
            !all.contains("tool_calls\":[{"),
            "no tool calls should be synthesized"
        );
    }

    #[tokio::test]
    async fn test_think_block_streams_live_then_resniffs() {
        let chunks: Vec<Result<Frame<Bytes>, Infallible>> = vec![
            Ok(Frame::data(Bytes::from(role_ev()))),
            Ok(Frame::data(Bytes::from(content_ev(
                "<think>The user wants rust info, I should search. Lots of reasoning here to pad this out beyond the guard.",
            )))),
            Ok(Frame::data(Bytes::from(content_ev("</think>")))),
            Ok(Frame::data(Bytes::from(content_ev(
                "{\"tool\": \"search\", \"args\": {\"q\": \"rust\"}}",
            )))),
            Ok(Frame::data(Bytes::from(finish_ev()))),
            Ok(Frame::data(Bytes::from(done_ev()))),
        ];
        let body = StreamBody::new(tokio_stream::iter(chunks));
        let mut stream = CommittedStream::new(body, machine());
        let mut all = String::new();
        let mut saw_think_early = false;
        let mut count = 0;
        while let Some(item) = stream.next().await {
            let s = String::from_utf8_lossy(&item.unwrap()).into_owned();
            count += 1;
            // The think block must be forwarded before the stream ends.
            if s.contains("<think>") && count <= 2 {
                saw_think_early = true;
            }
            all.push_str(&s);
        }
        assert!(saw_think_early, "think content must stream live: {all}");
        assert!(
            all.contains("call_rescued_0"),
            "post-think JSON call must be rescued: {all}"
        );
        assert!(
            !all.contains("{\"tool\":"),
            "raw tool JSON must not reach the client: {all}"
        );
    }

    #[tokio::test]
    async fn test_marker_straddling_chunk_boundary_within_event_stream() {
        let (_, all) = drive(vec![
            role_ev(),
            content_ev("Sure, searching now for the thing you asked about. "),
            content_ev("<function="),
            content_ev("search>\n<parameter=q>\nrust\n</parameter>\n</function>"),
            finish_ev(),
            done_ev(),
        ])
        .await;
        assert!(!all.contains("<function="), "xml call must not leak: {all}");
        assert!(
            all.contains("call_rescued_0"),
            "xml call must be rescued: {all}"
        );
        // arguments is a JSON-encoded string inside the delta, so quotes are escaped
        assert!(all.contains(r#"{\"q\":\"rust\"}"#), "{all}");
    }
}
