//! Rescue parsing — extract tool calls from text the model emitted instead of
//! native tool calls.  Ported from forge's `rescue_tool_call`
//! (src/forge/prompts/templates.py).  Strategies run in order; first hit wins.
//! Every strategy requires the parsed tool name to exist in the request's
//! `tools` array, which guards against rescuing look-alike JSON in prose.

use regex::Regex;
use std::collections::HashSet;
use std::sync::LazyLock;

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct ToolCall {
    pub name: String,
    /// Always a JSON object.
    pub args: serde_json::Value,
}

static THINK_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?is)\[THINK\].*?\[/THINK\]|<think>.*?</think>").unwrap());
static REHEARSAL_RE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"(\w+)\[ARGS\]\s*").unwrap());
static QWEN_FN_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?s)<function=([^>\s]+)>(.*?)</function>").unwrap());
static QWEN_PARAM_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"<parameter=([^>\s]+)>").unwrap());
static MISTRAL_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"\[TOOL_CALLS\](\w+)\s*").unwrap());

pub(crate) fn rescue_tool_calls(content: &str, tool_names: &HashSet<String>) -> Vec<ToolCall> {
    let stripped = THINK_RE.replace_all(content, "");
    let text = stripped.trim();
    if text.is_empty() {
        return Vec::new();
    }

    for strategy in [
        scan_json_candidates,
        scan_rehearsal,
        scan_qwen_xml,
        scan_mistral,
    ] {
        let calls = strategy(text, tool_names);
        if !calls.is_empty() {
            return calls;
        }
    }
    Vec::new()
}

/// Strategy 1: balanced-brace scan for bare or fenced JSON objects of shape
/// `{"tool":…,"args":{…}}` (forge style) or `{"name":…,"arguments":…}`
/// (OpenAI style, arguments either an object or a stringified object).
fn scan_json_candidates(text: &str, tool_names: &HashSet<String>) -> Vec<ToolCall> {
    let mut calls = Vec::new();
    let bytes = text.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] != b'{' {
            i += 1;
            continue;
        }
        let Some(end) = balanced_object_end(bytes, i) else {
            i += 1;
            continue;
        };
        match parse_candidate(&text[i..end], tool_names) {
            Some(call) => {
                calls.push(call);
                i = end;
            }
            None => i += 1,
        }
    }
    calls
}

fn parse_candidate(candidate: &str, tool_names: &HashSet<String>) -> Option<ToolCall> {
    let v: serde_json::Value = serde_json::from_str(candidate).ok()?;
    let obj = v.as_object()?;
    let (name, args) = if let (Some(name), Some(args)) = (obj.get("tool"), obj.get("args")) {
        (name.as_str()?, args.clone())
    } else if let (Some(name), Some(args)) = (obj.get("name"), obj.get("arguments")) {
        (name.as_str()?, args.clone())
    } else {
        return None;
    };
    let args = coerce_args_object(args)?;
    tool_names.contains(name).then(|| ToolCall {
        name: name.to_string(),
        args,
    })
}

/// Arguments must end up a JSON object; a string value is parsed once
/// (OpenAI serializes arguments as a JSON-encoded string).
fn coerce_args_object(args: serde_json::Value) -> Option<serde_json::Value> {
    match args {
        serde_json::Value::Object(_) => Some(args),
        serde_json::Value::String(s) => {
            let inner: serde_json::Value = serde_json::from_str(&s).ok()?;
            inner.is_object().then_some(inner)
        }
        _ => None,
    }
}

/// Strategy 2: rehearsal syntax `tool_name[ARGS]{…}`.
fn scan_rehearsal(text: &str, tool_names: &HashSet<String>) -> Vec<ToolCall> {
    scan_anchored(&REHEARSAL_RE, text, tool_names)
}

/// Strategy 4: Mistral bracket-tag `[TOOL_CALLS]tool_name {…}`.
fn scan_mistral(text: &str, tool_names: &HashSet<String>) -> Vec<ToolCall> {
    scan_anchored(&MISTRAL_RE, text, tool_names)
}

/// Shared shape for strategies 2 and 4: an anchor regex captures the tool
/// name and is immediately followed by a balanced JSON object of arguments.
fn scan_anchored(re: &Regex, text: &str, tool_names: &HashSet<String>) -> Vec<ToolCall> {
    let mut calls = Vec::new();
    for cap in re.captures_iter(text) {
        let name = &cap[1];
        if !tool_names.contains(name) {
            continue;
        }
        let after = cap.get(0).unwrap().end();
        if !text[after..].starts_with('{') {
            continue;
        }
        let Some(end) = balanced_object_end(text.as_bytes(), after) else {
            continue;
        };
        let Ok(args) = serde_json::from_str::<serde_json::Value>(&text[after..end]) else {
            continue;
        };
        if args.is_object() {
            calls.push(ToolCall {
                name: name.to_string(),
                args,
            });
        }
    }
    calls
}

/// Strategy 3: Qwen-coder XML
/// `<function=name><parameter=key>value</parameter>…</function>`.
/// Parameter values stay strings, trimmed of leading/trailing newlines.
fn scan_qwen_xml(text: &str, tool_names: &HashSet<String>) -> Vec<ToolCall> {
    let mut calls = Vec::new();
    for cap in QWEN_FN_RE.captures_iter(text) {
        let name = &cap[1];
        if !tool_names.contains(name) {
            continue;
        }
        let body = &cap[2];
        let mut args = serde_json::Map::new();
        let opens: Vec<_> = QWEN_PARAM_RE.captures_iter(body).collect();
        for (i, pcap) in opens.iter().enumerate() {
            let key = pcap[1].to_string();
            let val_start = pcap.get(0).unwrap().end();
            let val_end = opens
                .get(i + 1)
                .map(|next| next.get(0).unwrap().start())
                .unwrap_or(body.len());
            let mut value = &body[val_start..val_end];
            if let Some(idx) = value.find("</parameter>") {
                value = &value[..idx];
            }
            args.insert(
                key,
                serde_json::Value::String(value.trim_matches('\n').to_string()),
            );
        }
        calls.push(ToolCall {
            name: name.to_string(),
            args: serde_json::Value::Object(args),
        });
    }
    calls
}

/// Find the exclusive end of the balanced JSON object starting at `start`
/// (which must be `{`), respecting quoted strings and escapes.  All control
/// bytes are ASCII, so byte indexing stays on char boundaries.
pub(crate) fn balanced_object_end(bytes: &[u8], start: usize) -> Option<usize> {
    debug_assert_eq!(bytes[start], b'{');
    let mut depth = 0usize;
    let mut in_string = false;
    let mut escaped = false;
    for (offset, &b) in bytes[start..].iter().enumerate() {
        if in_string {
            if escaped {
                escaped = false;
            } else if b == b'\\' {
                escaped = true;
            } else if b == b'"' {
                in_string = false;
            }
            continue;
        }
        match b {
            b'"' => in_string = true,
            b'{' => depth += 1,
            b'}' => {
                depth -= 1;
                if depth == 0 {
                    return Some(start + offset + 1);
                }
            }
            _ => {}
        }
    }
    None
}

// ─── Tests ──────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn tools(names: &[&str]) -> HashSet<String> {
        names.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn test_rescue_fenced_forge_style_json() {
        let content = "I'll search for that.\n```json\n{\"tool\": \"search\", \"args\": {\"query\": \"rust\"}}\n```";
        let calls = rescue_tool_calls(content, &tools(&["search"]));
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "search");
        assert_eq!(calls[0].args["query"], "rust");
    }

    #[test]
    fn test_rescue_openai_style_with_string_arguments() {
        let content = r#"{"name": "search", "arguments": "{\"query\": \"rust\"}"}"#;
        let calls = rescue_tool_calls(content, &tools(&["search"]));
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].args["query"], "rust");
    }

    #[test]
    fn test_rescue_strips_think_blocks() {
        let content = "<think>{\"tool\":\"search\",\"args\":{}} maybe?</think>{\"tool\": \"lookup\", \"args\": {\"id\": 4}}";
        let calls = rescue_tool_calls(content, &tools(&["search", "lookup"]));
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "lookup");
    }

    #[test]
    fn test_rescue_rehearsal_syntax() {
        let content = "search[ARGS]{\"query\": \"rust\"}";
        let calls = rescue_tool_calls(content, &tools(&["search"]));
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "search");
        assert_eq!(calls[0].args["query"], "rust");
    }

    #[test]
    fn test_rescue_qwen_xml() {
        let content = "<function=search>\n<parameter=query>\nrust async\n</parameter>\n<parameter=limit>\n5\n</parameter>\n</function>";
        let calls = rescue_tool_calls(content, &tools(&["search"]));
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].args["query"], "rust async");
        assert_eq!(calls[0].args["limit"], "5");
    }

    #[test]
    fn test_rescue_mistral_bracket_tag() {
        let content = "[TOOL_CALLS]search {\"query\": \"rust\"}";
        let calls = rescue_tool_calls(content, &tools(&["search"]));
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "search");
    }

    #[test]
    fn test_rescue_unknown_tool_rejected() {
        let content = "{\"tool\": \"hack\", \"args\": {}}";
        assert!(rescue_tool_calls(content, &tools(&["search"])).is_empty());
    }

    #[test]
    fn test_rescue_plain_prose_no_calls() {
        let content = "The capital of France is Paris. Here is a set: {1, 2, 3}.";
        assert!(rescue_tool_calls(content, &tools(&["search"])).is_empty());
    }

    #[test]
    fn test_rescue_multiple_calls_in_one_text() {
        let content = "{\"tool\":\"search\",\"args\":{\"q\":\"a\"}}\n{\"tool\":\"search\",\"args\":{\"q\":\"b\"}}";
        let calls = rescue_tool_calls(content, &tools(&["search"]));
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[1].args["q"], "b");
    }

    #[test]
    fn test_balanced_scanner_escaped_and_nested() {
        let content =
            r#"{"tool": "search", "args": {"q": "brace \" } in {string}", "n": {"x": 1}}}"#;
        let calls = rescue_tool_calls(content, &tools(&["search"]));
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].args["q"], "brace \" } in {string}");
        assert_eq!(calls[0].args["n"]["x"], 1);
    }

    #[test]
    fn test_args_must_be_object() {
        let content = "{\"tool\": \"search\", \"args\": [1,2]}";
        assert!(rescue_tool_calls(content, &tools(&["search"])).is_empty());
    }
}
