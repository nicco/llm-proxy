//! Smart system-prompt injection.
//!
//! For tools-bearing chat-completions requests on opted-in models, inject
//! configured instruction blocks into the system message — the same effect
//! as embedding agent rules in the chat template, but visible in config,
//! per-alias, hot-swappable with a proxy restart (1s) instead of a model
//! reload (minutes), and selective: blocks can match on toolset keywords or
//! tool_choice, and injection is skipped entirely for clients that already
//! ship a large system prompt of their own.

use crate::config::InjectConfig;
use serde_json::Value;

/// Mutate `body` in place; returns whether anything was injected.
pub(crate) fn apply(body: &mut Value, cfg: &InjectConfig, req_id: u64) -> bool {
    let Some(tools) = body.get("tools").and_then(|v| v.as_array()) else {
        return false;
    };
    if tools.is_empty() {
        return false;
    }

    // Clients with big system prompts bring their own agent discipline.
    if let Some(limit) = cfg.skip_if_system_over {
        if existing_system_len(body) > limit {
            tracing::debug!("[#{req_id}] inject skipped: system message over {limit}B");
            return false;
        }
    }

    // Lowercased haystack of tool names + descriptions for keyword matching.
    let haystack: String = tools
        .iter()
        .flat_map(|t| {
            ["name", "description"]
                .iter()
                .filter_map(|k| t.pointer(&format!("/function/{k}"))?.as_str())
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>()
        .join(" ")
        .to_lowercase();
    let tool_choice = body
        .get("tool_choice")
        .and_then(|v| v.as_str())
        .unwrap_or("");

    let selected: Vec<&str> = cfg
        .blocks
        .iter()
        .filter(|b| {
            if let Some(tc) = &b.match_tool_choice {
                return tc == tool_choice;
            }
            if !b.match_tools.is_empty() {
                return b
                    .match_tools
                    .iter()
                    .any(|kw| haystack.contains(&kw.to_lowercase()));
            }
            true // no matcher = always
        })
        .filter_map(|b| b.text.as_deref())
        .map(str::trim)
        .collect();

    if selected.is_empty() {
        return false;
    }
    let rules = selected.join("\n\n");

    let Some(messages) = body.get_mut("messages").and_then(|v| v.as_array_mut()) else {
        return false;
    };

    // Merge into an existing leading system message when its content is a
    // plain string; otherwise insert a fresh system message up front.
    let merged = messages
        .first_mut()
        .filter(|m| m.get("role").and_then(|r| r.as_str()) == Some("system"))
        .and_then(|m| {
            let content = m.get("content")?.as_str()?.to_string();
            m["content"] = Value::String(format!("{content}\n\n{rules}"));
            Some(())
        })
        .is_some();
    if !merged {
        messages.insert(0, serde_json::json!({"role": "system", "content": rules}));
    }
    tracing::debug!(
        "[#{req_id}] injected {} rule block(s), {}B",
        selected.len(),
        rules.len()
    );
    true
}

fn existing_system_len(body: &Value) -> usize {
    body.get("messages")
        .and_then(|m| m.get(0))
        .filter(|m| m.get("role").and_then(|r| r.as_str()) == Some("system"))
        .map(|m| match m.get("content") {
            Some(Value::String(s)) => s.len(),
            Some(Value::Array(items)) => items
                .iter()
                .filter_map(|i| i.get("text")?.as_str())
                .map(str::len)
                .sum(),
            _ => 0,
        })
        .unwrap_or(0)
}

// ─── Tests ──────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{InjectBlock, InjectConfig};
    use serde_json::json;

    fn block(
        name: &str,
        text: &str,
        match_tools: &[&str],
        tool_choice: Option<&str>,
    ) -> InjectBlock {
        InjectBlock {
            name: name.into(),
            text: Some(text.into()),
            file: None,
            match_tools: match_tools.iter().map(|s| s.to_string()).collect(),
            match_tool_choice: tool_choice.map(String::from),
        }
    }

    fn cfg(blocks: Vec<InjectBlock>, skip_over: Option<usize>) -> InjectConfig {
        InjectConfig {
            skip_if_system_over: skip_over,
            blocks,
        }
    }

    fn tools_body(system: Option<&str>) -> Value {
        let mut messages = vec![];
        if let Some(s) = system {
            messages.push(json!({"role": "system", "content": s}));
        }
        messages.push(json!({"role": "user", "content": "hi"}));
        json!({
            "model": "m", "messages": messages,
            "tools": [{"type": "function", "function": {"name": "send_email", "description": "Send an email to someone"}}]
        })
    }

    #[test]
    fn test_always_block_injected_as_new_system_message() {
        let mut body = tools_body(None);
        assert!(apply(
            &mut body,
            &cfg(vec![block("a", "RULE-A", &[], None)], None),
            0
        ));
        assert_eq!(body["messages"][0]["role"], "system");
        assert_eq!(body["messages"][0]["content"], "RULE-A");
        assert_eq!(body["messages"][1]["role"], "user");
    }

    #[test]
    fn test_merges_into_existing_system_message() {
        let mut body = tools_body(Some("You are helpful."));
        assert!(apply(
            &mut body,
            &cfg(vec![block("a", "RULE-A", &[], None)], None),
            0
        ));
        assert_eq!(body["messages"][0]["content"], "You are helpful.\n\nRULE-A");
        assert_eq!(body["messages"].as_array().unwrap().len(), 2);
    }

    #[test]
    fn test_no_tools_no_injection() {
        let mut body = json!({"model": "m", "messages": [{"role": "user", "content": "hi"}]});
        assert!(!apply(
            &mut body,
            &cfg(vec![block("a", "RULE-A", &[], None)], None),
            0
        ));
    }

    #[test]
    fn test_match_tools_selects_relevant_blocks() {
        let mut body = tools_body(None);
        let c = cfg(
            vec![
                block("email", "EMAIL-RULES", &["email", "calendar"], None),
                block("search", "SEARCH-RULES", &["search", "web"], None),
            ],
            None,
        );
        assert!(apply(&mut body, &c, 0));
        let sys = body["messages"][0]["content"].as_str().unwrap();
        assert!(sys.contains("EMAIL-RULES"));
        assert!(
            !sys.contains("SEARCH-RULES"),
            "unmatched block must not inject: {sys}"
        );
    }

    #[test]
    fn test_match_tool_choice_required() {
        let c = cfg(vec![block("req", "MUST-CALL", &[], Some("required"))], None);
        let mut body = tools_body(None);
        assert!(!apply(&mut body, &c, 0), "no tool_choice → no injection");
        let mut body = tools_body(None);
        body["tool_choice"] = json!("required");
        assert!(apply(&mut body, &c, 0));
        assert!(body["messages"][0]["content"]
            .as_str()
            .unwrap()
            .contains("MUST-CALL"));
    }

    #[test]
    fn test_skip_when_system_message_large() {
        let big = "x".repeat(5000);
        let mut body = tools_body(Some(&big));
        let c = cfg(vec![block("a", "RULE-A", &[], None)], Some(4000));
        assert!(!apply(&mut body, &c, 0));
        assert_eq!(body["messages"][0]["content"].as_str().unwrap().len(), 5000);
    }

    #[test]
    fn test_multiple_blocks_joined_in_order() {
        let mut body = tools_body(None);
        let c = cfg(
            vec![
                block("a", "FIRST", &[], None),
                block("b", "SECOND", &[], None),
            ],
            None,
        );
        assert!(apply(&mut body, &c, 0));
        assert_eq!(body["messages"][0]["content"], "FIRST\n\nSECOND");
    }
}
