//! Conservative JSON-Schema argument checking for tool calls.
//!
//! Validates a call's arguments against the tool's `function.parameters`
//! schema from the request — only the unambiguous parts: `required`
//! properties, `additionalProperties: false`, and top-level `type` mismatches.
//! Anything the checker doesn't fully understand passes, so quirky schemas
//! can never cause spurious retries; a real violation costs one corrective
//! retry instead of a client-side hard failure.

use serde_json::Value;

/// Returns a list of human-readable violations (empty = acceptable).
pub(crate) fn violations(args: &Value, schema: &Value) -> Vec<String> {
    let mut out = Vec::new();
    let Some(args_obj) = args.as_object() else {
        return out; // non-object args are handled by the basic validator
    };
    let Some(schema_obj) = schema.as_object() else {
        return out;
    };
    // Only reason about plain object schemas.
    match schema_obj.get("type") {
        Some(Value::String(t)) if t == "object" => {}
        None if schema_obj.contains_key("properties") => {}
        _ => return out,
    }
    let properties = schema_obj.get("properties").and_then(|v| v.as_object());

    if let Some(required) = schema_obj.get("required").and_then(|v| v.as_array()) {
        for r in required.iter().filter_map(|v| v.as_str()) {
            if !args_obj.contains_key(r) {
                out.push(format!("missing required property \"{r}\""));
            }
        }
    }

    if schema_obj.get("additionalProperties") == Some(&Value::Bool(false)) {
        if let Some(props) = properties {
            let mut allowed: Vec<&str> = props.keys().map(String::as_str).collect();
            allowed.sort_unstable();
            for k in args_obj.keys() {
                if !props.contains_key(k) {
                    out.push(format!(
                        "unsupported property \"{k}\" (allowed: {})",
                        allowed.join(", ")
                    ));
                }
            }
        }
    }

    if let Some(props) = properties {
        for (k, v) in args_obj {
            let Some(expected) = props.get(k).and_then(|p| p.get("type")) else {
                continue;
            };
            let ok = match expected {
                Value::String(t) => type_matches(v, t),
                Value::Array(ts) => ts
                    .iter()
                    .filter_map(|t| t.as_str())
                    .any(|t| type_matches(v, t)),
                _ => true,
            };
            if !ok {
                out.push(format!(
                    "property \"{k}\" should be of type {expected} but got {}",
                    json_type_name(v)
                ));
            }
        }
    }
    out
}

fn type_matches(v: &Value, t: &str) -> bool {
    match t {
        "string" => v.is_string(),
        "number" => v.is_number(),
        "integer" => v.as_i64().is_some() || v.as_u64().is_some(),
        "boolean" => v.is_boolean(),
        "array" => v.is_array(),
        "object" => v.is_object(),
        "null" => v.is_null(),
        _ => true, // unknown type keyword — don't judge
    }
}

fn json_type_name(v: &Value) -> &'static str {
    match v {
        Value::Null => "null",
        Value::Bool(_) => "boolean",
        Value::Number(_) => "number",
        Value::String(_) => "string",
        Value::Array(_) => "array",
        Value::Object(_) => "object",
    }
}

// ─── Tests ──────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn edit_schema() -> Value {
        json!({
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "edits": {"type": "array"}
            },
            "required": ["path", "edits"],
            "additionalProperties": false
        })
    }

    #[test]
    fn test_missing_required_and_extra_property() {
        // The exact production failure: {newText, path} against the edit schema.
        let args = json!({"newText": "const x = 1;", "path": "/a/main.js"});
        let v = violations(&args, &edit_schema());
        assert!(
            v.iter()
                .any(|s| s.contains("missing required property \"edits\"")),
            "{v:?}"
        );
        assert!(
            v.iter()
                .any(|s| s.contains("unsupported property \"newText\"")),
            "{v:?}"
        );
    }

    #[test]
    fn test_valid_args_pass() {
        let args = json!({"path": "/a/main.js", "edits": [{"old": "a", "new": "b"}]});
        assert!(violations(&args, &edit_schema()).is_empty());
    }

    #[test]
    fn test_type_mismatch() {
        let args = json!({"path": 42, "edits": []});
        let v = violations(&args, &edit_schema());
        assert_eq!(v.len(), 1);
        assert!(v[0].contains("\"path\""));
        assert!(v[0].contains("got number"));
    }

    #[test]
    fn test_extra_property_allowed_when_not_closed() {
        let schema = json!({"type": "object", "properties": {"a": {"type": "string"}}});
        let args = json!({"a": "x", "b": "y"});
        assert!(violations(&args, &schema).is_empty());
    }

    #[test]
    fn test_integer_vs_number() {
        let schema = json!({"type": "object", "properties": {"n": {"type": "integer"}}});
        assert!(violations(&json!({"n": 5}), &schema).is_empty());
        assert_eq!(violations(&json!({"n": 5.5}), &schema).len(), 1);
    }

    #[test]
    fn test_union_type() {
        let schema = json!({"type": "object", "properties": {"v": {"type": ["string", "number"]}}});
        assert!(violations(&json!({"v": "x"}), &schema).is_empty());
        assert!(violations(&json!({"v": 3}), &schema).is_empty());
        assert_eq!(violations(&json!({"v": true}), &schema).len(), 1);
    }

    #[test]
    fn test_unknown_schema_shapes_pass() {
        // Non-object schema, exotic constructs — never judged.
        assert!(violations(&json!({"a": 1}), &json!({"anyOf": []})).is_empty());
        assert!(violations(&json!({"a": 1}), &json!({"type": "array"})).is_empty());
        assert!(violations(&json!("not an object"), &edit_schema()).is_empty());
    }
}
