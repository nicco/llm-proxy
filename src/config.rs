//! Runtime configuration loaded from ~/.llm-proxy/config.json
//!
//! Each model entry declares where its traffic goes and what params to attach.
//! The proxy has no opinions about what those params mean — it just passes them.

use serde::Deserialize;
use std::collections::HashMap;
use std::env;
use std::path::PathBuf;

#[derive(Debug, Clone, Deserialize)]
pub struct AppConfig {
    #[serde(default)]
    pub models: Vec<ModelConfig>,
    /// Fast O(1) lookup table built at startup from [`Self::models`].
    #[serde(skip)]
    pub model_index: HashMap<String, ModelConfig>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ModelConfig {
    pub name: String,
    pub target: String,
    pub served_model: String,
    pub api_key: Option<String>,
    /// Arbitrary key-value pairs injected into the request body as-is.
    /// The proxy does not interpret these fields.
    #[serde(default)]
    pub params: HashMap<String, serde_json::Value>,
    /// Tool-call fortification (forge-style validate/rescue/retry).
    /// `None` means pure passthrough — the proxy never parses responses.
    #[serde(default)]
    pub fortify: Option<FortifyConfig>,
    /// Smart system-prompt injection for tools-bearing chat requests.
    #[serde(default)]
    pub inject: Option<InjectConfig>,
}

/// Conditionally inject instruction blocks into the system message of
/// POST …/chat/completions requests that carry tools.
#[derive(Debug, Clone, Deserialize)]
pub struct InjectConfig {
    /// Skip all injection when the request already carries a system message
    /// larger than this many bytes — agent harnesses like Claude Code ship
    /// their own tool discipline and need no help.
    pub skip_if_system_over: Option<usize>,
    pub blocks: Vec<InjectBlock>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct InjectBlock {
    pub name: String,
    /// Inline rule text…
    pub text: Option<String>,
    /// …or a file path (relative paths resolve against the config file's
    /// directory). Loaded once at startup into `text`.
    pub file: Option<String>,
    /// Inject only when any keyword appears (case-insensitive) in a tool
    /// name or description. Empty = always (when tools are present).
    #[serde(default)]
    pub match_tools: Vec<String>,
    /// Inject only when the request's `tool_choice` equals this string
    /// (e.g. "required").
    pub match_tool_choice: Option<String>,
}

/// How fortified requests interact with streaming.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum FortifyMode {
    /// Stream upstream; forward prose live, hold from the first sign of a
    /// tool call, fix the held tail at end of stream.  Retry only happens
    /// when nothing has been forwarded yet.
    Hold,
    /// Force `stream:false` upstream, validate the complete response, and
    /// synthesize SSE for streaming clients.  Retry always possible.
    Buffer,
}

#[derive(Debug, Clone, Deserialize)]
pub struct FortifyConfig {
    #[serde(default = "default_true")]
    pub enabled: bool,
    #[serde(default = "default_mode")]
    pub mode: FortifyMode,
    #[serde(default = "default_max_retries")]
    pub max_retries: u32,
    #[serde(default = "default_true")]
    pub rescue: bool,
    /// Check tool-call arguments against each tool's JSON Schema
    /// (`required`, `additionalProperties: false`, top-level types) and
    /// retry with a precise nudge on violations.  Conservative: schema
    /// constructs the checker doesn't understand always pass.
    #[serde(default = "default_true")]
    pub validate_args: bool,
    #[serde(default)]
    pub inject_respond_tool: bool,
    /// After a plain-text response in a conversation that already contains
    /// tool results, make one extra round-trip asking the model to verify it
    /// completed every requested action (and to continue if not).  Such
    /// requests are routed through the buffered path so the check can run
    /// before any bytes reach the client.
    #[serde(default)]
    pub completion_check: bool,
}

fn default_true() -> bool {
    true
}

fn default_mode() -> FortifyMode {
    FortifyMode::Hold
}

fn default_max_retries() -> u32 {
    3
}

impl AppConfig {
    /// Resolve the default config path (~/.llm-proxy/config.json).
    ///
    /// Override with the `LLM_PROXY_CONFIG` environment variable.
    pub fn default_path() -> PathBuf {
        if let Ok(p) = env::var("LLM_PROXY_CONFIG") {
            return PathBuf::from(p);
        }
        let home = env::var("HOME").unwrap_or_else(|_| {
            // If HOME isn't set and neither is LLM_PROXY_CONFIG, fail with a clear message
            eprintln!("error: HOME not set and LLM_PROXY_CONFIG not set");
            eprintln!("  Set LLM_PROXY_CONFIG=/path/to/config.json or export HOME=<your-home>");
            std::process::exit(1);
        });
        PathBuf::from(home).join(".llm-proxy").join("config.json")
    }

    /// Load config from the given path.
    /// Builds an internal HashMap for O(1) model lookups.
    pub fn from_file(path: &str) -> Result<Self, anyhow::Error> {
        let content = std::fs::read_to_string(path)?;
        let mut config: Self = serde_json::from_str(&content)?;
        let base = std::path::Path::new(path).parent().map(|p| p.to_path_buf());
        config.resolve_inject_files(base.as_deref())?;
        config.rebuild_index();
        Ok(config)
    }

    /// Load `file`-based inject blocks into their `text` field, failing fast
    /// at startup on missing files rather than per-request.
    fn resolve_inject_files(
        &mut self,
        base: Option<&std::path::Path>,
    ) -> Result<(), anyhow::Error> {
        for m in &mut self.models {
            let Some(inject) = &mut m.inject else {
                continue;
            };
            for b in &mut inject.blocks {
                if let Some(file) = &b.file {
                    let p = std::path::Path::new(file);
                    let full = match base {
                        Some(dir) if p.is_relative() => dir.join(p),
                        _ => p.to_path_buf(),
                    };
                    b.text = Some(std::fs::read_to_string(&full).map_err(|e| {
                        anyhow::anyhow!("inject block '{}' ({}): {e}", b.name, full.display())
                    })?);
                }
                if b.text.as_deref().map(str::trim).unwrap_or("").is_empty() {
                    anyhow::bail!(
                        "inject block '{}' has neither text nor file content",
                        b.name
                    );
                }
            }
        }
        Ok(())
    }

    /// Rebuild the O(1) lookup index from the models vector.
    pub fn rebuild_index(&mut self) {
        self.model_index.clear();
        for m in &self.models {
            self.model_index.insert(m.name.clone(), m.clone());
        }
    }

    /// Find a model config by its name — O(1) via internal HashMap.
    pub fn find(&self, name: &str) -> Option<&ModelConfig> {
        self.model_index.get(name)
    }
}

// ─── Tests ──────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Serializes the tests that mutate process-global env vars — they race
    /// when the test harness runs them on parallel threads.
    static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    #[test]
    fn test_default_path_returns_homedir_config() {
        let _guard = ENV_LOCK.lock().unwrap();
        // Clear any env var that tests may have set
        env::remove_var("LLM_PROXY_CONFIG");
        let orig_home = env::var("HOME").ok();
        env::set_var("HOME", "/tmp/test-home");
        let path = AppConfig::default_path();
        assert_eq!(path, PathBuf::from("/tmp/test-home/.llm-proxy/config.json"));
        if let Some(h) = orig_home {
            env::set_var("HOME", h);
        } else {
            env::remove_var("HOME");
        }
    }

    #[test]
    fn test_default_path_respects_env_var() {
        let _guard = ENV_LOCK.lock().unwrap();
        // Clear HOME to avoid interference, and clean up the env var
        env::remove_var("HOME");
        env::remove_var("LLM_PROXY_CONFIG");
        env::set_var("LLM_PROXY_CONFIG", "/custom/path/config.json");
        let path = AppConfig::default_path();
        assert_eq!(path, PathBuf::from("/custom/path/config.json"));
        env::remove_var("LLM_PROXY_CONFIG");
        env::set_var("HOME", "/tmp/test-home"); // restore for other tests
    }

    #[test]
    fn test_find_returns_config_by_name() {
        let mut config = AppConfig {
            models: vec![
                ModelConfig {
                    name: "fast".into(),
                    target: "http://localhost:8000".into(),
                    served_model: "gpt-4".into(),
                    api_key: None,
                    params: HashMap::new(),
                    fortify: None,
                    inject: None,
                },
                ModelConfig {
                    name: "thinking".into(),
                    target: "http://localhost:8000".into(),
                    served_model: "claude".into(),
                    api_key: None,
                    params: HashMap::new(),
                    fortify: None,
                    inject: None,
                },
            ],
            model_index: HashMap::new(),
        };
        config.rebuild_index();
        assert_eq!(config.find("fast").unwrap().name, "fast");
        assert_eq!(config.find("thinking").unwrap().served_model, "claude");
        assert!(config.find("unknown").is_none());
    }

    #[test]
    fn test_find_last_dup_wins() {
        let mut config = AppConfig {
            models: vec![
                ModelConfig {
                    name: "dup".into(),
                    target: "http://localhost:8000".into(),
                    served_model: "first".into(),
                    api_key: None,
                    params: HashMap::new(),
                    fortify: None,
                    inject: None,
                },
                ModelConfig {
                    name: "dup".into(),
                    target: "http://localhost:8001".into(),
                    served_model: "second".into(),
                    api_key: None,
                    params: HashMap::new(),
                    fortify: None,
                    inject: None,
                },
            ],
            model_index: HashMap::new(),
        };
        config.rebuild_index();
        // HashMap semantics: last-writer wins for duplicate keys
        assert_eq!(config.find("dup").unwrap().served_model, "second");
    }

    #[test]
    fn test_app_config_empty_models() {
        let json = r#"{}"#;
        let mut config: AppConfig = serde_json::from_str(json).unwrap();
        config.rebuild_index();
        assert!(config.models.is_empty());
        assert!(config.model_index.is_empty());
    }

    #[test]
    fn test_model_config_defaults() {
        let json = r#"{
            "name": "test",
            "target": "http://localhost:8000",
            "served_model": "model-x"
        }"#;
        let model: ModelConfig = serde_json::from_str(json).unwrap();
        assert_eq!(model.name, "test");
        assert_eq!(model.api_key, None);
        assert!(model.params.is_empty());
    }

    #[test]
    fn test_model_config_with_api_key() {
        let json = r#"{
            "name": "test",
            "target": "http://localhost:8000",
            "served_model": "model-x",
            "api_key": "sk-test-key"
        }"#;
        let model: ModelConfig = serde_json::from_str(json).unwrap();
        assert_eq!(model.api_key, Some("sk-test-key".to_string()));
    }

    #[test]
    fn test_model_config_fortify_defaults() {
        let json = r#"{
            "name": "test",
            "target": "http://localhost:8000",
            "served_model": "model-x",
            "fortify": {}
        }"#;
        let model: ModelConfig = serde_json::from_str(json).unwrap();
        let f = model.fortify.unwrap();
        assert!(f.enabled);
        assert_eq!(f.mode, FortifyMode::Hold);
        assert_eq!(f.max_retries, 3);
        assert!(f.rescue);
        assert!(!f.inject_respond_tool);
    }

    #[test]
    fn test_model_config_fortify_buffer_mode() {
        let json = r#"{
            "name": "test",
            "target": "http://localhost:8000",
            "served_model": "model-x",
            "fortify": {"mode": "buffer", "max_retries": 1, "inject_respond_tool": true}
        }"#;
        let model: ModelConfig = serde_json::from_str(json).unwrap();
        let f = model.fortify.unwrap();
        assert_eq!(f.mode, FortifyMode::Buffer);
        assert_eq!(f.max_retries, 1);
        assert!(f.inject_respond_tool);
    }

    #[test]
    fn test_model_config_no_fortify_is_none() {
        let json = r#"{
            "name": "test",
            "target": "http://localhost:8000",
            "served_model": "model-x"
        }"#;
        let model: ModelConfig = serde_json::from_str(json).unwrap();
        assert!(model.fortify.is_none());
    }

    #[test]
    fn test_model_config_with_params() {
        let json = r#"{
            "name": "test",
            "target": "http://localhost:8000",
            "served_model": "model-x",
            "params": {"temperature": 0.7, "nested": {"a": 1}}
        }"#;
        let model: ModelConfig = serde_json::from_str(json).unwrap();
        assert_eq!(model.params.len(), 2);
        assert_eq!(model.params["temperature"], 0.7);
        assert_eq!(model.params["nested"]["a"], 1);
    }
}
