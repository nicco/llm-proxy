mod config;
mod forward;

use axum::{
    extract::State,
    http::{Request, StatusCode, Uri},
    response::Response,
    routing::{any, get},
    Router,
};
use http_body_util::BodyExt;
use hyper_util::client::legacy::Client;
use std::sync::Arc;
use tower_http::cors::{Any, CorsLayer};

struct AppState {
    config: config::AppConfig,
}

/// Catch-all: extract model from URL prefix or request body JSON.
async fn handle(
    State(state): State<Arc<AppState>>,
    uri: Uri,
    req: Request<axum::body::Body>,
) -> Result<Response<axum::body::Body>, (StatusCode, String)> {
    let path = uri.path();

    // Try to extract model name from first path segment
    let first_seg = path
        .trim_start_matches('/')
        .split('/')
        .next()
        .filter(|s| !s.is_empty());

    // Check if first segment is a known model
    let model_name = if let Some(seg) = first_seg {
        if state.config.find(seg).is_some() {
            seg.to_string()
        } else {
            // First segment isn't a model — try from body
            return infer_model_from_body(&state, path, req, &uri).await;
        }
    } else {
        return Err((StatusCode::BAD_REQUEST, "missing model name".to_string()));
    };

    let model_cfg = state
        .config
        .find(&model_name)
        .ok_or_else(|| (StatusCode::NOT_FOUND, format!("unknown model: {model_name}")))?;

    let remaining = &path[model_name.len() + 1..];
    let upstream_url = format!("{}{}", model_cfg.target.trim_end_matches('/'), remaining);

    let client = Client::builder(hyper_util::rt::TokioExecutor::new()).build_http();
    let method = req.method().clone();
    forward::proxy(&upstream_url, model_cfg, req, &client, method)
        .await
        .map_err(|e| (StatusCode::BAD_GATEWAY, e.to_string()))
}

/// Strip trailing path from a URL, keeping only scheme + host + port.
fn strip_url_path(url: &str) -> &str {
    if let Some(pos) = url.find("://") {
        let after_scheme = &url[pos + 3..];
        if let Some(slash) = after_scheme.find('/') {
            &url[..pos + 3 + slash]
        } else {
            url
        }
    } else {
        url
    }
}

/// Extract model from request body when URL doesn't have model prefix.
async fn infer_model_from_body(
    state: &AppState,
    path: &str,
    req: Request<axum::body::Body>,
    _uri: &Uri,
) -> Result<Response<axum::body::Body>, (StatusCode, String)> {
    let (parts, body) = req.into_parts();
    let body_bytes = axum::body::to_bytes(body, 4 * 1024 * 1024)
        .await
        .map_err(|e| (StatusCode::BAD_REQUEST, e.to_string()))?;

    // Extract model from JSON body
    let model_name = serde_json::from_slice::<serde_json::Value>(&body_bytes)
        .ok()
        .and_then(|v| v.get("model")?.as_str().map(|s| s.to_string()))
        .ok_or_else(|| (StatusCode::BAD_REQUEST, "model not found in body".to_string()))?;

    let model_cfg = state
        .config
        .find(&model_name)
        .ok_or_else(|| (StatusCode::NOT_FOUND, format!("unknown model: {model_name}")))?;

    // Reconstruct request with original body
    let req = Request::from_parts(parts, axum::body::Body::from(body_bytes));

    let upstream_url = format!(
        "{}{}",
        strip_url_path(&model_cfg.target),
        path
    );

    let client = Client::builder(hyper_util::rt::TokioExecutor::new()).build_http();
    let method = req.method().clone();
    forward::proxy(&upstream_url, model_cfg, req, &client, method)
        .await
        .map_err(|e| (StatusCode::BAD_GATEWAY, e.to_string()))
}

/// Fetch vLLM model list and return only proxy-configured models that match.
async fn list_models(State(state): State<Arc<AppState>>) -> Result<axum::Json<serde_json::Value>, (StatusCode, String)> {
    // Fetch upstream model list from first configured target
    let target = state.config.models.first().map(|m| &m.target).cloned().unwrap_or_default();
    let upstream_url = format!("{}/models", target.trim_end_matches('/'));

    let client = Client::builder(hyper_util::rt::TokioExecutor::new()).build_http();
    let req = Request::builder()
        .uri(&upstream_url)
        .body(axum::body::Body::empty())
        .map_err(|e| (StatusCode::BAD_GATEWAY, e.to_string()))?;

    let resp = client
        .request(req)
        .await
        .map_err(|e| (StatusCode::BAD_GATEWAY, e.to_string()))?;

    let collected = resp.into_body().collect().await
        .map_err(|e| (StatusCode::BAD_GATEWAY, e.to_string()))?;
    let body = collected.to_bytes();

    let upstream_models: Vec<String> = serde_json::from_slice::<serde_json::Value>(&body)
        .ok()
        .and_then(|v| v.get("data")?.as_array().map(|arr| {
            arr.iter()
                .filter_map(|m| m.get("id")?.as_str().map(|s| s.to_string()))
                .collect::<Vec<_>>()
        }))
        .unwrap_or_default();

    let upstream_set: std::collections::HashSet<&str> = upstream_models.iter().map(|s| s.as_str()).collect();

    let models: Vec<serde_json::Value> = state
        .config
        .models
        .iter()
        .filter(|m| upstream_set.contains(m.served_model.as_str()))
        .map(|m| {
            serde_json::json!({
                "id": m.name,
                "object": "model",
                "created": 0,
                "owned_by": "llm-proxy"
            })
        })
        .collect();

    Ok(axum::Json(serde_json::json!({"object": "list", "data": models})))
}

async fn health() -> &'static str {
    "ok"
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    let config = config::AppConfig::from_file(
        &config::AppConfig::default_path().to_string_lossy(),
    )?;

    let state = Arc::new(AppState { config });

    let addr: std::net::SocketAddr = "0.0.0.0:7878".parse()?;

    let cors = CorsLayer::new()
        .allow_methods([axum::http::Method::GET, axum::http::Method::POST])
        .allow_headers([axum::http::header::CONTENT_TYPE])
        .allow_origin(Any);

    let app = Router::new()
        .route("/v1/models", get(list_models))
        .route("/{*rest}", any(handle))
        .route("/health", get(health))
        .layer(cors)
        .with_state(state);

    let listener = tokio::net::TcpListener::bind(addr).await?;
    tracing::info!("listening on {addr}");
    axum::serve(listener, app).await?;

    Ok(())
}
