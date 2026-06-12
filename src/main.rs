mod config;
mod fortify;
mod forward;

use axum::{
    extract::State,
    http::{Request, StatusCode, Uri},
    response::Response,
    routing::{any, get},
    Router,
};
use http_body_util::BodyExt;
use hyper_util::client::legacy::{connect::HttpConnector, Client};
use std::sync::Arc;
use tower_http::cors::{Any, CorsLayer};

struct AppState {
    config: config::AppConfig,
    client: Client<HttpConnector, axum::body::Body>,
    max_body: usize,
    /// Max time to wait for upstream response *headers* (body streaming is
    /// never time-limited).  `None` disables the limit.
    header_timeout: Option<std::time::Duration>,
}

/// Catch-all: extract model from URL prefix or request body JSON.
async fn handle(
    State(state): State<Arc<AppState>>,
    uri: Uri,
    req: Request<axum::body::Body>,
) -> Result<Response<axum::body::Body>, (StatusCode, String)> {
    let path = uri.path();

    // Try to extract model name from first path segment
    let trimmed = path.trim_start_matches('/');
    let Some(seg) = trimmed.split('/').next().filter(|s| !s.is_empty()) else {
        return Err((StatusCode::BAD_REQUEST, "missing model name".to_string()));
    };
    let Some(model_cfg) = state.config.find(seg) else {
        return infer_model_from_body(&state, path, req).await;
    };

    let prefix_len = path.len() - trimmed.len(); // leading slash(es)
    let remaining = &path[prefix_len + seg.len()..];
    let mut upstream_url = format!("{}{}", model_cfg.target.trim_end_matches('/'), remaining);
    if let Some(q) = uri.query() {
        upstream_url.push('?');
        upstream_url.push_str(q);
    }

    forward::proxy(
        &upstream_url,
        model_cfg,
        req,
        &state.client,
        state.max_body,
        state.header_timeout,
    )
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
) -> Result<Response<axum::body::Body>, (StatusCode, String)> {
    let (parts, body) = req.into_parts();
    let body_bytes = axum::body::to_bytes(body, state.max_body)
        .await
        .map_err(|e| (StatusCode::BAD_REQUEST, e.to_string()))?;

    let model_name = serde_json::from_slice::<serde_json::Value>(&body_bytes)
        .ok()
        .and_then(|v| v.get("model")?.as_str().map(|s| s.to_string()))
        .ok_or_else(|| {
            (
                StatusCode::BAD_REQUEST,
                "model not found in body".to_string(),
            )
        })?;

    let model_cfg = state.config.find(&model_name).ok_or_else(|| {
        (
            StatusCode::NOT_FOUND,
            format!("unknown model: {model_name}"),
        )
    })?;

    let mut upstream_url = format!("{}{}", strip_url_path(&model_cfg.target), path);
    if let Some(q) = parts.uri.query() {
        upstream_url.push('?');
        upstream_url.push_str(q);
    }

    let req = Request::from_parts(parts, axum::body::Body::from(body_bytes));

    forward::proxy(
        &upstream_url,
        model_cfg,
        req,
        &state.client,
        state.max_body,
        state.header_timeout,
    )
    .await
    .map_err(|e| (StatusCode::BAD_GATEWAY, e.to_string()))
}

/// Fetch vLLM model list and return only proxy-configured models that match.
async fn list_models(
    State(state): State<Arc<AppState>>,
) -> Result<axum::Json<serde_json::Value>, (StatusCode, String)> {
    let target = state
        .config
        .models
        .first()
        .map(|m| &m.target)
        .cloned()
        .unwrap_or_default();
    let upstream_url = format!("{}/models", target.trim_end_matches('/'));

    let req = Request::builder()
        .uri(&upstream_url)
        .body(axum::body::Body::empty())
        .map_err(|e| (StatusCode::BAD_GATEWAY, e.to_string()))?;

    let resp = match state.client.request(req).await {
        Ok(r) => r,
        Err(e) => {
            tracing::debug!("upstream /v1/models failed: {e}");
            return Err((
                StatusCode::BAD_GATEWAY,
                format!("upstream models fetch failed: {e}"),
            ));
        }
    };

    let collected = resp
        .into_body()
        .collect()
        .await
        .map_err(|e| (StatusCode::BAD_GATEWAY, e.to_string()))?;
    let body = collected.to_bytes();

    let upstream_models: Vec<String> = serde_json::from_slice::<serde_json::Value>(&body)
        .ok()
        .and_then(|v| {
            v.get("data")?.as_array().map(|arr| {
                arr.iter()
                    .filter_map(|m| m.get("id")?.as_str().map(|s| s.to_string()))
                    .collect::<Vec<_>>()
            })
        })
        .unwrap_or_default();

    let upstream_set: std::collections::HashSet<&str> =
        upstream_models.iter().map(|s| s.as_str()).collect();

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

    Ok(axum::Json(
        serde_json::json!({"object": "list", "data": models}),
    ))
}

async fn health() -> &'static str {
    "ok"
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("warn")),
        )
        .init();

    let config =
        config::AppConfig::from_file(&config::AppConfig::default_path().to_string_lossy())?;

    if config.models.is_empty() {
        tracing::warn!("config loaded with zero models — all requests will 404");
    } else {
        tracing::info!("loaded {} model(s)", config.models.len());
    }

    let max_body: usize = std::env::var("LLM_PROXY_MAX_BODY")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(4 * 1024 * 1024); // 4 MiB default

    // How long to wait for upstream response headers before giving up.
    // Generous default: for non-streaming requests, time-to-headers includes
    // upstream queueing + the full generation.  0 disables the limit.
    let header_timeout = std::env::var("LLM_PROXY_HEADER_TIMEOUT_SECS")
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(300);
    let header_timeout =
        (header_timeout > 0).then(|| std::time::Duration::from_secs(header_timeout));

    // Shared HTTP client with keep-alive connection pooling.  The pool limit
    // only caps *idle* sockets per host — concurrent requests beyond it open
    // fresh connections, so parallel bursts never queue behind the pool.
    let mut connector = HttpConnector::new();
    connector.set_nodelay(true); // don't let Nagle delay small SSE chunks
    connector.set_connect_timeout(Some(std::time::Duration::from_secs(10)));

    let client: Client<HttpConnector, axum::body::Body> =
        Client::builder(hyper_util::rt::TokioExecutor::new())
            .pool_idle_timeout(std::time::Duration::from_secs(90))
            .pool_max_idle_per_host(32)
            .build(connector);

    let state = Arc::new(AppState {
        config,
        client,
        max_body,
        header_timeout,
    });

    let bind: std::net::SocketAddr = std::env::var("LLM_PROXY_BIND")
        .unwrap_or_else(|_| "0.0.0.0:7878".into())
        .parse()
        .expect("invalid LLM_PROXY_BIND address");

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

    let listener = tokio::net::TcpListener::bind(bind).await?;
    tracing::info!("listening on {bind}");
    axum::serve(listener, app)
        .with_graceful_shutdown(async {
            tokio::signal::ctrl_c().await.ok();
            tracing::info!("shutting down");
        })
        .await?;

    Ok(())
}
