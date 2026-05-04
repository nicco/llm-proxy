# llm-proxy

A lightweight, high-performance proxy that adds **per-model parameter injection** and **model name rewriting** on top of any OpenAI-compatible LLM API (vLLM, Ollama, llama.cpp, etc.).

Use it to serve multiple "models" — each with its own temperature, chat templates, system prompts, presence penalty, etc. — from a single upstream server.

## How it works

The proxy listens on port 7878 and supports two ways to route requests to a model:

### URL prefix (preferred)

```
GET http://localhost:7878/qwen3.6-thinking/v1/chat/completions
```

The first path segment (`qwen3.6-thinking`) is looked up in the config and the request is forwarded to that model's `target` URL. The rest of the path is appended as-is.

### Model name in body

```
curl http://localhost:7878/v1/chat/completions \
  -d '{"model":"qwen3.6-fast","messages":[...]}'
```

When the URL doesn't start with a known model name, the proxy reads the `model` field from the JSON body and routes accordingly.

### `GET /v1/models`

Fetches the full model list from the upstream server, then filters it to return only models whose upstream `id` matches a configured `served_model` field. The returned `id` is rewritten to the proxy-facing `name`.

## Configuration

Config lives at `~/.llm-proxy/config.json` (or `$HOME/.llm-proxy/config.json` inside the container):

```json
{
  "models": [
    {
      "name": "qwen3.6-thinking",
      "target": "http://host.docker.internal:8000/v1",
      "served_model": "Qwen3.6-35B-A3B",
      "params": {
        "temperature": 1.0,
        "chat_template_kwargs": {
          "enable_thinking": true,
          "preserve_thinking": true
        }
      }
    }
  ]
}
```

| Field | Description |
|---|---|
| `name` | Your proxy-facing model name — used in URLs and body `model` field |
| `target` | Upstream API base URL (scheme + host + port) |
| `served_model` | The real model name the upstream server actually serves (e.g. `"Qwen3.6-35B-A3B"`) |
| `api_key` | Optional — forwarded as `Authorization: Bearer <key>` header |
| `params` | Arbitrary key-value pairs merged into every request body. The proxy doesn't interpret them — it just passes them through |

### Request body transformation

For each request the proxy:

1. **Overwrites** the `model` field with `served_model` (so the upstream knows which real model to use)
2. **Merges** all `params` into the request body (overriding any client-provided values with the same key)

### SSE streaming

Streaming responses are forwarded in real-time with model name rewriting applied to every `data:` line. No intermediate buffering.

## Running

### Docker Compose (recommended)

```yaml
services:
  llm-proxy:
    build: .
    container_name: llm-proxy
    ports:
      - "27878:7878"
    volumes:
      - ~/.llm-proxy:/root/.llm-proxy
    extra_hosts:
      - "host.docker.internal:host-gateway"
    restart: unless-stopped
```

Then:

```bash
docker compose up -d
```

The proxy listens internally on port 7878 and is exposed on port 27878.

### Docker (manual)

```bash
docker build -t llm-proxy .
docker run -d \
  --name llm-proxy \
  -p 27878:7878 \
  -v ~/.llm-proxy:/root/.llm-proxy \
  --add-host host.docker.internal:host-gateway \
  llm-proxy
```

### Local development

```bash
cargo run
# listens on http://0.0.0.0:7878
```

## Endpoints

| Route | Method | Description |
|---|---|---|
| `/{model_name}/{*rest}` | ANY | Proxy all OpenAI API endpoints |
| `/v1/models` | GET | Returns proxy-configured model list |
| `/health` | GET | Returns `ok` |

## Example curl

```bash
# List models
curl http://localhost:7878/v1/models

# Chat completion via URL prefix
curl http://localhost:7878/qwen3.6-fast/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"say hi"}],"max_tokens":10}'

# Chat completion via body model field
curl http://localhost:7878/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3.6-fast","messages":[...],"max_tokens":10}'
```

> If running via Docker, replace `7878` with the host port (default: `27878`).

## Architecture

```
Client
  |
  v
llm-proxy :7878        ← Rust / Axum
  |  URL prefix matching
  |  Body model extraction
  |  Params injection
  |  SSE streaming
  |  Model name rewriting
  v
upstream (vLLM, etc.) :8000
```

- **Zero JSON parses on the response path** — SSE and non-streaming responses are forwarded as raw bytes with only the model name rewritten.
- **CORS enabled** — `Origin: *`, `GET` + `POST` allowed.
- **Static config** — the config file is read once at startup. Restart the container to pick up changes.

## License

[MIT](LICENSE)
