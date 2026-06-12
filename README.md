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

## Smart system-prompt injection

Optional per-model injection of instruction blocks into the system message of
tools-bearing `POST …/chat/completions` requests — the agent-template trick,
but visible in config, per-alias, selective, and hot-swappable with a proxy
restart instead of a model reload:

```json
"inject": {
  "skip_if_system_over": 6000,
  "blocks": [
    {"name": "security", "file": "rules/security.txt"},
    {"name": "channels", "file": "rules/side-effect-channels.txt",
     "match_tools": ["email", "calendar", "remind"]},
    {"name": "tool-required", "file": "rules/tool-required.txt",
     "match_tool_choice": "required"}
  ]
}
```

- Blocks with no matcher always inject (when tools are present); `match_tools`
  keywords match case-insensitively against tool names/descriptions;
  `match_tool_choice` matches the request's `tool_choice` string.
- `skip_if_system_over` skips injection entirely when the client already
  sends a system message larger than N bytes (agent harnesses like Claude
  Code ship their own tool discipline).
- `file` paths resolve relative to the config directory and load at startup;
  `text` works inline. Reference blocks live in [`rules/`](rules/).
- Injected blocks merge into an existing leading system message or are
  inserted as a new one. Requests without tools are untouched.

When fortify is also enabled, `tool_choice: "required"` additionally makes a
plain-text response retryable (the model is nudged to produce a tool call).

## Tool-call fortification (forge-style)

Optional per-model reliability layer for tool calling, ported from
[forge](https://github.com/antoinezambelli/forge): tool calls in responses
are **validated** against the request's `tools` array, malformed calls are
**rescue-parsed** out of text (JSON code fences, `[TOOL_CALLS]`, Qwen-coder
XML, `name[ARGS]{…}`), and invalid calls trigger **retries** with corrective
messages appended to the conversation.

```json
{
  "name": "qwen3.6-agent",
  "target": "http://host.docker.internal:8000/v1",
  "served_model": "Qwen3.6-35B-A3B",
  "fortify": {
    "enabled": true,
    "mode": "hold",
    "max_retries": 3,
    "rescue": true,
    "inject_respond_tool": false
  }
}
```

| Field | Default | Description |
|---|---|---|
| `enabled` | `true` | Master switch (the whole block is optional — omit `fortify` for pure passthrough) |
| `mode` | `"hold"` | `"hold"` streams prose live and buffers only tool calls; `"buffer"` forces non-streaming upstream and synthesizes SSE — retry always possible |
| `max_retries` | `3` | Upstream re-asks after an invalid tool call before degrading to the last response as-is |
| `rescue` | `true` | Attempt to parse tool calls out of malformed text responses |
| `inject_respond_tool` | `false` | Append a synthetic `respond` tool so the model always tool-calls; respond calls are converted back to plain text |

Fortification only engages for `POST …/chat/completions` requests carrying a
non-empty `tools` array (and `n` ≤ 1). Everything else — including all
requests to models without a `fortify` block — stays on the zero-parse
forwarding path.

### How streaming is preserved (`mode: "hold"`)

The upstream stays in streaming mode. Prose is forwarded live, a few deltas
behind the live edge; the moment the response turns into a tool call (native
`tool_calls` deltas or a malformed-call text marker), forwarding stops and
the rest is buffered, validated, and fixed at end of stream. `<think>` blocks
stream live. Retrying (re-asking the model) is physically possible only while
nothing has been forwarded yet:

| Response shape | Client experience | Retry? | Rescue? |
|---|---|---|---|
| Native tool call from the first delta | held → validated tool call | ✅ | ✅ |
| Malformed tool call at start of content | held → rescued tool call | ✅ | ✅ |
| Pure prose | streams live | n/a | n/a |
| Prose, then tool call | prose live → pause → fixed tool call | ❌ | ✅ |
| Tool call after streamed think/reasoning | reasoning live → then as above | ❌ | ✅ |

`mode: "buffer"` trades live prose for retry coverage on every row. In both
modes a response that can't be fixed is forwarded as-is — fortification never
turns a deliverable response into an error. Retries are sent non-streaming;
streaming clients then receive a synthesized SSE stream of the final
validated response, so client code works unmodified either way.

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
  |  Tool-call fortification (opt-in)
  v
upstream (vLLM, etc.) :8000
```

- **Zero JSON parses on the response path** — SSE and non-streaming responses are forwarded as raw bytes with only the model name rewritten. (Exception: fortified tool-calling requests, which parse responses by design.)
- **CORS enabled** — `Origin: *`, `GET` + `POST` allowed.
- **Static config** — the config file is read once at startup. Restart the container to pick up changes.

## License

[MIT](LICENSE)
