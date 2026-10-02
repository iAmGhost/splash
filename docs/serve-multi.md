# Multiple models

[Back to Splash](../README.md#multiple-models)

`splash serve-multi` serves several models from one port. It runs a thin
proxy that owns the port you pass; exactly one native engine runs at a
time. When a request names a model that is not loaded, the proxy waits for
requests already in flight on the current engine to complete, stops the
engine, waits for its memory to be reclaimed, loads the requested model,
and then serves the request — no client-side reconnection or retries
required, and no in-flight request is ever cut short.

## Usage

`models.json`:

```json
{
  "models": [
    {"model": "unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_M", "aliases": ["code-27b"]},
    {"model": "unsloth/Qwen3.6-35B-A3B-GGUF:UD-Q4_K_M",
     "aliases": ["code-35b-a3b"], "max_context": 131072}
  ]
}
```

```bash
splash serve-multi --config models.json
```

No engine runs until the first request arrives; that request's model (by
repo ID or alias) is what loads first. Any config model that is not
installed yet is downloaded and prepared before the proxy starts accepting
traffic.

### Config file

The top level is a JSON object with a single `models` list. Each entry:

| Key | Required | Purpose |
| --- | --- | --- |
| `model` | yes | Repo ID to serve, e.g. `OWNER/REPO` or `OWNER/REPO:VARIANT`. |
| `aliases` | no | Extra API model IDs, as in [`--served-model-name`](../DEVELOPMENT.md#api-model-aliases). |
| `max_context` | no | Context limit in tokens for this model. |

Shared CLI flags take precedence over per-model values, e.g.
`--max-context 100K` beats `"max_context": 131072`.

### How a switch works

POSTs to `/v1/chat/completions`, `/v1/completions`, `/v1/responses`, and
`/v1/messages` are inspected for their `model` field (a repo ID or an alias);
everything else is forwarded as-is:

- **In-flight requests are never cut** — once a switch is triggered, the
  supervisor waits for requests already being forwarded to the current
  engine to run to completion before stopping it. A switch therefore starts
  as soon as the current work is done, but never in the middle of it.
- **Requests are processed in arrival order** — a switch to a queued model
  does not start until every request that is waiting for, or being served
  by, the model it would stop has been processed. In particular the first
  request that triggers the initial load is always served before a switch
  requested while that load was still running, so an early request is
  never starved by a later model's switch.
- **Matching model or alias** — forwarded to the running engine immediately.
  If a switch is already pending (the current model is about to be
  replaced), these requests are held and served once the model becomes the
  active one again.
- **Another configured model** — a switch is started and the connection is
  held until that model is the active one, then the original request is
  forwarded. If the model is not ready within `--switch-timeout`
  (default 600s), the request gets a `503` with error code
  `model_switching` and a `Retry-After` header, so standard
  OpenAI/Anthropic SDK clients retry it unchanged.
- **Model queued behind a switch** — requests that arrive during an in-flight
  switch and name yet another model are queued in arrival order and held
  until their model loads.
- **Model not in the config** — an immediate `503` with error code
  `model_not_found`; the engine is not cycled.

If loading the new model fails, the supervisor restores the previous model
instead of staying down; requests held for the failed model get a `503`
with error code `model_load_failed` immediately (with `Retry-After`, so
SDK clients retry it — the retry starts a fresh load attempt), and the
model is removed from the queue so the supervisor moves on to the next
queued model instead of retrying it automatically.

### Options

Options specific to `serve-multi` (see `splash serve-multi --help` for the
full list):

| Option | Default | Purpose |
| --- | --- | --- |
| `--config FILE` | required | JSON config listing the models. |
| `--switch-timeout SECONDS` | `600` | How long a request waits while its model loads before a `503`. `0` answers `503` immediately. |
| `--switch-settle SECONDS` | `60` | How long to wait for the stopped engine's memory to be reclaimed before launching the next model; the wait ends early once the target is reached or memory is flat. `0` disables. |
| `--no-evict-cache` | off | Skip the stale-cache reclaim pass. Faster switches, higher risk of `Q4 buffer below plan` failures. |

The shared flags — `--host`, `--port`, `--default-reasoning-effort`,
`--kv-format`, `--max-memory`, `--max-cache-disk`, `--max-context`,
`--allowed-host`, `--max-request-size`, `--max-image-pixels`, `--api-key`,
`--no-webui` — have the same meaning as on `splash serve` and apply to every
engine instance. Per-model selection (`--revision`, `--draft-model`,
`--language-only`) is part of the model ID itself, not a shared flag.

### Memory between models

A stopped engine does not release memory instantly. Its file-backed weight
pages linger in active cache, and its GPU (wired) buffers stay counted until
the kernel reclaims them asynchronously; launching the next model into that
footprint is what causes `Q4 buffer below plan` failures. Before each launch
the supervisor:

1. measures available and wired memory via `vm_stat`,
2. briefly allocates and frees a large anonymous buffer to nudge macOS to
   demote the stale file cache (capped at 12 GiB; disabled by
   `--no-evict-cache`), and
3. polls until both pools have room for the next model (estimated from its
   weight file sizes), the machine is flat for ~15s with enough already
   available, or the `--switch-settle` budget runs out.

If the budget runs out before enough memory is available, it warns clearly
and proceeds anyway — the engine's planner reports the failure. On a machine
busy with other work, switches can take tens of seconds to a few minutes.

### Endpoints and health

Everything except the endpoints below is proxied to the active engine
unmodified, including the chat page, streaming (SSE) responses, and cache
management. While a switch is in progress (or no engine is ready):

| Endpoint | Behavior |
| --- | --- |
| `GET /ready` | `503` `{"status": "switching"}` |
| `GET /status` | `503` with `active_model`, `switch_target`, `queued_models`, and `in_flight` |
| `GET /v1/models` | `503` with an empty model list |
| `GET /metrics` | `503` |

When ready, all of these forward to the engine.

### Reliability

- Engine output is relayed to the supervisor's stdout unchanged;
  supervisor's own lines are marked `serve-multi ·`.
- An engine that exits unexpectedly is relaunched automatically; after three
  crashes within 60 seconds the supervisor parks and leaves a message
  instead of thrashing. A crash restart is queued exactly like a model
  switch: requests that arrive while the engine is (re)loading are held or
  queued in arrival order, and the in-progress load is never killed by a
  request for another model.
- If a model fails to load, the previous model is restored before any
  queued switch runs, and requests hold for it until it is back.
- `Ctrl+C` or `SIGTERM` stops the engine and the proxy cleanly.
