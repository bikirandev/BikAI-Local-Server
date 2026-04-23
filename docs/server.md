# server.py — AI Inference Server

## Overview

`server.py` is the **AI inference server** on port 8000.  
It loads GGUF models via `llama-cpp-python` and exposes an OpenAI-compatible REST API.

**It is spawned and killed by `controller.py`** — never started manually in normal usage.  
It can also be started directly for standalone use.

---

## Startup

```bash
# Normally started by controller via API
# But can be run directly:
python server.py --model models/gemma3-4b.gguf
python server.py --model models/gemma3-4b.gguf --parallel 4 --port 8000 --ctx 4096 --threads 8
```

On startup (`lifespan`):
1. Calls `ensure_api_key()` — generates and saves key if missing
2. Creates a `LlamaPool` — loads N model instances into memory
3. Creates a `ThreadPoolExecutor` with N workers (one per model instance)

---

## Configuration (via CLI args or .env)

| Arg | .env key | Default | Description |
|---|---|---|---|
| `--model` | `MODEL_PATH` | (required) | Path to `.gguf` file |
| `--parallel` | `N_PARALLEL` | 4 | Number of concurrent inference slots |
| `--ctx` | `N_CTX` | 4096 | Context window length per slot |
| `--threads` | `N_THREADS` | cpu_count | CPU threads for inference |
| `--port` | `PORT` | 8000 | Bind port |
| — | `API_KEY` | auto-generated | Auth key for all protected endpoints |
| — | `RATE_LIMIT` | 30/minute | Rate limit per IP |

---

## LlamaPool

The pool manages N Llama instances (one per parallel slot):

```python
class LlamaPool:
    def __init__(self, model_path, size, n_ctx, n_threads):
        # Divides threads evenly: per_instance = n_threads // size
        # Loads all instances upfront — startup takes time proportional to N
        self._queue = asyncio.Queue()

    async def acquire() -> Llama   # waits if all instances busy
    def release(instance: Llama)   # returns instance to pool
```

Each request:
1. `await pool.acquire()` — blocks until a slot is free
2. Runs inference in `ThreadPoolExecutor` (off the event loop)
3. `pool.release(llm)` in a `finally` block

**Threads are divided evenly**: if `N_THREADS=8` and `N_PARALLEL=4`, each instance gets 2 threads.

---

## API Endpoints

### Public

| Method | Path | Description |
|---|---|---|
| GET | `/health` | `{"status":"ok","model":"gemma3-4b"}` |
| OPTIONS | `/{path}` | CORS preflight — always 200 |

### Protected (`X-API-Key` required)

| Method | Path | Description |
|---|---|---|
| GET | `/v1/models` | OpenAI-compatible model list |
| POST | `/v1/chat/completions` | OpenAI-compatible chat (streaming + non-streaming) |
| POST | `/generate` | Simple text generation endpoint |

---

## `/v1/chat/completions` Request

```json
{
  "model": "optional",
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello!"}
  ],
  "stream": false,
  "temperature": 0.7,
  "max_tokens": 2048
}
```

**Non-streaming response:**
```json
{
  "id": "chatcmpl-abc123",
  "object": "chat.completion",
  "model": "gemma3-4b",
  "choices": [{
    "index": 0,
    "message": {"role": "assistant", "content": "Hello! How can I help?"},
    "finish_reason": "stop"
  }],
  "usage": {"prompt_tokens": 10, "completion_tokens": 8, "total_tokens": 18}
}
```

**Streaming (`"stream": true`)**: Returns `text/event-stream` SSE chunks:
```
data: {"id":"chatcmpl-...","object":"chat.completion.chunk","choices":[{"delta":{"content":"Hello"},...}]}
data: [DONE]
```

---

## `/generate` Request (Simple)

```json
{
  "prompt": "The capital of France is",
  "temperature": 0.7,
  "max_tokens": 100
}
```

Response:
```json
{
  "response": " Paris.",
  "model": "gemma3-4b",
  "usage": {...}
}
```

---

## Rate Limiting

Uses `slowapi` (wraps `limits`). Default: `30/minute` per IP.  
Change by setting `RATE_LIMIT=60/minute` in `.env` and restarting.

---

## Streaming Implementation

Streaming uses a producer/consumer pattern to avoid blocking the event loop:

```python
def _produce():
    for chunk in llm.create_chat_completion(..., stream=True):
        loop.call_soon_threadsafe(chunk_queue.put_nowait, chunk)
    loop.call_soon_threadsafe(chunk_queue.put_nowait, None)  # sentinel

loop.run_in_executor(executor, _produce)

while True:
    chunk = await chunk_queue.get()
    if chunk is None: break
    yield f"data: {json.dumps(sse_chunk)}\n\n"
```

The `_produce` function runs in a thread (so llama.cpp doesn't block the event loop), and puts chunks into an asyncio queue that the async generator consumes.

---

## Model Constraints

- Only GGUF format models supported (via llama-cpp-python)
- Context length capped at 8192 tokens (`min(n_ctx, 8192)`)
- CPU-only: `n_gpu_layers=0` (no CUDA/Metal)
- `verbose=False` suppresses llama.cpp internal logs

---

## Adding a New Inference Endpoint

1. Add a Pydantic request model
2. Create a sync helper function `_run_xxx_sync(llm, ...) -> dict`
3. In the async route: `await pool.acquire()` → `await _run_in_thread(_run_xxx_sync, llm, ...)` → `pool.release(llm)` in `finally`
4. Protect with `dependencies=[Depends(require_api_key)]`
