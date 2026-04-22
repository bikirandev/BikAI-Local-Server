"""
Gemma 2 API Server
==================
Runs Google's Gemma 2 model locally via llama-cpp-python (CPU-optimised,
no GPU required) and exposes it as a REST API (OpenAI-compatible format).

Parallel requests are handled natively: llama.cpp processes multiple
sequences simultaneously through its internal batch scheduler (n_parallel
slots), while FastAPI dispatches each blocking inference call to a thread-
pool executor so the event loop never stalls.

Usage:
  python server.py                               # Local only (http://localhost:8000)
  python server.py --model path/to/model.gguf   # Custom GGUF file
  python server.py --parallel 8                 # Allow 8 concurrent sequences

Expose publicly via nginx reverse proxy:
  bikai nginx --domain api.example.com          # HTTP
  bikai nginx --domain api.example.com --ssl    # HTTPS via Let's Encrypt

Recommended GGUF models (download from HuggingFace):
  bartowski/gemma-2-2b-it-GGUF  *Q4_K_M*  ~1.5 GB  (default)
  bartowski/gemma-2-9b-it-GGUF  *Q4_K_M*  ~5.4 GB
"""

import asyncio
import json
import os
import secrets
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator, List, Optional

import uvicorn
from dotenv import load_dotenv, set_key
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, Response, StreamingResponse
from fastapi.security import APIKeyHeader
from llama_cpp import Llama  # type: ignore[import-untyped]
from pydantic import BaseModel, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

load_dotenv()

ENV_FILE = Path(".env")

_config: dict = {
    "model_path": os.getenv("MODEL_PATH", ""),
    "api_key": os.getenv("API_KEY", ""),
    "rate_limit": os.getenv("RATE_LIMIT", "30/minute"),
    "n_parallel": int(os.getenv("N_PARALLEL", "4")),
    "n_ctx": int(os.getenv("N_CTX", "4096")),
    "n_threads": int(os.getenv("N_THREADS", str(os.cpu_count() or 4))),
}

# Pool of Llama instances — one per parallel slot (initialised in lifespan)
_state: dict = {"pool": None, "executor": None}

# ---------------------------------------------------------------------------
# Setup helpers
# ---------------------------------------------------------------------------


def ensure_api_key() -> None:
    if not _config["api_key"]:
        key = secrets.token_urlsafe(32)
        _config["api_key"] = key
        ENV_FILE.touch(exist_ok=True)
        set_key(str(ENV_FILE), "API_KEY", key)
        print(f"\n{'='*60}")
        print(f"  Generated API Key: {key}")
        print(f"  Saved to: {ENV_FILE.resolve()}")
        print("  Include this header in every request:")
        print(f"    X-API-Key: {key}")
        print(f"{'='*60}\n")
    else:
        print("[+] Using API key from .env")


def _make_instance(model_path: str, n_ctx: int, n_threads: int) -> Llama:
    return Llama(
        model_path=model_path,
        n_ctx=min(n_ctx, 8192),  # cap at Gemma 2's max training context
        n_threads=n_threads,
        n_gpu_layers=0,          # CPU-only (no CUDA on this machine)
        verbose=False,
    )


class LlamaPool:
    """Thread-safe pool of Llama instances. Each request borrows one instance."""

    def __init__(self, model_path: str, size: int, n_ctx: int, n_threads: int) -> None:
        if not model_path or not Path(model_path).is_file():
            raise RuntimeError(
                f"GGUF model not found at '{model_path}'.\n"
                "Download a model and pass --model /path/to/model.gguf\n"
                "Example:\n"
                "  hf download bartowski/gemma-2-2b-it-GGUF "
                "--include 'gemma-2-2b-it-Q4_K_M.gguf' --local-dir ./models"
            )
        # Divide threads evenly so instances don't fight for CPU
        per_instance_threads = max(1, n_threads // size)
        print(f"[*] Loading {size} model instance(s) from {model_path} ...")
        print(f"    n_ctx={min(n_ctx, 8192)}  threads_per_instance={per_instance_threads}")
        self._queue: asyncio.Queue = asyncio.Queue()
        for i in range(size):
            print(f"    [{i+1}/{size}] loading...", flush=True)
            self._queue.put_nowait(
                _make_instance(model_path, n_ctx, per_instance_threads)
            )
        print("[+] Model pool ready.")

    async def acquire(self) -> Llama:
        return await self._queue.get()

    def release(self, instance: Llama) -> None:
        self._queue.put_nowait(instance)


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------

limiter = Limiter(key_func=get_remote_address)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    ensure_api_key()
    _state["pool"] = LlamaPool(
        _config["model_path"],
        size=_config["n_parallel"],
        n_ctx=_config["n_ctx"],
        n_threads=_config["n_threads"],
    )
    # One thread per instance so blocking inference never stalls the event loop
    _state["executor"] = ThreadPoolExecutor(max_workers=_config["n_parallel"])
    yield
    _state["executor"].shutdown(wait=False)


app = FastAPI(
    title="Gemma 2 API",
    description="Local Gemma 2 inference via llama-cpp-python — OpenAI-compatible API",
    version="2.0.0",
    lifespan=lifespan,
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
    allow_credentials=False,
    expose_headers=["*"],
)

# ---------------------------------------------------------------------------
# Authentication
# ---------------------------------------------------------------------------

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=True)


async def require_api_key(key: str = Depends(_api_key_header)) -> str:
    if not _config["api_key"]:
        raise HTTPException(status_code=500, detail="Server API key not configured.")
    if not secrets.compare_digest(key, _config["api_key"]):
        raise HTTPException(status_code=401, detail="Invalid API key.")
    return key


# ---------------------------------------------------------------------------
# Catch-all OPTIONS handler — ensures CORS preflight always succeeds
# even when ngrok intercepts before FastAPI middleware can respond.
# ---------------------------------------------------------------------------


@app.options("/{path:path}")
async def options_handler(path: str):
    return Response(
        status_code=200,
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
            "Access-Control-Allow-Headers": "*, X-API-Key, Content-Type, ngrok-skip-browser-warning",
            "Access-Control-Max-Age": "86400",
        },
    )


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------


class Message(BaseModel):
    role: str = Field(..., pattern="^(system|user|assistant)$")
    content: str = Field(..., min_length=1, max_length=32_768)


class ChatRequest(BaseModel):
    model: Optional[str] = None
    messages: List[Message] = Field(..., min_length=1, max_length=50)
    stream: bool = False
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=2048, ge=1, le=8192)


class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=32_768)
    model: Optional[str] = None
    stream: bool = False
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=2048, ge=1, le=8192)


# ---------------------------------------------------------------------------
# Inference helpers (run in thread pool to keep event loop free)
# ---------------------------------------------------------------------------


def _run_chat_sync(llm: Llama, messages: list, temperature: float, max_tokens: int) -> dict:
    return llm.create_chat_completion(
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        stream=False,
    )


def _run_generate_sync(llm: Llama, prompt: str, temperature: float, max_tokens: int) -> dict:
    return llm.create_completion(
        prompt=prompt,
        temperature=temperature,
        max_tokens=max_tokens,
        stream=False,
    )


async def _run_in_thread(fn, *fn_args):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(_state["executor"], fn, *fn_args)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/health")
async def health():
    """Public health-check endpoint — no auth required."""
    return {"status": "ok", "model": Path(_config["model_path"]).name}


@app.get("/server/info", response_class=HTMLResponse)
async def server_info(request: Request):
    """Public server info page — shows model, endpoints, and usage."""
    model_name = Path(_config["model_path"]).name
    model_stem = Path(_config["model_path"]).stem
    host = request.headers.get("host", "localhost")
    scheme = request.headers.get("x-forwarded-proto", "http")
    base_url = f"{scheme}://{host}"
    parallel = _config["n_parallel"]
    ctx = _config["n_ctx"]
    threads = _config["n_threads"]
    uptime_since = getattr(_state, "started_at", "unknown")

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>Bik AI — Server Info</title>
  <style>
    *{{box-sizing:border-box;margin:0;padding:0}}
    body{{font-family:'Segoe UI',system-ui,sans-serif;background:#0f1117;color:#e2e8f0;min-height:100vh;padding:2rem}}
    .logo{{color:#38bdf8;font-weight:800;font-size:1.5rem;letter-spacing:.05em;margin-bottom:.25rem}}
    .sub{{color:#64748b;font-size:.875rem;margin-bottom:2rem}}
    .card{{background:#1e2330;border:1px solid #2d3748;border-radius:.75rem;padding:1.5rem;margin-bottom:1.25rem}}
    .card h2{{font-size:.75rem;font-weight:600;text-transform:uppercase;letter-spacing:.1em;color:#64748b;margin-bottom:1rem}}
    .row{{display:flex;justify-content:space-between;align-items:center;padding:.5rem 0;border-bottom:1px solid #2d3748}}
    .row:last-child{{border-bottom:none}}
    .label{{color:#94a3b8;font-size:.875rem}}
    .value{{font-size:.875rem;font-weight:500;color:#e2e8f0}}
    .value.green{{color:#4ade80}}
    .value.blue{{color:#38bdf8}}
    .value.yellow{{color:#fbbf24}}
    .endpoint{{background:#0f1117;border:1px solid #2d3748;border-radius:.5rem;padding:.75rem 1rem;margin-bottom:.75rem}}
    .method{{display:inline-block;padding:.2rem .6rem;border-radius:.25rem;font-size:.7rem;font-weight:700;margin-right:.75rem}}
    .get{{background:#1e3a5f;color:#60a5fa}}
    .post{{background:#1e3d2a;color:#4ade80}}
    .path{{font-family:monospace;font-size:.875rem;color:#e2e8f0}}
    .desc{{color:#64748b;font-size:.8rem;margin-top:.35rem;padding-left:3.5rem}}
    .curl-box{{background:#0f1117;border:1px solid #2d3748;border-radius:.5rem;padding:1rem;font-family:monospace;font-size:.8rem;color:#a5f3fc;overflow-x:auto;white-space:pre;margin-top:.75rem}}
    .badge{{display:inline-flex;align-items:center;gap:.4rem;padding:.3rem .8rem;border-radius:9999px;font-size:.75rem;font-weight:600}}
    .badge.running{{background:#14532d;color:#4ade80}}
    a{{color:#38bdf8;text-decoration:none}}
    a:hover{{text-decoration:underline}}
    @media(max-width:600px){{body{{padding:1rem}}.row{{flex-direction:column;align-items:flex-start;gap:.25rem}}}}
  </style>
</head>
<body>
  <div class="logo">BIK AI</div>
  <div class="sub">Local LLM Server &mdash; by <a href="https://bikiran.com" target="_blank">bikiran.com</a></div>

  <div class="card">
    <h2>Server Status</h2>
    <div class="row">
      <span class="label">Status</span>
      <span class="badge running">&#x25CF; Running</span>
    </div>
    <div class="row">
      <span class="label">Model</span>
      <span class="value blue">{model_name}</span>
    </div>
    <div class="row">
      <span class="label">Parallel slots</span>
      <span class="value">{parallel}</span>
    </div>
    <div class="row">
      <span class="label">Context window</span>
      <span class="value">{ctx:,} tokens</span>
    </div>
    <div class="row">
      <span class="label">CPU threads</span>
      <span class="value">{threads}</span>
    </div>
    <div class="row">
      <span class="label">Base URL</span>
      <span class="value"><a href="{base_url}">{base_url}</a></span>
    </div>
  </div>

  <div class="card">
    <h2>Endpoints</h2>
    <div class="endpoint">
      <span class="method get">GET</span><span class="path">/health</span>
      <div class="desc">Health check &mdash; no auth required</div>
    </div>
    <div class="endpoint">
      <span class="method get">GET</span><span class="path">/server/info</span>
      <div class="desc">This page</div>
    </div>
    <div class="endpoint">
      <span class="method get">GET</span><span class="path">/v1/models</span>
      <div class="desc">List loaded model &mdash; requires X-API-Key header</div>
    </div>
    <div class="endpoint">
      <span class="method post">POST</span><span class="path">/v1/chat/completions</span>
      <div class="desc">OpenAI-compatible chat &mdash; supports streaming</div>
    </div>
    <div class="endpoint">
      <span class="method post">POST</span><span class="path">/generate</span>
      <div class="desc">Simple text generation</div>
    </div>
    <div class="endpoint">
      <span class="method get">GET</span><span class="path">/docs</span>
      <div class="desc">Interactive API documentation (Swagger UI)</div>
    </div>
  </div>

  <div class="card">
    <h2>Example Request</h2>
    <div class="curl-box">curl {base_url}/v1/chat/completions \\
  -H "X-API-Key: YOUR_API_KEY" \\
  -H "Content-Type: application/json" \\
  -d '{{"messages":[{{"role":"user","content":"Hello!"}}],"stream":false}}'</div>
  </div>

  <div class="card">
    <h2>Authentication</h2>
    <div class="row">
      <span class="label">Header</span>
      <span class="value yellow" style="font-family:monospace">X-API-Key: &lt;your-key&gt;</span>
    </div>
    <div class="row">
      <span class="label">Get your key</span>
      <span class="value" style="font-family:monospace">bikai token show</span>
    </div>
  </div>
</body>
</html>"""
    return HTMLResponse(content=html)


@app.get("/v1/models", dependencies=[Depends(require_api_key)])
async def list_models():
    """List loaded model (OpenAI-compatible)."""
    return {
        "object": "list",
        "data": [
            {
                "id": Path(_config["model_path"]).stem,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "google",
            }
        ],
    }


@app.post("/v1/chat/completions", dependencies=[Depends(require_api_key)])
@limiter.limit(_config["rate_limit"])
async def chat_completions(request: Request, body: ChatRequest):
    """OpenAI-compatible chat completions — supports streaming."""
    _ = request
    messages = [{"role": m.role, "content": m.content} for m in body.messages]

    pool: LlamaPool = _state["pool"]

    if body.stream:
        return StreamingResponse(
            _stream_chat(pool, messages, body.temperature, body.max_tokens),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    llm = await pool.acquire()
    try:
        data = await _run_in_thread(
            _run_chat_sync, llm, messages, body.temperature, body.max_tokens
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Inference error: {exc}") from exc
    finally:
        pool.release(llm)

    choice = data["choices"][0]
    return {
        "id": f"chatcmpl-{secrets.token_hex(8)}",
        "object": "chat.completion",
        "model": Path(_config["model_path"]).stem,
        "choices": [
            {
                "index": 0,
                "message": choice["message"],
                "finish_reason": choice.get("finish_reason", "stop"),
            }
        ],
        "usage": data.get("usage", {}),
    }


async def _stream_chat(
    pool: LlamaPool, messages: list, temperature: float, max_tokens: int
) -> AsyncGenerator[str, None]:
    llm = await pool.acquire()
    loop = asyncio.get_event_loop()
    chunk_queue: asyncio.Queue = asyncio.Queue()

    def _produce():
        try:
            for chunk in llm.create_chat_completion(
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True,
            ):
                loop.call_soon_threadsafe(chunk_queue.put_nowait, chunk)
        finally:
            loop.call_soon_threadsafe(chunk_queue.put_nowait, None)  # sentinel
            pool.release(llm)

    loop.run_in_executor(_state["executor"], _produce)

    while True:
        chunk = await chunk_queue.get()
        if chunk is None:
            yield "data: [DONE]\n\n"
            break
        delta = chunk["choices"][0].get("delta", {})
        content = delta.get("content", "")
        if content:
            sse = {
                "id": f"chatcmpl-{secrets.token_hex(8)}",
                "object": "chat.completion.chunk",
                "choices": [
                    {
                        "index": 0,
                        "delta": {"role": "assistant", "content": content},
                        "finish_reason": None,
                    }
                ],
            }
            yield f"data: {json.dumps(sse)}\n\n"


@app.post("/generate", dependencies=[Depends(require_api_key)])
@limiter.limit(_config["rate_limit"])
async def generate(request: Request, body: GenerateRequest):
    """Simple single-turn text generation endpoint."""
    _ = request
    pool: LlamaPool = _state["pool"]
    llm = await pool.acquire()
    try:
        data = await _run_in_thread(
            _run_generate_sync, llm, body.prompt, body.temperature, body.max_tokens
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Inference error: {exc}") from exc
    finally:
        pool.release(llm)

    return {
        "response": data["choices"][0]["text"],
        "model": Path(_config["model_path"]).stem,
        "usage": data.get("usage", {}),
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Gemma 2 API Server (llama-cpp-python)")
    parser.add_argument("--host", default="0.0.0.0", help="Bind host (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="Bind port (default: 8000)")
    parser.add_argument(
        "--model",
        default=os.getenv("MODEL_PATH", ""),
        help="Path to GGUF model file",
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=int(os.getenv("N_PARALLEL", "4")),
        help="Number of parallel inference slots (default: 4)",
    )
    parser.add_argument(
        "--ctx",
        type=int,
        default=int(os.getenv("N_CTX", "4096")),
        help="Context length per slot in tokens (default: 4096)",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=int(os.getenv("N_THREADS", str(os.cpu_count() or 4))),
        help="CPU threads for inference (default: all cores)",
    )
    args = parser.parse_args()

    _config["model_path"] = args.model
    _config["n_parallel"] = args.parallel
    _config["n_ctx"] = args.ctx
    _config["n_threads"] = args.threads

    print(f"[*] Starting server on http://{args.host}:{args.port}")
    print(f"[*] API docs: http://localhost:{args.port}/docs\n")

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
