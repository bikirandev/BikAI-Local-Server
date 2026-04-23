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
from fastapi.responses import Response, StreamingResponse
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
    _state["started_at"] = time.time()
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
async def options_handler(_path: str = ""):
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


# ---------------------------------------------------------------------------
# Inference routes below — control UI moved to controller.py
# ---------------------------------------------------------------------------


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
