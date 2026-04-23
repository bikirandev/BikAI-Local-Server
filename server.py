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
import resource
import secrets
import signal
import subprocess
import sys
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


# ---------------------------------------------------------------------------
# Control UI — JSON API
# ---------------------------------------------------------------------------

def _read_env_val(key: str, default: str = "") -> str:
    load_dotenv(override=True)
    return os.getenv(key, default)


def _write_env_val(key: str, value: str) -> None:
    ENV_FILE.touch(exist_ok=True)
    set_key(str(ENV_FILE), key, value)


def _read_pid() -> int | None:
    pid_file = Path(".bikai.pid")
    if pid_file.exists():
        try:
            return int(pid_file.read_text().strip())
        except ValueError:
            pass
    return None


@app.get("/api/ui/status")
async def ui_status():
    """Return current server + config status for the control UI."""
    model_path = Path(_config["model_path"])
    pid = _read_pid()
    try:
        mem_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024
    except Exception:
        mem_mb = 0
    try:
        size_bytes = model_path.stat().st_size
        size_str = f"{size_bytes / 1_073_741_824:.2f} GB"
    except Exception:
        size_str = "unknown"

    started_at = _state.get("started_at")
    if started_at:
        elapsed = int(time.time() - started_at)
        h, m = divmod(elapsed // 60, 60)
        s = elapsed % 60
        uptime_str = f"{h}h {m}m {s}s" if h else f"{m}m {s}s"
    else:
        uptime_str = "—"

    return {
        "running": True,
        "pid": pid,
        "uptime": uptime_str,
        "model_name": model_path.name,
        "model_path": str(model_path),
        "model_size": size_str,
        "parallel": _config["n_parallel"],
        "ctx": _config["n_ctx"],
        "threads": _config["n_threads"],
        "rate_limit": _config["rate_limit"],
        "port": _read_env_val("PORT", "8000"),
        "domain": _read_env_val("DOMAIN", ""),
        "mem_mb": mem_mb,
    }


@app.get("/api/ui/models")
async def ui_models():
    """List downloaded GGUF models."""
    models_dir = Path("models")
    active = _read_env_val("MODEL_PATH", "")
    result = []
    if models_dir.exists():
        for f in sorted(models_dir.glob("**/*.gguf")):
            is_active = str(f) == active or (active and str(f.resolve()) == str(Path(active).resolve()))
            result.append({
                "name": f.name,
                "path": str(f),
                "size": f"{f.stat().st_size / 1_073_741_824:.2f} GB",
                "active": is_active,
            })
    return {"models": result}


class StartRequest(BaseModel):
    model: str
    parallel: int = 4
    port: int = 8000
    ctx: int = 4096
    threads: int = 4
    daemon: bool = True


@app.post("/api/ui/start", dependencies=[Depends(require_api_key)])
async def ui_start(req: StartRequest):
    """Start the server with new config (restarts if already running)."""
    # Stop current process if running
    pid = _read_pid()
    if pid:
        try:
            os.kill(pid, signal.SIGTERM)
            time.sleep(1)
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    Path(".bikai.pid").unlink(missing_ok=True)

    cmd = [
        sys.executable, "server.py",
        "--model", req.model,
        "--parallel", str(req.parallel),
        "--port", str(req.port),
        "--ctx", str(req.ctx),
        "--threads", str(req.threads),
    ]
    log = open("bikai-server.log", "w")
    proc = subprocess.Popen(cmd, stdout=log, stderr=log)
    Path(".bikai.pid").write_text(str(proc.pid))
    _write_env_val("PORT", str(req.port))
    _write_env_val("MODEL_PATH", req.model)
    return {"ok": True, "pid": proc.pid}


@app.post("/api/ui/stop", dependencies=[Depends(require_api_key)])
async def ui_stop():
    """Stop the running server process."""
    pid = _read_pid()
    if pid:
        try:
            os.kill(pid, signal.SIGTERM)
            time.sleep(1)
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        except ProcessLookupError:
            pass
        Path(".bikai.pid").unlink(missing_ok=True)
    port = _read_env_val("PORT", "8000")
    result = subprocess.run(["fuser", f"{port}/tcp"], capture_output=True, text=True)
    for p in result.stdout.strip().split():
        try:
            os.kill(int(p), signal.SIGTERM)
        except (ValueError, ProcessLookupError):
            pass
    return {"ok": True}


@app.post("/api/ui/restart", dependencies=[Depends(require_api_key)])
async def ui_restart():
    """Restart the server using current .env config."""
    model = _read_env_val("MODEL_PATH")
    if not model:
        raise HTTPException(status_code=400, detail="MODEL_PATH not set")
    req = StartRequest(
        model=model,
        parallel=int(_read_env_val("N_PARALLEL", "4")),
        port=int(_read_env_val("PORT", "8000")),
        ctx=int(_read_env_val("N_CTX", "4096")),
        threads=int(_read_env_val("N_THREADS", str(os.cpu_count() or 4))),
    )
    return await ui_start(req)


@app.get("/api/ui/logs")
async def ui_logs(lines: int = 100):
    """Return last N lines from the server log file."""
    log_file = Path("bikai-server.log")
    if not log_file.exists():
        return {"lines": []}
    try:
        result = subprocess.run(
            ["tail", f"-n{lines}", str(log_file)],
            capture_output=True, text=True
        )
        return {"lines": result.stdout.splitlines()}
    except Exception:
        return {"lines": []}


class DownloadRequest(BaseModel):
    type: str  # "gdrive" | "huggingface" | "url"
    id: str = ""
    repo: str = ""
    file: str = ""
    url: str = ""


@app.post("/api/ui/download", dependencies=[Depends(require_api_key)])
async def ui_download(req: DownloadRequest):
    """Trigger a model download in background via bikai CLI."""
    bikai_bin = Path(sys.executable).parent / "bikai"
    if not bikai_bin.exists():
        # Try installed path
        bikai_bin = Path.home() / ".local" / "bin" / "bikai"
    if not bikai_bin.exists():
        raise HTTPException(status_code=500, detail="bikai binary not found")

    if req.type == "gdrive":
        if not req.id:
            raise HTTPException(status_code=400, detail="id required for gdrive")
        cmd = [str(bikai_bin), "download", "-g", req.id, "--set-default"]
    elif req.type == "huggingface":
        if not req.repo or not req.file:
            raise HTTPException(status_code=400, detail="repo and file required")
        cmd = [str(bikai_bin), "download", "-r", req.repo, "-f", req.file, "--set-default"]
    elif req.type == "url":
        if not req.url:
            raise HTTPException(status_code=400, detail="url required")
        cmd = [str(bikai_bin), "download", "-u", req.url, "--set-default"]
    else:
        raise HTTPException(status_code=400, detail="Unknown type")

    log = open("bikai-server.log", "a")
    subprocess.Popen(cmd, stdout=log, stderr=log)
    return {"ok": True, "message": "Download started in background. Check logs for progress."}


class NginxRequest(BaseModel):
    domain: str = ""
    port: int = 8000
    ssl: bool = False


@app.post("/api/ui/nginx", dependencies=[Depends(require_api_key)])
async def ui_nginx(req: NginxRequest):
    """Configure nginx as a reverse proxy."""
    import re
    domain = req.domain.strip()
    if not domain:
        try:
            import urllib.request
            domain = urllib.request.urlopen("https://api.ipify.org", timeout=5).read().decode().strip()
        except Exception:
            raise HTTPException(status_code=500, detail="Could not detect public IP")

    is_ip = bool(re.match(r"^\d{1,3}(\.\d{1,3}){3}$", domain))
    server_name = "_" if is_ip else domain

    conf = f"""server {{
    listen 80;
    server_name {server_name};
    location / {{
        proxy_pass http://127.0.0.1:{req.port};
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_buffering off;
        proxy_cache off;
        proxy_read_timeout 300s;
        add_header 'Access-Control-Allow-Origin' '*' always;
        add_header 'Access-Control-Allow-Methods' 'GET, POST, OPTIONS' always;
        add_header 'Access-Control-Allow-Headers' 'X-API-Key, Content-Type, Authorization' always;
        if ($request_method = OPTIONS) {{
            add_header 'Access-Control-Allow-Origin' '*';
            return 204;
        }}
    }}
}}
"""
    proc = subprocess.run(
        ["sudo", "tee", "/etc/nginx/sites-available/bikai"],
        input=conf, text=True, capture_output=True,
    )
    if proc.returncode != 0:
        raise HTTPException(status_code=500, detail=f"nginx write failed: {proc.stderr}")

    subprocess.run(["sudo", "ln", "-sf", "/etc/nginx/sites-available/bikai",
                    "/etc/nginx/sites-enabled/bikai"], check=False)
    subprocess.run(["sudo", "rm", "-f", "/etc/nginx/sites-enabled/default"], check=False)
    subprocess.run(["sudo", "systemctl", "enable", "nginx"], check=False)
    subprocess.run(["sudo", "systemctl", "reload", "nginx"], check=False)
    _write_env_val("DOMAIN", domain)
    return {"ok": True, "domain": domain, "url": f"http://{domain}"}


class TokenRequest(BaseModel):
    confirm: bool = False


@app.post("/api/ui/token/new", dependencies=[Depends(require_api_key)])
async def ui_token_new():
    """Generate a new API key."""
    key = secrets.token_urlsafe(32)
    _write_env_val("API_KEY", key)
    _config["api_key"] = key
    return {"ok": True, "key": key}


@app.get("/api/ui/token", dependencies=[Depends(require_api_key)])
async def ui_token_show():
    """Return the current API key."""
    key = _read_env_val("API_KEY", "")
    return {"key": key}


@app.get("/server/info", response_class=HTMLResponse)
async def server_info(request: Request):
    """Public server info page — shows model, endpoints, and usage."""
    model_path = Path(_config["model_path"])
    model_name = model_path.name
    model_stem = model_path.stem
    host = request.headers.get("host", "localhost")
    scheme = request.headers.get("x-forwarded-proto", "http")
    base_url = f"{scheme}://{host}"
    parallel  = _config["n_parallel"]
    ctx       = _config["n_ctx"]
    threads   = _config["n_threads"]
    rate_limit = _config["rate_limit"]
    port      = os.getenv("PORT", "8000")

    # Model file size
    try:
        size_bytes = model_path.stat().st_size
        size_str = f"{size_bytes / 1_073_741_824:.2f} GB"
    except Exception:
        size_str = "unknown"

    # Uptime
    started_at = _state.get("started_at")
    if started_at:
        elapsed = int(time.time() - started_at)
        h, m = divmod(elapsed // 60, 60)
        s = elapsed % 60
        uptime_str = f"{h}h {m}m {s}s" if h else f"{m}m {s}s"
    else:
        uptime_str = "unknown"

    # CPU / RAM info
    import platform, resource
    cpu_count = os.cpu_count() or "?"
    try:
        mem_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024
        mem_str = f"{mem_mb:,} MB"
    except Exception:
        mem_str = "unknown"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>Bik AI — Server Info</title>
  <style>
    *{{box-sizing:border-box;margin:0;padding:0}}
    body{{font-family:'Segoe UI',system-ui,sans-serif;background:#0f1117;color:#e2e8f0;min-height:100vh;padding:2rem}}
    .logo{{color:#38bdf8;font-weight:800;font-size:1.6rem;letter-spacing:.05em;margin-bottom:.25rem}}
    .sub{{color:#64748b;font-size:.875rem;margin-bottom:2rem}}
    .grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:1.25rem;margin-bottom:1.25rem}}
    .card{{background:#1e2330;border:1px solid #2d3748;border-radius:.75rem;padding:1.5rem}}
    .card h2{{font-size:.7rem;font-weight:700;text-transform:uppercase;letter-spacing:.12em;color:#64748b;margin-bottom:1rem}}
    .row{{display:flex;justify-content:space-between;align-items:center;padding:.45rem 0;border-bottom:1px solid #1a2035}}
    .row:last-child{{border-bottom:none}}
    .label{{color:#94a3b8;font-size:.8rem}}
    .value{{font-size:.85rem;font-weight:500;color:#e2e8f0;text-align:right;word-break:break-all;max-width:60%}}
    .green{{color:#4ade80}} .blue{{color:#38bdf8}} .yellow{{color:#fbbf24}} .mono{{font-family:monospace}}
    .badge{{display:inline-flex;align-items:center;gap:.4rem;padding:.3rem .8rem;border-radius:9999px;font-size:.75rem;font-weight:600;background:#14532d;color:#4ade80}}
    .endpoint{{background:#0f1117;border:1px solid #2d3748;border-radius:.5rem;padding:.75rem 1rem;margin-bottom:.6rem}}
    .method{{display:inline-block;padding:.2rem .55rem;border-radius:.25rem;font-size:.68rem;font-weight:700;margin-right:.6rem}}
    .get{{background:#1e3a5f;color:#60a5fa}} .post{{background:#1e3d2a;color:#4ade80}}
    .path{{font-family:monospace;font-size:.85rem}}
    .desc{{color:#64748b;font-size:.78rem;margin-top:.3rem;padding-left:3.2rem}}
    .code{{background:#0f1117;border:1px solid #2d3748;border-radius:.5rem;padding:1rem;font-family:monospace;font-size:.78rem;color:#a5f3fc;overflow-x:auto;white-space:pre;margin-top:.75rem;line-height:1.6}}
    .cmd{{background:#1a1f2e;border:1px solid #38bdf855;border-radius:.4rem;padding:.5rem 1rem;font-family:monospace;font-size:.82rem;color:#38bdf8;display:inline-block;margin-top:.4rem}}
    a{{color:#38bdf8;text-decoration:none}} a:hover{{text-decoration:underline}}
    @media(max-width:600px){{body{{padding:1rem}}.row{{flex-direction:column;align-items:flex-start;gap:.2rem}}.value{{max-width:100%;text-align:left}}}}
  </style>
</head>
<body>
  <div class="logo">BIK AI</div>
  <div class="sub">Local LLM Server &mdash; by <a href="https://bikiran.com" target="_blank">bikiran.com</a></div>

  <div class="grid">

    <div class="card">
      <h2>Server Status</h2>
      <div class="row"><span class="label">Status</span><span class="badge">&#x25CF; Running</span></div>
      <div class="row"><span class="label">Uptime</span><span class="value green">{uptime_str}</span></div>
      <div class="row"><span class="label">Port</span><span class="value mono">{port}</span></div>
      <div class="row"><span class="label">Base URL</span><span class="value blue"><a href="{base_url}">{base_url}</a></span></div>
      <div class="row"><span class="label">Rate limit</span><span class="value">{rate_limit}</span></div>
    </div>

    <div class="card">
      <h2>Model</h2>
      <div class="row"><span class="label">File name</span><span class="value blue mono">{model_name}</span></div>
      <div class="row"><span class="label">Model ID</span><span class="value mono">{model_stem}</span></div>
      <div class="row"><span class="label">File size</span><span class="value">{size_str}</span></div>
      <div class="row"><span class="label">Full path</span><span class="value mono" style="font-size:.72rem">{model_path}</span></div>
    </div>

    <div class="card">
      <h2>Inference Config</h2>
      <div class="row"><span class="label">Parallel slots</span><span class="value yellow">{parallel}</span></div>
      <div class="row"><span class="label">Context window</span><span class="value">{ctx:,} tokens</span></div>
      <div class="row"><span class="label">CPU threads</span><span class="value">{threads}</span></div>
      <div class="row"><span class="label">Total CPU cores</span><span class="value">{cpu_count}</span></div>
      <div class="row"><span class="label">Memory usage</span><span class="value">{mem_str}</span></div>
    </div>

    <div class="card">
      <h2>Authentication</h2>
      <div class="row"><span class="label">Header</span><span class="value mono yellow">X-API-Key: &lt;key&gt;</span></div>
      <div class="row"><span class="label">Token required</span><span class="value">All routes except <span class="mono">/health</span> and <span class="mono">/server/info</span></span></div>
      <div class="row" style="flex-direction:column;align-items:flex-start;gap:.5rem;padding-top:.75rem">
        <span class="label">View your API key on the server:</span>
        <span class="cmd">bikai token show</span>
      </div>
    </div>

  </div>

  <div class="card" style="margin-bottom:1.25rem">
    <h2>Endpoints</h2>
    <div class="endpoint">
      <span class="method get">GET</span><span class="path">/health</span>
      <div class="desc">Health check &mdash; no auth required</div>
    </div>
    <div class="endpoint">
      <span class="method get">GET</span><span class="path">/server/info</span>
      <div class="desc">This page &mdash; no auth required</div>
    </div>
    <div class="endpoint">
      <span class="method get">GET</span><span class="path">/v1/models</span>
      <div class="desc">List loaded model &mdash; OpenAI-compatible, requires X-API-Key</div>
    </div>
    <div class="endpoint">
      <span class="method post">POST</span><span class="path">/v1/chat/completions</span>
      <div class="desc">Chat completions &mdash; OpenAI-compatible, supports streaming (SSE)</div>
    </div>
    <div class="endpoint">
      <span class="method post">POST</span><span class="path">/generate</span>
      <div class="desc">Simple text generation &mdash; requires X-API-Key</div>
    </div>
    <div class="endpoint">
      <span class="method get">GET</span><span class="path">/docs</span>
      <div class="desc">Interactive Swagger UI</div>
    </div>
  </div>

  <div class="card">
    <h2>Example Request</h2>
    <div class="code">curl {base_url}/v1/chat/completions \\
  -H "X-API-Key: YOUR_API_KEY" \\
  -H "Content-Type: application/json" \\
  -d '{{"model":"{model_stem}","messages":[{{"role":"user","content":"Hello!"}}],"stream":false}}'</div>
  </div>

  <div class="card">
    <h2>Custom Domain Setup</h2>
    <p style="color:#94a3b8;font-size:.82rem;margin-bottom:1rem">Point a domain or subdomain at this server, then run one command to configure nginx and optionally get a free SSL certificate.</p>

    <div style="margin-bottom:1rem">
      <div style="color:#64748b;font-size:.72rem;font-weight:700;text-transform:uppercase;letter-spacing:.1em;margin-bottom:.5rem">Step 1 &mdash; Point DNS</div>
      <p style="color:#94a3b8;font-size:.82rem">Add an <strong style="color:#e2e8f0">A record</strong> in your DNS provider pointing your domain to this server&apos;s IP:</p>
      <div class="code" style="margin-top:.5rem">A  api.yourdomain.com  →  {host.split(":")[0]}</div>
    </div>

    <div style="margin-bottom:1rem">
      <div style="color:#64748b;font-size:.72rem;font-weight:700;text-transform:uppercase;letter-spacing:.1em;margin-bottom:.5rem">Step 2 &mdash; Configure nginx (HTTP)</div>
      <div class="code">bikai nginx --domain api.yourdomain.com</div>
    </div>

    <div style="margin-bottom:1rem">
      <div style="color:#64748b;font-size:.72rem;font-weight:700;text-transform:uppercase;letter-spacing:.1em;margin-bottom:.5rem">Step 2 &mdash; Configure nginx + SSL (HTTPS) &mdash; recommended</div>
      <p style="color:#94a3b8;font-size:.82rem;margin-bottom:.5rem">Requires DNS to be live first. Gets a free Let&apos;s Encrypt certificate automatically.</p>
      <div class="code">bikai nginx --domain api.yourdomain.com --ssl</div>
    </div>

    <div>
      <div style="color:#64748b;font-size:.72rem;font-weight:700;text-transform:uppercase;letter-spacing:.1em;margin-bottom:.5rem">After setup</div>
      <div class="code">bikai url            # show current public URL
bikai nginx --status # check nginx config &amp; service</div>
    </div>
  </div>

</body>
</html>"""
    return HTMLResponse(content=html)


@app.get("/ui", response_class=HTMLResponse)
async def control_ui(request: Request):
    """Control panel — manage the server from a browser."""
    host = request.headers.get("host", "localhost")
    scheme = request.headers.get("x-forwarded-proto", "http")
    base_url = f"{scheme}://{host}"

    html = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1.0"/>
<title>Bik AI — Control Panel</title>
<style>
:root{
  --bg:#0d1117;--surface:#161b22;--border:#30363d;--border2:#21262d;
  --text:#e6edf3;--muted:#8b949e;--blue:#58a6ff;--green:#3fb950;
  --red:#f85149;--yellow:#d29922;--purple:#bc8cff;--orange:#ffa657;
  --radius:8px;--font:'Inter',system-ui,-apple-system,sans-serif;
}
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:var(--font);background:var(--bg);color:var(--text);min-height:100vh;font-size:14px}
a{color:var(--blue);text-decoration:none}
a:hover{text-decoration:underline}

/* Layout */
.shell{display:flex;min-height:100vh}
.sidebar{width:220px;background:var(--surface);border-right:1px solid var(--border);display:flex;flex-direction:column;flex-shrink:0;position:fixed;top:0;left:0;height:100vh;z-index:100}
.main{margin-left:220px;flex:1;display:flex;flex-direction:column;min-height:100vh}
.topbar{height:56px;background:var(--surface);border-bottom:1px solid var(--border);display:flex;align-items:center;padding:0 24px;gap:16px;position:sticky;top:0;z-index:50}
.content{padding:28px 32px;flex:1}

/* Sidebar */
.brand{padding:20px 18px 12px;border-bottom:1px solid var(--border2)}
.brand-name{font-size:16px;font-weight:700;color:var(--text);letter-spacing:.01em}
.brand-sub{font-size:11px;color:var(--muted);margin-top:2px}
.nav{padding:10px 8px;flex:1;overflow-y:auto}
.nav-section{font-size:10px;font-weight:600;text-transform:uppercase;letter-spacing:.08em;color:var(--muted);padding:10px 10px 4px}
.nav-item{display:flex;align-items:center;gap:10px;padding:8px 10px;border-radius:var(--radius);cursor:pointer;color:var(--muted);font-size:13px;font-weight:500;transition:all .15s;margin-bottom:1px;border:none;background:none;width:100%;text-align:left}
.nav-item:hover{background:#21262d;color:var(--text)}
.nav-item.active{background:#1f2937;color:var(--blue)}
.nav-item svg{width:15px;height:15px;flex-shrink:0;opacity:.8}
.nav-item.active svg{opacity:1}
.sidebar-footer{padding:14px 18px;border-top:1px solid var(--border2);font-size:11px;color:var(--muted)}

/* Topbar */
.topbar-title{font-weight:600;font-size:15px;flex:1}
.status-pill{display:inline-flex;align-items:center;gap:6px;padding:4px 12px;border-radius:20px;font-size:12px;font-weight:600}
.status-pill.running{background:#0d2818;color:var(--green);border:1px solid #1a3a25}
.status-pill.stopped{background:#2d1012;color:var(--red);border:1px solid #3d1a1c}
.dot{width:7px;height:7px;border-radius:50%;background:currentColor}

/* Panels */
.panel{display:none}
.panel.active{display:block}

/* Cards */
.card{background:var(--surface);border:1px solid var(--border);border-radius:var(--radius);padding:20px;margin-bottom:16px}
.card-title{font-size:12px;font-weight:700;text-transform:uppercase;letter-spacing:.08em;color:var(--muted);margin-bottom:14px}
.stat-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:12px}
.stat-box{background:var(--bg);border:1px solid var(--border2);border-radius:var(--radius);padding:14px}
.stat-label{font-size:11px;color:var(--muted);margin-bottom:4px}
.stat-value{font-size:18px;font-weight:700;color:var(--text)}
.stat-value.green{color:var(--green)}
.stat-value.blue{color:var(--blue)}
.stat-value.yellow{color:var(--yellow)}
.stat-value.purple{color:var(--purple)}

/* Tables */
.table{width:100%;border-collapse:collapse}
.table th{text-align:left;font-size:11px;font-weight:600;text-transform:uppercase;letter-spacing:.06em;color:var(--muted);padding:0 12px 10px;border-bottom:1px solid var(--border2)}
.table td{padding:11px 12px;border-bottom:1px solid var(--border2);font-size:13px;vertical-align:middle}
.table tr:last-child td{border-bottom:none}
.table tr:hover td{background:#1a1f27}
.badge{display:inline-flex;align-items:center;padding:2px 8px;border-radius:4px;font-size:11px;font-weight:600}
.badge.active{background:#0d2818;color:var(--green);border:1px solid #1a3a25}
.badge.inactive{background:#1a1a1a;color:var(--muted);border:1px solid var(--border2)}

/* Forms */
.form-row{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:14px;margin-bottom:14px}
.form-group{display:flex;flex-direction:column;gap:5px}
.form-label{font-size:12px;font-weight:500;color:var(--muted)}
.form-input,.form-select{background:var(--bg);border:1px solid var(--border);border-radius:var(--radius);color:var(--text);font-size:13px;padding:8px 11px;font-family:var(--font);transition:border .15s;width:100%}
.form-input:focus,.form-select:focus{outline:none;border-color:var(--blue)}
.form-input::placeholder{color:#4d5566}
.form-hint{font-size:11px;color:var(--muted);margin-top:2px}

/* Buttons */
.btn{display:inline-flex;align-items:center;gap:7px;padding:7px 14px;border-radius:var(--radius);font-size:13px;font-weight:600;cursor:pointer;border:none;transition:all .15s;font-family:var(--font)}
.btn:disabled{opacity:.5;cursor:not-allowed}
.btn-primary{background:#238636;color:#fff}
.btn-primary:hover:not(:disabled){background:#2ea043}
.btn-danger{background:#b62324;color:#fff}
.btn-danger:hover:not(:disabled){background:#d12a2b}
.btn-secondary{background:var(--border2);color:var(--text);border:1px solid var(--border)}
.btn-secondary:hover:not(:disabled){background:var(--border)}
.btn-ghost{background:transparent;color:var(--muted);border:1px solid var(--border2)}
.btn-ghost:hover:not(:disabled){border-color:var(--border);color:var(--text)}
.btn-sm{padding:5px 10px;font-size:12px}
.btn-group{display:flex;gap:8px;flex-wrap:wrap}

/* Log viewer */
.log-wrap{background:#0d1117;border:1px solid var(--border2);border-radius:var(--radius);padding:14px;font-family:'JetBrains Mono','Fira Code',monospace;font-size:12px;line-height:1.7;color:#c9d1d9;max-height:450px;overflow-y:auto;white-space:pre-wrap;word-break:break-all}
.log-wrap::-webkit-scrollbar{width:5px}
.log-wrap::-webkit-scrollbar-thumb{background:var(--border);border-radius:4px}

/* Token display */
.token-box{font-family:monospace;font-size:13px;background:var(--bg);border:1px solid var(--border2);border-radius:var(--radius);padding:10px 14px;color:var(--green);word-break:break-all;letter-spacing:.02em}

/* Alerts */
.alert{padding:10px 14px;border-radius:var(--radius);font-size:13px;margin-bottom:12px;display:none}
.alert.show{display:flex;align-items:center;gap:8px}
.alert-success{background:#0d2818;color:var(--green);border:1px solid #1a3a25}
.alert-error{background:#2d1012;color:var(--red);border:1px solid #3d1a1c}
.alert-info{background:#0d1b2e;color:var(--blue);border:1px solid #1a2e45}

/* Spinner */
.spin{animation:spin .7s linear infinite;display:inline-block}
@keyframes spin{to{transform:rotate(360deg)}}

/* Misc */
.mono{font-family:monospace}
.muted{color:var(--muted)}
.flex{display:flex}
.items-center{align-items:center}
.gap-8{gap:8px}
.gap-12{gap:12px}
.mb-4{margin-bottom:4px}
.mb-12{margin-bottom:12px}
.mb-16{margin-bottom:16px}
.section-header{display:flex;align-items:center;justify-content:space-between;margin-bottom:16px}
.section-title{font-size:16px;font-weight:700}
.divider{height:1px;background:var(--border2);margin:20px 0}
.copy-btn{background:none;border:none;color:var(--muted);cursor:pointer;padding:2px 6px;border-radius:4px;font-size:11px;font-family:var(--font)}
.copy-btn:hover{color:var(--text);background:var(--border2)}

/* Responsive */
@media(max-width:768px){
  .sidebar{width:100%;height:auto;position:relative;flex-direction:row;flex-wrap:wrap}
  .main{margin-left:0}
  .nav{display:flex;flex-wrap:wrap;padding:6px}
  .nav-section{display:none}
  .nav-item{flex:0 0 auto}
  .content{padding:16px}
}
</style>
</head>
<body>
<div class="shell">

<!-- Sidebar -->
<nav class="sidebar">
  <div class="brand">
    <div class="brand-name">Bik AI</div>
    <div class="brand-sub">Control Panel</div>
  </div>
  <div class="nav">
    <div class="nav-section">Overview</div>
    <button class="nav-item active" onclick="nav('dashboard')" id="nav-dashboard">
      <svg viewBox="0 0 16 16" fill="currentColor"><path d="M1.5 2a.5.5 0 0 0-.5.5v4a.5.5 0 0 0 .5.5h4a.5.5 0 0 0 .5-.5v-4a.5.5 0 0 0-.5-.5h-4zm0 7a.5.5 0 0 0-.5.5v4a.5.5 0 0 0 .5.5h4a.5.5 0 0 0 .5-.5v-4a.5.5 0 0 0-.5-.5h-4zm7-7a.5.5 0 0 0-.5.5v4a.5.5 0 0 0 .5.5h4a.5.5 0 0 0 .5-.5v-4a.5.5 0 0 0-.5-.5h-4zm0 7a.5.5 0 0 0-.5.5v4a.5.5 0 0 0 .5.5h4a.5.5 0 0 0 .5-.5v-4a.5.5 0 0 0-.5-.5h-4z"/></svg>
      Dashboard
    </button>
    <div class="nav-section">Server</div>
    <button class="nav-item" onclick="nav('server')" id="nav-server">
      <svg viewBox="0 0 16 16" fill="currentColor"><path d="M1 2a1 1 0 0 1 1-1h11a1 1 0 0 1 1 1v2a1 1 0 0 1-1 1H2a1 1 0 0 1-1-1V2zm0 5a1 1 0 0 1 1-1h11a1 1 0 0 1 1 1v2a1 1 0 0 1-1 1H2a1 1 0 0 1-1-1V7zm0 5a1 1 0 0 1 1-1h11a1 1 0 0 1 1 1v2a1 1 0 0 1-1 1H2a1 1 0 0 1-1-1v-2z"/></svg>
      Server Control
    </button>
    <button class="nav-item" onclick="nav('logs')" id="nav-logs">
      <svg viewBox="0 0 16 16" fill="currentColor"><path d="M3 0h10a2 2 0 0 1 2 2v12a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2v-1h1v1a1 1 0 0 0 1 1h10a1 1 0 0 0 1-1V2a1 1 0 0 0-1-1H3a1 1 0 0 0-1 1v1H1V2a2 2 0 0 1 2-2z"/><path d="M1 5v-.5a.5.5 0 0 1 1 0V5h.5a.5.5 0 0 1 0 1h-2a.5.5 0 0 1 0-1H1zm0 3v-.5a.5.5 0 0 1 1 0V8h.5a.5.5 0 0 1 0 1h-2a.5.5 0 0 1 0-1H1zm0 3v-.5a.5.5 0 0 1 1 0v.5h.5a.5.5 0 0 1 0 1h-2a.5.5 0 0 1 0-1H1z"/></svg>
      Logs
    </button>
    <div class="nav-section">Models</div>
    <button class="nav-item" onclick="nav('models')" id="nav-models">
      <svg viewBox="0 0 16 16" fill="currentColor"><path d="M8 1a2 2 0 0 1 2 2v4H6V3a2 2 0 0 1 2-2zm3 6V3a3 3 0 0 0-6 0v4a2 2 0 0 0-2 2v5a2 2 0 0 0 2 2h6a2 2 0 0 0 2-2V9a2 2 0 0 0-2-2z"/></svg>
      Models
    </button>
    <button class="nav-item" onclick="nav('download')" id="nav-download">
      <svg viewBox="0 0 16 16" fill="currentColor"><path d="M.5 9.9a.5.5 0 0 1 .5.5v2.5a1 1 0 0 0 1 1h12a1 1 0 0 0 1-1v-2.5a.5.5 0 0 1 1 0v2.5a2 2 0 0 1-2 2H2a2 2 0 0 1-2-2v-2.5a.5.5 0 0 1 .5-.5z"/><path d="M7.646 11.854a.5.5 0 0 0 .708 0l3-3a.5.5 0 0 0-.708-.708L8.5 10.293V1.5a.5.5 0 0 0-1 0v8.793L5.354 8.146a.5.5 0 1 0-.708.708l3 3z"/></svg>
      Download Model
    </button>
    <div class="nav-section">Config</div>
    <button class="nav-item" onclick="nav('nginx')" id="nav-nginx">
      <svg viewBox="0 0 16 16" fill="currentColor"><path d="M0 8a8 8 0 1 0 16 0A8 8 0 0 0 0 8zm7.5-6.923c-.67.204-1.335.82-1.887 1.855-.143.268-.276.56-.395.872.705.157 1.472.257 2.282.287V1.077zM4.249 3.539c.142-.384.304-.744.481-1.078a6.7 6.7 0 0 1 .597-.933A7.01 7.01 0 0 0 3.051 3.05c.362.184.763.349 1.198.49zM3.509 7.5c.036-1.07.188-2.087.436-3.008a9.124 9.124 0 0 1-1.565-.667A6.964 6.964 0 0 0 1.018 7.5h2.49zm1.4-2.741a12.344 12.344 0 0 0-.4 2.741H7.5V5.091c-.91-.03-1.783-.145-2.591-.332zM8.5 5.09V7.5h2.99a12.342 12.342 0 0 0-.399-2.741c-.808.187-1.681.301-2.591.332zM4.51 8.5c.035.987.176 1.914.399 2.741A13.612 13.612 0 0 1 7.5 10.91V8.5H4.51zm3.99 0v2.409c.91.03 1.783.145 2.591.332.223-.827.364-1.754.4-2.741H8.5zm-3.282 3.696c.12.312.252.604.395.872.552 1.035 1.218 1.65 1.887 1.855V11.91c-.81.03-1.577.13-2.282.287zm.11.233c-.31.322-.59.48-.676.48a.56.56 0 0 1-.249-.064c.135-.402.338-.78.554-1.15.116.207.238.408.37.604zm4.26.48c-.087 0-.366-.158-.677-.48.132-.196.254-.397.37-.604.217.37.42.748.555 1.15a.56.56 0 0 1-.248.065zM7.5 14.923c.67-.204 1.335-.82 1.887-1.855.143-.268.276-.56.395-.872A12.63 12.63 0 0 0 7.5 11.91v3.013zm2.282-.287c-.705-.157-1.472-.257-2.282-.287V11.91c.91.03 1.783.145 2.591.332-.224.828-.364 1.754-.4 2.741zM11.507 7.5c.036-1.07.188-2.087.436-3.008a9.124 9.124 0 0 0 1.565-.667A6.964 6.964 0 0 1 14.982 7.5h-3.475zm.492 1c-.036.987-.177 1.914-.4 2.741a9.13 9.13 0 0 0 1.565.667 6.963 6.963 0 0 0 1.507-3.408h-2.672z"/></svg>
      Nginx / Domain
    </button>
    <button class="nav-item" onclick="nav('token')" id="nav-token">
      <svg viewBox="0 0 16 16" fill="currentColor"><path d="M8 1a2 2 0 0 1 2 2v4H6V3a2 2 0 0 1 2-2zm3 6V3a3 3 0 0 0-6 0v4a2 2 0 0 0-2 2v5a2 2 0 0 0 2 2h6a2 2 0 0 0 2-2V9a2 2 0 0 0-2-2z"/></svg>
      API Token
    </button>
  </div>
  <div class="sidebar-footer">bikiran.com</div>
</nav>

<!-- Main -->
<div class="main">
  <header class="topbar">
    <span class="topbar-title" id="topbar-title">Dashboard</span>
    <span class="status-pill stopped" id="status-pill"><span class="dot"></span><span id="status-text">Loading…</span></span>
    <a href="/server/info" target="_blank" class="btn btn-ghost btn-sm">Info page ↗</a>
  </header>

  <div class="content">

    <!-- DASHBOARD -->
    <div class="panel active" id="panel-dashboard">
      <div id="dash-alert" class="alert"></div>
      <div class="stat-grid mb-16" id="dash-stats">
        <div class="stat-box"><div class="stat-label">Status</div><div class="stat-value" id="st-status">—</div></div>
        <div class="stat-box"><div class="stat-label">Uptime</div><div class="stat-value green" id="st-uptime">—</div></div>
        <div class="stat-box"><div class="stat-label">Model</div><div class="stat-value blue" id="st-model" style="font-size:13px;word-break:break-all">—</div></div>
        <div class="stat-box"><div class="stat-label">Parallel Slots</div><div class="stat-value yellow" id="st-parallel">—</div></div>
        <div class="stat-box"><div class="stat-label">Context Window</div><div class="stat-value" id="st-ctx">—</div></div>
        <div class="stat-box"><div class="stat-label">CPU Threads</div><div class="stat-value" id="st-threads">—</div></div>
        <div class="stat-box"><div class="stat-label">Memory Usage</div><div class="stat-value purple" id="st-mem">—</div></div>
        <div class="stat-box"><div class="stat-label">Rate Limit</div><div class="stat-value" id="st-rate">—</div></div>
      </div>
      <div class="card">
        <div class="card-title">Quick Actions</div>
        <div class="btn-group">
          <button class="btn btn-danger" onclick="stopServer()" id="btn-stop">Stop Server</button>
          <button class="btn btn-primary" onclick="restartServer()" id="btn-restart">Restart Server</button>
          <button class="btn btn-secondary" onclick="refreshStatus()">↻ Refresh</button>
          <button class="btn btn-ghost btn-sm" onclick="nav('logs')">View Logs →</button>
        </div>
      </div>
      <div class="card">
        <div class="card-title">Endpoints</div>
        <table class="table">
          <thead><tr><th>Method</th><th>Path</th><th>Auth</th><th>Description</th></tr></thead>
          <tbody>
            <tr><td><span class="badge inactive">GET</span></td><td class="mono">/health</td><td class="muted">No</td><td>Health check</td></tr>
            <tr><td><span class="badge inactive">GET</span></td><td class="mono">/server/info</td><td class="muted">No</td><td>Info page</td></tr>
            <tr><td><span class="badge inactive">GET</span></td><td class="mono">/ui</td><td class="muted">No</td><td>This control panel</td></tr>
            <tr><td><span class="badge inactive">GET</span></td><td class="mono">/v1/models</td><td class="muted">Yes</td><td>List model (OpenAI-compatible)</td></tr>
            <tr><td><span class="badge active">POST</span></td><td class="mono">/v1/chat/completions</td><td class="muted">Yes</td><td>Chat completions — supports streaming</td></tr>
            <tr><td><span class="badge active">POST</span></td><td class="mono">/generate</td><td class="muted">Yes</td><td>Simple text generation</td></tr>
            <tr><td><span class="badge inactive">GET</span></td><td class="mono">/docs</td><td class="muted">No</td><td>Swagger UI</td></tr>
          </tbody>
        </table>
      </div>
    </div>

    <!-- SERVER CONTROL -->
    <div class="panel" id="panel-server">
      <div id="server-alert" class="alert"></div>
      <div class="card">
        <div class="card-title">Start / Reconfigure Server</div>
        <div class="form-row">
          <div class="form-group" style="grid-column:1/-1">
            <label class="form-label">Model</label>
            <select class="form-select" id="srv-model"></select>
            <div class="form-hint">Select a downloaded model. Download more in the Models section.</div>
          </div>
        </div>
        <div class="form-row">
          <div class="form-group">
            <label class="form-label">Parallel Slots</label>
            <input type="number" class="form-input" id="srv-parallel" value="4" min="1" max="32"/>
            <div class="form-hint">Concurrent requests. More = more RAM.</div>
          </div>
          <div class="form-group">
            <label class="form-label">Port</label>
            <input type="number" class="form-input" id="srv-port" value="8000" min="1024" max="65535"/>
          </div>
          <div class="form-group">
            <label class="form-label">Context Length</label>
            <input type="number" class="form-input" id="srv-ctx" value="4096" min="512" max="8192"/>
            <div class="form-hint">Tokens per slot. Max 8192 for Gemma.</div>
          </div>
          <div class="form-group">
            <label class="form-label">CPU Threads</label>
            <input type="number" class="form-input" id="srv-threads" value="4" min="1" max="128"/>
          </div>
        </div>
        <div class="btn-group">
          <button class="btn btn-primary" onclick="startServer()" id="btn-start">Start Server</button>
          <button class="btn btn-danger" onclick="stopServer()">Stop Server</button>
          <button class="btn btn-secondary" onclick="restartServer()">Restart</button>
        </div>
      </div>
    </div>

    <!-- LOGS -->
    <div class="panel" id="panel-logs">
      <div class="section-header">
        <span class="section-title">Server Logs</span>
        <div class="btn-group">
          <select class="form-select" id="log-lines" style="width:auto;padding:5px 10px">
            <option value="50">Last 50 lines</option>
            <option value="100" selected>Last 100 lines</option>
            <option value="200">Last 200 lines</option>
            <option value="500">Last 500 lines</option>
          </select>
          <button class="btn btn-secondary btn-sm" onclick="loadLogs()">Refresh</button>
          <button class="btn btn-ghost btn-sm" id="btn-autoscroll" onclick="toggleAutoScroll()">Auto-scroll: ON</button>
        </div>
      </div>
      <div class="log-wrap" id="log-output">Loading logs…</div>
    </div>

    <!-- MODELS -->
    <div class="panel" id="panel-models">
      <div id="models-alert" class="alert"></div>
      <div class="card">
        <div class="section-header" style="margin-bottom:0">
          <span class="card-title" style="margin-bottom:0">Downloaded Models</span>
          <button class="btn btn-ghost btn-sm" onclick="loadModels()">↻ Refresh</button>
        </div>
        <div class="divider" style="margin:12px 0"></div>
        <table class="table">
          <thead><tr><th>Name</th><th>Size</th><th>Status</th><th>Action</th></tr></thead>
          <tbody id="models-tbody"><tr><td colspan="4" class="muted">Loading…</td></tr></tbody>
        </table>
      </div>
    </div>

    <!-- DOWNLOAD -->
    <div class="panel" id="panel-download">
      <div id="dl-alert" class="alert"></div>
      <div class="card">
        <div class="card-title">Download via Google Drive</div>
        <div class="form-row">
          <div class="form-group" style="grid-column:1/-1">
            <label class="form-label">Google Drive File ID</label>
            <input class="form-input" id="dl-gdrive" placeholder="e.g. 1aBcDeFgHiJkLmNoPqRsTuV"/>
            <div class="form-hint">From the share URL: drive.google.com/file/d/<strong>FILE_ID</strong>/view</div>
          </div>
        </div>
        <button class="btn btn-primary" onclick="downloadGdrive()" id="btn-dl-gdrive">Download</button>
      </div>
      <div class="card">
        <div class="card-title">Download from HuggingFace</div>
        <div class="form-row">
          <div class="form-group">
            <label class="form-label">Repository</label>
            <input class="form-input" id="dl-hf-repo" placeholder="bartowski/gemma-3-4b-it-GGUF"/>
          </div>
          <div class="form-group">
            <label class="form-label">Filename</label>
            <input class="form-input" id="dl-hf-file" placeholder="gemma-3-4b-it-Q4_K_M.gguf"/>
          </div>
        </div>
        <button class="btn btn-primary" onclick="downloadHF()" id="btn-dl-hf">Download</button>
      </div>
      <div class="card">
        <div class="card-title">Download from Direct URL</div>
        <div class="form-row">
          <div class="form-group" style="grid-column:1/-1">
            <label class="form-label">URL</label>
            <input class="form-input" id="dl-url" placeholder="https://your-storage.com/model.gguf"/>
          </div>
        </div>
        <button class="btn btn-primary" onclick="downloadURL()" id="btn-dl-url">Download</button>
      </div>
      <div id="dl-progress" style="display:none" class="card">
        <div class="card-title">Download Progress</div>
        <div class="log-wrap" id="dl-log" style="max-height:200px">Starting download…</div>
      </div>
    </div>

    <!-- NGINX -->
    <div class="panel" id="panel-nginx">
      <div id="nginx-alert" class="alert"></div>
      <div class="card">
        <div class="card-title">Configure Nginx Reverse Proxy</div>
        <p class="muted mb-12" style="font-size:13px">Expose the server on port 80 via nginx. If you have a domain, point its A record to this server's IP first.</p>
        <div class="form-row">
          <div class="form-group">
            <label class="form-label">Domain / IP</label>
            <input class="form-input" id="ng-domain" placeholder="Leave blank to auto-detect public IP"/>
            <div class="form-hint">e.g. api.yourdomain.com — leave blank to use your public IP</div>
          </div>
          <div class="form-group">
            <label class="form-label">Backend Port</label>
            <input type="number" class="form-input" id="ng-port" value="8000"/>
          </div>
        </div>
        <div class="form-row">
          <div class="form-group">
            <label class="form-label" style="display:flex;align-items:center;gap:8px;cursor:pointer">
              <input type="checkbox" id="ng-ssl" style="accent-color:var(--blue)"/> Enable HTTPS (Let's Encrypt)
            </label>
            <div class="form-hint">Requires a real domain name pointed at this server. Port 80 must be open.</div>
          </div>
        </div>
        <button class="btn btn-primary" onclick="configNginx()" id="btn-nginx">Apply Configuration</button>
      </div>
      <div class="card">
        <div class="card-title">DNS Setup (Custom Domain)</div>
        <p class="muted mb-12" style="font-size:13px">Add this A record in your DNS provider:</p>
        <table class="table">
          <thead><tr><th>Type</th><th>Name</th><th>Value</th><th>TTL</th></tr></thead>
          <tbody>
            <tr>
              <td><span class="badge inactive">A</span></td>
              <td class="mono">api  <span class="muted">(or @)</span></td>
              <td class="mono" id="ng-ip-display">—</td>
              <td class="muted">Auto</td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>

    <!-- TOKEN -->
    <div class="panel" id="panel-token">
      <div id="token-alert" class="alert"></div>
      <div class="card">
        <div class="card-title">Current API Key</div>
        <div class="token-box" id="token-display">Enter your API key below to authenticate and view the token.</div>
        <div style="margin-top:10px;display:flex;gap:8px">
          <button class="btn btn-secondary btn-sm" onclick="copyToken()">Copy</button>
          <button class="btn btn-ghost btn-sm" onclick="loadToken()">↻ Refresh</button>
        </div>
      </div>
      <div class="card">
        <div class="card-title">Generate New Key</div>
        <p class="muted mb-12" style="font-size:13px">This will invalidate the current key. All existing integrations using the old key will stop working.</p>
        <button class="btn btn-danger" onclick="newToken()">Generate New API Key</button>
      </div>
      <div class="card">
        <div class="card-title">Usage</div>
        <div class="log-wrap" style="max-height:none">curl BASE_URL/v1/chat/completions \
  -H "X-API-Key: YOUR_KEY" \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Hello!"}],"stream":false}'</div>
      </div>
    </div>

  </div><!-- /content -->
</div><!-- /main -->
</div><!-- /shell -->

<script>
// ── Auth ──────────────────────────────────────────────────────────────────
let API_KEY = localStorage.getItem('bikai_key') || '';
const BASE = '';

function getAuthHeader(){return{'X-API-Key':API_KEY,'Content-Type':'application/json'};}

// Prompt for API key if not set
function ensureKey(){
  if(!API_KEY){
    const k = prompt('Enter your API key to continue:\n(Run: bikai token show  on the server)');
    if(k){API_KEY=k.trim();localStorage.setItem('bikai_key',API_KEY);}
  }
  return !!API_KEY;
}

// ── Navigation ─────────────────────────────────────────────────────────────
const PANELS={};
const titles={dashboard:'Dashboard',server:'Server Control',logs:'Logs',models:'Models',download:'Download Model',nginx:'Nginx / Domain',token:'API Token'};
function nav(id){
  document.querySelectorAll('.nav-item').forEach(el=>el.classList.remove('active'));
  document.querySelectorAll('.panel').forEach(el=>el.classList.remove('active'));
  const navEl=document.getElementById('nav-'+id);
  if(navEl)navEl.classList.add('active');
  const panel=document.getElementById('panel-'+id);
  if(panel)panel.classList.add('active');
  document.getElementById('topbar-title').textContent=titles[id]||id;
  if(id==='dashboard')refreshStatus();
  if(id==='logs')loadLogs();
  if(id==='models')loadModels();
  if(id==='server')loadServerForm();
  if(id==='nginx')loadNginxPanel();
  if(id==='token'&&API_KEY)loadToken();
}

// ── Alerts ─────────────────────────────────────────────────────────────────
function showAlert(id,msg,type='success'){
  const el=document.getElementById(id);
  if(!el)return;
  el.textContent=msg;
  el.className='alert '+type+' show';
  setTimeout(()=>el.classList.remove('show'),5000);
}

// ── Status ─────────────────────────────────────────────────────────────────
async function refreshStatus(){
  try{
    const r=await fetch(BASE+'/api/ui/status');
    if(!r.ok)throw new Error();
    const d=await r.json();
    const pill=document.getElementById('status-pill');
    const txt=document.getElementById('status-text');
    pill.className='status-pill '+(d.running?'running':'stopped');
    txt.textContent=d.running?'Running':'Stopped';
    set('st-status',d.running?'Running':'Stopped');
    set('st-uptime',d.uptime||'—');
    set('st-model',d.model_name||'—');
    set('st-parallel',d.parallel??'—');
    set('st-ctx',(d.ctx||0).toLocaleString()+' tokens');
    set('st-threads',d.threads??'—');
    set('st-mem',d.mem_mb?d.mem_mb.toLocaleString()+' MB':'—');
    set('st-rate',d.rate_limit||'—');
  }catch(e){
    const pill=document.getElementById('status-pill');
    if(pill)pill.className='status-pill stopped';
    document.getElementById('status-text').textContent='Offline';
  }
}
function set(id,val){const el=document.getElementById(id);if(el)el.textContent=val;}

// ── Server control ─────────────────────────────────────────────────────────
async function startServer(){
  if(!ensureKey())return;
  const model=document.getElementById('srv-model').value;
  if(!model){alert('Select a model first.');return;}
  document.getElementById('btn-start').disabled=true;
  document.getElementById('btn-start').innerHTML='<span class="spin">↻</span> Starting…';
  try{
    const r=await fetch(BASE+'/api/ui/start',{method:'POST',headers:getAuthHeader(),
      body:JSON.stringify({
        model,
        parallel:+document.getElementById('srv-parallel').value,
        port:+document.getElementById('srv-port').value,
        ctx:+document.getElementById('srv-ctx').value,
        threads:+document.getElementById('srv-threads').value,
        daemon:true
      })});
    const d=await r.json();
    if(r.ok){showAlert('server-alert','Server started (PID '+d.pid+')','alert-success');refreshStatus();}
    else showAlert('server-alert','Error: '+(d.detail||JSON.stringify(d)),'alert-error');
  }catch(e){showAlert('server-alert','Request failed','alert-error');}
  document.getElementById('btn-start').disabled=false;
  document.getElementById('btn-start').textContent='Start Server';
}

async function stopServer(){
  if(!ensureKey())return;
  if(!confirm('Stop the running server?'))return;
  await fetch(BASE+'/api/ui/stop',{method:'POST',headers:getAuthHeader()});
  setTimeout(()=>{refreshStatus();showAlert('dash-alert','Server stopped','alert-info');},1000);
}

async function restartServer(){
  if(!ensureKey())return;
  const r=await fetch(BASE+'/api/ui/restart',{method:'POST',headers:getAuthHeader()});
  const d=await r.json();
  if(r.ok){showAlert('dash-alert','Server restarted (PID '+d.pid+')','alert-success');refreshStatus();}
  else showAlert('dash-alert','Restart failed: '+(d.detail||'unknown error'),'alert-error');
}

async function loadServerForm(){
  const r=await fetch(BASE+'/api/ui/status');
  const d=await r.json();
  document.getElementById('srv-parallel').value=d.parallel||4;
  document.getElementById('srv-port').value=d.port||8000;
  document.getElementById('srv-ctx').value=d.ctx||4096;
  document.getElementById('srv-threads').value=d.threads||4;
  // Load model list
  const mr=await fetch(BASE+'/api/ui/models');
  const md=await mr.json();
  const sel=document.getElementById('srv-model');
  sel.innerHTML='';
  if(!md.models.length){sel.innerHTML='<option>No models downloaded</option>';return;}
  md.models.forEach(m=>{
    const opt=document.createElement('option');
    opt.value=m.path;opt.textContent=m.name+' ('+m.size+')';
    if(m.active)opt.selected=true;
    sel.appendChild(opt);
  });
}

// ── Logs ───────────────────────────────────────────────────────────────────
let autoScroll=true;
let logInterval=null;
function toggleAutoScroll(){
  autoScroll=!autoScroll;
  document.getElementById('btn-autoscroll').textContent='Auto-scroll: '+(autoScroll?'ON':'OFF');
}
async function loadLogs(){
  const lines=document.getElementById('log-lines').value;
  const r=await fetch(BASE+'/api/ui/logs?lines='+lines);
  const d=await r.json();
  const box=document.getElementById('log-output');
  box.textContent=d.lines.join('\n')||'(no logs yet)';
  if(autoScroll)box.scrollTop=box.scrollHeight;
}
// Auto-refresh logs every 3s when logs panel visible
setInterval(()=>{
  if(document.getElementById('panel-logs').classList.contains('active'))loadLogs();
},3000);

// ── Models ─────────────────────────────────────────────────────────────────
async function loadModels(){
  const r=await fetch(BASE+'/api/ui/models');
  const d=await r.json();
  const tbody=document.getElementById('models-tbody');
  if(!d.models.length){
    tbody.innerHTML='<tr><td colspan="4" class="muted" style="text-align:center;padding:20px">No models found. Download one below.</td></tr>';
    return;
  }
  tbody.innerHTML=d.models.map(m=>`
    <tr>
      <td class="mono" style="font-size:12px;word-break:break-all">${m.name}</td>
      <td>${m.size}</td>
      <td>${m.active?'<span class="badge active">Active</span>':'<span class="badge inactive">Inactive</span>'}</td>
      <td><button class="btn btn-ghost btn-sm" onclick="useModel('${m.path.replace(/'/g,"\\'")}','${m.name.replace(/'/g,"\\'")}')">Use this model</button></td>
    </tr>`).join('');
}

function useModel(path,name){
  nav('server');
  setTimeout(()=>{
    const sel=document.getElementById('srv-model');
    for(const opt of sel.options)if(opt.value===path){opt.selected=true;break;}
  },200);
}

// ── Download ───────────────────────────────────────────────────────────────
async function triggerDownload(body){
  if(!ensureKey())return;
  document.getElementById('dl-progress').style.display='block';
  const log=document.getElementById('dl-log');
  log.textContent='Initiating download…\nThis runs on the server — check logs for live progress.';
  try{
    const r=await fetch(BASE+'/api/ui/download',{method:'POST',headers:getAuthHeader(),body:JSON.stringify(body)});
    const d=await r.json();
    if(r.ok){
      log.textContent='Download started (background).\nCheck the Logs panel for progress.\n\n'+JSON.stringify(d,null,2);
      showAlert('dl-alert','Download started — check Logs for progress','alert-success');
    }else{
      log.textContent='Error: '+(d.detail||JSON.stringify(d));
      showAlert('dl-alert','Download failed: '+(d.detail||'unknown'),'alert-error');
    }
  }catch(e){log.textContent='Request failed: '+e;}
}
function downloadGdrive(){
  const id=document.getElementById('dl-gdrive').value.trim();
  if(!id){showAlert('dl-alert','Enter a Google Drive file ID','alert-error');return;}
  triggerDownload({type:'gdrive',id});
}
function downloadHF(){
  const repo=document.getElementById('dl-hf-repo').value.trim();
  const file=document.getElementById('dl-hf-file').value.trim();
  if(!repo||!file){showAlert('dl-alert','Enter both repo and filename','alert-error');return;}
  triggerDownload({type:'huggingface',repo,file});
}
function downloadURL(){
  const url=document.getElementById('dl-url').value.trim();
  if(!url){showAlert('dl-alert','Enter a URL','alert-error');return;}
  triggerDownload({type:'url',url});
}

// ── Nginx ──────────────────────────────────────────────────────────────────
async function loadNginxPanel(){
  try{
    const r=await fetch('https://api.ipify.org');
    const ip=await r.text();
    set('ng-ip-display',ip.trim());
    if(!document.getElementById('ng-domain').value)
      document.getElementById('ng-domain').placeholder=ip.trim()+' (auto-detected)';
  }catch(e){}
}
async function configNginx(){
  if(!ensureKey())return;
  const btn=document.getElementById('btn-nginx');
  btn.disabled=true;btn.innerHTML='<span class="spin">↻</span> Applying…';
  try{
    const r=await fetch(BASE+'/api/ui/nginx',{method:'POST',headers:getAuthHeader(),
      body:JSON.stringify({
        domain:document.getElementById('ng-domain').value.trim(),
        port:+document.getElementById('ng-port').value||8000,
        ssl:document.getElementById('ng-ssl').checked
      })});
    const d=await r.json();
    if(r.ok)showAlert('nginx-alert','nginx configured. URL: '+d.url,'alert-success');
    else showAlert('nginx-alert','Error: '+(d.detail||JSON.stringify(d)),'alert-error');
  }catch(e){showAlert('nginx-alert','Request failed','alert-error');}
  btn.disabled=false;btn.textContent='Apply Configuration';
}

// ── Token ──────────────────────────────────────────────────────────────────
async function loadToken(){
  if(!ensureKey())return;
  const r=await fetch(BASE+'/api/ui/token',{headers:getAuthHeader()});
  if(r.ok){const d=await r.json();document.getElementById('token-display').textContent=d.key||'(not set)';}
  else document.getElementById('token-display').textContent='Authentication failed — check your key.';
}
function copyToken(){
  const t=document.getElementById('token-display').textContent;
  navigator.clipboard.writeText(t).then(()=>showAlert('token-alert','Copied to clipboard','alert-success'));
}
async function newToken(){
  if(!ensureKey())return;
  if(!confirm('Generate a new API key? The old key will stop working immediately.'))return;
  const r=await fetch(BASE+'/api/ui/token/new',{method:'POST',headers:getAuthHeader()});
  const d=await r.json();
  if(r.ok){
    API_KEY=d.key;localStorage.setItem('bikai_key',d.key);
    document.getElementById('token-display').textContent=d.key;
    showAlert('token-alert','New key generated and saved','alert-success');
  }else showAlert('token-alert','Failed: '+(d.detail||'unknown'),'alert-error');
}

// ── Init ───────────────────────────────────────────────────────────────────
refreshStatus();
setInterval(refreshStatus,10000);
loadNginxPanel();
</script>
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
