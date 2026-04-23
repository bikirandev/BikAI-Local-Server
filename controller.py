"""
Bik AI — Controller Server
===========================
A lightweight, always-running management server (default port 8001).
It is separate from the AI inference server so it survives restarts.

Routes:
  GET  /controller/ui          → React control-panel SPA
  GET  /api/controller/status  → AI server status (public)
  GET  /api/controller/models  → List downloaded models (public)
  POST /api/controller/start   → Start / reconfigure AI server
  POST /api/controller/stop    → Stop AI server
  POST /api/controller/restart → Restart AI server
  GET  /api/controller/logs    → Last N log lines
  GET  /api/controller/nginx   → Current nginx config
  POST /api/controller/nginx   → Write + reload nginx config
  GET  /api/controller/nginx/status → nginx service status
  GET  /api/controller/token   → Show API key
  POST /api/controller/token/new → Rotate API key
  POST /api/controller/download → Trigger model download

Usage:
  python controller.py               # port 8001
  python controller.py --port 9001   # custom port
"""

import argparse
import os
import re
import resource
import secrets
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

import uvicorn
from dotenv import load_dotenv, set_key
from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.security import APIKeyHeader
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).parent
ENV_FILE = BASE_DIR / ".env"
PID_FILE = BASE_DIR / ".bikai.pid"            # AI server PID
CTRL_PID_FILE = BASE_DIR / ".bikai-ctrl.pid"  # Controller PID
DOWNLOAD_PID_FILE = BASE_DIR / ".bikai-dl.pid"  # Download process PID
LOG_FILE = BASE_DIR / "bikai-server.log"
DOWNLOAD_LOG_FILE = BASE_DIR / "bikai-download.log"
MODELS_DIR = BASE_DIR / "models"
UI_DIST = BASE_DIR / "ui" / "dist"

# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------


def _load_env() -> None:
    load_dotenv(dotenv_path=str(ENV_FILE), override=True)


def _read_env(key: str, default: str = "") -> str:
    _load_env()
    return os.getenv(key, default)


def _write_env(key: str, value: str) -> None:
    ENV_FILE.touch(exist_ok=True)
    set_key(str(ENV_FILE), key, value)


def _read_pid() -> Optional[int]:
    if PID_FILE.exists():
        try:
            return int(PID_FILE.read_text().strip())
        except ValueError:
            pass
    return None


def _ai_is_running() -> bool:
    """Check if the AI server process is alive."""
    pid = _read_pid()
    if pid is None:
        return False
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        PID_FILE.unlink(missing_ok=True)
        return False


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=True)


async def require_api_key(key: str = Depends(_api_key_header)) -> str:
    stored = _read_env("API_KEY")
    if not stored:
        raise HTTPException(status_code=500, detail="API_KEY not configured.")
    if not secrets.compare_digest(key, stored):
        raise HTTPException(status_code=401, detail="Invalid API key.")
    return key


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Bik AI Controller",
    description="Management API for Bik AI Local Server",
    version="1.0.0",
    docs_url="/controller/docs",
    redoc_url=None,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    allow_credentials=False,
)

# ---------------------------------------------------------------------------
# Status
# ---------------------------------------------------------------------------


@app.get("/api/controller/status")
async def api_status():
    """Return current AI server status — public."""
    _load_env()
    running = _ai_is_running()
    pid = _read_pid()
    model_path_raw = _read_env("MODEL_PATH", "")
    model_path = Path(model_path_raw) if model_path_raw else None
    model_name = model_path.name if model_path else "—"
    model_size = "—"
    if model_path and model_path.is_file():
        try:
            model_size = f"{model_path.stat().st_size / 1_073_741_824:.2f} GB"
        except Exception:
            pass

    uptime_str = "—"
    if running and pid:
        try:
            # read process start time from /proc
            stat = Path(f"/proc/{pid}/stat").read_text().split()
            hz = os.sysconf("SC_CLK_TCK")
            boot = float(Path("/proc/uptime").read_text().split()[0])
            proc_start = float(stat[21]) / hz
            elapsed = int(boot - proc_start + time.time() - time.time())
            # simpler: use process creation time via stat
            start_ts = os.stat(f"/proc/{pid}").st_ctime
            elapsed = int(time.time() - start_ts)
            h, rem = divmod(elapsed, 3600)
            m, s = divmod(rem, 60)
            uptime_str = f"{h}h {m}m {s}s" if h else f"{m}m {s}s"
        except Exception:
            uptime_str = "running"

    try:
        mem_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024
    except Exception:
        mem_mb = 0

    # nginx status
    nginx_domain = _read_env("DOMAIN", "")
    nginx_conf = Path("/etc/nginx/sites-available/bikai")
    nginx_enabled = Path("/etc/nginx/sites-enabled/bikai").exists()
    nginx_active = False
    try:
        r = subprocess.run(["systemctl", "is-active", "nginx"], capture_output=True, text=True)
        nginx_active = r.stdout.strip() == "active"
    except Exception:
        pass

    return {
        "running": running,
        "pid": pid,
        "uptime": uptime_str,
        "model_name": model_name,
        "model_path": model_path_raw,
        "model_size": model_size,
        "parallel": int(_read_env("N_PARALLEL", "4")),
        "ctx": int(_read_env("N_CTX", "4096")),
        "threads": int(_read_env("N_THREADS", str(os.cpu_count() or 4))),
        "rate_limit": _read_env("RATE_LIMIT", "30/minute"),
        "port": int(_read_env("PORT", "8000")),
        "controller_port": int(_read_env("CONTROLLER_PORT", "8001")),
        "domain": nginx_domain,
        "mem_mb": mem_mb,
        "nginx": {
            "installed": nginx_conf.exists(),
            "enabled": nginx_enabled,
            "active": nginx_active,
            "domain": nginx_domain,
        },
    }


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


@app.get("/api/controller/models")
async def api_models():
    """List downloaded GGUF models — public."""
    _load_env()
    active = _read_env("MODEL_PATH", "")
    result = []
    if MODELS_DIR.exists():
        for f in sorted(MODELS_DIR.glob("**/*.gguf")):
            try:
                is_active = (
                    str(f) == active
                    or (active and str(f.resolve()) == str(Path(active).resolve()))
                )
                result.append({
                    "name": f.name,
                    "path": str(f),
                    "size": f"{f.stat().st_size / 1_073_741_824:.2f} GB",
                    "active": is_active,
                })
            except Exception:
                pass
    return {"models": result, "active_path": active}


# ---------------------------------------------------------------------------
# Start / Stop / Restart
# ---------------------------------------------------------------------------


class StartRequest(BaseModel):
    model: str
    parallel: int = 4
    port: int = 8000
    ctx: int = 4096
    threads: int = 4


def _kill_ai_server() -> None:
    """Terminate the AI server process gracefully."""
    pid = _read_pid()
    if pid:
        for sig in (signal.SIGTERM, signal.SIGKILL):
            try:
                os.kill(pid, sig)
                time.sleep(0.8)
            except ProcessLookupError:
                break
        PID_FILE.unlink(missing_ok=True)

    # Also kill anything still on the AI port
    port = _read_env("PORT", "8000")
    try:
        result = subprocess.run(["fuser", f"{port}/tcp"], capture_output=True, text=True)
        for p in result.stdout.strip().split():
            try:
                os.kill(int(p), signal.SIGTERM)
            except (ValueError, ProcessLookupError):
                pass
    except Exception:
        pass


@app.post("/api/controller/start", dependencies=[Depends(require_api_key)])
async def api_start(req: StartRequest):
    """Start (or reconfigure) the AI server. Controller stays alive."""
    _kill_ai_server()
    time.sleep(1)

    # Resolve model path
    model = req.model
    if not Path(model).is_file():
        candidates = [MODELS_DIR / model, MODELS_DIR / (model + ".gguf")]
        resolved = next((str(c) for c in candidates if c.is_file()), None)
        if not resolved:
            raise HTTPException(status_code=400, detail=f"Model not found: {model}")
        model = resolved

    # Save config to .env
    _write_env("MODEL_PATH", model)
    _write_env("N_PARALLEL", str(req.parallel))
    _write_env("PORT", str(req.port))
    _write_env("N_CTX", str(req.ctx))
    _write_env("N_THREADS", str(req.threads))

    cmd = [
        sys.executable,
        str(BASE_DIR / "server.py"),
        "--model", model,
        "--parallel", str(req.parallel),
        "--port", str(req.port),
        "--ctx", str(req.ctx),
        "--threads", str(req.threads),
    ]
    log_fh = LOG_FILE.open("w")
    proc = subprocess.Popen(cmd, stdout=log_fh, stderr=log_fh, cwd=str(BASE_DIR), start_new_session=True)
    PID_FILE.write_text(str(proc.pid))
    return {"ok": True, "pid": proc.pid, "model": model}


@app.post("/api/controller/stop", dependencies=[Depends(require_api_key)])
async def api_stop():
    """Stop the AI server."""
    _kill_ai_server()
    return {"ok": True}


@app.post("/api/controller/restart", dependencies=[Depends(require_api_key)])
async def api_restart():
    """Restart AI server with current .env config."""
    _load_env()
    model = _read_env("MODEL_PATH")
    if not model:
        raise HTTPException(status_code=400, detail="MODEL_PATH not set in .env")
    req = StartRequest(
        model=model,
        parallel=int(_read_env("N_PARALLEL", "4")),
        port=int(_read_env("PORT", "8000")),
        ctx=int(_read_env("N_CTX", "4096")),
        threads=int(_read_env("N_THREADS", str(os.cpu_count() or 4))),
    )
    return await api_start(req)


# ---------------------------------------------------------------------------
# Logs
# ---------------------------------------------------------------------------


@app.get("/api/controller/logs")
async def api_logs(lines: int = 200):
    """Return last N lines from the AI server log file."""
    if not LOG_FILE.exists():
        return {"lines": []}
    try:
        result = subprocess.run(
            ["tail", f"-n{lines}", str(LOG_FILE)],
            capture_output=True, text=True,
        )
        return {"lines": result.stdout.splitlines()}
    except Exception:
        return {"lines": []}


# ---------------------------------------------------------------------------
# Nginx
# ---------------------------------------------------------------------------

NGINX_TEMPLATE = """\
# Bik AI nginx configuration
# Generated by bikai controller — do not edit manually

# Main HTTP server block
server {{
    listen {listen_port};
    server_name {server_name};

    # Client request limits
    client_max_body_size {client_max_body_size};

    # Gzip compression
    {gzip_block}

    # ── AI inference API ──────────────────────────────────────
    location / {{
        proxy_pass http://127.0.0.1:{ai_port};
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # Streaming (SSE) support
        proxy_buffering off;
        proxy_cache off;
        proxy_read_timeout {read_timeout}s;
        chunked_transfer_encoding on;

        # CORS
        add_header 'Access-Control-Allow-Origin' '{cors_origin}' always;
        add_header 'Access-Control-Allow-Methods' 'GET, POST, OPTIONS' always;
        add_header 'Access-Control-Allow-Headers' 'X-API-Key, Content-Type, Authorization' always;
        add_header 'Access-Control-Max-Age' '86400' always;

        if ($request_method = OPTIONS) {{
            add_header 'Access-Control-Allow-Origin' '{cors_origin}';
            add_header 'Access-Control-Allow-Methods' 'GET, POST, OPTIONS';
            add_header 'Access-Control-Allow-Headers' 'X-API-Key, Content-Type, Authorization';
            add_header 'Access-Control-Max-Age' '86400';
            return 204;
        }}
    }}

    # ── Controller UI ─────────────────────────────────────────
    location /controller {{
        proxy_pass http://127.0.0.1:{ctrl_port};
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_read_timeout 60s;
    }}
}}
"""

GZIP_ON = """\
gzip on;
    gzip_types text/plain application/json application/javascript text/css;
    gzip_min_length 256;"""

GZIP_OFF = "# gzip off"


class NginxConfigRequest(BaseModel):
    domain: str = ""
    ai_port: int = 8000
    ctrl_port: int = 8001
    listen_port: int = 80
    ssl: bool = False
    cors_origin: str = "*"
    read_timeout: int = 300
    client_max_body_size: str = "10m"
    gzip: bool = True
    worker_processes: str = "auto"
    worker_connections: int = 1024


def _build_nginx_conf(req: NginxConfigRequest, domain: str) -> str:
    is_ip = bool(re.match(r"^\d{1,3}(\.\d{1,3}){3}$", domain))
    server_name = "_" if is_ip else domain
    gzip_block = GZIP_ON if req.gzip else GZIP_OFF
    return NGINX_TEMPLATE.format(
        listen_port=req.listen_port,
        server_name=server_name,
        ai_port=req.ai_port,
        ctrl_port=req.ctrl_port,
        cors_origin=req.cors_origin,
        read_timeout=req.read_timeout,
        client_max_body_size=req.client_max_body_size,
        gzip_block=gzip_block,
    )


def _get_nginx_main_conf() -> dict:
    """Read nginx.conf worker settings."""
    nginx_conf_path = Path("/etc/nginx/nginx.conf")
    result = {"worker_processes": "auto", "worker_connections": 1024}
    if nginx_conf_path.exists():
        text = nginx_conf_path.read_text()
        m = re.search(r"worker_processes\s+(\S+)\s*;", text)
        if m:
            result["worker_processes"] = m.group(1)
        m = re.search(r"worker_connections\s+(\d+)\s*;", text)
        if m:
            result["worker_connections"] = int(m.group(1))
    return result


@app.get("/api/controller/nginx")
async def api_nginx_get():
    """Return current nginx config and status."""
    conf_path = Path("/etc/nginx/sites-available/bikai")
    conf_text = ""
    if conf_path.exists():
        conf_text = conf_path.read_text()

    nginx_active = False
    try:
        r = subprocess.run(["systemctl", "is-active", "nginx"], capture_output=True, text=True)
        nginx_active = r.stdout.strip() == "active"
    except Exception:
        pass

    main = _get_nginx_main_conf()
    return {
        "installed": conf_path.exists(),
        "enabled": Path("/etc/nginx/sites-enabled/bikai").exists(),
        "active": nginx_active,
        "config_text": conf_text,
        "domain": _read_env("DOMAIN", ""),
        "worker_processes": main["worker_processes"],
        "worker_connections": main["worker_connections"],
    }


@app.get("/api/controller/nginx/status")
async def api_nginx_status():
    """Detailed nginx service status."""
    try:
        r = subprocess.run(
            ["sudo", "nginx", "-t"],
            capture_output=True, text=True,
        )
        config_valid = r.returncode == 0
        config_msg = (r.stderr or r.stdout).strip()
    except Exception:
        config_valid = False
        config_msg = "nginx not found"

    try:
        r2 = subprocess.run(
            ["systemctl", "status", "nginx", "--no-pager", "-l"],
            capture_output=True, text=True,
        )
        service_status = r2.stdout.strip()
    except Exception:
        service_status = ""

    return {"config_valid": config_valid, "config_message": config_msg, "service_status": service_status}


@app.post("/api/controller/nginx", dependencies=[Depends(require_api_key)])
async def api_nginx_apply(req: NginxConfigRequest):
    """Write nginx config and reload."""
    domain = req.domain.strip()
    if not domain:
        try:
            import urllib.request
            domain = urllib.request.urlopen("https://api.ipify.org", timeout=5).read().decode().strip()
        except Exception:
            raise HTTPException(status_code=500, detail="Could not detect public IP. Provide a domain.")

    conf = _build_nginx_conf(req, domain)

    # Write config via sudo tee
    proc = subprocess.run(
        ["sudo", "tee", "/etc/nginx/sites-available/bikai"],
        input=conf, text=True, capture_output=True,
    )
    if proc.returncode != 0:
        raise HTTPException(status_code=500, detail=f"Failed to write nginx config: {proc.stderr}")

    # Update worker_processes in nginx.conf if requested
    nginx_conf_path = Path("/etc/nginx/nginx.conf")
    if nginx_conf_path.exists():
        text = nginx_conf_path.read_text()
        text = re.sub(
            r"worker_processes\s+\S+\s*;",
            f"worker_processes {req.worker_processes};",
            text,
        )
        text = re.sub(
            r"worker_connections\s+\d+\s*;",
            f"worker_connections {req.worker_connections};",
            text,
        )
        proc2 = subprocess.run(
            ["sudo", "tee", "/etc/nginx/nginx.conf"],
            input=text, text=True, capture_output=True,
        )
        if proc2.returncode != 0:
            raise HTTPException(status_code=500, detail=f"Failed to update nginx.conf: {proc2.stderr}")

    # Enable site, remove default, reload
    subprocess.run(["sudo", "ln", "-sf",
                    "/etc/nginx/sites-available/bikai",
                    "/etc/nginx/sites-enabled/bikai"], check=False)
    subprocess.run(["sudo", "rm", "-f", "/etc/nginx/sites-enabled/default"], check=False)

    # Test config
    test = subprocess.run(["sudo", "nginx", "-t"], capture_output=True, text=True)
    if test.returncode != 0:
        raise HTTPException(status_code=500, detail=f"nginx config test failed:\n{test.stderr}")

    subprocess.run(["sudo", "systemctl", "enable", "nginx"], check=False)
    subprocess.run(["sudo", "systemctl", "reload", "nginx"], check=False)

    _write_env("DOMAIN", domain)

    # SSL via certbot
    if req.ssl and not re.match(r"^\d{1,3}(\.\d{1,3}){3}$", domain):
        certbot_result = subprocess.run(
            ["sudo", "certbot", "--nginx", "-d", domain,
             "--non-interactive", "--agree-tos", "--redirect"],
            capture_output=True, text=True,
        )
        if certbot_result.returncode != 0:
            return {
                "ok": True,
                "domain": domain,
                "ssl": False,
                "ssl_error": certbot_result.stderr[:500],
                "warning": "nginx configured but SSL cert failed. Check DNS and port 80.",
            }

    return {
        "ok": True,
        "domain": domain,
        "ssl": req.ssl,
        "url": f"{'https' if req.ssl else 'http'}://{domain}",
        "controller_url": f"{'https' if req.ssl else 'http'}://{domain}/controller/ui",
    }


# ---------------------------------------------------------------------------
# API Key
# ---------------------------------------------------------------------------


@app.get("/api/controller/token", dependencies=[Depends(require_api_key)])
async def api_token_show():
    return {"key": _read_env("API_KEY", "")}


@app.post("/api/controller/token/new", dependencies=[Depends(require_api_key)])
async def api_token_rotate():
    key = secrets.token_urlsafe(32)
    _write_env("API_KEY", key)
    return {"ok": True, "key": key}


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------


class DownloadRequest(BaseModel):
    type: str          # "gdrive" | "huggingface" | "url"
    id: str = ""       # gdrive file id
    repo: str = ""     # hf repo
    file: str = ""     # hf filename
    url: str = ""      # direct URL
    set_default: bool = True


@app.post("/api/controller/download", dependencies=[Depends(require_api_key)])
async def api_download(req: DownloadRequest):
    """Start a model download in background."""
    bikai_bin = None
    for candidate in [
        Path(sys.executable).parent / "bikai",
        Path.home() / ".local" / "bin" / "bikai",
        Path("/usr/local/bin/bikai"),
    ]:
        if candidate.exists():
            bikai_bin = str(candidate)
            break

    if req.type == "gdrive":
        if not req.id:
            raise HTTPException(status_code=400, detail="id required for gdrive")
        if bikai_bin:
            cmd = [bikai_bin, "download", "-g", req.id]
            if req.set_default:
                cmd.append("--set-default")
        else:
            # Fallback: use gdown directly
            MODELS_DIR.mkdir(parents=True, exist_ok=True)
            out = str(MODELS_DIR) + "/"
            cmd = [sys.executable, "-m", "gdown", req.id, "-O", out]
    elif req.type == "huggingface":
        if not req.repo or not req.file:
            raise HTTPException(status_code=400, detail="repo and file required")
        if bikai_bin:
            cmd = [bikai_bin, "download", "-r", req.repo, "-f", req.file]
            if req.set_default:
                cmd.append("--set-default")
        else:
            cmd = ["hf", "download", req.repo, req.file, "--local-dir", str(MODELS_DIR)]
    elif req.type == "url":
        if not req.url:
            raise HTTPException(status_code=400, detail="url required")
        if bikai_bin:
            cmd = [bikai_bin, "download", "-u", req.url]
            if req.set_default:
                cmd.append("--set-default")
        else:
            out_name = Path(req.url.split("?")[0]).name or "model.gguf"
            MODELS_DIR.mkdir(parents=True, exist_ok=True)
            cmd = ["wget", "-P", str(MODELS_DIR), req.url, "-O", str(MODELS_DIR / out_name)]
    else:
        raise HTTPException(status_code=400, detail=f"Unknown download type: {req.type}")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    dl_log_fh = DOWNLOAD_LOG_FILE.open("w")
    proc = subprocess.Popen(cmd, stdout=dl_log_fh, stderr=dl_log_fh, cwd=str(BASE_DIR), start_new_session=True)
    DOWNLOAD_PID_FILE.write_text(str(proc.pid))
    return {"ok": True, "message": "Download started."}


@app.get("/api/controller/download/status")
async def api_download_status():
    """Check if a download is running and return recent log lines."""
    active = False
    if DOWNLOAD_PID_FILE.exists():
        try:
            pid = int(DOWNLOAD_PID_FILE.read_text().strip())
            os.kill(pid, 0)   # raises if dead
            active = True
        except (ValueError, ProcessLookupError, OSError):
            DOWNLOAD_PID_FILE.unlink(missing_ok=True)

    lines: list[str] = []
    if DOWNLOAD_LOG_FILE.exists():
        try:
            result = subprocess.run(
                ["tail", "-n30", str(DOWNLOAD_LOG_FILE)],
                capture_output=True, text=True,
            )
            lines = result.stdout.splitlines()
        except Exception:
            pass

    return {"active": active, "lines": lines}


# ---------------------------------------------------------------------------
# Health check (always public)
# ---------------------------------------------------------------------------


@app.get("/health")
@app.get("/controller/health")
async def controller_health():
    return {"status": "ok", "service": "controller"}


# ---------------------------------------------------------------------------
# React SPA — serve built UI
# ---------------------------------------------------------------------------

if UI_DIST.exists():
    # Mount static assets
    app.mount("/controller/assets", StaticFiles(directory=str(UI_DIST / "assets")), name="assets")

    @app.get("/controller/ui")
    @app.get("/controller/ui/{path:path}")
    async def serve_ui(path: str = ""):
        index = UI_DIST / "index.html"
        if index.exists():
            return FileResponse(str(index))
        return JSONResponse(
            {"error": "UI not built. Run: cd ui && npm run build"},
            status_code=503,
        )
else:
    @app.get("/controller/ui")
    @app.get("/controller/ui/{path:path}")
    async def serve_ui_placeholder(path: str = ""):
        return JSONResponse(
            {
                "error": "UI not built yet.",
                "fix": "Run:  cd ui && npm run build",
                "controller_api": "/api/controller/status",
            },
            status_code=503,
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bik AI Controller Server")
    parser.add_argument("--port", type=int, default=int(os.getenv("CONTROLLER_PORT", "8001")))
    parser.add_argument("--host", default="0.0.0.0")
    args = parser.parse_args()

    # Auto-create .env with a generated API key if it doesn't exist or key is blank
    if not _read_env("API_KEY"):
        key = secrets.token_urlsafe(32)
        _write_env("API_KEY", key)
        print(f"[*] Generated API key: {key}")
        print(f"[*] Saved to {ENV_FILE}")

    # Write our own PID
    CTRL_PID_FILE.write_text(str(os.getpid()))
    _write_env("CONTROLLER_PORT", str(args.port))

    print(f"[*] Bik AI Controller starting on {args.host}:{args.port}")
    print(f"[*] UI: http://localhost:{args.port}/controller/ui")
    print(f"[*] API: http://localhost:{args.port}/api/controller/status")

    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
