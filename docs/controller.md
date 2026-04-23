# controller.py — Management Server

## Overview

`controller.py` is the **always-running management server** on port 8001.  
It serves the React UI, exposes a REST API for controlling the AI server, streams real-time metrics, and manages model downloads.

It is started once and **never restarted** unless you explicitly do so. The AI inference server (`server.py`) is spawned and killed by the controller.

---

## Startup

```bash
python controller.py               # default port 8001
python controller.py --port 9001   # custom port
```

On startup (`__main__`):
1. Calls `_ensure_ui_built()` — builds React UI if `ui/dist/` is missing
2. Auto-generates `API_KEY` in `.env` if not set
3. Writes its own PID to `.bikai-ctrl.pid`
4. Writes `CONTROLLER_PORT` to `.env`
5. Starts uvicorn

---

## Key Constants

```python
BASE_DIR = Path(__file__).parent       # project root
ENV_FILE = BASE_DIR / ".env"
PID_FILE = BASE_DIR / ".bikai.pid"            # AI server PID
CTRL_PID_FILE = BASE_DIR / ".bikai-ctrl.pid"  # Controller PID
DOWNLOAD_PID_FILE = BASE_DIR / ".bikai-dl.pid"
LOG_FILE = BASE_DIR / "bikai-server.log"
DOWNLOAD_LOG_FILE = BASE_DIR / "bikai-download.log"
MODELS_DIR = BASE_DIR / "models"
UI_DIST = BASE_DIR / "ui" / "dist"
```

---

## All API Endpoints

### Public (no API key needed)

| Method | Path | Description |
|---|---|---|
| GET | `/health` | Always returns `{"status":"ok"}` |
| GET | `/controller/health` | Same — used by cli.py health check |
| GET | `/api/controller/status` | Full server status, config, nginx state |
| GET | `/api/controller/models` | List all `.gguf` files in `models/` |
| GET | `/api/controller/metrics` | SSE stream of CPU/RAM/disk (live) |
| GET | `/api/controller/download/status` | Active download state + last 30 log lines |
| GET | `/api/controller/logs` | Last N lines of `bikai-server.log` |
| GET | `/controller/ui` | Serves React SPA `index.html` |
| GET | `/controller/ui/{path}` | Same — all sub-routes serve `index.html` |
| GET | `/controller/assets/*` | Static files (JS/CSS) for React UI |

### Protected (require `X-API-Key` header)

| Method | Path | Description |
|---|---|---|
| POST | `/api/controller/start` | Start/reconfigure AI server |
| POST | `/api/controller/stop` | Stop AI server |
| POST | `/api/controller/restart` | Restart with current `.env` config |
| POST | `/api/controller/download` | Start model download in background |
| DELETE | `/api/controller/models/{name}` | Delete a model file |
| GET | `/api/controller/token` | Show current API key |
| POST | `/api/controller/token/new` | Rotate API key |
| GET | `/api/controller/nginx` | Get nginx config + status |
| GET | `/api/controller/nginx/status` | Detailed nginx service status |
| POST | `/api/controller/nginx` | Write nginx config and reload |

---

## `/api/controller/status` Response Shape

```json
{
  "running": true,
  "pid": 12345,
  "uptime": "2h 15m 3s",
  "model_name": "gemma3-4b.gguf",
  "model_path": "/home/user/.bikai/models/gemma3-4b.gguf",
  "model_size": "2.32 GB",
  "parallel": 4,
  "ctx": 4096,
  "threads": 8,
  "rate_limit": "30/minute",
  "port": 8000,
  "controller_port": 8001,
  "domain": "api.example.com",
  "mem_mb": 512,
  "nginx": {
    "installed": true,
    "enabled": true,
    "active": true,
    "domain": "api.example.com"
  }
}
```

---

## `/api/controller/start` Request Body

```json
{
  "model": "gemma3-4b.gguf",
  "parallel": 4,
  "port": 8000,
  "ctx": 4096,
  "threads": 8
}
```

The controller:
1. Kills any running AI server (`_kill_ai_server()`)
2. Resolves model path (tries `models/<name>` and `models/<name>.gguf`)
3. Writes all config values to `.env`
4. Spawns `server.py` via `subprocess.Popen` with `start_new_session=True`
5. Writes AI server PID to `.bikai.pid`

---

## `/api/controller/metrics` — SSE Stream

Endpoint returns `text/event-stream`. Each event is a JSON object sent every ~1.5 seconds:

```json
{
  "cpu_pct": 23.4,
  "ram_total_mb": 16384,
  "ram_used_mb": 9200,
  "ram_free_mb": 7184,
  "ram_pct": 56.2,
  "ai_mem_mb": 3200,
  "disk_total_gb": 512.0,
  "disk_used_gb": 180.5,
  "disk_free_gb": 331.5,
  "disk_pct": 35.3
}
```

- **CPU**: calculated by reading `/proc/stat` twice with 0.5s gap (delta method)
- **RAM**: read from `/proc/meminfo` using `MemTotal` - `MemAvailable`
- **AI memory**: reads `VmRSS` from `/proc/<ai_pid>/status`
- **Disk**: uses `os.statvfs()` on the project directory

---

## `/api/controller/download` — Model Download

Request body:
```json
// Google Drive
{ "type": "gdrive", "id": "1kO_KTjQ...", "set_default": true }

// HuggingFace
{ "type": "huggingface", "repo": "bartowski/gemma-3-4b-it-GGUF", "file": "gemma-3-4b-it-Q4_K_M.gguf", "set_default": true }

// Direct URL
{ "type": "url", "url": "https://example.com/model.gguf", "set_default": true }
```

Downloads run as a background subprocess (stdout/stderr → `bikai-download.log`).  
Poll `/api/controller/download/status` to check progress.

---

## `/api/controller/models/{name}` DELETE

- Validates: no `/` or `..` in `model_name` (path traversal protection)
- Rejects if the model is currently active AND the AI server is running
- Clears `MODEL_PATH` in `.env` if the deleted model was the active one

---

## Nginx Management

The controller writes nginx config to `/etc/nginx/sites-available/bikai` using `sudo tee`.  
The config template proxies:
- `/` → port 8000 (AI server)
- `/controller` → port 8001 (controller)

All nginx write operations use `sudo` (user must have passwordless sudo for nginx commands, or it's set up during `setup.sh`).

---

## Auto-Build UI Logic

`_ensure_ui_built()` runs on every startup:

```python
if ui/dist/index.html exists → skip
if ui/package.json missing → warn and skip
if npm not found → warn and skip
if node_modules missing → run npm install
run npm run build
```

This ensures the UI is always available even on fresh clones or git pulls.

---

## Important Gotchas

1. **All `subprocess.run()` in async handlers must use `asyncio.to_thread()`** — otherwise it blocks the uvicorn event loop and ALL requests queue up (the "Loading..." hang bug)
2. **All daemon Popen calls must use `start_new_session=True`** — prevents SIGHUP suspension when the parent terminal closes
3. **`.env` is reloaded on every read** (`_read_env` calls `load_dotenv(override=True)`) — config changes take effect immediately without restart
