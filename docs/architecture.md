# BikAI Local Server — Architecture Overview

## Purpose

BikAI is a self-hosted local LLM (Large Language Model) inference server that:
- Runs GGUF models locally via `llama-cpp-python` (CPU-only, no GPU required)
- Exposes an OpenAI-compatible REST API
- Provides a React web UI control panel to manage everything

---

## Two-Process Architecture

The system runs as **two separate processes**. This is the most important design decision.

```
┌──────────────────────────────────────────────────────────┐
│                        nginx (port 80)                   │
│   /            → proxy → AI Server (port 8000)           │
│   /controller  → proxy → Controller (port 8001)          │
└──────────────────────────────────────────────────────────┘
          │                          │
          ▼                          ▼
┌──────────────────┐      ┌──────────────────────────────┐
│   server.py      │      │   controller.py              │
│   port 8000      │      │   port 8001                  │
│                  │      │                              │
│  AI inference    │      │  Always running              │
│  llama-cpp       │      │  Management UI               │
│  /v1/chat/...    │      │  /api/controller/*           │
│  /generate       │      │  /controller/ui  (React SPA) │
│  /v1/models      │      │                              │
└──────────────────┘      └──────────────────────────────┘
       ▲  spawned/killed
       │  by controller
       └──────────────────────────────┘
```

### Why two processes?
- The **controller** (`controller.py`) must stay alive even when the AI server is restarting or crashed
- The **AI server** (`server.py`) loads the model and is killed/restarted on config changes
- The controller spawns/kills the AI server via `subprocess.Popen`

---

## File Structure

```
BikAI-Local-Server/
├── controller.py          # Management server (port 8001) — ALWAYS RUNNING
├── server.py              # AI inference server (port 8000) — started by controller
├── cli.py                 # bikai CLI entry point (click-based)
├── setup.sh               # One-liner installer script
├── requirements.txt       # Python dependencies
├── pyproject.toml         # Python packaging (makes `bikai` CLI installable)
│
├── ui/                    # React + TypeScript + Vite frontend
│   ├── src/
│   │   ├── App.tsx        # Root app — lock screen, sidebar, routing
│   │   ├── api.ts         # All HTTP calls to /api/controller/*
│   │   └── pages/
│   │       ├── Dashboard.tsx  # Start/Stop/Restart + config form
│   │       ├── Models.tsx     # Model list, download, delete
│   │       ├── Stats.tsx      # Real-time CPU/RAM/disk metrics (SSE)
│   │       ├── Nginx.tsx      # nginx config form
│   │       ├── Logs.tsx       # Live log viewer
│   │       └── Settings.tsx   # API key management
│   ├── dist/              # Built UI (gitignored — auto-built on controller start)
│   └── package.json
│
├── models/                # Downloaded GGUF model files (gitignored)
├── .env                   # Runtime config (gitignored)
├── .bikai.pid             # PID of running AI server
├── .bikai-ctrl.pid        # PID of running controller
├── .bikai-dl.pid          # PID of running download process
├── bikai-server.log       # AI server logs
├── bikai-download.log     # Model download logs
└── docs/                  # This documentation
```

---

## Data Flow: Typical Request

1. Client sends `POST /v1/chat/completions` with `X-API-Key` header
2. nginx proxies to port 8000 (AI server)
3. `server.py` validates API key → acquires a Llama instance from the pool
4. Runs inference in a thread pool executor (keeps event loop free)
5. Returns response (or streams SSE chunks)

## Data Flow: Controller API Request

1. Client (React UI or curl) sends `GET /api/controller/status`
2. nginx proxies `/controller/*` to port 8001 (controller)
3. `controller.py` responds with current config, PID, status, etc.
4. API key required for mutating endpoints (`/start`, `/stop`, `/nginx`, etc.)

---

## Key Design Decisions

| Decision | Reason |
|---|---|
| Two processes | Controller survives AI server restarts |
| `start_new_session=True` on all Popen calls | Daemons survive terminal close / SIGHUP |
| `asyncio.to_thread()` for `subprocess.run` | Prevents blocking the uvicorn event loop |
| Auto-build UI on startup | `ui/dist/` is gitignored; controller builds it if missing |
| Hash-based routing (`#dashboard`) | SPA served from a sub-path `/controller/ui` |
| API key in `localStorage` | Persists across page reloads; logout clears it |

---

## Configuration (.env)

All runtime config lives in `.env` in the project root. It is auto-created on first run.

```
API_KEY=<generated token>       # Shared key for all protected endpoints
MODEL_PATH=models/gemma3-4b.gguf
PORT=8000                       # AI server port
CONTROLLER_PORT=8001            # Controller port
N_PARALLEL=4                    # Parallel inference slots
N_CTX=4096                      # Context length per slot
N_THREADS=8                     # CPU threads
RATE_LIMIT=30/minute            # AI API rate limit
DOMAIN=api.example.com          # Public domain (for nginx)
```

---

## Process Management (PID files)

| File | Purpose |
|---|---|
| `.bikai.pid` | PID of the running AI server (`server.py`) |
| `.bikai-ctrl.pid` | PID of the running controller (`controller.py`) |
| `.bikai-dl.pid` | PID of an active model download process |

The controller checks PID files + `os.kill(pid, 0)` to verify processes are alive.

---

## Authentication

- **Single API key** stored in `.env` as `API_KEY`
- All protected endpoints require `X-API-Key: <key>` header
- The React UI stores the key in `localStorage` under `bikai_api_key`
- The `/api/controller/status` and `/api/controller/models` endpoints are **public** (no key needed) — so the dashboard header can show server status before login
- Key rotation: `POST /api/controller/token/new` generates a new key and saves to `.env`
