# setup.sh — Installer & setup.sh Guide

## Overview

`setup.sh` is the **one-liner installer** that sets up everything from scratch on a fresh server.

```bash
# Run locally
bash setup.sh

# Or one-liner from anywhere (once published)
curl -fsSL https://raw.githubusercontent.com/bikirandev/BikAI-Local-Server/main/setup.sh | bash
```

It installs all dependencies, builds the UI, generates the API key, starts the servers, configures nginx, and sets up systemd auto-start.

---

## What It Does (Step by Step)

### Step 1 — Python 3.10+
Checks for Python 3.10+. If missing or too old, installs via the OS package manager.  
Supported: Debian/Ubuntu (`apt`), Fedora (`dnf`), RHEL (`yum`), Arch (`pacman`), SUSE (`zypper`), macOS (`brew`).

### Step 2 — Build Tools + Node.js
Installs: `cmake`, `gcc`, `nginx`, `openblas-dev`, `npm`/Node.js.  
On Debian, Node.js LTS is installed via NodeSource if `npm` is missing.

### Step 3 — Clone / Update Repo
If `~/.bikai/` doesn't exist: `git clone` the repo.  
If it exists: `git pull` to update.

### Step 4 — Python venv + pip install
```bash
python3 -m venv ~/.bikai/venv
CMAKE_ARGS="-DGGML_BLAS=ON -DGGML_BLAS_VENDOR=OpenBLAS" \
  pip install -r requirements.txt
pip install -e .   # installs bikai CLI entry point
```

`llama-cpp-python` is compiled from source with OpenBLAS for CPU acceleration.

### Step 5 — bikai command
Creates `~/.local/bin/bikai` launcher script:
```bash
#!/usr/bin/env bash
cd ~/.bikai
exec ~/.bikai/venv/bin/python ~/.bikai/cli.py "$@"
```
Adds `~/.local/bin` to `PATH` in the appropriate shell RC file.

### Step A — Generate API Key
Uses Python + `python-dotenv` to generate `secrets.token_urlsafe(32)` and write to `~/.bikai/.env`.  
Skips if key already exists.

### Step B — Build React UI
```bash
cd ~/.bikai/ui
npm install --silent
npm run build
```
If `npm` is missing, warns and skips (controller will auto-build later).

### Step C — bikai up
Runs `bikai up --api-port 8000 --ctrl-port 8001 --parallel 4`.  
Starts both the controller and AI server (if a model is configured).

### Step D — nginx
Runs `bikai nginx --port 8000 --ctrl-port 8001`.  
Writes nginx config, enables site, reloads nginx.

### Step E — systemd Auto-start
Creates two systemd services:

**`bikai-controller.service`** (always created):
```ini
[Service]
ExecStart=~/.bikai/venv/bin/python ~/.bikai/controller.py --port 8001
Restart=always
RestartSec=5
```

**`bikai.service`** (only if a model is configured):
```ini
[Service]
ExecStart=~/.bikai/venv/bin/python ~/.bikai/server.py --model <MODEL_FILE>
Restart=on-failure
```

Both are enabled with `systemctl enable` so they start on boot.

---

## Final Output

The installer prints:
```
  API Key:  <your-key>
  UI:       http://<your-ip>/controller/ui
  API:      http://<your-ip>/v1/chat/completions
  
  Recommended model (Google Drive):
  ID: 1kO_KTjQ-GcaarzLxqXnUyJkEmbM6UC3d
  bikai download -g 1kO_KTjQ-GcaarzLxqXnUyJkEmbM6UC3d
```

---

## Sudo Handling

The script detects root vs. non-root:
- If root: `SUDO=""` (no sudo prefix)
- If non-root with passwordless sudo: proceeds silently  
- If non-root with password sudo: prompts once, then keeps the timestamp alive with a background loop

---

## Re-running

The script is idempotent — safe to re-run:
- Skips steps that are already done (Python present, packages installed, etc.)
- `git pull` updates the code
- UI rebuild happens if needed
- Existing API key is preserved
- systemd services are overwritten with latest config

---

## Manual Install (Without setup.sh)

```bash
git clone https://github.com/bikirandev/BikAI-Local-Server.git ~/.bikai
cd ~/.bikai
python3 -m venv venv
CMAKE_ARGS="-DGGML_BLAS=ON -DGGML_BLAS_VENDOR=OpenBLAS" venv/bin/pip install -r requirements.txt
venv/bin/pip install -e .
venv/bin/python cli.py controller start --port 8001
```

The controller will auto-build the UI on first start.

---

## Production Deployment Checklist

1. Run `setup.sh` on a fresh Linux server (Ubuntu 22.04+ recommended)
2. Verify: `curl http://localhost:8001/api/controller/status`
3. Open firewall ports 80 (nginx) and optionally 443 (SSL)
4. Access UI: `http://<server-ip>/controller/ui`
5. Login with the API key shown at the end of `setup.sh`
6. Go to Models tab → Download a model (Gemma 3 4B recommended)
7. Go to Dashboard → Start server

---

## Environment Variables Summary

All stored in `~/.bikai/.env`:

| Variable | Default | Set By |
|---|---|---|
| `API_KEY` | auto-generated | setup.sh step A / controller startup |
| `MODEL_PATH` | (empty) | Set after model download |
| `PORT` | 8000 | `bikai start` / Dashboard |
| `CONTROLLER_PORT` | 8001 | `bikai controller start` |
| `N_PARALLEL` | 4 | `bikai start` / Dashboard |
| `N_CTX` | 4096 | `bikai start` / Dashboard |
| `N_THREADS` | cpu_count | `bikai start` / Dashboard |
| `RATE_LIMIT` | 30/minute | Manual edit |
| `DOMAIN` | (empty) | `bikai nginx` / Nginx page |
