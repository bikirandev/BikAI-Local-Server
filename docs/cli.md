# cli.py — bikai Command Line Interface

## Overview

`cli.py` is the entry point for the `bikai` CLI tool, built with [Click](https://click.palletsprojects.com/).  
It is installed as a system command by `setup.sh` at `~/.local/bin/bikai`.

All commands operate on the project directory (`~/.bikai/` in production, the repo root in dev).

---

## Installation

`setup.sh` creates a launcher script at `~/.local/bin/bikai`:
```bash
#!/usr/bin/env bash
cd ~/.bikai
exec ~/.bikai/venv/bin/python ~/.bikai/cli.py "$@"
```

It's also registered as a package entry point in `pyproject.toml`:
```toml
[project.scripts]
bikai = "cli:cli"
```

---

## All Commands

### Server Management

```bash
bikai start                              # Start AI server (foreground)
bikai start --daemon                     # Start as background daemon
bikai start --model models/gemma.gguf   # Use specific model
bikai start --parallel 2 --port 9000    # Custom options
bikai stop                               # Stop AI server
bikai restart                            # Restart AI server
bikai status                             # Show status
bikai logs                               # Show last 200 log lines
bikai logs -f                            # Follow log (daemon mode)
bikai logs -n 500                        # Show last 500 lines
```

### Controller Management

```bash
bikai controller start                   # Start controller on port 8001
bikai controller start --port 9001       # Custom port
bikai controller stop                    # Stop controller
bikai controller restart                 # Restart controller
bikai controller status                  # Show controller status
```

### Model Download

```bash
bikai download -r bartowski/gemma-3-4b-it-GGUF -f gemma-3-4b-it-Q4_K_M.gguf  # HuggingFace
bikai download -g 1kO_KTjQ-GcaarzLxqXnUyJkEmbM6UC3d                           # Google Drive
bikai download -u https://example.com/model.gguf                               # Direct URL
bikai download -r ... --set-default      # Download and set as active model
```

### Combined Up/Down

```bash
bikai up                                  # Start controller + AI server
bikai up --api-port 8000 --ctrl-port 8001 --parallel 4
bikai down                                # Stop everything
```

### Nginx

```bash
bikai nginx --domain api.example.com      # Setup HTTP reverse proxy
bikai nginx --domain api.example.com --ssl  # Setup with Let's Encrypt SSL
bikai nginx --status                       # Show nginx status
bikai nginx --port 8000 --ctrl-port 8001   # Custom ports
```

### Token / API Key

```bash
bikai token show      # Print current API key
bikai token new       # Generate and save a new API key
```

### Info

```bash
bikai models          # List all downloaded models
bikai url             # Show local + public API URL
bikai config          # Show current config from .env
```

---

## Key Internal Helpers

```python
_read_env(key, default)     # Read from .env (calls load_dotenv on every read)
_write_env(key, value)      # Write to .env (using python-dotenv's set_key)
_server_url()               # Returns http://localhost:<PORT>
_ctrl_url()                 # Returns http://localhost:<CONTROLLER_PORT>
_is_running()               # Checks AI server health (GET /health)
_ctrl_is_running()          # Checks controller health (GET /controller/health)
_read_pid()                 # Read .bikai.pid
_read_ctrl_pid()            # Read .bikai-ctrl.pid
_auto_build_ui()            # Build React UI if ui/dist missing
```

---

## `bikai up` Behaviour

1. Calls `_auto_build_ui()` — builds UI if missing
2. Starts controller (`controller.py`) as daemon with `start_new_session=True`
3. Waits up to 10s for controller health check to pass
4. If `MODEL_PATH` is set in `.env`, starts AI server as daemon
5. Waits up to 60s for AI server health check to pass

---

## `bikai controller start` Behaviour

1. Checks if controller is already running via health check
2. Calls `_auto_build_ui()` — builds UI if missing
3. Spawns `controller.py` with `subprocess.Popen(..., start_new_session=True)`
4. Writes PID to `.bikai-ctrl.pid`
5. Writes `CONTROLLER_PORT` to `.env`

---

## `bikai download` Behaviour

1. Creates `models/` directory if needed
2. **HuggingFace**: Uses `huggingface_hub` CLI (`hf download`) or Python API
3. **Google Drive**: Uses `gdown` Python package
4. **Direct URL**: Uses `wget` or `httpx` streaming download
5. Progress shown in terminal; `--set-default` writes `MODEL_PATH` to `.env`

---

## `bikai nginx` Behaviour

1. Reads current port config from `.env`
2. Generates nginx config from template (proxies `/` → AI, `/controller` → controller)
3. Writes via `sudo tee /etc/nginx/sites-available/bikai`
4. Enables site, removes default, runs `sudo nginx -t`, reloads nginx
5. Optionally runs `certbot --nginx` for SSL

---

## PID File Pattern

All daemon commands use this pattern:
```python
proc = subprocess.Popen(cmd, stdout=log_fh, stderr=log_fh, start_new_session=True)
PID_FILE.write_text(str(proc.pid))
```

Stopping uses:
```python
os.kill(pid, signal.SIGTERM)
time.sleep(0.8)
os.kill(pid, signal.SIGKILL)  # fallback
PID_FILE.unlink(missing_ok=True)
```

---

## Adding a New CLI Command

```python
@cli.command()
@click.option("--my-option", default="value", help="Description")
def my_command(my_option):
    """Short description shown in --help."""
    _header("My Command")
    # ... implementation
    _ok("Done!")
```

Use `_ok()`, `_info()`, `_warn()`, `_err()` for consistent styled output.

---

## Dependencies

| Package | Purpose |
|---|---|
| `click` | CLI framework (commands, options, groups) |
| `httpx` | HTTP client for health checks |
| `python-dotenv` | Read/write `.env` file |
| `gdown` | Google Drive downloads |
| `huggingface_hub` | HuggingFace downloads |
