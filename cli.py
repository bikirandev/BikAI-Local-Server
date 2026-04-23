#!/usr/bin/env python3
"""
Bik AI CLI
==========
Manage your local LLM server from the command line.

Usage:
  python cli.py <command> [options]
  ./bikai <command> [options]          # after chmod +x bikai

Commands:
  start       Start the API server
  stop        Stop the running server
  restart     Restart the server
  status      Show server status
  download    Download a model from HuggingFace
  models      List downloaded models
  token show  Show current API key
  token new   Generate and save a new API key
  url         Show API URL (local + ngrok if running)
  config      Show current configuration
"""

import json
import os
import secrets
import signal
import subprocess
import sys
import time
from pathlib import Path

import click
import httpx
from dotenv import load_dotenv, set_key

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ENV_FILE   = Path(".env")
PID_FILE   = Path(".bikai.pid")
LOG_FILE   = Path("bikai-server.log")
MODELS_DIR = Path("models")

load_dotenv()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _read_env(key: str, default: str = "") -> str:
    load_dotenv(override=True)
    return os.getenv(key, default)


def _write_env(key: str, value: str) -> None:
    ENV_FILE.touch(exist_ok=True)
    set_key(str(ENV_FILE), key, value)


def _server_url() -> str:
    port = _read_env("PORT", "8000")
    return f"http://localhost:{port}"


def _is_running() -> bool:
    try:
        r = httpx.get(f"{_server_url()}/health", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


def _read_pid() -> int | None:
    if PID_FILE.exists():
        try:
            return int(PID_FILE.read_text().strip())
        except ValueError:
            pass
    return None


def _ok(msg: str)   -> None: click.echo(click.style(f"  ✓  {msg}", fg="green"))
def _info(msg: str) -> None: click.echo(click.style(f"  →  {msg}", fg="cyan"))
def _warn(msg: str) -> None: click.echo(click.style(f"  !  {msg}", fg="yellow"))
def _err(msg: str)  -> None: click.echo(click.style(f"  ✗  {msg}", fg="red"), err=True)


def _header(title: str) -> None:
    click.echo()
    click.echo(click.style(f"  {'─'*50}", fg="bright_black"))
    click.echo(click.style(f"  {title}", bold=True))
    click.echo(click.style(f"  {'─'*50}", fg="bright_black"))


# ---------------------------------------------------------------------------
# CLI root
# ---------------------------------------------------------------------------

@click.group(context_settings={"help_option_names": ["-h", "--help"]})
def cli():
    """
    \b
    ██████╗ ██╗██╗  ██╗     █████╗ ██╗
    ██╔══██╗██║██║ ██╔╝    ██╔══██╗██║
    ██████╔╝██║█████╔╝     ███████║██║
    ██╔══██╗██║██╔═██╗     ██╔══██║██║
    ██████╔╝██║██║  ██╗    ██║  ██║██║
    ╚═════╝ ╚═╝╚═╝  ╚═╝    ╚═╝  ╚═╝╚═╝

    Bik AI — Local LLM Server by bikiran.com

    \b
    Quick start:
      bikai download -r bartowski/gemma-3-4b-it-GGUF -f gemma-3-4b-it-Q4_K_M.gguf
      bikai start --daemon
      bikai status

    \b
    Download examples:
      bikai download -r bartowski/gemma-3-4b-it-GGUF -f gemma-3-4b-it-Q4_K_M.gguf
      bikai download -u https://your-storage.com/gemma.gguf
      bikai download -g 1aBcDeFgHiJkLmNoPqRsTuV

    \b
    Start examples:
      bikai start --model ./models/gemma-3-4b-it-Q4_K_M.gguf
      bikai start --model ./models/gemma-3-4b-it-Q4_K_M.gguf --parallel 3
      bikai start --model ./models/gemma.gguf --parallel 2 --port 9000
      bikai start --daemon                         (background, uses MODEL_PATH from .env)

    \b
    Server control:
      bikai stop
      bikai restart
      bikai status
      bikai logs -f                                (live log, daemon mode only)
      bikai logs -n 100                            (last 100 lines)

    \b
    Expose publicly with nginx:
      bikai nginx --domain api.example.com         (HTTP reverse proxy)
      bikai nginx --domain api.example.com --ssl   (HTTPS via Let's Encrypt)
      bikai nginx --status                         (show nginx status)

    \b
    Models:
      bikai models                                 (list downloaded models)

    \b
    API key:
      bikai token show                             (show current key)
      bikai token new                              (generate new key)

    \b
    Info:
      bikai url                                    (show local + public URL)
      bikai config                                 (show all settings including parallel count)

    \b
    Parallel slots (how many users at once):
      Default is 4. Change with --parallel or set N_PARALLEL in .env
      RAM guide  →  2B model: 6  |  4B model: 4  |  9B model: 2

    \b
    Run 'bikai COMMAND -h' for full options of any command.
    """
    pass


# ---------------------------------------------------------------------------
# start
# ---------------------------------------------------------------------------

@cli.command()
@click.option("--model",    "-m", default=lambda: _read_env("MODEL_PATH"),    help="Path to GGUF model file")
@click.option("--parallel", "-p", default=lambda: int(_read_env("N_PARALLEL", "4")), show_default=True, help="Parallel instances")
@click.option("--port",           default=lambda: int(_read_env("PORT", "8000")),     show_default=True, help="Bind port")
@click.option("--ctx",            default=lambda: int(_read_env("N_CTX", "4096")),    show_default=True, help="Context length per slot")
@click.option("--threads",        default=lambda: int(_read_env("N_THREADS", str(os.cpu_count() or 4))), show_default=True, help="CPU threads")
@click.option("--daemon", "-d",   is_flag=True, default=False, help="Run in background (daemon mode)")
def start(model, parallel, port, ctx, threads, daemon):
    """Start the API server."""
    _header("Starting Bik AI Server")

    if not model:
        _err("No model specified. Use --model or set MODEL_PATH in .env")
        _info("Example: bikai start --model gemma-2-2b-it-Q4_K_M")
        sys.exit(1)

    # Resolve bare name: "gemma3-4b" → "./models/gemma3-4b.gguf"
    # Try exact path first, then models/ dir with/without .gguf extension
    if not Path(model).is_file():
        candidates = [
            MODELS_DIR / model,
            MODELS_DIR / (model + ".gguf"),
        ]
        resolved = next((str(p) for p in candidates if p.is_file()), None)
        if resolved:
            model = resolved
        else:
            _err(f"Model not found: {model}")
            _info(f"Available models:")
            for f in sorted(MODELS_DIR.glob("**/*.gguf")):
                _info(f"  {f.name}")
            sys.exit(1)

    if _is_running():
        _warn("Server is already running.")
        _info(f"Local URL: {_server_url()}")
        return

    cmd = [
        sys.executable, "server.py",
        "--model",    model,
        "--parallel", str(parallel),
        "--port",     str(port),
        "--ctx",      str(ctx),
        "--threads",  str(threads),
    ]

    # Save PORT to .env so other commands can find the server
    _write_env("PORT", str(port))

    if daemon:
        log = LOG_FILE.open("w")
        proc = subprocess.Popen(cmd, stdout=log, stderr=log, start_new_session=True)
        PID_FILE.write_text(str(proc.pid))
        _ok(f"Server started in background  (PID {proc.pid})")
        _info(f"Logs:      tail -f {LOG_FILE}")
        _info(f"Local URL: {_server_url()}")
        _info("Stop with: bikai stop")
    else:
        _ok(f"Server starting on port {port}  (Ctrl+C to stop)")
        _info(f"Local URL: {_server_url()}/docs")
        click.echo()
        try:
            subprocess.run(cmd, check=False)
        except KeyboardInterrupt:
            _info("Server stopped.")


# ---------------------------------------------------------------------------
# stop
# ---------------------------------------------------------------------------

@cli.command()
def stop():
    """Stop the running server."""
    _header("Stopping Bik AI Server")

    pid = _read_pid()
    stopped = False

    if pid:
        try:
            os.kill(pid, signal.SIGTERM)
            time.sleep(1)
            try:
                os.kill(pid, signal.SIGKILL)  # force if still alive
            except ProcessLookupError:
                pass
            PID_FILE.unlink(missing_ok=True)
            _ok(f"Process {pid} terminated.")
            stopped = True
        except ProcessLookupError:
            PID_FILE.unlink(missing_ok=True)

    # Also kill anything on the port
    port = _read_env("PORT", "8000")
    result = subprocess.run(["fuser", f"{port}/tcp"], capture_output=True, text=True)
    pids = result.stdout.strip().split()
    for p in pids:
        try:
            os.kill(int(p), signal.SIGTERM)
            stopped = True
        except (ValueError, ProcessLookupError):
            pass

    if stopped:
        _ok("Server stopped.")
    else:
        _warn("No running server found.")


# ---------------------------------------------------------------------------
# restart
# ---------------------------------------------------------------------------

@cli.command()
@click.pass_context
def restart(ctx):
    """Restart the server (daemon mode)."""
    ctx.invoke(stop)
    time.sleep(2)
    model = _read_env("MODEL_PATH")
    if model:
        ctx.invoke(start, model=model, daemon=True)
    else:
        _err("Cannot restart: MODEL_PATH not set in .env")


# ---------------------------------------------------------------------------
# status
# ---------------------------------------------------------------------------

@cli.command()
def status():
    """Show server status."""
    _header("Bik AI Server Status")

    running = _is_running()
    pid     = _read_pid()
    port    = _read_env("PORT", "8000")
    model   = _read_env("MODEL_PATH", "—")
    api_key = _read_env("API_KEY", "—")

    status_label = click.style("RUNNING", fg="green", bold=True) if running \
               else click.style("STOPPED", fg="red", bold=True)

    click.echo(f"\n  Status    : {status_label}")
    if pid:
        click.echo(f"  PID       : {pid}")
    click.echo(f"  Port      : {port}")
    click.echo(f"  Model     : {Path(model).name if model != '—' else '—'}")
    click.echo(f"  API Key   : {api_key[:12]}…" if len(api_key) > 12 else f"  API Key   : {api_key}")

    if running:
        click.echo(f"  Local URL : {_server_url()}")
        try:
            r = httpx.get(f"{_server_url()}/health", timeout=3)
            data = r.json()
            click.echo(f"  Health    : {data.get('status', 'ok')}")
        except Exception:
            pass
    click.echo()


# ---------------------------------------------------------------------------
# download
# ---------------------------------------------------------------------------

@cli.command()
@click.option("--repo", "-r", default=None, help="HuggingFace repo       e.g. bartowski/gemma-3-4b-it-GGUF")
@click.option("--file", "-f", default=None, help="GGUF filename          e.g. gemma-3-4b-it-Q4_K_M.gguf")
@click.option("--url",  "-u", default=None, help="Direct URL             e.g. https://your-storage.com/model.gguf")
@click.option("--gdrive", "-g", default=None, help="Google Drive file ID   e.g. 1aBcDeFgHiJkLmNoPqRsTuV")
@click.option("--dir",  "-d", default=str(MODELS_DIR), show_default=True, help="Download directory")
@click.option("--set-default", is_flag=True, default=False, help="Save as MODEL_PATH in .env after download")
def download(repo, file, url, gdrive, dir, set_default):
    """Download a GGUF model from HuggingFace, a direct URL, or Google Drive.

    \b
    Examples:
      # HuggingFace
      bikai download -r bartowski/gemma-3-4b-it-GGUF -f gemma-3-4b-it-Q4_K_M.gguf

      # Direct URL (S3, Cloudflare R2, VPS, any HTTPS link)
      bikai download -u https://your-storage.com/gemma.gguf

      # Google Drive  (use the file ID from the share link)
      bikai download -g 1aBcDeFgHiJkLmNoPqRsTuV
    """
    _header("Downloading Model")

    dest = Path(dir)
    dest.mkdir(parents=True, exist_ok=True)

    # ── Google Drive ───────────────────────────────────────────────
    if gdrive:
        # Accept full share URLs too: extract ID automatically
        if "drive.google.com" in gdrive:
            import re
            m = re.search(r"/d/([a-zA-Z0-9_-]+)", gdrive) or \
                re.search(r"id=([a-zA-Z0-9_-]+)", gdrive)
            if m:
                gdrive = m.group(1)
            else:
                _err("Could not parse Google Drive file ID from URL.")
                sys.exit(1)

        out_name = file  # user-provided name, or None to auto-detect from Drive
        _info(f"Source : Google Drive  (id: {gdrive})")
        _info(f"Dest   : {dest}/gdrive_<filename>")
        click.echo()

        try:
            import gdown
        except ImportError:
            _info("Installing gdown…")
            subprocess.run([sys.executable, "-m", "pip", "install", "gdown", "-q"], check=True)
            import gdown

        if out_name:
            # User specified a filename — download directly to that name
            model_path = str(dest / out_name)
            gdown.download(id=gdrive, output=model_path, quiet=False)
        else:
            # Let gdown use the real Drive filename by passing dest dir (trailing slash)
            result_path = gdown.download(id=gdrive, output=str(dest) + "/", quiet=False)
            if result_path and Path(result_path).exists():
                actual = Path(result_path)
                real_name = actual.name
                # Prefix with gdrive_ if not already
                if not real_name.startswith("gdrive_"):
                    final_path = dest / f"gdrive_{real_name}"
                    actual.rename(final_path)
                    model_path = str(final_path)
                else:
                    model_path = str(actual)
            else:
                _err("Download failed — gdown returned no file path.")
                sys.exit(1)

    # ── Direct URL ─────────────────────────────────────────────────
    elif url:
        out_name = file or Path(url.split("?")[0]).name
        if not out_name.endswith(".gguf"):
            out_name += ".gguf"
        model_path = str(dest / out_name)
        _info(f"Source : {url}")
        _info(f"Dest   : {model_path}")
        click.echo()

        try:
            subprocess.run(
                ["wget", "--progress=bar:force", "-O", model_path, url],
                check=True,
            )
        except FileNotFoundError:
            # fallback to curl
            try:
                subprocess.run(
                    ["curl", "-L", "--progress-bar", "-o", model_path, url],
                    check=True,
                )
            except FileNotFoundError:
                _err("Neither 'wget' nor 'curl' found. Install one and retry.")
                sys.exit(1)

    # ── HuggingFace ────────────────────────────────────────────────
    elif repo and file:
        model_path = str(dest / file)
        _info(f"Source : HuggingFace  {repo}")
        _info(f"File   : {file}")
        _info(f"Dest   : {dest.resolve()}")
        click.echo()

        # Remove incomplete/zero-byte file so hf doesn't skip it
        existing = dest / file
        if existing.exists() and existing.stat().st_size < 1_000_000:
            _warn(f"Removing incomplete file ({existing.stat().st_size} bytes): {existing.name}")
            existing.unlink()

        try:
            subprocess.run(
                ["hf", "download", repo, file, "--local-dir", str(dest)],
                check=True,
            )
        except FileNotFoundError:
            _err("'hf' command not found. Install with:  pip install huggingface-hub")
            sys.exit(1)
        except subprocess.CalledProcessError as e:
            _err(f"Download failed: {e}")
            sys.exit(1)

    else:
        _err("Provide one of:  --repo + --file  |  --url  |  --gdrive")
        _info("Run:  bikai download -h")
        sys.exit(1)

    # ── Validate downloaded file ────────────────────────────────────
    final = Path(model_path)
    if not final.exists() or final.stat().st_size < 1_000_000:
        size = final.stat().st_size if final.exists() else 0
        _err(f"Download failed or file is corrupt ({size} bytes). Please try again.")
        if final.exists():
            final.unlink()
        sys.exit(1)

    size_gb = final.stat().st_size / 1_073_741_824
    _ok(f"Downloaded to: {model_path}  ({size_gb:.2f} GB)")

    if set_default or (sys.stdin.isatty() and click.confirm("\n  Set as default model in .env?", default=True)):
        _write_env("MODEL_PATH", model_path)
        _ok("MODEL_PATH updated in .env")
        _info("Run server with:  bikai start")



# ---------------------------------------------------------------------------
# models
# ---------------------------------------------------------------------------

@cli.command()
def models():
    """List downloaded GGUF models."""
    _header("Downloaded Models")

    gguf_files = sorted(MODELS_DIR.glob("**/*.gguf")) if MODELS_DIR.exists() else []
    active = _read_env("MODEL_PATH")

    if not gguf_files:
        _warn("No models found in ./models/")
        _info("Download one with:  bikai download --repo bartowski/gemma-3-4b-it-GGUF --file gemma-3-4b-it-Q4_K_M.gguf")
        return

    click.echo()
    for f in gguf_files:
        size_gb = f.stat().st_size / 1_073_741_824
        is_active = str(f) == active or str(f.resolve()) == str(Path(active).resolve() if active else "")
        marker = click.style(" ← active", fg="green") if is_active else ""
        click.echo(f"  {f.name:<50}  {size_gb:.2f} GB{marker}")
    click.echo()


# ---------------------------------------------------------------------------
# token group
# ---------------------------------------------------------------------------

@cli.group()
def token():
    """Manage the API key."""
    pass


@token.command("show")
def token_show():
    """Show the current API key."""
    _header("API Key")
    key = _read_env("API_KEY")
    if key:
        click.echo(f"\n  {click.style(key, fg='bright_white', bold=True)}\n")
        _info("Include this in every request as:  X-API-Key: <key>")
    else:
        _warn("No API key set. Start the server once to auto-generate one.")
    click.echo()


@token.command("new")
def token_new():
    """Generate a new API key and save to .env."""
    _header("New API Key")
    old = _read_env("API_KEY")
    if old:
        if not click.confirm("  This will invalidate the current key. Continue?", default=False):
            _info("Cancelled.")
            return

    key = secrets.token_urlsafe(32)
    _write_env("API_KEY", key)
    click.echo(f"\n  {click.style(key, fg='green', bold=True)}\n")
    _ok("New key saved to .env")
    _warn("Restart the server for the new key to take effect:  bikai restart")
    click.echo()


# ---------------------------------------------------------------------------
# url
# ---------------------------------------------------------------------------

@cli.command()
def url():
    """Show API URLs."""
    _header("API URLs")

    port    = _read_env("PORT", "8000")
    domain  = _read_env("DOMAIN")
    running = _is_running()

    click.echo()
    local = f"http://localhost:{port}"
    click.echo(f"  Local  : {click.style(local, fg='cyan')}")

    if domain:
        public = f"https://{domain}"
        click.echo(f"  Public : {click.style(public, fg='green', bold=True)}")
    else:
        _warn("No domain set. Run:  bikai nginx --domain your-domain.com")

    if not running:
        click.echo()
        _warn("Server is not running. Start with:  bikai start")

    click.echo()


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------

@cli.command()
def config():
    """Show current configuration."""
    _header("Configuration")

    keys = [
        ("MODEL_PATH",   "Model file"),
        ("API_KEY",      "API key"),
        ("N_PARALLEL",   "Parallel instances"),
        ("N_CTX",        "Context length"),
        ("N_THREADS",    "CPU threads"),
        ("PORT",         "Port"),
        ("RATE_LIMIT",   "Rate limit"),
        ("DOMAIN",       "Public domain"),
    ]

    click.echo()
    for env_key, label in keys:
        val = _read_env(env_key, "—")
        # Mask sensitive values
        if env_key in ("API_KEY",) and len(val) > 8:
            val = val[:8] + "…" + val[-4:]
        click.echo(f"  {label:<22} {click.style(val, fg='bright_white')}")
    click.echo()


# ---------------------------------------------------------------------------
# nginx
# ---------------------------------------------------------------------------

NGINX_CONF_TEMPLATE = """\
server {{
    listen 80;
    server_name {server_name};

    # ── Controller UI & management API ────────────────────────
    location /controller {{
        proxy_pass http://127.0.0.1:{ctrl_port};
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_read_timeout 60s;
    }}

    location /api/controller {{
        proxy_pass http://127.0.0.1:{ctrl_port};
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_read_timeout 60s;
    }}

    # ── AI inference API ──────────────────────────────────────
    location / {{
        proxy_pass http://127.0.0.1:{port};
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # Streaming (SSE) support
        proxy_buffering off;
        proxy_cache off;
        proxy_read_timeout 300s;
        chunked_transfer_encoding on;

        # CORS — allow all origins
        add_header 'Access-Control-Allow-Origin' '*' always;
        add_header 'Access-Control-Allow-Methods' 'GET, POST, OPTIONS' always;
        add_header 'Access-Control-Allow-Headers' 'X-API-Key, Content-Type, Authorization' always;
        add_header 'Access-Control-Max-Age' '86400' always;

        if ($request_method = OPTIONS) {{
            add_header 'Access-Control-Allow-Origin' '*';
            add_header 'Access-Control-Allow-Methods' 'GET, POST, OPTIONS';
            add_header 'Access-Control-Allow-Headers' 'X-API-Key, Content-Type, Authorization';
            add_header 'Access-Control-Max-Age' '86400';
            return 204;
        }}
    }}
}}
"""


@cli.command()
@click.option("--domain", "-d", default=lambda: _read_env("DOMAIN"), required=False, help="Domain or public IP (omit to auto-detect public IP)")
@click.option("--port",         default=lambda: _read_env("PORT", "8000"),            help="AI server port to proxy to")
@click.option("--ctrl-port",    default=lambda: _read_env("CTRL_PORT", "8001"),        help="Controller port to proxy to")
@click.option("--ssl",          is_flag=True, default=False,                           help="Obtain HTTPS cert via Let's Encrypt (certbot, requires a domain)")
@click.option("--status",       is_flag=True, default=False,                           help="Show nginx status and exit")
def nginx(domain, port, ctrl_port, ssl, status):
    """Configure nginx as a reverse proxy for the API server.

    \b
    Examples:
      bikai nginx                                   # auto-detect public IP, HTTP
      bikai nginx --domain api.example.com          # specific domain, HTTP
      bikai nginx --domain api.example.com --ssl    # HTTPS via Let's Encrypt
      bikai nginx --status                          # check nginx status
    """
    _header("Nginx Setup")

    # Resolve nginx binary (may be at /usr/sbin/nginx, not in venv PATH)
    nginx_bin = None
    for candidate in ["nginx", "/usr/sbin/nginx", "/usr/local/sbin/nginx"]:
        if subprocess.run(["which", candidate] if "/" not in candidate else ["test", "-x", candidate],
                          capture_output=True).returncode == 0:
            nginx_bin = candidate
            break
    nginx_installed = nginx_bin is not None

    if status:
        if not nginx_installed:
            _warn("nginx is not installed.")
            _info("Install it with:  sudo apt install nginx")
            return
        result = subprocess.run(["sudo", nginx_bin, "-t"], capture_output=True, text=True)
        if result.returncode == 0:
            _ok("nginx config is valid")
        else:
            _err(f"nginx config error:\n{result.stderr}")
        subprocess.run(["systemctl", "status", "nginx", "--no-pager", "-l"], check=False)
        return

    # Auto-detect public IP if no domain given
    if not domain:
        _info("No domain specified — detecting public IP...")
        try:
            import urllib.request
            domain = urllib.request.urlopen("https://api.ipify.org", timeout=5).read().decode().strip()
            _ok(f"Public IP: {domain}")
        except Exception:
            _err("Could not detect public IP. Pass --domain manually.")
            sys.exit(1)
        if ssl:
            _warn("--ssl requires a domain name, not an IP. Disabling SSL.")
            ssl = False

    # Install nginx if not present
    if not nginx_installed:
        _info("nginx not found. Installing...")
        try:
            subprocess.run(["sudo", "apt-get", "install", "-y", "-q", "nginx"], check=True)
            nginx_bin = "/usr/sbin/nginx"
        except subprocess.CalledProcessError:
            _err("Could not install nginx. Install it manually: sudo apt install nginx")
            sys.exit(1)

    conf_name = "bikai"
    conf_path = f"/etc/nginx/sites-available/{conf_name}"
    link_path = f"/etc/nginx/sites-enabled/{conf_name}"

    # Use _ (catch-all) for bare IPs; use the domain name for named domains
    import re as _re
    is_ip = bool(_re.match(r"^\d{1,3}(\.\d{1,3}){3}$", domain))
    server_name = "_" if is_ip else domain

    conf_content = NGINX_CONF_TEMPLATE.format(server_name=server_name, port=port, ctrl_port=ctrl_port)

    _info(f"Writing nginx config: {conf_path}")
    try:
        # Write via sudo tee
        proc = subprocess.run(
            ["sudo", "tee", conf_path],
            input=conf_content,
            text=True,
            capture_output=True,
        )
        if proc.returncode != 0:
            _err(f"Failed to write nginx config: {proc.stderr}")
            sys.exit(1)
    except FileNotFoundError:
        _err("'sudo' not found. Run as root or install sudo.")
        sys.exit(1)

    # Enable site
    subprocess.run(["sudo", "ln", "-sf", conf_path, link_path], check=True)

    # Remove default site if present (avoids port 80 conflict)
    subprocess.run(["sudo", "rm", "-f", "/etc/nginx/sites-enabled/default"], check=False)

    # Test config
    result = subprocess.run(["sudo", nginx_bin, "-t"], capture_output=True, text=True)
    if result.returncode != 0:
        _err(f"nginx config test failed:\n{result.stderr}")
        sys.exit(1)

    # Reload nginx
    subprocess.run(["sudo", "systemctl", "enable", "nginx"], check=True)
    subprocess.run(["sudo", "systemctl", "reload", "nginx"], check=True)
    _ok(f"nginx configured and reloaded")
    _info(f"HTTP URL: http://{domain}")

    # Save domain to .env
    _write_env("DOMAIN", domain)

    # SSL via certbot
    if ssl:
        click.echo()
        _info("Setting up HTTPS via Let's Encrypt...")

        # Install certbot if needed
        if subprocess.run(["which", "certbot"], capture_output=True).returncode != 0:
            _info("Installing certbot...")
            subprocess.run(
                ["sudo", "apt-get", "install", "-y", "-q", "certbot", "python3-certbot-nginx"],
                check=True,
            )

        result = subprocess.run(
            ["sudo", "certbot", "--nginx", "-d", domain, "--non-interactive", "--agree-tos", "--redirect"],
            capture_output=False,
        )
        if result.returncode == 0:
            _ok(f"SSL certificate obtained!")
            _info(f"HTTPS URL: https://{domain}")
        else:
            _warn("certbot failed. Make sure:")
            _warn(f"  1. DNS for '{domain}' points to this server's IP")
            _warn("  2. Port 80 is open in your firewall")
            _warn("  Run manually: sudo certbot --nginx -d " + domain)

    click.echo()
    _info("Run 'bikai url' to see your public URL")


# ---------------------------------------------------------------------------
# logs
# ---------------------------------------------------------------------------

@cli.command()
@click.option("--lines", "-n", default=50, show_default=True, help="Number of lines to show")
@click.option("--follow", "-f", is_flag=True, default=False, help="Follow log output (like tail -f)")
def logs(lines, follow):
    """Show server logs (daemon mode only)."""
    _header("Server Logs")

    if not LOG_FILE.exists():
        _warn(f"Log file not found: {LOG_FILE}")
        _info("Logs are only written when server is started with --daemon")
        return

    if follow:
        subprocess.run(["tail", f"-n{lines}", "-f", str(LOG_FILE)], check=False)
    else:
        subprocess.run(["tail", f"-n{lines}", str(LOG_FILE)], check=False)


# ---------------------------------------------------------------------------
# controller group — manage the always-running management server
# ---------------------------------------------------------------------------

CTRL_PID_FILE = Path(".bikai-ctrl.pid")


def _ctrl_url() -> str:
    port = _read_env("CONTROLLER_PORT", "8001")
    return f"http://localhost:{port}"


def _ctrl_is_running() -> bool:
    try:
        r = httpx.get(f"{_ctrl_url()}/controller/health", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


def _auto_build_ui() -> None:
    """Build the React UI if ui/dist/index.html is missing."""
    base = Path(__file__).parent
    index = base / "ui" / "dist" / "index.html"
    if index.exists():
        return

    ui_src = base / "ui"
    if not (ui_src / "package.json").exists():
        _warn("ui/package.json not found — skipping UI build")
        return

    npm = subprocess.run(["which", "npm"], capture_output=True, text=True).stdout.strip()
    if not npm:
        _warn("npm not found — cannot auto-build UI. Install Node.js 18+ and run: cd ui && npm run build")
        return

    _info("UI not built. Building now (this takes ~15s)…")
    if not (ui_src / "node_modules").exists():
        _info("Installing UI dependencies…")
        r = subprocess.run([npm, "install", "--silent"], cwd=str(ui_src), capture_output=True, text=True)
        if r.returncode != 0:
            _warn(f"npm install failed:\n{r.stderr[:300]}")
            return

    r = subprocess.run([npm, "run", "build"], cwd=str(ui_src), capture_output=True, text=True)
    if r.returncode == 0:
        _ok("UI built successfully.")
    else:
        _warn(f"UI build failed:\n{r.stderr[:300]}")


def _read_ctrl_pid() -> int | None:
    if CTRL_PID_FILE.exists():
        try:
            return int(CTRL_PID_FILE.read_text().strip())
        except ValueError:
            pass
    return None


@cli.group()
def controller():
    """Manage the controller (management UI) server."""
    pass


@controller.command("start")
@click.option("--port", "-p", default=lambda: int(_read_env("CONTROLLER_PORT", "8001")),
              show_default=True, help="Controller port")
@click.option("--host", default="0.0.0.0", show_default=True, help="Bind host")
def controller_start(port, host):
    """Start the controller server in background (daemon mode)."""
    _header("Starting Bik AI Controller")

    if _ctrl_is_running():
        _warn("Controller is already running.")
        _info(f"UI: {_ctrl_url()}/controller/ui")
        return

    # Auto-build React UI if dist is missing
    _auto_build_ui()

    cmd = [
        sys.executable,
        str(Path(__file__).parent / "controller.py"),
        "--port", str(port),
        "--host", host,
    ]
    _write_env("CONTROLLER_PORT", str(port))
    log_path = Path("bikai-controller.log")
    log_fh = log_path.open("w")
    proc = subprocess.Popen(cmd, stdout=log_fh, stderr=log_fh, start_new_session=True)
    CTRL_PID_FILE.write_text(str(proc.pid))
    _ok(f"Controller started  (PID {proc.pid})")
    _info(f"UI:    http://localhost:{port}/controller/ui")
    _info(f"API:   http://localhost:{port}/api/controller/status")
    _info(f"Logs:  tail -f {log_path}")
    _info("Stop with:  bikai controller stop")


@controller.command("stop")
def controller_stop():
    """Stop the controller server."""
    _header("Stopping Controller")

    pid = _read_ctrl_pid()
    stopped = False

    if pid:
        for sig in (signal.SIGTERM, signal.SIGKILL):
            try:
                os.kill(pid, sig)
                time.sleep(0.8)
            except ProcessLookupError:
                break
        CTRL_PID_FILE.unlink(missing_ok=True)
        _ok(f"Process {pid} terminated.")
        stopped = True

    port = _read_env("CONTROLLER_PORT", "8001")
    result = subprocess.run(["fuser", f"{port}/tcp"], capture_output=True, text=True)
    for p in result.stdout.strip().split():
        try:
            os.kill(int(p), signal.SIGTERM)
            stopped = True
        except (ValueError, ProcessLookupError):
            pass

    if stopped:
        _ok("Controller stopped.")
    else:
        _warn("No running controller found.")


@controller.command("status")
def controller_status():
    """Show controller server status."""
    _header("Controller Status")
    running = _ctrl_is_running()
    pid = _read_ctrl_pid()
    port = _read_env("CONTROLLER_PORT", "8001")

    label = click.style("RUNNING", fg="green", bold=True) if running \
        else click.style("STOPPED", fg="red", bold=True)

    click.echo(f"\n  Status : {label}")
    if pid:
        click.echo(f"  PID    : {pid}")
    click.echo(f"  Port   : {port}")
    click.echo(f"  UI     : http://localhost:{port}/controller/ui")
    click.echo()


@controller.command("restart")
@click.pass_context
def controller_restart(ctx):
    """Restart the controller server."""
    ctx.invoke(controller_stop)
    time.sleep(1)
    ctx.invoke(controller_start)


# ---------------------------------------------------------------------------
# up  —  start everything with one command
# ---------------------------------------------------------------------------

@cli.command()
@click.option("--api-port",   default=lambda: int(_read_env("PORT", "8000")),            show_default=True, help="AI server port")
@click.option("--ctrl-port",  default=lambda: int(_read_env("CONTROLLER_PORT", "8001")), show_default=True, help="Controller port")
@click.option("--parallel",   default=lambda: int(_read_env("N_PARALLEL", "4")),         show_default=True, help="Parallel AI slots")
@click.option("--threads",    default=lambda: int(_read_env("N_THREADS", str(os.cpu_count() or 4))), show_default=True, help="CPU threads")
@click.option("--ctx",        default=lambda: int(_read_env("N_CTX", "4096")),           show_default=True, help="Context length")
def up(api_port, ctrl_port, parallel, threads, ctx):
    """Start everything: controller + AI server (if model is configured).

    \b
    This is the one-command startup. Run once, then use the browser UI.

      bikai up                  # start all with defaults from .env
      bikai up --api-port 8080  # custom port

    The controller UI is always available at http://localhost:<ctrl-port>/controller/ui
    From there you can download models, start/stop the AI server, and manage nginx.
    """
    _header("Starting Bik AI")

    # ── 1. Controller (always) ────────────────────────────────────────────
    ctrl_already = _ctrl_is_running()
    if ctrl_already:
        _warn(f"Controller already running on port {_read_env('CONTROLLER_PORT', str(ctrl_port))}")
    else:
        _write_env("CONTROLLER_PORT", str(ctrl_port))
        cmd = [
            sys.executable,
            str(Path(__file__).parent / "controller.py"),
            "--port", str(ctrl_port),
        ]
        log_path = Path("bikai-controller.log")
        proc = subprocess.Popen(cmd, stdout=log_path.open("w"), stderr=subprocess.STDOUT, start_new_session=True)
        CTRL_PID_FILE.write_text(str(proc.pid))
        _ok(f"Controller started (PID {proc.pid})")

        # Wait up to 10s for controller to be ready
        for _ in range(10):
            try:
                if httpx.get(f"http://localhost:{ctrl_port}/controller/health", timeout=1).status_code == 200:
                    break
            except Exception:
                pass
            time.sleep(1)
        else:
            _warn("Controller didn't respond in 10s — check logs: tail -f bikai-controller.log")

    # ── 2. AI server (only if model is configured) ────────────────────────
    model = _read_env("MODEL_PATH")
    ai_started = False

    if not model:
        click.echo()
        _info("No MODEL_PATH in .env — AI server not started.")
        _info("Download a model from the controller UI, then the server will start automatically.")
    else:
        # Resolve path
        if not Path(model).is_file():
            candidates = [MODELS_DIR / model, MODELS_DIR / (model + ".gguf")]
            model = next((str(p) for p in candidates if p.is_file()), model)

        if not Path(model).is_file():
            _warn(f"Model file not found: {model}")
            _info("Download a model from the controller UI first.")
        elif _is_running():
            _warn(f"AI server already running on port {_read_env('PORT', str(api_port))}")
        else:
            _write_env("PORT", str(api_port))
            cmd = [
                sys.executable, "server.py",
                "--model",    model,
                "--parallel", str(parallel),
                "--port",     str(api_port),
                "--ctx",      str(ctx),
                "--threads",  str(threads),
            ]
            log = LOG_FILE.open("w")
            proc = subprocess.Popen(cmd, stdout=log, stderr=log, start_new_session=True)
            PID_FILE.write_text(str(proc.pid))
            _ok(f"AI server started  (PID {proc.pid})  model: {Path(model).name}")
            ai_started = True

            # Wait up to 60s for AI server (model loading takes time)
            _info("Waiting for AI server to load model (this may take 30-60s)...")
            for _ in range(60):
                try:
                    if httpx.get(f"http://localhost:{api_port}/health", timeout=1).status_code == 200:
                        break
                except Exception:
                    pass
                time.sleep(1)
            else:
                _warn("AI server didn't respond in 60s — check logs: tail -f bikai-server.log")

    # ── Summary ───────────────────────────────────────────────────────────
    click.echo()
    click.echo(f"  {click.style('──────────────────────────────────────', fg='cyan')}")
    click.echo(f"  {click.style('Bik AI is running!', fg='green', bold=True)}")
    click.echo(f"  {click.style('──────────────────────────────────────', fg='cyan')}")
    click.echo()
    click.echo(f"  Controller UI  : {click.style(f'http://localhost:{ctrl_port}/controller/ui', fg='cyan')}")
    if ai_started or _is_running():
        click.echo(f"  AI API         : {click.style(f'http://localhost:{api_port}', fg='cyan')}")
        click.echo(f"  API docs       : {click.style(f'http://localhost:{api_port}/docs', fg='cyan')}")
    click.echo()
    click.echo(f"  Logs:  tail -f bikai-controller.log   tail -f bikai-server.log")
    click.echo(f"  Stop:  {click.style('bikai down', bold=True)}")
    click.echo()


# ---------------------------------------------------------------------------
# down  —  stop everything
# ---------------------------------------------------------------------------

@cli.command()
def down():
    """Stop everything: AI server + controller.

    \b
    Stops both the AI inference server and the controller UI server.
    """
    _header("Stopping Bik AI")

    # Stop AI server
    pid = _read_pid()
    if pid:
        for sig in (signal.SIGTERM, signal.SIGKILL):
            try:
                os.kill(pid, sig)
                time.sleep(0.8)
            except ProcessLookupError:
                break
        PID_FILE.unlink(missing_ok=True)
        _ok(f"AI server (PID {pid}) stopped.")
    else:
        port = _read_env("PORT", "8000")
        result = subprocess.run(["fuser", f"{port}/tcp"], capture_output=True, text=True)
        for p in result.stdout.strip().split():
            try:
                os.kill(int(p), signal.SIGTERM)
                _ok(f"AI server (PID {p}) stopped.")
            except (ValueError, ProcessLookupError):
                pass

    # Stop controller
    ctrl_pid = _read_ctrl_pid()
    if ctrl_pid:
        for sig in (signal.SIGTERM, signal.SIGKILL):
            try:
                os.kill(ctrl_pid, sig)
                time.sleep(0.8)
            except ProcessLookupError:
                break
        CTRL_PID_FILE.unlink(missing_ok=True)
        _ok(f"Controller (PID {ctrl_pid}) stopped.")
    else:
        port = _read_env("CONTROLLER_PORT", "8001")
        result = subprocess.run(["fuser", f"{port}/tcp"], capture_output=True, text=True)
        for p in result.stdout.strip().split():
            try:
                os.kill(int(p), signal.SIGTERM)
                _ok(f"Controller (PID {p}) stopped.")
            except (ValueError, ProcessLookupError):
                pass

    _ok("All stopped.")
    click.echo()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    cli()
