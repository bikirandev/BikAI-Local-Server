# Bik AI — Local LLM Server

Run AI models locally on any Linux or macOS machine (no GPU required) and expose them as an OpenAI-compatible REST API.

Built by [bikiran.com](https://bikiran.com)

---

## Requirements

- Linux (Ubuntu 20.04+, Debian, Fedora, Arch, openSUSE) or macOS
- 8 GB RAM minimum (16 GB recommended for 4B models)
- No GPU needed — runs on CPU

---

## Install (fresh server)

```bash
curl -fsSL https://raw.githubusercontent.com/bikirandev/BikAI-Local-Server/main/setup.sh | bash
```

The installer automatically:
- Installs Python 3.10+, cmake, OpenBLAS, git
- Clones the repo to `~/.bikai/`
- Creates a Python virtualenv
- Compiles and installs `llama-cpp-python` (CPU-optimised with OpenBLAS)
- Installs nginx
- Creates the `bikai` command in `~/.local/bin/`
- Reloads your shell so `bikai` works immediately

---

## Quick Start

### 1. Download a model

```bash
# Gemma 2 2B — fast, ~1.6 GB RAM
bikai download -r bartowski/gemma-2-2b-it-GGUF -f gemma-2-2b-it-Q4_K_M.gguf

# Gemma 3 4B — smarter, ~3 GB RAM
bikai download -r bartowski/google_gemma-3-4b-it-GGUF -f google_gemma-3-4b-it-Q4_K_M.gguf
```

### 2. Start the server

```bash
# Foreground (see logs live)
bikai start --model ~/.bikai/models/gemma-2-2b-it-Q4_K_M.gguf

# Background (daemon)
bikai start --model ~/.bikai/models/gemma-2-2b-it-Q4_K_M.gguf --daemon
```

### 3. Check it works

```bash
bikai status
```

```
  Status    : RUNNING
  Port      : 8000
  Model     : gemma-2-2b-it-Q4_K_M.gguf
  Local URL : http://localhost:8000
```

---

## All Commands

### Server control

```bash
bikai start    # start the server (see options below)
bikai stop     # stop the server
bikai restart  # stop + start in daemon mode
bikai status   # show status, PID, model, URL
bikai logs     # show last 50 lines of server log
bikai logs -f  # follow live log output
```

### `bikai start` options

| Option | Default | Description |
|--------|---------|-------------|
| `--model` / `-m` | `MODEL_PATH` from `.env` | Path to GGUF model file |
| `--parallel` / `-p` | `4` | How many users can run at once |
| `--port` | `8000` | Port to listen on |
| `--ctx` | `4096` | Context window in tokens |
| `--threads` | all cores | CPU threads for inference |
| `--daemon` / `-d` | off | Run in background |

### nginx (public access)

```bash
bikai nginx --domain api.example.com         # HTTP reverse proxy on port 80
bikai nginx --domain api.example.com --ssl   # HTTPS via Let's Encrypt
bikai nginx --status                         # check nginx status
```

### Models

```bash
bikai models          # list downloaded models
bikai download ...    # download a model (see below)
```

### API key

```bash
bikai token show      # show current API key
bikai token new       # generate a new API key
```

### Info

```bash
bikai url             # show local + public URL
bikai config          # show all settings
```

---

## Downloading Models

### From HuggingFace

```bash
bikai download -r <repo> -f <filename>
```

```bash
# Examples
bikai download -r bartowski/gemma-2-2b-it-GGUF      -f gemma-2-2b-it-Q4_K_M.gguf
bikai download -r bartowski/google_gemma-3-4b-it-GGUF -f google_gemma-3-4b-it-Q4_K_M.gguf
```

### From a direct URL (S3, Cloudflare R2, VPS, etc.)

```bash
bikai download -u https://your-server.com/model.gguf
```

### From Google Drive

```bash
# Use the file ID from the share link
bikai download -g 1aBcDeFgHiJkLmNoPqRsTuV
# Or paste the full share URL
bikai download -g "https://drive.google.com/file/d/1aBcDeFgHiJkLmNoPqRsTuV/view"
```

After downloading, you'll be asked whether to set it as the default model in `.env`.

---

## Recommended Models

| Model | Repo | File | Size | Best for |
|-------|------|------|------|----------|
| Gemma 2 2B | `bartowski/gemma-2-2b-it-GGUF` | `gemma-2-2b-it-Q4_K_M.gguf` | 1.6 GB | Fast responses, low RAM |
| Gemma 3 4B | `bartowski/google_gemma-3-4b-it-GGUF` | `google_gemma-3-4b-it-Q4_K_M.gguf` | 2.8 GB | Better quality |

RAM guide — parallel slots:
- 2B model → up to 6 parallel users
- 4B model → up to 4 parallel users
- 9B model → up to 2 parallel users

---

## Parallel Workers

`--parallel N` controls how many requests run at the same time. Each slot loads a separate model instance into RAM.

```bash
bikai start --model ./models/gemma-2-2b-it-Q4_K_M.gguf --parallel 4
```

For a 16 GB RAM machine running the 2B model, `--parallel 6` is a good maximum.

---

## Public Access via nginx

nginx is installed automatically by the installer.

### HTTP

```bash
bikai nginx --domain api.example.com
```

### HTTPS (Let's Encrypt)

```bash
bikai nginx --domain api.example.com --ssl
```

Requirements for SSL:
- Your domain's DNS A record must point to this server's IP
- Port 80 and 443 must be open in your firewall

The command:
1. Writes an nginx config with CORS headers and SSE streaming support
2. Enables the site and reloads nginx
3. Runs `certbot --nginx` to get a free TLS certificate and auto-redirect HTTP → HTTPS
4. Saves the domain to `.env`

```bash
bikai url   # shows https://api.example.com
```

---

## Configuration (.env)

All settings can be stored in `.env` in the install directory (`~/.bikai/.env`):

```env
MODEL_PATH=~/.bikai/models/gemma-2-2b-it-Q4_K_M.gguf
API_KEY=your-api-key
N_PARALLEL=4
N_CTX=4096
N_THREADS=8
PORT=8000
RATE_LIMIT=30/minute
DOMAIN=api.example.com
```

View current config:

```bash
bikai config
```

---

## API Usage

The server is OpenAI-compatible. Include your API key in every request:

```
X-API-Key: <your-key>
```

Get your key with `bikai token show`.

### Chat completions

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "X-API-Key: YOUR_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Hello!"}],
    "stream": false
  }'
```

### Streaming

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "X-API-Key: YOUR_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Write a poem about the sea"}],
    "stream": true
  }'
```

### Simple generate

```bash
curl http://localhost:8000/generate \
  -H "X-API-Key: YOUR_KEY" \
  -H "Content-Type: application/json" \
  -d '{"prompt": "The capital of France is"}'
```

### Health check (no auth)

```bash
curl http://localhost:8000/health
```

### List models

```bash
curl http://localhost:8000/v1/models -H "X-API-Key: YOUR_KEY"
```

### Interactive API docs

```
http://localhost:8000/docs
```

---

## Update

```bash
curl -fsSL https://raw.githubusercontent.com/bikirandev/BikAI-Local-Server/main/setup.sh | bash
```

Re-running the installer updates the code without touching your `.env` or downloaded models.

---

## Uninstall

```bash
rm -rf ~/.bikai ~/.local/bin/bikai
```

Then remove the `export PATH` line added to your `~/.bashrc` / `~/.zshrc`.
