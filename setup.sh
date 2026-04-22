#!/usr/bin/env bash
# ============================================================
#  Bik AI — Single-file installer
#  Usage:
#    bash setup.sh
#  Or, one-liner from anywhere:
#    curl -fsSL https://raw.githubusercontent.com/bikirandev/BikAI-Local-Server/main/setup.sh | bash
# ============================================================
set -e

INSTALL_DIR="$HOME/.bikai"
REPO="https://github.com/bikirandev/BikAI-Local-Server.git"
BIN_DIR="$HOME/.local/bin"
MIN_PY_MINOR=10   # Python 3.10+

# ── Colours ─────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; CYAN='\033[0;36m'
YELLOW='\033[1;33m'; BOLD='\033[1m'; NC='\033[0m'

ok()     { echo -e "  ${GREEN}✓${NC}  $1"; }
info()   { echo -e "  ${CYAN}→${NC}  $1"; }
warn()   { echo -e "  ${YELLOW}!${NC}  $1"; }
err()    { echo -e "  ${RED}✗${NC}  $1" >&2; exit 1; }
header() { echo -e "\n  ${CYAN}──────────────────────────────────────────────────${NC}\n  ${BOLD}$1${NC}\n  ${CYAN}──────────────────────────────────────────────────${NC}"; }
step()   { echo -e "\n  ${BOLD}[$1]${NC} $2"; }

# ── Sudo handling ────────────────────────────────────────────
# If already root, SUDO is a no-op so we never prompt.
# If not root, cache credentials once upfront so apt/dnf never prompts mid-install.
if [[ $EUID -eq 0 ]]; then
  SUDO=""
else
  SUDO="sudo"
  if sudo -n true 2>/dev/null; then
    : # passwordless sudo — no prompt needed
  else
    echo -e "  ${YELLOW}This installer needs sudo to install system packages.${NC}"
    echo -e "  ${YELLOW}Please enter your password once:${NC}"
    sudo -v || err "sudo access required. Re-run as root or grant sudo privileges."
    # Keep sudo timestamp alive in background for the duration of the script
    ( while true; do sudo -n true; sleep 50; done ) &
    SUDO_KEEPER_PID=$!
    trap 'kill $SUDO_KEEPER_PID 2>/dev/null' EXIT
  fi
fi

# ── Banner ───────────────────────────────────────────────────
echo ""
echo -e "  ${CYAN}██████╗ ██╗██╗  ██╗     █████╗ ██╗${NC}"
echo -e "  ${CYAN}██╔══██╗██║██║ ██╔╝    ██╔══██╗██║${NC}"
echo -e "  ${CYAN}██████╔╝██║█████╔╝     ███████║██║${NC}"
echo -e "  ${CYAN}██╔══██╗██║██╔═██╗     ██╔══██║██║${NC}"
echo -e "  ${CYAN}██████╔╝██║██║  ██╗    ██║  ██║██║${NC}"
echo -e "  ${CYAN}╚═════╝ ╚═╝╚═╝  ╚═╝    ╚═╝  ╚═╝╚═╝${NC}"
echo ""
echo -e "  ${BOLD}Bik AI Installer${NC}  —  by bikiran.com"
echo ""

# ── Detect OS / Package manager ─────────────────────────────
detect_os() {
  if   [[ "$OSTYPE" == "darwin"* ]];        then echo "macos"
  elif command -v apt-get &>/dev/null;       then echo "debian"
  elif command -v dnf     &>/dev/null;       then echo "fedora"
  elif command -v yum     &>/dev/null;       then echo "rhel"
  elif command -v pacman  &>/dev/null;       then echo "arch"
  elif command -v zypper  &>/dev/null;       then echo "suse"
  else                                            echo "unknown"
  fi
}
OS=$(detect_os)
info "Detected OS: $OS"

# ── Helpers ──────────────────────────────────────────────────
pkg_install() {
  case $OS in
    debian) $SUDO apt-get install -y -qq "$@" ;;
    fedora) $SUDO dnf install -y -q "$@" ;;
    rhel)   $SUDO yum install -y -q "$@" ;;
    arch)   $SUDO pacman -Sy --noconfirm "$@" ;;
    suse)   $SUDO zypper install -y "$@" ;;
    macos)
      if ! command -v brew &>/dev/null; then
        info "Installing Homebrew..."
        /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
      fi
      brew install "$@"
      ;;
    *) err "Cannot auto-install packages on this OS. Install manually: $*" ;;
  esac
}

# ── Step 1: Python ───────────────────────────────────────────
step "1/5" "Checking Python"

ensure_python() {
  if command -v python3 &>/dev/null; then
    VER=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    MINOR=$(echo "$VER" | cut -d. -f2)
    if [[ $MINOR -ge $MIN_PY_MINOR ]]; then
      ok "Python $VER found"; return
    fi
    warn "Python $VER found but 3.${MIN_PY_MINOR}+ required. Installing newer version..."
  else
    info "Python 3 not found. Installing..."
  fi

  case $OS in
    debian) pkg_install python3 python3-pip python3-venv ;;
    fedora) pkg_install python3 python3-pip ;;
    rhel)   pkg_install python3 python3-pip ;;
    arch)   pkg_install python python-pip ;;
    suse)   pkg_install python3 python3-pip ;;
    macos)  pkg_install python@3.12 ;;
    *)      err "Install Python 3.10+ manually from https://python.org" ;;
  esac
  ok "Python installed"
}
ensure_python

# ── Step 2: System build deps (for llama-cpp-python) ────────
step "2/5" "Checking build tools"

ensure_build_deps() {
  local need_cmake=false need_nginx=false need_blas=false need_cc=false
  command -v cmake   &>/dev/null || need_cmake=true
  command -v nginx   &>/dev/null || need_nginx=true
  command -v gcc     &>/dev/null || need_cc=true
  pkg-config --exists openblas 2>/dev/null   || need_blas=true

  if ! $need_cmake && ! $need_nginx && ! $need_blas && ! $need_cc; then
    ok "Build tools already installed — skipping"; return
  fi

  case $OS in
    debian) pkg_install build-essential cmake libopenblas-dev pkg-config nginx ;;
    fedora) pkg_install gcc gcc-c++ cmake openblas-devel nginx ;;
    rhel)   $SUDO yum groupinstall -y "Development Tools" &>/dev/null; pkg_install cmake openblas-devel nginx ;;
    arch)   pkg_install base-devel cmake openblas nginx ;;
    suse)   pkg_install gcc gcc-c++ cmake openblas-devel nginx ;;
    macos)  pkg_install cmake openblas; xcode-select --install 2>/dev/null || true ;;
    *)      warn "Skipping build tools. If install fails, install cmake + openblas manually." ;;
  esac
  ok "Build tools ready"
}
ensure_build_deps

# ── Step 3: git + clone / update ────────────────────────────
step "3/5" "Downloading Bik AI"

if ! command -v git &>/dev/null; then
  info "git not found. Installing..."
  case $OS in
    debian) pkg_install git ;;
    fedora) pkg_install git ;;
    rhel)   pkg_install git ;;
    arch)   pkg_install git ;;
    macos)  pkg_install git ;;
    *)      err "Install git manually." ;;
  esac
fi

if [[ -d "$INSTALL_DIR/.git" ]]; then
  info "Updating existing installation at $INSTALL_DIR ..."
  git -C "$INSTALL_DIR" pull --quiet
  ok "Updated to latest version"
else
  info "Cloning to $INSTALL_DIR ..."
  git clone --quiet "$REPO" "$INSTALL_DIR"
  ok "Downloaded"
fi

# ── Step 4: Virtual env + packages ──────────────────────────
step "4/5" "Installing Python packages"

cd "$INSTALL_DIR"
python3 -m venv venv
ok "Virtual environment created"

info "Installing packages (this takes 5-15 min — compiling llama-cpp-python)..."
CMAKE_ARGS="-DGGML_BLAS=ON -DGGML_BLAS_VENDOR=OpenBLAS" \
  "$INSTALL_DIR/venv/bin/pip" install -e . -q
ok "All packages installed"

# ── Step 5: bikai command ────────────────────────────────────
step "5/5" "Installing bikai command"

mkdir -p "$BIN_DIR"

cat > "$BIN_DIR/bikai" <<LAUNCHER
#!/usr/bin/env bash
cd "$INSTALL_DIR"
exec "$INSTALL_DIR/venv/bin/python" "$INSTALL_DIR/cli.py" "\$@"
LAUNCHER
chmod +x "$BIN_DIR/bikai"
ok "Created: $BIN_DIR/bikai"

# Add BIN_DIR to PATH in shell rc
add_to_path() {
  local RC="$1"
  if [[ -f "$RC" ]] && grep -q "$BIN_DIR" "$RC" 2>/dev/null; then
    return  # already there
  fi
  echo "" >> "$RC"
  echo "# Bik AI" >> "$RC"
  echo "export PATH=\"\$HOME/.local/bin:\$PATH\"" >> "$RC"
  ok "Added ~/.local/bin to PATH in $RC"
}

ADDED=false
[[ "$SHELL" == *"zsh"*  ]] && add_to_path "$HOME/.zshrc"  && ADDED=true
[[ "$SHELL" == *"bash"* ]] && add_to_path "$HOME/.bashrc" && ADDED=true
[[ "$SHELL" == *"bash"* ]] && [[ -f "$HOME/.bash_profile" ]] && add_to_path "$HOME/.bash_profile"
if ! $ADDED; then
  add_to_path "$HOME/.profile"
fi

# ── Done ─────────────────────────────────────────────────────
echo ""
echo -e "  ${GREEN}${BOLD}══════════════════════════════════════════════════${NC}"
echo -e "  ${GREEN}${BOLD}  Bik AI installed successfully!${NC}"
echo -e "  ${GREEN}${BOLD}══════════════════════════════════════════════════${NC}"
echo ""
echo "  Next steps:"
echo ""

# Detect which RC was written
if   [[ "$SHELL" == *"zsh"*  ]]; then RC_FILE="$HOME/.zshrc"
elif [[ "$SHELL" == *"bash"* ]]; then RC_FILE="$HOME/.bashrc"
else                                   RC_FILE="$HOME/.profile"
fi

# Source the RC file so bikai is available immediately in this session
export PATH="$BIN_DIR:$PATH"
# shellcheck disable=SC1090
[[ -f "$RC_FILE" ]] && source "$RC_FILE" 2>/dev/null || true
ok "Shell reloaded — bikai is ready to use"

# ── Auto-setup: model + nginx + start ────────────────────────
echo ""
header "Auto Setup"

GDRIVE_ID="1kO_KTjQ-GcaarzLxqXnUyJkEmbM6UC3d"
API_PORT="8000"
PARALLEL="4"

# Step A: Download model from Google Drive (skip if already present)
step "A" "Checking model..."
MODEL_KNOWN="$INSTALL_DIR/models/gdrive_gemma3-4b.gguf"
if [[ -f "$MODEL_KNOWN" ]]; then
  ok "Model already exists — skipping download"
else
  info "Downloading model from Google Drive..."
  "$BIN_DIR/bikai" download -g "$GDRIVE_ID" || {
    warn "Model download failed. You can retry with:  bikai download -g $GDRIVE_ID"
  }
fi

# Detect downloaded model (prefer known name, fall back to first .gguf found)
if [[ -f "$MODEL_KNOWN" ]]; then
  MODEL_FILE="$MODEL_KNOWN"
else
  MODEL_FILE=$(find "$INSTALL_DIR/models" -name "*.gguf" | sort | head -1)
fi
if [[ -z "$MODEL_FILE" ]]; then
  warn "No model found in $INSTALL_DIR/models — start the server manually after downloading a model."
else
  ok "Model: $(basename "$MODEL_FILE")"

  # Save model path to .env
  "$INSTALL_DIR/venv/bin/python" - <<PYEOF
from dotenv import set_key
set_key("$INSTALL_DIR/.env", "MODEL_PATH", "$MODEL_FILE")
set_key("$INSTALL_DIR/.env", "PORT", "$API_PORT")
set_key("$INSTALL_DIR/.env", "N_PARALLEL", "$PARALLEL")
PYEOF

  # Step B: Start the server in daemon mode
  step "B" "Starting Bik AI server (parallel=$PARALLEL, port=$API_PORT)..."
  # Disable set -e for the start command — we handle failure ourselves
  set +e
  "$BIN_DIR/bikai" start \
    --model "$MODEL_FILE" \
    --parallel "$PARALLEL" \
    --port "$API_PORT" \
    --daemon
  START_EXIT=$?
  set -e

  if [[ $START_EXIT -ne 0 ]]; then
    warn "bikai start returned an error. Last log lines:"
    echo ""
    tail -30 "$INSTALL_DIR/bikai-server.log" 2>/dev/null | sed 's/^/    /' || true
    echo ""
    warn "Fix the issue then run:  bikai start --model '$MODEL_FILE' --parallel $PARALLEL --daemon"
  else
    # Wait up to 30s for the health endpoint to respond
    info "Waiting for server to be ready..."
    READY=false
    for i in $(seq 1 30); do
      if curl -sf "http://localhost:$API_PORT/health" &>/dev/null; then
        READY=true
        break
      fi
      sleep 1
    done

    if $READY; then
      ok "Server is up and healthy"
    else
      warn "Server did not respond after 30 seconds. Last log lines:"
      echo ""
      tail -30 "$INSTALL_DIR/bikai-server.log" 2>/dev/null | sed 's/^/    /' || true
      echo ""
      warn "Fix the issue then run:  bikai start --model '$MODEL_FILE' --parallel $PARALLEL --daemon"
    fi
  fi
fi

# Step C: Setup nginx
step "C" "Configuring nginx reverse proxy..."
"$BIN_DIR/bikai" nginx --port "$API_PORT" || {
  warn "nginx setup failed. Run manually:  bikai nginx"
}

# Get public IP for display
PUBLIC_IP=$(curl -s --max-time 5 https://api.ipify.org 2>/dev/null || echo "your-server-ip")

echo ""
echo -e "  ${GREEN}${BOLD}══════════════════════════════════════════════════${NC}"
echo -e "  ${GREEN}${BOLD}  Bik AI is live!${NC}"
echo -e "  ${GREEN}${BOLD}══════════════════════════════════════════════════${NC}"
echo ""
echo -e "  Public IP  : ${CYAN}http://$PUBLIC_IP${NC}"
echo -e "  Server info: ${CYAN}http://$PUBLIC_IP/server/info${NC}"
echo -e "  API docs   : ${CYAN}http://$PUBLIC_IP/docs${NC}"
echo -e "  Health     : ${CYAN}http://$PUBLIC_IP/health${NC}"
echo ""
echo -e "  Your API key:"
"$BIN_DIR/bikai" token show 2>/dev/null || true
echo ""
echo -e "  ${CYAN}Full help:  bikai -h${NC}"
echo ""
