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
  local need_cmake=false need_nginx=false need_blas=false need_cc=false need_node=false
  command -v cmake   &>/dev/null || need_cmake=true
  command -v nginx   &>/dev/null || need_nginx=true
  command -v gcc     &>/dev/null || need_cc=true
  command -v npm     &>/dev/null || need_node=true
  pkg-config --exists openblas 2>/dev/null   || need_blas=true

  if ! $need_cmake && ! $need_nginx && ! $need_blas && ! $need_cc && ! $need_node; then
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

  # Node.js — required to build the React controller UI
  if ! command -v npm &>/dev/null; then
    info "Installing Node.js..."
    case $OS in
      debian)
        # Use NodeSource LTS (Node 20)
        curl -fsSL https://deb.nodesource.com/setup_lts.x | $SUDO -E bash - &>/dev/null
        $SUDO apt-get install -y -qq nodejs
        ;;
      fedora) $SUDO dnf install -y -q nodejs npm ;;
      rhel)   $SUDO yum install -y -q nodejs npm ;;
      arch)   $SUDO pacman -Sy --noconfirm nodejs npm ;;
      suse)   $SUDO zypper install -y nodejs npm ;;
      macos)  brew install node ;;
      *)      warn "Could not install Node.js automatically. Install Node.js 18+ and re-run." ;;
    esac
    command -v npm &>/dev/null && ok "Node.js installed" || warn "Node.js install failed — UI build will be skipped"
  else
    ok "Node.js already installed"
  fi
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
# Install requirements.txt first (with CMAKE_ARGS so llama-cpp-python compiles with OpenBLAS)
CMAKE_ARGS="-DGGML_BLAS=ON -DGGML_BLAS_VENDOR=OpenBLAS" \
  "$INSTALL_DIR/venv/bin/pip" install -r "$INSTALL_DIR/requirements.txt" -q
# Install CLI entry point
"$INSTALL_DIR/venv/bin/pip" install -e "$INSTALL_DIR" -q
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
CTRL_PORT="8001"
PARALLEL="4"

# Step A: Generate API key upfront so it's ready for the summary
step "A" "Generating API key..."
"$INSTALL_DIR/venv/bin/python" - <<PYEOF
import secrets, os
from pathlib import Path
from dotenv import set_key
env = Path("$INSTALL_DIR/.env")
env.touch(exist_ok=True)
# Only generate if not already set
current = ""
try:
    from dotenv import dotenv_values
    current = dotenv_values(str(env)).get("API_KEY", "")
except Exception:
    pass
if not current:
    key = secrets.token_urlsafe(32)
    set_key(str(env), "API_KEY", key)
    print(f"Generated API key: {key}")
else:
    print(f"Existing API key: {current}")
PYEOF
ok "API key ready"

# Step B: Build the React UI
step "B" "Building Controller UI..."
if command -v npm &>/dev/null && [[ -d "$INSTALL_DIR/ui" ]]; then
  info "Installing UI dependencies..."
  pushd "$INSTALL_DIR/ui" > /dev/null
  npm install --silent --no-progress 2>/dev/null || npm install --silent 2>&1 | grep -E 'error|warn' | head -5 || true
  ok "Dependencies installed."
  info "Compiling UI (this takes ~10s)..."
  npm run build --silent 2>/dev/null || npm run build 2>&1 | grep -E 'error|built in|✓' | tail -5 || warn "UI build failed — controller will serve without UI assets"
  popd > /dev/null
  ok "UI compiled."
else
  warn "npm not found — skipping UI build. Install Node.js 18+ and re-run setup.sh."
fi

# Step C: Start everything with a single bikai up
step "C" "Starting controller + AI server..."
set +e
"$BIN_DIR/bikai" up --api-port "$API_PORT" --ctrl-port "$CTRL_PORT" --parallel "$PARALLEL"
UP_EXIT=$?
set -e

if [[ $UP_EXIT -ne 0 ]]; then
  warn "bikai up failed. Try manually:  bikai up"
fi

# Step D: Setup nginx (proxies both / → AI port and /controller → controller port)
step "D" "Configuring nginx reverse proxy..."
"$BIN_DIR/bikai" nginx --port "$API_PORT" --ctrl-port "$CTRL_PORT" || {
  warn "nginx setup failed. Run manually:  bikai nginx"
}

# Step E: Create systemd services for auto-start on reboot
step "E" "Setting up auto-start on reboot (systemd)..."
if command -v systemctl &>/dev/null && systemctl is-system-running --quiet 2>/dev/null || \
   command -v systemctl &>/dev/null && [[ -d /run/systemd/system ]]; then

  SYSTEMD_DIR="/etc/systemd/system"
  CURRENT_USER="$(whoami)"

  # Controller service — always auto-start
  $SUDO tee "$SYSTEMD_DIR/bikai-controller.service" > /dev/null <<SERVICE_EOF
[Unit]
Description=BikAI Controller (management UI)
After=network.target
Wants=network-online.target

[Service]
Type=simple
User=$CURRENT_USER
WorkingDirectory=$INSTALL_DIR
ExecStart=$INSTALL_DIR/venv/bin/python $INSTALL_DIR/controller.py --port $CTRL_PORT
Restart=always
RestartSec=5
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
SERVICE_EOF

  # AI server service — auto-start only if a model is already configured
  if [[ -n "$MODEL_FILE" && -f "$MODEL_FILE" ]]; then
    $SUDO tee "$SYSTEMD_DIR/bikai.service" > /dev/null <<SERVICE_EOF
[Unit]
Description=BikAI Inference Server
After=network.target bikai-controller.service
Wants=bikai-controller.service

[Service]
Type=simple
User=$CURRENT_USER
WorkingDirectory=$INSTALL_DIR
ExecStart=$INSTALL_DIR/venv/bin/python $INSTALL_DIR/server.py --model $MODEL_FILE --parallel $PARALLEL --port $API_PORT
Restart=on-failure
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
SERVICE_EOF
    $SUDO systemctl daemon-reload
    $SUDO systemctl enable bikai-controller.service bikai.service 2>/dev/null
    ok "Systemd services enabled: bikai-controller + bikai (auto-start on reboot)"
  else
    $SUDO systemctl daemon-reload
    $SUDO systemctl enable bikai-controller.service 2>/dev/null
    ok "Systemd service enabled: bikai-controller (auto-start on reboot)"
    info "AI server will need to be started manually with: bikai start"
  fi
else
  warn "systemd not detected — skipping auto-start setup. Start manually with: bikai controller start"
fi

# Get public IP for display
PUBLIC_IP=$(curl -s --max-time 5 https://api.ipify.org 2>/dev/null || echo "your-server-ip")

# Read generated API key for display
API_KEY_DISPLAY=$(grep 'API_KEY' "$INSTALL_DIR/.env" 2>/dev/null | head -1 | cut -d'=' -f2- | tr -d "'\"" || echo "run: bikai token show")

echo ""
echo -e "  ${GREEN}${BOLD}══════════════════════════════════════════════════${NC}"
echo -e "  ${GREEN}${BOLD}  Bik AI is live!${NC}"
echo -e "  ${GREEN}${BOLD}══════════════════════════════════════════════════${NC}"
echo ""
echo -e "  Control Panel  : ${CYAN}${BOLD}http://$PUBLIC_IP/controller/ui${NC}"
echo -e "  AI API          : ${CYAN}http://$PUBLIC_IP${NC}"
echo ""
echo -e "  ${YELLOW}${BOLD}Your API key (save this now):${NC}"
echo -e "  ${GREEN}${BOLD}  $API_KEY_DISPLAY${NC}"
echo ""
echo -e "  Enter this key when prompted by the controller UI."
echo -e "  To get it again:  ${CYAN}bikai token show${NC}"
echo ""
echo -e "  ${BOLD}Next:${NC} Open the Control Panel and go to Models → Download"
echo -e "  Recommended model: ${CYAN}Gemma 3 4B (Q4_K_M, ~2.3 GB)${NC}"
echo -e "  Google Drive ID:   ${CYAN}${BOLD}1kO_KTjQ-GcaarzLxqXnUyJkEmbM6UC3d${NC}"
echo ""
echo -e "  ${CYAN}Full help:  bikai -h${NC}"
echo ""
