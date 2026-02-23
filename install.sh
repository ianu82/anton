#!/bin/sh
# Anton install script — curl -sSf https://raw.githubusercontent.com/mindsdb/anton/main/install.sh | sh
# Pure POSIX sh, no sudo, idempotent.

set -e

CYAN='\033[36m'
GREEN='\033[32m'
RED='\033[31m'
BOLD='\033[1m'
RESET='\033[0m'

LOCAL_BIN="$HOME/.local/bin"
REPO_URL="git+https://github.com/mindsdb/anton.git"

info()  { printf "%b\n" "$1"; }
error() { printf "${RED}error:${RESET} %s\n" "$1" >&2; }

# ── 1. Branded logo ────────────────────────────────────────────────
info ""
info "${CYAN} ▄▀█ █▄ █ ▀█▀ █▀█ █▄ █${RESET}"
info "${CYAN} █▀█ █ ▀█  █  █▄█ █ ▀█${RESET}"
info "${CYAN} autonomous coworker${RESET}"
info ""

# ── 2. Check prerequisites ──────────────────────────────────────────
if ! command -v git >/dev/null 2>&1; then
    error "git is required but not found."
    info "  Install it with your package manager:"
    info "    macOS:  xcode-select --install"
    info "    Ubuntu: sudo apt install git"
    info "    Fedora: sudo dnf install git"
    exit 1
fi

if ! command -v curl >/dev/null 2>&1; then
    error "curl is required but not found."
    info "  Install it with your package manager:"
    info "    Ubuntu: sudo apt install curl"
    info "    Fedora: sudo dnf install curl"
    exit 1
fi

# ── 3. Find or install uv ──────────────────────────────────────────
if command -v uv >/dev/null 2>&1; then
    info "  Found uv: $(command -v uv)"
elif [ -f "$HOME/.local/bin/uv" ]; then
    export PATH="$LOCAL_BIN:$PATH"
    info "  Found uv: $HOME/.local/bin/uv"
elif [ -f "$HOME/.cargo/bin/uv" ]; then
    export PATH="$HOME/.cargo/bin:$PATH"
    info "  Found uv: $HOME/.cargo/bin/uv"
else
    info "  Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1
    # Source uv's env setup if available
    if [ -f "$HOME/.local/bin/env" ]; then
        . "$HOME/.local/bin/env"
    else
        export PATH="$LOCAL_BIN:$PATH"
    fi
    info "  Installed uv: $(command -v uv)"
fi

# ── 4. Install anton via uv tool ───────────────────────────────────
info "  Installing anton..."
uv tool install "$REPO_URL" --force
info "  Installed anton"

# ── 5. Ensure ~/.local/bin is in PATH ──────────────────────────────
ensure_path() {
    # Check if ~/.local/bin is already in PATH
    case ":$PATH:" in
        *":$LOCAL_BIN:"*) return ;;
    esac

    # Detect shell config file
    SHELL_NAME="$(basename "$SHELL" 2>/dev/null || echo "sh")"
    case "$SHELL_NAME" in
        zsh)  SHELL_RC="$HOME/.zshrc" ;;
        bash)
            if [ -f "$HOME/.bash_profile" ]; then
                SHELL_RC="$HOME/.bash_profile"
            else
                SHELL_RC="$HOME/.bashrc"
            fi
            ;;
        fish) SHELL_RC="$HOME/.config/fish/config.fish" ;;
        *)    SHELL_RC="$HOME/.profile" ;;
    esac

    # Only append if not already present
    if [ -f "$SHELL_RC" ] && grep -qF '.local/bin' "$SHELL_RC" 2>/dev/null; then
        return
    fi

    if [ "$SHELL_NAME" = "fish" ]; then
        mkdir -p "$(dirname "$SHELL_RC")"
        printf '\n# Added by anton installer\nfish_add_path %s\n' "$LOCAL_BIN" >> "$SHELL_RC"
    else
        printf '\n# Added by anton installer\nexport PATH="$HOME/.local/bin:$PATH"\n' >> "$SHELL_RC"
    fi
    info "  Updated ${SHELL_RC}"
}

ensure_path

# ── 6. Success message ──────────────────────────────────────────────
info ""
info "${GREEN}  ✓ anton installed successfully!${RESET}"
info ""
info "  Open a new terminal, then:"
info ""
info "    anton                                          ${CYAN}# Dashboard${RESET}"
info "    anton run \"analyze last month's sales data\"    ${CYAN}# Give Anton a task${RESET}"
info ""
info "  Upgrade:    uv tool upgrade anton"
info "  Uninstall:  uv tool uninstall anton"
info ""
info "  Config: ~/.anton/.env"
info ""
