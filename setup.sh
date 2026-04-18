#!/usr/bin/env bash
set -euo pipefail

# ─────────────────────────────────────────────────
# GPT2.C — Setup Script
# ─────────────────────────────────────────────────

WEIGHTS_DIR="weights"
WEIGHTS_BASE_URL="https://huggingface.co/roeybh/gpt2-small-from-scratch-c/resolve/main"
TOKENIZER_URL="https://huggingface.co/openai-community/gpt2/resolve/main/tokenizer.json"

# Weight files: name -> filename
declare -A WEIGHT_FILES=(
    ["small"]="gpt2_c_weights.bin"
    ["medium"]="gpt2_medium_c_weights.bin"
    ["large"]="gpt2_large_c_weights.bin"
)

# Approximate download sizes
declare -A WEIGHT_SIZES=(
    ["small"]="~500 MB"
    ["medium"]="~1.4 GB"
    ["large"]="~3.0 GB"
)

# ─────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────

info()  { echo "==> $*"; }
ok()    { echo "  ✓ $*"; }
skip()  { echo "  - $* (already installed)"; }
err()   { echo "  ✗ ERROR: $*" >&2; }

detect_platform() {
    OS=$(uname -s)
    ARCH=$(uname -m)
    HAS_GPU=false

    if command -v nvidia-smi &>/dev/null && nvidia-smi &>/dev/null; then
        HAS_GPU=true
    fi

    echo ""
    info "Detected platform: $OS / $ARCH"
    if $HAS_GPU; then
        ok "NVIDIA GPU detected"
    else
        echo "  - No NVIDIA GPU detected (GPU build will not be available)"
    fi
    echo ""
}

# ─────────────────────────────────────────────────
# Step 1: Install uv
# ─────────────────────────────────────────────────

install_uv() {
    info "Step 1: Python package manager (uv)"

    if command -v uv &>/dev/null; then
        skip "uv $(uv --version 2>/dev/null | head -1)"
        return
    fi

    echo "  Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # Refresh PATH for this session
    export PATH="$HOME/.local/bin:$PATH"
    ok "uv installed"
}

# ─────────────────────────────────────────────────
# Step 2: Install system dependencies (jansson, etc)
# ─────────────────────────────────────────────────

install_system_deps() {
    info "Step 2: System dependencies (jansson, build tools)"

    case "$OS" in
        Darwin)
            if ! command -v brew &>/dev/null; then
                echo "  Homebrew is required on macOS. Install it with:"
                echo "    /bin/bash -c \"\$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\""
                exit 1
            fi

            if brew list jansson &>/dev/null; then
                skip "jansson"
            else
                echo "  Installing jansson via brew..."
                brew install jansson
                ok "jansson installed"
            fi
            ;;
        Linux)
            if command -v apt-get &>/dev/null; then
                # Check which packages are missing
                local pkgs_needed=()
                for pkg in build-essential libjansson-dev libopenblas-dev; do
                    if ! dpkg -s "$pkg" &>/dev/null; then
                        pkgs_needed+=("$pkg")
                    fi
                done

                if [ ${#pkgs_needed[@]} -eq 0 ]; then
                    skip "build-essential, libjansson-dev, libopenblas-dev"
                else
                    echo "  Installing ${pkgs_needed[*]} via apt..."
                    sudo apt-get install -y "${pkgs_needed[@]}"
                    ok "system packages installed"
                fi
            elif command -v brew &>/dev/null; then
                if brew list jansson &>/dev/null; then
                    skip "jansson"
                else
                    echo "  Installing jansson via brew..."
                    brew install jansson
                    ok "jansson installed"
                fi
            else
                err "No supported package manager found (apt or brew)."
                echo "       Please install jansson and openblas manually."
                exit 1
            fi
            ;;
        *)
            err "Unsupported OS: $OS"
            exit 1
            ;;
    esac
}

# ─────────────────────────────────────────────────
# Step 3: Download tokenizer
# ─────────────────────────────────────────────────

download_tokenizer() {
    info "Step 3: Tokenizer data"

    if [ -f "tokenizer.json" ]; then
        skip "tokenizer.json"
        return
    fi

    echo "  Downloading tokenizer.json..."
    curl -L -o tokenizer.json "$TOKENIZER_URL"
    ok "tokenizer.json downloaded"
}

# ─────────────────────────────────────────────────
# Step 4: Download model weights
# ─────────────────────────────────────────────────

download_weights() {
    info "Step 4: GPT-2 model weights"

    mkdir -p "$WEIGHTS_DIR"

    # Check which models are missing
    local missing=()
    for size in small medium large; do
        local file="${WEIGHT_FILES[$size]}"
        if [ -f "$WEIGHTS_DIR/$file" ]; then
            skip "$size weights"
        else
            missing+=("$size")
        fi
    done

    if [ ${#missing[@]} -eq 0 ]; then
        return
    fi

    echo ""
    echo "  Models available for download:"
    for size in "${missing[@]}"; do
        echo "    [$size] ${WEIGHT_SIZES[$size]}"
    done

    echo ""
    local default="${missing[0]}"
    read -rp "  Which models to download? (${missing[*]}/all/none) [$default]: " choice
    choice=${choice:-$default}

    if [ "$choice" = "none" ]; then
        echo "  Skipping weight downloads."
        return
    fi

    if [ "$choice" = "all" ]; then
        local sizes=("${missing[@]}")
    else
        IFS=',' read -ra sizes <<< "$choice"
    fi

    for size in "${sizes[@]}"; do
        size=$(echo "$size" | xargs)  # trim whitespace
        local file="${WEIGHT_FILES[$size]:-}"
        if [ -z "$file" ]; then
            err "Unknown model size: $size"
            continue
        fi

        if [ -f "$WEIGHTS_DIR/$file" ]; then
            skip "$size weights"
        else
            echo "  Downloading $size weights (${WEIGHT_SIZES[$size]})..."
            curl -L -o "$WEIGHTS_DIR/$file" "$WEIGHTS_BASE_URL/$file"
            ok "$size weights downloaded"
        fi
    done
}

# ─────────────────────────────────────────────────
# Step 5: Python environment
# ─────────────────────────────────────────────────

setup_python() {
    info "Step 5: Python environment (uv sync)"

    uv sync
    ok "Python environment ready"
}

# ─────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────

print_summary() {
    echo ""
    echo "========================================="
    echo "  Setup complete!"
    echo "========================================="
    echo ""
    echo "  Next steps:"
    echo ""
    echo "  1. Start the tokenizer server:"
    echo "     uv run python tokenizer.py"
    echo ""
    echo "     Or activate the environment first, then run directly:"
    echo "       source .venv/bin/activate"
    echo "       python tokenizer.py"
    echo ""
    echo "  2. Build and run (in another terminal):"
    echo ""
    echo "     CPU build:"
    echo "       make small          # GPT-2 Small (124M)"
    echo "       make medium         # GPT-2 Medium (355M)"
    echo "       make large          # GPT-2 Large (774M)"

    if $HAS_GPU; then
        echo ""
        echo "     GPU build (CUDA):"
        echo "       make gpu small     # GPT-2 Small (124M)"
        echo "       make gpu medium    # GPT-2 Medium (355M)"
        echo "       make gpu large     # GPT-2 Large (774M)"
    fi

    echo ""
    echo "  3. Run inference:"
    echo "     ./out/cpu/gpt2_small                                  # interactive mode"
    echo "     ./out/cpu/gpt2_small --prompt \"Once upon a time...\"   # one-shot mode"
    echo ""
    echo "     Available flags:"
    echo "       --prompt <text>           Input prompt (omit for interactive mode)"
    echo "       --req_out_tokens <n>      Number of output tokens to generate"
    echo "       --token_chunk_size <n>    Tokens per chunk for batched generation"
    echo "       --json_out_file <path>    Save performance metrics to JSON file"
    echo "       --no-stream               Disable streaming (show text at the end)"
    echo "       --verbose                 Show chunk-level stats during generation"
    echo ""
}

# ─────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────

echo ""
echo "╔═══════════════════════════════════════╗"
echo "║       GPT2.C — Project Setup          ║"
echo "╚═══════════════════════════════════════╝"

detect_platform
install_uv
install_system_deps
download_tokenizer
download_weights
setup_python
print_summary
