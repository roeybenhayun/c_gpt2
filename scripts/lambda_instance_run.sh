#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# lambda_instance_run.sh
#
# Runs DIRECTLY ON the Lambda H100 instance. Handles every step in the
# runbook (clone, setup, build, profiled benchmark run). After it finishes,
# manually rsync ~/c_gpt2/logs/ back to your laptop.
#
# Usage (typical flow):
#
#   # 1. Copy this script onto the instance:
#   scp scripts/lambda_instance_run.sh ubuntu@68.209.74.4:~
#
#   # 2. SSH in WITH AGENT FORWARDING so the GitHub clone uses your local key:
#   ssh -A ubuntu@<ip>
#
#   # 3. Run it (monitor live; ~10-15 min total):
#   bash ~/lambda_instance_run.sh
#
#   # 4. Back on your laptop, pull the logs (script prints the exact command
#   #    in its closing banner with the run-dir already templated):
#   cd ~/projects/c_gpt2
#   RUN_DIR="logs/lambda/h100/$(date +%Y%m%d_%H%M%S)_bf16_nsys_profile"
#   mkdir -p "$RUN_DIR"
#   rsync -avz --progress --exclude='*.sqlite' \
#     ubuntu@<ip>:~/c_gpt2/logs/ "$RUN_DIR/"
#
#   # 5. Terminate the instance from the Lambda dashboard.
#
# This script is a strict subset of lambda_h100_remote_run.sh — same pre-flight
# checks, same Makefile patch, same benchmarks. The difference is "you run
# rsync + terminate yourself" instead of orchestrating from the laptop.
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

log()  { printf '\n\033[1;36m==>\033[0m %s\n' "$*"; }
ok()   { printf '  \033[1;32m✓\033[0m %s\n' "$*"; }
warn() { printf '  \033[1;33m!\033[0m %s\n' "$*"; }
err()  { printf '  \033[1;31m✗\033[0m %s\n' "$*" >&2; }

# setup.sh installs uv into ~/.local/bin and exports PATH inside its own
# subshell only. Pre-pend now so subsequent `uv` invocations resolve, whether
# uv is being installed for the first time or was already present.
export PATH="$HOME/.local/bin:$PATH"

# ── 1. GPU + toolchain pre-flight ────────────────────────────────────────────
#
# Catch every "would we waste cloud minutes?" failure here, before the ~5 GB
# weight download in step 3. The nsys smoke test is the most important one —
# `--profile` invokes nsys via `eval`, so if nsys is missing or broken the
# benchmark JSONs still get produced and you only notice the missing
# .nsys-rep files after spending the full benchmark.

log "Pre-flight: GPU + toolchain"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
which nvcc || { err "nvcc not on PATH"; exit 1; }
nvcc --version | tail -2

if ! command -v nsys >/dev/null 2>&1; then
    warn "nsys not found, installing nsight-systems + target package"
    sudo apt-get update -qq
    # Lambda Stack 24.04 calls the package `nsight-systems` (NOT
    # `nsight-systems-cli` as on stock Ubuntu). The `-target` package is
    # what contains the QdstrmImporter binary that converts raw nsys
    # captures into .nsys-rep reports — without it, profiled runs produce
    # only .qdstrm files and `compare_profiles.py` can't read them.
    sudo apt-get install -y -qq nsight-systems nsight-systems-target
fi
nsys --version

# Even with both packages installed, nsys often can't *find* the importer
# because it ships at /usr/lib/nsight-systems/host-linux-x64/ but nsys
# looks for it as a sibling of its own bin/ dir. Symlink it next to nsys
# if the importer dir isn't already there.
NSYS_REAL="$(readlink -f "$(command -v nsys)")"
NSYS_PARENT="$(dirname "$(dirname "$NSYS_REAL")")"
if [ ! -e "$NSYS_PARENT/host-linux-x64" ] && [ -d /usr/lib/nsight-systems/host-linux-x64 ]; then
    warn "Linking QdstrmImporter into $NSYS_PARENT/ (Lambda packaging quirk)"
    sudo ln -sf /usr/lib/nsight-systems/host-linux-x64 "$NSYS_PARENT/host-linux-x64"
fi

# nsys can be installed and versioned yet still fail to produce a .nsys-rep
# (e.g. importer missing, libraries can't load). Smoke test with `sleep 0.5`
# (using `true` exits too fast for nsys 2024+ to attach) and
# --force-overwrite to handle stale files from re-runs.
rm -f /tmp/nsys_smoke.*
nsys profile --stats=false --force-overwrite=true -o /tmp/nsys_smoke sleep 0.5 >/dev/null 2>&1
if [ ! -f /tmp/nsys_smoke.nsys-rep ]; then
    err "nsys ran but produced no .nsys-rep — would silently waste benchmark time"
    err "Diagnose: ldd /usr/lib/nsight-systems/host-linux-x64/QdstrmImporter | grep 'not found'"
    exit 1
fi
rm -f /tmp/nsys_smoke.*
ok "GPU + toolchain ready (nvcc, nsys verified)"

# ── 2. Clone (or update) repo ────────────────────────────────────────────────

log "Cloning repo (relies on ssh -A agent forwarding for GitHub auth)"
cd ~
if [ -d c_gpt2/.git ]; then
    cd c_gpt2
    git fetch origin
    git checkout master
    git pull --ff-only
else
    git clone git@github.com:roeybenhayun/c_gpt2.git
    cd c_gpt2
fi
ok "Repo at $(git rev-parse --short HEAD) ($(git rev-parse --abbrev-ref HEAD))"

# ── 3. Setup script (non-interactive weight download) ────────────────────────

log "Running setup.sh (downloads ~5 GB of weights on first run)"
# Use `echo` (not `yes`) to answer the weight-prompt question. setup.sh
# reads stdin exactly once for that prompt and never again. With `yes all`
# the producer keeps running, gets SIGPIPE when setup.sh finishes, and
# `set -o pipefail` bubbles that 141 exit up — silently terminating THIS
# script right after a successful setup. `echo all` is single-shot and
# exits cleanly after writing one line.
echo all | ./setup.sh
export PATH="$HOME/.local/bin:$PATH"   # in case setup.sh just installed uv
ok "Setup complete"

# ── 4. Patch Makefile NVCC path (Lambda installs at /usr/bin/nvcc) ───────────

if grep -q '/usr/local/cuda-12.8/bin/nvcc' Makefile; then
    log "Patching Makefile NVCC path"
    sed -i 's|/usr/local/cuda-12.8/bin/nvcc|/usr/bin/nvcc|' Makefile
    ok "Patched: $(grep -n 'NVCC =' Makefile)"
else
    ok "Makefile already patched"
fi

# ── 5. Start tokenizer in background, wait for port to bind ──────────────────

log "Starting tokenizer server in background"
pkill -f 'python tokenizer.py' 2>/dev/null || true
sleep 1
nohup uv run python tokenizer.py > /tmp/tokenizer.log 2>&1 &
TOKENIZER_PID=$!

for i in $(seq 1 30); do
    if (echo > /dev/tcp/127.0.0.1/65432) 2>/dev/null; then
        ok "Tokenizer up on 127.0.0.1:65432 (pid $TOKENIZER_PID)"
        break
    fi
    if [ "$i" = "30" ]; then
        err "Tokenizer failed to bind on port 65432 within 30 sec"
        tail /tmp/tokenizer.log
        exit 1
    fi
    sleep 1
done

# Always clean up the tokenizer when this script exits, even on error.
trap 'kill $TOKENIZER_PID 2>/dev/null || true' EXIT

# ── 6. Run the profiled benchmark ────────────────────────────────────────────

log "Running benchmarks: --gpu --bf16 --profile across all 3 presets"
./scripts/run.sh --gpu --bf16 --profile --decode
./scripts/run.sh --gpu --bf16 --profile --prefill
./scripts/run.sh --gpu --bf16 --profile --balanced

# ── 7. Output summary + handoff to manual rsync ──────────────────────────────

log "Output summary"
JSON_COUNT=$(ls logs/*.json 2>/dev/null | wc -l)
NSYS_COUNT=$(ls logs/*.nsys-rep 2>/dev/null | wc -l)
SQLITE_COUNT=$(ls logs/*.sqlite 2>/dev/null | wc -l)
echo "JSON metric logs:  $JSON_COUNT"
echo "nsys-rep profiles: $NSYS_COUNT"
echo "sqlite siblings:   $SQLITE_COUNT"
du -sh logs/

# Best-effort detection of this instance's external IP (just to template the
# rsync command in the closing banner — purely informational).
INSTANCE_IP=$(curl -s --max-time 3 ifconfig.me 2>/dev/null || echo "<this_instance_ip>")

cat <<EOF

╔══════════════════════════════════════════════════════════════════════════╗
║  RUN COMPLETE                                                            ║
╠══════════════════════════════════════════════════════════════════════════╣
║                                                                          ║
║  From your laptop, pull the logs (skips .sqlite to halve transfer):      ║
║                                                                          ║
║    cd ~/projects/c_gpt2                                                  ║
║    RUN_DIR="logs/lambda/h100/\$(date +%Y%m%d_%H%M%S)_bf16_nsys_profile"  ║
║    mkdir -p "\$RUN_DIR"                                                  ║
║    rsync -avz --progress --exclude='*.sqlite' \\
║      ubuntu@$68.209.74.4:~/c_gpt2/logs/ "\$RUN_DIR/"
║                                                                          ║
║  Then TERMINATE this instance (Lambda bills until termination):          ║
║    https://cloud.lambda.ai/instances                                     ║
║                                                                          ║
╚══════════════════════════════════════════════════════════════════════════╝
EOF
