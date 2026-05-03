#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# regenerate_article_plots.sh
#
# Re-runs performance_analysis.py against a chosen log dir and copies the
# resulting PNGs into the FP32→BF16 article's assets/plots/ folder with the
# names the article actually references.
#
# Why this exists: performance_analysis.py writes to plots/<name>.png with
# fixed filenames (overlay, summary, ttft, ...) and OVERWRITES them on every
# run. So `plots/` only ever holds the most recent invocation. This script
# does the "run, snapshot, run again, snapshot" dance and renames files into
# article-friendly names so they survive future analyzer calls.
#
# Usage:
#   ./scripts/regenerate_article_plots.sh                    # default: --log-dir logs
#   ./scripts/regenerate_article_plots.sh logs/lambda/h100/20260502_full_sweep_bf16_fp32
#
# After this runs, the article's images render correctly in any markdown
# viewer.
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

LOG_DIR="${1:-logs}"
ASSETS="docs/articles/2026-05-fp32-to-bf16-gpu/assets/plots"

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

if [ ! -d "$LOG_DIR" ]; then
    echo "Log dir not found: $LOG_DIR" >&2
    exit 1
fi
mkdir -p "$ASSETS"

run_analyzer() {
    local preset="$1"
    echo ""
    echo "==> Running analyzer for --$preset against $LOG_DIR"
    MPLBACKEND=Agg uv run python scripts/performance_analysis.py \
        --gpu --bf16 --"$preset" --log-dir "$LOG_DIR" \
        2>&1 | grep -E 'Found|Summary|GPT-2|Saved|^---' | head -20
}

snapshot() {
    # snapshot <source-in-plots/> <dest-in-assets/>
    local src="plots/$1"
    local dst="$ASSETS/$2"
    if [ ! -f "$src" ]; then
        echo "  ! expected $src after analyzer run, but it's missing — skipping $2" >&2
        return
    fi
    cp "$src" "$dst"
    echo "  ✓ $dst"
}

# ─── Decode preset → 4 article images ────────────────────────────────────────

run_analyzer decode
echo "  Snapshotting decode plots:"
snapshot summary.png  decode_tps_bar.png
snapshot summary.png  gpu_fp32_vs_bf16_tps.png   # same chart, different article reference
snapshot overlay.png  decode_overlay.png
snapshot speedup.png  decode_speedup.png

# ─── Prefill preset → 2 article images ───────────────────────────────────────

run_analyzer prefill
echo "  Snapshotting prefill plots:"
snapshot ttft.png         ttft_prefill_bar.png
snapshot phase_decomp.png ttft_prefill_phase_decomp.png

# ─── Summary ─────────────────────────────────────────────────────────────────

echo ""
echo "==> Article assets/plots/ now contains:"
ls -lh "$ASSETS"
