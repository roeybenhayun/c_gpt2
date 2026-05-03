"""Render a side-by-side decode-TPS bar chart across multiple log sources.

The default `performance_analysis.py` only plots one source at a time. This
script overlays N sources × 2 dtypes (FP32 + BF16) on the same axes — useful
for cross-hardware comparisons (5080 vs H100, H100 vs B200, etc.) where the
question is "same code, same dtype: who's faster?".

Usage:
    uv run python scripts/compare_sources_decode_tps.py \\
        --source "RTX 5080:logs" \\
        --source "H100 SXM5:logs/lambda/h100/20260502_full_sweep_bf16_fp32" \\
        --output docs/articles/2026-05-fp32-to-bf16-gpu/assets/plots/h100_vs_5080_decode_tps.png

Each --source is "label:path" — colon-separated. Order matters (defines bar
group order on the plot).
"""

import argparse
import glob
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SIZES = ["small", "medium", "large"]
TAGS = ["gpu", "bf16"]
TAG_LABELS = {"gpu": "FP32", "bf16": "BF16"}

# Two-shade palette per source (saturated for FP32, pale for BF16).
SOURCE_PALETTES = [
    ("#2e86ab", "#90c5dc"),  # blue
    ("#c25555", "#e89a9a"),  # red
    ("#7caf50", "#b9d8a0"),  # green
    ("#9b59b6", "#cdb1d8"),  # purple
]


def latest_tps(log_dir: str, size: str, tag: str, preset: str = "decode"):
    pattern = f"{log_dir}/gpt2_{size}_{tag}_{preset}_*.json"
    files = sorted(glob.glob(pattern))
    if not files:
        return None
    with open(files[-1]) as f:
        return json.load(f).get("output_tps")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", action="append", required=True,
                   help='"label:path" pair, repeatable')
    p.add_argument("--output", required=True, help="output PNG path")
    p.add_argument("--preset", default="decode", choices=["decode", "prefill", "balanced"])
    p.add_argument("--title", default=None, help="chart title (auto-generated if omitted)")
    args = p.parse_args()

    sources = []
    for s in args.source:
        if ":" not in s:
            print(f"--source must be 'label:path', got: {s}", file=sys.stderr)
            sys.exit(1)
        label, path = s.split(":", 1)
        sources.append((label, path))

    if len(sources) > len(SOURCE_PALETTES):
        print(f"Only {len(SOURCE_PALETTES)} colour palettes defined", file=sys.stderr)
        sys.exit(1)

    data = {}
    for label, path in sources:
        for tag in TAGS:
            data[(label, tag)] = [latest_tps(path, sz, tag, args.preset) for sz in SIZES]

    n_models = len(SIZES)
    n_bars = len(sources) * len(TAGS)
    total_w = 0.84
    bar_w = total_w / n_bars
    x_pos = np.arange(n_models)
    offsets = [(-total_w / 2) + (i + 0.5) * bar_w for i in range(n_bars)]

    fig, ax = plt.subplots(figsize=(11, 6.5), dpi=150)

    i = 0
    for src_idx, (label, _) in enumerate(sources):
        fp32_color, bf16_color = SOURCE_PALETTES[src_idx]
        for tag, color in zip(TAGS, (fp32_color, bf16_color)):
            vals = data[(label, tag)]
            bars = ax.bar(x_pos + offsets[i], vals, bar_w,
                          label=f"{label} {TAG_LABELS[tag]}",
                          color=color, alpha=0.95)
            for bar, v in zip(bars, vals):
                if v:
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                            f"{v:.1f}", ha="center", va="bottom",
                            fontsize=8.5, fontweight="bold")
            i += 1

    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"GPT-2 {s.capitalize()}" for s in SIZES], fontsize=11)
    ax.set_ylabel("Tokens / second", fontsize=12)
    ax.set_title(args.title or f"Decode TPS comparison — {args.preset} preset", fontsize=13)
    ax.legend(loc="upper right", ncol=len(sources), fontsize=10)
    ax.grid(True, axis="y", alpha=0.3)

    valid = [v for k in data for v in data[k] if v is not None]
    if valid:
        ax.set_ylim(0, max(valid) * 1.20)

    plt.tight_layout()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.output, dpi=150)
    print(f"Saved {args.output}")

    # Print the underlying numbers for log/sanity
    print()
    print(f"{'Source':<14} {'Tag':<6} {'Small':>8} {'Medium':>8} {'Large':>8}")
    print("-" * 52)
    for label, _ in sources:
        for tag in TAGS:
            v = data[(label, tag)]
            row = " ".join(f"{x:>8.1f}" if x is not None else f"{'—':>8}" for x in v)
            print(f"{label:<14} {TAG_LABELS[tag]:<6} {row}")


if __name__ == "__main__":
    main()
