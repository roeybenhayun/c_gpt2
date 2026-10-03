#!/usr/bin/env python3
"""Plot round-by-round decode throughput against the memory-bandwidth ceiling.

Standalone on purpose: performance_analysis.py renders from JSON logs of a
single build, while this chart is a hand-curated series across seven commits.
Folding it into that script would mean teaching it about git history for one
article. Styling (figsize, dpi, grid, fonts, palette) matches the plots the
earlier articles shipped.

Usage:
    uv run python scripts/ralph_progress_plot.py [--out-dir <path>]
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# GPT-2 Large, GPU BF16, decode preset, RTX 5080. Measured values, in commit
# order; TPS is the figure quoted in each commit message.
STEPS = [
    # (label, ms/token, tokens/sec, round, iteration-within-round)
    ("Baseline",            16.63,  59.83, 0, None),
    ("One-pass top-k",      13.71,  72.49, 1, 1),
    ("Batched softmax",     10.88,  91.19, 1, 2),
    ("Batched attn GEMVs",   4.60, 213.25, 2, 1),
    ("Fused QKV",            4.27, 229.43, 3, 1),
    ("Drop dead memcpy",     4.12, 237.15, 3, 2),
    ("Fused bias+residual",  3.97, 246.80, 3, 3),
    ("Fused bias+GELU",      3.76, 259.67, 3, 4),
]

# Each Ralph round was a separate loop run with its own goal, its own verified
# baseline, and its own exit. Round 2 met its goal in a single iteration.
ROUNDS = [
    (1, "Round 1\ngoal +50%"),
    (2, "Round 2\ngoal 150"),
    (3, "Round 3\ngoal 250"),
]

# 1.54 GB of weights per token at ~960 GB/s on the RTX 5080.
FLOOR_MS = 1.61
FLOOR_TPS = 620

LARGE_COLOR = "#2ca02c"   # gpt2_large, matching performance_analysis.py
FLOOR_COLOR = "#c0392b"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir",
                    default="docs/articles/2026-09-ralph-loop-decode-perf/assets/plots")
    args = ap.parse_args()

    labels = [s[0] for s in STEPS]
    tps = np.array([s[2] for s in STEPS])
    x = np.arange(len(STEPS))

    fig, ax = plt.subplots(1, 1, figsize=(10, 7))

    # Shade each loop round behind the curve, so it is obvious which changes
    # came out of which run -- and that round 2 needed only one iteration.
    for rnd, title in ROUNDS:
        idxs = [i for i, s in enumerate(STEPS) if s[3] == rnd]
        lo, hi = min(idxs) - 0.5, max(idxs) + 0.5
        if rnd % 2 == 1:
            ax.axvspan(lo, hi, color="#7f8c8d", alpha=0.07, zorder=0)
        ax.axvline(lo, color="#bdc3c7", linewidth=1, zorder=0)
        ax.text((lo + hi) / 2, 520, title, ha="center", va="bottom",
                fontsize=9.5, color="#555555", fontweight="bold",
                linespacing=1.4)

    ax.plot(x, tps, "-o", color=LARGE_COLOR, linewidth=2.5, markersize=8,
            label="GPT-2 Large, GPU BF16 — tokens / sec", zorder=3)

    # The ceiling is the point of the chart: it is what the curve climbs towards.
    ax.axhline(FLOOR_TPS, color=FLOOR_COLOR, linestyle="--", linewidth=2,
               label=f"memory-bandwidth ceiling — ~{FLOOR_TPS} TPS ({FLOOR_MS} ms/token)")
    ax.fill_between([-0.5, len(STEPS) - 0.5], FLOOR_TPS, 700,
                    color=FLOOR_COLOR, alpha=0.07)
    ax.text(len(STEPS) - 0.6, FLOOR_TPS + 18, "unreachable — weights alone cost this much bandwidth",
            ha="right", va="bottom", fontsize=9, color=FLOOR_COLOR, style="italic")

    for xi, yi in zip(x, tps):
        ax.text(xi, yi + 16, f"{yi:.0f}", ha="center", va="bottom",
                fontsize=9, fontweight="bold", color=LARGE_COLOR)

    # Headroom at the two ends, which is the article's argument in two labels.
    # Headroom is quoted from decode-only TPOT, matching the article's floor
    # arithmetic. (TPS here is end-to-end and includes prefill, so deriving the
    # ratio from it would read ~0.1x differently and contradict the prose.)
    tpot = np.array([s[1] for s in STEPS])
    ax.annotate(f"{tpot[0] / FLOOR_MS:.0f}x below ceiling",
                xy=(0.05, tps[0] + 8), xytext=(0.6, 175),
                fontsize=10, color="#555555",
                arrowprops=dict(arrowstyle="->", color="#999999", lw=1.2))
    ax.annotate(f"{tpot[-1] / FLOOR_MS:.1f}x below ceiling",
                xy=(len(STEPS) - 1.05, tps[-1] + 10),
                xytext=(len(STEPS) - 3.2, tps[-1] + 95),
                fontsize=10, color="#555555",
                arrowprops=dict(arrowstyle="->", color="#999999", lw=1.2))

    ax.set_xticks(x)
    tick_labels = [
        s[0] if s[4] is None else f"{s[0]}\n(iter {s[4]})" for s in STEPS
    ]
    ax.set_xticklabels(tick_labels, fontsize=9.5, rotation=30, ha="right")
    ax.set_ylabel("Decode throughput (tokens / sec) — higher is better", fontsize=12)
    ax.set_title("GPT-2 Large decode: seven changes, and the wall they are "
                 "heading for", fontsize=14)
    ax.set_ylim(0, 700)
    ax.set_xlim(-0.5, len(STEPS) - 0.5)
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend(fontsize=10, loc="upper left")

    fig.tight_layout()
    os.makedirs(args.out_dir, exist_ok=True)
    path = os.path.join(args.out_dir, "decode_tps_vs_ceiling.png")
    fig.savefig(path, dpi=150)
    print(f"  -> Saved {path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
