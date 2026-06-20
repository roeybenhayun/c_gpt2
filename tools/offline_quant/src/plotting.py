"""Matplotlib helpers for --distrib (single tensor) and --compare (cross)."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: never need an interactive backend
import matplotlib.pyplot as plt
import numpy as np

from .quantizer import QuantizedTensor


def _save(fig, out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_distribution_with_amax(W: np.ndarray, label: str, out_path: Path) -> Path:
    w_max = float(W.max())
    w_min = float(W.min())
    amax = float(max(abs(w_max), abs(w_min)))
    fig, ax = plt.subplots(figsize=(7.5, 4))
    ax.hist(W.ravel(), bins=200, color="#4c78a8", edgecolor="none")
    ax.axvline(w_max, color="red", linestyle="--", label=f"max(W) = {w_max:+.4f}")
    ax.axvline(w_min, color="red", linestyle="--", label=f"min(W) = {w_min:+.4f}")
    ax.set_xlabel("FP32 value")
    ax.set_ylabel("count")
    ax.set_title(f"{label}: FP32 weight distribution  (amax = {amax:.4f})")
    ax.legend()
    return _save(fig, out_path)


def plot_dequant_error_hist(
    W_fp32: np.ndarray, W_recon: np.ndarray, label: str, out_path: Path
) -> Path:
    err = (W_fp32 - W_recon).ravel()
    rms = float(np.sqrt(np.mean(err ** 2)))
    maxabs = float(np.max(np.abs(err)))
    fig, ax = plt.subplots(figsize=(7.5, 4))
    ax.hist(err, bins=200, color="#54a24b", edgecolor="none")
    ax.set_xlabel("FP32 − dequant(int8)")
    ax.set_ylabel("count")
    ax.set_title(f"{label}: dequant error  (rms = {rms:.4e}, max|err| = {maxabs:.4e})")
    return _save(fig, out_path)


def plot_fp32_vs_dequant_scatter(
    W_fp32: np.ndarray,
    W_recon: np.ndarray,
    label: str,
    out_path: Path,
    sample: int = 5000,
) -> Path:
    flat_fp = W_fp32.ravel()
    flat_dq = W_recon.ravel()
    if flat_fp.size > sample:
        rng = np.random.default_rng(0)
        idx = rng.choice(flat_fp.size, size=sample, replace=False)
        flat_fp = flat_fp[idx]
        flat_dq = flat_dq[idx]
    lim = float(max(np.abs(flat_fp).max(), np.abs(flat_dq).max())) * 1.05
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.scatter(flat_fp, flat_dq, s=3, alpha=0.4, color="#4c78a8")
    ax.plot([-lim, lim], [-lim, lim], "k--", lw=1, label="y = x")
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel("FP32 value")
    ax.set_ylabel("dequant(int8) value")
    ax.set_aspect("equal")
    ax.set_title(f"{label}: FP32 vs dequant ({len(flat_fp)} sampled cells)")
    ax.legend()
    return _save(fig, out_path)


def plot_memory_breakdown(memory_summary: dict, label: str, out_path: Path) -> Path:
    """Stacked bar: source FP32 vs INT8 artifact, split into matmul / preserved / scales."""
    m = memory_summary
    matmul_src   = m["source_fp32_total_mb"] - m["preserved_fp32_mb"]   # 4× int8 size
    matmul_dst   = m["quantized_int8_mb"]
    preserved_mb = m["preserved_fp32_mb"]
    scales_mb    = m["scale_overhead_kb"] / 1024.0

    fig, ax = plt.subplots(figsize=(7, 5))
    x = np.arange(2)
    width = 0.55
    ax.bar(x, [preserved_mb, preserved_mb], width,
           label="preserved (FP32)", color="#9ecae1")
    ax.bar(x, [matmul_src, matmul_dst], width,
           bottom=[preserved_mb, preserved_mb],
           label="matmul tensors (FP32 → INT8)", color="#4c78a8")
    ax.bar(x, [0, scales_mb], width,
           bottom=[preserved_mb + matmul_src, preserved_mb + matmul_dst],
           label="per-channel scales (FP32)", color="#f58518")

    for i, total in enumerate([m["source_fp32_total_mb"], m["artifact_total_mb"]]):
        ax.text(i, total + total * 0.015, f"{total} MiB", ha="center", fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(["Source FP32 file", "INT8 artifact"])
    ax.set_ylabel("MiB")
    ax.set_title(f"{label}: memory breakdown")
    ax.legend(loc="upper right")
    return _save(fig, out_path)


def plot_cross_model_memory(
    summaries: list[tuple[str, dict]], out_path: Path
) -> Path:
    """Grouped bars: source FP32 vs INT8 artifact across model sizes."""
    n = len(summaries)
    labels    = [f"gpt2-{s[0]}" for s in summaries]
    sources   = [s[1]["source_fp32_total_mb"] for s in summaries]
    artifacts = [s[1]["artifact_total_mb"]    for s in summaries]

    fig, ax = plt.subplots(figsize=(7.5, 5))
    x = np.arange(n)
    width = 0.36
    b1 = ax.bar(x - width / 2, sources,   width, label="Source FP32", color="#a3a3a3")
    b2 = ax.bar(x + width / 2, artifacts, width, label="INT8 artifact", color="#4c78a8")

    for rect, v in zip(b1, sources):
        ax.text(rect.get_x() + rect.get_width() / 2, v, f"{v}", ha="center", va="bottom")
    for rect, v in zip(b2, artifacts):
        ax.text(rect.get_x() + rect.get_width() / 2, v, f"{v}", ha="center", va="bottom")

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("MiB")
    ax.set_title("Cross-model memory: source FP32 vs INT8 artifact")
    ax.legend()
    return _save(fig, out_path)


def plot_side_by_side_distributions(
    tensors: list[tuple[str, np.ndarray]], out_path: Path
) -> Path:
    """1×N row of distribution histograms with per-panel max/min annotations."""
    n = len(tensors)
    fig, axes = plt.subplots(1, n, figsize=(6.5 * n, 4), squeeze=False)
    for ax, (label, W) in zip(axes[0], tensors):
        w_max = float(W.max())
        w_min = float(W.min())
        amax = float(max(abs(w_max), abs(w_min)))
        ax.hist(W.ravel(), bins=200, color="#4c78a8", edgecolor="none")
        ax.axvline(w_max, color="red", linestyle="--", label=f"max(W) = {w_max:+.4f}")
        ax.axvline(w_min, color="red", linestyle="--", label=f"min(W) = {w_min:+.4f}")
        ax.set_xlabel("FP32 value")
        ax.set_ylabel("count")
        ax.set_title(f"{label}  (amax = {amax:.4f})")
        ax.legend(loc="upper right")
    return _save(fig, out_path)


def render_mapping_examples(
    W_fp32: np.ndarray,
    qt: QuantizedTensor,
    label: str,
    out_path: Path,
    n: int = 8,
    seed: int = 42,
) -> Path:
    """Random (row, col) cells: FP32 → amax → scale → INT8 → dequant → error."""
    rng = np.random.default_rng(seed)
    rows = rng.integers(0, W_fp32.shape[0], size=n)
    cols = rng.integers(0, W_fp32.shape[1], size=n)
    row_amax = np.max(np.abs(W_fp32), axis=1)
    cell_text = []
    for r, c in zip(rows, cols):
        fp = float(W_fp32[r, c])
        am = float(row_amax[r])
        sc = float(qt.scale[r])
        i8 = int(qt.int8[r, c])
        dq = i8 * sc
        cell_text.append([
            f"{r}", f"{c}",
            f"{fp: .6f}", f"{am: .6f}", f"{sc: .6f}",
            f"{i8:+d}", f"{dq: .6f}", f"{fp - dq: .6e}",
        ])
    headers = ["row", "col", "FP32", "amax (row)", "scale (row)", "INT8", "dequant", "error"]
    fig, ax = plt.subplots(figsize=(11.5, 0.5 * (n + 2) + 0.6))
    ax.axis("off")
    tbl = ax.table(cellText=cell_text, colLabels=headers, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.5)
    ax.set_title(
        f"{label}: mapping examples (random cells)\n"
        r"quantize:   $\mathrm{INT8} = \mathrm{round}(\mathrm{FP32}\;/\;\mathrm{scale})$"
        r",   $\mathrm{scale} = \mathrm{amax}\;/\;127$"
        "\n"
        r"dequantize: $\mathrm{FP32}_{\mathrm{recon}} = \mathrm{INT8} \times \mathrm{scale}$",
        pad=12,
        fontsize=10,
    )
    return _save(fig, out_path)
