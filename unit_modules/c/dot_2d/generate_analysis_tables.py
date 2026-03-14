#!/usr/bin/env python3
"""Generate publication-quality table images for GPT-2 dot_2d FLOP analysis."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

OUTPUT_DIR = Path(__file__).parent / "benchmark_results"
OUTPUT_DIR.mkdir(exist_ok=True)

HEADER_COLOR = "#2c3e50"
ROW_EVEN = "#f8f9fa"
ROW_ODD = "white"


def _apply_style():
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    plt.rcParams.update({
        "figure.dpi": 200,
        "savefig.dpi": 200,
        "font.family": "sans-serif",
        "figure.facecolor": "white",
    })


def _style_table(table, n_rows, n_cols):
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.7)
    table.auto_set_column_width(col=list(range(n_cols)))
    for col in range(n_cols):
        cell = table[0, col]
        cell.set_facecolor(HEADER_COLOR)
        cell.set_text_props(color="white", fontweight="bold")
    for row in range(1, n_rows + 1):
        color = ROW_EVEN if row % 2 == 0 else ROW_ODD
        for col in range(n_cols):
            table[row, col].set_facecolor(color)


def generate_operations_table():
    """Table 1: dot_2d operations per layer."""
    fig, ax = plt.subplots(figsize=(14, 5))
    fig.suptitle("dot_2d Operations per Transformer Layer",
                 fontsize=14, fontweight="bold", y=0.98)
    fig.text(0.5, 0.92, "8 operations per layer + 1 final logits projection",
             ha="center", fontsize=10, color="gray", style="italic")
    ax.axis("off")

    data = [
        ["1", "Q projection",  "[T, d_model] x [d_model, d_model]^T", "Input x W_q^T"],
        ["2", "K projection",  "[T, d_model] x [d_model, d_model]^T", "Input x W_k^T"],
        ["3", "V projection",  "[T, d_model] x [d_model, d_model]^T", "Input x W_v^T"],
        ["4", "Q x K^T",       "[T, head_dim] x [T, head_dim]^T",     "Per head, scaled"],
        ["5", "Weights x V",   "[T, T] x [T, head_dim]",              "Per head"],
        ["6", "Output proj",   "[T, d_model] x [d_model, d_model]^T", "attn_out x W_proj^T"],
        ["7", "FFN up (W1)",   "[T, d_model] x [d_ff, d_model]^T",    "d_ff = 4 x d_model"],
        ["8", "FFN down (W2)", "[T, d_ff] x [d_model, d_ff]^T",       "Back to d_model"],
        ["9", "Logits",        "[T, d_model] x [d_model, vocab]",      "After all layers (once)"],
    ]
    cols = ["#", "Operation", "Matrix Dimensions", "Description"]
    table = ax.table(cellText=data, colLabels=cols, loc="center", cellLoc="center")
    _style_table(table, len(data), len(cols))
    fig.savefig(OUTPUT_DIR / "analysis_operations.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {OUTPUT_DIR / 'analysis_operations.png'}")


def generate_config_table():
    """Table 2: Model configurations."""
    fig, ax = plt.subplots(figsize=(10, 4.5))
    fig.suptitle("GPT-2 Model Configurations",
                 fontsize=14, fontweight="bold", y=0.98)
    ax.axis("off")

    data = [
        ["d_model",            "768",   "1024",  "1280"],
        ["num_layers",         "12",    "24",    "36"],
        ["nof_heads",          "12",    "16",    "20"],
        ["head_dim",           "64",    "64",    "64"],
        ["d_ff (4 x d_model)", "3,072", "4,096", "5,120"],
        ["vocab_size",         "50,257","50,257","50,257"],
        ["ctx_len",            "1,024", "1,024", "1,024"],
        ["dot_2d calls",       "97",    "193",   "289"],
    ]
    cols = ["Parameter", "Small", "Medium", "Large"]
    table = ax.table(cellText=data, colLabels=cols, loc="center", cellLoc="center")
    _style_table(table, len(data), len(cols))
    fig.savefig(OUTPUT_DIR / "analysis_config.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {OUTPUT_DIR / 'analysis_config.png'}")


def generate_flops_table():
    """Table 3: FLOP breakdown with calculations."""
    fig, ax = plt.subplots(figsize=(16, 8))
    fig.suptitle("dot_2d FLOP Breakdown per Layer (T=1024, full context)",
                 fontsize=14, fontweight="bold", y=0.98)
    fig.text(0.5, 0.93, r"FLOPs per matmul = 2 $\times$ M $\times$ N $\times$ K",
             ha="center", fontsize=10, color="gray", style="italic")
    ax.axis("off")

    data = [
        ["Projections (Q,K,V,Out)",
         "4 x 2 x 1024 x 768 x 768",   "0.0048",
         "4 x 2 x 1024 x 1024 x 1024", "0.0086",
         "4 x 2 x 1024 x 1280 x 1280", "0.0134"],
        ["Attention (QK^T + WV)",
         "2 x 2 x 1024 x 1024 x 768",  "0.0032",
         "2 x 2 x 1024 x 1024 x 1024", "0.0043",
         "2 x 2 x 1024 x 1024 x 1280", "0.0054"],
        ["FFN (W1 + W2)",
         "2 x 2 x 1024 x 768 x 3072",  "0.0097",
         "2 x 2 x 1024 x 1024 x 4096", "0.0172",
         "2 x 2 x 1024 x 1280 x 5120", "0.0268"],
        ["Per layer total", "", "0.0177", "", "0.0301", "", "0.0456"],
        ["All layers (x N)",
         "x 12", "0.212",
         "x 24", "0.722",
         "x 36", "1.642"],
        ["Logits projection",
         "2 x 1024 x 768 x 50257",  "0.079",
         "2 x 1024 x 1024 x 50257", "0.105",
         "2 x 1024 x 1280 x 50257", "0.132"],
        ["GRAND TOTAL", "", "0.29", "", "0.83", "", "1.77"],
    ]
    cols = ["Operation",
            "Small Calculation", "Small\nTFLOPS",
            "Medium Calculation", "Medium\nTFLOPS",
            "Large Calculation", "Large\nTFLOPS"]
    table = ax.table(cellText=data, colLabels=cols, loc="center", cellLoc="center")
    _style_table(table, len(data), len(cols))

    # Bold the total rows
    for row in [4, 7]:  # "Per layer total" and "GRAND TOTAL" (1-indexed in table = data row + 1)
        for col in range(len(cols)):
            table[row, col].set_text_props(fontweight="bold")
    for col in range(len(cols)):
        table[len(data), col].set_text_props(fontweight="bold")
        table[len(data), col].set_facecolor("#e8f4f8")

    fig.savefig(OUTPUT_DIR / "analysis_flops.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {OUTPUT_DIR / 'analysis_flops.png'}")


def generate_speedup_table():
    """Table 4: Potential GPU speedup."""
    fig, ax = plt.subplots(figsize=(14, 4))
    fig.suptitle("Potential GPU Speedup for dot_2d Operations",
                 fontsize=14, fontweight="bold", y=0.98)
    fig.text(0.5, 0.90,
             "RTX 5080 (~40 TFLOPS/s cuBLAS) vs Ryzen 9 9950X3D (~3.5 TFLOPS/s OpenBLAS)",
             ha="center", fontsize=10, color="gray", style="italic")
    ax.axis("off")

    data = [
        ["Small",  "0.29", "0.29 / 3.5 = ~83ms",  "0.29 / 40 = ~7ms",  "~12x"],
        ["Medium", "0.83", "0.83 / 3.5 = ~236ms", "0.83 / 40 = ~21ms", "~11x"],
        ["Large",  "1.77", "1.77 / 3.5 = ~507ms", "1.77 / 40 = ~44ms", "~11x"],
    ]
    cols = ["Model", "Total TFLOPS", "CPU Time", "GPU Time", "Speedup"]
    table = ax.table(cellText=data, colLabels=cols, loc="center", cellLoc="center")
    _style_table(table, len(data), len(cols))

    # Highlight speedup column
    for row in range(1, len(data) + 1):
        table[row, 4].set_text_props(fontweight="bold", color="#27ae60")

    fig.savefig(OUTPUT_DIR / "analysis_speedup.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {OUTPUT_DIR / 'analysis_speedup.png'}")


def main():
    _apply_style()
    generate_operations_table()
    generate_config_table()
    generate_flops_table()
    generate_speedup_table()
    print("\nAll analysis tables generated.")


if __name__ == "__main__":
    main()
