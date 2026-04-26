import matplotlib.pyplot as plt
import os

# Kernel time distributions for GPT-2 Large from nsys profiles taken 2026-04-12.
# - Pre  : logs/gpt2_large_gpu_profile_20260412_202718.nsys-rep  (~29.3s total GPU time)
# - Post : logs/gpt2_large_gpu_profile_20260412_203340.nsys-rep  (~7.2s  total GPU time)

KERNELS_PRE = [
    ("softmax_kernel",          79.4),
    ("gemvx (variant 1)",        6.7),
    ("gemvNSP",                  5.4),
    ("gemvx (variant 2)",        3.0),
    ("gemvx (variant 3)",        2.9),
    ("cutlass sgemm (128x32)",   0.8),
    ("layernorm_kernel",         0.8),
    ("add_bias_kernel",          0.6),
]

KERNELS_POST = [
    ("gemvx (variant 1)",       27.4),
    ("gemvNSP",                 22.0),
    ("softmax_kernel",          16.1),
    ("gemvx (variant 2)",       12.1),
    ("gemvx (variant 3)",       12.0),
    ("cutlass sgemm (128x32)",   3.2),
    ("layernorm_kernel",         3.1),
    ("add_bias_kernel",          2.4),
    ("add_2d_kernel",            0.7),
    ("gelu_kernel",              0.5),
    ("concat_heads_kernel",      0.3),
]

COLORS = [
    "#e74c3c", "#3498db", "#2ecc71", "#9b59b6", "#f39c12",
    "#1abc9c", "#e67e22", "#95a5a6", "#bdc3c7", "#34495e",
    "#16a085", "#2c3e50",
]


def make_pie_chart(kernels, title, out_filename):
    threshold = 0.5
    main = [(name, pct) for name, pct in kernels if pct >= threshold]
    other_pct = sum(pct for _, pct in kernels if pct < threshold)
    remaining = 100.0 - sum(pct for _, pct in kernels)
    other_pct += remaining
    main.append(("Other", other_pct))

    labels = [k[0] for k in main]
    sizes = [k[1] for k in main]

    fig, ax = plt.subplots(figsize=(10, 8))

    pie_labels = ["softmax_kernel" if name == "softmax_kernel" else "" for name in labels]

    wedges, texts, autotexts = ax.pie(
        sizes,
        labels=pie_labels,
        autopct=lambda p: f'{p:.1f}%' if p >= 0.8 else '',
        colors=COLORS[:len(main)],
        startangle=90,
        pctdistance=0.82,
        explode=[0.05 if labels[i] == "softmax_kernel" else 0 for i in range(len(main))],
    )

    for t in autotexts:
        t.set_fontsize(10)
        t.set_fontweight('bold')
    for t in texts:
        t.set_fontsize(12)
        t.set_fontweight('bold')

    legend_labels = [f'{name} ({pct:.1f}%)' for name, pct in zip(labels, sizes)]
    ax.legend(wedges, legend_labels, title="CUDA Kernels", loc="center left",
              bbox_to_anchor=(1.0, 0.5), fontsize=10, title_fontsize=11)

    ax.set_title(title, fontsize=13, pad=20)

    PLOT_DIR = "plots"
    os.makedirs(PLOT_DIR, exist_ok=True)
    out_path = os.path.join(PLOT_DIR, out_filename)
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Saved to {out_path}")
    plt.close(fig)


make_pie_chart(
    KERNELS_PRE,
    "CUDA Kernel Time Distribution — GPT-2 Large (Pre-Softmax Optimization)",
    "kernel_profile_pre_softmax.png",
)
make_pie_chart(
    KERNELS_POST,
    "CUDA Kernel Time Distribution — GPT-2 Large (Post-Softmax Optimization)",
    "cuda_kernels_time_distribution.png",
)
