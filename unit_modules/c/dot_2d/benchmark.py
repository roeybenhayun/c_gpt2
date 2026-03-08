#!/usr/bin/env python3
"""
Benchmark runner and visualization for dot_2d (CPU vs GPU).
Runs the C benchmark binary for various matrix sizes corresponding to
real-world transformer context lengths, collects HW info and runtime stats,
and produces publication-quality plots.
"""

import json
import subprocess
import os
import sys
import platform
import time
from pathlib import Path
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

SCRIPT_DIR = Path(__file__).parent.resolve()
BINARY = SCRIPT_DIR / "out" / "dot_2d_gpu"
OUTPUT_DIR = SCRIPT_DIR / "benchmark_results"

# Matrix sizes mapped to real model context lengths / embedding dims
BENCHMARK_CONFIGS = [
    {"size":   512, "label": "GPT-2\nSmall embed\n(512)"},
    {"size":   768, "label": "GPT-2\nd_model\n(768)"},
    {"size":  1024, "label": "GPT-2\nctx_len\n(1024)"},
    {"size":  2048, "label": "GPT-3\nctx_len\n(2048)"},
    {"size":  4096, "label": "LLaMA-2 7B\nctx / d_model\n(4096)"},
    {"size":  8192, "label": "GPT-4\nctx_len\n(8192)"},
    {"size": 12288, "label": "GPT-3 175B\nd_model\n(12288)"},
    {"size": 16384, "label": "LLaMA-3 405B\nctx_len\n(16384)"},
]


def collect_hw_info():
    """Collect hardware information about CPU and GPU."""
    info = {}

    # CPU
    info["cpu_model"] = "Unknown"
    try:
        out = subprocess.check_output(
            ["lscpu"], text=True, stderr=subprocess.DEVNULL
        )
        for line in out.splitlines():
            if line.startswith("Model name:"):
                info["cpu_model"] = line.split(":", 1)[1].strip()
            elif line.startswith("CPU(s):"):
                info["cpu_threads"] = line.split(":", 1)[1].strip()
            elif line.startswith("Core(s) per socket:"):
                info["cpu_cores"] = line.split(":", 1)[1].strip()
            elif line.startswith("CPU max MHz:"):
                mhz = float(line.split(":", 1)[1].strip())
                info["cpu_max_ghz"] = f"{mhz / 1000:.2f}"
    except Exception:
        pass

    # RAM
    try:
        out = subprocess.check_output(
            ["free", "-h"], text=True, stderr=subprocess.DEVNULL
        )
        for line in out.splitlines():
            if line.startswith("Mem:"):
                info["ram_total"] = line.split()[1]
    except Exception:
        pass

    # GPU
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,compute_cap,driver_version",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        parts = [p.strip() for p in out.strip().split(",")]
        info["gpu_model"] = parts[0]
        info["gpu_vram_mib"] = parts[1]
        info["gpu_compute_cap"] = parts[2]
        info["gpu_driver"] = parts[3]
    except Exception:
        pass

    # CUDA version
    try:
        out = subprocess.check_output(
            ["nvidia-smi"], text=True, stderr=subprocess.DEVNULL
        )
        for line in out.splitlines():
            if "CUDA Version" in line:
                idx = line.index("CUDA Version:")
                info["cuda_version"] = line[idx:].split(":")[1].strip().split()[0]
    except Exception:
        pass

    # cuBLAS version from library
    try:
        out = subprocess.check_output(
            ["ls", "/usr/local/cuda-12.8/lib64/"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        for line in out.splitlines():
            if "libcublas.so." in line and not line.endswith(".so"):
                info["cublas_lib"] = line.strip()
                break
    except Exception:
        pass

    info["os"] = platform.platform()
    info["timestamp"] = datetime.now().isoformat()

    return info


def collect_runtime_stats():
    """Snapshot of CPU and GPU utilization."""
    stats = {}
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,utilization.memory,temperature.gpu,"
                "power.draw,clocks.sm,clocks.mem",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        parts = [p.strip() for p in out.strip().split(",")]
        stats["gpu_util_pct"] = parts[0]
        stats["gpu_mem_util_pct"] = parts[1]
        stats["gpu_temp_c"] = parts[2]
        stats["gpu_power_w"] = parts[3]
        stats["gpu_sm_clock_mhz"] = parts[4]
        stats["gpu_mem_clock_mhz"] = parts[5]
    except Exception:
        pass

    try:
        with open("/proc/loadavg") as f:
            load = f.read().split()
            stats["cpu_load_1m"] = load[0]
            stats["cpu_load_5m"] = load[1]
    except Exception:
        pass

    return stats


def run_benchmark(size):
    """Run the C benchmark binary for a given matrix size."""
    print(f"  Running N={size}...", end="", flush=True)

    # Collect GPU stats right before
    pre_stats = collect_runtime_stats()

    result = subprocess.run(
        [str(BINARY), str(size), "--json"],
        capture_output=True,
        text=True,
        timeout=600,
    )

    # Collect GPU stats right after
    post_stats = collect_runtime_stats()

    if result.returncode != 0:
        print(f" FAILED (exit code {result.returncode})")
        print(f"  stderr: {result.stderr}")
        return None

    data = json.loads(result.stdout)
    data["runtime_stats_pre"] = pre_stats
    data["runtime_stats_post"] = post_stats

    print(
        f" GPU={data['gpu_avg_sec']:.6f}s ({data['gpu_tflops']:.1f} TFLOPS) "
        f"CPU={data['cpu_avg_sec']:.6f}s ({data['cpu_tflops']:.1f} TFLOPS) "
        f"Speedup={data['speedup']:.1f}x "
        f"{'PASS' if data['validation_passed'] else 'FAIL'}"
    )
    return data


def _prep_plot_data(results):
    """Extract arrays from results for plotting."""
    sizes = [r["matrix_size"] for r in results]
    gpu_times = [r["gpu_avg_sec"] for r in results]
    cpu_times = [r["cpu_avg_sec"] for r in results]
    gpu_tflops = [r["gpu_tflops"] for r in results]
    cpu_tflops = [r["cpu_tflops"] for r in results]
    speedups = [r["speedup"] for r in results]
    return sizes, gpu_times, cpu_times, gpu_tflops, cpu_tflops, speedups


def _apply_style():
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    plt.rcParams.update({
        "figure.dpi": 200,
        "savefig.dpi": 200,
        "font.family": "sans-serif",
        "axes.titleweight": "bold",
        "figure.facecolor": "white",
    })


GPU_COLOR = "#e74c3c"
CPU_COLOR = "#3498db"
SPEEDUP_COLOR = "#2ecc71"


def _hw_subtitle(hw_info):
    return (
        f"{hw_info.get('gpu_model', '?')} ({hw_info.get('gpu_vram_mib', '?')} MiB) "
        f"vs {hw_info.get('cpu_model', '?')} "
        f"({hw_info.get('cpu_cores', '?')}C/{hw_info.get('cpu_threads', '?')}T)"
    )


def _fmt_time(val):
    if val < 0.001:
        return f"{val * 1e6:.0f}us"
    if val < 1:
        return f"{val * 1000:.1f}ms"
    return f"{val:.2f}s"


def _plot_execution_time(ax, sizes, gpu_times, cpu_times, standalone=False):
    ax.plot(sizes, gpu_times, "o-", color=GPU_COLOR, linewidth=2, markersize=8,
            label="GPU (cuBLAS)", zorder=5)
    ax.plot(sizes, cpu_times, "s-", color=CPU_COLOR, linewidth=2, markersize=8,
            label="CPU (OpenBLAS)", zorder=5)
    ax.fill_between(sizes, gpu_times, cpu_times, alpha=0.08, color="gray")
    ax.set_yscale("log")
    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{int(v)}"))
    ax.set_ylabel("Time (seconds, log scale)")
    if not standalone:
        ax.set_title("Execution Time")
    ax.set_xlabel("Matrix Size (N x N)")
    ax.legend(loc="upper left")

    for i, (s, g) in enumerate(zip(sizes, gpu_times)):
        offset = 12 if i % 2 == 0 else -16
        ax.annotate(_fmt_time(g), (s, g), textcoords="offset points",
                    xytext=(0, offset), ha="center", fontsize=7, color=GPU_COLOR,
                    fontweight="bold")
    for i, (s, c) in enumerate(zip(sizes, cpu_times)):
        offset = -16 if i % 2 == 0 else 10
        ax.annotate(_fmt_time(c), (s, c), textcoords="offset points",
                    xytext=(0, offset), ha="center", fontsize=7, color=CPU_COLOR,
                    fontweight="bold")


# Theoretical peak FP32 TFLOPS
# RTX 5080: 10752 CUDA cores × 2 (FMA) × 2.62 GHz boost = 56.3 TFLOPS
GPU_THEORETICAL_TFLOPS = 56.3
# Ryzen 9 9950X3D (Zen 5): 16 cores × 2 FMA units × 16 FP32/FMA × 2 ops × 4.5 GHz (AVX-512 all-core) = 4.6 TFLOPS
CPU_THEORETICAL_TFLOPS = 4.6


def _plot_throughput(ax, sizes, gpu_tflops, cpu_tflops, standalone=False):
    # Theoretical peak lines
    ax.axhline(y=GPU_THEORETICAL_TFLOPS, color=GPU_COLOR, linestyle="--", alpha=0.4, linewidth=1.5)
    ax.text(sizes[0], GPU_THEORETICAL_TFLOPS * 1.08,
            f"RTX 5080 theoretical peak ({GPU_THEORETICAL_TFLOPS:.1f} TFLOPS)",
            fontsize=7, color=GPU_COLOR, alpha=0.7, va="bottom")
    ax.axhline(y=CPU_THEORETICAL_TFLOPS, color=CPU_COLOR, linestyle="--", alpha=0.4, linewidth=1.5)
    ax.text(sizes[0], CPU_THEORETICAL_TFLOPS * 1.08,
            f"9950X3D theoretical peak ({CPU_THEORETICAL_TFLOPS:.1f} TFLOPS, AVX-512 @4.5GHz)",
            fontsize=7, color=CPU_COLOR, alpha=0.7, va="bottom")

    # Measured data
    ax.plot(sizes, gpu_tflops, "o-", color=GPU_COLOR, linewidth=2, markersize=8,
            label="GPU (cuBLAS)", zorder=5)
    ax.plot(sizes, cpu_tflops, "s-", color=CPU_COLOR, linewidth=2, markersize=8,
            label="CPU (OpenBLAS)", zorder=5)
    ax.fill_between(sizes, gpu_tflops, cpu_tflops, alpha=0.08, color="gray")
    ax.set_yscale("log")
    ax.set_ylabel("TFLOPS (log scale)")
    if not standalone:
        ax.set_title(r"Throughput    TFLOPS $= \frac{2N^3}{time} \times 10^{-12}$")
    ax.set_xlabel("Matrix Size (N x N)")
    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{int(v)}"))
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(
        lambda v, _: f"{v:.0f}" if v >= 1 else f"{v:.1f}"))
    ax.legend(loc="lower right")

    for i, (s, t) in enumerate(zip(sizes, gpu_tflops)):
        offset = 12 if i % 2 == 0 else -16
        ax.annotate(f"{t:.1f}", (s, t), textcoords="offset points",
                    xytext=(0, offset), ha="center", fontsize=7, color=GPU_COLOR,
                    fontweight="bold")
    for i, (s, t) in enumerate(zip(sizes, cpu_tflops)):
        offset = -16 if i % 2 == 0 else 10
        ax.annotate(f"{t:.1f}", (s, t), textcoords="offset points",
                    xytext=(0, offset), ha="center", fontsize=7, color=CPU_COLOR,
                    fontweight="bold")


def _plot_speedup(ax, sizes, speedups, standalone=False):
    x = range(len(sizes))
    bars = ax.bar(x, speedups, color=SPEEDUP_COLOR, alpha=0.85,
                  edgecolor="white", linewidth=0.5, width=0.6)
    ax.set_ylabel("Speedup (CPU time / GPU time)")
    if not standalone:
        ax.set_title("GPU Speedup over CPU")
    ax.set_xticks(list(x))
    ax.set_xticklabels([str(s) for s in sizes], fontsize=9)
    ax.set_xlabel("Matrix Size (N x N)")
    ax.axhline(y=1, color="gray", linestyle="--", alpha=0.5, label="Break-even")

    for bar, val in zip(bars, speedups):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{val:.0f}x", ha="center", va="bottom", fontsize=9, fontweight="bold")


def _plot_table(ax, results, standalone=False):
    ax.axis("off")
    if not standalone:
        ax.set_title("Real-World Model Reference", pad=20)

    table_data = []
    for r in results:
        for cfg in BENCHMARK_CONFIGS:
            if cfg["size"] == r["matrix_size"]:
                model_label = cfg["label"].replace("\n", " ")
                table_data.append([
                    model_label,
                    f"{r['matrix_size']:,}",
                    f"{r['gpu_avg_sec']:.4f}s" if r["gpu_avg_sec"] >= 0.001 else f"{r['gpu_avg_sec']*1e6:.0f}us",
                    f"{r['cpu_avg_sec']:.4f}s" if r["cpu_avg_sec"] >= 0.01 else f"{r['cpu_avg_sec']*1000:.2f}ms",
                    f"{r['gpu_tflops']:.1f}",
                    f"{r['speedup']:.0f}x",
                ])
                break

    table = ax.table(
        cellText=table_data,
        colLabels=["Model Reference", "N", "GPU Time", "CPU Time", "GPU TFLOPS", "Speedup"],
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.6)
    table.auto_set_column_width(col=list(range(6)))

    for col in range(6):
        cell = table[0, col]
        cell.set_facecolor("#2c3e50")
        cell.set_text_props(color="white", fontweight="bold")

    for row in range(1, len(table_data) + 1):
        color = "#f8f9fa" if row % 2 == 0 else "white"
        for col in range(6):
            table[row, col].set_facecolor(color)


def _save_individual_plots(results, hw_info):
    """Save each plot as a standalone high-res PNG."""
    sizes, gpu_times, cpu_times, gpu_tflops, cpu_tflops, speedups = _prep_plot_data(results)
    subtitle = _hw_subtitle(hw_info)
    saved = []

    def _make_fig(title):
        fig, ax = plt.subplots(figsize=(10, 6))
        fig.suptitle(title + "\n" + subtitle,
                     fontsize=13, fontweight="bold", y=1.0)
        return fig, ax

    # 1. Execution time
    fig, ax = _make_fig("Execution Time: GPU (cuBLAS) vs CPU (OpenBLAS)")
    _plot_execution_time(ax, sizes, gpu_times, cpu_times, standalone=True)
    path = OUTPUT_DIR / "plot_execution_time.png"
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    saved.append(path)

    # 2. Throughput
    fig, ax = _make_fig(r"Throughput: GPU (cuBLAS) vs CPU (OpenBLAS)    TFLOPS $= \frac{2N^3}{time} \times 10^{-12}$")
    _plot_throughput(ax, sizes, gpu_tflops, cpu_tflops, standalone=True)
    path = OUTPUT_DIR / "plot_throughput.png"
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    saved.append(path)

    # 3. Speedup
    fig, ax = _make_fig("GPU Speedup over CPU")
    _plot_speedup(ax, sizes, speedups, standalone=True)
    path = OUTPUT_DIR / "plot_speedup.png"
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    saved.append(path)

    # 4. Reference table
    fig, ax = plt.subplots(figsize=(12, 5))
    fig.suptitle("Real-World Model Reference\n" + subtitle,
                 fontsize=13, fontweight="bold", y=1.0)
    _plot_table(ax, results, standalone=True)
    path = OUTPUT_DIR / "plot_reference_table.png"
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    saved.append(path)

    return saved


def create_plots(results, hw_info):
    """Create publication-quality benchmark plots (combined + individual)."""
    _apply_style()

    sizes, gpu_times, cpu_times, gpu_tflops, cpu_tflops, speedups = _prep_plot_data(results)
    subtitle = _hw_subtitle(hw_info)

    # --- Combined 2x2 plot ---
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(
        "Matrix Multiplication Benchmark: GPU (cuBLAS) vs CPU (OpenBLAS)",
        fontsize=16, fontweight="bold", y=0.98,
    )
    fig.text(0.5, 0.945, subtitle, ha="center", fontsize=10, color="gray", style="italic")

    _plot_execution_time(axes[0, 0], sizes, gpu_times, cpu_times)
    _plot_throughput(axes[0, 1], sizes, gpu_tflops, cpu_tflops)
    _plot_speedup(axes[1, 0], sizes, speedups)
    _plot_table(axes[1, 1], results)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    combined_path = OUTPUT_DIR / "benchmark_plot.png"
    fig.savefig(combined_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"\nCombined plot saved to: {combined_path}")

    # --- Individual plots ---
    individual = _save_individual_plots(results, hw_info)
    for p in individual:
        print(f"Individual plot saved to: {p}")


def main():
    if not BINARY.exists():
        print(f"Binary not found at {BINARY}. Run 'make' first.")
        sys.exit(1)

    OUTPUT_DIR.mkdir(exist_ok=True)

    print("Collecting hardware info...")
    hw_info = collect_hw_info()
    print(f"  CPU: {hw_info.get('cpu_model', '?')}")
    print(f"  GPU: {hw_info.get('gpu_model', '?')} ({hw_info.get('gpu_vram_mib', '?')} MiB)")
    print(f"  RAM: {hw_info.get('ram_total', '?')}")
    print()

    print("Running benchmarks...")
    results = []
    for cfg in BENCHMARK_CONFIGS:
        data = run_benchmark(cfg["size"])
        if data:
            results.append(data)

    if not results:
        print("No successful benchmarks!")
        sys.exit(1)

    # Save full JSON report
    report = {
        "hardware": hw_info,
        "runtime_stats_idle": collect_runtime_stats(),
        "benchmarks": results,
    }
    json_path = OUTPUT_DIR / "benchmark_results.json"
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nJSON report saved to: {json_path}")

    # Create plots
    create_plots(results, hw_info)


if __name__ == "__main__":
    main()
