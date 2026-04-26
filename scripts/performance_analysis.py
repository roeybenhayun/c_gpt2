import json
import matplotlib.pyplot as plt
import numpy as np
import os
import glob
import sys

# --- Configuration ---

# To use MANUAL mode: Fill in the paths for the files you want to plot.
# To use AUTOMATIC mode: Leave all paths as empty strings ("").
files = {
    "gpt2_small": "",
    "gpt2_medium": "",
    "gpt2_large": ""
}

LOG_DIR = "logs"

colors = {
    "gpt2_small": "#1f77b4",
    "gpt2_medium": "#ff7f0e",
    "gpt2_large": "#2ca02c"
}

# --- Helper Functions ---

def discover_files(tag=None):
    """Discover latest log files, optionally filtered by tag (cpu/gpu)."""
    found = {}
    for model_name in files.keys():
        if tag:
            search_pattern = os.path.join(LOG_DIR, f"{model_name}_{tag}_*.json")
        else:
            search_pattern = os.path.join(LOG_DIR, f"{model_name}_*.json")
        matched = glob.glob(search_pattern)
        if not matched:
            continue
        latest_file = max(matched, key=os.path.getmtime)
        found[model_name] = latest_file
        print(f"  -> Found latest for '{model_name}' ({tag or 'any'}): {os.path.basename(latest_file)}")
    return found

def load_json(file_path):
    """Load a full JSON log file."""
    with open(file_path) as f:
        return json.load(f)

def load_run_data(file_path):
    """Load a JSON log file and return (x, y, metadata)."""
    data = load_json(file_path)
    x = np.array([chunk["total_context"] for chunk in data["chunks"]])
    y = np.array([chunk["chunk_seconds"] for chunk in data["chunks"]])
    kv_cache_enabled = data.get("kv_cache_enabled", 1)
    order = 1 if kv_cache_enabled == 1 else 2
    chunk_size = data.get("token_chunk_size", "N/A")
    return x, y, order, chunk_size

def fit_curve(x, y, order):
    """Return fitted x and y arrays for plotting."""
    coeffs = np.polyfit(x, y, order)
    poly = np.poly1d(coeffs)
    x_fit = np.linspace(min(x), max(x), 100)
    return x_fit, poly(x_fit)

def format_label(model_name):
    return model_name.replace("gpt2_", "GPT-2 ").title()

# --- Mode Selection & Discovery ---

is_manual_mode = any(files.values())

# Parse CLI flags: --cpu, --gpu, or both (default)
cli_args = set(sys.argv[1:])
want_cpu = "--cpu" in cli_args
want_gpu = "--gpu" in cli_args
if not want_cpu and not want_gpu:
    want_cpu = True
    want_gpu = True

if is_manual_mode:
    print("--- Running in MANUAL mode ---")
    manual_files = {model: path for model, path in files.items() if path}
    for model, path in manual_files.items():
        print(f"  -> Using manual path for '{model}': {path}")
    cpu_files = {}
    gpu_files = {}
    solo_files = manual_files
else:
    print("--- Running in AUTOMATIC discovery mode ---")
    cpu_files = discover_files("cpu") if want_cpu else {}
    gpu_files = discover_files("gpu") if want_gpu else {}
    solo_files = {}

    if not cpu_files and not gpu_files:
        print("  No cpu/gpu tagged files found, falling back to untagged discovery...")
        solo_files = discover_files()

has_comparison = bool(cpu_files) and bool(gpu_files)

all_files = cpu_files or gpu_files or solo_files
if not all_files:
    raise FileNotFoundError("No files to process. Please fill in paths manually or run the bash script to generate log files.")

print("\n--- Processing and plotting data ---")

# --- Load all data ---

cpu_data = {}
gpu_data = {}
cpu_json = {}
gpu_json = {}
chunk_size_label = None

for model_name in files.keys():
    if model_name in cpu_files:
        x, y, order, cs = load_run_data(cpu_files[model_name])
        cpu_data[model_name] = (x, y, order)
        cpu_json[model_name] = load_json(cpu_files[model_name])
        if chunk_size_label is None:
            chunk_size_label = cs
    if model_name in gpu_files:
        x, y, order, cs = load_run_data(gpu_files[model_name])
        gpu_data[model_name] = (x, y, order)
        gpu_json[model_name] = load_json(gpu_files[model_name])
        if chunk_size_label is None:
            chunk_size_label = cs
    if model_name in solo_files:
        x, y, order, cs = load_run_data(solo_files[model_name])
        cpu_data[model_name] = (x, y, order)
        cpu_json[model_name] = load_json(solo_files[model_name])
        if chunk_size_label is None:
            chunk_size_label = cs

# --- Check if new metrics are available ---

def has_new_metrics(data):
    return "ttft_s" in data and "per_token_latencies" in data

any_new_metrics = any(
    has_new_metrics(d) for d in list(cpu_json.values()) + list(gpu_json.values())
)

PLOT_DIR = "plots"

# --- Plotting functions ---

def plot_overlay(ax):
    """Plot 1: Overlay — CPU (solid) vs GPU (dashed)."""
    for model_name in files.keys():
        color = colors[model_name]
        label_name = format_label(model_name)

        if model_name in cpu_data:
            x, y, order = cpu_data[model_name]
            x_fit, y_fit = fit_curve(x, y, order)
            if has_comparison:
                ax.plot(x, y, 'o', color=color, alpha=0.4, markersize=5)
                ax.plot(x_fit, y_fit, '-', color=color, linewidth=2, label=f'{label_name} CPU')
            else:
                ax.plot(x, y, 'o', color=color, alpha=0.4, markersize=5)
                ax.plot(x_fit, y_fit, '-', color=color, linewidth=2, label=f'{label_name}')

        if model_name in gpu_data:
            x, y, order = gpu_data[model_name]
            x_fit, y_fit = fit_curve(x, y, order)
            ax.plot(x, y, 's', color=color, alpha=0.4, markersize=5)
            ax.plot(x_fit, y_fit, '--', color=color, linewidth=2, label=f'{label_name} GPU')

    ax.set_xlabel("Total Context Length", fontsize=12)
    ax.set_ylabel(f"Time for Last {chunk_size_label} Tokens (s)", fontsize=12)
    if has_comparison:
        ax.set_title("CPU vs GPU — Time to Generate Tokens", fontsize=14)
    else:
        ax.set_title("Time to Generate Tokens vs. Context Length", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

def plot_speedup(ax):
    """Plot 2: Speedup ratio (CPU / GPU)."""
    for model_name in files.keys():
        if model_name not in cpu_data or model_name not in gpu_data:
            continue

        color = colors[model_name]
        label_name = format_label(model_name)

        cpu_x, cpu_y, _ = cpu_data[model_name]
        gpu_x, gpu_y, _ = gpu_data[model_name]

        common_min = max(cpu_x.min(), gpu_x.min())
        common_max = min(cpu_x.max(), gpu_x.max())
        common_x = np.linspace(common_min, common_max, 50)

        cpu_interp = np.interp(common_x, cpu_x, cpu_y)
        gpu_interp = np.interp(common_x, gpu_x, gpu_y)

        speedup = cpu_interp / gpu_interp

        ax.plot(common_x, speedup, '-', color=color, linewidth=2, label=f'{label_name}')
        avg_speedup = np.mean(speedup)
        ax.axhline(y=avg_speedup, color=color, linestyle=':', alpha=0.5)
        ax.text(common_x[-1] + 10, avg_speedup, f'{avg_speedup:.1f}x',
                color=color, fontsize=11, fontweight='bold', va='center')

    ax.axhline(y=1.0, color='gray', linestyle='-', alpha=0.3)
    ax.set_xlabel("Total Context Length", fontsize=12)
    ax.set_ylabel("Speedup (CPU time / GPU time)", fontsize=12)
    ax.set_title("GPU Speedup over CPU", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

def plot_tpot(ax):
    """Plot 3: Per-token latency (TPOT) vs context length."""
    for tag, json_dict, style in [("CPU", cpu_json, '-'), ("GPU", gpu_json, '--')]:
        for model_name in files.keys():
            if model_name not in json_dict:
                continue
            data = json_dict[model_name]
            if not has_new_metrics(data):
                continue

            color = colors[model_name]
            label_name = format_label(model_name)
            tokens = data["per_token_latencies"]

            ctx = np.array([t["context_len"] for t in tokens])
            lat = np.array([t["latency_s"] for t in tokens])

            # Smooth with a rolling average for readability
            window = max(1, len(lat) // 30)
            if window > 1:
                lat_smooth = np.convolve(lat, np.ones(window)/window, mode='valid')
                ctx_smooth = ctx[:len(lat_smooth)]
            else:
                lat_smooth = lat
                ctx_smooth = ctx

            label = f'{label_name} {tag}' if has_comparison else label_name
            ax.plot(ctx_smooth, lat_smooth, style, color=color, linewidth=1.5, label=label, alpha=0.8)

    ax.set_xlabel("Total Context Length", fontsize=12)
    ax.set_ylabel("Per-Token Latency (s)", fontsize=12)
    ax.set_title("TPOT — Per-Token Latency vs. Context Length", fontsize=14)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

def plot_summary_bars(ax):
    """Plot 4: Bar charts — TTFT, Mean TPOT, TPS."""
    model_names = [m for m in files.keys() if m in cpu_json or m in gpu_json]
    n_models = len(model_names)
    bar_width = 0.35
    x_pos = np.arange(n_models)

    # Collect metrics
    metrics = {}
    for tag, json_dict in [("CPU", cpu_json), ("GPU", gpu_json)]:
        ttft_vals = []
        tpot_vals = []
        tps_vals = []
        for m in model_names:
            if m in json_dict and has_new_metrics(json_dict[m]):
                d = json_dict[m]
                ttft_vals.append(d["ttft_s"])
                tpot_vals.append(d["mean_tpot_s"] * 1000)  # convert to ms
                tps_vals.append(d["output_tps"])
            else:
                ttft_vals.append(0)
                tpot_vals.append(0)
                tps_vals.append(0)
        metrics[tag] = {"ttft": ttft_vals, "tpot_ms": tpot_vals, "tps": tps_vals}

    ax.set_title("Summary Metrics — CPU vs GPU", fontsize=14)

    tags_present = [t for t in ["CPU", "GPU"] if any(v > 0 for v in metrics[t]["tps"])]

    labels = [format_label(m) for m in model_names]

    if len(tags_present) == 2:
        offsets = [-bar_width/2, bar_width/2]
        bar_colors = ['#5dade2', '#48c9b0']
    else:
        offsets = [0]
        bar_colors = ['#5dade2']

    # TPS bar chart
    for i, tag in enumerate(tags_present):
        vals = metrics[tag]["tps"]
        bars = ax.bar(x_pos + offsets[i], vals, bar_width, label=f'{tag}', color=bar_colors[i], alpha=0.85)
        for bar, val in zip(bars, vals):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                        f'{val:.1f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("Tokens / Second", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

    # Print summary table to console
    print("\n--- Summary Metrics ---")
    print(f"{'Model':<18} {'Tag':<5} {'TTFT (s)':<12} {'Mean TPOT (ms)':<16} {'TPS':<10} {'E2E (s)':<10}")
    print("-" * 71)
    for tag, json_dict in [("CPU", cpu_json), ("GPU", gpu_json)]:
        if not json_dict:
            continue
        for m in model_names:
            if m in json_dict and has_new_metrics(json_dict[m]):
                d = json_dict[m]
                print(f"{format_label(m):<18} {tag:<5} {d['ttft_s']:<12.4f} {d['mean_tpot_s']*1000:<16.2f} {d['output_tps']:<10.2f} {d['e2e_latency_s']:<10.2f}")

def save_individual_plot(plot_func, filename, figsize=(10, 7)):
    """Create a standalone figure for a single plot and save it."""
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    plot_func(ax)
    fig.tight_layout()
    path = os.path.join(PLOT_DIR, filename)
    fig.savefig(path, dpi=150)
    print(f"  -> Saved {path}")
    plt.close(fig)

# --- Determine layout ---
# Row 1: [Overlay plot] [Speedup plot]          (always present when comparison)
# Row 2: [TPOT vs ctx]  [TTFT / TPS bar charts] (when new metrics available)

if has_comparison and any_new_metrics:
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    ax_overlay = axes[0][0]
    ax_speedup = axes[0][1]
    ax_tpot = axes[1][0]
    ax_bars = axes[1][1]
elif has_comparison:
    fig, (ax_overlay, ax_speedup) = plt.subplots(1, 2, figsize=(18, 7))
    ax_tpot = None
    ax_bars = None
elif any_new_metrics:
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    ax_overlay = axes[0]
    ax_tpot = axes[1]
    ax_bars = axes[2]
    ax_speedup = None
else:
    fig, ax_overlay = plt.subplots(1, 1, figsize=(10, 7))
    ax_speedup = None
    ax_tpot = None
    ax_bars = None

# Draw combined figure
plot_overlay(ax_overlay)
if ax_speedup is not None:
    plot_speedup(ax_speedup)
if ax_tpot is not None:
    plot_tpot(ax_tpot)
if ax_bars is not None:
    plot_summary_bars(ax_bars)

plt.tight_layout()

# --- Save individual plots ---
os.makedirs(PLOT_DIR, exist_ok=True)

print("\n--- Saving individual plots ---")
save_individual_plot(plot_overlay, "overlay.png")
if has_comparison:
    save_individual_plot(plot_speedup, "speedup.png")
if any_new_metrics:
    save_individual_plot(plot_tpot, "tpot.png")
    save_individual_plot(plot_summary_bars, "summary.png")

# Save combined figure
combined_path = os.path.join(PLOT_DIR, "combined.png")
fig.savefig(combined_path, dpi=150)
print(f"  -> Saved {combined_path}")

plt.show()
