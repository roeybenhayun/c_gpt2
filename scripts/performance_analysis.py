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

def discover_files(tag=None, preset=None):
    """Discover latest log files, optionally filtered by tag (cpu/gpu/bf16) and/or preset."""
    found = {}
    for model_name in files.keys():
        if tag and preset:
            search_pattern = os.path.join(LOG_DIR, f"{model_name}_{tag}_{preset}_*.json")
        elif tag:
            search_pattern = os.path.join(LOG_DIR, f"{model_name}_{tag}_*.json")
        else:
            search_pattern = os.path.join(LOG_DIR, f"{model_name}_*.json")
        matched = glob.glob(search_pattern)
        if not matched:
            continue
        latest_file = max(matched, key=os.path.getmtime)
        found[model_name] = latest_file
        scope = " ".join(filter(None, [tag, preset])) or "any"
        print(f"  -> Found latest for '{model_name}' ({scope}): {os.path.basename(latest_file)}")
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

def with_preset_suffix(title):
    """Append the active preset (DECODE / PREFILL / BALANCED) to a chart title."""
    return f"{title} — {preset_label} preset" if preset_label else title

def workload_shape():
    """Return (prompt_tokens, output_tokens) from any loaded JSON, or None if not available.
    All series in a preset run share the same workload, so any JSON works."""
    for series in (cpu_json, gpu_json, bf16_json):
        for d in series.values():
            if "initial_prompt_len" in d and "requested_out_tokens" in d:
                return d["initial_prompt_len"], d["requested_out_tokens"]
    return None

def title_with_workload(title):
    """Decorate a chart title with the preset suffix and a 'P prompt tokens + N output tokens' subtitle."""
    decorated = with_preset_suffix(title)
    shape = workload_shape()
    if shape is not None:
        p_tok, o_tok = shape
        decorated = f"{decorated}\n({p_tok} prompt tokens + {o_tok} output tokens)"
    return decorated

# --- Mode Selection & Discovery ---

is_manual_mode = any(files.values())

# Parse CLI flags.
#   Series flags: --cpu, --gpu, --bf16  (if none given, all three)
#   Preset flag : --decode | --prefill | --balanced  (matches run.sh's preset
#                  embedded in log filenames; if none given, no preset filter)
cli_args = set(sys.argv[1:])
want_cpu = "--cpu" in cli_args
want_gpu = "--gpu" in cli_args
want_bf16 = "--bf16" in cli_args
if not want_cpu and not want_gpu and not want_bf16:
    want_cpu = True
    want_gpu = True
    want_bf16 = True

preset_flags = {"--decode", "--prefill", "--balanced"} & cli_args
if len(preset_flags) > 1:
    print(f"Error: presets are mutually exclusive (got {sorted(preset_flags)})", file=sys.stderr)
    sys.exit(1)
preset_filter = next(iter(preset_flags), "").lstrip("-") or None  # "decode" | "prefill" | "balanced" | None
preset_label = preset_filter.upper() if preset_filter else None

if is_manual_mode:
    print("--- Running in MANUAL mode ---")
    manual_files = {model: path for model, path in files.items() if path}
    for model, path in manual_files.items():
        print(f"  -> Using manual path for '{model}': {path}")
    cpu_files = {}
    gpu_files = {}
    bf16_files = {}
    solo_files = manual_files
else:
    print("--- Running in AUTOMATIC discovery mode ---")
    if preset_filter:
        print(f"  Preset filter: {preset_filter}")
    cpu_files = discover_files("cpu", preset_filter) if want_cpu else {}
    gpu_files = discover_files("gpu", preset_filter) if want_gpu else {}
    bf16_files = discover_files("bf16", preset_filter) if want_bf16 else {}
    solo_files = {}

    if not cpu_files and not gpu_files and not bf16_files:
        if preset_filter:
            print(
                f"  No logs found for preset '{preset_filter}'. "
                f"Run `./scripts/run.sh --gpu --bf16 --{preset_filter}` first.",
                file=sys.stderr,
            )
            sys.exit(1)
        print("  No cpu/gpu/bf16 tagged files found, falling back to untagged discovery...")
        solo_files = discover_files()

has_comparison = sum(bool(x) for x in (cpu_files, gpu_files, bf16_files)) >= 2

all_files = cpu_files or gpu_files or bf16_files or solo_files
if not all_files:
    raise FileNotFoundError("No files to process. Please fill in paths manually or run the bash script to generate log files.")

print("\n--- Processing and plotting data ---")

# --- Load all data ---

cpu_data = {}
gpu_data = {}
bf16_data = {}
cpu_json = {}
gpu_json = {}
bf16_json = {}
chunk_size_label = None

def _load_into(model_name, file_path, data_dict, json_dict):
    global chunk_size_label
    x, y, order, cs = load_run_data(file_path)
    data_dict[model_name] = (x, y, order)
    json_dict[model_name] = load_json(file_path)
    if chunk_size_label is None:
        chunk_size_label = cs

for model_name in files.keys():
    if model_name in cpu_files:
        _load_into(model_name, cpu_files[model_name], cpu_data, cpu_json)
    if model_name in gpu_files:
        _load_into(model_name, gpu_files[model_name], gpu_data, gpu_json)
    if model_name in bf16_files:
        _load_into(model_name, bf16_files[model_name], bf16_data, bf16_json)
    if model_name in solo_files:
        _load_into(model_name, solo_files[model_name], cpu_data, cpu_json)

# --- Check if new metrics are available ---

def has_new_metrics(data):
    return "ttft_s" in data and "per_token_latencies" in data

any_new_metrics = any(
    has_new_metrics(d)
    for d in list(cpu_json.values()) + list(gpu_json.values()) + list(bf16_json.values())
)

# Series metadata: (data_dict, json_dict, label, line_style, marker, bar_color)
SERIES = [
    (cpu_data,  cpu_json,  "CPU",      "-",  "o", "#5dade2"),
    (gpu_data,  gpu_json,  "GPU FP32", "--", "s", "#48c9b0"),
    (bf16_data, bf16_json, "GPU BF16", ":",  "^", "#f5b041"),
]
# Filter to series that actually have data
ACTIVE_SERIES = [s for s in SERIES if s[0]]

PLOT_DIR = "plots"

# --- Plotting functions ---

def plot_overlay(ax):
    """Plot 1: Overlay — one line style per series (CPU/GPU/BF16)."""
    for model_name in files.keys():
        color = colors[model_name]
        label_name = format_label(model_name)
        for data_dict, _json, series_label, style, marker, _bc in ACTIVE_SERIES:
            if model_name not in data_dict:
                continue
            x, y, order = data_dict[model_name]
            x_fit, y_fit = fit_curve(x, y, order)
            ax.plot(x, y, marker, color=color, alpha=0.4, markersize=5)
            label = f'{label_name} {series_label}' if has_comparison else label_name
            ax.plot(x_fit, y_fit, style, color=color, linewidth=2, label=label)

    ax.set_xlabel("Total Context Length", fontsize=12)
    ax.set_ylabel(f"Time for Last {chunk_size_label} Tokens (s)", fontsize=12)
    if has_comparison:
        title_parts = [s[2] for s in ACTIVE_SERIES]
        ax.set_title(with_preset_suffix(f"{' vs '.join(title_parts)} — Time to Generate Tokens"), fontsize=14)
    else:
        ax.set_title(with_preset_suffix("Time to Generate Tokens vs. Context Length"), fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

def _plot_ratio(ax, baseline_data, baseline_label, accel_pairs, ylabel, title):
    """Generic speedup plot: baseline_time / accel_time for each (data, label, style)."""
    for model_name in files.keys():
        if model_name not in baseline_data:
            continue
        color = colors[model_name]
        label_name = format_label(model_name)
        base_x, base_y, _ = baseline_data[model_name]

        for data_dict, series_label, style in accel_pairs:
            if model_name not in data_dict:
                continue
            ax_x, ax_y, _ = data_dict[model_name]

            common_min = max(base_x.min(), ax_x.min())
            common_max = min(base_x.max(), ax_x.max())
            common_x = np.linspace(common_min, common_max, 50)

            base_interp = np.interp(common_x, base_x, base_y)
            acc_interp = np.interp(common_x, ax_x, ax_y)
            speedup = base_interp / acc_interp

            line_label = f'{label_name} {series_label}' if len(accel_pairs) > 1 else label_name
            ax.plot(common_x, speedup, style, color=color, linewidth=2, label=line_label)
            avg_speedup = np.mean(speedup)
            ax.axhline(y=avg_speedup, color=color, linestyle=':', alpha=0.3)
            ax.text(common_x[-1] + 10, avg_speedup, f'{avg_speedup:.1f}x',
                    color=color, fontsize=10, fontweight='bold', va='center')

    ax.axhline(y=1.0, color='gray', linestyle='-', alpha=0.3)
    ax.set_xlabel("Total Context Length", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(with_preset_suffix(title), fontsize=14)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

def _plot_speedup_pair(ax, baseline_data, baseline_label, accel_data, accel_label, title):
    """Plot 6 mutual-speedup curves: for each model, BOTH directions of the
    speedup ratio. The BF16 line is FP32_time / BF16_time (above 1× when BF16
    wins). The FP32 line is BF16_time / FP32_time (above 1× when FP32 wins).
    The two curves are reciprocals of each other, so they mirror around 1.0×
    and are visually distinct (no overlap on a flat baseline)."""
    for model_name in files.keys():
        base_xy = baseline_data.get(model_name)
        acc_xy = accel_data.get(model_name)
        if base_xy is None or acc_xy is None:
            continue

        color = colors[model_name]
        label_name = format_label(model_name)

        base_x, base_y, _ = base_xy
        acc_x, acc_y, _ = acc_xy

        common_min = max(base_x.min(), acc_x.min())
        common_max = min(base_x.max(), acc_x.max())
        common_x = np.linspace(common_min, common_max, 50)

        base_interp = np.interp(common_x, base_x, base_y)
        acc_interp = np.interp(common_x, acc_x, acc_y)

        # BF16 speedup over FP32 (>1 when BF16 wins).
        bf16_speedup = base_interp / acc_interp
        ax.plot(common_x, bf16_speedup, '-', color=color, linewidth=2,
                label=f'{label_name} {accel_label} speedup')

        # FP32 speedup over BF16 (>1 when FP32 wins).
        fp32_speedup = acc_interp / base_interp
        ax.plot(common_x, fp32_speedup, '--', color=color, linewidth=1.8,
                alpha=0.85,
                label=f'{label_name} {baseline_label} speedup')

        # Per-model mean BF16 speedup annotation on the right edge.
        mean_bf16 = float(np.mean(bf16_speedup))
        ax.text(common_max + (common_max - common_min) * 0.01, mean_bf16,
                f'{mean_bf16:.2f}×', color=color,
                fontsize=10, fontweight='bold', va='center')

    ax.axhline(y=1.0, color='gray', linestyle='-', alpha=0.3)
    ax.set_xlabel("Total Context Length", fontsize=12)
    ax.set_ylabel(f"Speedup (>1× = winner) — solid: {accel_label}, dashed: {baseline_label}",
                  fontsize=10)
    ax.set_title(with_preset_suffix(title), fontsize=14)
    ax.legend(fontsize=8, loc='upper left', ncol=2)
    ax.grid(True, alpha=0.3)

def plot_speedup(ax):
    """Plot 2: Speedup ratio.
    - If CPU data is present: GPU FP32/CPU + GPU BF16/CPU, 6 curves.
    - Else if both GPU dtypes present: per-model FP32 baseline (flat 1.0×)
      + BF16/FP32 speedup, 6 curves total (2 per model).
    - Else: empty.
    """
    if cpu_data:
        accel_pairs = [
            (gpu_data,  "GPU FP32", "-"),
            (bf16_data, "GPU BF16", "--"),
        ]
        accel_pairs = [p for p in accel_pairs if p[0]]
        _plot_ratio(ax, cpu_data, "CPU",
                    accel_pairs,
                    ylabel="Speedup (CPU time / GPU time)",
                    title="GPU Speedup over CPU")
    elif gpu_data and bf16_data:
        _plot_speedup_pair(ax, gpu_data, "GPU FP32", bf16_data, "GPU BF16",
                           title="GPU FP32 vs BF16 — Mutual Speedup")
    else:
        ax.text(0.5, 0.5, "Need at least two series for a speedup plot",
                ha='center', va='center', transform=ax.transAxes)

def plot_tpot(ax):
    """Plot 3: Per-token latency (TPOT) vs context length."""
    for _data, json_dict, series_label, style, _marker, _bc in ACTIVE_SERIES:
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

            label = f'{label_name} {series_label}' if has_comparison else label_name
            ax.plot(ctx_smooth, lat_smooth, style, color=color, linewidth=1.5, label=label, alpha=0.8)

    ax.set_xlabel("Total Context Length", fontsize=12)
    ax.set_ylabel("Per-Token Latency (s)", fontsize=12)
    ax.set_title(with_preset_suffix("TPOT — Per-Token Latency vs. Context Length"), fontsize=14)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

def plot_summary_bars(ax):
    """Plot 4: TPS bar chart — one bar group per series (CPU/GPU/BF16)."""
    all_jsons = [s[1] for s in ACTIVE_SERIES]
    model_names = [m for m in files.keys() if any(m in j for j in all_jsons)]
    n_models = len(model_names)
    n_series = len(ACTIVE_SERIES)

    if n_models == 0 or n_series == 0:
        return

    total_width = 0.8
    bar_width = total_width / n_series
    x_pos = np.arange(n_models)
    # Centre the bar group around each x_pos
    offsets = [(-total_width/2) + (i + 0.5) * bar_width for i in range(n_series)]

    ax.set_title(title_with_workload("Summary Metrics — TPS by series"), fontsize=14)
    labels = [format_label(m) for m in model_names]

    for i, (_data, json_dict, series_label, _style, _marker, bar_color) in enumerate(ACTIVE_SERIES):
        vals = []
        for m in model_names:
            if m in json_dict and has_new_metrics(json_dict[m]):
                vals.append(json_dict[m]["output_tps"])
            else:
                vals.append(0)
        bars = ax.bar(x_pos + offsets[i], vals, bar_width,
                      label=series_label, color=bar_color, alpha=0.85)
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
    print(f"{'Model':<18} {'Tag':<10} {'TTFT (s)':<12} {'Mean TPOT (ms)':<16} {'TPS':<10} {'E2E (s)':<10}")
    print("-" * 76)
    for _data, json_dict, series_label, _style, _marker, _bc in ACTIVE_SERIES:
        if not json_dict:
            continue
        for m in model_names:
            if m in json_dict and has_new_metrics(json_dict[m]):
                d = json_dict[m]
                print(f"{format_label(m):<18} {series_label:<10} {d['ttft_s']:<12.4f} {d['mean_tpot_s']*1000:<16.2f} {d['output_tps']:<10.2f} {d['e2e_latency_s']:<10.2f}")

def plot_ttft_bars(ax):
    """Chart A — bar chart of TTFT (pure prefill time) per (model, series).

    Reads `ttft_s` straight from each JSON. The "headline" metric for any
    workload that exercises a non-trivial prefill (especially --prefill).
    """
    all_jsons = [s[1] for s in ACTIVE_SERIES]
    model_names = [m for m in files.keys() if any(m in j for j in all_jsons)]
    n_models = len(model_names)
    n_series = len(ACTIVE_SERIES)

    if n_models == 0 or n_series == 0:
        ax.text(0.5, 0.5, "No data", ha='center', va='center', transform=ax.transAxes)
        return

    total_width = 0.8
    bar_width = total_width / n_series
    x_pos = np.arange(n_models)
    offsets = [(-total_width/2) + (i + 0.5) * bar_width for i in range(n_series)]

    ax.set_title(title_with_workload("TTFT — Time to First Token"), fontsize=14)
    labels = [format_label(m) for m in model_names]

    # Collect TTFTs per series so we can also annotate the BF16 / FP32 ratio.
    ttft_by_series = {}
    for i, (_data, json_dict, series_label, _style, _marker, bar_color) in enumerate(ACTIVE_SERIES):
        vals = []
        for m in model_names:
            if m in json_dict and has_new_metrics(json_dict[m]):
                vals.append(json_dict[m]["ttft_s"])
            else:
                vals.append(0)
        ttft_by_series[series_label] = vals
        bars = ax.bar(x_pos + offsets[i], vals, bar_width,
                      label=series_label, color=bar_color, alpha=0.9)
        for bar, val in zip(bars, vals):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                        f'{val*1000:.0f} ms',
                        ha='center', va='bottom', fontsize=9, fontweight='bold')

    # Annotate each model with the BF16-vs-FP32 time delta.
    #   Positive %  → BF16 takes more time (slower than FP32)
    #   Negative %  → BF16 takes less time (faster than FP32)
    # e.g. FP32=66ms, BF16=77ms → +16.7% (BF16 is 16.7% slower).
    if "GPU FP32" in ttft_by_series and "GPU BF16" in ttft_by_series:
        fp32_vals = ttft_by_series["GPU FP32"]
        bf16_vals = ttft_by_series["GPU BF16"]
        for i, (f, b) in enumerate(zip(fp32_vals, bf16_vals)):
            if f > 0 and b > 0:
                delta_pct = (b - f) / f * 100.0
                top = max(f, b) * 1.18
                # Color the badge so good (BF16 faster) and bad (BF16 slower)
                # are immediately readable.
                edge = '#7caf50' if delta_pct < 0 else '#c25555'
                ax.annotate(f'{delta_pct:+.1f}%',
                            xy=(x_pos[i], top),
                            ha='center', va='center',
                            fontsize=11, fontweight='bold',
                            color='#333333',
                            bbox=dict(boxstyle='round,pad=0.3',
                                      fc='white', ec=edge, lw=1.2))

    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("TTFT (seconds)", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    # Add headroom for the ratio annotation
    if any(any(v) for v in ttft_by_series.values()):
        max_v = max((v for s in ttft_by_series.values() for v in s), default=0)
        if max_v > 0:
            ax.set_ylim(0, max_v * 1.30)


def plot_phase_decomp_bars(ax):
    """Chart C — stacked bar showing where the wall clock goes.

    Each bar is split into:
      - bottom segment: TTFT  (prefill)
      - top segment:    e2e_latency_s - TTFT  (decode tail)
    The full bar height equals the e2e wall-clock time. Reading paired bars
    for FP32 and BF16, the prefill segment is where BF16 wins; the decode
    segment is where it doesn't.
    """
    all_jsons = [s[1] for s in ACTIVE_SERIES]
    model_names = [m for m in files.keys() if any(m in j for j in all_jsons)]
    n_models = len(model_names)
    n_series = len(ACTIVE_SERIES)

    if n_models == 0 or n_series == 0:
        ax.text(0.5, 0.5, "No data", ha='center', va='center', transform=ax.transAxes)
        return

    total_width = 0.8
    bar_width = total_width / n_series
    x_pos = np.arange(n_models)
    offsets = [(-total_width/2) + (i + 0.5) * bar_width for i in range(n_series)]

    ax.set_title(title_with_workload("Phase decomposition: prefill + decode"), fontsize=14)
    labels = [format_label(m) for m in model_names]

    # Track legend handles so each series shows up once with both phases.
    legend_seen = set()

    for i, (_data, json_dict, series_label, _style, _marker, bar_color) in enumerate(ACTIVE_SERIES):
        prefill_vals = []
        decode_vals = []
        for m in model_names:
            if m in json_dict and has_new_metrics(json_dict[m]):
                d = json_dict[m]
                ttft = d["ttft_s"]
                e2e = d["e2e_latency_s"]
                prefill_vals.append(ttft)
                decode_vals.append(max(0.0, e2e - ttft))
            else:
                prefill_vals.append(0)
                decode_vals.append(0)

        xs = x_pos + offsets[i]

        # Prefill (bottom) — full color, the "interesting" segment.
        prefill_label = f"{series_label} — prefill" if series_label not in legend_seen else None
        ax.bar(xs, prefill_vals, bar_width,
               color=bar_color, alpha=1.0,
               edgecolor='white', linewidth=0.5,
               label=prefill_label)

        # Decode (top) — same hue, faded, so it reads as "the tail".
        decode_label = f"{series_label} — decode" if series_label not in legend_seen else None
        ax.bar(xs, decode_vals, bar_width, bottom=prefill_vals,
               color=bar_color, alpha=0.35,
               edgecolor='white', linewidth=0.5,
               label=decode_label)

        legend_seen.add(series_label)

        # Per-segment labels in milliseconds, centred in each rectangle when
        # the segment is tall enough to fit the text (>=5% of the bar's total).
        for j, (px, p, dv) in enumerate(zip(xs, prefill_vals, decode_vals)):
            total = p + dv
            if total <= 0:
                continue
            if p > 0.05 * total:
                ax.text(px, p / 2, f'prefill\n{p*1000:.0f} ms',
                        ha='center', va='center',
                        fontsize=8, color='white', fontweight='bold')
            if dv > 0.05 * total:
                ax.text(px, p + dv / 2, f'decode\n{dv*1000:.0f} ms',
                        ha='center', va='center',
                        fontsize=8, color='#222222')
            # Total above the bar
            ax.text(px, total, f'{total*1000:.0f} ms',
                    ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("Wall clock (seconds)", fontsize=12)
    ax.legend(fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3, axis='y')


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
# - prefill preset: the sweep plots (overlay, speedup-vs-context, TPOT) are
#   degenerate at out_tokens=32, so swap them out for the new prefill-aware
#   charts (TTFT bar + phase decomposition).
# - everything else (decode preset / no preset / balanced): keep the existing
#   2×2 sweep+bars layout that works for multi-chunk decode runs.
is_prefill_layout = preset_filter == "prefill"

ax_overlay = ax_speedup = ax_tpot = ax_bars = ax_ttft = ax_phase = None

if is_prefill_layout and any_new_metrics:
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    ax_ttft = axes[0]
    ax_phase = axes[1]
elif preset_filter == "decode" and has_comparison and any_new_metrics:
    # Decode preset: prefill is tiny (~1% of wall time) so the TTFT-bars
    # and phase-decomp panels carry almost no signal. Keep just the four
    # panels that are genuinely informative for decode-bound runs.
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    ax_overlay = axes[0][0]   # raw FP32 + BF16 latency curves (6 lines)
    ax_speedup = axes[0][1]   # mutual BF16/FP32 speedup (6 lines)
    ax_tpot    = axes[1][0]   # mean time per output token vs context
    ax_bars    = axes[1][1]   # TPS summary bars
elif has_comparison and any_new_metrics:
    # Balanced (or no preset): everything is meaningful — show all 6 panels.
    # Row 0 — curves over context length (the sweep view).
    # Row 1 — summary bars (the at-a-glance view).
    fig, axes = plt.subplots(2, 3, figsize=(24, 14))
    ax_overlay = axes[0][0]   # raw FP32 + BF16 latency curves (6 lines)
    ax_speedup = axes[0][1]   # mutual BF16/FP32 speedup (6 lines)
    ax_tpot    = axes[0][2]   # mean time per output token vs context
    ax_bars    = axes[1][0]   # TPS summary bars
    ax_ttft    = axes[1][1]   # TTFT prefill bars (FP32 vs BF16)
    ax_phase   = axes[1][2]   # prefill vs decode phase decomposition
elif has_comparison:
    fig, (ax_overlay, ax_speedup) = plt.subplots(1, 2, figsize=(18, 7))
elif any_new_metrics:
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    ax_overlay = axes[0]
    ax_tpot = axes[1]
    ax_bars = axes[2]
else:
    fig, ax_overlay = plt.subplots(1, 1, figsize=(10, 7))

# Draw combined figure
if ax_overlay is not None:
    plot_overlay(ax_overlay)
if ax_speedup is not None:
    plot_speedup(ax_speedup)
if ax_tpot is not None:
    plot_tpot(ax_tpot)
if ax_bars is not None:
    plot_summary_bars(ax_bars)
if ax_ttft is not None:
    plot_ttft_bars(ax_ttft)
if ax_phase is not None:
    plot_phase_decomp_bars(ax_phase)

plt.tight_layout()

# --- Save individual plots ---
os.makedirs(PLOT_DIR, exist_ok=True)

print("\n--- Saving individual plots ---")
if is_prefill_layout and any_new_metrics:
    # For prefill, only the new charts are meaningful — skip the degenerate ones.
    save_individual_plot(plot_ttft_bars, "ttft.png")
    save_individual_plot(plot_phase_decomp_bars, "phase_decomp.png")
else:
    save_individual_plot(plot_overlay, "overlay.png")
    if has_comparison:
        save_individual_plot(plot_speedup, "speedup.png")
    if any_new_metrics:
        save_individual_plot(plot_tpot, "tpot.png")
        save_individual_plot(plot_summary_bars, "summary.png")
        # The phase / TTFT charts are also useful outside the prefill preset
        # (e.g. balanced) — render them as standalone PNGs for any preset that
        # has the new metrics.
        save_individual_plot(plot_ttft_bars, "ttft.png")
        save_individual_plot(plot_phase_decomp_bars, "phase_decomp.png")

# Save combined figure
combined_path = os.path.join(PLOT_DIR, "combined.png")
fig.savefig(combined_path, dpi=150)
print(f"  -> Saved {combined_path}")

plt.show()
