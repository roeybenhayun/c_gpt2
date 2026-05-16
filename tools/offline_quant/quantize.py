"""Entry point for the offline INT8 weight quantization tool.

Mode flags are mutually exclusive (one verb per invocation). Default
(no mode flag) quantizes the selected model(s), writes the artifacts to
out/quant_files/, and interactively prompts to copy them to weights/.
"""
from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import questionary

from src.reader import (
    CTX_LEN,
    MODEL_CONFIGS,
    VOCAB_SIZE,
    Weights,
    default_weights_dir,
    load_fp32_weights,
)
from src.quantizer import quantize_per_channel, quantize_weights, dequantize
from src.writer import (
    PRESERVED_TENSOR_NAMES,
    QUANTIZED_TENSOR_NAMES,
    compute_memory_summary,
    write_quantized,
)

TOOL_DIR = Path(__file__).resolve().parent
DEFAULT_QUANT_DIR = TOOL_DIR / "out" / "quant_files"

# Extra "virtual" tensor names: W_q/W_k/W_v are row-slices of the packed W_qkv
# buffer (rows [0,d), [d,2d), [2d,3d) respectively). For per-channel scales,
# the slice's quantization is identical to slicing the full W_qkv quant result.
DISTRIB_TENSOR_CHOICES = QUANTIZED_TENSOR_NAMES + ["W_q", "W_k", "W_v"]


def _select_tensor(weights: Weights, layer_idx: int, tensor: str) -> "np.ndarray":
    layer = weights.layers[layer_idx]
    d = weights.config.d_model
    if tensor == "W_q":
        return layer.W_qkv[:d, :]
    if tensor == "W_k":
        return layer.W_qkv[d : 2 * d, :]
    if tensor == "W_v":
        return layer.W_qkv[2 * d :, :]
    return getattr(layer, tensor)


def _resolve_out_dir(out_dir_arg: str | None) -> Path:
    if out_dir_arg:
        p = Path(out_dir_arg).resolve()
    else:
        p = TOOL_DIR / "out" / "runs" / time.strftime("%Y-%m-%dT%H-%M-%S")
    p.mkdir(parents=True, exist_ok=True)
    return p


def _resolve_models(arg: str) -> list[str]:
    return list(MODEL_CONFIGS) if arg == "all" else [arg]


def cmd_list() -> int:
    print("Quantized GEMM tensors per layer (per-channel symmetric INT8, scale = amax/127):")
    for n in QUANTIZED_TENSOR_NAMES:
        print(f"  - {n}")
    print("\nPreserved tensors (stored FP32 in file; runtime cast to BF16 at load):")
    for n in PRESERVED_TENSOR_NAMES:
        print(f"  - {n}")
    print("\nPer-model shapes:")
    for size, cfg in MODEL_CONFIGS.items():
        print(
            f"  gpt2-{size}: d_model={cfg.d_model}, d_ff={cfg.d_ff}, "
            f"num_layers={cfg.num_layers}, vocab_size={VOCAB_SIZE}, ctx_len={CTX_LEN}"
        )
    return 0


def cmd_stats(model: str, chart: bool, out_dir_arg: str | None) -> int:
    sizes = _resolve_models(model)
    header = (
        f"{'Model':<11} {'Src FP32':>10} {'INT8':>8} "
        f"{'Preserved':>10} {'Scale':>10} {'Artifact':>10} {'Ratio':>8}"
    )
    print(header)
    print("-" * len(header))
    any_ok = False
    summaries: list[tuple[str, dict]] = []
    for size in sizes:
        try:
            w = load_fp32_weights(size)
        except FileNotFoundError as e:
            print(f"{size}: SKIP — {e}", file=sys.stderr)
            continue
        any_ok = True
        qw = quantize_weights(w)
        m = compute_memory_summary(qw)
        summaries.append((size, m))
        ratio = m["artifact_total_mb"] / max(m["source_fp32_total_mb"], 1)
        print(
            f"{'gpt2-'+size:<11} "
            f"{m['source_fp32_total_mb']:>9}M {m['quantized_int8_mb']:>7}M "
            f"{m['preserved_fp32_mb']:>9}M {m['scale_overhead_kb']:>9}K "
            f"{m['artifact_total_mb']:>9}M {ratio*100:>7.1f}%"
        )
    if chart and summaries:
        from src.plotting import plot_memory_breakdown
        out_dir = _resolve_out_dir(out_dir_arg)
        for size, m in summaries:
            p = plot_memory_breakdown(m, f"gpt2-{size}", out_dir / f"{size}_memory_breakdown.png")
            print(f"  chart: {p}")
    return 0 if any_ok else 1


def _install(produced: list[tuple[Path, Path]], dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    for bin_path, json_path in produced:
        for src in (bin_path, json_path):
            tgt = dest / src.name
            shutil.copy2(src, tgt)
            print(f"  installed {tgt}")


def cmd_compare(out_dir_arg: str | None) -> int:
    from src.plotting import (
        plot_cross_model_memory,
        plot_side_by_side_distributions,
    )

    out_dir = _resolve_out_dir(out_dir_arg)
    print(f"writing compare plots to {out_dir}")

    summaries: list[tuple[str, dict]] = []
    loaded: dict[str, Weights] = {}
    for size in MODEL_CONFIGS:
        try:
            w = load_fp32_weights(size)
        except FileNotFoundError as e:
            print(f"  {size}: SKIP — {e}", file=sys.stderr)
            continue
        loaded[size] = w
        summaries.append((size, compute_memory_summary(quantize_weights(w))))

    if not summaries:
        return 1

    p = plot_cross_model_memory(summaries, out_dir / "cross_model_memory.png")
    print(f"  {p}")

    # W_q vs W1 distribution comparison (use small layer 0 if available, else first loaded).
    src = loaded.get("small") or next(iter(loaded.values()))
    d = src.config.d_model
    W_q = src.layers[0].W_qkv[:d, :]
    W1 = src.layers[0].W1
    p = plot_side_by_side_distributions(
        [(f"gpt2-{src.config.name} layer 0  W_q  [{W_q.shape[0]} × {W_q.shape[1]}]", W_q),
         (f"gpt2-{src.config.name} layer 0  W1  [{W1.shape[0]} × {W1.shape[1]}]",   W1)],
        out_dir / f"compare_W_q_vs_W1_{src.config.name}_l0.png",
    )
    print(f"  {p}")
    return 0


def cmd_valid(model: str) -> int:
    from src.validator import summarize, validate

    sizes = _resolve_models(model)
    any_ok = False
    for size in sizes:
        try:
            w = load_fp32_weights(size)
        except FileNotFoundError as e:
            print(f"{size}: SKIP — {e}", file=sys.stderr)
            continue
        any_ok = True
        qw = quantize_weights(w)
        bin_path, json_path = write_quantized(qw, DEFAULT_QUANT_DIR)
        print(f"[{size}] wrote {bin_path.name} + {json_path.name}")
        errors = validate(size, bin_path)
        s = summarize(errors)
        print(f"\n[{size}] per-tensor-type error summary (across {qw.config.num_layers} layers):")
        header = (
            f"{'Tensor':<10} {'#layers':>8} "
            f"{'mean_rmse_FP32':>16} {'max|err|_FP32':>16} "
            f"{'mean_rmse_BF16':>16} {'max|err|_BF16':>16}"
        )
        print(header)
        print("-" * len(header))
        for tname in QUANTIZED_TENSOR_NAMES:
            d = s[tname]
            print(
                f"{tname:<10} {d['n_layers']:>8d} "
                f"{d['mean_rmse_vs_fp32']:>16.4e} {d['max_maxabs_vs_fp32']:>16.4e} "
                f"{d['mean_rmse_vs_bf16']:>16.4e} {d['max_maxabs_vs_bf16']:>16.4e}"
            )
        # Aggregate RMSE for a per-channel-quant tensor with uniform rounding noise:
        #   E[err²] = mean_i(scale_i² / 12)  →  rmse = sqrt(mean(scale²)/12)
        l0 = qw.layers[0].W_qkv
        expected = float(np.sqrt(np.mean(l0.scale ** 2) / 12.0))
        rmse_l0 = float(np.sqrt(np.mean(
            (l0.int8.astype(np.float32) * l0.scale[:, None] - w.layers[0].W_qkv) ** 2
        )))
        print(f"\n  sanity: layer-0 W_qkv rmse={rmse_l0:.4e}  expected={expected:.4e}  ratio={rmse_l0/expected:.3f}")
        print()
    return 0 if any_ok else 1


def cmd_distrib(model: str, layer_arg: int | None, tensor_arg: str | None, out_dir_arg: str | None) -> int:
    from src.plotting import (
        plot_dequant_error_hist,
        plot_distribution_with_amax,
        plot_fp32_vs_dequant_scatter,
        render_mapping_examples,
    )

    if model == "all":
        size = questionary.select("Model size?", choices=list(MODEL_CONFIGS)).ask()
        if not size:
            return 1
    else:
        size = model

    try:
        w = load_fp32_weights(size)
    except FileNotFoundError as e:
        print(f"{size}: {e}", file=sys.stderr)
        return 1
    cfg = w.config

    if layer_arg is None:
        layer_pick = questionary.select(
            "Layer index?",
            choices=[str(i) for i in range(cfg.num_layers)],
        ).ask()
        if layer_pick is None:
            return 1
        layer_idx = int(layer_pick)
    else:
        if not (0 <= layer_arg < cfg.num_layers):
            print(f"--layer {layer_arg} out of range [0, {cfg.num_layers})", file=sys.stderr)
            return 1
        layer_idx = layer_arg

    if tensor_arg is None:
        tensor = questionary.select("Tensor?", choices=DISTRIB_TENSOR_CHOICES).ask()
        if tensor is None:
            return 1
    else:
        if tensor_arg not in DISTRIB_TENSOR_CHOICES:
            print(f"--tensor {tensor_arg!r} not in {DISTRIB_TENSOR_CHOICES}", file=sys.stderr)
            return 1
        tensor = tensor_arg

    W = _select_tensor(w, layer_idx, tensor)
    qt = quantize_per_channel(W)
    W_recon = dequantize(qt)
    out_dim, in_dim = W.shape
    label = f"gpt2-{size} layer {layer_idx} {tensor}  [{out_dim} × {in_dim}]"
    out_dir = _resolve_out_dir(out_dir_arg)
    stem = f"{size}_l{layer_idx}_{tensor}"
    print(f"writing 4 plots to {out_dir}")
    plot_distribution_with_amax(W, label,           out_dir / f"{stem}_distribution.png")
    plot_dequant_error_hist(W, W_recon, label,      out_dir / f"{stem}_dequant_error.png")
    plot_fp32_vs_dequant_scatter(W, W_recon, label, out_dir / f"{stem}_scatter.png")
    render_mapping_examples(W, qt, label,           out_dir / f"{stem}_mapping_table.png")
    print(f"  amax = {float(np.max(np.abs(W))):.6f}")
    print(
        f"  rmse = {float(np.sqrt(np.mean((W - W_recon) ** 2))):.4e}"
        f"   max|err| = {float(np.max(np.abs(W - W_recon))):.4e}"
    )
    return 0


def cmd_quantize(model: str, do_install: bool) -> int:
    sizes = _resolve_models(model)
    produced: list[tuple[Path, Path]] = []
    for size in sizes:
        try:
            w = load_fp32_weights(size)
        except FileNotFoundError as e:
            print(f"{size}: SKIP — {e}", file=sys.stderr)
            continue
        t0 = time.time()
        qw = quantize_weights(w)
        bin_path, json_path = write_quantized(qw, DEFAULT_QUANT_DIR)
        elapsed = time.time() - t0
        m = compute_memory_summary(qw)
        print(
            f"[{size}] wrote {bin_path.name} ({m['artifact_total_mb']} MiB) "
            f"+ {json_path.name} in {elapsed:.1f}s"
        )
        produced.append((bin_path, json_path))

    if not produced:
        return 1

    weights_dir = default_weights_dir()
    if do_install:
        _install(produced, weights_dir)
        return 0

    ok = questionary.confirm(
        f"Copy {len(produced)} quant artifact(s) to {weights_dir}?",
        default=False,
    ).ask()
    if ok:
        _install(produced, weights_dir)
    else:
        print(f"Skipped install. Artifacts remain in {DEFAULT_QUANT_DIR}.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Offline INT8 weight quantization for c_gpt2 (Phase 1 of W8A8).",
    )
    parser.add_argument(
        "--model", choices=list(MODEL_CONFIGS) + ["all"], default="all",
        help="model size to operate on (default: all)",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--list", action="store_true",
                      help="list quantized vs preserved tensors + per-model shapes")
    mode.add_argument("--stats", action="store_true",
                      help="memory before/after summary per model")
    mode.add_argument("--install", action="store_true",
                      help="quantize and copy artifacts to weights/ non-interactively")
    mode.add_argument("--distrib", action="store_true",
                      help="interactive: pick layer+tensor; emit 4 per-tensor plots")
    mode.add_argument("--valid", action="store_true",
                      help="quantize, write .bin+.json, then read back and verify per-tensor error")
    mode.add_argument("--compare", action="store_true",
                      help="batch: cross-tensor and cross-model composite plots")
    parser.add_argument("--layer", type=int, default=None,
                        help="(--distrib) layer index — skips interactive prompt")
    parser.add_argument("--tensor", default=None,
                        help=f"(--distrib) tensor name in {DISTRIB_TENSOR_CHOICES} — skips interactive prompt")
    parser.add_argument("--chart", action="store_true",
                        help="(--stats) also emit a memory-breakdown bar chart")
    parser.add_argument("--out-dir", default=None,
                        help="output directory for plots (default: out/runs/<timestamp>/)")
    args = parser.parse_args(argv)

    if args.list:
        return cmd_list()
    if args.stats:
        return cmd_stats(args.model, args.chart, args.out_dir)
    if args.distrib:
        return cmd_distrib(args.model, args.layer, args.tensor, args.out_dir)
    if args.valid:
        return cmd_valid(args.model)
    if args.compare:
        return cmd_compare(args.out_dir)
    return cmd_quantize(args.model, do_install=args.install)


if __name__ == "__main__":
    sys.exit(main())
