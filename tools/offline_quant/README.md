# offline_quant

Offline INT8 weight quantization tool for c_gpt2. Phase 1 of the W8A8 path
(see `docs/articles/2026-06-quant8-gpu/offline_quant_plan.md` for the full
design).

## What it does

Reads FP32 weights from `weights/gpt2[_<size>]_c_weights.bin`, quantizes the
4 GEMM matmul tensors per layer (`W_qkv` packed, `attn_proj`, `W1`, `W2`)
to per-channel symmetric INT8 (`scale = amax / 127`), passes the rest
(embeddings, LayerNorm params, biases) through as FP32, and writes
`gpt2_<size>_quant8.bin` plus a sidecar `quant_config.json` manifest.

## Quick start

```sh
cd tools/offline_quant
uv sync
uv run quantize.py --model small --valid       # quantize + validate one size
uv run quantize.py --list                       # tensor inventory
uv run quantize.py --stats                      # memory before/after
uv run quantize.py --distrib                    # interactive per-tensor plots
uv run quantize.py --compare                    # cross-tensor / cross-model plots
uv run quantize.py                              # default: quantize all sizes, prompt to install to weights/
```

Outputs land under `tools/offline_quant/out/{runs,scales,quant_files}/`.
Pass `--out-dir <path>` to redirect plot output (used to write article
assets directly into `docs/articles/2026-06-quant8-gpu/assets/plots/`).

## What does NOT belong here

The runtime activation-quant kernel, dequant kernel, and `cublasGemmEx`
INT8 wiring are deferred to a separate phase. This tool only produces the
weight-side artifact.
