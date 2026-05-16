# Offline INT8 Weight Quantization Tool — Phase 1 of W8A8

## Context

The c_gpt2 project finished FP32 → BF16 in article 4 (`docs/articles/2026-05-fp32-to-bf16-gpu/`). The next step is W8A8 INT8. The cuBLAS-only constraint forces W8A8 (cuBLAS has no W8A16 GEMM path), which splits cleanly:

- **Offline (this plan)**: read FP32 weights from existing `weights/gpt2_<size>_c_weights.bin`, produce a new `.bin` where the 4 large matmul tensors per layer are replaced with INT8 + per-channel FP32 scales; preserved tensors stay as FP32 (existing runtime cast to BF16 keeps working unchanged); package alongside `quant_config.json` manifest.
- **Runtime (future phase, not this plan)**: per-token dynamic activation quant kernel before each GEMM, dequant kernel after each GEMM, cuBLAS INT8 path.

Design principle: minimum runtime disruption. The new `.bin` is a surgical edit of the existing file format — only the 4 matmul tensors per layer change. Everything else (embeddings, LN params, biases) stays FP32 in the file, and the existing FP32 → BF16 cast at load (`fread_weights_or_exit`) continues to handle them. This keeps the runtime weight type decoupled from the file format and reuses the existing loader code for ~80% of tensors.

The artifact this plan produces is portable across all CUDA targets the engine supports (RTX 5080, H100, Jetson Xavier NX, etc.) — same file, different speed curves per device.

### cuBLAS W8A8 path

The W8A8 commitment comes from cuBLAS. The relevant kernel is `cublasGemmEx` with both inputs as `CUDA_R_8I` (INT8), output as `CUDA_R_32I` (INT32), and compute type `CUBLAS_COMPUTE_32I`. cuBLAS has no W8A16 / weight-only path — to use cuBLAS for INT8 GEMM, both A (weights) and B (activations) must be INT8. So to keep the heavy GEMM work on cuBLAS (no custom CUDA GEMM kernel) we commit to W8A8 across all matmuls. The offline tool produces the W side (this plan); a future runtime kernel produces the X side per token.

### Why only the GEMM weights are quantized

- **Tensor core acceleration applies only to GEMM.** INT8 tensor cores deliver real compute and bandwidth wins for `cublasGemmEx`. Other ops in the network — LayerNorm, GELU, softmax, residual add — are element-wise or non-linear and need float math (mean/variance, exp/erf, addition into a float residual stream). There's no INT8 kernel worth using for them.
- **Sensitivity vs. savings.** LN gamma/beta and biases add directly into activations with no GEMM accumulation to smooth quant noise. Quantizing them costs accuracy where it hurts most while saving only kilobytes — bad trade.
- **Embeddings (`wte`, `wpe`) are lookups, not GEMMs.** INT8 buys nothing on speed, and the noise would feed straight into LN1 with no smoothing.

### Offline ↔ runtime split

```
   OFFLINE (this plan)                          RUNTIME (future phase, per forward pass)
   ──────────────────                          ─────────────────────────────────────────

   FP32 weights from HF                         X_bf16  (from prev op: LN/GELU/residual)
        │                                            │
        ▼                                            ▼
   [quantize.py, per-channel /127]              [quant kernel: per-token amax → INT8]
        │                                            │
        ▼                                            ▼
   gpt2_<size>_quant8.bin                       X_int8 + scale_X
   ┌────────────────────────┐                       │
   │ INT8 W_qkv   + scale_W │                       │
   │ INT8 attn_proj+ scale  │  GPT-2 startup        │
   │ INT8 W1      + scale   │  (cudaMemcpy)         │
   │ INT8 W2      + scale   │ ─────────► [Device memory: W_int8 + scale_W]
   │ FP32 preserved         │                       │  per GEMM
   │   (wte, wpe, LN, b*)   │                       ▼  ▼
   └────────────────────────┘                cuBLAS INT8 GEMM (W · X)
                                                    │
                                                 Y_int32
                                                    │
                                                    ▼
                                             [dequant + bias kernel]
                                             scale_W · scale_X · Y_int32 + bias_bf16
                                                    │
                                                    ▼
                                                 Y_bf16
                                                    │
                                                    ▼
                                           LN / GELU / residual  (all BF16)
                                                    │
                                                    └─► next GEMM cycle
```

At GPT-2 startup, the loader reads the `.bin` and copies INT8 weights + `scale_W` into device memory via `cudaMemcpy`. They stay resident there for the process lifetime; cuBLAS reads from device memory on every GEMM call. The `scale_X` vector is recomputed per GEMM at runtime by the activation quant kernel. Bias and preserved tensors live in BF16 (cast at load from FP32) and never touch INT8.

## Architectural decisions (locked in via discussion)

| Decision | Choice |
|---|---|
| Quantization granularity | Per-channel symmetric, `scale_W = amax / 127` |
| Quantized tensors | `W_qkv` (packed), `attn_proj`, `W1`, `W2` per layer (4 per layer) |
| Preserved (FP32) tensors | `wte`, `wpe`, all LN gamma/beta, all biases — stored as FP32 in the file; existing runtime cast to BF16 unchanged |
| QKV layout | Quantize as packed `[d_model, 3*d_model]` — one INT8 buffer + one length-`3*d_model` scale vector per layer |
| Schema location | No binary header — sequential reads in fixed order, sizes derived from compile-time model config. `quant_config.json` is a sidecar manifest for human inspection only |
| Source for quantization | FP32 HF weights (highest precision) |
| Activation quant scheme (future) | Dynamic per-token at runtime — no offline calibration in this phase |
| Validation reference | Report both `error_total` (vs FP32) and `error_quant_only` (vs BF16) |

## Tool layout

```
tools/offline_quant/
├── pyproject.toml          # uv-managed, numpy + matplotlib + questionary + huggingface_hub
├── quantize.py             # CLI entry point
├── src/
│   ├── reader.py           # FP32 weight reader (existing weights/gpt2_<size>_c_weights.bin)
│   ├── quantizer.py        # per-channel symmetric: amax → scale → INT8 + clamp
│   ├── writer.py           # sequential tensor blocks (16-byte aligned) + JSON manifest sidecar
│   ├── validator.py        # noise measurement (vs FP32 and vs BF16)
│   └── plotting.py         # matplotlib for --distrib mode and article assets
├── out/
│   ├── runs/<timestamp>/   # per-run logs
│   ├── scales/             # standalone scale dumps from --distrib (debug only)
│   └── quant_files/        # final .bin + .json before promotion to weights/
└── README.md
```

Directory note: `tools/offline_quant/out/` is local to this tool and won't conflict with the project's top-level `out/` build directory.

## CLI design

```
quantize.py [--model {small,medium,large,all}]   # default: all
            [--list]                              # show quantized vs preserved tensor lists
            [--distrib]                           # interactive: pick layer+tensor; per-tensor plots (dist + amax, dequant error, scatter, mapping table)
            [--compare]                           # batch: cross-tensor and cross-model composite plots
            [--stats]                             # before/after memory breakdown per tensor type (with --chart for bar plot)
            [--valid]                             # noise validation against FP32 and BF16
            [--out-dir <path>]                    # destination for plot outputs (default: tools/offline_quant/out/runs/<timestamp>/)
            [--install]                           # copy .bin + .json to weights/ non-interactively
```

- Default (no flags): quantize all sizes, write `tools/offline_quant/out/quant_files/gpt2_<size>_quant8.{bin,json}`, then prompt to copy to `weights/` interactively.
- Flags compose: `--valid --model large` runs validation only for large.
- `--list/--show` collapsed to `--list` (one name).

## File format

### No binary header — sequential reads, sizes from compile-time defines

The `.bin` file has **no header, no tensor table, no magic bytes**. The loader reads sequentially in a fixed order, exactly like the existing FP32 file. Block sizes are computed from compile-time model config (`d_model`, `d_ff`, `vocab_size`, etc., via `-DGPT2_<size>_MODEL`). The contract is the byte order documented below — the reader and writer must agree on it.

This matches the existing `weights/gpt2_<size>_c_weights.bin` philosophy: minimal, position-based, no metadata in the file.

Wrong-file detection relies on the filename suffix (`_quant8.bin`). The build/runtime path is selected at compile time, so the loader code already knows which file to expect.

### Tensor body order (matches existing C loader at gpt2.c:1361–1404)

1. `wte` (FP32)
2. `wpe` (FP32)
3. For each layer `l = 0..num_layers-1`:
   - `ln1_gamma`, `ln1_beta` (FP32)
   - `W_qkv` (INT8 + scale of length `3*d_model`)
   - `b_qkv` (FP32, length `3*d_model`)
   - `attn_proj` (INT8 + scale of length `d_model`)
   - `attn_proj_bias` (FP32)
   - `ln2_gamma`, `ln2_beta` (FP32)
   - `W1` (INT8 + scale of length `d_ff`)
   - `b1` (FP32)
   - `W2` (INT8 + scale of length `d_model`)
   - `b2` (FP32)
4. `lnf_gamma`, `lnf_beta` (FP32)

All blocks padded to 16-byte alignment (writer pads with zeros; reader skips padding using known block sizes). FP32 preserved tensors are byte-for-byte identical to their counterparts in `weights/gpt2_<size>_c_weights.bin` — the existing C loader path for these (`fread_weights_or_exit` with its FP32 → BF16 cast) is reused unchanged. The new INT8 + scale blocks get a small new `fread_int8_with_scale` helper.

### `quant_config.json` (one per model size, alongside its `.bin`)

Each model size produces its own `.bin` + `.json` pair (`gpt2_small_quant8.{bin,json}`, `gpt2_medium_quant8.{bin,json}`, `gpt2_large_quant8.{bin,json}`). The JSON is a human-readable manifest for audit/inspection only — the loader does not read it (no header, no JSON; layout is derived from compile-time model config).

```json
{
  "tool_version": "0.1",
  "model": "gpt2-large",
  "scheme": "int8_per_channel_symmetric",
  "scale_convention": "amax/127",
  "scale_dtype": "fp32",
  "source_weights_path": "weights/gpt2_large_c_weights.bin",
  "quantized_tensors": ["W_qkv", "attn_proj", "W1", "W2"],
  "preserved_tensors": [
    "wte", "wpe",
    "ln1_gamma", "ln1_beta", "b_qkv", "attn_proj_bias",
    "ln2_gamma", "ln2_beta", "b1", "b2",
    "lnf_gamma", "lnf_beta"
  ],
  "memory_summary": {
    "source_fp32_total_mb": 3094,
    "quantized_int8_mb": 774,
    "preserved_fp32_mb": 256,
    "scale_overhead_kb": 920,
    "artifact_total_mb": 1031
  }
}
```

The JSON is human-readable manifest only — the loader does not read it. Useful for `--list`, `--stats`, and post-hoc inspection.

## Article assets — produced AFTER the tool, using the tool

Tool builds first, then is run to produce all article plots. All plot generation lives in the tool — no separate scripts directory. Outputs follow the existing article convention and land in `docs/articles/2026-06-quant8-gpu/assets/plots/` (per the project's article asset structure: `assets/{diagrams,plots,tables,videos}/`).

The tool grows two flags to cover everything:

- `--distrib` (interactive, single-tensor): distribution histogram + amax, dequant error histogram, FP32-vs-dequant scatter, mapping examples table.
- `--compare` (batch, multi-tensor / multi-model): side-by-side comparisons (e.g., attention W_q vs MLP W1; cross-model memory bar charts).

Both flags accept `--out-dir <path>` so the tool can write directly into `docs/articles/2026-06-quant8-gpu/assets/plots/` when generating article assets, or into `tools/offline_quant/out/runs/<timestamp>/` for ad-hoc inspection.

Plots needed for the article (all written to `docs/articles/2026-06-quant8-gpu/assets/plots/`):

| Plot | Produced by |
|---|---|
| Distribution + amax — attention W_q layer 0 | `--distrib` |
| Distribution + amax — MLP W1 layer 0 | `--distrib` |
| Mapping examples table (5–10 FP32 → INT8 → dequant rows) | `--distrib` (table mode) |
| Dequant error histogram for W_q | `--distrib` |
| W_fp32 vs W_dequant scatter (y=x reference) | `--distrib` |
| Side-by-side attention W_q vs MLP W1 | `--compare` |
| Memory bar chart (matmul vs preserved) | `--stats` (chart mode) |
| Cross-model memory comparison (small/medium/large) | `--compare` |

## Implementation steps

1. Set up `tools/offline_quant/` skeleton: `pyproject.toml` (uv-managed, deps: numpy, matplotlib, questionary, huggingface_hub), package layout, README.
2. Implement `reader.py`: load FP32 from `weights/gpt2_<size>_c_weights.bin` using the same byte layout as the C loader. Validate dimensions against config. Keep tensor identities (which 4 per layer get quantized vs which pass through as FP32).
3. Implement `quantizer.py`: per-channel symmetric for the 4 matmul tensors only (`W_qkv`, `attn_proj`, `W1`, `W2`). Axis = output dimension (per existing transpose convention, this is axis=1 on disk). Emit `(W_int8 ∈ [-127, 127], scale_W ∈ FP32)`. Preserved tensors are not touched.
4. Implement `writer.py`: write blocks sequentially in the documented order, 16-byte aligned (little-endian — same as the existing FP32 file; common across all target platforms). No header. For preserved tensors, copy FP32 bytes through unchanged. For quantized tensors, write INT8 block followed by FP32 scale vector. Write JSON manifest alongside (sidecar, not read by loader).
5. CLI plumbing in `quantize.py`: argparse-based; wire `--list`, `--stats`, `--model`, `--install`.
6. `--distrib` mode: interactive picker via `questionary` (consistent with existing project pattern); matplotlib plots saved to `out/runs/<timestamp>/`. Implements per-tensor distribution histogram + amax annotation, dequant error histogram, FP32-vs-dequant scatter, and a small mapping examples table.
7. `--stats` mode: per-tensor and per-tensor-type memory breakdown. Optional chart output (matplotlib bar) saved alongside the tabular print.
8. `--valid` mode: load resulting `.bin` + `.json`, dequant each quantized tensor, compute per-tensor RMSE and max-abs error against (a) original FP32 and (b) BF16 cast of source. Print summary table.
9. `--compare` mode: batch generator for cross-tensor (e.g., attention W_q vs MLP W1) and cross-model (small/medium/large memory bar) composite plots. Same `plotting.py` backend, different drivers.
10. **Generate article assets** by running the tool with `--out-dir docs/articles/2026-06-quant8-gpu/assets/plots/`: `--distrib` for per-tensor plots, `--stats --chart` for memory breakdown, `--compare` for composite plots. PNGs land directly in the article's `assets/plots/` per the project's existing article convention.
11. After manual HF upload by user: update `setup.sh` to download `gpt2_<size>_quant8.bin` + `.json` when an INT8 build path is selected.

## Critical files to modify or create

**New:**
- `tools/offline_quant/pyproject.toml`
- `tools/offline_quant/quantize.py`
- `tools/offline_quant/src/{reader,quantizer,writer,validator,plotting}.py`
- `tools/offline_quant/README.md`
- `docs/articles/2026-06-quant8-gpu/assets/plots/*.png` (article plot PNGs — written by tool runs into the existing article asset convention)
- `docs/articles/2026-06-quant8-gpu/article.md` (skeleton/draft, separate from this plan)

**Modified later (after manual HF upload):**
- `setup.sh` — add INT8 download branch.

**Not touched in this phase:**
- `gpt2.c`, `cuda/*.cu` — runtime work belongs to the next phase (activation quant kernel + dequant kernel + cuBLAS INT8 path wiring).
- `weights/*.bin` (FP32 source) — stays as the canonical source of truth.

## Verification

- Run end-to-end: `uv run quantize.py --model small --valid`. Confirm:
  - `.bin` and `.json` produced under `tools/offline_quant/out/quant_files/`.
  - All INT8 byte values lie in `[-127, 127]` (spot-check with `numpy.min/max` on each tensor).
  - `error_quant_only` (vs BF16): RMS roughly `scale_W / sqrt(12)` per channel — sanity check that the quant noise distribution matches uniform rounding.
  - `error_total` (vs FP32): ≈ `error_quant_only` plus a small BF16 cast contribution.
- Run `--list` and `--stats`; confirm reported totals match the JSON manifest.
- Verify file format: read the `.bin` back sequentially using the documented block order, dequantize one tensor and check it matches the FP32 source within tolerance.
- Re-run for `medium` and `large`; confirm all three sizes produce valid artifacts.
- After manual HF upload + `setup.sh` update: run a clean `setup.sh` flow on a fresh checkout, confirm the INT8 files land in `weights/`.

## Open questions deferred to runtime phase

These are out of scope for this plan but flagged for the next phase:
- Activation quant kernel (per-token dynamic).
- Dequant + bias-add kernel (fused).
- cuBLAS INT8 path wiring (`cublasGemmEx` with `CUDA_R_8I` × 2 → `CUDA_R_32I`).
- Per-arch INT8 capability detection at startup (Maxwell has no INT8 tensor cores; gracefully fall back or refuse).
- Build system: `out/gpu/int8/` directory and `make gpu int8 <size>` target.
