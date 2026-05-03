
# GPT2.C — C Implementation of GPT-2 Inference

A C implementation of GPT-2 inference, using Hugging Face weights converted to a custom binary format. Tokenization is handled via a Python server.

Supported platforms: ARM64 macOS, x86_64 Linux. Optional CUDA GPU acceleration on Linux.

---

## Quick Start

Run the setup script to install all dependencies, download model weights, and set up the Python environment:
```bash
./setup.sh
```

The script will:
1. Install [uv](https://docs.astral.sh/uv/) (Python package manager)
2. Install system dependencies (jansson, openblas, build tools) via brew or apt
3. Download tokenizer data
4. Download GPT-2 model weights (you choose which sizes)
5. Set up the Python environment (`uv sync`)

Once setup is complete, follow the printed instructions to build and run.

---

## Manual Setup

If you prefer to set things up manually:

### 1. System Dependencies

macOS:
```bash
brew install jansson
```

Linux (Ubuntu/Debian):
```bash
sudo apt-get install -y build-essential libjansson-dev libopenblas-dev
```

### 2. Python Environment

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) and sync dependencies:
```bash
uv sync
```

### 3. Download Model Weights

Place weight files under the `weights/` directory:

| Model | File | URL |
|-------|------|-----|
| Small (124M) | `gpt2_c_weights.bin` | https://huggingface.co/roeybh/gpt2-small-from-scratch-c/resolve/main/gpt2_c_weights.bin |
| Medium (355M) | `gpt2_medium_c_weights.bin` | https://huggingface.co/roeybh/gpt2-small-from-scratch-c/resolve/main/gpt2_medium_c_weights.bin |
| Large (774M) | `gpt2_large_c_weights.bin` | https://huggingface.co/roeybh/gpt2-small-from-scratch-c/resolve/main/gpt2_large_c_weights.bin |

### 4. Download Tokenizer

```bash
curl -L -o tokenizer.json https://huggingface.co/openai-community/gpt2/resolve/main/tokenizer.json
```

### 5. Create Weights from Hugging Face (Optional)

You can generate the weights yourself instead of downloading prebuilt files:
```bash
uv run python extract_weights.py --model-size small     # or medium / large
```

This requires the original HuggingFace model files under `transformers/models/gpt2/` (see `extract_weights.py` for details).

---

## Building

KV cache is enabled by default for all builds, providing O(N) generation complexity.

### CPU Build

```bash
make small          # GPT-2 Small (124M)
make medium         # GPT-2 Medium (355M)
make large          # GPT-2 Large (774M)
make                # All targets
```

### GPU Build (CUDA + cuBLAS)

Requires NVIDIA GPU with CUDA toolkit installed (Linux x86_64 only). The Makefile will check for GPU availability and report a clear error if not found.

```bash
make gpu small      # GPT-2 Small (124M)
make gpu medium     # GPT-2 Medium (355M)
make gpu large      # GPT-2 Large (774M)
make gpu            # All targets
```

### GPU Half-Precision Builds (BF16 / FP16)

The same source compiles to FP32 (default), BF16, or FP16 storage via Makefile flags. Compute stays in FP32 (FP32 accumulator) regardless of input dtype — only the in-memory weight/activation representation changes. Each dtype lands in its own output subdirectory so stale FP32 `.o` files can never link into a BF16 binary.

```bash
make gpu bf16 small      # BF16 build  → out/gpu/bf16/gpt2_small
make gpu fp16 medium     # FP16 build  → out/gpu/fp16/gpt2_medium
```

The `bf16` and `fp16` flags only apply to GPU builds; using them without `gpu` will fail with a clear error (the CPU path goes through `cblas_sgemm`, which has no half-precision form).

The on-disk weight `.bin` is FP32 in every build — the loader converts to the build's storage dtype on the fly, so the same weight file works for all three.

### Cleaning

```bash
make clean
```

---

## Running Inference

1. Start the tokenizer server:
```bash
uv run python tokenizer.py
```

2. Run a model (in another terminal):

One-shot prompt:
```bash
./out/cpu/gpt2_small --prompt "Once upon a time..."
```

Interactive mode:
```bash
./out/cpu/gpt2_small
```

GPU inference (FP32):
```bash
./out/gpu/gpt2_small --prompt "Once upon a time..."
```

GPU inference (BF16 / FP16):
```bash
./out/gpu/bf16/gpt2_small --prompt "Once upon a time..."
./out/gpu/fp16/gpt2_small --prompt "Once upon a time..."
```

---

## Running Performance Tests

The runner builds the requested binaries, then runs each model with a fixed workload and writes a per-run JSON log under `logs/`.

### Runner flags

```bash
./scripts/run.sh                        # All models, CPU + GPU FP32 + GPU BF16, default workload
./scripts/run.sh small                  # Specific model size
./scripts/run.sh --cpu                  # CPU FP32 only
./scripts/run.sh --gpu                  # GPU FP32 only
./scripts/run.sh --bf16                 # GPU BF16 only
./scripts/run.sh --gpu --bf16 large     # GPU FP32 and BF16, Large only
./scripts/run.sh --gpu --profile small  # GPU FP32 with nsys profiling
```

### Workload presets

The runner has three baked-in workload shapes that exercise the model's two execution regimes (prefill and decode) in different proportions. Presets are mutually exclusive; the default is `--decode`.

| Preset       | Prompt              | Output tokens | Phase mix          | Effective `M`              |
|--------------|---------------------|---------------|--------------------|----------------------------|
| `--decode`   | ~13 tokens (built-in) | 768          | ~99% decode        | 1                          |
| `--prefill`  | ~1000 tokens (`scripts/prompts/long_prompt.txt`) | 32 | prefill-dominated | ~1000 prefill / 1 decode   |
| `--balanced` | ~200 tokens (same file) | 200       | roughly 50/50      | ~200 prefill / 1 decode    |

```bash
./scripts/run.sh --gpu --bf16 --prefill   # Long-prompt run, GPU FP32 + BF16
./scripts/run.sh --gpu --bf16 --balanced  # Mixed-shape run
```

The preset is embedded in every output JSON filename (`gpt2_<size>_<tag>_<preset>_…json`) so the analyzer can render one regime at a time.

### Workload overrides

For ad-hoc testing without a preset:

```bash
./scripts/run.sh --gpu --bf16 --prompt-file my_prompt.txt --out-tokens 64 large
```

`--prompt-file` and `--out-tokens` override whatever the preset would have set.

### Analysing Results

The analyzer renders four plots (overlay, speedup, TPOT, TPS bar) and prints a summary table. Series flags select which runners to include; preset flags filter logs by workload shape and tag the chart titles.

```bash
uv run python scripts/performance_analysis.py                       # All series, all presets (latest of each)
uv run python scripts/performance_analysis.py --gpu --bf16          # GPU FP32 vs BF16
uv run python scripts/performance_analysis.py --gpu --bf16 --prefill # …filtered to the prefill preset, title gets "PREFILL preset" suffix
uv run python scripts/performance_analysis.py --cpu --gpu --bf16 --balanced
```

Series flags: `--cpu`, `--gpu`, `--bf16`. Preset flags: `--decode`, `--prefill`, `--balanced` (mutually exclusive). Override the directory globbed for JSONs with `--log-dir <path>` (default: `logs`) — useful for analysing archived runs (e.g. cloud results under `logs/lambda/h100/<run>/`) without disturbing your local `logs/`.

### Long-prompt prefill benchmark

For benchmarking prompt length and measuring TTFT (pure prefill time) directly:

```bash
./scripts/prefill_benchmark.sh                       # Large, default sizes
./scripts/prefill_benchmark.sh --profile             # Same, plus nsys at the largest size
./scripts/prefill_benchmark.sh medium                # Different model
```

Produces a side-by-side FP32 vs BF16 TTFT table at increasing prompt lengths.

### Comparing nsys profiles

After two profiled runs (FP32 and BF16, or before-and-after a code change), compare per-kernel time:

```bash
uv run python scripts/compare_profiles.py <baseline.nsys-rep> <target.nsys-rep>
```

Prints a kernel-family-bucketed table with `baseline | target | ratio | Δ time`, plus a total. Kernels with dtype-templated names (e.g. `softmax_kernel(float *, ...)` vs `softmax_kernel(__nv_bfloat16 *, ...)`) collapse into the same row.

---

## Directory Structure

```
.
├── setup.sh                        # One-step project setup
├── gpt2.c                          # Main C inference code
├── Makefile
├── pyproject.toml                  # Python dependencies (uv)
├── tokenizer.py                    # Tokenizer server
├── extract_weights.py              # Weight extraction from HuggingFace
├── TODO.md                         # Performance optimization tracker
├── cuda/                           # CUDA kernel source files
├── include/                        # Headers (cuda_kernels.h, model_config.h)
├── scripts/
│   ├── run.sh                      # End-to-end build + run + log
│   ├── prefill_benchmark.sh        # TTFT benchmark at varying prompt lengths
│   ├── performance_analysis.py     # Plots + summary tables from JSON logs
│   ├── compare_profiles.py         # nsys-rep diff (per-kernel time)
│   └── prompts/                    # Reusable prompt fixtures (long_prompt.txt)
├── weights/                        # Model weights
├── logs/                           # JSON logs and nsys profile reports
├── docs/
│   └── articles/                   # Long-form articles, one folder per article with its own assets/
├── out/
│   ├── cpu/                        # CPU binaries
│   └── gpu/                        # GPU FP32 binaries + CUDA object files
│       ├── bf16/                   # GPU BF16 binaries (built with `make gpu bf16 …`)
│       └── fp16/                   # GPU FP16 binaries (built with `make gpu fp16 …`)
└── transformers/
    └── models/                     # HuggingFace model files (optional)
```
