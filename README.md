
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

GPU inference:
```bash
./out/gpu/gpt2_small --prompt "Once upon a time..."
```

---

## Running Performance Tests

```bash
./scripts/run.sh                        # All models, CPU + GPU
./scripts/run.sh small                  # Specific model
./scripts/run.sh --gpu                  # GPU only
./scripts/run.sh --cpu                  # CPU only
./scripts/run.sh --gpu --profile small  # GPU with nsys profiling
```

### Analysing Results

```bash
python scripts/performance_analysis.py          # All results
python scripts/performance_analysis.py --gpu    # GPU-only
python scripts/performance_analysis.py --cpu    # CPU-only
```

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
├── cuda/                           # CUDA kernel source files
├── include/                        # Headers (cuda_kernels.h, model_config.h)
├── scripts/                        # Automation and performance analysis
├── weights/                        # Model weights
├── logs/                           # JSON logs and nsys profile reports
├── out/
│   ├── cpu/                        # CPU binaries
│   └── gpu/                        # GPU binaries + CUDA object files
└── transformers/
    └── models/                     # HuggingFace model files (optional)
```
