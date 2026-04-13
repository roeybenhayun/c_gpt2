
# GPT2.C — C Implementation of GPT-2 Inference

A C implementation of GPT-2 inference, using Hugging Face weights converted to a custom binary format. Tokenization is handled via a Python server. Using one external library for logging

---
## Setup Instructions

### 1. Python Environment

#### Create a virtual environment:
```bash
python3.9 -m venv transformers_env
```

Activate it:
```bash
source transformers_env/bin/activate
```

Install dependencies:
```bash
pip install -r requirements.in
```

⚠️ Due to Python package version mismatches, loading and caching Hugging Face models might fail.
You can avoid this by manually downloading the model files (see below).




### 2. Download Prebuilt GPT2.C Weights (Recommended)
Place these files under weights directory (create this dir in the project root):

GPT-2 Small:
https://huggingface.co/roeybh/gpt2-small-from-scratch-c/resolve/main/gpt2_c_weights.bin

GPT-2 Medium:
https://huggingface.co/roeybh/gpt2-small-from-scratch-c/resolve/main/gpt2_medium_c_weights.bin

GPT-2 Large:
https://huggingface.co/roeybh/gpt2-small-from-scratch-c/resolve/main/gpt2_large_c_weights.bin

### 3. Create Weights from Hugging Face Files (Optional)
You can generate the weights yourself using extract_weights.py.

The required directories (transformers/models/gpt2, etc.) are already part of the repo.
Just place the following files in the correct folder.

GPT-2 Small — transformers/models/gpt2:
```bash
cd transformers/models/gpt2

wget https://huggingface.co/gpt2/resolve/main/pytorch_model.bin
wget https://huggingface.co/gpt2/resolve/main/merges.txt
wget https://huggingface.co/gpt2/resolve/main/vocab.json
wget https://huggingface.co/gpt2/resolve/main/tokenizer_config.json
wget https://huggingface.co/gpt2/resolve/main/config.json
```

GPT-2 Medium — transformers/models/gpt2-medium:
```bash
cd transformers/models/gpt2-medium

wget https://huggingface.co/gpt2-medium/resolve/main/pytorch_model.bin
wget https://huggingface.co/gpt2-medium/resolve/main/merges.txt
wget https://huggingface.co/gpt2-medium/resolve/main/vocab.json
wget https://huggingface.co/gpt2-medium/resolve/main/tokenizer_config.json
wget https://huggingface.co/gpt2-medium/resolve/main/config.json
```

GPT-2 Large — transformers/models/gpt2-large:
```bash
cd transformers/models/gpt2-large

wget https://huggingface.co/gpt2-large/resolve/main/pytorch_model.bin
wget https://huggingface.co/gpt2-large/resolve/main/merges.txt
wget https://huggingface.co/gpt2-large/resolve/main/vocab.json
wget https://huggingface.co/gpt2-large/resolve/main/tokenizer_config.json
wget https://huggingface.co/gpt2-large/resolve/main/config.json
```

Then run:
```bash
python extract_weights.py --model-size small     # or medium / large
```


## Compiler & Build Instructions
Tested with:
```text
Apple clang version 16.0.0 (clang-1600.0.26.6)
Target: arm64-apple-darwin23.6.0
Thread model: posix

Linux 6.14.0-36-generic #36~24.04.1-Ubuntu SMP PREEMPT_DYNAMIC x86_64

```

Install Homebrew (if needed):
```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

Install Jansson (JSON library dependency):
OSX
```bash
brew install jansson
```
Ubuntu run the install_dependencies.sh
```bash
sudo ./install_dependencies.sh
```


## Build Targets
ARM64 OSX
X86_64

### Standard Build (Without KV Cache - only for comperisons)

To build the standard, non-cached versions of the models, which are useful for baseline comparisons:

Build GPT-2 Small:
```bash
make small
```

Build GPT-2 Medium:
```bash
make medium
```

Build GPT-2 Large:
```bash
make large
```

Build all targets
```bash
make 
```

### High-Performance Build (With KV Cache)
To enable the high-performance KV Cache, pass the ENABLE_KV_CACHE flag via CPPFLAGS. This changes the token generation complexity from O(N²) to O(N), resulting in a significant speedup

Build GPT-2 Small:
```bash
make small CPPFLAGS="-DENABLE_KV_CACHE"
```

Build GPT-2 Medium:
```bash
make medium CPPFLAGS="-DENABLE_KV_CACHE"
```

Build GPT-2 Large:
```bash
make large CPPFLAGS="-DENABLE_KV_CACHE"
```

Build all targets
```bash
make CPPFLAGS="-DENABLE_KV_CACHE"
```

### GPU Build (CUDA + cuBLAS)
Requires NVIDIA GPU with CUDA toolkit installed. Builds with KV cache enabled by default.

Build GPU GPT-2 Small:
```bash
make gpu small
```

Build GPU GPT-2 Medium:
```bash
make gpu medium
```

Build GPU GPT-2 Large:
```bash
make gpu large
```

Build all GPU targets:
```bash
make gpu
```

### Cleaning
To remove all compiled binaries from the project directory:

```bash
make clean
```


## Running Inference
1. Activate Python environment:
```bash
source transformers_env/bin/activate
```

2. Start the tokenizer server:
```bash
python tokenizer.py
```

3. Run a model:
One-shot prompt:
```bash
./out/gpt2_small --prompt "Once upon a time..."
```

Interactive mode:
```bash
./out/gpt2_small
```


## Running Performance Tests

Run all models (CPU + GPU):
```bash
./scripts/run.sh
```

Run a specific model:
```bash
./scripts/run.sh small
```

Run GPU only:
```bash
./scripts/run.sh --gpu
```

Run CPU only:
```bash
./scripts/run.sh --cpu
```

Run GPU with NVIDIA Nsight Systems profiling:
```bash
./scripts/run.sh --gpu --profile small
```

Flags can be combined with model sizes:
```bash
./scripts/run.sh --gpu small medium
```

### Analysing Results

Analyse all results (CPU + GPU):
```bash
python scripts/performance_analysis.py
```

Analyse GPU-only results:
```bash
python scripts/performance_analysis.py --gpu
```

Analyse CPU-only results:
```bash
python scripts/performance_analysis.py --cpu
```


📁 Directory Structure (Relevant Parts)
```bash
.
├── out/                            # Compiled binaries
│   ├── small/                      # CUDA object files (small model)
│   ├── medium/                     # CUDA object files (medium model)
│   └── large/                      # CUDA object files (large model)
├── cuda/                           # CUDA kernel source files
├── include/                        # Header files (cuda_kernels.h, model_config.h)
├── logs/                           # JSON logs and nsys profile reports
├── scripts/                        # Automation and performance analysis
├── tokenizer.py                    # Tokenizer server
├── extract_weights.py              # Script to extract weights from Hugging Face models
├── transformers/
│   └── models/
│       ├── gpt2/
│       ├── gpt2-medium/
│       └── gpt2-large/
├── gpt2.c                          # Main C code for inference
├── Makefile
├── requirements.txt                # Python modules
├── train_gpt2.py                   # Python GPT2 inference impl.
├── weights/                        # Model weights
├── install_dependencies.sh         # (Ubuntu 24.04 only)
└── README.md
```
