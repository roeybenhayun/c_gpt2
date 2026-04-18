# Compiler and flags
CC =
ARCH = arm64
CFLAGS = -Wall -O3
LDFLAGS = -ljansson
CPPFLAGS =
# Platform defs
PLATFORM_DEFS = -DENABLE_KV_CACHE

# --- Enforce OSx (Darwin) and arm64 architecture ---
# Gets the OS name (e.g., "Darwin" for macOS)
DETECTED_OS := $(strip $(shell uname -s))
$(info DETECTED_OS='$(value DETECTED_OS)')

# Gets the machine architecture (e.g., "arm64", "x86_64")
DETECTED_ARCH := $(strip $(shell uname -m))
$(info DETECTED_ARCH='$(value DETECTED_ARCH)')

# Output directories: out/cpu/ for CPU builds, out/gpu/ for GPU builds
OUTDIR = ./out
ifneq (,$(filter gpu,$(MAKECMDGOALS)))
    BINDIR = $(OUTDIR)/gpu
else
    BINDIR = $(OUTDIR)/cpu
endif

ifeq ($(DETECTED_OS),Darwin) # Check if the operating system is macOS
    ifeq ($(DETECTED_ARCH),arm64) # If it's macOS, now check if the architecture is arm64
        # --- Configuration for macOS arm64 ---
        CC = clang # Use clang as the compiler for macOS
        CFLAGS += -arch arm64 \
			-I/opt/homebrew/include \
			-L/opt/homebrew/lib
        LDFLAGS += -framework Accelerate
        PLATFORM_DEFS += -DUSE_ACCELERATE -DACCELERATE_NEW_LAPACK
    else
        $(error This Makefile is configured to support ONLY arm64 architecture on macOS. Detected OS: "$(DETECTED_OS)", Architecture: "$(DETECTED_ARCH)")

    endif
else ifeq ($(DETECTED_OS),Linux)
    ifeq ($(DETECTED_ARCH),x86_64)
        CC = gcc
        CFLAGS += -I/usr/include
        LDFLAGS += -L/usr/lib -lopenblas -lm
        PLATFORM_DEFS += -DUSE_ACCELERATE_X86

        ifneq (,$(filter gpu,$(MAKECMDGOALS)))
            # Inject GPU (CUDA) Flags
            CFLAGS += -DUSE_GPU -I/usr/local/cuda-12.8/include
            LDFLAGS += -L/usr/local/cuda-12.8/lib64 -lcudart -lcublas
            CUDA_SRC = $(wildcard cuda/*.cu)
            # Per-model object directories so each binary links against .o files
            # compiled with the matching -DGPT2_*_MODEL. Sharing one out/*.o across
            # small/medium/large silently links stale kernels into the wrong binary.
            CUDA_OBJ_SMALL  = $(patsubst cuda/%.cu, $(BINDIR)/small/%.o,  $(CUDA_SRC))
            CUDA_OBJ_MEDIUM = $(patsubst cuda/%.cu, $(BINDIR)/medium/%.o, $(CUDA_SRC))
            CUDA_OBJ_LARGE  = $(patsubst cuda/%.cu, $(BINDIR)/large/%.o,  $(CUDA_SRC))
            NVCC = /usr/local/cuda-12.8/bin/nvcc
            NVCC_FLAGS = -O3

            # Verify NVIDIA GPU and CUDA toolkit are available
            ifeq ($(wildcard $(NVCC)),)
                $(error GPU build requested but nvcc not found at $(NVCC). Install CUDA toolkit or adjust the NVCC path.)
            endif
            NVIDIA_GPU := $(shell nvidia-smi > /dev/null 2>&1 && echo yes || echo no)
            ifeq ($(NVIDIA_GPU),no)
                $(error GPU build requested but no NVIDIA GPU detected. Run 'nvidia-smi' to check your GPU setup.)
            endif
        endif

    .PHONY: gpu
    gpu: $(if $(filter small medium large,$(MAKECMDGOALS)),,all)
        @:
    else
        $(error This Makefile is configured to support ONLY x86_64 architecture on Linux. Detected OS: "$(DETECTED_OS)", Architecture: "$(DETECTED_ARCH)")

    endif

else
    # Error if the operating system is not macOS (e.g., Linux, Windows)
    $(error This Makefile is configured to support ONLY macOS (Darwin) operating system. Detected OS: "$(DETECTED_OS)")
endif

# Add platform-specific definitions to the main CFLAGS
CFLAGS += $(PLATFORM_DEFS)

# Source and output
SRC = gpt2.c

# Ensure the output directory exists
$(shell mkdir -p $(BINDIR))

# Targets
.PHONY: all small medium large clean

all: small medium large          # build everything with "make"

small: MODEL_DEF = -DGPT2_SMALL_MODEL
small: $(CUDA_OBJ_SMALL)
	$(CC) $(CPPFLAGS) $(CFLAGS) $(MODEL_DEF) $(SRC) $(CUDA_OBJ_SMALL) -o $(BINDIR)/gpt2_small $(LDFLAGS)

medium: MODEL_DEF = -DGPT2_MEDIUM_MODEL
medium: $(CUDA_OBJ_MEDIUM)
	$(CC) $(CPPFLAGS) $(CFLAGS) $(MODEL_DEF) $(SRC) $(CUDA_OBJ_MEDIUM) -o $(BINDIR)/gpt2_medium $(LDFLAGS)

large: MODEL_DEF = -DGPT2_LARGE_MODEL
large: $(CUDA_OBJ_LARGE)
	$(CC) $(CPPFLAGS) $(CFLAGS) $(MODEL_DEF) $(SRC) $(CUDA_OBJ_LARGE) -o $(BINDIR)/gpt2_large $(LDFLAGS)

$(BINDIR)/small/%.o: cuda/%.cu | $(BINDIR)/small
	$(NVCC) $(NVCC_FLAGS) -DGPT2_SMALL_MODEL -c $< -o $@

$(BINDIR)/medium/%.o: cuda/%.cu | $(BINDIR)/medium
	$(NVCC) $(NVCC_FLAGS) -DGPT2_MEDIUM_MODEL -c $< -o $@

$(BINDIR)/large/%.o: cuda/%.cu | $(BINDIR)/large
	$(NVCC) $(NVCC_FLAGS) -DGPT2_LARGE_MODEL -c $< -o $@

$(BINDIR)/small $(BINDIR)/medium $(BINDIR)/large:
	mkdir -p $@

clean:
	rm -rf $(OUTDIR)/cpu $(OUTDIR)/gpu
