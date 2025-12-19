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

        ifneq (,$(filter cuda,$(MAKECMDGOALS)))
            # Inject CUDA Flags
            CFLAGS += -DUSE_CUDA -I/usr/local/cuda-12.8/include
            LDFLAGS += -L/usr/local/cuda-12.8/lib64 -lcudart -lcublas
        endif

    .PHONY: cuda
    cuda: $(if $(filter small medium large,$(MAKECMDGOALS)),,all)
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
OUTDIR = ./out


# Ensure the output directory exists
$(shell mkdir -p $(OUTDIR))

# Targets
.PHONY: all small medium large clean

all: small medium large          # build everything with “make”

small:
	$(CC) $(CPPFLAGS) $(CFLAGS) -DGPT2_SMALL_MODEL $(SRC) -o $(OUTDIR)/gpt2_small $(LDFLAGS)

medium:
	$(CC) $(CPPFLAGS) $(CFLAGS) -DGPT2_MEDIUM_MODEL $(SRC) -o $(OUTDIR)/gpt2_medium $(LDFLAGS)

large:
	$(CC) $(CPPFLAGS) $(CFLAGS) -DGPT2_LARGE_MODEL $(SRC) -o $(OUTDIR)/gpt2_large $(LDFLAGS)

clean:
	rm -f $(OUTDIR)/gpt2_*
