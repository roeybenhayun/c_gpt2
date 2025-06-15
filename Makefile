# Compiler and flags
CC = gcc
CFLAGS = -Wall -O3 -I/opt/homebrew/include -L/opt/homebrew/lib -ljansson \
         -DUSE_ACCELERATE -DACCELERATE_NEW_LAPACK -framework Accelerate

# Source and output
SRC = gpt2.c
OUTDIR = ./out

# Ensure the output directory exists
$(shell mkdir -p $(OUTDIR))

# Targets
all: medium

small:
	$(CC) $(CFLAGS) -DGPT2_SMALL_MODEL $(SRC) -o $(OUTDIR)/gpt2_small

medium:
	$(CC) $(CFLAGS) -DGPT2_MEDIUM_MODEL $(SRC) -o $(OUTDIR)/gpt2_medium

large:
	$(CC) $(CFLAGS) -DGPT2_LARGE_MODEL $(SRC) -o $(OUTDIR)/gpt2_large

clean:
	rm -f $(OUTDIR)/gpt2_*
