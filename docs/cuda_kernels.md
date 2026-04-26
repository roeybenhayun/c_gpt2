# CUDA Kernels — GPT-2.C

## Overview

8 custom CUDA kernels implement the transformer operations that aren't handled by cuBLAS (matrix multiplications). Ordered from simplest to most complex.

## Kernel Descriptions

### 1. add_2d — Element-wise addition
The simplest kernel. Each thread handles one element: `out[i] = a[i] + b[i]`. A 2D grid maps threads to rows and columns. Pure embarrassingly parallel — no thread cooperation, no shared memory, no reductions.

### 2. add_bias — Broadcast addition
Almost identical to add_2d, but instead of adding two same-shaped matrices, it broadcasts a 1D bias vector across rows: `a[row][col] += b[col]`. Same 2D grid, same independence between threads. The only twist is the index math — bias is indexed by column only.

### 3. gelu — Element-wise activation
Same parallel pattern as add_2d (one thread per element, no cooperation), but with a more complex math expression per element:
```
0.5 * x * (1 + tanh(0.797.. * (x + 0.044715 * x^3)))
```
The complexity is in the arithmetic (tanh, cube), not the parallelism.

### 4. causal_masking — Row-wise fill
One thread per row (1D grid). Each thread writes `-INFINITY` to all positions above the diagonal: `row[j] = -INF for j > i`. Simple logic, but unlike the previous kernels, each thread does variable work (row 0 masks almost everything, last row masks nothing).

### 5. concat_heads — Data rearrangement
Reshapes multi-head attention output from `[heads][seq_len][head_dim]` to `[seq_len][heads * head_dim]`. One block per head, one thread per dimension within the head. The complexity is in the index math — translating between two different memory layouts — not in the computation.

### 6. embeddings — Lookup + addition
Each thread computes one element of the embedding: `embeddings[row][col] = wte[token_id][col] + wpe[row][col]`. The interesting part is the indirect memory access — `token_id` is looked up from an array, making the wte access data-dependent (scatter/gather pattern). Also supports partial computation via `start_row`/`n_rows` for the KV cache optimization.

### 7. layernorm — Cooperative reduction (2 passes)
First kernel requiring **thread cooperation via shared memory**. One block per row, threads within a block work together across 3 phases:
1. **Mean** — each thread sums a stride of the row, then tree reduction in shared memory to get total sum
2. **Variance** — same pattern: each thread computes partial sum of squared differences, tree reduction to get variance, thread 0 computes `rsqrtf`
3. **Normalize** — each thread applies `(x - mean) * inv_std * gamma + beta`

Uses `__syncthreads()` barriers between phases. Two full tree reductions make this significantly more complex than the element-wise kernels.

### 8. softmax — Cooperative reduction (3 passes)
The most complex kernel, and the one that was the performance bottleneck (79% of GPU time before rewrite). Same cooperative pattern as layernorm but with **3 reduction phases**:
1. **Find max** — tree reduction to find row maximum (for numerical stability)
2. **Sum exponentials** — compute `exp(x * inv_temp - max)` per element, tree reduction to get total sum
3. **Normalize** — divide each exponential by the sum

Also handles temperature scaling (`inv_temp`), stride (for non-contiguous rows in the attention matrix), and stores intermediate exponentials to avoid recomputation. The rewrite from 1-thread-per-row to 1-block-per-row with shared memory tree reduction was the single biggest optimization — the original version had one thread serially looping over up to 1024 columns, completely underutilizing the GPU.

---

## Thread and Block Configuration

### Two parallelism patterns

| Pattern | Kernels | Thread role |
|---------|---------|-------------|
| **One thread per element** (2D grid) | add_2d, add_bias, gelu, embeddings | Independent — no cooperation |
| **One thread per row** (1D grid) | causal_masking | Independent — serial loop over columns |
| **One block per row** (1D grid, shared memory) | layernorm, softmax, concat_heads | Cooperative — threads within a block share data via reductions |

### Block sizes

| Kernel | Block size | Total threads/block |
|--------|-----------|-------------------|
| add_2d | (16, 16) | 256 |
| add_bias | (16, 16) | 256 |
| gelu | (32, 32) | 1024 |
| causal_masking | (256, 1) | 256 |
| concat_heads | (head_dim, 1) = (64, 1) | 64 |
| embeddings | (16, 16) | 256 |
| layernorm | (1024, 1) | 1024 |
| softmax | (256, 1) | 256 |

---

## Grid Sizes by Model and Phase

GPT-2 model dimensions:

| | Small | Medium | Large |
|---|---|---|---|
| d_model | 768 | 1024 | 1280 |
| n_heads | 12 | 16 | 20 |
| head_dim | 64 | 64 | 64 |
| 4 x d_model (FFN) | 3072 | 4096 | 5120 |

Prefill assumes N=100 tokens as an example, generation is 1 new token.

### 1. add_2d — matrix is `(tokens x d_model)`

| | Small | Medium | Large |
|---|---|---|---|
| Prefill (100) | (48, 7) = 336 blocks | (64, 7) = 448 blocks | (80, 7) = 560 blocks |
| Generation (1) | (48, 1) = 48 blocks | (64, 1) = 64 blocks | (80, 1) = 80 blocks |

### 2. add_bias — matrix is `(tokens x d_model)` or `(tokens x 4*d_model)`

On attention/projection output `(tokens x d_model)`:

| | Small | Medium | Large |
|---|---|---|---|
| Prefill (100) | (48, 7) = 336 | (64, 7) = 448 | (80, 7) = 560 |
| Generation (1) | (48, 1) = 48 | (64, 1) = 64 | (80, 1) = 80 |

On FFN first layer `(tokens x 4*d_model)`:

| | Small | Medium | Large |
|---|---|---|---|
| Prefill (100) | (192, 7) = 1344 | (256, 7) = 1792 | (320, 7) = 2240 |
| Generation (1) | (192, 1) = 192 | (256, 1) = 256 | (320, 1) = 320 |

### 3. gelu — matrix is `(tokens x 4*d_model)`

| | Small | Medium | Large |
|---|---|---|---|
| Prefill (100) | (96, 4) = 384 | (128, 4) = 512 | (160, 4) = 640 |
| Generation (1) | (96, 1) = 96 | (128, 1) = 128 | (160, 1) = 160 |

### 4. causal_masking — 1D over tokens

| | Small | Medium | Large |
|---|---|---|---|
| Prefill (100) | (1) = 1 block | (1) = 1 block | (1) = 1 block |
| Generation (1) | (1) = 1 block | (1) = 1 block | (1) = 1 block |

Only needed during prefill. At 100 tokens, 100 < 256 so it fits in a single block.

### 5. concat_heads — one block per head

| | Small | Medium | Large |
|---|---|---|---|
| Prefill (100) | (12) = 12 blocks | (16) = 16 blocks | (20) = 20 blocks |
| Generation (1) | (12) = 12 blocks | (16) = 16 blocks | (20) = 20 blocks |

Always the same — grid only depends on `n_heads`, not sequence length. One of the smallest kernel launches.

### 6. embeddings — matrix is `(n_rows x d_model)`

| | Small | Medium | Large |
|---|---|---|---|
| Prefill (100) | (48, 7) = 336 | (64, 7) = 448 | (80, 7) = 560 |
| Generation (1) | (48, 1) = 48 | (64, 1) = 64 | (80, 1) = 80 |

### 7. layernorm — one block per token

| | Small | Medium | Large |
|---|---|---|---|
| Prefill (100) | (100) = 100 blocks | (100) = 100 blocks | (100) = 100 blocks |
| Generation (1) | (1) = 1 block | (1) = 1 block | (1) = 1 block |

Same across model sizes — grid only depends on token count. During generation, the entire layernorm runs in a single block of 1024 threads.

### 8. softmax — one block per row of attention scores `(tokens x tokens)`

| | Small | Medium | Large |
|---|---|---|---|
| Prefill (100) | (100) = 100 blocks | (100) = 100 blocks | (100) = 100 blocks |
| Generation (1) | (1) = 1 block | (1) = 1 block | (1) = 1 block |

Same as layernorm — grid depends on number of rows (tokens), not model size. During generation, **the entire softmax runs in 1 block of 256 threads** — this is why the old 1-thread-per-row implementation was catastrophic: it was literally 1 thread doing all the work serially.

---

## Key Takeaway

During token generation, most kernels collapse to tiny launches (1 block for layernorm/softmax, 48-80 blocks for element-wise ops). The GPU is massively underutilized at this phase — most SMs sit idle. This is why GPU inference is **memory-bandwidth bound** during generation, not compute bound. The grid scales with matrix dimensions which are determined at runtime based on model size and current sequence length.
