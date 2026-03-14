# GPT-2 dot_2d FLOP Analysis

## dot_2d operations per transformer layer

All three GPT-2 variants share the same 8 dot_2d operations per layer:

| # | Operation | Matrix A | Matrix B | Description |
|---|-----------|----------|----------|-------------|
| 1 | Q projection | [T, d_model] | [d_model, d_model]^T | Input x W_q^T |
| 2 | K projection | [T, d_model] | [d_model, d_model]^T | Input x W_k^T |
| 3 | V projection | [T, d_model] | [d_model, d_model]^T | Input x W_v^T |
| 4 | Q x K^T (attention scores) | [T, head_dim] | [T, head_dim]^T | Per head, scaled by 1/sqrt(head_dim) |
| 5 | Weights x V (attention context) | [T, T] | [T, head_dim] | Per head |
| 6 | Output projection | [T, d_model] | [d_model, d_model]^T | attn_output x W_proj^T |
| 7 | FFN up-projection (W1) | [T, d_model] | [d_ff, d_model]^T | d_ff = 4 x d_model |
| 8 | FFN down-projection (W2) | [T, d_ff] | [d_model, d_ff]^T | Back to d_model |

Plus one final call after all layers:

| # | Operation | Matrix A | Matrix B | Description |
|---|-----------|----------|----------|-------------|
| 9 | Logits projection | [T, d_model] | [d_model, vocab_size] | Maps to vocabulary (50257) |

## Model configurations

| | Small | Medium | Large |
|---|---|---|---|
| d_model | 768 | 1024 | 1280 |
| num_layers | 12 | 24 | 36 |
| nof_heads | 12 | 16 | 20 |
| head_dim | 64 | 64 | 64 |
| d_ff (4 x d_model) | 3072 | 4096 | 5120 |
| vocab_size | 50257 | 50257 | 50257 |
| ctx_len | 1024 | 1024 | 1024 |

## Total dot_2d calls per inference pass

| | Small | Medium | Large |
|---|---|---|---|
| Per layer | 8 | 8 | 8 |
| Total | 8 x 12 + 1 = 97 | 8 x 24 + 1 = 193 | 8 x 36 + 1 = 289 |

## FLOP breakdown per layer (T=1024)

FLOPs per matmul = 2 x M x N x K

### Projections (Q, K, V, Output) -- 4 calls of [T, d] x [d, d]

| | Calculation | Result |
|---|---|---|
| Small | 4 x 2 x 1024 x 768 x 768 / 10^12 | 0.0048 TFLOPS |
| Medium | 4 x 2 x 1024 x 1024 x 1024 / 10^12 | 0.0086 TFLOPS |
| Large | 4 x 2 x 1024 x 1280 x 1280 / 10^12 | 0.0134 TFLOPS |

### Attention matmuls (Q x K^T + Weights x V) -- summed across all heads

Each head: [T, 64] x [T, 64]. n_heads x head_dim = d_model, so total = 2 x 2 x T x T x d_model.

| | Calculation | Result |
|---|---|---|
| Small | 2 x 2 x 1024 x 1024 x 768 / 10^12 | 0.0032 TFLOPS |
| Medium | 2 x 2 x 1024 x 1024 x 1024 / 10^12 | 0.0043 TFLOPS |
| Large | 2 x 2 x 1024 x 1024 x 1280 / 10^12 | 0.0054 TFLOPS |

### FFN (W1 up + W2 down) -- 2 calls, d_ff = 4 x d_model

| | Calculation | Result |
|---|---|---|
| Small | 2 x 2 x 1024 x 768 x 3072 / 10^12 | 0.0097 TFLOPS |
| Medium | 2 x 2 x 1024 x 1024 x 4096 / 10^12 | 0.0172 TFLOPS |
| Large | 2 x 2 x 1024 x 1280 x 5120 / 10^12 | 0.0268 TFLOPS |

### Per layer total

| | Projections + Attention + FFN | Result |
|---|---|---|
| Small | 0.0048 + 0.0032 + 0.0097 | 0.0177 TFLOPS |
| Medium | 0.0086 + 0.0043 + 0.0172 | 0.0301 TFLOPS |
| Large | 0.0134 + 0.0054 + 0.0268 | 0.0456 TFLOPS |

FFN dominates each layer (~55% of FLOPS) because d_ff = 4 x d_model.

## Total inference FLOPS

### All layers

| | Calculation | Result |
|---|---|---|
| Small | 0.0177 x 12 | 0.212 TFLOPS |
| Medium | 0.0301 x 24 | 0.722 TFLOPS |
| Large | 0.0456 x 36 | 1.642 TFLOPS |

### Logits projection -- [T, d_model] x [d_model, vocab_size]

| | Calculation | Result |
|---|---|---|
| Small | 2 x 1024 x 768 x 50257 / 10^12 | 0.079 TFLOPS |
| Medium | 2 x 1024 x 1024 x 50257 / 10^12 | 0.105 TFLOPS |
| Large | 2 x 1024 x 1280 x 50257 / 10^12 | 0.132 TFLOPS |

### Grand total

| | Calculation | Result |
|---|---|---|
| Small | 0.212 + 0.079 | ~0.29 TFLOPS |
| Medium | 0.722 + 0.105 | ~0.83 TFLOPS |
| Large | 1.642 + 0.132 | ~1.77 TFLOPS |

## Potential GPU speedup

Benchmark results: ~3.5 TFLOPS/s CPU (OpenBLAS, AVX-512), ~40 TFLOPS/s GPU (cuBLAS).

| | CPU time | GPU time | Speedup |
|---|---|---|---|
| Small | 0.29 / 3.5 = ~83ms | 0.29 / 40 = ~7ms | ~12x |
| Medium | 0.83 / 3.5 = ~236ms | 0.83 / 40 = ~21ms | ~11x |
| Large | 1.77 / 3.5 = ~507ms | 1.77 / 40 = ~44ms | ~11x |

Hardware: RTX 5080 (56.3 TFLOPS theoretical) vs Ryzen 9 9950X3D (4.6 TFLOPS theoretical, AVX-512).

Note: estimates assume data stays on GPU between calls. Per-call host-device memcpy would reduce the speedup.
