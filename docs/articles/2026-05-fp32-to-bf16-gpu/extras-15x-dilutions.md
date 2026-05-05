# Where the 15× goes — three serial dilutions

> **Side reading for the FP32→BF16 article.** This was originally a section in
> `article.md` but was extracted as deep-dive material that goes beyond the
> article's main thesis. Kept here for the curious reader who wants to see the
> full Amdahl decomposition behind the headline 1.30× wall-clock win.

Pulling the 5080 and H100 data together raises one last question: BF16 tensor cores are advertised at ~15× the throughput of the FP32 CUDA-core path on the same hardware (true for both the 5080 and the H100). Why does the wall-clock end-to-end win come out to just 1.30× (1024-token Large prefill on the 5080) — and why doesn't the H100, with its 3.5× more bandwidth and ~15× more BF16 tensor-core peak, push that any higher?

The answer is that 15× is the **kernel-peak ratio for an idealized tensor-op kernel**, and three serial dilutions stand between that ratio and what a user sees in TTFT. Each one is real, each one is measurable from the data already in the article (5080 prefill profiles + H100 nsys breakdown), and each one corresponds to a different optimization lever.

```
15× peak BF16 vs FP32 tensor-core throughput
      ↓ (cuBLAS efficiency at our shapes)
~3.3× actual GEMM kernel speedup
      ↓ (non-GEMM kernels don't get the tensor-core win)
~1.84× total kernel-time speedup
      ↓ (host overhead is invariant across dtypes)
~1.30× end-to-end wall-clock speedup
```

## Layer 1: 15× → 3.3× — cuBLAS doesn't deliver peak at our shapes

Nvidia's "989 BF16 TFLOPS" (H100) or "419 BF16 TFLOPS" (RTX 5080) is a **peak number** that assumes ideal kernels at ideal sizes. Real cuBLAS GEMMs achieve a fraction of that, depending heavily on the M, K, N triple. For Large prefill at 1024 tokens, the typical GEMM is `1024 × 1280 × 1280` (attention) or `1024 × 1280 × 5120` (FFN). At those dimensions cuBLAS picks `cutlass_*_tensorop_*` kernels (verified in the 5080 post-fix profile and in the H100 `nvjet_tst_*` attribution) but achieves roughly 30–50% of peak tensor-core utilization — typical numbers for one-off, non-batched shapes where the algorithm-selection heuristic doesn't have batched amortization to help it.

Measured directly in the 5080 post-fix profile:
- FP32 GEMM kernel time: **73.2 ms** (using `cutlass simt GEMM` — the FP32 CUDA-core path)
- BF16 GEMM kernel time: **22.1 ms** (using `cutlass tensor-op GEMM` — tensor cores engaged)
- **GEMM-only speedup: 73.2 / 22.1 ≈ 3.3×**

The 15× → 3.3× drop here is **the largest single dilution in the chain**, and it has nothing to do with host overhead. It's pure peak-vs-achieved at the kernel level. Bigger M (batched inference, speculative decoding) would push this number toward peak; staying at M ≈ 1000 with one-off heuristic dispatch leaves it at ~3.3×.

## Layer 2: 3.3× → 1.84× — non-GEMM kernels stay put

After GEMM is sped up 3.3×, GEMM is no longer the dominant kernel. The other kernels — softmax, layernorm, gelu, add_bias, concat_heads — **don't benefit from BF16 compute** (covered earlier in the article: they're reductions or elementwise, no multiply-accumulate split, no tensor-core path available). Their absolute time stays roughly the same in BF16 — these are still GPU kernels that scale with memory bandwidth, but the dtype change doesn't move them.

From the same 5080 post-fix table:

| Kernel category | FP32 time | BF16 time | Speedup |
|---|---:|---:|---:|
| GEMM only (`cutlass simt` → `cutlass tensorop`) | 73.2 ms | 22.1 ms | **3.32×** |
| Everything else (softmax, layernorm, gelu, add_bias, concat_heads, splitKreduce, …) | 43.8 ms | 42.9 ms | 1.02× |
| **Total kernel time** | **117 ms** | **65 ms** | **1.80×** |

Pure Amdahl's law on the GPU side: GEMM was 62% of FP32 kernel time, so the maximum achievable kernel-total speedup (with infinite GEMM speedup, non-GEMM unchanged) would be `1 / (0.38 + 0) = 2.63×`. With a finite 3.3× GEMM speedup it lands at `1 / (0.38 + 0.62/3.3) = 1.81×`. Almost half the GEMM win evaporates here, before host overhead enters the picture at all.

## Layer 3: 1.84× → 1.30× — host overhead is invariant

The last dilution is the host-overhead one the H100 nsys breakdown in the article quantified directly (~50% of decode wall clock; ~45% on the mixed prefill preset). Host time per prefill — kernel launch dispatch, per-layer `cudaDeviceSynchronize`, the tokenizer socket round-trip, logit sampling on the CPU, JSON-logging glue — is **mostly invariant to the dtype change** because none of those things touch the GPU at all. (It would shift if we reorganised the execution shape — CUDA Graphs, async tokenizer — but in this experiment, swapping FP32 for BF16 doesn't move it.)

Solving from the 5080 1024-token prefill measurements:

```
host_overhead = TTFT_BF16 − kernel_BF16 = 172 ms − 65 ms = 107 ms

Sanity check: TTFT_FP32 = host_overhead + kernel_FP32
                       = 107 ms + 117 ms
                       = 224 ms ✓ (measured: 224.5 ms)

End-to-end speedup = (107 + 117) / (107 + 65)
                   = 224 / 172
                   = 1.30×
```

So 107 ms of constant host overhead reduces a 1.80× kernel speedup to 1.30× wall clock. That's roughly a 28% additional dilution — meaningful, but **the smallest of the three layers**, which often surprises people. The mental model "host overhead is the bottleneck" is correct in *direction* but understates the role of the prior two layers.

## What each layer responds to

Putting the chain in one table to make the optimization choices explicit:

| Dilution layer | What we lost | What lifts it |
|---|---|---|
| Peak vs cuBLAS achieved (15× → 3.3×) | ~78% | bigger M (batched inference, speculative decoding); `cublasLt` with explicit algo pinning; persistent kernels |
| GEMM share of kernel total (3.3× → 1.8×) | ~45% | kernel fusion (fold non-GEMM work into the GEMM epilogue); fuse adjacent custom kernels so they share a single launch |
| Host overhead invariance (1.8× → 1.3×) | ~28% | CUDA Graphs (capture-once, replay); async tokenizer; batched logit sampling; remove per-layer syncs |

Reaching 4–5× wall-clock speedup over FP32 on the same hardware would require chipping at all three layers, not just the host one. None of them are precision changes; all of them are *shape* and *orchestration* changes — which is also why the H100, with its 15× more BF16 tensor-core peak and 3.5× more bandwidth, doesn't break the 1.3× ceiling either. The dilution chain is a property of the *code*, not the silicon.
