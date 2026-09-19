# GPT-2 in C — a 4.3× decode speedup, found by an agent loop

_The sixth article in the series: from CPU baseline, to KV-cache, to GPU, to BF16, to INT8 — and now a performance round where the optimizations were found by Claude Code running in a Ralph loop. Measured on a local RTX 5080._

## Intro

The previous five articles were hand-written optimization work: KV-cache, cuBLAS, BF16, INT8. This one is different. I pointed [Claude Code](https://claude.com/claude-code) at the repo with a single instruction — *make GPT-2 Large BF16 decode 50% faster* — and let it run in a **Ralph loop**: the same prompt fed back after every turn, so the agent iterates against its own committed work until the goal is met.

Three rounds later, decode throughput on GPT-2 Large had gone from **59.8 to 259.7 tokens/sec — 4.3×** — across seven accepted changes.

The headline result is not the interesting part. The interesting part is *what kind* of optimization it found: **not one of the seven was arithmetic.** No hand-written GEMM, no new matmul kernel, no algorithmic change to attention. Every win was removing kernel launches, redundant trips through memory, or work that did not need doing at all.

> **Asset checklist**
> - [x] decode_tps_vs_ceiling.png

### How this article is organized

- **The loop** — what Ralph is, and why the brief file matters more than the prompt.
- **The results** — seven changes, round by round.
- **The pattern** — why it was all overhead, and how the bandwidth floor proved it.
- **What the loop got wrong** — the parts worth being honest about.
- **An accuracy surprise** — INT8 got *more* accurate.

---

## The loop

The mechanism is crude and that is the point. A stop hook intercepts the agent when it tries to finish and re-feeds the identical prompt. There is no memory between iterations except **what is on disk** — the files and the git history.

That constraint drives the whole design. Since the prompt never changes, all the state has to live in a file. I kept a `RALPH.md` at the repo root holding:

- the **goal** and a verified baseline number,
- the **exact** build / benchmark / analyze commands,
- a **noise floor** (2%) and the rule that anything in the 2–4% band gets re-run,
- a **correctness gate**,
- an append-only **results table** and a **"Tried and rejected"** section.

The last two matter most. Without them the loop re-tries the same idea forever, because from its point of view every iteration is the first one.

The prompt itself ended up being three sentences that delegate everything to the file. Every number I duplicated into the prompt was a number that later drifted out of sync with the file — the first run declared victory on a stale target that the file had already moved.

## The results

Baseline: GPT-2 Large, BF16, GPU decode, RTX 5080 — **59.83 TPS**, 16.63 ms per token.

| # | Change | TPS | Δ |
|---|--------|-----|---|
| 1 | `top_k_sample`: full-vocab qsort → one-pass selection | 72.49 | +21% |
| 2 | Batch the decode softmax across heads | 91.19 | +26% |
| 3 | Batch the per-head attention GEMVs | 213.25 | +133% |
| 4 | Fuse Q/K/V into one GEMM | 229.43 | +7.5% |
| 5 | Delete a dead per-layer `cudaMemcpy` | 237.15 | +3.1% |
| 6 | Fuse bias + residual at both joins | 246.80 | +4.0% |
| 7 | Fuse bias + GELU on the MLP projection | 259.67 | +5.3% |

Across all three sizes, and the other two dtypes:

| build | before | after | |
|---|---|---|---|
| BF16 small | 175.54 | **897.06** | 5.1× |
| BF16 medium | 98.19 | **457.48** | 4.7× |
| **BF16 large** | **59.83** | **259.67** | **4.3×** |
| FP32 large | 155.01 | 174.78 | +12.8% |
| INT8 large | 182.91 | 194.83 | +6.5% |

Three of the seven are worth describing.

**The qsort.** Top-k sampling sorted all 50,257 `(probability, index)` pairs on the CPU every single token — through a function-pointer comparator, into a 400 KB stack array — to read off the top 40 and discard the other 50,217. Replacing it with a single pass that keeps a running top-40 cost **2.76 ms per token**, 17% of the entire token budget, spent on the CPU while the GPU idled.

**The softmax.** Decode launched the softmax kernel once per attention head, per layer: 20 × 36 = **720 launches per token**, each doing about 2 µs of work on a single row. The heads were serialized only because they all shared one scratch row. Giving each head its own row — in a buffer that already had 1,024 of them — let all 20 go out in one launch. The kernel itself needed no modification: it was already written as one block per row.

**The same trick again, bigger.** The two GEMVs beside that softmax had the identical shape: 1,440 launches per token on matrices far too small to occupy the GPU. `cublasGemmStridedBatchedEx` collapsed them to 72. That single change was +133%.

## The pattern

The loop's first real act was arithmetic on a napkin, and it set up everything after.

GPT-2 Large reads about **1.54 GB of weights per token** at BF16. The RTX 5080 has roughly **960 GB/s**. So the hard floor is **~1.61 ms/token ≈ 620 TPS**.

The baseline was 16.63 ms — **10× above the floor.** That single number said decode was *not* memory bound, which is the default assumption for LLM inference and would have sent the work straight at the GEMMs. It was wrong here, and the profile confirmed why: cuBLAS was already achieving **78–83% of peak bandwidth** on the MLP and lm_head matrices. There was nothing to win there.

What there was instead: ~2,100 kernel launches per token, and at one point **44% of the token was not kernel execution at all** — just gaps between launches.

Seven changes later the token is 3.76 ms, **2.3× above the floor** instead of 10×.

![Decode throughput across the seven changes, against the memory-bandwidth ceiling](assets/plots/decode_tps_vs_ceiling.png)

_Shaded bands are the three loop runs; each was a separate Ralph session with
its own goal, its own verified baseline and its own exit. Round 2 met its target
in a single iteration — and that one change, batching the per-head attention
GEMVs, is two thirds of the entire gain. The curve flattens hard after it. The
dashed line is the physics: 1.54 GB of weights per token at ~960 GB/s._

> If you take one thing from this: compute the bandwidth floor *before* optimizing. It tells you whether you are fighting physics or fighting overhead, and those need completely different work.

## What the loop got wrong

A clean 4.3× makes the process sound smoother than it was.

**Its own ranking was wrong.** Going into the final round, CUDA graph capture was ranked the top candidate — it targeted a measured 1.24 ms launch gap. It was never implemented. Reading the code turned up a blocker nothing in the analysis had anticipated: every kernel in the project launches on the **default stream**, and the legacy default stream cannot be stream-captured. Graphs are gated behind a repo-wide refactor. Four other changes delivered the round instead.

**A profile attribution was off by 60%.** One round credited the per-head attention GEMVs with 1.78 ms across two cuBLAS kernel entries. They were actually spread across six entries totalling ~2.9 ms — which is why that change overshot its prediction so far. The fix was to write the error into the brief, so the next round trusted a fresh profile over the prose.

**Measurement variance nearly caused a false claim.** At 4 ms per token, run-to-run spread reached ~2.3% — up from under 0.5% at the start, because fixed per-run costs are a bigger fraction of a smaller number. Near a threshold, the mean is not good enough. The final goal was called on the **minimum** of five runs.

**And one deletion looked like a regression when it was the opposite** — see below.

## An accuracy surprise

Three of the final round's changes were not dtype-guarded, so FP32 and INT8 went through the new kernels too. Re-measuring both: no regression, both faster.

But INT8's greedy output **changed**, which is a red flag — greedy decoding is argmax, and argmax is normally robust to last-bit rounding.

It turned out to be an improvement. Measured against the high-precision BF16 reference output:

| INT8 build | agrees with BF16 for |
|---|---|
| before | **52 characters** |
| after | **996 characters** (the whole sample) |

The fused kernels compute `x + bias + residual` in float and round **once**. The unfused pair rounded to the storage dtype, wrote to memory, read it back, and rounded **again**. One rounding instead of two matters most where the activation error is already largest — which is exactly INT8.

The lesson generalizes: a fused kernel is not numerically neutral, and "the output changed" should be judged against the most accurate reference you have, not against the previous build of the same dtype.

## What's next

The biggest item still on the table is CUDA graph capture, blocked behind moving every kernel launch and the cuBLAS handle onto an explicit stream. Below that: the remaining `concat_heads` and `layernorm` launches, and the structural options — flash-attention-style decode, persistent kernels, or leaning harder on INT8 to move the bandwidth floor itself rather than closing the gap to it.

Decode sits at 2.3× above that floor. The easy overhead is gone.

## See also

- [GPT-2 in C — INT8 on GPU](../2026-06-quant8-gpu/article.md)
- [GPT-2 in C — FP32 to BF16 on GPU](../2026-05-fp32-to-bf16-gpu/article.md)
- [GPT-2 in C — now on GPU with 9× faster inference](../2026-04-gpu-inference/article.md)
- [Ralph Wiggum as a software engineer](https://ghuntley.com/ralph/) — the original technique
