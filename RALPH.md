# RALPH — GPT-2 Large BF16 GPU decode speedup

## Goal — ROUND 2 (active)

Raise **GPT-2 Large, GPU BF16, decode-phase TPS from 91.19 to >= 150 TPS**
(TPOT 10.88 -> <= 6.67 ms). Prefill performance is not the target, but must not
regress by more than 10%. Small/medium models are not the target either; keep
them working.

**91.19 TPS is the verified round-2 baseline** (commit 3f47f3c; four runs:
91.23, 91.26, 91.26, 91.19). Every delta is measured against it, not against
round 1's numbers. Do not spend an iteration re-measuring it.

150 was chosen over a more ambitious 200 deliberately: it is reachable on the
two highest-confidence candidates alone (see the menu), so the loop can exit
cleanly rather than grinding to max-iterations chasing the last few percent.
If it lands early with budget left, start a fresh round rather than moving the
goalposts mid-loop — a moved target invalidates every delta already recorded.

### Round 1 (complete, for reference)

59.83 -> 91.19 TPS (+52.5%) against a 50% goal. Both wins were pure overhead
removal, no new kernels and no math changed. Details in the results table.

Branch: `perf/bf16-large-decode`.

## Exact commands

```bash
# build (large only — keeps the loop fast)
make gpu bf16 large

# benchmark (decode preset is the default: ~13-token prompt, 768 generated tokens)
./scripts/run.sh --bf16 large

# analyze — read ONLY the "GPU BF16" large decode TPS column
uv run python scripts/performance_analysis.py --bf16 --headless
```

`--headless` is **required**. Without it the script ends in a blocking
`plt.show()` and the iteration hangs forever. Headless prints the summary table
for the latest log and exits in ~0.3s with no plots and no comparison charts.
`--gpu` is omitted on purpose: it pulls in stale FP32 logs and adds a row that
is not the target metric.

Before the final commit of the whole effort, run the full `make gpu bf16` and
`./scripts/run.sh --bf16` (all three sizes) once to confirm nothing else broke.

## Measurement rules

1. **The round-2 baseline is already verified: 91.19 TPS / 10.88 ms TPOT.**
   Do not spend an iteration re-measuring it. Iteration 1 should be a real
   optimization attempt.
2. **Noise floor: 2%.** A delta below 2% is noise, not an improvement. Do not
   commit it, do not record it as a win. If a change looks like +2-4%, re-run
   the benchmark before believing it.
3. Always compare against the **current best committed** TPS, not against the
   original baseline.
4. Keep the GPU in a comparable state between runs — do not benchmark while a
   build or another GPU job is running.

## Correctness gate (a change is not an improvement if it breaks this)

- Output text must stay coherent English, not repeated tokens or garbage.
- With `--temperature 0` (greedy), the same prompt must produce the **same
  tokens** as the baseline build for at least the first 32 tokens. **Captured
  in iteration 1** from commit 81d7293. It also lives at
  `logs/ralph_greedy_ref.txt`, but `logs/` is gitignored, so the authoritative
  copy is here:

  ```
  ./out/gpu/bf16/gpt2_large --prompt "Once upon a time, in a land far, far away, there was a small dragon." \
      --req_out_tokens 32 --token_chunk_size 32 --temperature 0 --no-stream
  ```
  must print exactly:
  ```
   He was a dragon of a very small size, and he was a very small dragon. He was a dragon of a very small size, and he was a
  ```

  This reference is still valid in round 2: every change committed so far
  reproduces it exactly, so it remains the gate for the whole effort.

- **The greedy gate does not exercise sampling.** `--temperature 0` takes an
  argmax path and never calls `top_k_sample`, so a change to the sampling code
  needs its own check. `rand()` is never seeded (no `srand` in `gpt2.c`), so the
  default top-k run is fully deterministic: compare `generated_text` in the new
  log against a saved baseline log and it must be **byte-identical** unless the
  change is deliberately numerical.

  **RETIRED as of round 2, iteration 1.** Batching the attention GEMMs moved
  cuBLAS onto a different kernel with a different accumulation order, which
  changes BF16 rounding in the last bits. Sampling amplifies that: the cumulative
  probability walk picks a different token within ~3 steps, so the sampled text
  now differs from the original baseline by design. **Do not treat sampled-text
  divergence as a failure from here on, and do not try to restore it.**

  **Use this instead — long greedy, compared across builds:**
  ```
  # current build
  ./out/gpu/bf16/gpt2_large --prompt "Once upon a time, in a land far, far away, there was a small dragon." \
      --req_out_tokens 256 --token_chunk_size 256 --temperature 0 --no-stream --json_out_file /tmp/greedy_new.json
  # then: git stash push gpt2.c, rebuild, rerun to /tmp/greedy_old.json, stash pop, rebuild
  ```
  and `generated_text` must match **byte for byte**. 256 consecutive argmax
  decisions over a 50,257-way distribution agreeing is strong evidence the math
  is right; argmax is far more robust to last-bit rounding than sampling is, so
  this separates "different rounding" from "broken kernel". A genuine indexing
  or stride bug diverges almost immediately under this test.
- Numerical changes that alter BF16 rounding may shift greedy output slightly;
  if that happens, judge by manual inspection of coherence and say so explicitly
  in the log row rather than silently accepting.

## Iteration protocol

One optimization per iteration. Do not batch two ideas into one measurement.

1. Read this file, including "Tried and rejected", so you do not repeat work.
2. Pick the highest value untried candidate from the menu below (or a better one
   you found in a profile — profiling counts as a legitimate iteration).
3. Implement it. Keep the change minimal and localized.
4. `make gpu bf16 large` → `./scripts/run.sh --bf16 large` →
   `uv run python scripts/performance_analysis.py --bf16 --headless`.
5. Append one row to the results table with the measured TPS.
6. **Improved (>2%) and correctness gate passes** → `git commit` with a message
   naming the change and the TPS delta, e.g.
   `Fuse QKV into one GEMM: 91.2 -> 101.3 TPS (+11%)`.
   **Otherwise** → revert the working tree (`git checkout -- <files>`) and add an
   entry under "Tried and rejected" explaining *why* it did not help, in enough
   detail that a future iteration does not retry it blindly.
7. When large decode TPS >= 150 (round-2 goal, from the verified 91.19
   baseline), output `<promise>PERF GOAL REACHED</promise>`.

Never rewrite or delete past rows in the results table or past entries in
"Tried and rejected". Append only. This file is the loop's only memory.

**Revert carefully.** Step 6's revert must name only the files the iteration
actually touched — never `git checkout -- .`, which would also discard unrelated
working-tree changes. (The `--headless` flag in `scripts/performance_analysis.py`
was uncommitted while the loop ran; it is committed now and documented in the
README.)

## Where the time goes (measured 2026-09-19, current build)

Fresh nsys profile of the **post-round-1** build (`make gpu bf16 large`,
64 tokens). Read this before picking anything — it overturns two assumptions
that the round-1 version of this file got wrong.

**1. Only ~56% of the token is kernel execution.** Kernel time sums to ~6.1 ms
against a 10.88 ms token. The other **~4.8 ms/token is gap** — launch overhead
and the blocking logit copy. At ~2,100 launches/token and ~2 us of CPU launch
cost each, the arithmetic matches the observed gap almost exactly. **This is
the single largest line item in the whole profile and nothing is computing
during it.**

**2. The big weight GEMVs are already near the hardware limit.** cuBLAS is not
the problem:

| work | ms/token | launches/token | achieved bandwidth |
|------|----------|----------------|--------------------|
| MLP GEMVs (W1, W2) | 1.26 | 72 | **749 GB/s = 78% of peak** |
| Q/K/V + attn_proj GEMVs | 0.99 | 144 | 475 GB/s = 50% |
| lm_head GEMV | 0.16 | 1 | **794 GB/s = 83% of peak** |
| per-head attention GEMVs | 1.78 | **720** | tiny matrices, launch-dominated |
| misc gemv variants | ~1.1 | ~570 | |
| layernorm / add_bias / softmax / add_2d | ~0.7 | ~600 | |

So the waste is in **launch count and small operations**, not arithmetic.
Round 1 won twice on exactly this and the pattern has not been exhausted.

**Bandwidth ceiling.** ~1.54 GB of weights per token (708M transformer params +
64M tied lm_head, at 2 bytes) at the 5080's ~960 GB/s = **~1.61 ms/token, i.e.
~620 TPS**. Current 10.88 ms is 6.8x above it. 620 is unreachable in practice;
70-80% of it (~430-500 TPS) is what an excellent implementation looks like.
The 150 goal is well inside physics — it is an engineering problem, not a
hardware one.

Caveats on the numbers above: the profile includes the 19-token prefill, which
slightly inflates per-token figures, and nsys adds its own overhead. The
ranking is solid; the specific ms values are +/-15%.

## Candidate menu (re-ranked against the profile above)

1. **CUDA graph capture of the decode step.** Targets the measured ~4.8 ms gap
   directly — the largest single item, and the only one backed by a direct
   measurement rather than inference. The decode kernel sequence is fixed at
   M=1, which is the case graphs exist for.

   **Read this before starting, it will save you two iterations:** kernel
   arguments are baked in at capture time, and `n_tokens` increments every
   decode step, so a naive capture breaks on the second token. Workable
   approaches: capture against fixed max-context buffers and pass the live
   length via device memory; capture per bucketed sequence length; or
   `cudaGraphExecUpdate` to patch args between launches. **If a first attempt
   fails on the shape problem, that is not evidence that graphs do not work
   here — do not log it under "Tried and rejected" as though it were.**

2. **Batch the per-head attention GEMVs.** 720 launches/token and 1.78 ms for
   matrices too small to saturate anything. `cublasGemmStridedBatched` collapses
   them to 2 per layer = 72/token. This is exactly the round-1 softmax win
   applied to the GEMVs beside it, and the per-head score rows that made that
   possible already exist (see iteration 2).

3. **Fuse Q/K/V into one GEMM.** 3 separate `cublasGemmEx` per layer at only 50%
   of peak bandwidth. One concatenated [3*d_model, d_model] weight = one launch
   and one pass over the activation, and the larger matrix should lift that 50%.

4. **Fuse the epilogues.** bias+GELU into the first MLP GEMV; residual add +
   layernorm (`cuda/add_2d.cu` + `cuda/layernorm.cu`). ~600 launches/token of
   trivial elementwise work. Individually small, collectively ~0.7 ms.

5. **GPU-side top-k.** Would also remove the blocking 100 KB D->H logit copy and
   its sync at `gpt2.c:2184`. But sampling is now only ~0.2 ms of a 10.88 ms
   token after round 1, so this is worth ~2% at best. Low priority despite
   being conceptually tidy.

6. **Custom BF16 GEMV kernels for M=1. DEMOTED — read before attempting.** The
   round-1 file listed this as a headline candidate. The profile says otherwise:
   cuBLAS already achieves 78-83% of peak on the MLP and lm_head GEMVs. Beating
   it there is hard and worth at most ~0.8 ms even if perfect. Only worth
   touching the Q/K/V-sized matrices (50% of peak), and candidate 3 probably
   fixes those more cheaply.

7. **Flash-attention-style decode / persistent kernels / INT8 weights.** A
   structurally different decode path, 300-400 TPS class, much bigger project.
   INT8 is the one lever that moves the bandwidth wall itself rather than
   improving efficiency against it. Out of scope for a 150 TPS goal.

8. **Re-profile when the ranking goes stale.** `./scripts/run.sh --bf16 large
   --profile` runs under nsys. The table above was measured on the round-1
   build; after two or three accepted changes it will no longer describe
   reality, and a fresh kernel-time breakdown is a perfectly good iteration
   output.

## Results

Append-only, across all rounds. Note the baseline changes between rounds:
round 1 deltas are against 59.83, round 2 deltas are against **91.19**.

### Round 1 — goal >= 89.7 TPS (complete)

| iter | change | large decode TPS | delta vs best | verdict |
|------|--------|------------------|---------------|---------|
| 0 | baseline, user-reported pre-loop | 56.7 | — | superseded by iter 0b |
| 0b | baseline re-measured on this machine (2 runs, both 59.83; TPOT 16.63 ms) | 59.83 | — | **reference** |
| — | `scripts/performance_analysis.py`: added `--headless` (tooling fix, not a perf change) | n/a | n/a | infrastructure |
| 1 | **top_k_sample: replace full-vocabulary qsort with one-pass insertion top-k** (candidate 2, CPU half). Measured cost of the old qsort: 2.76 ms/token. TPOT 16.63 -> 13.71 ms. Two runs: 72.49, 72.54. All 768 sampled tokens byte-identical to baseline; greedy gate matches. | **72.49** | +21.2% | **committed** ✅ |
| 2 | **Batch the decode softmax across heads** (candidate 7, found by profiling = candidate 8). nsys showed `softmax_kernel` at 19.4% of GPU time with 46,144 launches / 64 tokens = 721 per token (20 heads x 36 layers), ~2 us each, i.e. almost pure launch overhead. Split the decode attention into 3 passes (scores -> one batched softmax -> context) so each head owns a row of `scores_h_d`; softmax is one block per row, so rows=nof_heads batches it with no kernel change. 720 -> 36 softmax launches per token. TPOT 13.71 -> 10.88 ms. Four runs: 91.23, 91.26, 91.26, 91.19. Text byte-identical; greedy matches; prefill TTFT 0.1718 vs 0.1720 pre-change (no regression); small 376.47, medium 166.79 both fine. | **91.19-91.26** | +25.9% | **committed** ✅ |

**Round 1 closed.** Target was >= 89.7 TPS (1.5x the verified 59.83 baseline).
Measured 91.19-91.26 across four runs = **+52.5%**, goal met.

| model | round-1 baseline | round-1 final | speedup |
|-------|------------------|---------------|---------|
| small | 175.54 | 376.47 | +114% |
| medium | 98.19 | 166.79 | +70% |
| large | **59.83** | **91.19** | **+52.5%** |

Both wins were overhead, not arithmetic: 2.76 ms/token of CPU qsort, and ~700
redundant kernel launches per token. Neither needed a new CUDA kernel.

### Round 2 — goal >= 150 TPS (active)

| iter | change | large decode TPS | delta vs best | verdict |
|------|--------|------------------|---------------|---------|
| 0 | round-2 baseline = round-1 final, commit 3f47f3c (four runs: 91.23, 91.26, 91.26, 91.19; TPOT 10.88 ms) | **91.19** | — | **reference** |
| 1 | **Batch the per-head attention GEMVs** (candidate 2). Added `dot_2d_gpu_batched` (`cublasGemmStridedBatchedEx`, same operand mapping as `dot_2d_gpu` plus a stride per operand) and collapsed both decode attention loops into one batched call each. 1440 -> 72 attention GEMM launches per token. Profile attribution in the round-1 notes was too conservative: the per-head GEMVs were spread across six cuBLAS kernel entries totalling ~2.9 ms/token of kernel time plus ~2.9 ms of launch gap, not the 1.78 ms attributed to two entries. TPOT 10.88 -> 4.60 ms. Four runs: 213.10, 212.74, 213.99, 213.25. Greedy-256 byte-identical across builds; sampled text differs (BF16 rounding, see gate); prefill TTFT 0.1718 unchanged; small 717.16, medium 371.33. | **212.74-213.99** | +133% | **committed** ✅ |

**Round 2 closed at iteration 1.** Target was >= 150 TPS from the 91.19
baseline. Measured 212.74-213.99 across four runs = **+133%**, goal met with a
single change.

| model | round-2 baseline | round-2 final | speedup | vs original 2026-09-19 baseline |
|-------|------------------|---------------|---------|--------------------------------|
| small | 376.47 | 717.16 | +91% | 175.54 -> 717.16 (**4.1x**) |
| medium | 166.79 | 371.33 | +123% | 98.19 -> 371.33 (**3.8x**) |
| large | **91.19** | **213.25** | **+133%** | 59.83 -> 213.25 (**3.6x**) |

Three wins in a row, all the same shape: **the cost was kernel launches, not
arithmetic.** No new CUDA kernel has been written in either round.

Large decode is now at 4.60 ms/token against the ~1.61 ms/token bandwidth
floor — **2.9x above it, down from 10x**. The remaining candidates (CUDA
graphs, fused QKV, fused epilogues) are still untried and the gap is now small
enough that a fresh profile should precede any round 3: the round-2 profile in
this file describes a build that no longer exists, and its attribution was
already shown to be off once.

## Tried and rejected

_(append entries here: what was tried, measured TPS, and why it did not help.
Say which round. Be specific about **why** — "did not help" is useless to a
future iteration; "batched call had to fall back to a loop because the strides
are not uniform" is what stops the idea being retried blindly.)_

_Nothing rejected in round 1: both attempts landed._
