# RALPH — GPT-2 Large BF16 GPU decode speedup

## Goal — ROUND 3 (active)

Raise **GPT-2 Large, GPU BF16, decode-phase TPS from 213.25 to >= 250 TPS**
(TPOT 4.60 -> <= 4.00 ms). Prefill performance is not the target, but must not
regress by more than 10%. Small/medium models are not the target either; keep
them working.

**213.25 TPS is the verified round-3 baseline** (commit 2e5c184; four runs:
213.10, 212.74, 213.99, 213.25; TPOT 4.60 ms). Every delta is measured against
it. Do not spend an iteration re-measuring it.

**Read this before picking a target or judging progress.** Kernel time is now
~3.36 ms of the 4.60 ms token; the other ~1.24 ms is launch gap. Launch-overhead
work — which is what all three wins so far were — can therefore never do better
than **3.36 ms = 298 TPS**, and only by driving the gap to exactly zero, which
is impossible. 250 asks for ~0.6 ms of the ~1.24 ms available, leaving real
margin. Anything above ~280 requires reducing *kernel* time, not launches, and
is a different kind of change.

This round is harder than the last two. Rounds 1 and 2 each had 2-6 ms sitting
on the table and were won with small, surgical edits. The remaining overhead is
1.24 ms total and the leading candidate needs a structural change. Expect failed
attempts, and log them properly.

### Rounds 1-2 (complete, for reference)

59.83 -> 91.19 -> 213.25 TPS, a cumulative **3.6x**. Three accepted changes, all
the same shape: the cost was kernel launches, not arithmetic. No new CUDA kernel
was written in either round. Details in the results table.

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

1. **The round-3 baseline is already verified: 213.25 TPS / 4.60 ms TPOT.**
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
   `Fuse QKV into one GEMM: 213.2 -> 236.7 TPS (+11%)`.
   **Otherwise** → revert the working tree (`git checkout -- <files>`) and add an
   entry under "Tried and rejected" explaining *why* it did not help, in enough
   detail that a future iteration does not retry it blindly.
7. When large decode TPS >= 250 (round-3 goal, from the verified 213.25
   baseline), output `<promise>PERF GOAL REACHED</promise>`.

Never rewrite or delete past rows in the results table or past entries in
"Tried and rejected". Append only. This file is the loop's only memory.

**Revert carefully.** Step 6's revert must name only the files the iteration
actually touched — never `git checkout -- .`, which would also discard unrelated
working-tree changes. (The `--headless` flag in `scripts/performance_analysis.py`
was uncommitted while the loop ran; it is committed now and documented in the
README.)

## Where the time goes (measured 2026-09-19, post-round-2 build)

Fresh nsys profile of the **current** build (commit 2e5c184, 64 tokens). The
round-2 version of this section described a build that no longer exists and its
attribution was wrong by ~60% — re-profile before trusting any table here.

| | value |
|---|---|
| TPOT | **4.60 ms** |
| kernel time | **~3.36 ms/token** (73%) |
| launch gap | **~1.24 ms/token** (27%) |
| launches | **~780/token** (was ~2,100 pre-round-2) |
| implied overhead | ~1.6 us per launch |

**The hard cap on launch-overhead work is 3.36 ms = 298 TPS**, reachable only
with zero gap. Everything above that requires cutting kernel time.

Largest kernel line items per token:

| work | ms/token | launches/token | note |
|------|----------|----------------|------|
| mixed gemvx (MLP + batched attention) | ~1.28 | ~102 | 749 GB/s on the MLP pair = 78% of peak |
| Q/K/V + attn_proj GEMVs | 0.98 | 144 | **475 GB/s = 50% of peak** — the weakest GEMM |
| lm_head GEMV | 0.18 | 1 | 794 GB/s = 83% of peak |
| layernorm | 0.26 | 73 | |
| add_bias | 0.20 | 216 | trivial work, many launches |
| softmax | 0.18 | 48 | already batched in round 1 |
| add_2d / gelu / concat_heads | 0.15 | ~154 | trivial work, many launches |

**Bandwidth ceiling unchanged:** ~1.54 GB of weights per token at ~960 GB/s =
**~1.61 ms/token = ~620 TPS**. Current 4.60 ms is 2.9x above it, down from 10x
at the start of round 1. The easy overhead is gone; what remains is either
structural (graphs) or arithmetic efficiency (fused QKV).

Caveats: the profile includes the 19-token prefill and nsys adds its own
overhead. Ranking is solid, absolute ms are +/-15%.

## Candidate menu (re-ranked against the profile above)

1. **CUDA graph capture of the decode step.** The only candidate that addresses
   the full 1.24 ms gap. Realistically recovers 70-80% of it (graph replay still
   costs ~0.3-0.5 us per node), so expect ~3.75 ms = **~265 TPS**. This alone
   should clear the 250 goal.

   **PREREQUISITE FOUND IN ROUND 3 ITERATION 1 — read this first.** Every
   custom kernel in `cuda/*.cu` launches on the **default stream**
   (`kernel<<<grid, block>>>` with no stream argument), and cuBLAS has never had
   `cublasSetStream` called on it. **The legacy default stream cannot be
   stream-captured**, so `cudaStreamBeginCapture` cannot work until every launch
   and the cuBLAS handle are moved onto an explicit non-default stream. That is
   a mechanical but repo-wide refactor (9 wrapper functions plus the handle) and
   it must land *before* any capture attempt. Budget it as its own iteration and
   commit it as infrastructure — measured neutral is the expected and acceptable
   result for it, so do not revert it for failing to improve TPS.

   **The shape problem — read before starting, it will save you two iterations.**
   Kernel arguments are baked in at capture time, and `n_tokens` increments every
   decode step, so a naive capture breaks on the second token. Workable
   approaches: capture against fixed max-context buffers and pass the live length
   via device memory; capture per bucketed sequence length and re-capture on
   bucket crossings; or `cudaGraphExecUpdate` to patch args between launches.
   Note the sampling tail (D->H logit copy + CPU top-k) cannot be captured —
   capture the transformer forward and leave the tail outside.
   **If a first attempt fails on the shape problem, that is not evidence that
   graphs do not work here. Do not log it under "Tried and rejected" as such.**

2. **Fuse Q/K/V into one GEMM.** 144 launches/token at only **50% of peak
   bandwidth** — the least efficient GEMM left, and the only remaining item that
   cuts *kernel* time rather than launches. One concatenated [3*d_model, d_model]
   weight = one launch and one pass over the activation, and the larger matrix
   should lift that 50%. Worth ~0.3-0.4 ms. Combines with candidate 1 rather
   than competing with it.

3. **Fuse the elementwise epilogues.** add_bias (216 launches/token), add_2d,
   gelu, concat_heads, layernorm — ~600 tiny launches/token for ~0.6 ms of
   trivial work. Fusing bias+GELU into the MLP GEMV epilogue and residual+LN
   into one kernel cuts both launch count and HBM round trips. Note this
   overlaps candidate 1: if graphs land first, the launch half of this win is
   already banked and only the memory traffic remains.

4. **GPU-side top-k.** Removes the blocking 100 KB D->H logit copy and its sync
   at `gpt2.c:2184`. Sampling is ~0.2 ms of a 4.60 ms token, so ~4% — but it
   also removes the one thing that forces a sync per token, which may matter
   more once graphs are in. Reassess after candidate 1.

5. **Custom BF16 GEMV kernels for M=1. STILL DEMOTED.** cuBLAS achieves 78-83%
   of peak on the MLP and lm_head GEMVs; beating it there is hard and worth
   little. The 50%-efficiency Q/K/V GEMVs are the only tempting target, and
   candidate 2 likely fixes those more cheaply.

6. **INT8 weights / flash-attention decode / persistent kernels.** The only
   levers that move the 1.61 ms bandwidth floor itself rather than closing the
   gap to it. INT8 would halve weight traffic. Large project, out of scope for
   a 250 TPS goal, and the INT8 path already exists separately in this repo.

7. **Re-profile when the ranking goes stale.** `./scripts/run.sh --bf16 large
   --profile`. The table above describes commit 2e5c184; after one or two
   accepted changes it will no longer describe reality. A fresh kernel-time
   breakdown is a legitimate iteration output — and it has already corrected
   itself once this effort.

## Results

Append-only, across all rounds. The baseline changes between rounds:
round 1 deltas are against 59.83, round 2 against 91.19, round 3 against
**213.25**.

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

### Round 2 — goal >= 150 TPS (complete)

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

### Round 3 — goal >= 250 TPS (active)

| iter | change | large decode TPS | delta vs best | verdict |
|------|--------|------------------|---------------|---------|
| 0 | round-3 baseline = round-2 final, commit 2e5c184 (four runs: 213.10, 212.74, 213.99, 213.25; TPOT 4.60 ms = ~3.36 ms kernel + ~1.24 ms gap, ~780 launches/token) | **213.25** | — | **reference** |
| 1 | **Fuse Q/K/V into one GEMM** (candidate 2, taken ahead of candidate 1 because graphs turned out to be blocked on a prerequisite — see menu). One `[3*d_model, d_model]` weight built device-side at load, one GEMM into a packed scratch, then `qkv_bias_scatter_cuda` adds the fused bias and scatters into Q / K-cache / V-cache. 6 launches per layer (3 GEMM + 3 add_bias) -> 2, i.e. 216 -> 72 per token. TPOT 4.60 -> 4.27 ms. Four runs: 227.04, 230.24, 229.98, 229.43. Greedy-256 byte-identical across builds; prefill TTFT 0.1760 vs 0.1718 (+2.4%, inside the 10% allowance); small 805.08, medium 401.49. Costs ~354 MB extra VRAM for the fused copy (originals kept for the INT8/CPU paths). | **227.0-230.2** | +7.5% | **committed** ✅ |
| 2 | **Remove a dead per-layer `cudaMemcpy` from the decode path** (not on the menu; found while reading the residual join for candidate 3). A blocking D2D copy preserved rows 0..i-1 of the hidden state every layer. It was dead twice over: from layer 1 on `current_hidden_state_d` IS `residual2_out_d` so src==dst, and nothing reads those rows (LN1 and the residual touch only row i; the final LN reads only last_token_position; history lives in the KV caches). Cost up to 2 MB/layer, ~72 MB/token, and 36 synchronous copies/token. TPOT 4.27 -> 4.12 ms. Four runs: 235.54, 237.15, 236.76, 238.47. Greedy-256 identical; prefill TTFT 0.1755 vs 0.1718 (+2.2%); small 820.40, medium 415.02. Smaller than hoped — the syncs were not as costly as the 36-per-token count suggested. | **235.5-238.5** | +3.1% | **committed** ✅ |
| 3 | **Fuse add_bias + add_2d at both residual joins** (candidate 3, first half). `add_bias_residual_cuda` computes `residual + (x + bias)` in one pass, replacing two launches and a pointless HBM round trip at each join. 144 -> 72 launches/token. TPOT 4.12 -> 3.97 ms. **Six runs: 243.43, 248.77, 246.76, 248.64, 245.40, 246.35 — median 246.8, range 243-249.** Greedy-256 identical; prefill TTFT 0.1740 vs 0.1718; small 865.06, medium 439.41. **Note the spread: ~2.3% run-to-run at this TPOT, up from <0.5% earlier in the effort. Near-goal claims now need the *minimum* of several runs above the bar, not the mean.** | **246.8** (median) | +4.0% | **committed** ✅ |
| 4 | **Fuse add_bias + gelu on the MLP first projection** (candidate 3, second half). `bias_gelu_cuda` computes `gelu(x + bias)` in one in-place pass; the tanh approximation matches `gelu_kernel` exactly. Separately these wrote and re-read the full `[1 x d_ff]` activation — the widest in the layer — and cost an extra launch per layer. TPOT 3.97 -> 3.76 ms. **Five runs: 258.72, 259.62, 259.72, 260.07, 259.67 — minimum 258.72, spread back to ~0.5%.** Greedy-256 identical; prefill TTFT 0.1718 = baseline exactly; small 897.06, medium 457.48. | **258.7-260.1** | +5.3% | **committed** ✅ |

**Round 3 closed at iteration 4.** Target was >= 250 TPS from the 213.25
baseline. Measured **258.72-260.07 across five runs (minimum 258.72)** =
**+21.6%**, goal met. Judged on the minimum rather than the mean, per the
variance note in iteration 3.

| model | round-3 baseline | round-3 final | speedup | vs original baseline |
|-------|------------------|---------------|---------|----------------------|
| small | 376.47 | 897.06 | +138% | 175.54 -> 897.06 (**5.1x**) |
| medium | 166.79 | 457.48 | +174% | 98.19 -> 457.48 (**4.7x**) |
| large | **213.25** | **259.67** | **+21.8%** | 59.83 -> 259.67 (**4.3x**) |

Four accepted changes, none of them the menu's #1 candidate: CUDA graphs was
never attempted because it is blocked on a stream refactor (see candidate 1).
Three of the four were still launch/round-trip removal; the fourth was deleting
work that did not need doing at all.

Large decode is now ~3.76 ms against the ~1.61 ms bandwidth floor — **2.3x
above it**, from 10x at the start of round 1. Remaining untried: CUDA graphs
(needs the stream refactor first), the `concat_heads` and `layernorm` launches,
and the structural items in candidate 6. **Re-profile before round 4** — the
table above describes commit 012e2e9 and is already stale.

## Tried and rejected

_(append entries here: what was tried, measured TPS, and why it did not help.
Say which round. Be specific about **why** — "did not help" is useless to a
future iteration; "batched call had to fall back to a loop because the strides
are not uniform" is what stops the idea being retried blindly.)_

_Nothing rejected in round 1: both attempts landed._
