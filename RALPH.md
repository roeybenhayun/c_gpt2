# RALPH — GPT-2 Large BF16 GPU decode speedup

## Goal

Raise **GPT-2 Large, GPU BF16, decode-phase TPS by 50%**: from the verified
**59.83 TPS** baseline to **>= 89.7 TPS**. Prefill performance is not the
target, but must not regress by more than 10%. Small/medium models are not the
target either; keep them working.

The original 56.7 figure was a pre-loop measurement. The baseline was re-measured
on this machine on 2026-09-19 (2 runs, both 59.83 TPS, mean TPOT 16.63-16.64 ms)
and 59.83 is the number every delta is measured against.

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

1. **The baseline is already verified: 59.83 TPS.** Do not spend an iteration
   re-measuring it. Iteration 1 should be a real optimization attempt.
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

- **The greedy gate does not exercise sampling.** `--temperature 0` takes an
  argmax path and never calls `top_k_sample`, so a change to the sampling code
  needs its own check. `rand()` is never seeded (no `srand` in `gpt2.c`), so the
  default top-k run is fully deterministic: compare `generated_text` in the new
  log against a saved baseline log and it must be **byte-identical** unless the
  change is deliberately numerical.
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
   `Fuse QKV into one GEMM: 59.8 -> 66.4 TPS (+11%)`.
   **Otherwise** → revert the working tree (`git checkout -- <files>`) and add an
   entry under "Tried and rejected" explaining *why* it did not help, in enough
   detail that a future iteration does not retry it blindly.
7. When large decode TPS >= 89.7 (1.5x the verified 59.83 baseline), output
   `<promise>PERF GOAL REACHED</promise>`.

Never rewrite or delete past rows in the results table or past entries in
"Tried and rejected". Append only. This file is the loop's only memory.

**Revert carefully.** `scripts/performance_analysis.py` carries an intentional
uncommitted change (the `--headless` flag). Step 6's revert must name only the
files the iteration actually touched — never `git checkout -- .` — or the
tooling fix is lost and every later iteration hangs on `plt.show()`.

## Where the time goes (starting knowledge)

Decode is M=1 per step: every GEMM is really a GEMV, so the phase is
**memory-bandwidth and launch-overhead bound**, not FLOP bound. Earlier FP32
profiling put cuBLAS at ~77% of GPU time.

**Bandwidth ceiling (computed iter 1).** Per decode token the weights read are
~708M transformer params + ~64M for the tied lm_head projection = ~772M at
2 bytes = **~1.54 GB/token**. At the 5080's ~960 GB/s that is a floor of
**~1.61 ms/token, i.e. ~620 TPS**. The baseline sat at 16.63 ms — roughly
**10x above the bandwidth floor** — so decode is *not* bandwidth bound here and
there is a lot of headroom. Do not stop optimizing because "it must be memory
bound"; measure first.

**Corollary, confirmed in iter 1:** a meaningful share of the token budget was
not on the GPU at all. Host-side per-token work is a first-class suspect, and
`--temperature 0` (greedy, which skips `top_k_sample`) versus the default top-k
run is a zero-cost way to measure the sampling half of it.

## Candidate menu (not exhaustive, not ordered by certainty)

1. **Fuse Q/K/V into one GEMM.** `dot_2d_gpu` is called three times per layer
   with `W_q_d`/`W_k_d`/`W_v_d` (`gpt2.c:342`). One concatenated `[3*d_model, d_model]`
   weight = one launch instead of three, and one pass over the activation.
   36 layers x 2 saved launches per token.
2. **Kill the per-token logits round trip.** `gpt2.c:2184` copies the whole
   `vocab_size` logit row device→host every decode step, then `top_k_sample`
   scans 50257 elements on the CPU (`gpt2.c:2228`). Do top-k (k=40) on the GPU
   and copy back only the 40 candidates, or sample on device and copy back one
   int. Removes a blocking sync, ~100KB of PCIe traffic, and a CPU pass per token.
3. **CUDA graph capture of the decode step.** The per-token kernel sequence is
   fixed once shapes stabilize at M=1. Capturing it collapses hundreds of launch
   overheads per token into one graph launch. Highest payoff if profiling shows
   large gaps between kernels.
4. **Custom BF16 GEMV kernels for M=1.** cuBLAS `cublasGemmEx` is tuned for
   batched GEMM; at M=1 a hand-written kernel that streams weights with wide
   vectorized loads and does a warp-level reduction often beats it. Try on the
   biggest weight (the MLP `W1`/`W2`, 4*d_model) first.
5. **Fuse the MLP.** `W1` GEMV → bias → GELU → `W2` GEMV writes and re-reads the
   `4*d_model` intermediate through HBM. Fusing bias+GELU into the first GEMV's
   epilogue removes a full round trip per layer.
6. **Fuse residual add + layernorm.** `add_2d` then `layernorm` are separate
   kernels (`cuda/add_2d.cu`, `cuda/layernorm.cu`), each reading and writing the
   activation. Two fusions per layer, 36 layers.
7. **KV cache layout and the attention path.** Check whether the attention
   scores/softmax/concat_heads sequence at M=1 is doing strided or uncoalesced
   reads over the growing KV cache, and whether `casual_masking` is even needed
   when only one query row exists.
8. **Profile first if unsure.** `./scripts/run.sh --bf16 large --profile` runs
   under nsys. A kernel-time breakdown of the BF16 large decode phase is a
   perfectly good iteration output — it retargets everything after it.

## Results

| iter | change | large decode TPS | delta vs best | verdict |
|------|--------|------------------|---------------|---------|
| 0 | baseline, user-reported pre-loop | 56.7 | — | superseded by iter 0b |
| 0b | baseline re-measured on this machine (2 runs, both 59.83; TPOT 16.63 ms) | 59.83 | — | **reference** |
| — | `scripts/performance_analysis.py`: added `--headless` (tooling fix, not a perf change) | n/a | n/a | infrastructure |
| 1 | **top_k_sample: replace full-vocabulary qsort with one-pass insertion top-k** (candidate 2, CPU half). Measured cost of the old qsort: 2.76 ms/token. TPOT 16.63 -> 13.71 ms. Two runs: 72.49, 72.54. All 768 sampled tokens byte-identical to baseline; greedy gate matches. | **72.49** | +21.2% | **committed** ✅ |
| 2 | **Batch the decode softmax across heads** (candidate 7, found by profiling = candidate 8). nsys showed `softmax_kernel` at 19.4% of GPU time with 46,144 launches / 64 tokens = 721 per token (20 heads x 36 layers), ~2 us each, i.e. almost pure launch overhead. Split the decode attention into 3 passes (scores -> one batched softmax -> context) so each head owns a row of `scores_h_d`; softmax is one block per row, so rows=nof_heads batches it with no kernel change. 720 -> 36 softmax launches per token. TPOT 13.71 -> 10.88 ms. Four runs: 91.23, 91.26, 91.26, 91.19. Text byte-identical; greedy matches; prefill TTFT 0.1718 vs 0.1720 pre-change (no regression); small 376.47, medium 166.79 both fine. | **91.19-91.26** | +25.9% | **committed** ✅ |

### Goal reached

Target was >= 89.7 TPS (1.5x the verified 59.83 baseline). Measured 91.19-91.26
across four runs, i.e. **+52.5% over baseline** — the 50% goal is met.

Full picture, GPU BF16 decode preset, RTX 5080:

| model | baseline TPS | final TPS | speedup |
|-------|--------------|-----------|---------|
| small | 175.54 | 376.47 | +114% |
| medium | 98.19 | 166.79 | +70% |
| large | **59.83** | **91.19** | **+52.5%** |

Both wins were overhead, not arithmetic: 2.76 ms/token of CPU qsort, and ~700
redundant kernel launches per token. Large decode is still ~6.8x above the
~1.61 ms/token bandwidth floor, so candidates 1 and 3-6 remain untried if more
is wanted later.

## Tried and rejected

_(append entries here: what was tried, measured TPS, and why it did not help)_
