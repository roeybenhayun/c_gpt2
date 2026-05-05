# Codex Review: FP32 to BF16 GPU Article

Publication review for `article.md` before Substack publishing.

## P0: Fix Before Publishing

1. **Visible TODOs remain in the article.**

   `article.md` still contains plot replacement TODO comments around the decode and prefill chart sections. Even if the linked image paths are now correct, these comments can leak into markdown previews, source views, or copy/paste flows.

   Affected areas:
   - `article.md:210`
   - `article.md:219`
   - `article.md:363`
   - `article.md:376`

   Recommendation: remove the TODO comments entirely once the plot paths are final.

2. **Prompt length is inconsistent: 13 tokens vs 19 tokens.**

   The raw decode logs show `initial_prompt_len: 19`, and the H100 section correctly says "~19-token prompt". But the scoreboard/explanation sections still say "13-token prompt".

   Affected areas:
   - `article.md:349`
   - `article.md:417`
   - `article.md:452`

   Recommendation: standardize the decode workload wording to `~19-token prompt, 768 output tokens`, unless you intentionally want to describe an older run.

3. **Some table assets disagree with the markdown numbers.**

   `tables.py` contains older/default values for `default-benchmark-tps.png`:

   | Model | `tables.py` FP32/BF16 | `article.md` FP32/BF16 |
   |---|---:|---:|
   | Small | 155.0 / 184.8 | 153.9 / 179.6 |
   | Medium | 95.6 / 103.2 | 93.0 / 101.6 |
   | Large | 60.1 / 60.3 | 57.9 / 59.3 |

   Recommendation: update `tables.py` or remove/export-regenerate the stale PNG tables before embedding them in Substack.

4. **"Bit-identical to the previous article" is not supported by the current code.**

   The article says the default FP32 binary is bit-identical to the previous article, but the current code routes FP32 through `cublasGemmEx`, not the old `cublasSgemm` path.

   Affected areas:
   - `article.md:79`
   - `gpt2.c:311`

   Recommendation: soften the wording to something like: "the default remains FP32 and is behaviorally comparable to the previous article."

## P1: Technical Framing To Tighten

1. **"Compute stays in FP32" is too broad.**

   The article says storage choice only affects bytes streamed from VRAM. That is mostly right for custom kernels and reductions, but cuBLAS BF16 tensor-op GEMMs use BF16 inputs with FP32 accumulation.

   Affected areas:
   - `article.md:44`
   - `article.md:121`
   - `gpt2.c:318`

   Recommendation: use more precise wording: "reductions and accumulators stay FP32; GEMM uses BF16 inputs with FP32 accumulation."

2. **Be careful calling host/non-GPU overhead "fixed" and "independent of GPU."**

   Several places treat non-GEMM or host overhead as fixed across dtype and GPU. The direction is right for this experiment, but some of the cited time is custom GPU kernels, not host work.

   Affected areas:
   - `article.md:381`
   - `article.md:389`
   - `article.md:483`
   - `article.md:595`

   Recommendation: distinguish:
   - host overhead: tokenizer socket, sampling, launch dispatch, synchronization, JSON/logging
   - non-GEMM GPU kernels: softmax, layernorm, GELU, add-bias, concat-heads

   Use "mostly invariant in this experiment" rather than "fixed" where the data does not prove strict invariance.

3. **Line-number references inside prose will rot.**

   The article refers to "line 247", "line 321", and "line 367". These are useful while editing locally, but not useful on Substack and will become wrong as soon as the article changes.

   Affected areas:
   - `article.md:389`
   - `article.md:452`
   - `article.md:483`
   - `article.md:493`

   Recommendation: replace line-number references with section names, for example "the post-fix 1024-token profile table above".

4. **"No GPU upgrade exceeds that bound" is too absolute.**

   The Amdahl argument is strong, but absolute wording invites nitpicks. The bound applies to this workload and code structure, not every possible GPU upgrade scenario.

   Affected areas:
   - `article.md:67`
   - `article.md:511`
   - `article.md:544`

   Recommendation: qualify with "on this workload as currently structured" or "without changing the execution shape."

## P2: Editorial / Reader Experience

1. **The article is strong but long.**

   The narrative is good: failed BF16 expectation, profiling surprise, mask-kernel fix, `M`-dimension explanation, H100 cross-check. But the 14-item "Sections in this article" list slows the intro.

   Recommendation: shorten the roadmap or remove it for Substack. Let the hook land earlier.

2. **Tighten the H100 setup section.**

   The H100 data is valuable, but the setup/runbook details take space away from the result.

   Affected area:
   - `article.md:427`

   Recommendation: keep the instance details and result tables in the article, but move most operational steps to `docs/h100_lambda_run.md` and link out.

3. **Fix small copy issues.**

   Examples:
   - `TensorRT-LLM,and` needs a space.
   - The article mixes US and UK spelling: `optimization/optimising`, `artifact/artefact`, `utilized/utilised`.

   Recommendation: choose one style and normalize before publishing.

## Overall Assessment

The article is publishable after consistency and wording cleanup. The technical story is strong: the expected BF16 speedup fails, profiling identifies why, the mask fold changes the result, and the H100 run confirms the explanation is about workload shape rather than one consumer GPU.

The main pre-publication work is not a rewrite. It is:

1. Remove visible TODOs.
2. Make benchmark numbers and prompt lengths consistent.
3. Update stale generated table assets.
4. Qualify the strongest claims so they match exactly what the measurements prove.
