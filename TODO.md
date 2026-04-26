# Optimization TODOs

This file tracks performance optimizations to be implemented later. These optimizations focus on reducing redundant calculations during the generation phase (`n_new_tokens == 1`).

## 1. Embeddings Calculation — DONE
**Location:** Main generation loop (`gpt2.c:1596`).
**Issue:** The raw embeddings (`wte + wpe`) were recalculated for the entire sequence (`current_seq_len`) on every generation step.
**Fix:** `embeddings_cuda` is now called with `last_index` (start row) and `n_new_tokens`, so only the newly generated token is embedded and appended.

## 2. First LayerNorm (LN1) — DONE
**Location:** `transformer_block_gpu` / `transformer_block_cpu` (`gpt2.c:866`, `gpt2.c:1044`).
**Issue:** `layernorm_cuda` / `layernorm_2d` for LN1 processed all `n_tokens` during the generation phase.
**Fix:** LN1 now runs only on `n_new_tokens` rows starting at `cache_start_index`. The subsequent `dot_2d` calls for Q/K/V also operate on the same `n_new_tokens` slice (the K/V slice is appended into the KV cache).

## 3. Final LayerNorm (Before Logits) — DONE
**Location:** Main inference loop (`gpt2.c:1648-1649`).
**Issue:** The final LayerNorm computed mean/variance for the entire sequence before the final logits calculation.
**Fix:** During generation, `ln_rows = 1` and the LayerNorm + logits projection run only on the last token's hidden state.

---

## 4. Deduplicate `wte` / `lm_head` in weight files
**Location:** `extract_weights.py` (writer) and the C weight loader in `gpt2.c` (reader).
**Issue:** HuggingFace's `GPT2LMHeadModel.state_dict()` exposes both `transformer.wte.weight` and `lm_head.weight` even though they are tied (point to the same underlying tensor). The current loop writes every state dict entry verbatim, so each `.bin` stores the `[vocab_size, d_model]` matrix twice. Wasted bytes per model:
- Small: 50257 × 768 × 4  ≈ 154 MB
- Medium: 50257 × 1024 × 4 ≈ 206 MB
- Large: 50257 × 1280 × 4 ≈ 257 MB

**Fix options:**
- Skip `lm_head.weight` on write (`if k == "lm_head.weight": continue`) and on the C side alias the `lm_head` pointer to `wte` (transposed if needed) after loading.
- Or iterate `model.transformer.state_dict()` plus an explicit unembedding entry, so `lm_head` never appears.

Either approach shrinks each `.bin` by 150–260 MB with no behavioral change. The C loader will need to be updated in lockstep so existing files keep working (or bump a format version).
