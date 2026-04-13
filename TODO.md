# Optimization TODOs

This file tracks performance optimizations to be implemented later. These optimizations focus on reducing redundant calculations during the generation phase (`n_new_tokens == 1`).

## 1. Embeddings Calculation
**Location:** Main generation loop (around line 1523).
**Issue:** The raw embeddings (`wte + wpe`) are recalculated for the entire sequence (`current_seq_len`) on every generation step.
**Fix:** Only calculate the embedding for the newly generated token (the last token in the sequence) and append it to the embeddings buffer.

## 2. First LayerNorm (LN1)
**Location:** `transformer_block_gpu` / `transformer_block_cpu`
**Issue:** `layernorm_cuda` / `layernorm_2d` for LN1 processes all `n_tokens` during the generation phase instead of just the single new token.
**Fix:** 
- Add a branch for `n_new_tokens == 1` to only calculate LN1 for the last row (`i = n_tokens - 1`).
- *Important:* If LN1 is optimized to output only 1 row, the subsequent `dot_2d` calculations for $Q$, $K$, and $V$ must also be updated to process only that single row (similar to how the MLP layer handles it).

## 3. Final LayerNorm (Before Logits)
**Location:** Main inference loop (after the layer loop).
**Issue:** The final LayerNorm computes the mean/variance for the entire sequence before the final logits calculation.
**Fix:** Only calculate the final LayerNorm for the last token's hidden state, as that is the only one needed to predict the next word.
