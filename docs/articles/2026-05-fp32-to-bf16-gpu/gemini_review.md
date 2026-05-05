# Article Review: GPT-2 in C — FP32 to BF16 on GPU

**Target Directory:** `docs/articles/2026-05-fp32-to-bf16-gpu/`
**Review Date:** Sunday, May 3, 2026

## Executive Summary
The article is a high-quality, deep-dive technical piece that provides excellent insights into the transition from FP32 to BF16. The "serial dilutions" analysis and the $M=1$ bottleneck explanation are standout sections. The technical implementation in the codebase is sound and follows the article's narrative.

Below are the recommended improvements categorized by priority.

---

## Priority: High (Blockers for Publishing)

### 1. Resolve Missing Assets (Plots)
The `article.md` contains several `TODO` placeholders for plot paths (e.g., lines 135, 143, 219, 227). These must be replaced with the actual paths to the generated PNGs before publishing.
*   **Action:** Verify that `performance_analysis.py` has generated these files and update the Markdown links.

### 2. Scoreboard Consistency Check
Ensure the numbers in the "Scoreboard" (line 360) and the "After the fix — TTFT benchmark" (line 259) are consistent with the latest benchmark runs. There is a slight discrepancy between the numbers in `tables.py` and the text.
*   **Action:** Re-run the final benchmarks and update both `tables.py` and `article.md` to ensure they match exactly.

### 3. Path and Title Typo
The requested review mentioned a directory `fp32-tp-bf16-gpu`, but the project uses `2026-05-fp32-to-bf16-gpu`.
*   **Action:** Ensure all internal links (e.g., line 386) use the correct directory naming convention.

---

## Priority: Medium (Quality Improvements)

### 1. Codebase Cleanup (`gpt2.c`)
The `gpt2.c` file contains several old `TODO`s and large blocks of commented-out code (e.g., the `add_tensor_to_layer` function and old `load_layers_weights`). While this is a research project, readers of the article who check the source will appreciate a cleaner implementation.
*   **Action:** Clean up `gpt2.c` by removing dead code or moving experimental sections to a separate file.

### 2. Clarify "Balanced" Workload
The article notes that "Balanced" is a local naming. To make the article more accessible to a general audience on Substack, consider defining it more formally at its first mention (e.g., "Mixed Sequence Length (200/200)").
*   **Action:** Update the "Prefill vs decode" section to clarify that "Balanced" represents a common chat-like workload with roughly equal input and output lengths.

### 3. Verify H100 Runbook
The link to `../../h100_lambda_run.md` (line 386) is correct relative to the article, but ensure that the instructions in that file are fully up to date with the recent Blackwell-based 5080 results.
*   **Action:** Briefly review `docs/h100_lambda_run.md` to ensure it still makes sense in the context of the new article.

---

## Priority: Low (Polish & Future Work)

### 1. Explicitly Rank "What's Next"
The "What's next" section is honest and exploratory. However, based on the article's own conclusion that decode is bandwidth-bound, explicitly ranking "Weight-only quantization" as the high-impact path (over `cublasLt` tweaking) would reinforce the article's main lesson.
*   **Action:** Slightly adjust the wording in the final section to emphasize that quantization is the standard industry answer to the $M=1$ problem.

### 2. Table Alignment Consistency
Check the Markdown table alignments (e.g., line 251 vs line 269). Some use colons for alignment and some don't.
*   **Action:** Standardize table formatting for a more polished look on GitHub/Substack.

---

## Final Verdict
The article is **Ready to Publish** once the High Priority placeholders are resolved. The technical narrative is compelling and the data-driven approach to debunking the "BF16 speedup" myth for small batches is very valuable for the community.
