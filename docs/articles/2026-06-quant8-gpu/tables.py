"""Table definitions for the INT8-on-GPU article.

Consumed by scripts/render_article_tables_dw.py (Datawrapper) or
scripts/render_article_tables.py (matplotlib fallback). To regenerate:

    uv run python scripts/render_article_tables_dw.py --article 2026-06-quant8-gpu

Tables here are the ones worth having as standalone PNGs (Substack embeds —
GitHub-flavoured tables don't survive the Substack copy-paste, so each table
in `article.md` gets a parallel PNG export here). The article markdown itself
keeps the GitHub-flavoured tables; these PNGs are an export, not a dependency.

Entries are ordered to match each table's appearance in `article.md`. The
Datawrapper renderer auto-prepends a zero-padded NN- prefix to each output
filename based on this order, so PNGs in `assets/tables/` sort the same way
they appear in the article — convenient when bulk-uploading to Substack.
"""

TABLES = [
    # ── 1. What gets quantized (per-tensor INT8 vs FP32 in each transformer block) ──
    {
        "filename": "quantized-tensors.png",
        "headers": ["Tensor", "Shape (Small)", "Quantized?", "Why"],
        # NOTE: Shape column uses actual Small numbers (d_model=768, d_ff=3072,
        # vocab=50 257, ctx=1024) rather than symbolic identifiers. Datawrapper's
        # markdown renderer mangles `d_model` (interprets `_` as italic) and
        # especially breaks on `3*d_model` where the adjacent `*` and `_` collide.
        # Plain numbers sidestep the markdown parser entirely. _escape_md in the
        # renderer handles the underscores in the Tensor column names cleanly.
        "rows": [
            ["W_qkv (packed)",                                    "[768, 2304]",         "INT8", "Large matmul input — biggest win"],
            ["attn_proj",                                         "[768, 768]",          "INT8", "Large matmul input"],
            ["W1 (FFN up)",                                       "[768, 3072]",         "INT8", "The biggest matmul in the block"],
            ["W2 (FFN down)",                                     "[3072, 768]",         "INT8", "Mirror of W1"],
            ["b_qkv, attn_proj_bias, b1, b2",                     "small vectors",       "FP32", "Add directly to activations — quant noise unbuffered"],
            ["ln1_gamma/beta, ln2_gamma/beta, lnf_gamma/beta",    "[768]",               "FP32", "Same reason; also tiny"],
            ["wte, wpe",                                          "[50 257, 768], [1024, 768]", "FP32", "Lookups, not GEMMs — INT8 buys nothing on speed"],
        ],
        "alignments": ["left", "left", "center", "left"],
        "col_widths": [0.28, 0.20, 0.10, 0.42],
        "fig_width": 16,
    },

    # ── 2. Memory savings — FP32 source vs INT8 artifact across model sizes ─
    {
        "filename": "memory-savings.png",
        "headers": ["Model", "FP32 source", "INT8 artifact", "Saved", "Quantized (INT8)", "Preserved (FP32)", "Scales"],
        "rows": [
            ["Small",  "475 MB",   "232 MB",  "51%", "81 MB",  "151 MB",  "324 KB"],
            ["Medium", "1 354 MB", "490 MB",  "64%", "288 MB", "202 MB",  "864 KB"],
            ["Large",  "2 953 MB", "929 MB",  "69%", "675 MB", "253 MB", "1 620 KB"],
        ],
        "alignments": ["center", "right", "right", "right", "right", "right", "right"],
        "col_widths": [0.10, 0.13, 0.14, 0.10, 0.18, 0.18, 0.17],
        "fig_width": 14,
    },

    # ── 3. INT8 vs BF16 — six greedy paper-validation cases × three sizes ──
    {
        "filename": "accuracy-int8-vs-bf16-greedy.png",
        "headers": ["Size", "Identical", "≤ 1-token diff", "Mid-stream divergence (both coherent)"],
        "rows": [
            ["Small",       "1 / 6",   "1 / 6",   "4 / 6"],
            ["Medium",      "1 / 6",   "0 / 6",   "5 / 6"],
            ["Large",       "2 / 6",   "1 / 6",   "3 / 6"],
            ["**Total**",   "**4 / 18**", "**2 / 18**", "**12 / 18**"],
        ],
        "alignments": ["center", "right", "right", "right"],
        "col_widths": [0.18, 0.18, 0.24, 0.40],
        "fig_width": 12,
    },

    # ── 4. Decode TPS on RTX 5080 — FP32 / BF16 / INT8 head-to-head ────────
    {
        "filename": "decode-tps-5080.png",
        "headers": ["Model", "FP32 TPS", "BF16 TPS", "INT8 TPS", "INT8 vs FP32", "INT8 vs BF16"],
        "rows": [
            ["Small",  "153.9", "179.6", "179.3", "1.17×", "1.00×"],
            ["Medium",  "93.0", "101.6",  "98.5", "1.06×", "0.97×"],
            ["Large",   "57.9",  "59.3",  "57.1", "0.99×", "0.96×"],
        ],
        "alignments": ["center", "right", "right", "right", "right", "right"],
        "col_widths": [0.12, 0.15, 0.15, 0.15, 0.21, 0.22],
        "fig_width": 12,
    },

    # ── 5. INT8 across all three workload presets (decode / prefill / balanced) ──
    {
        "filename": "workload-presets-int8.png",
        "headers": ["Model", "Decode TPS", "Prefill TTFT", "Prefill TPS", "Balanced TPS"],
        "rows": [
            ["Small",  "179.3", "94.5 ms",  "120.3", "170.3"],
            ["Medium",  "98.5", "127.5 ms",  "71.6",  "96.7"],
            ["Large",   "57.1", "166.6 ms",  "44.4",  "57.2"],
        ],
        "alignments": ["center", "right", "right", "right", "right"],
        "col_widths": [0.14, 0.20, 0.22, 0.22, 0.22],
        "fig_width": 12,
    },
]
