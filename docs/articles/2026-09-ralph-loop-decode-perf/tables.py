"""Table definitions for the Ralph-loop decode-performance article.

Consumed by scripts/render_article_tables_dw.py (Datawrapper) or
scripts/render_article_tables.py (matplotlib fallback). To regenerate:

    uv run python scripts/render_article_tables_dw.py --article 2026-09-ralph-loop-decode-perf

Tables here are the ones worth having as standalone PNGs (Substack embeds —
GitHub-flavoured tables don't survive the Substack copy-paste, so each table
in `article.md` gets a parallel PNG export here). The article markdown itself
keeps the GitHub-flavoured tables; these PNGs are an export, not a dependency.

Entries are ordered to match each table's appearance in `article.md`.

NOTE on table 1: in `article.md` each change links to its commit on GitHub. A
PNG cannot carry links, so the short SHA is rendered as its own column instead
— that keeps the commit discoverable from the image. The markdown version keeps
the links.
"""

TABLES = [
    # ── 1. The seven accepted changes, in order, with the TPS after each ────
    {
        "filename": "seven-changes.png",
        "headers": ["#", "Change", "Commit", "TPS", "Δ"],
        "rows": [
            ["1", "top_k_sample: full-vocab qsort → one-pass selection", "aecc951",  "72.49", "+21%"],
            ["2", "Batch the decode softmax across heads",               "3f47f3c",  "91.19", "+26%"],
            ["3", "Batch the per-head attention GEMVs",                  "2e5c184", "213.25", "+133%"],
            ["4", "Fuse Q/K/V into one GEMM",                            "af4ac77", "229.43", "+7.5%"],
            ["5", "Delete a dead per-layer cudaMemcpy",                  "9c9250d", "237.15", "+3.1%"],
            ["6", "Fuse bias + residual at both joins",                  "012e2e9", "246.80", "+4.0%"],
            ["7", "Fuse bias + GELU on the MLP projection",              "2a183a7", "259.67", "+5.3%"],
        ],
        "alignments": ["center", "left", "left", "right", "right"],
        # TPS and Δ forced to text: Datawrapper's number sniffer strips the sign
        # and unit, rendering "+21%" as 21 and "246.80" as 246.8.
        "col_types": ["auto", "text", "text", "text", "text"],
        "col_widths": [0.05, 0.52, 0.13, 0.15, 0.15],
        "fig_width": 14,
    },

    # ── 2. Every size and dtype, before and after the three rounds ─────────
    {
        "filename": "across-sizes-and-dtypes.png",
        "headers": ["Build", "Before", "After", "Change"],
        "rows": [
            ["BF16 small",      "175.54",  "897.06", "5.1×"],
            ["BF16 medium",      "98.19",  "457.48", "4.7×"],
            ["**BF16 large**", "**59.83**", "**259.67**", "**4.3×**"],
            ["FP32 large",      "155.01",  "174.78", "+12.8%"],
            ["INT8 large",      "182.91",  "194.83", "+6.5%"],
        ],
        "alignments": ["left", "right", "right", "right"],
        "col_types": ["text", "text", "text", "text"],
        "col_widths": [0.34, 0.22, 0.22, 0.22],
        "fig_width": 10,
    },

    # ── 3. INT8 accuracy against the BF16 reference, before and after ──────
    {
        "filename": "int8-accuracy.png",
        "headers": ["INT8 build", "Agrees with the BF16 reference for"],
        "rows": [
            ["before", "**52 characters**"],
            ["after",  "**996 characters** (the whole sample)"],
        ],
        "alignments": ["left", "left"],
        "col_widths": [0.30, 0.70],
        "fig_width": 10,
    },
]
