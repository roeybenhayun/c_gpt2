"""Table definitions for the GPU inference article.

Consumed by scripts/render_article_tables.py. To regenerate the PNGs:

    uv run python scripts/render_article_tables.py --article 2026-04-gpu-inference
"""

TABLES = [
    {
        "filename": "model-sizes.png",
        "headers": ["Model", "Params", "d_model", "num_layers"],
        "rows": [
            ["Small",  "124M", "768",  "12"],
            ["Medium", "355M", "1024", "24"],
            ["Large",  "774M", "1280", "36"],
            ["XLarge", "1.5B", "1600", "(not supported)"],
        ],
        "alignments": ["center", "right", "right", "center"],
        "col_widths": [0.20, 0.20, 0.20, 0.40],
        "fig_width": 8,
    },
    {
        "filename": "weight-matrices.png",
        "headers": ["Weight", "Shape", "Floats"],
        "rows": [
            ["W_q",                     "d_model × d_model",   "d_model²"],
            ["W_k",                     "d_model × d_model",   "d_model²"],
            ["W_v",                     "d_model × d_model",   "d_model²"],
            ["attn_proj_weight",        "d_model × d_model",   "d_model²"],
            ["W1 (MLP up, c_fc)",       "d_model × 4·d_model", "4·d_model²"],
            ["W2 (MLP down, c_proj)",   "4·d_model × d_model", "4·d_model²"],
        ],
        "alignments": ["left", "left", "left"],
        "col_widths": [0.40, 0.35, 0.25],
        "fig_width": 10,
    },
    {
        "filename": "per-token-bytes.png",
        "headers": ["Model", "d_model", "n_layers", "Per-layer", "× n_layers", "+ LM head", "Total/token"],
        "rows": [
            ["Small",  "768",  "12", "27 MiB", "324 MiB",  "147 MiB", "471 MiB"],
            ["Medium", "1024", "24", "48 MiB", "1152 MiB", "196 MiB", "1.32 GiB"],
            ["Large",  "1280", "36", "75 MiB", "2700 MiB", "245 MiB", "2.88 GiB"],
        ],
        "alignments": ["center", "right", "right", "right", "right", "right", "right"],
        "fig_width": 12,
    },
    {
        "filename": "file-size-sanity-check.png",
        "headers": ["Model", "per-layer × n_layers", "+ LM head (wte_T)", "+ wte (gather)", "+ wpe + LN", "≈ file size", "actual file"],
        "rows": [
            ["Small",  "324 MiB",  "147 MiB", "147 MiB", "~3 MiB", "~621 MiB",  "622 MiB ✓"],
            ["Medium", "1152 MiB", "196 MiB", "196 MiB", "~4 MiB", "~1548 MiB", "1550 MiB ✓"],
            ["Large",  "2700 MiB", "245 MiB", "245 MiB", "~5 MiB", "~3196 MiB", "3198 MiB ✓"],
        ],
        "alignments": ["center", "right", "right", "right", "right", "right", "right"],
        "fig_width": 14,
    },
    {
        "filename": "theoretical-max-vs-measured.png",
        "headers": ["Model", "Per-token bytes", "Theoretical max TPS", "Measured TPS", "Utilization"],
        "rows": [
            ["Small",  "471 MiB",  "~2040", "158", "~7.7%"],
            ["Medium", "1.32 GiB", "~727",  "95",  "~13%"],
            ["Large",  "2.88 GiB", "~333",  "60",  "~18%"],
        ],
        "alignments": ["center", "right", "right", "right", "right"],
        "col_widths": [0.15, 0.20, 0.25, 0.20, 0.20],
        "fig_width": 10,
    },
    {
        "filename": "block-configurations.png",
        "headers": ["Kernel", "Block size", "Threads/block"],
        "rows": [
            ["add_2d",         "(16, 16)",  "256"],
            ["add_bias",       "(16, 16)",  "256"],
            ["gelu",           "(32, 32)",  "1024"],
            ["causal_masking", "(256, 1)",  "256"],
            ["concat_heads",   "(64, 1)",   "64"],
            ["embeddings",     "(16, 16)",  "256"],
            ["layernorm",      "(1024, 1)", "1024"],
            ["softmax",        "(256, 1)",  "256"],
        ],
        "alignments": ["left", "center", "right"],
        "col_widths": [0.45, 0.30, 0.25],
        "fig_width": 8,
    },
]
