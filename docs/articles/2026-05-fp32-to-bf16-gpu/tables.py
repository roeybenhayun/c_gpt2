"""Table definitions for the FP32→BF16 GPU article.

Consumed by scripts/render_article_tables.py. To regenerate the PNGs:

    uv run python scripts/render_article_tables.py --article 2026-05-fp32-to-bf16-gpu

Tables here are the ones worth having as standalone PNGs (substack embeds,
side-by-side comparisons during edits). The article markdown itself uses
GitHub-flavoured tables; these are an export, not a dependency.
"""

TABLES = [
    # ── Initial result — the article's "wait, what?" moment ──────────────────
    {
        "filename": "default-benchmark-tps.png",
        "headers": ["Model", "FP32 TPS", "BF16 TPS", "Speedup"],
        "rows": [
            ["Small",  "155.0", "184.8", "1.19×"],
            ["Medium", "95.6",  "103.2", "1.08×"],
            ["Large",  "60.1",  "60.3",  "~1.00×"],
        ],
        "alignments": ["center", "right", "right", "right"],
        "col_widths": [0.25, 0.25, 0.25, 0.25],
        "fig_width": 8,
    },

    # ── The sweet-spot regime curve ──────────────────────────────────────────
    {
        "filename": "bf16-sweet-spot-by-M.png",
        "headers": ["M (prompt tokens)", "BF16 / FP32 speedup", "Note"],
        "rows": [
            ["1 (decode)", "≈ 1.00×", "no tensor cores at all"],
            ["35",         "0.91×",   "overhead dominates, BF16 slightly slower"],
            ["123",        "1.05×",   "tensor cores starting to engage"],
            ["543",        "1.13×",   ""],
            ["1024",       "1.30×",   "fully in the tensor-core regime"],
        ],
        "alignments": ["center", "right", "left"],
        "col_widths": [0.22, 0.22, 0.56],
        "fig_width": 12,
    },

    # ── Pure prefill across model sizes (5080) ───────────────────────────────
    {
        "filename": "prefill-ttft-5080.png",
        "headers": ["Model", "FP32 TTFT", "BF16 TTFT", "Δ"],
        "rows": [
            ["Small",   "86 ms",  "103 ms", "+20.3% slower"],
            ["Medium", "133 ms",  "131 ms", "−1.6%"],
            ["Large",  "205 ms",  "172 ms", "−16.2% faster"],
        ],
        "alignments": ["center", "right", "right", "right"],
        "col_widths": [0.20, 0.25, 0.25, 0.30],
        "fig_width": 9,
    },

    # ── Pure decode across model sizes (5080) ────────────────────────────────
    {
        "filename": "decode-tps-5080.png",
        "headers": ["Model", "FP32 TPS", "BF16 TPS", "Speedup"],
        "rows": [
            ["Small",  "153.9", "179.6", "1.17×"],
            ["Medium", "93.0",  "101.6", "1.09×"],
            ["Large",  "57.9",  "59.3",  "1.02× — basically tied"],
        ],
        "alignments": ["center", "right", "right", "left"],
        "col_widths": [0.18, 0.20, 0.20, 0.42],
        "fig_width": 10,
    },

    # ── Final scoreboard (after mask fix) ────────────────────────────────────
    {
        "filename": "scoreboard.png",
        "headers": ["Workload", "Effective M", "Before BF16", "First BF16 attempt", "After mask fix"],
        "rows": [
            ["1024-token prefill TTFT, Large",        "1024",          "n/a (FP32-only)",  "0.84× — slower",   "1.30× — faster ✓"],
            ["1024-token prefill kernel total, Large", "1024",         "214 ms (FP32)",    "266 ms (BF16)",    "65 ms — 1.84× ✓"],
            ["Default run.sh TPS, Small",              "1 + ~13",      "156",              "185 (1.19×)",      "178 (1.14×)"],
            ["Default run.sh TPS, Medium",             "1 + ~13",      "96",               "103 (1.08×)",      "104 (1.10×)"],
            ["Default run.sh TPS, Large",              "1 + ~13",      "60",               "60 (1.00×)",       "60 (1.02×)"],
        ],
        "alignments": ["left", "center", "right", "right", "right"],
        "col_widths": [0.32, 0.12, 0.18, 0.18, 0.20],
        "fig_width": 16,
    },

    # ── H100 vs 5080 — decode TPS ───────────────────────────────────────────
    {
        "filename": "h100-vs-5080-decode-tps.png",
        "headers": ["Model", "5080 FP32", "H100 FP32", "5080 BF16", "H100 BF16"],
        "rows": [
            ["Small",  "153.9", "134.6", "179.6", "159.7"],
            ["Medium", "93.0",  "88.9",  "101.6", "98.8"],
            ["Large",  "57.9",  "60.3",  "59.3",  "61.6"],
        ],
        "alignments": ["center", "right", "right", "right", "right"],
        "col_widths": [0.20, 0.20, 0.20, 0.20, 0.20],
        "fig_width": 11,
    },

    # ── H100 vs 5080 — prefill TTFT ─────────────────────────────────────────
    {
        "filename": "h100-vs-5080-prefill-ttft.png",
        "headers": ["Model", "5080 FP32 (ms)", "H100 FP32 (ms)", "5080 BF16 (ms)", "H100 BF16 (ms)"],
        "rows": [
            ["Small",  "86",  "136", "103", "145"],
            ["Medium", "133", "173", "131", "178"],
            ["Large",  "205", "235", "172", "215"],
        ],
        "alignments": ["center", "right", "right", "right", "right"],
        "col_widths": [0.16, 0.21, 0.21, 0.21, 0.21],
        "fig_width": 12,
    },

    # ── H100 nsys — kernel time vs wall clock (Amdahl) ──────────────────────
    {
        "filename": "h100-amdahl-kernel-fraction.png",
        "headers": ["Quantity", "FP32", "BF16"],
        "rows": [
            ["End-to-end wall clock (E2E)",          "861 ms", "851 ms"],
            ["Sum of GPU kernel time",               "421 ms", "380 ms"],
            ["Host / non-kernel time (E2E − kernel)", "440 ms", "471 ms"],
            ["Kernel fraction of wall clock",        "49%",    "45%"],
        ],
        "alignments": ["left", "right", "right"],
        "col_widths": [0.50, 0.25, 0.25],
        "fig_width": 12,
    },

    # ── H100 nsys — tensor-op kernel attribution ────────────────────────────
    {
        "filename": "h100-tensor-op-kernels.png",
        "headers": ["Kernel", "BF16 time", "Calls", "What it is"],
        "rows": [
            ["nvjet_tst_64x8_64x16_4x1_v_bz_TNT",        "28.3 ms", "5,580", "Hopper-native tensor-op GEMM"],
            ["nvjet_tst_64x8_64x16_1x1_v_bz_NNT",        "10.7 ms", "2,880", "another tile shape"],
            ["nvjet_tst_64x8_64x16_4x1_v_bz_splitK_TNT", "8.7 ms",  "1,116", "split-K variant for large K"],
            ["cutlass tensorop GEMM (generic)",          "8.7 ms",  "1,440", "non-Hopper-specific CUTLASS tensor-op"],
            ["nvjet_tst_8x64_64x16_2x1_v_bz_TNN",        "7.4 ms",  "2,880", "another tile shape"],
            ["Sum (all tensor-core kernels)",            "~64 ms",  "—",     "replaces ~40 ms of FP32 sm80_xmma_gemm_*"],
        ],
        "alignments": ["left", "right", "right", "left"],
        "col_widths": [0.36, 0.12, 0.10, 0.42],
        "fig_width": 16,
    },
]
