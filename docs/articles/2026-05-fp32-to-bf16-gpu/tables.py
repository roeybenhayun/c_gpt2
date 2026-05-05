"""Table definitions for the FP32→BF16 GPU article.

Consumed by scripts/render_article_tables_dw.py (Datawrapper) or
scripts/render_article_tables.py (matplotlib fallback). To regenerate:

    uv run python scripts/render_article_tables_dw.py --article 2026-05-fp32-to-bf16-gpu

Tables here are the ones worth having as standalone PNGs (substack embeds,
side-by-side comparisons during edits). The article markdown itself uses
GitHub-flavoured tables; these are an export, not a dependency.

Entries are ordered to match each table's appearance in `article.md`. The
Datawrapper renderer auto-prepends a zero-padded NN- prefix to each output
filename based on this order, so the PNGs in `assets/tables/` sort the same
way they appear in the article — convenient when bulk-uploading to Substack.
The two trailing entries (bf16-sweet-spot-by-M, decode-tps-5080) are
supplementary summaries not directly embedded in the article.
"""

TABLES = [
    # ── 1. Workload shapes (Prefill vs decode primer) ───────────────────────
    {
        "filename": "workload-shapes.png",
        "headers": ["Shape", "Prompt", "Output", "Phase mix", "Real-world example"],
        "rows": [
            ["Prefill-dominated", "1000 tokens", "10 tokens",   "~99% prefill",  "RAG / long-context summarization"],
            ["Balanced",          "200 tokens",  "200 tokens",  "roughly 50/50", "Typical chat exchange"],
            ["Decode-dominated",  "10 tokens",   "1000 tokens", "~99% decode",   "\"Tell me a story…\" / long generation"],
        ],
        "alignments": ["left", "right", "right", "left", "left"],
        "col_widths": [0.18, 0.13, 0.13, 0.16, 0.40],
        "fig_width": 14,
    },

    # ── 2. Default benchmark — the article's "wait, what?" moment ──────────
    {
        "filename": "default-benchmark-tps.png",
        "headers": ["Model", "FP32 TPS", "BF16 TPS", "Speedup"],
        "rows": [
            ["Small",  "153.9", "179.6", "1.17×"],
            ["Medium", "93.0",  "101.6", "1.09×"],
            ["Large",  "57.9",  "59.3",  "1.02×"],
        ],
        "alignments": ["center", "right", "right", "right"],
        "col_widths": [0.25, 0.25, 0.25, 0.25],
        "fig_width": 8,
    },

    # ── 3. First nsys investigation — decode kernel mix (FP32 vs BF16) ──────
    {
        "filename": "decode-nsys-fp32-vs-bf16.png",
        "headers": ["Kernel family", "FP32 (ms)", "BF16 (ms)", "Δ"],
        "rows": [
            ["cublas gemvx (all variants)", "3713", "4773", "+1060"],
            ["cublas gemvNSP",              "1591",  "246", "−1345"],
            ["cublas splitKreduce",            "2",  "321",  "+319"],
            ["softmax_kernel",              "1156", "1236",   "+80"],
            ["layernorm_kernel",             "224",  "206",   "−18"],
            ["add_bias_kernel",              "172",  "152",   "−19"],
            ["**TOTAL kernel time**",       "**7208**", "**7075**", "**−133 (−1.8%)**"],
        ],
        "alignments": ["left", "right", "right", "right"],
        "col_widths": [0.45, 0.18, 0.18, 0.19],
        "fig_width": 12,
    },

    # ── 4. Prefill TTFT — before the mask fix ───────────────────────────────
    {
        "filename": "prefill-ttft-before-fix.png",
        "headers": ["Actual tokens", "FP32 TTFT", "BF16 TTFT", "Δ TTFT", "Speedup"],
        "rows": [
            ["35",   "0.0707 s", "0.0794 s", "+0.0087 s", "0.89×"],
            ["123",  "0.0904 s", "0.0906 s", "+0.0002 s", "1.00×"],
            ["543",  "0.1821 s", "0.2079 s", "+0.0258 s", "0.88×"],
            ["1024", "0.3165 s", "0.3771 s", "+0.0606 s", "0.84×"],
        ],
        "alignments": ["right", "right", "right", "right", "right"],
        "col_widths": [0.18, 0.20, 0.20, 0.22, 0.20],
        "fig_width": 12,
    },

    # ── 5. 1024-token prefill kernel profile — before the mask fix ──────────
    {
        "filename": "prefill-1024-kernels-before-fix.png",
        "headers": ["Kernel family", "FP32 (ms)", "BF16 (ms)", "Δ", "Note"],
        "rows": [
            ["cutlass simt GEMM (FP32 path)",       "69.2",  "0",     "−69.2", "replaced"],
            ["cutlass tensor-op GEMM (BF16 path)",   "0",   "22.1",  "+22.1", "tensor cores engaged ✓"],
            ["casual_masking_kernel",              "100.9", "**201.3**", "**+100.4**", "2.00× slower"],
            ["concat_heads",                        "31.0", "31.0",   "~0",   "unchanged"],
            ["softmax / layernorm / add_bias / gelu", "small", "small", "small", "as expected"],
            ["**TOTAL kernel time**",               "**214**",  "**266**",   "**+52**",  "BF16 1.24× slower"],
        ],
        "alignments": ["left", "right", "right", "right", "left"],
        "col_widths": [0.36, 0.12, 0.12, 0.12, 0.28],
        "fig_width": 14,
    },

    # ── 6. 1024-token prefill kernel profile — after the mask fix ───────────
    {
        "filename": "prefill-1024-kernels-after-fix.png",
        "headers": ["Kernel family", "FP32 (ms)", "BF16 (ms)", "Δ", "Note"],
        "rows": [
            ["cutlass simt GEMM (FP32 path)",       "73.2", "0",    "−73.2", "replaced"],
            ["cutlass tensor-op GEMM (BF16 path)",  "0",    "22.1", "+22.1", "tensor cores still engaging"],
            ["casual_masking_kernel",               "**gone**", "**gone**", "—",     "0 instances either way"],
            ["concat_heads",                        "31.7", "31.1", "−0.6",  "unchanged"],
            ["softmax_kernel",                      "5.1",  "5.3",  "+0.2",  "mask check is essentially free"],
            ["layernorm / add_bias / gelu",         "small","small","small", "unchanged"],
            ["cublas splitKreduce",                 "1.1",  "0",    "−1.1",  "also gone in BF16"],
            ["**TOTAL kernel time**",               "**117**",  "**65**",   "**−52**",   "**BF16 1.84× faster than FP32**"],
        ],
        "alignments": ["left", "right", "right", "right", "left"],
        "col_widths": [0.34, 0.10, 0.10, 0.10, 0.36],
        "fig_width": 14,
    },

    # ── 7. Prefill TTFT — after the mask fix (with before-fix comparison) ──
    {
        "filename": "prefill-ttft-after-fix.png",
        "headers": ["Actual tokens", "FP32 TTFT", "BF16 TTFT", "Δ TTFT", "Speedup", "Speedup before fix"],
        "rows": [
            ["35",   "0.0700 s", "0.0769 s", "+0.0069 s", "0.91×", "0.89×"],
            ["123",  "0.0900 s", "0.0854 s", "−0.0046 s", "1.05×", "1.00×"],
            ["543",  "0.1509 s", "0.1333 s", "−0.0177 s", "1.13×", "0.88×"],
            ["**1024**", "**0.2245 s**", "**0.1727 s**", "**−0.0519 s**", "**1.30×**", "0.84×"],
        ],
        "alignments": ["right", "right", "right", "right", "right", "right"],
        "col_widths": [0.13, 0.16, 0.16, 0.18, 0.13, 0.24],
        "fig_width": 14,
    },

    # ── 8. Pure prefill across model sizes (5080) ───────────────────────────
    {
        "filename": "prefill-ttft-5080.png",
        "headers": ["Model", "FP32 TTFT", "BF16 TTFT", "Δ"],
        "rows": [
            ["Small",   "86 ms",  "103 ms", "**+20.3% slower**"],
            ["Medium", "133 ms",  "131 ms", "−1.6%"],
            ["Large",  "205 ms",  "172 ms", "**−16.2% faster**"],
        ],
        "alignments": ["center", "right", "right", "right"],
        "col_widths": [0.20, 0.25, 0.25, 0.30],
        "fig_width": 9,
    },

    # ── 9. Final scoreboard (after mask fix) ────────────────────────────────
    {
        "filename": "scoreboard.png",
        "headers": ["Workload", "Effective M", "Before BF16", "First BF16 attempt", "After mask fix"],
        "rows": [
            ["1024-token prefill TTFT, Large",        "1024",          "n/a (FP32-only)",  "0.84× — slower",   "1.30× — faster ✓"],
            ["1024-token prefill kernel total, Large", "1024",         "214 ms (FP32)",    "266 ms (BF16)",    "65 ms — 1.84× ✓"],
            ["Default run.sh TPS, Small",              "1 + ~19",      "156",              "185 (1.19×)",      "178 (1.14×)"],
            ["Default run.sh TPS, Medium",             "1 + ~19",      "96",               "103 (1.08×)",      "104 (1.10×)"],
            ["Default run.sh TPS, Large",              "1 + ~19",      "60",               "60 (1.00×)",       "60 (1.02×)"],
        ],
        "alignments": ["left", "center", "right", "right", "right"],
        "col_widths": [0.32, 0.12, 0.18, 0.18, 0.20],
        "fig_width": 16,
    },

    # ── 10. H100 instance setup ─────────────────────────────────────────────
    {
        "filename": "h100-instance-setup.png",
        "headers": ["Field", "Value"],
        "rows": [
            ["Instance",                            "1× H100 80GB SXM5"],
            ["Host",                                "26 vCPUs, 225 GiB RAM, 2.8 TiB SSD"],
            ["Image",                               "Ubuntu 24.04 + Lambda Stack (driver 580.105.08, CUDA 12.8)"],
            ["Cost",                                "$4.29 / GPU / hr (May 2026) — billed per minute"],
            ["Total spend for the full benchmark", "~$2"],
        ],
        "alignments": ["left", "left"],
        "col_widths": [0.30, 0.70],
        "fig_width": 14,
    },

    # ── 11. H100 vs 5080 — decode TPS ───────────────────────────────────────
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

    # ── 12. H100 vs 5080 — prefill TTFT ─────────────────────────────────────
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

    # ── 13. H100 nsys — kernel time vs wall clock (Amdahl) ──────────────────
    {
        "filename": "h100-amdahl-kernel-fraction.png",
        "headers": ["Quantity", "FP32", "BF16"],
        "rows": [
            ["End-to-end wall clock (E2E)",          "861 ms", "851 ms"],
            ["**Sum of GPU kernel time**",           "**421 ms**", "**380 ms**"],
            ["Host / non-kernel time (E2E − kernel)", "440 ms", "471 ms"],
            ["**Kernel fraction of wall clock**",    "**49%**",    "**45%**"],
        ],
        "alignments": ["left", "right", "right"],
        "col_widths": [0.50, 0.25, 0.25],
        "fig_width": 12,
    },

    # ── 14. H100 nsys — tensor-op kernel attribution ────────────────────────
    {
        "filename": "h100-tensor-op-kernels.png",
        "headers": ["Kernel", "BF16 time", "Calls", "What it is"],
        "rows": [
            ["nvjet_tst_64x8_64x16_4x1_v_bz_TNT",        "28.3 ms", "5,580", "Hopper-native tensor-op GEMM"],
            ["nvjet_tst_64x8_64x16_1x1_v_bz_NNT",        "10.7 ms", "2,880", "another tile shape"],
            ["nvjet_tst_64x8_64x16_4x1_v_bz_splitK_TNT", "8.7 ms",  "1,116", "split-K variant for large K"],
            ["cutlass tensorop GEMM (generic)",          "8.7 ms",  "1,440", "non-Hopper-specific CUTLASS tensor-op"],
            ["nvjet_tst_8x64_64x16_2x1_v_bz_TNN",        "7.4 ms",  "2,880", "another tile shape"],
            ["**Sum (all tensor-core kernels)**",        "**~64 ms**",  "—",     "replaces ~40 ms of FP32 sm80_xmma_gemm_*"],
        ],
        "alignments": ["left", "right", "right", "left"],
        "col_widths": [0.36, 0.12, 0.10, 0.42],
        "fig_width": 16,
    },

    # ── 15. H100 — hypotheses confirmed by the cross-check ──────────────────
    {
        "filename": "h100-hypotheses-confirmed.png",
        "headers": ["Hypothesis from earlier section", "Confirmed?", "Numerical evidence"],
        "rows": [
            ["GPU isn't the bottleneck on this codebase",        "✓", "~50% of wall time is non-kernel"],
            ["Tensor cores engage on H100 BF16 prefill",         "✓", "nvjet_tst_* + cutlass tensorop GEMM = ~64 ms"],
            ["H100 advantage is bounded by Amdahl on host",      "✓", "Ceiling is 2.24× even with infinite GPU, on this workload as currently structured"],
            ["Next ~2× requires *shape* changes, not silicon",   "✓", "Confirmed quantitatively"],
        ],
        "alignments": ["left", "center", "left"],
        "col_widths": [0.40, 0.12, 0.48],
        "fig_width": 16,
    },

    # ── 16. Dilution layer 2 — kernel category breakdown (Amdahl on GPU) ────
    {
        "filename": "dilution-kernel-breakdown.png",
        "headers": ["Kernel category", "FP32 time", "BF16 time", "Speedup"],
        "rows": [
            ["GEMM only (cutlass simt → cutlass tensorop)",                                   "73.2 ms", "22.1 ms", "**3.32×**"],
            ["Everything else (softmax, layernorm, gelu, add_bias, concat_heads, splitKreduce, …)", "43.8 ms", "42.9 ms", "1.02×"],
            ["**Total kernel time**",                                                          "**117 ms**",  "**65 ms**",   "**1.80×**"],
        ],
        "alignments": ["left", "right", "right", "right"],
        "col_widths": [0.55, 0.15, 0.15, 0.15],
        "fig_width": 16,
    },

    # ── 17. Dilution layers and the optimization levers that lift each one ──
    {
        "filename": "dilution-layers-levers.png",
        "headers": ["Dilution layer", "What we lost", "What lifts it"],
        "rows": [
            ["Peak vs cuBLAS achieved (15× → 3.3×)",
             "~78%",
             "bigger M (batched inference, speculative decoding); cublasLt with explicit algo pinning; persistent kernels"],
            ["GEMM share of kernel total (3.3× → 1.8×)",
             "~45%",
             "kernel fusion (fold non-GEMM work into the GEMM epilogue); fuse adjacent custom kernels so they share a single launch"],
            ["Host overhead invariance (1.8× → 1.3×)",
             "~28%",
             "CUDA Graphs (capture-once, replay); async tokenizer; batched logit sampling; remove per-layer syncs"],
        ],
        "alignments": ["left", "center", "left"],
        "col_widths": [0.28, 0.10, 0.62],
        "fig_width": 18,
    },

    # ── 18. (supplementary, not in article) Sweet-spot regime curve ─────────
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

    # ── 19. (supplementary, not in article) Pure decode across sizes (5080) ─
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
]
