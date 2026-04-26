"""Render the data tables for an article as PNG images.

Generic renderer. Each article defines its tables in a `tables.py` next to its
`article.md`, exposing a top-level `TABLES` list of dicts. Output PNGs land in
`docs/articles/<slug>/assets/tables/`.

Usage:
    uv run python scripts/render_article_tables.py --article 2026-04-gpu-inference
"""

import argparse
import importlib.util
import os
import sys

import matplotlib.pyplot as plt

# Visual style — clean, modern, blog-friendly
HEADER_BG = "#34495e"
HEADER_FG = "#ffffff"
EVEN_ROW_BG = "#ecf0f1"
ODD_ROW_BG = "#ffffff"
BORDER_COLOR = "#bdc3c7"
FONT_SIZE = 11
DPI = 150


def render_table(
    *,
    headers,
    rows,
    filename,
    output_dir,
    col_widths=None,
    alignments=None,
    fig_width=10,
    cell_pad=1.4,
):
    """Render a styled data table to PNG."""
    n_rows = len(rows)
    n_cols = len(headers)

    if alignments is None:
        alignments = ["center"] * n_cols
    if col_widths is None:
        col_widths = [1.0 / n_cols] * n_cols

    fig_height = max(0.8, 0.45 * (n_rows + 1) + 0.3)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=DPI)
    ax.axis("off")

    table = ax.table(
        cellText=rows,
        colLabels=headers,
        colWidths=col_widths,
        loc="center",
        cellLoc="center",
    )

    table.auto_set_font_size(False)
    table.set_fontsize(FONT_SIZE)
    table.scale(1, cell_pad)

    for col_idx in range(n_cols):
        cell = table[0, col_idx]
        cell.set_facecolor(HEADER_BG)
        cell.set_text_props(color=HEADER_FG, weight="bold")
        cell.set_edgecolor(BORDER_COLOR)

    for row_idx in range(1, n_rows + 1):
        for col_idx in range(n_cols):
            cell = table[row_idx, col_idx]
            cell.set_facecolor(EVEN_ROW_BG if row_idx % 2 == 0 else ODD_ROW_BG)
            cell.set_edgecolor(BORDER_COLOR)
            cell.set_text_props(ha=alignments[col_idx])

    out_path = os.path.join(output_dir, filename)
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    print(f"Saved {out_path}")


def load_tables_module(tables_path):
    spec = importlib.util.spec_from_file_location("article_tables", tables_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description="Render article tables to PNG")
    parser.add_argument(
        "--article",
        required=True,
        help="Article slug, e.g. 2026-04-gpu-inference",
    )
    args = parser.parse_args()

    article_dir = os.path.join("docs", "articles", args.article)
    if not os.path.isdir(article_dir):
        sys.exit(f"Article folder not found: {article_dir}")

    tables_path = os.path.join(article_dir, "tables.py")
    if not os.path.isfile(tables_path):
        sys.exit(f"No tables.py found in {article_dir}")

    output_dir = os.path.join(article_dir, "assets", "tables")
    os.makedirs(output_dir, exist_ok=True)

    module = load_tables_module(tables_path)
    if not hasattr(module, "TABLES"):
        sys.exit(f"{tables_path} must define a top-level TABLES list")

    for spec in module.TABLES:
        render_table(output_dir=output_dir, **spec)


if __name__ == "__main__":
    main()
