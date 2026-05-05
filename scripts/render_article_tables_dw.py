#!/usr/bin/env python3
"""Render article tables via the Datawrapper API.

Reads `tables.py` (same format as scripts/render_article_tables.py) and creates
or updates a Datawrapper chart per entry. Each chart is published, its
iframe embed code is written to `assets/tables/embeds.md`, and a PNG snapshot
is downloaded to `assets/tables/<filename>` (same paths as the matplotlib
fallback renderer, so existing article references keep working).

Auth: requires the `DW_API_KEY` env var (Datawrapper API token with
`chart:read chart:write folder:read folder:write` scopes).

Idempotency: chart IDs are persisted in `assets/tables/.dw_chart_map.json`
so re-running the script updates existing charts instead of spawning new
ones. Delete the map file if you switch Datawrapper accounts (the old IDs
won't belong to the new account).

Watermark: free Datawrapper accounts include "Created with Datawrapper"
attribution on published charts. The user accepts this.

Usage:
    # one-time: ensure DW_API_KEY is in your shell env
    export DW_API_KEY=...      # or put it in /etc/environment

    uv run python scripts/render_article_tables_dw.py \\
        --article 2026-05-fp32-to-bf16-gpu

    # dry-run: parse tables.py, print what would be created/updated, no API calls
    uv run python scripts/render_article_tables_dw.py \\
        --article 2026-05-fp32-to-bf16-gpu --dry-run
"""

import argparse
import csv
import importlib.util
import io
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request

DW_BASE = "https://api.datawrapper.de"
DW_CHART_TYPE = "tables"
CHART_MAP_FILENAME = ".dw_chart_map.json"
EMBED_OUTPUT_FILENAME = "embeds.md"
ENV_VAR_NAME = "DW_API_KEY"


# ──────────────────────── token discovery (no .env file) ────────────────────

def _get_api_key():
    """Find DW_API_KEY by sourcing the user's bashrc.

    Token is expected to live in ~/.bashrc as `export DW_API_KEY=...`. The
    script may be launched from a long-lived parent process (Claude Code, IDE)
    whose env predates the bashrc edit, so we always spawn a fresh login
    interactive bash to source rc files and read the current value.

    Returns the token (str) or None if not found."""
    # First the cheap path: env var already inherited (works when shell propagated).
    key = os.environ.get(ENV_VAR_NAME)
    if key:
        return key

    # Otherwise: spawn a login interactive bash that sources ~/.bashrc.
    try:
        result = subprocess.run(
            ["bash", "-lic", f"echo -n ${ENV_VAR_NAME}"],
            capture_output=True, text=True, timeout=5,
        )
        candidate = result.stdout.strip()
        if candidate:
            return candidate
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        pass

    return None


# ───────────────────────── Datawrapper API client ──────────────────────────

class DatawrapperError(RuntimeError):
    pass


def _request(method, path, *, token, data=None, content_type=None):
    """Single HTTP call to the Datawrapper API.

    Returns parsed JSON for application/json responses, raw bytes otherwise.
    Retries once on 429 (rate limit) with a short backoff."""
    headers = {"Authorization": f"Bearer {token}"}
    body = None
    if data is not None:
        if isinstance(data, (dict, list)):
            body = json.dumps(data).encode("utf-8")
            headers["Content-Type"] = "application/json"
        elif isinstance(data, str):
            body = data.encode("utf-8")
            headers["Content-Type"] = content_type or "text/plain"
        else:
            body = data
            headers["Content-Type"] = content_type or "application/octet-stream"

    url = f"{DW_BASE}{path}"
    req = urllib.request.Request(url, method=method, data=body, headers=headers)

    for attempt in range(2):
        try:
            with urllib.request.urlopen(req) as resp:
                raw = resp.read()
                ct = resp.headers.get("Content-Type", "")
                if "application/json" in ct:
                    return json.loads(raw) if raw else None
                return raw
        except urllib.error.HTTPError as e:
            if e.code == 429 and attempt == 0:
                time.sleep(2)
                continue
            try:
                err_body = e.read().decode("utf-8")
            except Exception:
                err_body = ""
            raise DatawrapperError(f"{method} {path} → {e.code}: {err_body}") from e


def create_chart(title, *, token):
    return _request(
        "POST", "/v3/charts", token=token,
        data={"title": title, "type": DW_CHART_TYPE},
    )


def update_chart_meta(chart_id, *, token, title=None, metadata=None):
    # Lock chart type to "tables" on every update. Without this,
    # Datawrapper's auto-detection sometimes flips a table to
    # `d3-bars-stacked` when it sees "1 text + 2 numeric" columns
    # (observed with the h100-amdahl-kernel-fraction shape), which makes
    # the rendered embed/PNG show only the footer. Explicitly pinning the
    # type keeps it stable across re-runs.
    body = {"type": DW_CHART_TYPE}
    if title is not None:
        body["title"] = title
    if metadata is not None:
        body["metadata"] = metadata
    return _request("PATCH", f"/v3/charts/{chart_id}", token=token, data=body)


def upload_csv(chart_id, csv_text, *, token):
    return _request(
        "PUT", f"/v3/charts/{chart_id}/data", token=token,
        data=csv_text, content_type="text/csv",
    )


def publish_chart(chart_id, *, token):
    return _request("POST", f"/v3/charts/{chart_id}/publish", token=token)


def get_chart(chart_id, *, token):
    return _request("GET", f"/v3/charts/{chart_id}", token=token)


def export_png(chart_id, *, token, width=800, border=40, zoom=4):
    # plain=true strips the header (title + description) and footer from the
    # exported image, matching the title-suppressed iframe behavior.
    # borderWidth adds padding around the visualization so cell text doesn't
    # sit flush against the image edge.
    # zoom=4 produces a ~3200px-wide PNG (vs the default zoom=2 which gives
    # ~1600px) — keeps the table sharp on retina/4K screens and when readers
    # zoom in. File size grows ~3-4× but stays small per table (a few hundred KB).
    return _request(
        "GET",
        f"/v3/charts/{chart_id}/export/png?width={width}"
        f"&plain=true&borderWidth={border}&zoom={zoom}",
        token=token,
    )


# ───────────────────────── tables.py → Datawrapper ──────────────────────────

def _escape_md(text):
    """Escape markdown special chars that the user didn't intend.

    With `metadata.visualize.markdown: True`, Datawrapper interprets `_word_`
    as italic and would mangle identifiers like `add_bias` or `concat_heads`
    into "add*bias*" / "concat*heads*". Pre-escaping `_` to `\\_` keeps
    intentional `**bold**` markers working while preserving identifier
    text in the rendered output."""
    return str(text).replace("_", r"\_")


def _build_csv(headers, rows):
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow([_escape_md(h) for h in headers])
    for row in rows:
        writer.writerow([_escape_md(c) for c in row])
    return buf.getvalue()


def _build_metadata(headers, alignments, col_widths):
    """Translate the simple alignments/col_widths from tables.py into
    Datawrapper's per-column metadata structure.

    Datawrapper expects column keys to match header names. For tables charts,
    the visualize.columns map controls per-column type, alignment, and width
    (as a fraction of the chart width)."""
    columns = {}
    for header, align, width in zip(headers, alignments, col_widths):
        columns[str(header)] = {
            "align": align,            # 'left' / 'center' / 'right'
            "width": float(width),     # fraction of total width
            "type": "auto",            # let Datawrapper sniff number vs text
            "format": "0,0.[00]",      # nice number formatting where applicable
        }
    return {
        "visualize": {
            "columns": columns,
            "search": False,
            "pagination": {"enabled": False},
            "header": {"borderTop": "2px", "borderBottom": "1px", "style": "bold"},
            "rows": {"compact": False, "stripes": True},
            # Enable markdown parsing in cells so `**bold**` and `*italic*`
            # markers in tables.py rows render as actual bold/italic text.
            # The Datawrapper UI calls this "Parse markdown".
            "markdown": True,
        },
        "describe": {
            "intro": "",
            "byline": "",
            "source-name": "",
            "source-url": "",
        },
        # Title suppression is handled by setting the chart's `title` field to
        # an empty string and using `plain=true` on PNG export — Datawrapper's
        # "show-title" metadata flags aren't honored consistently across chart
        # types, so we don't rely on them here.
    }


# ───────────────────────── chart-id persistence ─────────────────────────────

def _load_chart_map(path):
    if not os.path.isfile(path):
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _save_chart_map(path, mapping):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(mapping, f, indent=2, sort_keys=True)
        f.write("\n")


# ─────────────────────────── tables.py loader ───────────────────────────────

def _load_tables_module(tables_path):
    spec = importlib.util.spec_from_file_location("article_tables", tables_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ───────────────────────────────── main ─────────────────────────────────────

def _prefixed_filename(filename, index, total):
    """Prepend a zero-padded NN- prefix so PNGs sort by article-appearance
    order in any file browser, making bulk-uploads to Substack easy.
    Width of the prefix scales with the table count (2 digits up to 99)."""
    width = max(2, len(str(total)))
    return f"{index + 1:0{width}d}-{filename}"


def render_one_table(spec, *, token, output_dir, chart_map, dry_run, index, total):
    """Create or update one Datawrapper chart for a tables.py entry.

    Returns (chart_id, embed_url, prefixed_filename) or None on failure
    (logged but non-fatal)."""
    filename = spec["filename"]
    prefixed = _prefixed_filename(filename, index, total)
    # Title is intentionally empty: the article already has a section heading
    # above each embedded table, so the chart-level title would be redundant
    # both in the iframe embed (which renders chart.title) and in the PNG.
    # Charts will appear as "Untitled" in the Datawrapper dashboard;
    # `.dw_chart_map.json` maps each filename → chart_id so they remain
    # findable from this side.
    title = ""
    headers = spec["headers"]
    rows = spec["rows"]
    alignments = spec.get("alignments") or ["center"] * len(headers)
    col_widths = spec.get("col_widths") or [1.0 / len(headers)] * len(headers)

    csv_text = _build_csv(headers, rows)
    metadata = _build_metadata(headers, alignments, col_widths)

    if dry_run:
        existing = chart_map.get(filename)
        action = "UPDATE" if existing else "CREATE"
        print(f"  [dry-run] {action} {prefixed!r} (chart_id={existing!r})")
        return None

    chart_id = chart_map.get(filename)
    try:
        if chart_id is None:
            created = create_chart(title, token=token)
            chart_id = created["id"]
            chart_map[filename] = chart_id
            print(f"  + Created chart {chart_id} for {filename}")
        else:
            print(f"  ~ Updating chart {chart_id} for {filename}")

        upload_csv(chart_id, csv_text, token=token)
        update_chart_meta(chart_id, token=token, title=title, metadata=metadata)
        publish_chart(chart_id, token=token)

        # Pull the chart back to grab the published embed URL.
        chart = get_chart(chart_id, token=token)
        public_url = chart.get("publicUrl") or (
            chart.get("publicVersion")
            and f"https://datawrapper.dwcdn.net/{chart_id}/{chart['publicVersion']}/"
        ) or f"https://datawrapper.dwcdn.net/{chart_id}/"

        # PNG snapshot — prefixed filename so file browsers sort in
        # article-appearance order (handy for Substack bulk-upload).
        png_bytes = export_png(chart_id, token=token, width=800)
        png_path = os.path.join(output_dir, prefixed)
        with open(png_path, "wb") as f:
            f.write(png_bytes)
        print(f"    PNG  → {png_path}")
        print(f"    URL  → {public_url}")
        return chart_id, public_url, prefixed
    except DatawrapperError as e:
        print(f"  ! Failed to render {filename}: {e}", file=sys.stderr)
        return None


def write_embed_index(embeds, output_path):
    """One-stop reference for copy-pasting iframe URLs into Substack."""
    lines = [
        "# Datawrapper embed URLs",
        "",
        "Generated by `scripts/render_article_tables_dw.py`. For each table, copy the",
        "URL into Substack via the `Embed → Datawrapper` block, or paste the iframe",
        "snippet directly into a custom HTML block.",
        "",
    ]
    for filename, url in embeds:
        lines.append(f"## `{filename}`")
        lines.append("")
        lines.append(f"- Embed URL: <{url}>")
        lines.append(f"- iframe: `<iframe src=\"{url}\" scrolling=\"no\" frameborder=\"0\"></iframe>`")
        lines.append("")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--article", required=True, help="Article slug, e.g. 2026-05-fp32-to-bf16-gpu")
    parser.add_argument("--dry-run", action="store_true", help="Parse tables.py and print what would be sent, but make no API calls")
    args = parser.parse_args()

    token = _get_api_key()
    if not token and not args.dry_run:
        sys.exit(
            f"{ENV_VAR_NAME} not found in env or in ~/.bashrc.\n"
            f"  Add `export {ENV_VAR_NAME}=...` to ~/.bashrc, or pass --dry-run.\n"
            f"  Generate a token at app.datawrapper.de/account/api-tokens with "
            f"scopes: chart:read chart:write folder:read folder:write"
        )

    article_dir = os.path.join("docs", "articles", args.article)
    if not os.path.isdir(article_dir):
        sys.exit(f"Article folder not found: {article_dir}")

    tables_path = os.path.join(article_dir, "tables.py")
    if not os.path.isfile(tables_path):
        sys.exit(f"No tables.py found in {article_dir}")

    output_dir = os.path.join(article_dir, "assets", "tables")
    os.makedirs(output_dir, exist_ok=True)
    map_path = os.path.join(output_dir, CHART_MAP_FILENAME)
    embed_path = os.path.join(output_dir, EMBED_OUTPUT_FILENAME)

    module = _load_tables_module(tables_path)
    if not hasattr(module, "TABLES"):
        sys.exit(f"{tables_path} must define a top-level TABLES list")

    chart_map = _load_chart_map(map_path)
    embeds = []

    total = len(module.TABLES)
    print(f"Rendering {total} table(s) from {tables_path}")
    for index, spec in enumerate(module.TABLES):
        result = render_one_table(
            spec, token=token, output_dir=output_dir,
            chart_map=chart_map, dry_run=args.dry_run,
            index=index, total=total,
        )
        if result is not None:
            _, url, prefixed = result
            embeds.append((prefixed, url))

    if not args.dry_run:
        _save_chart_map(map_path, chart_map)
        if embeds:
            write_embed_index(embeds, embed_path)
            print(f"\nEmbed index → {embed_path}")
            print(f"Chart-id map → {map_path}")


if __name__ == "__main__":
    main()
