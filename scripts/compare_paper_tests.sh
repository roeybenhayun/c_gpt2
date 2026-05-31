#!/bin/bash
# compare_paper_tests.sh
#
# Diff per-case generated_<target>_<size>.txt files from two precisions
# (default: bf16 vs int8) across one or more model sizes. Annotates each
# row with the case's sampling mode (greedy vs sampled) so byte-divergence
# can be interpreted correctly — greedy diffs are an accuracy signal,
# sampled diffs are RNG drift and expected.
#
# Typical usage (after run_paper_tests.sh has produced both sides):
#   ./scripts/compare_paper_tests.sh                          # bf16 vs int8, large
#   ./scripts/compare_paper_tests.sh small medium large        # all three sizes
#   ./scripts/compare_paper_tests.sh --left gpu --right bf16   # different precision pair
#   ./scripts/compare_paper_tests.sh --md report.md medium     # write markdown report

set -u

BUILD_DIR="$(cd "$(dirname "$0")/.." && pwd)"
TESTS_DIR="${BUILD_DIR}/tests/paper_validation"

LEFT="bf16"
RIGHT="int8"
SIZES=()
FILTER_CATEGORY=""
FILTER_CASE=""
MD_PATH=""

usage() {
    cat <<'EOF' >&2
Usage: compare_paper_tests.sh [options...] [size...]

     --case <substr>      Compare only cases whose dir contains <substr>
     --category <name>    Compare only cases in tests/paper_validation/<name>/
 -h, --help               This help text
     --left <target>      Left  precision (default: bf16). One of cpu|gpu|bf16|int8.
     --md <path>          Also emit a markdown report to <path>
     --right <target>     Right precision (default: int8). One of cpu|gpu|bf16|int8.
     --tests-dir <path>   Override the tests dir (default tests/paper_validation)

  size: small | medium | large (default: large). Pass multiple to compare back-to-back.
  Reads generated_<target>_<size>.txt for each case and reports per-row:
    IDENTICAL | DIFFER (with first-diff byte and per-side byte counts)
  Sampling mode (greedy vs sampled) is read from metadata.json.
EOF
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --left)
            if [ "$#" -lt 2 ]; then echo "--left needs a value" >&2; exit 1; fi
            LEFT="$2"; shift 2 ;;
        --right)
            if [ "$#" -lt 2 ]; then echo "--right needs a value" >&2; exit 1; fi
            RIGHT="$2"; shift 2 ;;
        --category)
            if [ "$#" -lt 2 ]; then echo "--category needs a value" >&2; exit 1; fi
            FILTER_CATEGORY="$2"; shift 2 ;;
        --case)
            if [ "$#" -lt 2 ]; then echo "--case needs a value" >&2; exit 1; fi
            FILTER_CASE="$2"; shift 2 ;;
        --md)
            if [ "$#" -lt 2 ]; then echo "--md needs a value" >&2; exit 1; fi
            MD_PATH="$2"; shift 2 ;;
        --tests-dir)
            if [ "$#" -lt 2 ]; then echo "--tests-dir needs a value" >&2; exit 1; fi
            TESTS_DIR="$2"; shift 2 ;;
        small|medium|large) SIZES+=("$1"); shift ;;
        --help|-h)  usage; exit 0 ;;
        *)          echo "Error: unknown arg '$1'" >&2; usage; exit 1 ;;
    esac
done

if [ ${#SIZES[@]} -eq 0 ]; then
    SIZES=("large")
fi

if [ "$LEFT" = "$RIGHT" ]; then
    echo "Error: --left and --right must differ (got '$LEFT' twice)." >&2
    exit 1
fi

if ! command -v jq >/dev/null 2>&1; then
    echo "Error: jq is required to parse metadata.json." >&2
    exit 1
fi

if [ ! -d "$TESTS_DIR" ]; then
    echo "Tests dir not found: $TESTS_DIR" >&2
    exit 1
fi

# Collect cases (same filter logic as run_paper_tests.sh)
shopt -s nullglob
case_dirs=()
for cat_dir in "$TESTS_DIR"/*/; do
    cat_name=$(basename "$cat_dir")
    if [ -n "$FILTER_CATEGORY" ] && [ "$cat_name" != "$FILTER_CATEGORY" ]; then
        continue
    fi
    for case_dir in "$cat_dir"*/; do
        case_name=$(basename "$case_dir")
        if [ -n "$FILTER_CASE" ] && [[ "$case_name" != *"$FILTER_CASE"* ]]; then
            continue
        fi
        if [ -f "${case_dir}metadata.json" ]; then
            case_dirs+=("${case_dir%/}")
        fi
    done
done

if [ ${#case_dirs[@]} -eq 0 ]; then
    echo "No matching cases found under $TESTS_DIR" >&2
    exit 1
fi

# Sampling-mode classifier: top_k=1 OR temperature=0 → greedy; else sampled.
# (Both flags collapse the sampling distribution to argmax, so either implies
# byte-identical streams across precisions in the absence of quantization noise.)
sampling_mode() {
    local meta="$1"
    local top_k temp
    top_k=$(jq -r '.params.top_k' "$meta")
    temp=$(jq -r '.params.temperature' "$meta")
    if [ "$top_k" = "1" ] || awk -v t="$temp" 'BEGIN { exit !(t == 0) }'; then
        echo "greedy"
    else
        echo "sampled"
    fi
}

# Per-case comparison. Echoes a single line with the result; the caller
# aggregates counters. Also appends a markdown row to MD_TMP if set.
MD_TMP=""
if [ -n "$MD_PATH" ]; then
    MD_TMP=$(mktemp)
fi

echo "Comparing $LEFT vs $RIGHT across sizes: ${SIZES[*]}"
echo "Tests dir: $TESTS_DIR"
echo ""

# Pre-render header
printf "  %-13s %-26s %-8s %-10s %-12s %-12s %-12s\n" \
    "category" "case" "size" "mode" "result" "bytes(L/R)" "1st_diff"
printf "  %-13s %-26s %-8s %-10s %-12s %-12s %-12s\n" \
    "--------" "----" "----" "----" "------" "----------" "--------"

declare -A n_identical n_differ n_missing
n_total=0

for size in "${SIZES[@]}"; do
    n_identical[$size]=0
    n_differ[$size]=0
    n_missing[$size]=0

    for case_dir in "${case_dirs[@]}"; do
        case_name=$(basename "$case_dir")
        cat_name=$(basename "$(dirname "$case_dir")")
        meta="$case_dir/metadata.json"
        mode=$(sampling_mode "$meta")

        left_file="$case_dir/generated_${LEFT}_${size}.txt"
        right_file="$case_dir/generated_${RIGHT}_${size}.txt"

        n_total=$((n_total+1))

        if [ ! -f "$left_file" ] || [ ! -f "$right_file" ]; then
            missing=""
            [ ! -f "$left_file" ]  && missing="${missing}L"
            [ ! -f "$right_file" ] && missing="${missing}R"
            printf "  %-13s %-26s %-8s %-10s %-12s %-12s %-12s\n" \
                "$cat_name" "$case_name" "$size" "$mode" \
                "MISSING($missing)" "-" "-"
            n_missing[$size]=$((n_missing[$size]+1))
            if [ -n "$MD_TMP" ]; then
                printf "| %s | %s | %s | %s | MISSING(%s) | - | - |\n" \
                    "$cat_name" "$case_name" "$size" "$mode" "$missing" >> "$MD_TMP"
            fi
            continue
        fi

        left_bytes=$(wc -c < "$left_file")
        right_bytes=$(wc -c < "$right_file")

        if cmp -s "$left_file" "$right_file"; then
            printf "  %-13s %-26s %-8s %-10s %-12s %-12s %-12s\n" \
                "$cat_name" "$case_name" "$size" "$mode" \
                "IDENTICAL" "$left_bytes/$right_bytes" "-"
            n_identical[$size]=$((n_identical[$size]+1))
            if [ -n "$MD_TMP" ]; then
                printf "| %s | %s | %s | %s | IDENTICAL | %s / %s | - |\n" \
                    "$cat_name" "$case_name" "$size" "$mode" \
                    "$left_bytes" "$right_bytes" >> "$MD_TMP"
            fi
        else
            # cmp prints: <left> <right> differ: byte N, line M
            first_diff=$(cmp "$left_file" "$right_file" 2>/dev/null | awk '{print $5}' | tr -d ',')
            printf "  %-13s %-26s %-8s %-10s %-12s %-12s %-12s\n" \
                "$cat_name" "$case_name" "$size" "$mode" \
                "DIFFER" "$left_bytes/$right_bytes" "byte $first_diff"
            n_differ[$size]=$((n_differ[$size]+1))
            if [ -n "$MD_TMP" ]; then
                printf "| %s | %s | %s | %s | DIFFER | %s / %s | byte %s |\n" \
                    "$cat_name" "$case_name" "$size" "$mode" \
                    "$left_bytes" "$right_bytes" "$first_diff" >> "$MD_TMP"
            fi
        fi
    done
done

echo ""
echo "Summary:"
for size in "${SIZES[@]}"; do
    total=$((n_identical[$size] + n_differ[$size] + n_missing[$size]))
    printf "  %-8s  %d identical, %d differ, %d missing (of %d)\n" \
        "$size" "${n_identical[$size]}" "${n_differ[$size]}" "${n_missing[$size]}" "$total"
done

# Materialize markdown report if requested
if [ -n "$MD_PATH" ]; then
    {
        echo "# Paper-validation comparison: $LEFT vs $RIGHT"
        echo
        echo "Generated by \`scripts/compare_paper_tests.sh\`."
        echo
        echo "- **Left**: \`$LEFT\`"
        echo "- **Right**: \`$RIGHT\`"
        echo "- **Sizes**: ${SIZES[*]}"
        echo
        echo "## How to read"
        echo
        echo "- **greedy** (top_k=1 or T=0): the two precisions should produce"
        echo "  byte-identical token streams unless quantization noise flips an"
        echo "  argmax. Divergence is a real accuracy signal."
        echo "- **sampled**: streams will diverge from RNG drift after the first"
        echo "  logit difference. Byte-equality not expected; check coherence by"
        echo "  reading the per-case \`generated_${LEFT}_<size>.txt\` and"
        echo "  \`generated_${RIGHT}_<size>.txt\` side-by-side."
        echo
        echo "## Results"
        echo
        echo "| Category | Case | Size | Mode | Result | Bytes ($LEFT / $RIGHT) | 1st diff |"
        echo "|---|---|---|---|---|---|---|"
        cat "$MD_TMP"
        echo
        echo "## Summary"
        echo
        echo "| Size | Identical | Differ | Missing |"
        echo "|---|---|---|---|"
        for size in "${SIZES[@]}"; do
            printf "| %s | %d | %d | %d |\n" \
                "$size" "${n_identical[$size]}" "${n_differ[$size]}" "${n_missing[$size]}"
        done
    } > "$MD_PATH"
    rm -f "$MD_TMP"
    echo ""
    echo "Markdown report written to $MD_PATH"
fi
