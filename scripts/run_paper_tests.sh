#!/bin/bash
# run_paper_tests.sh
#
# Walk tests/paper_validation/, read each case's metadata.json (top_k,
# temperature, out_tokens, prompt_file), invoke the chosen gpt2 binary once
# per case, and capture stdout/stderr into <case>/generated_<target>_<size>.txt
# for qualitative comparison against the paper's expected.txt.
#
# One gpt2 process per case (clean reset between cases — matches user request).
#
# Tokenizer service is NOT managed by this script; start it separately:
#   uv run python tokenizer.py     (in another shell, before invoking this)

set -u

BUILD_DIR="$(cd "$(dirname "$0")/.." && pwd)"
TESTS_DIR="${BUILD_DIR}/tests/paper_validation"

# Defaults
RUN_TARGET="gpu"
SIZES=()
FILTER_CATEGORY=""
FILTER_CASE=""
DO_BUILD=false
SKIP_TOKENIZER_CHECK=false
DO_CLEAN=false

usage() {
    cat <<'EOF' >&2
Usage: run_paper_tests.sh [options...] [size...]

     --bf16                  Use the GPU BF16 build
     --build                 Force a rebuild before running
     --case <substr>         Run only cases whose dir contains <substr>
     --category <name>       Run only cases in tests/paper_validation/<name>/
     --clean                 Delete per-run artifacts and exit
     --cpu                   Use the CPU build
     --gpu                   Use the GPU FP32 build (default)
 -h, --help                  This help text
     --int8                  Use the GPU INT8 build
     --skip-tokenizer-check  Skip the localhost:65432 pre-flight check
     --tests-dir <path>      Override the tests dir (default tests/paper_validation)

  size: small | medium | large (default: large). Pass multiple to run back-to-back.
  Outputs per case: <case>/{generated_*.txt, run_*.json, run_*.log}.
EOF
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --cpu)      RUN_TARGET="cpu";   shift ;;
        --gpu)      RUN_TARGET="gpu";   shift ;;
        --bf16)     RUN_TARGET="bf16";  shift ;;
        --int8)     RUN_TARGET="int8";  shift ;;
        --category)
            if [ "$#" -lt 2 ]; then echo "--category needs a value" >&2; exit 1; fi
            FILTER_CATEGORY="$2"; shift 2 ;;
        --case)
            if [ "$#" -lt 2 ]; then echo "--case needs a value" >&2; exit 1; fi
            FILTER_CASE="$2"; shift 2 ;;
        --build)    DO_BUILD=true; shift ;;
        --clean)    DO_CLEAN=true; shift ;;
        --tests-dir)
            if [ "$#" -lt 2 ]; then echo "--tests-dir needs a value" >&2; exit 1; fi
            TESTS_DIR="$2"; shift 2 ;;
        --skip-tokenizer-check) SKIP_TOKENIZER_CHECK=true; shift ;;
        small|medium|large) SIZES+=("$1"); shift ;;
        --help|-h)  usage; exit 0 ;;
        *)          echo "Error: unknown arg '$1'" >&2; usage; exit 1 ;;
    esac
done

if [ ${#SIZES[@]} -eq 0 ]; then
    SIZES=("large")
fi

# --clean: scrub per-run artifacts and exit. Runs before any build/tokenizer
# setup so it's cheap to invoke and doesn't depend on tokenizer being up.
# Matches the patterns added to tests/paper_validation/.gitignore.
if $DO_CLEAN; then
    if [ ! -d "$TESTS_DIR" ]; then
        echo "Tests dir not found: $TESTS_DIR" >&2
        exit 1
    fi
    # -print so we get a count of what was deleted; -delete is filesystem-atomic
    # per entry. Patterns match exactly what the harness writes per case.
    n_txt=$(find "$TESTS_DIR" -type f -name 'generated_*.txt' -print -delete | wc -l)
    n_json=$(find "$TESTS_DIR" -type f -name 'run_*_*.json'   -print -delete | wc -l)
    n_log=$(find "$TESTS_DIR" -type f -name 'run_*_*.log'     -print -delete | wc -l)
    echo "Cleaned per-run artifacts under $TESTS_DIR"
    echo "  generated_*.txt : $n_txt removed"
    echo "  run_*_*.json    : $n_json removed"
    echo "  run_*_*.log     : $n_log removed"
    exit 0
fi

# Map target to binary directory and make goals
case "$RUN_TARGET" in
    cpu)   BIN_DIR="out/cpu";      MAKE_GOALS="" ;;
    gpu)   BIN_DIR="out/gpu";      MAKE_GOALS="gpu" ;;
    bf16)  BIN_DIR="out/gpu/bf16"; MAKE_GOALS="gpu bf16" ;;
    int8)  BIN_DIR="out/gpu/int8"; MAKE_GOALS="gpu int8" ;;
esac

cd "$BUILD_DIR"

# Pre-flight: jq is required to parse metadata.json
if ! command -v jq >/dev/null 2>&1; then
    echo "Error: jq is required to parse metadata.json. Install it (apt-get install jq)." >&2
    exit 1
fi

# Pre-flight: tokenizer service must be reachable on port 65432
if ! $SKIP_TOKENIZER_CHECK; then
    if ! (echo > /dev/tcp/127.0.0.1/65432) 2>/dev/null; then
        echo "Error: tokenizer service not reachable on 127.0.0.1:65432." >&2
        echo "Start it in another shell with:  uv run python tokenizer.py" >&2
        echo "(Or pass --skip-tokenizer-check to bypass this check.)" >&2
        exit 1
    fi
fi

# Build any missing binaries
for size in "${SIZES[@]}"; do
    bin="$BIN_DIR/gpt2_$size"
    if $DO_BUILD || [ ! -x "$bin" ]; then
        echo "Building $bin (make $MAKE_GOALS $size) ..."
        # shellcheck disable=SC2086
        make $MAKE_GOALS "$size" || { echo "Build failed for $size" >&2; exit 1; }
    fi
done

# Collect cases matching filters
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
    [ -n "$FILTER_CATEGORY" ] && echo "  (filter --category $FILTER_CATEGORY)" >&2
    [ -n "$FILTER_CASE" ] && echo "  (filter --case $FILTER_CASE)" >&2
    exit 1
fi

echo "Target:    $RUN_TARGET"
echo "Sizes:     ${SIZES[*]}"
echo "Cases:     ${#case_dirs[@]}"
echo "Tests dir: $TESTS_DIR"
echo ""

# Run each (size × case)
n_ok=0
n_fail=0
for size in "${SIZES[@]}"; do
    bin="$BIN_DIR/gpt2_$size"
    for case_dir in "${case_dirs[@]}"; do
        case_name=$(basename "$case_dir")
        cat_name=$(basename "$(dirname "$case_dir")")
        meta="$case_dir/metadata.json"

        prompt_file_name=$(jq -r '.prompt_file' "$meta")
        prompt_file="$case_dir/$prompt_file_name"
        top_k=$(jq -r '.params.top_k' "$meta")
        temperature=$(jq -r '.params.temperature' "$meta")
        out_tokens=$(jq -r '.params.out_tokens' "$meta")
        # gpt2 writes a structured JSON (perf log + clean .generated_text) via
        # --json_out_file. We extract .generated_text into the .txt for the
        # qualitative comparison and keep the JSON alongside for per-case
        # ttft_s / mean_tpot_s / output_tps. --no-stream is on because we read
        # the result from JSON, not stdout — and skipping per-token decode
        # round-trips is a meaningful speedup for the long-decode cases.
        out_text="$case_dir/generated_${RUN_TARGET}_${size}.txt"
        out_json="$case_dir/run_${RUN_TARGET}_${size}.json"
        log_file="$case_dir/run_${RUN_TARGET}_${size}.log"

        if [ ! -f "$prompt_file" ]; then
            echo "[$cat_name/$case_name size=$size]  SKIP — prompt file missing: $prompt_file"
            n_fail=$((n_fail+1))
            continue
        fi

        echo "[$cat_name/$case_name size=$size]  k=$top_k T=$temperature N=$out_tokens"
        # Read the prompt file into a bash var and pass via --prompt. gpt2.c
        # JSON-escapes the argv on the way to the tokenizer, so quotes /
        # newlines / UTF-8 in the prompt are safe over the wire.
        prompt_text=$(<"$prompt_file")
        "./$bin" \
            --prompt        "$prompt_text" \
            --top_k         "$top_k" \
            --temperature   "$temperature" \
            --req_out_tokens "$out_tokens" \
            --json_out_file "$out_json" \
            --no-stream \
            > "$log_file" 2>&1
        ec=$?

        if [ $ec -ne 0 ] || [ ! -f "$out_json" ]; then
            echo "  FAIL (exit $ec) — stdout in $log_file"
            n_fail=$((n_fail+1))
            continue
        fi

        # Extract the clean generated text from the JSON. Falls back to copying
        # the log if jq somehow finds no field (shouldn't happen given gpt2.c
        # always sets generated_text).
        if ! jq -er '.generated_text' "$out_json" > "$out_text" 2>/dev/null; then
            echo "  FAIL — JSON missing .generated_text, see $out_json"
            n_fail=$((n_fail+1))
            continue
        fi

        bytes=$(wc -c < "$out_text")
        tps=$(jq -r '.output_tps | round * 10 / 10' "$out_json" 2>/dev/null)
        echo "  OK   ($bytes bytes, ${tps} tok/s) -> $out_text"
        n_ok=$((n_ok+1))
    done
done

echo ""
echo "Done. ok=$n_ok fail=$n_fail. Captured outputs in <case>/generated_${RUN_TARGET}_<size>.txt"
[ $n_fail -eq 0 ]
