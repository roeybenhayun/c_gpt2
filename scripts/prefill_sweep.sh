#!/bin/bash
# Sweep prompt lengths to compare FP32 vs BF16 prefill (TTFT).
#
# Goal: validate that BF16 wins on M > 1 (prefill) where tensor cores engage,
# even though it doesn't help in the M = 1 generation path.
#
# Method: feed prompts of increasing length, request only 1 output token, and
# read `ttft_s` from the JSON log — that field is pure prefill time.
#
# Usage:
#   ./scripts/prefill_sweep.sh                          # large, default sizes
#   ./scripts/prefill_sweep.sh medium                   # different model
#   ./scripts/prefill_sweep.sh large path/to/text.md    # custom source text
#   ./scripts/prefill_sweep.sh --profile                # also nsys-profile target 512
#   ./scripts/prefill_sweep.sh --profile --profile-target 1000 large

set -e

MODEL="large"
SOURCE_TEXT="docs/articles/2026-04-gpu-inference/article.md"
PROFILE=false
PROFILE_TARGET="${PREFILL_PROFILE_TARGET:-512}"

POSITIONAL=()
while [ "$#" -gt 0 ]; do
    case "$1" in
        --profile)
            PROFILE=true
            shift
            ;;
        --profile-target)
            if [ "$#" -lt 2 ]; then
                echo "--profile-target requires a value" >&2
                exit 1
            fi
            PROFILE_TARGET="$2"
            shift 2
            ;;
        --help|-h)
            sed -n '1,18p' "$0"
            exit 0
            ;;
        --*)
            echo "Unknown option: $1" >&2
            exit 1
            ;;
        *)
            POSITIONAL+=("$1")
            shift
            ;;
    esac
done

if [ "${#POSITIONAL[@]}" -gt 0 ]; then
    MODEL="${POSITIONAL[0]}"
fi
if [ "${#POSITIONAL[@]}" -gt 1 ]; then
    SOURCE_TEXT="${POSITIONAL[1]}"
fi
if [ "${#POSITIONAL[@]}" -gt 2 ]; then
    echo "Too many positional arguments: ${POSITIONAL[*]}" >&2
    exit 1
fi

# Approximate target prompt token counts. English averages ~4 chars/token, so
# the script slices the source file by byte count to hit each target.
# input_buffer is 8192 bytes (gpt2.c MAX_PROMPT_BYTES) and ctx_len is 1024,
# so the largest meaningful target is ~1000 tokens.
PROMPT_TOKEN_TARGETS=(32 128 512 1000)
BYTES_PER_TOKEN=4

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"

OUTPUT_DIR="logs/prefill_sweep"
mkdir -p "$OUTPUT_DIR"

# Pre-flight checks
if [ ! -f "$SOURCE_TEXT" ]; then
    echo "Source text not found: $SOURCE_TEXT" >&2
    echo "Pass a path as the 2nd arg: $0 $MODEL <path-to-long-text>" >&2
    exit 1
fi
for bin in "out/gpu/gpt2_$MODEL" "out/gpu/bf16/gpt2_$MODEL"; do
    if [ ! -x "$bin" ]; then
        echo "Binary not found: $bin" >&2
        echo "Build with: make gpu $MODEL && make gpu bf16 $MODEL" >&2
        exit 1
    fi
done
if ! pgrep -f "tokenizer.py" > /dev/null; then
    echo "Tokenizer server not running. Start with: uv run python tokenizer.py" >&2
    exit 1
fi
if ! command -v jq > /dev/null; then
    echo "jq is required for JSON parsing." >&2
    exit 1
fi
if $PROFILE && ! command -v nsys > /dev/null; then
    echo "nsys is required for --profile." >&2
    exit 1
fi

# Strip everything that could break the naive snprintf-into-JSON in gpt2.c
# OR confuse tokenizer.py when truncated:
#   - non-ASCII bytes (em-dashes / smart quotes split mid-codepoint = invalid UTF-8 → tokenizer hangs)
#   - control chars and tabs/newlines
#   - double quotes and backslashes (would unbalance the JSON string)
# Then collapse whitespace runs.
SOURCE_CONTENT=$(tr -cd '\40-\176' < "$SOURCE_TEXT" | tr -d '"\\' | tr -s ' ')
SOURCE_BYTES=${#SOURCE_CONTENT}

echo "Model:       gpt2_$MODEL"
echo "Source text: $SOURCE_TEXT ($SOURCE_BYTES bytes)"
if $PROFILE; then
    echo "Profile:     enabled for target ~$PROFILE_TARGET tokens"
fi
echo

# Header
printf "%-12s  %-12s  %-12s  %-12s  %-9s\n" \
    "actual_tok" "FP32 TTFT" "BF16 TTFT" "Δ TTFT" "speedup"
printf "%-12s  %-12s  %-12s  %-12s  %-9s\n" \
    "----------" "---------" "---------" "------" "-------"

run_one() {
    local bin="$1"
    local prompt="$2"
    local out_json="$3"
    local profile_prefix="${4:-}"
    local -a cmd=(
        "./$bin"
        --prompt "$prompt"
        --req_out_tokens 1
        --json_out_file "$out_json"
        --no-stream
    )

    if [ -n "$profile_prefix" ]; then
        echo "    profiling $bin -> ${profile_prefix}.nsys-rep" >&2
        cmd=(nsys profile --stats=true -f true -o "$profile_prefix" "${cmd[@]}")
    fi

    if [ -n "$profile_prefix" ]; then
        # Profiling adds startup/report overhead; keep a separate log with nsys
        # stats and any profiler errors.
        if timeout 300 "${cmd[@]}" > "${profile_prefix}.log" 2>&1; then
            return 0
        fi
    else
        # 120 s timeout — well above any legitimate prefill time on any model size.
        # If it hits this, something's wrong (e.g. tokenizer not responding).
        if timeout 120 "${cmd[@]}" > /dev/null 2>&1; then
            return 0
        fi
    fi

    local status=$?
    local timeout_s=120
    if [ -n "$profile_prefix" ]; then
        timeout_s=300
    fi
    if [ "$status" -eq 124 ]; then
        echo "  ! $bin timed out (>${timeout_s}s)" >&2
    else
        echo "  ! $bin failed (exit $status)" >&2
    fi
    if [ -f "$out_json" ]; then
        echo "    JSON log was still written: $out_json" >&2
    fi
    if [ -n "$profile_prefix" ] && [ -f "${profile_prefix}.log" ]; then
        echo "    nsys log: ${profile_prefix}.log" >&2
    fi
    return "$status"
}

for target in "${PROMPT_TOKEN_TARGETS[@]}"; do
    bytes=$((target * BYTES_PER_TOKEN))
    if [ "$bytes" -gt "$SOURCE_BYTES" ]; then
        echo "Skipping $target tokens (~$bytes bytes) — source text only $SOURCE_BYTES bytes" >&2
        continue
    fi
    PROMPT="${SOURCE_CONTENT:0:$bytes}"
    echo "  → target ~$target tokens (${bytes} bytes): \"${PROMPT:0:60}…\"" >&2

    TS=$(date +'%Y%m%d_%H%M%S')
    fp32_log="$OUTPUT_DIR/fp32_${MODEL}_${target}_${TS}.json"
    bf16_log="$OUTPUT_DIR/bf16_${MODEL}_${target}_${TS}.json"
    fp32_profile=""
    bf16_profile=""

    if $PROFILE && [ "$target" = "$PROFILE_TARGET" ]; then
        fp32_profile="$OUTPUT_DIR/fp32_${MODEL}_${target}_${TS}_profile"
        bf16_profile="$OUTPUT_DIR/bf16_${MODEL}_${target}_${TS}_profile"
    fi

    run_one "out/gpu/gpt2_$MODEL"        "$PROMPT" "$fp32_log" "$fp32_profile"
    run_one "out/gpu/bf16/gpt2_$MODEL"   "$PROMPT" "$bf16_log" "$bf16_profile"

    fp32_ttft=$(jq -r '.ttft_s' "$fp32_log")
    bf16_ttft=$(jq -r '.ttft_s' "$bf16_log")
    actual=$(jq -r '.initial_prompt_len' "$fp32_log")

    printf "%-12s  %-12s  %-12s  %-12s  %-9s\n" \
        "$actual" \
        "$(awk -v v="$fp32_ttft" 'BEGIN { printf "%.4fs", v }')" \
        "$(awk -v v="$bf16_ttft" 'BEGIN { printf "%.4fs", v }')" \
        "$(awk -v a="$fp32_ttft" -v b="$bf16_ttft" 'BEGIN { printf "%+.4fs", b - a }')" \
        "$(awk -v a="$fp32_ttft" -v b="$bf16_ttft" 'BEGIN { if (b > 0) printf "%.2fx", a/b; else print "—" }')"
done

echo
echo "Logs saved under $OUTPUT_DIR/"
if $PROFILE; then
    echo "Profiles saved under $OUTPUT_DIR/ for target ~$PROFILE_TARGET tokens."
fi
