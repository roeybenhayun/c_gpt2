#!/bin/bash
# Default workload (the "decode" preset shape):
#   ~13-token prompt, 768 generated tokens — decode-dominated, M ≈ 1 throughout.
INPUT_TEXT="Once upon a time, in a land far, far away, there was a small dragon."
OUT_TOKENS="768"
CHUNK_SIZE="32"
OUTPUT_DIR="logs"
BUILD_DIR="$(cd "$(dirname "$0")/.." && pwd)"
LONG_PROMPT_FILE="${BUILD_DIR}/scripts/prompts/long_prompt.txt"

if [ ! -d "$OUTPUT_DIR" ]; then
    echo "Creating $OUTPUT_DIR directory..."
    mkdir "$OUTPUT_DIR"
fi

# Parse flags
RUN_CPU=false
RUN_GPU=false
RUN_BF16=false
RUN_INT8=false
PROFILE=false
MODELS=()
PROMPT_FILE=""
PRESET=""                       # "" | "decode" | "prefill" | "balanced"
OUT_TOKENS_EXPLICIT=false       # tracks whether --out-tokens was passed by the user

usage() {
    cat <<'EOF' >&2
Usage: run.sh [options...] [size...]

     --balanced            Use the balanced preset (~200 prompt + ~200 output tokens)
     --bf16                Run GPU BF16 build (in default set)
     --cpu                 Run CPU FP32 build (in default set)
     --decode              Use the decode preset (default): ~13 prompt + 768 output tokens
     --gpu                 Run GPU FP32 build (in default set)
 -h, --help                This help text
     --int8                Run GPU INT8 build (opt-in; needs sm_75+ and quant8.bin)
     --out-tokens <N>      Override the preset's output token count
     --prefill             Use the prefill preset: ~1000 prompt + 32 output tokens
     --profile             Run under nsys (GPU runs only)
     --prompt-file <path>  Override the preset's prompt

  size: small | medium | large (default: all three).
  Presets --decode / --prefill / --balanced are mutually exclusive.
  Runners default to --cpu --gpu --bf16 if none specified.
EOF
}

set_preset() {
    if [ -n "$PRESET" ] && [ "$PRESET" != "$1" ]; then
        echo "Error: presets --decode / --prefill / --balanced are mutually exclusive." >&2
        exit 1
    fi
    PRESET="$1"
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --cpu)        RUN_CPU=true; shift ;;
        --gpu)        RUN_GPU=true; shift ;;
        --bf16)       RUN_BF16=true; shift ;;
        --int8)       RUN_INT8=true; shift ;;
        --profile)    PROFILE=true; shift ;;
        --decode)     set_preset decode; shift ;;
        --prefill)    set_preset prefill; shift ;;
        --balanced)   set_preset balanced; shift ;;
        --prompt-file)
            if [ "$#" -lt 2 ]; then
                echo "--prompt-file requires a path argument" >&2
                exit 1
            fi
            PROMPT_FILE="$2"
            shift 2
            ;;
        --out-tokens)
            if [ "$#" -lt 2 ]; then
                echo "--out-tokens requires a value argument" >&2
                exit 1
            fi
            OUT_TOKENS="$2"
            OUT_TOKENS_EXPLICIT=true
            shift 2
            ;;
        --help|-h)    usage; exit 0 ;;
        small|medium|large)
            MODELS+=("$1")
            shift
            ;;
        *)
            echo "Error: Invalid argument '$1'." >&2
            usage
            exit 1
            ;;
    esac
done

# Apply preset defaults. Each preset bakes a (prompt, output count) shape so the
# workload is reproducible. Explicit --prompt-file / --out-tokens override the preset.
PROMPT_BYTES_LIMIT=8000   # default cap (under input_buffer[MAX_PROMPT_BYTES=8192])
case "$PRESET" in
    "" | decode)
        # leave INPUT_TEXT and OUT_TOKENS at the defaults declared above
        :
        ;;
    prefill)
        # Large prompt → big M during prefill. 32 output tokens → small decode tail.
        # English BPE is closer to ~5 chars/token in practice, so ~5120 bytes
        # tokenises to ~1000 prompt tokens (verified empirically against
        # long_prompt.txt: 2048 bytes → 403 tokens).
        : "${PROMPT_FILE:=$LONG_PROMPT_FILE}"
        $OUT_TOKENS_EXPLICIT || OUT_TOKENS=32
        PROMPT_BYTES_LIMIT=5120
        ;;
    balanced)
        # Medium prompt + medium output. ~200 tokens of prompt = ~800 bytes,
        # ~200 output tokens for a roughly 50/50 phase mix on Large.
        : "${PROMPT_FILE:=$LONG_PROMPT_FILE}"
        $OUT_TOKENS_EXPLICIT || OUT_TOKENS=200
        PROMPT_BYTES_LIMIT=800
        ;;
esac

# If a prompt file was selected (by preset or --prompt-file), load + sanitize it.
# Sanitization mirrors prefill_benchmark.sh: strip non-ASCII and JSON-breaking chars
# so the naive snprintf-into-JSON in gpt2.c doesn't produce malformed requests.
if [ -n "$PROMPT_FILE" ]; then
    if [ ! -f "$PROMPT_FILE" ]; then
        echo "Prompt file not found: $PROMPT_FILE" >&2
        exit 1
    fi
    INPUT_TEXT=$(tr -cd '\40-\176' < "$PROMPT_FILE" | tr -d '"\\' | tr -s ' ')
    INPUT_TEXT="${INPUT_TEXT:0:$PROMPT_BYTES_LIMIT}"
    if [ -z "$INPUT_TEXT" ]; then
        echo "Sanitized prompt from $PROMPT_FILE was empty" >&2
        exit 1
    fi
fi

echo "Preset:      ${PRESET:-decode (default)}"
echo "Prompt:      ${#INPUT_TEXT} bytes  (source: ${PROMPT_FILE:-built-in default})"
echo "Out tokens:  $OUT_TOKENS"

# Default: run cpu + gpu fp32 + gpu bf16 if no runner flag specified.
# INT8 is opt-in only — it's a separate (W8A8) precision path with its own
# quant8.bin weights, so it's not in the default set.
if ! $RUN_CPU && ! $RUN_GPU && ! $RUN_BF16 && ! $RUN_INT8; then
    RUN_CPU=true
    RUN_GPU=true
    RUN_BF16=true
fi

# Default: all models if none specified
if [ ${#MODELS[@]} -eq 0 ]; then
    echo "No specific model size provided. Running ALL models."
    MODELS=("small" "medium" "large")
fi

run_models() {
    local bin_dir="$1"  # binary directory (e.g. "out/cpu", "out/gpu", "out/gpu/bf16")
    local tag="$2"      # tag for output filename
    local profile_ok="$3"  # "yes" if profiling is allowed for this mode

    # Embed the preset name in every output filename so the analyzer can
    # discover one regime at a time. Default (no preset given) maps to "decode".
    local preset_tag="${PRESET:-decode}"

    for size in "${MODELS[@]}"; do
        local model="gpt2_${size}"
        TIMESTAMP=$(date +'%Y%m%d_%H%M%S')
        output_file="${OUTPUT_DIR}/${model}_${tag}_${preset_tag}_req_out_tokens_${OUT_TOKENS}_token_chunk_size_${CHUNK_SIZE}_ts_${TIMESTAMP}.json"
        echo "Running $model ($tag, ${preset_tag})..."
        echo "Outputting to $output_file"

        local CMD="./${bin_dir}/$model --prompt \"$INPUT_TEXT\" \
                    --req_out_tokens \"$OUT_TOKENS\" \
                    --token_chunk_size \"$CHUNK_SIZE\" \
                    --json_out_file \"$output_file\" \
                    --no-stream --verbose"

        if $PROFILE && [ "$profile_ok" = "yes" ]; then
            local nsys_out="${OUTPUT_DIR}/${model}_${tag}_${preset_tag}_profile_${TIMESTAMP}"
            echo "Profiling with nsys → ${nsys_out}.nsys-rep"
            eval nsys profile --stats=true -o "$nsys_out" $CMD
        else
            eval $CMD
        fi
    done
}

cd "$BUILD_DIR"

# ───────── CPU build & run ─────────
if $RUN_CPU; then
    echo "========================================="
    echo "  Building CPU binaries"
    echo "========================================="
    make "${MODELS[@]}"
    if [ $? -ne 0 ]; then
        echo "CPU build failed!"
        exit 1
    fi

    echo "========================================="
    echo "  Running CPU inference"
    echo "========================================="
    run_models out/cpu cpu no
fi

# ───────── GPU FP32 build & run ─────────
if $RUN_GPU; then
    echo "========================================="
    echo "  Building GPU (fp32) binaries"
    echo "========================================="
    make gpu "${MODELS[@]}"
    if [ $? -ne 0 ]; then
        echo "GPU build failed!"
        exit 1
    fi

    echo "========================================="
    echo "  Running GPU (fp32) inference"
    echo "========================================="
    run_models out/gpu gpu yes
fi

# ───────── GPU BF16 build & run ─────────
if $RUN_BF16; then
    echo "========================================="
    echo "  Building GPU (bf16) binaries"
    echo "========================================="
    make gpu bf16 "${MODELS[@]}"
    if [ $? -ne 0 ]; then
        echo "GPU bf16 build failed!"
        exit 1
    fi

    echo "========================================="
    echo "  Running GPU (bf16) inference"
    echo "========================================="
    run_models out/gpu/bf16 bf16 yes
fi

# ───────── GPU INT8 build & run ─────────
if $RUN_INT8; then
    echo "========================================="
    echo "  Building GPU (int8) binaries"
    echo "========================================="
    make gpu int8 "${MODELS[@]}"
    if [ $? -ne 0 ]; then
        echo "GPU int8 build failed!"
        exit 1
    fi

    echo "========================================="
    echo "  Running GPU (int8) inference"
    echo "========================================="
    run_models out/gpu/int8 int8 yes
fi

echo "========================================="
echo "  Runs completed. (CPU=$RUN_CPU GPU=$RUN_GPU BF16=$RUN_BF16 INT8=$RUN_INT8)"
echo "========================================="
