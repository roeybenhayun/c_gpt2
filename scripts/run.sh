#!/bin/bash
INPUT_TEXT="Once upon a time, in a land far, far away, there was a small dragon."
# for performance I used out_token=768
OUT_TOKENS="768"
CHUNK_SIZE="32"
OUTPUT_DIR="logs"
BUILD_DIR="$(cd "$(dirname "$0")/.." && pwd)"

if [ ! -d "$OUTPUT_DIR" ]; then
    echo "Creating $OUTPUT_DIR directory..."
    mkdir "$OUTPUT_DIR"
fi

# Parse flags
RUN_CPU=false
RUN_GPU=false
PROFILE=false
MODELS=()

for arg in "$@"; do
    case $arg in
        --cpu)
            RUN_CPU=true
            ;;
        --gpu)
            RUN_GPU=true
            ;;
        --profile)
            PROFILE=true
            ;;
        small|medium|large)
            MODELS+=("$arg")
            ;;
        *)
            echo "Error: Invalid argument '$arg'."
            echo "Usage: ./run.sh [--cpu] [--gpu] [--profile] [small] [medium] [large]"
            exit 1
            ;;
    esac
done

# Default: run both if neither flag specified
if ! $RUN_CPU && ! $RUN_GPU; then
    RUN_CPU=true
    RUN_GPU=true
fi

# Default: all models if none specified
if [ ${#MODELS[@]} -eq 0 ]; then
    echo "No specific model size provided. Running ALL models."
    MODELS=("small" "medium" "large")
fi

run_models() {
    local mode="$1"  # "cpu" or "gpu"
    local tag="$2"   # tag for output filename

    for size in "${MODELS[@]}"; do
        local model="gpt2_${size}"
        TIMESTAMP=$(date +'%Y%m%d_%H%M%S')
        output_file="${OUTPUT_DIR}/${model}_${tag}_req_out_tokens_${OUT_TOKENS}_token_chunk_size_${CHUNK_SIZE}_ts_${TIMESTAMP}.json"
        echo "Running $model ($tag)..."
        echo "Outputting to $output_file"

        local bin_dir="./out/${mode}"
        local CMD="./${bin_dir}/$model --prompt \"$INPUT_TEXT\" \
                    --req_out_tokens \"$OUT_TOKENS\" \
                    --token_chunk_size \"$CHUNK_SIZE\" \
                    --json_out_file \"$output_file\" \
                    --no-stream --verbose"

        if $PROFILE && [ "$mode" = "gpu" ]; then
            local nsys_out="${OUTPUT_DIR}/${model}_${tag}_profile_${TIMESTAMP}"
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
    run_models cpu cpu
fi

# ───────── GPU build & run ─────────
if $RUN_GPU; then
    echo "========================================="
    echo "  Building GPU binaries"
    echo "========================================="
    make gpu "${MODELS[@]}"
    if [ $? -ne 0 ]; then
        echo "GPU build failed!"
        exit 1
    fi

    echo "========================================="
    echo "  Running GPU inference"
    echo "========================================="
    run_models gpu gpu
fi

echo "========================================="
echo "  Runs completed. (CPU=$RUN_CPU GPU=$RUN_GPU)"
echo "========================================="
