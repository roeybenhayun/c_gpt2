#!/bin/bash
INPUT_TEXT="Once upon a time, in a land far, far away, there was a small dragon."
# for performance I used out_token=768
OUT_TOKENS="768"
CHUNK_SIZE="32"
OUTPUT_DIR="logs"




if [ ! -d "$OUTPUT_DIR" ]; then
    echo "Creating $OUTPUT_DIR directory..."
    mkdir "$OUTPUT_DIR"
fi

# Model list
MODELS=("gpt2_small" "gpt2_medium" "gpt2_large")

for model in "${MODELS[@]}"; do    
    TIMESTAMP=$(date +'%Y%m%d_%H%M%S')
    output_file="${OUTPUT_DIR}/${model}_req_out_tokens_${OUT_TOKENS}_token_chunk_size_${CHUNK_SIZE}_ts_${TIMESTAMP}.json"
    echo "Running $model..."
    echo "Outputting to $output_file"

    ./out/$model --prompt "$INPUT_TEXT" \
                --req_out_tokens "$OUT_TOKENS" \
                --token_chunk_size "$CHUNK_SIZE" \
                --json_out_file "$output_file"
done

echo "All models processed."