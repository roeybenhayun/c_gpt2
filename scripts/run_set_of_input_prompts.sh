#!/bin/bash
# Set output token limit to 40

INPUT_TEXTS=(
    "Isaac Newton is famous for his work on"
    "Once upon a time, in a land far, far away, there was a small dragon."
    "The best way to learn a new programming language is to"
    "Person A: What did you do yesterday?\nPerson B: I went to the park and then"
)
OUT_TOKENS="40"
CHUNK_SIZE="32"
OUTPUT_DIR="logs"





# Model list
MODELS=("gpt2_small" "gpt2_medium" "gpt2_large")

for i in "${!INPUT_TEXTS[@]}"; do
    prompt="${INPUT_TEXTS[$i]}"
    for model in "${MODELS[@]}"; do
        TIMESTAMP=$(date +'%Y%m%d_%H%M%S')
        echo "Running $model with prompt #$((i + 1))..."
        output_file="${OUTPUT_DIR}/${model}_req_out_tokens_${OUT_TOKENS}_token_chunk_size_${CHUNK_SIZE}_ts_${TIMESTAMP}.json"        
        ./out/$model --prompt "$prompt" \
                --req_out_tokens "$OUT_TOKENS" \
                --token_chunk_size "$CHUNK_SIZE" \
                --json_out_file "$output_file"
    done
done
