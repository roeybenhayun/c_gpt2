#!/bin/bash
INPUT_TEXT="Once upon a time, in a land far, far away, there was a small dragon."

OUTPUT_DIR="logs"
if [ ! -d "$OUTPUT_DIR" ]; then
    echo "Creating $OUTPUT_DIR directory..."
    mkdir "$OUTPUT_DIR"
fi

# Model list
MODELS=("gpt2_small" "gpt2_medium" "gpt2_large")

for model in "${MODELS[@]}"; do
    echo "Running $model..."
    ./out/$model --prompt "$INPUT_TEXT" > "$OUTPUT_DIR/${model}.txt"
    echo "Output saved to $OUTPUT_DIR/${model}.txt"
done