#!/bin/bash
# Set output token limit to 40

INPUT_TEXTS=(
    "Isaac Newton is famous for his work on"
    "Once upon a time, in a land far, far away, there was a small dragon."
    "The best way to learn a new programming language is to"
    "Person A: What did you do yesterday?\nPerson B: I went to the park and then"
)



# Model list
MODELS=("gpt2_small" "gpt2_medium" "gpt2_large")

for i in "${!INPUT_TEXTS[@]}"; do
    prompt="${INPUT_TEXTS[$i]}"
    for model in "${MODELS[@]}"; do
        echo "Running $model with prompt #$((i + 1))..."
        ./out/$model --prompt "$prompt"
    done
done
