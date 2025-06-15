
# Setup 

## Python

### Create python env
* python3.9 -m venv transformers_env
### Activate it
* source transformers_env/bin/activate
### Install required python dependencies
* pip install -r requirements.txt

### Note 
Due to python packages dep miss match importing/loading to cache the transformer model failed. 
So solution was to download the model and other model files 
directly and use them

## Download GPT2.C weights
Download to the repo root directory
### Small
https://huggingface.co/roeybh/gpt2-small-from-scratch-c/resolve/main/gpt2_c_weights.bin

### Medium
https://huggingface.co/roeybh/gpt2-small-from-scratch-c/resolve/main/gpt2_medium_c_weights.bin

### Large


### Create GPT2.C weights
Or you can download the following files and use extract_weights script to create 
gpt2_c_weights.bin using extract_weights.py

There is no need to clone the transformer repo
you'll just need to download the GPT2 model fils and place them in "transformers/models/gpt2" directory

This file holds the weights (for GPT2 small)
wget https://huggingface.co/gpt2/resolve/main/pytorch_model.bin 

wget https://huggingface.co/gpt2/resolve/main/merges.txt

wget https://huggingface.co/gpt2/resolve/main/vocab.json

wget https://huggingface.co/gpt2/resolve/main/tokenizer_config.json

wget https://huggingface.co/gpt2/resolve/main/config.json


To get GPT2 medium weights use the following urls. Save those fils into "transformers/models/gpt2-medium"

wget https://huggingface.co/gpt2-medium/resolve/main/pytorch_model.bin

wget https://huggingface.co/gpt2-medium/resolve/main/merges.txt

wget https://huggingface.co/gpt2-medium/resolve/main/vocab.json

wget https://huggingface.co/gpt2-medium/resolve/main/tokenizer_config.json

wget https://huggingface.co/gpt2-medium/resolve/main/config.json

To get GPT2 medium weights use the following urls. Save those fils into "transformers/models/gpt2-medium"

## Compiler
* gcc -v
* Apple clang version 16.0.0 (clang-1600.0.26.6)
* Target: arm64-apple-darwin23.6.0
* Thread model: posix


### install ARM brew 
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
### install jansson (for now the code dependes on external json lib)
/opt/homebrew/bin/brew install jansson

### How to build?
gcc -Wall -I/opt/homebrew/include -L/opt/homebrew/lib -ljansson -O3 -DUSE_ACCELERATE -DACCELERATE_NEW_LAPACK -framework Accelerate gpt2.c -o ./out/gpt2

#### Activate venv:
* source transformers_env/bin/activate

#### Start the python Tokenizer server
* python tokenizer.py

#### Run GPT2
* ./out/gpt2.c
