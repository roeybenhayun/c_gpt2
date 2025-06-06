
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

### Get GPT2 weights
there is no need to clone the transformer repo
you'll just need to download the GPT2 model fils and place them in "transformers/models/gpt2" directory

wget https://huggingface.co/gpt2/resolve/main/pytorch_model.bin 

wget https://huggingface.co/gpt2/resolve/main/merges.txt

wget https://huggingface.co/gpt2/resolve/main/vocab.json

wget https://huggingface.co/gpt2/resolve/main/tokenizer_config.json

wget https://huggingface.co/gpt2/resolve/main/config.json



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

### How to run?
./out/gpt2
#### Activate venv:
* source transformers_env/bin/activate

#### Start the python Tokenizer server
* python tokenizer.py

#### Run GPT2
* ./out/gpt2.c


## To cleanup

train_gpt2- inference is completed --> next token prediction (this is based on Andrej video - link here)
Next, for better understanding I decided to implement GPT2 in C code without any dependencies.
I started with basic building blocks for scaled dot product attention from attention is all you need paper.
dot_product.c - started with basic 1d dot product, then moved to 2d product 
softmax.c - started with 1d vector then moved to 2d softmax operation
scaled_dot_product_attention.c - used above blocks plus layer norm and utils functions (such as mean, variance etc)

I used pytorch to verify each block.
dot_product.py
scaled_dot_product.py


Now let's breakdown the model arch:
GPT2Model - this is the base model without task-specific head attached to to. it outputs hidden states
GPT2LMHeadModel - LMH stands for Language Modeling Head on top of the base model. Output is Logits (next token prediction)


wte - Word Token Embedding 
wpe - Word Position Embedding
drop - Dropout layer
h - hidden layers, total of 12 layers of GPT2Block
    ln_1 - Layer Normalization
    attn - attention block
        c attn - convolution attention - this 
        c proj - convolution 

GPT2LMHeadModel(
  (transformer): GPT2Model(
    (wte): Embedding(50257, 768) // 50257 is the vocab size, 768 is d_model
    (wpe): Embedding(1024, 768)  // 1024 is the context size, 
    (drop): Dropout(p=0.1, inplace=False) // 
    (h): ModuleList(
      (0-11): 12 x GPT2Block(
        (ln_1): LayerNorm((768,), eps=1e-05, elementwise_affine=True) // in the attention is all you need paper this comes after
        (attn): GPT2Attention(
          (c_attn): Conv1D(nf=2304, nx=768)
          (c_proj): Conv1D(nf=768, nx=768)
          (attn_dropout): Dropout(p=0.1, inplace=False)
          (resid_dropout): Dropout(p=0.1, inplace=False)
        )
        (ln_2): LayerNorm((768,), eps=1e-05, elementwise_affine=True) // in the attention is all you need paper this comes after
        (mlp): GPT2MLP(
          (c_fc): Conv1D(nf=3072, nx=768)
          (c_proj): Conv1D(nf=768, nx=3072)
          (act): NewGELUActivation()
          (dropout): Dropout(p=0.1, inplace=False)
        )
      )
    )
    (ln_f): LayerNorm((768,), eps=1e-05, elementwise_affine=True)// additional layer norm added 
  )
  (lm_head): Linear(in_features=768, out_features=50257, bias=False)
)

Differences compared to the original attention is all you need paper
==> No cross attention (since only the decoder is used)
==> Layer norm moved to before Masked Multi head attention and before the FF 
==> Another layer norm added before the Linear layer
