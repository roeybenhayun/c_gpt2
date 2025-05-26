
import math
import os
import time
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
#import torch

def load_bin_weights_into_model(model, bin_path):
    with open(bin_path, "rb") as f:
        def read(shape, dtype=np.float32):
            numel = np.prod(shape)
            arr = np.frombuffer(f.read(numel * 4), dtype=dtype).reshape(shape)
            return torch.tensor(arr)

        state_dict = model.state_dict()
        new_sd = {}

        def assign_tensor(name, shape):
            tensor = read(shape)
            new_sd[name] = tensor

        # Embeddings
        assign_tensor("transformer.wte.weight", (50257, 768))  # token embeddings
        assign_tensor("transformer.wpe.weight", (1024, 768))   # position embeddings

        for layer in range(12):
            prefix = f"transformer.h.{layer}"
            assign_tensor(f"{prefix}.ln_1.weight", (768,))
            assign_tensor(f"{prefix}.ln_1.bias", (768,))

            # QKV packed: already transposed on disk → load directly
            assign_tensor(f"{prefix}.attn.c_attn.weight", (2304, 768))
            assign_tensor(f"{prefix}.attn.c_attn.bias", (2304,))

            assign_tensor(f"{prefix}.attn.c_proj.weight", (768, 768))
            assign_tensor(f"{prefix}.attn.c_proj.bias", (768,))

            assign_tensor(f"{prefix}.ln_2.weight", (768,))
            assign_tensor(f"{prefix}.ln_2.bias", (768,))

            assign_tensor(f"{prefix}.mlp.c_fc.weight", (3072, 768))
            assign_tensor(f"{prefix}.mlp.c_fc.bias", (3072,))
            assign_tensor(f"{prefix}.mlp.c_proj.weight", (768, 3072))
            assign_tensor(f"{prefix}.mlp.c_proj.bias", (768,))

        assign_tensor("transformer.ln_f.weight", (768,))
        assign_tensor("transformer.ln_f.bias", (768,))

        # Final head (tied weights)
        new_sd["lm_head.weight"] = new_sd["transformer.wte.weight"]

        model.load_state_dict(new_sd, strict=False)
        #print("PY attn_proj.weight[:10, :10]:")
        #print(model.transformer.h[0].attn.c_proj.weight[:10, :10])

        #print("Python W1[0][:10]:")
        #print(model.transformer.h[0].mlp.c_fc.weight[0, :10].detach().numpy())


@dataclass
class GPTConfig:
    block_size: int = 1024 # max sequence length
    vocab_size: int = 50257 # number of tokens: 50000 BPE merges + 256 bytes tokens + 1 
    n_layer: int = 12 # number of layers 
    n_head: int = 12 # number of attention heads
    n_embd: int = 768 # embedding dimension

class CausalSelfAttention(nn.Module):
    def __init__(self,config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd,3*config.n_embd)
        self.c_proj = nn.Linear(config.n_embd,config.n_embd)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.register_buffer('bias', torch.tril(torch.ones(config.block_size, config.block_size))
                             .view(1, 1, config.block_size, config.block_size))
    def forward(self,x):
        B,T,C = x.size()
        qkv = self.c_attn(x)
        q,k,v = qkv.split(self.n_embd,dim = 2)
        k = k.view(B,T,self.n_head,C // self.n_head).transpose(1,2) # (B,nh,T,hs)
        q = q.view(B,T,self.n_head,C // self.n_head).transpose(1,2) # (B,nh,T,hs)
        v = v.view(B,T,self.n_head,C // self.n_head).transpose(1,2) # (B,nh,T,hs)

        #print("ALPHA == ", 1.0 / math.sqrt(k.size(-1)))
        att = (q @ k.transpose(-2,-1)) * (1.0 / math.sqrt(k.size(-1)))
        print("PY Q[0][0][1][:10]:", q[0,0, 1, :10].detach().numpy())
        print("PY K[0][0][1][:10]:", k[0,0, 1, :10].detach().numpy())
        print("PY V[0][0][1][:10]:", v[0,0, 1, :10].detach().numpy())


        ## autoregresive masking -make sure tokens only attends to previous tokens
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        ## normalize the attention - sum to one always
        att = F.softmax(att, dim=-1)
        
        y = att @ v

        #print(y[0, -1, :10].detach().numpy())
        token_idx_to_check = 1

        print("\nPython context_heads equivalent for token {}:".format(token_idx_to_check))
        for h_idx in range(self.n_head):
            # Extract the slice for the current head, for the specified token
            py_context_h_out_token = y[0, h_idx, token_idx_to_check, :].detach().numpy()
            print(f"  Head {h_idx}: {py_context_h_out_token[:10]}") # Print first 10 elements

        y = y.transpose(1,2).contiguous().view(B,T,C)
        print("Python final_attention_output[1][:10]:")
        print(y[0, 1, :10].detach().numpy())

        # ✅ Add this block here:
        fa_out = y[0, 0, :].detach().numpy()  # the first token's attention output
        attn_proj_w = self.c_proj.weight.detach().numpy()  # [out_features, in_features]
        context_np = np.dot(fa_out, attn_proj_w.T)  # manual projection

        #print("Manual NumPy context[0][:10]:", context_np[:10])  # for comparison with C
        # ✅ Done adding

        #print("PY attn_proj.weight[:10, :10]:")
        #print(self.c_proj.weight[:10, :10].detach().numpy())

        y = self.c_proj(y)
        print("Python context[1][:10]:")
        print(y[0, 1, :10].detach().numpy())
        return y
                                          

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu    = nn.GELU(approximate='tanh')
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        x = self.c_fc(x)
        print("*****x1*****", x[0, 1, :10].detach().numpy())  # Add here
        x = self.gelu(x)
        print("*****x1 after GELU*****", x[0, 1, :10].detach().numpy())  # Add here
        x = self.c_proj(x)
        print("*****x1 after projection*****", x[0, 1, :10].detach().numpy())
        return x
    
class Block(nn.Module):
    def __init__(self,config):
        super().__init__()
        ## LN after 
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)
    
    def forward(self,x):
       # x = x + self.attn(self.ln_1(x))
       # x = x + self.mlp(self.ln_2(x))
        x_res = x  # Save input before attention

        # LayerNorm1
        x_ln1 = self.ln_1(x_res)
        print("Python X_norm1[1][:10]:", x_ln1[0, 1, :10].detach().numpy())
        # Attention
        attn_out = self.attn(x_ln1)

        # Residual1
        res1 = x_res + attn_out
        print("Python residual_out[1][:10]:", res1[0, 1, :10].detach().numpy())

        # LayerNorm2
        x_ln2 = self.ln_2(res1)
        print("Python X_norm2[1][:10]:", x_ln2[0, 1, :10].detach().numpy())
        #exit(0)

        # MLP
        mlp_out = self.mlp(x_ln2)

        # Residual2 (final output of the block)
        res2 = res1 + mlp_out
        print("Python residual2_out[1][:10]:", res2[0, 1, :10].detach().numpy())
        return res2
        #return x

class GPT(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))

        # final classifier head
        self.lm_head = nn.Linear(config.n_embd,config.vocab_size,bias = False)
    
    def forward(self, idx, targets=None):
        # idx is of shape (B, T)
        # example shape [5,17]; Batch size = 5, T = 17
        print("idx shape:", idx.size())
        #exit(0)
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
        # forward the token and positions embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device) # shape (T)
                
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (T, n_embd)
        print("pos_emb shape:", pos_emb.size())
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (B, T, n_embd)
        print("tok_emb shape:", tok_emb.size())
        #exit(0)
        x = tok_emb + pos_emb ## broadcasting hidden here
        # forward the blocks of the transformer
        print("Embedding input to transformer (token 0)[:10]:")
        print(x[0, 0, :10].detach().numpy())

        for i, block in enumerate(self.transformer.h):
            x = block(x)
            if i == 0:
                print("Python residual2_out[1][:10]:")
                print(x[0, 1, :10].detach().numpy())
                #exit()
        #print("Python residual2_out[-1][:10]:")
        #print(x[0, -1, :10].detach().numpy())

        # forward the final layernorm and the classifier
        x = self.transformer.ln_f(x)
        print("Python Xf_out[1][:10]:")
        print(x[0, 1, :10].detach().numpy())
        #exit()
        logits = self.lm_head(x) # (B, T, vocab_size)
        print("Python logits[1][:10]:")
        print(logits[0, 1, :10].detach().numpy()) 
        return logits
        #loss = None
        #if targets is not None:
        #    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        #return logits, loss
    
    @classmethod
    def from_pretrained(cls, model_type):
        """Loads pretrained GPT-2 model weights from huggingface"""
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print("***loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        load_bin_weights_into_model(model, "gpt2_weights.bin")
        #print("*************ln_f weight***************")
        #print(model.transformer.ln_f.weight[:10].detach().numpy())
        #print("ln_f bias")
        #print(model.transformer.ln_f.bias[:10].detach().numpy())
        return model
        ## load weights from bin 
        ##sd = model.state_dict()
        ## load weights from bin 
        ##sd_keys = sd.keys()
        #print(sd_keys)
        ## load weights from bin 
        ##sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param
        #print("after")
        #print(sd_keys)
        #exit
        
        # init a huggingface/transformers model
        # this is the model we will copy weights from
        # replace with import itself
        model_path = "../transformers/models/gpt2"
        model_hf = GPT2LMHeadModel.from_pretrained(model_path)
        print("loaded GPT2 model")
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        print(sd_keys_hf)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model
## Load the params from hf
max_return_sequence = 1
max_length = 10

model = GPT.from_pretrained('gpt2')
print("Model loaded successfully from huggingface gpt2!")
model.eval()
print("Model eval done")
#import tiktoken
from tokenizers import Tokenizer
#enc = tiktoken.get_encoding("gpt2")
enc = Tokenizer.from_file("../gpt2/tokenizer.json")

tokens = enc.encode("the sky is blue").ids
print("TOKENS:",tokens)
tokens = torch.tensor(tokens,dtype=torch.long)
# duplicate the tokens for the number of sequences we want to generate
tokens = tokens.unsqueeze(0).repeat(max_return_sequence,1)
x = tokens
print("tokens shape:", x.shape)
#exit(0)

torch.manual_seed(242)
while x.size(1) < max_length:
    with torch.no_grad():
        logits = model(x)
        # get the last token
        logits = logits[:,-1,:]
        # sample from the distribution
        probs = F.softmax(logits, dim=-1)
        topk_probs, topk_indices = torch.topk(probs, 50,dim=-1)
        ix = torch.multinomial(topk_probs, 1)
        xcol = torch.gather(topk_indices, -1, ix)
        x = torch.cat((x,xcol), dim=1)
        # Greedy decoding (argmax)
        #next_token = torch.argmax(logits, dim=-1, keepdim=True)
        #x = torch.cat((x, next_token), dim=1)


for i in range(max_return_sequence):
    tokens = x[i,:max_length].tolist()
    decoded = enc.decode(tokens)
    print(">",decoded)
        




