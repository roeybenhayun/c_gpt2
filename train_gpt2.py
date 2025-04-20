
import math
import os
import time
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F

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

        att = (q @ k.transpose(-2,-1)) * (1.0 / math.sqrt(k.size(-1)))
        ## autoregresive masking -make sure tokens only attends to previous tokens
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        ## normalize the attention - sum to one always
        att = F.softmax(att, dim=-1)
        y = att @ v
        y = y.transpose(1,2).contiguous().view(B,T,C)
        y = self.c_proj(y)
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
        x = self.gelu(x)
        x = self.c_proj(x)
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
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

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
        for block in self.transformer.h:
            x = block(x)
        # forward the final layernorm and the classifier
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x) # (B, T, vocab_size)
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
        print("loading weights from pretrained gpt: %s" % model_type)

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
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        # this is the model we will copy weights from
        # replace with import itself
        model_path = "../transformers/models/gpt2"
        model_hf = GPT2LMHeadModel.from_pretrained(model_path)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
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
max_return_sequence = 5
max_length = 30

model = GPT.from_pretrained('gpt2')
print("Model loaded successfully from huggingface gpt2!")
model.eval()

import tiktoken
enc = tiktoken.get_encoding("gpt2")
tokens = enc.encode("Hello, I have a lovely dog. He loves to play and play the guitar.")
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


for i in range(max_return_sequence):
    tokens = x[i,:max_length].tolist()
    decoded = enc.decode(tokens)
    print(">",decoded)
        
