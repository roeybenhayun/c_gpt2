from transformers import GPT2LMHeadModel # 124M parameters
import numpy as np
model_path = "./transformers/models/gpt2"
model = GPT2LMHeadModel.from_pretrained(model_path)
# raw tensors
sd_hf = model.state_dict()
save_tp_file = True

# List of keys whose tensors should be transposed before saving
transpose_keys = [
    "attn.c_attn.weight",   # QKV packed weight
    "attn.c_proj.weight",   # attention output projection
    "mlp.c_fc.weight",      # feedforward W1
    "mlp.c_proj.weight",    # feedforward W2
   # "lm_head.weight",       # final projection (tied with wte)
]

for k, v in sd_hf.items():
        print(f"Saving: {k} with shape {v.shape}")
#exit
if save_tp_file == True:
    # Open a binary file for writing
    with open("gpt2_c_weights.bin", "wb") as f:
        for k, v in sd_hf.items():
            print(f"Saving: {k} with shape {v.shape}")        
            # Convert tensor to numpy array of float32
            np_array = v.cpu().numpy().astype(np.float32)
            
            # Apply transpose where needed
            if any(k.endswith(name) for name in transpose_keys):
                print("TRANSPOSED")
                np_array = np_array.transpose()

            # Save the binary bytes directly
            f.write(np_array.tobytes())
