from transformers import GPT2LMHeadModel # 124M parameters
import numpy as np
target_model_size = 'medium'
if (target_model_size == 'small'):
    model_path = "./transformers/models/gpt2"
    out_file = "gpt2_c_weights.bin"
elif (target_model_size == 'medium'):
    model_path = "./transformers/models/gpt2-medium"     
    out_file = "gpt2_medium_c_weights.bin"
else:
    # Handle invalid choice: important for robustness
    raise ValueError("Invalid model size specified. Choose 'small' or 'medium'.")

model = GPT2LMHeadModel.from_pretrained(model_path) # default is GPT-2 small model (124M parameters)
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

if save_tp_file == True:
    # Open a binary file for writing
    with open(out_file, "wb") as f:
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
