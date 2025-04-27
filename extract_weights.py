from transformers import GPT2LMHeadModel # 124M parameters
import numpy as np
model_path = "./transformers/models/gpt2"
model = GPT2LMHeadModel.from_pretrained(model_path)
# raw tensors
sd_hf = model.state_dict()
save_tp_file = False

for k, v in sd_hf.items():
        print(f"Saving: {k} with shape {v.shape}")

if save_tp_file == True:
    # Open a binary file for writing
    with open("gpt2_weights.bin", "wb") as f:
        for k, v in sd_hf.items():
            print(f"Saving: {k} with shape {v.shape}")        
            # Convert tensor to numpy array of float32
            np_array = v.cpu().numpy().astype(np.float32)
            # Save the binary bytes directly
            f.write(np_array.tobytes())
