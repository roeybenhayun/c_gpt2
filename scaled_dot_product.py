import torch
import torch.nn.functional as F

# Define Q (3x4) as in C
Q = torch.tensor([
    [1.2, 2.3, 3.0, 4.9],
    [1.2, 2.3, 3.0, 4.0],
    [1.2, 2.3, 3.0, 4.9]
], dtype=torch.float32)  # (3, 4)

# Define K (3x4) — NOT transposed — so matches shape with Q
K = torch.tensor([
    [0.3, 0.9, 1.1],
    [0.4, 0.9, 1.1],
    [0.5, 0.9, 1.1],
    [0.5, 0.9, 1.1]
], dtype=torch.float32)  # shape: (4, 3)
K = K.T

# Define V (3x4) — same as C
V = torch.tensor([
    [0.3, 0.9, 1.1, 0.4],
    [0.4, 0.9, 1.1, 1.2],
    [0.5, 0.9, 1.1, 0.1]
], dtype=torch.float32)  # (3, 4)

# Reshape for (batch=1, heads=1, seq_len=3, d_k=4)
Q = Q.unsqueeze(0).unsqueeze(0)  # (1, 1, 3, 4)
K = K.unsqueeze(0).unsqueeze(0)  # (1, 1, 3, 4)
V = V.unsqueeze(0).unsqueeze(0)  # (1, 1, 3, 4)

# Causal mask: shape (1, 1, 3, 3), True = masked
causal_mask = torch.triu(torch.ones(3, 3, dtype=torch.bool), diagonal=1)
causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)  # (1, 1, 3, 3)

# Run scaled dot product attention with mask
context = F.scaled_dot_product_attention(Q, K, V, attn_mask=causal_mask, dropout_p=0.0)

# Remove batch/head dimensions
context = context.squeeze(0).squeeze(0)

# Print final context
print("Context (after masked scaled dot product attention):")
print(context.round(decimals=2))
