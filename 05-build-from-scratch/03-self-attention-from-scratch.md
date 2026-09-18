# 03 — Self-Attention From Scratch

## Single-Head Scaled Dot-Product Attention

```python
import torch
import torch.nn as nn
import math

class ScaledDotProductAttention(nn.Module):
    def __init__(self, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, Q, K, V, mask=None):
        # Q: (B, T, d_k)
        # K: (B, T, d_k)
        # V: (B, T, d_v)
        # mask: (B, T, T) or broadcastable — 0 = mask out, 1 = keep
        # Returns: (B, T, d_v), attention weights (B, T, T)

        B, T, d_k = Q.shape

        # Step 1: Compute QK^T
        scores = torch.matmul(Q, K.transpose(-2, -1))  # (B, T, T)

        # Step 2: Scale
        scores = scores / math.sqrt(d_k)

        # Step 3: Apply mask (if provided)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        # Step 4: Softmax
        attn_weights = torch.softmax(scores, dim=-1)  # (B, T, T)
        attn_weights = self.dropout(attn_weights)

        # Step 5: Weighted sum of V
        output = torch.matmul(attn_weights, V)  # (B, T, d_v)

        return output, attn_weights
```

## Test It

```python
B, T, d_k, d_v = 2, 10, 64, 64
Q = torch.randn(B, T, d_k)
K = torch.randn(B, T, d_k)
V = torch.randn(B, T, d_v)

attn = ScaledDotProductAttention()
out, weights = attn(Q, K, V)

print(f"Output shape: {out.shape}")       # (2, 10, 64)
print(f"Weights shape: {weights.shape}")  # (2, 10, 10)
print(f"Weights sum per row: {weights[0].sum(dim=-1)}")  # All 1.0
```

## Key Takeaway

Self-attention is matrix multiplication -> scaling -> masking -> softmax -> matrix multiplication. Understand these 5 steps and you understand attention.
