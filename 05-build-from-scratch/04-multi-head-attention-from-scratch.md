# 04 — Multi-Head Attention From Scratch

## Implementation

```python
import torch
import torch.nn as nn
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # 64 for d_model=512, h=8

        # Linear projections for Q, K, V
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)

        # Output projection
        self.W_O = nn.Linear(d_model, d_model)

        self.attention = ScaledDotProductAttention(dropout)
        self.dropout = nn.Dropout(dropout)

    def split_heads(self, x):
        # x: (B, T, d_model)
        # return: (B, h, T, d_k)
        B, T, _ = x.shape
        x = x.view(B, T, self.num_heads, self.d_k)
        return x.transpose(1, 2)  # (B, h, T, d_k)

    def combine_heads(self, x):
        # x: (B, h, T, d_k)
        # return: (B, T, d_model)
        B, _, T, _ = x.shape
        x = x.transpose(1, 2).contiguous()
        return x.view(B, T, self.d_model)

    def forward(self, query, key, value, mask=None):
        # query, key, value: (B, T, d_model)
        # mask: (B, 1, T, T) or broadcastable
        # return: (B, T, d_model), attention weights

        B, T, _ = query.shape

        # Step 1: Linear projections
        Q = self.W_Q(query)  # (B, T, d_model)
        K = self.W_K(key)
        V = self.W_V(value)

        # Step 2: Split into heads
        Q = self.split_heads(Q)  # (B, h, T, d_k)
        K = self.split_heads(K)
        V = self.split_heads(V)

        # Step 3: Apply attention
        attn_out, attn_weights = self.attention(Q, K, V, mask)
        # attn_out: (B, h, T, d_k)

        # Step 4: Combine heads
        attn_out = self.combine_heads(attn_out)  # (B, T, d_model)

        # Step 5: Output projection
        output = self.W_O(attn_out)  # (B, T, d_model)
        output = self.dropout(output)

        return output, attn_weights
```

## Tensor Shape Flow

```
Input:          (B, T, d_model)
  |
W_Q/W_K/W_V:    (B, T, d_model)
  |
split_heads:    (B, h, T, d_k)
  |
Attention:      (B, h, T, d_k), weights (B, h, T, T)
  |
combine_heads:  (B, T, d_model)
  |
W_O:            (B, T, d_model)
```

## Key Takeaway

Multi-head attention = project -> split -> attend -> combine -> project. The split/combine operations are purely reshaping — no learned parameters.
