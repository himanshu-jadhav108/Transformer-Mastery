# 02 — Embeddings From Scratch

## What We Build

Token embeddings + positional encoding = model input.

## Implementation

```python
import torch
import torch.nn as nn
import math

class TransformerEmbeddings(nn.Module):
    def __init__(self, vocab_size, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.d_model = d_model

        # Token embeddings: (V, d_model)
        self.token_embedding = nn.Embedding(vocab_size, d_model)

        # Positional encoding: (max_len, d_model)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        # div_term = 1 / (10000^(2i/d_model))
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * 
            (-math.log(10000.0) / d_model)
        )

        # Apply sin to even indices, cos to odd indices
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Register as buffer (not a parameter)
        self.register_buffer('pe', pe.unsqueeze(0))  # (1, max_len, d_model)

        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(d_model)

    def forward(self, x):
        # x: (B, T) token IDs
        # return: (B, T, d_model)
        B, T = x.shape

        # Token embeddings + scaling
        tok_emb = self.token_embedding(x) * self.scale  # (B, T, d_model)

        # Add positional encoding (first T positions)
        pos_emb = self.pe[:, :T, :]  # (1, T, d_model)

        return self.dropout(tok_emb + pos_emb)
```

## Tensor Shapes

```
Input x:          (B, T)
token_embedding:  (B, T, d_model)
pe[:, :T, :]:     (1, T, d_model)
Output:           (B, T, d_model)
```

## Why Scale by sqrt(d_model)?

Embeddings have variance ~1. Positional encodings have variance ~0.5. Scaling balances their magnitudes so neither dominates.

## Key Takeaway

Embeddings turn discrete tokens into continuous vectors. Positional encoding adds order information. The combination is the Transformer's input representation.
