# 05 — Transformer Block

## Position-Wise Feed-Forward Network

```python
class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()  # Modern: GELU; Original: ReLU

    def forward(self, x):
        # x: (B, T, d_model)
        # return: (B, T, d_model)
        x = self.linear1(x)       # (B, T, d_ff)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)       # (B, T, d_model)
        return x
```

## Pre-Norm Transformer Block

```python
class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = PositionWiseFeedForward(d_model, d_ff, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # x: (B, T, d_model)
        # mask: attention mask
        # return: (B, T, d_model)

        # Pre-Norm: norm -> sublayer -> residual
        attn_out, _ = self.attn(self.norm1(x), self.norm1(x), 
                               self.norm1(x), mask)
        x = x + self.dropout(attn_out)  # residual

        ffn_out = self.ffn(self.norm2(x))
        x = x + self.dropout(ffn_out)   # residual

        return x
```

## Tensor Shapes

```
Input:     (B, T, d_model)
  |
LayerNorm: (B, T, d_model)
  |
Self-Attn: (B, T, d_model)
  |
Residual:  (B, T, d_model)
  |
LayerNorm: (B, T, d_model)
  |
FFN:       (B, T, d_model)
  |
Residual:  (B, T, d_model)
```

## Key Takeaway

A Transformer block is: LayerNorm -> Attention -> Residual -> LayerNorm -> FFN -> Residual. Pre-norm is more stable than post-norm for deep networks.
