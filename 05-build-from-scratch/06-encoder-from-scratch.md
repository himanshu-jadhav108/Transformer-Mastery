# 06 — Encoder From Scratch

## Implementation

```python
class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, d_ff, 
                 num_layers, max_len=5000, dropout=0.1):
        super().__init__()
        self.embedding = TransformerEmbeddings(vocab_size, d_model, 
                                               max_len, dropout)
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        # x: (B, T) token IDs
        # mask: (B, T, T) padding mask
        # return: (B, T, d_model)
        x = self.embedding(x)  # (B, T, d_model)

        for layer in self.layers:
            x = layer(x, mask)  # (B, T, d_model)

        return self.norm(x)
```

## Tensor Shapes

```
Input:      (B, T_src)
Embedding:  (B, T_src, d_model)
Layer 1:    (B, T_src, d_model)
Layer 2:    (B, T_src, d_model)
...
Layer N:    (B, T_src, d_model)
Output:     (B, T_src, d_model)
```

## Key Takeaway

The encoder is a stack of identical Transformer blocks with bidirectional self-attention. The output is a context-rich representation of the entire input sequence.
