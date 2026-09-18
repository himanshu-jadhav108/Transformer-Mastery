# 07 — Decoder From Scratch

## Implementation

```python
class TransformerDecoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.masked_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.norm3 = nn.LayerNorm(d_model)
        self.ffn = PositionWiseFeedForward(d_model, d_ff, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_output, src_mask=None, tgt_mask=None):
        # x: (B, T_tgt, d_model)
        # enc_output: (B, T_src, d_model)
        # return: (B, T_tgt, d_model)

        # 1. Masked self-attention
        attn_out, _ = self.masked_attn(
            self.norm1(x), self.norm1(x), self.norm1(x), tgt_mask
        )
        x = x + self.dropout(attn_out)

        # 2. Cross-attention
        cross_out, _ = self.cross_attn(
            self.norm2(x), enc_output, enc_output, src_mask
        )
        x = x + self.dropout(cross_out)

        # 3. Feed-forward
        ffn_out = self.ffn(self.norm3(x))
        x = x + self.dropout(ffn_out)

        return x


class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, d_ff,
                 num_layers, max_len=5000, dropout=0.1):
        super().__init__()
        self.embedding = TransformerEmbeddings(vocab_size, d_model,
                                               max_len, dropout)
        self.layers = nn.ModuleList([
            TransformerDecoderBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, enc_output, src_mask=None, tgt_mask=None):
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x, enc_output, src_mask, tgt_mask)
        return self.norm(x)
```

## Causal Mask

```python
def generate_causal_mask(size):
    mask = torch.tril(torch.ones(size, size)).unsqueeze(0)
    return mask  # (1, size, size)
```

## Key Takeaway

The decoder has three sub-layers: masked self-attention, cross-attention, and FFN. The causal mask prevents looking at future tokens.
