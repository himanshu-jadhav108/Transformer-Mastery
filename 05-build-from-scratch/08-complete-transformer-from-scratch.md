# 08 — Complete Transformer From Scratch

## Full Implementation

```python
class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512,
                 num_heads=8, d_ff=2048, num_layers=6, max_len=5000,
                 dropout=0.1):
        super().__init__()
        self.d_model = d_model

        self.encoder = TransformerEncoder(
            src_vocab_size, d_model, num_heads, d_ff,
            num_layers, max_len, dropout
        )
        self.decoder = TransformerDecoder(
            tgt_vocab_size, d_model, num_heads, d_ff,
            num_layers, max_len, dropout
        )
        self.output_layer = nn.Linear(d_model, tgt_vocab_size)
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def make_src_mask(self, src):
        # (B, 1, 1, T_src)
        return (src != 0).unsqueeze(1).unsqueeze(2)

    def make_tgt_mask(self, tgt):
        tgt_pad = (tgt != 0).unsqueeze(1).unsqueeze(3)
        T = tgt.size(1)
        causal = torch.tril(torch.ones(T, T, device=tgt.device))
        causal = causal.unsqueeze(0).unsqueeze(0)
        return tgt_pad & causal  # (B, 1, T, T)

    def forward(self, src, tgt):
        # src: (B, T_src), tgt: (B, T_tgt)
        # return: (B, T_tgt, tgt_vocab_size)
        src_mask = self.make_src_mask(src)
        tgt_mask = self.make_tgt_mask(tgt)

        enc_out = self.encoder(src, src_mask)
        dec_out = self.decoder(tgt, enc_out, src_mask, tgt_mask)
        return self.output_layer(dec_out)
```

## Test

```python
B, T_src, T_tgt = 2, 20, 15
src_vocab, tgt_vocab = 10000, 10000

model = Transformer(src_vocab, tgt_vocab)
src = torch.randint(1, src_vocab, (B, T_src))
tgt = torch.randint(1, tgt_vocab, (B, T_tgt))

output = model(src, tgt)
print(output.shape)  # (2, 15, 10000)
```

## Key Takeaway

You have built a complete Transformer. Every component is explicit and understood. This is the foundation for all modern language models.
