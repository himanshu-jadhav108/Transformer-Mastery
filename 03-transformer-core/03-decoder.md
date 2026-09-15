# 03 — Decoder

## What is the Decoder?

A stack of **N identical layers** that generate output tokens autoregressively (one at a time).

## Single Decoder Layer

```
Input: (B, T, d_model)  ← previously generated tokens
  │
  ├─→ Masked Multi-Head Self-Attention ──→ Add & Norm ──┐
  │                                                      │
  └──────────────────────────────────────────────────────┘
  │
  ├─→ Multi-Head Cross-Attention ────────→ Add & Norm ──┐
  │   (Q from decoder, K,V from encoder)                  │
  └───────────────────────────────────────────────────────┘
  │
  ├─→ Feed-Forward Network ────────────────→ Add & Norm ──┐
  │                                                       │
  └───────────────────────────────────────────────────────┘
  │
Output: (B, T, d_model)
```

## Three Sub-Layers (vs Encoder's Two)

1. **Masked Self-Attention**: Decoder tokens attend only to previous decoder tokens
2. **Cross-Attention**: Decoder queries attend to encoder keys/values
3. **Feed-Forward Network**: Position-wise MLP

## Why Masked Self-Attention?

During training, the decoder sees the FULL target sequence. Masking prevents it from "cheating" by looking at future tokens.

During inference, tokens are generated one at a time, so masking is naturally enforced.

## Tensor Flow

```
Decoder Input:  (B, T_dec, d_model)
  ↓
Masked Self-Attn:(B, T_dec, d_model)
  ↓
Add & Norm:     (B, T_dec, d_model)
  ↓
Cross-Attention:(B, T_dec, d_model)
  ↓
Add & Norm:     (B, T_dec, d_model)
  ↓
FFN:            (B, T_dec, d_model)
  ↓
Add & Norm:     (B, T_dec, d_model)
  ↓
[Repeat N times]
  ↓
Output:         (B, T_dec, d_model)
```

## Key Takeaway

The decoder generates output autoregressively. Masked self-attention prevents looking ahead, while cross-attention allows the decoder to query the encoder's representations.
