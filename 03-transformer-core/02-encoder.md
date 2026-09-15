# 02 — Encoder

## What is the Encoder?

A stack of **N identical layers** that process the input sequence bidirectionally.

## Original Paper: N = 6 layers

## Single Encoder Layer

```
Input: (B, T, d_model)
  │
  ├─→ Multi-Head Self-Attention ──→ Add & Norm ──┐
  │                                               │
  └───────────────────────────────────────────────┘
  │
  ├─→ Feed-Forward Network ───────→ Add & Norm ──┐
  │                                              │
  └──────────────────────────────────────────────┘
  │
Output: (B, T, d_model)
```

## Components

1. **Multi-Head Self-Attention**: Each token attends to ALL other tokens (bidirectional)
2. **Add & Norm**: Residual connection + Layer Normalization
3. **Feed-Forward Network**: Position-wise MLP
4. **Repeat N times**

## Why Bidirectional?

The encoder has no causal mask. Every token sees every other token. This is why BERT (encoder-only) excels at understanding tasks.

## Tensor Flow

```
Input:     (B, T, d_model)
  ↓
Self-Attn: (B, T, d_model)
  ↓
Add & Norm:(B, T, d_model)
  ↓
FFN:       (B, T, d_model)
  ↓
Add & Norm:(B, T, d_model)
  ↓
[Repeat N times]
  ↓
Output:    (B, T, d_model)
```

## Key Takeaway

The encoder builds rich, bidirectional contextual representations of the input. It is the "understanding" half of the Transformer.
