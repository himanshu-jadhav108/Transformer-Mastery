# 05 — Masking

## Why Mask?

Two types of masking serve different purposes:

## 1. Padding Mask

**Problem**: Batches contain sequences of different lengths. Shorter sequences are padded.

**Solution**: Mask out padded positions so they don't contribute to attention.

```
Attention scores before mask:  (B, T, T)
Padding mask:                  (B, T, T)  ← 0 for pad, 1 for real
Masked scores:                 (B, T, T)  ← pad positions = -∞
After softmax:                 (B, T, T)  ← pad positions = 0
```

**Implementation**: Add `-1e9` (or `-inf`) to attention scores at padded positions before softmax.

## 2. Causal (Look-Ahead) Mask

**Problem**: During training, decoder sees full target sequence. It must not peek at future tokens.

**Solution**: Mask upper-triangle of attention matrix.

```
Causal mask for T=4:
  [[1, 0, 0, 0],
   [1, 1, 0, 0],
   [1, 1, 1, 0],
   [1, 1, 1, 1]]

Position 0 can attend to: 0
Position 1 can attend to: 0, 1
Position 2 can attend to: 0, 1, 2
Position 3 can attend to: 0, 1, 2, 3
```

## Combined Mask

In practice, both masks are combined:
```python
mask = padding_mask & causal_mask
```

## Tensor Shapes

```
Attention scores: (B, T, T)
Padding mask:     (B, 1, T) or (B, T, T)
Causal mask:      (1, T, T) or (T, T)
Combined:         (B, T, T)
```

## Common Mistake

Applying causal mask in the encoder. The encoder is **bidirectional** — no causal mask needed.

## Key Takeaway

- **Padding mask**: Ignores padding tokens
- **Causal mask**: Prevents looking at future tokens (decoder only)
- Combined mask = element-wise AND of both
