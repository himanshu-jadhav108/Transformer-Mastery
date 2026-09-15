# 07 — Residual Connections

## The Problem

Deep networks suffer from **vanishing gradients** and **degradation** (accuracy drops as depth increases).

## Solution: Residual Connection

```
Output = LayerNorm(x + Sublayer(x))
```

Where `Sublayer(x)` is either attention or FFN.

## Why It Works

1. **Gradient highway**: Gradients can flow directly through the skip connection
2. **Identity learning**: If `Sublayer(x) ≈ 0`, the layer acts as identity — easy to learn
3. **Ensemble effect**: Deep residual networks behave like implicit ensembles of shallow networks

## In the Transformer

Each sub-layer (attention, FFN) is wrapped:
```
x = LayerNorm(x + MultiHeadAttention(x))
x = LayerNorm(x + FFN(x))
```

## Pre-Norm vs Post-Norm

| Variant | Formula | Used In |
|---------|---------|---------|
| Post-Norm (original) | `x = LayerNorm(x + Sublayer(x))` | Original Transformer |
| Pre-Norm (modern) | `x = x + Sublayer(LayerNorm(x))` | GPT, BERT, most modern models |

**Pre-Norm is more stable** for very deep networks. Modern Transformers use Pre-Norm.

## Key Takeaway

Residual connections enable training of deep Transformers by creating gradient shortcuts and making identity mappings easy to learn. Pre-Norm is preferred for deep models.
