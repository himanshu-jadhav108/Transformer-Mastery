# 06 — Feed-Forward Networks

## What is the FFN?

A position-wise MLP applied to each token independently.

## Architecture

```
FFN(x) = max(0, x·W₁ + b₁)·W₂ + b₂
```

Or with GELU activation (modern):
```
FFN(x) = GELU(x·W₁ + b₁)·W₂ + b₂
```

## Dimensions

```
x:    (B, T, d_model)
W₁:   (d_model, d_ff)    where d_ff = 4 × d_model (typical)
b₁:   (d_ff,)
W₂:   (d_ff, d_model)
b₂:   (d_model,)
```

## Why Position-Wise?

The same MLP is applied to each position independently. No interaction between positions — that's the job of attention.

## Why d_ff = 4 × d_model?

The inner dimension is larger to increase model capacity. The bottleneck structure (expand → contract) is efficient.

## Modern Variant: SwiGLU

Used in LLaMA, PaLM:
```
SwiGLU(x) = (x·W · SiLU(x·V)) · W₂
```
Replaces single-gate FFN with gated activation.

## Tensor Flow

```
Input:   (B, T, d_model)
  ↓
Linear1: (B, T, d_ff)
  ↓
Activation (ReLU/GELU/SwiGLU)
  ↓
Linear2: (B, T, d_model)
  ↓
Output:  (B, T, d_model)
```

## Key Takeaway

The FFN adds non-linearity and increases model capacity. It processes each position independently, complementing the inter-position mixing done by attention.
