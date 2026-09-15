# 08 — Layer Normalization

## What is Layer Normalization?

Normalize across the **feature dimension** for each token independently.

## Formula

```
LayerNorm(x) = γ ⊙ (x - μ) / √(σ² + ε) + β

where:
  μ = mean(x) across features
  σ² = variance(x) across features
  γ, β = learned scale and shift parameters
  ε = small constant for numerical stability
```

## Why LayerNorm (Not BatchNorm)?

| Property | BatchNorm | LayerNorm |
|----------|-----------|-----------|
| Normalizes across | Batch dimension | Feature dimension |
| Requires | Large batch size | Works with any batch size |
| For sequences | Problematic (variable length) | Natural fit |
| Used in | CNNs | Transformers, RNNs |

## Tensor Shapes

```
Input:  (B, T, d_model)
μ, σ²:  (B, T, 1)        — computed across d_model dimension
γ, β:   (d_model,)        — learned parameters
Output: (B, T, d_model)
```

## Purpose in Transformers

1. **Stabilizes training**: Prevents internal covariate shift
2. **Controls activation scale**: Keeps values in reasonable range
3. **Enables deeper networks**: Without it, deep Transformers are unstable

## RMSNorm (Modern Alternative)

Used in LLaMA:
```
RMSNorm(x) = x / √(mean(x²) + ε) · γ
```

Simpler than LayerNorm (no mean subtraction, no bias). Empirically works as well or better.

## Key Takeaway

LayerNorm normalizes each token's features independently, stabilizing training and enabling deep architectures. RMSNorm is a modern simplification gaining popularity.
