# 03 — Softmax

## Definition

```
softmax(x_i) = exp(x_i) / Σ_j exp(x_j)
```

## Properties

1. **Output range**: (0, 1)
2. **Sum to 1**: `Σ_i softmax(x_i) = 1`
3. **Monotonic**: If `x_i > x_j`, then `softmax(x_i) > softmax(x_j)`
4. **Amplifies differences**: Large gaps in input become more extreme in output

## Numerical Stability

```python
def stable_softmax(x):
    x_max = x.max(dim=-1, keepdim=True)
    return torch.exp(x - x_max) / torch.exp(x - x_max).sum(dim=-1, keepdim=True)
```

Subtracting max prevents overflow in `exp()`.

## In Transformers

```
AttentionWeights = softmax(QK^T / √d_k, dim=-1)
```

- Applied **row-wise** across the sequence dimension
- Converts similarity scores to a probability distribution
- Allows gradient flow to all positions (unlike argmax)

## Gradient of Softmax

```
∂softmax(x_i)/∂x_j = softmax(x_i) · (δ_ij - softmax(x_j))
```

Where `δ_ij = 1` if `i=j`, else `0`.

## Common Mistake

Applying softmax across the wrong dimension. In attention, it must be across the **key dimension** (dim=-1), not the batch or feature dimension.

## Key Takeaway

Softmax turns arbitrary scores into a valid probability distribution. It is differentiable (unlike argmax), enabling end-to-end gradient-based training.
