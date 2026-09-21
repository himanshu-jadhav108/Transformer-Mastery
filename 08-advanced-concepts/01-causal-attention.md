# 01 — Causal Attention

## What is Causal Attention?

Attention where each position can only attend to previous positions (and itself).

## Why Causal?

For autoregressive generation, the model must not "see" future tokens during training.

## Implementation

```python
def causal_attention(Q, K, V):
    scores = Q @ K.transpose(-2, -1) / sqrt(d_k)  # (B, T, T)

    # Create causal mask: lower triangular
    T = scores.size(-1)
    mask = torch.tril(torch.ones(T, T, device=scores.device))
    scores = scores.masked_fill(mask == 0, float('-inf'))

    weights = softmax(scores, dim=-1)
    return weights @ V
```

## Attention Pattern

```
Position:  0   1   2   3   4
        0 [1   0   0   0   0]
        1 [1   1   0   0   0]
        2 [1   1   1   0   0]
        3 [1   1   1   1   0]
        4 [1   1   1   1   1]
```

## Efficiency Note

Causal attention computes the full (T,T) matrix then masks. Flash Attention optimizes this by skipping masked computations.

## Key Takeaway

Causal attention enforces autoregressive generation by preventing information flow from future to past. It is the defining characteristic of decoder-only models.
