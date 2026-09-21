# 03 — RoPE (Rotary Position Embeddings)

## The Problem with Absolute Positional Encoding

Adding fixed/learned vectors to embeddings:
- Cannot extrapolate beyond training length
- Position information is "baked in" early, hard to modify

## Solution: Rotate Q and K by Position

Instead of adding position to embeddings, rotate the Q and K vectors by an angle proportional to their position.

## Mathematical Formulation

For a 2D pair (x₁, x₂) at position m:
```
RoPE(x, m) = [x₁·cos(m·θ) - x₂·sin(m·θ),
              x₁·sin(m·θ) + x₂·cos(m·θ)]
```

Where θ = base^(-2i/d) for dimension pair i.

## In d dimensions

Apply rotation to each pair of dimensions:
```python
def apply_rotary_pos_emb(q, k, cos, sin):
    # q, k: (B, h, T, d_k)
    # cos, sin: (1, 1, T, d_k) precomputed

    # Split into pairs
    q1, q2 = q[..., ::2], q[..., 1::2]
    k1, k2 = k[..., ::2], k[..., 1::2]

    # Rotate
    q_rot = torch.stack([q1*cos - q2*sin, q1*sin + q2*cos], dim=-1)
    k_rot = torch.stack([k1*cos - k2*sin, k1*sin + k2*cos], dim=-1)

    return q_rot.flatten(-2), k_rot.flatten(-2)
```

## Why RoPE Works

1. **Relative positions**: `RoPE(q, m)·RoPE(k, n)` depends only on (m-n)
2. **Extrapolation**: Can generalize to longer sequences
3. **No extra parameters**: Angles are fixed
4. **Applied to Q,K only**: V remains unchanged

## Base Frequency

```
θ_i = 1 / (base^(2i/d_k))
```

Default base = 10000. Larger bases (e.g., 500000) improve long-context performance.

## Key Takeaway

RoPE encodes position by rotating Q and K vectors. It naturally encodes relative positions, requires no learned parameters, and extrapolates better than absolute positional encoding.
