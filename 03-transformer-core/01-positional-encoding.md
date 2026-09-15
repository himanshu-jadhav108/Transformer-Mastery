# 01 — Positional Encoding

## The Problem

Attention is **permutation-invariant**: `Attention(X) = Attention(shuffled_X)`.

But language order matters: "dog bites man" ≠ "man bites dog".

## Solution: Positional Encoding

Add position-dependent signals to embeddings so the model knows token order.

## Sinusoidal Positional Encoding (Original Paper)

```
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

Where:
- `pos` = token position (0, 1, 2, ...)
- `i` = dimension index
- `d_model` = embedding dimension

## Why Sinusoids?

1. **Unique encoding**: Every position gets a unique pattern
2. **Relative positions**: `PE(pos+k)` can be represented as linear function of `PE(pos)` (due to trig identities)
3. **Extrapolation**: Can generalize to longer sequences than seen in training
4. **Fixed**: No parameters to learn

## Tensor Shapes

```
Token embeddings: (B, T, d_model)
Positional enc:   (1, T, d_model)  or (T, d_model)
Final input:      (B, T, d_model)  = embeddings + positional_enc
```

## Learned Positional Embeddings

Alternative: Learn a matrix `PE` of shape `(max_len, d_model)` and add it.

| Approach | Pros | Cons |
|----------|------|------|
| Sinusoidal | Fixed, extrapolates | Less flexible |
| Learned | More flexible | Cannot extrapolate beyond max_len |

## Modern: Rotary Position Embeddings (RoPE)

See `08-advanced-concepts/03-rope.md`. RoPE encodes position by rotating Q and K vectors rather than adding to embeddings.

## Key Takeaway

Positional encoding injects order information into the otherwise order-agnostic attention mechanism. Sinusoidal encoding is parameter-free and theoretically elegant.
