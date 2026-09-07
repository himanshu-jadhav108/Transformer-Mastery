# 02 — Query, Key, Value

## What Are Q, K, V?

Three linear projections of the same input, each serving a distinct purpose:

| Symbol | Name | Role | Analogy |
|--------|------|------|---------|
| Q | Query | "What am I looking for?" | Your search terms |
| K | Key | "What do I contain?" | Document metadata |
| V | Value | "What information do I have?" | Document content |

## Why Three Projections?

The input embedding alone cannot simultaneously serve as:
1. The search term (Q)
2. The index/key (K)
3. The content (V)

Separate projections allow the model to learn specialized representations for each role.

## Mathematical View

Given input `X` of shape `(B, T, d_model)`:

```
Q = X · W_Q    where W_Q: (d_model, d_k)
K = X · W_K    where W_K: (d_model, d_k)
V = X · W_V    where W_V: (d_model, d_v)
```

## Tensor Shapes

```
X:   (B, T, d_model)
W_Q: (d_model, d_k)
Q:   (B, T, d_k)

W_K: (d_model, d_k)
K:   (B, T, d_k)

W_V: (d_model, d_v)
V:   (B, T, d_v)
```

## Intuition

Think of Q, K, V as three different "views" of the same information:
- **Q**: The asking perspective ("What do I need?")
- **K**: The answering perspective ("What can I offer?")
- **V**: The content perspective ("Here is my information")

## Why d_k and d_v?

- `d_k`: Dimension of the similarity space (typically `d_model / h`)
- `d_v`: Dimension of the output value (typically `d_model / h`)

In practice, `d_k = d_v = d_model / h` for simplicity.

## Common Mistake

Thinking Q, K, V come from different inputs. In **self-attention**, they all come from the SAME input via different linear projections.

## Key Takeaway

Q, K, V are learned projections that enable the attention mechanism to compare tokens (Q·K) and retrieve information (V) in a flexible, trainable way.
