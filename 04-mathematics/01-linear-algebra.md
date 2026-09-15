# 01 — Linear Algebra for Transformers

## Vectors

A vector is an ordered list of numbers. In Transformers, each token is a vector in d_model-dimensional space.

```
x ∈ ℝ^d_model

Example: x = [0.2, -0.5, 1.3, ..., 0.1]  (512 numbers)
```

## Matrices

A matrix is a 2D array. In Transformers, weight matrices transform vectors.

```
W ∈ ℝ^(m×n)

Example: W_Q ∈ ℝ^(512×64) transforms 512-dim embeddings to 64-dim queries
```

## Matrix Multiplication

```
C = A · B
C[i,j] = Σ_k A[i,k] · B[k,j]
```

**Transformer applications**:
- `Q = X · W_Q`: Project embeddings to query space
- `Scores = Q · K^T`: Compute all pairwise similarities
- `Output = AttentionWeights · V`: Weighted combination

## Dot Product

```
a · b = Σ_i a_i · b_i
```

Measures similarity: large positive = similar direction, zero = orthogonal, negative = opposite.

**In attention**: `Q · K^T` measures how much each query "matches" each key.

## Transpose

```
A: (m, n)  →  A^T: (n, m)
```

**In attention**: `K^T` changes shape from `(B, T, d_k)` to `(B, d_k, T)` so `Q · K^T` yields `(B, T, T)`.

## Tensor Shapes

| Operation | Input Shapes | Output Shape |
|-----------|-------------|--------------|
| Embedding lookup | IDs (B,T), E (V,d) | (B,T,d) |
| Linear projection | (B,T,d_model), W (d_model,d_k) | (B,T,d_k) |
| QK^T | Q (B,T,d_k), K^T (B,d_k,T) | (B,T,T) |
| Softmax × V | (B,T,T), V (B,T,d_v) | (B,T,d_v) |

## Key Takeaway

Transformers are 99% matrix multiplication. Understand shapes, and you understand the architecture.
