# 02 — Attention Mathematics

## Complete Derivation

### Given

```
X ∈ ℝ^(B×T×d_model)     — input embeddings
W_Q ∈ ℝ^(d_model×d_k)   — query projection
W_K ∈ ℝ^(d_model×d_k)   — key projection
W_V ∈ ℝ^(d_model×d_v)   — value projection
```

### Step 1: Project

```
Q = X · W_Q    ∈ ℝ^(B×T×d_k)
K = X · W_K    ∈ ℝ^(B×T×d_k)
V = X · W_V    ∈ ℝ^(B×T×d_v)
```

### Step 2: Compute Raw Scores

```
S = Q · K^T    ∈ ℝ^(B×T×T)

S[b,i,j] = Σ_k Q[b,i,k] · K[b,j,k]
```

`S[b,i,j]` = similarity between token i (as query) and token j (as key).

### Step 3: Scale

```
S' = S / √d_k
```

Variance of each dot product is d_k (for random unit vectors). Division by √d_k normalizes to variance 1.

### Step 4: Softmax

```
A = softmax(S', dim=-1)    ∈ ℝ^(B×T×T)

A[b,i,j] = exp(S'[b,i,j]) / Σ_k exp(S'[b,i,k])
```

Properties:
- Each row sums to 1: `Σ_j A[b,i,j] = 1`
- All values in [0, 1]
- Higher score → higher weight

### Step 5: Weighted Values

```
O = A · V    ∈ ℝ^(B×T×d_v)

O[b,i,:] = Σ_j A[b,i,j] · V[b,j,:]
```

Each output token is a weighted sum of all value vectors.

### Step 6: Output Projection

```
Output = O · W_O    ∈ ℝ^(B×T×d_model)
```

## Why This Works

1. **Q·K^T** measures compatibility
2. **Softmax** converts to probability distribution
3. **A·V** retrieves information proportionally to compatibility
4. **W_O** mixes head outputs (in multi-head)

## Key Takeaway

Attention is differentiable matrix algebra. Every step has a clear mathematical meaning and gradient flow.
