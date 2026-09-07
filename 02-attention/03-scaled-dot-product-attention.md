# 03 — Scaled Dot-Product Attention

## The Formula

```
Attention(Q, K, V) = softmax(QK^T / √d_k) · V
```

## Step-by-Step Breakdown

### Step 1: Compute Similarity Scores

```
Scores = Q · K^T
```

- Q shape: `(B, T, d_k)`
- K^T shape: `(B, d_k, T)`
- Scores shape: `(B, T, T)`

Each element `Scores[i, j]` = similarity between query at position i and key at position j.

### Step 2: Scale

```
Scaled = Scores / √d_k
```

**Why scale?** For large `d_k`, dot products grow large, pushing softmax into regions with extremely small gradients. Scaling stabilizes training.

**Intuition**: Variance of dot product of two random vectors of dimension d_k is d_k. Dividing by √d_k normalizes variance to 1.

### Step 3: Apply Softmax

```
AttentionWeights = softmax(Scaled, dim=-1)
```

Converts scores to probabilities (row-wise). Each row sums to 1.

Shape: `(B, T, T)`

### Step 4: Weighted Sum of Values

```
Output = AttentionWeights · V
```

- AttentionWeights: `(B, T, T)`
- V: `(B, T, d_v)`
- Output: `(B, T, d_v)`

Each output position is a weighted combination of all value vectors.

## Complete Tensor Flow

```
Input X:        (B, T, d_model)
  ↓
Q = XW_Q:       (B, T, d_k)
K = XW_K:       (B, T, d_k)
V = XW_V:       (B, T, d_v)
  ↓
QK^T:           (B, T, T)
  ↓
Scale /√d_k:    (B, T, T)
  ↓
Softmax:        (B, T, T)  ← attention weights
  ↓
· V:            (B, T, d_v) ← output
  ↓
Linear proj:    (B, T, d_model)
```

## Numerical Example

```
d_k = 4
Q = [[1, 0, 1, 0]]        (1, 4)
K = [[1, 0, 1, 0],        (2, 4)
     [0, 1, 0, 1]]
V = [[10, 20],            (2, 2)
     [30, 40]]

QK^T = [[2, 0]]           (1, 2)
Scale: [[2/2, 0/2]] = [[1, 0]]
Softmax: [[0.73, 0.27]]
Output: 0.73·[10,20] + 0.27·[30,40] = [15.4, 25.4]
```

## Why √d_k and Not d_k?

- `d_k` would over-normalize (too small values)
- `√d_k` preserves the right scale for gradient flow
- Empirically validated in the original paper

## Key Takeaway

Scaled dot-product attention computes a similarity matrix, normalizes it, and uses it to create weighted combinations of values. The entire operation is differentiable and parallelizable.
