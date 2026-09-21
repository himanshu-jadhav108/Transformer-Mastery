# 06 — Efficient Attention

## Beyond O(T²): Linear Attention Variants

### 1. Sparse Attention

Only attend to a subset of positions:

| Pattern | Attends to |
|---------|-----------|
| Strided | Every k-th position |
| Local/Sliding window | Nearby positions |
| Dilated | Sparse pattern within window |
| Random | Random subset |
| Factorized | Combination of patterns |

### 2. Linear Attention

Reformulate attention to avoid explicit O(T²) matrix:
```
Standard:  softmax(QK^T)V
Linear:    φ(Q) · (φ(K)^T · V)  where φ is a feature map
```

Complexity: O(T·d²) instead of O(T²·d)

### 3. Performer

Uses random feature maps (FAVOR+) to approximate softmax attention.

### 4. Linformer

Approximates K, V with low-rank projections:
```
K' = K · E_k  where E_k: (T, k) and k << T
V' = V · E_v
Attention = softmax(Q · K'^T) · V'
```

### 5. Longformer / BigBird

Combination of:
- Global attention (attend to all from specific tokens)
- Local sliding window
- Random attention

## Comparison

| Method | Complexity | Exact? | Best For |
|--------|-----------|--------|----------|
| Standard | O(T²) | Yes | General |
| Flash Attention | O(T²) compute, O(T) memory | Yes | Long sequences |
| Linear Attention | O(T) | Approximate | Very long sequences |
| Sparse | O(T·w) | Yes | Structured data |

## Key Takeaway

Standard attention is O(T²). For very long sequences, sparse patterns, linear approximations, or Flash Attention are necessary. The choice depends on whether exactness or scalability is more important.
