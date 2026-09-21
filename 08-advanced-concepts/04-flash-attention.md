# 04 — Flash Attention

## The Problem

Standard attention:
1. Compute S = QK^T → materialize (B, T, T) matrix
2. Apply softmax → materialize another (B, T, T)
3. Multiply with V → materialize output

Memory: O(T²) — becomes the bottleneck for long sequences.

## Solution: Tiling + Recomputation

Flash Attention computes attention in tiles, keeping data in fast SRAM (GPU shared memory) and avoiding materialization of the full attention matrix.

## Algorithm (Simplified)

```python
def flash_attention(Q, K, V, block_size):
    # Q, K, V: (T, d_k) — single sequence for clarity
    O = zeros(T, d_k)
    L = zeros(T)  # log-sum-exp for softmax

    for i in range(0, T, block_size):  # Tile over rows
        Qi = Q[i:i+block_size]

        for j in range(0, T, block_size):  # Tile over columns
            Kj = K[j:j+block_size]
            Vj = V[j:j+block_size]

            Sij = Qi @ Kj.T  # Small tile: (block, block)
            Pij = exp(Sij - row_max(Sij))

            # Online softmax: update running statistics
            # ... (complex state management)

            O[i:i+block_size] += Pij @ Vj

    return O
```

## Benefits

| Metric | Standard | Flash Attention |
|--------|----------|----------------|
| Memory | O(T²) | O(T) |
| IO operations | O(T²) | O(T²) but fused |
| Speed | Baseline | 2-4x faster |
| Exact? | Yes | Yes (mathematically identical) |

## Flash Attention v2

- Better parallelism across sequence length
- Reduced non-matmul operations
- Up to 2x faster than v1

## Usage

```python
from flash_attn import flash_attn_func

output = flash_attn_func(q, k, v, dropout_p=0.0, causal=True)
# q, k, v: (B, T, h, d_k) — note: heads in dim 2
```

## Key Takeaway

Flash Attention reduces attention memory from O(T²) to O(T) by tiling and fusing operations in GPU SRAM. It produces exact same results as standard attention but much faster.
