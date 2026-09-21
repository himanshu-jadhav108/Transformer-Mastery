# 02 — KV Cache

## The Problem

During autoregressive generation, we recompute K and V for ALL previous tokens at every step.

For sequence length T, total computation: O(1 + 2 + 3 + ... + T) = O(T²)

## Solution: Cache K and V

Store K and V matrices after computing them. Reuse in subsequent steps.

## Implementation

```python
class KVCache:
    def __init__(self, max_batch_size, max_seq_len, 
                 num_heads, head_dim):
        self.k_cache = torch.zeros(
            max_batch_size, num_heads, max_seq_len, head_dim
        )
        self.v_cache = torch.zeros(
            max_batch_size, num_heads, max_seq_len, head_dim
        )
        self.seq_len = 0

    def update(self, k, v):
        # k, v: (B, h, 1, d_k) for new token
        self.k_cache[:, :, self.seq_len, :] = k
        self.v_cache[:, :, self.seq_len, :] = v
        self.seq_len += 1

        # Return all cached K, V
        return (self.k_cache[:, :, :self.seq_len, :],
                self.v_cache[:, :, :self.seq_len, :])
```

## Complexity Comparison

| Method | Per-step Compute | Memory |
|--------|-----------------|--------|
| No cache | O(T²) | O(T) |
| KV Cache | O(T) | O(T) |

## Memory Cost

For batch B, heads h, seq T, dim d_k:
```
Memory = 2 * B * h * T * d_k * 2 bytes (FP16)
```

For LLaMA-7B (B=1, h=32, d_k=128, T=4096):
```
~ 2 * 1 * 32 * 4096 * 128 * 2 = 64 MB per layer
* 32 layers = ~2 GB
```

## Key Takeaway

KV Cache reduces per-step computation from O(T²) to O(T) during inference. It is essential for efficient autoregressive generation but increases memory usage proportionally to sequence length.
