# 01 — Tensor Basics for Transformers

## Understanding Tensor Shapes

Transformers manipulate tensors with 3-4 dimensions. Mastering shapes is essential.

## Standard Shape Convention

| Symbol | Meaning | Example |
|--------|---------|---------|
| B | Batch size | 32 |
| T | Sequence length (tokens) | 512 |
| C or d_model | Embedding dimension | 768 |
| h | Number of attention heads | 12 |
| d_k | Query/key dimension per head | 64 |
| d_v | Value dimension per head | 64 |
| V | Vocabulary size | 50,000 |

## Common Tensor Shapes

```
Token IDs:           (B, T)
Embeddings:          (B, T, d_model)
Q, K, V (single):    (B, T, d_model)
Q, K, V (per head):  (B, T, d_k)
Attention scores:    (B, T, T)
Attention weights:   (B, T, T)
Attention output:    (B, T, d_v)
FFN hidden:          (B, T, d_ff)
Logits:              (B, T, V)
```

## Multi-Head Reshaping

```python
# From (B, T, d_model) to (B, h, T, d_k)
x = x.view(B, T, h, d_k).transpose(1, 2)
# Result: (B, h, T, d_k)

# Back to (B, T, d_model)
x = x.transpose(1, 2).contiguous().view(B, T, d_model)
```

## Broadcasting Rules

PyTorch automatically expands dimensions of size 1:
```python
# (B, T, T) + (B, 1, T) -> (B, T, T)
# (B, h, T, T) + (1, 1, T, T) -> (B, h, T, T)
```

## Key Takeaway

Always track tensor shapes. If you know the shapes, you know the architecture. Draw shape diagrams for every component you implement.
