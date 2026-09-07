# 02 — Word Embeddings

## What Are Word Embeddings?

Word embeddings map discrete tokens (words/subwords) to continuous vector spaces where **semantic similarity = geometric proximity**.

## Why We Need Them

Neural networks operate on numbers, not text. Embeddings bridge this gap.

## How They Work

1. Build a vocabulary of size `V`
2. Create an embedding matrix `E` of shape `(V, d_model)`
3. Each token ID `i` maps to row `E[i]`
4. During training, `E` is learned end-to-end

## Mathematical View

```
Input: token IDs      →  shape (B, T)
Embedding lookup: E[x] →  shape (B, T, d_model)
```

## Example

```python
import torch
import torch.nn as nn

vocab_size = 10000
d_model = 512

embedding = nn.Embedding(vocab_size, d_model)
tokens = torch.randint(0, vocab_size, (2, 10))  # (B=2, T=10)
embeddings = embedding(tokens)                   # (2, 10, 512)
```

## Intuition

Think of embeddings as a "meaning coordinate system." Words with similar meanings cluster together. The model learns these coordinates during training.

## Common Mistake

Confusing embedding lookup with one-hot encoding. Embeddings are **learned dense vectors**, not fixed sparse indicators.

## Key Takeaway

Embeddings turn tokens into vectors. The rest of the Transformer operates on these vectors.
