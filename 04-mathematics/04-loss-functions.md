# 04 — Loss Functions

## Cross-Entropy Loss

The standard loss for sequence modeling and classification.

### Definition

```
CE(p, q) = -Σ_i p(i) · log(q(i))
```

Where:
- `p` = true distribution (one-hot for classification)
- `q` = predicted distribution (softmax output)

### For Language Modeling

```
Loss = -1/N Σ_t log P(y_t | y_{<t}, x)
```

Average negative log-likelihood of correct next tokens.

### In PyTorch

```python
criterion = nn.CrossEntropyLoss()
# logits: (B*T, vocab_size)
# targets: (B*T,)
loss = criterion(logits.view(-1, vocab_size), targets.view(-1))
```

## Label Smoothing

Prevents overconfidence:
```
Loss = -Σ_i [(1-ε)·p(i) + ε/V] · log(q(i))
```

Where `ε` (e.g., 0.1) spreads some probability mass uniformly.

## Why Cross-Entropy?

1. **Maximum likelihood**: Minimizing CE = maximizing likelihood
2. **Information theory**: Minimizes bits needed to encode true labels
3. **Well-behaved gradients**: Smooth, convex-ish landscape

## Tensor Shapes

```
Logits:     (B, T, vocab_size)
Targets:    (B, T)  — integer token IDs
Loss:       scalar
```

## Key Takeaway

Cross-entropy loss measures how well the model's predicted probability distribution matches the true distribution. For language modeling, it's the negative log-probability of the correct next token.
