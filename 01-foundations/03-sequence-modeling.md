# 03 — Sequence Modeling

## What is Sequence Modeling?

Predicting the next element (token) in a sequence given previous elements.

## Why It's Hard

- Variable-length inputs
- Long-range dependencies
- Context matters (order is crucial)

## Example

```
Input:  "The cat sat on the ..."
Target: "mat"
```

The model must understand grammar, semantics, and world knowledge to predict correctly.

## Formal Definition

Given sequence `x = (x₁, x₂, ..., x_T)`, model learns `P(x_t | x₁, ..., x_{t-1})`.

## Approaches Before Transformers

| Approach | Mechanism | Limitation |
|----------|-----------|------------|
| N-grams | Count-based | No long context |
| RNN | Hidden state | Sequential, slow |
| LSTM/GRU | Gated hidden state | Still sequential |

## The Core Problem

**Sequential processing is a bottleneck.** RNNs process one token at a time. For a sequence of length T, they need T steps.

## Key Takeaway

Sequence modeling requires understanding context. The challenge is doing so efficiently for long sequences.
