# 01 — Neural Networks Basics for Transformers

## What is a Neural Network?

A neural network is a function approximator composed of layers of neurons. Each neuron: `output = activation(weight · input + bias)`.

## Why Transformers Need Deep Networks

Transformers are **deep feed-forward networks with attention**. You need to understand:

- **Linear layers**: `y = xW + b` — the backbone of every component
- **Activation functions**: ReLU, GELU (used in Transformers)
- **Layer stacking**: Depth enables hierarchical feature learning
- **Backpropagation**: How gradients flow through the network

## Key Concepts for Transformers

| Concept | Role in Transformers |
|---------|---------------------|
| Linear projection | Creates Q, K, V matrices; output projections |
| Matrix multiplication | Attention scores, feed-forward layers |
| Softmax | Converts attention scores to probabilities |
| Gradient descent | Training the entire model |
| Loss functions | Cross-entropy for next-token prediction |

## Tensor Shape Basics

```
Input:  (B, T, C)     — batch of token sequences
Weight: (C, C_out)    — learned parameters
Output: (B, T, C_out) — transformed representations
```

## Key Takeaway

Transformers are built entirely from linear algebra operations. Master matrix multiplication and you master 80% of the architecture.
