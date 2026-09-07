# 04 — RNN (Recurrent Neural Network)

## What is an RNN?

An RNN processes sequences by maintaining a hidden state that updates at each time step.

## Architecture

```
At each time step t:
  h_t = tanh(W_hh · h_{t-1} + W_xh · x_t + b_h)
  y_t = W_hy · h_t + b_y
```

## Diagram

```
    x₁ → [RNN] → h₁ → [RNN] → h₂ → ... → h_T
              ↑_________↑
              hidden state passed forward
```

## Why RNNs Were Popular

- Natural fit for sequences
- Variable-length input handling
- Shared parameters across time

## Critical Problems

1. **Vanishing gradients**: Gradients shrink exponentially as they backprop through time
2. **Exploding gradients**: Gradients grow exponentially (less common)
3. **Slow training**: Cannot parallelize — must process sequentially
4. **Long-range dependency failure**: Information from early tokens gets diluted

## Tensor Shapes

```
x_t:   (B, input_size)
h_t:   (B, hidden_size)
y_t:   (B, output_size)
W_hh:  (hidden_size, hidden_size)
W_xh:  (input_size, hidden_size)
```

## Key Takeaway

RNNs introduced the idea of maintaining state across a sequence, but their sequential nature and gradient problems made them unsuitable for long sequences and large-scale training.
