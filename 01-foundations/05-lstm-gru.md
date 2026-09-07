# 05 — LSTM and GRU

## What Are LSTM and GRU?

Gated variants of RNNs designed to solve the vanishing gradient problem.

## LSTM (Long Short-Term Memory)

Introduces a **cell state** (long-term memory) and three gates:

| Gate | Function | Equation |
|------|----------|----------|
| Forget gate | What to discard from cell state | `f_t = σ(W_f · [h_{t-1}, x_t] + b_f)` |
| Input gate | What to store in cell state | `i_t = σ(W_i · [h_{t-1}, x_t] + b_i)` |
| Output gate | What to output | `o_t = σ(W_o · [h_{t-1}, x_t] + b_o)` |

Cell state update:
```
C_t = f_t ⊙ C_{t-1} + i_t ⊙ tanh(W_C · [h_{t-1}, x_t] + b_C)
h_t = o_t ⊙ tanh(C_t)
```

## GRU (Gated Recurrent Unit)

Simplified version with two gates:

| Gate | Function |
|------|----------|
| Update gate | Balance old and new information |
| Reset gate | How much past state to forget |

```
z_t = σ(W_z · [h_{t-1}, x_t])
r_t = σ(W_r · [h_{t-1}, x_t])
h_t = (1 - z_t) ⊙ h_{t-1} + z_t ⊙ tanh(W · [r_t ⊙ h_{t-1}, x_t])
```

## Comparison

| Feature | RNN | LSTM | GRU |
|---------|-----|------|-----|
| Gates | 0 | 3 | 2 |
| Parameters | Fewest | Most | Medium |
| Long-range deps | Poor | Good | Good |
| Training speed | Slow | Slower | Medium |
| Parallelization | No | No | No |

## Key Takeaway

LSTM and GRU improved RNNs but did NOT solve the fundamental problem: **sequential processing prevents parallelization and limits scalability.**
