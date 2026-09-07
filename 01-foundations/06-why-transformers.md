# 06 — Why Transformers?

## The Problem with RNNs/LSTM/GRU

1. **Sequential bottleneck**: O(T) time per layer — cannot parallelize
2. **Long-range dependency decay**: Information from distant tokens weakens
3. **Difficult to scale**: Adding depth increases training time linearly with sequence length

## The Attention Insight (Bahdanau et al., 2015)

What if every token could **directly attend to every other token** in a single step?

## Why Transformers Win

| Property | RNN Family | Transformer |
|----------|-----------|-------------|
| Parallelization | ❌ Sequential | ✅ Fully parallel |
| Long-range deps | ❌ Weak | ✅ Direct connections |
| Training speed | ❌ O(T) per layer | ✅ O(1) per layer (matrix ops) |
| Scalability | ❌ Limited | ✅ Excellent |
| Interpretability | ❌ Black box | ✅ Attention weights visible |

## The Transformer Revolution

"Attention Is All You Need" (Vaswani et al., 2017) showed that **attention alone** — without recurrence — can model sequences effectively.

## Core Idea

Replace sequential hidden state updates with **global attention**:
- Each token computes relationships with ALL other tokens simultaneously
- Information flows directly between any two positions
- Everything is matrix multiplication → GPU-friendly

## Trade-off

Transformers have O(T²) attention complexity (quadratic in sequence length). This is the main limitation modern research addresses (Flash Attention, sparse attention, linear attention).

## Key Takeaway

Transformers replaced recurrence with attention, enabling parallel training and direct long-range dependencies. This architectural choice is why modern LLMs exist.
