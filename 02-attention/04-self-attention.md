# 04 — Self-Attention

## What is Self-Attention?

Each token in the sequence attends to **all tokens in the same sequence** (including itself).

## Why "Self"?

Q, K, V all come from the **same input sequence**. The sequence attends to itself to build contextualized representations.

## Diagram

```
Input: [The] [cat] [sat]
         ↓     ↓     ↓
       ┌─────────────────┐
       │  Self-Attention  │
       └─────────────────┘
         ↓     ↓     ↓
Output: [The*] [cat*] [sat*]

* = contextualized representation
```

## How It Works

For each position i:
1. Compute Q_i (what token i is "asking")
2. Compute similarity with all K_j (what every token "offers")
3. Weight all V_j by these similarities
4. Output_i = weighted sum of all values

## Example: Contextualization

```
Input embeddings (before attention):
  "bank" = [0.2, 0.5, ...]  (same for river bank and money bank)

After self-attention:
  "river bank" → "bank" attends to "river" → representation shifts
  "bank account" → "bank" attends to "account" → different shift
```

## Tensor Shapes (Self-Attention)

```
X:      (B, T, d_model)
Q,K,V:  (B, T, d_k)   where d_k = d_model (single head)
Scores: (B, T, T)
Output: (B, T, d_v)   where d_v = d_model
```

## Self-Attention vs Cross-Attention

| | Self-Attention | Cross-Attention |
|---|----------------|-----------------|
| Q source | Same sequence | Decoder sequence |
| K,V source | Same sequence | Encoder sequence |
| Purpose | Contextualize within sequence | Align two sequences |
| Used in | Encoder, Decoder | Decoder only |

## Common Mistake

Thinking self-attention only attends to previous tokens. In the **encoder**, self-attention is **bidirectional** (attends to all tokens). Only the **decoder** uses causal masking to attend to previous tokens only.

## Key Takeaway

Self-attention transforms static embeddings into context-aware representations. Each token's new meaning is a weighted blend of all tokens' meanings, with weights learned from data.
