# 05 — Multi-Head Attention

## The Problem with Single-Head Attention

One attention pattern is insufficient. Language has multiple relationship types:
- Syntactic (subject-verb agreement)
- Semantic (word meaning)
- Coreference ("it" → "cat")
- Positional (nearby words)

## Solution: Multiple Heads

Run attention **h times in parallel** with different learned projections, then concatenate and project.

## Formula

```
MultiHead(Q, K, V) = Concat(head_1, ..., head_h) · W_O

where head_i = Attention(Q·W_Qi, K·W_Ki, V·W_Vi)
```

## Tensor Shapes

```
Input:          (B, T, d_model)

Per head:
  W_Qi:         (d_model, d_k)    where d_k = d_model / h
  W_Ki:         (d_model, d_k)
  W_Vi:         (d_model, d_v)    where d_v = d_model / h
  Q_i:          (B, T, d_k)
  K_i:          (B, T, d_k)
  V_i:          (B, T, d_v)
  head_i:       (B, T, d_v)

Concatenation:  (B, T, h × d_v) = (B, T, d_model)
W_O:            (d_model, d_model)
Final output:   (B, T, d_model)
```

## Diagram

```
        ┌─────────────┐
   X ──→│ Split into  │──→ Q₁,K₁,V₁ ──→ Attention₁ ──┐
        │   h heads   │──→ Q₂,K₂,V₂ ──→ Attention₂ ──┤
        └─────────────┘   ...                          ├──→ Concat ──→ Linear ──→ Output
                                    ──→ Attention_h ──┘
```

## Why Concatenate Then Project?

- Concatenation preserves information from all heads
- Final linear projection `W_O` mixes head outputs, allowing cross-head interaction

## Number of Heads

Typical values: `h = 8` or `h = 12` for `d_model = 512` or `768`.

Rule of thumb: `d_k = d_model / h` should be ≥ 64 for stable training.

## Common Mistake

Forgetting that multi-head attention is NOT just running attention h times on the same Q,K,V. Each head has **independent learned projections**.

## Key Takeaway

Multi-head attention allows the model to jointly attend to information from different representation subspaces at different positions. Each head specializes on different relationship patterns.
