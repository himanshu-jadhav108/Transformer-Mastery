# 04 — Cross-Attention

## What is Cross-Attention?

Attention where **Q comes from one sequence** and **K, V come from another**.

## In the Transformer

```
Q  ← Decoder output (what the decoder wants to know)
K  ← Encoder output (what the source contains)
V  ← Encoder output (what information to retrieve)
```

## Purpose

Align the decoder's generation with the encoder's understanding of the source.

## Example: Translation

```
Source (Encoder): "The cat sat on the mat"
Target (Decoder): "Le chat ..."

When generating "chat" (cat), the decoder queries the encoder.
Cross-attention weights will be high on "cat" in the source.
```

## Tensor Shapes

```
Q (decoder):  (B, T_dec, d_model)
K (encoder):  (B, T_enc, d_model)
V (encoder):  (B, T_enc, d_model)

W_Q: (d_model, d_k)
W_K: (d_model, d_k)
W_V: (d_model, d_v)

Q_proj:       (B, T_dec, d_k)
K_proj:       (B, T_enc, d_k)
V_proj:       (B, T_enc, d_v)

Scores:       (B, T_dec, T_enc)
Output:       (B, T_dec, d_v)
```

## Diagram

```
  Decoder State          Encoder Output
       │                       │
       ↓                       ↓
      [Q] ←──── Attention ──→ [K, V]
       │                       ↑
       └─────── Output ────────┘
```

## Self-Attention vs Cross-Attention

| | Self-Attention | Cross-Attention |
|---|----------------|-----------------|
| Q source | Same as K,V | Different from K,V |
| In Encoder | Yes | No |
| In Decoder | Yes (masked) | Yes |
| Purpose | Contextualize | Align sequences |

## Key Takeaway

Cross-attention is the bridge between encoder and decoder. It allows the decoder to "ask questions" about the source representation when generating each output token.
