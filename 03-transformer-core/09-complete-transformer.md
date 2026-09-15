# 09 — Complete Transformer

## Full Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      INPUT (Source)                          │
│                   Token IDs: (B, T_src)                      │
└─────────────────────────────────────────────────────────────┘
                              │
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                         ENCODER                               │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐       │
│  │ Embedding   │───→│ Positional  │───→│  Encoder    │       │
│  │ + Scaling   │    │ Encoding    │    │  ×N layers  │       │
│  └─────────────┘    └─────────────┘    └─────────────┘       │
│                              │                                │
│                         Output: (B, T_src, d_model)           │
└─────────────────────────────────────────────────────────────┘
                              │
                              ↓ K, V
┌─────────────────────────────────────────────────────────────┐
│                         DECODER                               │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐       │
│  │ Embedding   │───→│ Positional  │───→│  Decoder    │       │
│  │ + Scaling   │    │ Encoding    │    │  ×N layers  │       │
│  └─────────────┘    └─────────────┘    └─────────────┘       │
│                              │                                │
│                         Output: (B, T_tgt, d_model)          │
└─────────────────────────────────────────────────────────────┘
                              │
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                      OUTPUT LAYER                             │
│              Linear(d_model, vocab_size)                     │
│                         ↓                                    │
│                       Softmax                                  │
│                         ↓                                    │
│              Probabilities: (B, T_tgt, V)                      │
└─────────────────────────────────────────────────────────────┘
```

## Hyperparameters (Original Paper)

| Parameter | Value |
|-----------|-------|
| d_model | 512 |
| d_ff | 2048 |
| h (heads) | 8 |
| d_k = d_v | 64 |
| N (layers) | 6 |
| dropout | 0.1 |
| vocab_size | ~37k (shared) |

## Training Objective

Cross-entropy loss on next-token prediction:
```
Loss = -Σ log P(y_t | y_{<t}, x)
```

## Inference (Autoregressive Generation)

```
1. Encode source → get K, V
2. Start with <BOS> token
3. Decode one token at a time
4. Append to input, repeat until <EOS>
```

## Key Takeaway

The complete Transformer is an encoder-decoder architecture where the encoder builds bidirectional representations and the decoder generates output autoregressively via cross-attention to the encoder and causal self-attention.
