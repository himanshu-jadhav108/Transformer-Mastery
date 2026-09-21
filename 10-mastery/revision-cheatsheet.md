# Revision Cheatsheet

## Core Formula
```
Attention(Q,K,V) = softmax(QK^T / sqrt(d_k)) * V
```

## Tensor Shapes
| Tensor | Shape |
|--------|-------|
| Input | (B, T, d_model) |
| Q, K, V | (B, T, d_k) |
| QK^T | (B, T, T) |
| Attention Weights | (B, T, T) |
| Output | (B, T, d_v) |
| Multi-head concat | (B, T, d_model) |

## Architecture Components

| Component | Purpose |
|-----------|---------|
| Embedding | Token -> vector |
| Positional Encoding | Inject order |
| Self-Attention | Contextualize |
| Cross-Attention | Align sequences |
| FFN | Add capacity |
| LayerNorm | Stabilize |
| Residual | Gradient highway |
| Masking | Control information flow |

## Model Types

| Model | Type | Attention | Best For |
|-------|------|-----------|----------|
| BERT | Encoder-only | Bidirectional | Understanding |
| GPT | Decoder-only | Causal | Generation |
| T5 | Encoder-decoder | Both | Translation, summarization |

## Key Hyperparameters

| Param | Typical Value |
|-------|--------------|
| d_model | 512-4096 |
| d_ff | 4 x d_model |
| h (heads) | 8-128 |
| d_k = d_v | d_model / h |
| N (layers) | 6-96 |
| dropout | 0.1 |

## Modern Improvements

| Technique | Benefit |
|-----------|---------|
| RoPE | Better relative positions |
| RMSNorm | Simpler, stable normalization |
| SwiGLU | Better activation |
| GQA | Reduced KV memory |
| Flash Attention | O(T) memory attention |
| KV Cache | Fast autoregressive inference |
| MoE | Scale parameters efficiently |
