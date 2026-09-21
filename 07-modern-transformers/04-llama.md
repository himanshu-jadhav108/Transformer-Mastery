# 04 — LLaMA

## Architecture

Decoder-only Transformer with several modern improvements.

## Key Innovations

### 1. Pre-Norm with RMSNorm

Replaces LayerNorm with RMSNorm (no mean subtraction, no bias):
```
RMSNorm(x) = x / sqrt(mean(x^2) + eps) * gamma
```

### 2. Rotary Position Embeddings (RoPE)

Instead of adding positional encodings, rotate Q and K vectors by position-dependent angles.

See `08-advanced-concepts/03-rope.md` for details.

### 3. SwiGLU Activation

Replaces ReLU/GELU FFN:
```
SwiGLU(x) = (x * W * SiLU(x * V)) * W2
```

### 4. KV Cache

Store K and V matrices during inference to avoid recomputation.

See `08-advanced-concepts/02-kv-cache.md`.

### 5. Grouped Query Attention (GQA)

Multiple query heads share the same K and V heads, reducing memory bandwidth.

| Variant | Q Heads | K,V Heads | Ratio |
|---------|---------|-----------|-------|
| MHA | 32 | 32 | 1:1 |
| GQA | 32 | 8 | 4:1 |
| MQA | 32 | 1 | 32:1 |

## LLaMA Sizes

| Model | Params | d_model | Heads | Layers | Context |
|-------|--------|---------|-------|--------|---------|
| LLaMA-7B | 7B | 4096 | 32 | 32 | 2k |
| LLaMA-13B | 13B | 5120 | 40 | 40 | 2k |
| LLaMA-30B | 30B | 6656 | 52 | 60 | 2k |
| LLaMA-65B | 65B | 8192 | 64 | 80 | 2k |

## Key Takeaway

LLaMA-style architectures combine multiple efficiency improvements (RMSNorm, RoPE, SwiGLU, GQA) to create high-performance decoder-only models. These choices are now standard in open-source LLMs.
