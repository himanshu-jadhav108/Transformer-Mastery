# 06 — Multimodal Transformers

## Core Idea

Process multiple modalities (text, image, audio, video) with a single Transformer.

## Common Approaches

### 1. Early Fusion

Concatenate/embed all modalities into a single sequence:
```
[Image patches] [Text tokens] [Audio tokens]
       ↓
   Shared Transformer
       ↓
  Unified output
```

### 2. Late Fusion

Process each modality separately, then fuse:
```
Image → Vision Encoder →
                         → Cross-Attention → Output
Text  → Text Encoder   →
```

### 3. Encoder-Decoder

```
Image → Encoder → K, V
Text  → Decoder → Q → Cross-attention → Generate caption
```

## Examples

| Model | Modalities | Architecture |
|-------|-----------|-------------|
| CLIP | Text + Image | Dual encoders, contrastive |
| DALL-E | Text → Image | Decoder-only, discrete VAE |
| BLIP | Text + Image | Encoder-decoder + ITM/ITC |
| GPT-4V | Text + Image | Unified decoder |
| LLaVA | Text + Image | Projector + LLM |

## CLIP (Contrastive Learning)

```python
# Image encoder + Text encoder
image_features = image_encoder(images)   # (B, d)
text_features = text_encoder(texts)        # (B, d)

# Normalize
image_features = F.normalize(image_features, dim=-1)
text_features = F.normalize(text_features, dim=-1)

# Contrastive loss: maximize diagonal of similarity matrix
similarity = image_features @ text_features.T  # (B, B)
loss = cross_entropy(similarity, labels) + cross_entropy(similarity.T, labels)
```

## Key Takeaway

Multimodal Transformers extend the attention mechanism across modalities. The core idea remains the same: compute compatibility between representations, regardless of their source modality.
