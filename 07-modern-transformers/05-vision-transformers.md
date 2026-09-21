# 05 — Vision Transformers (ViT)

## Core Idea

Apply Transformers to images by treating image patches as tokens.

## Pipeline

```
Image (224x224x3)
  ↓
Split into 16x16 patches → 196 patches
  ↓
Flatten + Linear projection → Patch embeddings (196, d_model)
  ↓
Add [CLS] token + positional embeddings
  ↓
Transformer Encoder x12
  ↓
[CLS] token → MLP Head → Classification
```

## Implementation

```python
class PatchEmbedding(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_ch=3, d_model=768):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_ch, d_model, 
                              kernel_size=patch_size, 
                              stride=patch_size)

    def forward(self, x):
        # x: (B, 3, 224, 224)
        x = self.proj(x)  # (B, 768, 14, 14)
        x = x.flatten(2)  # (B, 768, 196)
        x = x.transpose(1, 2)  # (B, 196, 768)
        return x
```

## Tensor Shapes

```
Image:          (B, 3, 224, 224)
Patches:        (B, 196, 768)
[CLS] token:    (B, 1, 768)
Input to enc:   (B, 197, 768)
Encoder output: (B, 197, 768)
[CLS] output:   (B, 768)
Classification: (B, num_classes)
```

## Why It Works

1. **Patches as tokens**: Image patches are analogous to word tokens
2. **Global attention**: Every patch attends to every other patch
3. **Scales well**: Larger models + more data → better performance

## Key Takeaway

ViT proved that pure attention (no CNN) can achieve state-of-the-art vision performance when trained on large datasets. The patch embedding is the critical adaptation from NLP to vision.
