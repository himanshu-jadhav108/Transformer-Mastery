# 01 — BERT

## Architecture

**Encoder-only**, bidirectional Transformer.

```
Input → [Embedding + Positional] → [Encoder x12] → [Pooler] → Output
```

## Key Innovations

1. **Bidirectional attention**: Every token sees every other token
2. **Pre-training + Fine-tuning**: Train once, adapt to many tasks

## Pre-training Tasks

### Masked Language Modeling (MLM)
```
Input:  "The [MASK] sat on the mat"
Target: "cat"

Randomly mask 15% of tokens:
- 80% → [MASK]
- 10% → random token
- 10% → unchanged
```

### Next Sentence Prediction (NSP)
```
Input:  [CLS] Sentence A [SEP] Sentence B [SEP]
Target: Is B the actual next sentence? (Yes/No)
```

(Note: NSP was later found less useful; RoBERTa removed it.)

## Fine-Tuning

| Task | Output Layer |
|------|-------------|
| Classification | Linear(d_model, num_classes) on [CLS] token |
| NER | Linear(d_model, num_labels) per token |
| QA | Start/End span prediction |
| Sentence similarity | Cosine similarity of [CLS] vectors |

## Tensor Shapes

```
Input:      (B, T) token IDs
Embedding:  (B, T, 768)  # d_model=768 for BERT-base
Output:     (B, T, 768)
[CLS]:      (B, 768)     # First token for classification
```

## BERT Variants

| Model | Layers | Hidden | Heads | Parameters |
|-------|--------|--------|-------|-----------|
| BERT-Base | 12 | 768 | 12 | 110M |
| BERT-Large | 24 | 1024 | 16 | 340M |

## Key Takeaway

BERT proved that bidirectional pre-training creates powerful contextual representations. Fine-tuning adapts these representations to downstream tasks with minimal task-specific parameters.
