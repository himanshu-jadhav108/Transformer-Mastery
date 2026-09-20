# 01 — Data Pipeline

## Overview

A training pipeline for Transformers involves:
1. Raw text collection
2. Tokenization
3. Batching and padding
4. Creating input-target pairs

## Dataset Structure

```python
from torch.utils.data import Dataset, DataLoader

class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_len=512):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        tokens = self.tokenizer.encode(text, max_length=self.max_len,
                                       padding='max_length',
                                       truncation=True)
        return torch.tensor(tokens)
```

## Batching Strategy

```python
def collate_fn(batch):
    # batch: list of (T,) tensors
    # Stack to (B, T)
    return torch.stack(batch)

dataloader = DataLoader(dataset, batch_size=32, 
                        shuffle=True, collate_fn=collate_fn)
```

## For Translation (Encoder-Decoder)

```python
def collate_fn_translation(batch):
    # batch: list of (src_tokens, tgt_tokens) pairs
    src = pad_sequence([b[0] for b in batch], batch_first=True)
    tgt = pad_sequence([b[1] for b in batch], batch_first=True)
    return src, tgt
```

## Key Takeaway

The data pipeline converts raw text into batched tensors. Proper padding and masking are critical for correct attention computation.
