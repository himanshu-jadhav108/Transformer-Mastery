# 02 — Tokenization

## What is Tokenization?

Splitting text into subword units (tokens) that the model processes.

## Common Algorithms

### 1. Byte-Pair Encoding (BPE)

Iteratively merges the most frequent character pairs.

```
Initial: l o w </w>  l o w e r </w>  n e w e s t </w>
Merge 'e','s' → 'es':  newest → n e w es t </w>
Merge 'es','t' → 'est': newest → n e w est </w>
Merge 'l','o' → 'lo':   low → lo w </w>
...
```

Used in: GPT-2, RoBERTa

### 2. WordPiece

Similar to BPE but merges based on likelihood, not frequency.

Used in: BERT, DistilBERT

### 3. SentencePiece

Treats text as raw sequence of characters (language-agnostic).

Used in: T5, LLaMA, most modern models

### 4. TikToken (GPT-4)

BPE with optimized regex pre-tokenization.

## Vocabulary Size

Typical sizes: 32k (T5), 50k (GPT-2, BERT), 100k+ (modern LLMs)

## Special Tokens

| Token | Purpose |
|-------|---------|
| `<pad>` | Padding |
| `<sos>` / `<bos>` | Start of sequence |
| `<eos>` | End of sequence |
| `<unk>` | Unknown token |
| `<mask>` | Masked token (BERT) |

## In PyTorch

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
tokens = tokenizer("Hello world", return_tensors="pt")
# tokens['input_ids']: (1, 3)
# tokens['attention_mask']: (1, 3)
```

## Key Takeaway

Tokenization determines the model's vocabulary. Subword tokenization (BPE, WordPiece) balances vocabulary size and coverage. The tokenizer is as important as the model architecture.
