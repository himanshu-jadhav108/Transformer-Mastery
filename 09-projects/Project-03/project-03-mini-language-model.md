# Project 3: Mini Language Model

## Objective
Build a decoder-only Transformer that generates text.

## Concepts Used
- Causal self-attention
- Next-token prediction
- Autoregressive generation

## Architecture
```
Input -> Embedding -> Decoder x6 -> Linear -> Softmax -> Next Token
```

## Steps
1. Collect a small text corpus (e.g., Shakespeare, Wikipedia subset)
2. Tokenize with BPE or character-level
3. Build Decoder-only Transformer
4. Train with next-token prediction
5. Generate text with sampling

## Expected Learning Outcome
- Understand autoregressive training
- Implement causal masking correctly
- Experience training instability and debugging

## Possible Extensions
- Implement temperature sampling and top-k
- Add KV Cache for faster generation
- Train on a larger corpus
