# 02 — GPT

## Architecture

**Decoder-only**, causal Transformer.

```
Input → [Embedding + Positional] → [Decoder xN] → [LM Head] → Next Token
```

## Key Characteristics

1. **Causal attention**: Each token only attends to previous tokens
2. **Autoregressive**: Predicts one token at a time
3. **Left-to-right**: Unidirectional context

## Pre-training

**Objective**: Next token prediction
```
Loss = -log P(x_t | x_{<t})
```

## Generation

```python
def generate(model, prompt, max_new_tokens=100):
    tokens = tokenizer.encode(prompt)
    for _ in range(max_new_tokens):
        logits = model(tokens)  # (1, T, V)
        next_token_logits = logits[0, -1, :]  # Last position
        probs = softmax(next_token_logits)
        next_token = sample(probs)
        tokens.append(next_token)
    return tokenizer.decode(tokens)
```

## GPT Evolution

| Model | Year | Params | Key Feature |
|-------|------|--------|-------------|
| GPT-1 | 2018 | 117M | First decoder-only pre-training |
| GPT-2 | 2019 | 1.5B | Zero-shot transfer |
| GPT-3 | 2020 | 175B | In-context learning |
| GPT-4 | 2023 | ? | Multimodal, RLHF |

## Tensor Shapes

```
Input:      (B, T)
Embedding:  (B, T, d_model)
Output:     (B, T, d_model)
LM Head:    (B, T, vocab_size)
```

## Key Takeaway

GPT showed that a simple objective (next-token prediction) with enough scale produces emergent capabilities. Decoder-only models dominate modern LLMs due to their simplicity and scalability.
