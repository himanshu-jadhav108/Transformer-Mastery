# Project 4: Mini GPT From Scratch

## Objective
Build, train, and generate text with a GPT-style model entirely from scratch.

## Concepts Used
- All previous concepts
- Full training pipeline
- Generation strategies

## Architecture
```
GPT-style Decoder:
- d_model = 384
- num_layers = 6
- num_heads = 6
- d_ff = 1536
- context_length = 256
- vocab_size = 10,000 (BPE)
```

## Steps
1. Implement full decoder-only Transformer in PyTorch
2. Prepare dataset (TinyShakespeare, OpenWebText subset)
3. Train with AdamW + cosine schedule
4. Implement generation: greedy, temperature, top-k
5. Evaluate perplexity

## Training Config
```python
batch_size = 64
learning_rate = 3e-4
max_iters = 5000
warmup_steps = 500
eval_interval = 500
```

## Expected Learning Outcome
- End-to-end understanding of Transformer training
- Experience with hyperparameter tuning
- Understanding of generation quality vs. model size

## Possible Extensions
- Add RoPE instead of absolute PE
- Implement gradient accumulation for larger effective batch
- Add WandB logging
- Export to ONNX for inference optimization
