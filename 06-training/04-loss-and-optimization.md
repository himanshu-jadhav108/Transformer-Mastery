# 04 — Loss and Optimization

## Loss Function

```python
criterion = nn.CrossEntropyLoss(ignore_index=pad_token_id)
```

`ignore_index` skips padding tokens in loss computation.

## Optimizer: Adam / AdamW

```python
optimizer = torch.optim.AdamW(model.parameters(), 
                               lr=1e-4, 
                               betas=(0.9, 0.98),
                               eps=1e-9,
                               weight_decay=0.01)
```

| Parameter | Value | Purpose |
|-----------|-------|---------|
| lr | 1e-4 to 5e-4 | Learning rate |
| betas | (0.9, 0.98) | Momentum coefficients |
| eps | 1e-9 | Numerical stability |
| weight_decay | 0.01 | L2 regularization |

## Why AdamW?

AdamW decouples weight decay from gradient updates, providing better regularization than standard Adam.

## Gradient Clipping

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

Prevents exploding gradients in early training.

## Mixed Precision Training

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

with autocast():
    output = model(src, tgt)
    loss = criterion(output, targets)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

Uses FP16 for forward/backward, FP32 for updates. ~2x speedup, ~50% memory savings.

## Key Takeaway

AdamW with warmup, gradient clipping, and mixed precision is the standard optimization setup for Transformer training. Weight decay and learning rate scheduling are critical for convergence.
