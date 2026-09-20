# 06 — Debugging Transformers

## Common Issues and Solutions

### 1. Loss Not Decreasing

| Cause | Check | Fix |
|-------|-------|-----|
| Learning rate too high | LR > 1e-3 | Reduce to 1e-4 |
| No gradient flow | `p.grad is None` | Check requires_grad |
| Wrong loss computation | Shapes mismatch | Verify (B*T, V) vs (B*T,) |
| Masking error | Attention on padding | Check mask application |

### 2. NaN Loss

| Cause | Check | Fix |
|-------|-------|-----|
| Gradient explosion | grad_norm > 1000 | Clip gradients |
| Division by zero | d_k = 0 | Verify d_model % h == 0 |
| Bad initialization | Weights too large | Use Xavier init |
| Mixed precision overflow | Loss scale = 1 | Increase loss scale |

### 3. Attention Weights All Equal

**Cause**: Scaling factor too large, or Q/K initialized poorly.

**Fix**: Check `scores / sqrt(d_k)`. Verify d_k is correct.

### 4. Model Only Predicts <pad> or <unk>

**Cause**: Class imbalance, or target shift wrong.

**Fix**: Use `ignore_index` in loss. Verify target is shifted correctly.

### 5. Slow Training

| Cause | Fix |
|-------|-----|
| No GPU | Move model/data to CUDA |
| Small batch size | Increase or use gradient accumulation |
| No mixed precision | Enable autocast |
| Inefficient data loading | Use num_workers > 0 |

### 6. Overfitting

- Increase dropout (0.1 → 0.3)
- Add weight decay (0.01 → 0.1)
- Use larger dataset
- Reduce model size

## Debugging Checklist

- [ ] Verify tensor shapes at every layer
- [ ] Check attention weights sum to 1
- [ ] Confirm causal mask is lower-triangular
- [ ] Ensure padding mask excludes pad tokens
- [ ] Validate loss decreases on a tiny dataset
- [ ] Test with batch_size=1 first
- [ ] Compare outputs with reference implementation

## Key Takeaway

Most Transformer bugs are shape mismatches or masking errors. Always verify tensor shapes and test on a small dataset before scaling up.
