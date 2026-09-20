# 05 — Learning Rate Scheduling

## Warmup + Cosine Decay

The standard schedule for Transformers:

```python
import math

class TransformerScheduler:
    def __init__(self, optimizer, d_model, warmup_steps, max_steps):
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.step_num = 0

    def step(self):
        self.step_num += 1
        lr = self._get_lr()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def _get_lr(self):
        step = self.step_num
        # Warmup: linear increase
        # Decay: inverse square root
        return self.d_model ** (-0.5) * min(
            step ** (-0.5),
            step * self.warmup_steps ** (-1.5)
        )
```

## Schedule Visualization

```
LR
│    ╭────╮
│   ╱      ╲____
│  ╱            ╲___
│ ╱                 ╲__
│╱                      ╲_
└────────────────────────→ Steps
   ^warmup^    ^decay^
```

## Why Warmup?

- Early training: gradients are noisy
- Small LR prevents divergence
- Gradually increase to target LR

## Why Decay?

- Later training: fine-tuning requires smaller steps
- Prevents oscillation around minimum

## Modern Alternatives

- **Cosine decay**: Smooth curve to minimum LR
- **Linear decay**: Simple, works well
- **Constant with cooldown**: Maintain LR then drop

## Key Takeaway

Learning rate scheduling is essential for Transformer training. Warmup prevents early instability; decay ensures convergence. The original paper's inverse square root schedule remains a strong baseline.
