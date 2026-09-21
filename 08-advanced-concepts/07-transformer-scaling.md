# 07 — Transformer Scaling

## Scaling Laws

As model size increases, performance improves predictably:

```
Loss ∝ C^(-α)   where C = compute (FLOPs), α ≈ 0.05-0.07
Loss ∝ N^(-β)   where N = parameters, β ≈ 0.05-0.08
Loss ∝ D^(-γ)   where D = dataset size, γ ≈ 0.05-0.10
```

## Chinchilla Scaling Laws (2022)

Optimal training: Model size (N) and data (D) should scale equally:
```
Optimal D ≈ 20 × N   (tokens per parameter)
```

For a 70B parameter model: train on ~1.4T tokens.

## Compute-Optimal Training

Given a fixed compute budget C:
```
N_opt ∝ C^0.5
D_opt ∝ C^0.5
```

## Over-Training

Training longer than Chinchilla-optimal can improve inference efficiency:
- More compute during training
- Smaller model can match larger under-trained model
- Better for deployment (faster inference)

## Emergent Abilities

At certain scale thresholds, models suddenly gain new capabilities:
- In-context learning
- Chain-of-thought reasoning
- Multi-step problem solving

## Practical Implications

1. **Small models + more data** > **Large models + less data** (for fixed compute)
2. **Depth vs Width**: Deeper models (more layers) generalize better than wide models
3. **Batch size**: Larger batches allow larger learning rates

## Key Takeaway

Transformer performance follows predictable scaling laws. The Chinchilla optimal point balances model size and training data. Understanding scaling helps make efficient training decisions.
