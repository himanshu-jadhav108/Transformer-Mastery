# 05 — Backpropagation Through Attention

## The Challenge

Attention is a complex function. We need gradients to flow through:
```
Q, K, V projections → QK^T → scaling → softmax → matrix multiply with V
```

## Gradient Flow

### 1. Through Output Projection

```
∂L/∂O = ∂L/∂Output · W_O^T
```

### 2. Through A·V

```
∂L/∂A = ∂L/∂O · V^T
∂L/∂V = A^T · ∂L/∂O
```

### 3. Through Softmax

```
∂L/∂S' = A ⊙ (∂L/∂A - Σ_j A_j · ∂L/∂A_j)
```

The Jacobian of softmax is the attention weights times the centered gradient.

### 4. Through Scaling

```
∂L/∂S = (1/√d_k) · ∂L/∂S'
```

### 5. Through QK^T

```
∂L/∂Q = ∂L/∂S · K
∂L/∂K = (∂L/∂S)^T · Q
```

### 6. Through Projections

```
∂L/∂W_Q = X^T · ∂L/∂Q
∂L/∂W_K = X^T · ∂L/∂K
∂L/∂W_V = X^T · ∂L/∂V
```

## Vanishing Gradients in Attention?

Rare. Attention creates **direct paths** from output to any input position. The softmax can saturate, but skip connections (residuals) provide alternative gradient paths.

## Key Takeaway

Gradients flow through attention via the chain rule. The softmax Jacobian is the most complex part, but PyTorch handles it automatically. Understanding the flow helps debug training issues.
