# 05 — Mixture of Experts (MoE)

## The Problem

Scaling dense models is expensive. Every parameter is used for every input.

## Solution: Sparse Activation

Use many "expert" networks, but activate only a few per token.

## Architecture

```
Input Token
    ↓
[Router / Gate] → selects top-k experts
    ↓
Expert 1    Expert 2    ...    Expert N
    ↓           ↓                ↓
  [FFN]       [FFN]            [FFN]
    ↓           ↓                ↓
    └────── Weighted Sum ──────┘
                    ↓
                 Output
```

## Router

```python
class Router(nn.Module):
    def __init__(self, d_model, num_experts, top_k=2):
        super().__init__()
        self.gate = nn.Linear(d_model, num_experts)
        self.top_k = top_k

    def forward(self, x):
        # x: (B, T, d_model)
        logits = self.gate(x)  # (B, T, num_experts)
        weights, indices = torch.topk(
            F.softmax(logits, dim=-1), self.top_k, dim=-1
        )
        weights = weights / weights.sum(dim=-1, keepdim=True)
        return weights, indices  # (B, T, k), (B, T, k)
```

## MoE Layer

```python
class MoELayer(nn.Module):
    def __init__(self, d_model, d_ff, num_experts, top_k=2):
        super().__init__()
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_ff),
                nn.GELU(),
                nn.Linear(d_ff, d_model)
            ) for _ in range(num_experts)
        ])
        self.router = Router(d_model, num_experts, top_k)

    def forward(self, x):
        weights, indices = self.router(x)
        # Route each token to top-k experts and combine
        # ... (implementation details)
        return output
```

## Load Balancing

Problem: Router may send all tokens to the same few experts.

Solution: Add auxiliary loss to encourage uniform routing:
```
Loss_aux = num_experts * Σ_i (fraction_of_tokens_to_expert_i * average_router_prob_for_expert_i)
```

## Examples

| Model | Experts | Active per token | Total Params |
|-------|---------|-----------------|--------------|
| GShard | 2048 | 2 | 600B |
| Switch Transformer | 2048 | 1 | 1.6T |
| Mixtral 8x7B | 8 | 2 | 47B active / 13B unique |

## Key Takeaway

MoE scales model capacity without proportional compute cost. Only a subset of parameters is active per token. The router is the critical component requiring careful load balancing.
