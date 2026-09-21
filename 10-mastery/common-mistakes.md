# Common Mistakes

## Attention
- **Confusing Q, K, V**: Remember — Q asks, K indexes, V provides content.
- **Forgetting the scale factor**: Without /sqrt(d_k), softmax saturates and gradients vanish.
- **Wrong softmax dimension**: Must be over the KEY dimension (dim=-1), not features.
- **Incorrect masking**: Padding mask excludes pads; causal mask excludes future.

## Shapes
- **Multi-head reshape errors**: `view()` then `transpose()`, not the reverse.
- **Forgetting contiguous()**: After transpose, call `.contiguous()` before view.
- **Mismatched Q/K/V dimensions**: d_k and d_v can differ, but usually d_k = d_v.

## Architecture
- **Applying causal mask in encoder**: Encoder is bidirectional — no causal mask!
- **Wrong norm placement**: Pre-Norm is `x + Sublayer(Norm(x))`, not `Norm(x + Sublayer(x))`.
- **Missing residual connections**: Every sub-layer needs a skip connection.

## Training
- **Wrong target shift**: For next-token prediction, input is [0:T-1], target is [1:T].
- **Not ignoring padding in loss**: Use `ignore_index` in CrossEntropyLoss.
- **Learning rate too high**: Start with 1e-4, use warmup.
- **No gradient clipping**: Essential for stability, especially in early training.

## Positional Encoding
- **Assuming attention handles order**: Attention is permutation-invariant. PE is mandatory.
- **Adding PE to wrong place**: Add to embeddings BEFORE the first Transformer block.

## Inference
- **Forgetting KV Cache**: Without it, generation is O(T^2) per step.
- **Not using causal mask during eval**: Even at inference, causal mask prevents cheating.
