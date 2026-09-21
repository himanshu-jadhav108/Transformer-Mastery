# Interview Questions

## Basic
1. What is the difference between self-attention and cross-attention?
2. Why do we scale by sqrt(d_k) in attention?
3. What is the purpose of positional encoding?
4. Explain the difference between encoder and decoder.
5. Why is LayerNorm used instead of BatchNorm?

## Intermediate
6. Derive the attention formula step by step.
7. Why does multi-head attention use different projections per head?
8. What is causal masking and where is it used?
9. Explain residual connections and why Pre-Norm is preferred.
10. How does the FFN contribute to the Transformer?

## Advanced
11. What is the computational complexity of attention? How does Flash Attention improve it?
12. Explain RoPE and why it's better than absolute positional encoding.
13. What is KV Cache and what problem does it solve?
14. Compare MHA, GQA, and MQA. When would you use each?
15. Explain the Chinchilla scaling laws.

## Implementation
16. Write scaled dot-product attention in PyTorch.
17. How do you handle padding in attention?
18. Implement causal mask generation.
19. What tensor shapes do you expect at each layer?
20. How would you debug a Transformer that's not learning?

## Architecture-Specific
21. Why is BERT bidirectional but GPT unidirectional?
22. What pre-training objectives does T5 use?
23. Explain SwiGLU and why LLaMA uses it.
24. What is the advantage of decoder-only models?
25. How do Vision Transformers handle images?

## System Design
26. How would you serve a 70B parameter model efficiently?
27. Describe quantization strategies for inference.
28. How do you handle long-context windows (>100k tokens)?
29. Explain speculative decoding.
30. Design a distributed training setup for a 175B model.
