# Learning Roadmap

## Phase 1: Foundations (Days 1-2)
- Neural network basics for Transformers
- Word embeddings and vector spaces
- Sequence modeling necessity
- RNN, LSTM, GRU architectures
- **Why RNNs fail at long sequences**
- The attention revolution

**Key Insight:** RNNs process sequentially → slow, vanishing gradients. Transformers process in parallel → fast, direct connections.

## Phase 2: Attention Deep Dive (Days 3-4)
- What "attention" means intuitively
- Query, Key, Value decomposition
- Scaled dot-product attention
- Self-attention mechanism
- Multi-head attention

**Key Insight:** Attention = soft dictionary lookup. Q asks questions, K provides addresses, V returns content.

## Phase 3: Transformer Architecture (Days 5-6)
- Positional encoding (sinusoidal + learned)
- Encoder block internals
- Decoder block internals
- Cross-attention mechanism
- Masking (causal + padding)
- Feed-forward networks
- Residual connections + LayerNorm
- Complete Transformer assembly

## Phase 4: Mathematics (Parallel with Phase 3)
- Linear algebra for Transformers
- Attention score mathematics
- Softmax and probability distributions
- Cross-entropy loss
- Backpropagation through attention

## Phase 5: Build From Scratch (Days 7-10)
- Tensor shape discipline
- Embeddings + Positional encoding
- Self-attention (single + multi-head)
- Transformer block
- Encoder + Decoder
- Complete Transformer model
- Training pipeline

## Phase 6: Training (Days 11-12)
- Data pipeline design
- Tokenization (BPE, WordPiece)
- Training loop implementation
- Loss and optimization
- Learning rate scheduling (warmup + cosine)
- Debugging strategies

## Phase 7: Modern Architectures (Days 13-14)
- BERT (encoder-only, bidirectional)
- GPT (decoder-only, causal)
- T5 (encoder-decoder, text-to-text)
- LLaMA (RMSNorm, RoPE, SwiGLU, GQA)
- Vision Transformers (ViT)
- Multimodal Transformers

## Phase 8: Advanced Optimization (Days 15-16)
- Causal attention optimization
- KV Cache for inference
- Rotary Position Embeddings (RoPE)
- Flash Attention
- Mixture of Experts (MoE)
- Efficient attention variants
- Scaling laws

## Phase 9: Projects (Days 17-21)
- Project 1: Attention Visualizer
- Project 2: Text Classifier
- Project 3: Mini Language Model
- Project 4: Mini GPT From Scratch

## Phase 10: Mastery (Ongoing)
- Revision cheatsheet
- Interview preparation
- Common mistakes review
- Final mastery checklist

---

## Progress Tracker

| Phase | Status | Date |
|-------|--------|------|
| 1. Foundations | [ ] | |
| 2. Attention | [ ] | |
| 3. Transformer Core | [ ] | |
| 4. Mathematics | [ ] | |
| 5. Build From Scratch | [ ] | |
| 6. Training | [ ] | |
| 7. Modern Architectures | [ ] | |
| 8. Advanced Concepts | [ ] | |
| 9. Projects | [ ] | |
| 10. Mastery | [ ] | |
