# Transformer Mastery Course

> A complete, structured, self-contained course for mastering Transformer
> architecture — from foundations to advanced concepts, with theory and
> hands-on PyTorch code integrated side by side in every module.

This course merges two parts into one learning path:
- **Theory** (`*.md` lessons) — concepts, math, and architecture explanations
- **Code** (`code/` folders) — runnable PyTorch/NumPy implementations for each lesson

You never have to guess which script goes with which lesson — every module's
`code/` folder sits right next to the lessons it belongs to.

## Who This Is For

- Complete beginners to Transformers with basic Python + some ML exposure
- Undergraduate AI & Data Science students
- Python developers with basic deep learning knowledge
- Anyone seeking **complete, practical mastery** over Transformers

## Prerequisites

- Python (intermediate)
- Basic Linear Algebra (vectors, matrices)
- Basic Deep Learning (neural networks, backpropagation, loss functions)
- PyTorch basics (tensors, `nn.Module`) — module `04-mathematics/code` will
  warm you up on tensor mechanics if you're rusty

## Setup

See `GETTING_STARTED.md` for environment setup. Quick version:

```bash
pip install -r requirements.txt
```

## Course Map (read + run in this order)

| # | Module | What You'll Learn | Code Included | Est. Time |
|---|--------|--------------------|----------------|-----------|
| 00 | `00-roadmap/` | Full course roadmap & progress tracker | — | 15 min |
| 01 | `01-foundations/` | Neural nets, embeddings, RNN/LSTM/GRU, why Transformers won | — | 2-3 hrs |
| 02 | `02-attention/` | Attention intuition, QKV, scaled dot-product, self-attention, multi-head attention | attention from scratch, attention variants, multi-head/grouped/multi-query attention, attention heatmap visualizer | 4-5 hrs |
| 03 | `03-transformer-core/` | Positional encoding, encoder/decoder blocks, cross-attention, masking, feed-forward, residuals, LayerNorm, full assembly | embeddings, RMSNorm/LayerNorm, SwiGLU, encoder block, causal decoder block, full encoder-decoder model | 5-6 hrs |
| 04 | `04-mathematics/` | Linear algebra, attention math, softmax, loss functions, backprop through attention | tensor shapes, broadcasting, einsum, matrix ops drills | 2-3 hrs |
| 05 | `05-build-from-scratch/` | Guided rebuild of the whole Transformer, lesson by lesson | code-map pointing back into modules 02-04 (see `00-CODE-MAP.md`) | 6-8 hrs |
| 06 | `06-training/` | Data pipeline, tokenization, training loop, optimization, LR scheduling, debugging | training loop, checkpointing, LR scheduler, shape/masking/NaN-loss debugging scripts | 3-4 hrs |
| 07 | `07-modern-transformers/` | BERT, GPT, T5, LLaMA, ViT, multimodal Transformers | mini-BERT (encoder-only), mini-GPT (decoder-only) | 3-4 hrs |
| 08 | `08-advanced-concepts/` | KV cache, RoPE, Flash Attention, MoE, efficient attention, scaling laws | KV cache, RoPE, sliding-window & flash-attention concepts, MoE, sampling strategies, scaling experiments | 3-4 hrs |
| 09 | `09-projects/` | 4 hands-on projects tying everything together | full starter code per project | 8-10 hrs |
| 10 | `10-mastery/` | Revision cheatsheet, interview prep, common mistakes, final checklist | attention + multi-head attention test suites | 2-3 hrs |

**Total: ~40-50 hours** for complete mastery.

## Recommended Study Order

1. Read `00-roadmap/learning-roadmap.md` first and track your progress there.
2. Complete `01-foundations/` sequentially — no code yet, pure grounding.
3. Deep-dive into `02-attention/`, coding along with everything in its `code/`
   folder — this is the heart of the course.
4. Study `03-transformer-core/` alongside its `code/` folder.
5. Read `04-mathematics/` and run the tensor drills in its `code/` folder
   whenever a shape or gradient claim doesn't feel intuitive yet.
6. Work through `05-build-from-scratch/` — open `00-CODE-MAP.md` first, since
   this section is a guided rebuild that reuses code from modules 02-04
   rather than duplicating it.
7. Follow `06-training/` and actually run the training loop end to end.
8. Explore `07-modern-transformers/`, running mini-BERT and mini-GPT.
9. Study `08-advanced-concepts/` for the optimizations that show up in every
   modern LLM.
10. Build all four projects in `09-projects/`, in order — each one is
    slightly harder than the last.
11. Finish with `10-mastery/`: run the test suite, then work through the
    revision cheatsheet and interview questions.

## Final Outcomes

After completing this course, you will be able to:

- [ ] Derive and implement scaled dot-product attention from scratch
- [ ] Build a complete Transformer (Encoder + Decoder) in PyTorch
- [ ] Understand BERT, GPT, T5, and LLaMA architectures
- [ ] Implement KV Cache, RoPE, Flash Attention, and MoE
- [ ] Train and generate text with a Mini GPT
- [ ] Debug Transformer training issues (shape errors, masking bugs, NaN loss)
- [ ] Pass the self-test suite in `10-mastery/code/`
- [ ] Answer advanced Transformer interview questions

## Tensor Shape Convention

Used consistently across every lesson and script in this course:

- **B** = Batch size
- **T** = Sequence length (tokens)
- **C** or **d_model** = Embedding dimension
- **d_k** = Key/Query dimension per head
- **d_v** = Value dimension per head
- **h** = Number of attention heads

Standard shape: `(B, T, C)`

## Folder Structure

```
transformer-mastery-course/
├── README.md                  <- you are here
├── GETTING_STARTED.md
├── requirements.txt
├── 00-roadmap/
├── 01-foundations/
├── 02-attention/
│   └── code/
├── 03-transformer-core/
│   └── code/
├── 04-mathematics/
│   └── code/
├── 05-build-from-scratch/
│   └── 00-CODE-MAP.md         <- points back to 02-04's code
├── 06-training/
│   └── code/
├── 07-modern-transformers/
│   └── code/
├── 08-advanced-concepts/
│   └── code/
├── 09-projects/
│   └── code/
└── 10-mastery/
    └── code/
```

## Notes on This Edition

This edition merges the original theory and code repositories into a single
course and fixes a few rough edges found while combining them:

- Every module now has its code sitting alongside the matching lessons,
  instead of theory and code living in two separate places.
- `09-projects/`: two project files had filenames that didn't match their
  actual content (a "language model" file that was actually the text
  classifier, and vice versa) — renamed so filenames now match content and
  match the corresponding `code/` subfolder.
- Added `04-mathematics/code/`, pairing the tensor-mechanics drills
  (broadcasting, einsum, shapes, matrix ops) with the math lessons they
  actually support.
- Added attention/positional-encoding visualization scripts into the
  modules they visualize, so they get used exactly when they're most useful.
- All 55 code files were verified to compile cleanly with no syntax errors.

---

*Built for maximum learning value with minimum wasted time.*
