# Transformer Mastery Course

![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch&logoColor=white)
![License](https://img.shields.io/badge/license-learning%20resource-informational)

A structured, code-first course for understanding Transformer architecture from
the ground up. Each stage combines clear theory, mathematical intuition, and
runnable Python examples so that you can move from "what does attention do?"
to building and debugging Transformer systems yourself.

## Project Overview

The course is organized as a single learning path:

- Markdown lessons explain the concepts, equations, and architecture.
- Numbered Python scripts provide runnable PyTorch and NumPy examples.
- Projects and mastery material turn the concepts into practical skills.

The code lives beside the lessons it supports. Within each code directory,
scripts are numbered from `01-...` so you can execute them in a sensible local
sequence. See [GETTING_STARTED.md](GETTING_STARTED.md) for Windows PowerShell
commands and the complete setup workflow.

## Goals And Learning Outcomes

By the end of the course, you should be able to:

- Explain why attention-based models replaced recurrent architectures for many sequence tasks.
- Derive and implement scaled dot-product, self, cross, causal, and multi-head attention.
- Track tensor shapes through embeddings, projections, masks, residual paths, and normalization.
- Build a complete Encoder-Decoder Transformer in PyTorch.
- Compare BERT, GPT, T5, LLaMA, Vision Transformer, and multimodal designs.
- Understand practical techniques including RoPE, KV caching, Flash Attention, MoE, and efficient attention.
- Train, evaluate, generate from, and debug small Transformer models.
- Apply the material in four progressively harder projects and prepare for technical interviews.

## Audience And Prerequisites

This course is for Python developers, AI and Data Science students, and
engineers who want a practical understanding of Transformers rather than only
an API-level overview.

You should be comfortable with Python and have introductory knowledge of:

- Vectors, matrices, matrix multiplication, and basic probability.
- Neural networks, backpropagation, loss functions, and optimization.
- Python environments and basic PyTorch tensors and `nn.Module`.

If tensor mechanics are rusty, `04-mathematics/code/` provides focused shape,
broadcasting, matrix-operation, and `einsum` exercises.

## Quick Start

From the repository root in Windows PowerShell:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python 02-attention\code\01-attention_manual_math.py
```

For PowerShell execution-policy guidance, troubleshooting, and a fuller study
workflow, read [GETTING_STARTED.md](GETTING_STARTED.md).

## Course Roadmap

Study the modules in order. The estimated times are approximate and include
reading, coding, and review.

| Module | Focus | Key practice | Time |
| --- | --- | --- | --- |
| `00-roadmap/` | Course map and progress tracking | Plan the learning path | 15 min |
| `01-foundations/` | Neural networks, embeddings, RNNs, LSTMs, and GRUs | Build the intuition for Transformers | 2-3 hrs |
| `02-attention/` | QKV, scaled dot-product, self, cross, causal, and multi-head attention | Run attention implementations and visualizations | 4-5 hrs |
| `03-transformer-core/` | Positional encoding, encoder/decoder blocks, masking, FFNs, residuals, and normalization | Assemble core Transformer components | 5-6 hrs |
| `04-mathematics/` | Linear algebra, softmax, loss, and backpropagation | Practice tensor shapes and operations | 2-3 hrs |
| `05-build-from-scratch/` | Guided reconstruction of a Transformer | Follow `00-CODE-MAP.md` across modules 02-04 | 6-8 hrs |
| `06-training/` | Data pipelines, tokenization, training, optimization, scheduling, and debugging | Run training and diagnostic scripts | 3-4 hrs |
| `07-modern-transformers/` | BERT, GPT, T5, LLaMA, ViT, and multimodal models | Run mini-BERT and mini-GPT examples | 3-4 hrs |
| `08-advanced-concepts/` | KV cache, RoPE, Flash Attention, MoE, efficient attention, and scaling | Explore modern efficiency techniques | 3-4 hrs |
| `09-projects/` | Four applied Transformer projects | Build increasingly complete systems | 8-10 hrs |
| `10-mastery/` | Revision, common mistakes, interview questions, and final checks | Consolidate and test your knowledge | 2-3 hrs |

Expected total: approximately **40-50 hours**.

## Recommended Workflow

1. Read `00-roadmap/learning-roadmap.md` and record your starting point.
2. Complete each module's lessons before running its code.
3. Execute numbered scripts in each populated `code/` directory.
4. Write down the input and output shape of every major operation.
5. Reimplement important components without looking at the solution.
6. Use `09-projects/` to connect isolated ideas into working systems.
7. Finish with `10-mastery/` and revisit any unchecked outcome.

## Repository Structure

```text
transformer-mastery-course/
├── README.md
├── GETTING_STARTED.md
├── requirements.txt
├── 00-roadmap/                  # Roadmap and progress tracker
├── 01-foundations/              # Neural-network and sequence foundations
├── 02-attention/                # Attention theory and implementations
├── 03-transformer-core/         # Transformer building blocks
├── 04-mathematics/              # Math and tensor-operation practice
├── 05-build-from-scratch/       # Guided end-to-end reconstruction
├── 06-training/                 # Training workflows and debugging
├── 07-modern-transformers/      # BERT, GPT, T5, LLaMA, ViT, multimodal
├── 08-advanced-concepts/        # Efficiency and scaling techniques
├── 09-projects/                 # Applied projects and starter code
└── 10-mastery/                  # Review, tests, and interview preparation
```

Most modules with runnable examples contain a `code/` directory. Some also
contain focused subdirectories such as `visualizations/`, `debugging/`,
`generation/`, and `experiments/`. `05-build-from-scratch/code/` is reserved
for the guided code map and is currently intentionally empty.

## Tensor-Shape Conventions

The course uses these symbols consistently:

| Symbol | Meaning |
| --- | --- |
| `B` | Batch size |
| `T` | Sequence length, measured in tokens |
| `C` or `d_model` | Model or embedding dimension |
| `h` | Number of attention heads |
| `d_k` | Query/key dimension per head |
| `d_v` | Value dimension per head |

The standard token representation is `(B, T, C)`. Multi-head attention often
reshapes it to `(B, h, T, d_k)` or `(B, h, T, d_v)`. Always check whether a
script uses batch-first or sequence-first tensors before comparing shapes.

## Troubleshooting

- **PowerShell blocks activation:** run
  `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`, then activate
  `.\.venv\Scripts\Activate.ps1` again.
- **`python` or `py` is not recognized:** install Python 3.9+ and enable
  "Add Python to PATH", then open a new PowerShell window.
- **Imports fail:** confirm `(.venv)` appears in the prompt and run
  `python -m pip install -r requirements.txt`.
- **A file path fails:** use the numbered filename shown by `Get-ChildItem`.
  For example, attention starts with
  `python 02-attention\code\01-attention_manual_math.py`.
- **A tensor shape fails:** inspect `04-mathematics/code/` and the debugging
  scripts in `06-training/code/debugging/` before changing model logic.
- **A visualization does not open:** run the script from an interactive Python
  environment and confirm Matplotlib is installed in the active virtual environment.

## Maintainer

**Himanshu Jadhav**  
Artificial Intelligence & Data Science Engineer

Passionate about computer vision, real-world localized model deployment, and
high-performance pipeline architecture.

- GitHub: [himanshu-jadhav108](https://github.com/himanshu-jadhav108)
- LinkedIn: [Himanshu Jadhav](https://www.linkedin.com/in/himanshu-jadhav-328082339)
- Portfolio: [himanshu-jadhav-portfolio.vercel.app](https://himanshu-jadhav-portfolio.vercel.app/)
- Instagram: [@himanshu_jadhav_108](https://www.instagram.com/himanshu_jadhav_108)
