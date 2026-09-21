# Project 1: Mini Attention Visualizer

## Objective
Build a tool that visualizes attention weights for a given sentence.

## Concepts Used
- Self-attention
- Attention matrices
- Softmax

## Architecture
```
Input Sentence -> Tokenize -> Embedding -> Single Self-Attention -> Visualize (B, T, T) heatmap
```

## Steps
1. Load a small pre-trained model (e.g., DistilBERT)
2. Pass a sentence through it
3. Extract attention weights from a chosen layer/head
4. Plot as a heatmap (tokens x tokens)

## Expected Learning Outcome
- Understand what attention "looks like" in practice
- See how [CLS] attends broadly, while content words attend specifically
- Observe different patterns across heads

## Possible Extensions
- Compare attention across layers
- Visualize cross-attention in translation
- Build an interactive web app
