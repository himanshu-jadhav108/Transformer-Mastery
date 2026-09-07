# 01 — Attention Intuition

## What is Attention?

Attention is a **soft dictionary lookup mechanism** where a query searches a set of key-value pairs.

## Real-World Analogy

Imagine searching a library:
- **Query (Q)**: "I want books about neural networks"
- **Key (K)**: Book titles/topics (what each book is about)
- **Value (V)**: The actual book content

The librarian compares your query with all keys, finds the most relevant ones, and returns a weighted combination of their values.

## In Neural Networks

Attention allows the model to **focus on relevant parts of the input** when producing each output.

## Why "Soft" Lookup?

Instead of returning one book, attention returns a **weighted blend** of ALL books, where weights indicate relevance.

## Core Purpose

1. **Capture relationships**: How much does token A relate to token B?
2. **Contextualize representations**: Each token's meaning depends on its neighbors
3. **Enable parallelization**: All pairwise relationships computed simultaneously

## Example in Language

```
Sentence: "The animal didn't cross the street because it was too tired."

What does "it" refer to? Attention allows "it" to attend strongly to "animal"
and weakly to "street", resolving the reference.
```

## Key Takeaway

Attention = relevance-weighted information retrieval. It replaces the need for sequential processing by allowing every token to directly access every other token.
