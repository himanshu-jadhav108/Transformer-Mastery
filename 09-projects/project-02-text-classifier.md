# Project 2: Transformer Text Classifier

## Objective
Build a sentiment classifier using an encoder-based Transformer.

## Concepts Used
- Encoder-only architecture
- Classification head
- Fine-tuning

## Architecture
```
Input -> Embedding -> Encoder x6 -> [CLS] token -> Linear -> Softmax -> Class
```

## Steps
1. Prepare IMDB or SST-2 dataset
2. Build TransformerEncoder (from scratch or use HuggingFace)
3. Add classification head: Linear(d_model, num_classes)
4. Fine-tune on labeled data
5. Evaluate accuracy

## Expected Learning Outcome
- Understand how encoder representations are used for classification
- Practice fine-tuning workflows
- Learn about [CLS] token pooling

## Possible Extensions
- Multi-label classification
- Compare with BERT fine-tuning
- Add attention visualization for interpretability
