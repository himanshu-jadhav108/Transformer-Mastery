# 03 — Training Loop

## Standard Training Loop

```python
def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0

    for batch in dataloader:
        src = batch['src'].to(device)
        tgt = batch['tgt'].to(device)

        # tgt_input: all except last token
        # tgt_output: all except first token (shifted by 1)
        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:]

        optimizer.zero_grad()

        # Forward pass
        output = model(src, tgt_input)  # (B, T, V)

        # Compute loss
        loss = criterion(output.reshape(-1, output.size(-1)),
                        tgt_output.reshape(-1))

        # Backward pass
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)
```

## For Decoder-Only (GPT-style)

```python
def train_epoch_gpt(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0

    for batch in dataloader:
        x = batch.to(device)  # (B, T)

        optimizer.zero_grad()

        logits = model(x)  # (B, T, V)

        # Predict next token at each position
        logits = logits[:, :-1, :].reshape(-1, vocab_size)
        targets = x[:, 1:].reshape(-1)

        loss = criterion(logits, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)
```

## Key Takeaway

The training loop is standard supervised learning: forward → loss → backward → step. The key difference for Transformers is the causal target shift (predict token t+1 given tokens 1..t).
