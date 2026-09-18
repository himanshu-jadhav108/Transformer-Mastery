# Where's the Code for This Section?

This section is a **guided reading path** through code that already lives in earlier
module folders — it doesn't duplicate files, it tells you which file to open next
so the "from scratch" build feels continuous. Follow the table top to bottom,
alongside the `.md` lesson of the same number.

| Lesson | Read | Then Open This Code |
|---|---|---|
| 01 | `01-tensor-basics.md` | `../04-mathematics/code/tensor_shapes.py`, `broadcasting.py` |
| 02 | `02-embeddings-from-scratch.md` | `../03-transformer-core/code/token_embeddings.py`, `positional_encoding_sinusoidal.py`, `learned_positional_embeddings.py` |
| 03 | `03-self-attention-from-scratch.md` | `../02-attention/code/attention_manual_math.py`, `scaled_dot_product.py`, `self_attention.py`, `qkv_projection.py` |
| 04 | `04-multi-head-attention-from-scratch.md` | `../02-attention/code/mha_manual.py`, `mha_einsum.py` |
| 05 | `05-transformer-block.md` | `../03-transformer-core/code/residual_connections.py`, `layer_norm.py`, `feed_forward_relu.py`, `feed_forward_gelu.py` |
| 06 | `06-encoder-from-scratch.md` | `../03-transformer-core/code/encoder_block.py` |
| 07 | `07-decoder-from-scratch.md` | `../03-transformer-core/code/causal_transformer.py` |
| 08 | `08-complete-transformer-from-scratch.md` | `../03-transformer-core/code/transformer_encoder_decoder.py` |

**Why organized this way:** keeping one canonical copy of each script (instead of a
duplicate per section) means when you fix a bug or tweak a hyperparameter, there's
only one place to change it — and it matches the folder you'll actually keep
referring back to during the projects in `09-projects/`.
