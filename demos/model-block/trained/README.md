# Trained tiny byte model

This is a **real trained F32 checkpoint** for the model slice 3 graph. It is
deliberately tiny: a smoke model that proves the checkpoint, tokenizer, decode,
and placement paths, not a useful assistant.

```text
checkpoint   jotnar.model.checkpoint/1, F32, 22 tensors, 464,128 bytes
manifest     sha256 01b128cea5b4d2d12297503f2742186825d235ad4e432a0313ed60c7e9d0a4c4
architecture D=64, 2 blocks, 4 heads, head_dim=16, inter=128, vocab=256
context      16 bytes, learned position embedding
tokenizer    byte-level (identity on 0..255)
objective    next byte from a 16-byte context (CE on the last position)
corpus       110,491 bytes: README.md + docs/{USAGE,WORKLOAD_CONTROLLER,
             STREAMING,EXECUTOR_CONTRACT,LADDER_OPS,FLEET_ADMISSION,
             DOOM_TWO_NODE}.md
             corpus sha256 eb51498b55d742009b093d9a231e9739df49f80bf48da039949b940b126bf0a7
split        99,441 train / 11,050 val bytes
training     2,500 steps, batch 64, AdamW lr 2e-3 cosine, seed 0, CPU F32
             torch 2.14.1+cu130, 4 threads
loss         3.22 bits/byte train, 3.36 bits/byte val (uniform = 8.00)
```

The training objective matches the graph: the graph's attention has no causal
mask, so a causal-LM objective would leak the future. Instead each example is a
16-byte context and the loss is the cross-entropy of the following byte at the
last position. Greedy decode slides the window one byte at a time and reruns the
whole graph; there is no KV cache yet.

## Prompt fixture

`prompt.json` holds a 16-byte prompt taken from the **validation** split, plus
the next 8 bytes as the true continuation:

```text
prompt        "rity. GPU partit"      (726974792e2047505520706172746974)
continuation  "ioning o"              (696f6e696e67206f)
greedy 8      " the the"              (2074686520746865)
```

The greedy output is honest evidence of the model's quality: it learned byte
statistics, but it is not a language model. The decode gate checks the
distributed graph against numpy and torch; it does not require the greedy text
to equal the continuation. The continuation match is reported as a quality
signal only.

## Files

```text
manifest.json           checkpoint manifest, tensor hashes, provenance
tensors/*.bin           little-endian F32 tensor blobs
prompt.json             16-byte prompt fixture + true continuation
training.json           training configuration, losses, greedy sample
```

## Regenerate

```sh
python3 scripts/train_tiny_model.py --out demos/model-block/trained
```

The script is deterministic for a fixed torch/CPU stack (seed 0, CPU, F32,
`torch.use_deterministic_algorithms(True)`). The checked-in checkpoint is the
artifact the gates load; the manifest hash is recorded in every gate summary.

## Limits

- Not a chat model, not instruction-tuned, no quality claim.
- 16-byte context, no causal LM objective, no KV cache.
- The listing `demos/model-block/attn4.asm` fixes D=64 and four 16-wide heads;
  the checkpoint must keep that shape.
- The corpus is in-repo documentation. It is a provenance and reproducibility
  choice, not a claim that the model generalizes beyond it.
