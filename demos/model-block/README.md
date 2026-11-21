# Model block demo — pipeline-parallel transformer

The first model-flagship slice: a stack of single-head pre-norm transformer
blocks, split across the live CUDA/HIP fleet.

```text
CUDA  attention:  X -> RMSNorm -> QKV -> Q K^T -> softmax -> A V
                     -> out proj -> residual -> RMSNorm    outputs N2, RES
HIP   MLP:        N2 -> gate/up -> SiLU(gate)*up -> down -> +RES   output OUT
```

Block 0 consumes the input `X`; block i>0 consumes block i-1's `OUT` as `X`.
The workload controller holds the manifest, weights, and input as bundle
artifacts; each executor receives one listing plus payloads. No model semantics
or weights live in the executor.

Listings:

```text
demos/model-block/attn.asm    CUDA attention half
demos/model-block/mlp.asm     HIP MLP half
```

The block is `D=64`, `N=8` tokens, `inter=128`, single head, F32. The gate uses
a deterministic seed and a numpy reference.

## Gate

```sh
python3 scripts/model_block_gate.py --no-build --blocks 2 \
  --out /tmp/opencode/model-block-gate
```

Checked on the live fleet:

```text
placement   attn-0/attn-1 -> CUDA, mlp-0/mlp-1 -> HIP
exactness   block0 3.0e-8, block1 3.0e-8 max abs vs numpy
repeat      byte-identical block outputs
control     perturbed WQ matches the perturbed reference and differs from
            the original by 1.57e-2
receipt     all stages done, outputs and hashes recorded
```

`--blocks 1` is the minimal slice. A real checkpoint, multi-head attention,
and a full embedding->blocks->logits sentence are the next steps.

## Slice 2 — checkpoint, multi-head, full sentence

`attn4.asm` slices Q/K/V into four 16-wide heads, runs per-head attention, and
concatenates with `CONCAT`; `head.asm` is the final RMSNorm + logits matmul.
`scripts/model_sentence_gate.py` writes a `jotnar.model.checkpoint/1` fixture
(manifest + F32 tensor blobs), loads it back with hash checks, materializes the
token embedding, and runs:

```text
tokens -> embedding -> attn-0 (CUDA) -> mlp-0 (HIP) -> attn-1 (CUDA)
       -> mlp-1 (HIP) -> final RMSNorm + head (CUDA) -> logits [8,32]
```

```sh
python3 scripts/model_sentence_gate.py --no-build --blocks 2 \
  --out /tmp/opencode/model-sentence-gate
```

```text
logits/block outputs  2.4e-7 / 3.0e-8 max abs vs numpy
repeat                byte-identical
control               head + 0.5 differs from the original by 6.65
receipt               5 stages done, attn/head CUDA, mlp HIP
```

The checkpoint is a deterministic fixture; replace it with a trained F32
checkpoint without changing the graph.
