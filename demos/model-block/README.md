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

The slice-2 gate used a deterministic fixture; slice 3 replaces it with a
trained F32 checkpoint without changing the graph.

## Slice 3 — trained checkpoint, tokenizer, decode, measurement

`trained/` is a real trained F32 checkpoint (22 tensors, 464,128 bytes) for the
same graph. It is a tiny byte-level next-byte model: D=64, 2 blocks, 4 heads,
context 16, vocab 256, learned position embedding. It was trained from scratch
on the eight documentation files recorded in `training.json` — the pre-slice-3,
pre-Vulkan corpus (110,491 bytes; `corpus_sha256 eb51498b…`) — at 2,500 steps
and 8 threads. The current docs differ from those hashes, so retraining from
this checkout yields a different checkpoint; `trained/README.md` records the
exact flags and the historical corpus. Losses are 3.22 train / 3.36 val bits
per byte (uniform is 8.00). It is not a chat model; the generated sample is
reported honestly.

`scripts/model_decode_gate.py` owns the checkpoint, byte tokenizer, prompt, and
greedy policy in the controller, materializes `tok_embed + pos_embed`, and runs
eight decode steps through the graph. It checks per-step logits against numpy
and the torch training model, the exact sequence, every stage done, repeat
determinism, and a `block0.WQ + 0.5` control. It runs the pipeline and an
all-HIP placement; both match numpy, so the HIP attention path is parity-clean.

```sh
python3 scripts/model_decode_gate.py --no-build \
  --checkpoint demos/model-block/trained --decode 8 \
  --out /tmp/opencode/model-decode-gate
python3 scripts/model_perf_gate.py --no-build \
  --checkpoint demos/model-block/trained --repeats 10 --warmup 2 \
  --decode 8 --clients 1,2,4 --requests 4 \
  --out /tmp/opencode/model-perf-gate
```

Measured warm on the live fleet: single-CUDA p50 2343.8 ms, pipeline 1702.4 ms,
single-HIP 732.0 ms. The split beats single-CUDA (1.38x) and loses to single-HIP
(2.33x). With a warm build cache every stage is a separate process launch and
the executor reports `run_ms` about 440 ms on CUDA and 125 ms on HIP; dispatch
dominates the model. Data-parallel is only 1.20x the pipeline at four clients.
The honest verdict is in `docs/WORKLOAD_CONTROLLER.md`; a resident/fused path is
needed before the split can be a compute-scaling claim.
