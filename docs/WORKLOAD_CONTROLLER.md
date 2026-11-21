# Workload controller

**Status:** native and GPU stages run; DOOM checkpoint handoff, a CUDA → HIP
batch chain, and a pipeline-parallel transformer block are the acceptance tests
**Scope:** model-aware planning above the CPU, GPU, and native executors

## Boundary decision

DOOM is a native Rust RV32IM program, not a `jotnar-noded` listing. The first
DOOM test therefore adds a **native/process executor** rather than hiding a
process behind the Jotnar job contract:

```text
jotnar-noded          model-blind Jotnar listing execution
jotnar-gpu-runner     target-aware GPU compilation and execution
jotnar-native-runner  model-blind, allow-listed process execution
jotnar-workload       bundle, stage graph, placement, checkpoints, ledger
```

`jotnar-native-runner` knows nothing about DOOM, RV32IM, models, or stage
graphs. It stages named files, runs one allow-listed command in a private
directory, and returns named files through the common executor contract.
`jotnar-workload` owns the DOOM plan.

## Components

### Bundle

A bundle is a content-addressed manifest:

```json
{
  "schema": "jotnar.workload/1",
  "name": "doom-rv32im",
  "version": "1",
  "artifacts": {
    "doom-image": {"sha256": "...", "bytes": 847112, "path": "blobs/..."},
    "doom-wad":   {"sha256": "...", "bytes": 4196020, "path": "blobs/..."}
  },
  "stages": [
    {
      "name": "frame1",
      "depends_on": [],
      "executor": {"kind": "native"},
      "command": ["{{program}}", "--image", "{{input:image}}", "--wad",
                  "{{input:wad}}", "--frames", "1", "--out", "frame1"],
      "inputs": {
        "image": {"artifact": "doom-image", "path": "doom.bin"},
        "wad":   {"artifact": "doom-wad",   "path": "doom1.wad"}
      },
      "outputs": ["frame1.fb", "frame1.png", "frame1.mem", "frame1.x",
                  "frame1.pc", "frame1.cyc", "frame1.csr"],
      "timeout_ms": 600000
    },
    {
      "name": "frame2",
      "depends_on": ["frame1"],
      "executor": {"kind": "native"},
      "command": ["{{program}}", "--resume", "frame1", "--frames", "2",
                  "--out", "frame2"],
      "inputs": {
        "mem": {"stage": "frame1", "output": "frame1.mem", "path": "frame1.mem"},
        "x":   {"stage": "frame1", "output": "frame1.x",   "path": "frame1.x"},
        "pc":  {"stage": "frame1", "output": "frame1.pc",  "path": "frame1.pc"},
        "cyc": {"stage": "frame1", "output": "frame1.cyc", "path": "frame1.cyc"},
        "csr": {"stage": "frame1", "output": "frame1.csr", "path": "frame1.csr"}
      },
      "outputs": ["frame2.fb", "frame2.png", "frame2.mem", "frame2.x",
                  "frame2.pc", "frame2.cyc", "frame2.csr"],
      "timeout_ms": 600000
    }
  ]
}
```

Artifacts are copied into `blobs/<sha256>` and verified before use. Stage
inputs come from an artifact or a previous stage output. `{{program}}` is
resolved from the controller's `--program`; `{{input:key}}` becomes the staged
relative path.

`--frames N` in `jotnar-demo-doom` is an **absolute target frame**, so stage
`frameN` resumes `frameN-1` and passes `--frames N`. Passing `--frames 1` on
resume would re-emit frame 1.

`--program` is the executable path **on the executor host**. The controller
does not need that file locally; each native runner checks it against its own
`--allow` list.

### GPU stages

A GPU stage uses `executor.kind: "gpu"`, a target constraint, and a `gpu`
payload. It carries no argv. The executor contract is unchanged: the controller
sends the same `submit` header and body bytes to a target-aware GPU runner.

```json
{
  "name": "gpu-a-00",
  "depends_on": [],
  "executor": {"kind": "gpu", "target": "cuda"},
  "outputs": ["OUT"],
  "timeout_ms": 600000,
  "gpu": {
    "listing": {"artifact": "listing"},
    "payloads": [{"artifact": "input-00"}],
    "frames": 1,
    "count": 1
  }
}
```

A later stage can consume the previous stage's binary result as a payload
stream:

```json
{
  "name": "gpu-b-00",
  "depends_on": ["gpu-a-00"],
  "executor": {"kind": "gpu", "target": "any"},
  "outputs": ["OUT"],
  "timeout_ms": 600000,
  "gpu": {
    "listing": {"artifact": "listing"},
    "payloads": [
      {"stage": "gpu-a-00", "output": "OUT", "stream": "X"}
    ],
    "frames": 1,
    "count": 1
  }
}
```

Schema decisions for this slice:

- **Listing is an artifact.** It is content-addressed, hashed in the ledger,
  and read by the controller as UTF-8. There is no inline listing field.
- **Payload is either an artifact or a stage output.** An artifact's bytes are
  the exact `{"streams": {...}}` JSON object the runner accepts. A stage output
  is decoded by the controller from the result descriptor's `kind`, `shape`,
  and `frame`.
- **Intermediate GPU output stays binary in the ledger.** The controller writes
  the result body unchanged to `stages/<stage>/<output>`, then converts it to a
  JSON payload only when a later stage consumes it. This keeps the executor
  contract and the content-addressed ledger unchanged; the cost is JSON
  expansion on the controller-to-runner hop.
- **Target is `cuda`, `hip`, or `any`.** The controller matches it against the
  executor capability and rejects a mismatched explicit `--place`.
- **`frames` and `count` are 1 in v1.** The current result splitter keys
  outputs by name, so multi-frame output naming needs its own decision.
- **Timeout is the controller polling deadline.** A stage that ends in
  `failed`/`canceled` stops the run; there is no retry yet.
- The ledger's `command` field for a GPU stage is a plan descriptor, not argv:
  `gpu`, target, listing hash, and each payload source hash. Reuse compares that
  descriptor plus the input hashes and output files.

The binary-to-payload conversion supports `F32`, `F64`, `I64`, `U8`, and
`Bool`. Non-finite `F32`/`F64` values are rejected rather than serialized as
JSON `null`.

### Stage descriptors

A GPU stage may carry a compiler-emitted `jotnar.stage/1` artifact:

```json
"gpu": {
  "listing": {"artifact": "listing"},
  "descriptor": {"artifact": "descriptor-00"},
  "payloads": [{"artifact": "input-00"}],
  "frames": 1,
  "count": 1
}
```

The controller loads and hashes the descriptor, checks the listing hash, typed
inputs against the actual payload, and exact output names against the stage.
At placement time it rejects a target that is not in the descriptor's
`targets` **before submission**. The receipt records the descriptor hash,
declared targets/ops/determinism, executor, job id, output hashes, and the
exactness verdict. The ledger's plan descriptor includes `descriptor:<sha256>`.

`jotnar build --stage-out FILE [--descriptor-only]` emits the descriptor from
the same artifact and parsed program that produce the manifest. Unknown shapes
are reported as `null` rather than invented; `partitionable` is declarative
and not used for splitting yet.

`jotnar-workload bundle-stage-chain` builds a 1–3 stage GPU chain whose targets
and output names come from descriptors, with an intermediate artifact handoff
between stages.

### Stage graph and ledger

The graph is validated for unknown dependencies and cycles. The order is
deterministic, with declaration order breaking ties.

The ledger records, per stage:

- state, attempt count, executor address, job id;
- resolved command;
- input and output SHA-256 hashes;
- exit code, stdout, stderr, timestamps.

A completed stage is reused only when its command, input hashes, and output
files still match. `--force` starts a fresh ledger. A ledger is tied to the
bundle manifest hash, so changing the manifest requires a new run or `--force`.

### Placement

`--executor ADDR` adds a native executor to the default pool.
`--gpu-executor cuda=ADDR` and `--gpu-executor hip=ADDR` add GPU runners with
their target. `--place STAGE=ADDR` pins a stage and is checked against the
stage's kind and target. Without a pin, stages are assigned across matching
executors in stage order. `--discover-native` browses
`_jotnar-native._tcp.local.`; `--discover-gpu` browses
`_jotnar-gpu._tcp.local.`. `jotnar-net capabilities` includes all three record
kinds.

The controller checks capabilities only as inventory; it never compiles or
interprets a program itself.

### Fleet admission

`--fleet-dir DIR` turns the controller into a fleet controller for batch work:
before submitting a GPU stage it snapshots executor `health`, reads the shared
lease table, and admits or defers the stage against the target's declared
capacity. DOOM registers an interactive lease in the same directory.

```sh
jotnar-workload --token test run ... \
  --fleet-dir /tmp/opencode/fleet --fleet-slots 1 \
  --fleet-max-wait-ms 60000
```

The receipt (`fleet` block) records executor telemetry, per-stage admission
decisions, waits, deferral reasons, the leases seen, and whether enforcement
was advisory. See [`FLEET_ADMISSION.md`](FLEET_ADMISSION.md).

## Commands

Build the native pieces:

```sh
cargo build --release \
  --bin jotnar-demo-doom \
  --bin jotnar-native-runner \
  --bin jotnar-workload
```

Create a bundle:

```sh
./target/release/jotnar-workload bundle-doom \
  --image "$HOME/Documents/OpenCode/jotnar-main/programs/doom/build/doom.bin" \
  --wad "$HOME/Documents/OpenCode/jotnar-assets/doom1.wad" \
  --out /tmp/opencode/doom_bundle --frames 2
```

Run two executors:

```sh
./target/release/jotnar-native-runner --bind 127.0.0.1:9998 --token test \
  --allow "$PWD/target/release/jotnar-demo-doom" \
  --work-root /tmp/opencode/native-a --mdns
./target/release/jotnar-native-runner --bind 127.0.0.1:9999 --token test \
  --allow "$PWD/target/release/jotnar-demo-doom" \
  --work-root /tmp/opencode/native-b --mdns
```

Run frame 1 on A, frame 2 on B, and compare both framebuffers with a
single-node reference:

```sh
./target/release/jotnar-workload --token test run \
  --bundle /tmp/opencode/doom_bundle \
  --ledger /tmp/opencode/doom_run \
  --program "$PWD/target/release/jotnar-demo-doom" \
  --executor 127.0.0.1:9998 --executor 127.0.0.1:9999 \
  --place frame1=127.0.0.1:9998 --place frame2=127.0.0.1:9999 \
  --compare frame1:frame1.fb=/tmp/opencode/doom_ref/frame1.fb \
  --compare frame2:frame2.fb=/tmp/opencode/doom_ref/frame2.fb
```

The reproducible gate builds everything, starts two independent runner
processes on free ports, runs the reference, runs the handoff, and records the
receipt:

```sh
python3 scripts/doom_two_node_gate.py
```

Expected checks:

```text
frame1.frame1.mem: OK ...
frame1.frame1.x: OK ...
frame1.frame1.pc: OK ...
frame1.frame1.cyc: OK ...
frame1.frame1.fb: OK ...
frame2.frame2.mem: OK ...
frame2.frame2.x: OK ...
frame2.frame2.pc: OK ...
frame2.frame2.cyc: OK ...
frame2.frame2.fb: OK ...
RESULT: ALL OK
```

These compare the full RV32IM state, not just a screenshot: the frame-2
checkpoint resumed on node B is byte-identical to a single-node `--frames 2`
reference. `frame1` and `frame2` framebuffers can be identical on the static
title screen; the memory hash still changes.

`jotnar-workload status --ledger DIR` prints the ledger and receipt. The
receipt includes the bundle hash, stage inputs/outputs, executor addresses,
job ids, and comparison hashes.

### Batch GPU pipeline

Build a one-item bundle (the gates use more items):

```sh
./target/release/jotnar-workload bundle-gpu \
  --listing demos/gpu-batch/double.asm \
  --generate 64x64x3 --items 1 \
  --out /tmp/opencode/gpu_batch --name gpu-batch
```

Run the single-node CUDA reference:

```sh
./target/release/jotnar-workload --token test run \
  --bundle /tmp/opencode/gpu_batch \
  --ledger /tmp/opencode/gpu_ref \
  --gpu-executor cuda=192.168.1.116:9997 \
  --gpu-executor hip=192.168.1.195:9997 \
  --place gpu-a-00=192.168.1.116:9997 \
  --place gpu-b-00=192.168.1.116:9997
```

Run the CUDA → HIP chain and compare with the reference:

```sh
./target/release/jotnar-workload --token test run \
  --bundle /tmp/opencode/gpu_batch \
  --ledger /tmp/opencode/gpu_batch_run \
  --gpu-executor cuda=192.168.1.116:9997 \
  --gpu-executor hip=192.168.1.195:9997 \
  --place gpu-a-00=192.168.1.116:9997 \
  --place gpu-b-00=192.168.1.195:9997 \
  --compare gpu-b-00:OUT=/tmp/opencode/gpu_ref/stages/gpu-b-00/OUT
```

`--input FILE` uses an existing JSON payload artifact instead of `--generate`.
`--generate` makes a distinct deterministic payload per item. Stage `gpu-b-NN`
reads `gpu-a-NN`'s binary `OUT` as payload stream `X`; no DOOM binary or
DOOM-specific coordinator is involved. `bundle-gpu` currently emits a
CUDA-first chain; the stage schema also accepts `hip` for a hand-built or future
builder.

The gates:

```sh
python3 scripts/gpu_batch_gate.py --items 8 --size 64x64x3
python3 scripts/doom_gpu_contention_gate.py --items 16 --size 128x128x3 --clients 4
```

`gpu_batch_gate.py` runs a single-node CUDA reference, then the CUDA -> HIP
chain, and compares every item. `doom_gpu_contention_gate.py` warms both
runners, measures the pipeline alone, then runs DOOM on the same CUDA and HIP
runners while one or more pipeline clients run concurrently.

Measured on the live fleet at `192.168.1.116` / `192.168.1.195`:

```text
batch gate, 8 items, 16 stages:  16/16 byte-identical
contention, 4 clients x 16 items: pipeline 64/64 byte-identical
  DOOM p50 39 ms, p95 46 ms, max 239 ms, budget 66 ms
  parallel efficiency 3.94 (4 clients overlapped well)
  verdict: pipeline healthy, DOOM tail unhealthy
```

The single-client run at 64 items stayed inside the DOOM budget
(`max 52 ms`), so the current controller can coexist with DOOM at that scale.
The 4-client run exposes the missing controller layer: batch work is admitted
without priority or leases, so a DOOM frame can wait behind GPU jobs and miss
its budget even when p95 looks healthy.

## Model flagship: pipeline-parallel transformer block

The first model slice is a two-block, single-head pre-norm transformer run as a
stage graph on the live fleet. The controller holds the model manifest,
weights, and input as bundle artifacts; each stage gets one listing plus
payloads, so the executors stay model-blind.

```text
block i:  CUDA attention  X -> RMSNorm -> QKV -> QK^T -> softmax -> AV
                          -> out proj -> +X -> RMSNorm   (outputs N2, RES)
          HIP MLP         N2 -> gate/up -> SiLU(gate)*up -> down -> +RES
                                                       (output OUT)
```

`demos/model-block/attn.asm` and `mlp.asm` are the two listings. Block i>0's
attention consumes block i-1's `OUT` as `X`; the workload DAG carries the
activation. The ops are in the shared CUDA/HIP set (MATMUL, RMSNORM, SOFTMAX,
SILU, MUL, ADD, TRANSPOSE, SLICE, CONCAT). The earlier note that TRANSPOSE was
CUDA-only was stale: slice 3's all-HIP placement matches numpy on the full
graph.

Gate:

```sh
python3 scripts/model_block_gate.py --no-build --blocks 2
```

```text
model      2 blocks, D=64, N=8, inter=128, single head, F32, seed 7
reference  numpy, same seed/weights
CUDA       attn-0, attn-1
HIP        mlp-0, mlp-1
checks     block0 max_abs 3.0e-8, block1 max_abs 3.0e-8
repeat     byte-identical block outputs
control    perturbed WQ tracks the perturbed reference; differs from the
           original by 1.57e-2
receipt    all four stages done, placement and output hashes recorded
```

The block is not spatially partitionable (attention/reductions), so the
distribution axis is stage/pipeline parallelism, not the `partitionable` grid.

### Model slice 2 — checkpoint, multi-head, full sentence

A `jotnar.model.checkpoint/1` manifest and F32 tensor blobs live in the
controller. `scripts/model_sentence_gate.py` writes a deterministic fixture,
loads it back with hash checks, materializes the token embedding, and runs:

```text
tokens -> embedding -> attn-0 (CUDA) -> mlp-0 (HIP) -> attn-1 (CUDA)
       -> mlp-1 (HIP) -> final RMSNorm + head (CUDA) -> logits [8,32]
```

`attn4.asm` slices Q/K/V into four 16-wide heads, runs per-head
`Q·Kᵀ → softmax → A·V`, concatenates the heads with `CONCAT`, and applies the
output projection. `head.asm` adds the final RMSNorm and logits matmul.

```sh
python3 scripts/model_sentence_gate.py --no-build --blocks 2
```

```text
model      2 blocks, D=64, N=8, H=4, Dh=16, inter=128, V=32, F32
checkpoint jotnar.model.checkpoint/1, 28 tensors, hash-verified
CUDA       attn-0, attn-1, head
HIP        mlp-0, mlp-1
checks     block0/1 3.0e-8, logits 2.4e-7 max abs vs numpy
repeat     logits and block outputs byte-identical
control    head + 0.5 tracks the perturbed reference; differs by 6.65
receipt    all five stages done, placement and hashes recorded
```

The checkpoint is a deterministic fixture; the container, loader, graph,
placement, and reference paths are real. A trained checkpoint can replace the
fixture without changing the graph. A larger model and a decode loop are the
next slices.

### Model slice 3 — trained checkpoint, tokenizer, decode, measurement

`demos/model-block/trained/` now holds a real trained F32 checkpoint, not a
fixture. It is a deliberately tiny byte-level model:

```text
checkpoint   jotnar.model.checkpoint/1, F32, 22 tensors, 464,128 bytes
             manifest sha256 01b128cea5b4d2d12297503f2742186825d235ad4e432a0313ed60c7e9d0a4c4
model        D=64, 2 blocks, H=4, Dh=16, inter=128, vocab=256, context=16
tokenizer    byte-level (identity on 0..255)
objective    next byte from a 16-byte context, CE on the last position
corpus       110,491 bytes of in-repo README/docs (sha256 eb51498b...)
training     2,500 steps, batch 64, AdamW lr 2e-3 cosine, seed 0, CPU F32
loss         3.22 bits/byte train, 3.36 bits/byte val (uniform = 8.00)
```

The objective matches the graph. The attention has no causal mask, so a causal
LM objective would leak the future; instead the controller feeds a 16-byte
context and supervises the following byte at the last position. Greedy decode
takes the argmax, appends it, slides the window, and reruns the graph.

The decode gate keeps the checkpoint, tokenizer, prompt, and decode policy in
the controller. It materializes `tok_embed[tokens] + pos_embed[0..16]`, writes
the five-stage bundle, and runs eight steps. The reference is numpy and the
torch model that trained the checkpoint:

```sh
python3 scripts/model_decode_gate.py --no-build \
  --checkpoint demos/model-block/trained --decode 8 \
  --out /tmp/opencode/model-decode-gate
```

```text
prompt        "rity. GPU partit" (validation split; continuation "ioning o")
greedy 8      " the the"
pipeline      attn/head CUDA, mlp HIP: 2.17e-5 full / 4.29e-6 last vs numpy
all-HIP       attn/mlp/head HIP:       1.70e-5 full / 3.81e-6 last vs numpy
torch         numpy-vs-torch           9.63e-6; sequences identical
sequence      exact for both placements; 8 steps x 5 stages all done
repeat        byte-identical logits, noise floor 0.0
control       block0.WQ + 0.5: 3.70e-5 vs the perturbed reference,
              8.08 vs the original; generated sequence changes to " the sta"
bound         1e-4 absolute (observed < 1/4 of the bound)
```

The greedy text is weak -- the model learned byte statistics, not a language
model. The gate reports the true-continuation match as a quality signal, not a
pass criterion. The checkpoint provenance and limits are in
[`demos/model-block/trained/README.md`](../demos/model-block/trained/README.md).

**HIP attention parity is already real.** The HIP target has F32 `TRANSPOSE`,
`SLICE`, `CONCAT`, and `SOFTMAX`; the earlier "TRANSPOSE on CUDA" note was
stale. The all-HIP placement matches numpy on the full graph. No new lowering
was needed for slice 3.

`scripts/model_perf_gate.py` measures the same graph four ways and records what
ran where:

```sh
python3 scripts/model_perf_gate.py --no-build \
  --checkpoint demos/model-block/trained --repeats 10 --warmup 2 \
  --decode 8 --clients 1,2,4 --requests 4 \
  --out /tmp/opencode/model-perf-gate
```

```text
warm, controller load ~6 (32 CPUs), 10 repeats
placement      p50      p99      max      util
single-cuda    2343.8   2394.2   2394.2   0.96 CUDA
pipeline       1702.4   1780.2   1780.2   0.80 CUDA / 0.16 HIP
single-hip      732.0    756.6    756.6   0.91 HIP
decode-8       pipeline 14.05 s, single-hip 5.89 s
throughput     pipeline 0.58 / 0.74 / 0.74 req/s at 1/2/4 clients
               data-parallel 0.86 / 0.89 req/s at 2/4 clients
               data-parallel / pipeline = 1.20x at 4 clients
```

The stage clocks separate dispatch from model time. With a warm build cache the
executor reports `compile_ms=0`, `build_ms=0`, `cache_hit=true`, but `run_ms`
is about 440 ms per CUDA stage and 120-130 ms per HIP stage. Every stage is a
separate executor process launch, and the CUDA context setup dominates this tiny
graph's kernel time. The controller overhead is only 15-26 ms per forward.

Cold, with the runner build caches cleared before each placement:

```text
single-cuda  21.23 s   (3 CUDA builds at ~6.2 s each)
pipeline     16.66 s   (3 CUDA builds + 1 HIP build)
single-hip    7.34 s   (3 HIP builds at ~2.2 s each)
```

Verdict: **single-HIP is the best option, warm and cold.** The pipeline is
1.38x faster than single-CUDA but 2.33x slower than single-HIP. Data-parallel
is only 1.20x the pipeline at four clients because the CUDA half carries the
dispatch cost. This is a correctness and substrate result, not a
compute-scaling win. A resident/fused execution path is required before the
split can be judged on model compute; that is the next slice.

## Limits

- The trusted-LAN token gates executor requests; there is no TLS.
- The native runner refuses a program unless it was passed with `--allow`.
- Inputs and outputs must be relative paths with no `..` or absolute prefix.
  Stage names must be a single safe path component and must not contain `:` or
  `=`, because the ledger uses them as directory names and the CLI uses those
  separators.
- The controller copies full checkpoint files through its own process. The
  64 MB RV32IM memory image is within the frame limits, but this is not a
  chunked or range-fetch transport yet.
- The controller does not retry failed stages yet; the ledger records the
  failure and the run can be resumed after the cause is fixed.
- Fleet leases are advisory and GPU-only. A runner still accepts work from a
  client that ignores the shared directory; CPU stages are not admitted.
- There is no preemption of a running GPU job. Batch work is deferred while an
  interactive lease holds the declared slot.
- The bundle layer supports the native and GPU executor kinds. CPU
  (`jotnar-noded`) stages are not wired into the bundle schema yet.
- GPU `frames` and `count` are 1 in v1. Multi-frame results need an output
  naming rule before the ledger can store them.
- GPU payloads cross the controller-to-runner hop as JSON. A large intermediate
  can expand significantly; there is no binary payload/stream transport yet.
- The controller sends every payload inline. It does not yet use the runner's
  `payload-put`/`payload_id` cache for repeated inputs.
- There is no live executor telemetry, lease, priority, or admission control.
  The contention gate measures the gap; it does not fix it.

## Streaming companion

The bundle/stage layer is batch-oriented. The cooperative two-node DOOM demo
uses a separate streaming coordinator, `jotnar-doom-coord`, because a game
frame is a per-frame request rather than a whole-file stage. It keeps the same
executor contract and adds frame splitting, stitching, and latency receipts.
See [`docs/DOOM_TWO_NODE.md`](DOOM_TWO_NODE.md). A general streaming stage ABI
is future work.

The resource model, lease vocabulary, and measured placement policy that the
next layer needs are in
[`docs/WORKLOAD_COORDINATION.md`](WORKLOAD_COORDINATION.md). The first
implemented slice is `jotnar-net hardware` plus the adaptive CUDA/HIP DOOM
split.
