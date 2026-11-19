# Workload controller

**Status:** native and GPU stages run; DOOM checkpoint handoff and a CUDA → HIP
batch chain are the acceptance tests
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

Build a deterministic batch: eight items, each a CUDA stage feeding a HIP stage.

```sh
./target/release/jotnar-workload bundle-gpu \
  --listing demos/gpu-batch/double.asm \
  --generate 64x64x3 --items 8 \
  --out /tmp/opencode/gpu_batch --name gpu-batch

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
DOOM-specific coordinator is involved.

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

## Limits

- The trusted-LAN token gates executor requests; there is no TLS.
- The native runner refuses a program unless it was passed with `--allow`.
- Inputs and outputs must be relative paths with no `..` or absolute prefix.
- The controller copies full checkpoint files through its own process. The
  64 MB RV32IM memory image is within the frame limits, but this is not a
  chunked or range-fetch transport yet.
- The controller does not retry failed stages yet; the ledger records the
  failure and the run can be resumed after the cause is fixed.
- GPU stages support native and GPU executor kinds. CPU (`jotnar-noded`) stages
  are not wired into the bundle schema yet.
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
