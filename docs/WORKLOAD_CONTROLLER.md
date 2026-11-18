# Workload controller

**Status:** first native workload runs; DOOM frame 1 → frame 2 checkpoint
handoff is the acceptance test
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

`--executor ADDR` adds an executor to the default pool. `--place STAGE=ADDR`
pins a stage. Without a pin, stages are assigned across the pool in stage
order. `--discover-native` browses `_jotnar-native._tcp.local.` and adds the
advertised native runners. `jotnar-net capabilities` now includes native
records beside CPU and GPU records.

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

## Limits

- The trusted-LAN token gates executor requests; there is no TLS.
- The native runner refuses a program unless it was passed with `--allow`.
- Inputs and outputs must be relative paths with no `..` or absolute prefix.
- The controller copies full checkpoint files through its own process. The
  64 MB RV32IM memory image is within the frame limits, but this is not a
  chunked or range-fetch transport yet.
- The controller does not retry failed stages yet; the ledger records the
  failure and the run can be resumed after the cause is fixed.
- Only the native executor kind is implemented in the first bundle schema.
  CPU/GPU stages can be added without moving planning into those executors.
