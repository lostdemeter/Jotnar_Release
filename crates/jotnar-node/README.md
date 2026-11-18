# JotnarOS node crate

The node-side application layer: a model-blind job server, the TCP front door,
the payload supervisor, emitted system-module dispatch, and the multi-node
scheduler. It depends on the workspace root crate and adds no private copy of
the interpreter.

## Binaries

```text
jotnar-noded         TCP protocol front door (Node + async jobs + forwarding)
jotnar-supervisord   long-lived payload supervisor (root-only Unix socket)
jotnar-payload       controller CLI for payload/event/module commands
jotnar-osd           emitted control-plane daemon and step harness
jotnar-net           multi-node scheduler / discovery client
jotnar-native-runner model-blind native/process executor
jotnar-workload      model-aware bundle/stage/checkpoint controller
jotnar-doom-coord    cooperative two-node DOOM frame coordinator
jotnar-doom-view     SDL2 viewer and key forwarder
```

In the image, `jotnar-noded` is installed as `/usr/local/bin/jotnar-node` so
`jotnar-service` and init keep the same names. `jotnar-supervisord`,
`jotnar-payload`, and `jotnar-osd` are installed beside it. On the development
host the controller clients are `target/release/jotnar-ctl` and
`target/release/jotnar-payload` (the image build also copies them to
`artifacts/`).

## Process model

```text
controller
   |  TCP 9999, framed Jotnar protocol (token)
   v
jotnar-noded
   |  ping/program/store/list/drop/exec via jotnar::protocol::Node
   |  job.submit/status/result/cancel/list via the local job engine
   |  payload-* / event-* / module-* via /run/jotnar/supervisor.sock
   v
jotnar-supervisord
   |-- payload threads (TICK/STDIN in, STDOUT out)
   |-- event ring (4096, seq + byte segments)
   |-- capability shims (console/file/proc/timer/net/job)
   `-- /var/lib/jotnar/payloads/<name>/{manifest.json,program.asm,seed.json}
```

The supervisor is independent of the TCP node service: restarting
`jotnar-service` does not stop a running payload. At boot the supervisor scans
the data partition and starts `autostart` payloads.

## Jobs

`jotnar-noded` owns an async job engine (`src/jobs.rs`) over `Node::snapshot`:

- bounded worker pool and queue (`--workers`, `--queue`), `queue-full` rejection;
- submit, status, result, cancel, list, idempotency keys, timeouts;
- cooperative cancellation installed with `jotnar::runtime::CancelGuard`;
- inline results, or store-backed chunked results for large outputs;
- result hashes, reader leases, eviction, and bounded memory;
- the same `Node::handle("exec")` path as synchronous execution, so receipts
  and determinism stay honest.

The host gate is `crates/jotnar-node/test/qemu_jobs.sh`; the physical gate is
`crates/jotnar-node/test/physical_jobs.sh`. `jotnar-ctl` exposes the matching
`submit`, `job-status`, `job-result`, `job-cancel`, `job-wait`, `job-run`,
`job-batch`, `result-read`, and `job-list` commands.

## Host streams

| stream | kind | direction | behavior |
|---|---|---|---|
| `TICK` | I64 [1] | supervisor → payload | current Unix ms, one execution per `tick_ms` |
| `STDIN` | U8 [n] | supervisor → payload | seed value, or one `event-write` payload (one shot) |
| `STDOUT` | U8/I64 [n] | payload → supervisor | local console (`tty0`/`ttyS0`) + event ring |
| `STATE` streams | any | payload ↔ supervisor | carried across events and persisted to `state.ndjson` |

Integer ops may widen `U8` to `I64`; the supervisor treats `STDOUT` I64 elements
as bytes. If a payload declares no `TICK`, it runs only on `event-write`.

## Protocol additions

All additions are under the shared `PROTO_V = 1`; optional fields do not bump it.
`ping` reports the available features and, when the supervisor is reachable:

```json
"features": ["jobs.v1", "payload.v1", "events.v1"],
"supervisor": {"state": "ready", "payloads": 0},
"jobs": {"workers": 4, "queued": 0, "result_transport": "inline+chunked"}
```

Payload requests:

```text
payload-put     name, autostart, restart, tick_ms, console, seed, body=listing
payload-start   name
payload-stop    name
payload-reload  name
payload-status  [name]
payload-drop    name
event-poll      after_seq, max, kinds[]
event-write     name, target=stdin|file, path, mode=write|append, body=bytes
file-read       name, path            -> file-data (body=bytes)
```

Job requests:

```text
submit          handle, inputs{store|inline}, outputs, threads, policy,
                timeout_ms, idempotency_key
job-status      job_id
job-result      job_id, off, len
result-read     result_id, off, len
result-drop     result_id
job-cancel      job_id
job-list
```

`event-poll` is non-blocking and returns an `event-batch` with `seq`, `ts_ms`,
`kind`, `stream`, `off`, `len`, `next_seq`, `head_seq`, and `dropped`. Poll
with `after_seq = head_seq - 1` to include the next event.

`payload-put` can declare file capabilities with `files: ["notes.txt"]`.
`event-write target=file` and `file-read` are restricted to those relative
paths under `payloads/<name>/files/`; absolute paths, `..`, and undeclared files
are refused.

STATE streams (`STATE name` plus `IN name` seed) are carried across executions
and snapshotted atomically to `payloads/<name>/state.ndjson` after each run.
Starting a payload restores the snapshot, so stop/start and reboot continue
from the last value.

## Emitted system modules

`build.rs` assembles every module under
`image/rootfs/usr/local/share/jotnar/modules/` and emits Rust step functions
plus a `SystemModule` enum into `jotnar-osd`. The emitted path is the boot
control plane; the interpreted `module-step` path is the parity oracle.

```sh
jotnar-osd --event 10                 # one emitted step + capability command
jotnar-osd --events 10,11,12          # stateful sequence
```

`crates/jotnar-node/test/qemu_smoke.py --m8` checks all emitted modules;
`--m11` runs randomized interpreted/emitted parity.

## Scheduler and discovery

`jotnar-net` speaks the same protocol to several nodes:

- health checks, heartbeat registry, draining state, and live queue depth;
- placement and aggregation across nodes;
- durable per-batch ledger, cached summaries, resume, and retries;
- stable node IDs and mDNS advertisement/discovery (`_jotnar-node._tcp.local.`);
- `run` for distributed batches and `discover` / `--discover` for registration;
- `capabilities` merges `_jotnar-node`, `_jotnar-gpu`, and
  `_jotnar-native` advertisements into one CPU/GPU/native inventory without
  moving execution into the node. See
  [`docs/CAPABILITY_EXCHANGE.md`](../../docs/CAPABILITY_EXCHANGE.md).
- `hardware` normalizes the same records into `jotnar.hardware/1` and can
  probe TCP reachability/RTT with `--probe`. See
  [`docs/WORKLOAD_COORDINATION.md`](../../docs/WORKLOAD_COORDINATION.md).
- `gpu` routes one stage to a GPU runner through the common executor contract
  (`submit`/`job-status`/`job-result`/`job-cancel`), selecting a runner from
  discovery unless `--addr` is given. Binary result descriptors are validated
  before reporting. See
  [`docs/EXECUTOR_CONTRACT.md`](../../docs/EXECUTOR_CONTRACT.md).

## Native runner and workload controller

`jotnar-native-runner` runs one allow-listed local process per job. It stages
named input files into a private work directory, runs a relative argv, and
returns named output files with SHA-256 descriptors through the same job
contract. It is model-blind: DOOM, model bundles, and stage graphs are not in
this binary.

`jotnar-workload` is the model-aware layer. It validates a content-addressed
bundle manifest, orders a stage graph, places stages on executors, moves
checkpoint files between stages, and records a ledger and receipt. The first
bundle is DOOM on the RV32IM core:

```sh
cargo build --release --bin jotnar-demo-doom
cargo build --release -p jotnar-node-tools --bin jotnar-native-runner --bin jotnar-workload
python3 scripts/doom_two_node_gate.py
```

The gate runs frame 1 on one native executor, transfers the checkpoint to a
second executor, resumes for frame 2, and compares the full RV32IM state with
a single-node reference. See
[`docs/WORKLOAD_CONTROLLER.md`](../../docs/WORKLOAD_CONTROLLER.md).

`jotnar-doom-coord` is the streaming companion. It runs the RV32IM core on the
coordinator host, calibrates the CUDA and HIP runners, splits each frame by
measured throughput, runs the same Jotnar `INTERP` listing on both, and
stitches the results. `jotnar-doom-view` displays the stitched frame and
forwards keys over TCP. `scripts/doom_two_gpu_gate.py` records the annotated
GIF and latency receipt; `scripts/doom_failover_gate.py` closes one node's
proxy mid-run and checks that the survivor finishes the frames. See
[`docs/DOOM_TWO_NODE.md`](../../docs/DOOM_TWO_NODE.md) and
[`docs/WORKLOAD_COORDINATION.md`](../../docs/WORKLOAD_COORDINATION.md).

`crates/jotnar-node/test/sched_two_nodes.sh` is the host gate.
`crates/jotnar-node/test/qemu_two_nodes.sh` boots two disposable QEMU nodes.

## Image and tests

```sh
./crates/jotnar-node/image/build-node-image.sh
./crates/jotnar-node/test/run_qemu.sh
./crates/jotnar-node/test/qemu_smoke.py --ssh-key ~/.ssh/id_ed25519 --m8
./crates/jotnar-node/test/qemu_smoke.py --m11
./crates/jotnar-node/test/qemu_jobs.sh
./crates/jotnar-node/test/sched_two_nodes.sh
ADDR=192.168.1.116:9999 ./crates/jotnar-node/test/physical_jobs.sh
```

The image builder pins the Alpine minirootfs and kernel, builds the node
binaries for `x86_64-unknown-linux-musl`, and writes a GPT disk image under
`artifacts/`. It also records the workspace commit in `artifacts/build-info.txt`.
See `image/README.md`.

## Test fixtures

```text
test/heartbeat.asm      IN STDIN; STDOUT = IADD(STDIN, 0)
test/seed-tick.json     U8 "tick\n" seed
test/counter.asm        STATE COUNT; COUNT = IADD(COUNT, 1)
test/seed-counter.json  I64 COUNT = 0
test/hello.asm          IN STDIN; STDOUT = IADD(STDIN, 0)
test/seed-hello.json    U8 "Hello JotnarOS!\n"
```

## Boundaries

This crate implements Claim A: a Linux-hosted system layer described in Jotnar
and emitted to Rust. It is not a Jotnar-native kernel (Claim B). The shared
token is trusted-LAN convenience, not TLS/mTLS; result hashes use SHA-256 but
the job store is memory-backed and not a durable journal. See
`showcase/node-os/` for the current claim boundary and evidence.
