# Executor contract

**Status:** GPU and native runners implement the async job contract; CPU node
already did
**Scope:** controller-to-executor boundary

## Contract

Both executors speak the same framed job API:

```text
submit      -> submit-ok      { job_id, state }
job-status  -> job-status-ok  { job }
job-result  -> job-result-ok  { job_id, outputs, ... } + result payload
job-cancel  -> job-cancel-ok  { job_id, accepted, previous, state }
ping        -> pong           static capability probe
health      -> health-ok      { schema, limits, activity, jobs[] }
```

A `submit` header carries the executor-specific job spec. The result envelope
is executor-independent: `job-result-ok` preserves the target's output
descriptors and may carry a binary payload.

`health` is the live view used by the fleet controller: queue depth, running
jobs, warm workers, bounded recent-job timing, and a timestamp. It is reporting
only; the executor never places work. `ping` stays static and cheap, and live
telemetry is never put in mDNS adverts.

The controller-side trait is
`jotnar_node_tools::executor::Executor`:

```rust
fn capability(&self) -> &Json;
fn submit(&self, spec: &Json, payload: &[u8]) -> Result<Json, (String, String)>;
fn status(&self, job_id: &str) -> Result<Json, (String, String)>;
fn result(&self, job_id: &str) -> Result<(Json, Vec<u8>), (String, String)>;
fn cancel(&self, job_id: &str) -> Result<Json, (String, String)>;
fn telemetry(&self) -> Result<Json, (String, String)>;
```

## Binary payloads

A streaming controller can avoid JSON expansion by setting
`payload_format: "binary"` on a `submit`/`run` header:

```text
body = {"listing": "...", "payload": {"streams": {
         "X": {"kind": "F32", "shape": [H, W, C], "off": 0, "bytes": N}
       }}}
       \n
       raw little-endian values (F32/F64/I64/U8/Bool)
```

The runner decodes the metadata line and the bytes into the same `Feeds` the
JSON path produces, then compiles and runs the same listing. The JSON payload
path is unchanged and remains the default. A `payload_id` request still uses
the JSON cache; binary payloads are inline in this slice.

For streaming image results, `output_u8: true` asks the runner to round and
clamp F32/F64 output values to `U8` before transport. The result descriptor
says `kind: "U8"`, with one byte per value. This is a transport encoding, not
a different program: the listing still computes in F32. It cuts the four-tile
wall output from ~11 MB/frame to ~2.8 MB/frame.

The executor still receives one listing plus one payload per job. It does not
learn about streams, tiles, source adapters, or sinks.

`RemoteExecutor` pairs a discovery capability record with a `JobClient`.
`validate_binary_result` checks descriptor bounds, frame indices, kinds,
shapes, and byte widths before a controller trusts a binary result.

## CPU node

`jotnar-noded` already had `submit`, `job-status`, `job-result`, and
`job-cancel`. Jobs are queued against an immutable registry snapshot, get a
bounded worker pool, and retain inline/stored results for a lease. The
controller keeps the model-blind boundary.

## GPU runner

`jotnar-gpu-runner` now exposes the same four methods:

- `submit` queues a job and executes it in a background thread through the
  existing `run`/`run-batch` path;
- `job-status` reports queued/running/done/failed/canceled;
- `job-result` returns the stored `run-ok`/`run-batch-ok` header and payload;
- `job-cancel` cancels queued jobs; a running GPU job may reject cancellation.

The GPU job table is bounded and evicts completed jobs first, never queued or
running jobs. The executor still owns compilation, warm workers, caches, and
batch frames; the controller does not compile or run listings.

`jotnar-gpu-net` remains the GPU-specific batch controller and benchmark
driver. The async endpoint is the stable executor face for `jotnar-net` and the
future workload controller.

## Native/process runner

`jotnar-native-runner` is the third executor. It is model-blind and generic:

- `submit` takes `command[]`, `inputs[]`, `outputs[]`, and an optional
  `timeout_ms`;
- the body is the concatenated input files named by the descriptors;
- the runner writes inputs into a private work directory, runs an
  allow-listed program with a relative argv, then stores the requested output
  files;
- `job-result` uses `output_format: "files"`, one `U8` descriptor per output
  with an offset, byte count, and SHA-256, plus the concatenated body;
- `job-status` and `job-cancel` match the other executors; a running process is
  killed on cancel.

The runner refuses programs that were not passed with `--allow`, and it
refuses absolute paths or `..` in input/output names. It advertises
`_jotnar-native._tcp.local.` when `--mdns` is enabled. It contains no DOOM,
model, or stage-graph logic.

## Controller route

`jotnar-net` can route one GPU stage through the contract:

```sh
jotnar-net --token T gpu listing.asm \
  --target cuda \
  --payload payload.json --outputs OUT \
  --frames 1 --count 1
```

If `--addr` is omitted it selects a matching `kind:gpu` capability from
discovery. `--addr` pins one runner. The command submits, polls status, fetches
the result, validates binary descriptors, and prints a summary. It does not
compile, schedule batches, or interpret the model.

## Boundaries

- Discovery is inventory; execution is a separate call.
- Tokens still gate every executor request.
- CPU node stays model-blind.
- GPU runner stays target-aware.
- `jotnar-net` owns routing, not target-specific compile/run logic.
- Workload/model semantics belong to the next layer.

## Known limits

- GPU async results are inline; there is no chunked result store yet.
- GPU cancellation is best-effort for running jobs.
- The GPU executor supports one frame or a run-batch frame count per job.
- The native executor stages whole files in memory; the 64 MB RV32IM
  checkpoint fits, but there is no range-fetch or streaming transport yet.
- CPU stored-result chunk fetching remains in the CPU scheduler path.

## Next

The workload controller uses the capability list and the `Executor` trait to
place stages, move checkpoints, and write a ledger/receipt. See
[`docs/WORKLOAD_CONTROLLER.md`](WORKLOAD_CONTROLLER.md). DOOM across two
nodes is the first acceptance test.

The cooperative per-frame demo uses the same contract from a streaming
coordinator: node A runs the RV32IM core, and the CUDA and HIP runners execute
the same Jotnar upscale listing on half a frame each. See
[`docs/DOOM_TWO_NODE.md`](DOOM_TWO_NODE.md).
