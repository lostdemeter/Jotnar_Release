# Executor contract

**Status:** GPU runner implements the async job contract; CPU node already did
**Scope:** controller-to-executor boundary

## Contract

Both executors speak the same framed job API:

```text
submit      -> submit-ok      { job_id, state }
job-status  -> job-status-ok  { job }
job-result  -> job-result-ok  { job_id, outputs, ... } + result payload
job-cancel  -> job-cancel-ok  { job_id, accepted, previous, state }
```

A `submit` header carries the executor-specific job spec. The result envelope
is executor-independent: `job-result-ok` preserves the target's output
descriptors and may carry a binary payload.

The controller-side trait is
`jotnar_node_tools::executor::Executor`:

```rust
fn capability(&self) -> &Json;
fn submit(&self, spec: &Json, payload: &[u8]) -> Result<Json, (String, String)>;
fn status(&self, job_id: &str) -> Result<Json, (String, String)>;
fn result(&self, job_id: &str) -> Result<(Json, Vec<u8>), (String, String)>;
fn cancel(&self, job_id: &str) -> Result<Json, (String, String)>;
```

`RemoteExecutor` pairs a discovery capability record with a `JobClient`.

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
the result, and prints a summary. It does not compile, schedule batches, or
interpret the model.

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
- CPU stored-result chunk fetching remains in the CPU scheduler path.

## Next

A workload controller can use the capability list and the `Executor` trait to
place stages, then fetch results through the same envelope. DOOM across two
nodes is the first planned acceptance test.
