# Runtime determinism and threading (research note)

Status: language-repo research note, part of **exhibit 10** (interpreter +
model-blind node). It documents the model-blind runtime design and the
reproducible language-side gates. It is **not** a production serve claim.
See [`showcase/runtime/CLAIM_BOUNDARY.md`](../showcase/runtime/CLAIM_BOUNDARY.md)
for the certified/research split.

The one-line thesis:

> The runtime can stay generic and simple while approaching bespoke-program
> performance if we spend the effort on kernels, buffer reuse, deterministic
> scheduling, and optional acceleration backends — not on model-specific code
> or a JIT.

## What changed

All changes are model-blind: streams, shapes, kernels, and a per-job thread
budget. No application logic entered the runtime core.

| file | change |
|---|---|
| `src/profile.rs` | opt-in per-op wall/allocation/output profile and a counting allocator wrapper. |
| `src/runtime.rs` | `ExecOptions { threads }`, `ThreadPolicy`, `SCHED_VERSION`, `run_with_options`, `run_profiled`; per-op instrumentation. |
| `src/ops/mod.rs` | per-job thread budget (thread-local execution context) and `dispatch_with_threads`. |
| `src/ops/t2.rs` | routes MATMUL, CONV2D, DECONV2D, SOFTMAX through partitioned kernels. |
| `crates/phi-core/src/t2.rs` | output-partitioned MATMUL/CONV2D/DECONV2D/SOFTMAX; reused convolution scratch. |
| `src/bin/jotnar-repl.rs` | `--threads N`, `--profile`. |
| `src/bin/jotnar-det.rs` | exact-output determinism harness (threads, repeats, concurrency). |
| `src/bin/jotnar-ctl.rs` | exec `--threads`, `--policy`, `--receipt`. |
| `tests/determinism.rs` | CI gate: T2 kernels bit-identical across 1..16 threads and concurrent jobs. |
| `tests/protocol.rs` | T2 plane transport, receipts, thread policy, ping/build identity, frozen-T compatibility. |
| `scripts/bench_runtime_warm.sh` | `JOTNAR_THREADS`, `JOTNAR_PROFILE`. |

The thread budget is separate from the listing's `CONFIG threads` (a
compiled-target directive). `ExecOptions::default()` stays **one thread**;
auto-threading is an explicit boundary policy
(`ThreadPolicy::{Reference, DeterministicAuto}`).

## Where determinism actually lives

The kernels are close to pure per-output functions: an element of a MATMUL,
CONV, or SOFTMAX row is a function of its inputs and a fixed loop order, not
of the worker that ran it. The determinism that matters is at **boundaries
where order is semantically required**:

- an op publishes its complete output before any consumer reads it;
- graph edges define intra-job order; independent streams may finish in any
  order and that order is unobservable;
- `REPEAT`/STATE carry is the explicit serialisation point (iteration `n`
  before `n+1`);
- the partition map and thread budget are pure functions of shape/count;
- a job publishes outputs and a receipt as one unit, never partial;
- program/blob stores are immutable and content-addressed;
- framed transport is byte-exact, so local and remote are indistinguishable.

Everything else — worker identity, queue position, completion order, cache
state — must be allowed to vary. That is the rule the harness enforces.

## Parallelization rules

- Parallelize independent outputs only: MATMUL rows, CONV/DECONV output
  elements, SOFTMAX rows.
- Partition by a pure function of shape and thread count: contiguous blocks of
  `rows.div_ceil(threads)` (or `out_len.div_ceil(threads)`).
- Each output element keeps the serial exponent scan and ascending
  accumulation order. No reduction is split across threads.
- One scratch buffer per worker, reused across outputs; no cross-job state.
- `threads == 1` always takes the reference path.

## Harness and CI shape

`jotnar-det` is the model-blind harness. It assembles once, replays NDJSON
payloads, hashes the requested streams, and exits 1 on a mismatch:

```sh
jotnar-det <listing.asm> <payload.ndjson> --outputs OUT \
  --threads 1,4,16 --repeat 2
jotnar-det <listing.asm> <payload.ndjson> --outputs OUT \
  --threads 1,4,16 --repeat 1 --concurrency 1,2,8 \
  --job-threads 4 --stagger-ms 20
```

The short CI gates are `tests/determinism.rs`, `tests/protocol.rs`,
`tests/nonfpu_t2.rs`, and `tests/transport.rs`. Large application matrices
(12-block backbone, neck, head) are release/performance checks on companion
payloads, not CI gates. The companion depth-conversion repository records the
DAV2 application measurements; they are not reproduced here.

## Profile and kernel lesson

The profile is opt-in and advisory: per-op wall time, allocated bytes, input
bytes, and output bytes. It is never part of a determinism verdict.
Profiling found that a T2 convolution kernel was allocating a fresh term
vector per output pixel; one reused scratch buffer per worker removed the
allocation churn. At the layer level, MATMUL remains the dominant cost, so
kernel quality is the next CPU lever; an arena was not the layer-level
bottleneck.

## Protocol surface

- `exec` headers accept `threads: N` (literal job budget) and
  `thread_policy: "reference" | "auto"`; the default is the one-thread
  reference order.
- `exec-ok` carries a canonical `receipt`: program/blob/input/output hashes,
  determinism profile, backend/kernel, `runtime_version`
  (`<crate>+<short-git>`), `sched_version`, resolved threads, partition
  descriptor, and verdict. `profile_ops: true` adds advisory timing outside
  the receipt.
- T2 streams use `T/t2` (s:i8, e:i32, z:u8, m:i32 = 10 bytes/element);
  frozen T keeps the interleaved 24-byte `i64x3` form.
- `NodeHandle` uses a bounded request queue. It still has one worker, so
  `program`/`store` can queue behind a long `exec` until `Arc` snapshots
  land.

## What remains (Stage E research)

1. Concurrent TCP jobs, cancellation/timeouts, and retry/health behavior.
2. `Arc` snapshots that decouple `program`/`store` from a long `exec`.
3. A per-job memory policy and backpressure for large stores.
4. Chunked/segmented or store-backed results for outputs larger than one
   frame. A 64 MiB frame carries ~6.7M T2 elements; application neck/head
   outputs are much larger.
5. A stronger content address than FNV-1a 64-bit before claiming immutable
   dedup.
6. F32 first-class streams/kernels and optional GPU acceleration; the gap to
   fast-float is precision/acceleration, not interpreter dispatch.
7. A separate numbered exhibit if the job server is promoted beyond exhibit
   10, with the root README, `docs/LOOP.md`, manual chapter, and manual output
   updated together.

## Claim boundary

- Certified here: hash-identical model-blind T2 kernels across the tested
  thread counts/repeats/concurrent in-process jobs, canonical receipts,
  build provenance, explicit thread policy, T2 small-output transport, and
  the bounded queue.
- Not certified here: DAV2 full-depth/FPS numbers, live serve, large-output
  or concurrent TCP, cancellation/timeouts, memory policy, F32/GPU parity,
  or a cryptographic content address.

See [`showcase/runtime/CLAIM_BOUNDARY.md`](../showcase/runtime/CLAIM_BOUNDARY.md).
