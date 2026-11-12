# Runtime threading: first measured result

Status: research, branch `research/jotnar-job-server`. Not a gate. Build:
`cargo build --release --bin jotnar-repl --bin jotnar-det`, machine AMD
Ryzen 9 7950X3D (16 cores / 32 threads), 61 GiB RAM.

Question: can the generic warm runtime approach bespoke compiled performance
with deterministic kernels and scheduling, without model-specific code?

Short answer: **yes on the measured DAV2 stages.** The T2 layer-0 518 px
workload drops from ~29 s serial to ~2.7 s at 16 threads with a bit-identical
`OUT` hash. The 12-block backbone drops from ~350 s serial to ~28 s, inside
1.1x of the compiled 16-thread fast-int backbone (29.0 s). The neck's T2
convolution path was allocation-bound, not arithmetic-bound: reusing one term
buffer per worker cut the neck from ~64 s to ~5 s at 16 threads and removed
~460 GiB of per-pixel allocation churn.

## What changed

All changes are model-blind: streams, shapes, kernels, and an execution
thread budget. No DAV2 logic entered the runtime core.

| file | change |
|---|---|
| `src/profile.rs` | opt-in per-op wall/allocation/output profile and a counting allocator wrapper. |
| `src/runtime.rs` | `ExecOptions { threads }`, `run_with_options`, `run_profiled`; per-op instrumentation. |
| `src/ops/mod.rs` | per-job thread budget (thread-local execution context) and `dispatch_with_threads`. |
| `src/ops/t2.rs` | routes MATMUL, CONV2D, DECONV2D, SOFTMAX through partitioned kernels. |
| `crates/phi-core/src/t2.rs` | output-partitioned MATMUL/CONV2D/DECONV2D/SOFTMAX; reused convolution scratch. |
| `src/bin/jotnar-repl.rs` | `--threads N`, `--profile`. |
| `src/bin/jotnar-det.rs` | exact-output determinism harness (threads, repeats, concurrency). |
| `tests/determinism.rs` | CI gate: T2 kernels bit-identical across 1..16 threads and concurrent jobs. |
| `scripts/bench_runtime_warm.sh` | `JOTNAR_THREADS`, `JOTNAR_PROFILE`. |

The thread budget is separate from the listing's `CONFIG threads` (a compiled
target directive). It is set at the job boundary; the program stays portable.

## Determinism rules used

- Parallelize independent outputs only: MATMUL rows, CONV/DECONV output
  elements, SOFTMAX rows.
- Partition by a pure function of shape and thread count: contiguous blocks of
  `rows.div_ceil(threads)` (or `out_len.div_ceil(threads)`).
- Each output element keeps the serial exponent scan and the ascending `t`
  accumulation order. No reduction is split across threads.
- One scratch buffer per worker, reused across outputs; no cross-job state.
- `threads == 1` always takes the reference path.

### Where determinism actually lives

The kernels are close to pure per-output functions: an element of a MATMUL,
CONV, or SOFTMAX row is a function of its inputs and a fixed loop order, not of
the worker that ran it. The determinism that matters is at **boundaries where
order is semantically required**:

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

Evidence:

- `jotnar-det` on T2 layer 0 at 518 px: `OUT` hash `68960ef97c3c1960` at
  threads 1, 2, 4, 8, 16 (mismatch is a hard failure).
- `jotnar-det` on T2 neck at 518 px: `NA0..NA3` hash
  `811c687f1de57a9d` at threads 1 and 16.
- T2 layer-0 112 px matrix: threads 1,4,16, repeat 2, concurrency 2 and 8
  (job budget 4) all match the serial hash.
- `cargo test --release --test determinism` and `--test nonfpu_t2` pass. The
  latter still compares the interpreter against the compiled NonFpu C oracle.

## Measurements (warm process, one 518 px frame)

Baseline before this branch: `scripts/bench_runtime_warm.sh`, T2 layer 0,
~29.0 s and ~983 MiB RSS at 518 px.

Per-op profile of one T2 layer-0 518 px pass, 16 threads
(`--threads 16 --profile`):

| op | calls | wall | share | allocated | output |
|---|---:|---:|---:|---:|---:|
| MATMUL | 35 | 2.38 s | 90.4% | 175 MiB | 168 MiB |
| CONCAT | 9 | 0.073 s | 2.8% | 54 MiB | 52 MiB |
| GELU | 1 | 0.062 s | 2.4% | 24 MiB | 20 MiB |
| SOFTMAX_WIDE | 6 | 0.054 s | 2.1% | 114 MiB | 107 MiB |
| ADD | 2 | 0.031 s | 1.2% | 10 MiB | 10 MiB |
| LAYERNORM | 2 | 0.027 s | 1.0% | 11 MiB | 10 MiB |
| TRANSPOSE | 6 | 0.004 s | 0.2% | 6 MiB | 5 MiB |

Serial profile before threading: MATMUL was 98.5% of 32.1 s. Allocation volume
for MATMUL is ~1x its output, so copies/arenas are **not** the layer-level
bottleneck; kernel throughput and parallelism are.

| workload (518 px) | serial | 16 threads | speedup | note |
|---|---:|---:|---:|---|
| T2 layer 0 | 29.0 s | 2.65–2.72 s | ~10.9x | warm, 3 runs; ~1.06 GiB RSS |
| T2 12-block backbone | 335.4 s | 27.7 s | 12.1x | includes seam; 4 threads: 88.8 s |
| T2 neck | 63.9 s | 5.2 s | 12.3x | 27 CONV2D + 2 DECONV2D |
| T2 head | 39.0 s | 3.15 s | 12.4x | 3 CONV2D + resize |
| full depth (sum of stages) | 438.3 s | **36.1 s** | 12.1x | backbone + neck + head |
| backbone concurrency 2 x 8 threads | — | 52.9 s slowest | — | 2 jobs, hash-equal |

Compiled oracles from the prior handoff: single-thread T2 layer 0 ~28.3 s;
16-thread fast-int 12-block backbone 29.0 s; fast-float torch depth 75.4 fps.
The threaded runtime backbone (27.7 s) is now **at or slightly faster than the
compiled 16-thread fast-int backbone** on this machine. The prior compiled
full-depth number (166 s) includes builds and encode, so it is not a pure
compute oracle; stage-wise comparison is the honest one.

The remaining gap to fast-float (13 ms/frame) is GPU/F32, not interpreter
dispatch or CPU scheduling.

## The neck finding

The original `phi_core::t2::conv2d` allocated a fresh `Vec<(T2, T2)>` for every
output element. On a 3x3 convolution with 384 input channels that is 3456 heap
pushes per output pixel; one pre-fix profile reported **460.79 GiB allocated**
for 170 MiB of output. Reusing one scratch buffer per worker (same term order,
same accumulator) plus output partitioning brought allocation volume down to
183 MiB and the neck to **5.2 s at 16 threads / 63.9 s serial**. The head's
three convolutions went from a 43 s-class stage to 3.15 s at 16 threads. This
confirms the performance thesis at a more interesting level than dispatch: the
runtime was leaving an order of magnitude on the table in a kernel, not in the
language.

## Determinism matrix run

```sh
# T2 layer 0, 518 px, exact hash
target/release/jotnar-det /tmp/opencode/runtime_t2_518/layer0.asm \
  /tmp/opencode/runtime_t2_518/payload.json \
  --outputs OUT --threads 1,2,4,8,16 --repeat 1

# T2 12-block backbone, 518 px (serial reference ~5.5 min)
target/release/jotnar-det /tmp/opencode/depth_t2_518/backbone/backbone.asm \
  /tmp/opencode/depth_t2_518/backbone/payload.json \
  --outputs M0,M1,M2,M3 --threads 1,4,16 --repeat 1 \
  --concurrency 2 --job-threads 8

# T2 neck, 518 px
target/release/jotnar-det /tmp/opencode/depth_t2_518/neck/neck.asm \
  /tmp/opencode/depth_t2_518/neck/payload.json \
  --outputs NA0,NA1,NA2,NA3 --threads 1,16 --repeat 1

# T2 head, 518 px
target/release/jotnar-det /tmp/opencode/depth_t2_518/head/head.asm \
  /tmp/opencode/depth_t2_518/head/payload.json \
  --outputs OUT --threads 1,16 --repeat 1

# T2 layer 0, 112 px: repeats + staggered concurrent jobs
target/release/jotnar-det /tmp/opencode/runtime_t2_112/layer0.asm \
  /tmp/opencode/runtime_t2_112/payload.json \
  --outputs OUT --threads 1,4,16 --repeat 2 --concurrency 1,2,8 \
  --job-threads 4 --stagger-ms 20
```

The harness prints program/payload hashes, a per-thread median, and a verdict;
any mismatch exits 1.

Recorded hashes (this build): layer-0 `OUT` `68960ef97c3c1960` at 518 and
`7cf39ca993d814e0` at 112; backbone `M0..M3` `38fb36f90fbb25e5`; neck
`811c687f1de57a9d`; head `321940074394c7bb`.

## Protocol/node follow-up (review integration)

The T2/session review found two drifts in `src/protocol.rs`; both are fixed:

- `ping` now reports `ops::registry().len()` instead of a hardcoded 86.
- `Node::exec` honours execution options: `threads: N` is a literal job
  budget, and `thread_policy: "auto" | "reference"` is explicit. The default
  stays the 1-thread reference order. `ExecOptions::default()` in the library
  remains 1; auto-threading lives at the job/CLI boundary.
- `exec-ok` now carries a canonical `receipt` (BTreeMap-backed JSON, no
  timing): program/blob/input/output hashes, determinism profile, backend,
  kernel, runtime version, `sched_version`, resolved threads, partition
  descriptor, and verdict. `profile_ops: true` adds advisory per-op timing
  *outside* the receipt.
- T2 streams now round-trip all four planes over the protocol using a compact
  `T/t2` wire dtype (s:i8, e:i32, z:u8, m:i32 = 10 bytes/element). Frozen T
  keeps the existing `T/i64x3` format.
- `NodeHandle` uses a bounded request queue (`spawn_bounded`; default 64) so
  callers get backpressure. Program/store requests still share the single
  worker, and immutable `Arc` snapshots are the next job-server step.
- `jotnar-ctl exec` accepts `--threads`, `--policy reference|auto`, and
  `--receipt`. A single-job TCP smoke with the 112 px T2 layer produced
  identical output bytes for reference and auto; receipts matched on program
  and output hashes and differed only in resolved threads/partition.

**Transport size limit:** a 64 MiB frame carries about 6.7M T2 elements. A
296x296x384 neck output is ~336 MiB, so neck/head job outputs need
chunked/segmented transport (or a store-backed result) before TCP parity is
claimed. The in-process `NodeHandle` uses the same frames.

## What remains

1. **Backbone/full-depth determinism**: green at 1/4/16 threads (and a
   two-job concurrency check). Add the same run to CI on a shorter 2-block
   listing; keep the 12-block matrix as a release check.
2. **Kernel quality**: matmul is now ~90% of the layer at 16 threads. SIMD and
   cache blocking are the next measured steps; do not start with a JIT.
3. **Neck/head elementwise work**: `RESIZE2D`, `RELU`, `ADD` are serial and
   together are ~4–6% of the optimized neck/head. Cheap row/pixel partitions.
4. **Memory**: one layer peaks near 1.06 GiB RSS; the full 12-block store will
   be larger. A per-job arena and blob store are the job-server next step.
5. **F32 and optional GPU**: the remaining gap to fast-float is a precision and
   acceleration decision, not a scheduling one.
6. **Node/job server**: in-process `NodeHandle`, bounded queue, content-addressed
   blob cache, framed TCP. The runtime core is ready to sit behind it.

## Recommendation

Promote deterministic threading to the default CPU execution path for exact-int
T2: the layer, backbone, neck, and head matrices are green, including repeats
and concurrent jobs. Keep the compiled fast-int and fast-float paths as the
production oracles while the job-server path is built. The first experiment
supports the thesis: no model-specific code and no JIT were needed to move from
29 s to 2.7 s per layer, and a kernel fix moved the neck by ~12x. Next decisions
should be driven by the same profile-first loop — matmul kernel quality before
F32/GPU, and server mechanics before more optimization.
