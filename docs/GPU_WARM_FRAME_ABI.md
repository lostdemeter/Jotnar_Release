# GPU warm-frame ABI (design)

**Status:** proposed, not implemented
**Scope:** host-side `jotnar-gpu-runner`; CUDA and HIP targets
**Branch:** `gpu-warm-frame-design`

## Why

The current runner compiles once and caches the executable, but it spawns that
executable once per job. Each job pays process launch, GPU context creation,
kernel/module loading, and input/output file I/O.

Profiling at `efc4f0c`, one warm artifact, one runner per target:

| MATMUL size | HIP per job | CUDA per job |
|---|---:|---:|
| 64³ | ~126 ms | ~439 ms |
| 256×512×256 | ~129 ms | ~443 ms |
| 512³ | ~131 ms | ~424 ms |

The time is nearly flat across a 512× increase in work, so the fixed per-job
cost dominates. `--parallel` hides some of it by overlapping processes, but it
does not remove it.

The existing CUDA `CONFIG serve_input` / `serve_output` / `serve_shm` path is
the right seed, but it feeds **one dynamic input and one output**. MATMUL has
two changing inputs. The warm-frame ABI generalizes that path and makes it the
same on CUDA and HIP.

## Goals

- One GPU context, one compiled artifact, and one arena per warm worker.
- Many frames per worker.
- All per-frame inputs and outputs can change.
- Same ABI for CUDA and HIP.
- Runner falls back to the current process-per-job path when an artifact does
  not support warm frames.
- No change to the language contract, node OS, or scheduler protocol.
- No performance claim until correctness and stability are gated.

## Non-goals

- Dynamic shapes, T/triple streams, or fp16 warm frames.
- Multiple GPUs in one worker.
- Cross-host shared memory.
- TLS, artifact signing, or GPU passthrough.
- hipBLAS/MIOpen or kernel tuning. Those are separate performance changes.

## Shape

```text
scheduler
  -> runner process
       -> warm worker (one per artifact key)
            shared input/output buffers
            request/response pipe
            persistent GPU context, kernels, arena
```

The warm worker is the same generated executable in a different mode:

```text
artifact --warm-frame CONFIG
```

Normal mode remains `artifact input1 input2 ... output1 ...`.

## Artifact eligibility

Warm frames require:

- static shapes for every input and output;
- F32/F64 or I64 streams only (GPU storage is F32 for F);
- `use_fp16 == false` in v1;
- no T streams;
- no dynamic shape or dynamic output;
- all ops supported by the target's normal emitter;
- a compile-time `CONFIG warm_frame 1` opt-in.

If any condition fails, the runner keeps the process-per-job path.

`CONFIG warm_frame 1` is intentionally separate from the existing
`serve_input`/`serve_output` controls. The existing serve mode stays for
compatibility; the long-term path is warm frames.

## Buffer contract

Every stream gets one shared buffer for v1. Separate files are simpler than
packing offsets, and they keep the worker parser small.

For each stream the runner creates a file in a per-worker temp directory,
`ftruncate`s it to `bytes`, and `mmap`s it `PROT_READ|PROT_WRITE`,
`MAP_SHARED`. Files are mode 0600 and removed when the worker exits.

One byte count per element:

| kind | bytes/element | note |
|---|---:|---|
| F32 / F64 input or output | 4 | GPU codegen computes in F32 |
| I64 | 8 | integer streams |
| U8 / Bool / T | — | refused in warm-frame v1 |

Buffers are native little-endian, tightly packed in row-major order, matching
the generated executable's argv format.

## Worker config

The worker reads a line-oriented text config, not JSON. C++ JSON parsing would
add a dependency for no benefit.

```text
jotnar-warm 1
target hip
in a f32 256 512 524288 /dev/shm/jotnar-xxxx/a
in b f32 512 256 524288 /dev/shm/jotnar-xxxx/b
out OUT f32 256 256 262144 /dev/shm/jotnar-xxxx/OUT
```

Rules:

- `jotnar-warm 1` is the first line.
- `target` must match the executable.
- every `in`/`out` has name, dtype, shape, bytes, and path;
- shapes are checked against the compiled artifact;
- byte counts are checked against `shape × dtype`;
- duplicate names and unknown directives fail startup.

The runner writes this file itself, so the worker does not need to trust
caller-provided paths.

## Control protocol

The worker uses stdin/stdout only for control. Logs go to stderr.

Startup:

```text
worker: write 'R' (0x52) to stdout when buffers, context, and kernels are ready
runner: read 'R' with a timeout
```

Frame:

```text
runner: write all input bytes into the input mappings
runner: write 0x01 to worker stdin
worker: copy inputs H2D, run the ops, copy outputs D2H into the output mappings
worker: write 0x00 for OK or 0x01 for ERROR to stdout, then flush
runner: read the status byte and own the output mappings
```

Shutdown:

```text
runner: write 0x00 to stdin
worker: release buffers, exit 0
```

There is exactly one in-flight frame per worker. Concurrency comes from
multiple runner processes, and later from a runner-side worker pool.

A nonzero status restarts the worker. The runner records the failure and does
not use the output mappings from that frame.

## Generated worker

Both emitters need the same scaffold:

1. parse the config;
2. `open`/`mmap` every buffer;
3. allocate device buffers and a persistent arena if configured;
4. initialize the target context once;
5. handshake `R`;
6. loop: read token, copy inputs, run ops, copy outputs, write status;
7. handle `0x00` shutdown and unmap/free.

`src/codegen/warm.rs` should own one shared Rust template, parameterized by
the runtime names:

```text
CUDA: cudaMalloc/cudaMemcpy/cudaDeviceSynchronize/...
HIP:  hipMalloc/hipMemcpy/hipDeviceSynchronize/...
```

The existing `cuda.rs` and `hip.rs` remain the op emitters. The shared template
owns only the host scaffold, buffer parsing, and frame loop. This avoids a
second CUDA/HIP divergence.

The normal `main` input/output path stays unchanged.

## Runner worker lifecycle

The runner keeps a pool keyed by:

```text
artifact_key = source_hash + target + HIP_ARCH/CUDA_ARCH + fp16 flag
layout_key   = input names/shapes/kinds + output names/shapes/kinds
```

`source_hash` already exists in the build cache key. The runner creates a
worker only when the artifact advertises warm support and the payload shapes
match `layout_key`.

Lifecycle:

- start on first frame for the key;
- reuse for every matching frame;
- serialize frames per worker with a lock;
- restart on nonzero status or child exit;
- kill on idle timeout or runner shutdown;
- keep the compiled artifact cache independent, so a runner restart does not
  require a rebuild.

The first version uses one warm worker per runner process. Multiple runner
processes per GPU host remain the scheduling unit; a runner-side worker pool is
a later extension.

## Runner response

The runner response grows:

```json
{
  "outputs": {...},
  "timing": {
    "compile_ms": 0,
    "build_ms": 0,
    "worker_start_ms": 437,
    "frame_ms": 1.2,
    "warm": true
  }
}
```

`run_ms` remains the end-to-end time for the frame path. `warm` and
`worker_start_ms` distinguish startup from steady state.

The scheduler ledger records `warm` and `frame_ms` as defaulted fields so old
ledgers remain readable.

## Scheduler impact

No protocol change is required. The scheduler still sends a listing, payload,
outputs, and count. The runner decides whether to use a warm worker.

Later scheduler work:

- report warm/startup/frame timings in the batch summary;
- use the worker pool size as a placement hint;
- keep retry/ledger semantics unchanged.

Do not move warm-frame logic into `jotnar-noded`.

## Failure handling

- Worker startup timeout -> kill, retry once, then fail the frame.
- Nonzero frame status -> record error, kill worker, restart on the next frame.
- Child exit -> same.
- Runner shutdown -> send `0x00`, wait briefly, then kill.
- Stale shared buffers -> removed by the runner's temp-directory cleanup.
- Wrong shape/kind -> fail before mapping or at worker startup.

A warm worker is trusted local code, but the runner must still validate
layout and sizes before writing any buffer.

## Implementation phases

1. **Documentation and manifest**
   - this document;
   - `CONFIG warm_frame 1` parsing and artifact eligibility;
   - shared warm scaffold in `src/codegen/warm.rs`.

2. **CUDA warm worker**
   - generated `--warm-frame` mode;
   - runner-side CUDA worker pool of one;
   - warm vs process-per-job parity for 64³, 256×512×256, 512³.

3. **HIP warm worker**
   - same scaffold with HIP API names;
   - parity and startup measurement on the RX 7900 XTX.

4. **Runner/scheduler integration**
   - `warm`, `worker_start_ms`, `frame_ms` in responses and ledgers;
   - idle timeout and restart gate;
   - multi-worker pool only if measurements justify it.

5. **Performance gate**
   - warm frame throughput vs current process-per-job;
   - correctness, tolerance, and error/restart tests;
   - no claim until the gate is stable.

## Acceptance tests

- Warm and normal outputs match exactly for I64 and within the existing float
  tolerance for F.
- `CONFIG warm_frame 1` refuses T, fp16, dynamic shapes, and unsupported ops.
- A killed worker restarts and the next frame is correct.
- Stale output buffers are never returned after a failed frame.
- CUDA and HIP use the same config syntax and status protocol.
- `cargo test --workspace` remains green.
- No CI, no node-OS changes, no GPU passthrough.

## Open questions

- one shared file with offsets vs one file per stream (v1: one per stream);
- runner-side worker pool size (v1: one per runner process);
- whether to retire the old `serve_*` config after warm frames land;
- static-input residency for workloads where only one operand changes;
- whether MIOpen/hipBLAS tuning belongs to this track or a separate one.
