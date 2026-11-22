# Usage and technical details

This is the developer-facing companion to the README: prerequisites, build
and test, the CLI and payload boundary, parity harnesses, and the repository
layout. The README is the five-minute front door; this page is for when you
need the exact command or path.

## Prerequisites

| need | for |
|---|---|
| Rust stable + Cargo | everything |
| `gcc` | C backend, native DOOM core, Verilog test fixture |
| CUDA toolkit + GPU | CUDA target and GPU migration parity |
| ROCm + `hipcc` + AMD GPU | HIP target and `scripts/parity_hip.py` |
| SDL2 runtime | playable DOOM window |
| DOOM image + WAD | live DOOM; `run_doom_demo.sh` falls back otherwise |
| Python 3 + NumPy/Pillow | parity harnesses against the Python oracle |
| `jotnar-main` + `phi-core` checkout | differential parity only; demos run without them |

## Build and test

```sh
cargo build --release
cargo test --workspace
```

Targets: `c` (default), `nonfpu`, `cuda`, `hip`, `rust`, `vulkan`. Add `--fp16`
to CUDA or HIP builds to read F inputs from half-precision files (FP32 compute,
smaller files). CUDA `CONV2D` uses cuDNN NHWC when available (set
`JOTNAR_CUDNN=0` to force the im2col + cuBLAS fallback).
`emit-rust` and `build --target rust` need `rustc` on PATH; they emit a
self-contained std-only Rust program (F as f64, I as i64). Rust is an
F/I subset: T (triple) streams are refused loudly — the emitter may
change the schedule, not the arithmetic. Emitted `CONV2D` uses all available
cores by default; set `JOTNAR_THREADS=N` to cap the worker count. Output bytes
are independent of the thread count. `jotnar build` writes a sibling
`<name>.manifest.json` declaring the target, dtype, stream shapes, and optional
`serve_input`/`serve_output`; emitted Rust artifacts with those config keys also
support a warm `--serve` framing mode.
The CUDA build finds the toolkit relative to `nvcc`; set `NVCC` if it is not
on `PATH`. Set `CUDA_ARCH` (for example `sm_120`; default `sm_86`) to match the
GPU; the default preserves the original SM86 code generation.
The HIP build uses `hipcc` on `PATH` (falling back to `/opt/rocm/bin/hipcc`);
set `HIPCC` to override and `HIP_ARCH` (for example `gfx1100`) to pin the
offload target. The HIP target is a correctness-first port of the 51-mnemonic
CUDA slice as of `d98dfe2`; current CUDA has 65 mnemonics. See
`docs/ROCM_FEASIBILITY.md` for the spike report and the 14-op gap list.
The Vulkan build needs `glslc` on `PATH` and emits C with embedded SPIR-V that
links `libvulkan`. It is a correctness-first φ-ladder subset (T; T2 for
`MUL`/`SQUARE`/`ADD`/`SUB`/`MATMUL`; I; `T2I`; and F matmul); every other op
refuses, and shapes must be static. See `docs/VULKAN_FEASIBILITY.md` for the
supported-op table, numeric model, and known limits.

## Status and evidence

What works today:

- `IMPORT` / `DEF` / `CALL` textual macro expansion with per-call-site
  namespacing, `REPEAT` / `BREAK`, `CONFIG`, layouts, `STATE`, `RANGE`.
- Registry binding for **all 106 mnemonics**, including `SADD`, `IMAX`,
  `ISAR`, `IDIVU`, `IMODU`, `IEQ`, `ILT`, `ISELECT`.
- The static layout verifier with honest coverage reporting.
- The reference interpreter over `F32`/`F64`, `I64`, `U8`, bool masks,
  lattice triples, and `None` history.
- The lattice substrate (`encode`/`decode`/fixed-point/LUTs) matching the
  Python formulas.
- Six code generators: C99, integer-only C with a float trap, CUDA, HIP,
  Rust (std-only; scalar-right ops + MATMUL/GELU/SOFTMAX/LAYERNORM/RELU/CONV/CONV2D/DECONV2D/RESIZE2D/TRANSPOSE/CONCAT
  over F/I; LAYERNORM eps comes from CONFIG eps_ln);
  T refused, DECONV/INTERP/GATHER fail loud until the next slice), and Vulkan
  (φ-ladder subset; see `docs/VULKAN_FEASIBILITY.md`). HIP ports the
  51-mnemonic CUDA slice as of `d98dfe2`; current CUDA has 65, and none of
  HIP, CUDA, or Vulkan is the full 106-mnemonic surface.
- Differential parity:

```text
88/88 op cases (original 86)  scripts/parity_ops.py
4 extension gates       tests/int_geometry.rs
32/32 emitter cases     scripts/parity_emit.py
flagship + xf_block     scripts/parity_programs.py
CUDA units/iext/…       scripts/parity_cuda.py
HIP 32 cases (C vs HIP) scripts/parity_hip.py     # hipcc/AMD GPU, correctness only
Vulkan 19 cases (T/T2 exact) scripts/parity_vulkan.py  # glslc + Vulkan device, correctness only
Futhark 118/118         scripts/parity_futhark.py
protocol test suite      cargo test --test protocol
DOOM frame 1 / frame 2   scripts/parity_doom_native.py
workspace tests          487/487
```

The GPU parity gates fail if `nvcc`/`hipcc` is unavailable. Set `NVCC`/`HIPCC`
or install the toolchain; a missing toolchain is not a pass.

Float `F` paths can differ from numpy in the last ulp on some platforms
(different SIMD reduction order); triples and integer paths are compared
bit-exactly.

## Stream arena (opt-in)

The interpreter is the reference and the default. `JOTNAR_ARENA=1` makes the
runtime try a single-buffer execution plan first: one `Vec<i64>` holds every
I stream (`i64`) and F stream (`f64` bits), and 53 kernels write destinations in
place. A listing is planned only when every stream is I or F, every mnemonic has
a kernel, the working set fits `JOTNAR_ARENA_MAX_MB` (default 64), and the run
is single-threaded and unprofiled. Anything else falls back to the interpreter
before the first instruction; a runtime error is returned, never silently
re-run.

```sh
cargo test --workspace                     # arena off
JOTNAR_ARENA=1 cargo test --workspace      # same tests, arena path
JOTNAR_ARENA=1 ./scripts/arena_soak.sh     # interpreter vs arena, whole listings
./scripts/alloc_gate.sh                    # allocation tripwire
```

T and T2 streams, `F32`/`U8`/`Bool` payloads, and mnemonics without arena
kernels are not arena-backed; `docs/ARENA.md` records why, what is
covered, and the measured allocation numbers.

### CUDA liveness arena

The runtime stream arena above is interpreter-only. The compiled CUDA target has
its own opt-in liveness pool: `CONFIG cuda_arena 1` in the listing, or
`JOTNAR_ARENA=1` at **build** time (an already-built artifact does not read the
env). One `cudaMalloc` holds the intermediates and graph outputs; inputs stay
separate. On the DAV2-S 518 graph this turns 823 per-stream allocations
(2.2 GB sum) into one 93.3 MB pool, with bit-identical output. `CONFIG
cuda_arena 0` / no flag keeps the per-stream behavior.

## Linear attention (opt-in op)

`LINEAR_ATTN(Q,K,V,W1,B1,W2,B2,LNQ_W,LNQ_B,LNK_W,LNK_B,TEMP,NH,DH)` is a
per-head normalized-key linear attention with a shared positive feature MLP:

```text
q = LayerNorm(Q_head) * exp(TEMP[head])
k = LayerNorm(K_head)
fq = softplus(GELU(q W1 + B1) W2 + B2) + 1e-3
fk = softplus(GELU(k W1 + B1) W2 + B2) + 1e-3
out_head = fq (fk^T V_head) / (fq sum fk)
```

`Q/K/V` are `(N, NH*DH)`; `W1` is `(DH,HID)`, `W2` is `(HID,M)`, the LayerNorm
vectors are `(DH)`, and `TEMP` is `(NH)`. There is no `N x N` score matrix:
compute is `O(N*M*DH)` and feature memory is `O(N*M)`. The CUDA target and the
interpreter have reference implementations; the Rust/C/T2 targets refuse.
A distilled linear student is not in the tree yet; this op is the runtime
contract for it.

## T2 phi-float (opt-in)

`CONFIG t2 24` switches a listing to the four-plane T2 value format
(`s,e,m,z`) on the non-FPU target. It keeps a binary mantissa alongside
the φ exponent, so arithmetic results are not re-snapped to the K=512
lattice. It is opt-in: listings without `CONFIG t2` use the frozen T
path byte-for-byte. T2 lowers `MUL`, `SQUARE`, `ADD`, `SUB`, `MATMUL`,
`LAYERNORM`, `GELU`, `SOFTMAX`, `SOFTMAX_WIDE`, `TRANSPOSE`, and
`CONCAT`; other ops refuse loudly. `CONFIG bridge_f <F>` selects the
fixed-point bridge width (default 18); T2 uses 24. See
[`docs/T2_PHI_FLOAT.md`](T2_PHI_FLOAT.md).

## Not ported yet

- `jotnar-main`'s **device** emitter and the CUDA fleet kernel used by the
  original DOOM GPU demo. The RV32IM machine, checkpoint/MMIO path, native
  DOOM frame, C/non-FPU/CUDA targets, and Verilog backend are in place.
- Serve mode (`live=` resident loops), packed outputs,
  the codec ABI, CUDA graphs, tracing summaries, causal `overrides`, and
  the store-hook fused executor. The in-process frame codec, registry, blob
  shelf, deterministic EXEC path, and TCP transport (`jotnar-node`) are
  in place.
- The model-side phi-core modules (`calibrate`, `score`, `compose`, `stamp`,
  `zoo`); the substrate crate covers the lattice/triple/exact helpers and
  numpy-style ops Jotnar consumes.
- Speed work: the CUDA backend is correctness-first (one launch/op); tuned
  kernels/cuBLAS scheduling is a separate pass.

The async node application now lives in `crates/jotnar-node`: bounded worker
pool, cancellation/timeouts, `Arc` snapshots, chunked/store-backed results,
node identity, discovery, and a controller ledger. These are application
claims, not language-exhibit claims. The simple `jotnar-node` front door
remains one client at a time with a bounded in-process queue; exhibit 10 claims
TCP parity only for single-job, small-output T2.

The HIP/CUDA GPU runner and scheduler live in `crates/jotnar-gpu-runner`:
`jotnar-gpu-runner` compiles and runs listings on a local GPU host and
`jotnar-gpu-net` discovers runners, places batches, retries failures, and keeps
a resumable ledger. `--batch-frames N` sends up to N warm frames per request,
and `--prewarm` starts the workers before the measured batch.
Warm-frame workers are opt-in with `CONFIG warm_frame 1`; the reproducible
gate is [`scripts/gpu_warm_gate.py`](../scripts/gpu_warm_gate.py). See
[`crates/jotnar-gpu-runner/README.md`](../crates/jotnar-gpu-runner/README.md).

The native/process lane and workload controller live in `crates/jotnar-node`:
`jotnar-native-runner` executes one allow-listed command per job, and
`jotnar-workload` validates a content-addressed bundle, orders a stage graph,
places stages, moves checkpoint files, and writes a ledger/receipt. The first
acceptance test runs DOOM frame 1 on one native executor, transfers the RV32IM
checkpoint to a second executor, resumes frame 2, and compares the full state
with a single-node reference:

```sh
python3 scripts/doom_two_node_gate.py
```

See [`docs/WORKLOAD_CONTROLLER.md`](WORKLOAD_CONTROLLER.md).

The same controller also runs GPU stages. A batch chain sends an artifact
payload to a CUDA stage, decodes its binary result into the next stage's JSON
payload, runs that on a HIP stage, and compares every output with a single-node
CUDA reference:

```sh
python3 scripts/gpu_batch_gate.py --items 8 --size 64x64x3
python3 scripts/doom_gpu_contention_gate.py --items 16 --size 128x128x3 --clients 4
```

The first gate covers correctness; the second measures DOOM latency while one
or more batch clients share the same CUDA and HIP runners. See
[`docs/WORKLOAD_CONTROLLER.md`](WORKLOAD_CONTROLLER.md) and
[`demos/gpu-batch/`](../demos/gpu-batch/).

The cooperative streaming demo runs the RV32IM core on one node and splits
each frame between the CUDA and HIP nodes, each running the Jotnar
`INTERP` upscale. It calibrates both cards and chooses the split from measured
throughput, then records a GIF and per-frame latency:

```sh
python3 scripts/doom_two_gpu_gate.py
```

See [`docs/DOOM_TWO_NODE.md`](DOOM_TWO_NODE.md) and
[`showcase/doom/doom_two_nodes.gif`](../showcase/doom/doom_two_nodes.gif).
The same coordinator can open an SDL viewer with a live node/latency HUD and
forward keys. The one-command launcher is:

```sh
./demos/doom-two-nodes/run.sh
```

or run the pieces directly:

```sh
jotnar-doom-coord ... --viewer-listen 127.0.0.1:10000
jotnar-doom-view --addr 127.0.0.1:10000
```

The network-outage gate closes one local proxy mid-run, optionally restarts
it, and checks that the surviving node finishes the frames and the split
resumes after recovery:

```sh
python3 scripts/doom_failover_gate.py
```

The resource/placement model is in
[`docs/WORKLOAD_COORDINATION.md`](WORKLOAD_COORDINATION.md).

The Rust SDL player has intermittent frame-time spikes not seen in the
original Python/C player. This is recorded as future work; add a frame-time
trace and p99/max gate before calling the frontend finished.

## CLI

```text
jotnar check  <file.asm> [--stdlib DIR]
jotnar verify <file.asm> [--stdlib DIR] [--json]
jotnar run    <file.asm> --payload FILE [--out FILE] [--outputs a,b] [--threads N] [--stdlib DIR]
jotnar ops
```

```sh
./target/release/jotnar check  programs/xf_block.asm --stdlib stdlib
./target/release/jotnar verify programs/xf_block.asm --stdlib stdlib
./target/release/jotnar run programs/holo_flagship.asm --payload payload.json --out out.json

./target/release/jotnar emit-c programs/xf_block.asm \
  --payload payload.json --outputs OUT --out proj.c
./target/release/jotnar build programs/lm_headt.asm \
  --payload payload.json --outputs LOGITS,OUT --workdir /tmp/headt
./target/release/jotnar build programs/bigram_lm.asm \
  --payload payload.json --outputs OUT --target nonfpu --workdir /tmp/nf
./target/release/jotnar ops
```

`check` parses, expands, and binds. `verify` replays layout unification over
declared layouts and prints the DEF interface table. `run` executes and
writes every stream (or named outputs). `--stdlib DIR` overrides `IMPORT`
search; otherwise the frontend searches the listing directory and ancestors.

## Runtime binaries (`jotnar-repl`, `jotnar-det`, `jotnar-node`, `jotnar-ctl`, `jotnar-bench`)

```text
jotnar-repl <file.asm> [--stdlib DIR] [--outputs a,b] [--pretty] [--stats]
            [--threads N] [--profile]
jotnar-det  <file.asm> <payload.ndjson> [--outputs a,b]
            [--threads 1,4,16] [--repeat N]
            [--concurrency 1,2,8] [--job-threads N] [--stagger-ms N]
jotnar-node [--bind ADDR] [--allow lattice,...] [--max-frame N] [--token SECRET] [--stats]
jotnar-net  [--discover-timeout-ms N] capabilities
jotnar-net  [--discover-timeout-ms N] hardware [--probe]
jotnar-net  [--token T] gpu <listing.asm> [--addr ADDR] [--target hip|cuda] --payload FILE --outputs a,b
jotnar-native-runner --allow PROGRAM [--allow PROGRAM]... [--bind ADDR] [--token T] [--mdns]
jotnar-workload bundle-doom --image FILE --wad FILE --out DIR [--frames N]
jotnar-workload bundle-gpu --listing FILE --out DIR (--input PAYLOAD.json | --generate HxWxC)
                            [--items N] [--name NAME] [--timeout-ms N]
jotnar-workload bundle-stage-chain --listing FILE --descriptor FILE [--descriptor FILE]...
                            --out DIR (--input PAYLOAD.json | --generate HxWxC)
                            [--name NAME] [--timeout-ms N]
jotnar-workload run --bundle PATH --ledger DIR [--program FILE]
                    [--executor ADDR]... [--gpu-executor cuda=ADDR]... [--gpu-executor hip=ADDR]...
                    [--place STAGE=ADDR]... [--compare STAGE:OUTPUT=FILE]...
                    [--fleet-dir DIR] [--fleet-slots N] [--fleet-max-wait-ms N]
jotnar-stream --descriptor FILE --cuda ADDR --hip ADDR [--token T]
              [--frames N] [--clip DIR] [--out DIR] [--receipt DIR]
              [--reference DIR] [--fleet-dir DIR] [--all-cuda]
              [--request-timeout-ms N] [--quarantine-frames N] [--warm-fallbacks]
jotnar-doom-stream --image FILE --wad FILE --cuda ADDR --hip ADDR
                   [--token T] [--frames N] [--start-frame N] [--left-cols N]
                   [--no-calibrate] [--probe-iters N] [--min-cols N] [--max-cols N]
                   [--sink tcp|frames] [--out DIR] [--hud] [--viewer-listen ADDR] [--viewer]
                   [--request-timeout-ms N] [--quarantine-frames N] [--no-warm-fallbacks]
jotnar-doom-coord --image FILE --wad FILE --cuda ADDR --hip ADDR [--token T] --out DIR
                  [--fleet-dir DIR] [--lease-ms N]
jotnar build FILE --payload FILE --outputs a,b [--stage-out FILE] [--descriptor-only]
jotnar-doom-coord ... --viewer-listen ADDR
jotnar-doom-view --addr ADDR
jotnar-ctl  [--addr ADDR] [--token SECRET] ping | list
jotnar-ctl  [--addr ADDR] [--token SECRET] program <name> <file.asm> [--backend lattice]
jotnar-ctl  [--addr ADDR] [--token SECRET] store <name> --payload <payload.json> --stream <NAME>
jotnar-ctl  [--addr ADDR] [--token SECRET] exec <handle> --payload <payload.json> --outputs a,b
            [--threads N] [--policy reference|auto] [--receipt]
jotnar-ctl  [--addr ADDR] [--token SECRET] drop <name>
jotnar-bench [--jobs N] [--workload matmul2|vec|u8|f64|shelf] [--n N] [--modes inproc,tcp-frames,tcp-json]
```

```sh
cargo run --release --bin jotnar-repl -- showcase/language/saxpy.asm --outputs OUT
# then pipe NDJSON {"streams": {...}} lines on stdin

cargo run --release --bin jotnar-node -- --bind 127.0.0.1:9999
cargo run --release --bin jotnar-node -- --bind 0.0.0.0:9999 --token "$SECRET"  # non-loopback requires --token
cargo run --release --bin jotnar-net -- --discover-timeout-ms 5000 capabilities
cargo run --release --bin jotnar-net -- --token test gpu listing.asm --target cuda --payload payload.json --outputs OUT
cargo run --release --bin jotnar-ctl -- --addr 127.0.0.1:9999 ping
cargo run --release --bin jotnar-ctl -- --addr 127.0.0.1:9999 program tile showcase/language/matmul2.asm
cargo run --release --bin jotnar-ctl -- --addr 127.0.0.1:9999 exec tile --payload payload.json --outputs c00

cargo run --release --bin jotnar-bench -- --jobs 2000
cargo run --release --bin jotnar-bench -- --workload vec --n 16384 --jobs 100
cargo run --release --bin jotnar-bench -- --workload u8 --side 128 --jobs 100
cargo run --release --bin jotnar-bench -- --workload shelf --n 16384 --jobs 100
```

`jotnar-det` is the exact-output determinism harness. It assembles once and
replays NDJSON payloads across thread budgets, repeats, and staggered
concurrent jobs, hashing every requested stream and exiting 1 on any
mismatch. `--threads 1,4,16` is the usual matrix; `--repeat 2` checks
repeated runs; `--concurrency 2,8 --job-threads 4 --stagger-ms 20` checks
independent jobs with varying completion order.

Execution budgets are per job, not per listing: `ExecOptions::default()` is
**one thread** (the permanent oracle), and auto-threading is an explicit
boundary policy (`ThreadPolicy::{Reference, DeterministicAuto}`) used by
`jotnar run --threads`, `jotnar-repl --threads`, `jotnar-node` exec headers,
and `jotnar-ctl exec --policy`. `CONFIG threads` remains a compiled-target
directive only; the runtime never reuses it as the job budget. The
output-partitioned T2 kernels keep each output's accumulation order fixed, so
1/4/16 threads are hash-identical.

An `exec-ok` response carries a canonical `receipt` with program/blob/input/
output hashes, determinism profile, backend/kernel, runtime version (package
plus short git), scheduler version, resolved threads, partition descriptor,
and verdict. `profile_ops: true` adds advisory per-op timing outside the
receipt. T2 streams use the compact `T/t2` wire dtype (`s:i8, e:i32, z:u8,
m:i32` = 10 bytes/element); frozen T keeps the interleaved 24-byte
`i64x3` form. A 64 MiB frame carries about 6.7M T2 elements. The node
application's async result path adds chunked store-backed outputs with
SHA-256; see `showcase/node-os/`. The simple front door does not claim TCP
parity for large outputs.

`jotnar-repl` assembles once, then answers one JSON payload per stdin
line (errors are per-line `{"error": ...}`, the process stays alive).
`jotnar-node` is the model-blind worker: framed TCP (`u32BE || JSON ||
payload`), one client at a time, loopback by default, `--max-frame`
(64 MiB default) and `--token` for non-loopback binds.
`--bind 127.0.0.1:0` binds an OS-assigned ephemeral port and prints the
resolved address on stderr. `exec` headers accept `threads: N` (literal job
budget) and `thread_policy: "reference" | "auto"`; the default stays the
one-thread reference order. `ping` (pong) and `exec-ok` carry
`runtime_version=<crate>+<short-git>`. The in-process `NodeHandle` uses a
bounded queue (default 64), but one worker: program/store mutations can
still queue behind a long exec. Use `jotnar-noded` from
`crates/jotnar-node` for the async job engine and immutable snapshots. See
`docs/RUNTIME_RESEARCH.md` for transport measurements,
`docs/RUNTIME_THREADING.md` for the determinism/threading design,
`showcase/runtime/` for the exhibit and claim boundary, and
`showcase/node-os/` for the node application.

## Payload format

JSON is the portable boundary used by the CLI and parity harness:

```json
{
  "streams": {
    "x":  {"kind":"F64","shape":[2,3],"data":[0.1,0.2,0.3,0.4,0.5,0.6]},
    "pos":{"kind":"I64","shape":[2],"data":["0","1"]},
    "rgb":{"kind":"U8","shape":[200,320,3],"data":[0,1,2]},
    "t":  {"kind":"T","shape":[2,2],
           "s":[1,-1,1,1],"e":[30000,31000,32000,33000],"z":[0,0,0,0]}
  }
}
```

Integer entries may be JSON numbers or strings (strings preserve int64
values above 2^53). Output uses the same shape.

`jotnar repeat` threads `STATE` streams across host iterations. The payload
is `{"iterations":[{"streams":{...}},...]}`; non-STATE inputs come from each
iteration, while STATE streams seed from iteration 0 and carry forward.
`--grow a,b` allows append-only growth along axis 0.

## Parity harness

The harnesses need the Python oracle checkout and `phi-core`:

```sh
export PHI_CORE_DIR=/path/to/phi-core
export JOTNAR_PY=/path/to/jotnar

python3 scripts/parity_ops.py --bin target/release/jotnar
python3 scripts/parity_programs.py --bin target/release/jotnar
python3 scripts/parity_emit.py --bin target/release/jotnar
python3 scripts/parity_cuda.py --bin target/release/jotnar   # nvcc/GPU
python3 scripts/parity_hip.py --bin target/release/jotnar    # hipcc/AMD GPU
python3 scripts/parity_futhark.py
python3 scripts/parity_doom_native.py --gpu-work /tmp/opencode/doom_overlay
```

## Layout

```text
src/
  asm.rs        IMPORT/DEF/CALL expansion, REPEAT, parse, assemble, verify
  ir.rs         Config, IN declarations, instructions, bound program
  layout.rs     KIND:GEOM layouts and $VAR / $VAR^T / * unification
  value.rs      Value (F32/F64/I64/U8/Bool/Triple/None) and Arg literals
  lattice.rs    phi lattice: encode/decode, fixed point, frozen LUTs
  ops/          mnemonic registry + reference semantics
    int.rs      exact integer arithmetic and F2I/I2F
    element.rs  triples arithmetic, sigmoid, clip, mux
    linalg.rs   matmul, norms, softmax, RoPE, attention, moves
    bytes.rs    IWINDOW/IGATHER1/CRC/inflate/varint
    image.rs    sRGB, luma, gain, splat blur, conv/deconv, warp
  codegen/      backend drivers
    infer.rs    static shapes/dtypes for codegen
    c.rs        C99 patterns
    nonfpu.rs   integer-only patterns + baked tables + float trap
    cuda.rs     CUDA kernels and driver
    hip.rs      HIP kernels and driver (51-op CUDA slice from d98dfe2)
  rv32.rs       RV32IM machine + DOOM MMIO/checkpointing
  protocol.rs   in-process node protocol + frame codec
  showcase/     demos: language, image, Futhark, protocol, analog, DOOM
crates/phi-core/
  src/lattice.rs   the phi lattice substrate
  src/triple.rs    typed (s,e,z) stream value
  src/ops.rs       exact compare/clip/bin helpers
  src/numpy_ops.rs matmul/norms/softmax/gelu/scan substrate ops
manual/          field manual source, PDF/HTML, press, assets
crates/jotnar-lab/ trust-spine certificate/gate/verify harness
showcase/        curated visuals and per-exhibit READMEs
certificates/    checked-in certificate examples
tests/           frontend/runtime/RV32/protocol/certificate tests
scripts/         Python differential parity harnesses
docs/LOOP.md     normative trust-spine appendix
docs/PORTING.md  port map and milestones
docs/PORT_BACKLOG.md curated port/demo plan
```

