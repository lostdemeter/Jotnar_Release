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
| SDL2 runtime | playable DOOM window |
| DOOM image + WAD | live DOOM; `run_doom_demo.sh` falls back otherwise |
| Python 3 + NumPy/Pillow | parity harnesses against the Python oracle |
| `jotnar-main` + `phi-core` checkout | differential parity only; demos run without them |

## Build and test

```sh
cargo build --release
cargo test --workspace
```

Targets: `c` (default), `nonfpu`, `cuda`, `rust`. Add `--fp16` to CUDA builds to
read F inputs from half-precision files (FP32 compute, smaller files).
`emit-rust` and `build --target rust` need `rustc` on PATH; they emit a
self-contained std-only Rust program (F as f64, I as i64). Rust is an
F/I subset: T (triple) streams are refused loudly — the emitter may
change the schedule, not the arithmetic.
The CUDA build finds the toolkit relative to `nvcc`; set `NVCC` if it is not
on `PATH`.

## Status and evidence

What works today:

- `IMPORT` / `DEF` / `CALL` textual macro expansion with per-call-site
  namespacing, `REPEAT` / `BREAK`, `CONFIG`, layouts, `STATE`, `RANGE`.
- Registry binding for **all 90 mnemonics**, including `SADD`, `IMAX`,
  `ISAR`, `IDIVU`, `IMODU`, `IEQ`, `ILT`, `ISELECT`.
- The static layout verifier with honest coverage reporting.
- The reference interpreter over `F32`/`F64`, `I64`, `U8`, bool masks,
  lattice triples, and `None` history.
- The lattice substrate (`encode`/`decode`/fixed-point/LUTs) matching the
  Python formulas.
- Four code generators: C99, integer-only C with a float trap, CUDA, and Rust
  (std-only; scalar-right, MATMUL, SOFTMAX over F/I; GELU/LAYERNORM implemented
  but not yet gated);
  T refused, CONV/DECONV/INTERP/GATHER fail loud until the next slice).
- Differential parity:

```text
88/88 op cases (original 86)  scripts/parity_ops.py
4 extension gates       tests/int_geometry.rs
32/32 emitter cases     scripts/parity_emit.py
flagship + xf_block     scripts/parity_programs.py
CUDA units/iext/…       scripts/parity_cuda.py
Futhark 118/118         scripts/parity_futhark.py
protocol test suite      cargo test --test protocol
DOOM frame 1 / frame 2   scripts/parity_doom_native.py
workspace tests          198/198
```

Float `F` paths can differ from numpy in the last ulp on some platforms
(different SIMD reduction order); triples and integer paths are compared
bit-exactly.

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

The Rust SDL player has intermittent frame-time spikes not seen in the
original Python/C player. This is recorded as future work; add a frame-time
trace and p99/max gate before calling the frontend finished.

## CLI

```text
jotnar check  <file.asm> [--stdlib DIR]
jotnar verify <file.asm> [--stdlib DIR] [--json]
jotnar run    <file.asm> --payload FILE [--out FILE] [--outputs a,b] [--stdlib DIR]
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

## Runtime binaries (`jotnar-repl`, `jotnar-node`, `jotnar-ctl`, `jotnar-bench`)

```text
jotnar-repl <file.asm> [--stdlib DIR] [--outputs a,b] [--pretty] [--stats]
jotnar-node [--bind ADDR] [--allow lattice,...] [--max-frame N] [--token SECRET] [--stats]
jotnar-ctl  [--addr ADDR] [--token SECRET] ping | list
jotnar-ctl  [--addr ADDR] [--token SECRET] program <name> <file.asm> [--backend lattice]
jotnar-ctl  [--addr ADDR] [--token SECRET] store <name> --payload <payload.json> --stream <NAME>
jotnar-ctl  [--addr ADDR] [--token SECRET] exec <handle> --payload <payload.json> --outputs a,b
jotnar-ctl  [--addr ADDR] [--token SECRET] drop <name>
jotnar-bench [--jobs N] [--workload matmul2|vec|u8|f64|shelf] [--n N] [--modes inproc,tcp-frames,tcp-json]
```

```sh
cargo run --release --bin jotnar-repl -- showcase/language/saxpy.asm --outputs OUT
# then pipe NDJSON {"streams": {...}} lines on stdin

cargo run --release --bin jotnar-node -- --bind 127.0.0.1:9999
cargo run --release --bin jotnar-node -- --bind 0.0.0.0:9999 --token "$SECRET"  # non-loopback requires --token
cargo run --release --bin jotnar-ctl -- --addr 127.0.0.1:9999 ping
cargo run --release --bin jotnar-ctl -- --addr 127.0.0.1:9999 program tile showcase/language/matmul2.asm
cargo run --release --bin jotnar-ctl -- --addr 127.0.0.1:9999 exec tile --payload payload.json --outputs c00

cargo run --release --bin jotnar-bench -- --jobs 2000
cargo run --release --bin jotnar-bench -- --workload vec --n 16384 --jobs 100
cargo run --release --bin jotnar-bench -- --workload u8 --side 128 --jobs 100
cargo run --release --bin jotnar-bench -- --workload shelf --n 16384 --jobs 100
```

`jotnar-repl` assembles once, then answers one JSON payload per stdin
line (errors are per-line `{"error": ...}`, the process stays alive).
`jotnar-node` is the model-blind worker: framed TCP (`u32BE || JSON ||
payload`), one client at a time, loopback by default, `--max-frame`
(64 MiB default) and `--token` for non-loopback binds. See
`docs/RUNTIME_RESEARCH.md` for the transport measurements and
`showcase/runtime/` for the exhibit.

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

