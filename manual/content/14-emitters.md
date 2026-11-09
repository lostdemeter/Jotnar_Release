---
title: "Emitters: C, non-FPU, CUDA, Rust, Verilog"
number: "13"
kicker: FIELD NOTE / THE MACHINE
lede: The same bound graph can become portable C, integer-only C, a GPU kernel, a piece of hardware, or a 32-bit machine. The lowering is where the promises are tested.
columns: 2
---

An emitter consumes the bound program and produces a target artifact. It does not re-parse the source, and it does not get to reinterpret the graph. The registry’s semantics are the contract; the emitter’s job is to preserve them.

## The targets

| Target | Flag | What it emits | What it refuses |
|---|---|---|---|
| **C** | `--target c` (default) | Portable C99, float/transformer path plus exact moves. | Dynamic shapes it cannot bound. |
| **Rust** | `--target rust` / `emit-rust` | Self-contained std-only Rust (`Vec<f64>`/`Vec<i64>`); scalar-right ops + MATMUL/GELU/SOFTMAX/CONV/TRANSPOSE/CONCAT. LAYERNORM is implemented but not yet gated. F/I only (raw f64/i64 LE payload). | T streams (refused, not decoded); DECONV/INTERP/GATHER (next slice). |
| **non-FPU** | `--target nonfpu` | Integer-only C, baked lattice tables, fixed-point math, no `float`/`double`/libm. | Any float path; a `float_trap` gate catches leaks. |
| **CUDA** | `--target cuda` | FP32 correctness-first kernels, one launch per op. | Missing toolkit or unsupported op. |
| **Verilog** | `jotnar-verilog` | Structural hardware for the supported integer subset. | Multi-lane arithmetic that would cross lanes; reassignment. |
| **RV32IM** | `jotnar-demo-doom` | A 32-bit machine model running a flat image. | Anything outside the RV32IM model. |

The first four are selected by `jotnar build --target`. Verilog has its own binary because it needs a configuration file (widths, state, memories). The RV32IM path is a separate runner because it needs a compiled image and a WAD.

## C

The C emitter is the portable baseline. It infers static shapes, emits array declarations, reads inputs from binary files in argv order, and writes outputs the same way.

```sh
jotnar emit-c showcase/language/saxpy.asm \
  --payload /tmp/saxpy.json --outputs OUT --out /tmp/saxpy.c
```

The emitted loop is exactly what you would write by hand:

```c
/* ax = IMUL(A, X) 64-bit wrap */
for (int64_t e_i = 0; e_i < 4; ++e_i)
  ax[e_i] = (int64_t)((uint64_t)A[e_i] * (uint64_t)(X[e_i]));

/* OUT = IADD(ax, Y) 64-bit wrap */
for (int64_t e_i = 0; e_i < 4; ++e_i)
  OUT[e_i] = (int64_t)((uint64_t)ax[e_i] + (uint64_t)(Y[e_i]));
```

The casts are the wrapping contract. The `uint64_t` multiply is defined to wrap; the cast back to `int64_t` preserves the bit pattern. This is why the C target can claim exact integer parity across platforms.

::: note [STATIC SHAPES]
The C emitter needs static shapes. If a dimension is dynamic, it returns a loud “dynamic dim” error rather than emitting a buffer it cannot bound. That is a feature: a silent fixed-size guess is how portable code becomes non-portable.
:::

## non-FPU

The non-FPU target is the strict one. It emits C with no `float`, no `double`, and no libm. The lattice tables are baked into the source. Fixed-point arithmetic, matmul, norms, softmax, and the headt block are all integer-only.

```sh
jotnar build showcase/language/saxpy.asm \
  --payload /tmp/saxpy.json --outputs OUT --target nonfpu --workdir /tmp/nf
```

A source-level `float_trap` gate fails the build if a float slips into the generated code. That is the only way to keep the claim honest: the compiler cannot silently introduce a float if the source has none.

The non-FPU gates are exact for bigram decisions, T moves, fixed-point arithmetic, matmul, GT/SELECT, and the full headt block. That is a stronger claim than “close enough.”

The four integer extensions have an explicit lowering boundary: `T2I`, `IGET`,
`IMATMUL`, and `IREDUCE_SUM` are implemented in C and non-FPU. CUDA and
Verilog return a loud `NoPattern` for them. That is a declared limit, not a
silent fallback.

## CUDA

The CUDA target is correctness-first. It emits one launch per operation, with FP32 kernels for the unit, bigram, and headt paths. It is not tuned; there are no CUDA graphs, no fused executor, and no cuBLAS scheduling.

```sh
jotnar build programs/lm_headt.asm \
  --payload payload.json --outputs LOGITS,OUT --target cuda --workdir /tmp/cu
```

The CUDA gates compare units within 1e-4 of numpy, bigram argmax bit-exactly, and the headt/fp16/D32/D64 paths against the reference agreement classes. `--fp16` reads F inputs from half-precision files while computing in FP32.

::: warning [NOT A SPEED CLAIM]
One launch per op is a correctness strategy, not a performance strategy. A tuned kernel pass is a separate project. If you benchmark this target and call it slow, you are right, and you have missed the point.
:::

## Rust

The Rust target emits a single-file, std-only Rust program (`rustc`, edition
2021). Integer ops keep checked-division guards (exit 30 on zero /
`i64::MIN / -1`), scalar operands are supported on the right-hand side
(literal RHS, matching the upstream infer contract — scalar-left is refused
with a typed error), and the artefact interface mirrors the C target (raw
little-endian f64/i64 argv files). **T (triple) streams are refused loudly
with the file contract intact** — a triple may never silently become an f64
matmul. The parity gate is `scripts/parity_rust.py`. The current gate covers scalar-right
integer/float ops, `MATMUL`, `GELU`, `SOFTMAX`, `CONV`, `TRANSPOSE`, and
`CONCAT`. `LAYERNORM` is implemented but not yet gated. `CONV` is
same-size, single-channel, odd-kernel, edge-replicate; `TRANSPOSE` and
`CONCAT` are 2D (`CONCAT` along axis 1). `GELU` uses the A&S
approximation, within `1e-6` of exact erf.

```sh
jotnar emit-rust showcase/language/saxpy.asm \
  --payload /tmp/saxpy.json --outputs OUT --out /tmp/saxpy.rs
jotnar build showcase/language/saxpy.asm \
  --payload /tmp/saxpy.json --outputs OUT --target rust --workdir /tmp/rs
```

## Verilog

The Verilog backend emits structural hardware for the integer subset. It is not a CPU; it is the graph as combinational and sequential logic. The CPU8 fixture is the canonical example: a single-cycle 8-bit CPU expressed in Jotnar, emitted as Verilog, and checked against an independent ISA model.

```sh
jotnar-verilog tests/fixtures/cpu8.asm \
  --config showcase/verilog/cpu8.json --module cpu8
```

The config supplies widths, state registers, and memories. The backend refuses multi-lane arithmetic that would cross lanes and reassignment of a wire. Those are hardware constraints, not language constraints, and the emitter says so.

## RV32IM

The DOOM runner is the largest integration test: a compiled RV32IM image executing on a Rust machine model with MMIO for WAD, framebuffer, clock, frame counter, key ring, and UART/CLINT. It is not an x86 emulator and not DOS.

```sh
./run_doom_demo.sh --frames 100
```

The frame-1 state is exact against the C native core; the frame-2 checkpoint is exact against the GPU core. That is a witness at the scale of a real program, not a toy loop.

::: pullquote [EMITTER RULE]
An emitter may change the schedule. It may not change the arithmetic. If it cannot preserve the contract, it must refuse the op.
:::
