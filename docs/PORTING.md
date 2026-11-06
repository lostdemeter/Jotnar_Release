# Port map and next milestones

The Python implementation is the oracle. This document keeps the mapping
between the two codebases explicit so the port stays reviewable.

## What maps to what

| Python | Rust | Notes |
| --- | --- | --- |
| `chain/asm.py:expand/parse/assemble/verify` | `src/asm.rs` | textual macro expansion, origins, REPEAT, layout replay |
| `chain/asm.py:_check_layouts/_resolve` | `src/layout.rs` | `$VAR`, `$VAR^T`, `*`, UNKNOWN |
| `chain/asm.py:run/_check_shapes` | `src/runtime.rs` | loops, BREAK, kind cross-check, per-op shape rules |
| `chain/asm_ops.py:REGISTRY/SIGS` | `src/ops/mod.rs` | all 86 mnemonics and signatures, including `SADD/IMAX/ISAR/IDIVU/IMODU/IEQ/ILT/ISELECT` |
| `chain/asm_ops.py:op_*` | `src/ops/*.rs` | reference semantics |
| `chain/emit_c.py` | `src/codegen/{c,infer}.rs` | C99 patterns + static shapes |
| `chain/emit_nonfpu.py` | `src/codegen/nonfpu.rs` | integer-only patterns and tables |
| `chain/emit_cuda.py` | `src/codegen/cuda.rs` | FP32 kernels + driver |
| `phi_core/lattice.py` | `crates/phi-core/src/lattice.rs` | frozen LUT formulas, round-half-even |
| `phi_core/numpy_ops.py` | `crates/phi-core/src/numpy_ops.rs` + `src/ops/*.rs` | matmul_int, norms, softmax, gelu, scan, gather; image ops still in Jotnar ops |
| `chain/holo_phi.py` | `src/ops/element.rs`, `image.rs` | tmul/binop/clip/sigmoid/sqrt, gain |
| `chain/splat.py` | `src/ops/image.rs` | structure tensor, coherence buckets, bank, soft blend |
| `chain/control.py` | `src/ops/image.rs`, `ops::ctrl` | beta fields; CTRL values embedded as frozen doctrine |
| `chain/temporal.py` / `verdict.py` | `src/ops/image.rs` | warp, static mask |

## Intentional early differences

- The Rust crate has one merged value type instead of numpy dtype dispatch.
  `F32`/`F64` remain distinct where numpy distinguishes them.
- Trace summaries, causal `overrides`, and the store-hook fused executor
  are not ported yet.
- `STATE.repeat()` exists in the library; the CLI does not expose it yet.
- Payloads are JSON rather than numpy arrays; the CLI is the boundary.

## Milestones

0. **phi-core substrate**: `crates/phi-core` now carries the lattice,
   `Triple`, exact compare/clip/bin helpers, and the core numpy-style ops.
   Jotnar consumes it through re-exports; image-specific substrate helpers
   are the next migration pass.
1. **Runtime parity**: frontend + interpreter + all 86 op semantics.
   Gated by `scripts/parity_ops.py` and `scripts/parity_programs.py`.
2. **C emitter**: done for the reference lowering set, plus ROPE/SWIGLU/
   ATTN/TOPK extensions. Gated by `scripts/parity_emit.py` and the headt /
   widths calibration classes.
3. **Non-FPU emitter**: done for exact moves, integer/byte ops, fixed-point
   arithmetic/matmul/norms/softmax, GT/SELECT, and the full headt block.
   Gated bit-exactly and by the source-level float trap.
4. **CUDA emitter**: correctness-first FP32 kernels for the reference
   unit/bigram/headt/fp16/width gates. Gated by `scripts/parity_cuda.py`.
   Tuned kernels/cuBLAS scheduling is a performance pass, not a parity gap.
5. **Serve / codec paths**: `live=` resident loops, pack_outputs, codec ABI,
   CUDA graphs. This is the main remaining functional surface.
6. **CLI polish**: trace output, payload from `.npy`/raw binary.

## Adding a mnemonic

1. Add an entry in `src/ops/mod.rs` with arity and the `SIGS` pattern.
2. Add the reference function (or leave `run: None` to stay loud).
3. Add a case to `scripts/parity_ops.py`; make it pass against the oracle.
4. If the op has a shape rule, add it to `runtime::check_shapes`.
