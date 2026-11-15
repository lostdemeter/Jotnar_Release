# Stream arena — handoff

Status: **opt-in, working, uncommitted branch `feat/stream-arena`.**
Base: `d98dfe2`. 301/301 tests with `JOTNAR_ARENA` off and on.

## What this is

Three runtime-internal layers, no change to the listing language:

1. **Flat payloads / admission** (`src/flat.rs`, `src/arena.rs`, `src/wire.rs`)
   One codec, one size table, two named byte conventions (`Wire`, `Target`),
   an `Admission` witness, canonical JSON framing, and a per-job byte arena.
2. **Allocation discipline** (`src/profile.rs`, `tests/discipline.rs`)
   Per-thread allocation counters, `JobScope`, enforced ceilings
   (`alloc_max` on EXEC, `--alloc-gate` in `jotnar-det`), and a transport that
   passes the caller's reply buffer through the worker.
3. **Stream arena** (`src/stream_arena.rs`, `src/runtime.rs`, `src/protocol.rs`)
   `Plan` compiled once from `infer_shapes`, one `Vec<i64>` of 64-bit words that
   holds I streams (`i64`) and F streams (`f64` bits), offsets instead of
   pointers, one-store reset, 39 kernels.

T streams are **not** arena-backed; see "Why T2 is excluded".

## Run it

```sh
cargo test --workspace                     # arena off
JOTNAR_ARENA=1 cargo test --workspace      # arena on
./scripts/alloc_gate.sh                    # allocation tripwire
./scripts/arena_soak.sh                    # arena-off vs arena-on parity
```

Knobs:

| Knob | Meaning |
|---|---|
| `JOTNAR_ARENA=1` | enable the arena path (default off) |
| `JOTNAR_ARENA_MAX_MB` | working-set cap (default 64 MiB) |
| `JOTNAR_ARENA=0` | force the interpreter once the default flips |
| `stream_arena::set_enabled` / `set_max_bytes` | per-thread overrides for tests |
| `runtime::arena_stats()` | (hits, fallbacks) on this thread |

A listing is planned only when every stream is I or F, every mnemonic has a
kernel, and `plan.bytes() <= max_bytes()`. Otherwise it falls back to the
interpreter. A **runtime** failure is returned, never silently re-run.

## Coverage

39 kernels. Registry is 92 entries.

| Group | Total | Ported |
|---|---:|---:|
| I-first | 34 | 32 |
| F-first | 6 | 1 |
| Generic `$A` | 27 | 3 |
| Unconstrained `*` | 14 | 3 |
| T | 6 | 0 (excluded) |
| `$X` | 4 | 0 |
| U8 | 1 | 0 (excluded) |

Ported: 4 elementwise-shift/3-op groups (19), reductions (2), encoder (1),
matmul (1), index/window/find/gather/scatter/pack/CRC (8), F: RELU, F2I, I2F,
CONCAT, SLICE, RESIZE2D, CONV2D, DECONV2D.

### Remaining, plannable now (14)

Ordered by risk:

1. `RESHAPE2`, `RESHAPE3` — pure bit-copy; output dims from literals.
2. `LAYERNORM`, `RMSNORM`, `ROTARY` — F row-wise; `Config` carries eps.
3. `MATMUL`, `BATCH_MATMUL` — F analogue of `IMATMUL`; accumulation order matters.
4. `CONV`, `INTERP` — F, coordinate-transformed (same pattern as `RESIZE2D`).
5. `GATHER` (I), `GATHER3`, `SCATTER` — index ops; `GATHER3` is F/T.
6. `ARGMAX` — F/I; returns I indices (mixed kinds).
7. `TOPK` — F; **the riskiest**: tie-breaking and sort order must match exactly.

### Blocked on missing `infer_shapes` rules (not arena work)

`LUMA`, `PERMUTE3`, `DECONV`, `POOLAVG`, `GAIN`, `STATIC` have no inference
rule at all, so they can never be planned. Adding those rules unblocks them for
the C and Rust backends too — worth doing independently.

### Excluded by design

`SELECT` (T), `IINFLATE` (raw-DEFLATE codec), `SQRT` (F→**T**), `SRGB_ENCODE`/
`SRGB_DECODE` (U8 output), all T/T2 ops.

## Why T2 is excluded (measured)

A T2 `ADD` over 1024 elements cost 271,510 bytes; a frozen-T `ADD` cost 62,430.
**`Tables::new(24)` alone was 213,176 bytes and was rebuilt per instruction**
(78% of the T2 op). Caching it (`src/ops/t2.rs`, thread-local by `f`) dropped
the T2 op to 58,334 bytes (−78%) and the two-op job from 517,504 to 91,152.

After that fix, arena T2 storage would buy ~50 allocations/op and ~18 KB
resident — at the cost of a second kind system (`is_t2()` is a runtime property
that propagates through 11 ops, while `infer_shapes` only knows `CKind::T`) and
a duplicated exact-arithmetic implementation on the path defined by
bit-exactness. Not worth it.

## Decisions and why

- **Arena covers I and F only.** Words are raw 64-bit patterns; F kernels use
  `f64::from_bits`/`to_bits`. No `unsafe`, no second buffer.
- **Fallback only on plan failure.** A runtime error is surfaced, so the arena
  cannot hide a divergence.
- **Cap, not truncation.** Oversized working sets go back to the interpreter
  whole. Per-listing all-or-nothing.
- **Plan cache lives in `Node`** (keyed by handle and input shapes, failures
  cached too). One-shot `run_with_options` still pays compile (~46% win vs
  ~83% on the warm path).
- **Default off** until the runtime app reports a healthy hit rate.

## Lessons

1. **Measure where the bytes go before arena-izing a kind.** For I and F the
   cost was per-op buffers (arena-shaped). For T it was a LUT rebuild
   (not arena-shaped), and a 15-line cache beat two days of storage work.
2. **A gate that hides why it failed is half a gate.** The soak printed an
   empty mismatch because stderr was redirected; fixed.
3. **Accumulating kernels must zero their destination.** Arena memory is
   reused, not fresh (`DECONV2D` needed `out.fill(0)`).
4. **Canonical receipts exclude timing.** `wall_ms` floats do not survive
   parse→serialize, which is why the receipt has no timing and the diagnostic
   header is not canonicalised.
5. **The registry is the authority.** Arity comes from `ops::registry()`; a new
   mnemonic is reported "not ported" instead of silently missing.
6. **The fixed `matching` count** (how many leading operands share the output
   length) replaced op-specific validation: binary 2, shift 1, 3-op 3,
   reductions/encoders 0.
7. **Destination above operands** is what lets one buffer split into `&mut dst`
   plus immutable operands with no `unsafe`.

## Known defects found along the way

1. **`IASSEMBLE` debug panic** — `v - 2^64` in `u128` for an 8-byte window with
   the sign bit set (`src/ops/bytes.rs:129`). Debug panics; release is correct.
   Fix: `v as i64`.
2. **`LUMA` has no inference rule** — blocks the op for every backend.
3. **Scalar broadcast mismatch** — `infer_shapes` rejects `IADD(X, scalar)`
   while `op_iadd` supports it.
4. **T2 tables per thread** — 213 KB each; a process-wide `Arc` is better once
   many workers exist.
5. **One-shot plan compilation** — `jotnar-det`/`jotnar-bench` recompile per
   call; move the cache into `BoundProgram` or those callers.

## Numbers

| Metric | Value |
|---|---|
| Tests | 301 (off and on) |
| Kernels | 39 |
| Arena unit tests / migration tests | 26 / 5 |
| Soak fixtures | 7 |
| Warm node job | 837 → 140 allocations |
| EXEC reply | 254 → 139 |
| Arena steady state | 0 allocations |
| T2 `ADD` | 271,510 → 58,334 bytes |
