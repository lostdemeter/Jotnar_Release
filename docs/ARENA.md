# Stream arena

Status: **opt-in, default off, 53 kernels.**
Base: `d98dfe2`. 320/320 tests with `JOTNAR_ARENA` off and on.

## What this is

These are runtime ideas ported from the goose experiments: caller-owned
reusable storage, flat payloads with an admission witness, and allocation
budgets enforced by the runtime. Jotnar keeps its listing language and
interpreter; the arena is an opt-in realization of the same graph.

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
   pointers, one-store reset, 53 kernels.

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
kernel, `plan.bytes() <= max_bytes()`, and the run is single-threaded and
unprofiled. Otherwise it falls back to the interpreter. A **runtime** failure
is returned, never silently re-run.

`tests/arena_migration.rs::every_soak_fixture_exercises_the_path_it_claims`
pins which fixture takes which path, so the soak cannot pass by falling back
every time. `arena_fallback` is the one deliberate interpreter case (`LUMA`
infers, but has no arena kernel).

## Coverage

53 kernels. Registry is 94 entries.

| Group | Total | Ported |
|---|---:|---:|
| I-first | 34 | 32 |
| F-first | 6 | 1 |
| Generic `$A` | 27 | 6 |
| Unconstrained `*` | 14 | 11 |
| T | 6 | 0 (excluded) |
| `$X` | 4 | 3 |
| U8 | 1 | 0 (excluded) |
| No layout signature (`ATTN`, `BMMV`) | 2 | 0 |

Ported: 4 elementwise-shift/3-op groups (19), reductions (2), encoder (1),
matmul (1), index/window/find/gather/scatter/pack/CRC (8), F: RELU, F2I, I2F,
CONCAT, SLICE, RESIZE2D, CONV2D, DECONV2D, then the last 14:
`RESHAPE2`/`RESHAPE3`, `LAYERNORM`/`RMSNORM`/`ROTARY`, `MATMUL`/`BATCH_MATMUL`,
`CONV`/`INTERP`, `GATHER`/`GATHER3`/`SCATTER`, `ARGMAX`, `TOPK`.

### The reference gap that had to be fixed first

`infer_shapes` and the C/Rust emitters accepted F for ten of those fourteen,
but the interpreter's implementations still called `arg_triple` and refused F:
`RESHAPE2`, `RESHAPE3`, `LAYERNORM`, `RMSNORM`, `ROTARY`, `MATMUL`,
`BATCH_MATMUL`, `CONV`, `INTERP`, `GATHER`. Porting only the arena would have
made those listings succeed with `JOTNAR_ARENA=1` and fail with it off — the
fallback and soak invariant is exactly that this cannot happen.

The interpreter now has F paths for all ten, mirroring the existing C emitter
arithmetic (including `MATMUL`'s `ikj` accumulation order and the C norms'
reciprocal-sqrt form). `TOPK` was moved from `sort_by` to the C emitter's
single-pass insertion so ties, NaN ordering, and allocation are identical on
both paths. `SCATTER` remains F-only in `infer_shapes`; the task note's "F/I"
was wrong. `GATHER` now supports F and I in the interpreter, arena, and C emitter.

### Backend support for the last fourteen

`Arena` is a runtime path, not a codegen target; `—` means the backend refuses
the kind with a typed error rather than guessing.

| op | interpreter | C | Rust | arena |
|---|---|---|---|---|
| `RESHAPE2`, `RESHAPE3` | T/F/I | T/F/I | F/I | F/I |
| `LAYERNORM` | T/F | F | F | F |
| `RMSNORM` | T/F | F | — | F |
| `ROTARY` | T/F | F | — | F |
| `MATMUL`, `BATCH_MATMUL` | T/F | F | F | F |
| `CONV` | T/F | F | F | F |
| `INTERP` | T/F | F | — | F |
| `GATHER` | T/F/I | T/F/I | — | F/I |
| `GATHER3` | T/F | T/F | — | F |
| `SCATTER` | F | F | — | F |
| `ARGMAX` | T/F/I | T/F | — | F/I |
| `TOPK` | F | F | — | F |

The interpreter and arena rows grew the F paths in this change, and the C
emitter now accepts `GATHER` I as well. The Rust backend remains unported for
`GATHER` (`—` in the table).

### Rules added, arena kernels not yet

`LUMA`, `PERMUTE3`, `DECONV`, `POOLAVG`, `GAIN`, `STATIC` now have
`infer_shapes` rules, so shape inference no longer blocks them. None is an
arena kernel yet:

- `LUMA` — F→F; the only clean candidate. Do **not** port it speculatively:
  the trigger is a real listing (or a hit-rate report) that falls back because
  of it.
- `STATIC` — F→Bool; needs a Bool output representation.
- `GAIN` — consumes T and returns F32; needs both T and F32.
- `PERMUTE3`, `DECONV`, `POOLAVG` — T-only.

T/T2 remains excluded for the measured reasons below; adding any of these four
T consumers is a T-arena decision, not a kernel task.

### Excluded by design

`SELECT` (T), `IINFLATE` (raw-DEFLATE codec), `SQRT` (F→**T**), `SRGB_ENCODE`/
`SRGB_DECODE` (U8 output), all T/T2 ops.

## Why T2 is excluded (measured)

A T2 `ADD` over 1024 elements cost 271,510 bytes; a frozen-T `ADD` cost 62,430.
**`Tables::new(24)` alone was 213,176 bytes and was rebuilt per instruction**
(78% of the T2 op). A 15-line thread-local cache drops the T2 op to 58,334 bytes (−78%) and the
two-op job from 517,504 to 91,152.

After that cache, arena T2 storage would buy ~50 allocations/op and ~18 KB
resident — at the cost of a second kind system (`is_t2()` is a runtime property
that propagates through 11 ops, while `infer_shapes` only knows `CKind::T`) and
a duplicated exact-arithmetic implementation on the path defined by
bit-exactness. Not worth it.

## Decisions and why

- **Arena covers I and F only.** Words are raw 64-bit patterns; F kernels use
  `f64::from_bits`/`to_bits`. No `unsafe`, no second buffer.
- **No speculative arena kinds.** A kernel for an unused op is not progress.
  `LUMA` waits for a real fallback; `STATIC` needs Bool; the T consumers need a
  new T-arena decision. The inference rules are enough groundwork for now.
- **Fallback only on plan failure.** A runtime error is surfaced, so the arena
  cannot hide a divergence.
- **The arena never widens the language.** If a kind has no interpreter path,
  add the interpreter path first; otherwise the same listing succeeds or fails
  depending on `JOTNAR_ARENA`. Ten ops needed that fix before they could be
  planned honestly.
- **Plan carries the program `Config`.** Eps and `rope_base` come from the
  listing, not from a fresh `Config::new()`, so norms/ROTARY match the
  interpreter exactly.
- **Float literals are a distinct operand.** `Arg::ConstF` keeps `INTERP`'s
  `0.5` scales representable without weakening integer literals.
- **`TOPK` uses insertion, not a sort.** The C emitter's single-pass loop gives
  ties-first order, a defined NaN position, and no allocation; `sort_by` with a
  non-total comparator gives none of those guarantees.
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
8. **Parity is not possible across a missing reference.** `infer_shapes` and
   the C emitter promised F for ten ops while the interpreter refused it.
   Writing arena kernels first would have made the same listing succeed only
   when the arena was on. Fix the reference (or shrink the plan) first.

## Known defects found along the way

1. **Arena default stays off** — pending a hit-rate report from the runtime app.
2. **Six rule-enabled ops have no arena kernel** — see "Rules added, arena
   kernels not yet". None is worth speculative work: `LUMA` needs a defined
   need, `STATIC` needs Bool, and the other four need T.
3. **`jotnar-bench` still compiles its one-shot reference plan** — `jotnar-det`
   now uses `WarmRunner`, but the bench reference path does not.

`IASSEMBLE`'s debug `u128` underflow is fixed: the interpreter truncates the
low 64 bits (`v as i64`) instead of computing `v - 2^64` in `u128`. The
eight-byte sign-bit case has an interpreter test.

## Numbers

| Metric | Value |
|---|---|
| Tests | 320 (off and on) |
| Kernels | 53 |
| Arena unit tests / migration tests | 36 / 6 |
| Soak fixtures | 12 |
| Warm node job | 837 → 140 allocations |
| EXEC reply | 254 → 139 |
| Arena steady state | 0 allocations |
| T2 `ADD` | 271,510 → 58,334 bytes |
