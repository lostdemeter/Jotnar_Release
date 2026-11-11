# T2 ("φ-float"): T + binary mantissa, integer-only

Status: **opt-in, layer-0 canary passed** on branch `jotnar-rust-t2`.
The frozen T format is unchanged and remains the default. Pass
`CONFIG t2 24` to enable it; listings without it emit the pre-T2 T
path.

## Why

The frozen T triple stores `sign · φ^((e−32768)/512)`, i.e. a value on
the K=512 log-φ lattice (step φ^(1/512) ≈ 1.00094, ~0.094 %). The
φ-lattice itself is not the limiter for DAV2 layer 0 — quantizing inputs
and weights to it and running float still gives corr 1.000000. The
limiter is that **every arithmetic result is re-snapped to the lattice**.
Scores in layer 0 reach |s| ≈ 46, so a 0.094 % snap is ~0.02 absolute;
softmax amplifies that into ~1 % attention-weight error and ~1e-2 depth
error at 518 px. Widening the fixed-point bridge (`bridge_f`) alone is not a
path to the target: on the current `62b401b` code, F=24 T measures
~0.70 for the full layer at 112/518 (AT 0.42/0.60, MB 0.95) even
though isolated LN/MATMUL improve. The opt-in `bridge_f` is
unvalidated and is not the source of T2's precision.

T2 keeps the missing fraction explicitly. It is the non-FPU analogue of
the IEEE-754 mantissa: the φ exponent does the ladder multiply, the
binary mantissa is the φ-vs-power-of-2 correction. The width-parametric
bridge (`CONFIG bridge_f`, default 18) is infrastructure used by T2;
by itself it is not a precision format.

## Encoding

A value is four planes with the same shape:

| plane | type | meaning |
|---|---|---|
| `s` | int8 | sign, ±1 |
| `e` | int32 | φ exponent, 0..65535 |
| `m` | int32 | signed binary residual |
| `z` | uint8 | zero flag (1 ⇒ value 0) |

```text
v = z ? 0 : s · φ^((e − 32768)/512) · (1 + m / 2^M)
```

`M = CONFIG t2 <M>` (default/target 24). `e` is the nearest K=512
lattice rung for `|v|`, so the factor `1 + m/2^M` is near 1.
Encoding/decoding (sensor/display boundary only) uses `f64`:

```text
e = round(512 · ln|v| / ln φ) + 32768
m = clamp(round((|v| / φ^((e−32768)/512) − 1) · 2^M), −2^(M−1), 2^(M−1)−1)
```

At runtime the format is integer-only. Values produced by the bridge
([`from_fixed`]) are canonical (`m ∈ [−2^(M−1), 2^(M−1))`). A raw `MUL`
may use two guard bits (`m ∈ [−3·2^(M−2), 5·2^(M−2))`) because the
product of two factors in `[0.5, 1.5)` lies in `[0.25, 2.25)`; the
factor is still positive, so the formula is exact and no φ-step
renormalization loop is needed. Rounding is round-half-away-from-zero
on magnitudes; `from_fixed` clamps the residual if the coarse+fine
exponent estimate lands one rung off (it is accurate to <0.05 rung).

## Arithmetic

**MUL** — φ exponents add exactly, mantissas multiply:

```text
e = clamp(ea + eb − 32768, 0, 65535)
s = sa · sb
m = ma + mb + round(ma · mb / 2^M)
```

The last term is the φ-vs-2 correction; it is pure binary integer math,
no table and no FPU. (There is no attempt to renormalize a factor of 2
into the φ exponent; that is the mismatch the mantissa exists to
absorb.)

**ADD/SUB** — align to the larger φ exponent `m0 = max(ea, eb)`, convert
both values to the fixed-point bridge at `m0`, add, and convert back:

```text
q(v) = s · round(φ^((e−m0)/512) · 2^F · (1 + m/2^M))
acc  = q(a) ± q(b)
out  = from_fixed(acc, m0)
```

Values much smaller than `m0` are resolved relative to the sum, which
is the correct block-floating behaviour.

**MATMUL** — each output element is a block-floating dot product. Pass
one finds `Emax = max(e_a[k] + e_b[k] − 32768)`; pass two accumulates

```text
term = sign · round( φ^((Emax − e_a − e_b + 32768)/512) · 2^F
                     · (1+m_a/2^M)(1+m_b/2^M) )
```

and the result goes through `from_fixed(acc, Emax)`. This replaces the
single global `m_acc` scale, which under-resolved small weights.

**LAYERNORM** — per row, `m0 = max e` over non-zero inputs; convert x,
weight and bias at `m0`; mean/var/normalize in integers; residual
epsilon from `CONFIG eps_ln`; convert back with `from_fixed`.

**GELU** — convert to the BIAS-scale fixed bridge, index the frozen
exact-erf CDF LUT (`phi_lut`), multiply, convert back. The mantissa is
used in the input conversion (the frozen T path used only the
exponent).

**SOFTMAX / SOFTMAX_WIDE** — BIAS-scale conversion, frozen `exp` LUT,
integer normalisation, convert back. T2 has the full fixed range, so
`SOFTMAX_WIDE` and `SOFTMAX` share the same T2 lowering.

`TRANSPOSE` and `CONCAT` are exact 4-plane moves.

### Overflow boundary

`from_fixed` uses a `u128`/`i128` magnitude, so `i64::MIN` does not wrap.
Both `i64::MIN` and `i64::MAX` saturate at the high T2 exponent (the
representable bridge boundary); this is documented behavior, not a panic.
The add/matmul/layernorm clamps share that boundary.

## Reference semantics and gates

`phi_core::t2` is the integer reference. The interpreter routes any
T2 stream to `src/ops/t2.rs`, which calls it; an op without a T2
reference fails loud (never silently drops the mantissa). Supported:
`MUL`, `SQUARE`, `ADD`, `SUB`, `MATMUL`, `LAYERNORM`, `GELU`,
`SOFTMAX`, `SOFTMAX_WIDE`, `TRANSPOSE`, `CONCAT`.

- `crates/phi-core/tests/t2.rs`: encode/decode, MUL/ADD/SUB vs f64,
  fixed round-trip, MATMUL vs f64.
- `tests/nonfpu_t2.rs`: emitted integer C vs the interpreter,
  **bit-identical on all four planes**, for MUL, ADD, SUB, MATMUL,
  LAYERNORM, GELU, SOFTMAX; the source passes the NonFpu float trap.
- The DAV2 application gates layer 0 with `CONFIG t2 24`, stage-wise
  against the float oracle.

## Layer-0 results (DAV2 block 0, eps 1e-6, M=24, bridge F=24)

T2 gate: `corr ≥ 0.9999`, `max|err| ≤ 1e-3`, `mean|err| ≤ 5e-5`.

| input | stage | corr | max abs | mean abs |
|---|---|---|---|---|
| 112 px (65 tok) | attention `AT` | 1.000000000 | 1.52e-5 | 9.6e-8 |
| | MLP `MB` | 0.999999991 | 1.16e-4 | 6.2e-6 |
| | full `OUT` | 0.999999984 | 1.05e-4 | 6.2e-6 |
| 518 px (1370 tok) | attention `AT` | 0.999999998 | 7.88e-5 | 9.4e-7 |
| | MLP `MB` | 0.999999991 | 1.55e-4 | 6.7e-6 |
| | full `OUT` | 0.999999978 | 9.61e-5 | 6.3e-6 |

For comparison, the frozen T path is 0.999972 (112 px) / 0.999081
(518 px). T2 is at the float-oracle noise floor and exceeds the 0.99999
target. The width-parametric bridge (`CONFIG bridge_f`, default 18) is
infrastructure used by T2; widening it alone is not a precision format.

## Cost (518 px, gcc -O2, single core)

| | T (F=18) | T2 (M=24, F=24) |
|---|---|---|
| emitted C | 5.35 MB | 5.46 MB |
| `cc -O2` | 1.2 s | 1.7 s |
| run | 11.2 s | 28.3 s |
| peak RSS | 249 MB | 411 MB |

The C is dominated by baked LUTs (fraction, coarse/fine, exp, erf).
T2 costs ~2.5× runtime: an extra plane to load/store and a two-pass,
`__int128`-checked dot product.

## Migration notes (blocks 1–11, neck, head)

- The listing change is one line: `CONFIG t2 24` (and optionally
  `bridge_f 24`). Inputs/weights become 4-plane. The DAV2 conversion
  scripts in the application repo use this switch.
- Op coverage is already the layer-0 set. For the rest of the backbone
  only the same ops are needed; port block 1 by rerunning the existing
  per-block listing with the T2 switch and its stage gate.
- Neck/head need the image ops on T2: `CONV2D`, `DECONV2D`,
  `RESIZE2D`, `RELU`. They are not implemented for T2 yet and currently
  fail loud. They should use per-output block-floating accumulators
  (same as MATMUL) and the T2 `from_fixed` bridge; `RELU` is a sign/zero
  move.
- `BATCH_MATMUL`, `ROTARY`, `ARGMAX`, `RMSNORM`, `SILU`, `T2I`, `GT`,
  `TSHIFT` have no T2 lowering yet and refuse with a typed
  `Unimplemented` error. So do the image ops `CONV2D`, `DECONV2D`,
  `RESIZE2D`, and `RELU` (layer 0 does not need them).
- T2 emission-vs-interpreter is bit-identical on all four planes
  (`tests/nonfpu_t2.rs`); the frozen T gate is unchanged
  (`tests/nonfpu_t.rs`).
- Do not change the default T semantics: `CONFIG t2` is the only switch,
  and a listing without it emits the exact pre-T2 C byte-for-byte in
  the T paths.
