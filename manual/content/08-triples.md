---
title: Triples and the Phi Lattice
number: "07"
kicker: REFERENCE / THE INSTRUCTION SET
lede: Float at the edge; exact in the middle. A triple is a value that carries its own scale.
columns: 2
---

A **triple** is three planes with the same shape:

```text
s : sign       int8, -1 or +1
e : exponent   int32, biased by 32768
z : zero flag  uint8, 1 for exact zero
```

The value is reconstructed as:

```text
value = s * PHI^((e - 32768) / 512)
```

where `PHI = 1.618033988749895` and the exponent step is `1/512` of a phi-log unit. The zero flag is separate because zero has no sign and no exponent; trying to encode it as one would lose the distinction between “zero” and “very small.”

::: figure src="assets/diagrams/lattice.svg" caption="The phi lattice. Values are scale-free; the fixed-point bridge is where scale re-enters."
:::

## Why this exists

Floating point has a fixed number of significand bits. Every operation rounds. For a pipeline that composes dozens of operations, the rounding order matters, and different SIMD widths can produce different last bits. Jotnar’s triple path is designed to make the arithmetic **integer-only in the middle** and to push float to the sensor boundary and the display boundary.

The lattice is not a floating-point format. It is a logarithmic encoding with a shared, frozen set of lookup tables. Multiplication and division become exponent adds and subtracts; addition and subtraction go through a fixed-point bridge. The result is deterministic and cheap to implement in integer hardware.

::: note [NOT A FLOAT]
`T` is not a float kind. You cannot pass a triple to `F2I`; the operation rejects it. You cannot add a triple to a float with `ADD` unless the layouts agree. To leave the lattice, use `decode` at the boundary; to enter it, use `encode`.
:::

## The fixed-point bridge

Most triple operations do not work directly on `(s,e,z)` pairs. They convert to a fixed-point integer at a chosen scale `m`, do integer arithmetic, and convert back. The bridge is:

```text
to_fixed(t, m)   -> int64 in 2^-18 units
from_fixed(q, m) -> Triple
```

The scale `m` is a bias value, not a floating-point exponent. Two scales matter:

| Config | Default | Used for |
|---|---|---|
| `m_acc` | `31344` | accumulation, convolution, matmul |
| `m_cov` | `33280` | coverage, output, most elementwise ops |

Set them with `CONFIG`:

```asm
CONFIG m_acc 41200
CONFIG m_cov 41200
```

The bridge uses frozen LUTs for the fractional part and a fine LUT for the exponent reconstruction. Rounding is round-half-to-even, matching numpy. The emitters bake the same tables, which is why the non-FPU target can reproduce the triple path without a single `float` or `double`.

::: warning [SCALE IS PART OF THE PROGRAM]
Changing `m_acc` or `m_cov` changes the arithmetic. It is not a display setting. If you change the scale, re-run the parity gate. The frozen defaults are part of the reference contract.
:::

## Opt-in T2: a binary mantissa

The frozen triple is a pure log-φ value: `s · φ^((e−32768)/512)`. For
some models that re-snap every arithmetic result to the K=512 lattice is
the precision limit. `T2` adds an explicit binary residual, still
integer-only:

```text
v = z ? 0 : s · φ^((e−32768)/512) · (1 + m / 2^M)
CONFIG t2 24
```

The four planes are `s` (int8), `e` (int32), `m` (int32 residual), and
`z` (uint8). Multiplication adds the φ exponents and multiplies the
mantissas; the `m_a·m_b/2^M` term is the φ-vs-power-of-2 correction.
Addition aligns to the larger exponent and uses the fixed-point bridge.

T2 is **opt-in**. Listings without `CONFIG t2` emit the frozen T path
byte-for-byte. On the non-FPU target, T2 currently lowers `MUL`,
`SQUARE`, `ADD`, `SUB`, `MATMUL`, `LAYERNORM`, `GELU`, `SOFTMAX`,
`SOFTMAX_WIDE`, `TRANSPOSE`, and `CONCAT`; other ops refuse with a typed
`Unimplemented` error rather than falling back to float. The
fixed-point bridge uses a `u128` magnitude; `i64::MIN`/`i64::MAX`
saturate at the high T2 exponent (the documented overflow boundary)
instead of wrapping. See `docs/T2_PHI_FLOAT.md` for the format,
rounding, and gate.

::: warning [OPT-IN, NOT THE DEFAULT]
T2 changes the datapath. Use it when the lattice-snap precision is the
bottleneck, and gate it against the frozen T reference. The frozen T
format remains the default contract.
:::

## Arithmetic

| Mnemonic | Signature | Behavior |
|---|---|---|
| `ADD` | `$A, $A -> $A` | Elementwise add through the fixed-point bridge. |
| `SUB` | `$A, $A -> $A` | Elementwise subtract. |
| `MUL` | `$A, $A -> $A` | Elementwise multiply. |
| `DIV` | `$A, $A -> $A` | Elementwise divide. |
| `SQUARE` | `$A -> $A` | Elementwise square. |
| `SQRT` | `F:HW -> T:HW` | Square root, emitted as triples. |
| `SIGMOID` | `$A -> $A` | Logistic sigmoid through the sigmoid LUT. |
| `CLIP` | `$A, F:SCALAR, F:SCALAR -> $A` | Clamp to `[lo, hi]`. |
| `RESCALE` | `$A, F:SCALAR -> $A` | Change the fixed-point scale. |
| `GT` | `$A, $A -> I:*` | Elementwise greater-than, returning a mask. |
| `SELECT` | `I:*, $A, $A -> $A` | Masked select. |

`MUL` and `DIV` are not IEEE operations. They are lattice operations: exponents combine, signs multiply, and the zero flag dominates. That is why the result can be exact in a way a float multiply is not.

## A worked example

```asm
CONFIG m_cov 33280
IN a AS T:HW
IN b AS T:HW
IN mask AS I:HW

sum  = ADD(a, b)
sq   = SQUARE(a)
gate = GT(a, b)
out  = SELECT(mask, sq, sum)
OUT  = CLIP(out, 0.0, 1.0)
```

Every step is elementwise. `GT` produces an integer mask; `SELECT` chooses between two triple streams; `CLIP` stays in the lattice. No float appears in the listing. The emitters can lower this to integer-only C, CUDA FP32, or hardware, depending on the target.

::: pullquote [LATTICE RULE]
Float at the edge; exact in the middle. The triple is how the middle stays exact.
:::

## Reading a triple stream

A triple stream is three arrays of the same shape. In JSON:

```json
{"kind":"T","shape":[2,2],
 "s":[1,-1,1,1],
 "e":[30000,31000,32000,33000],
 "z":[0,0,0,0]}
```

The first element is positive with exponent 30000; the second is negative with exponent 31000; and so on. All four are non-zero. If `z` were 1, the value would be exactly zero regardless of `s` and `e`.

That is the whole representation. It is small, it is integer, and it is the same on every machine that Jotnar targets.

::: end
:::
