---
title: Exact Integers
number: "06"
kicker: REFERENCE / THE INSTRUCTION SET
lede: The integer family is the floor of the language. It wraps, it divides, it compares, and it never guesses.
columns: 2
---

Jotnar’s integer streams are signed 64-bit values. Addition, subtraction, and multiplication wrap modulo 2^64. That is not an accident or a platform detail; it is the contract. If you need saturation, use `SADD`. If you need unsigned division, use `IDIVU`. If you need a different width, mask with `IAND`.

## The core

| Mnemonic | Signature | Behavior |
|---|---|---|
| `IADD` | `I:*, * -> I:*` | Wrapping add. |
| `ISUB` | `I:*, * -> I:*` | Wrapping subtract. |
| `IMUL` | `I:*, * -> I:*` | Wrapping multiply. |
| `IMAD` | `I:*, *, * -> I:*` | `a*b + c`, wrapping. |
| `SADD` | `I:*, * -> I:*` | Saturating add. |
| `IMAX` | `I:*, * -> I:*` | Elementwise maximum. |
| `IDIV` | `I:*, * -> I:*` | Truncating signed division. |
| `IMOD` | `I:*, * -> I:*` | Remainder of truncating division. |
| `IDIVU` | `I:*, * -> I:*` | Unsigned 64-bit division. |
| `IMODU` | `I:*, * -> I:*` | Unsigned 64-bit remainder. |

`IDIV` and `IMOD` follow C’s truncation toward zero, not Python’s floor division. They raise on division by zero and on `INT64_MIN / -1`, which is unrepresentable. `IDIVU` and `IMODU` reinterpret the operands as unsigned 64-bit values before dividing.

## Bits and shifts

| Mnemonic | Signature | Behavior |
|---|---|---|
| `IAND` | `I:*, * -> I:*` | Bitwise and. |
| `IOR` | `I:*, * -> I:*` | Bitwise or. |
| `IXOR` | `I:*, * -> I:*` | Bitwise exclusive-or. |
| `ISHL` | `I:*, F:SCALAR -> I:*` | Logical left shift by 0..63. |
| `ISHR` | `I:*, F:SCALAR -> I:*` | Logical right shift by 0..63. |
| `ISAR` | `I:*, * -> I:*` | Arithmetic right shift by 0..63. |

`ISHL` and `ISHR` take a scalar shift amount. `ISAR` takes a stream or scalar. All three reject shift counts outside `0..=63`. There is no “shift by 64 means zero” rule; that is a bug waiting to happen.

::: warning [SHIFT RANGE]
A shift count of 64 is an error, not a zero. A negative count is an error. If you are generating shifts from data, clamp them first.
:::

## Comparisons and select

| Mnemonic | Signature | Behavior |
|---|---|---|
| `IEQ` | `I:*, * -> I:*` | `1` if equal, else `0`. |
| `ILT` | `I:*, * -> I:*` | `1` if `a < b`, else `0`. |
| `ISELECT` | `I:*, *, * -> I:*` | `mask != 0 ? a : b`. |
| `F2I` | `F:* -> I:*` | Float to int, truncating toward zero. |
| `I2F` | `I:* -> F:*` | Integer to F64. |

`ISELECT` is the branch-like primitive. It is not a branch; both arms are computed, and the mask chooses elementwise. That makes it safe for data-dependent control flow in a graph and easy to lower to hardware.

```asm
# A two-way select. One condition, two data inputs, no arithmetic.
IN C
IN A
IN B

OUT = ISELECT(C, A, B)
```

`F2I` is strict. It rejects `NaN`, `Inf`, values outside the int64 range, and triples. It truncates toward zero, so `-3.9` becomes `-3`. If you need rounding, do it explicitly before the conversion.

::: note [INTEGERS AND JSON]
The payload boundary accepts integers as JSON numbers or strings. Use strings for values above 2^53. The output side writes integers as strings for the same reason. The interpreter itself always uses int64; the string form is only a transport detail.
:::

## A worked example: wrapping

```asm
IN a
IN b
s = IADD(a, b)
t = IMUL(s, 3)
OUT = IAND(t, 255)
```

This is a wrapping add, a wrapping multiply by three, and an 8-bit mask. If `a` and `b` are bytes, `s` can carry into bit 8 and above; `IAND` keeps the low byte. That is the exact idiom used by the CPU8 listing in Chapter 16.

::: pullquote [INTEGER RULE]
If you want a different width, mask it. If you want saturation, ask for it. If you want a conversion, name it.
:::

## Why not just use floats?

Because floats are not a superset of integers. An int64 above 2^53 cannot be represented exactly in an F64. A checksum, an address, a byte count, or a hash is not a float. Jotnar keeps the integer path exact all the way to the emitter: the C target uses `uint64_t` arithmetic for wrapping ops, and the non-FPU target never touches a float for integer work.

The integer family is also the substrate for the CPU and byte chapters. Once you can add, mask, shift, compare, and select, you can build a machine.

::: end
:::
