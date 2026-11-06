---
title: Streams, Kinds, and Shapes
number: "03"
kicker: FIELD NOTE / THE LANGUAGE
lede: A stream is a name, a kind, a shape, and a layout. The first three are data; the fourth is a promise.
columns: 2
---

Every value in Jotnar is a **stream**. A stream has four pieces of identity:

1. a **name** chosen by the author;
2. a **kind** — what the elements are;
3. a **shape** — how many elements and how they are arranged;
4. a **layout** — a compact string that combines kind and geometry for the assembler.

The name is local to the listing. The kind and shape are checked at run time. The layout is checked at assembly time, when it is declared.

## The kinds

| Kind | Layout letter | Payload kind | What it holds |
|---|---|---|---|
| `F32` | `F` | `F32` | 32-bit floats. |
| `F64` | `F` | `F64` or `F` | 64-bit floats. |
| `I64` | `I` | `I64` or `I` | Signed 64-bit integers. |
| `U8` | `U8` | `U8` | Unsigned bytes. |
| `Bool` | `I` | `Bool` | Boolean masks. |
| `Triple` | `T` | `T` | Three planes: sign, exponent, zero flag. |
| `None` | `UNKNOWN` | `None` | Absence, used for history streams. |

`F32` and `F64` are distinct even though both use the layout letter `F`. The reference interpreter preserves the distinction where numpy does. A `Bool` stream is an integer mask at the layout level; `SELECT` and `ISELECT` accept it.

`Triple` is the important odd one. It is not a float, and it is not an integer. It is a scale-free value in the phi lattice. Chapter 08 is entirely about it.

::: note [SCALARS]
A scalar is not a separate kind. It is a stream with an empty shape. A bare number in argument position is a scalar literal. The layout resolver treats it as `F:SCALAR`. Operations that accept a scalar (shifts, clip bounds, axes, strides) read it as an integer or float depending on the operation.
:::

## Shapes

| Shape | Meaning | Example layout |
|---|---|---|
| `[]` | Scalar. | `F:SCALAR` |
| `[N]` | Vector of length `N`. | `F:SEQ`, `I:*` |
| `[H, W]` | 2D field or matrix. | `T:HW`, `F:HW` |
| `[H, W, C]` | Image with channels. | `F:HWC`, `U8:HWC` |
| `[B, H, W, C]` | Batch of images. | `*` |
| `[R, D]` | Rows of features. | `*` |

Jotnar does not have a fixed list of shape names. `HW`, `HWC`, `SEQ`, and `SCALAR` are conventions used by the registry’s signatures. `*` matches any shape and binds nothing.

::: warning [SHAPE IS NOT OPTIONAL]
The interpreter checks shapes for the operations that have shape rules: elementwise arithmetic, `MATMUL`, `WARP`, `SELECT`, `ARGMAX`, `GATHER`, and `GATHER3`. An operation without a shape rule can still fail loudly on its own terms. Never assume a wildcard in a signature means “any shape will work.”
:::

## Declaring inputs

An input declaration may carry a layout:

```asm
IN rgb AS U8:HWC
IN depth AS T:HW
IN pos AS I:SEQ
```

The layout is a promise that the assembler uses for unification. It is not a conversion. If the payload delivers an `F64` stream under a name declared `U8:HWC`, the runtime cross-check fails. The layout tells the assembler what to expect; the payload has to agree.

You can declare several names at once:

```asm
IN a, b AS T:SEQ
```

Each name gets the same layout. Duplicate input names are an error.

## The payload

The payload is a JSON object with a `streams` map. Each stream has a `kind`, a `shape`, and data. The data key depends on the kind:

```json
{
  "streams": {
    "x":   {"kind":"F64","shape":[2,3],
            "data":[0.1,0.2,0.3,0.4,0.5,0.6]},
    "pos": {"kind":"I64","shape":[2],"data":["0","1"]},
    "rgb": {"kind":"U8","shape":[2,2,3],"data":[0,1,2,3,4,5,6,7,8,9,10,11]},
    "t":   {"kind":"T","shape":[2,2],
            "s":[1,-1,1,1],"e":[30000,31000,32000,33000],"z":[0,0,0,0]}
  }
}
```

`F32` and `Bool` are accepted too. Integer entries may be numbers or strings; strings preserve int64 values above 2^53. The output uses the same shape, with integer values encoded as strings for the same reason.

::: note [NONE]
A stream may be `None`. The `WARP` operation passes `None` through, which is how a temporal pipeline represents “no history yet.” `None` is not zero. It is a distinct value that most operations reject.
:::

## Kinds are not coerced

There is no implicit cast from `F` to `I`. `F2I` is an explicit operation, and it truncates toward zero, rejects NaN and infinity, and rejects values outside the int64 range. `I2F` is explicit too, and produces `F64`. If you want a different kind, ask for it.

The same is true of `Bool` and `U8`: they are distinct at the payload boundary even though many operations accept both as integer-like masks or bytes. The fail-loud doctrine is not only about errors. It is about making every conversion visible.

::: end
:::
