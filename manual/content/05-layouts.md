---
title: "Layouts: Geometry as Type"
number: "04"
kicker: FIELD NOTE / THE LANGUAGE
lede: A layout is a small string that tells the assembler what a stream is shaped like — and lets it refuse the wrong graph before the runtime sees it.
columns: 2
---

A layout is a string of the form `KIND:GEOM`. `KIND` is one of `F`, `T`, `I`, `U8`, or `UNKNOWN`. `GEOM` is a shape convention such as `SCALAR`, `SEQ`, `HW`, or `HWC`. The full layout is a compact type for a stream.

```text
F:SCALAR      a float scalar
F:SEQ         a float vector
F:HWC         a float image with channels
T:HW          a triple field
I:*           an integer stream of any shape
U8:HWC        a byte image
```

The registry gives every mnemonic a **layout signature**: one pattern per input, one per output. The assembler tries to unify the patterns against the actual layouts. If it cannot, the program is rejected. If it can, every output gets a resolved layout, or `UNKNOWN` when the information is not there.

## The pattern grammar

| Pattern | Meaning |
|---|---|
| `*` | Matches anything. Binds nothing. |
| `$VAR` | Unifies a whole layout string. The same variable must get the same layout everywhere. |
| `$VAR^T` | The transpose-derived form of `$VAR`. |
| `KIND:GEOM` | A literal layout. Must match exactly. |

A simple elementwise operation uses one variable for all three positions:

```text
ADD   $A, $A -> $A
```

If you call `ADD(T:HW, T:HW)`, the assembler binds `$A = T:HW` and gives the output `T:HW`. If you call `ADD(T:HW, T:HWC)`, it reports a **layout conflict**: `$A` was already `T:HW`, and the second argument wants `T:HWC`.

::: warning [CONFLICT VS. MISMATCH]
A **layout conflict** means two uses of the same variable disagree. A **layout mismatch** means a literal pattern does not match the actual layout. Both are assembly errors. Neither is repaired by broadcasting, truncation, or a silent cast.
:::

## Transpose

`TRANSPOSE` is the clean example of a derived layout:

```text
TRANSPOSE   $A -> $A^T
```

If the input is `T:HW`, the output layout is `T:HW^T`. The actual operation swaps the last two dimensions, so the data is `W×H`; the `^T` suffix records that the layout was derived by transposition rather than declared independently. If the input is `UNKNOWN`, the output stays `UNKNOWN`.

::: note [WHY THE SUFFIX?]
The suffix is not a separate shape language. It is a marker that the output was derived from the input by transposition. This lets the assembler keep the relationship between the two streams even when the concrete geometry is not known yet.
:::

## Literal signatures

Some operations are specific about kind and geometry:

| Signature | Meaning |
|---|---|
| `U8:HWC -> F:HWC` | `SRGB_DECODE`: bytes in, floats out, same image shape. |
| `F:HW -> T:HW` | `SQRT`: float field in, triple field out. |
| `F:HW2 -> I:HW` | `STATIC`: a two-channel flow becomes a one-channel mask. |
| `I:*, * -> I:*` | `IADD`: an integer stream and a scalar-or-stream operand. |

The literal patterns are how the registry prevents nonsense like decoding sRGB from a triple field or computing a flow mask from a scalar.

## Wildcards are not promises

`MATMUL` is declared as `*, * -> *`. It deliberately does not unify layouts, because matrix multiplication cares about the inner dimension, not the spelling of the geometry. The runtime has a separate shape rule:

```text
a.shape[-1] == b.shape[-2]
```

If you forget the transpose, the runtime says:

```text
inner dims [B,M,K] vs [B,N,K] (did you forget TRANSPOSE?)
```

That is the division of labor. The assembler handles the relationships it can see. The runtime handles the data-dependent ones. Both are allowed to fail loudly.

## Verification coverage

`jotnar verify` replays layout unification over declared layouts only. It prints a coverage line and, optionally, a DEF interface table.

```text
$ jotnar verify showcase/language/saxpy.asm
layouts resolved: 2/5 (0 error(s))
```

Two of five streams have resolved layouts: `ax` and `OUT`. `A`, `X`, and `Y` have no declarations, so they are `UNKNOWN`. That is not an error; it is a coverage number. If you want a higher number, declare the inputs:

```asm
IN A AS I:SEQ
IN X AS I:SEQ
IN Y AS I:SEQ
```

Then `verify` can resolve all five.

::: note [UNKNOWN IS HONEST]
`UNKNOWN` means the assembler does not have enough information. It is not a wildcard, and it is not a failure by itself. A program can run with unknown layouts; it just gets less static checking. The runtime shape checks are still active.
:::

::: pullquote [LAYOUT RULE]
A layout is a promise the assembler can check before the machine runs. A shape is a fact the runtime checks before the operation runs.
:::

## A worked disagreement

Consider:

```asm
IN a AS T:SEQ
IN b AS T:HW
OUT = ADD(a, b)
```

The signature of `ADD` is `$A, $A -> $A`. The first argument binds `$A = T:SEQ`. The second wants `T:SEQ` but gets `T:HW`. The assembler reports a layout conflict at the `ADD` line. It does not add a sequence to a field, and it does not pretend the field is a sequence.

That is the whole type system in one example. It is small, but it catches a large class of mistakes before a payload is ever loaded.

::: end
:::
