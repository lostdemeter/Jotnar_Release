---
title: The Geometry of a Program
number: "02"
kicker: FIELD NOTE / THE LANGUAGE
lede: A Jotnar listing is a graph wearing the costume of a list. Learn to see the edges.
columns: 2
---

Start with the smallest useful program in the language:

::: listing [SAXPY / showcase/language/saxpy.asm]
```asm
# SAXPY: A*X + Y.  A linear elementwise specimen.
IN A
IN X
IN Y

ax = IMUL(A, X)
OUT = IADD(ax, Y)
```
:::

That is a complete listing. It declares three input streams, multiplies `A` by `X` elementwise, adds `Y`, and names the result `OUT`. The `#` lines are comments. The blank lines are for humans. The indentation is optional.

The important thing is not the order of the lines. It is the **shape of the graph**:

::: figure src="assets/diagrams/dataflow.svg" caption="SAXPY as a dataflow graph. The names are the edges; the operations are the nodes."
:::

`A`, `X`, and `Y` are sources. `IMUL` consumes `A` and `X` and produces `ax`. `IADD` consumes `ax` and `Y` and produces `OUT`. There is no register named `ax`, no stack slot, and no instruction pointer. `ax` is a name for a stream. If you never use it again, it is still a stream; the interpreter keeps it because another instruction might.

## The three verbs

The CLI has three verbs for a listing, and they answer different questions.

```sh
# Does it parse, expand, bind, and check arity?
jotnar check showcase/language/saxpy.asm

# Can the declared layouts be unified? What is the DEF interface?
jotnar verify showcase/language/saxpy.asm

# Run it against a payload.
jotnar run showcase/language/saxpy.asm \
  --payload /tmp/saxpy.json --outputs OUT
```

`check` is syntax and registry binding. `verify` replays layout unification over declared layouts only. `run` executes the graph. They are deliberately separate: a program can bind and run without declaring a single layout, and a program can declare layouts that `verify` resolves even if the runtime shape checks would later reject the data.

::: note [CHECK, VERIFY, RUN]
`check` says the listing is well-formed. `verify` says the declared layouts are consistent. `run` says the payload has the right shapes and the operations accept them. None of the three is a proof that the program computes what you meant.
:::

For SAXPY, `check` reports `2 instructions, 3 inputs, 0 state stream(s)`. `verify` reports `layouts resolved: 2/5 (0 error(s))` because only `ax` and `OUT` have layouts that can be resolved without declared input layouts. That is honest coverage reporting, not a failure.

## What the interpreter does

`run` reads the payload, binds each declared input name to a typed stream, and executes the instructions in listing order. Because every instruction consumes only streams that already exist, listing order is a valid topological order. The interpreter does not need a scheduler. It does not need to know that `ax` comes before `OUT`; the assembler already rejected the listing if it did not.

The payload is ordinary JSON:

```json
{"streams":{
  "A":{"kind":"I64","shape":[4],"data":["3","3","3","3"]},
  "X":{"kind":"I64","shape":[4],"data":["1","2","3","4"]},
  "Y":{"kind":"I64","shape":[4],"data":["10","20","30","40"]}
}}
```

The output is a stream map in the same shape. With `--outputs OUT`, the CLI writes only the named stream:

```json
{"streams":{"OUT":{"kind":"I64","shape":[4],
  "data":["13","26","39","52"]}}}
```

Integer data is printed as strings so that values above 2^53 survive JSON. If you are reading this in a language with a native int64, you may parse them as numbers.

## There are no registers

This is the mental model that matters. A register is a place. A stream is a name for a value. The language has no place. It has names, operations, and shapes.

That has consequences:

- **Order is a convenience.** The listing is written in a valid order, but the meaning is the graph.
- **Reuse is a name choice.** If you write `x = IADD(x, 1)`, you are rebinding the name, not mutating a register. The old value is still available to any instruction that already consumed it. *(Some targets, notably Verilog, reject reassignment because hardware wires are not variables. More on that later.)*
- **Dead names are allowed.** A stream that is never consumed is still computed by the reference interpreter. The emitters may eliminate it; the language does not require them to.
- **Shapes travel with values.** A stream is not just a kind; it has a shape. A scalar is a stream with an empty shape. A vector is one dimension. An image is `H×W×C`.

::: pullquote [THE GRAPH RULE]
The listing is a convenient serialization of a graph. If you can draw the graph, you can read the program.
:::

## The fail-loud moment

Try changing the arity:

```asm
IN A
IN X
OUT = IMUL(A)
```

The assembler refuses it. `IMUL` wants two inputs; you gave one. It does not insert a zero, broadcast a scalar, or hope for the best. The error names the operation and the source line. That is the contract.

The same contract applies to shapes. `IADD` requires its operands to have the same shape (or one to be a scalar literal). If you feed a `[4]` stream and a `[3]` stream, the runtime raises a shape mismatch. It does not truncate to the shorter one.

::: end
:::
