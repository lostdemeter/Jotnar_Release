---
title: Four First Listings
number: "15"
kicker: FIELD NOTES / HANDS ON
lede: The first fifteen minutes of Jotnar: a linear map, a reduction, a shared operand graph, and a branch-like select.
columns: 2
---

These four listings are the tour. They are small enough to run in a terminal and complete enough to show the language’s shapes. The files live in `showcase/language/`.

::: note [RUN THE WHOLE TOUR]
```sh
cargo run --release --bin jotnar-demo-tour
```
The tour runs all four and prints the results. You can also run each listing by hand with the CLI.
:::

## 1. SAXPY

```asm
# SAXPY: A*X + Y.  A linear elementwise specimen.
IN A
IN X
IN Y

ax = IMUL(A, X)
OUT = IADD(ax, Y)
```

Payload:

```json
{"streams":{
  "A":{"kind":"I64","shape":[4],"data":["3","3","3","3"]},
  "X":{"kind":"I64","shape":[4],"data":["1","2","3","4"]},
  "Y":{"kind":"I64","shape":[4],"data":["10","20","30","40"]}
}}
```

Expected: `[13, 26, 39, 52]`.

What to notice: `IMUL` and `IADD` are wrapping integer operations. The graph is two nodes deep. The output shape is the same as the input shape because both operations are elementwise.

## 2. Reduce

```asm
# 8-way tree reduction.  A balanced fan-in-2 tree.
IN A0
IN A1
IN A2
IN A3
IN A4
IN A5
IN A6
IN A7

s0 = IADD(A0, A1)
s1 = IADD(A2, A3)
s2 = IADD(A4, A5)
s3 = IADD(A6, A7)
t0 = IADD(s0, s1)
t1 = IADD(s2, s3)
OUT = IADD(t0, t1)
```

Expected for inputs `0..7`: `28`.

What to notice: the tree is balanced. The graph has depth 3, not 7. The names `s0`, `s1`, `t0`, `t1` are descriptive, not special. A different naming would produce the same graph.

## 3. Matmul2

```asm
# 2x2 matrix multiply. Four dot products with a shared operand graph.
IN a00
IN a01
IN a10
IN a11
IN b00
IN b01
IN b10
IN b11

t00 = IMUL(a00, b00)
t01 = IMUL(a01, b10)
c00 = IADD(t00, t01)
t10 = IMUL(a10, b00)
t11 = IMUL(a11, b10)
c10 = IADD(t10, t11)
t02 = IMUL(a00, b01)
t03 = IMUL(a01, b11)
c01 = IADD(t02, t03)
t12 = IMUL(a10, b01)
t13 = IMUL(a11, b11)
c11 = IADD(t12, t13)
```

For the identity times `[[1,2],[3,4]]`, expected `c00=1, c01=2, c10=3, c11=4`.

What to notice: the inputs are scalars, not matrices. The 2×2 structure is in the naming and the graph, not in a shape declaration. This is the smallest example of a shared operand graph: `a00` feeds both `t00` and `t02`.

## 4. Select

```asm
# A two-way select. One condition, two data inputs, no arithmetic.
IN C
IN A
IN B

OUT = ISELECT(C, A, B)
```

Expected: wherever `C` is nonzero, `OUT` takes `A`; elsewhere it takes `B`.

What to notice: `ISELECT` is not a branch. Both arms are values. The mask chooses elementwise. That is what makes the operation safe to reorder and easy to lower to hardware.

::: pullquote [TOUR RULE]
If you can name the nodes, you can read the graph. If you can read the graph, you can run it.
:::

## Run one by hand

```sh
cat > /tmp/saxpy.json <<'JSON'
{"streams":{
  "A":{"kind":"I64","shape":[4],"data":["3","3","3","3"]},
  "X":{"kind":"I64","shape":[4],"data":["1","2","3","4"]},
  "Y":{"kind":"I64","shape":[4],"data":["10","20","30","40"]}
}}
JSON

cargo run --release --bin jotnar -- run showcase/language/saxpy.asm \
  --payload /tmp/saxpy.json --outputs OUT
```

The same listing can be lowered to C:

```sh
cargo run --release --bin jotnar -- build showcase/language/saxpy.asm \
  --payload /tmp/saxpy.json --outputs OUT --workdir /tmp/saxpy --name saxpy
```

::: end
:::
