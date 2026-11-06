---
title: Linear Algebra and the Transformer
number: "08"
kicker: REFERENCE / THE INSTRUCTION SET
lede: Matrix multiply, norms, softmax, attention, and the small set of moves that make a transformer a graph.
columns: 2
---

The linear-algebra family is where the triple path earns its keep. A matmul is a long chain of multiplies and adds; if each one rounded independently, the result would depend on the order of accumulation. Jotnar accumulates in fixed-point integers at `m_acc`, then returns to the lattice at `m_cov`. The order is explicit, and the emitters preserve it.

## Matmul

| Mnemonic | Signature | Behavior |
|---|---|---|
| `MATMUL` | `*, * -> *` | 2D matrix multiply over triples. |
| `BATCH_MATMUL` | `*, * -> *` | Batched matmul over leading dimensions. |
| `BMMV` | `*13 -> *` | Fused batched matrix-vector kernel with explicit strides. |

The runtime checks the inner dimensions:

```text
a.shape[-1] == b.shape[-2]
```

The classic mistake is a missing transpose. Jotnar names it:

```text
inner dims [B,M,K] vs [B,N,K] (did you forget TRANSPOSE?)
```

A 2×2 multiply is small enough to read:

::: listing [MATMUL2 / showcase/language/matmul2.asm]
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
:::

The listing is a graph: `t00` and `t01` feed `c00`; `t10` and `t11` feed `c10`; and the second column reuses `a00`, `a01`, `a10`, `a11`. The order is chosen for reading, not for a machine’s pipeline. An emitter is free to schedule the independent products in any order that preserves the dataflow.

## Norms and activations

| Mnemonic | Signature | Behavior |
|---|---|---|
| `RMSNORM` | `$X, $W -> $X` | RMSNorm over the last axis. |
| `LAYERNORM` | `$X, $W, $W -> $X` | LayerNorm over the last axis; 2D input. |
| `SILU` | `$A -> $A` | `x * sigmoid(x)`. |
| `GELU` | `$A -> $A` | GELU through frozen LUTs. |
| `SIGMOID` | `$A -> $A` | Logistic sigmoid. |
| `SWIGLU` | `$A, F:SCALAR -> $A` | `silu(gate) * up`; the input is `2*inter` wide. |
| `SCAN` | `$X, $X, $X -> $X` | Recurrence step `h*abar + bx`. |

`RMSNORM` and `LAYERNORM` take their epsilon from `CONFIG` (`eps_rms`, `eps_ln`) or from explicit count values (`eps_rms_c`, `eps_ln_c`). The default is 4514 counts at the coverage scale. That is a deliberate, frozen calibration, not a magic float.

## Softmax and attention

| Mnemonic | Signature | Behavior |
|---|---|---|
| `SOFTMAX` | `$A -> $A` | Softmax over the last axis at BIAS scale. |
| `SOFTMAX_WIDE` | `$A -> $A` | Softmax with the wide fixed-point range. |
| `TSHIFT` | `$A -> $A` | Subtract the per-row maximum. |
| `ATTN` | `8 args -> $A` | Multi-head attention with a mask. |
| `ROTARY` | `$X, * -> $X` | Rotary embedding over triple pairs. |
| `ROPE` | `$A, *, F:SCALAR, F:SCALAR -> $A` | Float RoPE over `(rows, heads*head_dim)`. |
| `ARGMAX` | `$A, F:SCALAR -> I:*` | Index of the maximum along an axis. |
| `TOPK` | `$A, F:SCALAR -> $A, I:*` | Top-k values and indices. |

`SOFTMAX` requires BIAS-scale counts. If you feed it a stream at a different scale, it refuses. `TSHIFT` is the numerical-stability step: subtract the row maximum before exponentiating. The exp itself uses a frozen LUT, not `exp()` from libm.

`ATTN` takes query, key, value, a mask, and four integers: number of heads, number of KV heads, head dimension, and sequence length. The mask is `(per, S)`, where `per = nh / nkv`. It applies the mask by writing a large negative value into the scores, not by branching.

::: note [ROPE VS. ROTARY]
`ROPE` is the float path: input `(rows, heads*head_dim)`, positions, head count, head dimension. `ROTARY` is the triple path: input triples and a position stream. They implement the same rotation idea at different stages of the pipeline. Use the one that matches your kind.
:::

## Anatomy of a block

A transformer block is not a single mnemonic. It is a graph:

1. `RMSNORM` or `LAYERNORM` on the input.
2. `MATMUL` with the Q, K, V weights.
3. `ROTARY` or `ROPE` on Q and K.
4. `ATTN` with the mask.
5. `MATMUL` with the output projection.
6. Residual `ADD`.
7. Norm, then `SWIGLU` or `GELU`.
8. Down projection and residual `ADD`.

Every step is an ordinary stream operation. There is no “transformer” op. That is the point: the model is a graph, and the graph is visible.

::: pullquote [SCHEDULING RULE]
The listing fixes the dataflow, not the schedule. The emitter may reorder independent work; it may not change the accumulation order inside a matmul.
:::

## Moves around the math

The linalg path depends on a small set of shape moves:

| Mnemonic | Signature | Behavior |
|---|---|---|
| `TRANSPOSE` | `$A -> $A^T` | Swap the last two dimensions. |
| `RESHAPE2` | `*, F:SCALAR, F:SCALAR -> *` | Reshape to two dimensions. |
| `RESHAPE3` | `*, F:SCALAR, F:SCALAR, F:SCALAR -> *` | Reshape to three dimensions. |
| `PERMUTE3` | `*, F:SCALAR, F:SCALAR, F:SCALAR -> *` | Permute three dimensions. |
| `CONCAT` | `$A, $A, F:SCALAR -> $A` | Concatenate along an axis. |
| `SLICE` | `$A, F:SCALAR, F:SCALAR, F:SCALAR -> $A` | Slice along an axis. |
| `GATHER` | `*, I:* -> *` | Gather rows by index. |
| `SCATTER` | `$A, $A, I:* -> $A` | Scatter values into a copy. |

These are not cosmetic. A reshape changes the stride interpretation; a transpose changes the layout; a concat changes the axis length. The runtime checks what it can, and the emitters infer static shapes before lowering. If a shape is dynamic where the C emitter needs a constant, the emitter says so instead of guessing.

::: end
:::
