# The φ-ladder op set

This is the op set the Jotnar LLM and the Qwen-style backbone are allowed to
use. It is written for inclusion with Jotnar itself. The Qwen block uses the
`both` set (C **and** non-FPU); the C-only F64 ops are a separate path and are
not used by the φ-ladder block.

The short rule:

> Every operation must have an interpreter meaning and a C or non-FPU
> lowering. The φ-ladder is the arithmetic. The scale is part of the contract.
> An op marked `not lowered` is not allowed in the ladder contract yet.

## 1. The representation

A ladder value is a triple:

```text
Triple { s, e, z }
value = s * φ^((e - BIAS) / K)
```

| field | meaning |
| --- | --- |
| `s` | sign, `-1` or `+1` |
| `e` | exponent/rung, `0..65535` |
| `z` | zero flag |

with:

```text
φ    = 1.618033988749895
K    = 512
BIAS = 32768
```

`l_frac`, `l_coarse`, and `l_fine` turn a fixed-point integer into the
nearest ladder value. `l_fine` is the mantissa correction: the part that
makes the ladder multiplication agree with IEEE 754 to sub-epsilon.

### The φ-spoof identity

IEEE 754 multiplication and ladder multiplication have the same structure:

| component | IEEE 754 | φ-ladder |
| --- | --- | --- |
| sign | XOR | XOR |
| exponent | add with bias | rung add with bias |
| mantissa | multiply | fine-lattice correction |

This is the φ-spoof result from the phi paper (§6.2). It means a
φ-spoofed value is a valid float whose multiplication already computes
φ-multiplication. In Jotnar the triple is the ladder; the fine lattice is the
mantissa correction.

## 2. The scale contract

The scale is the single most important part of the contract. A value larger
than the scale saturates:

```text
max_value(m) = φ^((m - BIAS) / K)
```

| `m_acc` / `m_cov` | max value | use |
| ---: | ---: | --- |
| `31344` | ~0.17 | normalized image coverage |
| `32768` | 1.0 | unit-range values |
| `35800` | ~18 | small counts |
| `38000` | ~136 | the 24-case agent corpus |
| `39000` | ~460 | larger generated corpora |

The Qwen backbone sets both scales explicitly:

```asm
CONFIG m_acc 38000
CONFIG m_cov 38000
```

A task generator must either keep values inside the declared scale or raise
the scale. This is a model/contract parameter, not a hidden constant.

## 3. Ladder ops

### Linear algebra

| op | signature | behavior | lowering |
| --- | --- | --- | --- |
| `MATMUL` | `*, * -> *` | 2D matrix multiply, accumulates at `m_acc` | both |
| `BATCH_MATMUL` | `*, * -> *` | batched matrix multiply | both |
| `BMMV` | 13 args -> `*` | fused batched matrix-vector kernel | C |
| `TRANSPOSE` | `$A -> $A^T` | swap last two dims | both |
| `SLICE` | `$A, F, F, F -> $A` | slice along an axis | both |
| `CONCAT` | `$A, $A, F -> $A` | concatenate along an axis | both |
| `RESHAPE2` | `*, F, F -> *` | reshape to 2D | not lowered |
| `RESHAPE3` | `*, F, F, F -> *` | reshape to 3D | not lowered |
| `PERMUTE3` | `*, F, F, F -> *` | permute 3D | not lowered |

### Elementwise ladder arithmetic

| op | signature | behavior | lowering |
| --- | --- | --- | --- |
| `ADD` | `$A, $A -> $A` | ladder add | both |
| `SUB` | `$A, $A -> $A` | ladder subtract | both |
| `MUL` | `$A, $A -> $A` | ladder multiply | both |
| `DIV` | `$A, $A -> $A` | ladder divide | C |
| `SQUARE` | `$A -> $A` | square, clipped | both |
| `CLIP` | `$A, F, F -> $A` | clip to `[lo, hi]` | not lowered |
| `RESCALE` | `$A, F -> $A` | change the fixed scale | not lowered |

### Norms and activations

| op | signature | behavior | lowering |
| --- | --- | --- | --- |
| `RMSNORM` | `$X, $W -> $X` | RMSNorm over the last axis | both |
| `LAYERNORM` | `$X, $W, $W -> $X` | LayerNorm over the last axis | C |
| `SILU` | `$A -> $A` | `x * sigmoid(x)` | both |
| `SIGMOID` | `$A -> $A` | logistic sigmoid | not lowered |
| `GELU` | `$A -> $A` | GELU through frozen LUTs | not lowered |
| `SWIGLU` | `$A, F -> $A` | `silu(gate) * up`; input is `2*inter` wide | C |

### Attention

| op | signature | behavior | lowering |
| --- | --- | --- | --- |
| `ROTARY` | `$X, * -> $X` | rotary embedding over triple pairs | both |
| `ROPE` | `$A, *, F, F -> $A` | float RoPE over `(rows, heads*head_dim)` | C |
| `TSHIFT` | `$A -> $A` | subtract the per-row maximum | both |
| `SOFTMAX` | `$A -> $A` | softmax at BIAS scale | both |
| `SOFTMAX_WIDE` | `$A -> $A` | softmax with the wide fixed range | both |
| `TBETA` | `$A -> $A` | constant `beta_b` field of the input shape | both |
| `BETA` | `$A -> $A` | constant `beta` field | both |
| `ATTN` | 8 args -> `$A` | multi-head attention with a mask (F64 path) | C |

### Bridges

| op | signature | behavior | lowering |
| --- | --- | --- | --- |
| `T2I` | `$A -> I:*` | decode ladder to int64, round | both |
| `IGET` | `I:*, F -> I:*` | extract one integer element | both |
| `F2I` | `F:* -> I:*` | binary64 bits to int64, truncating | both |
| `I2F` | `I:* -> F:*` | int64 to binary64 | both |

`T2I` is the bridge from the ladder transformer path back to the integer
agent/harness path. `IGET` turns the `[1, D]` latent into scalar `h0_i`
streams.

## 4. Worked example: the Qwen block

This is the graph used by `Backbone::qwen_named`:

```asm
CONFIG m_acc 38000
CONFIG m_cov 38000
CONFIG beta_b 0.31622776601683794

IN tokens AS T:*
IN pos AS I:SEQ
IN emb AS T:*
IN pos_code AS T:*
IN wq AS T:*
IN wk AS T:*
IN wv AS T:*
IN wo AS T:*
IN w_gate_up AS T:*
IN w_down AS T:*
IN rms1 AS T:*
IN rms2 AS T:*
IN pool AS T:*
IN w_out AS T:*

qwen_x0 = MATMUL(tokens, emb)
qwen_x  = MUL(qwen_x0, pos_code)
qwen_n1 = RMSNORM(qwen_x, rms1)

qwen_q = MATMUL(qwen_n1, wq)
qwen_k = MATMUL(qwen_n1, wk)
qwen_v = MATMUL(qwen_n1, wv)

qwen_qr = ROTARY(qwen_q, pos)
qwen_kr = ROTARY(qwen_k, pos)
qwen_kt = TRANSPOSE(qwen_kr)
qwen_sc = MATMUL(qwen_qr, qwen_kt)
qwen_bt = TBETA(qwen_sc)
qwen_sc2 = MUL(qwen_sc, qwen_bt)
qwen_sh = TSHIFT(qwen_sc2)
qwen_pr = SOFTMAX_WIDE(qwen_sh)
qwen_at = MATMUL(qwen_pr, qwen_v)
qwen_ao = MATMUL(qwen_at, wo)
qwen_res = ADD(qwen_x, qwen_ao)

qwen_n2 = RMSNORM(qwen_res, rms2)
qwen_gu = MATMUL(qwen_n2, w_gate_up)
qwen_gate = SLICE(qwen_gu, 1, 0, 16)
qwen_up = SLICE(qwen_gu, 1, 16, 32)
qwen_silu = SILU(qwen_gate)
qwen_act = MUL(qwen_silu, qwen_up)
qwen_dn = MATMUL(qwen_act, w_down)
qwen_out = ADD(qwen_res, qwen_dn)

qwen_pooled = MATMUL(pool, qwen_out)
qwen_latent_t = MATMUL(qwen_pooled, w_out)
qwen_latent = T2I(qwen_latent_t)

h0_0 = IGET(qwen_latent, 0)
...
h0_8 = IGET(qwen_latent, 8)
```

The first slice sets `wo = 0` and `w_down = 0`. The residual stream carries
the token codes, so the block is an exact pass-through of the named latent.
The attention and SwiGLU branches are present and ready to train.

## 5. Adding a ladder op

A new op is only complete when all four exist:

1. an interpreter implementation in `src/ops/`;
2. a layout signature in `src/ops/mod.rs`;
3. a C and/or non-FPU lowering in `src/codegen/c.rs` or
   `src/codegen/nonfpu.rs`;
4. a compile/parity test that proves the lowering.

For the φ-ladder block, require the non-FPU lowering (and preferably both).
Do not add an opcode that only the interpreter can run. Do not add a
floating-point-only opcode to solve a precision problem; fix the ladder
scale or add a ladder-native op.

## 6. References

- The φ-spoof identity and the IEEE 754 equivalence: `phi_paper_full.md`
  §6.2, “The φ-Spoof Encoding”.
- The lattice substrate: `crates/phi-core/src/lattice.rs`.
- The Qwen block: `crates/jotnar-agent/src/backbone.rs`.
- The non-FPU bridge: `src/codegen/nonfpu.rs`.
