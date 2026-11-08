# The task-encoder backbone

The task encoder used to begin with a fixed named-selection matrix:

```
h0 = W_enc * task + b_enc
```

That matrix is legible, but it is not a model. The backbone abstraction replaces
it with a Jotnar listing fragment:

```
task features t0..t19 -> backbone -> h0..h8
```

Everything downstream is unchanged: the looped recurrent block, the action
head, the fused budget head, the driver, and the exact gate.

## Interface

A `Backbone` owns two layers:

```text
w1: H x T   hidden projection
b1: H
w2: D x H   latent readout (fixed point)
b2: D       (fixed point)
scale, out_scale, relu
```

`build_h0(lines)` emits the fragment and returns the names of the initial
latent streams. The looped encoder splices those names into its existing
recurrent block.

Two backbones are implemented:

| kind | hidden | readout | purpose |
| --- | --- | --- | --- |
| `Named` | identity, no ReLU | the named `W_enc` | exact baseline |
| `Mlp` | mixed task features, ReLU | fitted | first learned backbone |

## The MLP slice

The MLP hidden layer is deterministic:

```
u_j = (sum_i w1[j][i] * t_i + b1[j]) / scale
h_j = relu(u_j)
h0_l = (sum_j w2[l][j] * h_j + b2[l]) / out_scale
```

`w1` mixes each task feature with its neighbour (`mix`), and the readout is
fitted with a ridge correction around the named readout, targeting the named
latent on the calibration task vectors. The readout is stored in fixed point
(`out_scale=64`), so the fit can express fractional weights while the listing
still uses integers and one final `IDIV`.

The first `T` hidden units are seeded with the named readout; extra hidden
units start at zero and the fit decides whether to use them. With `mix=0` this
is an exact reparameterization of the named selection. A small `mix` gives the
fit a little room; a larger mix makes the hidden features drift enough that
the held-out gate rejects the result (see below). `--lr` is the ridge strength:
higher values allow a larger correction, while the default 1 keeps the readout
close to the named baseline.

```sh
cargo run -q -p jotnar-agent -- fit-backbone \
  --out runs/backbone --corpus --hidden 32 --mix 1
```

```
backbone fit: 18 samples, hidden=32, mix=1, lr=1, max latent error=2
named: 14 correct, 4 wrong -> 18 correct, 0 wrong (adopted=true)
mlp:   14 correct, 4 wrong -> 18 correct, 0 wrong (adopted=true)
heldout named=6 correct, 0 wrong; mlp=6 correct, 0 wrong
```

Run the fitted backbone:

```sh
cargo run -q -p jotnar-agent -- run --out runs/backbone-run \
  --proposer task --corpus \
  --looped-encoder-weights runs/encoder/looped-encoder.weights.json \
  --backbone runs/backbone/backbone.weights.json
```

`fit-backbone` also writes `looped-encoder-backbone.weights.json`, a single
file with the backbone and the action head. Use it directly when you do not
need the fused budget head, or add `--budget-predictor` to attach the budget
head at run time.

The gate is the same one used everywhere: 18/18 selection, 6/6 held-out, and a
positive adoption flag. The named baseline is evaluated in the same run, so the
backbone has to earn its place.

## The Qwen slice

The second backbone is a Qwen-style block over the task tokens. It is a real
attention + SwiGLU graph, expressed entirely in the φ-ladder:

```text
tokens -> MATMUL(emb) -> MUL(pos_code)
       -> RMSNORM -> Q/K/V -> ROTARY -> MATMUL scores -> TBETA -> MUL
       -> TSHIFT -> SOFTMAX_WIDE -> MATMUL(V) -> MATMUL(wo) -> ADD residual
       -> RMSNORM -> MATMUL(gate+up) -> SLICE -> SILU -> MUL -> MATMUL(down)
       -> ADD residual -> pool -> readout -> T2I -> IGET -> h0
```

The φ-spoof result is the key: IEEE 754 multiplication is structurally the
φ-ladder multiplication — sign, exponent, and mantissa correction. The triple
stream is that ladder. The earlier failure was not a missing float opcode; it
was the wrong lattice scale. `CONFIG m_acc 38000` and `m_cov 38000` put the
task counts (up to ~128) inside the representable range, and `T2I` decodes
back to the integer latent. The block uses the existing `MATMUL`, `RMSNORM`,
`ROTARY`, `TBETA`, `TSHIFT`, `SOFTMAX_WIDE`, `SILU`, `MUL`, `ADD`, and `SLICE`
ops, plus the new `T2I`/`IGET` bridge. Both bridge ops have C and non-FPU
lowerings; `qwen_backbone_compiles_to_nonfpu` checks that the Qwen listing
lowers to the integer target.

The first slice initializes the attention output projection and the MLP down
projection to zero. The residual stream carries the token codes, so the block
is an exact pass-through of the named latent; the branches are present and
ready to train. This is the standard zero-init residual trick, not a bypass:
the listing really executes the attention and SwiGLU graph.

```sh
cargo run -q -p jotnar-agent -- fit-backbone \
  --out runs/backbone-qwen --corpus --qwen --dim 10 --inter 16
```

```
backbone fit: 18 samples, kind=qwen, dim=10, inter=16, max latent error=0
named:    14 correct, 4 wrong -> 18 correct, 0 wrong (adopted=true)
backbone: 14 correct, 4 wrong -> 18 correct, 0 wrong (adopted=true)
heldout named=6 correct, 0 wrong; backbone=6 correct, 0 wrong
```

The attention and SwiGLU branches are zero-initialized. Training them is the
next step; the gate accepts only if the trained block preserves or improves
18/18 selection and 6/6 held-out.

The Qwen block is not a new floating-point opcode family. It stays on the
φ-ladder and therefore inherits the C/non-FPU lowering; the only new bridge
ops are `T2I` and `IGET`, both with C and non-FPU primitives. The exact gate is
still the reference interpreter. The full op table and the scale contract are
in [`LADDER_OPS.md`](../../../docs/LADDER_OPS.md).

### Calibration fit

`fit-qwen` trains the branch readouts on the calibration split. The Q/K/V and
gate/up features are deterministic random; the attention output projection
(`wo`) and the MLP down projection (`w_down`) are fit with ridge least squares
on **masked** calibration tasks. The target is the named latent of the full
task, so the attention/MLP branches learn to recover information from the
remaining tokens.

```sh
cargo run -q -p jotnar-agent -- fit-qwen \
  --out runs/qwen --masks 8 --mask-rate 0.3 \
  --ridge 0.001 --rounds 3 --seed 1 --scale 0.2
```

```
qwen fit: 6 calibration cases, 48 masked samples, dim=10, inter=16,
          seed=1, ridge=0.001, rounds=3,
          masked latent error: before=63.0000, after=41.3479
named:   14 correct, 4 wrong -> 18 correct, 0 wrong
trained: 14 correct, 4 wrong -> 18 correct, 0 wrong
heldout named=6 correct, 0 wrong; trained=6 correct, 0 wrong; adopted=true
```

The gate accepts the trained block. This is a calibration readout fit, not
end-to-end backprop: the branch features are fixed and only the output
projections are fitted. The next step is end-to-end/teacher training on a
larger trace set.

The fused budget head is calibrated jointly. Refit it against the trained
backbone:

```sh
cargo run -q -p jotnar-agent -- fit-budget-predictor \
  --out runs/qwen-budget --corpus \
  --backbone runs/qwen/backbone-qwen-trained.weights.json
```

`loops_for_task` uses `max(predicted_budget, structural_prior)`: the learned
head may raise the family/depth budget, but it may not drop below the
structural prior. This keeps the trained backbone from being under-budgeted.
The fused file reaches 18/18 selection and 6/6 held-out.

## What the gate rejects

A more aggressive hidden mix fits the training latent better but fails the
held-out gate:

| mix | max latent error | selection | held-out |
| ---: | ---: | ---: | ---: |
| 1 | 2 | 18/18 | 6/6 |
| 2 | 3 | 14/18 | 4/6 |
| 4 | 2 | 14/18 | 4/6 |

The fitted readout interpolates the 18 training latents, but the looped action
head is sensitive to the latent scale and the held-out cases expose the
overfit. The accepted first slice keeps `mix=1` and the gate. This is the
calibration problem the attention block has to solve, not a reason to loosen
the gate.

## Why this is the right first slice

- It changes one block, not the loop, heads, driver, or certificates.
- It is still a Jotnar listing; `asm::assemble` and the exact gate apply.
- It has a named baseline that is exactly reproducible.
- It gives the Qwen-style block a fixed contract: task vector in, 9-channel
  latent out.

## Next

The MLP is a placeholder for the attention block, not the destination. The
next slice replaces `w1`/`w2` with a Qwen-style block over the task tokens:

1. embed the 20 task features as tokens,
2. one self-attention head (Q/K/V, softmax, output projection),
3. a SwiGLU/RMSNorm block,
4. the same `h0..h8` readout and the same gate.

The interface, the corpus, the budget head, and the exact gate stay fixed. The
backbone grows only if it preserves the 18/18 and 6/6 results.
