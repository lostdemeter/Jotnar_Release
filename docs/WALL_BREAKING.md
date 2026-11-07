# Wall-Breaking Protocol in Jotnar

This is the first Jotnar-native slice of the Wall-Breaking Protocol (WBP). It
exists because the state model is additive/linear and real interactions are
not. Instead of adding a nonlinear layer, WBP asks a different question:

> Is the residual structured? If it is, peel the self-similar layers and use
> them as a basis. If it looks ergodic, inject an irrational harmonic and look
> again. If it still looks ergodic, the wall is genuine.

The exact gate is untouched: WBP is a proposer/diagnostic, not a judge.

## The mapping

| WBP idea | Jotnar expression |
| --- | --- |
| `resfrac_score` | an AR(1)/AR(2) listing over an `I:SEQ` signal |
| `FractalPeeler` | host loop over the `resfrac` listing; each layer is a listing run |
| `ErgodicJump` | a baked irrational harmonic added by an `IADD` listing, then `resfrac` again |
| self-similar probe/feature | `x_i * x_{i-lag}` via `IWINDOW` + `IMUL` |
| φ-clock / φ-structured probes | baked integer tables (data), projected by `IMATMUL` |

The listing uses only existing integer ops: `IWINDOW`, `IMUL`, `ISUM`, `IADD`,
`ISUB`, `IDIV`, `IMAX`, `IEQ`, `ISELECT`, `IGET`. No new opcodes and no floats
at runtime. Values are scaled by `SCALE = 1024`; `rho^2` is returned scaled by
`SCALE` so the Rust side can take the square root for reporting.

## Commands

```sh
# Audit an arbitrary integer signal (comma-separated or JSON array file).
cargo run -q -p jotnar-agent -- wbp-audit \
  --file runs/wbp-signal.json --order 2 --layers 5

cargo run -q -p jotnar-agent -- wbp-audit \
  --signal "1,2,3,5,8,13,21,34,55,89,144,233"
```

The audit prints the peeled `rho` trajectory and, if the final residual is
ergodic, the `1/sqrt(5)`, `1/phi`, and `1/pi` injection deltas.

## The synthetic wall result

The first experiment is deliberately small and honest. The target is a product
interaction: a linear scorer on two features cannot separate
`sign(x0*x1) > 0` (an XOR of signs). The WBP self-similar product at lag 1 is
computed by the Jotnar listing, not by Rust.

```text
WBP synthetic wall: linear=0.583 wbp-product=1.000
```

That is the protocol's core claim in miniature: the wall is a basis problem, not
necessarily a model-capacity problem. The test lives in
`crates/jotnar-agent/tests/wall_breaking.rs`.

## The agent application

The current 46-state / 1275-transition table is linearly separable at
`discriminating top-1 = 1.000`, so there is **no residual wall to peel yet**.
That is the honest reading: the current model does not need WBP on the table it
has.

The next wall is the deeper table (depth 3 / repeated `depth + 1` and
`refinements + 1`), where the additive model can no longer represent the
interaction between a capability and a state. The planned loop is:

1. Fit the linear `StateEffectModel` on a held-out transition split (hold out
   cases or state groups, not random rows).
2. Compute the score residual on the held-out groups.
3. Run `wbp-audit` on the residual sequence.
4. If `rho` is low (structured), add the self-similar product feature at the
   peeled lag and refit. The product is computed by the Jotnar listing.
5. If `rho -> 1`, apply the `1/sqrt(5)` / `1/phi` / `1/pi` jumps. If they do not
   drop `rho`, the wall is ergodic; do not claim a linear feature.

The exact gate still decides adoption. The hard split stays audit-only.

## Limits

- The first slice uses AR(1)/AR(2), not the protocol's default order 5.
- Fixed-point integer division truncates; the `rho` thresholds are calibrated on
  the synthetic signals, not on a large benchmark.
- The self-similar feature is a product at one lag, not a full φ-clock basis.
  The baked clock table and projection are the next extension.
- No new opcodes were needed. If the clock projection later needs a primitive,
  it must meet the four-part opcode rule (interpreter, signature, C/non-FPU
  lowering, parity test).
