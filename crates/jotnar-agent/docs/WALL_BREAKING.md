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

## The deeper-table result

We then built the deeper, repeated-action table:

- `reachable_states` is a bounded BFS over action sequences, not distinct
  subsets; `depth + 1` and `refinements + 1` repeat up to their caps.
- depth 3: **205 states / 7790 transitions / 10 actions**, still
  `discriminating top-1 = 1.000` for the plain linear model.
- `affine_s64` is the wall: a 64x deep chain that needs three `depth + 1`
  steps. The default depth-2 harness misses it; the exact gate solves it at 6
  steps.

The wall turned out to be in **proposal composition**, not per-transition
linearity. The per-transition model was already 1.000; the set-union candidate
collapsed the repeated `depth + 1` actions to one. The fix is a learned
**path-union** candidate: union the top beam path per wrong case and preserve
repeated depth/refinement counts. Result:

```text
generated 19/19 selection, 6/6 held-out
hard      13/13 (affine_s64 at 6 steps; held-out pair -> select_min)
learned-only the same
```

WBP's self-similar interaction lag 35 (`refinements * depth_delta`) is
available on `StateEffectModel` (`with_interactions(vec![35])`), assembles as a
Jotnar listing, and is neutral on this wall. That is the honest result: the
wall was in the search aggregation, not in the residual of the per-transition
model. WBP remains the tool for a wall that actually lives in the residual.

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

For the honest assessment and the future-work checklist, see
[`WBP_CRITIQUE.md`](WBP_CRITIQUE.md).
