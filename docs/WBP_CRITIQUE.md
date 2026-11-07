# WBP critique and future work

This is the honest assessment of the Wall-Breaking Protocol (WBP) after the
first Jotnar port. It is written for future WBP work, not for marketing.

## Summary

WBP is a useful **workbench**, not a theory of nonlinearity. Its strongest
contributions are diagnostic: distinguish a structured residual from an ergodic
one, try a basis before adding capacity, and calibrate symbolic results against
decoys. Its weakest parts are the number-theory ontology and the assumption that
the residual has a natural 1-D order.

The Jotnar port is a good substrate: deterministic integer listings, no hidden
float library, and a proposed basis that is itself an inspectable program. That
is the part worth keeping and extending.

## What works

1. **Residual as signal.** `resfrac`/`FractalPeeler` answers “is there structure
   here worth modeling?” before “which bigger model do I add?”. In the agent, it
   correctly showed that the deep `affine_s64` wall was not in the per-transition
   residual; it was in the proposal composition. That is a valuable negative
   result.
2. **Self-similar product basis.** `x_i * x_{i-lag}` is a principled way to add
   the interaction the residual indicates. The synthetic product wall
   (`linear=0.583 -> wbp-product=1.000`) is a real demonstration that the basis
   can break a product interaction.
3. **Calibration guardrail.** The 5× precision re-check, restricted basis,
   decoy test, and ratio threshold are the best scientific hygiene in the
   protocol. They should be mandatory, not optional.
4. **Deterministic Jotnar expression.** Fixed-point integer listings, no new
   opcodes, no runtime floats. The same signal gives the same `rho`, layers, and
   jump deltas. This is the right way to keep the protocol reviewable.

## What is weak or oversold

1. **It is not a general nonlinearity breaker.** AR peel finds linear
   predictability in a chosen order. A φ-clock projection is still linear. The
   only thing that truly breaks a product interaction is the product feature.
   The interesting question is whether the protocol *selects* the right product
   from the residual. Our first port used a hand-chosen lag (35); that is not
   yet a discovery result.
2. **The ordering is a hidden degree of freedom.** AR/resfrac assumes a
   sequence. The transition table is an unordered set of `(case, state, action)`
   rows. Imposing an order changes the diagnostic. On graphs or sets, the result
   is not permutation-invariant. This must be addressed before claiming general
   residual structure.
3. **The magic constants are priors, not laws.** `1/√5`, `1/φ`, `1/π`, ζ zeros,
   and prime impulses are probes. A drop in `rho` after injection does not prove
   the constant is meaningful. The protocol’s own decision log contains
   numerology failures (137/30, ζ = π/4). The honest version is a frequency
   sweep plus the calibration guardrail, with φ/√5/π as one prior among many.
4. **The ontology oversells the math.** “Holographic bound”, “Kerr horizon”,
   “frame dragging”, and “helicity” are mostly metaphor. The underlying
   diagnostics — two-piece breakpoint, sign correlation, multi-projection
   agreement — can stay, but the names should not imply a physical mechanism.
5. **It can miss walls outside the residual.** The deep-table wall was in the
   set-union composition, not the per-transition model. WBP on the residual
   said “nothing to peel” — correctly. A path/value residual diagnostic is
   needed for search-composition walls.
6. **The synthetic wall is close to tautological.** Adding `x0*x1` to a linear
   model solves an XOR-like target. That validates the basis, not the selection
   procedure. A stronger test is: hold out the interaction, run the protocol,
   and see whether it selects the right lag without being told.

## Future work, in priority order

1. **Port the calibration guardrail to Jotnar.** Precision re-check, restricted
   basis, decoy test, ratio. Make it a required suffix for any φ/ζ/prime/PSLQ
   result.
2. **Replace the magic constants with a frequency sweep.** Keep φ/√5/π as
   pre-registered priors, but compare against random frequencies and require the
   result to beat the sweep’s decoys.
3. **Make the residual ordering explicit and permutation-aware.** Support
   multiple orders, graph orders, and a permutation-invariant summary. Report
   whether the diagnosis survives the order change.
4. **Select lags by held-out gain, not by hand.** Enumerate candidate lags,
   fit on train, score on held-out transition groups, and report the selected
   lag plus the decoy distribution. No hand-picked lag 35.
5. **Extend `resfrac` beyond AR(1)/AR(2).** Order-k Levinson or a least-squares
   AR; axis-aware reductions; a scale/height sweep.
6. **Add a path/value residual diagnostic.** For search-composition walls like
   the repeated `depth + 1` case, diagnose the beam/path score residual, not
   only the per-transition residual.
7. **Build a real residual-wall benchmark.** A dataset where the nonlinearity
   is genuinely in the held-out labels, not injected by the feature we add.
   Report negative results as first-class outcomes.
8. **Test outside the agent.** Time series, tabular interactions, and a genuine
   black-box linear layer (PEP). If the protocol only works on our synthetic
   examples, say so.
9. **Keep the exact gate as the judge.** WBP is a proposer/diagnostic. It never
   grades itself and never overrides the gate.

## Scorecard for future WBP claims

A WBP result should report:

- the signal/table and its ordering;
- `rho` before/after each peel and jump;
- the candidate basis/lag and how it was selected;
- the held-out metric before/after, not the training metric;
- the decoy/calibration ratio;
- the exact-gate result, retention, and held-out non-regression;
- the negative result if it did not help;
- whether the wall was in the residual, in the search composition, or in the
  model capacity. These are different problems.

## The one-line lesson

Use WBP to decide **where** the wall is before deciding **what** to add. In the
deep-table case it said the wall was in proposal composition, and that was the
right answer.
