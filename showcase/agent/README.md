# Certified harness tuning (`jotnar-agent`)

> **Experimental research exhibit.** The language contract, if accepted, is
> the four integer primitives (`T2I`, `IGET`, `IMATMUL`, `IREDUCE_SUM`) and
> their documented boundaries. The agent, proposer menus, weights, corpora,
> demos, and the backend pipeline search are research, not part of the
> language contract. This is a certified harness/backend tuner, not an
> open-ended self-improving agent.


A bounded proposer edits a harness of composable Jotnar families and steps;
the exact interpreter gate is the only judge. The loop composes independent
fixes, keeps a per-case certificate, retains every incumbent-correct case, and
checks a held-out split before adoption.

```sh
cargo run -q -p jotnar-agent -- demo --out runs/demo
cargo run -q -p jotnar-agent -- backend-showcase --out runs/backend-showcase
```

## What it proves

- **Composition:** two independent failures are solved by composing the top
  proposals, not by one hand-picked edit.
- **Gate independence:** the proposer only ranks; the exact interpreter gate
  decides. A correctly impossible task produces a first-class refusal
  certificate.
- **Retention and held-out non-regression:** adoption requires every
  incumbent-correct case to stay correct and the frozen held-out split not to
  regress.
- **Backend tuning:** a real integer-IR pipeline (DCE, CSE, power-of-two
  strength reduction, `IMUL`+`IADD` -> `IMAD` fusion) is selected on training
  kernels and lowered through both C and non-FPU with exact parity.

Measured results:

```text
demo:       quick 3/5 -> 5/5; generated 19/19 + 6/6; corpus 18/18 + 6/6;
            hard audit 13/13; learned-only 19/19 + 6/6 + 13/13
backend:    selected cse -> dce -> strength -> fuse_imad;
            train ops 27 -> 20 (25.9%), weighted 58 -> 35 (39.7%);
            held-out ops 13 -> 11, weighted 33 -> 23;
            C parity 18/18, non-FPU parity 18/18; regressions 0
```

## What it does not prove

- It is not an open-ended self-improving agent. The action/pass/feature menus
  are fixed and human-authored.
- The hard and held-out sets are curated, not external benchmarks.
- Certificates are finite-sample exactness, not universal equivalence.
- The cost weights and emitted code-line counts are synthetic proxies, not
  hardware timing claims.
- The backend showcase uses an exhaustive bounded search, not the learned
  state model; the reusable contribution is the certified loop.

## Read more

- [`crates/jotnar-agent/README.md`](../../crates/jotnar-agent/README.md)
- [`docs/DEMO.md`](../../crates/jotnar-agent/docs/DEMO.md)
- [`docs/BACKEND_SHOWCASE.md`](../../crates/jotnar-agent/docs/BACKEND_SHOWCASE.md)
- [`docs/WORLD_MODEL.md`](../../crates/jotnar-agent/docs/WORLD_MODEL.md)
- [`docs/WBP_CRITIQUE.md`](../../crates/jotnar-agent/docs/WBP_CRITIQUE.md)
