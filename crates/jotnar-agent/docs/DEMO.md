# The demo

One command runs the whole consolidated story: propose, exact gate, compose,
blind select, adopt, retain, and audit on held-out and hard streams.

```sh
cd "$(git rev-parse --show-toplevel)"
cargo run -q -p jotnar-agent -- demo --out runs/demo
```

It fits the state-dependent model from the generated training split, then runs
the learned beam on the generated and corpus streams, checks retention and
held-out non-regression, runs the learned-only ablation, and audits the adopted
harness on the separate hard set. It writes `report.md` and `summary.json` and
exits non-zero if any expectation fails.

## Expected result

```text
[1/5] quick loop: 5 cases, deterministic proposer
      base 3/5 -> adopted 5/5; held-out 3/5 -> 5/5; selected: compose: add compose[reduce_sum] + depth + 1
[2/5] fit the state-dependent model on the generated training split
      19 cases, 205 states, 7790 transitions, 1026 groups, top-1 1.000
[3/5] generated stream: learned beam with exact-gate adoption
      base 11/19 -> adopted 19/19; held-out 6/6; hard 13/13
[4/5] original 24-case acceptance corpus
      base 14/18 -> adopted 18/18; held-out 6/6
[5/5] learned-only ablation: no complete/singles/pairs safety nets
      adopted 19/19; held-out 6/6; hard 13/13

demo: quick 3/5 -> 5/5 (held-out 5/5)
demo: generated 11/19 -> 19/19 (held-out 6/6)
demo: corpus 14/18 -> 18/18 (held-out 6/6)
demo: hard audit 13/13; state model 1.000
demo: learned-only 19/19 (held-out 6/6, hard 13/13)
demo: report runs/demo/report.md
```

Runtime is about 35 seconds on the development machine; most of it is fitting
the 7,790-row depth-3 transition table.

For the focused compiler/backend application — a real optimizer pipeline over
integer kernels with C and non-FPU parity — see
[`BACKEND_SHOWCASE.md`](BACKEND_SHOWCASE.md).

## What the demo shows

- The exact interpreter gate is the only judge; every candidate gets a
  certificate.
- Two independent failures are solved by composing the top proposals, not by a
  single hand-picked edit.
- The quick stream includes a **correct refusal** for an impossible task; a
  refusal is a first-class certificate, not an error.
- The learned `(task, state, action)` model drives the deep/shaped cases and
  keeps the held-out stream clean.
- One geometric `reduce_sum` action transfers to the held-out `[3,3]`, `[4,4]`,
  and `[16]` shapes.
- The learned-only run (no completeness/singles/pairs safety nets) preserves
  the result.

## What it does not show

- It does not invent opcodes, families, or task features. The 10-action menu
  and the 39-feature task vector are human-authored.
- The hard set is 13 hand-authored audit cases, not an external benchmark.
- The deepest refine cases still use the coverage/neutral-drift tie-break.
- It is not an open-ended self-improving agent. The human still supplies the
  mechanisms, training cases, and feature schema.
- No hardware cost or performance claim: the cost head predicts bounded steps,
  not machine time.

## Artifacts

```text
runs/demo/
  report.md
  summary.json
  hard.json
  state-effect-model.weights.json
  state-effect-model.jotnar
  quick/
    round.json
    traces.jsonl
    certificates/{base,adopted}-{train,heldout}/
  generated/
    round.json
    traces.jsonl
    certificates/...
  corpus/
    ...
  learned-only/
    ...
```

`summary.json` contains the exact counts, selected labels, model fit metadata,
and the list of demo checks with an `all_ok` flag. The certificates are the
same schema used by the rest of the project: `status`, `outcome`, `verdict`,
`gate`, `witness`, `proof`.

## How to read the result honestly

The demo is a **certified harness tuner**. It is good at selecting and
composing edits from a fixed, geometry-first menu, and at proving the result
with an exact gate, retention, and held-out checks. The impressive-looking
numbers come from that narrow loop, not from open-ended capability discovery.
If you want to test the stronger claim, freeze the menu/features/corpora and
introduce a new held-out mechanism without adding a training case or feature;
that is the autonomy audit described in `HANDOFF.md`.
