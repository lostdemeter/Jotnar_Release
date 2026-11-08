# Bit-exact backend tuning showcase

This is the concrete compiler/backend application of the certified-harness
loop. It tunes a small optimizer pipeline over a corpus of integer Jotnar
kernels, checks every candidate against the reference interpreter, and lowers
the adopted pipeline through both the C and non-FPU backends for exact parity.

```sh
cd "$(git rev-parse --show-toplevel)"
cargo run -q -p jotnar-agent -- backend-showcase --out runs/backend-showcase
```

## What it runs

- **Corpus:** 10 training kernels, 5 held-out kernels, and 3 edge kernels.
  Kernels are ordinary Jotnar listings: affine/FIR-style integer arithmetic,
  duplicate expressions, dead temporaries, power-of-two multiplies,
  matmul+reduce, and select/min.
- **Pass menu:** DCE, CSE, strength reduction (`IMUL(x, 2^k)` -> `ISHL(x, k)`),
  and producer-consumer fusion (`IMUL` + `IADD` -> `IMAD`).
- **Search:** all ordered pass pipelines up to four stages are evaluated on the
  training split. Selection maximizes weighted-op savings, then instruction
  savings, then prefers the shorter pipeline. The held-out and edge kernels are
  never used for selection.
- **Exact gate:** every optimized listing is run through the reference
  interpreter on every supplied example and compared exactly.
- **Backend parity:** the adopted pipeline is emitted to C and to the
  integer-only non-FPU backend, compiled, run, and compared exactly. The
  lowering with fewer emitted code lines on the training split is adopted;
  both are reported.
- **Certificates:** one `jotnar.certificate` trust-spine object per kernel with
  reference/optimized hashes, proves/does-not-prove, assumptions, unknowns,
  falsifiers, nulls, bench labels, provenance, artifacts, and C/non-FPU parity.
  At least one is checked with `jotnar-lab verify` on every run.

## Expected output

```text
[1/4] selecting a pipeline on the training split
      selected: cse -> dce -> strength -> fuse_imad (train weighted 58 -> 35, ops 27 -> 20)
[2/4] exact gate + C/non-FPU parity
      emitted code lines: c=521, nonfpu=485; adopted nonfpu
[3/4] writing certificates and reports
[4/4] wrote runs/backend-showcase/report.md
```

Report summary:

```text
| split   | kernels | ops before -> after | weighted before -> after | multiply-class |
| ------- | ------: | ------------------: | -----------------------: | -------------: |
| train   |      10 |        27 -> 20 (25.9%) |          58 -> 35 (39.7%) |         11 -> 3 |
| heldout |       5 |        13 -> 11 (15.4%) |          33 -> 23 (30.3%) |          6 -> 2 |
| edge    |       3 |         5 -> 3 (40.0%)  |           7 -> 5 (28.6%)  |          1 -> 1 |

Exact gate: all kernels pass. Backend parity: C 18/18, non-FPU 18/18.
Regressions: 0.
```

## Artifacts

```text
runs/backend-showcase/
  report.md
  summary.json
  certificates/{train,heldout,edge}/*.json   # jotnar.certificate objects
  optimized/{train,heldout,edge}/*.asm
  backend/                  # emitted C sources and binaries for both targets
```

A certificate is a real trust-spine object, not a private parity record:

```json
{
  "schema": "jotnar.certificate",
  "id": "agent/backend/train/train-scale2",
  "status": "current",
  "outcome": "solved",
  "verdict": "pass",
  "gate": {
    "name": "bit_exact_interpreter_c_nonfpu",
    "can_say_no": true
  },
  "proves": [
    "the optimized listing is bit-exact on 2 example(s) for the interpreter, C, and non-FPU targets"
  ],
  "does_not_prove": [
    "the function on inputs outside the supplied examples",
    "universal equivalence for every possible program"
  ],
  "falsifiers": [
    {
      "corruption": "change one optimized output byte",
      "expected": "fail",
      "actual": "fail",
      "caught_by": "bit_exact_interpreter_c_nonfpu",
      "not_caught": false
    }
  ],
  "nulls": [
    {
      "name": "identity pipeline",
      "control": "baseline listing with no passes",
      "expected": "pass",
      "actual": "pass",
      "effect": "none",
      "status": "tested"
    }
  ],
  "bench": {
    "clock": "not_measured",
    "scope": "backend_pipeline",
    "kind": "not_measured"
  },
  "artifacts": [
    { "role": "optimized_listing", "path": "../../optimized/train/train-scale2.asm" }
  ]
}
```

Verify one with the trust-spine verifier:

```sh
cargo run -q -p jotnar-lab -- verify \
  runs/backend-showcase/certificates/train/train-scale2.json
```

## Why this is a real fit

- The reference interpreter is a **golden model**, and the C/non-FPU emitters
  are the backends. That is the same structure a compiler team already uses for
  differential testing.
- The passes are ordinary IR rewrites, not per-kernel answers.
- The system **selects and composes** the pipeline; the exact gate decides.
- The held-out split measures whether a pipeline selected on one set of kernels
  generalizes to a different set.
- Every accepted kernel result has a trust-spine certificate, and the
  no-regression count is explicit.

## How it relates to the learned agent

This showcase uses an exhaustive bounded search over 65 ordered pipelines, not
the learned state-effect model. At this scale exhaustive search is the stronger
choice. The reusable contribution is the certified loop around it: exact
interpreter gate, held-out evaluation, C/non-FPU parity, per-program
certificates, and no-regression accounting. The learned model is for larger
action spaces where exhaustive enumeration is not practical.

## What it does not show

- It is not a general compiler optimizer. The pass menu and corpus are fixed
  and human-authored; layout-changing and loop-level passes are future work.
- Certificates are finite-sample exactness, not universal equivalence.
- The weighted cost table and emitted code-line count are synthetic proxies,
  not hardware timing claims.
- The held-out/edge split is a curated sibling split, not an external compiler
  benchmark.

## Adapting it to another backend

Replace the `corpus()` listings and the `Pass` menu in
`crates/jotnar-agent/src/backend_showcase.rs`. Keep the shape of the loop:

1. reference interpreter outputs per kernel,
2. candidate pipeline applied to the IR,
3. exact gate on the training examples,
4. held-out and edge evaluation after selection,
5. C/non-FPU (or another target) parity for the adopted pipeline,
6. one certificate per kernel.

The key constraint is the same as the rest of the project: the proposer
searches, but the exact gate is the only judge.
