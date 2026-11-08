# The 24-case corpus

Six capability families, two train / one calibration / one held-out case each.
The held-out split is by parameter and family, not by duplicate.

| family | train | calibration | held-out | capability |
| --- | --- | --- | --- | --- |
| elementwise | `saxpy→double`, `affine→s16` | `saxpy→oct` | `saxpy→s16` | compose depth + deep fix |
| reduction | `reduce8→32`, `reduce8→64` | `reduce16→64` | `reduce16→128` | fit `reduce(n)` |
| matrix | `matmul2→sum`, `matmul2→colsum` | `matmul2→sum` | `matmul2→colsum` | compose `sum4` |
| select | `select→max` (refuse), `select→min` (refuse) | `select→select` (identity) | `select→matmul` (refuse) | honest refusal / control |
| refine | `saxpy_dup→saxpy`, `affine_dup→affine` | `reduce8_dup→reduce8` | `matmul2_dup→matmul2` | duplicate removal |
| control | `saxpy→matmul` (refuse), `reduce8→matmul` (refuse) | `saxpy→saxpy` (identity) | `affine→matmul` (refuse) | no-op / refusal |

## Base profile

On all 24 cases, the incumbent harness is **18 correct, 6 wrong**:

- matrix: 4 wrong refusals (needs `sum4`)
- elementwise: 2 wrong refusals (needs the deep fix)
- reduction, select, refine, control: all correct

The 18-case selection stream (train + calibration) is **14 correct, 4 wrong**:
three matrix failures and one deep elementwise failure.

## Corpus run

```sh
cargo run -q -p jotnar-agent -- run --out runs/corpus \
  --proposer deterministic --corpus
```

Result:

```
selection: 14/18 correct, 4 wrong -> 18/18 correct, 0 wrong
heldout:   6/6 correct, 0 wrong
selected:  compose: add compose[sum4] + refinements + 1
```

The linear corpus prior and the looped 2-loop corpus prior both reach the same
adopted harness:

```sh
cargo run -q -p jotnar-agent -- run --out runs/corpus-linear \
  --proposer jotnar --corpus

cargo run -q -p jotnar-agent -- run --out runs/corpus-looped \
  --proposer looped --loops 2 --corpus
```

## Per-family features

`FEATURE_NAMES` now includes six wrong-refusal counts:

```
wrong_refusal_elementwise, wrong_refusal_reduction, wrong_refusal_matrix,
wrong_refusal_select, wrong_refusal_refine, wrong_refusal_control
```

The ordinary 5-case prior is deliberately left untouched for the Dory
experiments. The corpus uses `JotnarProposer::corpus_prior()` /
`LoopedProposer::corpus_prior(loops)`, which reads the matrix and elementwise
counts instead of letting the matrix decoy crowd out the deep fix.

## The ceiling this exposes

The first attempt with the ordinary prior selected only `sum4` on the 24-case
stream: the deep action was third or fourth because the aggregate decoy grew
with the matrix count. Per-family counts fix that for this corpus, but the
underlying limitation remains: **one aggregate feature vector per round cannot
say which case needs which action.**

The next real step is per-case task vectors:

- start inputs/outputs/ops
- target inputs/outputs/ops
- failure type and expectation
- tags and family

The proposer then scores actions per case; the driver unions/composes the top
actions. That is also where a Jotnar task encoder plugs into the loop.

## Honest boundaries

- 24 cases, with near-duplicate matrix and refine variants.
- The select family is expected-refusal/control, not a solved capability.
- The gate is the reference interpreter; C / non-FPU parity is still separate.
- The corpus exposes the aggregate-feature ceiling, not a general benchmark.
