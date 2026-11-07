# The action-effect world model

The teacher can choose actions; the world model learns what actions **do**.
There are two models now:

1. the original `(task, action) -> outcome` linear effect model;
2. the state-dependent `(task, harness_state, action) -> outcome` model, with a
   beam/triple proposer on top.

Both are ordinary Jotnar listings. Neither grades itself: the exact
interpreter gate produces every label and chooses every adopted harness.

## The first model

`EffectModel` is a Jotnar-native linear map:

```text
20 task features -> 11 action scores
```

It is trained on the exact effect table from `gen-effects`:

- for each case, actions that keep or make the case correct are positives;
- actions that break it are negatives;
- a per-case ranking perceptron updates the action rows.

```sh
cargo run -q -p jotnar-agent -- gen-effects \
  --out runs/action-effects.jsonl

cargo run -q -p jotnar-agent -- fit-effect-model \
  --out runs/effect-model --generated --epochs 10 --lr 1
```

Generated result:

```text
effect model: 17 cases, 10 actions, epochs=10, lr=1, train top-1 accuracy=1.000
```

`WorldModelProposer` ranks single actions and proposes every action **pair**, so
the exact gate can choose a composition:

```sh
cargo run -q -p jotnar-agent -- run --out runs/world \
  --proposer world --generated \
  --effect-weights runs/effect-model/effect-model.weights.json
```

```text
selection: 11/18 correct, 7 wrong -> 18/18 correct, 0 wrong
heldout:   6/6 correct, 0 wrong
selected:  world: add compose[reduce_sum] + refinements + 1 + add family select_min
adopted:   true
```

The original 24-case corpus with the same generated-trained model:

```text
selection: 14/18 correct, 4 wrong -> 18/18 correct, 0 wrong
heldout:   6/6 correct, 0 wrong
selected:  world: add compose[reduce_sum] + depth + 1
adopted:   true
```

## The state-dependent model

The first model cannot tell that `reduce_sum` is useless once the consumer is already
present, or that after a composition the remaining fix is `depth + 1`. The
state-dependent model takes the harness with the task:

```text
32 geometric task features + 14 harness-state features -> 10 action scores
```

State features (`Harness::state_features`): `has_reduce_sum`, `has_select_min`,
`has_select_max`, the three duplicate families, family/consumer counts, depth,
refinements, refine/tag flags, the depth cap, and `has_add2`.

The consumer is one geometric action: `add compose[reduce_sum]`. Its
applicability is a shape contract, not an arity name: the start must have one
output stream with a non-empty shape, and the target must be a scalar. The task
vector carries `start_rank`, `start_numel`, `target_rank`, `target_numel`,
`start_is_field`, `target_is_scalar`, `shape_match`, `reduce_applicable`, and
the start op family (`start_imatmul`, `start_iselect`, `start_iadd`,
`start_ireduce_sum`). The model learns from those features; `action_applies`
only enforces the contract. There is no `sum4`/`sum9`/`sum16` action.

`gen-transitions` builds the causal transition table with the exact gate. It
enumerates every distinct harness reachable with up to `--depth` actions
(default 2), evaluates every state on every training case, and records
`(case, state, action) -> child outcome`.

```sh
cargo run -q -p jotnar-agent -- gen-transitions \
  --out runs/transitions.jsonl --depth 2

cargo run -q -p jotnar-agent -- fit-state-effect-model \
  --out runs/state-effect-model --generated \
  --depth 2 --epochs 10 --lr 1
```

```text
state effect model: 18 cases, 205 states, 7380 transitions, 10 actions,
depth=2, epochs=10, lr=1, discriminating top-1 accuracy=1.000
```

The listing is `IN t0..t19, u0..u13 -> s0..s9`. `fit` is a per-`(case, state)`
ranking perceptron: correct children are positives, wrong children are
negatives, and breaking an already-correct state is weighted double. The
`discriminating` accuracy counts only groups that actually contain a choice,
not the many already-solved groups where every action is a safe no-op.

## Beam and triple composition

`BeamProposer` is the compositional planner:

1. rank the single actions from the base state;
2. union the top action per wrong case (the first candidate, so it wins exact
   ties);
3. expand the top-`beam` states, rank by the state model's predicted effect,
   and keep the best paths (pairs and triples);
4. include the exhaustive pair set as a cheap safety net;
5. hand every candidate to the exact gate.

```sh
cargo run -q -p jotnar-agent -- run --out runs/beam \
  --proposer beam --generated \
  --state-effect-weights runs/state-effect-model/state-effect-model.weights.json \
  --beam 4 --depth 2
```

```text
selection: 11/18 correct, 7 wrong -> 18/18 correct, 0 wrong
heldout:   6/6 correct, 0 wrong
selected:  beam-coverage: add compose[reduce_sum] + depth + 1 + refinements + 1
           + add family select_min
adopted:   true
```

The coverage candidate keeps the second capability alive: the top union is
`reduce_sum + refinements + select_min`, but the deep case also supports `depth + 1`,
and the coverage union adds it. The geometric `reduce_sum` action is learned
from the shaped matrix/vector training cases, and the exact gate accepts the
four-capability harness at the same steps.

Depth-3 training (`--depth 3`) and `--depth 3` beam expansion produce the
explicit triple `compose[reduce_sum] + depth + 1 + select_min` with the same
selection and held-out results.

The 24-case corpus:

```text
selection: 14/18 correct, 4 wrong -> 18/18 correct, 0 wrong
heldout:   6/6 correct, 0 wrong
selected:  beam-complete: add compose[reduce_sum] + depth + 1 + refinements + 1
adopted:   true
```

## Closing the generated held-out miss

The first stream was 5/6: `gen-elem-affine-s16-holdout` was a deep failure with
no deep failure in training. The effect model never learned that a deeper
target needs `depth + 1`.

The fix is a **training-stream** member, not a held-out change:

```rust
make("gen-elem-saxpy-s16-train", "saxpy", "saxpy_s16", ..., Train)
```

The held-out row is still `gen-elem-affine-s16-holdout` (`affine -> affine_s16`)
and is never trained on. The new row is a different start family and example
set, so the model sees the deep mechanism and generalizes to the affine target.
The generated stream is now 11 train / 7 calibration / 6 held-out = 24 cases.

Result: generated `beam` and `joint` reach **18/18 selection, 6/6 held-out**;
the legacy single-action `world` model excludes the repeated-depth `s64` case.

## The joint latent + causal model

`JointProposer` keeps the Qwen-style backbone's latent and the causal state
model:

```text
score(action) = 4 * effect(task, state, action)
              + 1 * <latent(task), prototype(action)>
```

The prototypes are the mean latent of the cases each action fixes, fitted from
the exact transition table. The spine block is the Qwen φ-ladder backbone from
`fit-backbone --qwen`; the latent is the identity channel, the effect model is
the causal channel.

```sh
cargo run -q -p jotnar-agent -- fit-backbone \
  --out runs/backbone-qwen --generated --qwen --dim 10 --inter 16

cargo run -q -p jotnar-agent -- fit-joint-model \
  --out runs/joint-model --generated \
  --state-effect-weights runs/state-effect-model/state-effect-model.weights.json \
  --backbone runs/backbone-qwen/backbone.weights.json \
  --depth 2 --beam 4

cargo run -q -p jotnar-agent -- run --out runs/joint \
  --proposer joint --generated \
  --joint-weights runs/joint-model/joint-model.weights.json
```

```text
selection: 11/18 correct, 7 wrong -> 18/18 correct, 0 wrong
heldout:   6/6 correct, 0 wrong
selected:  joint-coverage: add compose[reduce_sum] + depth + 1 + refinements + 1
           + add family select_min
adopted:   true
```

Corpus: 18/18 selection, 6/6 held-out. Hard audit: 12/12.

## The quality-diversity archive

The driver adopts one harness at a time. The archive keeps several gate-correct
attractors, selected by **quality** (exact correctness, then held-out) and
**novelty** (L1 distance in the 14-feature harness-state descriptor):

```sh
cargo run -q -p jotnar-agent -- run --out runs/beam-archive \
  --proposer beam --generated \
  --state-effect-weights runs/state-effect-model/state-effect-model.weights.json \
  --archive runs/archive.json --archive-novelty 1 --archive-capacity 16

cargo run -q -p jotnar-agent -- archive --file runs/archive.json
```

The first archive run kept 16 entries: the 18/18 + 6/6 path-union harness,
the 14/18 pair attractors (each missing a different capability), and
the novel near-misses.
That is the point: `reduce_sum + refinements` and `reduce_sum + select_min` are different
basins, even when both are one case short. The archive is analysis/proposal
state; adoption still requires the gate, retention, and held-out
non-regression.

## The generalization audit

The first held-out split was a **sibling** split, not a feature-held-out split.
`feature-audit` makes that measurable:

```sh
cargo run -q -p jotnar-agent -- feature-audit --generated
```

Min L1 distance from each held-out case to the training task vectors:

```text
gen-matmul-sum-holdout          0
gen-select-min-holdout          0
gen-elem-affine-s16-holdout     2
gen-refine-affine-oct-holdout   2
gen-refuse-saxpy-select-holdout 3
gen-reduce-16-32-holdout       67
```

Four of six held-out cases are within three feature units of training, and two
are exact duplicates. The held-out split is still useful for example-level
retention, but it is not evidence of feature generalization.

The audit adds a separate **hard** set, `hard_corpus()`, with a `Hard` split
that is never used for fitting, selection, or adoption:

- four novel `depth + refinements` compositions (`saxpy_dup -> saxpy_s16`,
  `affine_dup -> affine_s16`, and two cross-start variants). Their minimum
  solver is `depth + refinements`, a combination no training case requires;
- three high-distance retention cases (`reduce128 -> reduce8` distance 493,
  `reduce32 -> reduce64` distance 193, `matmul2_sum -> matmul2_colsum` 15);
- three **feature-distant** compose cases across two mechanisms:

  | case | shape | min-dist | minimum solver |
  | --- | --- | ---: | --- |
  | `matmul3 -> matmul3_sum` | `[3,3]` field -> scalar | `add compose[reduce_sum]` |
  | `matmul4 -> matmul4_sum` | `[4,4]` field -> scalar | `add compose[reduce_sum]` |
  | `select16 -> select16_sum` | `[16]` select field -> scalar | `add compose[reduce_sum]` |

- one high-distance refusal control (`reduce128 -> matmul2`, distance 493).

The consumer is taught by `gen-vec9-sum-train` and `gen-vec16-sum-train`
(shaped vectors that reduce to a scalar) and by the shaped `matmul2` training
cases. The audit shapes are different rank/size/mechanism signatures. Nothing
in the hard split is trained on.

```sh
cargo run -q -p jotnar-agent -- feature-audit --generated \
  --state-effect-weights runs/state-effect-model/state-effect-model.weights.json \
  --top-k 3

cargo run -q -p jotnar-agent -- run --out runs/beam \
  --proposer beam --generated \
  --state-effect-weights runs/state-effect-model/state-effect-model.weights.json \
  --hard
```

The learned top-1 on each distant shape is the taught consumer:

```text
hard-matmul3-sum    learned top-1 add compose[reduce_sum] (240)
hard-matmul4-sum    learned top-1 add compose[reduce_sum] (296)
hard-select16-sum   learned top-1 add compose[reduce_sum] (296)
```

`--hard` only audits the final adopted harness:

```text
hard audit: 12/12 correct, 0 wrong (never used for selection or adoption)
```

Progression:

```text
base harness:       4/11  (retention + refusal control only)
learned union:     12/12  (reduce_sum from the shape features)
learned-only:      12/12  (no completeness/singles/pairs safety net)
```

Before the coverage candidate the run solved only the retention/control cases
and failed all four refine/deep compositions plus the three distant compose
cases. The gap was real and measurable, not hidden by the sibling held-out.

## Capability coverage / neutral drift

The top union is `reduce_sum + refinements + select_min`; it is 18/18
on the selection stream, but it does not carry `depth + 1`. The deep case
supports both `refinements` and `depth`, so the proposer adds a **coverage
union**: the top action from each capability group (compose, depth, refinements,
select) whose predicted score is positive. The driver's selection then uses
capability coverage as a **tie-break after correctness and step count**:

```text
(correct, wrong, steps, capability_coverage, candidate index)
```

The coverage harness has the same 18/18 and step count as the union, so the
tie-break selects it. That is neutral drift: the gate cannot tell the two apart
on the selection stream, so the loop keeps the harness that preserves the second
capability.

The **compose-completeness** candidate is still available: it adds every
configured consumer for free (a mismatched consumer is a no-op). But it is no
longer what solves the distant cases. `reduce_sum` is the learned per-case top
action for every shaped field-to-scalar case. The exact gate still verifies the
result; the completeness net is a fallback, not the explanation.

The reduction is shape-bound, not arity-bound. `action_applies` requires a
single shaped start output and a scalar target. The fit only compares actions
the transition table offers from a state, and the proposer filters
inapplicable actions. The model sees the shape/op-family features and learns
when the reduction is the right edit; it is not handed a per-arity table.

## Learned vs safety-net ablation

`run` can disable the structural safety nets:

```sh
cargo run -q -p jotnar-agent -- run --out runs/beam-learned \
  --proposer beam --generated \
  --state-effect-weights runs/state-effect-model/state-effect-model.weights.json \
  --no-complete --no-singles --no-pairs --hard
```

Both runs select the same four-capability harness:

```text
with safety nets:   18/18 selection, 6/6 held-out, hard 12/12
learned-only:       18/18 selection, 6/6 held-out, hard 12/12
```

The ablation is the scientific result: the distant compose cases are reached by
the learned per-case ranking, not by the complete/singles safety net. The four
refine/deep hard cases still use the coverage union's structural deep prior to
keep `depth + 1` beside `refinements + 1`; that prior is a separate, pre-existing
coverage heuristic, and the learned model ranks `refinements + 1` first there.

## Cost-aware planning

`CostModel` is a Jotnar-native cost head over the same `(task, state)`
features. It predicts bounded steps, is fitted by ridge regression on the
**correct** transition rows, and lowers to a listing with `IDIV` by
`COST_SCALE`:

```sh
cargo run -q -p jotnar-agent -- fit-cost-model \
  --out runs/cost-model --generated --depth 2 --lambda 1

cargo run -q -p jotnar-agent -- run --out runs/beam-cost \
  --proposer beam --generated \
  --state-effect-weights runs/state-effect-model/state-effect-model.weights.json \
  --cost-weights runs/cost-model/cost-model.weights.json --cost-weight 4
```

```text
cost model: 18 cases, 7380 transitions, depth=3, lambda=1, mae=0.482
```

The beam score is `effect - cost_weight * predicted_steps`. The exact gate and
the driver's actual-step tie-break still decide; the cost head only shapes
which candidates the beam proposes. On this small stream the gate already finds
the cheapest correct harness, so the cost head's value is in larger search
spaces, not in changing the current winner.

## Basin hopping from the archive

`ArchiveProposer` seeds the loop with the best archive entries and proposes
their bounded mutations (including the neutral-drift toggles):

```sh
cargo run -q -p jotnar-agent -- run --out runs/qd \
  --proposer qd --generated --archive runs/archive.json --qd-seeds 4 --hard
```

The first run with the geometric consumer started from the 16-entry archive and
selected a hop from the `beam-union` seed adding `depth + 1`:

```text
selected:  qd-hop[beam-union: ...]: depth + 1
adopted:   true
hard audit: 12/12 correct, 0 wrong
```

That is the archive doing real work: the loop improves by starting from a
previously found basin and adding the missing capability, not by collapsing
back to the incumbent.

## Why it works

- The transition table gives **state-dependent causal labels**, not action
  imitation.
- Beam search ranks compositions by predicted effect; the exact gate verifies.
- The deep training member supplies the missing mechanism, not the held-out
  answer.
- The joint model separates identity (latent) from causality (effect).
- The archive prevents the loop from collapsing back to one incumbent.
- In every path the model **proposes**; the exact gate **judges**.

## Known limits

- State enumeration is over distinct action subsets; repeated `depth + 1` is
  represented by one action.
- The transition table grows with `--depth` (205 states / 7380 transitions at
  depth 3 with 10 actions, including repeated depth/refinements).
- The effect model is linear. The state split helps, but the joint interaction
  is still additive.
- The Qwen backbone is an exact zero-init/pass-through block on the current
  slice; the joint model's latent channel is the named latent, not a
  backprop-trained attention representation.
- The archive records gate-correct/near-correct candidates and now seeds basin
  hopping, but it is not yet a full quality-diversity search with continuous
  descriptors.
- The hard set is twelve audit cases, not a benchmark. Its four refine/deep
  compositions share one mechanism (`*_dup -> *_s16`) and their count distance
  is small. The three distant compose cases are distinct in rank, numel, and op
  family (`[3,3]` matmul, `[4,4]` matmul, `[16]` select).
- One `reduce_sum` action handles every field-to-scalar reduction. It is taught
  on shaped matrix/vector cases and held out on larger/different shapes. It is
  not a general axis-reduction language yet; arbitrary `reduce(axis)` is future
  work.
- The shape contract (`start_is_field && target_is_scalar`) is structural. It
  tells the scorer which edits are meaningful; the learned fit still has to
  rank `reduce_sum` above the other applicable actions using the shape and
  op-family features.
- The four refine/deep hard cases still use the coverage union's structural
  deep prior to keep `depth + 1` beside the learned `refinements + 1`.
- The coverage/completeness tie-breaks are heuristics. They are safe because
  they run after exact correctness and actual steps, and the held-out check
  still gates adoption, but they can keep a capability that is irrelevant to
  the current stream.
- The cost head is soft. On this tiny stream the exact gate's step tie-break
  already selects the cheapest correct harness; the cost head matters when the
  search space is larger.
