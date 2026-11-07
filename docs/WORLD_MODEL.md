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
20 task features -> 10 action scores
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
effect model: 15 cases, 10 actions, epochs=10, lr=1, train top-1 accuracy=1.000
```

`WorldModelProposer` ranks single actions and proposes every action **pair**, so
the exact gate can choose a composition:

```sh
cargo run -q -p jotnar-agent -- run --out runs/world \
  --proposer world --generated \
  --effect-weights runs/effect-model/effect-model.weights.json
```

```text
selection: 11/15 correct, 4 wrong -> 15/15 correct, 0 wrong
heldout:   6/6 correct, 0 wrong
selected:  world: add compose[sum4] + depth + 1 + add family select_min
adopted:   true
```

The original 24-case corpus with the same generated-trained model:

```text
selection: 14/18 correct, 4 wrong -> 18/18 correct, 0 wrong
heldout:   6/6 correct, 0 wrong
selected:  world: add compose[sum4] + depth + 1
adopted:   true
```

## The state-dependent model

The first model cannot tell that `sum4` is useless once the consumer is already
present, or that after a composition the remaining fix is `depth + 1`. The
state-dependent model takes the harness with the task:

```text
20 task features + 14 harness-state features -> 10 action scores
```

State features (`Harness::state_features`): `has_sum4`, `has_select_min`,
`has_select_max`, the three duplicate families, family/consumer counts, depth,
refinements, refine/tag flags, the depth cap, and `has_add2`.

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
state effect model: 15 cases, 46 states, 615 transitions, 10 actions,
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
selection: 11/15 correct, 4 wrong -> 15/15 correct, 0 wrong
heldout:   6/6 correct, 0 wrong
selected:  beam-union: add compose[sum4] + refinements + 1 + add family select_min
adopted:   true
```

Depth-3 training (`--depth 3`, 130 states, 1755 transitions) and `--depth 3`
beam expansion produce the explicit triple
`compose[sum4] + depth + 1 + select_min` with the same 15/15 and 6/6.

The 24-case corpus:

```text
selection: 14/18 correct, 4 wrong -> 18/18 correct, 0 wrong
heldout:   6/6 correct, 0 wrong
selected:  beam-union: add compose[sum4] + refinements + 1
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
The generated stream is now 8 train / 7 calibration / 6 held-out = 21 cases.

Result: generated `world`, `beam`, and `joint` all reach **15/15 selection,
6/6 held-out**.

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
selection: 11/15 correct, 4 wrong -> 15/15 correct, 0 wrong
heldout:   6/6 correct, 0 wrong
selected:  joint-union: add compose[sum4] + refinements + 1 + add family select_min
adopted:   true
```

Corpus: 18/18 selection, 6/6 held-out.

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

The first archive run kept 16 entries: the 15/15 + 6/6 triple, the 14/15 pair
attractors (each missing a different capability), and the novel near-misses.
That is the point: `sum4 + refinements` and `sum4 + select_min` are different
basins, even when both are one case short. The archive is analysis/proposal
state; adoption still requires the gate, retention, and held-out
non-regression.

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
- The transition table grows with `--depth` (46 states at depth 2, 130 at
  depth 3); depth 4 is not yet bounded.
- The effect model is linear. The state split helps, but the joint interaction
  is still additive.
- The Qwen backbone is an exact zero-init/pass-through block on the current
  slice; the joint model's latent channel is the named latent, not a
  backprop-trained attention representation.
- The archive records gate-correct/near-correct candidates; it is not yet a
  basin-hopping proposer.
