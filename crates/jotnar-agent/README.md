# jotnar-agent

A listing-level autonomous circuit for Jotnar.

> New session? Read
> [`HANDOFF.md`](../../../harness_selfimprove/HANDOFF.md) for the current
> state, artifacts, known gaps, and next-session plan, then
> [`NEXT_SESSION_PROMPT.md`](../../../harness_selfimprove/NEXT_SESSION_PROMPT.md)
> for the self-contained next prompt.

This crate applies the *shape* of Harness²'s optimizer loop to the Rust Jotnar
release, without importing Harness²'s benchmarks or runtime:

```
propose -> probe -> compose -> blind select -> reflect -> adopt
```

What changes is the **harness** (the family set, the bounded step set, the
search depth, the refinement policy, and the experience memory). What stays
fixed is the **substrate**: candidate programs are ordinary Jotnar listings,
resolved by the reference interpreter and certified by `jotnar-lab`.

## The circuit

```
CircuitIntent -> specify_or_solve -> plan -> exact gate -> certificate
                        ^                                  |
                        +------ bounded refinement --------+
```

A state is an assembled `jotnar::BoundProgram` plus provenance
(`family`, `params`, `kind`). The bounded steps are:

- `fit` — instantiate a family from the target's parameters
- `compose` — bind the current output into a consumer (`add2`, `reduce_sum`)
- `refine` — conservative CSE, accepted only if outputs stay exact

The gate runs both the candidate and the target on the supplied examples and
compares every output stream exactly. A refusal is a first-class result: the
carrier is a `jotnar-lab` certificate with `outcome=refused`, `verdict=bottom`,
and a typed refusal object.

## The proposer

The first proposer is itself a Jotnar listing. It is a tiny linear scoring
unit:

- inputs: 15 evidence features (`wrong_refusal`, `wrong_refusal_4out`,
  `wrong_refusal_deep`, `correct`, `steps`, `has_reduce_sum`, `depth`,
  `refinements`, …)
- outputs: one integer score per bounded harness action
  (`add compose[reduce_sum]`, `depth + 1`, `refine off`, `tags off`,
  `add family select_max`, `refinements + 1`)
- the driver reads the top applicable scores and probes those actions

The weights are a prior, not a trained model. The listing is written to
`proposer.jotnar` and its feature/score trace to `proposer.json` beside the
run. `--proposer deterministic` keeps the earlier bounded mutation order as a
control.

This is the first concrete "Jotnar LLM for Jotnar improvement" unit: the
proposer is an ordinary Jotnar program, gated by the same exact gate as every
candidate.

## From linear to looped

The next proposer is a tiny **looped latent-recurrent** model in the spirit of
NVIDIA Dory: an input block, a shared recurrent block applied `N` times with
the input reinjected every loop, and an output block. The recurrent block uses
a residual + ReLU + clip, because a purely linear loop would only reparameterize
the linear model. `N` is the elastic compute knob.

```sh
cargo run -q -p jotnar-agent -- run --out runs/agent --proposer looped --loops 2
cargo run -q -p jotnar-agent -- run --out runs/agent --proposer looped --loops 4 --fit
cargo run -q -p jotnar-agent -- fit-recurrent --out runs/fit --deep-train --efficient
cargo run -q -p jotnar-agent -- run --out runs/agent --proposer looped --loops 2 \
  --weights runs/fit/looped.weights.json
```

The output block is fitted with `--fit`. The shared recurrent block can be
grid-fitted with `fit-recurrent`; `--weights` loads the result. Two levers make
the recurrent fit preserve the held-out threshold:

- `--deep-train` adds a *different* deep case (`affine → affine_s16`) to
  calibration, while `refine-s16` stays held out;
- `--efficient` prefers coverage, then retention at the largest loop budget,
  then early loop coverage, then margin.

Together they select `(diag=5, cross=-4)`, which is miss/hit/hit on the
held-out case across 1/2/4 loops. Either lever alone shifts the threshold to
four loops or ignores the deep case. See [`docs/DORY.md`](docs/DORY.md) for the
2×2 experiment.

The headline experiment: fit on four cases, hold out the deep `refine-s16`
case, then evaluate on all five. The fit reinforces only `reduce_sum`; the deep fix
is not taught. **One loop still picks the decoy** (`tags off`) as its second
proposal; **two and four loops pick `refinements + 1`** and adopt the composed
fix. That is a generalization effect from the repeated block, not memorization
of the calibration set.

```sh
cargo run -q -p jotnar-agent -- run --out runs/agent --proposer looped --loops 1
cargo run -q -p jotnar-agent -- run --out runs/agent --proposer looped --loops 2
```

## The 24-case corpus

`--corpus` runs a stratified 24-case stream: six capability families, two
train / one calibration / one held-out case each. Held-out is by parameter and
family, not duplicate.

```sh
cargo run -q -p jotnar-agent -- run --out runs/corpus \
  --proposer deterministic --corpus
```

```
selection: 14/18 correct, 4 wrong -> 18/18 correct, 0 wrong
heldout:   6/6 correct, 0 wrong
selected:  compose: add compose[reduce_sum] + refinements + 1
```

The corpus adds six per-family wrong-refusal features. The ordinary 5-case
prior is kept for the Dory experiments; the corpus uses
`JotnarProposer::corpus_prior()` / `LoopedProposer::corpus_prior(loops)`.
See [`docs/CORPUS.md`](docs/CORPUS.md) for the case table, splits, and the
aggregate-feature ceiling.

## The task encoder

The aggregate proposer cannot say *which* case needs *which* action. The task
encoder works per case:

```
CaseView -> 20-dim task vector -> 9 named latent channels -> 6 action scores
```

The latent channels are legible (`start_outputs`, `depth_delta`,
`wrong_refusal`, `start_is_four`, …), and the encoder is an ordinary Jotnar
listing. For every wrong refusal it selects the top applicable action, unions
the selections across cases, and hands the union to the exact gate.

```sh
cargo run -q -p jotnar-agent -- run --out runs/corpus \
  --proposer task --corpus
```

```
selection: 14/18 correct, 4 wrong -> 18/18 correct, 0 wrong
heldout:   6/6 correct, 0 wrong
selected:  case-union: add compose[reduce_sum] + refinements + 1
```

On the 5-case stream it also adopts 5/5. The matrix case and the deep case are
different task vectors, so they select different actions.

The selection matrix can be fitted with a per-case ranking perceptron:

```sh
cargo run -q -p jotnar-agent -- fit-encoder --out runs/encoder --corpus
cargo run -q -p jotnar-agent -- run --out runs/fitted --proposer task \
  --corpus --encoder-weights runs/encoder/task-encoder.weights.json
```

The fitted encoder reaches 18/18 selection and 6/6 held-out. It can then be
looped:

```sh
cargo run -q -p jotnar-agent -- run --out runs/one-loop --proposer task \
  --corpus --encoder-loops 1

cargo run -q -p jotnar-agent -- run --out runs/two-loops --proposer task \
  --corpus --encoder-loops 2
```

`LoopedTaskEncoder` reuses the input/output blocks and applies a shared
recurrent block with the task vector reinjected every loop.

The prior is deliberately loop-sensitive in **two families**:

- matrix: the linear/1-loop encoder picks the stable `target_outputs_1`
  decoy; `start_is_four` grows and `reduce_sum` takes over at two loops.
- deep elementwise: the same decoy wins at one loop; `start_is_one` grows and
  a deep action takes over at two loops.

| mode | selection | held-out |
| --- | ---: | ---: |
| linear | 14/18 | 4/6 |
| 1 loop | 14/18 | 4/6 |
| 2 loops | **18/18** | **6/6** |
| 3 loops | **18/18** | **6/6** |

The recurrent block can be grid-fitted:

```sh
cargo run -q -p jotnar-agent -- fit-looped-encoder \
  --out runs/encoder --corpus --max-loops 3

cargo run -q -p jotnar-agent -- run --out runs/fitted-loop --proposer task \
  --corpus --looped-encoder-weights runs/encoder/looped-encoder.weights.json
```

The fit now includes a **loop-cost objective**: reject any config that already
covers a family at one loop, then take the smallest budget in `2..=max_loops`
that covers every calibration case. The result:

```
diag 4..6, cross -4..-1  ->  2 families, cost 4, 18/18, 6/6
chosen: diag=4 cross=-2, budgets={elementwise: 2, matrix: 2}
```

The distance tie-break keeps the prior `(4,-2)`.

The budget can also be learned per case:

```sh
cargo run -q -p jotnar-agent -- fit-budget-predictor \
  --out runs/encoder --corpus --max-loops 3

cargo run -q -p jotnar-agent -- run --out runs/budget \
  --proposer task --corpus \
  --budget-predictor runs/encoder/budget-predictor.weights.json
```

The predictor is a Jotnar listing over the task vector. It fits 18/18
calibration labels (`{1: 14, 2: 4}`), predicts budget 2 for every wrong-refusal
held-out case, and drives the corpus to 18/18 selection and 6/6 held-out.

The budget head is fused into the looped encoder listing: one program emits
`s0..s5` action scores and `p0..p2` budget scores. `scores_and_budget` reads
both; the fitted budget weights are saved with the looped encoder. See
[`docs/ENCODER.md`](docs/ENCODER.md).

## The backbone

The input block is now a `Backbone` fragment: task features in, the same
9-channel latent out. The named-selection matrix is the exact baseline; the
first learned backbone is a two-layer MLP with a ReLU hidden layer and a
fixed-point readout fitted (ridge correction) to reproduce the named latent.

```sh
cargo run -q -p jotnar-agent -- fit-backbone \
  --out runs/backbone --corpus --hidden 32 --mix 1

cargo run -q -p jotnar-agent -- run --out runs/backbone-run \
  --proposer task --corpus \
  --looped-encoder-weights runs/encoder/looped-encoder.weights.json \
  --backbone runs/backbone/backbone.weights.json
```

The gate is unchanged: 18/18 selection and 6/6 held-out for both the named
baseline and the fitted MLP. A larger hidden mix fits the training latent
better but fails the held-out gate, so the first slice keeps `mix=1`.
`fit-backbone` also writes `looped-encoder-backbone.weights.json`, a single
file with the backbone and the action head.

The second backbone is a Qwen-style block on the φ-ladder: `MATMUL`,
`RMSNORM`, `ROTARY`, `TBETA`, `TSHIFT`, `SOFTMAX_WIDE`, `SILU`, `MUL`, `ADD`,
and `SLICE`, bridged back to the integer latent with `T2I` + `IGET`. The
φ-spoof identity means this is the same multiplication structure as IEEE 754;
the lattice scale is set to cover the task counts, so no float opcode is
needed. The attention output projection and the MLP down projection are
zero-initialized, so the block is an exact pass-through of the named latent
while the full graph is present:

```sh
cargo run -q -p jotnar-agent -- fit-backbone \
  --out runs/backbone-qwen --corpus --qwen --dim 10 --inter 16
```

Result: 18/18 selection, 6/6 held-out, max latent error 0. The listing also
lowers to the non-FPU target; `T2I`/`IGET` have C and non-FPU primitives. See
[`docs/BACKBONE.md`](docs/BACKBONE.md).

The branch readouts can be trained on masked calibration tasks:

```sh
cargo run -q -p jotnar-agent -- fit-qwen \
  --out runs/qwen --masks 8 --mask-rate 0.3 \
  --ridge 0.001 --rounds 3 --seed 1 --scale 0.2
```

```
masked latent error: before=63.0000, after=41.3479
named:   18/18, 0 wrong
trained: 18/18, 0 wrong
heldout: 6/6; adopted=true
```

This is a ridge readout fit on fixed random branch features, not end-to-end
backprop. The gate accepts the trained block. The fused budget head is
recalibrated with `fit-budget-predictor --backbone`; the learned budget can
raise, but not lower, the family/depth prior. The fused file also reaches
18/18 and 6/6. See [`docs/LADDER_OPS.md`](../../docs/LADDER_OPS.md) for the op
set.

## Traces and generated tasks

Every `run` writes `traces.jsonl` into the round directory. A trace is one
`(candidate, case, phase)` record with the task, harness, exact outcome,
`jotnar-lab` certificate, cost, and provenance.

```sh
cargo run -q -p jotnar-agent -- gen-tasks --out runs/tasks.jsonl

cargo run -q -p jotnar-agent -- run --out runs/generated \
  --proposer task --generated
```

The generated corpus recombines library start/target pairs into 11 train /
7 calibration / 6 held-out cases. The task prior is 11/18 with 3/6 held-out;
the causal planner reaches 18/18 and 6/6. The held-out gain comes from the
deep **training** member (`saxpy -> saxpy_s16`); the held-out member
(`affine -> affine_s16`) is unchanged. The wide consumers are taught by the
`vec9`/`vec16` training fans, not by the audit shapes. See
[`docs/TRACES.md`](docs/TRACES.md).

Teacher distillation uses the same task/latent JSONL interface:

```sh
cargo run -q -p jotnar-agent -- gen-teacher \
  --out runs/teacher.jsonl --generated --source named

cargo run -q -p jotnar-agent -- distill-qwen \
  --teacher runs/teacher.jsonl --out runs/qwen-distilled \
  --dim 10 --inter 16 --seed 1 --scale 0.2
```

Dory can write the same file with `"source":"dory"`. `distill-qwen` defaults
to an end-to-end path: warm-start the readouts, backprop through Q/K/V and
gate/up, then keep the best of zero-init, readout, and end-to-end. With
`--corpus` or `--generated`, the selection is gate-aware. The real Dory model
is `nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-GGUF`; see
[`docs/DORY_TEACHER.md`](../../docs/DORY_TEACHER.md). Dory's latent fits to
~2.5e-6 teacher error, but the exact gate rejects the latent-only student
because `start_is_one` differs; the zero-init candidate wins. That is the
next co-training target.

## The action-effect world model

The teacher gives action effects; the world model learns them. `EffectModel`
is a Jotnar-native linear map from the 32 geometric task features to 10 action scores,
trained on the exact `gen-effects` table.

```sh
cargo run -q -p jotnar-agent -- fit-effect-model \
  --out runs/effect-model --generated --epochs 10 --lr 1

cargo run -q -p jotnar-agent -- run --out runs/world \
  --proposer world --generated \
  --effect-weights runs/effect-model/effect-model.weights.json
```

`WorldModelProposer` ranks actions and also proposes every action **pair**, so
the exact gate can choose a composition:

```
generated: 18/18 selection, 6/6 held-out, adopted=true
corpus:    18/18 selection, 6/6 held-out, adopted=true
```

## The state-dependent model and beam

The first model is `(task, action)`. The state-dependent model is
`(task, harness_state, action)`: `gen-transitions` records the exact child
outcome of every applicable action from every harness state reachable with up
to `--depth` actions, and `fit-state-effect-model` trains a ranking perceptron
on the transition table.

```sh
cargo run -q -p jotnar-agent -- gen-transitions \
  --out runs/transitions.jsonl --depth 2

cargo run -q -p jotnar-agent -- fit-state-effect-model \
  --out runs/state-effect-model --generated --depth 2 --epochs 10 --lr 1

cargo run -q -p jotnar-agent -- run --out runs/beam \
  --proposer beam --generated \
  --state-effect-weights runs/state-effect-model/state-effect-model.weights.json \
  --beam 4 --depth 2
```

`BeamProposer` unions the predicted best action per wrong case, adds a
**coverage union** (the best action from each capability group) and an optional
**compose-completeness union** (all compose consumers), expands the top-`beam`
states, keeps the best paths, and still hands every candidate to the exact
gate. The driver prefers the more capable harness only after correctness and
actual steps tie. `--no-complete --no-singles --no-pairs` disables the
structural safety nets for the learned-only ablation.

```
generated: 18/18 selection, 6/6 held-out, adopted=true
corpus:    18/18 selection, 6/6 held-out, adopted=true
selected:  reduce_sum + depth + refinements + select_min
learned-only (no safety nets): 18/18, 6/6, hard 12/12
```

The generated held-out miss is closed by a deep **training** member
(`saxpy -> saxpy_s16`), not by moving the frozen held-out case
(`affine -> affine_s16`). The coverage union keeps `depth + 1` alive beside
`refinements + 1`. The consumer is the geometric `reduce_sum` action: applicability is the input
shape, and the model learns it from the shaped training cases. The
completeness net is still available but is no longer the explanation. See
[`docs/WORLD_MODEL.md`](../../docs/WORLD_MODEL.md).

## The generalization audit

The original held-out split is a **sibling** split: four of six cases are
within three feature units of a training case, and two are exact duplicates.
`feature-audit` prints the min L1 distance per case. A separate `Hard` split
(`hard_corpus()`) is audit-only: it is never used for fitting, selection, or
adoption.

```sh
cargo run -q -p jotnar-agent -- feature-audit --generated

cargo run -q -p jotnar-agent -- run --out runs/hard \
  --proposer beam --generated \
  --state-effect-weights runs/state-effect-model/state-effect-model.weights.json \
  --hard
```

The hard set has four novel `depth + refinements` compositions
(`*_dup -> *_s16`), three high-distance retention cases, one refusal control,
and **three feature-distant compose cases across two mechanisms**:

| case | shape | min-dist | min solver |
| --- | --- | ---: | --- |
| `matmul3 -> matmul3_sum` | `[3,3]` field -> scalar | `add compose[reduce_sum]` |
| `matmul4 -> matmul4_sum` | `[4,4]` field -> scalar | `add compose[reduce_sum]` |
| `select16 -> select16_sum` | `[16]` select field -> scalar | `add compose[reduce_sum]` |

The wide consumers are taught on `vec9`/`vec16` training fans, never on these
shapes. `feature-audit --state-effect-weights FILE` prints the learned top-k;
the learned top-1 on each distant case is the taught consumer.

Progression:

```
base harness:       4/11  (only retention/control)
learned union:     12/12  (reduce_sum from the shape features)
learned-only:      12/12  (no completeness/singles/pairs safety net)
```

The hard set measures the gap; it does not tune against it. The distant
consumers are reached by the learned per-case ranking. The four refine/deep
cases still use the coverage union's structural deep prior for `depth + 1`,
which is a separate pre-existing heuristic.

## Cost-aware planning

`CostModel` is a Jotnar-native cost head over the same `(task, state)`
features, fitted by ridge regression on the correct transition rows and lowered
to a listing with `IDIV`.

```sh
cargo run -q -p jotnar-agent -- fit-cost-model \
  --out runs/cost-model --generated --depth 2 --lambda 1

cargo run -q -p jotnar-agent -- run --out runs/beam-cost \
  --proposer beam --generated \
  --state-effect-weights runs/state-effect-model/state-effect-model.weights.json \
  --cost-weights runs/cost-model/cost-model.weights.json --cost-weight 4
```

The beam score is `effect - cost_weight * predicted_steps`; the exact gate and
the driver's actual-step tie-break still decide.

## The joint latent + causal model

`JointProposer` combines the Qwen-style backbone latent with the state-effect
scores: the latent is the identity channel, the effect model is the causal
channel. Action prototypes are the mean latent of the cases each action fixes.

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

```
generated: 18/18 selection, 6/6 held-out, adopted=true
corpus:    18/18 selection, 6/6 held-out, adopted=true
selected:  reduce_sum + depth + refinements + select_min
hard:      12/12
learned-only (no safety nets): 18/18, 6/6, hard 12/12
```

`run --archive FILE` records gate-competitive candidates in a QD archive,
selected by quality (exact correctness, then held-out) and novelty (L1
distance in the harness-state descriptor). `archive --file FILE` prints it.
The archive keeps the `18/18 + 6/6` path-union harness beside the 14/17
pair attractors that each miss a different capability; adoption still requires
the exact gate, retention, and held-out non-regression.

## The quality-diversity archive

`--proposer qd --archive FILE` seeds a later round with the best archived
attractors and their bounded mutations (basin hopping + neutral drift):

```sh
cargo run -q -p jotnar-agent -- run --out runs/qd \
  --proposer qd --generated --archive runs/archive.json --qd-seeds 4 --hard
```

The first run selected a hop from the `beam-union` seed adding `depth + 1`,
then passed the hard audit 12/12.

## The driver

The driver proposes bounded mutations of the harness (from the Jotnar
proposer or the deterministic control), probes each on the task stream,
chooses by exact evidence only (no labels), and adopts only when:

1. it improves the number of correct outcomes,
2. every incumbent-correct case stays correct, and
3. the frozen held-out examples do not regress.

The adopted harness and its lesson are appended to `experience.jsonl`; every
case gets a certificate under `certificates/`.

## First result

Run:

```sh
cargo run -q -p jotnar-agent -- run --out runs/agent
cargo run -q -p jotnar-agent -- run --out runs/agent --proposer deterministic
cargo test -p jotnar-agent
```

The base harness solves the reduction fit and the SAXPY composition, correctly
refuses the SAXPY→matmul task, and cannot solve two independent failures:

- `compose-matmul` needs the geometric `compose[reduce_sum]` action;
- `refine-s16` needs one more refinement round (`refinements + 1`).

The driver probes the proposals, then **composes the top two**. The composed
harness solves both:

```
case                     incumbent    adopted
fit-reduce               solved       solved
compose-saxpy            solved       solved
refine-s16               wrong-refusal solved
compose-matmul           wrong-refusal solved
refuse-saxpy-matmul      refused      refused

incumbent: 3/5 correct, 2 wrong
adopted:   5/5 correct, 0 wrong
held-out:  3/5 -> 5/5
selected:  compose: add compose[reduce_sum] + refinements + 1
```

That is a capability extension discovered by the loop, with retention and
held-out checks — not a hand-coded feature.

## Held-out transition evaluation

`eval-state-effect` splits the exact transition rows and reports discriminating
top-1 on train and held-out groups:

```sh
cargo run -q -p jotnar-agent -- eval-state-effect --generated \
  --depth 3 --holdout-cases 4 --interaction-lags 35
```

The first held-out split (the select/vec cases) gives `train 1.000 / heldout
0.682`; the WBP lag-35 interaction is neutral. The residual is structured in
row order but the peel layer is not a feature-space attribution; see
[`docs/WBP_CRITIQUE.md`](../../docs/WBP_CRITIQUE.md) and
[`docs/WORLD_MODEL.md`](../../docs/WORLD_MODEL.md). The wall is mechanism/
representation, not model capacity.

## Wall-Breaking Protocol (side quest)

The additive state model is still linear. The Wall-Breaking Protocol asks
whether the residual is structured before adding a nonlinear model: peel the
self-similar AR layers, use the residual-selected product feature as a basis, and
if the residual looks ergodic, inject an irrational harmonic and look again.

The first Jotnar-native slice lives in
[`src/wall_breaking.rs`](src/wall_breaking.rs) and is documented in
[`docs/WALL_BREAKING.md`](../../docs/WALL_BREAKING.md). It uses only existing
integer ops (`IWINDOW`, `IMUL`, `ISUM`, `IADD`, `ISUB`, `IDIV`, `IMAX`, `IEQ`,
`ISELECT`, `IGET`) at a fixed-point scale.

```sh
cargo run -q -p jotnar-agent -- wbp-audit --file runs/wbp-signal.json
```

The synthetic wall result is `linear=0.583 -> wbp-product=1.000`: a product
interaction the additive scorer cannot represent is broken by the Jotnar
self-similar product feature. The deeper repeated-action table is now in place
(205 states / 7380 transitions, depth 3). Its wall — `affine_s64` needing three
`depth + 1` steps — was in proposal composition, not per-transition
linearity; the learned path-union fixes it (18/18 + 6/6, hard 12/12). The WBP
interaction lag 35 is available and neutral on that wall.

## What it proves

- A bounded circuit can be improved by editing its harness, not its weights.
- Two independent failures can be solved by composing the top two proposals.
- Jotnar's interpreter is a sufficient exact gate for this slice.
- A refusal can be a successful, schema-valid certificate.
- The experience log is append-only and carries the evidence for adoption.
- A looped proposer fitted on four cases and evaluated on a held-out fifth
  changes its second proposal with the loop count: one loop misses the deep
  fix, two and four loops find it.
- A 24-case stratified stream with per-family evidence adopts a two-action
  harness and carries it to the held-out family members (6/6).
- A per-case task encoder selects different actions for matrix and deep
  failures, unions them, and reaches 24/24 without a corpus-specific hand
  tuning of the aggregate decoy.
- The encoder's selection matrix can be fitted from per-case outcomes and then
  looped with a per-case compute budget, while keeping 18/18 selection and
  6/6 held-out.
- The per-case Dory crossover is real in two families: linear and one loop miss
  both the matrix `reduce_sum` and the deep fix; two loops find both.
- The loop-cost fit chooses the smallest family budgets that still cover:
  `{matrix: 2, elementwise: 2}`, total cost 4, and preserves the held-out 6/6.
- A learned per-case budget predictor over the task vector fits 18/18
  calibration labels and drives the corpus to 18/18 selection and 6/6
  held-out.
- The budget head is fused into the looped encoder listing: one Jotnar program
  emits both action scores and budget scores.
- The input block is a `Backbone` fragment. The named baseline is exact, a
  two-layer MLP fitted to its latent keeps 18/18 selection and 6/6 held-out,
  and a Qwen-style attention/SwiGLU block passes the same gate as an exact
  zero-init pass-through on the φ-ladder, with a non-FPU lowering.
- The recurrent block controls the loop-count threshold: the prior needs two
  loops; a calibration-margin fit needs four.
- A training deep case plus a coverage/retention/loop-cost objective fits a
  recurrent block that generalizes to the held-out deep case at two loops and
  retains the fix at four.
- A state-dependent `(task, harness_state, action)` world model learns the
  exact transition table and a beam/triple proposer reaches 18/18 + 6/6 on the
  generated stream and 18/18 + 6/6 on the corpus.
- The first causal planner composes three capabilities
  (`reduce_sum + depth/refinements + select_min`), not just a pair.
- The generated held-out miss is closed by adding a deep **training** member
  (`saxpy -> saxpy_s16`); the frozen held-out `affine -> affine_s16` is not
  trained on.
- A joint latent + effect proposer (Qwen φ-ladder backbone + causal state
  model) reaches the same 18/18 + 6/6 and 18/18 + 6/6.
- A quality-diversity archive keeps the best four-capability harness beside the
  novel 14/17 pair attractors, so the loop does not collapse to one incumbent.
- `feature-audit` shows the original held-out split is a sibling split: two
  cases are exact feature duplicates and four are within three units of a
  training task.
- A separate audit-only hard set (four novel `depth + refinements`
  compositions, three high-distance retention cases, one refusal control, and
  three feature-distant compose cases across matrix and select) measures the
  real generalization gap. The base harness scores 4/11; the learned union
  reaches 12/12 without training on the hard set.
- The three distant compose cases (`[3,3]` matmul, `[4,4]` matmul, `[16]`
  select) are all field-to-scalar reductions. One geometric `reduce_sum`
  action handles all three; the model learns its applicability from shape and
  op-family features (`ima(t)mul`, `iselect`, `reduce_applicable`).
- With `--no-complete --no-singles --no-pairs`, the learned-only run still
  reaches 18/18 selection, 6/6 held-out, and hard 12/12: the distant cases are
  not relying on the completeness/singles safety net.
- A Jotnar-native cost head (`(task, state) -> predicted steps`, ridge fit,
  `IDIV` scale) lets the beam rank by `effect - lambda * cost`.
- `--proposer qd --archive` performs basin hopping from archived attractors;
  a hop from the `beam-union` seed adding `depth + 1` reaches the hard audit
  12/12.

## What it does not prove

- C / non-FPU / CUDA parity for the adopted paths (the gate is the interpreter).
- Cost or performance claims (the cost model is not implemented).
- The spatial-map/refine layer from the Python circuit.
- Generalization beyond the supplied and frozen examples.
- That the chosen path is the only path.
- That this generalizes beyond the supplied corpus. The recurrent fit is a
  two-parameter grid; the training deep case is a near neighbour. The 24-case
  corpus has near-duplicate matrix/refine variants and expected-refusal select
  cases. This is a mechanism demonstration, not a benchmark.
- That the loop advantage survives fitting. The fitted linear encoder learns
  the flat rule and removes the need for the loop; the crossover is a
  prior/zero-shot result.
- That the crossover generalizes beyond the two demonstrated families. It is
  matrix and deep elementwise on a 24-case corpus; more families are needed.
- That the budget predictor is a general policy. It is fitted on 18 labels and
  mostly predicts budget 2; the one held-out mismatch is an already-correct
  case where the budget is unused.
- That the task encoder is fully learned. The linear selection matrix is
  fitted; the Qwen-style block is an exact pass-through with zero-initialized
  attention/MLP branches, not a trained transformer; the recurrent block is a
  grid-fitted prior.
- That the state model is a general planner. It enumerates distinct action
  subsets up to `--depth`; repeated `depth + 1` is one action, and depth 4 is
  not yet bounded. The joint latent channel is the named/Qwen pass-through
  latent on this slice, not a backprop-trained attention representation.
- That the archive is a full quality-diversity search. It seeds basin hopping
  and ranks by quality + novelty, but its descriptor is the 14-feature harness
  state, not a continuous behavior space.
- That the hard set is a benchmark. It is twelve audit cases; the four
  refine/deep compositions share one mechanism (`*_dup -> *_s16`) and their
  feature distance is only 3–7. The three distant compose cases span two
  mechanisms but are one consumer each; that is a real but small generalization
  measurement.
- That the geometric consumer transfer is broad. One `reduce_sum` action is
  taught on shaped matrix/vector fields and held out on larger/different
  shapes; the shape contract is structural, and the model still has to rank
  the action above other applicable edits.
- That the four refine/deep hard cases are solved by the learned ranking alone.
  The coverage union keeps its structural deep prior beside the learned
  `refinements + 1`.
- That the coverage/completeness candidates are learned policy. The
  completeness union is a structural safety net; the learned-only ablation
  shows it is no longer needed for the distant cases, but the coverage tie-break
  is still a heuristic.
- That the coverage tie-break is a general policy. It is safe because it runs
  after exact correctness and actual steps, and the held-out check still gates
  adoption, but it can keep a capability that is irrelevant to the current
  stream.
- That the cost head changes the current winner. On this small stream the
  exact gate's step tie-break already selects the cheapest correct harness;
  the cost head matters when the search space is larger.

## Where the LLM goes

This crate is the **harness** the future Jotnar LLM will improve. The intended
split is:

- **substrate**: a Qwen-style block or the smaller geometric unit, expressed as
  Jotnar listings and lowered by the Rust backends
- **circuit**: this crate's intents, steps, gates, and certificates
- **proposer**: a model that reads `experience.jsonl`, refusal certificates,
  and circuit traces, then proposes harness edits or new listings
- **gate**: `run_text` + `jotnar-lab`; the proposer never grades itself

The tiny linear proposer is the first unit in that slot. The backbone contract
is now fixed: task vector in, 9-channel latent out, as a Jotnar listing. The
Qwen-style attention/SwiGLU block is in place; its branches are zero-initialized
and still need training. The next step is the data loop: export the experience
log and certificates into traces, add a task generator with held-out families,
and train the branches against the exact gate. The acceptance contract does not
change: every proposal must survive the exact gate, the retention check, and
the held-out stream.

## Files

| file | role |
| --- | --- |
| `src/state.rs` | `BoundProgram` state and output discovery |
| `src/family.rs` | bounded family builders and fits |
| `src/gate.rs` | exact interpreter gate |
| `src/step.rs` | `fit` / `compose` / `refine` |
| `src/solve.rs` | `specify_or_solve -> plan -> certificate` |
| `src/harness.rs` | editable harness and deterministic mutations |
| `src/proposer.rs` | linear Jotnar proposer, fitting, read-out |
| `src/looped.rs` | looped latent-recurrent proposer (Dory-shaped) |
| `src/driver.rs` | propose/probe/compose/select/reflect/adopt |
| `src/certificate.rs` | `jotnar-lab` certificate construction |
| `src/library.rs` | seed listings, examples, and the acceptance corpus |
| `src/corpus.rs` | the 24-case stratified corpus and its splits |
| `src/view.rs` | per-case `CaseView` provenance |
| `src/task_encoder.rs` | Jotnar-native per-case task encoder and fit |
| `src/looped_encoder.rs` | looped task encoder with a per-case loop budget |
| `src/budget_predictor.rs` | learned per-case loop-budget predictor |
| `src/backbone.rs` | named/MLP/Qwen-style input backbones and Qwen calibration fit |
| `docs/BACKBONE.md` | backbone interface, MLP slice, Qwen φ-ladder block |
| `docs/LADDER_OPS.md` | the φ-ladder op set and the C/non-FPU contract |
| `src/trace.rs` | execution-grounded JSONL trace exporter |
| `docs/TRACES.md` | trace schema, generated task manifest, data loop |
| `docs/DORY_TEACHER.md` | Dory model, llama.cpp build, teacher rows, gate result |
| `scripts/gen_teacher_ollama.py` | local Ollama teacher rows |
| `scripts/gen_teacher_dory.py` | Dory llama-server latent teacher rows |
| `scripts/gen_teacher_dory_actions.py` | Dory action teacher rows (program + effect aware) |
| `scripts/compare_teachers.py` | loop-count / teacher latent comparison |
| `src/effects.rs` | per-case action-effect table and state transition table |
| `src/effect_model.rs` | linear action-effect world model and pair-composition proposer |
| `src/state_effect_model.rs` | `(task, state, action)` world model and beam/triple proposer |
| `src/cost_model.rs` | Jotnar-native `(task, state) -> steps` cost head |
| `src/joint.rs` | Qwen latent + state-effect joint proposer |
| `src/archive.rs` | quality-diversity archive and basin-hopping proposer |
| `src/wall_breaking.rs` | WBP resfrac/peel/ergodic-jump/self-similar listings |
| `docs/WALL_BREAKING.md` | WBP mapping, synthetic wall result, agent application |
| `docs/WORLD_MODEL.md` | effect model, state model, beam, joint, archive, hard audit, results |
