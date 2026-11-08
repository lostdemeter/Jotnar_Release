# The task encoder

The aggregate proposer asked: *how many cases failed?* The task encoder asks:
*which case failed, and what shape does it have?*

## The case view

Each case becomes a `CaseView`:

```
case_id, family, expect_solve, outcome,
start_inputs, start_outputs, start_ops,
target_inputs, target_outputs, target_ops,
steps, refined, tags
```

The encoder turns that into a 20-dim task vector:

```
start_inputs, start_outputs, start_ops,
target_inputs, target_outputs, target_ops,
depth_delta, output_delta, input_delta,
expect_solve, wrong_refusal, wrong_solve,
steps, refined,
start_outputs_1, start_outputs_4,
target_outputs_1, has_sum4, refinements, tags_count
```

## The encoder

The encoder is an ordinary Jotnar listing:

```
task vector -> 9 named latent channels -> one score per bounded action
```

The latent channels are deliberately legible:

```
start_outputs, start_ops, target_ops, depth_delta,
wrong_refusal, start_is_four, start_is_one, has_sum4
```

The prior is a selection matrix, not an opaque model. It is written to
`proposer.jotnar` beside the run.

## Per-case selection

For every **wrong refusal**:

1. encode the task vector,
2. score all six actions,
3. take the highest positive action that is still applicable,
4. union the selected actions across cases,
5. evaluate the union with the exact gate.

The driver still does blind selection, retention, and held-out adoption. The
encoder only chooses what to try.

## Example scores

Matrix failure (`matmul2 -> sum`):

```
latent:  start_outputs=4, start_ops=12, target_ops=15,
         depth_delta=3, wrong_refusal=1, start_is_four=1
scores:  sum4=12, depth=9, refinements=10, ...
choice:  add compose[sum4]
```

Deep elementwise failure (`saxpy -> s16`):

```
latent:  start_outputs=1, start_ops=2, target_ops=6,
         depth_delta=4, wrong_refusal=1, start_is_one=1
scores:  refinements=18, depth=13, sum4=0, ...
choice:  refinements + 1
```

The union is `sum4 + refinements + 1`.

## Results

24-case corpus:

```
selection: 14/18 correct, 4 wrong -> 18/18 correct, 0 wrong
heldout:   6/6 correct, 0 wrong
selected:  case-union: add compose[sum4] + refinements + 1
```

5-case stream:

```
incumbent: 3/5 correct, 2 wrong
adopted:   5/5 correct, 0 wrong
held-out:  3/5 -> 5/5
```

Run it:

```sh
cargo run -q -p jotnar-agent -- run --out runs/corpus \
  --proposer task --corpus
```

## Why this is better than the aggregate prior

The corpus prior needed hand-tuned per-family weights so the matrix decoy did
not crowd out the deep fix. The task encoder does not need that: the matrix
case and the deep case are different vectors, so they select different actions.

## Fitting the matrix

`calibration_dataset(cases, base)` measures every `(case, action)` pair:

- `+1` if the action turns a wrong case correct,
- `-1` if it breaks a correct case,
- `0` if it changes nothing.

The fit is a per-case ranking perceptron:

- group samples by case;
- if a positive action is already top, do nothing;
- otherwise boost the best positive action and penalize the competing top
  action.

This matters: a naive per-action update lets three matrix samples overwhelm
the one deep sample. The ranking update keeps the deep action competitive.

```sh
cargo run -q -p jotnar-agent -- fit-encoder --out runs/encoder --corpus
```

```
encoder fit: 90 samples (5 positive, 0 negative), epochs=3, lr=1
  positive action top-1: 4/5; negative action avoided: 0/0
```

The fifth positive sample is an alternative deep action (`depth + 1` vs
`refinements + 1`), so 4/5 top-1 is not a miss: the case still gets a valid
action. The fitted encoder reaches 18/18 selection and 6/6 held-out.

```sh
cargo run -q -p jotnar-agent -- run --out runs/fitted --proposer task \
  --corpus --encoder-weights runs/encoder/task-encoder.weights.json
```

## Looping the encoder

`LoopedTaskEncoder` uses the fitted linear encoder as the input and output
block, then applies a shared recurrent block with the task vector reinjected
every loop:

```
task -> input block -> h0
h_{t+1} = clip(relu(h_t + (Wr h_t + Win task + br)/scale), 0, 64)
scores = Wout h_N + bout
```

The loop count is a **per-case budget**:

| case | loops |
| --- | ---: |
| correct / refused | 1 |
| matrix-shaped wrong refusal | 2 |
| deep wrong refusal | 3 |

### The per-case crossover

The prior output block is deliberately loop-sensitive:

```
sum4         = wrong_refusal*2 + start_is_four*4 - has_sum4*20
refinements  = wrong_refusal*2 + start_is_one*6
tags off     = wrong_refusal*1 + target_outputs_1*10
```

`target_outputs_1` is stable across loops. `start_is_four` and `start_is_one`
are reinjected and grow with the shared block.

| family | linear / 1 loop | 2 loops | 3 loops |
| --- | --- | --- | --- |
| matrix | `tags off` decoy | **sum4** | **sum4** |
| deep elementwise | `tags off` decoy | **depth / refinements** | **depth / refinements** |

Both families are wrong at one loop and right at two/three. On the corpus:

| mode | selection | held-out |
| --- | ---: | ---: |
| linear | 14/18 | 4/6 |
| 1 loop | 14/18 | 4/6 |
| 2 loops | **18/18** | **6/6** |
| 3 loops | **18/18** | **6/6** |

```sh
cargo run -q -p jotnar-agent -- run --out runs/one-loop --proposer task \
  --corpus --encoder-loops 1

cargo run -q -p jotnar-agent -- run --out runs/two-loops --proposer task \
  --corpus --encoder-loops 2
```

### Fitting the recurrent block with a loop-cost objective

`fit-looped-encoder` grids the shared block’s `diag` and `cross`:

```sh
cargo run -q -p jotnar-agent -- fit-looped-encoder \
  --out runs/encoder --corpus --max-loops 3
```

The objective is now a loop-cost fit:

1. reject a config if any family is already covered at one loop (that would
   erase the crossover);
2. for each family, take the **smallest** loop count in `2..=max_loops` that
   covers every calibration case;
3. prefer calibration correctness, then coverage, then **lowest total loop
   cost**, then distance from the prior.

Result:

```
diag 4..6, cross -4..-1  ->  2 families, cost 4, 18/18 calibration, 6/6 held-out
chosen: diag=4 cross=-2, budgets={elementwise: 2, matrix: 2}
```

The loop-cost fit chooses **two loops for both families** and a total budget of
4. The prior-distance tie-break keeps `(4,-2)`.

Note: fitting the **linear** selection matrix (`fit-encoder`) learns the flat
rule directly and removes the need for the loop. The crossover is a
prior/zero-shot result; fitting on the same task distribution can make one
loop sufficient again.

Run the fitted looped encoder:

```sh
cargo run -q -p jotnar-agent -- run --out runs/fitted-loop \
  --proposer task --corpus \
  --looped-encoder-weights runs/encoder/looped-encoder.weights.json
```

### Learned per-case budget

`fit-budget-predictor` learns the loop budget directly from the task vector:

```sh
cargo run -q -p jotnar-agent -- fit-budget-predictor \
  --out runs/encoder --corpus --max-loops 3
```

When refitting against a trained backbone, add
`--backbone runs/qwen/backbone-qwen-trained.weights.json`. Labels are the
smallest loop count where the looped encoder’s top action is a positive action
for that case; correct/refused cases are label 1. The predictor
is another ordinary Jotnar listing: task vector -> one score per budget class
-> argmax. It is fitted with the same ranking perceptron.

```
budget fit: 18 samples, epochs=10, train accuracy 18/18
budget distribution: {1: 14, 2: 4}
heldout budget matches: 5/6
```

The one held-out mismatch is `refine-matmul-holdout`: it is already correct,
so the encoder uses one loop regardless of the predicted budget. Every
wrong-refusal held-out case gets budget 2.

Run the corpus with the learned predictor:

```sh
cargo run -q -p jotnar-agent -- run --out runs/budget \
  --proposer task --corpus \
  --budget-predictor runs/encoder/budget-predictor.weights.json
```

Result: 18/18 selection, 6/6 held-out.

The per-case predictor can raise the family/depth prior for wrong-refusal
cases, but it cannot lower it: `loops_for_task` uses
`max(predicted, structural_prior)`. Correct/refused cases still use one loop
by construction. The safety floor is what lets a trained backbone and a
recalibrated budget head compose without under-budgeting the deep cases.

### One program, two outputs

The budget head is now fused into the looped encoder listing. The same program
emits:

```
s0..s5   action scores (from the final looped latent)
p0..p2   budget scores (from the task vector, loop-independent)
```

`scores_and_budget(task, loops)` reads both in one run. The fitted budget
weights live in `LoopEncoderWeights` and are saved with the looped encoder, so
a single `looped-encoder.weights.json` carries the circuit and the budget head.

```sh
cargo run -q -p jotnar-agent -- fit-budget-predictor \
  --out runs/encoder --corpus --max-loops 3

cargo run -q -p jotnar-agent -- run --out runs/fused \
  --proposer task --corpus \
  --budget-predictor runs/encoder/budget-predictor.weights.json
```

Result: 18/18 selection, 6/6 held-out. The generated `proposer.jotnar` contains
both `s*` and `p*` output groups.

## Next

1. **More loop-sensitive families.** The crossover is measured on matrix and
   deep elementwise; add families where one loop is provably insufficient.
2. **Pool over cases.** A small attention/pooling head over the case views
   rather than independent per-case decisions.
3. **Train the Qwen branches.** The Qwen-style attention/SwiGLU block is in
   place as an exact zero-init pass-through (see
   [`BACKBONE.md`](BACKBONE.md)). `fit-qwen` now fits the attention/MLP
   readouts on masked calibration tasks; the trained block keeps 18/18 and
   6/6. The next step is end-to-end/teacher training on a larger trace set.
4. **Trace schema + task generator.** Export the experience log and
   certificates into a dataset, then add family/parameter splits with held-out
   families. That is the data flywheel the Jotnar LLM needs.

## Honest boundary

The linear selection matrix is fitted; the input block is a `Backbone`
fragment (named baseline, fitted two-layer MLP, or zero-init Qwen-style
pass-through), the recurrent block is a small grid-fitted prior, and the
corpus is 24 cases. The Qwen attention/MLP branches are not trained yet. The
crossover is demonstrated on two families. This is the per-case interface plus
a working fit and loop, not a general model.
