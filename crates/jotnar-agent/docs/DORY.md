# From linear proposer to looped latent recurrence

NVIDIA's Dory1.0-4B is a **looped (latent-recurrent) normalized-GPT**:

- an **input block** (8 layers),
- a **shared recurrent block** (20 layers) applied 1–4 times,
- an **output block** (8 layers),
- the input **reinjected at every loop**,
- **elastic inference compute**: one checkpoint, different loop counts,
- **last-loop KV caching**: one cache per recurrent attention layer,
- nGPT backbone, native NVFP4 pre-training, 256K context.

The interesting part for us is not the 4B scale. It is the shape:

```
input -> [input block] -> h0
h_{t+1} = shared_block(h_t, input)     # applied N times, input reinjected
scores  = [output block] h_N
```

That shape maps directly onto the circuit harness.

## Mapping

| Dory | `jotnar-agent` |
| --- | --- |
| input block | `h0 = (Win f + bin) / scale` |
| shared recurrent block | `Wr`, `Wf` reused for every loop |
| input reinjection | `Wf f` added at every loop |
| output block | `scores = Wout h_N + bout` |
| elastic loop count | `--loops 1..8` |
| normalization | integer `IDIV` scale + clip |
| nonlinearity | residual + `IMAX(.,0)` + bounded select |
| last-loop KV cache | the planner keeps only the current state; `visited` is the cache |
| nGPT | integer latent lanes; not yet a transformer |

The first looped model is deliberately small: `latent=13`, one input block, one
shared recurrent block, one output block. It is an ordinary Jotnar listing and
is run through the same `run_text` gate as every candidate.

## Why the nonlinearity matters

A looped **linear** model is only a reparameterized linear model: the effective
map is `Wout (Wr)^N`, still linear in the features. Dory's gain comes from the
nonlinear block. The first version therefore uses:

```
d        = (Wr h_t + Wf f + br) / scale
h_{t+1}  = clip(relu(h_t + d), 0, 64)
```

The residual keeps the state alive, the ReLU gates cross-lane terms, and the
clip bounds growth. This is the smallest honest stand-in for the normalized
recurrent block.

## The loop-count crossover

The corpus now has two independent failures:

- `compose-matmul` needs a `compose[sum4]` consumer;
- `refine-s16` needs one more refinement round (`refinements + 1`).

The driver probes the top proposals, then composes the top two. The second
proposal is where the loop count matters.

| loops | scores `(sum4, depth, refine, tags, select_max, refinements)` | second proposal | outcome |
| ---: | --- | --- | --- |
| 0 (linear) | — | `tags off` | decoy; no adoption |
| 1 | `[16, 3, 1, 11, -2, 10]` | `tags off` | decoy; no adoption |
| 2 | `[18, 4, 0, 13, -1, 16]` | `refinements + 1` | adopted 5/5 |
| 4 | `[28, 9, 0, 23, 4, 39]` | `refinements + 1` | adopted 5/5 |

`tags off` is a deliberate decoy: it reads the refusal count and the
four-output flag, but it deepens nothing. The `refinements` row reads the
family count, which the recurrent block amplifies. At one loop the decoy wins;
at two loops the deep fix overtakes it; at four loops the deep fix is the top
score.

This is the Dory claim in miniature: **the same weights, a larger loop budget,
a different decision, and a better outcome.** The composed harness is
`sum4 + refinements + 1`, and it reaches 5/5 with zero wrong outcomes.

## Fitted on four, tested on the fifth

The first fit used the same cases it was evaluated on, which erases the
crossover by teaching the one-loop model the answer. The honest split is:

- **calibration**: `fit-reduce`, `compose-saxpy`, `compose-matmul`,
  `refuse-saxpy-matmul`
- **held out**: `refine-s16` (the deep fix)

The fit reinforces only `sum4`, because the deep case is absent. The held-out
case still has to be inferred from the recurrent block:

| loops | scores `(sum4, depth, refine, tags, select_max, refinements)` | outcome |
| ---: | --- | --- |
| 1 | `[159, 3, 1, 11, -2, 10]` | decoy second; no adoption |
| 2 | `[360, 4, 0, 13, -1, 16]` | composes `sum4 + refinements + 1`; 5/5 |
| 4 | `[1849, 9, 0, 23, 4, 38]` | composes the same fix; 5/5 |

The deep fix was not in calibration. Two loops discover it; one does not.
That is a **generalization** effect from the repeated block, not memorization
of the calibration set.

The honest claim:

- the three-block looped structure is implemented and runs as a Jotnar listing;
- the output block is fitted; the recurrent block is now grid-fitted;
- one loop misses the held-out deep fix, two and four loops find it;
- the corpus is five cases; this is a mechanism demonstration, not a benchmark.

## Fitting the recurrent block: both fixes

The shared block is parameterized by two integers: the diagonal (`diag`) and
the cross-lane coefficient (`cross`). The fit grid is `diag 2..6`,
`cross -4..1`. The held-out `refine-s16` is never used to select.

Two independent levers were tested:

1. **Training deep case** — `affine -> affine_s16`, a *different* 1-output
   start that needs the same depth class. It is added to calibration; the
   held-out `refine-s16` stays out.
2. **Efficient objective** (`--efficient`) — a lexicographic score:
   ```
   (covered, retained, -loop_cost, margin_total, prior_distance)
   ```
   - `covered`: the calibration requirements (`sum4` and a deep action) enter
     the top two at some loop count;
   - `retained`: they are still present at the largest loop budget (4);
   - `loop_cost`: the first loop count where coverage holds (earlier is better).

The 2×2 experiment:

| fit | chosen | calibration coverage | held-out 1 / 2 / 4 |
| --- | --- | --- | --- |
| margin only | `(6,-4)` | `sum4` only | miss / miss / hit |
| margin + deep-train | `(6,1)` | none (margin ignores deep) | miss / miss / miss |
| efficient | `(6,-4)` | `sum4` at 1, retained | miss / miss / hit |
| **efficient + deep-train** | **`(5,-4)`** | both at 2, retained at 4 | **miss / hit / hit** |

The result is that **both** are needed:

- the deep training case gives the fit a deep signal;
- the efficient/retention objective stops it from choosing a high-margin block
  that either ignores the deep case or loses it again at four loops;
- together they select `(5,-4)`, which generalizes to the held-out
  `refine-s16` at two loops and retains the fix at four.

Full-loop verification with `(5,-4)`:

| loops | outcome |
| ---: | --- |
| 1 | 3/5, no adoption |
| 2 | 5/5, adopted |
| 4 | 5/5, adopted |

That is the monotone elastic-compute behavior we wanted: one loop fails, two
loops succeeds, four loops retains.

Run the fit and use the weights:

```sh
cargo run -q -p jotnar-agent -- fit-recurrent --out runs/fit \
  --deep-train --efficient
cargo run -q -p jotnar-agent -- run --out runs/loops2 \
  --proposer looped --loops 2 --weights runs/fit/looped.weights.json
```

Honest boundary: this is a two-parameter recurrent grid on a five-case corpus.
The training deep case is a near neighbour (same depth class, different
arithmetic). It is a mechanism demonstration, not a benchmark.

## Next

1. **A larger corpus.** The mechanism is now demonstrated on one held-out deep
   case. Add more independent failures and decoys so the loop-count effect is
   measured over a distribution.
2. **Fit more of the recurrent block.** The grid fits two parameters
   (`diag`, `cross`). Fit the full `Wr`/`Wf` matrices once the corpus can
   constrain them.
3. **Feature interactions.** The current input is a flat feature vector. Add a
   small interaction block so the recurrent layers can compose evidence, not
   just rescale it.
4. **Attention / SWA.** Replace the dense `Wr`/`Wf` blocks with a bounded
   attention-like operation when the feature set grows.
5. **Last-loop state.** Make the circuit's per-loop state explicit: carry only
   the latest state and a bounded trace, not the whole path, so the memory cost
   is independent of the loop count.
