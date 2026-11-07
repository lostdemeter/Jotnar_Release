# Traces and the task generator

The data flywheel has two halves:

1. **Traces** — every candidate, task, exact outcome, certificate, and cost.
2. **Tasks** — generated start/target pairs with parameter/family splits.

Both are plain JSONL so they can be consumed without importing the agent.

## Trace records

`run` writes `traces.jsonl` into each round directory:

```text
runs/agent/round-1/traces.jsonl
```

Each line is one `(candidate, case, phase)` record:

```json
{
  "task": {
    "case": "matmul-sum",
    "family": "matrix",
    "expect": "solve",
    "split": "train",
    "phase": "selection",
    "depth": 2,
    "start_outputs": 4,
    "start_ops": 3,
    "target_outputs": 1,
    "target_ops": 4
  },
  "candidate": {
    "index": 1,
    "label": "case-union: add compose[reduce_sum] + depth + 1",
    "selected": true,
    "adopted": true,
    "harness": {
      "families": ["saxpy", "reduce", "matmul2", "select", "add2"],
      "compose_consumers": ["add2", "reduce_sum"],
      "max_depth": 3,
      "max_refinements": 2,
      "refine_enabled": true,
      "use_tags": false
    },
    "train_correct": 18,
    "train_wrong": 0,
    "heldout_correct": 6,
    "heldout_wrong": 0
  },
  "gate": {
    "verdict": "pass",
    "outcome": "solved",
    "certificate": { "schema": "jotnar-lab/certificate", "...": "..." }
  },
  "cost": { "steps": 3, "refined": false },
  "provenance": {
    "proposer": "task",
    "round": 1,
    "source": "candidate"
  }
}
```

| field | meaning |
| --- | --- |
| `task.split` | `train`, `calibration`, `heldout`, or `stream` |
| `task.phase` | `selection`, `heldout-examples`, or `heldout-split` |
| `candidate.index` | 0 is the incumbent; higher indices are proposals |
| `candidate.selected` | blind-selected candidate |
| `candidate.adopted` | candidate was actually adopted |
| `gate.verdict` | `pass`, `fail`, or `bottom` |
| `gate.outcome` | `solved`, `refused`, `wrong-solve`, or `wrong-refusal` |
| `gate.certificate` | the full `jotnar-lab` certificate |
| `cost.steps` | bounded steps used by the case |
| `provenance.source` | `candidate` or `adopted` |

The adopted harness is also evaluated on the frozen held-out cases, so the
trace contains the real held-out outcome, not just the held-out examples
embedded in the selection cases.

## The task generator

`gen-tasks` writes the generated task manifest:

```sh
cargo run -q -p jotnar-agent -- gen-tasks --out runs/tasks.jsonl
```

A row:

```json
{
  "id": "gen-elem-saxpy-quad",
  "start": "saxpy",
  "target": "saxpy_quad",
  "family": "elementwise",
  "expect": "solve",
  "depth": 2,
  "examples": "saxpy_quad",
  "split": "train",
  "start_outputs": 1,
  "start_ops": 2,
  "target_outputs": 1,
  "target_ops": 4
}
```

The generated corpus recombines library start/target pairs and assigns
splits by parameter/family, not by duplicate. It is a 25-case stream:

- 12 train
- 7 calibration
- 6 held-out

The deep elementwise failures are present in training (`saxpy -> saxpy_s16`
and the deeper `saxpy -> saxpy_s64`) and held out on a different start family
(`affine -> affine_s16`, `affine -> affine_s64`). The `s64` case needs repeated
`depth + 1` steps, so the transition table is a bounded BFS over action
sequences rather than distinct subsets.

The select mechanism is taught on a different start shape:
`gen-select-min-cross-train` (`select_max -> select_min`) is a training case;
the held-out `pair -> select_min` shape is not trained on. The target op family
and `ISELECT` operand orientation are task features, so the model can carry the
mechanism across starts.

The geometric reduction is taught the same way: `vec9 -> vec9_sum` and
`vec16 -> vec16_sum` are shaped vector-to-scalar **training** cases. The shaped
`matmul2` cases also exercise `reduce_sum`. `matmul3 -> matmul3_sum`,
`matmul4 -> matmul4_sum`, and `select16 -> select16_sum` are held out in the
`Hard` split.

Run it with:

```sh
cargo run -q -p jotnar-agent -- run --out runs/generated \
  --proposer task --generated
```

The current prior does **not** solve the generated matrix/select variants.
That is the point: the generated stream is training data, not another
hand-tuned acceptance corpus. The exact gate labels every case; the trace
file records what the model got wrong.

### The hard audit split

`hard_corpus()` is a separate `Hard` split that is never used for fitting,
selection, or adoption. It has four `depth + refinements` compositions
(`*_dup -> *_s16`), three high-distance reduction/matrix retention cases, one
refusal control, three feature-distant compose cases, and one held-out select
mechanism case:

| case | shape | min solver |
| --- | --- | --- |
| `hard-matmul3-sum` | `[3,3]` matmul field -> scalar | `add compose[reduce_sum]` |
| `hard-matmul4-sum` | `[4,4]` matmul field -> scalar | `add compose[reduce_sum]` |
| `hard-select16-sum` | `[16]` select field -> scalar | `add compose[reduce_sum]` |
| `hard-affine-s64` | `affine` -> 64x deep chain | `depth + 1` x3 (repeated) |
| `hard-select-min-pair` | `pair` -> min select | `add family select_min` |

The geometric reduction is taught on `vec9`/`vec16` shaped training cases, and
repeated depth is taught on `saxpy_s64`; none of the hard shapes is trained on.
The depth-3 table has 205 states / 7790 transitions; the learned path-union
candidate preserves repeated depth/refinement actions. `feature-audit` prints the count distance and
the geometry signature (rank, numel, op family, reduction applicability) from
each held-out/hard case, and (with `--state-effect-weights`) the model's learned
top-k action scores. `run --hard` evaluates the adopted harness on the hard set
and writes `hard.json`. Both are audits: they measure generalization, they do
not tune against it.

## Teacher distillation

A teacher row is task vector in, latent out:

```json
{"task":[3,1,2,3,1,4,2,0,0,1,0,0,2,0,1,0,1,0,1,0],
 "latent":[1.0,2.0,4.0,2.0,0.0,0.0,0.0,0.0,1.0],
 "split":"train","source":"named"}
```

`gen-teacher` writes the named-encoder baseline; Dory can write the same JSONL
with `"source":"dory"`.

```sh
cargo run -q -p jotnar-agent -- gen-teacher \
  --out runs/teacher.jsonl --generated --source named

cargo run -q -p jotnar-agent -- distill-qwen \
  --teacher runs/teacher.jsonl --out runs/qwen-distilled \
  --dim 10 --inter 16 --seed 1 --scale 0.2
```

`distill-qwen` defaults to an end-to-end path:

1. warm-start the attention/MLP readouts with the ridge fit;
2. backprop through the f64 Qwen block and update Q/K/V and gate/up;
3. keep the best of `zero-init`, `readout`, and `end-to-end` by teacher
   latent error.

```sh
cargo run -q -p jotnar-agent -- distill-qwen \
  --teacher runs/teacher.jsonl --out runs/qwen-distilled \
  --end-to-end --epochs 20 --lr 0.001 \
  --dim 10 --inter 16 --seed 1 --scale 0.2
```

`--readout` keeps the older readout-only fit. The end-to-end path uses
batched gradients, clipping, and a small learning rate; the readout is the
warm start. The named teacher is the baseline; the Dory teacher is the same
interface with a different latent distribution. The exact gate still decides
whether the distilled student is adopted.

The real Dory model is `nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-GGUF`; the
Dory-enabled llama.cpp build and the `gen_teacher_dory.py` script are described
in [`DORY_TEACHER.md`](../../../docs/DORY_TEACHER.md). Dory's 9-channel latent
fits to ~2.5e-6 teacher error, but the exact gate rejects the distilled student
because `start_is_one` differs from the named contract. The zero-init candidate
wins; see `DORY_TEACHER.md` for the gate table. This is the key distillation
result: a teacher latent must be paired with action-head co-training or an
action/listing target, not just latent MSE.

## The data loop

```text
gen-tasks -> run --generated -> traces.jsonl
                ^                    |
                |                    v
        better listing  <-  train Jotnar LLM
```

Every accepted/rejected candidate is a training row. The certificate is the
label; the exact gate is the teacher. The next model does not grade itself.
