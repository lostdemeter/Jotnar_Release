# The Dory teacher

The actual Dory model is available on Hugging Face:

- `nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16`
- `nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-GGUF` (`Q5_K_M`, 3,673,288,864 bytes)

Dory is a looped latent-recurrent 4B model: 16 input layers, 40 recurrent
layers, 16 output layers, 2 recurrent loops by default. It is the natural
teacher/reference for the Jotnar looped encoder.

## Running Dory locally

Dory support is not in the released `llama.cpp` yet. The model card points at
the Dory-enabled PR:

```text
https://github.com/brianchmielNV/llama.cpp
revision 98a74c97ea09ac72f6ceda1559808fc0a2f447ce
```

Build `llama-server` with CUDA and run the GGUF with two loops:

```sh
git clone https://github.com/brianchmielNV/llama.cpp.git llama.cpp-dory
cd llama.cpp-dory
git checkout 98a74c97ea09ac72f6ceda1559808fc0a2f447ce

cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86 \
  -DLLAMA_BUILD_SERVER=ON -DLLAMA_BUILD_UI=OFF \
  -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF
cmake --build build --target llama-server --parallel

./build/bin/llama-server \
  -m ../dory-model/NVIDIA-Nemotron-Labs-Dory1.0-4B-Q5_K_M.gguf \
  --alias dory --host 127.0.0.1 --port 8080 \
  -ngl 999 --ctx-size 4096 --parallel 1 \
  --batch-size 512 --ubatch-size 512 \
  --cache-type-k f16 --cache-type-v f16 \
  --flash-attn off --no-context-shift \
  --jinja --reasoning on \
  --override-kv "dory.num_loops=int:2,dory.recurrent_kv_cache_mode=str:per_loop"
```

The local environment needed the CUDA runtime headers from the Python
`nvidia-cu13` package plus CCCL headers from `cuda-cccl`; the standard CUDA
toolkit layout is simpler.

## Generating teacher rows

`jotnar-agent gen-teacher` writes the task vectors and the named baseline:

```sh
cargo run -q -p jotnar-agent -- gen-teacher \
  --out runs/teacher-tasks.jsonl --generated --source named
```

`scripts/gen_teacher_dory.py` reads that file, prompts the Dory server for the
9 latent channels, and writes the same schema with `"source":"dory"`:

```sh
python3 scripts/gen_teacher_dory.py \
  --tasks runs/teacher-tasks.jsonl \
  --out runs/teacher-dory.jsonl
```

The generated 20-row file is kept at:

```text
/home/balrog/student_runs/depth_jotnaros_data/teacher-dory.jsonl
```

## Loop probe

Dory was run at 1, 2, and 4 recurrent loops on the 20 generated tasks. The
short JSON prompt is already in a stable basin:

| comparison | mean absolute delta | rows changed |
| --- | ---: | ---: |
| 1-loop vs 2-loop | 0.033 | 1/17 |
| 1-loop vs 4-loop | 0.026 | 1/17 |
| 2-loop vs 4-loop | 0.0065 | 1/17 |
| named vs 2-loop | 0.111 | 17/17 |

The loop count barely moves the teacher latent. The meaningful difference is
the attractor: Dory sets `start_is_one=1` and `start_is_four=0`; the named
contract sets `start_is_one=0` and has a nonzero `start_is_four` for three
cases. This is why latent MSE is not enough.

## Action teacher

`scripts/gen_teacher_dory_actions.py` asks Dory for one harness action per
case. With the program-aware prompt it includes the start listing, target
listing, examples, task features, and the 10-action menu. The server needs a
larger context (`--ctx-size 16384`) because the reduce-128 examples are large.

```sh
python3 scripts/gen_teacher_dory_actions.py \
  --tasks runs/teacher-tasks.jsonl \
  --programs runs/gen-tasks.jsonl \
  --out runs/teacher-actions-dory.jsonl
```

The 20-row action file is at:

```text
/home/balrog/student_runs/depth_jotnaros_data/teacher-actions-dory-programs.jsonl
```

Run the teacher through the exact gate:

```sh
cargo run -q -p jotnar-agent -- run --out runs/dory-actions \
  --proposer teacher --generated \
  --teacher-actions runs/teacher-actions-dory-programs.jsonl
```

Result:

```text
selection: 11/14 correct, 3 wrong -> 11/14 correct, 3 wrong
heldout:   3/6 correct, 3 wrong
selected:  incumbent; adopted=false
```

Dory's action distribution was varied (`refinements + 1` 6, `saxpy_dup` 4,
`sum4` 3, `affine_dup` 2, `select_min` 2, `matmul2_dup` 2, `select_max` 1),
but the gate still rejects the union. The model does not know the exact
harness action effects; the teacher needs feedback/action-effect examples or
the action head needs co-training.

## Action-effect feedback

`gen-effects` records what every bounded action does on every case:

```sh
cargo run -q -p jotnar-agent -- gen-effects \
  --out runs/action-effects.jsonl
```

Each row contains the base outcome and a 10-entry action table: applicable,
outcome, correct/wrong, steps, and the gate note. The generated table is at:

```text
/home/balrog/student_runs/depth_jotnaros_data/action-effects.jsonl
```

The Dory action prompt now includes the start listing, target listing,
examples, task features, and the action-effect table. Dory produced varied
actions, but the exact gate still rejected the union:

```text
selection: 11/14 correct, 3 wrong -> 11/14 correct, 3 wrong
heldout:   3/6 correct, 3 wrong
selected:  teacher: add compose[sum4] + add family select_min; adopted=false
```

Action-effect feedback alone is not enough for Dory. Dory still does not know
how its action choice composes with the rest of the harness.

The effect data is enough for a **world model**, though. The Jotnar-native
linear effect model and pair-composition proposer reach **14/14 generated
selection, 5/6 held-out** and **18/18 corpus, 6/6 held-out**. See
[`WORLD_MODEL.md`](WORLD_MODEL.md). Dory remains the teacher for the effect
labels; the exact gate remains the judge.

## Distillation result

`distill-qwen` now supports a gate-aware selection. It builds three candidates:

1. zero-init / random-pass baseline
2. readout-only fit
3. end-to-end Q/K/V/gate-up fit

and evaluates each with the exact gate on `--corpus` or `--generated`.

```sh
cargo run -q -p jotnar-agent -- distill-qwen \
  --teacher runs/teacher-dory.jsonl \
  --out runs/qwen-dory --end-to-end --corpus \
  --epochs 20 --lr 0.001 --dim 10 --inter 16
```

Dory's latent is very close to the named latent (teacher error after the
readout fit is ~2.5e-6), but it differs in the `start_is_one` channel. The
exact gate rejects the distilled student:

```text
gate zero-init: train 18/18 (0 wrong), held 6/6
gate readout:  train 15/18 (3 wrong), held 5/6
gate end-to-end: train 15/18 (3 wrong), held 5/6
selected=zero-init
```

On the generated stream:

```text
gate zero-init: train 13/14 (1 wrong), held 5/6
gate readout:  train 11/14 (3 wrong), held 3/6
gate end-to-end: train 11/14 (3 wrong), held 3/6
selected=zero-init
```

This is the important result: **latent distillation is not enough**. Dory's
9-channel latent is close, but the downstream action head and loop are
sensitive to the exact channel values. The teacher must either:

- be constrained to the named latent contract, or
- co-train the action head / loop with the student, or
- teach **actions/listings** rather than only the latent.

The exact gate is doing its job: it rejects a student that fits the teacher
latent but loses the task behavior.

## Next

1. Co-train the action head with the Dory latent so the student can absorb a
   different but consistent latent.
2. Give Dory **action-effect feedback**: show it what each action did on the
   examples, not just the action menu; then gate the proposal.
3. Use Dory's loop count as a teacher only on tasks where 1/2/4 loops actually
   change the answer; the current generated prompts are already a fixed basin.
4. Keep a quality-diversity archive of students/action heads so the loop does
   not converge to a single incumbent attractor.
