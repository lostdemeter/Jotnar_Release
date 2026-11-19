# Batch GPU pipeline

This is workload #2 for the workload controller. It is a short batch pipeline
through `jotnar-workload`, not a DOOM-specific coordinator:

```text
payload artifact
  -> gpu-a-NN (CUDA)  OUT = ADD(X, X)
  -> gpu-b-NN (any)   reuses gpu-a-NN's binary OUT as payload stream X
  -> OUT
```

The controller decodes the first stage's binary F32 result into a JSON payload
for the second stage. The bundle is content-addressed; the receipt records each
stage's target, executor, job id, and input/output hashes.

`double.asm` is the listing. `scripts/gpu_batch_gate.py` builds a deterministic
payload, runs a single-node CUDA reference, runs the CUDA -> HIP chain, and
compares every output byte-for-byte. `scripts/doom_gpu_contention_gate.py` runs
the same chain while DOOM plays on the same CUDA and HIP runners.

```sh
python3 scripts/gpu_batch_gate.py
python3 scripts/doom_gpu_contention_gate.py --items 8 --size 64x64x3
```

These gates need the live CUDA and HIP runners. The in-repo controller test
(`crates/jotnar-node/tests/gpu_workload.rs`) covers the same stage handoff with
a fake GPU executor and no network.
