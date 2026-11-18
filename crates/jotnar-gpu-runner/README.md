# Jotnar GPU runner

A host-side HIP/CUDA runner for Jotnar listings. It is deliberately separate
from `jotnar-noded`: the model-blind CPU node stays unchanged. This service is
target-aware, compiles with the existing Jotnar codegen, runs the generated
binary on the local GPU, and returns typed streams.

## Protocol

One frame per TCP connection, using the node frame shape:

```text
u32BE length || JSON header || '\n' || payload
```

`ping`:

```json
{"v":1, "id":1, "type":"ping", "auth":"..."}
```

`payload-put` uploads a parsed payload once; `run` can then use its id:

```json
{"v":1, "id":1, "type":"payload-put", "auth":"..."}
{"v":1, "id":1, "type":"run", "target":"hip", "outputs":["OUT"],
 "payload_id":"a86e11ef3b2b8bf5", "output_format":"binary", "auth":"..."}
```

`output_format:"binary"` returns raw stream bytes plus descriptors in the
response header; the Rust scheduler requests it. Without it the response body
is the JSON stream form used by the debug scripts.

`run-batch` runs N frames on one warm worker in one request:

```json
{"v":1, "id":1, "type":"run-batch", "target":"hip", "outputs":["OUT"],
 "frames":8, "payload_id":"a86e11ef3b2b8bf5",
 "output_format":"binary", "auth":"..."}
```

The binary response uses `type:"run-batch-ok"`, a `frames` count, and one
descriptor per frame with a `frame` index. Without binary transport the body
is `{"frames":[{"outputs":...}, ...], "timing":...}`. The runner executes the
frames serially on the warm worker and returns per-frame `frame_ms`.

`warm-prepare` compiles/builds and starts the warm worker without running a
frame:

```json
{"v":1, "id":1, "type":"warm-prepare", "target":"hip", "outputs":["OUT"],
 "payload_id":"a86e11ef3b2b8bf5", "auth":"..."}
```

It returns `warm-prepare-ok` with `warm`, `worker_start_ms`, `warmup_ms`,
`warmup`, `worker_reused`, `compile_ms`, `build_ms`, and `cache_hit`. The
request runs one warm-up frame by default so the first real frame does not pay
lazy kernel initialization; send `"warmup": false` to only start the worker.
Repeated prepares reuse the worker. If the runner is started with `--no-warm`,
it returns `warm:false` after the build.

payload:

```json
{
  "listing": "IN a\nIN b\nOUT = MATMUL(a, b)\n",
  "payload": {"streams": {"a": {"kind":"F64","shape":[2,2],"data":[...]}, ...}},
  "use_fp16": false
}
```

response:

```json
{"v":1, "id":1, "type":"run-ok", "ok":true, "target":"hip",
 "runtime_version":"...", "compile_ms":0, "build_ms":900, "run_ms":12,
 "cache_hit":false}
```

with a body containing typed `outputs`.

## Build and run

```sh
cargo build --release -p jotnar-gpu-runner
```

HIP host:

```sh
env HIPCC=/opt/rocm-7.2.4/bin/hipcc HIP_ARCH=gfx1100 \
  PATH=/opt/rocm-7.2.4/bin:$PATH \
  ./jotnar-gpu-runner --bind 0.0.0.0:9997 --target hip --token "$TOKEN" \
  --cache-dir /tmp/jotnar-gpu-runner-cache --mdns
```

CUDA host:

```sh
env NVCC="$HOME/.local/lib/python3.12/site-packages/nvidia/cu13/bin/nvcc" \
  CUDA_ARCH=sm_120 \
  ./jotnar-gpu-runner --bind 0.0.0.0:9997 --target cuda --token "$TOKEN" \
  --cache-dir /tmp/jotnar-gpu-runner-cache --mdns
```

`--no-warm` disables persistent workers. `--warm-cache-workers N` bounds the
worker pool (default 8, least-recently-used eviction).
`--payload-cache-bytes` bounds parsed payloads (default 256 MiB), and
`--artifact-cache-entries` bounds compiled artifacts (default 64, LRU).
Workers use `/dev/shm` when available; at startup the runner sweeps
directories owned by dead processes, plus unowned directories older than ten
minutes (the pre-owner layout), and shutdown kills a worker after a two-second
grace period.

Client and two-host comparison:

```sh
python3 scripts/gpu_runner_client.py --addr HOST:9997 --token "$TOKEN" \
  --target hip --listing job.asm --payload payload.json --outputs OUT

python3 scripts/gpu_runner_compare.py \
  --a 192.168.1.195:9997 --target-a hip \
  --b 192.168.1.116:9997 --target-b cuda \
  --token "$TOKEN" --listing job.asm --payload payload.json --outputs OUT
```

## Boundaries

- Compile-on-host with a source/arch cache, bounded parsed-payload and compiled
  artifact caches, and binary output transport; no signed/prebuilt artifact
  transport yet.
- Trusted-LAN shared token; no TLS/mTLS. A non-loopback bind requires
  `--token`, and an empty token is rejected.
- No GPU passthrough into the Alpine node.
- Target-aware by design; not part of the model-blind node or the language
  contract.
- Current slices: HIP 51 CUDA-equivalent mnemonics; CUDA 65. Integer results
  can be compared exactly; float cross-vendor results need tolerance.
- The target-aware scheduler lives in this crate (`jotnar-gpu-net`) and stays
  separate from `jotnar-net`; capability exchange is still an open design
  choice.

## Scheduler

`jotnar-gpu-net` is the target-aware controller for these runners. It is a
companion to `jotnar-net`; the CPU node remains model-blind.

```sh
# auto-discover advertised runners
./jotnar-gpu-net --discover --discover-timeout-ms 3000 --token "$TOKEN" health

# health of each explicit runner
./jotnar-gpu-net --runner hip=192.168.1.195:9997 \
  --runner cuda=192.168.1.116:9997 --token "$TOKEN" health

# run the same job on both targets and compare
./jotnar-gpu-net --runner hip=192.168.1.195:9997 \
  --runner cuda=192.168.1.116:9997 --token "$TOKEN" \
  run job.asm --payload payload.json --outputs OUT \
  --target both --count 20 --parallel 4 \
  --ledger artifacts/gpu-batches --batch-id demo --max-attempts 3
```

`--target hip` or `--target cuda` runs only that target. `both` runs one
runner per target and compares outputs using the scheduler's `atol`/`rtol`.
Repeated `--runner` entries form a pool; jobs are assigned round-robin across
the pool for each selected target. `--count` submits N copies and `--parallel`
sets the number of concurrent job workers. `--batch-frames N` sends up to N
jobs per `run-batch` request (default 1, maximum 64); `--parallel` is the
number of concurrent chunks. `--prewarm` prepares every selected runner before
the measured batch, including one warm-up frame, so worker startup and lazy
kernel initialization are reported separately as `prewarm_ms` and excluded
from `wall_ms`. `--ledger` defaults to `artifacts/gpu-batches`; a
repeated `--batch-id` resumes pending jobs and replays a completed batch from
its cached summary. Failures retry across the target pool up to
`--max-attempts`. If a batch request fails or the runner does not support it,
those jobs fall back to the normal per-job request path. Within a batch, a
runner that fails a request is skipped by later jobs and chunks; if every
runner for a target has failed, the scheduler tries the group again.

The controller uploads the batch payload once per runner with
`payload-put`, then sends only the payload id per job. The runner caches parsed
payloads and compiled artifacts; large outputs use binary transport. The summary
reports `batch_frames`, `chunks`, and `warm_runs` alongside wall time and
throughput.

## Warm frames

Warm frames are implemented for CUDA and HIP. The runner uses a persistent
worker automatically for eligible artifacts; `--no-warm` forces the old
process-per-job path. The ABI and boundaries are documented in
[`docs/GPU_WARM_FRAME_ABI.md`](../../docs/GPU_WARM_FRAME_ABI.md).

## Gate

[`scripts/gpu_warm_gate.py`](../../scripts/gpu_warm_gate.py) warms each
artifact, runs uncached repeats in per-job and batch modes, verifies identical
outputs, and reports medians and p95:

```sh
./scripts/gpu_warm_gate.py \
  --runner hip=192.168.1.195:9997 \
  --runner hip=192.168.1.195:9996 \
  --runner cuda=192.168.1.116:9997 \
  --runner cuda=192.168.1.116:9996 --token "$TOKEN" \
  --probe-dir artifacts/gpu-runner/warm-frame/probe \
  --repeats 5 --parallel 4 --prewarm
```

Batch mode auto-sizes frames to `ceil(count / parallel)`. The gate fails unless
every expected run used a warm worker; pass `--allow-cold` for a process-per-job
pool. `--prewarm` removes worker startup from the measured wall time. Use batch
frames for small jobs where connection overhead dominates;
large outputs remain bandwidth-bound and usually favor the per-job path.
