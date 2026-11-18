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

`run`:

```json
{"v":1, "id":1, "type":"run", "target":"hip", "outputs":["OUT"], "auth":"..."}
```

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

- Compile-on-host with a source/arch cache; no signed/prebuilt artifact
  transport yet.
- Trusted-LAN shared token; no TLS/mTLS.
- No GPU passthrough into the Alpine node.
- Target-aware by design; not part of the model-blind node or the language
  contract.
- Current slices: HIP 51 CUDA-equivalent mnemonics; CUDA 65. Integer results
  can be compared exactly; float cross-vendor results need tolerance.
- No scheduler integration yet; the next step is target-aware placement in
  `jotnar-net`.

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
sets the number of concurrent job workers. `--ledger` defaults to
`artifacts/gpu-batches`; a repeated `--batch-id` resumes pending jobs and
replays a completed batch from its cached summary. Failures retry across the
target pool up to `--max-attempts`.
