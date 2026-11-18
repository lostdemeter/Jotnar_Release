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
 "runtime_version":"...", "build_ms":0, "run_ms":12}
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
  ./jotnar-gpu-runner --bind 0.0.0.0:9997 --target hip --token "$TOKEN"
```

CUDA host:

```sh
env NVCC="$HOME/.local/lib/python3.12/site-packages/nvidia/cu13/bin/nvcc" \
  CUDA_ARCH=sm_120 \
  ./jotnar-gpu-runner --bind 0.0.0.0:9997 --target cuda --token "$TOKEN"
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

- Compile-on-host; no signed/prebuilt artifact transport yet.
- Trusted-LAN shared token; no TLS/mTLS.
- No GPU passthrough into the Alpine node.
- Target-aware by design; not part of the model-blind node or the language
  contract.
- Current slices: HIP 51 CUDA-equivalent mnemonics; CUDA 65. Integer results
  can be compared exactly; float cross-vendor results need tolerance.
- No scheduler integration yet; the next step is target-aware placement in
  `jotnar-net`.
