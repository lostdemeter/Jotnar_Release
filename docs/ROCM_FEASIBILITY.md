# ROCm/HIP feasibility spike

Date: 2026-11-17
Status: **spike complete; correctness-only**
Host: `mintsession2` (`192.168.1.195`)
License: GPL-3.0-only. No CI was added.

## Question

Can the existing Jotnar CUDA lowering be made to run on the RX 7900 XTX
through HIP/ROCm, and what is the smallest honest backend that proves it?

Answer for this spike: **yes, for the CUDA-equivalent opcode set**, with a
direct source-level HIP emitter and an exact C-vs-HIP parity gate. This is not
an "all Jotnar opcodes on AMD" result, not a performance result, and not a
node-image integration.

## Boundaries

This spike did **not** do any of the following:

- change the Alpine node image or the node OS;
- add GPU passthrough to either QEMU node;
- touch, stop, or reconfigure the running QEMU nodes;
- add ROCm to the node payload/runtime;
- implement Claim B or a Jotnar-native AMD kernel;
- claim performance, throughput, or speedup;
- push or merge any branch;
- add CI.

The only host changes were ROCm packages (plus their normal Ubuntu dependency
point-release upgrades) and a host user group membership (`render`, `video`) so
the unprivileged user can open `/dev/kfd` and `/dev/dri/renderD128`. No reboot
was performed.

Local branch state: the spike was originally authored on `38aa759` (based on
`d98dfe2`) and ported onto current `main` as `ec0c05b`. The integration is
merged; this document is added with the follow-up fixes.

QEMU safety note: the host also runs QEMU nodes. The spike did not stop or
reconfigure them.

## Host and GPU verification

| item | observed |
|---|---|
| OS | Linux Mint 22.3 (Zena), Ubuntu 24.04 base |
| kernel | `6.14.0-37-generic` |
| CPU | AMD Ryzen 9 3950X |
| GPU | Navi 31 / Radeon RX 7900 XTX (`1002:744c`), `amdgpu` |
| ROCm | 7.2.4 (AMD `repo.radeon.com` `noble`) |
| HIP compiler | `7.2.53211-97f5574fe2`, AMD clang 22.0.0git |
| HSA runtime | 1.18 |
| `rocminfo` agent | `gfx1100`, 96 CUs, wavefront 32, L2 6 MiB |
| `rocm-smi` | `Radeon RX 7900 XTX`, GFX version `gfx1100` |
| `HSA_OVERRIDE_GFX_VERSION` | **not set / not needed** |

The GPU enumerates as `gfx1100` directly. No `HSA_OVERRIDE_GFX_VERSION`
workaround, no kernel argument, and no DKMS build were required.

### Install commands

```sh
# on mintsession2, from a normal login with passwordless sudo
mkdir -p ~/rocm-spike && cd ~/rocm-spike
curl -fsSLO https://repo.radeon.com/amdgpu-install/7.2.4/ubuntu/noble/amdgpu-install_7.2.4.70204-1_all.deb
sudo apt-get update
sudo apt-get install -y ./amdgpu-install_7.2.4.70204-1_all.deb

# Runtime + hipcc + hipBLAS + rocminfo + rocm-smi-lib.
# --no-dkms keeps the in-tree amdgpu driver; the host already has it loaded.
yes | sudo amdgpu-install --usecase=hiplibsdk --no-dkms

# rocm-smi is shipped in the versioned tree; rocm-utils is installed for the
# supporting tools.  /usr/bin/hipcc and /usr/bin/rocminfo come from
# update-alternatives.
sudo apt-get install -y rocm-utils

# Allow the normal user to open KFD/DRM render nodes without sudo.
sudo usermod -aG render,video mintsession2
```

Log out and back in after the group change (or use `sg render -c ...` for the
current shell).

### Verification commands

```sh
rocminfo | grep -E 'Name:|Marketing Name|Uuid'
hipcc --version
/opt/rocm-7.2.4/bin/rocm-smi --showproductname
```

Observed:

```text
Name:                    gfx1100
Marketing Name:          Radeon RX 7900 XTX
Vendor Name:             AMD
...
HIP version: 7.2.53211-97f5574fe2
...
GPU[0] : Card Series: Radeon RX 7900 XTX
GPU[0] : GFX Version: gfx1100
```

A minimal `hipMalloc` / `hipMemcpy` / `hipLaunchKernelGGL` vector-add was
compiled with `hipcc` and produced `11 22 33 44` before the Jotnar backend was
exercised.

### Installer notes and workarounds

- `amdgpu-install` printed `WARNING: amdgpu dkms failed for running kernel`
  during the `amdgpu-install` package setup. This is expected here: the
  in-tree `amdgpu` module is already loaded and the spike used `--no-dkms`.
  No reboot or module replacement was done.
- The install is under `/opt/rocm-7.2.4`, not `/opt/rocm`; this install did
  not create the unversioned `/opt/rocm` symlink. `/usr/bin/hipcc` and
  `/usr/bin/rocminfo` work through `update-alternatives`. `rocm-smi` is not on
  `PATH`; use `/opt/rocm-7.2.4/bin/rocm-smi`. The Jotnar HIP builder finds
  `hipcc` on `PATH` and falls back to `/opt/rocm/bin/hipcc`; neither workaround
  changes emitted Jotnar code.
- The host packages are ROCm 7.2.4 (`hiplibsdk`) plus `rocm-utils`. MIOpen is
  not required by this spike and was not used.

## Chosen approach

**A. Direct HIP target.** The CUDA emitter is already a Rust source generator
with one kernel per op and a simple host driver. Porting it by replacing the
runtime header, allocator, copy, launch, and build APIs is smaller and more
reversible than adding a hipify step at build time. The CUDA target remains
untouched, so CUDA parity is not put at risk by this spike.

Implemented files:

- `src/codegen/hip.rs` — the HIP emitter (copy/adaptation of
  `cuda.rs`; same kernels and opcode semantics).
- `src/codegen/mod.rs` — `Target::Hip`, `"hip"` parsing, compile and
  build dispatch.
- `scripts/parity_hip.py` — the minimal C-vs-HIP gate.
- `tests/hip_emit.rs` — no-GPU checks for target wiring and emitted
  API surface.

`hip.rs` is deliberately a mechanical port: kernel bodies, index arithmetic,
launch geometry, static-shape assumptions, error codes, and one-launch-per-op
structure are preserved. The only intended semantic difference is FP32
execution on the GPU versus the C target's FP64 host reference, which the
parity gate measures rather than hides.

## CUDA → HIP API mapping

| CUDA surface | HIP/ROCm surface | Notes |
|---|---|---|
| `#include <cuda_runtime.h>` | `#include <hip/hip_runtime.h>` | runtime |
| `#include <cuda_fp16.h>` | `#include <hip/hip_fp16.h>` | `__half`, `__half2float` |
| `#include <cublas_v2.h>` | `#include <hipblas/hipblas.h>` | include only; neither emitter calls cuBLAS/hipBLAS in this spike |
| cuDNN | MIOpen | **deferred**, unused |
| `cudaError_t`, `cudaSuccess`, `cudaGetErrorString` | `hipError_t`, `hipSuccess`, `hipGetErrorString` | error macro |
| `cudaMalloc`, `cudaMemset`, `cudaFree` | `hipMalloc`, `hipMemset`, `hipFree` | device buffers |
| `cudaMemcpy`, `cudaMemcpyHostToDevice`, `cudaMemcpyDeviceToHost`, `cudaMemcpyDeviceToDevice` | `hipMemcpy` and the corresponding `hipMemcpy*` kinds | I/O and SCATTER copy |
| `cudaGetLastError`, `cudaDeviceSynchronize` | `hipGetLastError`, `hipDeviceSynchronize` | after every launch |
| `kernel<<<grid, block>>>(args)` | `hipLaunchKernelGGL(kernel, dim3(grid), dim3(block), 0, 0, args)` | explicit HIP launch |
| `atomicExch`, `rsqrtf`, `expf`, `powf`, `sinf`, `cosf`, `floorf` | unchanged | HIP device intrinsics |
| `nvcc`, `.cu`, `-gencode arch=compute_86,code=sm_86` | `hipcc`, `.hip`, auto-detected offload arch | optional `HIP_ARCH=gfx1100` to pin |
| `-l:libcublas.so.13` | none | no library symbols are emitted |

The CUDA `grid_expr` 65535-block cap is retained. AMD grid-x is much larger,
but keeping the CUDA geometry is part of preserving this as a CUDA-equivalent
port rather than an optimized ROCm emitter.

## Supported opcode set

The HIP target ports the **51-mnemonic CUDA slice as of `d98dfe2`** (including
aliases). Current CUDA has 65 mnemonics; the 14 not in this HIP slice are
`GELU`, `RELU`, `CONV2D`, `CONV2D_RELU`, `DECONV2D`, `RESIZE2D`, `RESHAPE2`,
`RESHAPE3`, `MATMUL_T`, `MATMUL_BIAS`, `MATMUL_BIAS_RESIDUAL`,
`MATMUL_BIAS_GELU`, `SELF_ATTN`, and `LINEAR_ATTN`. HIP is intentionally not
the full 105-mnemonic language:

```text
ADD SUB MUL DIV SILU
BETA TBETA
GATHER ARGMAX GATHER3
MATMUL BATCH_MATMUL BMMV ATTN
RMSNORM LAYERNORM TSHIFT SOFTMAX SOFTMAX_WIDE
ROTARY ROPE SWIGLU
TRANSPOSE SLICE CONCAT SELECT SCATTER TOPK
CONV INTERP
F2I I2F
IADD ISUB IMUL IXOR IAND IOR IEQ ILT IMAX SADD IMAD
ISHL ISHR ISAR IDIV IMOD IDIVU IMODU ISELECT
```

`TBETA` remains untestable in this target because it produces a T stream, and
both the CUDA and HIP emitters refuse T streams (as the CUDA target already
did). `IMATMUL` is **not** in the CUDA target for the same F-only boundary and
is therefore not in this first HIP port. It is a candidate for the next slice,
not a hidden part of this one.

## Parity gate

`scripts/parity_hip.py` builds the same listing twice with the Jotnar
CLI — once with `--target c` and once with `--target hip` — then feeds both
executables the same payload and compares outputs. The C target is the
reference; no Python chain, numpy, or vendor library is used as an oracle.

- Float tolerance: `abs(hip-c) <= 1e-4 + 1e-4 * abs(c)`.
- Missing `hipcc` is a hard failure; the gate exits nonzero rather than
  reporting a green skip.
- Integer outputs: bit-exact.
- The C target stores F streams as `double`; HIP stores them as `float`. The
  observed float differences are the expected FP32 rounding, not an opcode
  disagreement.

Final result on `mintsession2` (32 cases, 2026-11-17):

```text
hip-elem: OK OUT: maxabs=9.748e-08
hip-saxpy: OK OUT: maxabs=1.174e-07
hip-matmul: OK OUT: maxabs=3.608e-07
hip-bmm: OK OUT: maxabs=1.799e-07
hip-rmsnorm: OK OUT: maxabs=2.626e-07
hip-softmax: OK OUT: maxabs=2.777e-08
hip-argmax: OK
hip-int6: OK
hip-isar: OK
hip-layernorm: OK OUT: maxabs=3.003e-07
hip-conv: OK OUT: maxabs=8.149e-08
hip-rotary: OK OUT: maxabs=4.679e-08
hip-rope: OK OUT: maxabs=6.738e-08
hip-transpose: OK OUT: maxabs=2.962e-08
hip-concat: OK OUT: maxabs=2.908e-08
hip-slice: OK OUT: maxabs=2.958e-08
hip-select: OK OUT: maxabs=2.628e-08
hip-scatter: OK OUT: maxabs=2.965e-08
hip-topk: OK V: maxabs=2.951e-08
hip-f2i_i2f: OK G: maxabs=0.000e+00
hip-intdiv: OK
hip-gather: OK OUT: maxabs=2.848e-08
hip-gather3: OK OUT: maxabs=1.873e-08
hip-swiglu: OK OUT: maxabs=2.763e-08
hip-tshift: OK OUT: maxabs=6.267e-08
hip-attn: OK OUT: maxabs=7.575e-08
hip-misc: OK S: maxabs=5.273e-08; D: maxabs=7.849e-08; B: maxabs=0.000e+00
hip-arithint: OK
hip-interp: OK OUT: maxabs=5.992e-08
hip-bitwise: OK
hip-sat: OK
hip-bmmv: OK OUT: maxabs=8.814e-08
FAILURES: none
```

The gate covers every CUDA/HIP pattern family except the T-stream `TBETA`
(there is no T stream in either GPU target). Exact integer cases include
`SADD` near `INT64_MIN`/`INT64_MAX`, shifts, unsigned div/mod, and
`ISELECT`.

Raw evidence is under `artifacts/rocm/` (`parity_hip.log`,
`environment.log`, `parity_hip.py`). Those artifacts are host-generated and
not part of the node build.

### Reproduce

```sh
# build host, in the Jotnar checkout
cargo build --release

# copy the CLI and gate to the ROCm host
ssh mintsession2 'mkdir -p ~/rocm-spike/jotnar/bin ~/rocm-spike/jotnar/scripts'
scp target/release/jotnar mintsession2:~/rocm-spike/jotnar/bin/jotnar
scp scripts/parity_hip.py mintsession2:~/rocm-spike/jotnar/scripts/parity_hip.py

# on the ROCm host
ssh mintsession2
cd ~/rocm-spike/jotnar
python3 scripts/parity_hip.py --bin bin/jotnar
```

The no-GPU wiring tests run in the normal Rust checkout:

```sh
cargo test --release --test hip_emit
```

## CUDA surface not covered

- cuBLAS/hipBLAS calls: neither emitter calls a library today; histograms and
  matmul are custom kernels. `hipblas` is installed and the header is included
  so a later library path does not need an install change.
- cuDNN/MIOpen: unused; fused convolution/MIOpen work is deferred.
- CUDA streams, events, graphs, multi-GPU/peer access, and graph capture.
- Dynamic shapes and dynamic grid sizing (both emitters require static shapes).
- T/triple streams and `TBETA`; `IMATMUL`, `IREDUCE_SUM`, and the rest of the
  non-CUDA opcode surface.
- `REPEAT`/`BREAK` semantics: the CUDA emitter already collapses `REPEAT` to a
  single iteration and ignores `BREAK`. The HIP port preserves that limitation
  rather than hiding it.
- FP16 compute: `--fp16` is storage conversion only, matching CUDA.
- MIOpen, rocBLAS/hipBLASLt tuning, memory pools, and any performance work.

## Viability and next step

The RX 7900 XTX is a viable correctness target for the CUDA-equivalent Jotnar
lowering under ROCm 7.2.4 on this Mint 22.3 host. The GPU is recognized as
`gfx1100` with no override, `hipcc` builds the emitted source, and 32 parity
cases pass with only FP32-vs-FP64 tolerance.

Recommended next slice, still correctness-first:

1. Decide whether `MATMUL`/`BATCH_MATMUL` should call hipBLAS or keep the naive
   kernels; do not mix a performance claim into that decision.
2. Add `IMATMUL`/integer geometry only after defining its HIP semantics and a
   parity gate, since it is outside today's CUDA-equivalent set.
3. Add dynamic-shape and `REPEAT`/`BREAK` support in both GPU emitters
   together, so the targets stay comparable.
4. Keep node GPU passthrough as a separate question. This spike runs on the
   host only.
5. Measure performance only in a later, explicitly labeled milestone.
