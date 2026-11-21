# Vulkan compute feasibility spike (Steam Frame)

Date: 2026-11-20
Status: **spike complete; correctness-only, ladder-first**
Host: Steam Frame (`192.168.1.164`), aarch64
License: GPL-3.0-only. No CI was added.

## Question

Can Jotnar run on the Steam Frame's GPU, and what is the smallest honest
backend that proves it?

Answer for this spike: **yes**, with a direct Vulkan-compute emitter and a
bit-exact gate against the non-FPU reference. The primary numeric path is the
**φ-ladder (T)**, not FP32. This is not "all 105 mnemonics on Vulkan", not a
performance result beyond the numbers below, and not a node-image
integration.

## Boundaries

This spike did **not**:

- implement T2 beyond `MUL`/`SQUARE`/`ADD`/`SUB` (no T2 `MATMUL`, norms,
  softmax, or image ops);
- implement batched matmul, broadcasting, reductions, norms, softmax,
  rotary, or the image ops;
- support T streams in the CUDA/HIP emitters (unchanged) or change any
  existing backend;
- claim frame-pacing, power, or thermal results;
- add CI;
- push or merge any branch.

Branch state: three commits on `vulkan-backend` (`8147d22`, `32172cd`,
`4d33d2b`, `4444340`).

## Host and device verification

| item | observed |
|---|---|
| OS | SteamOS 0.5.9 (`holo`, `ID_LIKE=arch`), BUILD_ID 20261113 |
| kernel | `7.2.0-g344f27110f05` aarch64 |
| CPU | 8 cores, 15 GiB RAM |
| storage | `/` 10 G read-only, `/home` 927 G writable |
| GPU | `Turnip Adreno (TM) 750` (`freedreno_icd.aarch64.json`) |
| Vulkan | loader 1.4.363, device apiVersion 1.4.362 |
| shaderc | `glslc 2026.1` (shaderc 1.4.341) |
| compiler | `cc` GCC 15.2.1 |
| Rust | 1.100.0 via `rustup` into `~/.cargo` (rootfs untouched) |

Device features that decided the numeric model:

```text
shaderFloat64 = false     -> F streams are f32, no f64 path
shaderInt64   = true      -> I and T stay exact 64-bit
shaderInt16   = true
maxComputeWorkGroupInvocations = 2048
maxComputeSharedMemorySize     = 32768
maxStorageBufferRange          = 134217728
maxPerStageDescriptorStorageBuffers = 16777216 (effectively unbounded)
```

`glslc` and `gcc` are preinstalled, so the frame needs no cross toolchain:
builds run natively on the device.

## Chosen approach

**Direct Vulkan compute target.** `src/codegen/vulkan.rs` emits GLSL, runs
`glslc`, and embeds the resulting SPIR-V in a generated C host. The runtime
needs only `libvulkan`; there is no shader compiler on the target. This
matches how `build_rust()` already requires `rustc` and `build_cu()` requires
`nvcc`.

Wiring is the existing seam: `Target::Vulkan`, `compile_vulkan()`,
`build_vulkan()` (`cc -lvulkan`).

Design for this spike:

- one compute dispatch per instruction, one command buffer, a global memory
  barrier between dispatches, one submit, one wait;
- host-visible coherent buffers, no staging copies;
- buffers prefer `HOST_CACHED|HOST_COHERENT` (see measurements);
- `CONFIG serve_input`/`serve_output` emits a resident stdin/stdout loop that
  initializes Vulkan once.

Streams are multi-plane:

| kind | planes | GPU element |
|---|---|---|
| T | `s`, `e`, `z` | i32, i32, u32 |
| I | one | i64 |
| F | one | f32 |

Planes are 32-bit on the GPU even when the file format is `i8`/`u8`, so no
`shaderInt8` / `shaderInt16` feature is required. Conversion happens at the
I/O boundary.

## φ-ladder semantics that the gate corrected

Two assumptions were wrong until the parity gate disagreed:

1. **ADD/SUB do not align to the larger rung.** Frozen T uses the declared
   `CONFIG m_cov` scale (default `33280`) as a fixed bridge:
   `from_fixed(to_fixed(a, m_cov) ± to_fixed(b, m_cov), m_cov)`. The max-rung
   alignment in `T2_PHI_FLOAT.md` belongs to T2, not frozen T.
2. **MATMUL sums per-term ladder products**, each converted at `m_acc`, then
   calls `from_fixed` once. It is not `to_fixed(a) * to_fixed(b)`.

`from_fixed` is followed exactly, including the zero case (`s=-1`,
`e = m0 + coarse[45]`), so all three planes match.

Tables (`frac` 13313, `coarse` 193, `fine` 16384) are generated from
`phi_core::lattice` at emit time and uploaded as SSBOs, so they cannot drift
from the reference.

## i128

**Frozen T needs no 128-bit arithmetic.** The reference bounds are
`assert_bound < 2^62` and `from_fixed < 2^53`; both fit in `i64`, and the
shader saturates at `2^53` rather than trapping.

**T2 also needs none**, which was the point of the experiment. The reference
uses `i128` in `to_fixed`/`from_fixed` because `(a << M) / base` can reach
~2^77, and naively that means emulating 128-bit division on the Adreno. Two
facts remove the need:

- `base` is capped by the hi-ladder table at `φ^16 · 2^F` (~2^35), and
- the mantissa is clamped to `[-2^(M-1), 2^(M-1))`, so once `a/base` leaves
  `[0.5, 1.5)` the result is the clamp bound and the exact quotient is
  discarded.

So an explicit clamp test before the division keeps the operand in 64 bits.
Checked against the reference on **43,488 vectors** (powers of two, their
neighbours, and random `q` across `m0 ∈ {31344, 32768, 33280, 38000, 39000,
65535}`): **zero mismatches**, and `a << M` peaks at **60 bits** inside the
non-clamped branch. `MUL` also stays in i64: `ma*mb` fits, and
`round_div_signed` is exact.

This holds for the documented `M == F = 24`; it is not a claim for arbitrary
widths.

## Supported opcode set

```text
T   (frozen): MUL SQUARE ADD SUB MATMUL
T2  (CONFIG t2 M): MUL SQUARE ADD SUB
I:  IADD ISUB IMUL IMATMUL
F:  MATMUL
```

Everything else refuses with a typed `Unimplemented` error. In particular T
streams are accepted (unlike CUDA/HIP, which refuse them), but only for the
ops above.

## Parity gate

`verify_backend.py` (kept outside the repo during the spike) builds a listing
with a given target, replays the reference interpreter, and compares every
output plane. T comparisons are exact; F comparisons honour the manifest
dtype (`f32`) with a tolerance.

| listing | non-FPU | Vulkan |
|---|---|---|
| `MUL` | PASS | PASS (bit-identical) |
| `ADD` | PASS | PASS (bit-identical) |
| `SUB` | PASS | PASS (bit-identical) |
| `ADD` then `SUB` (intermediate stream) | PASS | PASS (bit-identical) |
| `MATMUL` (T) | PASS | PASS (bit-identical) |
| `IADD`/`IMUL`, `IMATMUL`, `MATMUL` (F) | PASS | PASS |
| T2 `MUL` then `ADD` (all four planes) | PASS | PASS (bit-identical) |

## Measurements

1,048,576 elements, `S = MUL(A,B)` then `OUT = ADD(S,B)`, three-plane T
streams, median of 5 runs.

| path | time |
|---|---|
| non-FPU, one process | 19.4 ms |
| Vulkan, one process | 37.9 ms |
| Vulkan, resident (`serve`) | **8.04 ms/frame** |

Vulkan one-shot phase breakdown:

```text
init=9.6  buffers=12.7  load=7.1  pipelines=0.2  dispatch=3.4  store=4.0 ms
```

- GPU dispatch is **3.4 ms**, faster than the CPU's entire 19.4 ms run.
- T2 vs frozen T, same listing and size, resident:

  | format | planes | time |
  |---|---|---|
  | frozen T | 3 | 8.70 ms/frame |
  | T2 | 4 | **10.80 ms/frame** (1.24x) |

  T2 is cheap on the GPU because the frame is I/O-bound at this size; the
  ~2.5x CPU ratio in `T2_PHI_FLOAT.md` does not carry over.
- The first measurement had `store=125 ms`: reading results out of
  write-combined coherent memory cost more than everything else combined.
  `HOST_CACHED|HOST_COHERENT` fixed it.
- Per-process setup (~30 ms) dominated the one-shot number; the resident
  loop removes it. Cold dispatch was 10.9 ms vs 3.3 ms warm.

Build times: Vulkan 0.78 s, non-FPU 0.37 s (shader compilation + `cc`).

## Known limits

- Rank-2 `MATMUL` only; no `BATCH_MATMUL`.
- No broadcasting between different element counts.
- No reductions, norms, softmax, rotary, or image ops.
- T2 covers `MUL`/`SQUARE`/`ADD`/`SUB` only; no `T2I`/`IGET`, no
  `ROTARY`/`TSHIFT`.
- No bounds-fault channel yet: over-range accumulators saturate instead of
  failing the run. The reference raises an error.
- Host I/O is scalar conversion plus a copy; a real embedding would write
  straight into the planes.

## Next

1. T2 `MATMUL` (block-floating two-pass), which is the shape the Qwen block
   needs; the bridge and mantissa arithmetic already exist.
2. `RMSNORM`, `ROTARY`, `TSHIFT`, `TBETA`, `SOFTMAX_WIDE`, `SILU`,
   `TRANSPOSE`, `SLICE` — the rest of the Qwen block.
3. The Qwen block listing as the acceptance gate, plane-for-plane against
   non-FPU.
4. Move the parity harness into the repo as `scripts/parity_vulkan.py`.

## Reproduce

```sh
# on the frame
rustup toolchain install stable
git clone https://github.com/lostdemeter/Jotnar_Release.git
cd Jotnar_Release && cargo build --release

./target/release/jotnar build listing.asm --payload p.json \
    --target vulkan --outputs OUT --workdir /tmp/vk
/tmp/vk/prog <input planes...> <output planes...>
```
