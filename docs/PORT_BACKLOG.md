# Port backlog and example curation

This is the working plan for turning the Python Jotnar surface into a
Rust-owned, carefully curated project. It is a backlog, not a promise to
port every historical script.

Guiding rules:

1. Port **capabilities**, not every one-off research script.
2. Every public example must own a distinct concept. If two examples teach
   the same thing, keep the smaller/clearer one and archive the other.
3. Keep the language, tools, examples, demos, machines, research, and assets
   separate even while they live in one workspace.
4. Prefer examples that can run in seconds with checked-in or small
   downloaded data. Put the heavy demos behind a clear “needs X” label.
5. Use the Python implementation and its gates as the oracle until a Rust
   replacement has its own differential gate.

## Current Rust baseline

Done:

- 90-mnemonic registry, parser/expander/assembler/verifier, interpreter,
  `REPEAT`/`BREAK`, `repeat()`.
- C, non-FPU, and CUDA code generators (correctness-first).
- `phi-core` crate: lattice, triples, exact helpers, numpy-style substrate
  ops, calibration/scoring.
- Differential harnesses: ops, programs, emitters, CUDA.

Open:

- Verilog and device emitters.
- Builder.
- Debugger, profiler, census, fusion advisor.
- Spatial map / refinement.
- Serve/live, codec ABI, graphs, protocol.
- RV32/RV64/FP machine models and host loops.
- Curated examples/demos and the final directory split.

## 1. Jotnar-LAN / language core

- [x] `asm.py`: IMPORT, DEF/CALL, REPEAT/BREAK, parse, assemble, verify
- [x] `asm_ops.py`: all 90 opcodes and signatures
- [x] `run()` / `repeat()` reference interpreter
- [ ] `builder.py`: programmatic listing construction
- [ ] `debugger.py`: breakpoints, watch, step, lesion
- [ ] `census.py`: stream census / coverage
- [ ] `geo_profiler.py`: static + traced cost attribution
- [ ] `fusion.py` / `fuse_exec.py`: fusion advisor and fused executor
- [ ] `ranges.py`, `oracle.py`, `traces.py`
- [ ] `qtypes.py`, `composite.py`, `fields.py`
- [ ] `serve.py`, `batch.py`, `codec` mode, protocol
- [ ] trace summaries and causal `overrides`

## 2. Jotnar-MAC / targets

- [x] C99 (`emit_c.py`)
- [x] integer-only C (`emit_nonfpu.py`)
- [x] CUDA FP32 (`emit_cuda.py`), correctness-first
- [ ] Verilog combinational/sequential backend (`verilog.py`)
- [ ] device emitter / direct-encoding path (`emit_device.py`)
- [ ] spatial map + refinement (`spatial.py`, `refine.py`)
- [ ] RV32/RV64/FP machine models and host loops
- [ ] OS, desktop, and DOOM adapters

## 3. phi-core

- [x] lattice, triples, fixed-point bridge, LUTs
- [x] exact compare/clip/bin helpers
- [x] matmul, norms, softmax, SiLU, GELU, scan, gather, avgpool
- [x] conv, deconv, warp, interpolate, groupnorm, IP, tiled scan
- [x] calibration (`m_of`, batchnorm fold) and scoring (PSNR, lattice bars)
- [ ] `compose`, `probe`, `zoo`, `stamp`
- [ ] migrate the remaining image-specific substrate helpers out of
      `jotnar/src/ops/` into `phi-core`
- [ ] decide whether the Rust project owns C/CUDA/SIMD substrate lowerings
      or treats the existing `phi-core/c_core` as an external reference

## 4. Tools

- [ ] builder CLI / Rust API
- [ ] debugger
- [ ] profiler / census / cost model
- [ ] spatial map + refined Jotnar map viewer
- [ ] Verilog co-simulation harness
- [ ] codec/serve/protocol tools
- [ ] trace viewer
- [x] `check`, `verify`, `run`, `repeat`, `emit-c`, `build`, `ops`
- [ ] `npy` / raw-binary payload adapters

## 5. Example listings

Existing corpus to pick from:

- `programs/shapes/`: `saxpy`, `saxpy_double`, `matmul2`, `matmul2_sum`,
  `reduce`, `blur`, `gray`, `copy4`, `copy4_sum`, `select`
- `stdlib/attention.asm`, `stdlib/mlp.asm`
- `programs/holo_flagship.asm`
- `programs/xf_block.asm`
- `programs/bigram_lm.asm`
- `programs/lm_headt.asm`, `lm_d32_headt.asm`, `lm_d64_headt.asm`
- `programs/tensor_disc.asm`, `stabilize_mgd.asm`, `scan_demo.asm`
- `programs/crc32.asm`, `gf_mul8.asm`, `ntt_bfly.asm`
- `programs/cpu8.asm`, `monitor.asm`
- `programs/rv32.asm`, `rv64.asm`, `rv64_fp_*.asm`
- `programs/jotnar_on_jotnar/small.asm`
- old backup corpus: BPE, assoc memory, attention variants, depth-4,
  dual-head, siphon, yarnball (archive source, select only if unique)

Curation rule: no two public examples should demonstrate the same lesson.

## 6. Demos

### Tier 1 — first-contact, high-signal, low dependency

- Image pipeline: one 43-op listing → C / CUDA / non-FPU, with gallery
- Bigram LM: smallest complete “creature” (`GATHER` + `ARGMAX`)
- CPU8 → Verilog: language as structural hardware
- Jotnar-on-Jotnar: listing → VM and emitted C, bit-exact
- Direct encoding: map is the placement; fused CUDA speed evidence

### Tier 2 — strong but dependency-heavy

- Transformer block / headt / D32 / D64
- Qwen chat + decode/prefill serving
- DOOM on RV32, native → GPU migration
- Desktop on RV32
- Linux/X/VNC on RV64
- Depth/SBS video (`depth_revisited`)
- Lachesis/Atropos board witness (`depth_jotnaros_experiment`)
- File repair via Jotnar codecs (`fillin`)

### Tier 3 — research / historical

- old 21-demo LLM suite in `holographic_enhancement_backup`
- contract model, shape/field guide, Yarnball 2.0
- board v2 and vector/search/roles/method experiments
- Echion board port, Mandelbrot, misc ISA diagnostics

## 7. Machines and cores

- [ ] CPU8 listing, model, monitor, shell
- [ ] RV32IM datapath + native/GPU kernels
- [ ] RV64IMAFDC datapath + FP listings
- [ ] Linux boot, X, VNC, desktop, DOOM
- [ ] Mandelbrot demo
- [ ] Jotnar-on-Jotnar build

## 8. Applications

- [ ] image/temporal demos (from `holographic_enhancement`)
- [ ] depth/stereo/SBS (`depth_revisited`)
- [ ] Lachesis/Atropos (`depth_jotnaros_experiment`)
- [ ] file repair codecs (`fillin`)
- [ ] Qwen chat/serve/protocol

## 9. Assets and data

- `jotnar-assets/`: `doom1.wad`, `doom-demo.mp4`, `desktop-demo.mp4`,
  `doomgeneric/`
- galleries and sample images
- depth results/visualizations
- fillin result artifacts
- keep large model/checkpoint data out of the language repo

## First story — completed

1. **Builder + the small language tour** — done
   - `src/builder.rs`: `Prog`, declarations, ops, DEF/CALL, stdlib composites.
   - `showcase/language/`: `saxpy`, `reduce`, `matmul2`, `select` + guide.
   - `src/bin/jotnar-demo-tour.rs`: one-command interpreter tour.
   - Evidence: generated image listing is byte-identical to Python;
     `tests/builder.rs` passes.

2. **Image pipeline demo** — done
   - `src/showcase/image_demo.rs`: builds the 43-op pipeline, runs C,
     non-FPU, and CUDA, compares outputs, and writes the gallery.
   - `src/bin/jotnar-demo-img.rs`.
   - Evidence: listing equals Python `demo_img.py` text; non-FPU PSNR
     56–69dB; CUDA-vs-C maxabs ~1e-4.

3. **Verilog + CPU8** — done
   - `src/verilog.rs`: combinational + sequential emitter and plan-level
     simulator, supporting the integer subset, compares, memory, and ICRC32.
   - `src/bin/jotnar-verilog.rs`.
   - `tests/verilog.rs`: CPU8 90-cycle trace matches the Python
     `chain.verilog`/`cpu8_model` reference exactly.
   - Evidence: emitted CPU8 Verilog is structurally identical up to map
     iteration order; all architectural state and RAM match.

4. **Jotnar-SIM analog field** — done
   - `src/showcase/analog_demo.rs`: 2D RC field, DAC/ADC sweeps, settling,
     noise, lane vector, tolerance gate, and an RJM-shaped artifact.
   - `src/bin/jotnar-demo-analog.rs`; `showcase/analog/`.
   - Evidence: ideal RC matches digital Laplace (1.39e-6 hole L1);
     8-bit DAC passes the tolerance gate; `tests/analog.rs` passes.
   - Framing: Jotnar-SIM, not Jotnar-ANA. The live path needs a DAC; the
     baked-map profile removes it from the *application* path only.

## Second leg — started

1. **Futhark as one inference unit** — done
   - `src/showcase/futhark.rs`, `src/bin/jotnar-demo-futhark.rs`.
   - Trained 117,505-entry lexicon + compiled 4-character fallback table +
     phoneme/rune table, all ordinary Jotnar `I` payload streams.
   - `scripts/parity_futhark.py`: 118/118 exact against the original
     Python/JS rule reference (known words, contractions, unknown words).
   - Boundary: word-level unit; DOM/text rendering stays host-side.

2. **Protocol determinism** — done
   - `src/protocol.rs`: frame codec, model-blind `Node`, program registry,
     blob shelf, typed EXEC, in-process worker threads.
   - `src/showcase/protocol_demo.rs`, `src/bin/jotnar-demo-protocol.rs`.
   - `tests/protocol.rs` (5) + `tests/protocol_demo.rs` (1).
   - Evidence: 3 batches, 12 independent jobs, 4 nodes, different
     completion/placement orders, byte-identical outputs; 5/5 protocol
     unit tests green.
   - Boundary: framed (in-process and TCP) transport over the same
     wire contract and model-blind node; serve-mode resident loops,
     codec ABI, CUDA graphs, and packed outputs are NOT done.

3. **DOOM on RV32IM** — native core done
   - `src/rv32.rs`: RV32I+M, CSR/Sv32, DOOM MMIO, checkpointing, frame output.
   - `src/bin/jotnar-demo-doom.rs`, `showcase/doom/`.
   - `scripts/parity_doom_native.py` and `tests/rv32.rs`.
   - Evidence: frame 1 is **24,502,445 instructions, PC 0x00048860**, and
     byte-identical to the C native core; the Rust frame-1 checkpoint
     resumed to frame 2 is byte-identical to the CUDA GPU checkpoint and
     framebuffer.
   - Not ported: `rv32_kernel.cu` fleet kernel, full DOOM toolchain.
   - Follow-up: the Rust SDL2 player has intermittent frame-time spikes not
     seen in the original Python/C player. Record a frame-time trace and
     gate p99/max before calling the frontend finished.

4. **Analog filter performance/limits** — done
   - `src/showcase/analog_filter.rs`, `src/bin/jotnar-demo-analog-filter.rs`.
   - `tests/analog_filter.rs`; `showcase/analog_filter/` includes the panel
     and metrics from a full run.
   - Evidence: synthetic optimum PM rational `k=0.08` gives hole 0.0686
     vs Laplace 0.0715 (~4% relative) at ~10× simulated cost; real-hole
     reference is worse for PM (0.1777 vs 0.1729).
   - Framing: quality, simulated cost, and physical estimate are separate;
     the positive hybrid remains surface + Laplace seam correction.

## Trust spine — current

- `docs/LOOP.md` — the loop and vocabulary.
- `docs/appendix/TRUST_SPINE.md` — certificate schema and first slice.
- `crates/jotnar-lab` — `Certificate`, `Verdict`/`Outcome`/`Status`,
  `Gate`, `Refusal`, falsifiers, nulls, benches, verify.
- `certificates/doom/native-gpu-migration.json` — first certificate.
- Evidence: `certify doom` reads existing native/GPU artifacts; `falsify
  doom` includes `not_caught`/vacuity rows; `null doom` includes a confounded
  title-frame control; `bench doom` attaches measured p50/p99/max; `verify`
  passes with hashes and reports `BLOCKED` when artifacts are absent.
- Boundary: trust spine belongs in the appendix/tooling, not the language.

## Next wave

After those:

5. Bigram + transformer examples (`bigram_lm`, `xf_block`, `lm_headt`);
   then the Qwen chat/serve path.
6. Desktop/Linux RV32/RV64 machine demos (DOOM itself is done).
7. Depth/SBS and Lachesis/Atropos.
8. File-repair (`fillin`) and protocol/pipelined demos.
9. Research/contract/shape/Yarnball material as a separate archive.

## What not to do

- Do not make all 56 old listings public examples.
- Do not ship three image demos, three transformer demos, and three LLM
  demos that teach the same thing.
- Do not put the DOOM stack or Qwen snapshot in the language repo.
- Do not port research scripts before the language, builder, and gates are
  stable.
- Do not call an example “Jotnar” alone; name the aspect (LAN/MAC/WIT) and
  the flavor (DIG/SIM/ANA/HYB) when it matters.


## Rust target

`--target rust` / `emit-rust` is an F/I-subset target (see manual 14-emitters). Needs `rustc` on PATH; emit uses `--edition 2021`. Next slices: CONV/DECONV/INTERP/GATHER, then the geometric DAV2 listing.
