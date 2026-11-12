---
title: "Field Notes: Four Exhibits"
number: "19"
kicker: FIELD NOTES / FOUR STORIES
lede: Four exhibits that are deliberately different: a real game, a word-level inference unit, a model-blind fleet, and an honest analog filter.
columns: 2
---

A language is only as convincing as the things people build with it. The full index has ten exhibits; Part IV is a **selected set of four**: the 43-op image pipeline and the DAC → RC → ADC analog field have their own chapters (17 and 18), the certified agent/backend-tuning exhibit lives in `showcase/agent/`, and the interpreter/model-blind node exhibit — now including the deterministic job-server research — lives in `showcase/runtime/`. These four stories are not benchmarks. They are stories about the same stack under different pressures.

## DOOM on the RV32IM core

A real compiled program on a real machine model. `doomgeneric` is built with Zig clang/lld for `riscv32-freestanding -mcpu=generic_rv32+m`, producing an 847,112-byte flat image. The Rust core implements RV32I + M, the CSR/trap surface, and the MMIO the game needs: WAD, framebuffer, clock, frame counter, key ring, UART/CLINT.

```text
frame 1: 24,502,445 instructions, pc=0x00048860
framebuffer: BYTE-IDENTICAL
```

Frame 1 is exact against the C native core. The frame-1 checkpoint can be resumed by the C core, and the frame-2 state is byte-identical to the GPU core’s checkpoint and framebuffer. The player is playable; it is not yet smooth. That is an honest known issue, not a correctness issue.

::: figure src="assets/img/doom-halftone.png" caption="The title screen, halftoned. The image is not the witness; the framebuffer bytes are."
:::

## Futhark: one word, one listing

The Futhark exhibit treats a word as a request to a Jotnar listing. The host encodes characters as integer IDs and renders rune codepoints; the translation decision lives in the program.

The listing contains three things:

1. a 128-bit fingerprint index (FNV-1a plus a polynomial) over a fixed 28-character word, searched in 17–20 binary-search steps;
2. the JavaScript greedy spelling rules compiled into a dense action table over a four-character window, remaining length, `has_output`, and `position==0`;
3. a rune table mapping each phoneme to up to eight codepoints.

Correctness is gated at **118/118 words**, including trained words, exception contractions, and unknown words that exercise the fallback table. The Python oracle was itself checked 118/118 against the actual JavaScript in headless Chromium before it was used as the oracle.

The boundary is explicit: DOM traversal, text splitting, settings, fonts, and the final separator stay host-side. Jotnar does the decision, not the browser.

## Protocol: same bytes, any order

The protocol exhibit keeps the contract and drops the network transport. Nodes are real worker threads behind the same frame codec: `u32BE || JSON header || payload`.

A node never sees “Futhark,” “matmul,” or a model name. It receives a listing, named blobs, an `EXEC` binding of input stream names to blobs or inline byte segments, and output names to return.

The controller runs the same 12 jobs in original order, reverse order, and deterministic shuffled orders, with different node placements and simulated jitter. The completion order changes. The per-job output bytes do not:

```text
original: completion=[0, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1] exact=12/12
reverse:  completion=[11, 0, 3, 6, 9, 1, 4, 7, 10, 2, 5, 8] exact=12/12
shuffle:  completion=[4, 0, 3, 7, 5, 6, 11, 10, 1, 9, 8, 2] exact=12/12
determinism: 3 batches, 12 jobs, 4 node assignments, byte-identical
RESULT: ALL OK
```

The interesting contract is not TCP. It is: **same program + same input bytes = same output bytes, independent of placement, submission order, and completion order.**

## Analog: the honest filter

The analog-filter exhibit asks whether a Perona–Malik anisotropic diffusion can help an image problem better than digital Laplace. It is a simulated PDE experiment, **not** the DAC → RC → ADC field of Chapter 18. It is not Jotnar-ANA, and it makes no claim about physical hardware speed.

| Filter | Hole L1 | Edge-band L1 | Simulated time |
|---|---:|---:|---:|
| Laplace | 0.0715 | 0.1181 | ~0.034 s |
| RC | 0.0715 | 0.1181 | ~0.266 s |
| PM exp `k=0.08` | 0.2380 | 0.1801 | ~0.478 s |
| PM rational `k=0.08` | **0.0686** | **0.1128** | ~0.344 s |
| PM rational `k=0.03` | 0.2072 | 0.2011 | ~0.343 s |
| PM rational `k=0.16` | 0.0708 | 0.1171 | ~0.342 s |

The best synthetic gain is about 4% relative on the hole and 4.5% on the edge band, at roughly **10×** the Laplace solve time. On the real self-occlusion hole, PM is **worse** (0.1777 vs. 0.1729). A local PDE cannot invent the missing sphere side.

::: note [BENCH LABELS]
These are a representative current run on this host. Bench labels: `clock=host_wall` (measured), `scope=end_to_end`, `kind=measured`. The field is simulated, not measured analog hardware. `showcase/analog_filter/metrics.json` is the authority if the numbers move again.
:::

::: figure src="assets/img/analog-filter-duotone.png" caption="TARGET | LAPLACE | RC | BEST PM .08. The red outline is the hole boundary. The duotone is for the zine; the numbers are in the table."
:::

The useful interpretation is the same as the earlier work: not the fill itself; possibly boundary/seam correction around a surface completion; possibly edge-aware depth/JBU filtering; possibly temporal integration. The positive hybrid remains digital surface geometry plus Laplace seam correction.

::: note [WHY INCLUDE A NEGATIVE RESULT?]
Because a manual that only shows wins is marketing. The analog filter is a real experiment with a real limit. The stack is more credible when it reports the limit in the same voice as the win.
:::

::: pullquote [SHOWCASE RULE]
A demo is a claim with a witness. If the witness is missing, it is a poster.
:::

::: end
:::
