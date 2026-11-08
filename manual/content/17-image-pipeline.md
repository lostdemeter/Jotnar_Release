---
title: The Image Pipeline
number: "17"
kicker: FIELD NOTES / THREE TARGETS
lede: One 43-op listing, the same text on C, integer-only C, and CUDA. The image is the evidence, not the benchmark.
columns: 2
---

The image pipeline is the shortest way to see the same graph on three machines. It is one 43-op Jotnar listing, built by `src/showcase/image_demo.rs`, and it is byte-identical to the Python-generated listing. The C, non-FPU, and CUDA outputs are compared in the same run.

```sh
cargo run --release --bin jotnar-demo-img -- \
  --input samples/input_example.png --gallery gallery
```

The gallery contains the C output, the integer-only output, and the CUDA output. The source is a 352×288 photograph; the C strip shows the crop/blur path, and the CUDA image shows the unsharp result.

::: figure src="assets/img/image-targets.png" caption="One listing, three targets: the input, the C crop/blur strip, and the CUDA unsharp result. The claim is the same graph, not pixel-identical floats on every platform."
:::

## What the run does

The listing decodes sRGB, computes luma, builds triples, applies the structure-tensor splat blur and beta field, applies a coherence-gated gain, and encodes back to sRGB. Chapter 10 explains the operations. This page is the exhibit claim.

| Target | What it emits | What the run compares |
|---|---|---|
| **C** | Portable C99, float/transformer path plus exact moves. | The C gallery image and the Python emitter output. |
| **non-FPU** | Integer-only C with baked lattice tables; no `float`, no `double`, no libm. | The integer gallery image and the source-level float trap. |
| **CUDA** | FP32 correctness-first kernels, one launch per op. | Units within 1e-4 of numpy; decisions and exact moves. |

The three outputs are not required to be bit-identical everywhere. Float `F` paths can differ from numpy in the last ulp on some platforms because of a different SIMD reduction order. Triples and integer paths are compared bit-exactly; exact moves are exact.

::: note [WHAT THE FIGURE IS NOT]
The figure is a visual check, not a witness. The witness is the comparison in `scripts/parity_emit.py` and the gallery run. A pretty image can hide a wrong stream; a byte comparison cannot.
:::

## What it proves

- The same 43-op listing can be lowered to three different targets without changing the graph.
- The exact moves and integer/triple paths survive the lowering.
- The non-FPU target can run the image path without a float in the generated source.
- The CUDA target is correct-first: one launch per operation, no tuned-kernel claims.

## What it does not prove

- **Hardware speed.** The CUDA backend is correctness-first. A correctness run is not a performance result.
- **Every opcode on every backend.** The image listing is a selected 43-op path, not the full 90-mnemonic register.
- **Pixel identity across every platform.** Floats can differ in the last ulp; the comparison classes are explicit.
- **That the image is what you meant.** The gate checks the lowering against the reference, not the artistic intent.

::: pullquote [IMAGE EXHIBIT RULE]
The image is the evidence that the graph survived. It is not the proof that the graph is right.
:::

::: end
:::
