---
title: Images, Fields, and Vision
number: "09"
kicker: REFERENCE / THE INSTRUCTION SET
lede: An image is a stream with channels. A field is a stream with a geometry. The image family is where the two meet.
columns: 2
---

The image family works on `H×W` and `H×W×C` streams. The usual pipeline is:

```text
U8:HWC  --SRGB_DECODE-->  F:HWC  --LUMA-->  F:HW
F:HW    --SQRT-->         T:HW   --SPLAT_BLUR-->  T:HW + T:HW
T:HW    --BETA-->         T:HW   --GAIN-->        F:HWC
F:HWC   --SRGB_ENCODE-->  U8:HWC
```

Every arrow is a stream. Every stage can be inspected, replaced, or run on a different target. There is no hidden image object.

## Color and luma

| Mnemonic | Signature | Behavior |
|---|---|---|
| `SRGB_DECODE` | `U8:HWC -> F:HWC` | Normalize bytes to `[0,1]`, then `pow(2.2)`. |
| `SRGB_ENCODE` | `F:HWC -> U8:HWC` | `pow(1/2.2)`, scale to 0..255, clamp. |
| `LUMA` | `F:HWC -> F:HW` | Rec.709 weights: 0.2126, 0.7152, 0.0722. |
| `GAIN` | `F:HWC, F:HW, T:HW -> F:HWC` | Coherence-gated gain. |

`sRGB` is not a linear-light space. Decoding before filtering and encoding after is the difference between a blur that looks right and a blur that looks muddy. The gamma is a fixed 2.2 approximation, not the piecewise sRGB curve; that is the frozen contract.

## Filtering

| Mnemonic | Signature | Behavior |
|---|---|---|
| `GAUSS` | `$A, F:SCALAR, F:SCALAR -> $A` | Gaussian blur with radius and sigma. |
| `ISO_BLUR` | `T:HW -> T:HW` | Fixed 5×5 Gaussian (radius 2, sigma 1). |
| `CONV` | `*, * -> *` | 2D convolution with an explicit kernel. |
| `CONV2D` | `*, *, *, F:SCALAR, F:SCALAR -> *` | Multi-channel HWC convolution with stride, padding, and bias. |
| `DECONV` | `*, *, F:SCALAR, F:SCALAR -> *` | Transposed convolution with stride and padding. |
| `DECONV2D` | `*, *, *, F:SCALAR, F:SCALAR -> *` | Multi-channel HWC transposed convolution with stride, padding, and bias. |
| `INTERP` | `*, F:SCALAR, F:SCALAR -> *` | Bilinear resize by dyadic scales. |
| `RESIZE2D` | `*, F:SCALAR, F:SCALAR, F:SCALAR -> *` | Bilinear resize to an explicit size with an align_corners flag. |
| `POOLAVG` | `* -> *` | Global average pool over H,W; returns `(C)`. |
| `PRELU` | `$A, $A -> $A` | PReLU with scalar or per-channel slope. |

`INTERP` only accepts dyadic scales (powers of two). A non-dyadic scale is a lowering error, not a silent approximation. That is a deliberate constraint: it keeps the fixed-point coordinate math exact.

## Structure and coherence

`SPLAT_BLUR` is the big one. It computes a structure tensor (Sobel gradients, then a Gaussian window), derives coherence, buckets the local orientation, and blends several oriented splats. It returns two streams: the filtered image and the coherence field.

| Mnemonic | Signature | Behavior |
|---|---|---|
| `SPLAT_BLUR` | `T:HW -> T:HW, T:HW` | Structure-tensor splat blur; image + coherence. |
| `BETA` | `$A -> $A` | Fill with the scalar `beta`. |
| `BETA_V5` | `T:HW, T:HW -> T:HW` | Beta field v5 from depth and coherence. |
| `MIXDYAD` | `T:HW, T:HW, I:HW -> T:HW` | Blend current and history under a mask. |
| `WARP` | `T:HW, F:HW2 -> T:HW` | Bilinear warp by a flow field. |
| `STATIC` | `F:HW2 -> I:HW` | Mask true where flow magnitude is zero. |

Coherence is a number in `[0,1]`: how strongly the local gradient points in one direction. The beta field uses it to decide how much to trust the current estimate versus the history. `BETA_V5` adds a depth term. `MIXDYAD` blends under a mask, and `WARP` resamples the previous field by a flow.

::: figure src="assets/img/portrait-halftone.png" caption="A face through the image path, reduced to a halftone. The original is a 360×288 photograph; the portrait is not the point, the pipeline is. Image: Jotnar samples/input_example.png."
:::

## The 43-op demo

The image demo runs one 43-op geometric listing through C, non-FPU, and CUDA. It is the shortest way to see the same graph on three machines:

```sh
cargo run --release --bin jotnar-demo-img -- \
  --input samples/input_example.png --gallery gallery
```

The gallery contains the C output, the non-FPU output, and the CUDA output. The point is not that the images are pixel-identical across every target; floats can differ in the last ulp. The point is that the graph is the same, the shapes are the same, and the exact moves are exact.

::: note [WHAT “COHERENCE” IS NOT]
Coherence is not confidence. It is a local orientation measure. A high-coherence edge is a direction, not a truth claim. The beta field uses it as one input among several; it does not turn it into a probability.
:::

## Why images belong in an assembly manual

Because the image path is where the language’s constraints become visible. You cannot blur a `U8:HWC` stream directly; you decode first. You cannot resize by an arbitrary scale; you use a dyadic scale or you get an error. You cannot treat a triple as a float; you decode at the boundary. The image family is a stress test for the type system, and it passes by refusing the wrong graph. The field-note version of the 43-op run is Chapter 17.

::: pullquote [IMAGE RULE]
Decode at the edge, filter in the middle, encode at the edge. The middle is where the exactness lives.
:::

::: end
:::
