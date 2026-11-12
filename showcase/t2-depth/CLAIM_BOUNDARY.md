# Claim boundary: T2 depth companion showcase

## Decision

The project accepts this companion showcase for inclusion in the language
repository **only as documentation of external application evidence**. It is
not a numbered exhibit, and it does not alter the repository's ten-exhibit
count, registry, manual, or certified gates.

Merging this docs-only branch does **not** promote the full-depth or
performance numbers to language-repo gates. They remain application-reported
evidence from a local-only companion repository.

## What this repository certifies

The following are reproducible from this branch and are the only claims the
language repository makes:

- T2 `CONV2D`, `DECONV2D`, `RESIZE2D`, `RELU` and seam moves `SLICE`,
  `RESHAPE2`, `RESHAPE3` are bit-identical on all four planes to the T2
  interpreter.
- `CONFIG threads N` (1..=256; 0 typed) emits OpenMP over T2 `MATMUL` rows and
  produces identical T2 output.
- **225/225** workspace tests, **26/26** Rust parity, `cargo fmt --check`
  clean, `cargo test --test registry_docs` clean.

## What this repository does not certify

- The composed full-depth correlation/error numbers.
- The production profile and FPS numbers, including the 75.4 fps fast-float
  figure, the 166 s/frame exact-T2 figure, and the SBS/webcam FPS numbers.
- The companion application, model weights, phi bake, fidelity cards, SBS
  videos, or webcam demo.
- Temporal filtering, calibration, JBU, disparity limiting, provenance/learned
  inpaint, or Rust-host behavior.
- That fast-float is non-FPU (it is explicitly waived), that exact T2 is live,
  or that the Rust target accepts T2.

## Evidence provenance

- Source: the companion depth-conversion application, currently local-only.
- Model: production checkpoint `dav2s_mixed_far_best.pt`, real-frame runs at
  112×112 and 518×518.
- Canonical numbers: recorded once in
  [`README.md`](README.md#application-reported-external-companion-repository).
- No companion certificate, model scripts, bake, or reproducible command is
  checked into the language repository.
- The panel is illustrative. It is derived from the repository's existing
  `showcase/image/input.png`; it is not a gate and introduces no new
  third-party asset.

## Acceptance

> **The project accepts the external, non-reproducible application boundary
> above for this companion showcase. The language repository does not certify
> the full-depth or FPS numbers; they are documented external evidence only.**

## Change control

- Do not move the canonical full-depth/FPS numbers into language-gated docs or
  the root exhibit table while the companion application is local-only.
- Do not cite this companion showcase as proof of production or live
  performance, or as a language-repo certificate.
- If the companion application is published with reproducible artifacts, this
  boundary can be replaced by a linked certificate in a separate change. Only
  then should the item be proposed as a numbered exhibit, with the root README,
  `docs/LOOP.md`, manual showcase chapter, and manual output updated together.
