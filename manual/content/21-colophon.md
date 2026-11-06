---
title: Colophon
number: "D"
kicker: APPENDIX D / THE END MATTER
lede: Who made this, what it is set in, and what it is licensed under.
columns: 1
---

**JOTNAR FIELD MANUAL** — Issue 01, Digital, First Rough Draft. November 2026.

Published by **The Deterministic Press** for the Jotnar Working Group.

::: note [EDITORIAL]
This is a first rough draft of a manual for a language under active development. The examples are real, the gaps are real, and the assembler is the authority. Corrections should be made against the source, not the PDF.
:::

## Type and tools

| Element | Choice |
|---|---|
| Display | Anton |
| Condensed labels | Barlow Condensed |
| Body | Courier Prime |
| Code | IBM Plex Mono |
| Masthead / quotes | Special Elite |
| Stamps | Stardos Stencil |
| Numbers | Bebas Neue |
| Setting | Markdown, Rust, Paged.js 0.4.3, Chromium |
| Textures and halftones | Deterministic PIL generator (`scripts/make_assets.py`) |
| Page size | 840 × 660 pt (11.67 × 9.17 in), landscape |

The fonts are open-licensed and vendored in `assets/fonts/`. The page size is deliberately the same landscape format as a folded zine spread. The cover, part dividers, and feature openers are full-bleed; the content pages use a measured single column with a red rail, and the contents/register pages use their own grids.

## Licenses

The Jotnar implementation and the code excerpts in this manual are **GPL-3.0-only**, matching the upstream project. The manual text is **CC BY-SA 4.0**. The fonts are under their respective open licenses (SIL Open Font License and Apache 2.0 where applicable). Paged.js is MIT.

The visual language is inspired by xerox zines, 1990s computer magazines, and cut-and-paste hardware manuals. **No reference text, logos, or artwork from any existing publication is reproduced.** The halftone images are derived from the Jotnar project’s own samples and showcase assets.

## Source

The manual source lives beside this PDF:

```text
book.toml
content/
data/ops.tsv
press/
theme/book.css
assets/
scripts/
```

Build it with:

```sh
cargo run --manifest-path press/Cargo.toml --release -- --preview
```

The Jotnar language lives in the parent repository. The Python implementation is the oracle. The parity harness is the witness.

::: pullquote [COLOPHON RULE]
No silent guesses. No reproduced logos. Rebuild the book.
:::

::: stamp [END OF ISSUE 01]
:::

::: end
:::
