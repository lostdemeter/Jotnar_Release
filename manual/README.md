# Jotnar Field Manual

A zine-style manual for **Jotnar**, the geometric assembly language implemented in
[the parent repository](../). The book is written as Markdown, rendered
to HTML by a small Rust press, paginated by Paged.js, and printed to PDF by
Chromium through the DevTools Protocol.

The first draft is **101 pages**, landscape 840 × 660 pt, built around a two-page
spread rhythm: a red cover, a black masthead, a contents page, six part dividers,
a full register of all 94 mnemonics, ten showcase exhibits, and a trust-spine
appendix that records what each claim proves and does not prove.

## Build

Requirements:

- Rust (2021 edition)
- Python 3 with `websockets`
- Chromium or Chrome on `PATH`
- `pdftoppm` (poppler) for `--preview`

```sh
cargo run --manifest-path press/Cargo.toml --release -- --preview
```

Output:

```text
out/manual.html
out/jotnar-field-manual.pdf
out/preview/page-*.png
```

Flags:

| Flag | Meaning |
|---|---|
| `--preview` | Render 72-dpi page previews with `pdftoppm`. |
| `--html-only` | Write `out/manual.html` and stop. |
| `--no-pdf` | Same as `--html-only`. |
| `--out PATH` | Change the PDF output path. |
| `--book PATH` | Change the manifest (default `book.toml`). |

## Layout

```text
book.toml                 metadata, parts, chapter order, themes
content/*.md              one Markdown file per chapter, with frontmatter
data/ops.tsv              the 94-mnemonic register (name, arity, sig, family, note)
press/src/main.rs         Markdown -> HTML, directives, syntax highlighting, TOC
press/print_pdf.py        CDP driver: wait for fonts + Paged.js, then print
press/vendor/             Paged.js 0.4.3 (vendored)
theme/book.css            the zine theme: page boxes, type, color, code, tables
assets/fonts/             Anton, Barlow Condensed, Courier Prime, IBM Plex Mono,
                          Special Elite, Stardos Stencil, Bebas Neue
assets/textures/          paper, red paper, dark paper, grain, halftone, page pill
assets/diagrams/          execution loop, trust loop, dataflow, phi lattice
assets/img/               halftone and duotone source images
scripts/make_assets.py    deterministic texture/halftone generator
scripts/witness.sh        run the four first listings through Jotnar
out/                      generated HTML, PDF, preview
```

## Editing

A chapter looks like this:

```markdown
---
title: The Geometry of a Program
number: "02"
kicker: FIELD NOTE / THE LANGUAGE
lede: A program is a graph wearing the costume of a list.
columns: 2
---

Body text, `listings`, tables, and directives.
```

Frontmatter keys: `title`, `number`, `kicker`, `lede`, `columns` (1 or 2),
`feature` (full-page opener), `theme` (`red`, `dark`, `paper`), `art`,
`art_caption`, `stamp`. Add the file to the relevant `[[part]]` in `book.toml`.

Directives are deliberately plain: `::: note`, `::: warning`, `::: sidebar`,
`::: pullquote`, `::: figure`, `::: listing`, `::: op-table`, `::: stamp`,
`::: end`, `::: pagebreak`. See Appendix B in the PDF for the source.

## The Jotnar connection

The press is host code because Jotnar is a dataflow language over typed streams:
it has no filesystem, strings, or subprocess boundary. The manual does not
pretend otherwise. What Jotnar *can* do is witness the examples:

```sh
JOTNAR_REPO=.. ./scripts/witness.sh
```

This runs the four first listings through the reference interpreter and checks
the outputs quoted in Chapter 15. It is a small, honest use of the language in
the book about the language.

## Trust spine

Appendix A of the PDF is the long-form trust spine: the loop, the three axes
(`verdict`, `outcome`, `status`), gate independence and the vacuity test,
falsifiers/nulls/benches, and why `BLOCKED` is never `PASS`. The harness itself
lives in the parent Rust repo:

```sh
cargo run -p jotnar-lab -- verify certificates/example/certificate.json
```

The self-contained example needs no external assets. The real DOOM certificate
references local `/tmp` artifacts; if they are absent, `verify` prints
`BLOCKED`. That is the design, not a bug.

## Design

The visual language is a xerox-zine / 1990s computer-magazine pastiche: cream
paper, black ink, one red, typewriter body, condensed display type, halftone
images, stamps, and cut-out labels. It is inspired by the provided reference PDF
but reproduces no text, logos, or artwork from it. The halftones and duotones are
generated from the Jotnar project's own samples and showcase assets.

## Licenses

- Jotnar implementation and code excerpts: **GPL-3.0-only**.
- Manual text: **CC BY-SA 4.0**.
- Fonts: SIL Open Font License / Apache 2.0 as applicable.
- Paged.js: MIT.
