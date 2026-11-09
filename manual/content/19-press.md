---
title: Building This Manual
number: "B"
kicker: APPENDIX B / THE PRESS
lede: The book is a build artifact. Here is the pipeline, the source, and the reason it is Rust rather than Jotnar.
columns: 1
---

This manual is built from Markdown files by a small Rust program called **jotnar-press**. The press reads `book.toml`, renders each chapter, generates the cover, masthead, table of contents, part dividers, and the 94-mnemonic register, then hands the HTML to Paged.js and Chromium for pagination and PDF output.

```text
content/*.md
   -> jotnar-press (pulldown-cmark + directives)
   -> out/manual.html
   -> Paged.js (page boxes, running heads, page numbers)
   -> Chromium (print-to-PDF, CSS @page)
   -> out/jotnar-field-manual.pdf
```

The whole thing is deliberately inspectable. There is no layout engine to learn, no binary template, and no proprietary typesetting step. If a page is wrong, it is wrong in the Markdown, the CSS, or the builder.

## The tree

| Path | What it is |
|---|---|
| `book.toml` | Book metadata, part order, chapter order, themes. |
| `content/*.md` | One Markdown file per chapter, with a small frontmatter block. |
| `data/ops.tsv` | The 94-mnemonic register: name, arity, signature, family, one-line contract. |
| `press/src/main.rs` | The press: frontmatter, directives, syntax highlighting, HTML, PDF. |
| `press/vendor/paged.polyfill.js` | Paged.js 0.4.3, vendored so the build is offline. |
| `theme/book.css` | The zine theme: page boxes, type, color, code, tables, cover. |
| `assets/` | Fonts, textures, halftone images, diagrams, cover art. |
| `scripts/make_assets.py` | Deterministic texture and halftone generator. |
| `scripts/witness.sh` | Runs the four first listings through the Jotnar interpreter. |
| `out/` | Generated HTML, PDF, and preview images. |

## Build

```sh
cargo run --manifest-path press/Cargo.toml --release -- \
  --preview
```

The `--preview` flag writes 72-dpi PNGs to `out/preview/` after the PDF. `--html-only` stops before Chromium; `--no-pdf` is the same; `--out` changes the PDF path.

## Add a chapter

1. Create `content/22-my-chapter.md` with frontmatter:

```yaml
---
title: My Chapter
number: "18"
kicker: FIELD NOTE / EXPERIMENT
lede: One sentence that tells the reader why this chapter exists.
columns: 2
---
```

2. Add the file to the appropriate `[[part]]` in `book.toml`.
3. Rebuild. The TOC page number is filled by the Paged.js `after` hook, so you do not need to edit it.

The frontmatter keys are: `title`, `number`, `kicker`, `lede`, `columns` (1 or 2), `feature` (true for a full-page opener), `theme` (`red`, `dark`, or `paper`), `art`, `art_caption`, and `stamp`.

## Directives

The press adds a few blocks on top of Markdown. They are deliberately plain:

| Directive | Result |
|---|---|
| `::: note [TITLE]` | A cream note box with a red rule. |
| `::: warning [TITLE]` | A black warning box with a red rule. |
| `::: sidebar [TITLE]` | A red sidebar box. |
| `::: pullquote [ATTRIBUTION]` | A large uppercase quote with a citation. |
| `::: figure src="..." caption="..."` | A figure with a caption; `class="bleed"` for full width. |
| `::: listing [CAPTION]` | A code block with a label. |
| `::: op-table family="int"` | A generated table for one family, or `all`. |
| `::: stamp [TEXT]` | A rotated stencil stamp. |
| `::: end` | The black `END` mark. |
| `::: pagebreak` | A forced page break. |

Markdown inside a directive is rendered normally, so a note can contain lists, code, and tables.

## The builder, in one function

The heart of the press is a small block renderer. It receives the directive name, the attributes, and the rendered body:

```rust
match name {
    "note" => format!(
        r#"<aside class="note span-all"><div class="aside-title">{}</div>{inner}</aside>"#,
        escape(&label)
    ),
    "warning" => format!(
        r#"<aside class="warning span-all"><div class="aside-title">{}</div>{inner}</aside>"#,
        escape(&label)
    ),
    "listing" => format!(
        r#"<div class="listing span-all"><div class="listing-label">{}</div>{inner}</div>"#,
        escape(&label)
    ),
    _ => format!(r#"<div class="block {name} span-all">{inner}</div>"#),
}
```

The `span-all` class is kept for the two-column experiments; the current prose grid is a measured single column. The rest is CSS.

## The pagination hook

Paged.js handles the page boxes, the running heads, and the page counter. The only dynamic piece is the TOC page number. The press injects a tiny `after` hook:

```js
window.PagedConfig = {
  auto: true,
  after: function () {
    var pages = [].slice.call(document.querySelectorAll('.pagedjs_page'));
    var pageOf = new Map();
    pages.forEach(function (page, idx) {
      page.querySelectorAll('[id]').forEach(function (el) {
        if (!pageOf.has(el.id)) pageOf.set(el.id, idx + 1);
      });
    });
    document.querySelectorAll('.toc-page[data-target]').forEach(function (el) {
      var n = pageOf.get(el.getAttribute('data-target'));
      if (n) el.textContent = String(n).padStart(2, '0');
    });
  }
};
```

That is the entire coupling between the content and the pagination. There is no two-pass build and no manual page-number table.

## Why not write the press in Jotnar?

This is the honest answer: Jotnar is a dataflow language over typed streams. Its boundary is a payload of streams and a set of named outputs. It has no filesystem, no string type, no Markdown parser, and no subprocess. Writing a Markdown-to-PDF builder in Jotnar would mean building an operating system, a text model, and a font rasterizer first. That is a different project, and a good one, but it is not this one.

What Jotnar *can* do is witness the examples. `scripts/witness.sh` runs the four first listings through the reference interpreter and checks the outputs the manual quotes. The press builds the book; Jotnar checks the claims the book makes about itself.

::: note [THE HONEST SPLIT]
Host code does I/O, parsing, and rendering. Jotnar does the typed computation. The boundary is the payload. Keeping that split explicit is the same discipline the language asks of its users.
:::

## Make it yours

The theme is a single CSS file. The tokens at the top control the paper, ink, red, and the font stacks. The cover is generated from `book.toml` plus `assets/cover-art.svg`. The part dividers use `theme = "red" | "dark" | "paper"` in `book.toml`. The halftones and textures are regenerated deterministically by `scripts/make_assets.py`.

If you change the page size, change `@page { size: ... }` and the fixed-size cover/part sections together. If you change the font, add a `@font-face` and update the token. If you change the register, edit `data/ops.tsv`; the chapter body is only a directive.

::: pullquote [PRESS RULE] class="full"
A manual you can rebuild is a manual you can trust.
:::

::: end
:::
