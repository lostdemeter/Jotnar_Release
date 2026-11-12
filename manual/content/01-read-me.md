---
title: Read This First
number: "00"
kicker: EDITORIAL / FIELD KIT
lede: This is a first rough draft of a manual for a language that is still being built. That is the point.
columns: 2
---

You are holding a field manual for **Jotnar**, a geometric assembly language for typed streams. It is not a tutorial for a finished product. It is a working document for a language that is being ported, tested, and argued with in public.

The code is real. The examples run. The edges are still sharp.

**Jotnar** is the family/stack. This manual documents the Rust implementation of **Jotnar-LAN/DIG**: a program format, a reference interpreter, and a set of emitters for C, integer-only C, CUDA, Verilog, and an RV32IM machine. The Python implementation is the oracle. This manual follows the same doctrine as the code: **unknown mnemonics, arity mismatches, use-before-def, layout conflicts, and shape mismatches are errors, never silent guesses.**

::: note [HOW TO READ THIS BOOK]
There are three passes. **First pass:** read the manifesto, then the geometry chapter, then the four first listings. You will be able to read a Jotnar listing without running it. **Second pass:** read the instruction-set chapters with the register open. **Third pass:** read the machine chapters and build something. The appendices are for when you want to change the book itself.
:::

## The field kit

You need a terminal, a Rust toolchain, and a payload. The payload is a JSON file that binds names to typed streams. That is the whole boundary. There is no hidden state, no environment variable that changes arithmetic, no configuration file that silently rescales your data.

```sh
# build the reference implementation
cargo build --release

# check, verify, run
./target/release/jotnar check  showcase/language/saxpy.asm
./target/release/jotnar verify showcase/language/saxpy.asm
./target/release/jotnar run    showcase/language/saxpy.asm \
  --payload /tmp/saxpy.json --outputs OUT
```

The payload for that run is small enough to read in one sitting:

```json
{"streams":{
  "A":{"kind":"I64","shape":[4],"data":["3","3","3","3"]},
  "X":{"kind":"I64","shape":[4],"data":["1","2","3","4"]},
  "Y":{"kind":"I64","shape":[4],"data":["10","20","30","40"]}
}}
```

The output is `[13, 26, 39, 52]`. If you got that, the rest of the book is elaboration.

## Prerequisites

| Need | For |
|---|---|
| Rust stable + Cargo | everything. |
| `gcc` | the C backend, the native DOOM core, and the Verilog fixture. |
| CUDA toolkit + GPU | the CUDA target and GPU migration parity. |
| SDL2 runtime | the playable DOOM window. |
| DOOM image + WAD | live DOOM; `run_doom_demo.sh` falls back otherwise. |
| Python 3 + NumPy/Pillow | the parity harnesses against the Python oracle. |
| `jotnar-main` + `phi-core` checkout | differential parity only; the demos run without them. |

`run_doom_demo.sh` is **playable by default**. `--frames N` is the headless path. If the external image or WAD is absent, the script falls back to the checked-in reference title and does not pretend the live run happened.

::: warning [DRAFT NOTICE]
This is **issue 01**, a first rough draft. Some chapters describe behavior that is gated by parity scripts but not yet exposed in the CLI. Some commands are stable; some are scaffolding. Where the manual and the assembler disagree, the assembler wins. Where the assembler and the oracle disagree, the parity harness wins. Where the harness and reality disagree, file a bug.
:::

::: note [STATUS AT THIS DRAFT / 2026-11-08]
The Rust workspace is green at **225/225 tests**. The self-contained trust-spine example verifies **PASS**. The real DOOM certificate verifies **PASS on this host**, and correctly prints **BLOCKED** when its local `/tmp` artifacts are absent. The manual's witness script passes **4/4** listings. The trust spine has its own appendix now: **Appendix A**.
:::

## Conventions

| Convention | Meaning |
|---|---|
| `UPPERCASE` | A mnemonic, directive, or keyword. |
| `lowercase` | A stream name chosen by the author. |
| `$VAR` | A layout variable, unified by the assembler. |
| `KIND:GEOM` | A concrete layout, such as `T:HW` or `F:HWC`. |
| `*` | A wildcard layout: matches anything, binds nothing. |
| `→` | “produces” in a signature; `2 → 1` means two inputs, one output. |
| **bold** | A term defined in the glossary. |
| `#` | A comment, to the end of the line. |

Jotnar listings are line-oriented. An assignment names one or more outputs on the left and one operation on the right. There are no registers, no stack, and no instruction pointer. The listing order is for the reader; the **dataflow** is the program.

## The fail-loud promise

Most languages try to be helpful when you make a mistake. Jotnar tries to be honest. A wrong arity is an error. A layout conflict is an error. A use-before-def is an error. A shape mismatch at run time is an error. An integer division by zero is an error. A float that is NaN or infinite where an integer is required is an error.

This is not masochism. It is the only way to keep a language that lowers to five different machines honest. If the frontend guesses, the emitters inherit the guess, and the witness is worthless.

::: pullquote [THE HOUSE RULE]
If the assembler has to guess, the program is already wrong.
:::

## What is not here yet

The manual describes the parts that exist. The honest gaps are also part of the documentation:

- the `jotnar-main` **device** emitter and the CUDA fleet kernel used by the original DOOM GPU demo;
- TCP transport and serve mode (`live=` resident loops), packed outputs, the codec ABI, CUDA graphs, tracing summaries, causal `overrides`, and the store-hook fused executor;
- the model-side phi-core modules (`calibrate`, `score`, `compose`, `stamp`, `zoo`);
- tuned CUDA kernels and cuBLAS scheduling. The current CUDA backend is correctness-first: one launch per operation;
- the Rust SDL player's intermittent frame-time spikes. It is playable; it is not yet smooth. A frame-time trace and a p99/max-frame gate are follow-ups, not core-exactness gaps.

The trust spine is **not** on this list. It is here: Appendix A. If you need one of the remaining gaps, you are reading the right draft at the wrong time. The next issue is the one to watch.

::: stamp [FAIL LOUD]
:::

::: end
:::
