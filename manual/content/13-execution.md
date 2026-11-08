---
title: "The Execution Loop: LAN → MAC → WIT"
number: "12"
kicker: FIELD NOTE / THE MACHINE
lede: A plan is not a machine. A machine is not a proof. The loop is what makes the difference visible.
columns: 2
---

Running a Jotnar program is a loop with three stations:

1. **Jotnar-LAN** — the program is parsed, expanded, bound, and (optionally) verified.
2. **Jotnar-MAC** — the bound graph is executed by the reference interpreter or lowered to a target.
3. **Jotnar-WIT** — the outputs are compared against an oracle, byte for byte where the contract allows it.

The stations are not a pipeline in the software-engineering sense. They are aspects of the same claim: the program you wrote, the machine that ran it, and the evidence that the result is what the plan said.

::: figure src="assets/diagrams/loop.svg" caption="The execution loop. Same program + same input bytes = same output bytes. The witness is the third station, not an afterthought."
:::

## What `jotnar run` does

The reference path is intentionally boring:

```text
read listing
  -> expand IMPORT / DEF / REPEAT
  -> parse directives and assignments
  -> bind each mnemonic to the registry (arity + signature)
  -> read payload JSON
  -> bind input streams (kind + shape)
  -> execute the bound instructions in order
  -> re-check shapes at each op that has a rule
  -> write every stream (or the named outputs) as JSON
```

There is no JIT, no speculative execution, and no hidden caching. The interpreter walks the bound instruction list. `REPEAT` with a runtime count and `BREAK` are the only control flow; everything else is dataflow.

::: note [STATE AND REPEAT]
`STATE` streams are seeded from the payload and carried across host iterations by `jotnar repeat`. Inside a single run, a `REPEAT` loop can update a name; the interpreter threads the new value into the next iteration. The state contract is checked at assembly time: a `STATE` name must also be an `IN` and must be assigned as an output.
:::

## The artifacts

| Artifact | Aspect | What it is |
|---|---|---|
| A listing (`.asm`) | Jotnar-LAN | The program text. |
| A bound program | Jotnar-LAN | Parsed instructions plus registry signatures. |
| A target source (`.c`, `.cu`, `.v`) | Jotnar-MAC | The lowered machine program. |
| A payload (`.json`) | Jotnar-MAP | Typed input streams at rest. |
| An output map (`.json`) | Jotnar-MAP | Typed output streams. |
| A parity report | Jotnar-WIT | Comparisons against the oracle. |
| A trace / certificate | Jotnar-WIT | Evidence of execution, not a proof of intent. |

The vocabulary matters because the claims are different. “It compiles” is a LAN claim. “It runs” is a MAC claim. “It matches the oracle” is a WIT claim. Do not use one to stand in for another.

## What the witness actually checks

The Rust port is gated by differential harnesses. The Python oracle covers the original 86 mnemonics; the four integer extensions (`T2I`, `IGET`, `IMATMUL`, `IREDUCE_SUM`) have no Python oracle and are gated separately by interpreter/C/non-FPU differential tests:

| Gate | What it compares | Current result |
|---|---|---|
| `parity_ops.py` | One small listing per original mnemonic, Rust vs. Python. | 88/88 cases for the original 86-mnemonic oracle. |
| `tests/int_geometry.rs` | The four integer extensions: interpreter vs. emitted C and non-FPU, plus T2I’s baked table and IGET bounds. | Exact within the stated test domains. |
| `parity_emit.py` | Emitted C and non-FPU binaries vs. Python-emitted binaries. | 32 emitter cases. |
| `parity_programs.py` | The flagship programs, including the image listing. | Streams match within the stated class. |
| `parity_cuda.py` | CUDA units, bigram, argmax, headt/fp16/D32/D64. | Units within 1e-4; decisions bit-exact. |
| `parity_futhark.py` | The word-level Futhark listing vs. the JS rule engine. | 118/118 words. |
| `parity_doom_native.py` | Frame 1/2 state and framebuffer vs. C and GPU cores. | Byte-identical where gated. |

Triples and integers are compared bit-exactly. Floats are compared within a tight tolerance, because a different SIMD reduction order can change the last ulp. The manual says “exact” only where the gate is exact.

::: warning [THE WITNESS RULE]
The board is a witness, not a proof. Byte-identity against an oracle means the lowering preserved the plan under the tested inputs. It does not mean the plan is correct, the inputs are representative, or the model is honest. Those are separate claims. Appendix A turns this rule into a certificate schema.
:::

## A small trace

A `verify` run is the cheapest witness:

```text
$ jotnar verify tests/fixtures/cpu8.asm
layouts resolved: 53/59 (0 error(s))
```

That line says the declared layouts unify. It does not say the CPU executes correctly. For that, the Verilog backend emits the same integer program as structural hardware and checks the cycle trace against an independent ISA model. The trace is the witness; the 53/59 is a static check.

For a run, the witness is the output map:

```json
{"streams":{"OUT":{"kind":"I64","shape":[4],
  "data":["13","26","39","52"]}}}
```

If the oracle says the same bytes, the run is witnessed. If it says something else, the run is a bug report.

::: pullquote [LOOP RULE] class="full"
Program, machine, witness. Name the aspect or the claim is ambiguous.
:::

::: end
:::
