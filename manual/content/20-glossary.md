---
title: Glossary
number: "C"
kicker: APPENDIX C / VOCABULARY
lede: The words are load-bearing. Use them the way the stack uses them.
columns: 2
---

| Term | Meaning |
|---|---|
| **arity** | The number of inputs and outputs a mnemonic requires. Wrong arity is an assembly error. |
| **assembler** | The frontend that expands, parses, binds, and verifies a listing. |
| **bench** | A cost certificate: frozen inputs, repeated runs, a distribution, and a worst case, with mandatory `clock`, `scope`, and `kind` labels. Not a leaderboard. |
| **blocked** | An outcome or status: the artifacts, runtime, or reference are absent. Honest, never a pass. |
| **bottom** | A gate verdict: the gate could not decide. Never a pass. |
| **bound program** | A parsed program with each instruction bound to a registry signature. |
| **CALL** | Textual expansion of a `DEF` at a call site, with namespaced internal names. |
| **certificate** | The recorded witness/proof/derivation, with scope, unknowns, falsifiers, nulls, bench labels, and provenance. |
| **coherence** | A local orientation measure in `[0,1]` produced by the structure tensor. |
| **CONFIG** | A program-global constant such as `m_acc`, `m_cov`, or `beta`. |
| **DEF** | A named reusable subgraph with formals and declared outputs. |
| **derivation** | The trace that makes an answer legible. |
| **DIG** | Digital Jotnar: exact, bit-identical, C/CUDA/RV64. |
| **does_not_prove** | The required boundary of a certificate’s claim. An empty boundary is an overclaim. |
| **emitter** | A backend that lowers a bound program to a target artifact. |
| **fail loud** | The doctrine that errors are errors, never silent guesses. |
| **falsifier** | A test of the gate’s sensitivity: corrupt the proposal and see whether the gate catches it. |
| **field** | A spatial stream/lattice: depth, normals, IDs, provenance. Jotnar-FLD. |
| **fixed-point bridge** | `to_fixed` / `from_fixed`: the conversion between triples and int64 at a scale. |
| **gate** | An independent predicate on `(map, sample, reference)`. It must be able to say no. |
| **gate independence** | A gate with its own reference, not `m == P(spec)`. A gate that cannot reject a wrong proposal is not a gate. |
| **Jotnar** | The family/stack: program, machine, witness. Not a project name. |
| **Jotnar-LAN** | The language/IR/op set/plan. The program. |
| **Jotnar-MAC** | The machine/target: ISA, memory, lanes, OS, board. |
| **Jotnar-WIT** | The witness/contract: trace, certificate, provenance, byte-identity. |
| **kind** | `F32`, `F64`, `I64`, `U8`, `Bool`, `Triple`, `None`. |
| **layout** | A `KIND:GEOM` string such as `T:HW` or `F:HWC`. |
| **lattice** | The phi-value substrate: sign, exponent, zero flag; integer arithmetic in the middle. |
| **map** | A stored field/artifact: shape, dtype, layout, hash. Jotnar-MAP. |
| **mnemonic** | An uppercase instruction name, bound to the registry. |
| **NoPattern** | A loud emitter error: the target has no lowering for this mnemonic. |
| **not_caught** | A falsifier row the gate did not catch. It names the trust boundary; it is not dishonesty. |
| **not_tested** | A deliberately skipped control or outcome. Honest, not a pass. |
| **null control** | A deliberate control that should not produce the effect. It tests the claim’s specificity. |
| **outcome** | `solved`, `refused`, `blocked`, or `not_tested`. The process result, separate from the gate verdict. |
| **payload** | The JSON boundary: a `streams` map of typed input values. |
| **plan** | The executable binding of maps (inputs/outputs) to vectors (work). Jotnar-PLN. |
| **proof** | A witness plus an independent gate plus a declared contract. |
| **proposal** | The candidate produced by a proposer, with provenance. |
| **refusal** | A typed statement that the process could not produce a proposal. A refusal is a result. |
| **registry** | The authoritative table of 90 mnemonics, arities, and signatures. |
| **shape** | The dimensions of a stream, e.g. `[4]`, `[H,W]`, `[H,W,C]`. |
| **signature** | A layout pattern per input and output, with `$VAR`, `*`, and literals. |
| **space** | The declared sample space, reference or bound, and provenance. Part of the contract. |
| **STATE** | A stream carried across host iterations by `jotnar repeat`. |
| **status** | `current`, `stale`, `historical`, `blocked`, or `superseded`. A stale or historical certificate is not a current claim. |
| **stream** | A name for a value with a kind and a shape. The central object. |
| **triple** | Three planes: sign, exponent, zero flag. A scale-free lattice value. |
| **unification** | Matching a signature pattern against an actual layout, binding `$VAR`s. |
| **vacuity test** | Replace the gate with `m == P(spec)` and show that a wrong map still passes. If it does not, the test is broken. |
| **vector** | A costed batch of data-parallel work. Jotnar-VEC. |
| **verdict** | `pass`, `fail`, or `bottom`. The gate’s decision, separate from the outcome. |
| **witness** | A fact: map, input, output, trace. A witness is not a proof until an independent gate and a contract are attached. |

::: note [NAMING DISCIPLINE]
When a sentence uses “Jotnar” without a qualifier, it means the family/stack. A specific claim names the aspect: **Jotnar-LAN**, **Jotnar-MAC**, or **Jotnar-WIT**. A digital/analog claim names the flavor: **DIG**, **SIM**, **ANA**, or **HYB**.
:::

::: end
:::
