# The loop

This is a **normative appendix**, not part of the language. Jotnar-LAN
defines programs; the trust spine defines how we are allowed to say a claim
about a program (or a machine, a map, a vector, a plan, a certificate) is
true.

The rule is one sentence:

> Construct the small space, propose, resolve exactly, gate independently,
> keep the certificate — and when you cannot, refuse.

The unity is in the loop, not in the artifacts. A demo is an instance of the
loop at one layer.

## The loop

```text
construct a small space
        |
        v
propose a candidate
        |
        v
resolve it exactly
        |
        v
gate it independently
        |
        v
keep the certificate
        |
        +--> refuse loudly when the space, data, or runtime is absent
```

Every stage is allowed to stop. A refusal is a result. A block is a result.
A failed gate is a result. None of them is a pass.

## Vocabulary

| term | meaning |
|---|---|
| **Space** | The declared sample space, reference or bound, and provenance. It is part of the contract, not an implementation detail. |
| **Proposal** | The candidate produced by a proposer. It has provenance. |
| **Gate** | An independent predicate on `(map, sample, reference)`. It must be able to say no. |
| **Witness** | A fact: map, input, output, trace. |
| **Proof** | A witness plus an independent gate plus a declared contract. |
| **Derivation** | The trace that makes the answer legible. |
| **Certificate** | The recorded witness/proof/derivation, with scope, unknowns, falsifiers, nulls, and provenance. |
| **Refusal** | A typed statement that the process could not produce a proposal. |

A gate is not `m == P(spec)`. That is vacuous: it restates the proposal. A
real gate has its own reference and can reject a wrong proposal. The
certificate records `gate.can_say_no`; a gate that cannot say no is not a
gate.

## Three axes, not one

**Gate verdict**

```text
pass | fail | bottom
```

`bottom` means the gate could not decide. It is never a pass.

**Process outcome**

```text
solved | refused | blocked | not_tested
```

A correct refusal is a successful certificate. A blocked run is an honest
certificate. A deliberately skipped control is `not_tested`, not a pass.

**Certificate status**

```text
current | stale | historical | blocked | superseded
```

A stale or historical certificate is evidence about the past, not a current
claim. A superseded certificate points at its successor.

A valid certificate can therefore be:

```text
outcome = refused
verdict = bottom
status  = current
```

and `verify` must accept it as a successful refusal — while still refusing
to call it a pass.

## The trust boundary

Every certificate must say both:

- `proves` — the positive claim;
- `does_not_prove` — the boundary of that claim.

The board is the model: the certificate checks what the plan says about the
rows it carries; it cannot check rows the plan omits. The omitted rows are
the boundary. The certificate must name them.

Required fields include:

- declared assumptions;
- unknown fields;
- the gate identity and `can_say_no`;
- falsifier rows;
- null controls;
- bench labels (`clock`, `scope`, `kind`);
- provenance: command, cwd, commit, host, environment;
- artifact paths and hashes.

## Three process utilities

They are duals, not one bucket.

### Falsify

Tests the **gate's sensitivity**.

Corrupt the proposal, plan, or certificate and see whether the gate catches
it. The interesting result is a `not_caught` row: it names the trust
boundary.

Every suite must include the vacuity test:

```text
replace the gate with   m == P(spec)
show that a wrong map still passes
```

If the wrong map does not pass, the test is broken.

### Null

Tests the **claim's specificity**.

Run a deliberate control that should not produce the effect. `not_tested` is
an honest state. If the control produces the effect, the claim is
confounded.

### Bench

Tests the **cost claim**.

Frozen inputs, repeated runs, a distribution, and a worst case. The labels
are mandatory:

```text
clock: rdcycle (emulated) | host_wall (measured) |
       cuda_event (measured) | analog (simulated) |
       calibrated (0.185 ns/node)
scope: apply | total | startup | end_to_end | worst_frame
kind:  measured | emulated | simulated | calibrated
```

A benchmark is a cost certificate, not a leaderboard.

## Refusal taxonomy

At minimum:

```text
under_specified
contradictory_invariant
unsolved
unsupported_capability
runtime_absent
reference_absent
```

A refusal carries `code`, `message`, and `missing` names. Refusing is not an
exception path; it is a first-class result.

## Current exhibit registry

The Rust rewrite currently exposes ten exhibits. They are instances of the
same loop at different layers:

| exhibit | small space | gate | does not prove |
|---|---|---|---|
| language tour | four listings | interpreter / C output | all programs |
| image targets | 43-op image | C / non-FPU / CUDA parity | hardware speed |
| Verilog / CPU8 | integer subset | structural co-sim vs ISA | all RTL |
| analog field | RC/DAC/ADC grid | digital Laplace + tolerance | geometry completion |
| Futhark | word corpus | JavaScript reference | DOM / arbitrary contractions |
| protocol | independent jobs | byte compare + placement | multi-stage pipelines / serve mode / other backends |
| DOOM | RV32IM checkpoint | C native + GPU state | all programs / plan completeness |
| analog filter | edge/hole field | L1 + cost + nulls | physical analog hardware |
| certified agent | harness/pipeline edits | exact interpreter + C/non-FPU parity | open-ended self-improvement / universal equivalence |
| interpreter + node + job server | warm listings / framed TCP / threaded T2 | exact bytes across 1/4/16 threads, repeats, concurrent jobs + receipts | live serve / large-output or concurrent TCP / TLS / memory policy / host-side placement |

The broader eight from the other Jotnar projects — board v2, solver, field
guide, Yarnball, Echion, resonance/Hopfield, depth — are registered later
from their own repos. The certificate format must not require them to live
in this repo.

## What the harness must not do

- It must not call a blocked run a pass.
- It must not accept a certificate without `does_not_prove`.
- It must not accept a gate that cannot say no.
- It must not turn a vacuous gate into evidence.
- It must not require a positive verdict to call a certificate valid; an
  honest fail, refusal, or block is a certificate too.

## First slice

1. `docs/LOOP.md` — this page.
2. `crates/jotnar-lab` — `Certificate`, `Verdict`, `Outcome`, `Status`,
   `Refusal`, falsifiers, nulls, bench, verify.
3. `certify doom` — read existing native/GPU artifacts and emit the
   certificate.
4. `falsify doom` — the corrupt-one-byte table, including `not_caught`.
5. `null doom` — the control table, including `not_tested`.
6. `bench doom` — measured p50/p99/max with the clock/scope/kind labels.
7. `verify certificates/` — schema, hashes, artifacts, and honest status.

Then the schema's real test: a plan-omission boundary (board v2 or the
equivalent) where the resolved path cannot see a candidate the plan omits.
