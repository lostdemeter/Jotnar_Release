---
title: The Trust Spine
number: "A"
kicker: APPENDIX A / WHAT A RESULT PROVES
lede: A demo is not a claim. The trust spine records what a result proves, what it does not, and when to refuse.
columns: 2
---

The trust spine is **not a language feature**. It is the development harness that lets us say a claim about a program, a machine, a map, a vector, a plan, or a certificate is true. Jotnar-LAN defines programs; the trust spine defines the discipline for claiming something about them.

The rule is one sentence:

> Construct the small space, propose, resolve exactly, gate independently, keep the certificate — and when you cannot, refuse.

The unity is in the loop, not in the artifacts. A demo is an instance of the loop at one layer. The normative summary is `docs/LOOP.md`; the schema is `docs/appendix/TRUST_SPINE.md`; the implementation is `crates/jotnar-lab`. This appendix is the reader’s version.

::: figure src="assets/diagrams/trust-loop.svg" caption="The trust loop. Every stage may stop; a refusal, a block, and a failed gate are results, but none of them is a pass."
:::

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

| Term | Meaning |
|---|---|
| **Space** | The declared sample space, reference or bound, and provenance. It is part of the contract. |
| **Proposal** | The candidate produced by a proposer, with provenance. |
| **Gate** | An independent predicate on `(map, sample, reference)`. It must be able to say no. |
| **Witness** | A fact: map, input, output, trace. |
| **Proof** | A witness plus an independent gate plus a declared contract. |
| **Derivation** | The trace that makes the answer legible. |
| **Certificate** | The recorded witness/proof/derivation, with scope, unknowns, falsifiers, nulls, provenance. |
| **Refusal** | A typed statement that the process could not produce a proposal. |

A gate is **not** `m == P(spec)`. That is vacuous: it restates the proposal. A real gate has its own reference and can reject a wrong proposal. The certificate records `gate.can_say_no`; a gate that cannot say no is not a gate.

## Three axes, not one

| Axis | Values | What it means |
|---|---|---|
| **Verdict** | `pass`, `fail`, `bottom` | The gate's decision. `bottom` means the gate could not decide. It is never a pass. |
| **Outcome** | `solved`, `refused`, `blocked`, `not_tested` | The process result. A correct refusal is a successful certificate. A block is an honest certificate. A skipped control is `not_tested`. |
| **Status** | `current`, `stale`, `historical`, `blocked`, `superseded` | The certificate's standing. A stale or historical certificate is evidence about the past, not a current claim. |

A valid certificate can therefore be:

```text
outcome = refused
verdict = bottom
status  = current
```

`verify` must accept that as a successful refusal — while still refusing to call it a pass.

::: warning [THE FIRST RULE OF THE SPINE]
A blocked run is not a pass. A skipped control is not a pass. A gate that cannot say no is not a gate. A certificate without `does_not_prove` is an overclaim.
:::

## Gate independence and the vacuity test

The gate must be able to reject the proposal for the right reason. The vacuity test is the cheapest way to catch a fake gate:

```text
replace the gate with   m == P(spec)
show that a wrong map still passes
```

If the wrong map **does not** pass, the test is broken. The DOOM certificate records this as a falsifier row: “replace gate with `m == P(spec)`” → wrong map passes → `not_caught` by construction. That is the independence test, not a failure of the gate.

## Required honesty fields

`Certificate::validate` rejects a certificate that omits:

- `proves` or `does_not_prove` (an empty boundary is an overclaim);
- declared assumptions and unknowns;
- the gate hash and `can_say_no = true`;
- at least one falsifier row and one null row (`not_tested` is allowed);
- bench `clock`, `scope`, and `kind` labels;
- provenance (command, cwd, commit, host, environment) and artifact paths/hashes.

A certificate with `outcome = refused` must carry a refusal. A certificate with `status = superseded` must carry `superseded_by`. `certificate_sha256` is computed over canonical JSON without the hash field, then written; `verify` recomputes it.

## Falsify, null, bench

They are duals, not one bucket.

| Tool | Tests | Honest non-result |
|---|---|---|
| **Falsify** | The gate's sensitivity: corrupt the proposal and see whether the gate catches it. | `not_caught` names the trust boundary. |
| **Null** | The claim's specificity: run a control that should not produce the effect. | `not_tested` is an honest state; a confounded control is a finding. |
| **Bench** | The cost claim: frozen inputs, repeated runs, a distribution, a worst case. | A benchmark is a cost certificate, not a leaderboard. |

Bench labels are mandatory:

```text
clock: rdcycle (emulated) | host_wall (measured) |
       cuda_event (measured) | analog (simulated) |
       calibrated (0.185 ns/node)
scope: apply | total | startup | end_to_end | worst_frame
kind:  measured | emulated | simulated | calibrated
```

A simulated analog wall-clock is not measured hardware time. The label says which one it is.

## The DOOM certificate

The real certificate is `certificates/doom/native-gpu-migration.json`. It records:

- frame 1: Rust vs. C native on the first 64 MB, registers, PC, cycles, and framebuffer;
- frame 2: Rust checkpoint vs. the GPU checkpoint on the first 64 MB, registers, PC, and framebuffer;
- **does not prove**: the GPU kernel for all programs, plan completeness, memory beyond 64 MB, or fleet throughput.

The falsifier table deliberately includes boundary rows:

| Corruption | Expected | Actual | `not_caught` |
|---|---|---|---|
| framebuffer byte changed | fail | fail | false |
| register byte changed | fail | fail | false |
| PC changed | fail | fail | false |
| memory byte changed (no frame effect) | fail | fail | false |
| memory byte changed beyond 64 MB | fail | pass | **true** |
| compare 128 MB native with 64 MB GPU | fail (space mismatch) | fail (full) / pass (64 MB slice) | **true** |
| wrong frame | fail | pass (frames identical) | **true** |
| replace gate with `m == P(spec)` | pass (wrong map must pass the vacuous gate) | pass | **true** |

The null controls include `wrong_frame` (expected reject, actual pass because the title frame is static; status `not_tested`) and `shifted_frame` (expected reject, actual reject; status `pass`). The bench is `clock=host_wall`, `scope=end_to_end`, `kind=measured`, three runs, with p50/p99/max.

::: listing [THE DOOM CERTIFICATE / CONDENSED]
```text
schema:          jotnar.certificate
status:          current
outcome:         solved
verdict:         pass
gate:            byte_identical_migration
can_say_no:      true
proves:          Rust frame 1 matches C native on 64 MB/regs/PC/cyc/fb;
                 Rust frame 2 matches the GPU checkpoint on 64 MB/regs/PC/fb
does_not_prove:  GPU kernel for all programs; plan completeness;
                 memory beyond 64 MB; fleet throughput
falsifiers:      8 rows, including 4 not_caught (space boundaries + vacuity)
nulls:           2 rows, including wrong_frame = not_tested (static title)
bench:           clock=host_wall, scope=end_to_end, kind=measured
```
:::

## Verify, and why BLOCKED is not PASS

`verify` checks the schema, the certificate hash, and the artifact hashes by default. `--no-hashes` skips only the artifact check; the schema and certificate hash are still checked.

| Certificate state | `verify` says |
|---|---|
| `outcome=solved`, `verdict=pass`, `status=current`, artifacts present | `PASS` |
| `outcome=refused` with a refusal | `REFUSED (valid)` |
| `outcome=blocked` | `BLOCKED` |
| `outcome=not_tested` | `NOT TESTED` |
| `verdict=fail` | `FAIL (certified negative)` |
| `verdict=bottom` | `BOTTOM (gate could not decide)` |
| `status=stale/historical/superseded` | `STALE` / `HISTORICAL` / `SUPERSEDED` (not a current claim) |
| artifacts missing | `BLOCKED`, exit non-zero |

The self-contained example needs no external assets. Run these commands from the Jotnar repository root (the parent directory):

```sh
cargo run -p jotnar-lab -- verify certificates/example/certificate.json
```

The DOOM certificate references local `/tmp` artifacts. On this host it verifies **PASS**. Without those artifacts it prints **BLOCKED** and exits non-zero. That is intentional: a certificate about files that are no longer present is not a pass. A caller may choose to accept a block for its own purpose, but that is the caller's policy, not the certificate's claim.

The real commands, also from the Jotnar repo root:

```sh
# Build the bundle from the native/C/GPU parity run.
python3 scripts/parity_doom_native.py \
  --gpu-work /tmp/opencode/doom_overlay \
  --bundle /tmp/opencode/doom_bundle

# Read existing artifacts; do not re-run the CPU.
cargo run -p jotnar-lab -- certify doom \
  --bundle /tmp/opencode/doom_bundle/manifest.json \
  --out certificates/doom/native-gpu-migration.json

cargo run -p jotnar-lab -- falsify doom \
  --bundle /tmp/opencode/doom_bundle/manifest.json

cargo run -p jotnar-lab -- null doom \
  --bundle /tmp/opencode/doom_bundle/manifest.json

cargo run -p jotnar-lab -- bench doom \
  --bundle /tmp/opencode/doom_bundle/manifest.json \
  --runner target/release/jotnar-demo-doom --runs 3 \
  --certificate certificates/doom/native-gpu-migration.json
```

::: note [STATUS AT THIS DRAFT / 2026-11-08]
Workspace tests: **221/221**. Self-contained example certificate: **PASS**. DOOM certificate: **PASS on this host**, and **BLOCKED** when the local artifacts are absent. Manual witness: **4/4** listings. The trust spine does not make every Jotnar claim certified; it makes the claim’s boundary explicit.
:::

## What the harness must not do

- It must not call a blocked run a pass.
- It must not accept a certificate without `does_not_prove`.
- It must not accept a gate that cannot say no.
- It must not turn a vacuous gate into evidence.
- It must not require a positive verdict to call a certificate valid; an honest fail, refusal, or block is a certificate too.

::: pullquote [TRUST SPINE RULE]
A certificate is a claim with its boundary attached. If the boundary is missing, the claim is not certified.
:::

::: end
:::
