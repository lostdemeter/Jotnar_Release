# Claim boundary: deterministic job server (runtime exhibit research)

## Decision

The job-server work is part of **exhibit 10, interpreter + model-blind node**.
It is a first-class language/runtime showcase, but it is **research, not a
production serve claim**. Adding it does not change the ten-exhibit count and
does not promote application performance or live-service behavior to a
language-repo gate.

The asynchronous job lifecycle, store-backed results, cancellation, node OS,
and distributed scheduler are application evidence in
[`showcase/node-os/`](../node-os/CLAIM_BOUNDARY.md) and
[`crates/jotnar-node`](../../crates/jotnar-node). They use the runtime's
`Node::snapshot`, `Node::authorize`, and `CancelGuard`, but they are not added
to the language contract.

## What this repository certifies

Reproducible from this repository:

- T2 `MATMUL`, `CONV2D`, `DECONV2D`, and `SOFTMAX` are output-partitioned and
  hash-identical across 1/4/16 threads, repeated runs, and in-process
  concurrent jobs (`tests/determinism.rs`, `jotnar-det`).
- `ExecOptions::default()` is one thread. Auto-threading is an explicit
  `ThreadPolicy::{Reference, DeterministicAuto}` boundary choice;
  `CONFIG threads` is not reused as the runtime job budget.
- `exec-ok` receipts are canonical and name the build
  (`runtime_version = <crate>+<short-git>`), scheduler version, resolved
  threads, partition descriptor, and program/blob/input/output hashes.
  Timing, worker identity, queue position, completion order, and cache state
  are outside the verdict.
- T2 streams round-trip all four planes over the protocol as `T/t2`
  (10 bytes/element); frozen T keeps the interleaved 24-byte `i64x3` form.
- `NodeHandle` uses a bounded request queue; `jotnar-node --bind
  127.0.0.1:0` reports the resolved ephemeral address.
- `ping` reports the live registry size and build identity; compatibility tests
  cover the frozen-T encoder/decoder and unknown message types.
- `Node::snapshot` shares bound programs and stored payloads behind `Arc`, and
  `Node::authorize` is the single token check used by `handle`/`handle_into`.
  `CancelGuard` installs a cooperative flag that the interpreter, `REPEAT`
  loops, and the stream-arena step loop check.
- The workspace test suite, protocol/determinism/transport/nonfpu suites, and
  the node crate tests are green on the integration branch.

## What this repository does not certify

- DAV2 full-depth correlation/error or FPS numbers. Those are
  companion-application evidence and are not reproducible from this
  repository; the canonical benchmark note lives with the companion depth
  repository.
- Live service, production replacement, or SBS integration. `fast-float`
  remains the production default; exact T2 is a canary/reference path.
- TLS beyond the shared-token trusted-LAN model, per-node fingerprints, or key
  rotation.
- Cross-host transport latency or throughput; the measured numbers are
  one-machine observations.
- The node application's async/store-backed behavior as a *language* claim.
  Those paths are gated in `crates/jotnar-node`, with the claim boundary in
  `showcase/node-os/`.
- F32/GPU performance or cross-backend float bit-identity.
- That the FNV-1a 64-bit hashes are a cryptographic content address. They are
  fingerprints for receipts/regression, not dedup or security. Result payload
  chunks use SHA-256 in the node application, but receipts remain fingerprints.

## Evidence provenance

- Language-side claims: checked-in Rust tests, `jotnar-det`, and
  `showcase/runtime/` commands.
- Node application claims: `crates/jotnar-node` tests and the QEMU/physical
  gates under `crates/jotnar-node/test/`, summarized in `showcase/node-os/`.
- Application-side depth claims: the companion depth-conversion repository
  (currently local-only). The language repository does not include its
  payloads, model weights, or benchmark scripts.

## Change control

- Do not cite this exhibit as proof of live serve, production depth
  performance, transport security, or a two-physical-node benchmark.
- Do not move companion application numbers into language-gated receipts, the
  root exhibit table, or the manual as if they were reproducible here.
- Async lifecycle, snapshots, cancellation, chunked results, scheduler, and
  node OS evidence belong in `showcase/node-os/`; keep this exhibit's language
  gates focused on determinism, receipts, and transport.
- Promotion to a separate numbered exhibit would also require the root README,
  `showcase/README.md`, `docs/LOOP.md`, the manual showcase chapter, and the
  manual output to be updated together.
