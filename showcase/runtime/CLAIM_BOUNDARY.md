# Claim boundary: deterministic job server (runtime exhibit research)

## Decision

The job-server work is part of **exhibit 10, interpreter + model-blind node**.
It is a first-class language/runtime showcase, but it is **research, not a
production serve claim**. Adding it does not change the ten-exhibit count and
does not promote application performance or live-service behavior to a
language-repo gate.

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
- `ping` reports the live registry size (94), and the frozen-T encoder and
  decoder agree on the pre-T2 wire layout.
- Release gates: **231/231** workspace tests, `cargo fmt --check` clean,
  and the protocol/determinism/transport/nonfpu suites green.

## What this repository does not certify

- DAV2 full-depth correlation/error or FPS numbers. Those are
  companion-application evidence and are not reproducible from this
  repository; the canonical benchmark note lives with the companion depth
  repository.
- Live service, production replacement, or SBS integration. `fast-float`
  remains the production default; exact T2 is a canary/reference path.
- TCP parity for large outputs. A 64 MiB frame carries ~6.7M T2 elements;
  application neck/head outputs are much larger. Chunked/segmented or
  store-backed result transport is Stage E.
- Concurrent TCP jobs, cancellation/timeouts, `Arc` snapshots that decouple
  `program`/`store` from a long `exec`, or a per-job memory policy.
- F32/GPU performance or cross-backend float bit-identity.
- That the FNV-1a 64-bit hashes are a cryptographic content address. They
  are fingerprints for receipts/regression, not dedup or security.
- Cross-host transport latency or throughput; the measured numbers are
  one-machine observations.

## Evidence provenance

- Language-side claims: checked-in Rust tests, `jotnar-det`, and
  `showcase/runtime/` commands.
- Application-side claims: the companion depth-conversion repository
  (currently local-only). The language repository does not include its
  payloads, model weights, or benchmark scripts.

## Change control

- Do not cite this exhibit as proof of live serve, large-output TCP parity,
  or production depth performance.
- Do not move companion application numbers into language-gated receipts,
  the root exhibit table, or the manual as if they were reproducible here.
- When Stage E lands (concurrent TCP, cancellation/timeouts, `Arc` snapshots,
  memory policy, chunked large outputs, stronger content hash), update this
  boundary and the exhibit together. Promotion to a separate numbered exhibit
  would also require the root README, `docs/LOOP.md`, the manual showcase
  chapter, and the manual output to be updated together.
