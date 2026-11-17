# Claim boundary: JotnarOS node OS

## Decision

JotnarOS is an **application showcase**, not an eleventh language exhibit and
not a production operating-system claim. It implements **Claim A**: a
Linux-hosted system layer described in Jotnar and emitted to Rust, with the
interpreter as the parity oracle. It does not implement **Claim B**, a
Jotnar-native kernel, bootloader, or driver model.

The language contract remains the ten exhibits and the tests under `tests/`,
`src/`, and `crates/jotnar-*`. The node application adds evidence for the
runtime and distribution paths without changing the language-repo claim.

## What this repository certifies for the node

Reproducible from this repository:

- **Model-blind execution.** The node's only execution path is
  `Node::handle("exec")` over a `Node::snapshot`; the job engine does not get a
  second interpreter. Programs, blobs, and streams are opaque to it.
- **Async lifecycle.** Bounded queue/worker pool, submit/status/result/cancel/
  list, idempotency, timeouts, cooperative cancellation, and queue-full
  rejection are covered by `crates/jotnar-node` tests and the host N5 gate.
- **Large results.** Store-backed chunked results carry SHA-256 and byte
  counts; reader leases and eviction are tested. Both the frozen and ported
  images passed a 67,280,000-byte / 65-chunk physical result.
- **Capability mediation.** The emitted `job-policy` module dispatches through
  the supervisor to the local job endpoint with the node token. Interpreted
  and emitted `module-step` agree on action/command/state, and the host N5
  gate exercises submit/status/list/cancel.
- **Multi-node scheduling.** Heartbeat health, draining, placement, retries,
  durable batch ledger, cached replay, resume, aggregation, and receipts are
  covered by `crates/jotnar-node/test/sched_two_nodes.sh`.
- **Identity and discovery.** Stable node IDs, mDNS advertisement, and
  controller discovery are implemented. Both the frozen and ported physical
  nodes were found with `auth=required`, 4 workers, and queue 64.
- **Image reproducibility.** `image/build-node-image.sh` pins Alpine and the
  kernel, builds node binaries for `x86_64-unknown-linux-musl`, and writes a
  GPT image plus a build manifest. The image hash is recorded in the manifest.

Host gates run on the integration branch:

```sh
cargo test -p jotnar-node-tools
./crates/jotnar-node/test/n5_job_capability.sh
./crates/jotnar-node/test/sched_two_nodes.sh
```

QEMU gates also passed on the integration branch at runtime
`0.1.0+a3bab886588b`:

```sh
./crates/jotnar-node/test/qemu_smoke.py --ssh-key ~/.ssh/id_ed25519 --m8
./crates/jotnar-node/test/qemu_smoke.py --m11
./crates/jotnar-node/test/qemu_jobs.sh
# two QEMU nodes, then:
ADDR_A=<host>:9998 ADDR_B=<host>:9999 COUNT=64 ./crates/jotnar-node/test/n4_imatmul.sh
```

Results: all 8 modules emitted with M8 parity; 96 M11 parity steps; N1/N2a
async submit/timeout/cancel, 200-job batch, and a 67,280,000-byte / 65-chunk
verified result; N4 64-job `IMATMUL` across two nodes with balanced 32/32
placement and identical outputs.

The ported image was flashed to USB and booted on the physical node
(`node-e10e85bc6f1c20c3`, runtime `0.1.0+a3bab886588b`). The physical gate
passed:

```sh
ADDR=192.168.1.116:9999 ./crates/jotnar-node/test/physical_jobs.sh
```

It reports 4 workers, queue 64, all 8 emitted modules including `job-policy`,
action-18 `job.submit` parity, submit → `done`, status/list, long-job cancel →
`canceled`, a 67,280,000-byte / 65-chunk verified result, a 200-job matmul2
batch, and a 50-job `vec n=65536` batch. mDNS discovery found the node with
`auth=required`. The earlier frozen-image physical evidence remains the N0–N5
freeze reference.

## What is not certified

- **No production service claim.** This is research/tooling evidence, not a
  live-service replacement.
- **No transport security.** `--token` is a cleartext shared secret in the JSON
  header. There is no TLS, mTLS, per-node fingerprint, trust-on-first-use, or
  key rotation. Discovery auto-registration trusts the LAN token.
- **No node durability.** The controller batch ledger survives restarts; the
  node's jobs and memory-backed result store do not. There is no node-side job
  journal.
- **No production scheduling.** Placement is a simple controller policy, not a
  scheduler with quotas, priorities, preemption, or fair-share guarantees.
- **No strong receipt address.** Result payloads use SHA-256; receipts still
  use FNV-1a fingerprints. Treat receipts as regression identifiers, not
  content-addressed security.
- **No physical multi-node claim yet.** N4's distributed IMATMUL run was on two
  QEMU nodes. The physical node has been gated alone. A two-physical-node run
  remains open.
- **No workload-performance claim.** Job rates and image results prove paths,
  not language or hardware performance.
- **No Claim B.** Linux provides boot, drivers, networking, filesystems,
  process isolation, and the console substrate. The emitted system layer is
  above that boundary. A Jotnar-native kernel is not started.

## Change control

- Keep concurrent TCP/jobs, cancellation, `Arc` snapshots, chunked results, and
  scheduler evidence in this application boundary; do not move them into the
  language contract or a numbered exhibit.
- When TLS/fingerprint trust, a node journal, or a physical two-node gate land,
  update this file and `showcase/runtime/CLAIM_BOUNDARY.md` together.
- If the node OS is later promoted to a numbered exhibit, update the root
  README, `showcase/README.md`, `docs/LOOP.md`, and the manual together.
