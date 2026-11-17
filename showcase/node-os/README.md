# JotnarOS node OS

> **Application showcase, not an eleventh language exhibit.** This page
> describes a system built on the language/runtime: a Linux-hosted node that
> runs a model-blind job server, a payload supervisor, and an emitted system
> layer. The language claims stay in exhibits 1–10. The boundary between the
> Linux substrate and a Jotnar-native kernel is in
> [`CLAIM_BOUNDARY.md`](CLAIM_BOUNDARY.md).

JotnarOS is a small bootable x86-64 node. The boot control plane is emitted
Rust (`jotnar-osd`), not the reference interpreter. The interpreter remains the
parity oracle. The node sees a listing, named blobs, and typed stream bytes — it
never sees a workload name, model name, or placement decision.

```text
controller / scheduler (smart: owns workload + placement)
   |
   |  framed Jotnar protocol, TCP 9999, shared token
   v
jotnar-noded (model-blind)
   |-- Node::snapshot -> async job engine -> Node::handle("exec")
   |-- payload-* / event-* / module-* -> supervisor socket
   v
jotnar-supervisord
   |-- payload threads (TICK/STDIN in, STDOUT out)
   |-- event ring + capability shims (console/file/proc/timer/net/job)
   `-- /var/lib/jotnar/payloads/<name>/...
```

## What the node adds

- **Async jobs.** Bounded queue and worker pool, submit/status/result/cancel/
  list, idempotency keys, timeouts, and cooperative cancellation between
  instructions and `REPEAT` iterations.
- **Immutable job snapshots.** Programs and stored payloads are `Arc`-shared;
  a job sees the registry at submission time while later uploads/drops proceed.
- **Store-backed results.** Inline results for small outputs; chunked,
  SHA-256-verified results for large outputs, with reader leases and eviction.
- **Capabilities.** The emitted system layer can submit and manage jobs without
  knowing the scheduler. The supervisor and emitted osd use one local job
  endpoint, with interpreted/emitted parity.
- **Supervision.** Long-lived payloads survive node-service restarts; STATE is
  snapshotted to the data partition.
- **Node identity and discovery.** Stable node IDs and mDNS advertisements
  (`_jotnar-node._tcp.local.`) expose capabilities, workers, queue depth, auth
  policy, result transport, and runtime version.
- **Distribution.** `jotnar-net` adds heartbeat health, draining, placement,
  retries, a durable per-batch ledger, cached summaries, resume, aggregation,
  and receipts.

## Modules

Eight emitted policy modules ship in the image:

```text
console-policy  system-log-policy  denied-policy  file-policy
proc-policy     timer-policy       net-policy     job-policy
```

`jotnar-osd --serve` runs the emitted steps; `jotnar-payload module-step` runs
the interpreted module as the oracle. `module-step` and `module-step-emitted`
must agree on action, command, and state.

## Build and run

```sh
# Build the bootable image (pinned Alpine + kernel; workspace runtime)
./crates/jotnar-node/image/build-node-image.sh

# QEMU: interactive console, or the automated gates
./crates/jotnar-node/test/run_qemu.sh
./crates/jotnar-node/test/qemu_smoke.py --ssh-key ~/.ssh/id_ed25519 --m8
./crates/jotnar-node/test/qemu_smoke.py --m11
./crates/jotnar-node/test/qemu_jobs.sh

# Host scheduler gate (two local nodes)
./crates/jotnar-node/test/sched_two_nodes.sh

# Physical node on the LAN
ADDR=192.168.1.116:9999 ./crates/jotnar-node/test/physical_jobs.sh
```

The builder writes `artifacts/jotnar-node-x86_64.img`, `node.token`,
`jotnar-ctl`, `jotnar-payload`, `jotnar-osd`, and `build-info.txt`.
`crates/jotnar-node/image/flash-usb.sh` verifies the `.sha256` sidecar before
writing a USB disk.

## Evidence

The frozen N0–N5 image is
`b0d52c257aa1fe9d10ea610daa3f16360b2416024eec70c3e63e1bc202440785`
(516 MiB, runtime `723a56b76f59`). On the frozen physical USB image
(`node-e75cfaea5d5a0826`, 192.168.1.116):

- `ping` reports 4 workers, queue 64, `inline+chunked`, and the `jobs.v1`,
  `payload.v1`, `events.v1` features;
- all 8 emitted modules are present, including `job-policy`;
- interpreted and emitted `job.submit` agree on action 18; interpreted submit
  reaches `done`; emitted status/list work; a long job cancels to `canceled`;
- mDNS discovery found the node with `auth=required`, and
  `jotnar-net --discover health` registered and checked it;
- the N2a large-result gate still passes: 67,280,000 bytes in 65 verified
  chunks, and 50 × `vec n=65536` jobs at 55.1 jobs/s.

On QEMU, the same image passes M8 (8 modules), M11 standalone (96 randomized
parity steps including `job-policy`), and the N2a job gate. The N4 distributed
IMATMUL benchmark ran 64 jobs across two QEMU nodes at 12.94 jobs/s with
identical outputs; duplicate replay and mid-batch node-loss retries pass.

The integration branch was QEMU-gated on 2026-11-17 at runtime
`0.1.0+a3bab886588b`:

- M8: all 8 modules emitted, whole-system interpreted/emitted parity PASS;
- M11 standalone: 96 randomized parity steps PASS;
- N1/N2a jobs gate: async submit, timeout, cancel, 200-job batch, and a
  67,280,000-byte / 65-chunk verified result PASS;
- N4: 64 `IMATMUL` jobs across two QEMU nodes, balanced 32/32, identical
  outputs, 17.17 jobs/s.

The workspace tests and host gates (`n5_job_capability.sh`,
`sched_two_nodes.sh`) also pass.

The ported image was flashed to USB and booted on the physical node
(`node-e10e85bc6f1c20c3`, runtime `0.1.0+a3bab886588b`): all 8 modules are
emitted, `physical_jobs.sh` passes, mDNS discovery reports `auth=required`, and
the N5 path passes action-18 parity plus submit/status/list/cancel. The earlier
frozen-image evidence below remains the reference for the N0–N5 freeze. See
[`CLAIM_BOUNDARY.md`](CLAIM_BOUNDARY.md).

## What this is not

- Not a Jotnar-native kernel or bootloader. Linux is the substrate; the system
  layer above it is described in Jotnar and emitted to Rust. That is Claim A.
- Not a production cluster. The shared token is cleartext, there is no TLS or
  mTLS, no per-node fingerprint trust, and no key rotation.
- Not durable at the node. The controller has a batch ledger; node jobs and
  memory-backed results do not survive a node restart.
- Not a performance claim. The numbers above are one-machine observations to
  prove the paths work, not a benchmark of the language or hardware.
- Not the language contract. Certified kernels, receipts, and thread
  determinism remain in [`../runtime/`](../runtime/) and
  [`../protocol/`](../protocol/).

## Source

- [`crates/jotnar-node/`](../../crates/jotnar-node/) — daemon, supervisor, job
  engine, scheduler, osd, image builder, gates.
- [`src/protocol.rs`](../../src/protocol.rs) — model-blind `Node`, `snapshot`,
  `authorize`, and the in-process frame contract.
- [`src/runtime.rs`](../../src/runtime.rs) — cancellation plumbing.
- [`src/bin/jotnar-ctl.rs`](../../src/bin/jotnar-ctl.rs) — control client with
  job/result commands.
