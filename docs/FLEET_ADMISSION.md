# Fleet admission: telemetry, leases, priority, and the interactive SLO

**Status:** implemented and gated on the live CUDA/HIP fleet
**Scope:** controller-side planning above the executors; executors only report

The workload controller is now a **fleet controller** for batch and interactive
work on the same executors. It can see live executor state, reserve capacity,
admit or defer batch work, and explain every decision in the receipt.

> North star: a distributed machine that runs typed dataflow graphs and can
> prove, for every stage, what ran, where, why, and whether it matched the
> reference.

## Boundary

```text
jotnar-noded          model-blind listing execution; reports health
jotnar-gpu-runner     target-aware GPU execution; reports health
jotnar-native-runner  model-blind process execution; reports health
jotnar-workload       planning, leases, admission, ledger, receipt
jotnar-doom-coord     interactive streaming coordinator; registers leases
```

Discovery stays static inventory. Live telemetry is a request/response call.
No executor gains placement policy; no placement decision moves into a runner.

## Telemetry

`health` is a normal framed request on the job contract, authenticated by the
existing token:

```json
{"v": 1, "id": 7, "type": "health", "auth": "..."}
```

```json
{
  "type": "health-ok", "ok": true,
  "schema": "jotnar.executor.health/1",
  "kind": "gpu", "target": "cuda", "arch": "sm_120",
  "ts_ms": 1763700000000, "uptime_ms": 812345,
  "limits": {"workers": null, "queue": null, "max_frame": 67108864},
  "activity": {"queued": 0, "running": 1, "retained": 14,
               "warm_workers": 2, "warm_artifacts": 2},
  "jobs": [{"id": "gpu-...-41", "state": "running",
            "created_ms": 0, "started_ms": 0, "finished_ms": null,
            "elapsed_ms": 90}]
}
```

`jobs` is a bounded window (last 32). `workers`/`queue` are `null` where the
executor has no fixed pool; the GPU and native runners spawn one thread per job
today. The controller snapshots the reply and records `age_ms`; a failed
health call is recorded, not fatal.

The CPU node, GPU runner, and native runner all implement `health`. `ping`
remains the cheap static probe; `job-list` remains result-store bookkeeping.
No live telemetry is advertised over mDNS.

## Leases

A lease is a reservation, not a lock, VM, or container. All participants share
one `--fleet-dir`; the controller and the DOOM coordinator write the same
directory.

`jotnar.lease/1`:

```json
{
  "schema": "jotnar.lease/1",
  "id": "doom-23142-cuda-1763700000000",
  "owner": "jotnar-doom-coord:23142",
  "class": "interactive",
  "target": "cuda",
  "slots": 1,
  "created_ms": 0, "renewed_ms": 0, "expires_ms": 0,
  "pid": 23142,
  "note": "two-node DOOM frame loop"
}
```

- Expiry is mandatory. A lease that is not renewed is gone; a killed owner
  leaks at most one expiry interval.
- The DOOM coordinator registers one interactive lease per GPU target before
  its frame loop (before `frames.jsonl` exists), renews it each frame, and
  revokes it on exit. The contention script does not fabricate leases.
- The batch controller holds an expiring admission per in-flight GPU stage.
- Enforcement is **advisory**: the controller refuses to over-admit, but the
  runner still accepts what it is given. The receipt says so.
- DOOM calibration uses 7 probes by default. Three probes produced splits from
  69 to 115 CUDA columns across runs because the CUDA samples were noisy;
  7 kept the interactive p99 under budget in consecutive alone runs.

## Priority and admission

| class | work | policy |
|---|---|---|
| interactive | DOOM frames, deadline-bound | holds the declared slot for its lease duration |
| batch | GPU pipeline stages, deferrable | admitted only into capacity not reserved by a live interactive lease |

`--fleet-slots` is the controller's declared capacity per target (default `1`).
An interactive lease with `slots = 1` therefore consumes the target while the
lease is live; batch work defers rather than sharing the GPU. This is a
measured choice: on the current fleet one co-resident batch stage pushed DOOM
p99 from ~58 ms to ~71 ms against a 66 ms budget. With exclusive reservation,
the batch load no longer causes interactive misses. The batch work is deferred,
not rejected, and proceeds FIFO when the lease ends.

Algorithm (under one fleet lock):

1. reap expired leases, admissions, and waiters;
2. sum live interactive slots for the target;
3. grant if `in_flight_batch < capacity - interactive_slots` and this waiter is
   the oldest waiter for the target;
4. otherwise refresh the waiter and retry with a short backoff;
5. after `--fleet-max-wait-ms` reject with a recorded reason.

Waiters are ordered by original `requested_ms` (tie: id), which gives FIFO and
round-robin across independent clients. A dead waiter expires in 2 s; an
admission expires with the stage; a rejection is a recorded policy outcome,
never a silent skip.

Per-stage receipt record:

```json
{
  "target": "cuda", "class": "batch",
  "decision": "admitted",
  "waited_ms": 10893, "attempts": 220, "deferrals": 219,
  "reason": "batch slot available after 219 deferral(s): target cuda has 1 interactive slot(s) reserved by ...",
  "capacity": 1, "interactive_slots": 1, "in_flight_batch": 0,
  "interactive_leases": [{"id": "...", "owner": "...", "expires_ms": 0}]
}
```

The record keeps the leases seen during deferral even if they expired before
the grant, so the receipt shows *why* the work waited.

## Interactive SLO

Budget = `1000 / fps` (66 ms at 15 fps).

- **Window:** every frame the coordinator records while batch clients are
  running.
- **p99:** ≤ budget.
- **max:** ≤ 2 × budget (132 ms), zero tolerance above it.
- **Paired control rule:** the contended p99 must stay within 10 ms of the
  DOOM-alone p99 (and within budget), and the contended max must not exceed the
  alone max or 2 × budget. This separates admission-control failures from
  executor-internal stalls.
- Batch side: every output matches the single-node reference, every client
  completes, and every admission is recorded with telemetry.

The gate records both the absolute SLO and the paired control rule, plus every
frame above the ceiling and which half (core, CUDA, HIP) dominated it.

For a `jotnar.stage/1` stage, the controller also validates the descriptor's
declared output kind/shape/bytes against the executor result. The descriptor's
`determinism` class is recorded; the current comparison mode is exact bytes,
so an `exact` verdict is a stronger outcome than a `float-agreement` claim.

## Commands

```sh
# One shared directory for the coordinator and all batch clients.
python3 scripts/doom_gpu_contention_gate.py --no-build \
  --items 16 --size 128x128x3 --clients 4 \
  --out /tmp/opencode/fleet-a/contention

# Manual run:
jotnar-doom-coord ... --fleet-dir /tmp/opencode/fleet --lease-ms 3000
jotnar-workload --token test run ... \
  --fleet-dir /tmp/opencode/fleet --fleet-slots 1 \
  --fleet-max-wait-ms 60000
```

Without `--fleet-dir`, behavior is unchanged (best effort).

## Limits and stop conditions

- Advisory enforcement: a runner still accepts work from a client that does
  not honor the lease table. The token is trusted-LAN, not identity.
- The fleet directory must be on shared storage visible to all controllers and
  the coordinator; timestamps assume roughly synchronized clocks.
- Capacity is an integer concurrency budget, not GPU memory or bandwidth.
  There is no preemption of a running GPU kernel.
- A co-resident batch stage on the current CUDA/HIP pair materially degrades
  the interactive tail; the implemented policy defers it. Finer concurrency
  needs runner-side priority/preemption or GPU partitioning.
- Rare 200 ms+ executor stalls are present even in the DOOM-alone baseline on
  an idle-clock GPU; the receipt records them separately. The strict max SLO
  needs a host-clock fix (persistence/boost) before it can be claimed.
- CPU stages are not yet admitted through the lease table; the current
  interactive competitor is GPU-only.

See [`PROPOSAL.md`](../artifacts/jotnar-fleet/PROPOSAL.md) for the design
decision record and [`RESULTS.md`](../artifacts/jotnar-fleet/RESULTS.md) for
measurements.
