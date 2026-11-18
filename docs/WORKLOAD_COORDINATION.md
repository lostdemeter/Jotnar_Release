# Workload coordination

**Status:** first slice implemented; hardware inventory, adaptive split,
iterative rebalance, interactive HUD/viewer, failover, and recovery
**Scope:** model-aware planning above the CPU, GPU, and native executors

## Position

The controller should not become a generic VM or container orchestrator. The
first useful layer is smaller:

1. a typed resource view of every node;
2. a compiler-derived contract for every stage;
3. a measured feedback loop that turns receipts into placement decisions.

“Virtual hardware” here means a normalized capability/lease view. A node is a
set of typed resources (CPU, GPU, programs, queue limits, links), not a machine
we pretend is identical to every other machine.

The executors stay unchanged:

```text
jotnar-noded          model-blind listing execution
jotnar-gpu-runner     target-aware GPU execution
jotnar-native-runner  model-blind process execution
jotnar-workload       batch bundles, stage graph, ledger
jotnar-doom-coord     streaming two-node frame coordinator
```

Planning stays above them. No model manifests, weights, or placement policy
belong in an executor.

## 1. Resource model

`jotnar.hardware/1` normalizes discovery into a stable view:

```json
{
  "schema": "jotnar.hardware/1",
  "nodes": [
    {
      "node_id": "Jotnar GPU cuda 23142",
      "kind": "gpu",
      "addrs": ["192.168.1.116:9997"],
      "targets": ["cuda"],
      "gpus": [{"target": "cuda", "arch": "sm_120",
                "toolchain": "nvcc", "pid": "23142"}],
      "limits": {"workers": null, "queue": null, "max_frame": null},
      "telemetry": {"queued": null, "running": null, "warm_artifacts": null},
      "runtime_version": "0.1.0+5cbc157b6bdf",
      "auth": "required"
    }
  ],
  "links": [
    {"from_node": "...", "to_node": "...",
     "from": "192.168.1.116:9997", "to": "192.168.1.195:9997",
     "reachable": true, "rtt_us": 320}
  ]
}
```

Implemented now:

```sh
jotnar-net --discover-timeout-ms 5000 hardware
jotnar-net --discover-timeout-ms 5000 hardware --probe
```

`--probe` measures TCP connect RTT between node pairs. It is a placement hint,
not a performance claim; protocol-level ping and throughput probes come later.

The schema has room for measured profiles and live telemetry. The important
fields are not the architecture strings; they are the measured costs and the
current queue state.

## 2. Leases

A placement should reserve a slice of a node, not the whole node:

```text
lease = node + workers + gpu_slots + memory + bandwidth + expiry
```

This gives role isolation before we need cgroups, GPU MPS, or containers. A
lease also gives the controller a way to admit or reject a workload without
starving another one.

The first slice does not enforce leases yet. It defines the resource vocabulary
and uses measured throughput for the DOOM split.

## 3. Stage contract from Jotnar

The compiler already knows the facts placement needs. A stage descriptor should
carry:

```json
{
  "role_hint": "stateful-core|data-parallel|post-process|transport|store",
  "ops": ["INTERP"],
  "kind": "F",
  "shape_in": [200, 161, 3],
  "shape_out": [400, 322, 3],
  "bytes_in": 386400,
  "bytes_out": 1545600,
  "targets": ["hip", "cuda", "cpu"],
  "parallelism": "spatial",
  "state_bytes": 0,
  "determinism": "exact-float",
  "partitionable": {"axis": "columns", "min": 32, "max": 288}
}
```

The RV32IM core is the opposite shape: `state_bytes=64MB`,
`parallelism=sequential`, `role_hint=stateful-core`. That distinction is what
stops the controller from treating the game core and the upscale stage as
interchangeable.

The compiler should emit legal partitions and their costs. The controller
should not guess how to split a Jotnar stage; it should ask the language layer.

## 4. Roles and placement

| role | examples | placement bias |
|---|---|---|
| stateful-core | RV32IM/DOOM, interactive loops | low-latency node, pinned, checkpointable |
| data-parallel | INTERP, MATMUL, CONV, image ops | GPUs, split by measured throughput |
| transport | frame relay, compression, chunking | near producer/consumer, link-aware |
| store | checkpoints, artifacts, blobs | fast local disk or content-addressed store |
| sink/display | encode/GIF/display | near the user or final consumer |

Workload classes:

- **interactive**: deadline, p95 latency, failover, leases, cancellation;
- **batch**: throughput, retries, queueing, cost per job;
- **stream**: per-frame, backpressure, drop/duplicate policy.

The placement score should combine predicted latency, queue delay, transfer
time, warm-cache affinity, failure risk, and energy/cost. Hard constraints come
first: required ops, precision, memory, state size, security, deadline.

Links are first-class resources. Large state or high-rate frames should stay on
short/fast links; small requests can tolerate more hops.

## 5. Feedback loop

Every run already produces receipts. Use them:

- per-op, per-target timings;
- per-link transfer time and bytes;
- queue delay vs execution time;
- p50/p95/max;
- failure and retry rates.

Label bottlenecks:

```text
compute-bound -> replicate or move to a faster target
memory-bound  -> co-locate with data, reduce copies
network-bound -> split differently, compress, prefetch, or move the stage
queue-bound   -> add workers/leases or back off
```

Update the cost model from measured p50/p95 with an EWMA. Do not trust static
architecture names.

## 6. Implemented first slice: adaptive DOOM split

The two-node DOOM coordinator now calibrates both cards before recording:

1. warm the full-frame fallback on both nodes;
2. warm the even-split workers;
3. run the same frame on both cards three times;
4. take the median per-card time;
5. choose columns proportional to throughput;
6. warm the chosen shapes;
7. measure at the chosen split and adjust once if the per-column balance is
   materially different;
8. record frames, calibration, rebalance, and receipts.

`choose_split` is pure and tested. The overlap column is preserved for every
split, so bilinear upscaling stays continuous at the moving boundary.

Checked run:

```text
CUDA probe p50  41 ms
HIP  probe p50  18 ms
chosen split    98 CUDA / 222 HIP columns
rebalance       72 CUDA / 248 HIP columns
total           p50 42 ms, p95 44 ms
frame budget    66 ms at 15 fps
real time       yes
```

Before the adaptive split the fixed 50/50 run was `p50 71 ms, p95 77 ms` at
12 fps. The split is a real placement decision driven by measurement, not a
constant. The rebalance step catches the fact that per-job overhead is not
perfectly linear in column count.

The coordinator records the calibration, the chosen split, the rebalance, per
frame job ids, and per-card latency in `run.json` and `frames.jsonl`.

The same run now has three operational checks:

- **interactive path**: `jotnar-doom-view` displays the stitched frame with a
  live HUD (nodes, roles, columns, per-node latency, frame budget, rolling
  p50/p95, failover/recovery state) and forwards keypresses to the RV32IM core
  over TCP;
- **outage path**: both nodes pre-warm a full-frame fallback. If a node fails,
  the coordinator marks it down, records the failure, and finishes the frame
  on the survivor. The checked outage gate completes 45/45 frames in both
  directions with exit 0. CUDA outage falls back to HIP at
  `p50 46 ms / p95 50 ms`; HIP outage falls back to CUDA at
  `p50 114 ms / p95 135 ms`.
- **recovery path**: every `--recover-every` frames the coordinator probes a
  failed node in a background thread; if it answers, it warms the split shapes
  and returns to split mode. The frame loop does not block on a dead network
  path. The checked recovery scenarios return to `split` at frame 221 with one
  recovery event each.

This is the first measured role-filling loop: calibration decides the split,
runtime failure decides the fallback role, and recovery restores the split
when the node returns.

## 7. Next slices

1. **Telemetry**: executors report queue depth, running jobs, GPU utilization,
   memory pressure, and warm artifacts on `ping`/capability records.
2. **Stage descriptors**: compiler emits `jotnar.stage/1` requirements and
   legal partitions.
3. **Leases**: reserve workers/GPU slots/memory/bandwidth; enforce admission.
4. **Continuous rebalance**: rebalance while the game is running, not only at
   startup and one follow-up; use EWMA and hysteresis to avoid shape churn.
5. **General streaming ABI**: move the DOOM-specific coordinator logic into a
   reusable streaming stage contract.
6. **Link-aware placement**: use the measured RTT/bandwidth matrix for stage
   placement, not only for display.

## 8. Boundaries and risks

- Keep the CPU node model-blind and the GPU runner target-aware.
- Do not let static capability strings drive placement; measure.
- Do not hide placement changes from the receipt.
- Do not over-centralize: executors must remain useful if the controller is
  gone.
- Capability advertisements can be wrong or stale; gate with probes and
  runtime checks.
- Measurement noise is real; use p50/p95 and EWMA, not one sample.
- Determinism must survive placement: receipts record executor, target, and
  exactness verdict.
- The trusted-LAN token is not TLS.
