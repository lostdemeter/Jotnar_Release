# Capability exchange

**Status:** discovery implemented in `jotnar-net`; execution stays separate
**Scope:** control-plane inventory for CPU nodes and GPU runners

## Why

A workload controller must answer one question before placement:

> can this node run this stage, with this precision, within these limits, and
> how warm is it?

Today the CPU node and GPU runner advertise through separate mDNS services.
`jotnar-net capabilities` merges them into one list without moving execution
into the node daemon and without making the CPU node target-aware.

## Services

| kind | mDNS service | execution owner |
|---|---|---|
| CPU | `_jotnar-node._tcp.local.` | `jotnar-noded` |
| GPU | `_jotnar-gpu._tcp.local.` | `jotnar-gpu-runner` |
| native | `_jotnar-native._tcp.local.` | `jotnar-native-runner` |

`jotnar-net` browses all three, normalizes the records, sorts them, and prints
JSON. It never submits a job and never bypasses a token check.

## Record

Common fields:

```json
{
  "kind": "cpu|gpu",
  "target": "cpu|hip|cuda",
  "addrs": ["host:port"],
  "port": 9999,
  "auth": "required|none",
  "runtime_version": "0.1.0+...",
  "fullname": "_service._tcp.local.",
  "instance": "advertised instance"
}
```

CPU adds:

```json
{"node_id": "...", "caps": "jobs.v1", "workers": "8", "queue": "64",
 "result_transport": "inline+chunked"}
```

GPU adds:

```json
{"arch": "sm_120|gfx1100", "toolchain": "nvcc|hipcc", "pid": "..."}
```

Native adds:

```json
{"target": "process", "programs": "jotnar-demo-doom", "platform": "linux-x86_64",
 "pid": "..."}
```

`programs` is a comma-separated allow-list of executable basenames. A
controller can use it as a placement hint; the runner still enforces the
allow-list when a job is submitted.

String-valued numeric fields remain strings in discovery so old adverts stay
readable. A workload controller can parse them as needed.

## Command

```sh
jotnar-net --discover-timeout-ms 5000 capabilities
```

The timeout is split between the three service browsers. Example on the live
fleet plus one local CPU node:

```text
count 6  cpu=1 gpu=4 native=1
gpu cuda 192.168.1.116:9996 sm_120 nvcc required
gpu cuda 192.168.1.116:9997 sm_120 nvcc required
gpu hip  192.168.1.195:9996 gfx1100 hipcc required
gpu hip  192.168.1.195:9997 gfx1100 hipcc required
native process 127.0.0.1:9998 jotnar-demo-doom required
cpu cpu  127.0.0.1:9999      node-... jobs.v1 workers=8 queue=64
```

`jotnar-net discover` remains CPU-only, `jotnar-gpu-net` remains the GPU
job controller, and `jotnar-native-runner` owns native process execution.
Capabilities are an inventory, not a new execution path.

`jotnar-net hardware` normalizes the same records into `jotnar.hardware/1`
with typed CPU/GPU/native nodes and optional link probes:

```sh
jotnar-net --discover-timeout-ms 5000 hardware --probe
```

The profile is the controller's resource view; see
[`docs/WORKLOAD_COORDINATION.md`](WORKLOAD_COORDINATION.md).

## Boundaries

- The CPU node stays model-blind.
- The GPU controller stays target-aware and separate.
- Capability discovery is informational; trusted-LAN tokens still gate work.
- No model semantics, weights, or stage planning live here.

## Next

The executor interface is documented in
[`docs/EXECUTOR_CONTRACT.md`](EXECUTOR_CONTRACT.md): `submit`, `job-status`,
`job-result`, and `job-cancel`, implemented by the CPU node, the GPU runner,
and the native runner. A workload controller consumes the capability list,
picks an executor, and uses that contract. Discovery, execution, and workload
planning stay separate. See
[`docs/WORKLOAD_CONTROLLER.md`](WORKLOAD_CONTROLLER.md) for the next layer.
