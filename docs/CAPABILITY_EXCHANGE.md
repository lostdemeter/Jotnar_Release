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

`jotnar-net` browses both, normalizes the records, sorts them, and prints JSON.
It never submits a job and never bypasses a token check.

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

String-valued numeric fields remain strings in discovery so old adverts stay
readable. A workload controller can parse them as needed.

## Command

```sh
jotnar-net --discover-timeout-ms 5000 capabilities
```

The timeout is split between the two service browsers. Example on the live
fleet plus one local CPU node:

```text
count 5  cpu=1 gpu=4
gpu cuda 192.168.1.116:9996 sm_120 nvcc required
gpu cuda 192.168.1.116:9997 sm_120 nvcc required
gpu hip  192.168.1.195:9996 gfx1100 hipcc required
gpu hip  192.168.1.195:9997 gfx1100 hipcc required
cpu cpu  127.0.0.1:9999      node-... jobs.v1 workers=8 queue=64
```

`jotnar-net discover` remains CPU-only and `jotnar-gpu-net` remains the GPU
job controller. Capabilities are an inventory, not a new execution path.

## Boundaries

- The CPU node stays model-blind.
- The GPU controller stays target-aware and separate.
- Capability discovery is informational; trusted-LAN tokens still gate work.
- No model semantics, weights, or stage planning live here.

## Next

1. Add an executor interface that both `jotnar-noded` and the GPU runner can
   implement.
2. Let a workload controller consume the capability list and submit stages to
   the correct executor.
3. Keep discovery, execution, and workload planning as separate layers.
