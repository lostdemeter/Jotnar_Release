# Runtime research: warm interpreter + network node

Status: merged runtime research note. The warm `jotnar-repl`, TCP
`jotnar-node`/`jotnar-ctl`, transport bench, and deterministic threading
work are in place; see [`RUNTIME_THREADING.md`](RUNTIME_THREADING.md) for the
model-blind determinism design. Application-scale measurements live in the
companion depth-conversion repository, not here. The goal is twofold:
(1) a small-footprint processor that takes Jotnar listings and processes
them like an interpreted language locally, and (2) a network stack so
the same processor can live on another machine, model-blind, and do work
for a smart controller.

This note records what the evidence says today, the candidate shapes for
(1) and (2), and what we are building first to let measurement arbitrate.

## 1. What exists today

| piece | where | note |
|---|---|---|
| assembler / verifier | `src/asm.rs` | IMPORT/DEF/CALL expansion, REPEAT/BREAK, `check`/`verify` |
| reference interpreter | `src/runtime.rs` (`run`, `run_text`, `repeat`) | `run_text` reassembles every call; `run(&BoundProgram, &Feeds)` is the warm reuse path |
| JSON payload codec | `src/payload.rs` | `{"streams": {...}}` with `F64/F32/I64/U8/Bool/T` |
| model-blind node | `src/protocol.rs` (`Node`, `NodeHandle`) | in-process boundary; TCP front end in `jotnar-node` |
| frame codec | `src/protocol.rs` (`encode_frame`/`decode_frame`) | `u32BE \|\| JSON header \|\| payload`, `PROTO_V=1` |
| CLI | `src/bin/jotnar.rs` | `check verify run emit-c build repeat ops`; one-shot: file + `--payload` → process → exit |

The frame format is TCP-ready and now has a TCP front end: `showcase/runtime/`
documents `jotnar-node` wrapping the same `Node::handle` boundary.

## 2. Measurements (release build, this machine)

CLI one-shot (`jotnar run`, includes process spawn):

```text
matmul2 (12 ops)   median 0.83ms over 50 runs, min 0.79ms
saxpy  (2 ops)     similar; check-only median ~0.75ms
futhark.asm (5514 instructions, 16 inputs) check median 7.92ms over 20 runs
```

Compile flow for the same tiny listing:

```text
emit-c  ~1.1ms
build   ~54.8ms (gcc)
run exe ~0.5ms (plus reading payload format of interest)
```

Warm protocol fan-out (`jotnar-demo-protocol`, 4 nodes / 12 jobs):

```text
12 jobs in ~10ms per batch, byte-identical across orders (RESULT: ALL OK)
```

Reading: the interpreter already beats compile-then-run for one-shot work
(~1ms vs ~56ms). Assembly scales roughly linearly (~1.4µs per instruction
from the futhark check). The per-command cost the user complains about is
mostly process spawn + reassembly + JSON round-trips when driving many
payloads from a host loop, not op execution.

Binary sizes (release): `jotnar` 4.8M, `jotnar-demo-protocol` 4.4M.
Both pull in `image` (png), Verilog, RV32/DOOM, CUDA/C codegen, etc.
A small-footprint processor has room to feature-gate those out later.

## 3. Candidate shapes

### 3a. Local interpreter ("Jotnar as Python")

Goal: feed commands in, get results out, no compile step.

- **A. Whole-program warm runner (chosen for prototype 1).**
  Assemble once at startup, then loop over NDJSON payloads on stdin,
  write NDJSON results on stdout. This skips per-command process spawn
  and reassembly, reuses `runtime::run(&BoundProgram, &Feeds)`, and maps
  directly onto the existing JSON codec. A Python driver can `subprocess`
  it, or a human can pipe payloads through it. Fail-loud per request
  (error JSON out, process stays alive) instead of CLI exit.

- **B. Line-at-a-time stepper.**
  Every line `OUT = IADD(...)` executes against a persistent stream
  table. Feels more like Python, but the frontend does IMPORT/DEF/CALL
  expansion globally and the verifier is whole-program, so this needs
  either incremental binding (smallest useful step, unknown effort) or
  reassembly on every line (no warm win). Defer until A shows a real
  pain point.

- **C. Persistent `repeat()`/STATE loops.**
  Host-driven resident pipelines already exist (`jotnar repeat`). The
  warm runner subsumes this; no separate binary needed.

### 3b. Network node ("model-blind worker")

Goal: ship the processor to another machine that never sees the workload.

Wire facts: header `{"v":1,"id":...,"type":...}` + length-prefixed payload;
message types `ping/pong`, `program/program-ok`, `store/store-ok`,
`drop/drop-ok`, `list/list-ok`, `exec/exec-ok`, `error`; streams are
`{"kind","dtype","shape"}` + little-endian bytes; `exec` binds inputs from
the blob shelf or inline segments and returns outputs as inline segments.
`Node::handle` is already the whole server logic; transport is a drop-in.

Candidates to benchmark before committing (as agreed):

- **TCP + framed codec (default expectation).** One client per server,
  one request at a time (matches `protocol.rs` comment: "fan-out lives
  in the controller"). Lowest new code: read `u32BE`, read body, call
  `Node::handle`, write reply. No TLS in v1; the threat model is a
  trusted LAN/cluster. Must add a max-frame limit (currently `decode_frame`
  requires exact length match and will allocation-fail on a huge prefix).
- **TCP + JSON-only.** Simpler to debug with `curl`, but integers-in-JSON
  lose the exact `i64` transport (`"data":["..."]` string form exists for
  that reason) and binary payloads need base64. Only wins if the frame
  codec turns out to bite in practice.

Benchmark plan: same jobs through (i) in-process `NodeHandle`, (ii) TCP
frames, (iii) TCP JSON. Measure throughput (jobs/s), p50/p99 latency,
and bytes-on-wire for a representative payload (e.g. a 256×256 U8 image).
Determinism is the gate: same program + same input bytes must give same
output bytes regardless of transport.

## 4. Prototype 1: `jotnar-repl` (this branch)

```sh
cargo run --release --bin jotnar-repl -- showcase/language/saxpy.asm [--outputs OUT]
# then, per line on stdin:
{"streams":{"A":{"kind":"I64","shape":[4],"data":["3","3","3","3"]},"X":...,"Y":...}}
# per line on stdout:
{"streams":{"OUT":{"kind":"I64","shape":[4],"data":["13","26","39","52"]}}}
# errors stay on the same stream so the process survives a bad request:
{"error":"..."}
```

Design notes:

- Reads the listing path from argv, `--stdlib DIR` overrides `IMPORT`
  search exactly like `jotnar` (`AsmOptions`, origin = listing path).
- Assembles once with `asm::assemble`, then reuses `runtime::run` for
  every payload. `--stats` prints assemble/run timings to stderr.
- `--outputs a,b` filters results; default emits every stream.
- Errors are emitted as `{"error": msg}` per line and the loop
  continues; assemble failure at startup exits non-zero.
- Empty lines are skipped; `{"cmd":"exit"}` / EOF ends the loop.

Known limits (intentional for the prototype):

- No `IMPORT` watch or reassembly on file change (explicit restart).
- No raw-binary/`npy` payload adapters yet (backlog item); JSON only.
- Single listing per process; multi-program jobs belong on the network
  node (Section 3b), which already does program/blob registry per node.

## 5. Phase 1 results: TCP node + transport benchmark

Added on this branch: `jotnar-node` (TCP server wrapping `Node::handle`),
`jotnar-ctl` (framed shell client), `jotnar-bench` (head-to-head).

```sh
cargo run --release --bin jotnar-node -- --bind 127.0.0.1:9999
cargo run --release --bin jotnar-ctl -- --addr 127.0.0.1:9999 ping
cargo run --release --bin jotnar-ctl -- --addr 127.0.0.1:9999 program tile showcase/language/matmul2.asm
cargo run --release --bin jotnar-bench -- --jobs 2000
```

Bench: same jobs, single sequential client per mode, `Node::handle`
underneath in all three. Small payload (matmul2, 2000 jobs of 8× `i64`):

```text
mode        median  p99     jobs/s   bytes/job (req->resp)
inproc        0.029   0.050  33017.1         995
tcp-frames    0.039   0.062  24140.6         995
tcp-json      0.061   0.088  16182.3         728
determinism: all modes matched reference (2000 jobs)
```

Large payload (`vec`: `OUT = IADD(X, Y)`, `n` `i64` per stream) to test
bandwidth scaling:

```text
n=16384, 100 jobs:
mode        median  p99     jobs/s   bytes/job
inproc        0.186   0.579   4969.2    393566
tcp-frames    0.495   0.701   1950.1    393566
tcp-json      4.393   7.661    216.5    366324

n=65536, 20 jobs:
mode        median  p99     jobs/s   bytes/job
inproc        0.854   3.069    995.5   1573214
tcp-frames    1.949   3.640    499.6   1573214
tcp-json     17.060  34.813     55.5   1561035
```

The frame-vs-JSON gap widens to ~9× at 64K elements and holds: in-process
scales linearly (0.022/0.186/0.854 ms at 1K/16K/64K), frames stay within
~2× of it, JSON is CPU-bound by the `Feeds` → JSON → frame → `EXEC`
double conversion plus integer-string parsing. Bytes-on-wire for small
ints can slightly favour JSON (728 vs 995 small; 366K vs 394K at 16K),
but speed still favours frames throughout. No crossover found.

Two early false starts are recorded so nobody re-litigates them:

1. Without `TCP_NODELAY`, the NDJSON path measured ~41 ms/job — a Nagle
   + delayed-ACK stall, not JSON cost. All TCP endpoints now set
   `set_nodelay(true)` and coalesce the JSON reply into one write.
2. The JSON variant preserves `i64` exactly via the string data form
   (`"data":["..."]`), so it is exact here; its cost is the double
   conversion (`Feeds` → JSON → frame → `EXEC` → frame → JSON → `Feeds`)
   plus integer-string parsing, not base64.

Verdict: **framed TCP is the canonical transport.** It costs ~10 µs over
in-process on loopback and is ~1.6× faster than JSON with zero extra
conversion. JSON stays a benchmark arm and a debugging aid, not a
second supported wire. `jotnar-node` defaults to loopback, caps frames
at 64 MiB by default (`--max-frame`, oversize → `error/too-large` and
connection close since the NM prefix cannot be trusted), and persists
one `Node` (program registry + blob shelf) across sequential clients.

Preliminary observation (single run, not a gate): warm REPL vs cold CLI
for 50× matmul2 payloads was 1.7 ms total vs 45.8 ms total (~26×), i.e.
0.03 ms vs 0.92 ms per payload. No checked-in reproducible warm-vs-cold
command yet.

Footprint check (single-machine one-off observation, not a gate):
`jotnar-node`/`jotnar-repl` are both 4.2M (3.3M stripped),
`jotnar-ctl` is 804K because it never links the interpreter. Dead-code
elimination already removes rv32/showcase/codegen/image from node/REPL
builds. The remaining text (~3.2M) is the registry + `phi-core`
substrate, i.e. the actual language. A feature split would save only a
few percent of already-eliminated code and would fork maintenance for
it, so it is deferred until measurements say otherwise. Ship stripped
binaries.

## 6. Phase 2 results: image/float payloads, blob shelf, auth

`jotnar-bench --workload u8|f64|shelf` added:

```text
u8 128x128 (16KB U8 -> F32), 100 jobs:
inproc        0.273   0.353   3452.4    82199 B/job
tcp-frames    0.410   0.560   2596.5    82199 B/job
tcp-json      1.400   2.571    660.6   380881 B/job   (4.6x bloat vs frames)

u8 256x256 (65KB U8), 50 jobs:
inproc        1.212   1.798    798.0   327960 B/job
tcp-frames    1.543   1.883    634.2   327960 B/job
tcp-json      5.784  10.574    158.9  1522897 B/job   (4.6x bloat vs frames)

f64 (I2F, n=16384, 100 jobs):
inproc        0.286   0.483   3337.2   262421 B/job
tcp-frames    0.373   0.554   2611.7   262421 B/job
tcp-json      2.241   4.215    423.3   263427 B/job

shelf (B n=16384 stored once, S scalar/job), 100 jobs:
inproc-inline 0.289   0.438   3240.9   262494 B/job
inproc-shelf  0.268   0.406   3294.4   131367 B/job
tcp-inline    0.338   0.494   2842.1   262494 B/job
tcp-shelf     0.330   0.481   2970.3   131367 B/job
```

Takeaways:

- Image blobs make JSON inflation visible (4.6× bytes vs frames at
  256×256). Float outputs are byte-identical in-process vs over the wire
  on the same engine, which is the correct scope: integer/triple are
  cross-target exact; float is exact for identical engines/orders, not
  across reduction orders. The node never claims cross-backend float
  identity.
- The blob shelf halves steady-state bytes for this workload (262K ->
  131K per job) and speeds jobs slightly, since the 131K store is paid
  once and steady-state `EXEC` references `B` by name with zero input
  bytes for it. The wins grow with jobs served and with output/input
  size ratios that favour reuse (weights, maps).

Security: `jotnar-node` defaults to loopback. Non-loopback binds are
refused without `--token` (shared-secret header `"auth"`, rejected as
`error/unauthorized` by `Node::handle` before dispatch). `jotnar-ctl`
sends it via `--token`. Threat model is a trusted LAN/cluster; there is
no TLS in v1. `PROTO_V` discipline: optional header fields (like
`"auth"`) do not bump the version; breaking header-semantics changes
do, with a migration note. Frames over `--max-frame` are rejected as
`error/too-large` and the connection is closed.

## 7. Remaining open questions

1. Line-at-a-time (3a/B) interaction and `npy`/raw payload adapters are
   still deferred until a real host workload demands them.
2. TLS/mTLS is deferred until a deployment beyond loopback + token
   appears. The token field is the seam: it would ride the same header
   without touching `Node::handle` dispatch.

## 8. How to extend this without rework

- The REPL loop is the same `Node::handle` boundary minus the registry:
  swapping stdin for TCP and swapping NDJSON payloads for framed
  `exec` requests keeps the interpreter core untouched.
- If line-at-a-time (3a/B) proves necessary, slot it in behind the same
  NDJSON loop as a different `cmd` (e.g. `{"cmd":"step",...}`), not as a
  new binary.
- The bench (Sections 5-6) answered the open transport question: frames
  won, so the frame path stays the canonical one for the determinism gate
  and the JSON arm is kept only as a benchmark/debugging aid.

---

Merged. See `showcase/runtime/` for the exhibit and `docs/USAGE.md` for the
CLI. All timings are local one-machine observations (labels:
`clock=host_wall, scope=end_to_end, kind=measured`), not cross-host gates.
