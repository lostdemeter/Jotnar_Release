# Runtime research: warm interpreter + network node

Status: research scaffolding. Local branch only (`runtime-interpreter`),
no commits to the parent checkout. The goal is twofold: (1) a small-footprint
processor that takes Jotnar listings and processes them like an interpreted
language locally, and (2) a network stack so the same processor can live on
another machine, model-blind, and do work for a smart controller.

This note records what the evidence says today, the candidate shapes for
(1) and (2), and what we are building first to let measurement arbitrate.

## 1. What exists today

| piece | where | note |
|---|---|---|
| assembler / verifier | `src/asm.rs` | IMPORT/DEF/CALL expansion, REPEAT/BREAK, `check`/`verify` |
| reference interpreter | `src/runtime.rs` (`run`, `run_text`, `repeat`) | `run_text` reassembles every call; `run(&BoundProgram, &Feeds)` is the warm reuse path |
| JSON payload codec | `src/payload.rs` | `{"streams": {...}}` with `F64/F32/I64/U8/Bool/T` |
| model-blind node | `src/protocol.rs` (`Node`, `NodeHandle`) | in-process only, no sockets by design |
| frame codec | `src/protocol.rs` (`encode_frame`/`decode_frame`) | `u32BE \|\| JSON header \|\| payload`, `PROTO_V=1` |
| CLI | `src/bin/jotnar.rs` | `check verify run emit-c build repeat ops`; one-shot: file + `--payload` → process → exit |

The frame format is already TCP-ready: `showcase/protocol/README.md` says
"Channels preserve the frame bytes and the model-blind boundary. A future
TCP transport can replace the channel without changing `Node::handle`."
The Open backlog (`docs/PORT_BACKLOG.md`, `docs/USAGE.md`) lists
"TCP transport and serve mode" as not ported yet.

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

## 5. Open questions / next measurements

1. Python-driver latency: spawn `jotnar-repl` once and feed N payloads vs
   N× `jotnar run`. Expect the warm loop to win by the process-spawn +
   reassembly margin; measure at N=100 with a mid-size listing.
2. Float vs integer paths: integer/triple are exact; float can differ in
   last ulp across reduction order. The node must not claim bit-identity
   for float across backends — docs already scope this; the transport
   benchmark must compare like-for-like.
3. Footprint: try a feature-gated minimal binary (no `image`, no `rv32`,
   no `codegen`) and record size/time deltas. If the win is <20% it is
   not worth the maintenance split.
4. Security: the framed node has no auth. Before exposing beyond
   localhost, decide: bind-address default to loopback, bearer token,
   or mTLS. Protocol `v` bump discipline if the header gains fields.
5. Versioning: `PROTO_V=1`; any new message type (e.g. `hello`) must keep
   `Node::handle` source-compatible or bump the version with a migration
   note.

## 6. How to extend this without rework

- The REPL loop is the same `Node::handle` boundary minus the registry:
  swapping stdin for TCP and swapping NDJSON payloads for framed
  `exec` requests keeps the interpreter core untouched.
- If line-at-a-time (3a/B) proves necessary, slot it in behind the same
  NDJSON loop as a different `cmd` (e.g. `{"cmd":"step",...}`), not as a
  new binary.
- If JSON transport somehow beats frames on the benchmark (Section 3b),
  the service can offer both behind a flag; the frame path stays the
  canonical one for the determinism gate.

---

Generated locally on branch `runtime-interpreter`. No push intended until
reviewed.
