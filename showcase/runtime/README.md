# Interpreter + model-blind node (warm execution over TCP)

> **Status: language-repo exhibit 10.** The deterministic job-server section
> below is first-class research/tooling, not a production serve claim. The
> language repository certifies the hash-identical kernels, receipts, build
> provenance, and small-output transport; application performance and Stage E
> live-service behavior are not certified here. See
> [`CLAIM_BOUNDARY.md`](CLAIM_BOUNDARY.md).

The warm interpreter and the network node are one processor with two
front doors:

```text
host (smart: owns workload + placement)
  |-- stdin NDJSON --> jotnar-repl  (local, Python-like)
  |-- TCP frames -----> jotnar-node (model-blind worker)
                            |
                            v
                    Node::handle (listing + blobs + streams)
```

Both assemble once and execute many payloads. Neither compiles
anything at runtime. The node never sees "Futhark", "matmul", or a
model name — it sees a listing to assemble, named blobs, and typed
stream bytes.

## Run

```sh
# Local interpreter: assemble saxpy once, then feed payloads on stdin
cargo run --release --bin jotnar-repl -- showcase/language/saxpy.asm --outputs OUT

# Network node: ping, upload, execute
cargo run --release --bin jotnar-node -- --bind 127.0.0.1:9999 &
cargo run --release --bin jotnar-ctl -- --addr 127.0.0.1:9999 ping
cargo run --release --bin jotnar-ctl -- --addr 127.0.0.1:9999 program tile showcase/language/matmul2.asm
cargo run --release --bin jotnar-ctl -- --addr 127.0.0.1:9999 exec tile --payload payload.json --outputs c00

# Transport benchmark: in-process vs TCP frames vs TCP JSON
cargo run --release --bin jotnar-bench -- --jobs 2000
cargo run --release --bin jotnar-bench -- --workload vec --n 16384 --jobs 100
cargo run --release --bin jotnar-bench -- --workload u8  --side 128 --jobs 100
cargo run --release --bin jotnar-bench -- --workload shelf --n 16384 --jobs 100
```

## What the numbers say

Labels on every bench line: `clock=host_wall, scope=end_to_end,
kind=measured, machine=local` (one machine, release build).

Warm vs cold (preliminary single-run observation, not a gate: matmul2,
50 payloads, cold `jotnar run` 45.8 ms total vs warm `jotnar-repl`
1.7 ms total, ~26×; no checked-in reproducible command yet).

Transport head-to-head (same `Node::handle`, 2000 matmul2 jobs):

```text
mode        median  jobs/s   bytes/job
inproc        0.029   33017      995
tcp-frames    0.039   24140      995
tcp-json      0.061   16182      728
```

Larger payloads (frames vs JSON):

```text
vec  n=16384 : tcp-frames 0.495ms / tcp-json 4.39ms / inproc 0.186ms
vec  n=65536 : tcp-frames 1.95ms  / tcp-json 17.1ms  / inproc 0.85ms
u8   256x256 : tcp-frames 1.54ms  / tcp-json 5.78ms  (4.6x bloat vs frames)
shelf (131K blob stored once): 131K B/job vs 262K inline
```

Framed TCP costs ~10 µs over in-process and stays ~9× faster than
JSON at 64K elements; image blobs inflate JSON to 4.6× the frame
bytes. The blob shelf pays the store once and references it by name
thereafter.

## Deterministic job server (research)

The node boundary now carries the deterministic threading work. This is
language/runtime research, not a production serve claim:

- **Per-job execution budget.** `ExecOptions::default()` is one thread (the
  permanent oracle). Auto-threading is explicit at the boundary:
  `ThreadPolicy::{Reference, DeterministicAuto}`, `jotnar run/repl --threads`,
  `jotnar-ctl exec --policy`, or the node's `thread_policy` header.
  `CONFIG threads` remains a compiled-target directive and is never reused as
  the job budget.
- **Output-partitioned T2 kernels.** MATMUL rows, CONV2D/DECONV2D flattened
  outputs, and SOFTMAX rows are split into stable contiguous blocks. Each
  output keeps the serial exponent scan and ascending accumulation order, so
  1/4/16 threads, repeats, and concurrent jobs are hash-identical.
- **Exact harness.** `jotnar-det` replays NDJSON payloads across thread
  budgets, repeats, and staggered concurrent jobs, hashing the requested
  streams and failing loudly on mismatch. `tests/determinism.rs` is the
  short CI gate.
- **Receipts and provenance.** `exec-ok` carries a canonical receipt:
  program/blob/input/output hashes, determinism profile, backend/kernel,
  runtime version (`<crate>+<short-git>`), scheduler version, resolved
  threads, partition descriptor, and verdict. Per-op timing is advisory and
  never enters the receipt.
- **Transport and queue.** T2 streams use the compact `T/t2` wire dtype
  (`s:i8, e:i32, z:u8, m:i32` = 10 B/element); frozen T keeps the
  interleaved 24-byte `i64x3` form. `NodeHandle` has a bounded request queue
  (default 64). `jotnar-node --bind 127.0.0.1:0` reports its resolved
  ephemeral address.

Reproduce the language-side claims without application payloads:

```sh
cargo test --release --test determinism      # 1..16 threads, repeats, concurrent jobs
cargo test --release --test protocol         # T2 transport, receipts, ping/build identity
cargo run --release --bin jotnar-det -- <listing.asm> <payload.ndjson> \
  --outputs OUT --threads 1,4,16 --repeat 2
```

Application-level DAV2 layer-0/backbone/neck/head measurements are
companion-repository evidence and are not reproduced from this repository.
See [`CLAIM_BOUNDARY.md`](CLAIM_BOUNDARY.md) for the certified/research
split and the Stage E blockers.

## Proves

Local warm execution and remote model-blind execution are the same
deterministic boundary: same program + same input bytes = same output
bytes, with compile cost removed from the per-command path. The T2 kernels
certified here are hash-identical across the tested thread counts, repeats,
and in-process concurrent jobs, and the receipt names the build and
partition.

## Does not prove

Cross-backend float bit-identity (integer/triple are cross-target
exact; float is exact on identical engines/orders), TLS beyond the
loopback + `--token` trusted-LAN model, or host-side workload
placement (that lives in the controller, by design). It does not prove
large-output TCP parity, concurrent TCP, cancellation/timeouts, a memory
quota, or live service.

## Boundaries (explicit)

- Loopback by default (`--bind 127.0.0.1:9999` needs no token).
  Non-loopback binds are refused without `--token`; the token is a
  cleartext shared secret in the JSON header, a trusted-LAN
  convenience, not a cryptographic authenticator. No TLS.
- One client at a time; fan-out lives in the controller.
- No memory quota: programs/stores persist across clients until
  `drop`. Oversize frames (`--max-frame`, 64 MiB default) are refused
  as `error/too-large` and the connection closes.
- The JSON arm is a benchmark/debugging aid, not a supported wire.
- Numbers above are one-machine observations, not cross-host gates.
- **Stage E research, not claimed:** concurrent TCP jobs,
  cancellation/timeouts, `Arc` snapshots for program/store, a per-job memory
  policy, and chunked/segmented or store-backed results for outputs larger
  than one frame. A 64 MiB frame carries ~6.7M T2 elements; application
  neck/head outputs are much larger, so TCP parity is not claimed for them.
- `NodeHandle` has a bounded queue but one worker; a long `exec` can still
  delay `program`/`store` until snapshots land.
- FNV-1a 64-bit is a fast fingerprint, not a cryptographic content address.
  Do not use these hashes for immutable dedup or security without a stronger
  hash.

## Footprint (one-off observation, reproducible)

```sh
cargo build --release --bin jotnar-repl --bin jotnar-node --bin jotnar-ctl
ls -lh target/release/jotnar-repl target/release/jotnar-node target/release/jotnar-ctl
nm target/release/jotnar-repl | grep -c png
strings target/release/jotnar-repl | grep -E 'DOOM|Verilog' || echo "no DOOM/Verilog strings"
```

One machine observed `jotnar-repl`/`jotnar-node` at 4.2M (3.3M stripped)
and `jotnar-ctl` at 804K, with zero `png` symbols and no DOOM/Verilog
strings (dead-code elimination removes rv32/showcase/codegen/image
from these builds).
