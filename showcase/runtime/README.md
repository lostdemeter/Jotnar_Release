# Interpreter + model-blind node (warm execution over TCP)

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
cargo run --release --bin jotnar-bench -- --workload u8  --n 128   --jobs 100
cargo run --release --bin jotnar-bench -- --workload shelf --n 16384 --jobs 100
```

## What the numbers say

Warm vs cold, one listing, many payloads (matmul2, 50 payloads):
cold `jotnar run` 45.8 ms total vs warm `jotnar-repl` 1.7 ms total
(~26×, 0.92 ms vs 0.03 ms per payload).

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

## Proves

Local warm execution and remote model-blind execution are the same
deterministic boundary: same program + same input bytes = same output
bytes, with compile cost removed from the per-command path.

## Does not prove

Cross-backend float bit-identity (integer/triple are cross-target
exact; float is exact on identical engines/orders), TLS beyond the
loopback + `--token` trusted-LAN model, or host-side workload
placement (that lives in the controller, by design).
