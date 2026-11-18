# DOOM across two nodes

**Status:** cooperative per-frame pipeline with measured split, iterative
rebalance, live HUD viewer, failover, and recovery; GIF recorded on the live
CUDA + HIP fleet
**Artifact:** [`showcase/doom/doom_two_nodes.gif`](../showcase/doom/doom_two_nodes.gif)

## Shape

```text
node A (coordinator host)            node B (CUDA)             node C (HIP)
  RV32IM native core                   Jotnar INTERP 2x          Jotnar INTERP 2x
  framebuffer 320x200 BGRA   ->        left window        ->     right window
        |                                   |                         |
        +-------- frame windows over TCP ---+                         |
        |                                                             |
        +<--------------- RGB results -------------------------------+
        |
  crop/stitch -> 640x400 RGB -> raw frames + latency metadata -> GIF
```

Node A runs the responsive native RV32IM core. Nodes B and C each run the same
Jotnar listing on one overlapping window of the frame. The coordinator
calibrates both cards and chooses the window widths from measured throughput;
on this fleet CUDA gets about 32% of the columns and HIP about 68%. The
overlap keeps the bilinear upscale continuous at the split.

## Why two nodes

This is the first test where the nodes cooperate on **one frame**, not just
hand a checkpoint from one to the other:

- the game state stays on node A;
- each GPU node does real Jotnar work every frame;
- the coordinator owns placement, frame splitting, stitching, and the
  latency receipt;
- the executor contract is unchanged: `submit`, `job-status`, `job-result`.

It is a streaming coordinator above the same executor contract, not a new
model or stage format.

## Listing

[`demos/doom-two-nodes/upscale-half.asm`](../demos/doom-two-nodes/upscale-half.asm):

```text
CONFIG warm_frame 1
IN X
OUT = INTERP(X, 2.0, 2.0)
```

`CONFIG warm_frame 1` keeps the GPU context, compiled artifact, and buffers
resident. The per-frame cost is the Jotnar op plus copies, not process
startup. The runner reports the F32 output descriptor and binary body; the
coordinator validates it before use.

## Commands

```sh
cargo build --release -p jotnar-node-tools --bin jotnar-doom-coord

./target/release/jotnar-doom-coord \
  --image "$HOME/Documents/OpenCode/jotnar-main/programs/doom/build/doom.bin" \
  --wad "$HOME/Documents/OpenCode/jotnar-assets/doom1.wad" \
  --cuda 192.168.1.116:9997 \
  --hip 192.168.1.195:9997 \
  --token test \
  --start-frame 200 --frames 60 --fps 15 \
  --out /tmp/opencode/doom_two_nodes
```

`--no-auto-split` forces the even 160/160 split for comparison. The
reproducible gate also annotates the frames and writes the GIF:

```sh
python3 scripts/doom_two_gpu_gate.py
```

## Interactive viewer

The quick launcher is:

```sh
./demos/doom-two-nodes/run.sh
```

The launcher uses `--no-record` by default so a long play session does not fill
the disk. Pass `--record` to write raw frames and `frames.jsonl`.

Start the coordinator with a viewer listener and open the SDL window on this
machine:

```sh
./target/release/jotnar-doom-coord \
  --image "$HOME/Documents/OpenCode/jotnar-main/programs/doom/build/doom.bin" \
  --wad "$HOME/Documents/OpenCode/jotnar-assets/doom1.wad" \
  --cuda 192.168.1.116:9997 --hip 192.168.1.195:9997 \
  --token test --start-frame 200 --frames 10000 --fps 20 \
  --viewer-listen 127.0.0.1:10000 \
  --out /tmp/opencode/doom_live

./target/release/jotnar-doom-view --addr 127.0.0.1:10000
```

The viewer loads SDL2 through its C ABI. It displays the stitched frame in a
960x699 window, with a top status band and a bottom latency band:

```text
JOTNAR DOOM  MODE SPLIT  FRAME 235  TOTAL 46MS/66MS
A RV32IM  B CUDA 87C 27MS  C HIP 253C 24MS
B 192.168.1.116:9997   C 192.168.1.195:9997

P50 45MS  P95 47MS  |  ALL NODES HEALTHY
```

The HUD is drawn by the coordinator into the viewer frame, so the live display
shows which node is running which part, the current split, per-node latency,
the frame budget, rolling p50/p95, and failover/recovery state. The raw frames
written for the GIF stay clean.

The viewer forwards key events to the coordinator over TCP. Arrows move, ctrl
fires, space uses, shift runs, esc opens the menu, and q quits.

The coordinator runs the RV32IM core and the GPU pipeline; the viewer only
displays and forwards keys. The startup script still advances to gameplay, so
the viewer can connect before or during the run.

![two-node viewer](../showcase/doom/doom_two_nodes_view.png)

After a failure and recovery the bottom band shows the transition:

![two-node viewer recovered](../showcase/doom/doom_two_nodes_recovered.png)

## Failover and recovery

Both nodes pre-warm a full-frame fallback shape at startup. If a node fails,
the coordinator marks it down, records the error, and retries that frame on
the surviving node with the full 320-column input. Later frames stay in
single-node mode.

Every `--recover-every` frames (default 20), the coordinator pings a failed
node. If it answers, the coordinator warms the split shapes again and returns
to split mode. The recovery is recorded separately from the failure.

The reproducible outage gate uses local TCP proxies in front of the real GPU
runners, closes one proxy mid-run, and optionally restarts it:

```sh
python3 scripts/doom_failover_gate.py
```

Checked results:

```text
CUDA outage  -> hip-full   p50 47 ms, p95 59 ms, 45/45 frames, exit 0
HIP outage   -> cuda-full  p50 119 ms, p95 126 ms, 45/45 frames, exit 0
CUDA recover -> split      p50 43 ms, p95 54 ms, 45/45 frames, 1 recovery
HIP recover  -> split      p50 49 ms, p95 120 ms, 45/45 frames, 1 recovery
```

The HIP outage is slower because CUDA takes the full frame, but it still
completes. Recovery returns to the measured split without restarting the
RV32IM core.

## Receipt

The checked adaptive run:

```text
node A  RV32IM native        core p50 33 ms
node B  CUDA 192.168.1.116   p50 29 ms
node C  HIP  192.168.1.195   p50 26 ms
total                        p50 42 ms, p95 44 ms
frame budget at 15 fps       66 ms
real time at target          yes
split                        72 CUDA / 248 HIP columns (23% / 77%)
frames                       60
warmup                       CUDA 6644 ms, HIP 2316 ms (first run)
```

The calibration probes the even split three times and uses the median:

```text
CUDA probe p50  41 ms
HIP  probe p50  18 ms
chosen split    98 / 222
rebalance       72 / 248
```

The GIF is 640x454 (640x400 frame plus a HUD), 60 frames, 15 fps. The HUD
shows the frame number, the measured CUDA/HIP split, per-card latency, and the
frame budget.

The coordinator writes:

```text
run.json        hashes, node addresses, split, calibration, latency p50/p95/max
frames.jsonl    one record per frame: input keys, per-card job ids/timings/hashes
frames/*.rgb    raw 640x400 RGB frames
poster.png      annotated first frame
contact.png     six-frame contact sheet
```

The checked evidence is copied to
`/home/balrog/Documents/OpenCode/jotnaros/artifacts/doom-two-nodes/`:

```text
run.json, frames.jsonl, hardware.json, poster.png, contact.png
viewer.png, viewer-key-test.jsonl
viewer-failover.png
failover/summary.json, failover/cuda-outage.log, failover/hip-outage.log
```

## Limits

- The GPU op is a 2x bilinear upscale, not a full distributed renderer. The
  RV32IM renderer still runs on node A.
- The split is chosen at startup and rebalanced once from a measured probe. It
  does not continuously rebalance during play.
- Recovery probes a failed node every `--recover-every` frames; it does not
  migrate state between nodes.
- The full-frame fallback is pre-warmed, so the surviving node does not pay
  artifact build in the failover frame.
- The viewer protocol is a simple length-prefixed frame stream and JSON key
  messages; it has no authentication beyond the trusted-LAN assumption.
- The coordinator is DOOM-specific. A general streaming stage ABI is future
  work; the batch bundle/stage layer is separate.
- The first frame pays artifact build and warm-worker startup. The receipt
  separates warmup from steady-state latency.
- The 64 MB RV32IM state is not moved between nodes in this demo; only frame
  pixels move. Checkpoint migration is the separate two-node gate.
- The trusted-LAN token is not TLS.
