# Streaming layer

**Status:** C0 and D0 implemented; DOOM port (C1) and live productization (D1)
are the next slices
**Scope:** a general streaming controller above the batch stage/executor layer

`jotnar-workload` runs typed stage graphs to completion. DOOM originally had a
separate, DOOM-specific coordinator for per-frame work. This layer extracts the
streaming path: a stream has a clock, a source adapter, a tiled GPU stage, a
fan-in/composite, a sink adapter, and a per-frame receipt. The executor contract
does not change; the runner still runs one listing and one payload per job.

```text
jotnar-noded          model-blind listing execution
jotnar-gpu-runner     target-aware GPU execution
jotnar-native-runner  model-blind process execution
jotnar-workload       batch planning, placement, leases, ledger
jotnar-stream         streaming graph, clock, source/sink, receipts
jotnar-doom-coord     DOOM source/sink adapter (port in progress)
```

## Descriptor

`jotnar.stream/1` is a controller artifact, not an executor instruction:

```json
{
  "schema": "jotnar.stream/1",
  "name": "camera-wall",
  "clock": {"fps": 15, "deadline_ms": 66, "max_in_flight": 1},
  "source": {"kind": "camera", "device": "/dev/video0",
             "width": 640, "height": 360, "channels": 3, "format": "rgb24",
             "capture_width": 1280, "capture_height": 720, "capture_fps": 30},
  "stage": {"listing": "demos/doom-two-nodes/upscale-half.asm",
            "targets": ["cuda", "hip"]},
  "tiles": {"rows": 2, "cols": 2, "halo": 1, "bezel": 0},
  "placement": {"cuda": 1, "hip": 3},
  "sink": {"kind": "tcp", "listen": "127.0.0.1:9990"},
  "merge": "tile-grid",
  "backpressure": "drop-oldest",
  "determinism": "exact",
  "receipt": {"kind": "every-frame", "dir": "/tmp/opencode/wall-receipt"}
}
```

`placement` is measured policy, not a constant: on this fleet one CUDA tile is
slower than three HIP tiles for the wall shape, so the wall uses 1:3. An empty
placement means equal round-robin.

## Frame envelope and adapters

A frame is RGB24 plus `index`, `ts_ms`, kind/shape/bytes, and a SHA-256. Source
and sink are adapters:

```text
FrameSource   frames-dir | camera | (RV32IM for DOOM, pending)
FrameSink     frames-dir | tcp (existing SDL viewer) | null
```

The camera adapter spawns `ffmpeg` at MJPEG 1280x720 30 fps and reads raw
RGB24 at the stream size. The base frame is downscaled once; the stream clock
then paces at 15 fps.

## Tiles, halo, and composite

The wall takes a 640x360 source and splits it into four 320x180 quadrants. Each
tile carries a one-pixel halo, is upscaled 2x (`INTERP`), and the 2x halo is
cropped before compositing into 1280x720. The halo keeps interior tile edges
continuous. Source edges are edge-replicated.

The same tiled plan runs on one CUDA node as the reference. The distributed run
places one tile on CUDA and three on HIP. Because both runs use the same tiler,
the comparison proves placement/vendor agreement, not agreement between
different resampling algorithms.

## Transport

The controller sends binary F32 payloads (`payload_format: "binary"`) and asks
the runner for U8 outputs (`output_u8: true`). The tile input is ~0.7 MB and the
U8 tile output is ~0.7 MB; the four-tile wall is ~2.8 MB/frame at 15 fps. The
JSON path would expand the input by roughly 10x and does not fit the wall's
frame budget. See [`EXECUTOR_CONTRACT.md`](EXECUTOR_CONTRACT.md).

## Receipt

`jotnar-stream` writes `frames.jsonl` (one line per frame: work time, deadline
verdict, wall hash/bytes, optional reference match, and per-tile row/col,
target, executor, job id, timestamps, hash, and bytes) and `summary.json`
(descriptor hash, listing hash, source/sink, clock, geometry, percentiles,
exactness counts, leases). A stream registers/interactive lease per target when
`--fleet-dir` is supplied.

## Wall measurements

Gate command:

```sh
python3 scripts/video_wall_gate.py --no-build --frames 30 \
  --out /tmp/opencode/wall-gate
```

Result on the live CUDA/HIP fleet (640x360 source, 1280x720 wall, 15 fps,
1 CUDA + 3 HIP tiles):

```text
reference (all CUDA)  p50 65, p95 74, p99 75, max 75 ms (correctness only)
distributed           p50 50, p95 65, p99 66, max 66 ms
                      budget 66, frames over budget 0
exactness             30/30 wall frames byte-identical
tiles                 4 per frame, both targets, job ids recorded
```

The reference is slower because all four tiles serialize on one GPU; it is a
correctness reference, not the SLO target.

## Commands

```sh
cd /home/balrog/Documents/OpenCode/jotnar-hip-release
cargo build --release -p jotnar-node-tools --bin jotnar-stream

# live camera + viewer (one command)
./target/release/jotnar-stream --descriptor demos/video-wall/camera-wall.json \
  --cuda 192.168.1.116:9997 --hip 192.168.1.195:9997 --token test --viewer

# or two terminals: stream first, then the viewer
./target/release/jotnar-doom-view --addr 127.0.0.1:9990

# recorded clip
./target/release/jotnar-stream --descriptor demos/video-wall/frames-wall.json \
  --cuda 192.168.1.116:9997 --hip 192.168.1.195:9997 --token test \
  --clip /tmp/opencode/wall-clip --frames 120
```

## Limits

- The tiler is currently a uniform tile grid; the compiler's
  `partitionable` claim is not consumed yet. 2D partitionability is the next
  C/debt slice.
- No failover/recovery in the stream controller yet; a failed tile aborts the
  run. DOOM still owns its failover path until C1.
- No runner-side priority: batch work defers while the wall holds the
  interactive lease. A concurrent export needs the runner priority debt.
- The source/compositor run on the controller host; the GPU lease does not
  cover that CPU/state owner. The receipt records the controller owner.
- The camera is local to the controller host (`/dev/video0`, MJPEG 720p30).
  The frames-dir source is the deterministic gate path.
- The SDL viewer is the existing `jotnar-doom-view`; the stream speaks its frame
  protocol. A wall-specific viewer/HUD is productization work.
