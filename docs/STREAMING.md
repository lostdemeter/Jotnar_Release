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

The compiler now emits a partition claim for `INTERP` and elementwise stages:

```json
"partitionable": {
  "axes": [
    {"axis": "rows", "index": 0, "min": 16, "max": 360, "overlap": 1},
    {"axis": "columns", "index": 1, "min": 16, "max": 640, "overlap": 1}
  ]
}
```

The stream controller loads the referenced `jotnar.stage/1` document, checks the
listing hash and target subset, and validates that the tile size is within
`min..=max` and the halo is at least the claimed overlap. It does not yet
derive the grid from the claim; the 2x2 grid is still a descriptor decision.

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

## Persistent transport

`JobClient` keeps one TCP session and the GPU runner's `handle` serves requests
in a loop until EOF. The stream controller creates one executor per tile slot
and reuses it across frames; submit, status polling, and result fetch share the
same connection. A status/result call retries once if a session was reset; a
submit retries once on `job-unavailable`/`job-invalid` (a duplicate job whose
ack was lost is possible, but the first result is ignored, so correctness is
unchanged).

This removed the ~210 ms new-connection submit stalls. The remaining wake
latency is on the runner host: under the CPU `powersave` governor and deep idle
states, a persistent connection thread could take ~30 ms to wake for a status
poll even though the GPU job is 1-12 ms. The CUDA host now runs the performance
governor with C1E+ disabled; the final live DOOM 100-frame gate is p99 61 /
max 63, and the wall is p99 29.

## Failover

Each tile is attempted on its planned target and then on the other targets in
order. The receipt records `attempts`, `failover_from`, and the errors for every
tile. A dead target therefore costs a failed connect, not a dropped frame; the
planned target is retried on the next frame, so recovery is automatic.

The wall gate includes a failover leg: the HIP endpoint is pointed at a closed
port (`127.0.0.1:1`), the three HIP tiles must retry on CUDA, and the two
composited frames must still match the reference. A blackholed target (accepts
but never responds) still depends on the client timeout; tuning that is future
work.

## DOOM column split (C1a)

The generic controller also runs the DOOM shape: a 320x200 source frame, a
one-column overlap, two column tiles, and a stitch back to 640x400. The
descriptor uses:

```json
"tiles": {"rows": 1, "cols": 2, "halo": 1, "bezel": 0},
"split": {"left_cols": 72, "right_cols": 248, "overlap": 1},
"placement": {"cuda": 1, "hip": 1},
"merge": "columns"
```

The coordinator was extended with `--record-source DIR` so its RV32IM
framebuffer (320x200 RGB) and stitched output (640x400 RGB) can be recorded.
`scripts/doom_stream_gate.py` runs the coordinator with `--no-auto-split`
(EVEN) and then replays the recorded source through the generic column-split
path at 72/248. The outputs are byte-identical, which also proves the stitch is
independent of the split point.

```text
coordinator    EVEN split, 20 frames, p50 74 / p99 82 ms total
generic path   72/248, 20 frames, GPU p50 15 / p99 16 ms
matches        20/20 byte-identical
failover       HIP dead; right tile retried on CUDA; 2/2 exact
projected full core_p50 36 + GPU p99 16 = 52 ms (provisional)
```

The stream path here consumes recorded source frames, so its latency is the GPU
path only. C1b is the live RV32 source adapter; until then the coordinator
remains the source/sink process and the 52 ms figure is a projection, not a
measured full-frame SLO.

## Live DOOM (C1b)

`jotnar-doom-stream` makes the RV32IM core a live `FrameSource` and the DOOM
viewer a live `FrameSink`. The generic controller still owns the clock, column
split, GPU placement, stitch, failover, leases, and receipt. The viewer keys
are forwarded through a shared queue; the scripted `start_keys`/`game_keys`
schedule is preserved.

```sh
./target/release/jotnar-doom-stream \
  --image "$HOME/Documents/OpenCode/jotnar-main/programs/doom/build/doom.bin" \
  --wad "$HOME/Documents/OpenCode/jotnar-assets/doom1.wad" \
  --listing demos/doom-two-nodes/upscale-half.asm \
  --cuda 192.168.1.116:9997 --hip 192.168.1.195:9997 --token test \
  --sink tcp --viewer
```

Omit `--frames` for an unlimited live run; Ctrl-C stops it. The SDL window is
the existing `jotnar-doom-view`; keys are forwarded back to the core.

Full-frame gate:

```sh
python3 scripts/doom_live_gate.py --no-build --frames 100 \
  --out /tmp/opencode/doom-live-gate
```

The live `work_ms` includes the core and the GPU path. 72/248 split, 15 fps,
100 measured frames after 3 warm-up frames:

```text
live         p50 47, p95 56, p99 61, max 63 ms
             budget 66, frames over budget 0
matches      100/100 byte-identical to the coordinator (EVEN reference)
failover     HIP dead; right tile retried on CUDA; 2/2 exact
             p50 64, p99 65 ms (survivor carries both halves)
stage claim  79905e7d… rows/columns overlap 1
```

This is with persistent transport and the CPU performance setting. The earlier
powersave run showed the same GPU work plus ~30 ms status-wake stalls; the host
setting removes them.

The SLO gate uses the frames sink. The TCP viewer sink can add a further tail
when the SDL reader falls behind (a 30-frame viewer run showed two ~250 ms
frames); that is sink backpressure. The HUD and the coordinator's
calibration/rebalance policy are not ported; the viewer title currently shows
mode/frame only.

## Receipt

`jotnar-stream` writes `frames.jsonl` (one line per frame: work time, deadline
verdict, wall hash/bytes, optional reference match, and per-tile row/col,
target, executor, job id, timestamps, hash, attempts, failover origin, and
bytes) and `summary.json` (descriptor and stage-descriptor hashes, listing hash,
source/sink, clock, geometry, percentiles, failover count, exactness counts,
leases, and the controller/source/state owner). A stream registers an
interactive lease per target when `--fleet-dir` is supplied.

## Wall measurements

Gate command:

```sh
python3 scripts/video_wall_gate.py --no-build --frames 30 \
  --out /tmp/opencode/wall-gate
```

Result on the live CUDA/HIP fleet (640x360 source, 1280x720 wall, 15 fps,
1 CUDA + 3 HIP tiles):

```text
reference (all CUDA)  p50 61, p95 65, p99 65, max 65 ms (correctness only)
distributed           p50 27, p95 28, p99 29, max 29 ms
                      budget 66, frames over budget 0
exactness             30/30 wall frames byte-identical
stage descriptor      ab0715db…  partitionable rows/columns overlap 1
failover leg          HIP dead, 6 tiles retried on CUDA, 2/2 exact
                      p50 58, p99 60, max 60 ms
```

Three frames are processed through the full tile path as warm-up
(`--warmup-frames 3`) and excluded from the sink, receipt, and SLO. The
reference is slower because all four tiles serialize on one GPU; it is a
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

# DOOM split/stitch equivalence (C1a)
python3 scripts/doom_stream_gate.py --no-build --frames 20 \
  --out /tmp/opencode/doom-stream-gate

# live DOOM full-frame gate (C1b)
python3 scripts/doom_live_gate.py --no-build --frames 20 \
  --out /tmp/opencode/doom-live-gate
```

## Limits

- The tiler is a uniform tile grid. The compiler's `partitionable` claim is
  loaded and validated (size range and overlap), but the grid is still a
  descriptor decision rather than derived from the claim. The gate's stage
  claim is compiled for CUDA; HIP tiles are covered by exactness, not by a
  second compiled descriptor.
- Failover retries a failed tile on the other targets and retries the planned
  target on the next frame, so recovery is automatic. A blackholed target that
  accepts and never responds still depends on the client timeout; the gate uses
  a closed port. A mid-run proxy/blackhole gate is future work.
- C1a proves the column split/stitch path byte-identical to DOOM. C1b runs the
  RV32IM source live through the same controller with a full-frame SLO. The
  legacy coordinator still owns the HUD and calibration/rebalance policy.
- The live source duplicates the coordinator's scripted key schedule; move it
  to a shared DOOM adapter module before changing either.
- The TCP viewer sink has no drop/backpressure policy; a slow SDL reader can
  add a frame tail. The SLO gate uses the frames sink.
- No runner-side priority: batch work defers while the wall holds the
  interactive lease. A concurrent export needs the runner priority debt.
- The source/compositor run on the controller host; the GPU lease does not
  cover that CPU/state owner. The receipt records the controller/source/state
  owner, but a full CPU lease is future work.
- The camera is local to the controller host (`/dev/video0`, MJPEG 720p30).
  The frames-dir source is the deterministic gate path.
- The SDL viewer is the existing `jotnar-doom-view`; the stream speaks its frame
  protocol. A wall-specific viewer/HUD is productization work.
- Three warm-up frames are processed before the measured run; they are excluded
  from the sink, receipt, and SLO. A long-running live stream has no such
  boundary, so its first frame still pays worker startup.
