# Streaming layer

**Status:** C0, D0, C1, and D1 implemented and gated; C2 (grid derivation) is the
next slice
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
FrameSource   frames-dir | camera | RV32IM (DOOM)
FrameSink     frames-dir | frames-hud | tcp (existing SDL viewer) | null
```

The camera adapter spawns `ffmpeg` at MJPEG 1280x720 30 fps and reads raw
RGB24 at the stream size. The base frame is downscaled once; the stream clock
then paces at 15 fps.

## Frame metadata and the HUD hook

`FrameSink` has a second write path:

```rust
fn write_meta(&mut self, frame: &Frame, meta: &FrameMeta) -> Result<(), String> {
    self.write(frame) // default: metadata is optional
}
```

`FrameMeta` is controller reporting, not planning. It carries the frame index,
`work_ms`, deadline, running p50/p95/p99, the active split (left/right/share),
cumulative failovers and recoveries, the failed/recovered targets, a human note,
and one entry per tile: row/col, planned target, actual target and address,
owned columns, tile latency, worker time when the runner reports it, submit/
status/result timings, attempts, failover origin, and job id.

The default delegates to `write`, so frames-dir and null sinks are byte-for-byte
unchanged. The receipt line adds the same fields without removing the existing
ones. A sink that wants a HUD overrides `write_meta`:

- `jotnar-doom-stream --sink tcp` renders the DOOM HUD (`hud::render`) and sends
  the 640x466 frame over the existing viewer protocol.
- `--sink frames --hud` writes the same composited HUD frames headlessly; the
  live gate uses it to prove the bands are drawn and the game region is
  unchanged without requiring SDL.

The HUD is rendering only. Tile geometry, split selection, and recording stay in
the controller; the executor still sees one listing and one payload per job.

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
`min..=max` and the halo is at least the claimed overlap.

If the descriptor omits `tiles`, the controller derives the grid (C2): the
number of tiles is the placement slot count, `rows*cols` must divide the source,
both tile dimensions must fit the claim, and the tile aspect ratio closest to
the source wins (ties: wider tiles). `halo` is the largest claimed overlap. If
no legal factorization exists, the controller fails with the slots, source, and
claim constraints rather than inventing a shape. `summary.json` records
`grid.derived`, the slots, the claim, the derived tiles, and the reason. A
declared `tiles` block still takes the validation path; the wall gate runs the
derived path and keeps a `--declared-grid` control.

The camera and frames wall descriptors omit `tiles` and reference the checked-in
`demos/video-wall/wall-stage.json` claim. The derived grid is 2x2 tiles of
320x180 with halo 1, identical to the previous declared grid, so the gate stays
byte-identical.

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
planned target is retried on the next frame, so recovery is automatic. The
controller tracks the failed-target set between measured frames and records a
`recovered_targets` entry (and a cumulative `recoveries` count) when a target
that failed last frame succeeds again.

The wall gate includes a failover leg: the HIP endpoint is pointed at a closed
port (`127.0.0.1:1`), the three HIP tiles must retry on CUDA, and the two
composited frames must still match the reference. The live DOOM gate adds a
recovery leg: HIP is refused for the first measured frames, then a local proxy
brings it up on the same port; the later frames must return to `attempts == 1`
with a recorded recovery.

A blackholed target (accepts but never responds) is bounded by the controller's
per-request deadline, not the 60 s client default: interactive tiles use
`--request-timeout-ms` (default 1000; the blackhole gate uses 500) for submit,
status, and result. After a real failed attempt the controller quarantines that
target for `--quarantine-frames` (default 4), uses a persistent fallback
session, and retries after the window. Fallback tile shapes are warmed on the
other target before the measured loop (`--warm-fallbacks`, on by default for
`jotnar-doom-stream`), so the first failover does not pay a multi-second
artifact compile. `scripts/doom_blackhole_gate.py` proves detection, exactness,
quarantine, and rejoin.

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
live         p50 39, p95 47, p99 49, max 52 ms
             budget 66, frames over budget 0
matches      100/100 byte-identical to the coordinator (EVEN reference)
HUD          640x466, game region identical
calibration  declared 72/248; chosen 113/207 after one rebalance
failover     HIP dead; right tile retried on CUDA; 2/2 exact
stage claim  79905e7d… rows/columns overlap 1
```

This is with persistent transport and the CPU performance setting. The earlier
powersave run showed the same GPU work plus ~30 ms status-wake stalls; the host
setting removes them.

The SLO gate uses the frames sink. The TCP viewer sink can add a further tail
when the SDL reader falls behind (a 30-frame viewer run showed two ~250 ms
frames); that is sink backpressure. The viewer now renders the HUD bands from
the controller metadata; `--sink frames --hud` is the headless equivalent.

### Split calibration (D1a)

A column-split descriptor can carry a calibration block:

```json
"calibration": {"probe_iters": 7, "min_cols": 32, "max_cols": 288, "rebalance": true}
```

`stream::SplitPolicy` runs on the first real source frame, before the measured
loop: warm even, probe even `N` times (p50 per side), choose a split, warm it,
then one rebalance pass (normalize to half-width, accept a change of at least
four columns). It writes `summary.calibration` (`policy`, `declared`, `probe`,
`chosen`, `split_warmup`, `rebalance`) and every measured receipt line carries
the active split. A probe failure aborts calibration and keeps the declared
split, so per-frame failover still runs when one target is dead.
`jotnar-doom-stream` enables calibration by default; `--no-calibrate` pins the
declared `--left-cols`.

The controller never sends the policy to an executor: it only submits the probe
and frame tiles it already owns. On this fleet the chosen split is about 94/226
for a declared 72/248. The binary/U8 stage's kernel time is sub-millisecond, so
the probe balances the observable end-to-end tile latency (transport plus host
wake), not pure kernel time. The RV32 core dominates the DOOM frame, so the
split is a weak lever here; the point of the gate is that the decision is made
on the controller and recorded, not that it moves the SLO.

The legacy coordinator is now a comparison tool: `doom_live_gate.py` still uses
it to produce the deterministic EVEN reference, but the demo and the receipt
path are `jotnar-doom-stream`.

## Receipt

`jotnar-stream` writes `frames.jsonl` (one line per frame: work time, deadline
verdict, mode, active split, per-target placement, running p50/p95/p99, failover
and recovery counts, failed/recovered targets, a note, wall hash/bytes, optional
reference match, and per-tile row/col, target, executor, job id, owned columns,
timestamps, hash, attempts, failover origin, and bytes) and `summary.json`
(descriptor and stage-descriptor hashes, listing hash, source/sink, clock,
geometry, the calibration report if enabled, percentiles, failover/recovery
counts, exactness counts, leases, and the controller/source/state owner). A
stream registers an interactive lease per target when `--fleet-dir` is
supplied.

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
distributed           p50 29, p95 31, p99 31, max 31 ms
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
