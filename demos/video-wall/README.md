# Video wall

The first streaming vertical: a 640x360 source is split into four 320x180
quadrants with a one-pixel halo, each quadrant is upscaled 2x on the GPU fleet,
and the four 640x360 results are composited into a 1280x720 wall at 15 fps.

```text
source 640x360 RGB24
  -> 2x2 tiles, 1px halo
  -> INTERP 2x  (1 CUDA tile + 3 HIP tiles)
  -> crop halo, composite 1280x720
  -> SDL viewer or raw frames
```

The source can be the local camera (`/dev/video0`, MJPEG 1280x720 @ 30 fps,
downscaled to 640x360) or a directory of `frame_%06d.rgb` files. The gate uses a
recorded clip so its reference is deterministic.

## Live camera

One command (starts the SDL viewer as a child; requires `DISPLAY`):

```sh
cd /home/balrog/Documents/OpenCode/jotnar-hip-release
./target/release/jotnar-stream --descriptor demos/video-wall/camera-wall.json \
  --cuda 192.168.1.116:9997 --hip 192.168.1.195:9997 --token test --viewer
```

Or two terminals. The stream is the frame server; it does not create a window
by itself.

```sh
# terminal 1
./target/release/jotnar-stream --descriptor demos/video-wall/camera-wall.json \
  --cuda 192.168.1.116:9997 --hip 192.168.1.195:9997 --token test
# terminal 2
./target/release/jotnar-doom-view --addr 127.0.0.1:9990
```

The viewer retries the connection for 15 seconds, so either order works. The
stream keeps running if the viewer disconnects. The viewer must run in the
desktop session that owns the camera/display (`DISPLAY=:0` here).

## Recorded clip

```sh
./target/release/jotnar-stream --descriptor demos/video-wall/frames-wall.json \
  --cuda 192.168.1.116:9997 --hip 192.168.1.195:9997 --token test \
  --clip /tmp/opencode/wall-clip --frames 120
```

## Gate

```sh
python3 scripts/video_wall_gate.py --no-build --frames 30 \
  --out /tmp/opencode/wall-gate
```

The gate warms the tile workers, runs the same tiled plan on one CUDA node as a
reference, then runs 1 CUDA + 3 HIP tiles and compares every 1280x720 frame
byte-for-byte. It requires p99 ≤ 66 ms, max ≤ 132 ms, zero over-budget frames,
four tiles per frame, both targets, and job ids in the receipt.

The 1:3 placement is measured, not fixed: the CUDA half is the slow side on
this fleet, and three HIP tiles finish before one CUDA tile in this workload.
