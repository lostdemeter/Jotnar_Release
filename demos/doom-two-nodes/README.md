# Two-node DOOM demo

This directory is the standalone demo. It contains the launcher, the Jotnar
listing, and slots for the binaries and assets:

```text
run.sh              launcher
build.sh            build coordinator + viewer into bin/
package.sh          create a distributable tarball
config.env          fleet addresses, token, fps, viewer address
upscale-half.asm    Jotnar INTERP program both GPU nodes run
bin/                jotnar-doom-coord and jotnar-doom-view
assets/             doom.bin and doom1.wad
out/                logs and receipt
```

## Quick start

From a source checkout:

```sh
./run.sh
```

The first run builds the binaries into `bin/` and stages `assets/` from the
standard locations if needed.

## Standalone package

```sh
./package.sh
```

This creates:

```text
dist/doom-two-nodes-demo.tar.gz
```

The tarball contains the launcher, both binaries, the listing, the DOOM image
and WAD, and this README. Extract it anywhere and run `./run.sh`; it still needs
the CUDA/HIP runners, a display, and `libSDL2-2.0.so.0`.

## What runs

```text
node A  RV32IM core on this machine
node B  CUDA runner, one overlapping frame window
node C  HIP runner, the other window
viewer  SDL2 window + key forwarding
```

The coordinator calibrates both cards, chooses the split from measured
throughput, rebalances once, and records the decision. The HUD shows node
roles, columns, per-node latency, frame budget, rolling p50/p95, and
failover/recovery state.

## Controls

```text
arrows move
ctrl fires
space uses
shift runs
esc opens the menu
q quits
```

## Options

```text
--cuda ADDR        CUDA runner address
--hip ADDR         HIP runner address
--token T          trusted-LAN token
--fps N            target frame rate
--start-frame N    frame to start the game at
--frames N         coordinator frame budget
--viewer-addr A    local viewer listener
--failover-timeout-ms N  fail over when a split job exceeds this deadline
--out DIR          logs, receipt, optional frames
--build            force a rebuild into bin/
--no-build         require existing bin/ binaries
--no-viewer        run the coordinator headless
--record           write raw frames and frames.jsonl
```

Defaults are in `config.env`; environment variables override them.

## Output

```text
out/coordinator.log
out/viewer.log
out/run.json
out/frames/         only with --record
```

The demo does not record frames by default, so a long play session does not
fill the disk.

## Troubleshooting

- `CUDA runner is not reachable` — check the address and that
  `jotnar-gpu-runner` is running.
- `no DISPLAY/WAYLAND_DISPLAY` — run from a desktop session or use
  `--no-viewer`.
- `missing bin/...` — run `./build.sh` or `./package.sh`.
- `missing assets/...` — put `doom.bin`/`doom1.wad` in `assets/` or set
  `DOOM_IMAGE`/`DOOM_WAD`.
- Blank window for a few seconds — warm-worker startup on the first run.
- One node loses its network — the survivor takes the full frame and the HUD
  changes to `cuda-full`/`hip-full`. Split jobs hit the failover deadline
  (default 500 ms) once, then the game keeps running; recovery probes happen in
  the background.

The full design and measured results are in the source repository under
`docs/DOOM_TWO_NODE.md` and `docs/WORKLOAD_COORDINATION.md`.
