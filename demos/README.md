# Demos

## Two-node DOOM

```sh
./demos/doom_two_nodes.sh
```

This launches playable DOOM with the RV32IM core on this machine and the frame
work split across the CUDA and HIP GPU nodes. The viewer shows a live HUD with
node roles, column split, per-node latency, frame budget, rolling p50/p95, and
failover/recovery state.

### Prerequisites

- The CUDA and HIP `jotnar-gpu-runner` processes are running and reachable.
- DOOM assets exist:

  ```text
  $HOME/Documents/OpenCode/jotnar-main/programs/doom/build/doom.bin
  $HOME/Documents/OpenCode/jotnar-assets/doom1.wad
  ```

- A display is available (`DISPLAY` or `WAYLAND_DISPLAY`).
- `libSDL2-2.0.so.0` is installed.

The defaults match the current fleet:

```text
CUDA 192.168.1.116:9997
HIP  192.168.1.195:9997
token test
```

### Options

```text
--cuda ADDR        CUDA runner address
--hip ADDR         HIP runner address
--token T          trusted-LAN token
--fps N            target frame rate (default 20)
--start-frame N    frame to start the game at (default 200)
--frames N         coordinator frame budget
--viewer-addr A    local viewer listener
--out DIR          logs and raw frames
--no-build         use existing release binaries
--no-viewer        run the coordinator headless
```

Environment equivalents: `JOTNAR_CUDA`, `JOTNAR_HIP`, `JOTNAR_TOKEN`,
`JOTNAR_ROOT`, `DOOM_IMAGE`, `DOOM_WAD`, `DOOM_DEMO_OUT`.

### Controls

```text
arrows move
ctrl fires
space uses
shift runs
esc opens the menu
q quits
```

### What happens

```text
node A  RV32IM core (this machine)
node B  CUDA, one overlapping frame window
node C  HIP,  the other overlapping frame window
viewer  SDL2 display + key forwarding
```

The coordinator calibrates both cards, chooses the split from measured
throughput, rebalances once, and then runs the game loop. If a node fails, the
survivor takes the full frame; if it returns, the coordinator rejoins the split.

Logs and the receipt are written to the output directory:

```text
/tmp/opencode/doom_demo/coordinator.log
/tmp/opencode/doom_demo/viewer.log
/tmp/opencode/doom_demo/run.json
/tmp/opencode/doom_demo/frames/
```

### Troubleshooting

- `CUDA runner is not reachable` — check the runner address and that
  `jotnar-gpu-runner` is running on that host.
- `no DISPLAY/WAYLAND_DISPLAY` — run from a desktop session, or use
  `--no-viewer` for a headless coordinator.
- Window is blank for a few seconds — the first run may be building the warm
  workers; subsequent runs are faster.
- Port already in use — pass another `--viewer-addr`, for example
  `127.0.0.1:10001`.

The full design and measured results are in
[`docs/DOOM_TWO_NODE.md`](../docs/DOOM_TWO_NODE.md) and
[`docs/WORKLOAD_COORDINATION.md`](../docs/WORKLOAD_COORDINATION.md).
