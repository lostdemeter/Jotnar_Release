# Demos

## Two-node DOOM

The standalone demo lives in [`doom-two-nodes/`](doom-two-nodes/):

```text
demos/doom-two-nodes/run.sh
demos/doom-two-nodes/build.sh
demos/doom-two-nodes/package.sh
demos/doom-two-nodes/config.env
demos/doom-two-nodes/upscale-half.asm
demos/doom-two-nodes/bin/
demos/doom-two-nodes/assets/
```

Quick start from a source checkout:

```sh
./demos/doom-two-nodes/run.sh
```

The older `demos/doom_two_nodes.sh` path is a compatibility wrapper.

### Standalone package

```sh
./demos/doom-two-nodes/package.sh
```

This builds the binaries, stages the DOOM image and WAD, and writes:

```text
demos/doom-two-nodes/dist/doom-two-nodes-demo.tar.gz
```

The tarball is self-contained apart from the external runtime prerequisites:
the CUDA/HIP GPU runners, a display, and `libSDL2-2.0.so.0`.

### What it does

```text
node A  RV32IM core on this machine
node B  CUDA runner, one overlapping frame window
node C  HIP runner, the other window
viewer  SDL2 window + key forwarding + live HUD
```

The coordinator calibrates both cards, chooses the split from measured
throughput, rebalances once, and records the decision. The HUD shows node
roles, columns, per-node latency, frame budget, rolling p50/p95, and
failover/recovery state.

### Controls

```text
arrows move
ctrl fires
space uses
shift runs
esc opens the menu
q quits
```

See [`doom-two-nodes/README.md`](doom-two-nodes/README.md) for options,
troubleshooting, and packaging details.
