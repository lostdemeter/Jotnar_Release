# DOOM on the RV32IM Jotnar core

This is the third second-leg exhibit: a real compiled program on a real
machine model. It is **RV32IM**, a 386-class 32-bit CPU — not x86, not DOS,
and not a 386 emulator.

```text
doomgeneric (C)
  -> Zig clang/lld, riscv32-freestanding -mcpu=generic_rv32+m
  -> flat RV32IM image (847,112 bytes)
  -> RV32IM machine model
  -> 320x200 RGBA framebuffer
```

The Rust core lives in `src/rv32.rs`; the runner is
`src/bin/jotnar-demo-doom.rs`.

## One command for everyone

```sh
./run_doom_demo.sh
```

The default is **playable DOOM** in a 960×600 window, running on the Rust
RV32IM core. Arrows move, ctrl fires, space uses, shift runs, esc opens the
menu, and q quits.

Headless frame runs use a frame count:

```sh
./run_doom_demo.sh --frames 100
./run_doom_demo.sh --no-open        # same as --frames 1
```

The script finds the DOOM assets, builds the Rust core, and opens the
window. If no display is available it falls back to a headless frame.

## Player

The player is `src/bin/jotnar-doom-player.rs`. It uses the same Rust core and
loads SDL2 through its C ABI, so no SDL headers or Rust SDL crate are needed.

## Run frame 1

Assets are external to this repo:

```sh
cargo run --release --bin jotnar-demo-doom -- \
  --image "$HOME/Documents/OpenCode/jotnar-main/programs/doom/build/doom.bin" \
  --wad   "$HOME/Documents/OpenCode/jotnar-assets/doom1.wad" \
  --frames 1 \
  --out /tmp/opencode/doom_rust/frame1 \
  --compare /tmp/opencode/doom_demo/frame1.fb
```

Reference result:

```text
frame 1: 24502445 instructions, pc=0x00048860
framebuffer: BYTE-IDENTICAL
```

The native C core produces the same instruction count, PC, registers,
first 64 MB of memory, and framebuffer.

## Checkpoint / migration gate

```sh
python3 scripts/parity_doom_native.py --gpu-work /tmp/opencode/doom_overlay
```

With the CUDA overlay present, the gate proves:

```text
frame1-mem64: OK
frame1-x: OK
frame1-pc: OK
frame1-cyc: OK
frame1-fb: OK
frame2-mem64: OK
frame2-x: OK
frame2-pc: OK
frame2-cyc: OK
frame2-fb: OK
gpu-frame2-mem64: OK
gpu-frame2-x: OK
gpu-frame2-pc: OK
gpu-frame2-fb: OK
RESULT: ALL OK
```

That means the Rust core's frame-1 checkpoint can be resumed by the C
native core, and the frame-2 state is byte-identical to the GPU Jotnar
core's checkpoint and framebuffer.

## Known issue — player frame spikes

The Rust player has intermittent frame-time spikes that were not observed in
the original Python/C player. It is playable, but not yet smooth enough to
call finished. Future work:

- record a frame-time trace (`Instant` around step loop, texture update,
  render, event poll, sleep) instead of judging by eye;
- compare against the same trace from the original ctypes player;
- check whether the spikes come from SDL texture upload/present, the
  64 MB-vs-128 MB memory layout, timer/clock handoff, or host scheduling;
- add a p99 / max-frame-time gate once the cause is known.

This is a player issue, not a RV32 core exactness issue: frame 1/2 state and
framebuffer bytes still gate against the C and GPU references.

## What is ported, and what is not

Ported to Rust:

- RV32I + M interpreter, including the spec edge cases for multiply/divide
- CSR/Sv32 surface and traps used by the freestanding runtime
- DOOM MMIO: WAD, framebuffer, clock, frame counter, key ring, UART/CLINT
- checkpoint save/load
- framebuffer PNG and raw output
- frame-1 and frame-2 exactness against the existing native and GPU demos

Not ported:

- the CUDA `rv32_kernel.cu` fleet kernel
- the playable SDL2 frontend
- the full DOOM toolchain (the 847 KB image is the artifact we run)

The Rust binary is the native exact core; the existing CUDA overlay remains
the GPU witness. The important migration contract is verified: the modeled
64 MB machine state is portable and byte-exact.
