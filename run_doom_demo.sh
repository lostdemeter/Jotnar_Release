#!/usr/bin/env bash
# One-command DOOM demo.
#
# Default: play DOOM in an SDL2 window on the Rust RV32IM core.
# --frames N: run N frames headless and save the last frame.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -d "$HOME/.cargo/bin" ]]; then
  PATH="$HOME/.cargo/bin:$PATH"
fi

JOTNAR_ROOT="${JOTNAR_ROOT:-$HOME/Documents/OpenCode/jotnar-main}"
WAD="${DOOM_WAD:-$HOME/Documents/OpenCode/jotnar-assets/doom1.wad}"
IMAGE="${DOOM_IMAGE:-$JOTNAR_ROOT/programs/doom/build/doom.bin}"
OUT="${DOOM_DEMO_OUT:-$ROOT/demo_out/doom}"
REFERENCE="$ROOT/showcase/doom/title.png"
EXPECTED_FB="4c40d22ef99cbf87eb28fb4c23c7c73d3a1462ea189cc10c037d1e8f5ce9fff3"

MODE=play
FRAMES=1
while [[ $# -gt 0 ]]; do
  arg="$1"
  shift
  case "$arg" in
    --frames)
      MODE=frames
      FRAMES="${1:-}"
      if [[ -z "$FRAMES" || ! "$FRAMES" =~ ^[0-9]+$ || "$FRAMES" -lt 1 ]]; then
        echo "--frames needs a positive integer" >&2
        exit 2
      fi
      shift
      ;;
    --no-open)
      MODE=frames
      FRAMES=1
      ;;
    --play)
      MODE=play
      ;;
    -h|--help)
      cat <<'EOF'
jotnar DOOM demo

  ./run_doom_demo.sh                  play DOOM in a window
  ./run_doom_demo.sh --frames 100     run 100 frames headless
  ./run_doom_demo.sh --no-open        headless frame 1 (same as --frames 1)
  ./run_doom_demo.sh --play           force the windowed player

  DOOM_WAD=...      use a different WAD
  JOTNAR_ROOT=...   use a different jotnar-main checkout
  DOOM_DEMO_OUT=... write frame images/checkpoints elsewhere
EOF
      exit 0
      ;;
    *)
      echo "unknown option: $arg" >&2
      exit 2
      ;;
  esac
done

banner() {
  printf '\n%s\n' "============================================================"
  printf '%s\n' "$1"
  printf '%s\n' "============================================================"
}

show_image() {
  local path="$1"
  if [[ -n "${DISPLAY:-}${WAYLAND_DISPLAY:-}" ]] && command -v xdg-open >/dev/null 2>&1; then
    xdg-open "$path" >/dev/null 2>&1 &
  else
    echo "Frame: $path"
  fi
}

fallback() {
  mkdir -p "$OUT"
  if [[ -f "$REFERENCE" ]]; then
    cp "$REFERENCE" "$OUT/title-reference.png"
    echo
    echo "Showing the checked-in reference title screen instead:"
    echo "  $OUT/title-reference.png"
    echo
    echo "To run live, make these files available:"
    echo "  $IMAGE"
    echo "  $WAD"
    show_image "$OUT/title-reference.png"
    exit 0
  fi
  exit 1
}

banner "Jotnar DOOM demo"
cat <<'EOF'
DOOM is running on a 32-bit RISC-V machine model written in Rust.
This is RV32IM, a 386-class CPU — not x86, not DOS, and not an emulator
of a commercial 386. The frame is built by the real DOOM shareware binary
compiled for that machine.
EOF

if [[ ! -f "$IMAGE" ]]; then
  echo
  echo "DOOM image not found: $IMAGE"
  fallback
fi
if [[ ! -f "$WAD" ]]; then
  echo
  echo "DOOM WAD not found: $WAD"
  fallback
fi
if ! command -v cargo >/dev/null 2>&1; then
  echo
  echo "cargo was not found; cannot build the Rust core."
  fallback
fi

mkdir -p "$OUT"
BUILD_LOG="$OUT/build.log"

if [[ "$MODE" == "play" && -z "${DISPLAY:-}${WAYLAND_DISPLAY:-}" ]]; then
  echo
  echo "No display found; running one frame headless instead."
  MODE=frames
  FRAMES=1
fi

if [[ "$MODE" == "play" ]]; then
  echo
  echo "Building the playable Rust core..."
  if ! (cd "$ROOT" && cargo build --release --quiet --bin jotnar-doom-player) >"$BUILD_LOG" 2>&1; then
    echo "Build failed. Details:"
    cat "$BUILD_LOG"
    exit 1
  fi
  echo "Starting DOOM..."
  echo
  echo "Arrows move, ctrl fires, space uses, shift runs, esc menu, q quits."
  if ! "$ROOT/target/release/jotnar-doom-player" --image "$IMAGE" --wad "$WAD"; then
    echo
    echo "The playable window could not start (SDL2/display)."
    echo "Trying a headless frame instead..."
    MODE=frames
    FRAMES=1
  else
    exit 0
  fi
fi

echo
echo "Building the headless Rust core..."
if ! (cd "$ROOT" && cargo build --release --quiet --bin jotnar-demo-doom) >"$BUILD_LOG" 2>&1; then
  echo "Build failed. Details:"
  cat "$BUILD_LOG"
  exit 1
fi

echo "Rendering $FRAMES frame(s)..."
"$ROOT/target/release/jotnar-demo-doom" \
  --image "$IMAGE" \
  --wad "$WAD" \
  --frames "$FRAMES" \
  --out "$OUT/frame$FRAMES"

if command -v sha256sum >/dev/null 2>&1; then
  HASH="$(sha256sum "$OUT/frame$FRAMES.fb" | awk '{print $1}')"
elif command -v shasum >/dev/null 2>&1; then
  HASH="$(shasum -a 256 "$OUT/frame$FRAMES.fb" | awk '{print $1}')"
else
  HASH=""
fi

if [[ "$FRAMES" == "1" && -n "$HASH" && "$HASH" == "$EXPECTED_FB" ]]; then
  echo
  echo "Exactness check: BYTE-IDENTICAL to the reference frame."
elif [[ -n "$HASH" ]]; then
  echo
  echo "Frame hash: $HASH"
fi

cat <<EOF

Saved frame $FRAMES to:
  $OUT/frame$FRAMES.png
  $OUT/frame$FRAMES.fb
  $OUT/frame$FRAMES.mem
  $OUT/frame$FRAMES.x
  $OUT/frame$FRAMES.pc

The Rust checkpoint resumes to the next frame exactly. The GPU fleet
kernel remains the optional CUDA witness.
EOF

show_image "$OUT/frame$FRAMES.png"
