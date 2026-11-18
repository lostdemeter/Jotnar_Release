#!/usr/bin/env bash
# Launch playable two-node DOOM.
#
# Node A (this machine) runs the RV32IM core and coordinates the frame split.
# Node B (CUDA) and node C (HIP) run the same Jotnar INTERP program on their
# half of each frame. The SDL viewer displays the stitched frame and forwards
# your keys to the RV32IM core.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -d "$HOME/.cargo/bin" ]]; then
  PATH="$HOME/.cargo/bin:$PATH"
fi

JOTNAR_ROOT="${JOTNAR_ROOT:-$HOME/Documents/OpenCode/jotnar-main}"
WAD="${DOOM_WAD:-$HOME/Documents/OpenCode/jotnar-assets/doom1.wad}"
IMAGE="${DOOM_IMAGE:-$JOTNAR_ROOT/programs/doom/build/doom.bin}"
LISTING="$ROOT/showcase/doom/upscale-half.asm"
CUDA="${JOTNAR_CUDA:-192.168.1.116:9997}"
HIP="${JOTNAR_HIP:-192.168.1.195:9997}"
TOKEN="${JOTNAR_TOKEN:-test}"
FPS=20
START_FRAME=200
FRAMES=1000000
VIEWER_ADDR="127.0.0.1:10000"
OUT="${DOOM_DEMO_OUT:-/tmp/opencode/doom_demo}"
BUILD=1
VIEWER=1

usage() {
  cat <<'EOF'
Jotnar DOOM — play over two GPU nodes

  demos/doom_two_nodes.sh [options]

  --cuda ADDR        CUDA runner address (default 192.168.1.116:9997)
  --hip ADDR         HIP runner address  (default 192.168.1.195:9997)
  --token T          trusted-LAN token (default test)
  --fps N            target frame rate (default 20)
  --start-frame N    frame to start the game at (default 200)
  --frames N         coordinator frame budget (default effectively unlimited)
  --viewer-addr A    local viewer listener (default 127.0.0.1:10000)
  --out DIR          logs and raw frames (default /tmp/opencode/doom_demo)
  --no-build         use existing release binaries
  --no-viewer        run the coordinator without opening a window
  -h, --help         this text

Environment:
  JOTNAR_ROOT, DOOM_WAD, DOOM_IMAGE, JOTNAR_CUDA, JOTNAR_HIP, JOTNAR_TOKEN,
  DOOM_DEMO_OUT

Controls in the viewer:
  arrows move, ctrl fires, space uses, shift runs, esc opens the menu, q quits
EOF
}

while [[ $# -gt 0 ]]; do
  arg="$1"
  shift
  case "$arg" in
    --cuda) CUDA="${1:-}"; shift ;;
    --hip) HIP="${1:-}"; shift ;;
    --token) TOKEN="${1:-}"; shift ;;
    --fps) FPS="${1:-}"; shift ;;
    --start-frame) START_FRAME="${1:-}"; shift ;;
    --frames) FRAMES="${1:-}"; shift ;;
    --viewer-addr) VIEWER_ADDR="${1:-}"; shift ;;
    --out) OUT="${1:-}"; shift ;;
    --no-build) BUILD=0 ;;
    --no-viewer) VIEWER=0 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown option: $arg" >&2; usage >&2; exit 2 ;;
  esac
done

die() {
  echo "error: $*" >&2
  exit 1
}

[[ -f "$IMAGE" ]] || die "DOOM image not found: $IMAGE (set DOOM_IMAGE or JOTNAR_ROOT)"
[[ -f "$WAD" ]] || die "DOOM WAD not found: $WAD (set DOOM_WAD)"
[[ -f "$LISTING" ]] || die "Jotnar listing not found: $LISTING"
[[ "$FPS" =~ ^[0-9]+$ && "$FPS" -gt 0 ]] || die "--fps needs a positive integer"
[[ "$START_FRAME" =~ ^[0-9]+$ ]] || die "--start-frame needs an integer"
[[ "$FRAMES" =~ ^[0-9]+$ && "$FRAMES" -gt 0 ]] || die "--frames needs a positive integer"

if [[ "$VIEWER" == 1 && -z "${DISPLAY:-}${WAYLAND_DISPLAY:-}" ]]; then
  die "no DISPLAY/WAYLAND_DISPLAY; use --no-viewer for a headless run"
fi

port_open() {
  local addr="$1" host port
  host="${addr%:*}"
  port="${addr##*:}"
  timeout 2 bash -c "exec 3<>/dev/tcp/$host/$port" 2>/dev/null
}

if ! port_open "$CUDA"; then
  die "CUDA runner is not reachable at $CUDA (start jotnar-gpu-runner or pass --cuda)"
fi
if ! port_open "$HIP"; then
  die "HIP runner is not reachable at $HIP (start jotnar-gpu-runner or pass --hip)"
fi

if [[ "$BUILD" == 1 ]]; then
  echo "building coordinator and viewer..."
  (cd "$ROOT" && cargo build --release -p jotnar-node-tools \
    --bin jotnar-doom-coord --bin jotnar-doom-view)
fi

COORD="$ROOT/target/release/jotnar-doom-coord"
VIEW="$ROOT/target/release/jotnar-doom-view"
[[ -x "$COORD" ]] || die "missing $COORD (run without --no-build)"
[[ -x "$VIEW" ]] || die "missing $VIEW (run without --no-build)"

mkdir -p "$OUT"
COORD_LOG="$OUT/coordinator.log"
COORD_OUT="$OUT/coordinator.out"
VIEW_LOG="$OUT/viewer.log"
COORD_PID=""
VIEW_PID=""

cleanup() {
  local code=$?
  trap - EXIT INT TERM
  if [[ -n "$VIEW_PID" ]] && kill -0 "$VIEW_PID" 2>/dev/null; then
    kill "$VIEW_PID" 2>/dev/null || true
  fi
  if [[ -n "$COORD_PID" ]] && kill -0 "$COORD_PID" 2>/dev/null; then
    kill "$COORD_PID" 2>/dev/null || true
  fi
  wait "$VIEW_PID" 2>/dev/null || true
  wait "$COORD_PID" 2>/dev/null || true
  echo
  echo "logs:   $COORD_LOG"
  echo "frames: $OUT/frames"
  echo "receipt: $OUT/run.json"
  exit "$code"
}
trap cleanup EXIT INT TERM

cat <<EOF

============================================================
  Jotnar DOOM — one RV32IM core, two GPU nodes
============================================================
  node A   RV32IM core (this machine)
  node B   CUDA  $CUDA
  node C   HIP   $HIP
  target   $FPS fps
  viewer   $VIEWER_ADDR

EOF

echo "starting coordinator..."
"$COORD" \
  --image "$IMAGE" \
  --wad "$WAD" \
  --listing "$LISTING" \
  --cuda "$CUDA" \
  --hip "$HIP" \
  --token "$TOKEN" \
  --start-frame "$START_FRAME" \
  --frames "$FRAMES" \
  --fps "$FPS" \
  --viewer-listen "$VIEWER_ADDR" \
  --recover-every 20 \
  --out "$OUT" \
  >"$COORD_OUT" 2>"$COORD_LOG" &
COORD_PID=$!

echo "warming GPU workers and calibrating the split (this can take a few seconds)..."
for _ in $(seq 1 600); do
  if ! kill -0 "$COORD_PID" 2>/dev/null; then
    echo "coordinator exited during startup:" >&2
    tail -40 "$COORD_LOG" >&2 || true
    exit 1
  fi
  if grep -q 'placement mode:' "$COORD_LOG" 2>/dev/null; then
    break
  fi
  sleep 0.1
done
if ! grep -q 'placement mode:' "$COORD_LOG" 2>/dev/null; then
  echo "coordinator did not finish calibration:" >&2
  tail -40 "$COORD_LOG" >&2 || true
  exit 1
fi

if [[ "$VIEWER" == 1 ]]; then
  echo "opening viewer..."
  "$VIEW" --addr "$VIEWER_ADDR" --title "Jotnar DOOM - two nodes" \
    >"$VIEW_LOG" 2>&1 &
  VIEW_PID=$!
  echo
  echo "DOOM is running. Arrows move, ctrl fires, space uses, shift runs, esc menu, q quits."
  echo "Close the window or press q to stop."
  echo
  wait "$VIEW_PID" || true
  VIEW_PID=""
  # Let the coordinator observe the quit and write run.json before cleanup.
  for _ in $(seq 1 30); do
    kill -0 "$COORD_PID" 2>/dev/null || break
    sleep 0.1
  done
else
  echo "coordinator running without a viewer; press Ctrl-C to stop."
  wait "$COORD_PID" || true
  COORD_PID=""
fi
