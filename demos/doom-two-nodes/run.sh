#!/usr/bin/env bash
# Standalone launcher for two-node DOOM.
#
# Keep this directory together: run.sh, upscale-half.asm, bin/, assets/.
# bin/ and assets/ can be populated by build.sh and package.sh.
set -euo pipefail

DEMO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$DEMO_DIR/../.." && pwd)"
if [[ -d "$HOME/.cargo/bin" ]]; then
  PATH="$HOME/.cargo/bin:$PATH"
fi

# Defaults live in config.env; real environment variables win.
# shellcheck source=config.env
source "$DEMO_DIR/config.env"

CUDA="${JOTNAR_CUDA}"
HIP="${JOTNAR_HIP}"
TOKEN="${JOTNAR_TOKEN}"
FPS="${DOOM_FPS}"
START_FRAME="${DOOM_START_FRAME}"
FRAMES=1000000
VIEWER_ADDR="${DOOM_VIEWER_ADDR}"
OUT="${DOOM_DEMO_OUT:-$DEMO_DIR/out}"
BUILD=auto
VIEWER=1
RECORD=0

usage() {
  cat <<'EOF'
Jotnar DOOM — play over two GPU nodes

  ./run.sh [options]

  --cuda ADDR        CUDA runner address
  --hip ADDR         HIP runner address
  --token T          trusted-LAN token
  --fps N            target frame rate
  --start-frame N    frame to start the game at
  --frames N         coordinator frame budget
  --viewer-addr A    local viewer listener
  --out DIR          logs, receipt, and optional frames
  --build            force a rebuild into bin/
  --no-build         never build; require bin/ binaries
  --no-viewer        run the coordinator headless
  --record           write raw frames and frames.jsonl
  -h, --help         this text

Assets and defaults:
  config.env         fleet addresses, token, fps, viewer address
  assets/doom.bin    RV32IM image
  assets/doom1.wad   DOOM WAD
  upscale-half.asm   the Jotnar program both GPU nodes run
  bin/               jotnar-doom-coord and jotnar-doom-view

Controls:
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
    --build) BUILD=1 ;;
    --no-build) BUILD=0 ;;
    --no-viewer) VIEWER=0 ;;
    --record) RECORD=1 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown option: $arg" >&2; usage >&2; exit 2 ;;
  esac
done

die() {
  echo "error: $*" >&2
  exit 1
}

[[ "$FPS" =~ ^[0-9]+$ && "$FPS" -gt 0 ]] || die "--fps needs a positive integer"
[[ "$START_FRAME" =~ ^[0-9]+$ ]] || die "--start-frame needs an integer"
[[ "$FRAMES" =~ ^[0-9]+$ && "$FRAMES" -gt 0 ]] || die "--frames needs a positive integer"
if [[ "$VIEWER" == 1 && -z "${DISPLAY:-}${WAYLAND_DISPLAY:-}" ]]; then
  die "no DISPLAY/WAYLAND_DISPLAY; use --no-viewer for a headless run"
fi

BIN_DIR="$DEMO_DIR/bin"
COORD="$BIN_DIR/jotnar-doom-coord"
VIEW="$BIN_DIR/jotnar-doom-view"
ASSET_DIR="$DEMO_DIR/assets"
IMAGE="$ASSET_DIR/doom.bin"
WAD="$ASSET_DIR/doom1.wad"
LISTING="$DEMO_DIR/upscale-half.asm"

stage_assets() {
  mkdir -p "$ASSET_DIR"
  local default_image="${DOOM_IMAGE:-${JOTNAR_ROOT:-$HOME/Documents/OpenCode/jotnar-main}/programs/doom/build/doom.bin}"
  local default_wad="${DOOM_WAD:-$HOME/Documents/OpenCode/jotnar-assets/doom1.wad}"
  if [[ ! -f "$IMAGE" ]]; then
    if [[ -f "$default_image" ]]; then
      cp "$default_image" "$IMAGE"
      echo "staged image: $IMAGE"
    fi
  fi
  if [[ ! -f "$WAD" ]]; then
    if [[ -f "$default_wad" ]]; then
      cp "$default_wad" "$WAD"
      echo "staged WAD:   $WAD"
    fi
  fi
}

if [[ "$BUILD" == 1 || ( "$BUILD" == auto && ( ! -x "$COORD" || ! -x "$VIEW" ) ) ]]; then
  "$DEMO_DIR/build.sh"
fi
[[ -x "$COORD" ]] || die "missing $COORD (run ./build.sh or package with binaries)"
[[ -x "$VIEW" ]] || die "missing $VIEW (run ./build.sh or package with binaries)"

stage_assets
[[ -f "$IMAGE" ]] || die "missing $IMAGE (put doom.bin in assets/ or set DOOM_IMAGE)"
[[ -f "$WAD" ]] || die "missing $WAD (put doom1.wad in assets/ or set DOOM_WAD)"
[[ -f "$LISTING" ]] || die "missing $LISTING"

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
  echo "logs:    $COORD_LOG"
  echo "receipt: $OUT/run.json"
  if [[ "$RECORD" == 1 ]]; then
    echo "frames:  $OUT/frames"
  fi
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
  record   $([[ "$RECORD" == 1 ]] && echo yes || echo no)

EOF

COORD_ARGS=(
  --image "$IMAGE"
  --wad "$WAD"
  --listing "$LISTING"
  --cuda "$CUDA"
  --hip "$HIP"
  --token "$TOKEN"
  --start-frame "$START_FRAME"
  --frames "$FRAMES"
  --fps "$FPS"
  --viewer-listen "$VIEWER_ADDR"
  --recover-every 20
  --out "$OUT"
)
if [[ "$RECORD" == 0 ]]; then
  COORD_ARGS+=(--no-record)
fi

echo "starting coordinator..."
"$COORD" "${COORD_ARGS[@]}" >"$COORD_OUT" 2>"$COORD_LOG" &
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
