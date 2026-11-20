#!/usr/bin/env bash
# One-command Jotnar Wall demo.
#
# Runs the camera wall on the live CUDA/HIP fleet and starts the SDL viewer with
# the wall HUD (tiles, targets, per-tile latency, p50/p95/p99, placement).
#
#   ./scripts/video_wall_demo.sh
#
# Overrides:
#   CUDA=HOST:PORT HIP=HOST:PORT TOKEN=SECRET
#   DESCRIPTOR=demos/video-wall/camera-wall.json
#   RECEIPT=/tmp/opencode/wall-demo-receipt
#   NO_VIEWER=1      # serve frames only; run jotnar-doom-view yourself
#
# Extra arguments are passed to jotnar-stream.
set -euo pipefail

cd "$(dirname "$0")/.."

CUDA=${CUDA:-192.168.1.116:9997}
HIP=${HIP:-192.168.1.195:9997}
TOKEN=${TOKEN:-test}
DESCRIPTOR=${DESCRIPTOR:-demos/video-wall/camera-wall.json}
RECEIPT=${RECEIPT:-/tmp/opencode/wall-demo-receipt}
NO_VIEWER=${NO_VIEWER:-0}

if [[ ! -e /dev/video0 && "$DESCRIPTOR" == *camera-wall.json ]]; then
  echo "no /dev/video0; set DESCRIPTOR=demos/video-wall/frames-wall.json CLIP=DIR" >&2
  exit 2
fi

cargo build --release -p jotnar-node-tools --bin jotnar-stream --bin jotnar-doom-view

args=(
  --descriptor "$DESCRIPTOR"
  --cuda "$CUDA"
  --hip "$HIP"
  --token "$TOKEN"
  --receipt "$RECEIPT"
)
if [[ "$NO_VIEWER" != "1" ]]; then
  args+=(--viewer)
else
  echo "NO_VIEWER=1: start ./target/release/jotnar-doom-view --addr 127.0.0.1:9990"
fi

exec ./target/release/jotnar-stream "${args[@]}" "$@"
