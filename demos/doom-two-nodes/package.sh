#!/usr/bin/env bash
# Build a self-contained demo tarball with binaries, assets, listing, and docs.
set -euo pipefail

DEMO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$DEMO_DIR/../.." && pwd)"
if [[ -d "$HOME/.cargo/bin" ]]; then
  PATH="$HOME/.cargo/bin:$PATH"
fi

"$DEMO_DIR/build.sh"

ASSET_DIR="$DEMO_DIR/assets"
IMAGE="$ASSET_DIR/doom.bin"
WAD="$ASSET_DIR/doom1.wad"
default_image="${DOOM_IMAGE:-${JOTNAR_ROOT:-$HOME/Documents/OpenCode/jotnar-main}/programs/doom/build/doom.bin}"
default_wad="${DOOM_WAD:-$HOME/Documents/OpenCode/jotnar-assets/doom1.wad}"

mkdir -p "$ASSET_DIR"
if [[ ! -f "$IMAGE" ]]; then
  [[ -f "$default_image" ]] || {
    echo "error: missing $IMAGE and $default_image" >&2
    exit 1
  }
  cp "$default_image" "$IMAGE"
fi
if [[ ! -f "$WAD" ]]; then
  [[ -f "$default_wad" ]] || {
    echo "error: missing $WAD and $default_wad" >&2
    exit 1
  }
  cp "$default_wad" "$WAD"
fi

STAGE="$DEMO_DIR/dist/doom-two-nodes"
DIST="$DEMO_DIR/dist"
rm -rf "$STAGE"
mkdir -p "$STAGE/bin" "$STAGE/assets"
cp "$DEMO_DIR/run.sh" "$STAGE/run.sh"
cp "$DEMO_DIR/config.env" "$STAGE/config.env"
cp "$DEMO_DIR/upscale-half.asm" "$STAGE/upscale-half.asm"
cp "$DEMO_DIR/README.md" "$STAGE/README.md"
cp "$DEMO_DIR/bin/jotnar-doom-coord" "$STAGE/bin/"
cp "$DEMO_DIR/bin/jotnar-doom-view" "$STAGE/bin/"
cp "$IMAGE" "$STAGE/assets/doom.bin"
cp "$WAD" "$STAGE/assets/doom1.wad"
chmod +x "$STAGE/run.sh" "$STAGE/bin/jotnar-doom-coord" "$STAGE/bin/jotnar-doom-view"

TAR="$DIST/doom-two-nodes-demo.tar.gz"
tar -czf "$TAR" -C "$DIST" doom-two-nodes
echo "package: $TAR"
ls -lh "$TAR"
