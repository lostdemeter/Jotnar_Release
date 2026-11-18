#!/usr/bin/env bash
# Build the coordinator and viewer into this demo's bin/ directory.
set -euo pipefail

DEMO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$DEMO_DIR/../.." && pwd)"
if [[ -d "$HOME/.cargo/bin" ]]; then
  PATH="$HOME/.cargo/bin:$PATH"
fi

if [[ ! -f "$ROOT/Cargo.toml" ]]; then
  echo "error: source checkout not found at $ROOT" >&2
  echo "This demo directory needs its bin/ populated, or run it from the repo." >&2
  exit 1
fi

mkdir -p "$DEMO_DIR/bin"
echo "building jotnar-doom-coord and jotnar-doom-view..."
(cd "$ROOT" && cargo build --release -p jotnar-node-tools \
  --bin jotnar-doom-coord --bin jotnar-doom-view)

cp "$ROOT/target/release/jotnar-doom-coord" "$DEMO_DIR/bin/jotnar-doom-coord"
cp "$ROOT/target/release/jotnar-doom-view" "$DEMO_DIR/bin/jotnar-doom-view"
chmod +x "$DEMO_DIR/bin/jotnar-doom-coord" "$DEMO_DIR/bin/jotnar-doom-view"
echo "installed:"
echo "  $DEMO_DIR/bin/jotnar-doom-coord"
echo "  $DEMO_DIR/bin/jotnar-doom-view"
