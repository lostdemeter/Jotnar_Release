#!/usr/bin/env bash
# Compatibility wrapper: the standalone demo lives in doom-two-nodes/.
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$DIR/doom-two-nodes/run.sh" "$@"
