#!/usr/bin/env sh
# Arena soak: the opt-in arena path must produce exactly what the interpreter
# produces. Each fixture is run twice -- once with the arena off, once on --
# and the reference hashes must match.
#
# This is the gate that keeps the migration honest: unit tests cover kernels,
# these fixtures cover whole listings (mixed families, multi-output, fallback).
set -eu
cd "$(dirname "$0")/.."

refs() {
    JOTNAR_ARENA="$1" cargo run --quiet --bin jotnar-det -- \
        "tests/fixtures/$2.asm" "tests/fixtures/$2.ndjson" \
        --repeat "${3:-2}" 2>/dev/null | grep '^ref' | sed 's/ wall=.*//'
}

fail=0
for base in arena_soak arena_matmul arena_f arena_img arena_conv arena_fallback alloc_gate; do
    off=$(refs 0 "$base")
    on=$(refs 1 "$base")
    if [ "$off" = "$on" ] && [ -n "$on" ]; then
        echo "arena soak $base: match"
    else
        echo "arena soak $base: MISMATCH" >&2
        echo "  arena off: $off" >&2
        echo "  arena on : $on" >&2
        fail=1
    fi
done
[ "$fail" -eq 0 ] || exit 1

# The allocation tripwire has to hold with the arena enabled too.
JOTNAR_ARENA=1 ./scripts/alloc_gate.sh
echo "arena soak: OK"
