#!/usr/bin/env bash
# Warm-runtime benchmark: assemble once, feed the same payload N times.
#
# usage: scripts/bench_runtime_warm.sh LISTING PAYLOAD [ITERS]
#
# The payload must be one JSON object per line (the format written by
# `jotnar run --payload`). Prints the interpreter's run times and max RSS.
set -euo pipefail

listing=${1:?usage: bench_runtime_warm.sh LISTING PAYLOAD [ITERS]}
payload=${2:?usage: bench_runtime_warm.sh LISTING PAYLOAD [ITERS]}
iters=${3:-3}
repl=${JOTNAR_REPL:-target/release/jotnar-repl}
outputs=${JOTNAR_OUTPUTS:-OUT}

if [ ! -x "$repl" ]; then
  echo "missing $repl; run: cargo build --release --bin jotnar-repl" >&2
  exit 2
fi
if [ ! -f "$listing" ] || [ ! -f "$payload" ]; then
  echo "missing listing or payload" >&2
  exit 2
fi

tmp=$(mktemp)
trap 'rm -f "$tmp"' EXIT

for _ in $(seq "$iters"); do
  cat "$payload"
  printf '\n'
done | /usr/bin/time -v "$repl" "$listing" --outputs "$outputs" --stats \
  >/dev/null 2>"$tmp"

echo "listing: $listing"
echo "payload: $payload ($(du -h "$payload" | cut -f1))"
grep -E 'assembled|^run ' "$tmp"
grep -E 'Maximum resident set size' "$tmp" || true
