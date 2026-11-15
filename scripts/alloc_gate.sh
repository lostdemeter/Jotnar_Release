#!/usr/bin/env sh
# Allocation tripwire for the warm runtime.
#
# The exact discipline is asserted by `cargo test --workspace`:
#   * `tests/discipline.rs` pins the boundary (frame assembly) at 0,
#     the warm job count to a band, and the per-mnemonic rate to <= 32;
#   * `tests/arena_witness.rs` pins arena steady state at 0.
#
# This gate is deliberately looser. It exists to catch an order-of-magnitude
# regression on a new toolchain, not to re-litigate the band. Override with
# ALLOC_GATE_CEILING when a compiler change legitimately moves the number.
set -eu
cd "$(dirname "$0")/.."
CEILING="${ALLOC_GATE_CEILING:-900}"
exec cargo run --quiet --bin jotnar-det -- \
    tests/fixtures/alloc_gate.asm \
    tests/fixtures/alloc_gate.ndjson \
    --outputs "${ALLOC_GATE_OUTPUTS:-OUT}" \
    --alloc-gate "$CEILING"
