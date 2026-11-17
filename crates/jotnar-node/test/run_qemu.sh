#!/bin/sh
# Interactive QEMU/OVMF launcher for the JotnarOS node (run on the Mint).
#
#   test/run_qemu.sh [IMAGE]
#
# Serial console is stdio by default, so Ctrl-A X exits QEMU. The node
# service listens on guest TCP 9999; it is forwarded to host 127.0.0.1:9999.
#
# For the automated M0 gate use crates/jotnar-node/crates/jotnar-node/test/qemu_smoke.py instead.
set -eu

PROJECT=$(CDPATH= cd -- "$(dirname -- "$0")/../../.." && pwd)
IMAGE=${1:-"$PROJECT/artifacts/jotnar-node-x86_64.img"}
PORT=${PORT:-9999}
VARS="$PROJECT/artifacts/OVMF_VARS_4M.fd"
OVMF_CODE=${OVMF_CODE:-/usr/share/OVMF/OVMF_CODE_4M.fd}
OVMF_VARS=${OVMF_VARS:-/usr/share/OVMF/OVMF_VARS_4M.fd}

[ -f "$IMAGE" ] || { echo "missing image: $IMAGE" >&2; exit 1; }
[ -f "$OVMF_CODE" ] || { echo "missing OVMF code: $OVMF_CODE" >&2; exit 1; }
[ -f "$OVMF_VARS" ] || { echo "missing OVMF vars: $OVMF_VARS" >&2; exit 1; }
mkdir -p "$PROJECT/artifacts"
cp -f "$OVMF_VARS" "$VARS"

ACCEL=tcg
[ -r /dev/kvm ] && [ -w /dev/kvm ] && ACCEL=kvm
CPU=max
[ "$ACCEL" = kvm ] && CPU=host

OVERLAY="$PROJECT/artifacts/qemu-interactive.qcow2"
rm -f "$OVERLAY"
qemu-img create -f qcow2 -b "$(readlink -f "$IMAGE")" -F raw "$OVERLAY" >/dev/null

qemu-system-x86_64 \
    -name jotnar-node \
    -machine q35 \
    -accel "$ACCEL" \
    -cpu "$CPU" \
    -m 1024 \
    -smp 2 \
    -drive "if=pflash,format=raw,readonly=on,file=$OVMF_CODE" \
    -drive "if=pflash,format=raw,file=$VARS" \
    -drive "file=$OVERLAY,format=qcow2,if=virtio" \
    -netdev "user,id=net0,hostfwd=tcp:127.0.0.1:${PORT}-:9999,hostfwd=tcp:127.0.0.1:2222-:22" \
    -device virtio-net-pci,netdev=net0 \
    -serial mon:stdio
status=$?
rm -f "$OVERLAY"
exit "$status"
