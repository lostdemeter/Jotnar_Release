#!/bin/sh
# Boot two disposable JotnarOS QEMU nodes for N3/N4 testing.
#
# Run on the build/QEMU host (Mint). Leaves both VMs running and prints their
# PIDs and host ports. Stop them with:
#   kill $(cat artifacts/n4-qemu/a.pid) $(cat artifacts/n4-qemu/b.pid)
set -eu

PROJECT=$(CDPATH= cd -- "$(dirname -- "$0")/../../.." && pwd)
IMAGE=${IMAGE:-"$PROJECT/artifacts/jotnar-node-x86_64.img"}
WORK=${WORK:-"$PROJECT/artifacts/n4-qemu"}
A_PORT=${A_PORT:-9998}
B_PORT=${B_PORT:-9999}
A_SSH=${A_SSH:-2222}
B_SSH=${B_SSH:-2223}
OVMF_CODE=${OVMF_CODE:-/usr/share/OVMF/OVMF_CODE_4M.fd}
OVMF_VARS=${OVMF_VARS:-/usr/share/OVMF/OVMF_VARS_4M.fd}
ACCEL=${ACCEL:-kvm}
SMP=${SMP:-2}
MEM=${MEM:-1024}

[ -f "$IMAGE" ] || { echo "missing image: $IMAGE" >&2; exit 1; }
[ -w /dev/kvm ] || ACCEL=tcg
CPU=max
[ "$ACCEL" = kvm ] && CPU=host
mkdir -p "$WORK"

start_node() {
  name=$1
  port=$2
  ssh_port=$3
  overlay="$WORK/$name.qcow2"
  serial="$WORK/$name.log"
  pid="$WORK/$name.pid"
  rm -f "$overlay" "$serial" "$pid"
  cp -f "$OVMF_VARS" "$WORK/$name.vars"
  qemu-img create -f qcow2 -b "$(readlink -f "$IMAGE")" -F raw "$overlay" >/dev/null
  qemu-system-x86_64 \
    -name "jotnar-$name" -machine q35 -accel "$ACCEL" -cpu "$CPU" \
    -m "$MEM" -smp "$SMP" \
    -drive "if=pflash,format=raw,readonly=on,file=$OVMF_CODE" \
    -drive "if=pflash,format=raw,file=$WORK/$name.vars" \
    -drive "file=$overlay,format=qcow2,if=virtio" \
    -netdev "user,id=net0,hostfwd=tcp:0.0.0.0:$port-:9999,hostfwd=tcp:0.0.0.0:$ssh_port-:22" \
    -device virtio-net-pci,netdev=net0 -display none \
    -serial "file:$serial" -pidfile "$pid" -daemonize
  echo "$name: pid $(cat "$pid"), job port $port, ssh $ssh_port, log $serial"
}

start_node a "$A_PORT" "$A_SSH"
start_node b "$B_PORT" "$B_SSH"

for name in a b; do
  ok=0
  for _ in $(seq 1 180); do
    if grep -q "JotnarOS node ready" "$WORK/$name.log" 2>/dev/null; then ok=1; break; fi
    sleep 1
  done
  if [ "$ok" != 1 ]; then
    echo "node $name failed to boot; serial tail:" >&2
    tail -40 "$WORK/$name.log" >&2 || true
    kill "$(cat "$WORK/a.pid")" "$(cat "$WORK/b.pid")" 2>/dev/null || true
    exit 1
  fi
  echo "node $name booted"
done

echo "Two nodes up. From balrog:"
echo "  ./target/release/jotnar-net --node 192.168.1.116:$A_PORT --node 192.168.1.116:$B_PORT ..."
echo "Stop: kill \$(cat $WORK/a.pid) \$(cat $WORK/b.pid)"
