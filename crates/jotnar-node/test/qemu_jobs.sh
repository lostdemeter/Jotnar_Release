#!/bin/sh
# QEMU gate for the N1 async job API.
#
# Run on the build/QEMU host (Mint):
#   crates/jotnar-node/test/qemu_jobs.sh
#
# Boots a disposable overlay of artifacts/jotnar-node-x86_64.img under
# QEMU/OVMF, waits for the node service, then drives the job API over the
# hostfwd on 127.0.0.1:${PORT:-9999}. Requires target/release/jotnar-ctl
# and artifacts/node.token.
#
# Environment:
#   PORT=9999        hostfwd job port
#   SSH_PORT=2222    hostfwd SSH port
#   ACCEL=kvm|tcg    default kvm when /dev/kvm is writable
#   IMAGE=...        default artifacts/jotnar-node-x86_64.img
set -eu

PROJECT=$(CDPATH= cd -- "$(dirname -- "$0")/../../.." && pwd)
IMAGE=${IMAGE:-"$PROJECT/artifacts/jotnar-node-x86_64.img"}
CTL=${CTL:-"$PROJECT/target/release/jotnar-ctl"}
TOKEN_FILE=${TOKEN_FILE:-"$PROJECT/artifacts/node.token"}
PORT=${PORT:-9999}
SSH_PORT=${SSH_PORT:-2222}
OVMF_CODE=${OVMF_CODE:-/usr/share/OVMF/OVMF_CODE_4M.fd}
OVMF_VARS=${OVMF_VARS:-/usr/share/OVMF/OVMF_VARS_4M.fd}
WORK="$PROJECT/artifacts/n1-gate"
OVERLAY="$PROJECT/artifacts/qemu-jobs.qcow2"
SERIAL="$WORK/serial.log"
PIDFILE="$WORK/qemu.pid"
ACCEL=${ACCEL:-kvm}

[ -f "$IMAGE" ] || { echo "missing image: $IMAGE" >&2; exit 1; }
[ -x "$CTL" ] || { echo "missing ctl: $CTL" >&2; exit 1; }
[ -s "$TOKEN_FILE" ] || { echo "missing token: $TOKEN_FILE" >&2; exit 1; }
command -v qemu-system-x86_64 >/dev/null || { echo "qemu-system-x86_64 missing" >&2; exit 1; }
command -v qemu-img >/dev/null || { echo "qemu-img missing" >&2; exit 1; }
[ -w /dev/kvm ] || ACCEL=tcg

mkdir -p "$WORK"
rm -f "$OVERLAY" "$SERIAL" "$PIDFILE"
cp -f "$OVMF_VARS" "$WORK/OVMF_VARS.fd"
qemu-img create -f qcow2 -b "$(readlink -f "$IMAGE")" -F raw "$OVERLAY" >/dev/null

CPU=max
[ "$ACCEL" = kvm ] && CPU=host
qemu-system-x86_64 \
  -name jotnar-jobs -machine q35 -accel "$ACCEL" -cpu "$CPU" -m 1024 -smp 2 \
  -drive "if=pflash,format=raw,readonly=on,file=$OVMF_CODE" \
  -drive "if=pflash,format=raw,file=$WORK/OVMF_VARS.fd" \
  -drive "file=$OVERLAY,format=qcow2,if=virtio" \
  -netdev "user,id=net0,hostfwd=tcp:127.0.0.1:$PORT-:9999,hostfwd=tcp:127.0.0.1:$SSH_PORT-:22" \
  -device virtio-net-pci,netdev=net0 -display none \
  -serial "file:$SERIAL" -pidfile "$PIDFILE" -daemonize

QPID=$(cat "$PIDFILE")
cleanup() {
  kill "$QPID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

echo "[jobs] qemu pid $QPID ($ACCEL), waiting for boot"
ok=0
for _ in $(seq 1 180); do
  if grep -q "JotnarOS node ready" "$SERIAL" 2>/dev/null; then ok=1; break; fi
  kill -0 "$QPID" 2>/dev/null || break
  sleep 1
done
if [ "$ok" != 1 ]; then
  echo "[jobs] boot failed; serial tail:" >&2
  tail -40 "$SERIAL" >&2 || true
  exit 1
fi

TOKEN=$(cat "$TOKEN_FILE")
ADDR="127.0.0.1:$PORT"
MATMUL="$WORK/matmul.json"
LONG="$WORK/long.json"
VEC="$WORK/vec.json"
RESIZE="$WORK/resize.json"
python3 - "$MATMUL" "$LONG" "$VEC" "$RESIZE" <<'PY'
import json, sys
matmul = {"streams": {
    "a00": {"kind": "I64", "shape": [1], "data": ["3"]},
    "a01": {"kind": "I64", "shape": [1], "data": ["1"]},
    "a10": {"kind": "I64", "shape": [1], "data": ["2"]},
    "a11": {"kind": "I64", "shape": [1], "data": ["4"]},
    "b00": {"kind": "I64", "shape": [1], "data": ["5"]},
    "b01": {"kind": "I64", "shape": [1], "data": ["7"]},
    "b10": {"kind": "I64", "shape": [1], "data": ["6"]},
    "b11": {"kind": "I64", "shape": [1], "data": ["8"]},
}}
json.dump(matmul, open(sys.argv[1], "w"))
json.dump({"streams": {
    "X": {"kind": "I64", "shape": [1], "data": ["0"]},
    "N": {"kind": "I64", "shape": [1], "data": ["2000000000"]},
}}, open(sys.argv[2], "w"))
n = 65536
json.dump({"streams": {
    "X": {"kind": "I64", "shape": [n], "data": [str(i) for i in range(n)]},
    "Y": {"kind": "I64", "shape": [n], "data": [str(2 * i) for i in range(n)]},
}}, open(sys.argv[3], "w"))
json.dump({"streams": {
    "X": {"kind": "F64", "shape": [1, 1, 1], "data": [1.0]},
}}, open(sys.argv[4], "w"))
PY

printf 'IN X\nIN N\nREPEAT N\nOUT = IADD(X, 1)\nEND\n' >"$WORK/long.asm"
printf 'IN X\nIN Y\nOUT = IADD(X, Y)\n' >"$WORK/vec.asm"
printf 'IN X\nOUT = RESIZE2D(X, 2900, 2900, 0)\n' >"$WORK/resize.asm"

run() { "$CTL" --addr "$ADDR" --token "$TOKEN" "$@"; }

run ping | python3 -c 'import json,sys; d=json.load(sys.stdin)["pong"]; print("[jobs] runtime", d["runtime_version"], "workers", d["jobs"]["workers"], "features", d["features"])'
run program tile "$PROJECT/showcase/language/matmul2.asm" >/dev/null
run program long "$WORK/long.asm" >/dev/null
run program vec "$WORK/vec.asm" >/dev/null
run program resize "$WORK/resize.asm" >/dev/null

echo "[jobs] sync exec"
run exec tile --payload "$MATMUL" --outputs c00,c01,c10,c11 | python3 -c \
  'import json,sys; d=json.load(sys.stdin)["streams"]; got={k:v["data"] for k,v in d.items()}; want={"c00":["21"],"c01":["29"],"c10":["34"],"c11":["46"]}; assert got==want,(got,want); print("[jobs] sync exec OK")'

echo "[jobs] async submit + wait"
OUT=$(run submit tile --payload "$MATMUL" --outputs c00,c01,c10,c11)
JID=$(printf '%s' "$OUT" | python3 -c 'import json,sys; print(json.load(sys.stdin)["job_id"])')
run job-wait "$JID" --receipt 2>"$WORK/receipt.json" \
  | python3 -c 'import json,sys; d=json.load(sys.stdin)["streams"]; assert d["c00"]["data"]==["21"]; print("[jobs] async submit+wait OK")'
python3 -c 'import json; r=json.load(open("'"$WORK"'/receipt.json")); assert r["verdict"]=="exact"; print("[jobs] receipt", r["runtime_version"])'

echo "[jobs] batch 200 matmul2"
run job-batch tile --payload "$MATMUL" --outputs c00,c01,c10,c11 --count 200 | tee "$WORK/batch.json"
python3 -c 'import json; d=json.load(open("'"$WORK"'/batch.json")); assert d["done"]==200 and d["failed"]==0 and d["identical_outputs"]; print("[jobs] batch OK", d["jobs_per_s"], "jobs/s")'

echo "[jobs] timeout"
set +e
run job-run long --payload "$LONG" --outputs OUT --timeout-ms 50 --wait-ms 10000 >"$WORK/timeout.out" 2>&1
rc=$?
set -e
[ "$rc" -ne 0 ] || { echo "timeout unexpectedly succeeded" >&2; exit 1; }
grep -q "timed_out" "$WORK/timeout.out" || { cat "$WORK/timeout.out" >&2; exit 1; }
echo "[jobs] timeout OK"

echo "[jobs] cancel running"
OUT=$(run submit long --payload "$LONG" --outputs OUT --timeout-ms 0)
JID=$(printf '%s' "$OUT" | python3 -c 'import json,sys; print(json.load(sys.stdin)["job_id"])')
sleep 0.2
run job-status "$JID" | python3 -c 'import json,sys; j=json.load(sys.stdin)["job"]; assert j["state"] in ("queued","running"); print("[jobs] status", j["state"])'
run job-cancel "$JID" >/dev/null
set +e
run job-wait "$JID" --wait-ms 10000 >"$WORK/cancel.out" 2>&1
rc=$?
set -e
[ "$rc" -ne 0 ] || { echo "cancel unexpectedly succeeded" >&2; exit 1; }
grep -q "canceled" "$WORK/cancel.out" || { cat "$WORK/cancel.out" >&2; exit 1; }
echo "[jobs] cancel OK"

echo "[jobs] large store-backed result (> 64 MiB frame)"
run job-run resize --payload "$RESIZE" --outputs OUT --verify --wait-ms 180000 \
  | tee "$WORK/large.json"
python3 -c 'import json; d=json.load(open("'"$WORK"'/large.json"))["result"]; assert d["transport"]=="store", d; assert d["bytes"]==67280000, d; assert d["chunks"]==65, d; assert d["verified"]; print("[jobs] large result OK", d["bytes"], "bytes", d["chunks"], "chunks")'

echo "[jobs] vec 64K"
run job-batch vec --payload "$VEC" --outputs OUT --count 25 | tee "$WORK/vec.json.out"
python3 -c 'import json; d=json.load(open("'"$WORK"'/vec.json.out")); assert d["done"]==25 and d["identical_outputs"]; print("[jobs] vec OK", d["jobs_per_s"], "jobs/s")'

run job-list | python3 -c 'import json,sys; d=json.load(sys.stdin); print("[jobs] stats", d["stats"]); assert d["stats"]["running"]==0'
echo "[jobs] PASS"
