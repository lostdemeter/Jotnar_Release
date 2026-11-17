#!/bin/sh
# Physical N1/N2a job gate: drive an already-booted node over the LAN.
#
# From balrog:
#   ADDR=192.168.1.116:9999 ./crates/jotnar-node/test/physical_jobs.sh
#
# From Mint, override CTL if needed. Requires target/release/jotnar-ctl
# and artifacts/node.token (or TOKEN/TOKEN_FILE).
set -eu

PROJECT=$(CDPATH= cd -- "$(dirname -- "$0")/../../.." && pwd)
ADDR=${ADDR:-192.168.1.116:9999}
CTL=${CTL:-"$PROJECT/target/release/jotnar-ctl"}
TOKEN_FILE=${TOKEN_FILE:-"$PROJECT/artifacts/node.token"}
WORK=${WORK:-"$PROJECT/artifacts/physical-gate"}
TOKEN=${TOKEN:-$(cat "$TOKEN_FILE")}

[ -x "$CTL" ] || { echo "missing ctl: $CTL" >&2; exit 1; }
[ -n "$TOKEN" ] || { echo "missing token" >&2; exit 1; }

mkdir -p "$WORK"
MATMUL="$WORK/matmul.json"
LONG="$WORK/long.json"
VEC="$WORK/vec.json"
RESIZE="$WORK/resize.json"
python3 - "$MATMUL" "$LONG" "$VEC" "$RESIZE" <<'PY'
import json, sys
json.dump({"streams": {
    "a00": {"kind": "I64", "shape": [1], "data": ["3"]},
    "a01": {"kind": "I64", "shape": [1], "data": ["1"]},
    "a10": {"kind": "I64", "shape": [1], "data": ["2"]},
    "a11": {"kind": "I64", "shape": [1], "data": ["4"]},
    "b00": {"kind": "I64", "shape": [1], "data": ["5"]},
    "b01": {"kind": "I64", "shape": [1], "data": ["7"]},
    "b10": {"kind": "I64", "shape": [1], "data": ["6"]},
    "b11": {"kind": "I64", "shape": [1], "data": ["8"]},
}}, open(sys.argv[1], "w"))
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

echo "[physical] ping $ADDR"
run ping | python3 -c 'import json,sys; d=json.load(sys.stdin)["pong"]; j=d["jobs"]; print("[physical] runtime",d["runtime_version"],"workers",j["workers"],"transport",j["result_transport"]); assert j["result_transport"]=="inline+chunked"'

run program tile "$PROJECT/showcase/language/matmul2.asm" >/dev/null
run program long "$WORK/long.asm" >/dev/null
run program vec "$WORK/vec.asm" >/dev/null
run program resize "$WORK/resize.asm" >/dev/null

echo "[physical] sync exec"
run exec tile --payload "$MATMUL" --outputs c00,c01,c10,c11 | python3 -c \
  'import json,sys; d=json.load(sys.stdin)["streams"]; got={k:v["data"] for k,v in d.items()}; want={"c00":["21"],"c01":["29"],"c10":["34"],"c11":["46"]}; assert got==want,(got,want); print("[physical] sync exec OK")'

echo "[physical] async submit + wait"
OUT=$(run submit tile --payload "$MATMUL" --outputs c00,c01,c10,c11)
JID=$(printf '%s' "$OUT" | python3 -c 'import json,sys; print(json.load(sys.stdin)["job_id"])')
run job-wait "$JID" --receipt 2>"$WORK/receipt.json" \
  | python3 -c 'import json,sys; d=json.load(sys.stdin)["streams"]; assert d["c00"]["data"]==["21"]; print("[physical] async OK")'
python3 -c 'import json; r=json.load(open("'"$WORK"'/receipt.json")); assert r["verdict"]=="exact"; print("[physical] receipt",r["runtime_version"])'

echo "[physical] batch 200 matmul2"
run job-batch tile --payload "$MATMUL" --outputs c00,c01,c10,c11 --count 200 | tee "$WORK/batch.json"
python3 -c 'import json; d=json.load(open("'"$WORK"'/batch.json")); assert d["done"]==200 and d["failed"]==0 and d["identical_outputs"]; print("[physical] batch OK",d["jobs_per_s"],"jobs/s")'

echo "[physical] timeout / cancel"
set +e
run job-run long --payload "$LONG" --outputs OUT --timeout-ms 50 --wait-ms 10000 >"$WORK/timeout.out" 2>&1
rc=$?
set -e
[ "$rc" -ne 0 ] || { echo "timeout unexpectedly succeeded" >&2; exit 1; }
grep -q "timed_out" "$WORK/timeout.out" || { cat "$WORK/timeout.out" >&2; exit 1; }
OUT=$(run submit long --payload "$LONG" --outputs OUT --timeout-ms 0)
JID=$(printf '%s' "$OUT" | python3 -c 'import json,sys; print(json.load(sys.stdin)["job_id"])')
sleep 0.2
run job-cancel "$JID" >/dev/null
set +e
run job-wait "$JID" --wait-ms 10000 >"$WORK/cancel.out" 2>&1
rc=$?
set -e
[ "$rc" -ne 0 ] || { echo "cancel unexpectedly succeeded" >&2; exit 1; }
grep -q "canceled" "$WORK/cancel.out" || { cat "$WORK/cancel.out" >&2; exit 1; }
echo "[physical] timeout/cancel OK"

echo "[physical] large store-backed result"
run job-run resize --payload "$RESIZE" --outputs OUT --verify --wait-ms 180000 \
  | tee "$WORK/large.json"
python3 -c 'import json; d=json.load(open("'"$WORK"'/large.json"))["result"]; assert d["transport"]=="store",d; assert d["bytes"]==67280000,d; assert d["chunks"]==65,d; assert d["verified"]; print("[physical] large result OK",d["bytes"],"bytes",d["chunks"],"chunks")'

echo "[physical] vec 64K batch"
run job-batch vec --payload "$VEC" --outputs OUT --count 50 | tee "$WORK/vec.json.out"
python3 -c 'import json; d=json.load(open("'"$WORK"'/vec.json.out")); assert d["done"]==50 and d["identical_outputs"]; print("[physical] vec OK",d["jobs_per_s"],"jobs/s")'

run job-list | python3 -c 'import json,sys; d=json.load(sys.stdin); print("[physical] stats",d["stats"]); assert d["stats"]["running"]==0'
echo "[physical] PASS"
