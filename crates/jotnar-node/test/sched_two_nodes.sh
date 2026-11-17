#!/bin/sh
# N3 scheduler gate: two local nodes, aggregation, one-node-down, and
# mid-batch node loss with retry.
#
#   crates/jotnar-node/test/sched_two_nodes.sh
#
# Assumes target/release/{jotnar-noded,jotnar-net} are built.
set -eu

PROJECT=$(CDPATH= cd -- "$(dirname -- "$0")/../../.." && pwd)
NODED=${NODED:-"$PROJECT/target/release/jotnar-noded"}
NET=${NET:-"$PROJECT/target/release/jotnar-net"}
WORK=${WORK:-"$PROJECT/artifacts/sched-gate"}
PORT_A=${PORT_A:-9994}
PORT_B=${PORT_B:-9995}
PORT_C=${PORT_C:-9993}
TOKEN=${TOKEN:-}

[ -x "$NODED" ] || { echo "missing $NODED; build node bins first" >&2; exit 1; }
[ -x "$NET" ] || { echo "missing $NET; build node bins first" >&2; exit 1; }
mkdir -p "$WORK"
rm -rf "$WORK/ledger"

MATMUL="$WORK/matmul.json"
LONG="$WORK/long.json"
python3 - "$MATMUL" "$LONG" <<'PY'
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
    "N": {"kind": "I64", "shape": [1], "data": ["10000000"]},
}}, open(sys.argv[2], "w"))
PY
printf 'IN X\nIN N\nREPEAT N\nOUT = IADD(X, 1)\nEND\n' >"$WORK/long.asm"

"$NODED" --bind "127.0.0.1:$PORT_A" --workers 2 --queue 64 \
  --node-id-file "$WORK/node-a.id" --mdns \
  --supervisor /tmp/no-supervisor.sock >"$WORK/node-a.log" 2>&1 &
PA=$!
"$NODED" --bind "127.0.0.1:$PORT_B" --workers 2 --queue 64 \
  --node-id-file "$WORK/node-b.id" --mdns \
  --supervisor /tmp/no-supervisor.sock >"$WORK/node-b.log" 2>&1 &
PB=$!
cleanup() {
  kill "$PA" "$PB" 2>/dev/null || true
  [ -n "${PC:-}" ] && kill "$PC" 2>/dev/null || true
}
trap cleanup EXIT INT TERM
sleep 0.6

NETARGS="--heartbeat-ms 100 --down-after 2 --ledger $WORK/ledger --node 127.0.0.1:$PORT_A --node 127.0.0.1:$PORT_B"
[ -n "$TOKEN" ] && NETARGS="--token $TOKEN $NETARGS"

echo "[sched] health"
$NET $NETARGS health | python3 -c 'import json,sys; d=json.load(sys.stdin); assert len(d)==2 and all(n["healthy"] for n in d), d; assert all("draining" in n for n in d), d; print("[sched] both healthy, draining state present")'

echo "[sched] mdns discovery"
$NET --discover-timeout-ms 2500 discover >"$WORK/discover.json"
python3 -c 'import json; d=json.load(open("'"$WORK"'/discover.json")); assert len(d)>=2, d; assert all(x["node_id"] and x["caps"]=="jobs.v1" for x in d), d; print("[sched] discovery OK,", len(d), "advert(s)")'

echo "[sched] balanced matmul batch"
$NET $NETARGS run tile --program "$PROJECT/showcase/language/matmul2.asm" \
  --payload "$MATMUL" --outputs c00,c01,c10,c11 --count 40 --wait-ms 30000 \
  --batch-id sched-happy --receipts "$WORK/receipts.json" >"$WORK/happy.json"
python3 -c 'import json; d=json.load(open("'"$WORK"'/happy.json")); assert d["done"]==40 and d["failed"]==0 and d["identical_outputs"], d; assert all(n["done"]>0 for n in d["nodes"]), d; print("[sched] happy OK", d["jobs_per_s"], "jobs/s")'

echo "[sched] duplicate batch replay"
$NET $NETARGS run tile --program "$PROJECT/showcase/language/matmul2.asm" \
  --payload "$MATMUL" --outputs c00,c01,c10,c11 --count 40 --wait-ms 30000 \
  --batch-id sched-happy >"$WORK/replay.json"
python3 -c 'import json; d=json.load(open("'"$WORK"'/replay.json")); assert d["done"]==40 and d["failed"]==0 and d["cached"], d; print("[sched] ledger cache OK, cached", d["cached"])'
rm -f "$WORK/ledger/sched-happy.json"
$NET $NETARGS run tile --program "$PROJECT/showcase/language/matmul2.asm" \
  --payload "$MATMUL" --outputs c00,c01,c10,c11 --count 40 --wait-ms 30000 \
  --batch-id sched-happy >"$WORK/replay2.json"
python3 -c 'import json; d=json.load(open("'"$WORK"'/replay2.json")); assert d["done"]==40 and d["failed"]==0 and d["replayed"]==40 and not d["cached"], d; print("[sched] node replay OK, replayed", d["replayed"])'

echo "[sched] one node down at start"
$NET --node 127.0.0.1:$PORT_A --node 127.0.0.1:9999 run tile \
  --payload "$MATMUL" --outputs c00,c01,c10,c11 --count 10 --wait-ms 30000 \
  >"$WORK/one-down.json"
python3 -c 'import json; d=json.load(open("'"$WORK"'/one-down.json")); assert d["done"]==10 and d["failed"]==0, d; assert any(not n["healthy"] for n in d["nodes"]), d; print("[sched] one-down OK")'

echo "[sched] ledger resume"
$NET $NETARGS run long --program "$WORK/long.asm" --payload "$LONG" \
  --outputs OUT --count 4 --wait-ms 120000 --timeout-ms 60000 --poll-ms 25 \
  --batch-id sched-resume >"$WORK/resume-first.json" 2>"$WORK/resume-first.err" &
RSCHED=$!
sleep 1.3
kill -9 "$RSCHED" 2>/dev/null || true
wait "$RSCHED" 2>/dev/null || true
$NET $NETARGS run long --program "$WORK/long.asm" --payload "$LONG" \
  --outputs OUT --count 4 --wait-ms 120000 --timeout-ms 60000 --poll-ms 25 \
  --batch-id sched-resume >"$WORK/resume-second.json"
python3 -c 'import json; d=json.load(open("'"$WORK"'/resume-second.json")); assert d["done"]==4 and d["failed"]==0 and d["resumed"], d; print("[sched] resume OK, done", d["done"], "retried", d["retried"])'

echo "[sched] mid-batch node loss"
$NET $NETARGS run long --program "$WORK/long.asm" --payload "$LONG" \
  --outputs OUT --count 4 --wait-ms 120000 --timeout-ms 60000 --poll-ms 25 \
  >"$WORK/failure.json" 2>"$WORK/failure.err" &
SCHED=$!
sleep 1.0
kill -9 "$PB" 2>/dev/null || true
wait "$SCHED"
python3 -c 'import json; d=json.load(open("'"$WORK"'/failure.json")); assert d["done"]==4 and d["failed"]==0, d; assert d["retried"]>=1, d; assert any(not n["healthy"] for n in d["nodes"]), d; print("[sched] retry OK, retried", d["retried"], "wall", d["wall_ms"], "ms")'

echo "[sched] heartbeat draining"
"$NODED" --bind "127.0.0.1:$PORT_C" --workers 1 --queue 1 \
  --supervisor /tmp/no-supervisor.sock >"$WORK/node-c.log" 2>&1 &
PC=$!
sleep 0.5
CTL=${CTL:-"$PROJECT/target/release/jotnar-ctl"}
CTLARGS=""
[ -n "$TOKEN" ] && CTLARGS="--token $TOKEN"
$CTL --addr "127.0.0.1:$PORT_C" $CTLARGS program long "$WORK/long.asm" >/dev/null
J=$($CTL --addr "127.0.0.1:$PORT_C" $CTLARGS submit long --payload "$LONG" \
  --outputs OUT --timeout-ms 0 | python3 -c 'import json,sys; print(json.load(sys.stdin)["job_id"])')
for _ in $(seq 1 50); do
  state=$($CTL --addr "127.0.0.1:$PORT_C" $CTLARGS job-status "$J" \
    | python3 -c 'import json,sys; print(json.load(sys.stdin)["job"]["state"])')
  [ "$state" = "running" ] && break
  sleep 0.05
done
$CTL --addr "127.0.0.1:$PORT_C" $CTLARGS submit long --payload "$LONG" \
  --outputs OUT --timeout-ms 0 >/dev/null
sleep 0.3
$NET --node "127.0.0.1:$PORT_C" health | python3 -c 'import json,sys; d=json.load(sys.stdin)[0]; assert d["draining"] and d["queued"]>=d["queue_capacity"], d; print("[sched] draining OK", d["queued"], "/", d["queue_capacity"])'
kill "$PC" 2>/dev/null || true
PC=""

echo "[sched] PASS"
