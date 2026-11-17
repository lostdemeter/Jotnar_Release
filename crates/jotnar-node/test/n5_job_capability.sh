#!/bin/sh
# N5 job-capability gate: interpreted + emitted dispatch through the
# supervisor to the local job endpoint, with token auth.
set -eu

PROJECT=$(CDPATH= cd -- "$(dirname -- "$0")/../../.." && pwd)
NODED=${NODED:-"$PROJECT/target/release/jotnar-noded"}
SUPERVISOR=${SUPERVISOR:-"$PROJECT/target/release/jotnar-supervisord"}
OSD=${OSD:-"$PROJECT/target/release/jotnar-osd"}
PAYLOAD=${PAYLOAD:-"$PROJECT/target/release/jotnar-payload"}
CTL=${CTL:-"$PROJECT/target/release/jotnar-ctl"}
WORK=${WORK:-"$PROJECT/artifacts/n5-gate"}
PORT=${PORT:-9998}
TOKEN=${TOKEN:-n5test-token}

for b in "$NODED" "$SUPERVISOR" "$OSD" "$PAYLOAD" "$CTL"; do
  [ -x "$b" ] || { echo "missing $b; build workspace bins first" >&2; exit 1; }
done
rm -rf "$WORK"
mkdir -p "$WORK/data/modules" "$WORK/modules" "$WORK/osd"
cp -r "$PROJECT/crates/jotnar-node/image/rootfs/usr/local/share/jotnar/modules/job-policy" "$WORK/data/modules/"
cp -r "$PROJECT"/crates/jotnar-node/image/rootfs/usr/local/share/jotnar/modules/* "$WORK/modules/"
printf '%s\n' "$TOKEN" >"$WORK/token"

printf 'IN X\nOUT = IADD(X, X)\n' >"$WORK/add.asm"
printf 'IN X\nIN N\nREPEAT N\nOUT = IADD(X, 1)\nEND\n' >"$WORK/long.asm"
python3 - "$WORK" <<'PY'
import json, sys
w = sys.argv[1]
n = 100
json.dump({"streams": {"X": {"kind": "I64", "shape": [n],
          "data": [str(i) for i in range(n)]}}}, open(w + "/add.json", "w"))
json.dump({"streams": {
    "X": {"kind": "I64", "shape": [1], "data": ["0"]},
    "N": {"kind": "I64", "shape": [1], "data": ["2000000000"]},
}}, open(w + "/long.json", "w"))
PY

cleanup() {
  [ -n "${PN:-}" ] && kill "$PN" 2>/dev/null || true
  [ -n "${PS:-}" ] && kill "$PS" 2>/dev/null || true
  [ -n "${PO:-}" ] && kill "$PO" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

"$NODED" --bind "127.0.0.1:$PORT" --workers 2 --queue 8 --token "$TOKEN" \
  --module-socket "$WORK/osd.sock" --supervisor "$WORK/supervisor.sock" \
  >"$WORK/noded.log" 2>&1 &
PN=$!
sleep 0.5
"$SUPERVISOR" --socket "$WORK/supervisor.sock" --data-dir "$WORK/data" \
  --job-endpoint "127.0.0.1:$PORT" --job-token-file "$WORK/token" \
  --log "$WORK/supervisor.log" >"$WORK/supervisor.out" 2>&1 &
PS=$!
"$OSD" --serve --socket "$WORK/osd.sock" --supervisor "$WORK/supervisor.sock" \
  --state-root "$WORK/osd" --manifest-dir "$WORK/modules" \
  >"$WORK/osd.out" 2>&1 &
PO=$!
sleep 0.8

run_ctl() { "$CTL" --addr "127.0.0.1:$PORT" --token "$TOKEN" "$@"; }
run_payload() { "$PAYLOAD" --addr "127.0.0.1:$PORT" --token "$TOKEN" "$@"; }

run_ctl program add "$WORK/add.asm" >/dev/null
run_ctl store X --payload "$WORK/add.json" --stream X >/dev/null
ARGS='{"handle":"add","inputs":{"X":{"store":"X"}},"outputs":["OUT"]}'

echo "[n5] module-step parity (job.submit)"
I=$(run_payload module-step job-policy job.submit --args "$ARGS")
E=$(run_payload module-step-emitted job-policy job.submit --args "$ARGS")
python3 -c 'import json,sys; i=json.loads(sys.argv[1]); e=json.loads(sys.argv[2]); assert i["action"]==e["action"]==18, (i,e); assert i["command"]["capability"]==e["command"]["capability"]=="job"; print("[n5] parity OK, action", i["action"])' "$I" "$E"

echo "[n5] interpreted submit + emitted status"
OUT=$(run_payload module-event-interpreted job-policy job.submit --args "$ARGS")
JID=$(printf '%s' "$OUT" | python3 -c 'import json,sys; print(json.load(sys.stdin)["result"]["result"]["job_id"])')
run_payload module-event job-policy job.status --args "{\"job_id\":\"$JID\"}" \
  | python3 -c 'import json,sys; r=json.load(sys.stdin)["result"]["result"]; assert r["job_id"]==sys.argv[1] and r["state"] in ("queued","running","done"), r; print("[n5] status OK", r["state"])' "$JID"

echo "[n5] emitted submit + list"
run_payload module-event job-policy job.submit --args "$ARGS" >/dev/null
run_payload module-event job-policy job.list \
  | python3 -c 'import json,sys; r=json.load(sys.stdin)["result"]["result"]; assert r["stats"]["retained"]>=2, r; print("[n5] list OK", r["stats"]["retained"])'

echo "[n5] submit + cancel a long job"
run_ctl program long "$WORK/long.asm" >/dev/null
run_ctl store X --payload "$WORK/long.json" --stream X >/dev/null
run_ctl store N --payload "$WORK/long.json" --stream N >/dev/null
LONGARGS='{"handle":"long","inputs":{"X":{"store":"X"},"N":{"store":"N"}},"outputs":["OUT"],"timeout_ms":0}'
OUT=$(run_payload module-event-interpreted job-policy job.submit --args "$LONGARGS")
JID=$(printf '%s' "$OUT" | python3 -c 'import json,sys; print(json.load(sys.stdin)["result"]["result"]["job_id"])')
sleep 0.2
run_payload module-event job-policy job.cancel --args "{\"job_id\":\"$JID\"}" \
  | python3 -c 'import json,sys; r=json.load(sys.stdin)["result"]["result"]; assert r["accepted"], r; print("[n5] cancel accepted")'
sleep 0.3
run_payload module-event job-policy job.status --args "{\"job_id\":\"$JID\"}" \
  | python3 -c 'import json,sys; r=json.load(sys.stdin)["result"]["result"]; assert r["state"]=="canceled", r; print("[n5] canceled OK")'

echo "[n5] PASS"
