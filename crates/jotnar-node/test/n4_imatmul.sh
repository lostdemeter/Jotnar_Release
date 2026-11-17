#!/bin/sh
# N4 representative workload: IMATMUL across two nodes.
#
# Start nodes with crates/jotnar-node/test/qemu_two_nodes.sh (or point at two existing nodes), then:
#   ADDR_A=192.168.1.116:9998 ADDR_B=192.168.1.116:9999 ./crates/jotnar-node/test/n4_imatmul.sh
#
# Defaults: 256x512 * 512x256 I64, 32 jobs, 1 thread per job.
set -eu

PROJECT=$(CDPATH= cd -- "$(dirname -- "$0")/../../.." && pwd)
NET=${NET:-"$PROJECT/target/release/jotnar-net"}
TOKEN_FILE=${TOKEN_FILE:-"$PROJECT/artifacts/node.token"}
TOKEN=${TOKEN:-$(cat "$TOKEN_FILE")}
ADDR_A=${ADDR_A:-192.168.1.116:9998}
ADDR_B=${ADDR_B:-192.168.1.116:9999}
COUNT=${COUNT:-32}
THREADS=${THREADS:-1}
WORK=${WORK:-"$PROJECT/artifacts/n4-imatmul"}
BATCH_ID=${BATCH_ID:-n4-$(date +%s)}

[ -x "$NET" ] || { echo "missing $NET; build jotnar-net first" >&2; exit 1; }
mkdir -p "$WORK"
PAYLOAD="$WORK/imatmul.json"
LISTING="$WORK/imatmul.asm"
printf 'IN A\nIN B\nOUT = IMATMUL(A, B)\n' >"$LISTING"
python3 - "$PAYLOAD" <<'PY'
import json, sys
M, K, N = 256, 512, 256
A = [[(i * 7 + j * 3) % 17 + 1 for j in range(K)] for i in range(M)]
B = [[(i * 5 + j * 11) % 13 + 1 for j in range(N)] for i in range(K)]
json.dump({"streams": {
    "A": {"kind": "I64", "shape": [M, K], "data": [str(x) for row in A for x in row]},
    "B": {"kind": "I64", "shape": [K, N], "data": [str(x) for row in B for x in row]},
}}, open(sys.argv[1], "w"))
PY

echo "[n4] $COUNT IMATMUL jobs on $ADDR_A + $ADDR_B"
"$NET" --token "$TOKEN" --node "$ADDR_A" --node "$ADDR_B" run imatmul \
  --program "$LISTING" --payload "$PAYLOAD" --outputs OUT \
  --count "$COUNT" --threads "$THREADS" --wait-ms 300000 --poll-ms 50 \
  --batch-id "$BATCH_ID" --receipts "$WORK/receipts.json" \
  | tee "$WORK/summary.json"
python3 -c 'import json; d=json.load(open("'"$WORK"'/summary.json")); assert d["done"]=='"$COUNT"' and d["failed"]==0 and d["identical_outputs"], d; print("[n4] PASS", d["jobs_per_s"], "jobs/s", d["output_bytes"], "bytes", d["output_hash"])'
