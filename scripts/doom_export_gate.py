#!/usr/bin/env python3
"""Interactive + batch coexistence gate (D1b).

Runs the live DOOM stream with an interactive lease while a batch export runs on
the same CUDA/HIP fleet. The controller admits the export into a reserved batch
slot (capacity 2 = 1 interactive + 1 batch); the GPU runner executes one job at
a time, serves interactive frames first, and gives batch work the idle window
between frames (with a starvation fallback).

Required:
  * the interactive 100-frame SLO holds and every frame is exact;
  * the export completes and every output matches the single-node reference;
  * every export stage has an admission record with a reason;
  * the export span overlaps the live span (it is not a sequential afterthought).
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import doom_live_gate as live_gate  # noqa: E402
import doom_gpu_contention_gate as batch_gate  # noqa: E402

CONTROLLER = ROOT / "target" / "release" / "jotnar-workload"


def reference_command(
    token: str, bundle: Path, ledger: Path, cuda: str, hip: str, items: int
) -> list[str]:
    cmd = [
        str(CONTROLLER),
        "--token",
        token,
        "run",
        "--bundle",
        str(bundle),
        "--ledger",
        str(ledger),
        "--gpu-executor",
        f"cuda={cuda}",
        "--gpu-executor",
        f"hip={hip}",
    ]
    for i in range(items):
        # Single-node reference: both stages on CUDA.
        cmd += [
            "--place",
            f"gpu-a-{i:02}={cuda}",
            "--place",
            f"gpu-b-{i:02}={cuda}",
        ]
    return cmd


def live_command(
    cuda: str,
    hip: str,
    token: str,
    frames: int,
    out: Path,
    receipt: Path,
    stage_descriptor: Path,
    fleet: Path,
) -> list[str]:
    return [
        str(live_gate.DOOM_STREAM),
        "--image",
        str(live_gate.IMAGE),
        "--wad",
        str(live_gate.WAD),
        "--listing",
        str(live_gate.LISTING),
        "--stage-descriptor",
        str(stage_descriptor),
        "--cuda",
        cuda,
        "--hip",
        hip,
        "--token",
        token,
        "--frames",
        str(frames),
        "--start-frame",
        "200",
        "--fps",
        "15",
        "--left-cols",
        "72",
        "--sink",
        "frames",
        "--out",
        str(out),
        "--receipt",
        str(receipt),
        "--warmup-frames",
        str(live_gate.WARMUP),
        "--fleet-dir",
        str(fleet),
    ]


def span(receipts: list[dict]) -> tuple[int, int]:
    starts = [int(r.get("started_ms") or 0) for r in receipts if r.get("started_ms")]
    ends = [int(r.get("finished_ms") or 0) for r in receipts if r.get("finished_ms")]
    if not starts or not ends:
        return (0, 0)
    return (min(starts), max(ends))


def live_span(lines: list[dict]) -> tuple[int, int]:
    starts = [int(l.get("started_ms") or 0) for l in lines if l.get("started_ms")]
    if not starts:
        return (0, 0)
    ends = [
        int(l.get("started_ms") or 0) + int(l.get("work_ms") or 0)
        for l in lines
        if l.get("started_ms")
    ]
    return (min(starts), max(ends))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cuda", default=live_gate.DEFAULT_CUDA)
    ap.add_argument("--hip", default=live_gate.DEFAULT_HIP)
    ap.add_argument("--token", default="test")
    ap.add_argument("--frames", type=int, default=100)
    ap.add_argument("--items", type=int, default=24)
    ap.add_argument(
        "--clients",
        type=int,
        default=2,
        help="concurrent export clients; more than one exercises deferrals",
    )
    ap.add_argument(
        "--size",
        default="64x64x3",
        help="export payload HxWxC; must fit the interactive inter-frame gap",
    )
    ap.add_argument("--fleet-slots", type=int, default=2)
    ap.add_argument("--fleet-max-wait-ms", type=int, default=60000)
    ap.add_argument("--out", default="/tmp/opencode/doom-export-gate")
    ap.add_argument("--no-build", action="store_true")
    args = ap.parse_args()

    for addr in (args.cuda, args.hip):
        if not live_gate.reachable(addr):
            print(f"GPU runner is not reachable at {addr}")
            return 2
    for path in (live_gate.IMAGE, live_gate.WAD):
        if not path.exists():
            print(f"missing {path}")
            return 2
    if not args.no_build:
        live_gate.build()
        r = subprocess.run(
            ["cargo", "build", "--release", "-p", "jotnar-node-tools",
             "--bin", "jotnar-workload"],
            cwd=ROOT, capture_output=True, text=True,
        )
        if r.returncode != 0:
            print(r.stdout)
            print(r.stderr)
            raise SystemExit("build failed")
    for path in (CONTROLLER, live_gate.COORD, live_gate.DOOM_STREAM, live_gate.JOTNAR):
        if not path.exists():
            print(f"missing {path}; run without --no-build")
            return 2
    if args.frames < 2 or args.items < 1:
        print("--frames must be >= 2 and --items >= 1")
        return 2

    out = Path(args.out).absolute()
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    fleet = out / "fleet"
    fleet.mkdir()

    # 1. Deterministic live reference.
    coord = out / "coord"
    total = args.frames + live_gate.WARMUP
    live_gate.run(
        [
            str(live_gate.COORD),
            "--image", str(live_gate.IMAGE),
            "--wad", str(live_gate.WAD),
            "--cuda", args.cuda,
            "--hip", args.hip,
            "--token", args.token,
            "--frames", str(total),
            "--fps", "15",
            "--no-auto-split",
            "--no-warmup",
            "--out", str(coord),
        ]
    )
    stage_descriptor = live_gate.emit_stage_descriptor(out)

    # 2. Batch export bundle and single-node reference.
    bundle = out / "bundle"
    live_gate.run(
        [
            str(CONTROLLER),
            "bundle-gpu",
            "--listing", str(live_gate.LISTING),
            "--generate", args.size,
            "--items", str(args.items),
            "--out", str(bundle),
            "--name", "doom-export",
        ]
    )
    live_gate.run(
        reference_command(
            args.token, bundle, out / "reference", args.cuda, args.hip, args.items
        )
    )
    # Warm both runners with the same distributed shape, without a fleet lease.
    live_gate.run(
        batch_gate.pipeline_command(
            str(CONTROLLER), args.token, bundle, out / "warmup",
            args.cuda, args.hip, args.items, None,
        )
    )

    # 3. Start the live stream with the interactive lease.
    live_dir = out / "live"
    live_receipt = out / "live-receipt"
    live_log = open(out / "live.log", "w")
    live_cmd = live_command(
        args.cuda, args.hip, args.token, args.frames,
        live_dir, live_receipt, stage_descriptor, fleet,
    )
    live_proc = subprocess.Popen(
        live_cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=live_log, text=True
    )
    lines_path = live_receipt / "frames.jsonl"
    deadline = time.time() + 180
    while time.time() < deadline and len(live_gate.receipt_lines(lines_path)) < 1:
        if live_proc.poll() is not None:
            print("live stream exited before its frame loop")
            print((live_proc.stdout or "").read()[-2000:])
            return 1
        time.sleep(0.03)

    # 4. Run the batch export on the same fleet while the live stream is active.
    #    Two clients submit at once so the reserved batch slot is exercised and
    #    at least one stage records a deferral.
    export_started = time.time()
    export_procs = []
    for client in range(args.clients):
        ledger = out / f"export-{client:02}"
        cmd = batch_gate.pipeline_command(
            str(CONTROLLER), args.token, bundle, ledger,
            args.cuda, args.hip, args.items, out / "reference",
            fleet=fleet,
            fleet_slots=args.fleet_slots,
            fleet_max_wait_ms=args.fleet_max_wait_ms,
        )
        export_procs.append(
            (
                client,
                ledger,
                subprocess.Popen(
                    cmd,
                    cwd=ROOT,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                ),
            )
        )
    export_results = []
    for client, ledger, proc in export_procs:
        stdout, stderr = proc.communicate(timeout=300)
        export_results.append(
            {
                "client": f"{client:02}",
                "ledger": ledger,
                "returncode": proc.returncode,
                "stdout": stdout,
                "stderr": stderr,
            }
        )
    export_s = time.time() - export_started

    live_stdout, _ = live_proc.communicate(timeout=180)
    live_log.close()
    if live_proc.returncode != 0:
        print("live stream failed")
        print((live_stdout or "")[-2000:])
        return 1
    failed_clients = [r for r in export_results if r["returncode"] != 0]
    if failed_clients:
        for failed in failed_clients:
            print(f"export client {failed['client']} failed")
            print(failed["stdout"][-3000:])
            print(failed["stderr"][-3000:])
        return 1

    live_summary = json.loads(live_stdout)
    export_receipts = [
        json.loads((r["ledger"] / "receipt.json").read_text())
        for r in export_results
    ]

    # 5. Exactness, SLO, admission, overlap.
    comparisons = live_gate.compare_live(coord, live_dir, args.frames)
    matches = sum(1 for item in comparisons if item["match"])
    latency = live_summary.get("latency_ms", {})
    p99 = latency.get("p99", 0)
    max_ms = latency.get("max", 0)
    over = latency.get("over_budget", 0)

    all_stages: dict[str, dict] = {}
    admissions = []
    receipts_ok = True
    comparisons_ok = True
    expected_stages = args.clients * args.items * 2
    for client_index, receipt in enumerate(export_receipts):
        client = f"{client_index:02}"
        receipts_ok = receipts_ok and bool(receipt.get("ok"))
        stages = receipt.get("stages") or {}
        for name, record in stages.items():
            all_stages[f"{client}:{name}"] = record
        comps = receipt.get("comparisons") or []
        comparisons_ok = comparisons_ok and bool(comps) and all(
            item.get("match") for item in comps
        )
        fleet_stages = (receipt.get("fleet") or {}).get("stages") or {}
        for name, entry in sorted(fleet_stages.items()):
            admission = entry.get("admission") or {}
            record = stages.get(name) or {}
            admissions.append(
                {
                    "client": client,
                    "stage": name,
                    "decision": admission.get("decision"),
                    "target": admission.get("target"),
                    "waited_ms": admission.get("waited_ms"),
                    "attempts": admission.get("attempts"),
                    "deferrals": admission.get("deferrals"),
                    "reason": admission.get("reason"),
                    "capacity": admission.get("capacity"),
                    "interactive_slots": admission.get("interactive_slots"),
                    "in_flight_batch": admission.get("in_flight_batch"),
                    "interactive_leases": [
                        lease.get("id")
                        for lease in admission.get("interactive_leases") or []
                    ],
                    "state": record.get("state"),
                    "started_ms": record.get("started_ms"),
                    "finished_ms": record.get("finished_ms"),
                }
            )
    admitted = [a for a in admissions if a["decision"] == "admitted"]
    rejected = [a for a in admissions if a["decision"] == "rejected"]
    deferred = [a for a in admitted if (a.get("deferrals") or 0) > 0]
    saw_interactive_lease = any(a["interactive_leases"] for a in admitted)
    all_done = len(all_stages) == expected_stages and all(
        record.get("state") == "done" for record in all_stages.values()
    )

    export_span = span(list(all_stages.values()))
    live_span_ms = live_span(live_gate.receipt_lines(lines_path))
    overlap = bool(
        export_span[0] and live_span_ms[0]
        and export_span[0] < live_span_ms[1]
        and export_span[1] > live_span_ms[0]
    )

    live_ok = bool(
        live_summary.get("frames") == args.frames
        and matches == args.frames
        and p99 <= 66
        and max_ms <= 132
        and over == 0
    )
    export_ok = bool(
        receipts_ok
        and all_done
        and comparisons_ok
        and len(admitted) == expected_stages
        and not rejected
    )
    admission_ok = bool(
        len(admissions) == expected_stages
        and all(a["reason"] for a in admitted)
        and saw_interactive_lease
    )
    ok = bool(live_ok and export_ok and admission_ok and overlap)

    summary = {
        "schema": "jotnar.stream.doom_export_gate/1",
        "ok": ok,
        "cuda": args.cuda,
        "hip": args.hip,
        "frames": args.frames,
        "export_items": args.items,
        "export_stages": args.items * 2,
        "export_size": args.size,
        "export_seconds": round(export_s, 2),
        "fleet": {
            "dir": str(fleet),
            "capacity_per_target": args.fleet_slots,
            "max_wait_ms": args.fleet_max_wait_ms,
        },
        "live": {
            "frames": live_summary.get("frames"),
            "latency_ms": latency,
            "matches": matches,
            "split": live_summary.get("split"),
            "calibration": live_summary.get("calibration"),
            "receipt": str(lines_path),
            "summary": str(live_receipt / "summary.json"),
        },
        "export": {
            "clients": args.clients,
            "expected_stages": expected_stages,
            "receipts": [
                str(r["ledger"] / "receipt.json") for r in export_results
            ],
            "ledgers": [
                str(r["ledger"] / "ledger.json") for r in export_results
            ],
            "ok": receipts_ok,
            "stages": len(all_stages),
            "done": sum(1 for r in all_stages.values() if r.get("state") == "done"),
            "comparisons_ok": comparisons_ok,
            "span_ms": export_span,
            "admitted": len(admitted),
            "deferred": len(deferred),
            "rejected": len(rejected),
            "saw_interactive_lease": saw_interactive_lease,
            "max_wait_ms": max((a.get("waited_ms") or 0) for a in admitted) if admitted else 0,
            "admissions": admissions,
        },
        "live_span_ms": live_span_ms,
        "overlap": overlap,
        "checks": {
            "live_slo_and_exactness": live_ok,
            "export_complete_and_exact": export_ok,
            "admission_records": admission_ok,
            "no_starvation": export_ok and bool(admitted),
            "concurrent_overlap": overlap,
        },
        "evidence": {
            "live_frames": str(live_dir),
            "live_receipt": str(lines_path),
            "export_receipts": [
                str(r["ledger"] / "receipt.json") for r in export_results
            ],
            "export_ledgers": [
                str(r["ledger"] / "ledger.json") for r in export_results
            ],
            "summary": str(out / "doom-export-gate.json"),
        },
    }
    (out / "doom-export-gate.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
