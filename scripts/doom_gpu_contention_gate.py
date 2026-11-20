#!/usr/bin/env python3
"""DOOM + batch GPU pipeline contention gate.

Runs the CUDA -> HIP batch pipeline alone, then again while DOOM plays on the
same CUDA and HIP runners. The receipt records DOOM p50/p95/max and frame
budget, pipeline wall time and slowdown, placement, and failover state.

The expected honest result is that the current controller has no telemetry,
leases, priority, or admission control: both workloads share the runners
best-effort. A DOOM miss or a large pipeline slowdown is the evidence the next
controller layer needs.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CONTROLLER = ROOT / "target" / "release" / "jotnar-workload"
COORD = ROOT / "target" / "release" / "jotnar-doom-coord"
LISTING = ROOT / "demos" / "gpu-batch" / "double.asm"
JOTNAR = Path(
    os.environ.get(
        "JOTNAR_ROOT", str(Path.home() / "Documents" / "OpenCode" / "jotnar-main")
    )
)
IMAGE = JOTNAR / "programs" / "doom" / "build" / "doom.bin"
WAD = Path(
    os.environ.get(
        "DOOM_WAD",
        str(Path.home() / "Documents" / "OpenCode" / "jotnar-assets" / "doom1.wad"),
    )
)
DEFAULT_CUDA = os.environ.get("JOTNAR_CUDA", "192.168.1.116:9997")
DEFAULT_HIP = os.environ.get("JOTNAR_HIP", "192.168.1.195:9997")


def reachable(addr: str) -> bool:
    host, _, port = addr.partition(":")
    try:
        with socket.create_connection((host, int(port)), timeout=2):
            return True
    except OSError:
        return False


def build() -> None:
    r = subprocess.run(
        [
            "cargo",
            "build",
            "--release",
            "-p",
            "jotnar-node-tools",
            "--bin",
            "jotnar-workload",
            "--bin",
            "jotnar-doom-coord",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    if r.returncode != 0:
        print(r.stdout)
        print(r.stderr)
        raise SystemExit("build failed")


def run(cmd: list[str]) -> dict:
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    if r.returncode != 0:
        print("command failed:", " ".join(cmd))
        print(r.stdout)
        print(r.stderr)
        raise SystemExit(r.returncode)
    return json.loads(r.stdout)


def stage_names(items: int) -> list[str]:
    return [f"gpu-a-{i:02}" for i in range(items)] + [
        f"gpu-b-{i:02}" for i in range(items)
    ]


def percentile(samples: list[float], p: int) -> float | None:
    if not samples:
        return None
    s = sorted(samples)
    rank = max(1, (len(s) * p + 99) // 100)
    return s[min(rank, len(s)) - 1]


def frame_totals(path: Path) -> list[float]:
    if not path.exists():
        return []
    totals = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        value = record.get("total_ms")
        if isinstance(value, (int, float)):
            totals.append(float(value))
    return totals


def frame_stalls(path: Path, ceiling: float) -> list[dict]:
    """Frames above the hard ceiling, with the half that dominated."""
    if not path.exists():
        return []
    out = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        total = record.get("total_ms")
        if not isinstance(total, (int, float)) or total <= ceiling:
            continue
        out.append(
            {
                "index": record.get("index"),
                "frame": record.get("frame"),
                "total_ms": total,
                "core_ms": record.get("core_ms"),
                "cuda_ms": record.get("cuda_ms"),
                "hip_ms": record.get("hip_ms"),
                "mode": record.get("mode"),
            }
        )
    return out


def doom_command(
    coord: str,
    image: Path,
    wad: Path,
    cuda: str,
    hip: str,
    token: str,
    frames: int,
    fps: int,
    out: Path,
    fleet: Path | None,
) -> list[str]:
    cmd = [
        coord,
        "--image",
        str(image),
        "--wad",
        str(wad),
        "--cuda",
        cuda,
        "--hip",
        hip,
        "--token",
        token,
        "--frames",
        str(frames),
        "--fps",
        str(fps),
        "--out",
        str(out),
    ]
    if fleet is not None:
        cmd += ["--fleet-dir", str(fleet), "--lease-ms", "3000"]
    return cmd


def admission_summary(receipts: list[dict]) -> dict:
    stages = []
    for index, receipt in enumerate(receipts):
        client = f"{index:02}"
        fleet = receipt.get("fleet") or {}
        for name, entry in sorted((fleet.get("stages") or {}).items()):
            admission = entry.get("admission") or {}
            telemetry = entry.get("telemetry") or {}
            if admission:
                stages.append(
                    {
                        "client": client,
                        "stage": name,
                        "decision": admission.get("decision"),
                        "target": admission.get("target"),
                        "waited_ms": admission.get("waited_ms"),
                        "attempts": admission.get("attempts"),
                        "deferrals": admission.get("deferrals"),
                        "reason": admission.get("reason"),
                        "interactive_slots": admission.get("interactive_slots"),
                        "in_flight_batch": admission.get("in_flight_batch"),
                        "leases": [
                            lease.get("id")
                            for lease in admission.get("interactive_leases") or []
                        ],
                    }
                )
            elif entry.get("admission_error"):
                stages.append(
                    {
                        "client": client,
                        "stage": name,
                        "decision": "error",
                        "reason": entry.get("admission_error"),
                    }
                )
    telemetry = []
    for index, receipt in enumerate(receipts):
        fleet = receipt.get("fleet") or {}
        for name, entry in sorted((fleet.get("stages") or {}).items()):
            snapshot = entry.get("telemetry") or {}
            telemetry.append(
                {
                    "client": f"{index:02}",
                    "stage": name,
                    "ok": snapshot.get("ok"),
                    "addr": snapshot.get("addr"),
                    "age_ms": snapshot.get("age_ms"),
                    "queued": ((snapshot.get("health") or {}).get("activity") or {}).get(
                        "queued"
                    ),
                    "running": ((snapshot.get("health") or {}).get("activity") or {}).get(
                        "running"
                    ),
                    "error": snapshot.get("error"),
                }
            )
    admitted = [s for s in stages if s.get("decision") == "admitted"]
    deferred = [s for s in admitted if (s.get("deferrals") or 0) > 0]
    rejected = [s for s in stages if s.get("decision") == "rejected"]
    per_client = {}
    for stage in admitted:
        client = stage["client"]
        item = per_client.setdefault(
            client, {"admitted": 0, "deferred": 0, "max_wait_ms": 0, "leases_seen": []}
        )
        item["admitted"] += 1
        if (stage.get("deferrals") or 0) > 0:
            item["deferred"] += 1
        item["max_wait_ms"] = max(item["max_wait_ms"], stage.get("waited_ms") or 0)
        item["leases_seen"].extend(stage.get("leases") or [])
    for item in per_client.values():
        item["leases_seen"] = sorted(set(item["leases_seen"]))
    leases_seen = sorted(
        {
            lease
            for stage in admitted
            for lease in (stage.get("leases") or [])
        }
    )
    starved = [
        f"{index:02}"
        for index in range(len(receipts))
        if not (per_client.get(f"{index:02}") or {}).get("leases_seen")
    ]
    return {
        "stages": len(stages),
        "admitted": len(admitted),
        "deferred": len(deferred),
        "rejected": len(rejected),
        "per_client": per_client,
        "leases_seen": leases_seen,
        "clients_with_progress_under_lease": len(per_client) - len(starved),
        "starved_clients": starved,
        "rejections": rejected[:8],
        "telemetry_snapshots": len(telemetry),
        "telemetry_ok": sum(1 for t in telemetry if t.get("ok") is True),
        "telemetry_failed": [
            t for t in telemetry if t.get("ok") is not True
        ][:8],
        "max_wait_ms": max(
            (s.get("waited_ms") or 0 for s in admitted), default=0
        ),
    }


def pipeline_command(
    controller: str,
    token: str,
    bundle: Path,
    ledger: Path,
    cuda: str,
    hip: str,
    items: int,
    reference: Path | None,
    fleet: Path | None = None,
    fleet_slots: int = 2,
    fleet_max_wait_ms: int = 60_000,
) -> list[str]:
    cmd = [
        controller,
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
    if fleet is not None:
        cmd += [
            "--fleet-dir",
            str(fleet),
            "--fleet-slots",
            str(fleet_slots),
            "--fleet-max-wait-ms",
            str(fleet_max_wait_ms),
        ]
    for i in range(items):
        cmd += [
            "--place",
            f"gpu-a-{i:02}={cuda}",
            "--place",
            f"gpu-b-{i:02}={hip}",
        ]
        if reference is not None:
            cmd += [
                "--compare",
                f"gpu-b-{i:02}:OUT={reference}/stages/gpu-b-{i:02}/OUT",
            ]
    return cmd


def wait_for_frame_loop(proc: subprocess.Popen, marker: Path, timeout: float) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            raise SystemExit("DOOM coordinator exited before its frame loop")
        if marker.exists():
            return
        time.sleep(0.05)
    raise SystemExit("DOOM coordinator did not reach its frame loop")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cuda", default=DEFAULT_CUDA)
    ap.add_argument("--hip", default=DEFAULT_HIP)
    ap.add_argument("--token", default="test")
    ap.add_argument("--listing", default=str(LISTING))
    ap.add_argument("--size", default="64x64x3", help="generated HxWxC payload")
    ap.add_argument("--items", type=int, default=8)
    ap.add_argument(
        "--clients",
        type=int,
        default=1,
        help="concurrent pipeline clients during the DOOM run",
    )
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--doom-frames", type=int, default=0)
    ap.add_argument("--fleet-slots", type=int, default=1)
    ap.add_argument("--fleet-max-wait-ms", type=int, default=60_000)
    ap.add_argument("--skip-doom-alone", action="store_true")
    ap.add_argument("--out", default="/tmp/opencode/doom_gpu_contention")
    ap.add_argument("--no-build", action="store_true")
    args = ap.parse_args()

    for addr in (args.cuda, args.hip):
        if not reachable(addr):
            print(f"GPU runner is not reachable at {addr}")
            return 2
    for path in (IMAGE, WAD):
        if not path.exists():
            print(f"missing {path}")
            return 2
    if not args.no_build:
        build()
    for path in (CONTROLLER, COORD):
        if not path.exists():
            print(f"missing {path}; run without --no-build")
            return 2

    if args.items <= 0 or args.clients <= 0:
        print("--items and --clients must be >= 1")
        return 2
    items = args.items
    clients = args.clients
    out = Path(args.out).absolute()
    listing = Path(args.listing).absolute()
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    bundle = out / "bundle"
    fleet = out / "fleet"
    fleet.mkdir(parents=True, exist_ok=True)
    built = run(
        [
            str(CONTROLLER),
            "bundle-gpu",
            "--listing",
            str(listing),
            "--generate",
            args.size,
            "--items",
            str(items),
            "--out",
            str(bundle),
            "--name",
            "gpu-batch-contention",
        ]
    )

    # Single-node CUDA reference for every item.
    reference_cmd = [
        str(CONTROLLER),
        "--token",
        args.token,
        "run",
        "--bundle",
        str(bundle),
        "--ledger",
        str(out / "reference"),
        "--gpu-executor",
        f"cuda={args.cuda}",
        "--gpu-executor",
        f"hip={args.hip}",
    ]
    for name in stage_names(items):
        reference_cmd += ["--place", f"{name}={args.cuda}"]
    t0 = time.time()
    run(reference_cmd)
    reference_s = time.time() - t0

    # Warm both runners before measuring. The reference only touched CUDA, so
    # the first distributed run would otherwise pay the HIP worker startup and
    # make the contended run look artificially fast.
    run(
        pipeline_command(
            str(CONTROLLER),
            args.token,
            bundle,
            out / "warmup",
            args.cuda,
            args.hip,
            items,
            None,
        )
    )

    # Pipeline alone, both runners warm.
    t0 = time.time()
    run(
        pipeline_command(
            str(CONTROLLER),
            args.token,
            bundle,
            out / "baseline",
            args.cuda,
            args.hip,
            items,
            out / "reference",
        )
    )
    baseline_s = time.time() - t0

    doom_frames = args.doom_frames or max(
        30, math.ceil(baseline_s * clients * args.fps * 1.5)
    )
    if doom_frames < 100:
        # p99 is only meaningful with enough samples; below 100 frames the
        # nearest-rank p99 is the maximum. Record the shortfall rather than
        # pretending.
        print(f"note: {doom_frames} DOOM frames; p99 equals max below 100 frames")

    doom_alone = None
    if not args.skip_doom_alone:
        alone_out = out / "doom-alone"
        alone = run(
            doom_command(
                str(COORD),
                IMAGE,
                WAD,
                args.cuda,
                args.hip,
                args.token,
                doom_frames,
                args.fps,
                alone_out,
                fleet,
            )
        )
        alone_latency = alone.get("latency_ms", {})
        alone_budget = alone.get("frame_budget_ms")
        alone_max = alone_latency.get("total_max")
        alone_p99 = alone_latency.get("total_p99")
        doom_alone = {
            "latency_ms": alone_latency,
            "frame_budget_ms": alone_budget,
            "p99_ok": (
                alone_p99 <= alone_budget
                if alone_p99 is not None and alone_budget
                else None
            ),
            "max_ok": (
                alone_max <= 2 * alone_budget
                if alone_max is not None and alone_budget
                else None
            ),
            "frames": alone.get("frames"),
            "wall_ms": alone.get("wall_ms"),
            "run": str(alone_out / "run.json"),
        }
        print(
            "DOOM alone:",
            f"p50 {alone_latency.get('total_p50')} p95 {alone_latency.get('total_p95')}",
            f"p99 {alone_p99} max {alone_max} budget {alone_budget}",
        )

    doom_out = out / "doom"
    doom_log = open(out / "doom.log", "w")
    doom_cmd = doom_command(
        str(COORD),
        IMAGE,
        WAD,
        args.cuda,
        args.hip,
        args.token,
        doom_frames,
        args.fps,
        doom_out,
        fleet,
    )
    print("coordinator:", " ".join(doom_cmd))
    proc = subprocess.Popen(doom_cmd, cwd=ROOT, stdout=doom_log, stderr=doom_log)
    contended_receipts = []
    contended_s = 0.0
    try:
        wait_for_frame_loop(proc, doom_out / "frames.jsonl", timeout=120)
        time.sleep(0.2)
        clients_procs = []
        try:
            for client in range(clients):
                cmd = pipeline_command(
                    str(CONTROLLER),
                    args.token,
                    bundle,
                    out / f"contended-{client:02}",
                    args.cuda,
                    args.hip,
                    items,
                    out / "reference",
                    fleet,
                    args.fleet_slots,
                    args.fleet_max_wait_ms,
                )
                clients_procs.append(
                    (
                        client,
                        subprocess.Popen(
                            cmd,
                            cwd=ROOT,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                            text=True,
                        ),
                    )
                )
            t0 = time.time()
            for client, client_proc in clients_procs:
                stdout, stderr = client_proc.communicate()
                if client_proc.returncode != 0:
                    print(f"pipeline client {client} failed")
                    print(stdout)
                    print(stderr)
                    raise SystemExit(client_proc.returncode)
                contended_receipts.append(json.loads(stdout))
            contended_s = time.time() - t0
        finally:
            for _, client_proc in clients_procs:
                if client_proc.poll() is None:
                    client_proc.kill()
                    client_proc.wait()
        try:
            proc.wait(timeout=300)
        except subprocess.TimeoutExpired:
            print("DOOM coordinator did not finish within 300s")
            return 1
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()
        doom_log.close()

    if proc.returncode != 0:
        print((out / "doom.log").read_text())
        return 1
    doom = json.loads((doom_out / "run.json").read_text())
    baseline_receipt = json.loads((out / "baseline" / "receipt.json").read_text())
    comparisons = [
        comparison
        for receipt in contended_receipts
        for comparison in receipt.get("comparisons", [])
    ]
    pipeline_ok = (
        baseline_receipt.get("ok") is True
        and all(r.get("ok") is True for r in contended_receipts)
        and len(comparisons) == items * clients
        and all(c.get("match") is True for c in comparisons)
    )
    serial_equivalent = baseline_s * clients
    parallel_efficiency = (
        serial_equivalent / contended_s if contended_s else None
    )
    latency = doom.get("latency_ms", {})
    budget = doom.get("frame_budget_ms")
    totals = frame_totals(doom_out / "frames.jsonl")
    p99 = latency.get("total_p99")
    if p99 is None:
        p99 = percentile(totals, 99)
    max_ms = latency.get("total_max")
    max_ceiling = budget * 2 if budget else None
    p99_ok = bool(p99 is not None and budget and p99 <= budget)
    max_ok = bool(max_ms is not None and max_ceiling and max_ms <= max_ceiling)
    slo_ok = p99_ok and max_ok
    frames_over_budget = latency.get("frames_over_budget")
    if frames_over_budget is None and budget:
        frames_over_budget = sum(1 for value in totals if value > budget)
    stalls = frame_stalls(doom_out / "frames.jsonl", max_ceiling) if max_ceiling else []
    alone_latency = (doom_alone or {}).get("latency_ms", {})
    alone_p99 = alone_latency.get("total_p99")
    alone_max = alone_latency.get("total_max")
    alone_stalls = (
        frame_stalls(out / "doom-alone" / "frames.jsonl", max_ceiling)
        if max_ceiling and not args.skip_doom_alone
        else []
    )
    # Paired no-regression rule: batch load may not make the interactive tail
    # worse than its own baseline. This is reported next to the absolute SLO.
    control_p99_limit = (
        max(alone_p99 + 10, budget)
        if alone_p99 is not None and budget
        else None
    )
    control_max_limit = (
        max(alone_max, max_ceiling)
        if alone_max is not None and max_ceiling
        else None
    )
    control_ok = bool(
        p99 is not None
        and control_p99_limit is not None
        and p99 <= control_p99_limit
        and (max_ms is None or control_max_limit is None or max_ms <= control_max_limit)
    )
    admissions = admission_summary(contended_receipts)
    registered_leases = [
        lease.get("id")
        for lease in ((doom.get("interactive_lease") or {}).get("leases") or [])
    ]
    admission_ok = bool(
        admissions["rejected"] == 0
        and not admissions["starved_clients"]
        and admissions["admitted"] == items * 2 * clients
        and admissions["telemetry_ok"] > 0
        and len(registered_leases) >= 2
        and len(admissions["leases_seen"]) >= 1
    )
    ok = bool(pipeline_ok and slo_ok and admission_ok)
    summary = {
        "schema": "jotnar.workload.doom_gpu_contention/2",
        "ok": ok,
        "cuda": args.cuda,
        "hip": args.hip,
        "bundle": {
            "path": str(bundle),
            "sha256": built["sha256"],
            "items": items,
            "size": args.size,
        },
        "pipeline": {
            "reference_seconds": round(reference_s, 3),
            "reference_note": "single-node CUDA reference; correctness only",
            "baseline_seconds": round(baseline_s, 3),
            "baseline_note": "one warm CUDA+HIP pipeline alone",
            "clients": clients,
            "contended_seconds": round(contended_s, 3),
            "serial_equivalent_seconds": round(serial_equivalent, 3),
            "parallel_efficiency": (
                round(parallel_efficiency, 2) if parallel_efficiency else None
            ),
            "comparisons": len(comparisons),
            "matches": all(c.get("match") is True for c in comparisons),
            "placement": "gpu-a-* on CUDA, gpu-b-* on HIP",
        },
        "doom_alone": doom_alone,
        "doom": {
            "requested_frames": doom_frames,
            "fps": args.fps,
            "frame_budget_ms": budget,
            "latency_ms": latency,
            "frames_over_budget": frames_over_budget,
            "p50_over_budget": (
                latency.get("total_p50", 0) > budget if budget else None
            ),
            "p95_over_budget": (
                latency.get("total_p95", 0) > budget if budget else None
            ),
            "p99_over_budget": (
                p99 > budget if p99 is not None and budget else None
            ),
            "max_over_budget": (
                max_ms > budget if max_ms is not None and budget else None
            ),
            "real_time_at_target": doom.get("real_time_at_target"),
            "mode": doom.get("mode"),
            "failover": doom.get("failover"),
            "wall_ms": doom.get("wall_ms"),
            "interactive_lease": doom.get("interactive_lease"),
        },
        "slo": {
            "method": "nearest-rank percentile over the recorded frame loop",
            "budget_ms": budget,
            "p99_ms": p99,
            "max_ms": max_ms,
            "max_ceiling_ms": max_ceiling,
            "p99_ok": p99_ok,
            "max_ok": max_ok,
            "frames": len(totals),
            "frames_over_budget": frames_over_budget,
            "stalls_over_ceiling": stalls,
            "verdict": "pass" if slo_ok else "fail",
        },
        "control_slo": {
            "method": "paired: contended tail must not regress beyond the DOOM-alone baseline",
            "alone_p99_ms": alone_p99,
            "alone_max_ms": alone_max,
            "contended_p99_ms": p99,
            "contended_max_ms": max_ms,
            "p99_limit_ms": control_p99_limit,
            "max_limit_ms": control_max_limit,
            "alone_stalls": alone_stalls,
            "contended_stalls": stalls,
            "verdict": "pass" if control_ok else "fail",
        },
        "fleet": {
            "dir": str(fleet),
            "capacity_per_target": args.fleet_slots,
            "max_wait_ms": args.fleet_max_wait_ms,
            "registered_leases": registered_leases,
            "admissions": admissions,
            "admission_ok": admission_ok,
        },
        "verdict": {
            "pipeline_healthy": pipeline_ok,
            "slo_healthy": slo_ok,
            "control_healthy": control_ok,
            "admission_healthy": admission_ok,
            "contention_observed": bool(
                parallel_efficiency is not None and parallel_efficiency < 0.8
            )
            or not slo_ok,
            "controls": {
                "live_executor_telemetry": admissions["telemetry_ok"] > 0,
                "leases": len(registered_leases) >= 2
                and len(admissions["leases_seen"]) >= 1,
                "priority": admission_ok,
                "admission_control": admissions["admitted"] > 0,
            },
        },
        "evidence": {
            "reference_ledger": str(out / "reference"),
            "warmup_ledger": str(out / "warmup"),
            "baseline_ledger": str(out / "baseline"),
            "contended_ledgers": [
                str(out / f"contended-{client:02}") for client in range(clients)
            ],
            "fleet_dir": str(fleet),
            "doom_alone_run": str(out / "doom-alone" / "run.json"),
            "doom_run": str(doom_out / "run.json"),
            "doom_log": str(out / "doom.log"),
            "summary": str(out / "contention.json"),
        },
    }
    (out / "contention.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
