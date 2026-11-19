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


def pipeline_command(
    controller: str,
    token: str,
    bundle: Path,
    ledger: Path,
    cuda: str,
    hip: str,
    items: int,
    reference: Path | None,
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

    items = max(1, args.items)
    clients = max(1, args.clients)
    out = Path(args.out)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    bundle = out / "bundle"
    built = run(
        [
            str(CONTROLLER),
            "bundle-gpu",
            "--listing",
            str(Path(args.listing)),
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
    doom_out = out / "doom"
    doom_log = open(out / "doom.log", "w")
    doom_cmd = [
        str(COORD),
        "--image",
        str(IMAGE),
        "--wad",
        str(WAD),
        "--cuda",
        args.cuda,
        "--hip",
        args.hip,
        "--token",
        args.token,
        "--frames",
        str(doom_frames),
        "--fps",
        str(args.fps),
        "--out",
        str(doom_out),
    ]
    print("coordinator:", " ".join(doom_cmd))
    proc = subprocess.Popen(doom_cmd, cwd=ROOT, stdout=doom_log, stderr=doom_log)
    try:
        wait_for_frame_loop(proc, doom_out / "frames.jsonl", timeout=120)
        time.sleep(0.2)
        clients_procs = []
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
        contended_receipts = []
        for client, client_proc in clients_procs:
            stdout, stderr = client_proc.communicate()
            if client_proc.returncode != 0:
                print(f"pipeline client {client} failed")
                print(stdout)
                print(stderr)
                raise SystemExit(client_proc.returncode)
            contended_receipts.append(json.loads(stdout))
        contended_s = time.time() - t0
        proc.wait(timeout=300)
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
    doom_ok = doom.get("real_time_at_target") is True
    max_over_budget = (
        latency.get("total_max", 0) > budget if budget else None
    )
    summary = {
        "schema": "jotnar.workload.doom_gpu_contention/1",
        "ok": pipeline_ok,
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
            "reference_note": "cold CUDA reference; correctness only",
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
        "doom": {
            "requested_frames": doom_frames,
            "fps": args.fps,
            "frame_budget_ms": budget,
            "latency_ms": latency,
            "p50_over_budget": (
                latency.get("total_p50", 0) > budget if budget else None
            ),
            "p95_over_budget": (
                latency.get("total_p95", 0) > budget if budget else None
            ),
            "max_over_budget": max_over_budget,
            "real_time_at_target": doom.get("real_time_at_target"),
            "mode": doom.get("mode"),
            "failover": doom.get("failover"),
            "wall_ms": doom.get("wall_ms"),
        },
        "verdict": {
            "pipeline_healthy": pipeline_ok,
            "doom_healthy": doom_ok,
            "doom_tail_healthy": not max_over_budget
            if max_over_budget is not None
            else None,
            "contention_observed": bool(
                parallel_efficiency is not None and parallel_efficiency < 0.8
            )
            or not doom_ok
            or bool(max_over_budget),
            "missing_controls": [
                term
                for term, present in (
                    ("live executor telemetry", False),
                    ("leases", False),
                    ("priority", False),
                    ("admission control", False),
                )
                if not present
            ],
        },
        "evidence": {
            "reference_ledger": str(out / "reference"),
            "warmup_ledger": str(out / "warmup"),
            "baseline_ledger": str(out / "baseline"),
            "contended_ledgers": [
                str(out / f"contended-{client:02}") for client in range(clients)
            ],
            "doom_run": str(doom_out / "run.json"),
            "doom_log": str(out / "doom.log"),
            "summary": str(out / "contention.json"),
        },
    }
    (out / "contention.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0 if pipeline_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
