#!/usr/bin/env python3
"""Blackholed-target gate (D1c).

Runs the live DOOM stream through local proxies. Mid-run the HIP proxy keeps the
TCP connection open but forwards nothing: the client must detect the blackhole
with its bounded per-request deadline, fail over to CUDA on that frame, keep
producing exact frames, quarantine the dead target, and rejoin when the target
returns. The default 60 s client read timeout must never be the bound.

The test uses `--request-timeout-ms 500` and `--quarantine-frames 20`.
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
import doom_failover_gate as proxy_gate  # noqa: E402


def live_command(
    cuda: str,
    hip: str,
    token: str,
    frames: int,
    out: Path,
    receipt: Path,
    stage_descriptor: Path,
    request_timeout_ms: int,
    quarantine_frames: int,
) -> list[str]:
    return [
        str(live_gate.DOOM_STREAM),
        "--image", str(live_gate.IMAGE),
        "--wad", str(live_gate.WAD),
        "--listing", str(live_gate.LISTING),
        "--stage-descriptor", str(stage_descriptor),
        "--cuda", cuda,
        "--hip", hip,
        "--token", token,
        "--frames", str(frames),
        "--start-frame", "200",
        "--fps", "15",
        "--left-cols", "72",
        "--sink", "frames",
        "--out", str(out),
        "--receipt", str(receipt),
        "--warmup-frames", str(live_gate.WARMUP),
        "--request-timeout-ms", str(request_timeout_ms),
        "--quarantine-frames", str(quarantine_frames),
    ]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cuda", default=live_gate.DEFAULT_CUDA)
    ap.add_argument("--hip", default=live_gate.DEFAULT_HIP)
    ap.add_argument("--token", default="test")
    ap.add_argument("--frames", type=int, default=40)
    ap.add_argument("--request-timeout-ms", type=int, default=500)
    ap.add_argument("--quarantine-frames", type=int, default=20)
    ap.add_argument("--bound-ms", type=int, default=2000)
    ap.add_argument("--out", default="/tmp/opencode/doom-blackhole-gate")
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
    for path in (live_gate.COORD, live_gate.DOOM_STREAM, live_gate.JOTNAR):
        if not path.exists():
            print(f"missing {path}; run without --no-build")
            return 2
    if args.frames < 12:
        print("--frames must be >= 12 to stage the outage and recovery")
        return 2

    out = Path(args.out).absolute()
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    # Deterministic reference.
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

    cport = live_gate.free_port()
    hport = live_gate.free_port()
    cproxy = proxy_gate.Proxy(cport, live_gate.parse_addr(args.cuda))
    hproxy = proxy_gate.Proxy(hport, live_gate.parse_addr(args.hip))

    live_dir = out / "blackhole"
    live_receipt = out / "blackhole-receipt"
    log = open(out / "blackhole.log", "w")
    proc = subprocess.Popen(
        live_command(
            f"127.0.0.1:{cport}",
            f"127.0.0.1:{hport}",
            args.token,
            args.frames,
            live_dir,
            live_receipt,
            stage_descriptor,
            args.request_timeout_ms,
            args.quarantine_frames,
        ),
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=log,
        text=True,
    )
    lines_path = live_receipt / "frames.jsonl"
    deadline = time.time() + 240
    try:
        # Let the frame loop settle, then blackhole HIP.
        while time.time() < deadline and len(live_gate.receipt_lines(lines_path)) < 5:
            time.sleep(0.03)
        blackhole_at = len(live_gate.receipt_lines(lines_path))
        hproxy.blackhole()
        # Wait for the detection frame plus one failover frame.
        while time.time() < deadline and len(live_gate.receipt_lines(lines_path)) < blackhole_at + 2:
            time.sleep(0.03)
        # The target returns: replace the blackholed proxy on the same port.
        hproxy.stop()
        hproxy = proxy_gate.Proxy(hport, live_gate.parse_addr(args.hip))
        restored_at = len(live_gate.receipt_lines(lines_path))
        stdout, _ = proc.communicate(timeout=240)
    finally:
        hproxy.stop()
        cproxy.stop()
        log.close()
        if proc.poll() is None:
            proc.kill()
            proc.wait()
    if proc.returncode != 0:
        print("live stream failed")
        print((stdout or "")[-3000:])
        return 1

    summary = json.loads(stdout)
    lines = live_gate.receipt_lines(lines_path)
    comparisons = live_gate.compare_live(coord, live_dir, args.frames)
    matches = sum(1 for item in comparisons if item["match"])

    def has_hip_failover(line: dict) -> bool:
        return any(
            tile.get("failover_from") == "hip" for tile in (line.get("tiles") or [])
        )

    def hip_tile(line: dict) -> dict:
        return next(
            (tile for tile in (line.get("tiles") or []) if tile.get("col") == 1),
            {},
        )

    detection = next(
        (
            line
            for line in lines
            if has_hip_failover(line) and hip_tile(line).get("attempts", 1) > 1
        ),
        None,
    )
    quarantined = [
        line
        for line in lines
        if detection
        and line["index"] > detection["index"]
        and has_hip_failover(line)
        and hip_tile(line).get("attempts", 1) == 1
    ]
    recovered = [
        line for line in lines if "hip" in (line.get("recovered_targets") or [])
    ]
    rejoined = [
        line
        for line in lines
        if detection
        and line["index"] > detection["index"]
        and not has_hip_failover(line)
        and hip_tile(line).get("target") == "hip"
        and hip_tile(line).get("attempts", 1) == 1
    ]
    over_bound = [
        line for line in lines if (line.get("work_ms") or 0) > args.bound_ms
    ]
    exact = matches == args.frames and len(lines) == args.frames
    detection_ok = bool(
        detection and (detection.get("work_ms") or 0) <= args.bound_ms
    )
    quarantine_ok = len(quarantined) >= 3
    recovery_ok = bool(recovered and rejoined and summary.get("recoveries", 0) >= 1)
    ok = bool(exact and detection_ok and quarantine_ok and recovery_ok and not over_bound)

    # SLO view excluding the single detection frame.
    work = [
        line["work_ms"] for line in lines if line is not detection and line.get("work_ms")
    ]
    work_sorted = sorted(work)
    def pct(p: int) -> int:
        if not work_sorted:
            return 0
        rank = max(1, (len(work_sorted) * p + 99) // 100)
        return work_sorted[rank - 1]

    report = {
        "schema": "jotnar.stream.doom_blackhole_gate/1",
        "ok": ok,
        "cuda": args.cuda,
        "hip": args.hip,
        "frames": args.frames,
        "request_timeout_ms": args.request_timeout_ms,
        "quarantine_frames": args.quarantine_frames,
        "bound_ms": args.bound_ms,
        "blackhole_at_frame": blackhole_at,
        "restored_at_frame": restored_at,
        "matches": matches,
        "summary": summary,
        "detection": {
            "frame": detection.get("index") if detection else None,
            "work_ms": detection.get("work_ms") if detection else None,
            "attempts": hip_tile(detection).get("attempts") if detection else None,
            "bounded": detection_ok,
        },
        "quarantined_frames": [line["index"] for line in quarantined],
        "recovered_frames": [line["index"] for line in recovered],
        "rejoined_frames": [line["index"] for line in rejoined],
        "over_bound_frames": [
            {"index": line["index"], "work_ms": line["work_ms"]} for line in over_bound
        ],
        "slo_excluding_detection": {
            "p50": pct(50),
            "p95": pct(95),
            "p99": pct(99),
            "max": max(work) if work else 0,
        },
        "checks": {
            "byte_identical": exact,
            "detection_bounded": detection_ok,
            "quarantine_recorded": quarantine_ok,
            "recovery_and_rejoin": recovery_ok,
            "no_frame_over_bound": not over_bound,
        },
        "evidence": {
            "frames": str(live_dir),
            "receipt": str(lines_path),
            "summary": str(out / "doom-blackhole-gate.json"),
            "log": str(out / "blackhole.log"),
        },
    }
    (out / "doom-blackhole-gate.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
