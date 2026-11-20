#!/usr/bin/env python3
"""Live DOOM streaming gate (C1b).

Runs the existing coordinator once with a fixed even split to produce the
reference frames. Then runs `jotnar-doom-stream`: the RV32IM core is the live
source, the generic stream controller owns the 72/248 column split, GPU
placement, stitch, failover, leases, and per-frame receipt. Every measured
frame is compared byte-for-byte to the coordinator's corresponding frame.

This is the full-frame SLO gate: `work_ms` includes the core and the GPU path.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import socket
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
COORD = ROOT / "target" / "release" / "jotnar-doom-coord"
DOOM_STREAM = ROOT / "target" / "release" / "jotnar-doom-stream"
JOTNAR = ROOT / "target" / "release" / "jotnar"
LISTING = ROOT / "demos" / "doom-two-nodes" / "upscale-half.asm"
JOTNAR_ROOT = Path(
    os.environ.get(
        "JOTNAR_ROOT", str(Path.home() / "Documents" / "OpenCode" / "jotnar-main")
    )
)
IMAGE = JOTNAR_ROOT / "programs" / "doom" / "build" / "doom.bin"
WAD = Path(
    os.environ.get(
        "DOOM_WAD",
        str(Path.home() / "Documents" / "OpenCode" / "jotnar-assets" / "doom1.wad"),
    )
)
DEFAULT_CUDA = "192.168.1.116:9997"
DEFAULT_HIP = "192.168.1.195:9997"
WARMUP = 3


def reachable(addr: str) -> bool:
    host, _, port = addr.partition(":")
    try:
        with socket.create_connection((host, int(port)), timeout=2):
            return True
    except OSError:
        return False


def build() -> None:
    for cmd in (
        ["cargo", "build", "--release", "--bin", "jotnar"],
        [
            "cargo",
            "build",
            "--release",
            "-p",
            "jotnar-node-tools",
            "--bin",
            "jotnar-doom-coord",
            "--bin",
            "jotnar-doom-stream",
        ],
    ):
        r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
        if r.returncode != 0:
            print(r.stdout)
            print(r.stderr)
            raise SystemExit("build failed")


def run(cmd: list[str], expect_ok: bool = True) -> subprocess.CompletedProcess:
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    if expect_ok and r.returncode != 0:
        print("command failed:", " ".join(cmd))
        print(r.stdout)
        print(r.stderr)
        raise SystemExit(r.returncode)
    return r


def emit_stage_descriptor(out: Path) -> Path:
    shape = out / "stage-shape.json"
    shape.write_text(
        json.dumps({"streams": {"X": {"kind": "F32", "shape": [200, 320, 3]}}})
        + "\n"
    )
    work = out / "stage-build"
    work.mkdir(parents=True, exist_ok=True)
    descriptor = out / "doom-stage.json"
    run(
        [
            str(JOTNAR),
            "build",
            str(LISTING),
            "--payload",
            str(shape),
            "--shape-only",
            "--outputs",
            "OUT",
            "--target",
            "cuda",
            "--workdir",
            str(work),
            "--name",
            "doom-stage",
            "--stage-out",
            str(descriptor),
            "--descriptor-only",
        ]
    )
    if not json.loads(descriptor.read_text()).get("partitionable"):
        raise SystemExit("stage descriptor has no partitionable claim")
    return descriptor


def live(
    cuda: str,
    hip: str,
    token: str,
    frames: int,
    out: Path,
    receipt: Path,
    stage_descriptor: Path,
    *,
    left_cols: int = 72,
    warmup: int = WARMUP,
) -> dict:
    cmd = [
        str(DOOM_STREAM),
        "--image",
        str(IMAGE),
        "--wad",
        str(WAD),
        "--listing",
        str(LISTING),
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
        "--left-cols",
        str(left_cols),
        "--sink",
        "frames",
        "--out",
        str(out),
        "--receipt",
        str(receipt),
        "--warmup-frames",
        str(warmup),
    ]
    result = run(cmd)
    return json.loads(result.stdout)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cuda", default=DEFAULT_CUDA)
    ap.add_argument("--hip", default=DEFAULT_HIP)
    ap.add_argument("--token", default="test")
    ap.add_argument("--frames", type=int, default=20)
    ap.add_argument("--left-cols", type=int, default=72)
    ap.add_argument("--out", default="/tmp/opencode/doom-live-gate")
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
    for path in (COORD, DOOM_STREAM, JOTNAR, LISTING):
        if not path.exists():
            print(f"missing {path}; run without --no-build")
            return 2
    if args.frames < 2 or not (1 <= args.left_cols < 320):
        print("--frames must be >= 2 and --left-cols in 1..319")
        return 2

    out = Path(args.out).absolute()
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    coord = out / "coord"
    total = args.frames + WARMUP
    run(
        [
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
            str(total),
            "--fps",
            "15",
            "--no-auto-split",
            "--no-warmup",
            "--out",
            str(coord),
        ]
    )
    coord_summary = json.loads((coord / "run.json").read_text())
    stage_descriptor = emit_stage_descriptor(out)

    live_summary = live(
        args.cuda,
        args.hip,
        args.token,
        args.frames,
        out / "live",
        out / "live-receipt",
        stage_descriptor,
        left_cols=args.left_cols,
    )
    comparisons = []
    matches = 0
    for i in range(args.frames):
        got_path = out / "live" / f"frame_{i:06}.rgb"
        want_path = coord / "frames" / f"frame_{i + WARMUP:04}.rgb"
        got = got_path.read_bytes()
        want = want_path.read_bytes()
        match = got == want
        matches += match
        comparisons.append(
            {
                "index": i,
                "coord_frame": i + WARMUP,
                "bytes": len(got),
                "sha256": hashlib.sha256(got).hexdigest(),
                "reference_sha256": hashlib.sha256(want).hexdigest(),
                "match": match,
            }
        )

    fail_summary = live(
        args.cuda,
        "127.0.0.1:1",
        args.token,
        2,
        out / "failover",
        out / "failover-receipt",
        stage_descriptor,
        left_cols=args.left_cols,
    )
    fail_matches = 0
    for i in range(2):
        got = (out / "failover" / f"frame_{i:06}.rgb").read_bytes()
        want = (coord / "frames" / f"frame_{i + WARMUP:04}.rgb").read_bytes()
        fail_matches += got == want
    fail_lines = [
        json.loads(line)
        for line in (out / "failover-receipt" / "frames.jsonl").read_text().splitlines()
        if line.strip()
    ]

    latency = live_summary.get("latency_ms", {})
    p99 = latency.get("p99", 0)
    max_ms = latency.get("max", 0)
    over = latency.get("over_budget", 0)
    exact = matches == args.frames
    failover_ok = (
        fail_matches == 2
        and fail_summary.get("failovers", 0) >= 2
        and all(
            any(t.get("failover_from") == "hip" for t in (line.get("tiles") or []))
            for line in fail_lines
        )
    )
    ok = bool(
        exact
        and failover_ok
        and p99 <= 66
        and max_ms <= 132
        and over == 0
        and live_summary.get("frames") == args.frames
        and (live_summary.get("stage_descriptor") or {}).get("sha256")
    )
    summary = {
        "schema": "jotnar.stream.doom_live_gate/1",
        "ok": ok,
        "cuda": args.cuda,
        "hip": args.hip,
        "frames": args.frames,
        "warmup_frames": WARMUP,
        "split": {"left_cols": args.left_cols, "right_cols": 320 - args.left_cols},
        "coordinator": {
            "frames": coord_summary.get("frames"),
            "latency_ms": coord_summary.get("latency_ms"),
            "run": str(coord / "run.json"),
        },
        "live": {
            "frames": live_summary.get("frames"),
            "latency_ms": latency,
            "failovers": live_summary.get("failovers"),
            "source": live_summary.get("source"),
            "stage_descriptor": live_summary.get("stage_descriptor"),
            "receipt": str(out / "live-receipt" / "frames.jsonl"),
        },
        "comparisons": comparisons,
        "matches": matches,
        "failover": {
            "frames": 2,
            "matches": fail_matches,
            "failovers": fail_summary.get("failovers"),
            "latency_ms": fail_summary.get("latency_ms"),
            "receipt": str(out / "failover-receipt" / "frames.jsonl"),
        },
        "checks": {
            "byte_identical": exact,
            "failover_exact": fail_matches == 2,
            "failover_recorded": failover_ok,
            "p99_ok": p99 <= 66,
            "max_ok": max_ms <= 132,
            "over_budget_zero": over == 0,
            "stage_claim": bool(
                (live_summary.get("stage_descriptor") or {}).get("sha256")
            ),
        },
        "evidence": {
            "coordinator_frames": str(coord / "frames"),
            "live_frames": str(out / "live"),
            "live_receipt": str(out / "live-receipt" / "frames.jsonl"),
            "failover_receipt": str(out / "failover-receipt" / "frames.jsonl"),
            "stage_descriptor": str(stage_descriptor),
            "summary": str(out / "doom-live-gate.json"),
        },
    }
    (out / "doom-live-gate.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
