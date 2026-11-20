#!/usr/bin/env python3
"""DOOM streaming equivalence gate (C1).

Runs the existing DOOM coordinator with a fixed even split and records both its
source framebuffer (320x200 RGB) and its stitched frames (640x400 RGB). Then it
replays those source frames through the generic `jotnar.stream/1` column-split
path on the same CUDA + HIP fleet and compares every stitched frame
byte-for-byte.

This proves the generic controller reproduces DOOM's GPU split/stitch path.
The coordinator remains the source/sink adapter in this slice; a live RV32
source adapter is the next C1 step.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import socket
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
COORD = ROOT / "target" / "release" / "jotnar-doom-coord"
STREAM = ROOT / "target" / "release" / "jotnar-stream"
JOTNAR = ROOT / "target" / "release" / "jotnar"
LISTING = ROOT / "demos" / "doom-two-nodes" / "upscale-half.asm"
JOTNAR_ROOT = Path(
    __import__("os").environ.get(
        "JOTNAR_ROOT", str(Path.home() / "Documents" / "OpenCode" / "jotnar-main")
    )
)
IMAGE = JOTNAR_ROOT / "programs" / "doom" / "build" / "doom.bin"
WAD = Path(
    __import__("os").environ.get(
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
            "jotnar-stream",
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


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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
    value = json.loads(descriptor.read_text())
    if not value.get("partitionable"):
        raise SystemExit("stage descriptor has no partitionable claim")
    return descriptor


def describe(out: Path, source: Path, sink: Path, left_cols: int, stage_descriptor: Path) -> Path:
    descriptor = {
        "schema": "jotnar.stream/1",
        "name": "doom-columns",
        "clock": {"fps": 15, "deadline_ms": 66, "max_in_flight": 1},
        "source": {
            "kind": "frames",
            "dir": str(source),
            "width": 320,
            "height": 200,
            "channels": 3,
            "format": "rgb24",
        },
        "stage": {
            "listing": str(LISTING),
            "descriptor": str(stage_descriptor),
            "targets": ["cuda", "hip"],
        },
        "tiles": {"rows": 1, "cols": 2, "halo": 1, "bezel": 0},
        "split": {"left_cols": left_cols, "right_cols": 320 - left_cols, "overlap": 1},
        "placement": {"cuda": 1, "hip": 1},
        "sink": {"kind": "frames", "dir": str(sink)},
        "merge": "columns",
        "backpressure": "drop-oldest",
        "determinism": "exact",
        "receipt": {"kind": "every-frame", "dir": str(out / f"receipt-{sink.name}")},
    }
    path = out / f"stream-{sink.name}.json"
    path.write_text(json.dumps(descriptor, indent=2) + "\n")
    return path


def stream(
    descriptor: Path,
    cuda: str,
    hip: str,
    token: str,
    source: Path,
    sink: Path,
    frames: int,
    *,
    warmup: int = WARMUP,
) -> tuple[subprocess.CompletedProcess, dict]:
    cmd = [
        str(STREAM),
        "--descriptor",
        str(descriptor),
        "--cuda",
        cuda,
        "--hip",
        hip,
        "--token",
        token,
        "--clip",
        str(source),
        "--out",
        str(sink),
        "--frames",
        str(frames),
        "--warmup-frames",
        str(warmup),
    ]
    result = run(cmd)
    return result, json.loads(result.stdout)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cuda", default=DEFAULT_CUDA)
    ap.add_argument("--hip", default=DEFAULT_HIP)
    ap.add_argument("--token", default="test")
    ap.add_argument("--frames", type=int, default=20)
    ap.add_argument("--left-cols", type=int, default=72)
    ap.add_argument("--out", default="/tmp/opencode/doom-stream-gate")
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
    for path in (COORD, STREAM, JOTNAR, LISTING):
        if not path.exists():
            print(f"missing {path}; run without --no-build")
            return 2
    if args.frames < 1 or not (1 <= args.left_cols < 320):
        print("--frames must be >= 1 and --left-cols in 1..319")
        return 2

    out = Path(args.out).absolute()
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    source = out / "source"
    coord = out / "coord"
    source.mkdir()

    total = args.frames + WARMUP
    coord_cmd = [
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
        "--record-source",
        str(source),
        "--out",
        str(coord),
    ]
    run(coord_cmd)
    coord_summary = json.loads((coord / "run.json").read_text())

    stage_descriptor = emit_stage_descriptor(out)
    descriptor = describe(out, source, out / "stream", args.left_cols, stage_descriptor)
    _, stream_summary = stream(
        descriptor,
        args.cuda,
        args.hip,
        args.token,
        source,
        out / "stream",
        args.frames,
    )

    comparisons = []
    matches = 0
    for i in range(args.frames):
        got_path = out / "stream" / f"frame_{i:06}.rgb"
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

    fail_descriptor = describe(
        out, source, out / "failover", args.left_cols, stage_descriptor
    )
    _, fail_summary = stream(
        fail_descriptor,
        args.cuda,
        "127.0.0.1:1",
        args.token,
        source,
        out / "failover",
        2,
    )
    fail_matches = 0
    fail_lines = [
        json.loads(line)
        for line in (out / "receipt-failover" / "frames.jsonl").read_text().splitlines()
        if line.strip()
    ]
    for i in range(2):
        got = (out / "failover" / f"frame_{i:06}.rgb").read_bytes()
        want = (coord / "frames" / f"frame_{i + WARMUP:04}.rgb").read_bytes()
        fail_matches += got == want

    latency = stream_summary.get("latency_ms", {})
    p99 = latency.get("p99", 0)
    max_ms = latency.get("max", 0)
    over = latency.get("over_budget", 0)
    exact = matches == args.frames
    core_p50 = (coord_summary.get("latency_ms") or {}).get("core_p50", 0)
    projected = core_p50 + p99
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
        and stream_summary.get("frames") == args.frames
        and (stream_summary.get("stage_descriptor") or {}).get("sha256")
    )
    summary = {
        "schema": "jotnar.stream.doom_gate/1",
        "ok": ok,
        "cuda": args.cuda,
        "hip": args.hip,
        "clip_frames": args.frames,
        "warmup_frames": WARMUP,
        "split": {"left_cols": args.left_cols, "right_cols": 320 - args.left_cols},
        "coordinator": {
            "frames": coord_summary.get("frames"),
            "latency_ms": coord_summary.get("latency_ms"),
            "mode": coord_summary.get("mode"),
            "split": coord_summary.get("split"),
            "run": str(coord / "run.json"),
        },
        "stream": {
            "frames": stream_summary.get("frames"),
            "latency_ms": latency,
            "failovers": stream_summary.get("failovers"),
            "stage_descriptor": stream_summary.get("stage_descriptor"),
            "receipt": stream_summary.get("receipt"),
        },
        "comparisons": comparisons,
        "matches": matches,
        "projected_full_frame_ms": {
            "note": "core_p50 from the coordinator plus the stream GPU p99; the live RV32 source adapter is not wired yet",
            "core_p50": core_p50,
            "gpu_path_p99": p99,
            "total": projected,
        },
        "failover": {
            "frames": 2,
            "matches": fail_matches,
            "failovers": fail_summary.get("failovers"),
            "latency_ms": fail_summary.get("latency_ms"),
            "receipt": str(out / "receipt-failover" / "frames.jsonl"),
        },
        "checks": {
            "byte_identical": exact,
            "failover_exact": fail_matches == 2,
            "failover_recorded": failover_ok,
            "p99_ok": p99 <= 66,
            "max_ok": max_ms <= 132,
            "over_budget_zero": over == 0,
            "projected_full_frame_ok": projected <= 66,
            "stage_claim": bool(
                (stream_summary.get("stage_descriptor") or {}).get("sha256")
            ),
        },
        "evidence": {
            "source": str(source),
            "coordinator_frames": str(coord / "frames"),
            "stream_frames": str(out / "stream"),
            "stream_receipt": str(out / "receipt-stream" / "frames.jsonl"),
            "stage_descriptor": str(stage_descriptor),
            "summary": str(out / "doom-stream-gate.json"),
        },
    }
    (out / "doom-stream-gate.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
