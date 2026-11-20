#!/usr/bin/env python3
"""Video-wall gate.

Runs a deterministic 640x360 clip through the 2x2 tiled upscale wall on the
live CUDA + HIP fleet. A single-node CUDA run of the same tiled plan is the
reference; every composited 1280x720 frame must match byte-for-byte.

The wall is two pixels shy of a monolithic upscale by construction: each
320x180 tile carries a one-pixel halo, is upscaled 2x, and is cropped back.
Both the reference and the distributed run use the same tiler, so the gate is
proving placement/vendor agreement, not a comparison against a different
algorithm.

The first frame of a cold worker is a warm-up, not part of the measured SLO.
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import socket
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STREAM = ROOT / "target" / "release" / "jotnar-stream"
JOTNAR = ROOT / "target" / "release" / "jotnar"
LISTING = ROOT / "demos" / "doom-two-nodes" / "upscale-half.asm"
DEFAULT_CUDA = "192.168.1.116:9997"
DEFAULT_HIP = "192.168.1.195:9997"


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
        ["cargo", "build", "--release", "-p", "jotnar-node-tools", "--bin", "jotnar-stream"],
    ):
        r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
        if r.returncode != 0:
            print(r.stdout)
            print(r.stderr)
            raise SystemExit("build failed")


def emit_stage_descriptor(out: Path, width: int, height: int) -> Path:
    shape = out / "stage-shape.json"
    shape.write_text(
        json.dumps({"streams": {"X": {"kind": "F32", "shape": [height, width, 3]}}})
        + "\n"
    )
    work = out / "stage-build"
    work.mkdir(parents=True, exist_ok=True)
    descriptor = out / "wall-stage.json"
    r = run(
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
            "wall-stage",
            "--stage-out",
            str(descriptor),
            "--descriptor-only",
        ]
    )
    value = json.loads(descriptor.read_text())
    if not value.get("partitionable"):
        raise SystemExit("stage descriptor has no partitionable claim")
    return descriptor


def run(cmd: list[str], expect_ok: bool = True) -> subprocess.CompletedProcess:
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    if expect_ok and r.returncode != 0:
        print("command failed:", " ".join(cmd))
        print(r.stdout)
        print(r.stderr)
        raise SystemExit(r.returncode)
    return r


def make_clip(out: Path, frames: int, width: int, height: int) -> None:
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    for f in range(frames):
        data = bytearray(width * height * 3)
        for y in range(height):
            row = y * width * 3
            for x in range(width):
                i = row + x * 3
                data[i] = (x + f * 17) % 256
                data[i + 1] = (y + f * 9) % 256
                data[i + 2] = ((x + y) // 2 + f * 23) % 256
        (out / f"frame_{f:06}.rgb").write_bytes(data)


def describe(
    out: Path,
    clip: Path,
    frames: int,
    width: int,
    height: int,
    deadline_ms: int,
    placement: dict[str, int],
    stage_descriptor: Path,
) -> Path:
    descriptor = {
        "schema": "jotnar.stream/1",
        "name": "camera-wall",
        "clock": {"fps": round(1000 / deadline_ms), "deadline_ms": deadline_ms, "max_in_flight": 1},
        "source": {
            "kind": "frames",
            "dir": str(clip),
            "width": width,
            "height": height,
            "channels": 3,
            "format": "rgb24",
        },
        "stage": {
            "listing": str(LISTING),
            "descriptor": str(stage_descriptor),
            "targets": ["cuda", "hip"],
        },
        "tiles": {"rows": 2, "cols": 2, "halo": 1, "bezel": 0},
        "placement": placement,
        "sink": {"kind": "frames", "dir": str(out / "reference")},
        "merge": "tile-grid",
        "backpressure": "drop-oldest",
        "determinism": "exact",
        "receipt": {"kind": "every-frame", "dir": str(out / "reference-receipt")},
    }
    path = out / "wall.json"
    path.write_text(json.dumps(descriptor, indent=2) + "\n")
    return path


def stream(
    descriptor: Path,
    cuda: str,
    hip: str,
    token: str,
    clip: Path,
    sink: Path,
    receipt: Path,
    frames: int,
    *,
    warmup: int = 3,
    all_cuda: bool = False,
    reference: Path | None = None,
    fleet: Path | None = None,
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
        str(clip),
        "--out",
        str(sink),
        "--receipt",
        str(receipt),
        "--frames",
        str(frames),
        "--warmup-frames",
        str(warmup),
    ]
    if all_cuda:
        cmd.append("--all-cuda")
    if reference is not None:
        cmd += ["--reference", str(reference)]
    if fleet is not None:
        cmd += ["--fleet-dir", str(fleet)]
    result = run(cmd)
    return result, json.loads(result.stdout)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cuda", default=DEFAULT_CUDA)
    ap.add_argument("--hip", default=DEFAULT_HIP)
    ap.add_argument("--token", default="test")
    ap.add_argument("--frames", type=int, default=30)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=360)
    ap.add_argument("--deadline-ms", type=int, default=66)
    ap.add_argument("--clip", default=None, help="existing frames-dir; generated if omitted")
    ap.add_argument("--out", default="/tmp/opencode/wall-gate")
    ap.add_argument("--no-build", action="store_true")
    args = ap.parse_args()

    for addr in (args.cuda, args.hip):
        if not reachable(addr):
            print(f"GPU runner is not reachable at {addr}")
            return 2
    if not args.no_build:
        build()
    if not STREAM.exists() or not JOTNAR.exists() or not LISTING.exists():
        print(f"missing {STREAM}, {JOTNAR}, or {LISTING}; run without --no-build")
        return 2
    if args.frames < 2:
        print("--frames must be >= 2")
        return 2

    out = Path(args.out).absolute()
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    clip = Path(args.clip).absolute() if args.clip else out / "clip"
    warmup = 3
    if args.clip is None:
        make_clip(clip, args.frames + warmup, args.width, args.height)
    elif not clip.is_dir():
        print(f"clip {clip} is not a directory")
        return 2

    deadline_ms = args.deadline_ms
    stage_descriptor = emit_stage_descriptor(out, args.width, args.height)
    descriptor = describe(
        out,
        clip,
        args.frames,
        args.width,
        args.height,
        deadline_ms,
        {"cuda": 1, "hip": 3},
        stage_descriptor,
    )
    fleet = out / "fleet"
    fleet.mkdir()

    ref_dir = out / "reference"
    ref_receipt = out / "reference-receipt"
    _, ref_summary = stream(
        descriptor,
        args.cuda,
        args.hip,
        args.token,
        clip,
        ref_dir,
        ref_receipt,
        args.frames,
        all_cuda=True,
        fleet=fleet,
    )
    dist_dir = out / "distributed"
    dist_receipt = out / "distributed-receipt"
    _, dist_summary = stream(
        descriptor,
        args.cuda,
        args.hip,
        args.token,
        clip,
        dist_dir,
        dist_receipt,
        args.frames,
        reference=ref_dir,
        fleet=fleet,
    )

    # Failover: point the HIP endpoint at a closed port. HIP tiles must retry
    # on CUDA, the wall must keep producing exact frames, and the receipt must
    # say which target was replaced.
    fail_dir = out / "failover"
    fail_receipt = out / "failover-receipt"
    _, fail_summary = stream(
        descriptor,
        args.cuda,
        "127.0.0.1:1",
        args.token,
        clip,
        fail_dir,
        fail_receipt,
        2,
        reference=ref_dir,
    )
    fail_lines = [
        json.loads(line)
        for line in (fail_receipt / "frames.jsonl").read_text().splitlines()
        if line.strip()
    ]
    fail_exact = fail_summary.get("exactness") or {}
    failover_ok = bool(
        fail_summary.get("frames") == 2
        and fail_exact.get("matches") == 2
        and fail_exact.get("mismatches") == 0
        and fail_summary.get("failovers", 0) >= 6
        and all(len(line.get("tiles") or []) == 4 for line in fail_lines)
        and all(
            sum(
                1
                for tile in (line.get("tiles") or [])
                if tile.get("failover_from") == "hip"
            )
            >= 3
            for line in fail_lines
        )
    )

    receipt_lines = [
        json.loads(line)
        for line in (dist_receipt / "frames.jsonl").read_text().splitlines()
        if line.strip()
    ]
    latency = dist_summary.get("latency_ms", {})
    p99 = latency.get("p99", 0)
    max_ms = latency.get("max", 0)
    over = latency.get("over_budget", 0)
    exact = dist_summary.get("exactness") or {}
    tile_counts = {len(line.get("tiles") or []) for line in receipt_lines}
    targets = {
        tile.get("target")
        for line in receipt_lines
        for tile in (line.get("tiles") or [])
    }
    job_ids_ok = all(
        tile.get("job_id")
        for line in receipt_lines
        for tile in (line.get("tiles") or [])
    )
    frames_ok = (
        len(receipt_lines) == args.frames
        and exact.get("matches") == args.frames
        and exact.get("mismatches") == 0
    )
    wall_ok = all(
        (dist_dir / f"frame_{i:06}.rgb").is_file()
        and (ref_dir / f"frame_{i:06}.rgb").is_file()
        for i in range(args.frames)
    )
    p99_ok = p99 <= deadline_ms
    max_ok = max_ms <= deadline_ms * 2
    ok = bool(
        frames_ok
        and wall_ok
        and p99_ok
        and max_ok
        and over == 0
        and tile_counts == {4}
        and targets == {"cuda", "hip"}
        and job_ids_ok
        and dist_summary.get("descriptor_sha256")
        and (dist_summary.get("stage_descriptor") or {}).get("sha256")
        and failover_ok
        and ref_summary.get("frames") == args.frames
    )
    summary = {
        "schema": "jotnar.stream.wall_gate/1",
        "ok": ok,
        "cuda": args.cuda,
        "hip": args.hip,
        "geometry": {
            "source": [args.width, args.height],
            "wall": [args.width * 2, args.height * 2],
            "tiles": {"rows": 2, "cols": 2, "halo": 1},
            "placement": {"cuda": 1, "hip": 3},
            "deadline_ms": deadline_ms,
            "fps": round(1000 / deadline_ms),
            "clip_frames": args.frames,
        },
        "descriptor": {
            "path": str(descriptor),
            "sha256": dist_summary.get("descriptor_sha256"),
            "stage": dist_summary.get("stage_descriptor"),
            "stage_path": str(stage_descriptor),
        },
        "reference": {
            "frames": ref_summary.get("frames"),
            "latency_ms": ref_summary.get("latency_ms"),
            "note": "all four tiles on CUDA; correctness only",
        },
        "distributed": {
            "frames": dist_summary.get("frames"),
            "latency_ms": latency,
            "exactness": exact,
            "tile_counts": sorted(tile_counts),
            "targets": sorted(targets),
            "job_ids": job_ids_ok,
            "summary": dist_summary,
        },
        "failover": {
            "ok": failover_ok,
            "frames": fail_summary.get("frames"),
            "failovers": fail_summary.get("failovers"),
            "exactness": fail_exact,
            "latency_ms": fail_summary.get("latency_ms"),
            "note": "HIP endpoint is 127.0.0.1:1; HIP tiles retry on CUDA",
        },
        "checks": {
            "frames_and_exactness": frames_ok,
            "failover": failover_ok,
            "wall_files": wall_ok,
            "p99_ok": p99_ok,
            "max_ok": max_ok,
            "over_budget_zero": over == 0,
            "four_tiles": tile_counts == {4},
            "both_targets": targets == {"cuda", "hip"},
            "job_ids": job_ids_ok,
        },
        "evidence": {
            "clip": str(clip),
            "reference_dir": str(ref_dir),
            "distributed_dir": str(dist_dir),
            "distributed_receipt": str(dist_receipt),
            "fleet": str(fleet),
            "summary": str(out / "wall.json"),
        },
    }
    (out / "wall-gate.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
