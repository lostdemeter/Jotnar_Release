#!/usr/bin/env python3
"""Batch GPU pipeline gate.

Builds a deterministic payload, runs the CUDA -> HIP chain through
`jotnar-workload`, and compares every output with a single-node CUDA reference.
The controller decodes the CUDA stage's binary F32 result into the HIP stage's
JSON payload; the two stages are represented in the bundle, not in a DOOM
binary.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CONTROLLER = ROOT / "target" / "release" / "jotnar-workload"
LISTING = ROOT / "demos" / "gpu-batch" / "double.asm"
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cuda", default=DEFAULT_CUDA)
    ap.add_argument("--hip", default=DEFAULT_HIP)
    ap.add_argument("--token", default="test")
    ap.add_argument("--listing", default=str(LISTING))
    ap.add_argument("--size", default="64x64x3", help="generated HxWxC payload")
    ap.add_argument("--items", type=int, default=8)
    ap.add_argument("--out", default="/tmp/opencode/gpu_batch_gate")
    ap.add_argument("--no-build", action="store_true")
    args = ap.parse_args()

    if not reachable(args.cuda):
        print(f"CUDA runner is not reachable at {args.cuda}")
        return 2
    if not reachable(args.hip):
        print(f"HIP runner is not reachable at {args.hip}")
        return 2
    if not args.no_build:
        build()
    if not CONTROLLER.exists():
        print(f"missing {CONTROLLER}; run without --no-build")
        return 2

    if args.items <= 0:
        print("--items must be >= 1")
        return 2
    items = args.items
    out = Path(args.out).absolute()
    listing = Path(args.listing).absolute()
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    bundle = out / "bundle"
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
            "gpu-batch-gate",
        ]
    )

    names = stage_names(items)
    base = [
        str(CONTROLLER),
        "--token",
        args.token,
        "run",
        "--bundle",
        str(bundle),
        "--gpu-executor",
        f"cuda={args.cuda}",
        "--gpu-executor",
        f"hip={args.hip}",
    ]

    reference_cmd = list(base) + ["--ledger", str(out / "reference")]
    for name in names:
        reference_cmd += ["--place", f"{name}={args.cuda}"]
    t0 = time.time()
    run(reference_cmd)
    reference_s = time.time() - t0

    distributed_cmd = list(base) + ["--ledger", str(out / "distributed")]
    for i in range(items):
        distributed_cmd += [
            "--place",
            f"gpu-a-{i:02}={args.cuda}",
            "--place",
            f"gpu-b-{i:02}={args.hip}",
            "--compare",
            f"gpu-b-{i:02}:OUT={out}/reference/stages/gpu-b-{i:02}/OUT",
        ]
    t0 = time.time()
    run(distributed_cmd)
    distributed_s = time.time() - t0

    reference = json.loads((out / "reference" / "receipt.json").read_text())
    receipt = json.loads((out / "distributed" / "receipt.json").read_text())
    comparisons = receipt.get("comparisons", [])
    ok = (
        reference.get("ok") is True
        and receipt.get("ok") is True
        and len(comparisons) == items
        and all(c.get("match") is True for c in comparisons)
    )
    summary = {
        "schema": "jotnar.workload.gpu_batch_gate/1",
        "ok": ok,
        "cuda": args.cuda,
        "hip": args.hip,
        "bundle": {
            "path": str(bundle),
            "sha256": built["sha256"],
            "items": items,
            "size": args.size,
            "stages": len(names),
        },
        "reference": {
            "ledger": str(out / "reference"),
            "seconds": round(reference_s, 3),
            "placement": "all stages on CUDA",
        },
        "distributed": {
            "ledger": str(out / "distributed"),
            "seconds": round(distributed_s, 3),
            "placement": "gpu-a-* on CUDA, gpu-b-* on HIP",
        },
        "comparisons": comparisons,
        "speedup": round(reference_s / distributed_s, 2) if distributed_s else None,
        "evidence": {
            "reference_receipt": str(out / "reference" / "receipt.json"),
            "distributed_receipt": str(out / "distributed" / "receipt.json"),
            "summary": str(out / "gate.json"),
        },
    }
    (out / "gate.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
