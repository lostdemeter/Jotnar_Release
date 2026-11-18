#!/usr/bin/env python3
"""Two-node DOOM checkpoint-handoff gate.

Builds the native runner, workload controller, and RV32IM DOOM demo, starts
two independent native executor processes, runs frame 1 on node A and frame 2
on node B, then compares both framebuffers with a single-node reference.

This is the first real acceptance test for the workload layer: it proves the
controller moves the RV32IM checkpoint over the network and that the resumed
frame is byte-identical to the reference.
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
JOTNAR = Path(
    os.environ.get(
        "JOTNAR_ROOT", str(Path.home() / "Documents" / "OpenCode" / "jotnar-main")
    )
)
WAD = Path(
    os.environ.get(
        "DOOM_WAD",
        str(Path.home() / "Documents" / "OpenCode" / "jotnar-assets" / "doom1.wad"),
    )
)
IMAGE = JOTNAR / "programs" / "doom" / "build" / "doom.bin"
WORK = Path(os.environ.get("DOOM_TWO_NODE_WORK", "/tmp/opencode/doom_two_node"))
TOKEN = "test"


def run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


def free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def wait_port(port: int, timeout: float = 20.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.5):
                return
        except OSError:
            time.sleep(0.05)
    raise RuntimeError(f"native runner on port {port} did not start")


def build(bins: list[str]) -> None:
    root_bins = [b for b in bins if b == "jotnar-demo-doom"]
    node_bins = [b for b in bins if b != "jotnar-demo-doom"]
    if root_bins:
        r = run(["cargo", "build", "--release", *[f"--bin={b}" for b in root_bins]], cwd=ROOT)
        if r.returncode != 0:
            print(r.stdout)
            print(r.stderr)
            raise SystemExit("build failed")
    if node_bins:
        r = run(
            ["cargo", "build", "--release", "-p", "jotnar-node-tools",
             *[f"--bin={b}" for b in node_bins]],
            cwd=ROOT,
        )
        if r.returncode != 0:
            print(r.stdout)
            print(r.stderr)
            raise SystemExit("build failed")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-build", action="store_true")
    ap.add_argument("--frames", type=int, default=2)
    ap.add_argument("--keep", action="store_true", help="keep the work directory")
    args = ap.parse_args()
    if args.frames < 2:
        print("--frames must be at least 2")
        return 2
    if not IMAGE.exists() or not WAD.exists():
        print(f"missing DOOM image or WAD\n  image: {IMAGE}\n  wad:   {WAD}")
        return 2
    if not args.no_build:
        build(["jotnar-demo-doom", "jotnar-native-runner", "jotnar-workload"])
    demo = ROOT / "target" / "release" / "jotnar-demo-doom"
    runner = ROOT / "target" / "release" / "jotnar-native-runner"
    controller = ROOT / "target" / "release" / "jotnar-workload"
    for p in (demo, runner, controller):
        if not p.exists():
            print(f"missing {p}; run without --no-build")
            return 2

    if WORK.exists() and not args.keep:
        shutil.rmtree(WORK)
    WORK.mkdir(parents=True, exist_ok=True)
    procs: list[subprocess.Popen] = []
    try:
        ports = [free_port(), free_port()]
        for i, port in enumerate(ports):
            node_work = WORK / f"runner{i + 1}"
            node_work.mkdir(parents=True, exist_ok=True)
            log = open(WORK / f"runner{i + 1}.log", "w")
            p = subprocess.Popen(
                [
                    str(runner),
                    "--bind", f"127.0.0.1:{port}",
                    "--token", TOKEN,
                    "--allow", str(demo),
                    "--work-root", str(node_work),
                    "--port-file", str(WORK / f"runner{i + 1}.port"),
                    "--max-frame", str(512 * 1024 * 1024),
                ],
                stdout=log,
                stderr=log,
            )
            procs.append(p)
        for port in ports:
            wait_port(port)

        ref1 = WORK / "reference" / "frame1"
        ref2 = WORK / "reference" / "frame2"
        r = run([
            str(demo), "--image", str(IMAGE), "--wad", str(WAD),
            "--frames", "1", "--out", str(ref1),
        ])
        if r.returncode != 0:
            print(r.stdout)
            print(r.stderr)
            return 1
        r = run([
            str(demo), "--image", str(IMAGE), "--wad", str(WAD),
            "--frames", str(args.frames), "--out", str(ref2),
        ])
        if r.returncode != 0:
            print(r.stdout)
            print(r.stderr)
            return 1
        print(f"reference: {ref2}.fb")

        bundle = WORK / "bundle"
        ledger = WORK / "ledger"
        r = run([
            str(controller), "bundle-doom",
            "--image", str(IMAGE), "--wad", str(WAD),
            "--out", str(bundle), "--frames", str(args.frames),
        ])
        if r.returncode != 0:
            print(r.stdout)
            print(r.stderr)
            return 1
        print(r.stdout.strip())

        cmd = [
            str(controller), "--token", TOKEN, "run",
            "--bundle", str(bundle),
            "--ledger", str(ledger),
            "--program", str(demo),
            "--executor", f"127.0.0.1:{ports[0]}",
            "--executor", f"127.0.0.1:{ports[1]}",
            "--place", f"frame1=127.0.0.1:{ports[0]}",
            "--place", f"frame{args.frames}=127.0.0.1:{ports[1]}",
        ]
        for frame, ref in ((1, ref1), (args.frames, ref2)):
            for ext in ("mem", "x", "pc", "cyc", "fb"):
                cmd.extend(["--compare", f"frame{frame}:frame{frame}.{ext}={ref}.{ext}"])
        r = run(cmd)
        print(r.stdout.strip())
        if r.returncode != 0:
            print(r.stderr)
            return 1

        receipt = json.loads((ledger / "receipt.json").read_text())
        ok = receipt.get("ok") is True
        for comp in receipt.get("comparisons", []):
            print(
                f"{comp['stage']}.{comp['output']}: "
                f"{'OK' if comp['match'] else 'FAIL'} {comp['sha256']}"
            )
        print("receipt:", ledger / "receipt.json")
        print("RESULT:", "ALL OK" if ok else "FAIL")
        return 0 if ok else 1
    finally:
        for p in procs:
            p.terminate()
        for p in procs:
            try:
                p.wait(timeout=5)
            except subprocess.TimeoutExpired:
                p.kill()


if __name__ == "__main__":
    raise SystemExit(main())
