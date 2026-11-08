#!/usr/bin/env python3
"""Differential gate: Rust RV32IM native core vs the C native core.

Frame 1 must match the C native state and framebuffer. The Rust checkpoint
is then padded to the C core's 128 MB layout, resumed by the C core, and a
second frame is compared. If `--gpu-work` points at an overlay run, the Rust
frame 2 is also compared with the GPU checkpoint (`demo_g2.*`).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
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
NATIVE_C = JOTNAR / "programs" / "rv32_native.c"
WORK = Path(os.environ.get("DOOM_PARITY_WORK", "/tmp/opencode/doom_parity"))
RUST = ROOT / "target" / "release" / "jotnar-demo-doom"
MEM = 64 << 20
C_MEM = 128 << 20
FB = 0x03000000
FB_BYTES = 320 * 200 * 4


def check(tag: str, cond: bool, extra: str = "") -> bool:
    print(f"{tag}: {'OK' if cond else 'FAIL'} {extra}")
    return cond


def run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu-work", default="", help="overlay DOOM_WORK with demo_g2.*")
    ap.add_argument("--build", action="store_true", help="build the Rust binary")
    ap.add_argument("--bundle", default="", help="write a doom certificate bundle manifest here")
    args = ap.parse_args()
    if not IMAGE.exists() or not WAD.exists():
        print("missing DOOM image or WAD")
        return 2
    WORK.mkdir(parents=True, exist_ok=True)
    gcc = WORK / "rv32_native"
    r = run(["gcc", "-O2", "-o", str(gcc), str(NATIVE_C)])
    if r.returncode != 0:
        print(r.stderr)
        return 1
    if args.build:
        r = run(["cargo", "build", "--release", "--bin", "jotnar-demo-doom"], cwd=ROOT)
        if r.returncode != 0:
            print(r.stderr)
            return 1
    if not RUST.exists():
        print(f"missing {RUST}; run with --build")
        return 1

    c1 = WORK / "c1"
    r1 = WORK / "r1"
    r = run([str(gcc), "--dump", str(IMAGE), str(WAD), "1", str(c1)])
    if r.returncode != 0:
        print(r.stderr)
        return 1
    r = run([str(RUST), "--image", str(IMAGE), "--wad", str(WAD),
             "--frames", "1", "--out", str(r1)])
    if r.returncode != 0:
        print(r.stderr or r.stdout)
        return 1
    print(r.stdout.strip())

    cm = (WORK / "c1.mem").read_bytes()
    rm = (WORK / "r1.mem").read_bytes()
    ok = True
    ok &= check("frame1-mem64", cm[:MEM] == rm)
    ok &= check("frame1-x", (WORK / "c1.x").read_bytes() == (WORK / "r1.x").read_bytes())
    ok &= check("frame1-pc", (WORK / "c1.pc").read_bytes() == (WORK / "r1.pc").read_bytes())
    ok &= check("frame1-cyc", (WORK / "c1.cyc").read_bytes() == (WORK / "r1.cyc").read_bytes())
    ok &= check("frame1-fb", cm[FB:FB + FB_BYTES] == (WORK / "r1.fb").read_bytes())

    # Resume the Rust checkpoint through the C core. The C core still allocates
    # 128 MB; pad the modeled 64 MB with zeros.
    pad = WORK / "cfromrust"
    (pad.with_suffix(".mem")).write_bytes(rm + bytes(C_MEM - MEM))
    for ext in ("x", "pc", "cyc"):
        shutil.copyfile(WORK / f"r1.{ext}", WORK / f"cfromrust.{ext}")
    c2 = WORK / "c2"
    r = run([str(gcc), "--resume", str(pad), "2", str(c2)])
    if r.returncode != 0:
        print(r.stderr)
        return 1
    r2 = WORK / "r2"
    r = run([str(RUST), "--resume", str(r1), "--frames", "2", "--out", str(r2)])
    if r.returncode != 0:
        print(r.stderr or r.stdout)
        return 1
    print(r.stdout.strip())
    c2m = (WORK / "c2.mem").read_bytes()
    r2m = (WORK / "r2.mem").read_bytes()
    ok &= check("frame2-mem64", c2m[:MEM] == r2m)
    ok &= check("frame2-x", (WORK / "c2.x").read_bytes() == (WORK / "r2.x").read_bytes())
    ok &= check("frame2-pc", (WORK / "c2.pc").read_bytes() == (WORK / "r2.pc").read_bytes())
    ok &= check("frame2-cyc", (WORK / "c2.cyc").read_bytes() == (WORK / "r2.cyc").read_bytes())
    ok &= check("frame2-fb", c2m[FB:FB + FB_BYTES] == (WORK / "r2.fb").read_bytes())

    if args.gpu_work:
        g = Path(args.gpu_work)
        if (g / "demo_g2.mem").exists():
            gm = (g / "demo_g2.mem").read_bytes()
            ok &= check("gpu-frame2-mem64", gm == r2m)
            ok &= check("gpu-frame2-x", (g / "demo_g2.x").read_bytes() == (WORK / "r2.x").read_bytes())
            ok &= check("gpu-frame2-pc", (g / "demo_g2.pc").read_bytes() == (WORK / "r2.pc").read_bytes())
            ok &= check("gpu-frame2-fb", (g / "demo_g2.fb").read_bytes() == (WORK / "r2.fb").read_bytes())
        else:
            print(f"gpu-frame2: SKIP (no demo_g2.mem in {g})")

    if args.bundle:
        bdir = Path(args.bundle)
        bdir.mkdir(parents=True, exist_ok=True)
        manifest = {
            "image": str(IMAGE),
            "wad": str(WAD),
            "rust_frame1": str(r1),
            "rust_frame2": str(r2),
            "c_frame1": str(c1),
            "c_frame2": str(c2),
        }
        g = Path(args.gpu_work) if args.gpu_work else None
        if g and (g / "demo_g2.mem").exists():
            manifest["gpu_frame2"] = str(g / "demo_g2")
        (bdir / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
        print(f"bundle: {bdir / 'manifest.json'}")

    print("RESULT:", "ALL OK" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
