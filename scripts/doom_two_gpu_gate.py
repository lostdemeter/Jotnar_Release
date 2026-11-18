#!/usr/bin/env python3
"""Two-node DOOM GIF gate.

Node A runs the RV32IM core. Nodes B and C run the same Jotnar upscale program
on the CUDA and HIP GPUs, one half of the frame each. The coordinator records
raw frames and per-frame latency; this script annotates them and writes a GIF.

The per-frame budget is printed explicitly. A run is "real time at target"
only when the p95 total is within 1000/fps ms.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import socket
import subprocess
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
COORD = ROOT / "target" / "release" / "jotnar-doom-coord"
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
DEFAULT_OUT = Path("/tmp/opencode/doom_two_nodes")
DEFAULT_GIF = ROOT / "showcase" / "doom" / "doom_two_nodes.gif"
W, H = 640, 400
TOP, BOT = 30, 24


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    path = Path("/usr/share/fonts/truetype/dejavu") / name
    if path.exists():
        return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


def reachable(addr: str) -> bool:
    host, _, port = addr.partition(":")
    try:
        with socket.create_connection((host, int(port)), timeout=2):
            return True
    except OSError:
        return False


def build() -> None:
    r = subprocess.run(
        ["cargo", "build", "--release", "-p", "jotnar-node-tools",
         "--bin", "jotnar-doom-coord"],
        cwd=ROOT, capture_output=True, text=True,
    )
    if r.returncode != 0:
        print(r.stdout)
        print(r.stderr)
        raise SystemExit("build failed")


def annotate(im: Image.Image, meta: dict, run: dict) -> Image.Image:
    canvas = Image.new("RGB", (W, TOP + H + BOT), (12, 12, 12))
    canvas.paste(im, (0, TOP))
    d = ImageDraw.Draw(canvas)
    f13 = font(13)
    f11 = font(11)
    f11b = font(11, bold=True)
    split = run.get("split", {"left_cols": 160, "right_cols": 160})
    left_cols = int(split["left_cols"])
    right_cols = int(split["right_cols"])
    left_pct = round(100 * left_cols / max(1, left_cols + right_cols))
    right_pct = 100 - left_pct
    d.text(
        (8, 7),
        f"Jotnar DOOM  |  node A RV32IM  ->  CUDA {left_pct}% + HIP {right_pct}%",
        fill=(232, 232, 232), font=f13,
    )
    label = f"frame {meta['frame']}"
    tw = d.textlength(label, font=f13)
    d.text((W - tw - 8, 7), label, fill=(232, 232, 232), font=f13)

    # Show the measured split and name the two contributing nodes.
    split_x = max(1, min(W - 1, left_cols * 2))
    d.line([(split_x, TOP), (split_x, TOP + H)], fill=(0, 190, 240), width=1)
    d.rectangle([(6, TOP + 6), (92, TOP + 22)], fill=(0, 0, 0))
    d.text((10, TOP + 7), f"CUDA {left_cols} cols", fill=(0, 220, 255), font=f11b)
    d.rectangle([(W - 92, TOP + 6), (W - 6, TOP + 22)], fill=(0, 0, 0))
    d.text((W - 88, TOP + 7), f"HIP {right_cols} cols", fill=(255, 210, 90), font=f11b)

    y = TOP + H + 5
    d.text((8, y), f"CUDA {meta['cuda_ms']} ms", fill=(0, 220, 255), font=f11)
    total = f"total {meta['total_ms']} ms  ({run['frame_budget_ms']} ms budget)"
    tw = d.textlength(total, font=f11)
    d.text(((W - tw) / 2, y), total, fill=(220, 220, 220), font=f11)
    right = f"HIP {meta['hip_ms']} ms"
    tw = d.textlength(right, font=f11)
    d.text((W - tw - 8, y), right, fill=(255, 210, 90), font=f11)
    return canvas


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cuda", default=DEFAULT_CUDA)
    ap.add_argument("--hip", default=DEFAULT_HIP)
    ap.add_argument("--token", default="test")
    ap.add_argument("--frames", type=int, default=60)
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--start-frame", type=int, default=200)
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--gif", default=str(DEFAULT_GIF))
    ap.add_argument("--no-auto-split", action="store_true")
    ap.add_argument("--no-build", action="store_true")
    args = ap.parse_args()

    if not IMAGE.exists() or not WAD.exists():
        print(f"missing DOOM image or WAD\n  image: {IMAGE}\n  wad:   {WAD}")
        return 2
    if not reachable(args.cuda):
        print(f"CUDA runner is not reachable at {args.cuda}")
        return 2
    if not reachable(args.hip):
        print(f"HIP runner is not reachable at {args.hip}")
        return 2
    if not args.no_build:
        build()
    if not COORD.exists():
        print(f"missing {COORD}; run without --no-build")
        return 2

    out = Path(args.out)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    log = out / "coordinator.log"
    cmd = [
        str(COORD),
        "--image", str(IMAGE),
        "--wad", str(WAD),
        "--cuda", args.cuda,
        "--hip", args.hip,
        "--token", args.token,
        "--start-frame", str(args.start_frame),
        "--frames", str(args.frames),
        "--fps", str(args.fps),
        "--out", str(out),
    ]
    if args.no_auto_split:
        cmd.append("--no-auto-split")
    print("coordinator:", " ".join(cmd))
    with open(log, "w") as f:
        r = subprocess.run(cmd, cwd=ROOT, stdout=subprocess.PIPE,
                           stderr=f, text=True)
    if r.returncode != 0:
        print((out / "coordinator.log").read_text())
        return 1
    run = json.loads((out / "run.json").read_text())
    metas = [
        json.loads(line)
        for line in (out / "frames.jsonl").read_text().splitlines()
        if line.strip()
    ]
    frames = []
    for meta in metas:
        raw = (out / meta["file"]).read_bytes()
        im = Image.frombytes("RGB", (W, H), raw)
        frames.append(annotate(im, meta, run))

    gif = Path(args.gif)
    gif.parent.mkdir(parents=True, exist_ok=True)
    duration = max(10, round(1000 / max(1, args.fps)))
    frames[0].save(
        gif,
        save_all=True,
        append_images=frames[1:],
        duration=duration,
        loop=0,
        optimize=True,
        disposal=2,
    )
    poster = out / "poster.png"
    frames[0].save(poster)
    # A compact contact sheet is useful in the receipt.
    picks = sorted(set([0, len(frames) // 5, 2 * len(frames) // 5,
                        3 * len(frames) // 5, 4 * len(frames) // 5,
                        len(frames) - 1]))
    cols, rows = 3, 2
    sheet = Image.new("RGB", (W * cols, (TOP + H + BOT) * rows), (12, 12, 12))
    for k, idx in enumerate(picks[: cols * rows]):
        sheet.paste(frames[idx], ((k % cols) * W, (k // cols) * (TOP + H + BOT)))
    sheet.save(out / "contact.png")

    summary = {
        "gif": str(gif),
        "gif_bytes": gif.stat().st_size,
        "poster": str(poster),
        "contact": str(out / "contact.png"),
        "run": run,
        "latency_ms": run["latency_ms"],
        "real_time_at_target": run["real_time_at_target"],
        "frames": len(frames),
        "fps": args.fps,
    }
    (out / "gif.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
