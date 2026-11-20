#!/usr/bin/env python3
"""Live DOOM streaming gate (D1a).

Runs the existing coordinator once with a fixed even split to produce the
reference frames. Then runs `jotnar-doom-stream`: the RV32IM core is the live
source, the generic stream controller owns the column split (including
calibration), GPU placement, stitch, failover, leases, and per-frame receipt.
Every measured frame is compared byte-for-byte to the coordinator's
corresponding frame.

Three legs:
  * the full-frame SLO run (100 frames by default), with adaptive split;
  * a HUD run through a frames-dir HUD sink, checked headlessly;
  * a mid-run HIP outage through local proxies, then restore, to prove
    failover and rejoin.

This is the full-frame SLO gate: `work_ms` includes the core and the GPU path.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import select
import shutil
import socket
import subprocess
import sys
import threading
import time
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
FB_W = 320
FB_H = 200
OUT_W = FB_W * 2
OUT_H = FB_H * 2
HUD_TOP = 44
HUD_BOTTOM = 22


def reachable(addr: str) -> bool:
    host, _, port = addr.partition(":")
    try:
        with socket.create_connection((host, int(port)), timeout=2):
            return True
    except OSError:
        return False


def free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def parse_addr(addr: str) -> tuple[str, int]:
    host, _, port = addr.partition(":")
    return host, int(port)


class Proxy:
    """One TCP proxy. `stop()` closes the listener and all live connections so
    the controller sees a real outage and fails over; a new Proxy on the same
    port restores reachability."""

    def __init__(self, listen_port: int, target: tuple[str, int]):
        self.target = target
        self.stop_event = threading.Event()
        self.lock = threading.Lock()
        self.pairs: list[tuple[socket.socket, socket.socket]] = []
        self.listener = socket.socket()
        self.listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.listener.bind(("127.0.0.1", listen_port))
        self.listener.listen(32)
        self.listener.setblocking(False)
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def _loop(self) -> None:
        while not self.stop_event.is_set():
            with self.lock:
                socks = [self.listener]
                for client, remote in self.pairs:
                    socks.extend((client, remote))
            try:
                readable, _, _ = select.select(socks, [], [], 0.1)
            except (OSError, ValueError):
                break
            for sock in readable:
                if sock is self.listener:
                    try:
                        client, _ = self.listener.accept()
                    except OSError:
                        continue
                    try:
                        remote = socket.create_connection(self.target, timeout=3)
                    except OSError:
                        client.close()
                        continue
                    with self.lock:
                        self.pairs.append((client, remote))
                    continue
                with self.lock:
                    pair = next(
                        ((c, r) for c, r in self.pairs if sock in (c, r)), None
                    )
                if pair is None:
                    continue
                client, remote = pair
                peer = remote if sock is client else client
                try:
                    data = sock.recv(65536)
                except OSError:
                    data = b""
                if not data:
                    self._close_pair(client, remote)
                    continue
                try:
                    peer.sendall(data)
                except OSError:
                    self._close_pair(client, remote)
        self._close_all()

    def _close_pair(self, client: socket.socket, remote: socket.socket) -> None:
        with self.lock:
            if (client, remote) in self.pairs:
                self.pairs.remove((client, remote))
        for sock in (client, remote):
            try:
                sock.close()
            except OSError:
                pass

    def _close_all(self) -> None:
        with self.lock:
            pairs = list(self.pairs)
            self.pairs.clear()
        for client, remote in pairs:
            for sock in (client, remote):
                try:
                    sock.close()
                except OSError:
                    pass

    def stop(self) -> None:
        self.stop_event.set()
        try:
            self.listener.close()
        except OSError:
            pass
        self._close_all()
        self.thread.join(timeout=2)


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
        json.dumps({"streams": {"X": {"kind": "F32", "shape": [FB_H, FB_W, 3]}}})
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


def live_command(
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
    fps: int = 15,
    hud: bool = False,
) -> list[str]:
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
        "--fps",
        str(fps),
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
    if hud:
        cmd.append("--hud")
    return cmd


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
    hud: bool = False,
) -> dict:
    result = run(
        live_command(
            cuda,
            hip,
            token,
            frames,
            out,
            receipt,
            stage_descriptor,
            left_cols=left_cols,
            warmup=warmup,
            hud=hud,
        )
    )
    return json.loads(result.stdout)


def receipt_lines(path: Path) -> list[dict]:
    try:
        text = path.read_text()
    except FileNotFoundError:
        return []
    lines = []
    for line in text.splitlines():
        if not line.strip():
            continue
        try:
            lines.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return lines


def reference_frame(coord: Path, measured: int) -> bytes:
    return (coord / "frames" / f"frame_{measured + WARMUP:04}.rgb").read_bytes()


def compare_live(coord: Path, live_dir: Path, frames: int) -> list[dict]:
    comparisons = []
    for i in range(frames):
        got = (live_dir / f"frame_{i:06}.rgb").read_bytes()
        want = reference_frame(coord, i)
        comparisons.append(
            {
                "index": i,
                "coord_frame": i + WARMUP,
                "bytes": len(got),
                "sha256": hashlib.sha256(got).hexdigest(),
                "reference_sha256": hashlib.sha256(want).hexdigest(),
                "match": got == want,
            }
        )
    return comparisons


def hud_leg(
    cuda: str,
    hip: str,
    token: str,
    frames: int,
    out: Path,
    stage_descriptor: Path,
    plain_dir: Path,
) -> dict:
    """Render the HUD through the frames sink and verify it headlessly."""
    frames = max(1, min(3, frames))
    hud_dir = out / "hud"
    hud_receipt = out / "hud-receipt"
    summary = live(
        cuda,
        hip,
        token,
        frames,
        hud_dir,
        hud_receipt,
        stage_descriptor,
        hud=True,
    )
    expected_bytes = OUT_W * (OUT_H + HUD_TOP + HUD_BOTTOM) * 3
    checks = []
    ok = True
    for i in range(frames):
        hud = (hud_dir / f"frame_{i:06}.rgb").read_bytes()
        plain = (plain_dir / f"frame_{i:06}.rgb").read_bytes()
        game = hud[HUD_TOP * OUT_W * 3 : (HUD_TOP + OUT_H) * OUT_W * 3]
        top = hud[: HUD_TOP * OUT_W * 3]
        bottom = hud[(HUD_TOP + OUT_H) * OUT_W * 3 :]
        check = {
            "index": i,
            "bytes": len(hud),
            "size_ok": len(hud) == expected_bytes,
            "game_region_matches_plain": game == plain,
            "top_band_pixels": sum(1 for b in top if b > 20),
            "bottom_band_pixels": sum(1 for b in bottom if b > 20),
        }
        check["ok"] = bool(
            check["size_ok"]
            and check["game_region_matches_plain"]
            and check["top_band_pixels"] > 0
            and check["bottom_band_pixels"] > 0
        )
        ok = ok and check["ok"]
        checks.append(check)
    return {
        "ok": ok,
        "frames": frames,
        "sink": "frames-hud",
        "dir": str(hud_dir),
        "receipt": str(hud_receipt / "frames.jsonl"),
        "summary": summary,
        "checks": checks,
    }


def recovery_leg(
    cuda: str,
    hip: str,
    token: str,
    frames: int,
    coord: Path,
    out: Path,
    stage_descriptor: Path,
) -> dict:
    """Run with HIP unreachable, then bring it up mid-run.

    A refused connection fails over immediately, so this leg proves failover and
    rejoin without depending on the D1c blackhole bounding work. The true
    mid-run blackhole is a separate D1c gate.
    """
    frames = max(4, min(10, frames))
    cport = free_port()
    hport = free_port()
    cproxy = Proxy(cport, parse_addr(cuda))
    hproxy = None
    rec_dir = out / "recovery"
    rec_receipt = out / "recovery-receipt"
    log_path = out / "recovery.log"
    log = open(log_path, "w")
    cmd = live_command(
        f"127.0.0.1:{cport}",
        f"127.0.0.1:{hport}",
        token,
        frames,
        rec_dir,
        rec_receipt,
        stage_descriptor,
        fps=8,
    )
    proc = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=log, text=True)
    lines_path = rec_receipt / "frames.jsonl"
    deadline = time.time() + 180
    try:
        while time.time() < deadline and len(receipt_lines(lines_path)) < 3:
            time.sleep(0.03)
        up_at = len(receipt_lines(lines_path))
        # The target returns on the same port.
        hproxy = Proxy(hport, parse_addr(hip))
        stdout, _ = proc.communicate(timeout=180)
    finally:
        if hproxy is not None:
            hproxy.stop()
        cproxy.stop()
        log.close()
        if proc.poll() is None:
            proc.kill()
            proc.wait()
    if proc.returncode != 0:
        return {
            "ok": False,
            "error": f"stream exited {proc.returncode}",
            "log": str(log_path),
            "stdout": (stdout or "")[-2000:],
        }
    summary = json.loads(stdout)
    lines = receipt_lines(lines_path)
    comparisons = compare_live(coord, rec_dir, frames)
    failover_frames = [
        line["index"]
        for line in lines
        if any((tile.get("failover_from") == "hip") for tile in (line.get("tiles") or []))
    ]
    rejoined_frames = [
        line["index"]
        for line in lines
        if not any(
            (tile.get("failover_from") == "hip") for tile in (line.get("tiles") or [])
        )
        and all(
            tile.get("attempts", 1) == 1
            for tile in (line.get("tiles") or [])
            if tile.get("col") == 1
        )
    ]
    recovered_frames = [
        line["index"]
        for line in lines
        if "hip" in (line.get("recovered_targets") or [])
    ]
    recovered_after_failover = any(
        rejoin > min(failover_frames) for rejoin in rejoined_frames
    ) if failover_frames else False
    exact = all(item["match"] for item in comparisons)
    ok = bool(
        exact
        and len(lines) == frames
        and summary.get("frames") == frames
        and summary.get("recoveries", 0) >= 1
        and failover_frames
        and recovered_frames
        and recovered_after_failover
    )
    return {
        "ok": ok,
        "frames": frames,
        "fps": 8,
        "up_at_frame": up_at,
        "matches": sum(1 for item in comparisons if item["match"]),
        "failover_frames": failover_frames,
        "rejoined_frames": rejoined_frames,
        "recovered_frames": recovered_frames,
        "summary": summary,
        "comparisons": comparisons,
        "dir": str(rec_dir),
        "receipt": str(lines_path),
        "log": str(log_path),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cuda", default=DEFAULT_CUDA)
    ap.add_argument("--hip", default=DEFAULT_HIP)
    ap.add_argument("--token", default="test")
    ap.add_argument("--frames", type=int, default=100)
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
    if args.frames < 2 or not (1 <= args.left_cols < FB_W):
        print("--frames must be >= 2 and --left-cols in 1..319")
        return 2

    out = Path(args.out).absolute()
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    coord = out / "coord"
    # The recovery leg may use up to ten measured frames; make sure the
    # coordinator reference covers them even on a short smoke run.
    total = max(args.frames, 10) + WARMUP
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
    comparisons = compare_live(coord, out / "live", args.frames)
    matches = sum(1 for item in comparisons if item["match"])

    summary_path = out / "live-receipt" / "summary.json"
    live_run = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    calibration = live_summary.get("calibration") or {}
    chosen = calibration.get("chosen") or {}
    adaptive_ok = bool(
        calibration.get("ok")
        and calibration.get("probe")
        and chosen
        and chosen.get("left_cols", 0) + chosen.get("right_cols", 0) == FB_W
        and live_summary.get("split") == chosen
    )

    hud = hud_leg(
        args.cuda,
        args.hip,
        args.token,
        args.frames,
        out,
        stage_descriptor,
        out / "live",
    )
    recovery = recovery_leg(
        args.cuda,
        args.hip,
        args.token,
        args.frames,
        coord,
        out,
        stage_descriptor,
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
        fail_matches += got == reference_frame(coord, i)
    fail_lines = receipt_lines(out / "failover-receipt" / "frames.jsonl")

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
        and adaptive_ok
        and hud.get("ok")
        and recovery.get("ok")
        and failover_ok
        and p99 <= 66
        and max_ms <= 132
        and over == 0
        and live_summary.get("frames") == args.frames
        and (live_summary.get("stage_descriptor") or {}).get("sha256")
    )
    summary = {
        "schema": "jotnar.stream.doom_live_gate/2",
        "ok": ok,
        "cuda": args.cuda,
        "hip": args.hip,
        "frames": args.frames,
        "warmup_frames": WARMUP,
        "split": {"left_cols": args.left_cols, "right_cols": FB_W - args.left_cols},
        "coordinator": {
            "frames": coord_summary.get("frames"),
            "latency_ms": coord_summary.get("latency_ms"),
            "run": str(coord / "run.json"),
        },
        "live": {
            "frames": live_summary.get("frames"),
            "latency_ms": latency,
            "failovers": live_summary.get("failovers"),
            "recoveries": live_summary.get("recoveries"),
            "mode": live_summary.get("mode"),
            "split": live_summary.get("split"),
            "source": live_summary.get("source"),
            "stage_descriptor": live_summary.get("stage_descriptor"),
            "calibration": calibration,
            "receipt": str(out / "live-receipt" / "frames.jsonl"),
            "run": live_run,
        },
        "adaptation": {
            "ok": adaptive_ok,
            "declared": calibration.get("declared"),
            "probe": calibration.get("probe"),
            "chosen": chosen,
            "rebalance": calibration.get("rebalance"),
            "warmup": calibration.get("split_warmup"),
            "policy": calibration.get("policy"),
        },
        "hud": hud,
        "recovery": recovery,
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
            "adaptive_split": adaptive_ok,
            "hud_rendered": bool(hud.get("ok")),
            "recovery": bool(recovery.get("ok")),
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
            "calibration": str(out / "live-receipt" / "summary.json"),
            "hud_frames": str(out / "hud"),
            "recovery_frames": str(out / "recovery"),
            "recovery_receipt": str(out / "recovery-receipt" / "frames.jsonl"),
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
