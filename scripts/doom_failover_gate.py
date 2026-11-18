#!/usr/bin/env python3
"""Network-outage gate for the two-node DOOM coordinator.

Starts two local TCP proxies in front of the real CUDA and HIP runners, runs
the coordinator through them, then closes one proxy mid-run. The coordinator
must mark that node down, keep the RV32IM loop alive, and finish the run on the
surviving node. The proxy makes the failure a real TCP failure without root,
iptables, or touching the remote runners.
"""
from __future__ import annotations

import argparse
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
WORK = Path(os.environ.get("DOOM_FAILOVER_WORK", "/tmp/opencode/doom_failover"))


def free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


class Proxy:
    """One TCP proxy. stop() closes the listener and every live connection."""

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
                for c, r in self.pairs:
                    socks.extend((c, r))
            try:
                readable, _, _ = select.select(socks, [], [], 0.1)
            except (OSError, ValueError):
                break
            for s in readable:
                if s is self.listener:
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
                    pair = next(((c, r) for c, r in self.pairs if s in (c, r)), None)
                if pair is None:
                    continue
                client, remote = pair
                peer = remote if s is client else client
                try:
                    data = s.recv(65536)
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
        for s in (client, remote):
            try:
                s.close()
            except OSError:
                pass

    def _close_all(self) -> None:
        with self.lock:
            pairs = list(self.pairs)
            self.pairs.clear()
        for c, r in pairs:
            for s in (c, r):
                try:
                    s.close()
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


def parse_addr(addr: str) -> tuple[str, int]:
    host, _, port = addr.partition(":")
    return host, int(port)


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


def run_scenario(name: str, fail_node: str, cuda: str, hip: str,
                 token: str, frames: int, fps: int, restart: bool = False) -> dict:
    out = WORK / name
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    ports = [free_port(), free_port()]
    proxies = [
        Proxy(ports[0], parse_addr(cuda)),
        Proxy(ports[1], parse_addr(hip)),
    ]
    cmd = [
        str(COORD),
        "--image", str(IMAGE),
        "--wad", str(WAD),
        "--cuda", f"127.0.0.1:{ports[0]}",
        "--hip", f"127.0.0.1:{ports[1]}",
        "--token", token,
        "--start-frame", "200",
        "--frames", str(frames),
        "--fps", str(fps),
        "--job-timeout-ms", "3000",
        "--recover-every", "10",
        "--out", str(out),
    ]
    log_path = out / "coordinator.log"
    log = open(log_path, "w")
    proc = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE,
                            stderr=log, text=True)
    try:
        deadline = time.time() + 60
        ready = False
        while time.time() < deadline:
            if proc.poll() is not None:
                break
            if log_path.exists() and "placement mode: split" in log_path.read_text():
                ready = True
                break
            time.sleep(0.05)
        if not ready:
            proc.wait(timeout=5)
            raise RuntimeError(f"{name}: coordinator did not reach split mode\n"
                               + log_path.read_text())
        time.sleep(0.5)
        target = 0 if fail_node == "cuda" else 1
        print(f"{name}: closing {fail_node} proxy at frame ~{frames // 4}")
        proxies[target].stop()
        if restart:
            time.sleep(0.8)
            target_addr = parse_addr(cuda if fail_node == "cuda" else hip)
            proxies[target] = Proxy(ports[target], target_addr)
            print(f"{name}: restarted {fail_node} proxy")
        stdout, _ = proc.communicate(timeout=120)
        rc = proc.returncode
    finally:
        for p in proxies:
            p.stop()
        log.close()
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=5)
    log_text = log_path.read_text()
    if rc != 0:
        raise RuntimeError(f"{name}: coordinator exited {rc}\n{log_text}")
    run = json.loads((out / "run.json").read_text())
    meta = [
        json.loads(line)
        for line in (out / "frames.jsonl").read_text().splitlines()
        if line.strip()
    ]
    expected_mode = "split" if restart else ("hip-full" if fail_node == "cuda" else "cuda-full")
    events = run.get("failover", {}).get("events", [])
    recoveries = run.get("failover", {}).get("recoveries", [])
    if not events:
        raise RuntimeError(f"{name}: no failover event\n{log_text}")
    if restart and not recoveries:
        raise RuntimeError(f"{name}: no recovery event\n{log_text}")
    if run.get("mode") != expected_mode:
        raise RuntimeError(f"{name}: mode {run.get('mode')} != {expected_mode}")
    if len(meta) != frames:
        raise RuntimeError(f"{name}: {len(meta)} frames, expected {frames}")
    last = out / meta[-1]["file"]
    if last.stat().st_size != 640 * 400 * 3:
        raise RuntimeError(f"{name}: bad frame size {last.stat().st_size}")
    print(
        f"{name}: OK mode={run['mode']} events={len(events)} recoveries={len(recoveries)} "
        f"p50={run['latency_ms']['total_p50']}ms p95={run['latency_ms']['total_p95']}ms"
    )
    return {
        "scenario": name,
        "failed_node": fail_node,
        "mode": run["mode"],
        "events": events,
        "recoveries": recoveries,
        "frames": len(meta),
        "latency_ms": run["latency_ms"],
        "log": str(log_path),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cuda", default=DEFAULT_CUDA)
    ap.add_argument("--hip", default=DEFAULT_HIP)
    ap.add_argument("--token", default="test")
    ap.add_argument("--frames", type=int, default=45)
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--no-build", action="store_true")
    args = ap.parse_args()
    if not IMAGE.exists() or not WAD.exists():
        print(f"missing DOOM image or WAD\n  image: {IMAGE}\n  wad:   {WAD}")
        return 2
    if not args.no_build:
        build()
    if not COORD.exists():
        print(f"missing {COORD}; run without --no-build")
        return 2
    WORK.mkdir(parents=True, exist_ok=True)
    results = []
    try:
        results.append(
            run_scenario("cuda-outage", "cuda", args.cuda, args.hip,
                         args.token, args.frames, args.fps)
        )
        results.append(
            run_scenario("hip-outage", "hip", args.cuda, args.hip,
                         args.token, args.frames, args.fps)
        )
        results.append(
            run_scenario("cuda-recover", "cuda", args.cuda, args.hip,
                         args.token, args.frames, args.fps, restart=True)
        )
        results.append(
            run_scenario("hip-recover", "hip", args.cuda, args.hip,
                         args.token, args.frames, args.fps, restart=True)
        )
    except Exception as e:
        print(e)
        return 1
    summary = {"schema": "jotnar.doom.failover/1", "results": results}
    (WORK / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print("RESULT: ALL OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
