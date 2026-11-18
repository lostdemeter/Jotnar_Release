#!/usr/bin/env python3
"""Reproducible warm-frame gate for the GPU scheduler.

For each probe size and scheduler mode, warm the workers, run a fixed number of
uncached batches, verify exact/identical outputs, and report medians. The
script does not claim a winner; it records the evidence needed to make one.
"""
import argparse
import hashlib
import json
import os
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path


def parse_counts(raw):
    counts = {}
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        name, value = item.split("=", 1)
        counts[name] = int(value)
    return counts


def parse_modes(raw):
    modes = [m.strip() for m in raw.split(",") if m.strip()]
    for mode in modes:
        if mode not in ("perjob", "batch"):
            raise SystemExit(f"unknown mode {mode!r}")
    return modes


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def run_gate(args, size, mode, count, batch_id, batch_frames):
    cmd = [args.binary]
    for runner in args.runner:
        cmd += ["--runner", runner]
    if args.token:
        cmd += ["--token", args.token]
    cmd += [
        "run",
        str(Path(args.probe_dir) / "matmul-warm.asm"),
        "--payload",
        str(Path(args.probe_dir) / f"{size}.json"),
        "--outputs",
        args.outputs,
        "--target",
        args.target,
        "--count",
        str(count),
        "--parallel",
        str(args.parallel),
        "--max-attempts",
        str(args.max_attempts),
        "--ledger",
        args.ledger,
        "--batch-id",
        batch_id,
    ]
    if mode == "batch":
        cmd += ["--batch-frames", str(batch_frames)]

    started = time.monotonic()
    proc = subprocess.run(cmd, text=True, capture_output=True, timeout=args.timeout)
    wall_s = time.monotonic() - started
    if proc.returncode != 0:
        raise RuntimeError(
            f"{size}/{mode}: exit {proc.returncode}\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
        )
    text = proc.stdout
    start = text.find("{")
    if start < 0:
        raise RuntimeError(f"{size}/{mode}: no JSON summary\n{text}\n{proc.stderr}")
    try:
        summary = json.loads(text[start:])
    except json.JSONDecodeError as e:
        raise RuntimeError(f"{size}/{mode}: bad JSON: {e}\n{text}") from e
    if summary.get("cached"):
        raise RuntimeError(f"{size}/{mode}: unexpected cached summary")
    if summary.get("done") != count or summary.get("failed") != 0:
        raise RuntimeError(f"{size}/{mode}: {summary}")
    if summary.get("identical_outputs") is not True:
        raise RuntimeError(f"{size}/{mode}: outputs differ: {summary}")
    return {
        "size": size,
        "mode": mode,
        "cmd": cmd,
        "count": count,
        "batch_frames": batch_frames if mode == "batch" else 1,
        "wall_ms": summary["wall_ms"],
        "wall_s": round(wall_s, 3),
        "jobs_per_s": summary["jobs_per_s"],
        "retries": summary.get("retries", 0),
        "output_bytes": summary.get("output_bytes", 0),
        "targets": summary.get("targets", []),
        "stderr": proc.stderr.strip(),
    }


def summarize(rows):
    walls = [r["wall_ms"] for r in rows]
    rates = [r["jobs_per_s"] for r in rows]
    ordered = sorted(walls)
    p95 = ordered[min(len(ordered) - 1, int(round(0.95 * (len(ordered) - 1))))]
    return {
        "runs": len(rows),
        "wall_ms_min": min(walls),
        "wall_ms_median": statistics.median(walls),
        "wall_ms_mean": round(statistics.mean(walls), 3),
        "wall_ms_p95": p95,
        "wall_ms_max": max(walls),
        "pairs_per_s_median": statistics.median(rates),
        "pairs_per_s_min": min(rates),
        "pairs_per_s_max": max(rates),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runner", action="append", required=True, help="hip=ADDR or cuda=ADDR")
    ap.add_argument("--token")
    ap.add_argument("--binary", default="target/release/jotnar-gpu-net")
    ap.add_argument("--probe-dir", required=True)
    ap.add_argument("--sizes", default="mm64,mm256,mm512")
    ap.add_argument("--counts", default="mm64=20,mm256=10,mm512=10")
    ap.add_argument("--modes", default="perjob,batch")
    ap.add_argument("--target", default="both")
    ap.add_argument("--outputs", default="OUT")
    ap.add_argument("--parallel", type=int, default=4)
    ap.add_argument(
        "--batch-frames",
        type=int,
        default=0,
        help="frames per batch request; 0 auto-sizes to ceil(count/parallel)",
    )
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--max-attempts", type=int, default=3)
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--ledger", default=tempfile.mkdtemp(prefix="jotnar-gpu-gate-"))
    ap.add_argument("--json-out", help="write the full evidence JSON here")
    args = ap.parse_args()

    sizes = [s.strip() for s in args.sizes.split(",") if s.strip()]
    counts = parse_counts(args.counts)
    modes = parse_modes(args.modes)
    if args.batch_frames != 0 and "batch" not in modes:
        modes.append("batch")
    if args.batch_frames == 1 and "batch" in modes:
        raise SystemExit("batch mode needs --batch-frames >1 or 0 (auto)")
    if args.batch_frames < 0:
        raise SystemExit("--batch-frames must be >=0")
    for size in sizes:
        if size not in counts:
            raise SystemExit(f"missing count for {size}")

    Path(args.ledger).mkdir(parents=True, exist_ok=True)
    probe = Path(args.probe_dir)
    evidence = {
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "binary": os.path.abspath(args.binary),
        "runners": args.runner,
        "target": args.target,
        "parallel": args.parallel,
        "batch_frames": args.batch_frames if args.batch_frames else "auto",
        "repeats": args.repeats,
        "listing_sha256": sha256(probe / "matmul-warm.asm"),
        "payload_sha256": {s: sha256(probe / f"{s}.json") for s in sizes},
        "results": [],
        "summary": {},
    }

    stamp = time.time_ns()
    for size in sizes:
        count = counts[size]
        batch_frames = args.batch_frames
        if batch_frames == 0:
            batch_frames = max(1, min(64, (count + args.parallel - 1) // args.parallel))
        for mode in modes:
            # One untimed warmup with the same shape as the measured runs.
            run_gate(
                args,
                size,
                mode,
                count,
                f"gate-{size}-{mode}-warm-{stamp}",
                batch_frames,
            )
            rows = []
            for rep in range(args.repeats):
                rows.append(
                    run_gate(
                        args,
                        size,
                        mode,
                        count,
                        f"gate-{size}-{mode}-{stamp}-{rep}",
                        batch_frames,
                    )
                )
            evidence["results"].extend(rows)
            summary = summarize(rows)
            summary["batch_frames"] = batch_frames if mode == "batch" else 1
            evidence["summary"][f"{size}/{mode}"] = summary

    out = Path(args.json_out) if args.json_out else Path(args.ledger) / "gate.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(evidence, indent=2) + "\n")

    print(f"{'size/mode':<16} {'runs':>4} {'median ms':>10} {'p95 ms':>9} {'pairs/s':>9}")
    for key, s in evidence["summary"].items():
        print(
            f"{key:<16} {s['runs']:>4} {s['wall_ms_median']:>10.1f} "
            f"{s['wall_ms_p95']:>9.1f} {s['pairs_per_s_median']:>9.1f}"
        )
    print(f"\nevidence: {out}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
