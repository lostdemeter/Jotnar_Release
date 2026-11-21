#!/usr/bin/env python3
"""Model slice 3: honest single-GPU vs pipeline vs data-parallel measurement.

The checkpoint, tokenizer, and placement policy stay in the controller. This
gate measures the same five-stage sentence graph four ways:

  single-cuda    attn/head + mlp on one CUDA runner
  pipeline       attn/head CUDA, mlp HIP (the slice-2 placement)
  single-hip     the whole graph on one HIP runner
  data-parallel  each request runs the whole model on one GPU; concurrent
                 requests are split across the CUDA and HIP runners

It records end-to-end controller latency, per-stage clocks, stage utilization,
the critical path, and the executor-reported compile/build/run milliseconds.
Throughput is measured with 1/2/4 concurrent controller clients. The verdict is
written into the JSON and a Markdown table; losing options are reported.

This is a tiny model and each stage is a separate process launch. The numbers
measure the system as built -- including that dispatch cost -- not a fused
kernel. No performance claim is made without this gate.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import model_common as mc  # noqa: E402
import model_sentence_gate as msg  # noqa: E402

DEFAULT_CHECKPOINT = ROOT / "demos" / "model-block" / "trained"
CACHE_DIR = "/tmp/jotnar-gpu-runner-cache-9997"


def load_prompt(path: Path | None, checkpoint: dict) -> list[int]:
    context = int(checkpoint["config"]["context"])
    if path is None:
        tokens = list(checkpoint["tokens"])
    else:
        fixture = json.loads(path.read_text())
        tokens = (
            [int(x) for x in fixture["bytes"]]
            if "bytes" in fixture
            else list(str(fixture["text"]).encode("utf-8"))
        )
    if len(tokens) != context:
        raise SystemExit(f"prompt must be {context} bytes, got {len(tokens)}")
    return tokens


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(fraction * len(ordered)) - 1))
    return ordered[index]


def stats(values: list[float]) -> dict:
    if not values:
        return {}
    return {
        "n": len(values),
        "min": min(values),
        "p50": percentile(values, 0.50),
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
        "max": max(values),
        "mean": sum(values) / len(values),
    }


def timed_run(bundle: Path, ledger: Path, kind: str, cuda: str, hip: str, token: str, blocks: int):
    if ledger.exists():
        shutil.rmtree(ledger)
    cmd = [
        str(mc.CONTROLLER),
        "--token",
        token,
        "run",
        "--bundle",
        str(bundle),
        "--ledger",
        str(ledger),
        "--gpu-executor",
        f"cuda={cuda}",
        "--gpu-executor",
        f"hip={hip}",
    ]
    for stage, addr in mc.placement(kind, cuda, hip, blocks).items():
        cmd += ["--place", f"{stage}={addr}"]
    started = time.perf_counter()
    result = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    wall_ms = (time.perf_counter() - started) * 1000.0
    if result.returncode != 0:
        raise RuntimeError(
            f"controller failed ({result.returncode}): {result.stdout[-2000:]}{result.stderr[-2000:]}"
        )
    receipt = json.loads((ledger / "receipt.json").read_text())
    return wall_ms, receipt


def stage_medians(walls: list[tuple[float, dict]]) -> dict:
    names = sorted({name for _, receipt in walls for name in receipt.get("stages") or {}})
    out = {}
    for name in names:
        durations = []
        timings = []
        for _, receipt in walls:
            record = (receipt.get("stages") or {}).get(name) or {}
            if record.get("started_ms") and record.get("finished_ms"):
                durations.append(record["finished_ms"] - record["started_ms"])
            timings.append(record.get("timing") or {})
        out[name] = {
            "duration_ms": stats(durations),
            "timing": {
                key: stats([float(item[key]) for item in timings if item.get(key) is not None])
                for key in ("compile_ms", "build_ms", "run_ms")
            },
        }
    return out


def forward_measure(
    bundle: Path,
    out: Path,
    kind: str,
    repeats: int,
    warmup: int,
    cuda: str,
    hip: str,
    token: str,
    blocks: int,
    cold: bool = False,
    cold_ssh_cuda: str = "",
    cold_ssh_hip: str = "",
    cold_cache_dir: str = CACHE_DIR,
) -> dict:
    if cold:
        # Clear before each placement so a cold number includes that
        # placement's own builds, not the previous placement's cache.
        cold_clear(cold_ssh_cuda, cold_cache_dir)
        cold_clear(cold_ssh_hip, cold_cache_dir)
    for index in range(warmup):
        timed_run(bundle, out / f"warmup-{kind}-{index}", kind, cuda, hip, token, blocks)
    runs = []
    for index in range(repeats):
        wall, receipt = timed_run(
            bundle, out / f"run-{kind}-{index}", kind, cuda, hip, token, blocks
        )
        runs.append((wall, receipt))
    walls = [wall for wall, _ in runs]
    first_wall = walls[0]
    warm_walls = walls[1:] if len(walls) > 1 else walls
    utilizations = [mc.utilization(receipt) for _, receipt in runs]
    timing = [mc.timing_classes(receipt) for _, receipt in runs]
    stage_span = [
        (max(item["finish"] for item in mc.stage_intervals(receipt).values())
         - min(item["start"] for item in mc.stage_intervals(receipt).values() if item["start"]))
        for _, receipt in runs
    ]
    return {
        "placement": kind,
        "wall_ms": stats(walls),
        "first_wall_ms": first_wall,
        "warm_wall_ms": stats(warm_walls),
        "receipt_span_ms": stats([float(x) for x in stage_span]),
        "controller_overhead_ms": stats(
            [wall - span for wall, span in zip(walls, stage_span)]
        ),
        "utilization": utilizations[-1],
        "critical_path_ms": stats(
            [float(item["critical_path_ms"]) for item in utilizations]
        ),
        "executor_timing": stage_medians(runs),
        "cache": {
            "stages_built": sum(item["built"] for item in timing),
            "stages_cache_hit": sum(item["cache_hit"] for item in timing),
            "stage_runs": sum(item["stages"] for item in timing),
        },
    }


def timed_decode(
    checkpoint: dict,
    prompt: list[int],
    decode: int,
    kind: str,
    root: Path,
    cuda: str,
    hip: str,
    token: str,
) -> dict:
    context = int(checkpoint["config"]["context"])
    blocks = int(checkpoint["config"]["blocks"])
    vocab = int(checkpoint["config"]["vocab"])
    seq = list(prompt)
    step_times = []
    start = time.perf_counter()
    for index in range(decode):
        window = seq[-context:]
        x = mc.encode_x(checkpoint, window)
        bundle = root / f"step-{index:02d}" / "bundle"
        ledger = root / f"step-{index:02d}" / "run"
        msg.write_bundle(bundle, checkpoint, x=x)
        wall, receipt = timed_run(bundle, ledger, kind, cuda, hip, token, blocks)
        logits = mc.read_output(ledger, "head", "LOGITS", (len(window), vocab))
        seq.append(int(np.argmax(logits[-1])))
        step_times.append(wall)
    total = (time.perf_counter() - start) * 1000.0
    return {
        "placement": kind,
        "steps": decode,
        "total_ms": total,
        "step_ms": stats(step_times),
        "sequence_hex": bytes(seq).hex(),
    }


def throughput_leg(
    bundle: Path,
    out: Path,
    mode: str,
    clients: int,
    requests: int,
    cuda: str,
    hip: str,
    token: str,
    blocks: int,
) -> dict:
    """Run `clients` concurrent controllers; each completes `requests` forwards."""
    if mode == "pipeline":
        kinds = ["pipeline"] * clients
    elif mode == "data-parallel":
        kinds = ["cuda-client" if index % 2 == 0 else "hip-client" for index in range(clients)]
    else:
        kinds = [mode] * clients

    def client(index: int) -> dict:
        kind = kinds[index]
        walls = []
        for request in range(requests):
            wall, _ = timed_run(
                bundle,
                out / f"client-{index:02d}" / f"request-{request:02d}",
                kind,
                cuda,
                hip,
                token,
                blocks,
            )
            walls.append(wall)
        return {"kind": kind, "walls": walls}

    started = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=clients) as pool:
        results = list(pool.map(client, range(clients)))
    wall = time.perf_counter() - started
    completed = sum(len(item["walls"]) for item in results)
    return {
        "mode": mode,
        "clients": clients,
        "requests_per_client": requests,
        "completed": completed,
        "wall_ms": wall * 1000.0,
        "throughput_per_s": completed / wall if wall else 0.0,
        "per_client": [
            {
                "kind": item["kind"],
                "p50_ms": stats(item["walls"])["p50"],
                "p99_ms": stats(item["walls"])["p99"],
            }
            for item in results
        ],
    }


def cold_clear(host: str, cache_dir: str) -> None:
    cmd = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5", host,
           f"rm -rf {cache_dir}/*"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"cold cache clear on {host} failed: {result.stderr.strip()}"
        )


def markdown(summary: dict) -> str:
    lines = [
        "# Model performance gate",
        "",
        f"- checkpoint: `{summary['checkpoint']['name']}` ({summary['checkpoint']['kind']})",
        f"- controller load: {summary['load']['load1']:.2f} (1 min), "
        f"{summary['load']['cpus']} CPUs",
        f"- repeats: {summary['repeats']} warm, warmup {summary['warmup']}",
        "",
        "| placement | p50 ms | p99 ms | max ms | util CUDA | util HIP | built | cache hits |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for kind, item in summary["forward"].items():
        cuda_util = next(
            (v["utilization"] for k, v in item["utilization"]["executors"].items()
             if k == summary["cuda"]),
            0.0,
        )
        hip_util = next(
            (v["utilization"] for k, v in item["utilization"]["executors"].items()
             if k == summary["hip"]),
            0.0,
        )
        lines.append(
            f"| {kind} | {item['wall_ms']['p50']:.1f} | {item['wall_ms']['p99']:.1f} | "
            f"{item['wall_ms']['max']:.1f} | {cuda_util:.2f} | {hip_util:.2f} | "
            f"{item['cache']['stages_built']} | {item['cache']['stages_cache_hit']} |"
        )
    lines += [
        "",
        "| throughput | clients | requests | wall ms | req/s |",
        "|---|---:|---:|---:|---:|",
    ]
    for item in summary["throughput"]:
        lines.append(
            f"| {item['mode']} | {item['clients']} | {item['completed']} | "
            f"{item['wall_ms']:.0f} | {item['throughput_per_s']:.3f} |"
        )
    lines += ["", "## Verdict", "", summary["verdict"]["text"], ""]
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    ap.add_argument("--prompt", default=None)
    ap.add_argument("--cuda", default=mc.DEFAULT_CUDA)
    ap.add_argument("--hip", default=mc.DEFAULT_HIP)
    ap.add_argument("--token", default="test")
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--decode", type=int, default=8)
    ap.add_argument("--decode-repeats", type=int, default=3)
    ap.add_argument("--clients", default="1,2,4")
    ap.add_argument("--requests", type=int, default=4)
    ap.add_argument(
        "--placements", default="single-cuda,pipeline,single-hip"
    )
    ap.add_argument("--no-build", action="store_true")
    ap.add_argument("--cold", action="store_true",
                    help="clear the runner build caches over SSH before each placement")
    ap.add_argument("--forward-only", action="store_true",
                    help="skip decode and throughput legs (used for cold runs)")
    ap.add_argument("--cold-ssh-cuda", default="mintsession@192.168.1.116")
    ap.add_argument("--cold-ssh-hip", default="mintsession2@192.168.1.195")
    ap.add_argument("--cold-cache-dir", default=CACHE_DIR)
    ap.add_argument("--out", default="/tmp/opencode/model-perf-gate")
    args = ap.parse_args()

    for addr in (args.cuda, args.hip):
        if not mc.reachable(addr):
            print(f"GPU runner is not reachable at {addr}")
            return 2
    if not args.no_build:
        mc.build()
    if not mc.CONTROLLER.exists():
        print(f"missing {mc.CONTROLLER}; run without --no-build")
        return 2

    checkpoint_dir = Path(args.checkpoint).absolute()
    checkpoint = mc.load_checkpoint(checkpoint_dir)
    prompt_path = Path(args.prompt) if args.prompt else None
    if prompt_path is None and (checkpoint_dir / "prompt.json").exists():
        prompt_path = checkpoint_dir / "prompt.json"
    prompt = load_prompt(prompt_path, checkpoint)
    blocks = int(checkpoint["config"]["blocks"])

    out = Path(args.out).absolute()
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    bundle = out / "bundle"
    msg.write_bundle(bundle, checkpoint, x=mc.encode_x(checkpoint, prompt))

    placements = [item.strip() for item in args.placements.split(",") if item.strip()]
    forward = {
        kind: forward_measure(
            bundle, out, kind, args.repeats, args.warmup,
            args.cuda, args.hip, args.token, blocks,
            cold=args.cold,
            cold_ssh_cuda=args.cold_ssh_cuda,
            cold_ssh_hip=args.cold_ssh_hip,
            cold_cache_dir=args.cold_cache_dir,
        )
        for kind in placements
    }

    decodes = []
    for kind in (() if args.forward_only else ("pipeline", "single-hip")):
        for index in range(args.decode_repeats):
            decodes.append(
                timed_decode(
                    checkpoint, prompt, args.decode, kind,
                    out / f"decode-{kind}-{index}",
                    args.cuda, args.hip, args.token,
                )
            )
    decode_summary = {}
    for kind in sorted({item["placement"] for item in decodes}):
        items = [item for item in decodes if item["placement"] == kind]
        decode_summary[kind] = {
            "runs": len(items),
            "total_ms": stats([item["total_ms"] for item in items]),
            "step_p50_ms": stats([item["step_ms"]["p50"] for item in items]),
            "step_p99_ms": stats([item["step_ms"]["p99"] for item in items]),
            "sequences": [item["sequence_hex"] for item in items],
        }

    client_counts = (
        [] if args.forward_only
        else [int(item) for item in args.clients.split(",") if item.strip()]
    )
    throughput = []
    for clients in client_counts:
        throughput.append(
            throughput_leg(
                bundle, out / f"tp-pipeline-{clients}", "pipeline", clients,
                args.requests, args.cuda, args.hip, args.token, blocks,
            )
        )
        if clients >= 2:
            throughput.append(
                throughput_leg(
                    bundle, out / f"tp-data-{clients}", "data-parallel", clients,
                    args.requests, args.cuda, args.hip, args.token, blocks,
                )
            )

    load1 = os.getloadavg()[0]
    cpus = os.cpu_count() or 1
    best = min(forward, key=lambda kind: forward[kind]["wall_ms"]["p50"])
    pipeline = forward.get("pipeline")
    single_cuda = forward.get("single-cuda")
    single_hip = forward.get("single-hip")
    dp = [item for item in throughput if item["mode"] == "data-parallel"]
    pl = [item for item in throughput if item["mode"] == "pipeline"]
    dp_max = max(dp, key=lambda item: item["clients"]) if dp else None
    pl_same = next(
        (item for item in pl if dp_max and item["clients"] == dp_max["clients"]),
        None,
    )
    verdict = {
        "best_forward_p50": best,
        "pipeline_vs_single_cuda": (
            (single_cuda["wall_ms"]["p50"] / pipeline["wall_ms"]["p50"])
            if pipeline and single_cuda else None
        ),
        "pipeline_vs_single_hip": (
            (single_hip["wall_ms"]["p50"] / pipeline["wall_ms"]["p50"])
            if pipeline and single_hip else None
        ),
        "data_parallel_vs_pipeline_throughput": (
            (dp_max["throughput_per_s"] / pl_same["throughput_per_s"])
            if dp_max and pl_same and pl_same["throughput_per_s"] else None
        ),
        "losing_options": [],
        "notes": [
            "Each stage is a separate controller subprocess and a separate GPU "
            "runner process launch; per-stage compile/build/run timing is kept "
            "in the JSON so dispatch cost is not confused with model compute.",
            f"Control-host load was {load1:.2f} at the end of the run; these "
            "are loaded-controller numbers, not a quiet-host claim.",
        ],
    }
    for kind, item in forward.items():
        if kind != best:
            verdict["losing_options"].append(
                f"{kind} p50 {item['wall_ms']['p50']:.1f} ms is "
                f"{item['wall_ms']['p50'] / forward[best]['wall_ms']['p50']:.2f}x "
                f"the best ({best})"
            )
    if pipeline and single_hip and pipeline["wall_ms"]["p50"] > single_hip["wall_ms"]["p50"]:
        verdict["pipeline_loses_to_single_hip"] = True
    parts = [f"Best forward p50: {best} ({forward[best]['wall_ms']['p50']:.1f} ms)."]
    if verdict["pipeline_vs_single_cuda"]:
        parts.append(
            f"Pipeline is {verdict['pipeline_vs_single_cuda']:.2f}x faster than "
            "single-CUDA."
        )
    if verdict["pipeline_vs_single_hip"]:
        parts.append(
            f"Pipeline is {verdict['pipeline_vs_single_hip']:.2f}x "
            + ("faster" if verdict["pipeline_vs_single_hip"] >= 1 else "slower")
            + " than single-HIP."
        )
    if verdict["data_parallel_vs_pipeline_throughput"]:
        parts.append(
            f"Data-parallel throughput is "
            f"{verdict['data_parallel_vs_pipeline_throughput']:.2f}x pipeline at "
            f"{dp_max['clients']} clients."
        )
    parts.append(
        "Dispatch overhead dominates this tiny graph; the split is not a "
        "compute-scaling win at this scale."
    )
    parts.extend(verdict["losing_options"])
    verdict["text"] = " ".join(parts)

    summary = {
        "schema": "jotnar.model.perf_gate/1",
        "ok": True,
        "cuda": args.cuda,
        "hip": args.hip,
        "checkpoint": {
            "path": str(checkpoint_dir),
            "name": checkpoint.get("name"),
            "sha256": checkpoint["sha256"],
            "kind": "trained" if checkpoint.get("training") else "fixture",
            "config": checkpoint["config"],
        },
        "repeats": args.repeats,
        "warmup": args.warmup,
        "cold": bool(args.cold),
        "load": {"load1": load1, "load5": os.getloadavg()[1], "cpus": cpus},
        "forward": forward,
        "decode": decode_summary,
        "throughput": throughput,
        "verdict": verdict,
        "evidence": {
            "root": str(out),
            "summary": str(out / "model-perf-gate.json"),
            "markdown": str(out / "model-perf-gate.md"),
        },
    }
    (out / "model-perf-gate.json").write_text(json.dumps(summary, indent=2) + "\n")
    (out / "model-perf-gate.md").write_text(markdown(summary) + "\n")
    print(json.dumps(summary["verdict"], indent=2))
    print((out / "model-perf-gate.md").read_text())
    return 0


if __name__ == "__main__":
    sys.exit(main())
