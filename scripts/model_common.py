#!/usr/bin/env python3
"""Shared helpers for the model decode and performance gates.

Slice 2's gate keeps the checkpoint loader, the numpy reference block, the
payload encoder, and the bundle writer. This module adds the pieces slices 3+
need without forking them:

  - controller-side token embedding plus the learned position embedding;
  - named placement policies (pipeline, single-CUDA, single-HIP, data-parallel);
  - a sentence runner that accepts an explicit placement map.

Executors still receive one listing plus payloads; the checkpoint, tokenizer,
decode policy, and position embedding stay in the controller.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np

import model_block_gate as mb
import model_sentence_gate as msg

CONTROLLER = msg.CONTROLLER
JOTNAR = msg.JOTNAR
ATTN4 = msg.ATTN4
MLP = msg.MLP
HEAD = msg.HEAD
DEFAULT_CUDA = msg.DEFAULT_CUDA
DEFAULT_HIP = msg.DEFAULT_HIP

load_checkpoint = msg.load_checkpoint
block_weights = msg.block_weights
reference_block = msg.reference_block
reference_from_x = msg.reference_from_x
write_bundle = msg.write_bundle
compare = mb.compare
read_output = mb.read_output
reachable = mb.reachable
run = mb.run
build = mb.build


def encode_x(checkpoint: dict, tokens: list[int]) -> np.ndarray:
    """Controller-side embedding: token table plus learned position table."""
    x = checkpoint["tensors"]["embedding"][tokens].astype(np.float32)
    pos = checkpoint["tensors"].get("position_embedding")
    if pos is not None:
        x = (x + pos[: len(tokens)]).astype(np.float32)
    return x


def is_trained(checkpoint: dict) -> bool:
    return bool(checkpoint.get("training")) if "training" in checkpoint else False


def placement(kind: str, cuda: str, hip: str, blocks: int) -> dict[str, str]:
    """Return stage -> address for a named placement policy."""
    places = {}
    if kind == "pipeline":
        for index in range(blocks):
            places[f"attn-{index}"] = cuda
            places[f"mlp-{index}"] = hip
        places["head"] = cuda
    elif kind in ("single-cuda", "cuda"):
        for index in range(blocks):
            places[f"attn-{index}"] = cuda
            places[f"mlp-{index}"] = cuda
        places["head"] = cuda
    elif kind in ("single-hip", "hip"):
        for index in range(blocks):
            places[f"attn-{index}"] = hip
            places[f"mlp-{index}"] = hip
        places["head"] = hip
    elif kind == "cuda-client" or kind == "hip-client":
        addr = cuda if kind == "cuda-client" else hip
        for index in range(blocks):
            places[f"attn-{index}"] = addr
            places[f"mlp-{index}"] = addr
        places["head"] = addr
    else:
        raise ValueError(f"unknown placement {kind!r}")
    return places


def run_placed(
    bundle: Path,
    ledger: Path,
    cuda: str,
    hip: str,
    token: str,
    blocks: int,
    kind: str,
) -> dict:
    if ledger.exists():
        shutil.rmtree(ledger)
    cmd = [
        str(CONTROLLER),
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
    for stage, addr in placement(kind, cuda, hip, blocks).items():
        cmd += ["--place", f"{stage}={addr}"]
    run(cmd)
    return json.loads((ledger / "receipt.json").read_text())


def stage_intervals(receipt: dict) -> dict[str, dict]:
    stages = receipt.get("stages") or {}
    return {
        name: {
            "start": int(record.get("started_ms") or 0),
            "finish": int(record.get("finished_ms") or 0),
            "executor": record.get("executor"),
            "state": record.get("state"),
            "timing": record.get("timing"),
        }
        for name, record in stages.items()
    }


def utilization(receipt: dict) -> dict:
    """Busy time per executor and makespan from the receipt's stage clocks."""
    intervals = stage_intervals(receipt)
    if not intervals:
        return {"makespan_ms": 0, "executors": {}, "critical_path_ms": 0}
    start = min(item["start"] for item in intervals.values() if item["start"])
    finish = max(item["finish"] for item in intervals.values())
    makespan = max(0, finish - start)
    per_exec: dict[str, int] = {}
    for item in intervals.values():
        if not item["start"] or not item["finish"]:
            continue
        per_exec[item["executor"]] = per_exec.get(item["executor"], 0) + (
            item["finish"] - item["start"]
        )
    critical = sum(
        item["finish"] - item["start"]
        for item in intervals.values()
        if item["start"] and item["finish"]
    )
    return {
        "start_ms": start,
        "finish_ms": finish,
        "makespan_ms": makespan,
        "critical_path_ms": critical,
        "executors": {
            addr: {
                "busy_ms": busy,
                "utilization": (busy / makespan) if makespan else 0.0,
            }
            for addr, busy in per_exec.items()
        },
    }


def timing_classes(receipt: dict) -> dict[str, int]:
    """How many stages reported a compile/build or a warm cache hit."""
    build = warm = 0
    for record in (receipt.get("stages") or {}).values():
        timing = record.get("timing") or {}
        if (timing.get("compile_ms") or 0) or (timing.get("build_ms") or 0):
            build += 1
        if timing.get("cache_hit"):
            warm += 1
    return {"stages": len(receipt.get("stages") or {}), "built": build, "cache_hit": warm}
