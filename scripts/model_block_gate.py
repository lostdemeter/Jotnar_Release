#!/usr/bin/env python3
"""Model flagship slice 1: a pipeline-parallel transformer block stack.

Attention halves run on CUDA, MLP halves on HIP. The controller holds the model
manifest, weights, and input as bundle artifacts and sends each stage a listing
plus payloads; the executors see no model semantics. The reference is numpy.

Each block (single-head, D=64, N=8 tokens, inter=128) is:

  CUDA:  X -> RMSNorm -> QKV -> Q K^T -> softmax -> A V -> out proj
             -> residual -> RMSNorm                    outputs N2, RES
  HIP:   N2 -> gate/up -> SiLU(gate)*up -> down -> +RES           output OUT

Blocks chain through the workload graph: block i's attention consumes block
i-1's OUT as X. The gate checks every block output against numpy, both targets,
a repeat run for determinism, and a perturbed-weight negative control.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import socket
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
CONTROLLER = ROOT / "target" / "release" / "jotnar-workload"
JOTNAR = ROOT / "target" / "release" / "jotnar"
ATTN = ROOT / "demos" / "model-block" / "attn.asm"
MLP = ROOT / "demos" / "model-block" / "mlp.asm"
DEFAULT_CUDA = "192.168.1.116:9997"
DEFAULT_HIP = "192.168.1.195:9997"

D = 64
N = 8
INTER = 128
EPS = 1e-5


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
        [
            "cargo",
            "build",
            "--release",
            "-p",
            "jotnar-node-tools",
            "--bin",
            "jotnar-workload",
        ],
    ):
        r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
        if r.returncode != 0:
            print(r.stdout)
            print(r.stderr)
            raise SystemExit("build failed")


def run(cmd: list[str]) -> subprocess.CompletedProcess:
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    if r.returncode != 0:
        print("command failed:", " ".join(cmd))
        print(r.stdout[-4000:])
        print(r.stderr[-4000:])
        raise SystemExit(r.returncode)
    return r


def make_inputs(seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    scale = 1.0 / np.sqrt(D)
    return {
        "X": rng.uniform(-0.3, 0.3, (N, D)).astype(np.float32),
        "WQ": (rng.uniform(-0.02, 0.02, (D, D)) * scale).astype(np.float32),
        "WK": rng.uniform(-0.02, 0.02, (D, D)).astype(np.float32),
        "WV": rng.uniform(-0.02, 0.02, (D, D)).astype(np.float32),
        "WO": rng.uniform(-0.02, 0.02, (D, D)).astype(np.float32),
        "W1": rng.uniform(-0.02, 0.02, (D, INTER)).astype(np.float32),
        "WUP": rng.uniform(-0.02, 0.02, (D, INTER)).astype(np.float32),
        "W2": rng.uniform(-0.02, 0.02, (INTER, D)).astype(np.float32),
        "R1": np.ones(D, dtype=np.float32),
        "R2": np.ones(D, dtype=np.float32),
    }


def rmsnorm(x: np.ndarray, gain: np.ndarray) -> np.ndarray:
    return (
        x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + EPS) * gain
    ).astype(np.float32)


def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return (e / e.sum(axis=-1, keepdims=True)).astype(np.float32)


def reference_block(x: np.ndarray, w: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    n1 = rmsnorm(x, w["R1"])
    q = n1 @ w["WQ"]
    k = n1 @ w["WK"]
    v = n1 @ w["WV"]
    p = softmax(q @ k.T)
    o = (p @ v) @ w["WO"]
    res = (x + o).astype(np.float32)
    n2 = rmsnorm(res, w["R2"])
    g = n2 @ w["W1"]
    u = n2 @ w["WUP"]
    m = ((g * (1.0 / (1.0 + np.exp(-g)))) * u).astype(np.float32)
    d = m @ w["W2"]
    out = (res + d).astype(np.float32)
    return {"N1": n1, "N2": n2, "RES": res, "OUT": out}


def reference_chain(blocks: list[dict[str, np.ndarray]]) -> list[dict[str, np.ndarray]]:
    outputs = []
    x = blocks[0]["X"]
    for w in blocks:
        block = reference_block(x, w)
        outputs.append(block)
        x = block["OUT"]
    return outputs


def payload(arrays: dict[str, np.ndarray]) -> bytes:
    streams = {}
    for name, value in arrays.items():
        streams[name] = {
            "kind": "F32",
            "shape": list(value.shape),
            "data": [float(x) for x in value.reshape(-1)],
        }
    return json.dumps({"streams": streams}, separators=(",", ":")).encode()


def add_blob(bundle: Path, data: bytes) -> dict:
    digest = hashlib.sha256(data).hexdigest()
    rel = f"blobs/{digest}"
    dest = bundle / rel
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        dest.write_bytes(data)
    return {"sha256": digest, "bytes": len(data), "path": rel}


def write_bundle(bundle: Path, blocks: list[dict[str, np.ndarray]]) -> Path:
    bundle.mkdir(parents=True, exist_ok=True)
    artifacts = {
        "attn-listing": add_blob(bundle, ATTN.read_bytes()),
        "mlp-listing": add_blob(bundle, MLP.read_bytes()),
    }
    stages = []
    for index, w in enumerate(blocks):
        attn_weights = {key: w[key] for key in ("WQ", "WK", "WV", "WO", "R1", "R2")}
        if index == 0:
            attn_weights["X"] = w["X"]
        mlp_weights = {key: w[key] for key in ("W1", "WUP", "W2")}
        artifacts[f"attn-input-{index}"] = add_blob(bundle, payload(attn_weights))
        artifacts[f"mlp-weights-{index}"] = add_blob(bundle, payload(mlp_weights))
        attn_name = f"attn-{index}"
        mlp_name = f"mlp-{index}"
        attn_payloads = [{"artifact": f"attn-input-{index}"}]
        depends = []
        if index > 0:
            depends = [f"mlp-{index - 1}"]
            attn_payloads.append(
                {
                    "stage": f"mlp-{index - 1}",
                    "output": "OUT",
                    "stream": "X",
                }
            )
        stages.append(
            {
                "name": attn_name,
                "depends_on": depends,
                "executor": {"kind": "gpu", "target": "any"},
                "command": [],
                "inputs": {},
                "outputs": ["N2", "RES"],
                "timeout_ms": 120000,
                "gpu": {
                    "listing": {"artifact": "attn-listing"},
                    "payloads": attn_payloads,
                    "frames": 1,
                    "count": 1,
                },
            }
        )
        stages.append(
            {
                "name": mlp_name,
                "depends_on": [attn_name],
                "executor": {"kind": "gpu", "target": "any"},
                "command": [],
                "inputs": {},
                "outputs": ["OUT"],
                "timeout_ms": 120000,
                "gpu": {
                    "listing": {"artifact": "mlp-listing"},
                    "payloads": [
                        {"artifact": f"mlp-weights-{index}"},
                        {"stage": attn_name, "output": "N2", "stream": "N2"},
                        {"stage": attn_name, "output": "RES", "stream": "RES"},
                    ],
                    "frames": 1,
                    "count": 1,
                },
            }
        )
    manifest = {
        "schema": "jotnar.workload/1",
        "name": "model-block",
        "version": "1",
        "description": "pipeline-parallel transformer blocks: attention on CUDA, MLP on HIP",
        "artifacts": artifacts,
        "stages": stages,
    }
    path = bundle / "bundle.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    return path


def run_bundle(
    bundle: Path,
    ledger: Path,
    cuda: str,
    hip: str,
    token: str,
    blocks: int,
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
    for index in range(blocks):
        cmd += [
            "--place",
            f"attn-{index}={cuda}",
            "--place",
            f"mlp-{index}={hip}",
        ]
    run(cmd)
    return json.loads((ledger / "receipt.json").read_text())


def read_output(ledger: Path, stage: str, name: str, shape: tuple[int, ...]) -> np.ndarray:
    path = ledger / "stages" / stage / name
    return np.fromfile(path, dtype="<f4").reshape(shape)


def compare(got: np.ndarray, want: np.ndarray) -> dict:
    diff = np.abs(got.astype(np.float64) - want.astype(np.float64))
    return {
        "max_abs": float(diff.max()) if diff.size else 0.0,
        "mean_abs": float(diff.mean()) if diff.size else 0.0,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cuda", default=DEFAULT_CUDA)
    ap.add_argument("--hip", default=DEFAULT_HIP)
    ap.add_argument("--token", default="test")
    ap.add_argument("--blocks", type=int, default=2)
    ap.add_argument("--atol", type=float, default=2e-3)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="/tmp/opencode/model-block-gate")
    ap.add_argument("--no-build", action="store_true")
    args = ap.parse_args()

    for addr in (args.cuda, args.hip):
        if not reachable(addr):
            print(f"GPU runner is not reachable at {addr}")
            return 2
    for path in (ATTN, MLP):
        if not path.exists():
            print(f"missing {path}")
            return 2
    if args.blocks < 1:
        print("--blocks must be >= 1")
        return 2
    if not args.no_build:
        build()
    for path in (CONTROLLER, JOTNAR):
        if not path.exists():
            print(f"missing {path}; run without --no-build")
            return 2

    out = Path(args.out).absolute()
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    blocks = [make_inputs(args.seed + index) for index in range(args.blocks)]
    expected = reference_chain(blocks)
    bundle = write_bundle(out / "bundle", blocks)

    ledger = out / "run"
    receipt = run_bundle(
        bundle, ledger, args.cuda, args.hip, args.token, args.blocks
    )
    stages = receipt.get("stages") or {}
    outputs = {
        index: read_output(ledger, f"mlp-{index}", "OUT", (N, D))
        for index in range(args.blocks)
    }
    checks = {
        f"block{index}": compare(outputs[index], expected[index]["OUT"])
        for index in range(args.blocks)
    }
    exact = all(item["max_abs"] <= args.atol for item in checks.values())

    # Determinism: the same bundle must produce byte-identical block outputs.
    ledger2 = out / "run-repeat"
    receipt2 = run_bundle(
        bundle, ledger2, args.cuda, args.hip, args.token, args.blocks
    )
    repeat = {
        f"block{index}": {
            "byte_identical": (
                (ledger / "stages" / f"mlp-{index}" / "OUT").read_bytes()
                == (ledger2 / "stages" / f"mlp-{index}" / "OUT").read_bytes()
            )
        }
        for index in range(args.blocks)
    }
    deterministic = all(item["byte_identical"] for item in repeat.values())

    # Negative control: perturb the last block's WQ. The final output must
    # track the new weights and differ materially from the original reference.
    perturbed = [{k: v.copy() for k, v in block.items()} for block in blocks]
    perturbed[-1]["WQ"] = (perturbed[-1]["WQ"] + 0.5).astype(np.float32)
    control_expected = reference_chain(perturbed)
    control_bundle = write_bundle(out / "control-bundle", perturbed)
    control_ledger = out / "control"
    control_receipt = run_bundle(
        control_bundle, control_ledger, args.cuda, args.hip, args.token, args.blocks
    )
    control_out = read_output(
        control_ledger, f"mlp-{args.blocks - 1}", "OUT", (N, D)
    )
    control_vs_perturbed = compare(control_out, control_expected[-1]["OUT"])
    control_vs_original = compare(control_out, expected[-1]["OUT"])
    control_ok = bool(
        control_vs_perturbed["max_abs"] <= args.atol
        and control_vs_original["max_abs"] > 1e-2
    )

    attn_targets = [
        (stages.get(f"attn-{index}") or {}).get("executor")
        for index in range(args.blocks)
    ]
    mlp_targets = [
        (stages.get(f"mlp-{index}") or {}).get("executor")
        for index in range(args.blocks)
    ]
    both_targets = bool(
        all(target == args.cuda for target in attn_targets)
        and all(target == args.hip for target in mlp_targets)
    )
    done = bool(stages) and all(
        (record or {}).get("state") == "done" for record in stages.values()
    )
    ok = bool(
        receipt.get("ok")
        and done
        and exact
        and deterministic
        and control_ok
        and both_targets
    )

    summary = {
        "schema": "jotnar.model.block_gate/1",
        "ok": ok,
        "cuda": args.cuda,
        "hip": args.hip,
        "model": {
            "kind": "pre-norm transformer block stack, single head",
            "blocks": args.blocks,
            "tokens": N,
            "dim": D,
            "inter": INTER,
            "eps": EPS,
            "seed": args.seed,
            "attention": "CUDA",
            "mlp": "HIP",
        },
        "atol": args.atol,
        "receipt": str(ledger / "receipt.json"),
        "stages": {
            name: {
                "state": (stages.get(name) or {}).get("state"),
                "executor": (stages.get(name) or {}).get("executor"),
                "job_id": (stages.get(name) or {}).get("job_id"),
                "outputs": (stages.get(name) or {}).get("outputs"),
            }
            for name in (
                f"{kind}-{index}"
                for index in range(args.blocks)
                for kind in ("attn", "mlp")
            )
        },
        "checks": checks,
        "exact": exact,
        "determinism": {"ok": deterministic, "outputs": repeat},
        "control": {
            "ok": control_ok,
            "vs_perturbed": control_vs_perturbed,
            "vs_original": control_vs_original,
        },
        "placement": receipt.get("placement"),
        "both_targets": both_targets,
        "evidence": {
            "bundle": str(bundle),
            "ledger": str(ledger),
            "repeat_ledger": str(ledger2),
            "control_bundle": str(control_bundle),
            "control_ledger": str(control_ledger),
            "summary": str(out / "model-block-gate.json"),
        },
    }
    (out / "model-block-gate.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
