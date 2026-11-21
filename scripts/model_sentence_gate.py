#!/usr/bin/env python3
"""Model flagship slice 2: checkpoint + multi-head + full sentence.

A two-block, four-head pre-norm transformer sentence:

  tokens -> embedding (controller) -> X
  X -> attn-0 (CUDA) -> mlp-0 (HIP) -> attn-1 (CUDA) -> mlp-1 (HIP)
    -> final RMSNorm + head (CUDA) -> logits [8,32]

The controller loads a `jotnar.model.checkpoint/1` manifest and F32 tensor
blobs, materializes the token embedding, and sends each stage one listing plus
payloads. The executors see no model manifest or weights.

The checkpoint here is a deterministic F32 fixture (the tree has no trained F32
transformer checkpoint); the container, loader, placement, and reference paths
are real. A trained checkpoint can replace the fixture without changing the
graph. The reference is numpy.
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
sys.path.insert(0, str(ROOT / "scripts"))

import model_block_gate as mb  # noqa: E402

CONTROLLER = ROOT / "target" / "release" / "jotnar-workload"
JOTNAR = ROOT / "target" / "release" / "jotnar"
ATTN4 = ROOT / "demos" / "model-block" / "attn4.asm"
MLP = ROOT / "demos" / "model-block" / "mlp.asm"
HEAD = ROOT / "demos" / "model-block" / "head.asm"
DEFAULT_CUDA = "192.168.1.116:9997"
DEFAULT_HIP = "192.168.1.195:9997"

D = 64
N = 8
HEADS = 4
DH = 16
INTER = 128
VOCAB = 32
EPS = 1e-5
TOKENS = [1, 4, 2, 8, 5, 3, 7, 6]


def make_checkpoint(seed: int, blocks: int) -> dict:
    rng = np.random.default_rng(seed)
    scale = 1.0 / np.sqrt(DH)
    tensors = {
        "embedding": rng.uniform(-0.3, 0.3, (VOCAB, D)).astype(np.float32),
        "final_norm": np.ones(D, dtype=np.float32),
        "head": rng.uniform(-0.05, 0.05, (D, VOCAB)).astype(np.float32),
    }
    for index in range(blocks):
        prefix = f"block{index}."
        tensors[prefix + "WQ"] = (
            rng.uniform(-0.02, 0.02, (D, D)) * scale
        ).astype(np.float32)
        for name in ("WK", "WV", "WO"):
            tensors[prefix + name] = rng.uniform(-0.02, 0.02, (D, D)).astype(
                np.float32
            )
        tensors[prefix + "W1"] = rng.uniform(-0.02, 0.02, (D, INTER)).astype(
            np.float32
        )
        tensors[prefix + "WUP"] = rng.uniform(-0.02, 0.02, (D, INTER)).astype(
            np.float32
        )
        tensors[prefix + "W2"] = rng.uniform(-0.02, 0.02, (INTER, D)).astype(
            np.float32
        )
        tensors[prefix + "R1"] = np.ones(D, dtype=np.float32)
        tensors[prefix + "R2"] = np.ones(D, dtype=np.float32)
    return {
        "config": {
            "blocks": blocks,
            "d_model": D,
            "tokens": N,
            "heads": HEADS,
            "head_dim": DH,
            "inter": INTER,
            "vocab": VOCAB,
            "eps": EPS,
            "seed": seed,
        },
        "tokens": TOKENS,
        "tensors": tensors,
    }


def write_checkpoint(directory: Path, checkpoint: dict) -> Path:
    if directory.exists():
        shutil.rmtree(directory)
    (directory / "tensors").mkdir(parents=True)
    manifest_tensors = {}
    for name, value in checkpoint["tensors"].items():
        data = value.astype("<f4").tobytes()
        rel = f"tensors/{name}.bin"
        (directory / rel).write_bytes(data)
        manifest_tensors[name] = {
            "dtype": "F32",
            "shape": list(value.shape),
            "sha256": hashlib.sha256(data).hexdigest(),
            "path": rel,
        }
    manifest = {
        "schema": "jotnar.model.checkpoint/1",
        "name": "tiny-transformer-fixture",
        "config": checkpoint["config"],
        "tokens": checkpoint["tokens"],
        "tensors": manifest_tensors,
    }
    path = directory / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    return path


def load_checkpoint(directory: Path) -> dict:
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != "jotnar.model.checkpoint/1":
        raise SystemExit("checkpoint schema is not jotnar.model.checkpoint/1")
    tensors = {}
    for name, entry in manifest["tensors"].items():
        path = directory / entry["path"]
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != entry["sha256"]:
            raise SystemExit(f"checkpoint tensor {name} hash mismatch")
        if entry["dtype"] != "F32":
            raise SystemExit(f"checkpoint tensor {name} is not F32")
        values = np.frombuffer(data, dtype="<f4")
        shape = tuple(int(x) for x in entry["shape"])
        if values.size != int(np.prod(shape)):
            raise SystemExit(f"checkpoint tensor {name} size mismatch")
        tensors[name] = values.reshape(shape).copy()
    return {
        "config": manifest["config"],
        "tokens": manifest["tokens"],
        "tensors": tensors,
        "sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
    }


def block_weights(checkpoint: dict, index: int) -> dict[str, np.ndarray]:
    prefix = f"block{index}."
    return {
        key: checkpoint["tensors"][prefix + key]
        for key in ("WQ", "WK", "WV", "WO", "W1", "WUP", "W2", "R1", "R2")
    }


def reference_block(x: np.ndarray, w: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    n1 = mb.rmsnorm(x, w["R1"])
    q = n1 @ w["WQ"]
    k = n1 @ w["WK"]
    v = n1 @ w["WV"]
    heads = []
    for head in range(HEADS):
        start = head * DH
        stop = start + DH
        scores = q[:, start:stop] @ k[:, start:stop].T
        probs = mb.softmax(scores)
        heads.append(probs @ v[:, start:stop])
    o = np.concatenate(heads, axis=1) @ w["WO"]
    res = (x + o).astype(np.float32)
    n2 = mb.rmsnorm(res, w["R2"])
    g = n2 @ w["W1"]
    u = n2 @ w["WUP"]
    m = ((g * (1.0 / (1.0 + np.exp(-g)))) * u).astype(np.float32)
    d = m @ w["W2"]
    out = (res + d).astype(np.float32)
    return {"N2": n2, "RES": res, "OUT": out}


def reference(checkpoint: dict) -> tuple[list[dict], np.ndarray, np.ndarray]:
    tokens = checkpoint["tokens"]
    x = checkpoint["tensors"]["embedding"][tokens].astype(np.float32)
    outputs = []
    for index in range(checkpoint["config"]["blocks"]):
        block = reference_block(x, block_weights(checkpoint, index))
        outputs.append(block)
        x = block["OUT"]
    final = mb.rmsnorm(x, checkpoint["tensors"]["final_norm"])
    logits = (final @ checkpoint["tensors"]["head"]).astype(np.float32)
    return outputs, logits, x


def add_blob(bundle: Path, data: bytes) -> dict:
    digest = hashlib.sha256(data).hexdigest()
    rel = f"blobs/{digest}"
    dest = bundle / rel
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        dest.write_bytes(data)
    return {"sha256": digest, "bytes": len(data), "path": rel}


def write_bundle(bundle: Path, checkpoint: dict) -> Path:
    bundle.mkdir(parents=True, exist_ok=True)
    blocks = checkpoint["config"]["blocks"]
    tokens = checkpoint["tokens"]
    x = checkpoint["tensors"]["embedding"][tokens].astype(np.float32)
    artifacts = {
        "attn4-listing": add_blob(bundle, ATTN4.read_bytes()),
        "mlp-listing": add_blob(bundle, MLP.read_bytes()),
        "head-listing": add_blob(bundle, HEAD.read_bytes()),
    }
    stages = []
    for index in range(blocks):
        prefix = f"block{index}."
        tensors = checkpoint["tensors"]
        attn_payload = {
            key: tensors[prefix + key]
            for key in ("WQ", "WK", "WV", "WO", "R1", "R2")
        }
        if index == 0:
            attn_payload["X"] = x
        mlp_payload = {
            key: tensors[prefix + key] for key in ("W1", "WUP", "W2")
        }
        artifacts[f"attn-weights-{index}"] = add_blob(
            bundle, mb.payload(attn_payload)
        )
        artifacts[f"mlp-weights-{index}"] = add_blob(
            bundle, mb.payload(mlp_payload)
        )
        attn_name = f"attn-{index}"
        mlp_name = f"mlp-{index}"
        attn_payloads = [{"artifact": f"attn-weights-{index}"}]
        depends = []
        if index > 0:
            depends = [f"mlp-{index - 1}"]
            attn_payloads.append(
                {"stage": f"mlp-{index - 1}", "output": "OUT", "stream": "X"}
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
                    "listing": {"artifact": "attn4-listing"},
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
    head_payload = {
        "R": checkpoint["tensors"]["final_norm"],
        "HEAD": checkpoint["tensors"]["head"],
    }
    artifacts["head-weights"] = add_blob(bundle, mb.payload(head_payload))
    stages.append(
        {
            "name": "head",
            "depends_on": [f"mlp-{blocks - 1}"],
            "executor": {"kind": "gpu", "target": "any"},
            "command": [],
            "inputs": {},
            "outputs": ["LOGITS"],
            "timeout_ms": 120000,
            "gpu": {
                "listing": {"artifact": "head-listing"},
                "payloads": [
                    {"artifact": "head-weights"},
                    {
                        "stage": f"mlp-{blocks - 1}",
                        "output": "OUT",
                        "stream": "X",
                    },
                ],
                "frames": 1,
                "count": 1,
            },
        }
    )
    manifest = {
        "schema": "jotnar.workload/1",
        "name": "model-sentence",
        "version": "1",
        "description": "four-head transformer sentence: attention/head on CUDA, MLP on HIP",
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
    cmd += ["--place", f"head={cuda}"]
    mb.run(cmd)
    return json.loads((ledger / "receipt.json").read_text())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cuda", default=DEFAULT_CUDA)
    ap.add_argument("--hip", default=DEFAULT_HIP)
    ap.add_argument("--token", default="test")
    ap.add_argument("--blocks", type=int, default=2)
    ap.add_argument("--atol", type=float, default=2e-3)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="/tmp/opencode/model-sentence-gate")
    ap.add_argument("--no-build", action="store_true")
    args = ap.parse_args()

    for addr in (args.cuda, args.hip):
        if not mb.reachable(addr):
            print(f"GPU runner is not reachable at {addr}")
            return 2
    for path in (ATTN4, MLP, HEAD):
        if not path.exists():
            print(f"missing {path}")
            return 2
    if args.blocks < 1:
        print("--blocks must be >= 1")
        return 2
    if not args.no_build:
        mb.build()
    for path in (CONTROLLER, JOTNAR):
        if not path.exists():
            print(f"missing {path}; run without --no-build")
            return 2

    out = Path(args.out).absolute()
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    fixture = make_checkpoint(args.seed, args.blocks)
    checkpoint_path = write_checkpoint(out / "checkpoint", fixture)
    checkpoint = load_checkpoint(out / "checkpoint")
    expected_blocks, expected_logits, _ = reference(checkpoint)
    bundle = write_bundle(out / "bundle", checkpoint)

    ledger = out / "run"
    receipt = run_bundle(
        bundle, ledger, args.cuda, args.hip, args.token, args.blocks
    )
    stages = receipt.get("stages") or {}
    logits = mb.read_output(ledger, "head", "LOGITS", (N, VOCAB))
    block_checks = {
        f"block{index}": mb.compare(
            mb.read_output(ledger, f"mlp-{index}", "OUT", (N, D)),
            expected_blocks[index]["OUT"],
        )
        for index in range(args.blocks)
    }
    logits_check = mb.compare(logits, expected_logits)
    checks = dict(block_checks)
    checks["logits"] = logits_check
    exact = all(item["max_abs"] <= args.atol for item in checks.values())

    ledger2 = out / "run-repeat"
    receipt2 = run_bundle(
        bundle, ledger2, args.cuda, args.hip, args.token, args.blocks
    )
    repeated = (
        (ledger / "stages" / "head" / "LOGITS").read_bytes()
        == (ledger2 / "stages" / "head" / "LOGITS").read_bytes()
    )
    blocks_repeated = all(
        (ledger / "stages" / f"mlp-{index}" / "OUT").read_bytes()
        == (ledger2 / "stages" / f"mlp-{index}" / "OUT").read_bytes()
        for index in range(args.blocks)
    )

    control = json.loads(json.dumps(fixture, default=lambda x: x.tolist()))
    control["tensors"]["head"] = (
        np.asarray(control["tensors"]["head"], dtype=np.float32) + 0.5
    ).tolist()
    control_tensors = {
        name: np.asarray(value, dtype=np.float32)
        for name, value in control["tensors"].items()
    }
    control_checkpoint = {
        "config": control["config"],
        "tokens": control["tokens"],
        "tensors": control_tensors,
        "sha256": "control",
    }
    _, control_expected_logits, _ = reference(control_checkpoint)
    control_bundle = write_bundle(out / "control-bundle", control_checkpoint)
    control_ledger = out / "control"
    control_receipt = run_bundle(
        control_bundle, control_ledger, args.cuda, args.hip, args.token, args.blocks
    )
    control_logits = mb.read_output(control_ledger, "head", "LOGITS", (N, VOCAB))
    control_vs_perturbed = mb.compare(control_logits, control_expected_logits)
    control_vs_original = mb.compare(control_logits, expected_logits)
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
    head_target = (stages.get("head") or {}).get("executor")
    both_targets = bool(
        all(target == args.cuda for target in attn_targets)
        and all(target == args.hip for target in mlp_targets)
        and head_target == args.cuda
    )
    done = bool(stages) and all(
        (record or {}).get("state") == "done" for record in stages.values()
    )
    deterministic = bool(repeated and blocks_repeated)
    ok = bool(
        receipt.get("ok")
        and done
        and exact
        and deterministic
        and control_ok
        and both_targets
    )

    summary = {
        "schema": "jotnar.model.sentence_gate/1",
        "ok": ok,
        "cuda": args.cuda,
        "hip": args.hip,
        "checkpoint": {
            "schema": "jotnar.model.checkpoint/1",
            "manifest": str(checkpoint_path),
            "sha256": checkpoint["sha256"],
            "config": checkpoint["config"],
            "tokens": checkpoint["tokens"],
            "note": "deterministic F32 fixture; replace with a trained checkpoint",
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
            for name in [f"{kind}-{index}" for index in range(args.blocks) for kind in ("attn", "mlp")]
            + ["head"]
        },
        "checks": checks,
        "exact": exact,
        "determinism": {"ok": deterministic, "logits": repeated, "blocks": blocks_repeated},
        "control": {
            "ok": control_ok,
            "vs_perturbed": control_vs_perturbed,
            "vs_original": control_vs_original,
        },
        "placement": receipt.get("placement"),
        "both_targets": both_targets,
        "evidence": {
            "checkpoint": str(out / "checkpoint"),
            "bundle": str(bundle),
            "ledger": str(ledger),
            "repeat_ledger": str(ledger2),
            "control_bundle": str(control_bundle),
            "control_ledger": str(control_ledger),
            "summary": str(out / "model-sentence-gate.json"),
        },
    }
    (out / "model-sentence-gate.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
