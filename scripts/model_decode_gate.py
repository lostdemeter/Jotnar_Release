#!/usr/bin/env python3
"""Model slice 3: trained checkpoint, tokenizer, and greedy decode gate.

The controller owns the checkpoint, the byte tokenizer, the prompt, and the
decode policy. It materialises the token embedding plus the learned position
embedding, then sends the existing five-stage sentence graph one listing plus
payloads per decode step:

  window [C] -> embedding+position -> attn-0 -> mlp-0 -> attn-1 -> mlp-1
             -> final norm + head -> logits [C, 256]
  argmax(logits[-1]) -> append -> slide the window -> rerun

The gate checks:

  - the checkpoint manifest and every tensor hash;
  - per-step logits against numpy *and* the torch model that trained it;
  - the generated token sequence is exact;
  - every stage of every step is done;
  - a repeat decode is byte-identical;
  - the pipeline placement and an all-HIP placement both match the reference
    (the HIP attention parity check);
  - a perturbed-head control tracks the perturbed reference and diverges from
    the original beyond the measured noise floor.

This is a tiny next-byte model, not a chat model. When the checkpoint has no
`training` block the gate still runs and labels it a fixture; it must not be
reported as trained.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import model_block_gate as mb  # noqa: E402
import model_common as mc  # noqa: E402
import model_sentence_gate as msg  # noqa: E402

DEFAULT_CHECKPOINT = ROOT / "demos" / "model-block" / "trained"


def load_prompt(path: Path | None, checkpoint: dict) -> tuple[list[int], dict]:
    context = int(checkpoint["config"]["context"])
    if path is None:
        tokens = list(checkpoint["tokens"])
        fixture = {}
    else:
        fixture = json.loads(path.read_text())
        if "bytes" in fixture:
            tokens = [int(x) for x in fixture["bytes"]]
        else:
            tokens = list(str(fixture["text"]).encode("utf-8"))
    if len(tokens) != context:
        raise SystemExit(
            f"prompt has {len(tokens)} tokens; the checkpoint context is {context}"
        )
    return tokens, fixture


def numpy_reference(checkpoint: dict, prompt: list[int], decode: int) -> dict:
    context = int(checkpoint["config"]["context"])
    vocab = int(checkpoint["config"]["vocab"])
    seq = list(prompt)
    steps = []
    for _ in range(decode):
        window = seq[-context:]
        x = mc.encode_x(checkpoint, window)
        _, logits, _ = mc.reference_from_x(checkpoint, x)
        last = logits[-1]
        order = np.argsort(last)[::-1]
        nxt = int(order[0])
        steps.append(
            {
                "logits": logits,
                "argmax": nxt,
                "top": [int(v) for v in order[:5]],
                "margin": float(last[order[0]] - last[order[1]]),
            }
        )
        seq.append(nxt)
    return {"sequence": seq, "steps": steps}


def load_torch(checkpoint_dir: Path, checkpoint: dict):
    try:
        import torch  # noqa: F401
    except ImportError as exc:  # pragma: no cover - torch is part of the gate stack
        raise SystemExit(f"the torch reference needs torch: {exc}") from exc
    import train_tiny_model as tt

    model = tt.Tiny(checkpoint["config"])
    state = {
        "tok": checkpoint["tensors"]["embedding"],
        "pos": checkpoint["tensors"]["position_embedding"],
        "Rf": checkpoint["tensors"]["final_norm"],
        "HEAD": checkpoint["tensors"]["head"],
    }
    for index in range(int(checkpoint["config"]["blocks"])):
        for name in ("WQ", "WK", "WV", "WO", "W1", "WUP", "W2", "R1", "R2"):
            state[f"blocks.{index}.{name}"] = checkpoint["tensors"][
                f"block{index}.{name}"
            ]
    import torch

    model.load_state_dict({key: torch.tensor(value) for key, value in state.items()})
    model.eval()
    return model


def torch_reference(model, prompt: list[int], decode: int) -> dict:
    import torch

    context = int(model.config["context"])
    seq = list(prompt)
    steps = []
    with torch.no_grad():
        for _ in range(decode):
            window = seq[-context:]
            idx = torch.tensor([window], dtype=torch.long)
            logits = model(idx)[0].numpy().astype(np.float32)
            last = logits[-1]
            order = np.argsort(last)[::-1]
            nxt = int(order[0])
            steps.append(
                {
                    "logits": logits,
                    "argmax": nxt,
                    "top": [int(v) for v in order[:5]],
                    "margin": float(last[order[0]] - last[order[1]]),
                }
            )
            seq.append(nxt)
    return {"sequence": seq, "steps": steps}


def run_decode(
    checkpoint: dict,
    prompt: list[int],
    decode: int,
    kind: str,
    root: Path,
    cuda: str,
    hip: str,
    token: str,
    label: str,
) -> dict:
    context = int(checkpoint["config"]["context"])
    vocab = int(checkpoint["config"]["vocab"])
    blocks = int(checkpoint["config"]["blocks"])
    seq = list(prompt)
    steps = []
    receipts = []
    stage_states = {}
    for index in range(decode):
        window = seq[-context:]
        x = mc.encode_x(checkpoint, window)
        step_dir = root / label / f"step-{index:02d}"
        bundle = step_dir / "bundle"
        ledger = step_dir / "run"
        msg.write_bundle(bundle, checkpoint, x=x)
        receipt = mc.run_placed(bundle, ledger, cuda, hip, token, blocks, kind)
        logits = mc.read_output(ledger, "head", "LOGITS", (len(window), vocab))
        np.save(step_dir / "logits.npy", logits)
        last = logits[-1]
        order = np.argsort(last)[::-1]
        nxt = int(order[0])
        states = {
            name: (record or {}).get("state")
            for name, record in (receipt.get("stages") or {}).items()
        }
        stage_states[index] = states
        steps.append(
            {
                "window_hex": bytes(window).hex(),
                "argmax": nxt,
                "top": [int(v) for v in order[:5]],
                "margin": float(last[order[0]] - last[order[1]]),
                "logits": logits,
                "logits_path": str(step_dir / "logits.npy"),
                "receipt": str(ledger / "receipt.json"),
                "executors": {
                    name: (record or {}).get("executor")
                    for name, record in (receipt.get("stages") or {}).items()
                },
            }
        )
        seq.append(nxt)
        receipts.append(str(ledger / "receipt.json"))
    return {
        "placement": kind,
        "sequence": seq,
        "steps": steps,
        "receipts": receipts,
        "stage_states": stage_states,
        "utilization": mc.utilization(json.loads(Path(receipts[-1]).read_text())),
    }


def continuation_report(fixture: dict, generated: list[int]) -> dict | None:
    """Quality signal only: how much of the true val continuation was guessed."""
    raw = fixture.get("continuation_hex")
    if not raw:
        return None
    want = list(bytes.fromhex(raw))
    count = min(len(want), len(generated))
    prefix = 0
    while prefix < count and want[prefix] == generated[prefix]:
        prefix += 1
    return {
        "hex": raw,
        "text": fixture.get("continuation_text"),
        "first_byte_match": bool(count and want[0] == generated[0]),
        "prefix_matches": prefix,
        "position_matches": sum(
            1 for index in range(count) if want[index] == generated[index]
        ),
        "compared": count,
    }


def compare_logits(a: list[dict], b: list[dict]) -> dict:
    full = max(float(np.abs(x["logits"] - y["logits"]).max()) for x, y in zip(a, b))
    last = max(
        float(np.abs(x["logits"][-1] - y["logits"][-1]).max()) for x, y in zip(a, b)
    )
    return {"max_abs_full": full, "max_abs_last": last}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    ap.add_argument("--prompt", default=None, help="prompt JSON (else checkpoint tokens)")
    ap.add_argument("--decode", type=int, default=8)
    ap.add_argument("--cuda", default=mc.DEFAULT_CUDA)
    ap.add_argument("--hip", default=mc.DEFAULT_HIP)
    ap.add_argument("--token", default="test")
    ap.add_argument("--atol", type=float, default=1e-4)
    ap.add_argument(
        "--placements",
        default="pipeline,hip",
        help="comma-separated: pipeline,hip,single-cuda,single-hip",
    )
    ap.add_argument("--no-build", action="store_true")
    ap.add_argument("--out", default="/tmp/opencode/model-decode-gate")
    args = ap.parse_args()
    if args.decode < 1:
        raise SystemExit("--decode must be >= 1")

    for addr in (args.cuda, args.hip):
        if not mc.reachable(addr):
            print(f"GPU runner is not reachable at {addr}")
            return 2
    if not args.no_build:
        mc.build()
    for path in (mc.CONTROLLER, mc.JOTNAR):
        if not path.exists():
            print(f"missing {path}; run without --no-build")
            return 2

    checkpoint_dir = Path(args.checkpoint).absolute()
    checkpoint = mc.load_checkpoint(checkpoint_dir)
    blocks = int(checkpoint["config"]["blocks"])
    vocab = int(checkpoint["config"]["vocab"])
    context = int(checkpoint["config"]["context"])
    trained = bool(checkpoint.get("training"))
    prompt_path = Path(args.prompt) if args.prompt else None
    if prompt_path is None and (checkpoint_dir / "prompt.json").exists():
        prompt_path = checkpoint_dir / "prompt.json"
    prompt, prompt_fixture = load_prompt(prompt_path, checkpoint)

    out = Path(args.out).absolute()
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    np_ref = numpy_reference(checkpoint, prompt, args.decode)
    torch_model = load_torch(checkpoint_dir, checkpoint)
    torch_ref = torch_reference(torch_model, prompt, args.decode)
    references = {
        "numpy_vs_torch": compare_logits(np_ref["steps"], torch_ref["steps"]),
        "numpy_sequence_hex": bytes(np_ref["sequence"]).hex(),
        "torch_sequence_hex": bytes(torch_ref["sequence"]).hex(),
        "sequence_match": np_ref["sequence"] == torch_ref["sequence"],
    }

    placements = [item.strip() for item in args.placements.split(",") if item.strip()]
    runs = {}
    for kind in placements:
        runs[kind] = run_decode(
            checkpoint,
            prompt,
            args.decode,
            kind,
            out,
            args.cuda,
            args.hip,
            args.token,
            f"run-{kind}",
        )
    repeat = run_decode(
        checkpoint,
        prompt,
        args.decode,
        placements[0],
        out,
        args.cuda,
        args.hip,
        args.token,
        "run-repeat",
    )

    control = copy.deepcopy(checkpoint)
    control["tensors"]["block0.WQ"] = (
        control["tensors"]["block0.WQ"].astype(np.float32) + 0.5
    ).astype(np.float32)
    control_ref = numpy_reference(control, prompt, args.decode)
    control_run = run_decode(
        control,
        prompt,
        args.decode,
        placements[0],
        out,
        args.cuda,
        args.hip,
        args.token,
        "run-control",
    )

    checks = {}
    for kind, run in runs.items():
        checks[kind] = {
            "sequence_exact": run["sequence"] == np_ref["sequence"],
            "logits_numpy": compare_logits(run["steps"], np_ref["steps"]),
            "logits_torch": compare_logits(run["steps"], torch_ref["steps"]),
            "all_stages_done": all(
                state == "done"
                for states in run["stage_states"].values()
                for state in states.values()
            ),
        }
    repeated = runs[placements[0]]
    repeat_ok = (
        repeated["sequence"] == repeat["sequence"]
        and all(
            (Path(a["logits_path"]).read_bytes() == Path(b["logits_path"]).read_bytes())
            for a, b in zip(repeated["steps"], repeat["steps"])
        )
    )
    noise = compare_logits(repeated["steps"], repeat["steps"])["max_abs_full"]
    control_vs_ref = compare_logits(control_run["steps"], control_ref["steps"])
    control_vs_original = compare_logits(control_run["steps"], np_ref["steps"])
    divergence = max(1e-3, 100.0 * noise)
    control_ok = bool(
        control_vs_ref["max_abs_full"] <= args.atol
        and (
            control_run["sequence"] != np_ref["sequence"]
            or control_vs_original["max_abs_full"] > divergence
        )
    )

    logits_ok = all(
        item["logits_numpy"]["max_abs_full"] <= args.atol
        and item["logits_torch"]["max_abs_full"] <= args.atol
        for item in checks.values()
    )
    ok = bool(
        references["sequence_match"]
        and logits_ok
        and all(
            item["sequence_exact"] and item["all_stages_done"]
            for item in checks.values()
        )
        and repeat_ok
        and control_ok
    )

    summary = {
        "schema": "jotnar.model.decode_gate/1",
        "ok": ok,
        "cuda": args.cuda,
        "hip": args.hip,
        "checkpoint": {
            "path": str(checkpoint_dir),
            "name": checkpoint.get("name"),
            "sha256": checkpoint["sha256"],
            "schema": "jotnar.model.checkpoint/1",
            "kind": "trained" if trained else "fixture",
            "config": checkpoint["config"],
            "tokenizer": checkpoint.get("tokenizer"),
            "training": checkpoint.get("training"),
        },
        "prompt": {
            "path": str(prompt_path) if prompt_path else None,
            "hex": bytes(prompt).hex(),
            "text": bytes(prompt).decode("utf-8", errors="replace"),
            "source": prompt_fixture.get("source"),
            "continuation_hex": prompt_fixture.get("continuation_hex"),
            "continuation_text": prompt_fixture.get("continuation_text"),
        },
        "decode": {
            "steps": args.decode,
            "sequence": np_ref["sequence"],
            "sequence_hex": bytes(np_ref["sequence"]).hex(),
            "sequence_text": bytes(np_ref["sequence"]).decode("utf-8", errors="replace"),
            "generated_hex": bytes(np_ref["sequence"][-args.decode :]).hex(),
            "generated_text": bytes(np_ref["sequence"][-args.decode :]).decode(
                "utf-8", errors="replace"
            ),
            "continuation": continuation_report(
                prompt_fixture, np_ref["sequence"][-args.decode :]
            ),
            "per_step": [
                {
                    "step": index,
                    "argmax": step["argmax"],
                    "top": step["top"],
                    "margin": step["margin"],
                    "numpy_vs_torch": float(
                        np.abs(
                            np_ref["steps"][index]["logits"]
                            - torch_ref["steps"][index]["logits"]
                        ).max()
                    ),
                }
                for index, step in enumerate(np_ref["steps"])
            ],
        },
        "atol": args.atol,
        "references": references,
        "placements": {
            kind: {
                "sequence_hex": bytes(run["sequence"]).hex(),
                "steps": [
                    {
                        key: value
                        for key, value in step.items()
                        if key not in ("logits", "window_hex")
                    }
                    for step in run["steps"]
                ],
                "utilization": run["utilization"],
            }
            for kind, run in runs.items()
        },
        "checks": checks,
        "determinism": {
            "ok": repeat_ok,
            "noise_floor_max_abs": noise,
            "repeat_sequence_hex": bytes(repeat["sequence"]).hex(),
        },
        "control": {
            "ok": control_ok,
            "perturbation": "block0.WQ + 0.5",
            "vs_perturbed_reference": control_vs_ref,
            "vs_original": control_vs_original,
            "divergence_threshold": divergence,
            "sequence_changed": control_run["sequence"] != np_ref["sequence"],
            "sequence_hex": bytes(control_run["sequence"]).hex(),
        },
        "evidence": {
            "root": str(out),
            "summary": str(out / "model-decode-gate.json"),
        },
    }
    (out / "model-decode-gate.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
