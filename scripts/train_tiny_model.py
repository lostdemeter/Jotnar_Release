#!/usr/bin/env python3
"""Train and export the tiny byte-level model used by model slices 3+.

The architecture is the one the checked-in listings implement:

  tokens [N] -> tok_embed + pos_embed -> X [N, D]
  X -> pre-norm attention block (4 heads, no causal mask) x blocks
    -> final RMSNorm -> logits [N, V]

The training objective is *next byte from a fixed 16-byte context*: the loss is
cross-entropy on the logits at the last position against the byte that follows
the context. This matches the graph exactly -- the graph has no causal mask, so
training a causal LM on it would leak the future. Greedy decode slides the
16-byte window one byte at a time and reruns the whole graph (no KV cache yet).

The corpus is a fixed list of in-repo documentation files. The export is a
`jotnar.model.checkpoint/1` container with F32 tensor blobs, a byte tokenizer,
the prompt fixture, and provenance (corpus hash, steps, losses). This is a tiny
trained model, not a chat model; the report must say so.

Usage:
  python3 scripts/train_tiny_model.py --out demos/model-block/trained
  python3 scripts/train_tiny_model.py --steps 3000 --device cuda --out /tmp/tiny
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = ROOT / "demos" / "model-block" / "trained"
CORPUS_FILES = (
    "README.md",
    "docs/USAGE.md",
    "docs/WORKLOAD_CONTROLLER.md",
    "docs/STREAMING.md",
    "docs/EXECUTOR_CONTRACT.md",
    "docs/LADDER_OPS.md",
    "docs/FLEET_ADMISSION.md",
    "docs/DOOM_TWO_NODE.md",
)

CONFIG = {
    "blocks": 2,
    "d_model": 64,
    "context": 16,
    "heads": 4,
    "head_dim": 16,
    "inter": 128,
    "vocab": 256,
    "eps": 1e-5,
    "pos": "learned",
}


def rmsnorm(x: torch.Tensor, gain: torch.Tensor, eps: float) -> torch.Tensor:
    return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps) * gain


class Block(torch.nn.Module):
    def __init__(self, d: int, heads: int, head_dim: int, inter: int, eps: float):
        super().__init__()
        self.heads = heads
        self.head_dim = head_dim
        self.eps = eps
        self.R1 = torch.nn.Parameter(torch.ones(d))
        self.R2 = torch.nn.Parameter(torch.ones(d))
        self.WQ = torch.nn.Parameter(torch.randn(d, d) * 0.02)
        self.WK = torch.nn.Parameter(torch.randn(d, d) * 0.02)
        self.WV = torch.nn.Parameter(torch.randn(d, d) * 0.02)
        self.WO = torch.nn.Parameter(torch.randn(d, d) * 0.02)
        self.W1 = torch.nn.Parameter(torch.randn(d, inter) * 0.02)
        self.WUP = torch.nn.Parameter(torch.randn(d, inter) * 0.02)
        self.W2 = torch.nn.Parameter(torch.randn(inter, d) * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n1 = rmsnorm(x, self.R1, self.eps)
        q = n1 @ self.WQ
        k = n1 @ self.WK
        v = n1 @ self.WV
        b, n, _ = q.shape
        q = q.view(b, n, self.heads, self.head_dim).transpose(1, 2)
        k = k.view(b, n, self.heads, self.head_dim).transpose(1, 2)
        v = v.view(b, n, self.heads, self.head_dim).transpose(1, 2)
        p = torch.softmax(q @ k.transpose(-1, -2), dim=-1)
        a = (p @ v).transpose(1, 2).reshape(b, n, self.heads * self.head_dim)
        res = x + a @ self.WO
        n2 = rmsnorm(res, self.R2, self.eps)
        m = F.silu(n2 @ self.W1) * (n2 @ self.WUP)
        return res + m @ self.W2


class Tiny(torch.nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.config = dict(config)
        d = int(config["d_model"])
        self.tok = torch.nn.Parameter(torch.randn(int(config["vocab"]), d) * 0.02)
        self.pos = torch.nn.Parameter(torch.randn(int(config["context"]), d) * 0.01)
        self.blocks = torch.nn.ModuleList(
            [
                Block(
                    d,
                    int(config["heads"]),
                    int(config["head_dim"]),
                    int(config["inter"]),
                    float(config["eps"]),
                )
                for _ in range(int(config["blocks"]))
            ]
        )
        self.Rf = torch.nn.Parameter(torch.ones(d))
        self.HEAD = torch.nn.Parameter(torch.randn(d, int(config["vocab"])) * 0.02)
        # GPT-style residual scaling: keep the output projections smaller.
        residual_std = 0.02 / math.sqrt(2 * int(config["blocks"]))
        for block in self.blocks:
            with torch.no_grad():
                block.WO.normal_(0.0, residual_std)
                block.W2.normal_(0.0, residual_std)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        x = self.tok[idx] + self.pos[: idx.shape[1]]
        for block in self.blocks:
            x = block(x)
        x = rmsnorm(x, self.Rf, float(self.config["eps"]))
        return x @ self.HEAD


def load_corpus() -> tuple[bytes, list[dict]]:
    chunks = []
    files = []
    for rel in CORPUS_FILES:
        path = ROOT / rel
        data = path.read_bytes()
        chunks.append(data)
        files.append(
            {
                "path": rel,
                "bytes": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
            }
        )
    corpus = b"\n\n".join(chunks) + b"\n"
    return corpus, files


def make_batches(
    data: torch.Tensor, context: int, count: int, generator: torch.Generator
) -> tuple[torch.Tensor, torch.Tensor]:
    starts = torch.randint(0, data.numel() - context - 1, (count,), generator=generator)
    offsets = starts[:, None] + torch.arange(context + 1)
    windows = data[offsets]
    return windows[:, :context], windows[:, context]


def fixed_batches(
    data: torch.Tensor, context: int, count: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Evenly spaced windows: a stable eval set."""
    span = data.numel() - context - 1
    starts = torch.linspace(0, span - 1, steps=count).long()
    offsets = starts[:, None] + torch.arange(context + 1)
    windows = data[offsets]
    return windows[:, :context], windows[:, context]


@torch.no_grad()
def evaluate(
    model: Tiny, data: torch.Tensor, context: int, count: int
) -> float:
    idx, target = fixed_batches(data, context, count)
    logits = model(idx)
    return float(F.cross_entropy(logits[:, -1, :], target).item())


@torch.no_grad()
def greedy(model: Tiny, prompt: list[int], count: int) -> list[int]:
    context = int(model.config["context"])
    seq = list(prompt)
    if len(seq) != context:
        raise SystemExit(f"prompt must be exactly {context} bytes")
    out = []
    for _ in range(count):
        idx = torch.tensor([seq[-context:]], dtype=torch.long)
        logits = model(idx)
        nxt = int(torch.argmax(logits[0, -1, :]).item())
        out.append(nxt)
        seq.append(nxt)
    return out


def find_prompt(corpus: bytes, context: int, continuation: int, val_start: int) -> tuple[bytes, bytes]:
    """Pick a printable 16-byte prompt (plus continuation) from the val split."""
    end = len(corpus) - context - continuation
    for start in range(val_start + 64, end):
        window = corpus[start : start + context + continuation]
        if all(32 <= byte < 127 for byte in window) and b"  " not in window:
            return window[:context], window[context:]
    start = max(val_start + 64, end // 2)
    return (
        corpus[start : start + context],
        corpus[start + context : start + context + continuation],
    )


def export(
    out: Path,
    model: Tiny,
    config: dict,
    tokens: list[int],
    training: dict,
) -> Path:
    if out.exists():
        shutil.rmtree(out)
    (out / "tensors").mkdir(parents=True)
    tensors = {
        "embedding": model.tok.detach().cpu().numpy(),
        "position_embedding": model.pos.detach().cpu().numpy(),
        "final_norm": model.Rf.detach().cpu().numpy(),
        "head": model.HEAD.detach().cpu().numpy(),
    }
    for index, block in enumerate(model.blocks):
        prefix = f"block{index}."
        for name in ("WQ", "WK", "WV", "WO", "W1", "WUP", "W2", "R1", "R2"):
            value = getattr(block, name).detach().cpu().numpy()
            tensors[prefix + name] = value
    manifest_tensors = {}
    for name, value in tensors.items():
        value = np.asarray(value, dtype="<f4")
        data = value.tobytes()
        rel = f"tensors/{name}.bin"
        (out / rel).write_bytes(data)
        manifest_tensors[name] = {
            "dtype": "F32",
            "shape": list(value.shape),
            "sha256": hashlib.sha256(data).hexdigest(),
            "path": rel,
        }
    manifest = {
        "schema": "jotnar.model.checkpoint/1",
        "name": "jotnar-tiny-byte-2x4",
        "config": config,
        "tokens": tokens,
        "tokenizer": {"kind": "byte", "vocab": int(config["vocab"])},
        "training": training,
        "tensors": manifest_tensors,
    }
    path = out / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    return path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--context", type=int, default=16)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--eval-every", type=int, default=200)
    ap.add_argument("--decode", type=int, default=8)
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device cuda requested but CUDA is unavailable")
    torch.set_num_threads(max(1, args.threads))
    torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(True)
    device = torch.device(args.device)

    corpus, files = load_corpus()
    corpus_sha = hashlib.sha256(corpus).hexdigest()
    data = torch.tensor(list(corpus), dtype=torch.long)
    split = int(len(corpus) * 0.9)
    train_data = data[:split]
    val_data = data[split:]
    config = dict(CONFIG, context=args.context)
    context = int(config["context"])
    if context < 2:
        raise SystemExit("--context must be >= 2")

    prompt, continuation = find_prompt(corpus, context, args.decode, split)
    tokens = list(prompt)

    model = Tiny(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, args.steps), eta_min=args.lr * 0.1
    )
    generator = torch.Generator(device="cpu").manual_seed(args.seed + 1)

    best_val = float("inf")
    best_state = None
    for step in range(1, args.steps + 1):
        model.train()
        idx, target = make_batches(train_data, context, args.batch, generator)
        idx, target = idx.to(device), target.to(device)
        logits = model(idx)
        loss = F.cross_entropy(logits[:, -1, :], target)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        if step == 1 or step % args.eval_every == 0 or step == args.steps:
            val = evaluate(model, val_data, context, 256)
            if val < best_val:
                best_val = val
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                }
            print(
                f"step {step:5d}  train {float(loss.item()):.4f} nats "
                f"({float(loss.item()) / math.log(2):.3f} b/B)  "
                f"val {val:.4f} nats ({val / math.log(2):.3f} b/B)  "
                f"lr {scheduler.get_last_lr()[0]:.2e}",
                flush=True,
            )

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    train_eval = evaluate(model, train_data, context, 512)
    val_eval = evaluate(model, val_data, context, 512)
    generated = greedy(model, tokens, args.decode)
    generated_hex = bytes(generated).hex()
    continuation_hex = continuation.hex()

    training = {
        "method": "next-byte-from-context",
        "context": context,
        "corpus_sha256": corpus_sha,
        "corpus_bytes": len(corpus),
        "corpus_files": files,
        "train_bytes": split,
        "val_bytes": len(corpus) - split,
        "steps": args.steps,
        "batch": args.batch,
        "lr": args.lr,
        "seed": args.seed,
        "device": str(device),
        "threads": args.threads,
        "train_bits_per_byte": train_eval / math.log(2),
        "val_bits_per_byte": val_eval / math.log(2),
        "uniform_bits_per_byte": 8.0,
        "prompt_source": "val",
        "prompt_hex": prompt.hex(),
        "continuation_hex": continuation_hex,
        "greedy_hex": generated_hex,
        "torch_version": torch.__version__,
    }
    out = Path(args.out).absolute()
    manifest = export(
        out,
        model,
        dict(config, tokens=context),
        tokens,
        training,
    )
    (out / "prompt.json").write_text(
        json.dumps(
            {
                "text": prompt.decode("ascii", errors="replace"),
                "bytes": tokens,
                "hex": prompt.hex(),
                "source": "val",
                "continuation_text": continuation.decode("ascii", errors="replace"),
                "continuation_bytes": list(continuation),
                "continuation_hex": continuation_hex,
            },
            indent=2,
        )
        + "\n"
    )
    summary = {
        "ok": True,
        "manifest": str(manifest),
        "bytes": sum(
            (out / entry["path"]).stat().st_size
            for entry in json.loads(manifest.read_text())["tensors"].values()
        ),
        "config": dict(config, tokens=context),
        "training": training,
        "greedy_text": bytes(generated).decode("ascii", errors="replace"),
        "continuation_text": continuation.decode("ascii", errors="replace"),
    }
    (out / "training.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
