#!/usr/bin/env python3
"""`jotnar.stage/1` descriptor gate.

Compiles two upscale descriptors per target (CUDA and HIP), builds a two-stage
chain whose targets and outputs come from those descriptors, runs a
single-node CUDA reference, runs the CUDA -> HIP chain with an intermediate
artifact handoff, and compares both outputs byte-for-byte.

It also proves the rejection path: a stage whose executor target is not in the
descriptor's `targets` must fail before submission and leave a receipt.

No executor learns about the descriptor. The GPU runner still receives a
listing plus payload.
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

ROOT = Path(__file__).resolve().parent.parent
JOTNAR = ROOT / "target" / "release" / "jotnar"
CONTROLLER = ROOT / "target" / "release" / "jotnar-workload"
LISTING = ROOT / "demos" / "doom-two-nodes" / "upscale-half.asm"
DEFAULT_CUDA = "192.168.1.116:9997"
DEFAULT_HIP = "192.168.1.195:9997"


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


def run(cmd: list[str], expect_ok: bool = True) -> subprocess.CompletedProcess:
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    if expect_ok and r.returncode != 0:
        print("command failed:", " ".join(cmd))
        print(r.stdout)
        print(r.stderr)
        raise SystemExit(r.returncode)
    return r


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_shape(path: Path, shape: list[int]) -> None:
    path.write_text(
        json.dumps({"streams": {"X": {"kind": "F32", "shape": shape}}}) + "\n"
    )


def emit_descriptor(out: Path, target: str, name: str, shape: list[int]) -> Path:
    shape_file = out / f"shape-{name}.json"
    write_shape(shape_file, shape)
    work = out / f"build-{name}"
    work.mkdir(parents=True, exist_ok=True)
    descriptor = out / f"{name}.stage.json"
    run(
        [
            str(JOTNAR),
            "build",
            str(LISTING),
            "--payload",
            str(shape_file),
            "--shape-only",
            "--outputs",
            "OUT",
            "--target",
            target,
            "--workdir",
            str(work),
            "--name",
            name,
            "--stage-out",
            str(descriptor),
            "--descriptor-only",
        ]
    )
    value = json.loads(descriptor.read_text())
    if value.get("schema") != "jotnar.stage/1":
        raise SystemExit(f"{descriptor}: bad schema")
    if value.get("targets") != [target]:
        raise SystemExit(f"{descriptor}: expected targets [{target!r}]")
    if value.get("listing_sha256") != sha256(LISTING):
        raise SystemExit(f"{descriptor}: listing hash mismatch")
    return descriptor


def bundle_chain(
    out: Path,
    name: str,
    descriptors: list[Path],
    size: str,
) -> dict:
    dest = out / f"bundle-{name}"
    r = run(
        [
            str(CONTROLLER),
            "bundle-stage-chain",
            "--listing",
            str(LISTING),
            "--out",
            str(dest),
            "--generate",
            size,
            "--name",
            name,
        ]
        + sum((["--descriptor", str(d)] for d in descriptors), [])
    )
    return json.loads(r.stdout)


def run_chain(
    bundle: Path,
    ledger: Path,
    token: str,
    cuda: str,
    hip: str,
    placements: dict[str, str],
    compares: dict[str, Path],
    expect_ok: bool = True,
) -> subprocess.CompletedProcess:
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
    for stage, addr in placements.items():
        cmd += ["--place", f"{stage}={addr}"]
    for key, reference in compares.items():
        cmd += ["--compare", f"{key}={reference}"]
    return run(cmd, expect_ok=expect_ok)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cuda", default=DEFAULT_CUDA)
    ap.add_argument("--hip", default=DEFAULT_HIP)
    ap.add_argument("--token", default="test")
    ap.add_argument("--size", default="200x80x3", help="first-stage HxWxC")
    ap.add_argument("--out", default="/tmp/opencode/fleet-b/stage-descriptor")
    ap.add_argument("--no-build", action="store_true")
    args = ap.parse_args()

    for addr in (args.cuda, args.hip):
        if not reachable(addr):
            print(f"GPU runner is not reachable at {addr}")
            return 2
    if not args.no_build:
        build()
    for path in (JOTNAR, CONTROLLER, LISTING):
        if not path.exists():
            print(f"missing {path}; run without --no-build")
            return 2

    h, _, rest = args.size.partition("x")
    w, _, c = rest.partition("x")
    try:
        shape1 = [int(h), int(w), int(c)]
    except ValueError:
        print("--size needs HxWxC")
        return 2
    if len(shape1) != 3 or any(d <= 0 for d in shape1):
        print("--size needs three positive dims")
        return 2
    shape2 = [shape1[0] * 2, shape1[1] * 2, shape1[2]]

    out = Path(args.out).absolute()
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    d1_cuda = emit_descriptor(out, "cuda", "upscale1-cuda", shape1)
    d1_hip = emit_descriptor(out, "hip", "upscale1-hip", shape1)
    d2_cuda = emit_descriptor(out, "cuda", "upscale2-cuda", shape2)
    d2_hip = emit_descriptor(out, "hip", "upscale2-hip", shape2)

    reference_bundle = bundle_chain(out, "ref-upscale", [d1_cuda, d2_cuda], args.size)
    distributed_bundle = bundle_chain(out, "dist-upscale", [d1_cuda, d2_hip], args.size)
    ref_bundle_path = Path(reference_bundle["bundle"])
    dist_bundle_path = Path(distributed_bundle["bundle"])

    ref_ledger = out / "reference"
    dist_ledger = out / "distributed"
    run_chain(
        ref_bundle_path,
        ref_ledger,
        args.token,
        args.cuda,
        args.hip,
        {
            "ref-upscale-00": args.cuda,
            "ref-upscale-01": args.cuda,
        },
        {},
    )
    ref_receipt = json.loads((ref_ledger / "receipt.json").read_text())
    first = ref_ledger / "stages" / "ref-upscale-00" / "OUT"
    second = ref_ledger / "stages" / "ref-upscale-01" / "OUT"
    if not first.is_file() or not second.is_file():
        print("reference outputs are missing")
        return 1

    run_chain(
        dist_bundle_path,
        dist_ledger,
        args.token,
        args.cuda,
        args.hip,
        {
            "dist-upscale-00": args.cuda,
            "dist-upscale-01": args.hip,
        },
        {
            "dist-upscale-00:OUT": first,
            "dist-upscale-01:OUT": second,
        },
    )
    dist_receipt = json.loads((dist_ledger / "receipt.json").read_text())
    dist_ledger_json = json.loads((dist_ledger / "ledger.json").read_text())

    # Negative control: a CUDA descriptor, stage target relaxed to `any`, placed
    # on HIP. The controller must reject it before any submit.
    bad_bundle = out / "bundle-reject"
    shutil.copytree(dist_bundle_path.parent, bad_bundle)
    bad_json = json.loads((bad_bundle / "bundle.json").read_text())
    for stage in bad_json["stages"]:
        if stage["name"] == "dist-upscale-00":
            stage["executor"]["target"] = "any"
    (bad_bundle / "bundle.json").write_text(json.dumps(bad_json, indent=2) + "\n")
    bad_ledger = out / "rejected"
    bad = run_chain(
        bad_bundle / "bundle.json",
        bad_ledger,
        args.token,
        args.cuda,
        args.hip,
        {
            "dist-upscale-00": args.hip,
            "dist-upscale-01": args.hip,
        },
        {},
        expect_ok=False,
    )
    bad_receipt_path = bad_ledger / "receipt.json"
    bad_receipt = (
        json.loads(bad_receipt_path.read_text()) if bad_receipt_path.is_file() else {}
    )
    bad_stage = (bad_receipt.get("stages") or {}).get("dist-upscale-00") or {}

    expected_sizes = {
        "dist-upscale-00": shape1[0] * 2 * shape1[1] * 2 * shape1[2] * 4,
        "dist-upscale-01": shape2[0] * 2 * shape2[1] * 2 * shape2[2] * 4,
    }
    comparisons_ok = (
        ref_receipt.get("ok") is True
        and dist_receipt.get("ok") is True
        and len(dist_receipt.get("comparisons", [])) == 2
        and all(c.get("match") is True for c in dist_receipt.get("comparisons", []))
    )
    exactness = dist_receipt.get("exactness") or {}
    exactness_ok = all(
        (exactness.get(name) or {}).get("verdict") == "exact"
        for name in ("dist-upscale-00", "dist-upscale-01")
    )
    descriptors = dist_receipt.get("descriptors") or {}
    descriptors_ok = all(
        len((descriptors.get(name) or {}).get("sha256", "")) == 64
        and (descriptors.get(name) or {}).get("target")
        == ("cuda" if name == "dist-upscale-00" else "hip")
        and (descriptors.get(name) or {}).get("job_id")
        for name in ("dist-upscale-00", "dist-upscale-01")
    )
    ledger_ok = all(
        any(
            entry.startswith("descriptor:")
            for entry in (dist_ledger_json.get("stages", {}).get(name) or {}).get(
                "command", []
            )
        )
        for name in ("dist-upscale-00", "dist-upscale-01")
    )
    rejection_ok = (
        bad.returncode != 0
        and bad_stage.get("state") == "descriptor-target-rejected"
        and bad_receipt.get("ok") is False
        and not bad_stage.get("job_id")
    )
    ok = bool(
        comparisons_ok and exactness_ok and descriptors_ok and ledger_ok and rejection_ok
    )

    summary = {
        "schema": "jotnar.stage.descriptor_gate/1",
        "ok": ok,
        "listing": {
            "path": str(LISTING),
            "sha256": sha256(LISTING),
        },
        "descriptors": {
            "stage1_cuda": {
                "path": str(d1_cuda),
                "sha256": sha256(d1_cuda),
                "value": json.loads(d1_cuda.read_text()),
            },
            "stage1_hip": {
                "path": str(d1_hip),
                "sha256": sha256(d1_hip),
                "value": json.loads(d1_hip.read_text()),
            },
            "stage2_cuda": {
                "path": str(d2_cuda),
                "sha256": sha256(d2_cuda),
                "value": json.loads(d2_cuda.read_text()),
            },
            "stage2_hip": {
                "path": str(d2_hip),
                "sha256": sha256(d2_hip),
                "value": json.loads(d2_hip.read_text()),
            },
        },
        "bundles": {
            "reference": reference_bundle,
            "distributed": distributed_bundle,
        },
        "reference": {
            "ok": ref_receipt.get("ok"),
            "outputs": {
                "stage1": {
                    "bytes": first.stat().st_size,
                    "sha256": sha256(first),
                },
                "stage2": {
                    "bytes": second.stat().st_size,
                    "sha256": sha256(second),
                },
            },
        },
        "distributed": {
            "ok": dist_receipt.get("ok"),
            "comparisons": [
                {
                    "stage": c.get("stage"),
                    "output": c.get("output"),
                    "bytes": c.get("bytes"),
                    "match": c.get("match"),
                    "sha256": c.get("sha256"),
                }
                for c in dist_receipt.get("comparisons", [])
            ],
            "descriptors": {
                name: {
                    "sha256": value.get("sha256"),
                    "name": value.get("name"),
                    "targets": value.get("targets"),
                    "target": value.get("target"),
                    "executor": value.get("executor"),
                    "job_id": value.get("job_id"),
                    "ops": value.get("ops"),
                    "determinism": value.get("determinism"),
                }
                for name, value in descriptors.items()
            },
            "exactness": exactness,
            "expected_bytes": expected_sizes,
        },
        "rejection": {
            "exit_code": bad.returncode,
            "state": bad_stage.get("state"),
            "error": bad_receipt.get("error"),
            "job_id": bad_stage.get("job_id"),
            "ok": rejection_ok,
        },
        "checks": {
            "comparisons_ok": comparisons_ok,
            "exactness_ok": exactness_ok,
            "descriptors_ok": descriptors_ok,
            "ledger_records_descriptor_hash": ledger_ok,
            "target_rejection_ok": rejection_ok,
        },
        "evidence": {
            "reference_ledger": str(ref_ledger),
            "distributed_ledger": str(dist_ledger),
            "rejected_ledger": str(bad_ledger),
            "summary": str(out / "stage-descriptor.json"),
        },
    }
    (out / "stage-descriptor.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
