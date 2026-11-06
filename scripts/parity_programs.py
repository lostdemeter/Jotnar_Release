#!/usr/bin/env python3
"""Program-level parity: the Python oracle vs the Rust binary.

Runs the two model listings from the reference repo:

* `programs/holo_flagship.asm` — needs `samples/input_example.png` + PIL.
* `programs/xf_block.asm`     — synthetic tensors, needs `stdlib/` IMPORTs.

Every stream is compared. Triples must be bit-exact by plane; floats are
compared with a tight tolerance (numpy may use different SIMD reduction
order for the F paths); ints/bools exactly.

Usage:
  JOTNAR_PY=/path/to/jotnar python3 scripts/parity_programs.py [--bin ...]
"""
import argparse, json, os, subprocess, sys, tempfile
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
def _first_dir(marker, *cands):
    for c in cands:
        if c and os.path.isdir(os.path.join(c, marker)):
            return c
    return None
PHI_CORE = _first_dir("phi_core", os.environ.get("PHI_CORE_DIR"),
                      os.path.join(ROOT, "..", "phi-core"),
                      os.path.join(ROOT, "phi-core"))
JOTNAR_PY = _first_dir("chain", os.environ.get("JOTNAR_PY"),
                       os.path.join(ROOT, "..", "jotnar"),
                       os.path.join(ROOT, "..", "jotnar-main"),
                       ROOT)
if not PHI_CORE or not JOTNAR_PY:
    sys.exit("set PHI_CORE_DIR and JOTNAR_PY to the phi-core / jotnar checkouts")
sys.path.insert(0, PHI_CORE)
sys.path.insert(0, JOTNAR_PY)
import phi_core.lattice as L
from chain import asm as ASM
from chain.asm_ops import REGISTRY, SIGS


def enc(v):
    if isinstance(v, tuple) and len(v) == 3 and hasattr(v[0], "shape"):
        s, e, z = v
        return {"kind": "T", "shape": list(s.shape),
                "s": [int(x) for x in s.reshape(-1)],
                "e": [int(x) for x in e.reshape(-1)],
                "z": [int(x) for x in z.reshape(-1)]}
    a = np.ascontiguousarray(v)
    if a.dtype == bool:
        return {"kind": "Bool", "shape": list(a.shape),
                "data": [bool(x) for x in a.reshape(-1)]}
    if a.dtype == np.uint8:
        return {"kind": "U8", "shape": list(a.shape),
                "data": [int(x) for x in a.reshape(-1)]}
    if np.issubdtype(a.dtype, np.integer):
        return {"kind": "I64", "shape": list(a.shape),
                "data": [str(int(x)) for x in a.reshape(-1)]}
    if a.dtype == np.float32:
        return {"kind": "F32", "shape": list(a.shape),
                "data": [float(x) for x in a.reshape(-1)]}
    return {"kind": "F64", "shape": list(a.shape),
            "data": [float(x) for x in a.reshape(-1)]}


def dec(o):
    k = o["kind"]
    if k == "T":
        sh = tuple(o["shape"])
        return (np.asarray(o["s"], np.int8).reshape(sh),
                np.asarray(o["e"], np.int32).reshape(sh),
                np.asarray(o["z"], np.uint8).reshape(sh))
    if k == "Bool":
        return np.asarray(o["data"], bool)
    if k == "U8":
        return np.asarray(o["data"], np.uint8)
    if k == "I64":
        return np.asarray([int(x) for x in o["data"]], np.int64)
    return np.asarray(o["data"], np.float32 if k == "F32" else np.float64)


def compare(py, rust, tol=1e-12):
    names = [n for n in py if n != "CONFIG"]
    problems = []
    for n in names:
        if n not in rust:
            problems.append(f"{n}: missing from Rust output")
            continue
        a = py[n]
        b = dec(rust[n])
        if isinstance(a, tuple):
            if not all(np.array_equal(x, y) for x, y in zip(a, b)):
                bad = [p for p, (x, y) in enumerate(zip(a, b)) if not np.array_equal(x, y)]
                problems.append(f"{n}: triple planes differ {bad}")
            continue
        aa = np.ascontiguousarray(a)
        if aa.size == b.size and aa.shape != b.shape:
            b = b.reshape(aa.shape)
        if aa.dtype.kind == "f":
            d = np.abs(aa.astype(np.float64) - b.astype(np.float64))
            if d.size and d.max() > tol:
                problems.append(f"{n}: float maxabs={d.max():.3g}")
        elif not np.array_equal(aa, b):
            problems.append(f"{n}: value mismatch")
    return problems


def run_program(binary, repo, text, payload, workdir, name):
    asm_path = os.path.join(workdir, name + ".asm")
    pay_path = os.path.join(workdir, name + ".json")
    out_path = os.path.join(workdir, name + ".out.json")
    open(asm_path, "w").write(text)
    json.dump({"streams": {k: enc(v) for k, v in payload.items()}}, open(pay_path, "w"))
    py = ASM.run_text(text, REGISTRY, payload, sigs=SIGS,
                      basedir=os.path.dirname(asm_path),
                      stdlib=os.path.join(repo, "stdlib"))
    r = subprocess.run(
        [binary, "run", asm_path, "--payload", pay_path, "--out", out_path,
         "--stdlib", os.path.join(repo, "stdlib")],
        capture_output=True, text=True)
    if r.returncode != 0:
        return [f"RUST FAILED: {r.stderr.strip()[:400]}"]
    return compare(py, json.load(open(out_path))["streams"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", default=os.path.join(HERE, "..", "target", "debug", "jotnar"))
    ap.add_argument("--repo", default=JOTNAR_PY)
    args = ap.parse_args()
    binary = os.path.abspath(args.bin)
    repo = os.path.abspath(args.repo)
    fails = 0
    with tempfile.TemporaryDirectory() as td:
        # flagship
        png = os.path.join(repo, "samples", "input_example.png")
        if os.path.exists(png):
            from PIL import Image
            rgb = np.asarray(Image.open(png).convert("RGB"))
            text = open(os.path.join(repo, "programs", "holo_flagship.asm")).read()
            probs = run_program(binary, repo, text, {"rgb": rgb}, td, "holo")
            print("holo_flagship:", "ok" if not probs else probs)
            fails += bool(probs)
        else:
            print("holo_flagship: SKIP (no sample image)")
        # xf block
        rng = np.random.default_rng(0)
        S, D = 4, 8
        def trip(shape):
            return L.encode(rng.normal(0, 0.3, shape).astype(np.float64))
        payload = {
            "x": trip((S, D)), "pos": np.arange(S, dtype=np.int64),
            "wq": trip((D, D)), "wk": trip((D, D)), "wv": trip((D, D)),
            "wo": trip((D, D)), "wup": trip((D, 16)), "wgate": trip((D, 16)),
            "wdown": trip((16, D)), "rms_w1": trip((D,)), "rms_w2": trip((D,)),
        }
        text = open(os.path.join(repo, "programs", "xf_block.asm")).read()
        probs = run_program(binary, repo, text, payload, td, "xf_block")
        print("xf_block:", "ok" if not probs else probs)
        fails += bool(probs)
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
