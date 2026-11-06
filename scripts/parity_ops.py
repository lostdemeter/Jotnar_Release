#!/usr/bin/env python3
"""Differential parity harness: Python Jotnar (oracle) vs the Rust binary.

Each case is a listing plus a payload. The Python interpreter runs it,
the Rust binary runs the same text, and streams are compared. Triples are
compared bit-exactly by plane; integers/bools exactly; floats with a
tolerance (0 = exact).

Usage: python3 scripts/parity_ops.py [--bin target/debug/jotnar] [filter]
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
        return (np.asarray(o["s"], np.int8), np.asarray(o["e"], np.int32),
                np.asarray(o["z"], np.uint8))
    if k == "Bool":
        return np.asarray(o["data"], bool)
    if k == "U8":
        return np.asarray(o["data"], np.uint8)
    if k == "I64":
        return np.asarray([int(x) for x in o["data"]], np.int64)
    if k == "F32":
        return np.asarray(o["data"], np.float32)
    return np.asarray(o["data"], np.float64)

class Case:
    def __init__(self, name, text, payload, outputs=None, tol=0.0):
        self.name, self.text, self.payload = name, text, payload
        self.outputs, self.tol = outputs, tol

CASES = []
def case(name, text, payload, outputs=None, tol=1e-12):
    CASES.append(Case(name, text, payload, outputs, tol))

def tri(a):
    return L.encode(np.ascontiguousarray(a, dtype=np.float64))

def run_case(c: Case, binary, workdir):
    asm_path = os.path.join(workdir, c.name + ".asm")
    pay_path = os.path.join(workdir, c.name + ".json")
    out_path = os.path.join(workdir, c.name + ".out.json")
    open(asm_path, "w").write(c.text)
    json.dump({"streams": {k: enc(v) for k, v in c.payload.items()}}, open(pay_path, "w"))
    py = ASM.run_text(c.text, REGISTRY, c.payload, sigs=SIGS,
                      basedir=workdir, stdlib=os.path.join(JOTNAR_PY, "stdlib"))
    args = [binary, "run", asm_path, "--payload", pay_path, "--out", out_path,
            "--stdlib", os.path.join(JOTNAR_PY, "stdlib")]
    if c.outputs:
        args += ["--outputs", ",".join(c.outputs)]
    r = subprocess.run(args, capture_output=True, text=True)
    if r.returncode != 0:
        return f"RUST FAILED: {r.stderr.strip()[:300]}"
    got = json.load(open(out_path))["streams"]
    names = c.outputs or [n for n in py if n != "CONFIG"]
    for n in names:
        if n not in got:
            return f"missing stream {n}"
        a = dec({"kind": "T", **{k: enc(py[n])[k] for k in ("s","e","z")}}) if isinstance(py[n], tuple) else py[n]
        b = dec(got[n])
        if isinstance(py[n], tuple):
            if not all(np.array_equal(x, y) for x, y in zip(a, b)):
                return f"{n}: triple mismatch"
            continue
        a = np.ascontiguousarray(py[n])
        if b.size == a.size and b.shape != a.shape:
            b = b.reshape(a.shape)
        if a.dtype.kind == "f":
            d = np.abs(np.asarray(a, np.float64) - np.asarray(b, np.float64))
            if d.size and d.max() > c.tol:
                return f"{n}: float maxabs={d.max():.3g} > {c.tol}"
        else:
            if not np.array_equal(np.asarray(a), np.asarray(b)):
                return f"{n}: value mismatch"
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", default=os.path.join(os.path.dirname(__file__), "..", "target", "debug", "jotnar"))
    ap.add_argument("filter", nargs="?", default="")
    args = ap.parse_args()
    binary = os.path.abspath(args.bin)
    ok = fail = 0
    with tempfile.TemporaryDirectory() as td:
        for c in CASES:
            if args.filter and args.filter not in c.name:
                continue
            err = run_case(c, binary, td)
            if err:
                fail += 1
                print(f"FAIL {c.name}: {err}")
            else:
                ok += 1
                print(f"ok   {c.name}")
    print(f"\n{ok} passed, {fail} failed, {len(CASES)} cases")
    return 1 if fail else 0

# ---------------------------------------------------------------------------
# Cases
# ---------------------------------------------------------------------------
rng = np.random.default_rng(7)

def float_case(name, text, payload, outputs=None, tol=1e-12):
    case(name, text, payload, outputs, tol)

# integer family
a = rng.integers(-1000, 1000, (3, 4), dtype=np.int64)
b = rng.integers(1, 20, (3, 4), dtype=np.int64)
for op in ["IADD", "ISUB", "IMUL", "IDIV", "IMOD", "IXOR", "IAND", "IOR"]:
    case(op, f"IN a\nIN b\nOUT = {op}(a, b)\n", {"a": a, "b": b}, ["OUT"])
case("IMAD", "IN a\nIN b\nIN c\nOUT = IMAD(a, b, c)\n", {"a": a, "b": b, "c": a}, ["OUT"])
case("ISHL", "IN a\nOUT = ISHL(a, 3)\n", {"a": a}, ["OUT"])
case("ISHR", "IN a\nOUT = ISHR(a, 3)\n", {"a": a}, ["OUT"])
f = rng.normal(0, 10, (3, 4)).astype(np.float64)
case("F2I", "IN f\nOUT = F2I(f)\n", {"f": f}, ["OUT"])
case("I2F", "IN a\nOUT = I2F(a)\n", {"a": a}, ["OUT"])

# bytes
data = rng.integers(0, 256, 32, dtype=np.int64)
case("IWINDOW", "IN x\nOUT = IWINDOW(x, 3, 6)\n", {"x": data}, ["OUT"])
case("IWINDOW_DYN", "IN x\nIN off\nOUT = IWINDOW(x, off, 4)\n", {"x": data, "off": np.array([5], np.int64)}, ["OUT"])
idx = rng.integers(0, 32, 8, dtype=np.int64)
case("IGATHER1", "IN x\nIN ids\nOUT = IGATHER1(x, ids)\n", {"x": data, "ids": idx}, ["OUT"])
case("IASSEMBLE", "IN x\nOUT = IASSEMBLE(x, 2, 4)\n", {"x": data}, ["OUT"])
case("ISUM", "IN x\nOUT = ISUM(x, 2, 6)\n", {"x": data}, ["OUT"])
case("ISCATTER1", "IN x\nIN ids\nIN v\nOUT = ISCATTER1(x, ids, v)\n",
     {"x": data.copy(), "ids": idx, "v": rng.integers(0, 256, 8, dtype=np.int64)}, ["OUT"])
case("ICRC32", "IN x\nOUT = ICRC32(x)\n", {"x": data}, ["OUT"])
case("ICRC32C", "IN x\nIN seed\nOUT = ICRC32C(x, seed)\n", {"x": data, "seed": np.array([0x1234], np.int64)}, ["OUT"])
pat = data[7:11].copy()
case("IFIND", "IN x\nIN pat\nOUT = IFIND(x, pat)\n", {"x": data, "pat": pat}, ["OUT"])
case("IVARINT", "IN x\nOUT, LEN = IVARINT(x, 2)\n", {"x": np.array([0x81,0x80,0x81,0x2a,0,0,0,0,0], np.int64)}, ["OUT", "LEN"])

# extended integer family (jotnar-main 86-op surface)
ia = np.array([0, 1, -1, 7, -7, (1 << 63) - 1, -(1 << 63), 123456789,
               -123456789, 42], np.int64)
ib = np.array([1, 2, 3, 7, -3, 2, -5, 1000, 999, (1 << 63) - 1], np.int64)
ish = np.array([0, 1, 2, 3, 4, 5, 6, 62, 63, 1], np.int64)
for op in ["SADD", "IMAX", "IDIVU", "IMODU", "IEQ", "ILT"]:
    case(op, f"IN a\nIN b\nOUT = {op}(a, b)\n", {"a": ia, "b": ib}, ["OUT"])
case("ISAR", "IN a\nIN s\nOUT = ISAR(a, s)\n", {"a": ia, "s": ish}, ["OUT"])
case("ISELECT", "IN c\nIN a\nIN b\nOUT = ISELECT(c, a, b)\n",
     {"c": np.array([0, 1, -1, 2, 0, 1, 0, 1, 0, 1], np.int64), "a": ia, "b": ib},
     ["OUT"])

# moves
t = tri(rng.normal(0, 0.3, (3, 4)))
case("TRANSPOSE", "IN t AS T:SEQ\nOUT = TRANSPOSE(t)\n", {"t": t}, ["OUT"])
case("RESHAPE2", "IN t AS T:SEQ\nOUT = RESHAPE2(t, 4, 3)\n", {"t": t}, ["OUT"])
case("RESHAPE3", "IN t AS T:SEQ\nOUT = RESHAPE3(t, 2, 3, 2)\n", {"t": t}, ["OUT"])
t3 = tri(rng.normal(0, 0.3, (2, 3, 2)))
case("PERMUTE3", "IN t AS T:SEQ\nOUT = PERMUTE3(t, 1, 0, 2)\n", {"t": t3}, ["OUT"])
case("CONCAT", "IN a AS T:SEQ\nIN b AS T:SEQ\nOUT = CONCAT(a, b, 0)\n",
     {"a": tri(rng.normal(0, .2, (2, 4))), "b": tri(rng.normal(0, .2, (3, 4)))}, ["OUT"])
case("SLICE", "IN t AS T:SEQ\nOUT = SLICE(t, 0, 1, 3)\n", {"t": t}, ["OUT"])
table = tri(rng.normal(0, 0.3, (5, 2)))
ids = np.array([0, 4, 2], np.int64)
case("GATHER", "IN w AS T:SEQ\nIN ids\nOUT = GATHER(w, ids)\n", {"w": table, "ids": ids}, ["OUT"])
bank = tri(rng.normal(0, 0.3, (3, 4, 2)))
case("GATHER3", "IN w AS T:SEQ\nIN i\nIN j\nOUT = GATHER3(w, i, j)\n",
     {"w": bank, "i": np.array([0,2], np.int64), "j": np.array([1,3], np.int64)}, ["OUT"])
mask = np.array([[True, False], [False, True], [True, True], [False, False], [True, False]], bool)
case("SELECT", "IN m\nIN a AS T:SEQ\nIN b AS T:SEQ\nOUT = SELECT(m, a, b)\n",
     {"m": mask, "a": table, "b": tri(rng.normal(0, 0.3, (5, 2)))}, ["OUT"])
case("GT", "IN a AS T:SEQ\nIN b AS T:SEQ\nOUT = GT(a, b)\n", {"a": table, "b": tri(rng.normal(0, 0.3, (5, 2)))}, ["OUT"])
case("ARGMAX_T", "IN t AS T:SEQ\nOUT = ARGMAX(t, 1)\n", {"t": t}, ["OUT"])
f2 = rng.normal(0, 1, (3, 5)).astype(np.float64)
case("ARGMAX_F", "IN f\nOUT = ARGMAX(f, 1)\n", {"f": f2}, ["OUT"])
case("TOPK", "IN f\nOUT, IDX = TOPK(f, 3)\n", {"f": f2}, ["OUT", "IDX"])
base = rng.normal(0, 1, (4, 3)).astype(np.float64)
case("SCATTER", "IN b\nIN r\nIN i\nOUT = SCATTER(b, r, i)\n",
     {"b": base, "r": np.array([[9.,8.,7.]]), "i": np.array([2], np.int64)}, ["OUT"])

# triples arithmetic
x = tri(rng.normal(0, 0.3, (3, 4)))
y = tri(rng.normal(0, 0.3, (3, 4)))
for op in ["ADD", "SUB", "MUL", "DIV"]:
    case(op, f"IN a AS T:SEQ\nIN b AS T:SEQ\nOUT = {op}(a, b)\n", {"a": x, "b": y}, ["OUT"])
case("SQUARE", "IN a AS T:SEQ\nOUT = SQUARE(a)\n", {"a": x}, ["OUT"])
case("SQRT", "IN f\nOUT = SQRT(f)\n", {"f": np.abs(rng.normal(0.2, 0.1, (3,4)))}, ["OUT"])
case("BETA", "IN a AS T:SEQ\nCONFIG beta 0.37\nOUT = BETA(a)\n", {"a": x}, ["OUT"])
case("TBETA", "IN a AS T:SEQ\nCONFIG beta 0.37\nCONFIG beta_b 0.8\nOUT = TBETA(a)\n", {"a": x}, ["OUT"])
case("CLIP", "IN a AS T:SEQ\nOUT = CLIP(a, -0.1, 0.2)\n", {"a": x}, ["OUT"])
case("SIGMOID", "IN a AS T:SEQ\nOUT = SIGMOID(a)\n", {"a": x}, ["OUT"])
case("RESCALE", "IN a AS T:SEQ\nOUT = RESCALE(a, 30000)\n", {"a": x}, ["OUT"])

# linalg / norms
A = tri(rng.normal(0, 0.3, (3, 4)))
B = tri(rng.normal(0, 0.3, (4, 2)))
case("MATMUL", "IN a AS T:SEQ\nIN b AS T:SEQ\nOUT = MATMUL(a, b)\n", {"a": A, "b": B}, ["OUT"])
Ab = tri(rng.normal(0, 0.3, (2, 3, 4)))
Bb = tri(rng.normal(0, 0.3, (1, 4, 2)))
case("BATCH_MATMUL", "IN a AS T:SEQ\nIN b AS T:SEQ\nOUT = BATCH_MATMUL(a, b)\n", {"a": Ab, "b": Bb}, ["OUT"])
w = tri(rng.uniform(0.5, 1.5, 4))
case("RMSNORM", "IN x AS T:SEQ\nIN w AS T:SEQ\nOUT = RMSNORM(x, w)\n", {"x": A, "w": w}, ["OUT"])
bias = tri(rng.normal(0, 0.1, 4))
case("LAYERNORM", "IN x AS T:SEQ\nIN w AS T:SEQ\nIN b AS T:SEQ\nOUT = LAYERNORM(x, w, b)\n",
     {"x": A, "w": w, "b": bias}, ["OUT"])
scores = tri(rng.normal(0, 0.1, (3, 5)))
case("SOFTMAX", "IN s AS T:SEQ\nOUT = SOFTMAX(s)\n", {"s": scores}, ["OUT"])
case("SOFTMAX_WIDE", "IN s AS T:SEQ\nOUT = SOFTMAX_WIDE(s)\n", {"s": scores}, ["OUT"])
case("TSHIFT", "IN s AS T:SEQ\nOUT = TSHIFT(s)\n", {"s": scores}, ["OUT"])
case("SILU", "IN a AS T:SEQ\nOUT = SILU(a)\n", {"a": x}, ["OUT"])
case("GELU", "IN a AS T:SEQ\nOUT = GELU(a)\n", {"a": x}, ["OUT"])
h = tri(rng.normal(0, 0.1, (2, 3)))
abar = tri(rng.uniform(0.1, 0.9, (2, 3)))
bx = tri(rng.normal(0, 0.1, (2, 3)))
case("SCAN", "IN h AS T:SEQ\nIN ab AS T:SEQ\nIN bx AS T:SEQ\nOUT = SCAN(h, ab, bx)\n",
     {"h": h, "ab": abar, "bx": bx}, ["OUT"])
q = tri(rng.normal(0, 0.3, (4, 4)))
case("ROTARY", "IN q AS T:SEQ\nIN pos\nOUT = ROTARY(q, pos)\n",
     {"q": q, "pos": np.arange(4, dtype=np.int64)}, ["OUT"])
case("ROPE", "IN q\nIN pos\nOUT = ROPE(q, pos, 2, 4)\n",
     {"q": rng.normal(0, 0.3, (4, 8)).astype(np.float64), "pos": np.arange(4, dtype=np.int64)}, ["OUT"])
case("SWIGLU", "IN a\nOUT = SWIGLU(a, 4)\n", {"a": rng.normal(0, 0.5, (3, 8)).astype(np.float64)}, ["OUT"])

# image / tensor helpers
case("SRGB_DECODE", "IN rgb\nOUT = SRGB_DECODE(rgb)\n",
     {"rgb": rng.integers(0, 256, (3, 4, 3), dtype=np.uint8)}, ["OUT"])
case("LUMA", "IN lin\nOUT = LUMA(lin)\n",
     {"lin": rng.uniform(0, 1, (3, 4, 3)).astype(np.float64)}, ["OUT"])
case("SRGB_ENCODE", "IN lin\nOUT = SRGB_ENCODE(lin)\n",
     {"lin": rng.uniform(0, 1, (3, 4, 3)).astype(np.float64)}, ["OUT"])
case("GAIN", "IN lin\nIN y\nIN yenh AS T:HW\nOUT = GAIN(lin, y, yenh)\n",
     {"lin": rng.uniform(0, 1, (3, 4, 3)).astype(np.float32),
      "y": rng.uniform(0.1, 0.9, (3, 4)).astype(np.float64),
      "yenh": tri(rng.uniform(0.1, 0.9, (3, 4)))}, ["OUT"])
img = tri(rng.uniform(0.1, 0.9, (5, 6)))
case("GAUSS", "IN a AS T:HW\nOUT = GAUSS(a, 1, 0.8)\n", {"a": img}, ["OUT"])
case("ISO_BLUR", "IN a AS T:HW\nOUT = ISO_BLUR(a)\n", {"a": img}, ["OUT"])
case("CONV", "IN a AS T:HW\nIN k\nOUT = CONV(a, k)\n",
     {"a": img, "k": np.array([[0.,1.,0.],[1.,2.,1.],[0.,1.,0.]])/5}, ["OUT"])
case("POOLAVG", "IN a AS T:HWC\nOUT = POOLAVG(a)\n", {"a": tri(rng.uniform(0.1,0.9,(3,4,2)))}, ["OUT"])
case("PRELU", "IN a AS T:SEQ\nIN s AS T:SEQ\nOUT = PRELU(a, s)\n",
     {"a": x, "s": tri(np.full((1,), 0.2))}, ["OUT"])
case("INTERP", "IN a AS T:HWC\nOUT = INTERP(a, 2.0, 2.0)\n", {"a": tri(rng.uniform(0.1,0.9,(3,4,2)))}, ["OUT"])
flow = rng.normal(0, 0.5, (5,6,2)).astype(np.float64)
case("WARP", "IN d AS T:HW\nIN flow\nOUT = WARP(d, flow)\n", {"d": img, "flow": flow}, ["OUT"])
case("STATIC", "IN flow\nOUT = STATIC(flow)\n", {"flow": flow}, ["OUT"])
case("MIXDYAD", "IN d AS T:HW\nIN w AS T:HW\nIN m\nOUT = MIXDYAD(d, w, m)\n",
     {"d": img, "w": tri(rng.uniform(0.1,0.9,(5,6))), "m": rng.integers(0,2,(5,6)).astype(np.int64)}, ["OUT"])

# fused / heavier ops
case("DECONV", "IN a AS T:HWC\nIN w AS T:SEQ\nOUT = DECONV(a, w, 2, 1)\n",
     {"a": tri(rng.uniform(0.1, 0.9, (3, 4, 2))),
      "w": tri(rng.normal(0, 0.2, (2, 3, 3, 2)))}, ["OUT"])
case("SPLAT_BLUR", "IN a AS T:HW\nAS, COH = SPLAT_BLUR(a)\n",
     {"a": tri(rng.uniform(0.1, 0.9, (5, 6)))}, ["AS", "COH"])
case("BETA_V5", "IN d AS T:HW\nIN c AS T:HW\nCONFIG beta 0.5\nOUT = BETA_V5(d, c)\n",
     {"d": tri(rng.normal(0, 0.1, (5, 6))), "c": tri(rng.uniform(0, 1, (5, 6)))}, ["OUT"])
q2 = rng.normal(0, 0.3, (2, 8)).astype(np.float64)
kk = rng.normal(0, 0.3, (3, 4)).astype(np.float64)
vv = rng.normal(0, 0.3, (3, 4)).astype(np.float64)
msk = np.array([[1,1,0],[1,1,1]], np.int64)
case("ATTN", "IN q\nIN k\nIN v\nIN m\nOUT = ATTN(q, k, v, m, 2, 1, 4, 3)\n",
     {"q": q2, "k": kk, "v": vv, "m": msk}, ["OUT"])
Abm = tri(rng.normal(0, 0.3, (4, 3)))
Bbm = tri(rng.normal(0, 0.3, (6, 2)))
case("BMMV", "IN a AS T:SEQ\nIN b AS T:SEQ\nOUT = BMMV(a, b, 2, 2, 2, 3, 3, 6, 2, 6, 2, 4, 0)\n",
     {"a": Abm, "b": Bbm}, ["OUT"])
import zlib
raw = b"hello hello hello, deflate raw stream"
packed = zlib.compress(raw, 6, wbits=-15)
case("IINFLATE", "IN x\nOUT = IINFLATE(x, %d)\n" % len(raw),
     {"x": np.frombuffer(packed, dtype=np.uint8).astype(np.int64)}, ["OUT"])

if __name__ == "__main__":
    sys.exit(main())
