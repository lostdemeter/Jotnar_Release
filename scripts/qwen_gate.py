#!/usr/bin/env python3
"""Qwen block acceptance gate.

Builds the block from docs/LADDER_OPS.md section 4 with a small deterministic
payload (8 tokens, 32 hidden, 16-wide SwiGLU, 9 latent outputs) and compares
every intermediate stream against the reference interpreter, plane for plane.

    python3 scripts/qwen_gate.py                # Vulkan (the new backend)
    python3 scripts/qwen_gate.py --target nonfpu # the integer reference

Exits non-zero if any plane differs. T streams are exact; I streams are exact.
"""
import argparse
import json
import math
import os
import struct
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

PHI = 1.6180339887498949
K = 512.0
BIAS = 32768
NLVL = 65536

S, D, GU, DOWN, OUT = 8, 32, 32, 16, 9

LISTING = """CONFIG m_acc 38000
CONFIG m_cov 38000
CONFIG beta_b 0.31622776601683794

IN tokens AS T:*
IN pos AS I:SEQ
IN emb AS T:*
IN pos_code AS T:*
IN wq AS T:*
IN wk AS T:*
IN wv AS T:*
IN wo AS T:*
IN w_gate_up AS T:*
IN w_down AS T:*
IN rms1 AS T:*
IN rms2 AS T:*
IN pool AS T:*
IN w_out AS T:*

qwen_x0 = MATMUL(tokens, emb)
qwen_x  = MUL(qwen_x0, pos_code)
qwen_n1 = RMSNORM(qwen_x, rms1)

qwen_q = MATMUL(qwen_n1, wq)
qwen_k = MATMUL(qwen_n1, wk)
qwen_v = MATMUL(qwen_n1, wv)

qwen_qr = ROTARY(qwen_q, pos)
qwen_kr = ROTARY(qwen_k, pos)
qwen_kt = TRANSPOSE(qwen_kr)
qwen_sc = MATMUL(qwen_qr, qwen_kt)
qwen_bt = TBETA(qwen_sc)
qwen_sc2 = MUL(qwen_sc, qwen_bt)
qwen_sh = TSHIFT(qwen_sc2)
qwen_pr = SOFTMAX_WIDE(qwen_sh)
qwen_at = MATMUL(qwen_pr, qwen_v)
qwen_ao = MATMUL(qwen_at, wo)
qwen_res = ADD(qwen_x, qwen_ao)

qwen_n2 = RMSNORM(qwen_res, rms2)
qwen_gu = MATMUL(qwen_n2, w_gate_up)
qwen_gate = SLICE(qwen_gu, 1, 0, 16)
qwen_up = SLICE(qwen_gu, 1, 16, 32)
qwen_silu = SILU(qwen_gate)
qwen_act = MUL(qwen_silu, qwen_up)
qwen_dn = MATMUL(qwen_act, w_down)
qwen_out = ADD(qwen_res, qwen_dn)

qwen_pooled = MATMUL(pool, qwen_out)
qwen_latent_t = MATMUL(qwen_pooled, w_out)
qwen_latent = T2I(qwen_latent_t)

h0_0 = IGET(qwen_latent, 0)
h0_1 = IGET(qwen_latent, 1)
h0_2 = IGET(qwen_latent, 2)
h0_3 = IGET(qwen_latent, 3)
h0_4 = IGET(qwen_latent, 4)
h0_5 = IGET(qwen_latent, 5)
h0_6 = IGET(qwen_latent, 6)
h0_7 = IGET(qwen_latent, 7)
h0_8 = IGET(qwen_latent, 8)
"""

STAGES = [
    "qwen_x0", "qwen_x", "qwen_n1", "qwen_q", "qwen_k", "qwen_v",
    "qwen_qr", "qwen_kr", "qwen_kt", "qwen_sc", "qwen_bt", "qwen_sc2",
    "qwen_sh", "qwen_pr", "qwen_at", "qwen_ao", "qwen_res", "qwen_n2",
    "qwen_gu", "qwen_gate", "qwen_up", "qwen_silu", "qwen_act", "qwen_dn",
    "qwen_out", "qwen_pooled", "qwen_latent_t", "qwen_latent",
    "h0_0", "h0_1", "h0_2", "h0_3", "h0_4", "h0_5", "h0_6", "h0_7", "h0_8",
]


class Lcg:
    def __init__(self, seed):
        self.state = seed & ((1 << 64) - 1)

    def vals(self, n, lo, hi):
        out = []
        for _ in range(n):
            self.state = (self.state * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)
            out.append(lo + (hi - lo) * ((self.state >> 11) / float(1 << 53)))
        return out


def encode(v):
    s = -1 if v < 0 else 1
    raw = K * math.log(abs(v) + 1e-15) / math.log(PHI)
    e = max(0, min(NLVL - 1, round(raw) + BIAS))
    return s, e, 0


def tstream(shape, values):
    s, e, z = [], [], []
    for v in values:
        a, b, c = encode(v)
        s.append(a)
        e.append(b)
        z.append(c)
    return {"kind": "T", "shape": list(shape), "s": s, "e": e, "z": z}


def payload():
    rng = Lcg(20261120)
    return {"streams": {
        "tokens": tstream([S, D], rng.vals(S * D, -0.5, 0.5)),
        "pos": {"kind": "I64", "shape": [S], "data": list(range(S))},
        "emb": tstream([D, D], rng.vals(D * D, -0.5, 0.5)),
        "pos_code": tstream([S, D], rng.vals(S * D, 0.5, 1.0)),
        "wq": tstream([D, D], rng.vals(D * D, -0.2, 0.2)),
        "wk": tstream([D, D], rng.vals(D * D, -0.2, 0.2)),
        "wv": tstream([D, D], rng.vals(D * D, -0.2, 0.2)),
        "wo": tstream([D, D], rng.vals(D * D, -0.2, 0.2)),
        "w_gate_up": tstream([D, GU], rng.vals(D * GU, -0.2, 0.2)),
        "w_down": tstream([DOWN, D], rng.vals(DOWN * D, -0.2, 0.2)),
        "rms1": tstream([D], rng.vals(D, 0.8, 1.2)),
        "rms2": tstream([D], rng.vals(D, 0.8, 1.2)),
        "pool": tstream([1, S], rng.vals(S, -4.0, 4.0)),
        "w_out": tstream([D, OUT], rng.vals(D * OUT, -0.2, 0.2)),
    }}


def planes(stream):
    kind = stream["kind"]
    if kind in ("T", "T2"):
        if "m" in stream:
            return [("s", "b", "s"), ("e", "i", "e"), ("m", "i", "m"), ("z", "B", "z")]
        return [("s", "b", "s"), ("e", "i", "e"), ("z", "B", "z")]
    if kind == "I64":
        return [("", "q", "data")]
    return [("", "d", "data")]


def emit(path, fmt, values):
    vals = [float(v) if fmt in "df" else int(v) for v in values]
    open(path, "wb").write(struct.pack("<" + fmt * len(vals), *vals))


def load(path, fmt, count):
    return list(struct.unpack("<" + fmt * count, open(path, "rb").read()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", default=os.path.join(ROOT, "target", "release", "jotnar"))
    ap.add_argument("--target", default="vulkan")
    args = ap.parse_args()

    work = tempfile.mkdtemp(prefix="qwen-gate-")
    listing = os.path.join(work, "qwen.asm")
    open(listing, "w").write(LISTING)
    pay = payload()
    ppath = os.path.join(work, "payload.json")
    json.dump(pay, open(ppath, "w"))

    print(f"{args.target}: {S} tokens, D={D}, gate_up={GU}, down={DOWN}, out={OUT}")
    print("interpreter...")
    exp = json.loads(subprocess.check_output(
        [args.bin, "run", listing, "--payload", ppath], text=True))["streams"]

    subprocess.run(
        [args.bin, "build", listing, "--payload", ppath, "--workdir", work,
         "--name", "blk", "--target", args.target, "--outputs", ",".join(STAGES)],
        check=True, stdout=subprocess.DEVNULL)
    man = json.load(open(os.path.join(work, "blk.manifest.json")))
    exe = os.path.join(work, man["executable"])

    argv = []
    for name in man["inputs"]:
        for suffix, fmt, key in planes(pay["streams"][name]):
            p = os.path.join(work, f"in.{name}{suffix}.bin")
            emit(p, fmt, pay["streams"][name][key])
            argv.append(p)
    out_paths = []
    for name in man["outputs"]:
        for suffix, fmt, _ in planes(exp[name]):
            p = os.path.join(work, f"out.{name}{suffix}.bin")
            out_paths.append((name, suffix, fmt, p))
            argv.append(p)

    subprocess.run([exe] + argv, check=True, stdout=subprocess.DEVNULL)

    ok = True
    planes_ok = 0
    for name, suffix, fmt, path in out_paths:
        count = 1
        for d in exp[name]["shape"]:
            count *= d
        got = load(path, fmt, count)
        key = "data"
        for suf, _, k in planes(exp[name]):
            if suf == suffix:
                key = k
        want = [int(x) for x in exp[name][key]]
        if got != want:
            ok = False
            bad = sum(1 for a, b in zip(got, want) if a != b)
            print(f"  FAIL {name}{('.' + suffix) if suffix else ''}: "
                  f"{bad}/{len(want)} elements differ")
        else:
            planes_ok += 1
    print(f"planes exact: {planes_ok}/{len(out_paths)}")
    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
