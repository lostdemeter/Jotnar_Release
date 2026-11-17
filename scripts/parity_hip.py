#!/usr/bin/env python3
"""Minimal HIP parity gate: HIP target vs the C target on identical listings.

This is deliberately narrower than ``parity_cuda.py``. It covers the
CUDA-equivalent opcode set this spike ports (elementwise, MATMUL /
BATCH_MATMUL, normalisation, softmax, ARGMAX, integer arithmetic) and compares
the emitted HIP executable against the emitted C executable on the same
payload. No Python chain, numpy, or GPU vendor library is used as a reference.

Run on a host with ``hipcc`` and the target GPU::

    python3 scripts/parity_hip.py --bin target/release/jotnar

The C target stores F streams as float64; the HIP target stores them as
float32, so float comparisons use an absolute/relative tolerance. Integer
outputs are compared bit-exactly.
"""
import argparse
import array
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

# Deterministic LCG so the gate is reproducible without numpy.
def floats(n, seed, lo=-1.0, hi=1.0):
    state = seed & ((1 << 64) - 1)
    out = []
    span = hi - lo
    for _ in range(n):
        state = (state * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)
        out.append(lo + span * (state >> 11) / float(1 << 53))
    return out


def ints(n, seed, lo=-31, hi=32):
    state = seed & ((1 << 64) - 1)
    out = []
    span = hi - lo
    for _ in range(n):
        state = (state * 1103515245 + 12345) & ((1 << 64) - 1)
        out.append(lo + (state >> 17) % span)
    return out


def payload(kind, shape, data):
    count = 1
    for d in shape:
        count *= d
    assert count == len(data), (shape, len(data))
    if kind == "f":
        return {"kind": "F64", "shape": list(shape), "data": data}
    return {"kind": "I64", "shape": list(shape),
            "data": [str(int(x)) for x in data]}


def write_inputs(td, inputs, target):
    argv = []
    for name, (kind, shape, data) in inputs.items():
        fn = os.path.join(td, f"in_{name}_{target}.bin")
        if kind == "f":
            fmt = "d" if target == "c" else "f"
            with open(fn, "wb") as fh:
                array.array(fmt, data).tofile(fh)
        else:
            with open(fn, "wb") as fh:
                array.array("q", data).tofile(fh)
        argv.append(fn)
    return argv


def read_output(path, kind, target):
    if kind == "f":
        fmt = "d" if target == "c" else "f"
        a = array.array(fmt)
    else:
        a = array.array("q")
    with open(path, "rb") as fh:
        a.fromfile(fh, os.path.getsize(path) // a.itemsize)
    return a


def build(binary, td, name, listing, outputs, target):
    d = os.path.join(td, f"{name}_{target}")
    os.makedirs(d, exist_ok=True)
    asm = os.path.join(d, "case.asm")
    pay = os.path.join(d, "payload.json")
    with open(asm, "w") as fh:
        fh.write(listing)
    with open(pay, "w") as fh:
        json.dump({"streams": {k: payload(*v) for k, v in CASE_INPUTS[name].items()}},
                  fh)
    cmd = [binary, "build", asm, "--payload", pay, "--outputs",
           ",".join(outputs), "--target", target, "--workdir", d,
           "--name", name]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError(f"{target} build rc={r.returncode}: {r.stderr[-500:]}")
    return r.stdout.strip().splitlines()[-1], d


def run_case(binary, name, atol, rtol):
    case = CASES[name]
    listing = case["listing"]
    inputs = case["inputs"]
    CASE_INPUTS[name] = inputs
    outputs = case["outputs"]
    results = {}
    with tempfile.TemporaryDirectory(prefix=f"hip_parity_{name}_") as td:
        for target in ("c", "hip"):
            exe, d = build(binary, td, name, listing, list(outputs), target)
            argv = [exe]
            argv += write_inputs(d, inputs, target)
            out_files = {}
            for o in outputs:
                fn = os.path.join(d, f"out_{o}.bin")
                out_files[o] = fn
                argv.append(fn)
            r = subprocess.run(argv, capture_output=True, text=True)
            if r.returncode:
                raise RuntimeError(
                    f"{target} run rc={r.returncode}: {r.stderr[-500:]}")
            results[target] = {
                o: read_output(out_files[o], outputs[o], target) for o in outputs
            }
    details = []
    ok = True
    for o, kind in outputs.items():
        got = results["hip"][o]
        want = results["c"][o]
        if len(got) != len(want):
            details.append(f"{o}: length {len(got)} != {len(want)}")
            ok = False
            continue
        if kind == "i":
            bad = [i for i, (g, w) in enumerate(zip(got, want)) if g != w]
            if bad:
                i = bad[0]
                details.append(
                    f"{o}: {len(bad)} mismatches (first [{i}] hip={got[i]} c={want[i]})")
                ok = False
        else:
            worst = 0.0
            worst_i = -1
            for i, (g, w) in enumerate(zip(got, want)):
                if not (math.isfinite(g) and math.isfinite(w)):
                    if g != w:
                        details.append(f"{o}: non-finite mismatch at [{i}]")
                        ok = False
                    continue
                err = abs(g - w)
                if err > worst:
                    worst, worst_i = err, i
            if worst > atol + rtol * max(1.0, abs(want[worst_i]) if worst_i >= 0 else 1.0):
                details.append(f"{o}: maxabs={worst:.3e} at [{worst_i}]")
                ok = False
            else:
                details.append(f"{o}: maxabs={worst:.3e}")
    return ok, details


CASES = {
    "elem": {
        "listing": "IN a\nIN b\nC = ADD(a, b)\nOUT = MUL(C, a)\n",
        "inputs": {
            "a": ("f", [4, 32], floats(128, 1)),
            "b": ("f", [4, 32], floats(128, 2)),
        },
        "outputs": {"OUT": "f"},
    },
    "saxpy": {
        "listing": "IN a\nIN x\nIN y\nAX = MUL(a, x)\nOUT = ADD(AX, y)\n",
        "inputs": {
            "a": ("f", [256], floats(256, 3)),
            "x": ("f", [256], floats(256, 4)),
            "y": ("f", [256], floats(256, 5)),
        },
        "outputs": {"OUT": "f"},
    },
    "matmul": {
        "listing": "IN a\nIN b\nOUT = MATMUL(a, b)\n",
        "inputs": {
            "a": ("f", [8, 16], floats(128, 6)),
            "b": ("f", [16, 32], floats(512, 7)),
        },
        "outputs": {"OUT": "f"},
    },
    "bmm": {
        "listing": "IN a\nIN b\nOUT = BATCH_MATMUL(a, b)\n",
        "inputs": {
            "a": ("f", [2, 4, 8], floats(64, 8)),
            "b": ("f", [2, 8, 16], floats(256, 9)),
        },
        "outputs": {"OUT": "f"},
    },
    "rmsnorm": {
        "listing": "CONFIG eps_rms 1e-6\nIN x\nIN w\nOUT = RMSNORM(x, w)\n",
        "inputs": {
            "x": ("f", [4, 32], floats(128, 10)),
            "w": ("f", [32], floats(32, 11, 0.5, 1.5)),
        },
        "outputs": {"OUT": "f"},
    },
    "softmax": {
        "listing": "IN x\nOUT = SOFTMAX_WIDE(x)\n",
        "inputs": {"x": ("f", [4, 64], floats(256, 12, -3.0, 3.0))},
        "outputs": {"OUT": "f"},
    },
    "argmax": {
        "listing": "IN x\nOUT = ARGMAX(x, 1)\n",
        "inputs": {"x": ("f", [8, 16], floats(128, 13))},
        "outputs": {"OUT": "i"},
    },
    "int6": {
        "listing": (
            "IN a\nIN b\n"
            "SU = ISUB(a, b)\nAD = IADD(a, b)\nMU = IMUL(a, b)\n"
            "EQ = IEQ(a, b)\nLT = ILT(a, b)\nMX = IMAX(a, b)\n"
        ),
        "inputs": {
            "a": ("i", [64], ints(64, 14)),
            "b": ("i", [64], ints(64, 15, 1, 32)),
        },
        "outputs": {"SU": "i", "AD": "i", "MU": "i", "EQ": "i",
                    "LT": "i", "MX": "i"},
    },
    "isar": {
        "listing": "IN a\nIN s\nOUT = ISAR(a, s)\n",
        "inputs": {
            "a": ("i", [32], ints(32, 16, -(1 << 40), 1 << 40)),
            "s": ("i", [32], ints(32, 17, 0, 64)),
        },
        "outputs": {"OUT": "i"},
    },
    "layernorm": {
        "listing": ("CONFIG eps_ln 1e-5\nIN x\nIN w\nIN b\n"
                    "OUT = LAYERNORM(x, w, b)\n"),
        "inputs": {
            "x": ("f", [4, 32], floats(128, 18)),
            "w": ("f", [32], floats(32, 19, 0.5, 1.5)),
            "b": ("f", [32], floats(32, 20, -0.25, 0.25)),
        },
        "outputs": {"OUT": "f"},
    },
    "conv": {
        "listing": "IN x\nIN k\nOUT = CONV(x, k)\n",
        "inputs": {
            "x": ("f", [8, 8], floats(64, 21)),
            "k": ("f", [3, 3], floats(9, 22, -0.5, 0.5)),
        },
        "outputs": {"OUT": "f"},
    },
    "rotary": {
        "listing": "IN x\nIN pos\nOUT = ROTARY(x, pos)\n",
        "inputs": {
            "x": ("f", [3, 8], floats(24, 23)),
            "pos": ("i", [3], [0, 1, 2]),
        },
        "outputs": {"OUT": "f"},
    },
    "rope": {
        "listing": "IN x\nIN pos\nOUT = ROPE(x, pos, 2, 4)\n",
        "inputs": {
            "x": ("f", [3, 8], floats(24, 24)),
            "pos": ("i", [3], [0, 1, 2]),
        },
        "outputs": {"OUT": "f"},
    },
    "transpose": {
        "listing": "IN x\nOUT = TRANSPOSE(x)\n",
        "inputs": {"x": ("f", [4, 8], floats(32, 25))},
        "outputs": {"OUT": "f"},
    },
    "concat": {
        "listing": "IN a\nIN b\nOUT = CONCAT(a, b, 1)\n",
        "inputs": {
            "a": ("f", [4, 8], floats(32, 26)),
            "b": ("f", [4, 8], floats(32, 27)),
        },
        "outputs": {"OUT": "f"},
    },
    "slice": {
        "listing": "IN x\nOUT = SLICE(x, 1, 1, 5)\n",
        "inputs": {"x": ("f", [4, 8], floats(32, 28))},
        "outputs": {"OUT": "f"},
    },
    "select": {
        "listing": "IN m\nIN a\nIN b\nOUT = SELECT(m, a, b)\n",
        "inputs": {
            "m": ("i", [16], [i % 2 for i in range(16)]),
            "a": ("f", [16], floats(16, 29)),
            "b": ("f", [16], floats(16, 30)),
        },
        "outputs": {"OUT": "f"},
    },
    "scatter": {
        "listing": "IN base\nIN row\nIN idx\nOUT = SCATTER(base, row, idx)\n",
        "inputs": {
            "base": ("f", [5, 4], floats(20, 31)),
            "row": ("f", [1, 4], floats(4, 32)),
            "idx": ("i", [1], [2]),
        },
        "outputs": {"OUT": "f"},
    },
    "topk": {
        "listing": "IN x\nV, I = TOPK(x, 3)\n",
        "inputs": {"x": ("f", [4, 8], floats(32, 33))},
        "outputs": {"V": "f", "I": "i"},
    },
    "f2i_i2f": {
        "listing": "IN x\nIN i\nF = F2I(x)\nG = I2F(i)\n",
        "inputs": {
            "x": ("f", [8], [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0]),
            "i": ("i", [8], [-3, -2, -1, 0, 1, 2, 3, 4]),
        },
        "outputs": {"F": "i", "G": "f"},
    },
    "intdiv": {
        "listing": ("IN a\nIN b\nIN c\nQ = IDIV(a, b)\nR = IMOD(a, b)\n"
                    "S = ISELECT(c, a, b)\n"),
        "inputs": {
            "a": ("i", [32], ints(32, 34, -100, 100)),
            "b": ("i", [32], ints(32, 35, 1, 31)),
            "c": ("i", [32], [i % 2 for i in range(32)]),
        },
        "outputs": {"Q": "i", "R": "i", "S": "i"},
    },
    "gather": {
        "listing": "IN t\nIN ids\nOUT = GATHER(t, ids)\n",
        "inputs": {
            "t": ("f", [5, 4], floats(20, 36)),
            "ids": ("i", [3], [0, 4, 2]),
        },
        "outputs": {"OUT": "f"},
    },
    "gather3": {
        "listing": "IN t\nIN ii\nIN jj\nOUT = GATHER3(t, ii, jj)\n",
        "inputs": {
            "t": ("f", [2, 3, 4], floats(24, 37)),
            "ii": ("i", [2], [0, 1]),
            "jj": ("i", [2], [2, 0]),
        },
        "outputs": {"OUT": "f"},
    },
    "swiglu": {
        "listing": "IN x\nOUT = SWIGLU(x, 4)\n",
        "inputs": {"x": ("f", [3, 8], floats(24, 38))},
        "outputs": {"OUT": "f"},
    },
    "tshift": {
        "listing": "IN x\nOUT = TSHIFT(x)\n",
        "inputs": {"x": ("f", [4, 8], floats(32, 39))},
        "outputs": {"OUT": "f"},
    },
    "attn": {
        "listing": ("IN qr\nIN kf\nIN vf\nIN mask\n"
                    "OUT = ATTN(qr, kf, vf, mask, 4, 2, 8, 3)\n"),
        "inputs": {
            "qr": ("f", [1, 32], floats(32, 40)),
            "kf": ("f", [3, 16], floats(48, 41)),
            "vf": ("f", [3, 16], floats(48, 42)),
            "mask": ("i", [2, 3], [1, 1, 1, 1, 1, 1]),
        },
        "outputs": {"OUT": "f"},
    },
    "misc": {
        "listing": ("CONFIG beta 0.5\nIN x\nIN a\nIN b\n"
                    "S = SILU(x)\nD = DIV(a, b)\nB = BETA(x)\n"),
        "inputs": {
            "x": ("f", [4, 8], floats(32, 43)),
            "a": ("f", [4, 8], floats(32, 44)),
            "b": ("f", [4, 8], floats(32, 45, 0.5, 1.5)),
        },
        "outputs": {"S": "f", "D": "f", "B": "f"},
    },
    "arithint": {
        "listing": ("IN a\nIN b\nIN c\nM = IMAD(a, b, c)\n"
                    "L = ISHL(a, 3)\nR = ISHR(a, 3)\n"
                    "QU = IDIVU(a, b)\nRM = IMODU(a, b)\n"),
        "inputs": {
            "a": ("i", [32], ints(32, 46, -100, 100)),
            "b": ("i", [32], ints(32, 47, 1, 31)),
            "c": ("i", [32], ints(32, 48, -100, 100)),
        },
        "outputs": {"M": "i", "L": "i", "R": "i", "QU": "i", "RM": "i"},
    },
    "interp": {
        "listing": "IN x\nOUT = INTERP(x, 2.0, 2.0)\n",
        "inputs": {"x": ("f", [4, 4, 2], floats(32, 49))},
        "outputs": {"OUT": "f"},
    },
    "bitwise": {
        "listing": ("IN a\nIN b\nXO = IXOR(a, b)\nAN = IAND(a, b)\n"
                    "OR = IOR(a, b)\nSA = SADD(a, b)\n"),
        "inputs": {
            "a": ("i", [32], ints(32, 50, -1000, 1000)),
            "b": ("i", [32], ints(32, 51, -1000, 1000)),
        },
        "outputs": {"XO": "i", "AN": "i", "OR": "i", "SA": "i"},
    },
    "sat": {
        "listing": "IN a\nIN b\nOUT = SADD(a, b)\n",
        "inputs": {
            "a": ("i", [8], [-(1 << 63), (1 << 63) - 1, -(1 << 63), 5,
                         (1 << 63) - 1, -5, 0, 123456789]),
            "b": ("i", [8], [-1, 1, 1, (1 << 63) - 1, (1 << 63) - 1,
                         -(1 << 63), 0, -987654321]),
        },
        "outputs": {"OUT": "i"},
    },
    "bmmv": {
        "listing": ("IN a\nIN b\n"
                    "OUT = BMMV(a, b, 2, 2, 1, 3, 3, 6, 1, 3, 0, 0, 0)\n"),
        "inputs": {
            "a": ("f", [2, 2, 3], floats(12, 52)),
            "b": ("f", [2, 3], floats(6, 53)),
        },
        "outputs": {"OUT": "f"},
    },
}

CASE_INPUTS = {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", default=os.path.join(ROOT, "target", "release", "jotnar"))
    ap.add_argument("--atol", type=float, default=1e-4)
    ap.add_argument("--rtol", type=float, default=1e-4)
    ap.add_argument("--case", action="append", default=[])
    args = ap.parse_args()
    binary = os.path.abspath(args.bin)
    if not os.path.isfile(binary):
        sys.exit(f"missing binary: {binary}")
    hipcc = os.environ.get("HIPCC", "hipcc")
    hipcc_ok = (
        os.path.isfile(hipcc) and os.access(hipcc, os.X_OK)
        if os.path.isabs(hipcc)
        else shutil.which(hipcc) is not None
    )
    if not hipcc_ok:
        sys.exit(
            "missing hipcc: HIP parity was not run; install ROCm or set HIPCC"
        )
    names = args.case or list(CASES)
    failed = []
    for name in names:
        try:
            ok, details = run_case(binary, name, args.atol, args.rtol)
        except Exception as exc:  # noqa: BLE001
            ok, details = False, [str(exc)[:300]]
        print(f"hip-{name}: {'OK' if ok else 'FAIL'} {'; '.join(details)}",
              flush=True)
        if not ok:
            failed.append(name)
    print("FAILURES:", failed if failed else "none")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
