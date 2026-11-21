#!/usr/bin/env python3
"""Vulkan parity gate: the Vulkan target against an established reference.

T and T2 are compared with the non-FPU emitter and must be bit-identical on
every plane. Integer cases are compared with the C target, also bit-exactly.
F streams are float32 on Vulkan and float64 on C, so those use a tolerance:
a declared agreement class, not an exact machine.

Run on the Steam Frame (or any host with glslc, cc and libvulkan)::

    python3 scripts/parity_vulkan.py --bin target/release/jotnar

Exits non-zero if any case mismatches.
"""
import argparse
import json
import os
import shutil
import struct
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def t3(s, e, z, shape):
    return {"kind": "T", "shape": shape, "s": s, "e": e, "z": z}


def t2(s, e, m, z, shape):
    return {"kind": "T", "shape": shape, "s": s, "e": e, "m": m, "z": z}


def i64(data, shape):
    return {"kind": "I64", "shape": shape, "data": data}


def f64(data, shape):
    return {"kind": "F64", "shape": shape, "data": data}


def gen_t2(rows, cols, seed):
    """Deterministic canonical T2 stream, including scattered zero terms."""
    n = rows * cols
    return t2(
        [1 if (i * 7 + seed) % 3 else -1 for i in range(n)],
        [32768 + ((i * 37 + seed * 11) % 900) for i in range(n)],
        [((i * 911 + seed * 17) % 2000) - 1000 for i in range(n)],
        [1 if (i * 13 + seed) % 17 == 0 else 0 for i in range(n)],
        [rows, cols],
    )


# name, listing, input streams, output kind+shape, reference target, exact?
CASES = [
    (
        "t_chain",
        """IN A AS T:*
IN B AS T:*
S = MUL(A, B)
OUT = ADD(S, B)
""",
        lambda: {
            "A": t3([1, 1, -1, 1, 1, -1, 1, -1],
                    [32768, 33000, 32500, 32768, 32000, 33500, 34000, 32768],
                    [0, 0, 0, 1, 0, 0, 0, 0], [8]),
            "B": t3([1, -1, 1, 1, -1, -1, 1, 1],
                    [32768, 32800, 33000, 32768, 34000, 33500, 32768, 33000],
                    [0, 0, 0, 0, 0, 0, 0, 0], [8]),
        },
        {"kind": "T", "shape": [8]},
        "nonfpu",
        True,
    ),
    (
        "t_matmul",
        """IN A AS T:*
IN B AS T:*
OUT = MATMUL(A, B)
""",
        lambda: {
            "A": t3([1, 1, 1, -1], [32768, 33000, 32500, 32768], [0, 0, 0, 0], [2, 2]),
            "B": t3([1, -1, 1, 1], [32768, 32800, 33000, 32768], [0, 0, 0, 0], [2, 2]),
        },
        {"kind": "T", "shape": [2, 2]},
        "nonfpu",
        True,
    ),
    (
        "t2_chain",
        """CONFIG t2 24
IN A AS T:*
IN B AS T:*
S = MUL(A, B)
OUT = ADD(S, B)
""",
        lambda: {
            "A": t2([1, 1, -1, 1, 1, -1],
                    [32768, 33000, 32500, 32768, 32000, 33500],
                    [0, 100000, -250000, 8388607, -8388608, 0],
                    [0, 0, 0, 1, 0, 0], [6]),
            "B": t2([1, -1, 1, 1, -1, -1],
                    [32768, 32800, 33000, 32768, 34000, 33500],
                    [50000, -70000, 120000, 0, 400000, -900000],
                    [0, 0, 0, 0, 0, 0], [6]),
        },
        {"kind": "T", "shape": [6], "t2": True},
        "nonfpu",
        True,
    ),
    (
        "t2_matmul",
        """CONFIG t2 24
IN A AS T:*
IN B AS T:*
OUT = MATMUL(A, B)
""",
        lambda: {"A": gen_t2(3, 5, 1), "B": gen_t2(5, 4, 2)},
        {"kind": "T", "shape": [3, 4], "t2": True},
        "nonfpu",
        True,
    ),
    (
        "t_transpose",
        """IN A AS T:*
OUT = TRANSPOSE(A)
""",
        lambda: {"A": t3([1, 1, -1, 1, -1, 1],
                         [32768, 33000, 32500, 32768, 34000, 33200],
                         [0, 0, 0, 1, 0, 0], [2, 3])},
        {"kind": "T", "shape": [3, 2]},
        "nonfpu",
        True,
    ),
    (
        "t_slice",
        """IN A AS T:*
OUT = SLICE(A, 1, 1, 3)
""",
        lambda: {"A": t3([1, 1, -1, 1, -1, 1, -1, 1],
                         [32768, 33000, 32500, 32768, 34000, 33200, 32900, 33100],
                         [0, 0, 0, 1, 0, 0, 0, 0], [2, 4])},
        {"kind": "T", "shape": [2, 2]},
        "nonfpu",
        True,
    ),
    (
        "t_tshift",
        """IN A AS T:*
OUT = TSHIFT(A)
""",
        lambda: {"A": t3([1, 1, -1, 1, -1, 1],
                         [32768, 33000, 32500, 32768, 34000, 33200],
                         [0, 0, 0, 1, 0, 0], [2, 3])},
        {"kind": "T", "shape": [2, 3]},
        "nonfpu",
        True,
    ),
    (
        "t_tbeta",
        """CONFIG beta_b 0.31622776601683794
IN A AS T:*
OUT = TBETA(A)
""",
        lambda: {"A": t3([1, 1, -1, 1], [32768, 33000, 32500, 32768],
                         [0, 0, 0, 1], [4])},
        {"kind": "T", "shape": [4]},
        "nonfpu",
        True,
    ),
    (
        "t_silu",
        """IN A AS T:*
OUT = SILU(A)
""",
        lambda: {"A": t3([1, -1, 1, 1, -1, 1],
                         [32768, 33000, 32500, 32768, 34000, 30000],
                         [0, 0, 0, 1, 0, 0], [6])},
        {"kind": "T", "shape": [6]},
        "nonfpu",
        True,
    ),
    (
        "t_softmax_wide",
        """IN A AS T:*
OUT = SOFTMAX_WIDE(A)
""",
        lambda: {"A": t3([1, 1, -1, 1, -1, 1, -1, 1],
                         [32768, 33000, 32500, 32768, 34000, 33200, 32900, 33100],
                         [0, 0, 0, 1, 0, 0, 0, 0], [2, 4])},
        {"kind": "T", "shape": [2, 4]},
        # non-FPU is off by one rung on element 4 of this case; the
        # interpreter is the oracle, so gate against it.
        "interp",
        True,
    ),
    (
        "t_rmsnorm",
        """IN X AS T:*
IN W AS T:*
OUT = RMSNORM(X, W)
""",
        lambda: {
            "X": t3([1, 1, -1, 1, -1, 1, -1, 1],
                    [32768, 33000, 32500, 32768, 34000, 33200, 32900, 33100],
                    [0, 0, 0, 1, 0, 0, 0, 0], [2, 4]),
            "W": t3([1, 1, 1, -1], [32768, 32900, 33000, 32768],
                    [0, 0, 0, 0], [4]),
        },
        {"kind": "T", "shape": [2, 4]},
        "nonfpu",
        True,
    ),
    (
        "i_t2i",
        """IN A AS T:*
OUT = T2I(A)
""",
        lambda: {"A": t3([1, -1, 1, 1, -1, 1],
                         [32768, 33000, 32500, 32768, 34000, 30000],
                         [0, 0, 0, 1, 0, 0], [6])},
        {"kind": "I64", "shape": [6]},
        "nonfpu",
        True,
    ),
    (
        "i_iget",
        """IN X AS I:*
OUT = IGET(X, 2)
""",
        lambda: {"X": i64([11, 22, 33, 44], [4])},
        {"kind": "I64", "shape": [1]},
        "c",
        True,
    ),
    (
        "i_chain",
        """IN A AS I:*
IN B AS I:*
S = IADD(A, B)
OUT = IMUL(S, S)
""",
        lambda: {
            "A": i64([1, -2, 3, -4], [4]),
            "B": i64([10, 20, -30, 40], [4]),
        },
        {"kind": "I64", "shape": [4]},
        "c",
        True,
    ),
    (
        "f_matmul",
        """IN A AS F:*
IN B AS F:*
OUT = MATMUL(A, B)
""",
        lambda: {
            "A": f64([1, 2, 3, 4], [2, 2]),
            "B": f64([5, 6, 7, 8], [2, 2]),
        },
        {"kind": "F64", "shape": [2, 2]},
        "c",
        False,
    ),
]

FILE_FMT = {"b": 1, "B": 1, "i": 4, "q": 8, "d": 8, "f": 4}


def planes(stream):
    """(suffix, struct format, key) for each plane in argv order."""
    kind = stream["kind"]
    if kind in ("T", "T2"):
        if stream.get("m") is not None:
            return [("s", "b", "s"), ("e", "i", "e"),
                    ("m", "i", "m"), ("z", "B", "z")]
        return [("s", "b", "s"), ("e", "i", "e"), ("z", "B", "z")]
    if kind == "I64":
        return [("", "q", "data")]
    return [("", "d", "data")]


def out_planes(spec):
    """Plane layout for an output spec (no data required)."""
    if spec["kind"] in ("T", "T2"):
        if spec.get("t2"):
            return [("s", "b"), ("e", "i"), ("m", "i"), ("z", "B")]
        return [("s", "b"), ("e", "i"), ("z", "B")]
    if spec["kind"] == "I64":
        return [("", "q")]
    return [("", "d")]


def numel(shape):
    n = 1
    for d in shape:
        n *= d
    return n


def emit(path, fmt, values):
    vals = [float(v) if fmt in "df" else int(v) for v in values]
    open(path, "wb").write(struct.pack("<" + fmt * len(vals), *vals))


def load(path, fmt, count):
    return list(struct.unpack("<" + fmt * count, open(path, "rb").read()))


def run_case(bin_path, case, workroot):
    name, listing, make_streams, out_spec, ref_target, exact = case
    work = os.path.join(workroot, name)
    os.makedirs(work, exist_ok=True)

    streams = make_streams()
    listing_path = os.path.join(work, "case.asm")
    open(listing_path, "w").write(listing)
    payload_path = os.path.join(work, "payload.json")
    json.dump({"streams": streams}, open(payload_path, "w"))

    def build(target, exe_name):
        subprocess.run(
            [bin_path, "build", listing_path, "--payload", payload_path,
             "--workdir", work, "--name", exe_name, "--target", target,
             "--outputs", "OUT"],
            check=True, stdout=subprocess.DEVNULL,
        )
        man = json.load(open(os.path.join(work, exe_name + ".manifest.json")))
        return os.path.join(work, man["executable"]), man

    vk_exe, vk_man = build("vulkan", "vk")
    ref_exe = ref_man = None
    if ref_target != "interp":
        ref_exe, ref_man = build(ref_target, "ref")

    def argv_for(exe, man, tag):
        n = numel(out_spec["shape"])
        argv = []
        for s_name in streams:
            s = streams[s_name]
            for suffix, fmt, key in planes(s):
                if fmt == "d" and man["dtype"] == "f32":
                    fmt = "f"
                p = os.path.join(work, f"in.{tag}.{s_name}{suffix}.bin")
                emit(p, fmt, s[key])
                argv.append(p)
        outs = []
        for suffix, fmt in out_planes(out_spec):
            if fmt == "d" and man["dtype"] == "f32":
                fmt = "f"
            p = os.path.join(work, f"out.{tag}{suffix}.bin")
            outs.append((suffix, fmt, p))
            argv.append(p)
        return argv, outs, n

    vk_argv, vk_outs, _ = argv_for(vk_exe, vk_man, "vk")

    if ref_target == "interp":
        # Compare against the interpreter itself. Used where an emitter and the
        # oracle disagree; the oracle wins.
        subprocess.run([vk_exe] + vk_argv, check=True, stdout=subprocess.DEVNULL)
        exp = json.loads(subprocess.check_output(
            [bin_path, "run", listing_path, "--payload", payload_path],
            text=True))["streams"]["OUT"]
        ok = True
        for suffix, fmt, path in vk_outs:
            key = suffix if suffix else "data"
            raw = exp[key]
            want = [float(x) for x in raw] if fmt in ("d", "f") else [int(x) for x in raw]
            got = load(path, fmt, len(want))
            if got != want:
                ok = False
                label = name + ("." + suffix if suffix else "")
                print(f"  FAIL {label}: interp={want[:8]} vk={got[:8]}")
        print(f"{'PASS' if ok else 'FAIL'} {name} (interpreter reference, exact)")
        return ok

    ref_argv, ref_outs, count = argv_for(ref_exe, ref_man, "ref")
    subprocess.run([ref_exe] + ref_argv, check=True, stdout=subprocess.DEVNULL)
    subprocess.run([vk_exe] + vk_argv, check=True, stdout=subprocess.DEVNULL)

    ok = True
    for (suffix, rfmt, rpath), (_, vfmt, vpath) in zip(ref_outs, vk_outs):
        a = load(rpath, rfmt, count)
        b = load(vpath, vfmt, count)
        if exact:
            same = a == b
        else:
            same = len(a) == len(b) and all(
                abs(x - y) <= 1e-5 * max(1.0, abs(x)) for x, y in zip(a, b))
        if not same:
            ok = False
            label = name + ("." + suffix if suffix else "")
            print(f"  FAIL {label}: ref={a[:8]} vk={b[:8]}")
    print(f"{'PASS' if ok else 'FAIL'} {name} "
          f"({ref_target} reference, {'exact' if exact else 'tolerance 1e-5'})")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", default=os.path.join(ROOT, "target", "release", "jotnar"))
    ap.add_argument("--keep", action="store_true")
    args = ap.parse_args()

    if not os.path.exists(args.bin):
        print(f"missing jotnar binary: {args.bin}", file=sys.stderr)
        return 2

    workroot = tempfile.mkdtemp(prefix="parity-vulkan-")
    ok = True
    for case in CASES:
        ok = run_case(args.bin, case, workroot) and ok
    if not args.keep:
        shutil.rmtree(workroot, ignore_errors=True)
    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
