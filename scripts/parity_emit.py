#!/usr/bin/env python3
"""Emitter parity: Python `chain.emit_c` vs the Rust codegen.

For each case both compilers emit C for the same listing/sample, both
binaries run on the same input files, and outputs are compared. Triples and
integers must match exactly; floats use a per-case tolerance.

Usage:
  PHI_CORE_DIR=... JOTNAR_PY=... python3 scripts/parity_emit.py \
      --bin target/debug/jotnar [filter]
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
                       os.path.join(ROOT, "..", "jotnar-main"), ROOT)
if not PHI_CORE or not JOTNAR_PY:
    sys.exit("set PHI_CORE_DIR and JOTNAR_PY")
sys.path.insert(0, PHI_CORE)
sys.path.insert(0, JOTNAR_PY)
import phi_core.lattice as L
from chain.emit_c import compile_program as py_compile, build as py_build


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
    if np.issubdtype(a.dtype, np.integer):
        return {"kind": "I64", "shape": list(a.shape),
                "data": [str(int(x)) for x in a.reshape(-1)]}
    return {"kind": "F64", "shape": list(a.shape),
            "data": [float(x) for x in a.reshape(-1)]}


class Case:
    def __init__(self, name, target, text, payload, outputs, tol=0.0,
                 basedir=None, stdlib=None):
        self.name, self.target = name, target
        self.text, self.payload, self.outputs = text, payload, outputs
        self.tol = tol
        self.basedir = basedir or os.path.join(JOTNAR_PY, "programs")
        self.stdlib = stdlib or os.path.join(JOTNAR_PY, "stdlib")


CASES = []


def case(name, target, text, payload, outputs, **kw):
    CASES.append(Case(name, target, text, payload, outputs, **kw))


def write_inputs(td, art, payload):
    argv = []
    for n in art["inputs"]:
        k, _, sh = art["streams"][n]
        if k == "T":
            for comp, dt in (("s", np.int8), ("e", np.int32), ("z", np.uint8)):
                fn = os.path.join(td, f"in_{n}_{comp}.bin")
                np.ascontiguousarray(payload[n]["sez".index(comp)]).astype(dt).tofile(fn)
                argv.append(fn)
        elif k == "F":
            fn = os.path.join(td, f"in_{n}.bin")
            np.ascontiguousarray(payload[n], dtype=np.float64).tofile(fn)
            argv.append(fn)
        else:
            fn = os.path.join(td, f"in_{n}.bin")
            np.ascontiguousarray(payload[n], dtype=np.int64).tofile(fn)
            argv.append(fn)
    return argv


def output_paths(td, art):
    argv, paths = [], []
    for o in art["outputs"]:
        ok = art["streams"][o][0]
        if ok == "T":
            for comp, dt in (("s", np.int8), ("e", np.int32), ("z", np.uint8)):
                fn = os.path.join(td, f"out_{o}_{comp}.bin")
                argv.append(fn)
                paths.append((o, comp, fn, dt))
        else:
            fn = os.path.join(td, f"out_{o}.bin")
            argv.append(fn)
            paths.append((o, None, fn, np.int64 if ok == "I" else np.float64))
    return argv, paths


def run_case(c, binary, td):
    for d in (os.path.join(td, "py"), os.path.join(td, "rs")):
        os.makedirs(d, exist_ok=True)
    py_src = py_compile(c.text, c.target, sample=c.payload,
                        outputs=c.outputs, basedir=c.basedir)
    py_exe = py_build(py_src["source"], os.path.join(td, "py"), name=c.name,
                      libs=[] if c.target == "nonfpu" else None)
    pay = os.path.join(td, "payload.json")
    json.dump({"streams": {k: enc(v) for k, v in c.payload.items()}},
              open(pay, "w"))
    rs_build = subprocess.run(
        [binary, "build", os.path.join(td, "case.asm"), "--payload", pay,
         "--outputs", ",".join(c.outputs), "--target", c.target,
         "--workdir", os.path.join(td, "rs"), "--name", c.name,
         "--stdlib", c.stdlib],
        capture_output=True, text=True)
    if rs_build.returncode != 0:
        return f"RUST BUILD FAILED: {rs_build.stderr.strip()[:400]}"
    rs_exe = rs_build.stdout.strip().splitlines()[-1]

    pay_py = write_inputs(os.path.join(td, "py"), py_src, c.payload)
    pay_rs = write_inputs(os.path.join(td, "rs"), py_src, c.payload)
    out_py, paths = output_paths(os.path.join(td, "py"), py_src)
    out_rs, _ = output_paths(os.path.join(td, "rs"), py_src)
    rp = subprocess.run([py_exe] + pay_py + out_py, capture_output=True,
                        text=True)
    rr = subprocess.run([rs_exe] + pay_rs + out_rs, capture_output=True,
                        text=True)
    if rp.returncode != 0:
        return f"PY RUN FAILED rc={rp.returncode} {rp.stderr[:200]}"
    if rr.returncode != 0:
        return f"RUST RUN FAILED rc={rr.returncode} {rr.stderr[:200]}"
    for o, comp, fn_py, dt in paths:
        got = np.fromfile(fn_py.replace(os.sep + "py" + os.sep,
                                        os.sep + "rs" + os.sep), dt)
        ref = np.fromfile(fn_py, dt)
        if ref.size != got.size:
            return f"{o}/{comp}: size {ref.size} vs {got.size}"
        if dt == np.float64:
            d = np.abs(got - ref)
            if d.size and d.max() > c.tol:
                return f"{o}/{comp}: maxabs={d.max():.3g} > {c.tol}"
        elif not np.array_equal(got, ref):
            return f"{o}/{comp}: value mismatch"
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", default=os.path.join(ROOT, "target", "debug", "jotnar"))
    ap.add_argument("filter", nargs="?", default="")
    args = ap.parse_args()
    binary = os.path.abspath(args.bin)
    ok = fail = 0
    with tempfile.TemporaryDirectory() as root:
        for c in CASES:
            if args.filter and args.filter not in c.name:
                continue
            td = os.path.join(root, c.name)
            os.makedirs(td, exist_ok=True)
            open(os.path.join(td, "case.asm"), "w").write(c.text)
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
rng = np.random.default_rng(11)


def tri(a):
    return L.encode(np.ascontiguousarray(a, dtype=np.float64))


# --- C backend F paths ---
case("c_silu", "c", "IN a\nOUT = SILU(a)\n",
     {"a": rng.normal(0, 1, (3, 4))}, ["OUT"], tol=0.0)
case("c_rmsnorm", "c", "CONFIG eps_rms 1e-6\nIN x\nIN w\nOUT = RMSNORM(x, w)\n",
     {"x": rng.normal(0, 1, (2, 8)), "w": rng.uniform(0.5, 1.5, 8)},
     ["OUT"], tol=0.0)
case("c_layernorm", "c", "CONFIG eps_ln 1e-5\nIN x\nIN w\nIN b\nOUT = LAYERNORM(x, w, b)\n",
     {"x": rng.normal(0, 1, (2, 8)), "w": rng.uniform(0.5, 1.5, 8),
      "b": rng.normal(0, 0.1, 8)}, ["OUT"], tol=0.0)
case("c_matmul", "c", "IN a\nIN b\nOUT = MATMUL(a, b)\n",
     {"a": rng.normal(0, 1, (3, 4)), "b": rng.normal(0, 1, (4, 2))},
     ["OUT"], tol=0.0)
case("c_batch_matmul", "c", "IN a\nIN b\nOUT = BATCH_MATMUL(a, b)\n",
     {"a": rng.normal(0, 1, (2, 3, 4)), "b": rng.normal(0, 1, (1, 4, 2))},
     ["OUT"], tol=0.0)
case("c_bmmv", "c",
     "IN a\nIN b\nOUT = BMMV(a, b, 2, 2, 2, 3, 3, 6, 2, 6, 2, 4, 0)\n",
     {"a": rng.normal(0, 1, (4, 3)), "b": rng.normal(0, 1, (6, 2))},
     ["OUT"], tol=0.0)
case("c_rotary", "c", "IN q\nIN pos\nOUT = ROTARY(q, pos)\n",
     {"q": rng.normal(0, 1, (4, 8)), "pos": np.arange(4, dtype=np.int64)},
     ["OUT"], tol=0.0)
for op in ("TSHIFT", "SOFTMAX_WIDE"):
    case(f"c_{op.lower()}", "c", f"IN s\nOUT = {op}(s)\n",
         {"s": rng.normal(0, 2, (3, 5))}, ["OUT"], tol=0.0)
case("c_conv", "c", "IN a\nIN k\nOUT = CONV(a, k)\n",
     {"a": rng.uniform(0, 1, (5, 6)),
      "k": np.array([[1., 2., 1.], [2., 4., 2.], [1., 2., 1.]]) / 16.0},
     ["OUT"], tol=0.0)
case("c_interp", "c", "IN a\nOUT = INTERP(a, 2.0, 2.0)\n",
     {"a": rng.uniform(0, 1, (3, 4, 2))}, ["OUT"], tol=0.0)
case("c_scatter", "c", "IN b\nIN r\nIN i\nOUT = SCATTER(b, r, i)\n",
     {"b": rng.normal(0, 1, (4, 3)), "r": rng.normal(0, 1, (1, 3)),
      "i": np.array([2], np.int64)}, ["OUT"])
case("c_gather_f", "c", "IN w\nIN ids\nOUT = GATHER(w, ids)\n",
     {"w": rng.normal(0, 1, (5, 2)), "ids": np.array([0, 4, 2], np.int64)},
     ["OUT"])
case("c_gather3_f", "c", "IN w\nIN i\nIN j\nOUT = GATHER3(w, i, j)\n",
     {"w": rng.normal(0, 1, (3, 4, 2)), "i": np.array([0, 2], np.int64),
      "j": np.array([1, 3], np.int64)}, ["OUT"])
case("c_slice_f", "c", "IN a\nOUT = SLICE(a, 0, 1, 3)\n",
     {"a": rng.normal(0, 1, (4, 3))}, ["OUT"])
case("c_concat_f", "c", "IN a\nIN b\nOUT = CONCAT(a, b, 1)\n",
     {"a": rng.normal(0, 1, (3, 2)), "b": rng.normal(0, 1, (3, 4))}, ["OUT"])
case("c_transpose_f", "c", "IN a\nOUT = TRANSPOSE(a)\n",
     {"a": rng.normal(0, 1, (3, 4))}, ["OUT"])
case("c_select_f", "c", "IN m\nIN a\nIN b\nOUT = SELECT(m, a, b)\n",
     {"m": np.array([[1, 0, 1], [0, 1, 1]], np.int64),
      "a": rng.normal(0, 1, (2, 3)), "b": rng.normal(0, 1, (2, 3))}, ["OUT"])
case("c_beta", "c", "CONFIG beta 0.37\nIN a\nOUT = BETA(a)\n",
     {"a": rng.normal(0, 1, (2, 3))}, ["OUT"])
case("c_tbeta", "c", "CONFIG beta 0.37\nCONFIG beta_b 0.8\nIN a\nOUT = TBETA(a)\n",
     {"a": rng.normal(0, 1, (2, 3))}, ["OUT"])
case("c_f2i", "c", "IN a\nOUT = F2I(a)\n",
     {"a": rng.normal(0, 10, (2, 3))}, ["OUT"])
case("c_i2f", "c", "IN a\nOUT = I2F(a)\n",
     {"a": rng.integers(-1000, 1000, (2, 3), dtype=np.int64)}, ["OUT"])
case("c_argmax_f", "c", "IN a\nOUT = ARGMAX(a, 1)\n",
     {"a": rng.normal(0, 1, (3, 5))}, ["OUT"])

# --- non-FPU exact paths ---
mov = ("IN t\nIN msk\nTR = TRANSPOSE(t)\nSL = SLICE(TR, 1, 1, 3)\n"
       "CT = CONCAT(SL, SL, 1)\nMS = SELECT(msk, CT, CT)\n")
rng2 = np.random.default_rng(3)
t = (rng2.integers(-5, 6, size=(4, 6)).astype(np.int8),
     rng2.integers(32000, 33500, size=(4, 6)).astype(np.int32),
     (rng2.random(size=(4, 6)) < 0.2).astype(np.uint8))
msk = np.array([[1, 0, 1, 1]] * 6, np.int64)
case("nf_moves", "nonfpu", mov, {"t": t, "msk": msk}, ["MS"])

arith = ("CONFIG m_cov 35048\nIN a\nIN b\n"
         "C = ADD(a, b)\nD = MUL(C, a)\nE = SUB(D, b)\nQ = SQUARE(a)\n")
def grid(g):
    ss = np.zeros_like(g, np.int8); ee = np.zeros_like(g, np.int32)
    zz = np.zeros_like(g, np.uint8)
    for v in np.unique(g):
        ws, we, wz = L.encode(np.array([float(v)])); m = g == v
        ss[m], ee[m], zz[m] = ws[0], we[0], wz[0]
    return (np.ascontiguousarray(ss), np.ascontiguousarray(ee), np.ascontiguousarray(zz))
rng3 = np.random.default_rng(5)
av = rng3.integers(-3, 4, size=(2, 8)); bv = rng3.integers(-3, 4, size=(2, 8))
case("nf_arith", "nonfpu", arith, {"a": grid(av), "b": grid(bv)}, ["E", "Q"])

mm = "CONFIG m_acc 36118\nCONFIG m_cov 35048\nIN a\nIN b\nOUT = MATMUL(a, b)\n"
rng4 = np.random.default_rng(9)
ma = rng4.integers(-3, 4, size=(3, 8)); mb = rng4.integers(-3, 4, size=(8, 5))
case("nf_matmul", "nonfpu", mm, {"a": grid(ma), "b": grid(mb)}, ["OUT"])

gt = "IN a\nIN b\nIN c\nM = GT(a, b)\nY = SELECT(M, a, c)\n"
rng5 = np.random.default_rng(7)
fa = rng5.normal(size=(3, 5)); fb = rng5.normal(size=(3, 5))
fa[0, 0], fb[0, 0] = 0.3, 0.3
fa[0, 1], fb[0, 1] = 0.0, 0.0
fa[0, 2], fb[0, 2] = -0.5, 0.5
case("nf_gt", "nonfpu", gt,
     {"a": tri(fa), "b": tri(fb), "c": tri(rng5.normal(size=(3, 5)))},
     ["M", "Y"])

# extended integer family (86-op surface)
ia = np.array([0, 1, -1, 7, -7, (1 << 63) - 1, -(1 << 63), 123456789,
               -123456789, 42], np.int64)
ib = np.array([1, 2, 3, 7, -3, 2, -5, 1000, 999, (1 << 63) - 1], np.int64)
ish = np.array([0, 1, 2, 3, 4, 5, 6, 62, 63, 1], np.int64)
icond = np.array([0, 1, -1, 2, 0, 1, 0, 1, 0, 1], np.int64)
iext_text = ("IN a\nIN b\nIN s\nIN c\n"
             "SADD_OUT = SADD(a, b)\nIMAX_OUT = IMAX(a, b)\n"
             "ISAR_OUT = ISAR(a, s)\nIDIVU_OUT = IDIVU(a, b)\n"
             "IMODU_OUT = IMODU(a, b)\nIEQ_OUT = IEQ(a, b)\n"
             "ILT_OUT = ILT(a, b)\nISELECT_OUT = ISELECT(c, a, b)\n")
iext_pay = {"a": ia, "b": ib, "s": ish, "c": icond}
iext_outs = ["SADD_OUT", "IMAX_OUT", "ISAR_OUT", "IDIVU_OUT", "IMODU_OUT",
             "IEQ_OUT", "ILT_OUT", "ISELECT_OUT"]
case("c_iext", "c", iext_text, iext_pay, iext_outs)
case("nf_iext", "nonfpu", iext_text, iext_pay, iext_outs)

# --- data-backed gates ---
DATA = os.path.join(JOTNAR_PY, "data")


def _load_headt():
    d = np.load(os.path.join(DATA, "lm_svd_IvoQ.npz"))
    b = np.load(os.path.join(DATA, "bankhn.npz"))
    ids = [12, 471, 59]
    toks = np.array(ids, np.int64)
    pos = np.arange(len(ids), dtype=np.int64)
    cm = np.tril(np.ones((len(ids), len(ids)), dtype=np.int64))
    return d, b, toks, pos, cm


def _headt_payload(fbits):
    d, b, toks, pos, cm = _load_headt()
    def e(a):
        a = np.ascontiguousarray(a, dtype=np.float64)
        return a if fbits else L.encode(a)
    return {"tok": toks, "pos": pos, "cmask": cm,
            "emb": e(d["emb"]), "wq": e(d["wq"]), "wk": e(d["wk"]),
            "wv": e(d["wv"]), "wo": e(d["wo"]), "wup": e(d["wup"]),
            "wgate": e(d["wgate"]), "wdown": e(d["wdown"]),
            "rms_w1": e(d["rms1"]), "rms_w2": e(d["rms2"]),
            "wlog": e(d["wlog"]), "ukt": e(b["ukt"]), "evb": e(b["evb"])}


if os.path.exists(os.path.join(DATA, "lm_svd_IvoQ.npz")):
    CFG = ("CONFIG m_acc 36118\nCONFIG m_cov 35048\nCONFIG beta -30.0\n"
           "CONFIG beta_b 0.25\nCONFIG eps_rms 1e-6\n")
    htext = CFG + open(os.path.join(
        JOTNAR_PY, "programs", "lm_headt.asm")).read()
    case("c_headt", "c", htext, _headt_payload(True),
         ["LOGITS", "OUT"], tol=6e-2)
    case("nf_headt", "nonfpu", htext, _headt_payload(False),
         ["LOGITS", "OUT"], tol=0.0)

if os.path.exists(os.path.join(DATA, "lm_bigrams.npz")):
    counts = np.load(os.path.join(DATA, "lm_bigrams.npz"))["counts"]
    bs = np.zeros_like(counts, np.int8)
    be = np.zeros_like(counts, np.int32)
    bz = np.zeros_like(counts, np.uint8)
    for v in np.unique(counts):
        if v == 0:
            continue
        ws, we, wz = L.encode(np.array([float(v)]))
        m = counts == v
        bs[m], be[m], bz[m] = ws[0], we[0], wz[0]
    bz[counts == 0] = 1
    bank = (np.ascontiguousarray(bs), np.ascontiguousarray(be),
            np.ascontiguousarray(bz))
    btext = open(os.path.join(
        JOTNAR_PY, "programs", "bigram_lm.asm")).read()
    rr = np.random.default_rng(0)
    nz = [i for i in range(counts.shape[0]) if counts[i].sum() > 0]
    toks = np.array(sorted(rr.choice(nz, size=200, replace=False).tolist()),
                    np.int64)
    case("nf_bigram", "nonfpu", btext, {"tok": toks, "bank": bank}, ["OUT"])

if __name__ == "__main__":
    sys.exit(main())
