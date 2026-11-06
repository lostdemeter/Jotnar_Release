#!/usr/bin/env python3
"""CUDA backend parity: Rust CUDA emission vs the lattice/numpy reference.

Covers the same gates as the reference `tests/test_emit_cuda.py`:
bigram decisions, compute units, ARGMAX ties, the headt block, fp16 storage,
and D32/D64 widths.

Usage:
  PHI_CORE_DIR=... JOTNAR_PY=... python3 scripts/parity_cuda.py \
      --bin target/release/jotnar [--skip-widths]
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
sys.path.insert(0, os.path.join(JOTNAR_PY, "scripts"))
import phi_core.lattice as L
from chain import asm as ASM
from chain.asm_ops import REGISTRY, SIGS

FAIL = []
EPS_F32 = 1e-4


def check(tag, cond, extra=""):
    print(f"{tag}: {'OK' if cond else 'FAIL'} {extra}", flush=True)
    if not cond:
        FAIL.append(tag)


def enc(v):
    if isinstance(v, tuple) and len(v) == 3 and hasattr(v[0], "shape"):
        s, e, z = v
        return {"kind": "T", "shape": list(s.shape),
                "s": [int(x) for x in s.reshape(-1)],
                "e": [int(x) for x in e.reshape(-1)],
                "z": [int(x) for x in z.reshape(-1)]}
    a = np.ascontiguousarray(v)
    if np.issubdtype(a.dtype, np.integer):
        return {"kind": "I64", "shape": list(a.shape),
                "data": [str(int(x)) for x in a.reshape(-1)]}
    return {"kind": "F64", "shape": list(a.shape),
            "data": [float(x) for x in a.reshape(-1)]}


def dec(t):
    return (L.decode(np.ascontiguousarray(t[0]), np.ascontiguousarray(t[1]))
            * (1 - np.ascontiguousarray(t[2]).astype(np.float64)))


def emit_run(binary, name, text, payload, outputs, dtmap, fp16=False,
             basedir=None, stdlib=None, tmp=None):
    td = tmp or tempfile.mkdtemp(prefix="cu_parity_")
    os.makedirs(td, exist_ok=True)
    asm = os.path.join(td, "case.asm")
    pay = os.path.join(td, "payload.json")
    open(asm, "w").write(text)
    json.dump({"streams": {k: enc(v) for k, v in payload.items()}},
              open(pay, "w"))
    args = [binary, "build", asm, "--payload", pay, "--outputs",
            ",".join(outputs), "--target", "cuda", "--workdir", td,
            "--name", name, "--stdlib", stdlib or os.path.join(JOTNAR_PY, "stdlib")]
    if fp16:
        args.append("--fp16")
    r = subprocess.run(args, capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError(r.stderr[-600:])
    exe = r.stdout.strip().splitlines()[-1]
    import re
    names = []
    for line in text.splitlines():
        code = line.split("#")[0].strip()
        m = re.match(r"IN\s+(.+?)(?:\s+AS\s+.*)?$", code)
        if m:
            names += [x.strip() for x in m.group(1).split(",") if x.strip()]
    argv = [exe]
    for n in names:
        fn = os.path.join(td, f"in_{n}.bin")
        v = payload[n]
        if np.issubdtype(np.asarray(v).dtype, np.integer):
            np.asarray(v, np.int64).tofile(fn)
        elif fp16:
            np.asarray(v, np.float64).astype(np.float16).tofile(fn)
        else:
            np.asarray(v, np.float64).astype(np.float32).tofile(fn)
        argv.append(fn)
    outs = []
    for o in outputs:
        fn = os.path.join(td, f"out_{o}.bin")
        outs.append(fn)
        argv.append(fn)
    rr = subprocess.run(argv, capture_output=True, text=True)
    if rr.returncode:
        raise RuntimeError(f"run rc={rr.returncode} {rr.stderr[-400:]}")
    return {o: np.fromfile(fn, dtmap[o]) for o, fn in zip(outputs, outs)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", default=os.path.join(ROOT, "target", "release", "jotnar"))
    ap.add_argument("--skip-widths", action="store_true")
    args = ap.parse_args()
    binary = os.path.abspath(args.bin)
    if subprocess.run(["which", "nvcc"], capture_output=True).returncode:
        print("SKIP (no nvcc)")
        return 0
    rng = np.random.default_rng(7)
    dd = os.path.join(JOTNAR_PY, "data")
    sdir = os.path.join(JOTNAR_PY, "programs")

    # units
    units = [
        ("mm", "IN a\nIN b\nOUT = MATMUL(a, b)\n",
         {"a": rng.normal(size=(8, 16)), "b": rng.normal(size=(16, 32))},
         lambda p: p["a"] @ p["b"], 0),
        ("bmm", "IN a\nIN b\nOUT = BATCH_MATMUL(a, b)\n",
         {"a": rng.normal(size=(2, 4, 8)), "b": rng.normal(size=(2, 8, 16))},
         lambda p: p["a"] @ p["b"], 0),
        ("rms", "CONFIG eps_rms 1e-6\nIN x\nIN w\nOUT = RMSNORM(x, w)\n",
         {"x": rng.normal(size=(4, 32)), "w": rng.normal(size=(32,))},
         lambda p: p["x"] / np.sqrt((p["x"] ** 2).mean(-1, keepdims=True) + 1e-6) * p["w"], 0),
        ("sm", "IN x\nOUT = SOFTMAX_WIDE(x)\n",
         {"x": rng.normal(size=(4, 64)) * 3},
         lambda p: (lambda e: e / e.sum(-1, keepdims=True))(np.exp(p["x"] - p["x"].max(-1, keepdims=True))), 0),
        ("elem", "IN a\nIN b\nC = ADD(a, b)\nOUT = MUL(C, a)\n",
         {"a": rng.normal(size=(4, 32)), "b": rng.normal(size=(4, 32))},
         lambda p: (p["a"] + p["b"]) * p["a"], 0),
    ]
    for name, text, payload, ref_fn, _ in units:
        payload = {k: np.ascontiguousarray(v, np.float64) for k, v in payload.items()}
        try:
            got = emit_run(binary, name, text, payload, ["OUT"], {"OUT": np.float32})
            ref = np.ascontiguousarray(ref_fn(payload))
            d = float(np.abs(got["OUT"].reshape(ref.shape) - ref).max())
            check(f"cu-{name}", d < EPS_F32, f"maxabs={d:.3e}")
        except Exception as e:  # noqa: BLE001
            check(f"cu-{name}", False, str(e)[:250])

    # argmax ties
    xa = rng.normal(size=(4, 64)); xa[2, :] = 1.0; xa[2, 0] = 1.0
    try:
        got = emit_run(binary, "am", "IN x\nOUT = ARGMAX(x, 1)\n",
                       {"x": xa}, ["OUT"], {"OUT": np.int64})
        check("cu-argmax", np.array_equal(got["OUT"], np.argmax(xa, axis=1)))
    except Exception as e:  # noqa: BLE001
        check("cu-argmax", False, str(e)[:250])

    # extended integer family
    ia = np.array([0, 1, -1, 7, -7, (1 << 63) - 1, -(1 << 63),
                   123456789, -123456789, 42], np.int64)
    ib = np.array([1, 2, 3, 7, -3, 2, -5, 1000, 999, (1 << 63) - 1], np.int64)
    ish = np.array([0, 1, 2, 3, 4, 5, 6, 62, 63, 1], np.int64)
    ic = np.array([0, 1, -1, 2, 0, 1, 0, 1, 0, 1], np.int64)
    iext = ("IN a\nIN b\nIN s\nIN c\n"
            "SA = SADD(a,b)\nMX = IMAX(a,b)\nSR = ISAR(a,s)\n"
            "QU = IDIVU(a,b)\nRM = IMODU(a,b)\nEQ = IEQ(a,b)\n"
            "LT = ILT(a,b)\nSE = ISELECT(c,a,b)\n")
    outs = ["SA", "MX", "SR", "QU", "RM", "EQ", "LT", "SE"]
    want = {
        "SA": np.clip(ia.astype(object) + ib.astype(object), -(1 << 63),
                      (1 << 63) - 1).astype(np.int64),
        "MX": np.maximum(ia, ib),
        "SR": ia >> ish,
        "QU": (ia.astype(np.uint64) // ib.astype(np.uint64)).astype(np.int64),
        "RM": (ia.astype(np.uint64) % ib.astype(np.uint64)).astype(np.int64),
        "EQ": (ia == ib).astype(np.int64),
        "LT": (ia < ib).astype(np.int64),
        "SE": np.where(ic != 0, ia, ib).astype(np.int64),
    }
    try:
        got = emit_run(binary, "iext", iext,
                       {"a": ia, "b": ib, "s": ish, "c": ic}, outs,
                       {k: np.int64 for k in outs})
        bad = [k for k in outs if not np.array_equal(got[k], want[k])]
        check("cu-iext", not bad, f"mismatch {bad}")
    except Exception as e:  # noqa: BLE001
        check("cu-iext", False, str(e)[:250])

    # bigram
    counts = np.load(os.path.join(dd, "lm_bigrams.npz"))["counts"]
    text = open(os.path.join(sdir, "bigram_lm.asm")).read()
    bank_f = np.ascontiguousarray(counts, np.float64)
    nz = [i for i in range(counts.shape[0]) if counts[i].sum() > 0]
    toks = np.array(sorted(rng.choice(nz, size=200, replace=False).tolist()), np.int64)
    bs = np.zeros_like(counts, np.int8); be = np.zeros_like(counts, np.int32)
    bz = np.zeros_like(counts, np.uint8)
    for v in np.unique(counts):
        if v == 0:
            continue
        ws, we, wz = L.encode(np.array([float(v)])); m = counts == v
        bs[m], be[m], bz[m] = ws[0], we[0], wz[0]
    bz[counts == 0] = 1
    ref = np.ascontiguousarray(ASM.run_text(
        text, REGISTRY, {"tok": toks, "bank": (bs, be, bz)},
        sigs=SIGS, basedir=sdir)["OUT"]).reshape(-1)
    try:
        got = emit_run(binary, "bigram", text,
                       {"tok": toks, "bank": bank_f}, ["OUT"], {"OUT": np.int64},
                       basedir=sdir)
        check("cu-bigram", np.array_equal(got["OUT"], ref))
    except Exception as e:  # noqa: BLE001
        check("cu-bigram", False, str(e)[:250])

    # headt + fp16
    if os.path.exists(os.path.join(dd, "lm_svd_IvoQ.npz")):
        cfg = ("CONFIG m_acc 36118\nCONFIG m_cov 35048\nCONFIG beta -30.0\n"
               "CONFIG beta_b 0.25\n")
        htext = cfg + open(os.path.join(sdir, "lm_headt.asm")).read()
        d = np.load(os.path.join(dd, "lm_svd_IvoQ.npz"))
        b = np.load(os.path.join(dd, "bankhn.npz"))
        toks = np.array([12, 471, 59], np.int64)
        pos = np.arange(3, dtype=np.int64)
        cm = np.tril(np.ones((3, 3), dtype=np.int64))
        def e3(a):
            return L.encode(np.ascontiguousarray(a, np.float64))
        trip = {"emb": e3(d["emb"]), "wq": e3(d["wq"]), "wk": e3(d["wk"]),
                "wv": e3(d["wv"]), "wo": e3(d["wo"]), "wup": e3(d["wup"]),
                "wgate": e3(d["wgate"]), "wdown": e3(d["wdown"]),
                "rms_w1": e3(d["rms1"]), "rms_w2": e3(d["rms2"]),
                "wlog": e3(d["wlog"]), "ukt": e3(b["ukt"]), "evb": e3(b["evb"])}
        pay = {"tok": toks, "pos": pos, "cmask": cm}; pay.update(trip)
        ref_f = ASM.run_text(htext, REGISTRY, pay, sigs=SIGS, basedir=sdir)
        ref_logits = dec(ref_f["LOGITS"])
        ref_out = np.ascontiguousarray(ref_f["OUT"]).reshape(-1)
        fpay = {"tok": toks, "pos": pos, "cmask": cm}
        fpay.update({k: np.ascontiguousarray(dec(v)) for k, v in trip.items()})
        for fp16 in (False, True):
            tag = "cu-headt16" if fp16 else "cu-headt"
            try:
                got = emit_run(binary, tag, htext, fpay, ["LOGITS", "OUT"],
                               {"LOGITS": np.float32, "OUT": np.int64},
                               fp16=fp16, basedir=sdir)
                dmax = float(np.abs(got["LOGITS"].reshape(ref_logits.shape)
                                    - ref_logits).max())
                check(tag + "-logits", dmax < 6e-2, f"maxabs={dmax:.3e}")
                check(tag + "-out", np.array_equal(got["OUT"], ref_out))
            except Exception as e:  # noqa: BLE001
                check(tag, False, str(e)[:250])

    # widths
    if not args.skip_widths:
        from d64_lib import encode as benc, load_bpe
        vocab, rank = load_bpe()
        cases = {"d32": ("lm_piece32_fit.npz", "bankp32_64.npz", "lm_d32_headt.asm"),
                 "d64": ("lm_piece64_fit.npz", "bankpiece64_d64.npz", "lm_d64_headt.asm")}
        cfg = ("CONFIG m_acc 36118\nCONFIG m_cov 35048\nCONFIG beta -30.0\n"
               "CONFIG beta_b 0.25\n")
        for tag, (fit, bank, listing) in cases.items():
            ids = benc("alexander founded alexandria and ruled egypt well", vocab, rank)[:8]
            toks = np.array(ids, np.int64)
            pos = np.arange(len(ids), dtype=np.int64)
            cm = np.tril(np.ones((len(ids), len(ids)), dtype=np.int64))
            w = np.load(os.path.join(dd, fit)); b = np.load(os.path.join(dd, bank))
            text = cfg + open(os.path.join(sdir, listing)).read()
            def e3(a):
                return L.encode(np.ascontiguousarray(a, np.float64))
            trip = {"emb": e3(w["emb"]), "wq": e3(w["wq"]), "wk": e3(w["wk"]),
                    "wv": e3(w["wv"]), "wo": e3(w["wo"]), "wup": e3(w["wup"]),
                    "wgate": e3(w["wgate"]), "wdown": e3(w["wdown"]),
                    "rms_w1": e3(w["rms1"]), "rms_w2": e3(w["rms2"]),
                    "wlog": e3(w["wlog"]), "ukt": e3(b["ukt"]), "evb": e3(b["evb"])}
            pay = {"tok": toks, "pos": pos, "cmask": cm}; pay.update(trip)
            ref_f = ASM.run_text(text, REGISTRY, pay, sigs=SIGS, basedir=sdir)
            fpay = {"tok": toks, "pos": pos, "cmask": cm}
            fpay.update({k: np.ascontiguousarray(dec(v)) for k, v in trip.items()})
            try:
                got = emit_run(binary, tag, text, fpay, ["H2", "LOGITS", "OUT"],
                               {"H2": np.float32, "LOGITS": np.float32, "OUT": np.int64},
                               basedir=sdir)
                d2 = float(np.abs(got["H2"].reshape(dec(ref_f["H2"]).shape)
                                  - dec(ref_f["H2"])).max())
                check(f"cu-{tag}-h2", d2 < 2e-2, f"maxabs={d2:.3e}")
                check(f"cu-{tag}-out",
                      np.array_equal(got["OUT"],
                                     np.ascontiguousarray(ref_f["OUT"]).reshape(-1)))
            except Exception as e:  # noqa: BLE001
                check(f"cu-{tag}", False, str(e)[:250])

    print("FAILURES:", FAIL if FAIL else "none")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
