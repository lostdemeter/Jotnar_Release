#!/usr/bin/env python3
"""Cost per dispatch: a chain of N ladder MULs, resident, fixed element count."""
import json
import os
import struct
import subprocess
import sys
import tempfile
import time

N_ELEM = int(os.environ.get("N", "4096"))
FRAMES = int(os.environ.get("FRAMES", "50"))
CHAIN = [int(x) for x in os.environ.get("CHAIN", "1,4,8,16,32").split(",")]
BIN = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "target", "release", "jotnar")


def listing(n):
    lines = ["CONFIG serve_input A", "CONFIG serve_output OUT",
             "IN A AS T:*", "IN B AS T:*"]
    prev = "A"
    for i in range(n):
        lines.append(f"v{i} = MUL({prev}, B)")
        prev = f"v{i}"
    lines.append(f"OUT = MUL({prev}, B)")
    return "\n".join(lines) + "\n"


def main():
    for n in CHAIN:
        work = tempfile.mkdtemp(prefix=f"chain-{n}-")
        lp = os.path.join(work, "c.asm")
        open(lp, "w").write(listing(n))
        s = [1] * N_ELEM
        e = [32768 + (i % 200) for i in range(N_ELEM)]
        z = [0] * N_ELEM
        pay = {"streams": {
            "A": {"kind": "T", "shape": [N_ELEM], "s": s, "e": e, "z": z},
            "B": {"kind": "T", "shape": [N_ELEM], "s": s, "e": e, "z": z}}}
        pp = os.path.join(work, "p.json")
        json.dump(pay, open(pp, "w"))
        subprocess.run([BIN, "build", lp, "--payload", pp, "--workdir", work,
                        "--name", "c", "--target", "vulkan", "--outputs", "OUT"],
                       check=True, stdout=subprocess.DEVNULL)
        man = json.load(open(os.path.join(work, "c.manifest.json")))
        exe = os.path.join(work, man["executable"])

        argv = []
        for name in ("A", "B"):
            for suffix, fmt, key in (("s", "b", "s"), ("e", "i", "e"), ("z", "B", "z")):
                p = os.path.join(work, f"{name}.{suffix}.bin")
                vals = pay["streams"][name][key]
                if fmt in "bB":
                    open(p, "wb").write(bytes([v & 0xFF for v in vals]))
                else:
                    open(p, "wb").write(struct.pack("<" + fmt * len(vals), *vals))
                argv.append(p)
        for suffix in ("s", "e", "z"):
            p = os.path.join(work, f"o.{suffix}.bin")
            argv.append(p)

        frame = b"".join(open(os.path.join(work, f"A.{x}.bin"), "rb").read()
                         for x in ("s", "e", "z"))
        blob = frame * FRAMES
        t = time.perf_counter()
        subprocess.run([exe] + argv, input=blob, stdout=subprocess.DEVNULL, check=True)
        ms = (time.perf_counter() - t) / FRAMES * 1000
        print(f"chain={n:2d} dispatches, {N_ELEM} elems: {ms:7.3f} ms/frame")


if __name__ == "__main__":
    main()
