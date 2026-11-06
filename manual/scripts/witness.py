#!/usr/bin/env python3
"""Run the manual's first four listings through the Jotnar interpreter.

This is the small, honest part of the build that uses the language itself:
the prose is host code, the examples are witnessed by the reference
implementation.
"""
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REPO = Path(os.environ.get("JOTNAR_REPO", ROOT.parent))
if not (REPO / "Cargo.toml").is_file():
    REPO = ROOT.parent / "rewrite_in_rust"
BIN = Path(os.environ.get("JOTNAR_BIN", REPO / "target" / "release" / "jotnar"))
LANG = REPO / "showcase" / "language"

I64 = lambda xs: {"kind": "I64", "shape": [len(xs)], "data": [str(x) for x in xs]}

CASES = [
    {
        "name": "saxpy",
        "file": LANG / "saxpy.asm",
        "streams": {"A": I64([3, 3, 3, 3]), "X": I64([1, 2, 3, 4]), "Y": I64([10, 20, 30, 40])},
        "outputs": ["OUT"],
        "expect": {"OUT": [13, 26, 39, 52]},
    },
    {
        "name": "reduce",
        "file": LANG / "reduce.asm",
        "streams": {f"A{i}": I64([i]) for i in range(8)},
        "outputs": ["OUT"],
        "expect": {"OUT": [28]},
    },
    {
        "name": "matmul2",
        "file": LANG / "matmul2.asm",
        "streams": {
            "a00": I64([1]), "a01": I64([0]), "a10": I64([0]), "a11": I64([1]),
            "b00": I64([1]), "b01": I64([2]), "b10": I64([3]), "b11": I64([4]),
        },
        "outputs": ["c00", "c01", "c10", "c11"],
        "expect": {"c00": [1], "c01": [2], "c10": [3], "c11": [4]},
    },
    {
        "name": "select",
        "file": LANG / "select.asm",
        "streams": {
            "C": I64([1, 0, 1, 0]),
            "A": I64([10, 20, 30, 40]),
            "B": I64([-1, -2, -3, -4]),
        },
        "outputs": ["OUT"],
        "expect": {"OUT": [10, -2, 30, -4]},
    },
]


def values(stream):
    return [int(x) for x in stream["data"]]


def main():
    if not BIN.is_file():
        print(f"jotnar binary not found: {BIN}", file=sys.stderr)
        print("set JOTNAR_BIN or build rewrite_in_rust first", file=sys.stderr)
        return 2
    failures = 0
    for case in CASES:
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump({"streams": case["streams"]}, f)
            payload = f.name
        try:
            proc = subprocess.run(
                [
                    str(BIN), "run", str(case["file"]),
                    "--payload", payload,
                    "--outputs", ",".join(case["outputs"]),
                ],
                capture_output=True,
                text=True,
            )
        finally:
            os.unlink(payload)
        if proc.returncode != 0:
            print(f"FAIL {case['name']}: {proc.stderr.strip()}")
            failures += 1
            continue
        got = json.loads(proc.stdout)["streams"]
        ok = True
        for name, want in case["expect"].items():
            have = values(got[name])
            if have != want:
                print(f"FAIL {case['name']}.{name}: want {want}, got {have}")
                ok = False
        print(f"{'PASS' if ok else 'FAIL'} {case['name']}")
        failures += 0 if ok else 1
    print(f"\n{len(CASES) - failures}/{len(CASES)} listings witnessed")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
