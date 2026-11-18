#!/usr/bin/env python3
"""Run one listing on two GPU runners and compare the typed outputs."""
import argparse
import json
import math
import socket
import sys

from gpu_runner_client import recv_frame, send_frame


def run(addr, token, target, listing, payload, outputs):
    host, port = addr.rsplit(":", 1)
    sock = socket.create_connection((host, int(port)), timeout=600)
    try:
        header = {
            "v": 1,
            "id": 1,
            "type": "run",
            "target": target,
            "outputs": outputs,
        }
        if token:
            header["auth"] = token
        body = json.dumps({"listing": listing, "payload": payload}).encode()
        send_frame(sock, header, body)
        response, response_body = recv_frame(sock)
        if not response.get("ok"):
            raise RuntimeError(f"{target}: {response}")
        return json.loads(response_body)["outputs"]
    finally:
        sock.close()


def values(stream):
    kind = stream["kind"]
    if kind in ("I64", "U8", "Bool"):
        return [int(x) for x in stream["data"]]
    if kind in ("F32", "F64"):
        return [float(x) for x in stream["data"]]
    raise RuntimeError(f"unsupported stream kind {kind}")


def compare(name, a, b, atol, rtol):
    if a.get("kind") != b.get("kind") or a.get("shape") != b.get("shape"):
        raise AssertionError(
            f"{name}: kind/shape {a.get('kind')}/{a.get('shape')} != "
            f"{b.get('kind')}/{b.get('shape')}"
        )
    av = values(a)
    bv = values(b)
    if a["kind"] in ("I64", "U8", "Bool"):
        if av != bv:
            raise AssertionError(f"{name}: integer outputs differ")
        return f"{name}: exact ({len(av)} elements)"
    worst = 0.0
    for x, y in zip(av, bv):
        if not (math.isfinite(x) and math.isfinite(y)):
            if x != y:
                raise AssertionError(f"{name}: non-finite mismatch")
            continue
        worst = max(worst, abs(x - y))
        if abs(x - y) > atol + rtol * abs(y):
            raise AssertionError(f"{name}: {x} vs {y} exceeds tolerance")
    return f"{name}: maxabs={worst:.3e} ({len(av)} elements)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="first runner ADDR")
    ap.add_argument("--b", required=True, help="second runner ADDR")
    ap.add_argument("--target-a", default="hip")
    ap.add_argument("--target-b", default="cuda")
    ap.add_argument("--token", default=None)
    ap.add_argument("--listing", required=True)
    ap.add_argument("--payload", required=True)
    ap.add_argument("--outputs", required=True)
    ap.add_argument("--atol", type=float, default=1e-4)
    ap.add_argument("--rtol", type=float, default=1e-4)
    args = ap.parse_args()

    listing = open(args.listing, "r", encoding="utf-8").read()
    payload = json.load(open(args.payload, "r", encoding="utf-8"))
    outputs = [x for x in args.outputs.split(",") if x]
    a = run(args.a, args.token, args.target_a, listing, payload, outputs)
    b = run(args.b, args.token, args.target_b, listing, payload, outputs)
    print(f"{args.target_a} {args.a}  vs  {args.target_b} {args.b}")
    for name in outputs:
        print(compare(name, a[name], b[name], args.atol, args.rtol))
    print("PASS")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        sys.exit(1)
