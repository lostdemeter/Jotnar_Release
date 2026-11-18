#!/usr/bin/env python3
"""Send one job to a jotnar-gpu-runner.

The frame format is the same as the node protocol:
    u32BE length || JSON header || '\n' || payload

Usage:
    python3 scripts/gpu_runner_client.py --addr 127.0.0.1:9997 \
        --target cuda --listing job.asm --payload payload.json \
        --outputs OUT
"""
import argparse
import json
import socket
import struct
import sys


def send_frame(sock, header, payload):
    body = json.dumps(header, separators=(",", ":")).encode() + b"\n" + payload
    sock.sendall(struct.pack(">I", len(body)) + body)


def recv_frame(sock):
    hdr = b""
    while len(hdr) < 4:
        chunk = sock.recv(4 - len(hdr))
        if not chunk:
            raise RuntimeError("connection closed before frame header")
        hdr += chunk
    n = struct.unpack(">I", hdr)[0]
    body = b""
    while len(body) < n:
        chunk = sock.recv(n - len(body))
        if not chunk:
            raise RuntimeError("connection closed before frame body")
        body += chunk
    header, _, payload = body.partition(b"\n")
    return json.loads(header), payload


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--addr", default="127.0.0.1:9997")
    ap.add_argument("--token", default=None)
    ap.add_argument("--target", choices=["hip", "cuda"], default="cuda")
    ap.add_argument("--listing")
    ap.add_argument("--payload")
    ap.add_argument("--outputs")
    ap.add_argument("--ping", action="store_true")
    args = ap.parse_args()

    host, port = args.addr.rsplit(":", 1)
    sock = socket.create_connection((host, int(port)), timeout=300)
    try:
        if args.ping:
            header = {"v": 1, "id": 1, "type": "ping"}
            if args.token:
                header["auth"] = args.token
            send_frame(sock, header, b"")
            header, payload = recv_frame(sock)
            print(json.dumps(header, indent=2))
            return 0 if header.get("ok") else 1

        if not args.listing or not args.payload or not args.outputs:
            raise SystemExit("--listing, --payload, and --outputs are required")
        listing = open(args.listing, "r", encoding="utf-8").read()
        payload_obj = json.load(open(args.payload, "r", encoding="utf-8"))
        body = json.dumps({"listing": listing, "payload": payload_obj}).encode()
        header = {
            "v": 1,
            "id": 1,
            "type": "run",
            "target": args.target,
            "outputs": [x for x in args.outputs.split(",") if x],
        }
        if args.token:
            header["auth"] = args.token
        send_frame(sock, header, body)
        header, payload = recv_frame(sock)
        print(json.dumps(header, indent=2))
        if payload:
            print(payload.decode())
        return 0 if header.get("ok") else 1
    finally:
        sock.close()


if __name__ == "__main__":
    sys.exit(main())
