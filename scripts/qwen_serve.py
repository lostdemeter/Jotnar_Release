#!/usr/bin/env python3
"""Qwen block: resident Vulkan vs the emitted C binary, at a chosen token count."""
import json
import os
import subprocess
import sys
import tempfile
import time

SCRIPTS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPTS)
import qwen_gate as q  # noqa: E402

S = int(os.environ.get("S", "8"))
FRAMES = int(os.environ.get("FRAMES", "20"))
RUNS = int(os.environ.get("RUNS", "5"))

D, GU, DOWN, OUT = 32, 32, 16, 9
H = [f"h0_{i}" for i in range(OUT)]
SERVE_IN = ["tokens", "pos"]

WORK = tempfile.mkdtemp(prefix=f"qwen-serve-{S}-")


def make_streams():
    rng = q.Lcg(20261120)
    return {
        "tokens": q.tstream([S, D], rng.vals(S * D, -0.5, 0.5)),
        "pos": {"kind": "I64", "shape": [S], "data": list(range(S))},
        "emb": q.tstream([D, D], rng.vals(D * D, -0.5, 0.5)),
        "pos_code": q.tstream([S, D], rng.vals(S * D, 0.5, 1.0)),
        "wq": q.tstream([D, D], rng.vals(D * D, -0.2, 0.2)),
        "wk": q.tstream([D, D], rng.vals(D * D, -0.2, 0.2)),
        "wv": q.tstream([D, D], rng.vals(D * D, -0.2, 0.2)),
        "wo": q.tstream([D, D], rng.vals(D * D, -0.2, 0.2)),
        "w_gate_up": q.tstream([D, GU], rng.vals(D * GU, -0.2, 0.2)),
        "w_down": q.tstream([DOWN, D], rng.vals(DOWN * D, -0.2, 0.2)),
        "rms1": q.tstream([D], rng.vals(D, 0.8, 1.2)),
        "rms2": q.tstream([D], rng.vals(D, 0.8, 1.2)),
        "pool": q.tstream([1, S], rng.vals(S, -4.0, 4.0)),
        "w_out": q.tstream([D, OUT], rng.vals(D * OUT, -0.2, 0.2)),
    }


def build(bin_path, listing, ppath, target, name):
    subprocess.run(
        [bin_path, "build", listing, "--payload", ppath, "--workdir", WORK,
         "--name", name, "--target", target, "--outputs", ",".join(H)],
        check=True, stdout=subprocess.DEVNULL)
    man = json.load(open(os.path.join(WORK, name + ".manifest.json")))
    return os.path.join(WORK, man["executable"]), man


def main():
    listing = os.path.join(WORK, "qwen_serve.asm")
    open(listing, "w").write(
        f"CONFIG serve_input {','.join(SERVE_IN)}\n"
        f"CONFIG serve_output {','.join(H)}\n" + q.LISTING)
    streams = make_streams()
    ppath = os.path.join(WORK, "payload.json")
    json.dump({"streams": streams}, open(ppath, "w"))

    bin_path = os.path.join(os.path.dirname(SCRIPTS), "target", "release", "jotnar")
    vk, vk_man = build(bin_path, listing, ppath, "vulkan", "vk")
    ref, ref_man = build(bin_path, listing, ppath, "nonfpu", "ref")

    # one file per plane, in manifest order
    argv = []
    for name in ref_man["inputs"]:
        for suffix, fmt, key in q.planes(streams[name]):
            p = os.path.join(WORK, f"in.{name}{suffix}.bin")
            q.emit(p, fmt, streams[name][key])
            argv.append(p)
    outs = []
    for name in ref_man["outputs"]:
        p = os.path.join(WORK, f"out.{name}.bin")
        argv.append(p)
        outs.append(p)

    subprocess.run([ref] + argv, check=True, stdout=subprocess.DEVNULL)
    expect = b"".join(open(p, "rb").read() for p in outs)

    # a frame is the serve-input planes concatenated
    frame = b""
    for name in SERVE_IN:
        for suffix, fmt, key in q.planes(streams[name]):
            p = os.path.join(WORK, f"in.{name}{suffix}.bin")
            frame += open(p, "rb").read()

    one = subprocess.run([vk] + argv, input=frame, stdout=subprocess.PIPE, check=True)
    print(f"S={S}: frame={len(frame)} B  output={len(one.stdout)} B  "
          f"matches one-shot: {one.stdout == expect}")

    t = time.perf_counter()
    subprocess.run([vk] + argv, input=frame * FRAMES,
                   stdout=subprocess.DEVNULL, check=True)
    vk_ms = (time.perf_counter() - t) / FRAMES * 1000

    times = []
    for _ in range(RUNS):
        t = time.perf_counter()
        subprocess.run([ref] + argv, check=True, stdout=subprocess.DEVNULL)
        times.append(time.perf_counter() - t)
    times.sort()
    c_ms = times[len(times) // 2] * 1000

    print(f"resident vulkan : {vk_ms:7.2f} ms/frame")
    print(f"non-FPU process : {c_ms:7.2f} ms/run")
    print(f"speedup         : {c_ms/vk_ms:6.2f}x")


if __name__ == "__main__":
    main()
