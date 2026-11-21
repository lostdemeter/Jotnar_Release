#!/usr/bin/env python3
"""Time the emitted Qwen block for each target (build once, run N times)."""
import json
import os
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import qwen_gate as q  # noqa: E402

RUNS = int(os.environ.get("RUNS", "5"))


def bench(target, bin_path):
    work = tempfile.mkdtemp(prefix=f"qwen-bench-{target}-")
    listing = os.path.join(work, "qwen.asm")
    open(listing, "w").write(q.LISTING)
    pay = q.payload()
    ppath = os.path.join(work, "payload.json")
    json.dump(pay, open(ppath, "w"))

    t0 = time.perf_counter()
    subprocess.run(
        [bin_path, "build", listing, "--payload", ppath, "--workdir", work,
         "--name", "blk", "--target", target, "--outputs", ",".join(q.STAGES)],
        check=True, stdout=subprocess.DEVNULL)
    build_s = time.perf_counter() - t0

    man = json.load(open(os.path.join(work, "blk.manifest.json")))
    exe = os.path.join(work, man["executable"])
    argv = []
    for name in man["inputs"]:
        for suffix, fmt, key in q.planes(pay["streams"][name]):
            p = os.path.join(work, f"in.{name}{suffix}.bin")
            q.emit(p, fmt, pay["streams"][name][key])
            argv.append(p)
    # output plane layout comes from the interpreter's stream kinds
    exp = json.loads(subprocess.check_output(
        [bin_path, "run", listing, "--payload", ppath], text=True))["streams"]
    for name in man["outputs"]:
        for suffix, fmt, _ in q.planes(exp[name]):
            p = os.path.join(work, f"out.{name}{suffix}.bin")
            argv.append(p)

    if os.environ.get("SHOW_PHASES"):
        env = dict(os.environ, JOTNAR_VK_TIME="1")
        r = subprocess.run([exe] + argv, stdout=subprocess.DEVNULL,
                           stderr=subprocess.PIPE, text=True, env=env)
        print(r.stderr.strip())

    times = []
    for _ in range(RUNS):
        t = time.perf_counter()
        subprocess.run([exe] + argv, check=True, stdout=subprocess.DEVNULL)
        times.append(time.perf_counter() - t)
    times.sort()
    med = times[len(times) // 2]
    print(f"{target:8s} build={build_s:6.2f}s  run={med*1000:8.2f} ms (median of {RUNS})")
    return med


def main():
    bin_path = os.path.join(os.path.dirname(HERE), "target", "release", "jotnar")
    only = os.environ.get("TARGETS")
    if only == "vulkan":
        bench("vulkan", bin_path)
        return
    a = bench("nonfpu", bin_path)
    b = bench("vulkan", bin_path)
    print(f"vulkan / nonfpu = {b/a:.2f}x")


if __name__ == "__main__":
    main()
