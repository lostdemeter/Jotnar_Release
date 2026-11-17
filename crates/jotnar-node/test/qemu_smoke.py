#!/usr/bin/env python3
"""Boot the JotnarOS node image under QEMU/OVMF and run the M0 smoke test.

Run this on a host with QEMU, OVMF, and KVM (the Mint build host). It:

  1. boots artifacts/jotnar-node-x86_64.img via UEFI,
  2. waits for the serial console and the node listener,
  3. runs `jotnar-ctl ping`, `program saxpy`, and `exec saxpy`,
  4. checks the expected output ([13, 26, 39, 52]),
  5. sends `poweroff -f` on the serial console and waits for QEMU to exit.

Usage:
    crates/jotnar-node/test/qemu_smoke.py [--image IMG] [--token-file FILE] [--ctl PATH]
                       [--runtime DIR] [--port N] [--timeout SEC]

Exit status is 0 only if every check passes.
"""

import argparse
import json
import os
import select
import shutil
import socket
import subprocess
import sys
import time

OVMF_CODE = os.environ.get("OVMF_CODE", "/usr/share/OVMF/OVMF_CODE_4M.fd")
OVMF_VARS = os.environ.get("OVMF_VARS", "/usr/share/OVMF/OVMF_VARS_4M.fd")


def log(msg):
    print(f"[smoke] {msg}", flush=True)


def wait_for_socket(path, timeout):
    end = time.time() + timeout
    while time.time() < end:
        if os.path.exists(path):
            try:
                s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                s.connect(path)
                s.setblocking(False)
                return s
            except OSError:
                pass
        time.sleep(0.2)
    raise RuntimeError(f"serial socket {path} did not appear")


def drain(sock, buf, deadline):
    """Read whatever is available until `deadline` or EOF."""
    while time.time() < deadline:
        r, _, _ = select.select([sock], [], [], 0.25)
        if not r:
            continue
        try:
            chunk = sock.recv(65536)
        except BlockingIOError:
            continue
        if not chunk:
            return
        buf.extend(chunk)


def wait_for(sock, buf, needle, timeout):
    end = time.time() + timeout
    while time.time() < end:
        if needle in bytes(buf):
            return
        drain(sock, buf, min(end, time.time() + 1.0))
    raise RuntimeError(f"timed out waiting for {needle!r} in serial output")


def send_keys(path, keys):
    """Inject keys through the QEMU HMP monitor (sendkey per key)."""
    sock = None
    deadline = time.time() + 10
    while time.time() < deadline:
        try:
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            sock.settimeout(2)
            sock.connect(path)
            break
        except OSError:
            if sock is not None:
                sock.close()
            sock = None
            time.sleep(0.2)
    if sock is None:
        raise RuntimeError(f"monitor socket {path} did not appear")

    time.sleep(0.3)
    try:
        sock.recv(65536)  # HMP banner/prompt
    except socket.timeout:
        pass
    for key in keys:
        sock.sendall(f"sendkey {key}\n".encode())
        time.sleep(0.15)
        try:
            sock.recv(4096)
        except socket.timeout:
            pass
    sock.close()


def run_ctl(ctl, addr, token, args, expect_ok=True):
    cmd = [ctl, "--addr", addr, "--token", token] + args
    shown = [("***" if part == token else part) for part in cmd]
    log(" ".join(shown))
    last = None
    for _ in range(12):
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if proc.returncode == 0:
            break
        last = proc
        if "Connection refused" not in (proc.stderr or ""):
            break
        time.sleep(0.5)
    else:
        proc = last
    if proc.returncode != 0:
        raise RuntimeError(f"ctl failed ({proc.returncode}): {proc.stderr.strip()}")
    try:
        parsed = json.loads(proc.stdout)
    except json.JSONDecodeError:
        raise RuntimeError(f"ctl produced non-JSON output: {proc.stdout!r}")
    if expect_ok and parsed.get("ok") is not True:
        raise RuntimeError(f"ctl reported failure: {proc.stdout}")
    return parsed


def run_payload(ctl, addr, token, args, expect_ok=True):
    cmd = [ctl, "--addr", addr, "--token", token] + args
    shown = [("***" if part == token else part) for part in cmd]
    log(" ".join(shown))
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    if proc.returncode != 0:
        raise RuntimeError(f"payload ctl failed ({proc.returncode}): {proc.stderr.strip()}")
    try:
        parsed = json.loads(proc.stdout)
    except json.JSONDecodeError:
        raise RuntimeError(f"payload ctl produced non-JSON output: {proc.stdout!r}")
    if expect_ok and parsed.get("ok") is not True:
        raise RuntimeError(f"payload ctl reported failure: {proc.stdout}")
    return parsed


def poll_console(payload_ctl, addr, token, after, needle, timeout=15):
    end = time.time() + timeout
    out = None
    while time.time() < end:
        out = run_payload(payload_ctl, addr, token,
                          ["poll", "--after", str(after), "--max", "64",
                           "--kinds", "console-out"])
        texts = [e.get("text", "") for e in out.get("events", [])]
        if any(needle in t for t in texts):
            return out
        time.sleep(0.4)
    raise RuntimeError(f"timed out waiting for {needle!r} in console events: {out}")


def collect_console(payload_ctl, addr, token, after, count, timeout=15):
    end = time.time() + timeout
    out = None
    while time.time() < end:
        out = run_payload(payload_ctl, addr, token,
                          ["poll", "--after", str(after), "--max", "64",
                           "--kinds", "console-out"])
        if len(out.get("events", [])) >= count:
            return out
        time.sleep(0.3)
    raise RuntimeError(f"timed out waiting for {count} console events: {out}")


def ssh_run(key, user, port, command, timeout=45):
    cmd = [
        "ssh", "-p", str(port), "-i", key,
        "-o", "StrictHostKeyChecking=no",
        "-o", "UserKnownHostsFile=/dev/null",
        "-o", "BatchMode=yes",
        "-o", "PreferredAuthentications=publickey",
        "-o", "ConnectTimeout=10",
        "-o", "LogLevel=ERROR",
        f"{user}@127.0.0.1",
        command,
    ]
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


def main():
    node_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    project = os.path.dirname(os.path.dirname(node_dir))
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default=os.path.join(project, "artifacts", "jotnar-node-x86_64.img"))
    ap.add_argument("--token-file", default=os.path.join(project, "artifacts", "node.token"))
    ap.add_argument("--ctl", default=os.path.join(project, "artifacts", "jotnar-ctl"))
    ap.add_argument("--runtime", default=project)
    ap.add_argument("--port", type=int, default=9999)
    ap.add_argument("--timeout", type=int, default=120)
    ap.add_argument("--keyboard", action="store_true",
                    help="power off by typing on the emulated local keyboard instead of serial")
    ap.add_argument("--no-overlay", action="store_true",
                    help="boot the image read-write instead of a disposable qcow2 overlay")
    ap.add_argument("--ssh-key", default=None,
                    help="private key for a post-boot SSH development check")
    ap.add_argument("--ssh-user", default="root")
    ap.add_argument("--ssh-port", type=int, default=2222)
    ap.add_argument("--m2", action="store_true",
                    help="run the M2 config/log/service acceptance checks over SSH")
    ap.add_argument("--m3", action="store_true",
                    help="run the M3 payload supervisor acceptance checks")
    ap.add_argument("--m4", action="store_true",
                    help="run the M4 console chroot bundle acceptance checks")
    ap.add_argument("--m5", action="store_true",
                    help="run the M5 local Hello JotnarOS! payload check")
    ap.add_argument("--m7", action="store_true",
                    help="run the M7a interpreted console-policy capability check")
    ap.add_argument("--m7b", action="store_true",
                    help="run the M7b emitted console-policy step check")
    ap.add_argument("--m7c", action="store_true",
                    help="run the M7c interpreted/emitted parity check")
    ap.add_argument("--m7d", action="store_true",
                    help="run the M7d emitted control-plane boot check")
    ap.add_argument("--m7e", action="store_true",
                    help="run the M7e second-module + denied-capability check")
    ap.add_argument("--m8", action="store_true",
                    help="run the M8 multi-module emitted-system parity check")
    ap.add_argument("--m8b", action="store_true",
                    help="run the M8b file-capability parity check")
    ap.add_argument("--m8c", action="store_true",
                    help="run the M8c proc-capability parity check")
    ap.add_argument("--m8d", action="store_true",
                    help="run the M8d timer-capability parity check")
    ap.add_argument("--m9", action="store_true",
                    help="run the M9 net-capability parity check")
    ap.add_argument("--m10", action="store_true",
                    help="run the M10 console productization check")
    ap.add_argument("--m11", action="store_true",
                    help="run the M11 randomized parity/fuzz check")
    ap.add_argument("--payload-ctl", default=os.path.join(project, "artifacts", "jotnar-payload"))
    ap.add_argument("--serial-log", default=os.path.join(project, "artifacts", "serial.log"))
    args = ap.parse_args()

    if args.m3 and not args.ssh_key:
        raise RuntimeError("--m3 requires --ssh-key (for the node-service restart check)")
    if args.m4 and not args.ssh_key:
        raise RuntimeError("--m4 requires --ssh-key (for the chroot/process checks)")

    for path in (args.image, args.token_file, args.ctl, OVMF_CODE, OVMF_VARS):
        if not os.path.exists(path):
            raise RuntimeError(f"missing required file: {path}")

    token = open(args.token_file, encoding="utf-8").read().strip()
    if not token:
        raise RuntimeError("empty token file")

    os.makedirs(os.path.dirname(args.serial_log), exist_ok=True)
    vars_copy = os.path.join(project, "artifacts", "OVMF_VARS_4M.fd")
    shutil.copyfile(OVMF_VARS, vars_copy)
    serial_sock = f"/tmp/jotnaros-serial-{os.getpid()}.sock"
    monitor_sock = f"/tmp/jotnaros-monitor-{os.getpid()}.sock"
    for path in (serial_sock, monitor_sock):
        if os.path.exists(path):
            os.unlink(path)

    accel = "kvm" if os.access("/dev/kvm", os.R_OK | os.W_OK) else "tcg"
    boot_image = args.image
    overlay = None
    if not args.no_overlay and shutil.which("qemu-img"):
        overlay = os.path.join(project, "artifacts", f"qemu-overlay-{os.getpid()}.qcow2")
        if os.path.exists(overlay):
            os.unlink(overlay)
        subprocess.run(
            ["qemu-img", "create", "-f", "qcow2", "-b", os.path.abspath(args.image),
             "-F", "raw", overlay],
            check=True, capture_output=True, text=True,
        )
        boot_image = overlay
        log("using disposable qcow2 overlay")
    cmd = [
        "qemu-system-x86_64",
        "-name", "jotnar-node",
        "-machine", "q35",
        "-accel", accel,
        "-cpu", "host" if accel == "kvm" else "max",
        "-m", "1024",
        "-smp", "2",
        "-drive", f"if=pflash,format=raw,readonly=on,file={OVMF_CODE}",
        "-drive", f"if=pflash,format=raw,file={vars_copy}",
        "-drive", f"file={boot_image},format={'qcow2' if overlay else 'raw'},if=virtio",
        "-netdev", f"user,id=net0,hostfwd=tcp:127.0.0.1:{args.port}-:9999,hostfwd=tcp:127.0.0.1:{args.port - 1}-:9998,hostfwd=tcp:127.0.0.1:{args.ssh_port}-:22",
        "-device", "virtio-net-pci,netdev=net0",
        "-display", "none",
        "-monitor", f"unix:{monitor_sock},server=on,wait=off",
        "-serial", f"unix:{serial_sock},server=on,wait=off",
    ]
    log(f"starting QEMU ({accel})")
    qemu = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)

    buf = bytearray()
    sock = None
    success = False
    try:
        sock = wait_for_socket(serial_sock, 15)
        wait_for(sock, buf, b"JotnarOS node ready", args.timeout)
        log("node booted")

        addr = f"127.0.0.1:{args.port}"
        pong = run_ctl(args.ctl, addr, token, ["ping"])
        log(f"ping ok (runtime {pong['pong']['runtime_version']})")

        saxpy = os.path.join(args.runtime, "showcase", "language", "saxpy.asm")
        payload = os.path.join(project, "artifacts", "saxpy.json")
        with open(payload, "w", encoding="utf-8") as f:
            json.dump({"streams": {
                "A": {"kind": "I64", "shape": [4], "data": ["3", "3", "3", "3"]},
                "X": {"kind": "I64", "shape": [4], "data": ["1", "2", "3", "4"]},
                "Y": {"kind": "I64", "shape": [4], "data": ["10", "20", "30", "40"]},
            }}, f)

        run_ctl(args.ctl, addr, token, ["program", "saxpy", saxpy])
        out = run_ctl(args.ctl, addr, token,
                      ["exec", "saxpy", "--payload", payload, "--outputs", "OUT"],
                      expect_ok=False)
        got = out["streams"]["OUT"]["data"]
        want = ["13", "26", "39", "52"]
        if got != want:
            raise RuntimeError(f"saxpy output mismatch: got {got}, want {want}")
        log(f"saxpy ok: {got}")

        if args.ssh_key:
            log(f"ssh check as {args.ssh_user}")
            proc = ssh_run(args.ssh_key, args.ssh_user, args.ssh_port,
                           "uname -a; df -h /var/lib/jotnar; /usr/local/sbin/jotnar-service status")
            if proc.returncode != 0:
                raise RuntimeError(f"ssh check failed ({proc.returncode}): {proc.stderr.strip()}")
            if "jotnar-node running" not in proc.stdout:
                raise RuntimeError(f"ssh check did not see a running node: {proc.stdout!r}")
            log("ssh ok: " + proc.stdout.splitlines()[0])

            if args.m2:
                log("m2: log files on jotnar-data")
                proc = ssh_run(args.ssh_key, args.ssh_user, args.ssh_port,
                               "test -s /var/lib/jotnar/log/init.log && "
                               "test -s /var/lib/jotnar/log/node.log && echo logs-ok")
                if proc.returncode != 0 or "logs-ok" not in proc.stdout:
                    raise RuntimeError(f"m2 log check failed: {proc.stdout} {proc.stderr}")

                log("m2: persistent config override (port 9998)")
                proc = ssh_run(args.ssh_key, args.ssh_user, args.ssh_port,
                               "mkdir -p /var/lib/jotnar/etc && "
                               "printf 'JOTNAR_PORT=9998\\n' > /var/lib/jotnar/etc/node.conf && "
                               "/usr/local/sbin/jotnar-service restart && "
                               "/usr/local/sbin/jotnar-service status")
                if proc.returncode != 0 or ":9998" not in proc.stdout:
                    raise RuntimeError(f"m2 override failed: {proc.stdout} {proc.stderr}")
                run_ctl(args.ctl, f"127.0.0.1:{args.port - 1}", token, ["ping"])

                log("m2: restore default port")
                proc = ssh_run(args.ssh_key, args.ssh_user, args.ssh_port,
                               "rm -f /var/lib/jotnar/etc/node.conf && "
                               "/usr/local/sbin/jotnar-service restart && "
                               "/usr/local/sbin/jotnar-service status")
                if proc.returncode != 0 or ":9999" not in proc.stdout:
                    raise RuntimeError(f"m2 restore failed: {proc.stdout} {proc.stderr}")
                run_ctl(args.ctl, addr, token, ["ping"])

                log("m2: service stop/start")
                proc = ssh_run(args.ssh_key, args.ssh_user, args.ssh_port,
                               "/usr/local/sbin/jotnar-service stop; "
                               "/usr/local/sbin/jotnar-service status || echo stopped-ok")
                if "stopped-ok" not in proc.stdout:
                    raise RuntimeError(f"m2 stop failed: {proc.stdout} {proc.stderr}")
                proc = ssh_run(args.ssh_key, args.ssh_user, args.ssh_port,
                               "/usr/local/sbin/jotnar-service start && "
                               "/usr/local/sbin/jotnar-service status")
                if proc.returncode != 0 or "running" not in proc.stdout:
                    raise RuntimeError(f"m2 start failed: {proc.stdout} {proc.stderr}")
                run_ctl(args.ctl, addr, token, ["ping"])
                log("m2: PASS")

            if args.m3:
                if not os.path.exists(args.payload_ctl):
                    raise RuntimeError(f"missing payload ctl: {args.payload_ctl}")
                pong = run_ctl(args.ctl, addr, token, ["ping"])
                features = pong.get("pong", {}).get("features", [])
                if "payload.v1" not in features:
                    raise RuntimeError(f"node did not advertise payload.v1: {pong}")
                log("m3: ping advertises payload.v1")

                heartbeat = os.path.join(project, "node", "test", "heartbeat.asm")
                seed = os.path.join(project, "node", "test", "seed-tick.json")
                run_payload(args.payload_ctl, addr, token,
                            ["put", "heartbeat", heartbeat, "--tick-ms", "500",
                             "--autostart", "--console", "both", "--seed", seed])
                run_payload(args.payload_ctl, addr, token, ["start", "heartbeat"])
                st = run_payload(args.payload_ctl, addr, token, ["status", "heartbeat"])
                if st.get("payloads", [{}])[0].get("state") != "running":
                    raise RuntimeError(f"heartbeat not running: {st}")
                out = poll_console(args.payload_ctl, addr, token, 0, "tick")
                seq = out.get("next_seq", 0)
                log("m3: heartbeat console output ok")

                run_payload(args.payload_ctl, addr, token,
                            ["write", "heartbeat", "--data", "hello\n"])
                out = poll_console(args.payload_ctl, addr, token, seq, "hello")
                seq = out.get("next_seq", seq)
                log("m3: event-write/stdin output ok")

                # One-shot exec must still work while the payload runs.
                run_ctl(args.ctl, addr, token,
                        ["program", "saxpy2", os.path.join(args.runtime, "showcase", "language", "saxpy.asm")])
                run_ctl(args.ctl, addr, token,
                        ["exec", "saxpy2", "--payload",
                         os.path.join(project, "artifacts", "saxpy.json"), "--outputs", "OUT"],
                        expect_ok=False)
                log("m3: exec while payload runs ok")

                # Restarting the TCP node must not kill the supervisor payload.
                proc = ssh_run(args.ssh_key, args.ssh_user, args.ssh_port,
                               "/usr/local/sbin/jotnar-service restart")
                if proc.returncode != 0:
                    raise RuntimeError(f"node restart failed: {proc.stdout} {proc.stderr}")
                time.sleep(2)
                run_ctl(args.ctl, addr, token, ["ping"])
                st = run_payload(args.payload_ctl, addr, token, ["status", "heartbeat"])
                if st.get("payloads", [{}])[0].get("state") != "running":
                    raise RuntimeError(f"payload did not survive node restart: {st}")
                poll_console(args.payload_ctl, addr, token, seq, "tick")
                log("m3: payload survives node-service restart")

                run_payload(args.payload_ctl, addr, token, ["stop", "heartbeat"])
                st = run_payload(args.payload_ctl, addr, token, ["status", "heartbeat"])
                if st.get("payloads", [{}])[0].get("state") != "stopped":
                    raise RuntimeError(f"heartbeat did not stop: {st}")
                run_payload(args.payload_ctl, addr, token, ["drop", "heartbeat"])
                log("m3: PASS")

                log("s4/s5: stateful counter + scoped file")
                counter = os.path.join(project, "node", "test", "counter.asm")
                cseed = os.path.join(project, "node", "test", "seed-counter.json")
                head = run_payload(args.payload_ctl, addr, token,
                                   ["poll", "--after", "0", "--max", "1"]).get("head_seq", 0)
                start_after = max(0, head - 1)
                run_payload(args.payload_ctl, addr, token,
                            ["put", "counter", counter, "--tick-ms", "300",
                             "--seed", cseed, "--files", "notes.txt", "--console", "both"])
                run_payload(args.payload_ctl, addr, token, ["start", "counter"])
                out = collect_console(args.payload_ctl, addr, token, start_after, 3)
                first = [e.get("hex") for e in out.get("events", [])]
                if first[:3] != ["31", "32", "33"]:
                    raise RuntimeError(f"unexpected counter start: {first}")
                seq = out.get("next_seq", 0)
                run_payload(args.payload_ctl, addr, token, ["stop", "counter"])
                run_payload(args.payload_ctl, addr, token, ["start", "counter"])
                out = collect_console(args.payload_ctl, addr, token, seq, 2)
                resumed = [int(e.get("hex"), 16) for e in out.get("events", [])]
                if not resumed or min(resumed) <= 51:  # '3'
                    raise RuntimeError(f"state did not continue after restart: {resumed}")
                log(f"s5: state continued after stop/start: {resumed}")

                run_payload(args.payload_ctl, addr, token,
                            ["write", "counter", "--target", "file",
                             "--path", "notes.txt", "--data", "hello state\n"])
                out = run_payload(args.payload_ctl, addr, token,
                                  ["read", "counter", "--path", "notes.txt"])
                if out.get("text") != "hello state\n":
                    raise RuntimeError(f"file read mismatch: {out}")
                proc = ssh_run(args.ssh_key, args.ssh_user, args.ssh_port,
                               "test -s /var/lib/jotnar/payloads/counter/state.ndjson && "
                               "test -s /var/lib/jotnar/payloads/counter/files/notes.txt && echo state-ok")
                if proc.returncode != 0 or "state-ok" not in proc.stdout:
                    raise RuntimeError(f"state/file files missing: {proc.stdout} {proc.stderr}")
                run_payload(args.payload_ctl, addr, token, ["stop", "counter"])
                run_payload(args.payload_ctl, addr, token, ["drop", "counter"])
                log("s4/s5: PASS")

            if args.m4:
                bundle = os.path.join(project, "artifacts",
                                      "console-alpine-minirootfs-3.24.4.tar.gz")
                if not os.path.exists(bundle):
                    raise RuntimeError(f"missing console bundle: {bundle}")
                log("m4: bundle-put console")
                out = run_payload(args.payload_ctl, addr, token,
                                  ["bundle-put", "console", bundle])
                if not out.get("sha256"):
                    raise RuntimeError(f"bundle-put failed: {out}")

                log("m4: console-start on VT2")
                out = run_payload(args.payload_ctl, addr, token,
                                  ["console-start", "console", "--vt", "2"])
                pid = out.get("pid")
                if not pid:
                    raise RuntimeError(f"console-start failed: {out}")
                st = run_payload(args.payload_ctl, addr, token, ["console-status", "console"])
                if st.get("consoles", [{}])[0].get("state") != "running":
                    raise RuntimeError(f"console not running: {st}")

                # Type into the chroot shell on the active VT. touch is enough
                # to prove keyboard + shell + chroot filesystem.
                time.sleep(1.5)
                send_keys(monitor_sock,
                          list("touch") + ["spc", "slash", "p", "a", "y", "l",
                                           "o", "a", "d", "minus", "o", "k", "ret"])
                time.sleep(1.0)
                rootfs = "/var/lib/jotnar/payloads/console/rootfs"
                proc = ssh_run(args.ssh_key, args.ssh_user, args.ssh_port,
                               f"readlink /proc/{pid}/root; "
                               f"test -f {rootfs}/payload-ok && echo payload-ok")
                if "payload-ok" not in proc.stdout or rootfs not in proc.stdout:
                    raise RuntimeError(f"chroot console check failed: {proc.stdout} {proc.stderr}")
                log("m4: chroot shell + keyboard marker ok")

                run_ctl(args.ctl, addr, token, ["ping"])
                run_ctl(args.ctl, addr, token,
                        ["exec", "saxpy", "--payload",
                         os.path.join(project, "artifacts", "saxpy.json"), "--outputs", "OUT"],
                        expect_ok=False)
                log("m4: uplink/exec while console runs ok")

                run_payload(args.payload_ctl, addr, token, ["console-stop", "console"])
                st = run_payload(args.payload_ctl, addr, token, ["console-status", "console"])
                if st.get("consoles", [{}])[0].get("state") != "stopped":
                    raise RuntimeError(f"console did not stop: {st}")
                proc = ssh_run(args.ssh_key, args.ssh_user, args.ssh_port,
                               f"test -d /proc/{pid} && echo still-running || echo stopped")
                if "stopped" not in proc.stdout:
                    raise RuntimeError(f"console process survived stop: {proc.stdout}")
                log("m4: PASS")

        if args.m5:
            hello = os.path.join(project, "node", "test", "hello.asm")
            seed = os.path.join(project, "node", "test", "seed-hello.json")
            head = run_payload(args.payload_ctl, addr, token,
                               ["poll", "--after", "0", "--max", "1"]).get("head_seq", 0)
            run_payload(args.payload_ctl, addr, token,
                        ["put", "hello", hello, "--seed", seed, "--console", "both"])
            run_payload(args.payload_ctl, addr, token, ["start", "hello"])
            run_payload(args.payload_ctl, addr, token, ["write", "hello", "--data", ""])
            out = poll_console(args.payload_ctl, addr, token, max(0, head - 1),
                               "Hello JotnarOS!")
            texts = [e.get("text", "") for e in out.get("events", [])]
            if "Hello JotnarOS!\n" not in texts:
                raise RuntimeError(f"M5 text mismatch: {texts}")
            st = run_payload(args.payload_ctl, addr, token, ["status", "hello"])
            if st.get("payloads", [{}])[0].get("state") != "running":
                raise RuntimeError(f"hello payload not running: {st}")
            log("m5: local Hello JotnarOS! PASS")
            run_payload(args.payload_ctl, addr, token, ["stop", "hello"])
            run_payload(args.payload_ctl, addr, token, ["drop", "hello"])

        if args.m7:
            if not args.ssh_key:
                raise RuntimeError("--m7 requires --ssh-key")
            bundle = os.path.join(project, "artifacts",
                                  "console-alpine-minirootfs-3.24.4.tar.gz")
            run_payload(args.payload_ctl, addr, token, ["bundle-put", "console", bundle])
            st = run_payload(args.payload_ctl, addr, token, ["module-status", "console-policy"])
            mod = st.get("modules", [{}])[0]
            if mod.get("name") != "console-policy":
                raise RuntimeError(f"console-policy module missing: {st}")
            before = mod.get("actions", 0)

            log("m7: module-event console.start")
            out = run_payload(args.payload_ctl, addr, token,
                              ["module-event", "console-policy", "console.start"])
            if out.get("result", {}).get("ok") is not True:
                raise RuntimeError(f"module console.start failed: {out}")
            pid = out.get("result", {}).get("result", {}).get("pid")
            if not pid:
                raise RuntimeError(f"module did not return a console pid: {out}")

            st = run_payload(args.payload_ctl, addr, token, ["console-status", "console"])
            if st.get("consoles", [{}])[0].get("state") != "running":
                raise RuntimeError(f"console not running after module start: {st}")
            rootfs = "/var/lib/jotnar/payloads/console/rootfs"
            proc = ssh_run(args.ssh_key, args.ssh_user, args.ssh_port,
                           f"readlink /proc/{pid}/root")
            if rootfs not in proc.stdout:
                raise RuntimeError(f"console root mismatch: {proc.stdout} {proc.stderr}")

            out = run_payload(args.payload_ctl, addr, token,
                              ["module-event", "console-policy", "console.status"])
            if out.get("result", {}).get("ok") is not True:
                raise RuntimeError(f"module console.status failed: {out}")

            out = run_payload(args.payload_ctl, addr, token,
                              ["module-event", "console-policy", "console.stop"])
            if out.get("result", {}).get("ok") is not True:
                raise RuntimeError(f"module console.stop failed: {out}")
            st = run_payload(args.payload_ctl, addr, token, ["console-status", "console"])
            if st.get("consoles", [{}])[0].get("state") != "stopped":
                raise RuntimeError(f"console did not stop via module: {st}")

            st = run_payload(args.payload_ctl, addr, token, ["module-status", "console-policy"])
            mod = st.get("modules", [{}])[0]
            if mod.get("actions", 0) < before + 3:
                raise RuntimeError(f"module action count did not advance: {st}")
            proc = ssh_run(args.ssh_key, args.ssh_user, args.ssh_port,
                           "test -s /var/lib/jotnar/modules/console-policy/state.ndjson")
            if proc.returncode != 0:
                raise RuntimeError(f"module state file missing: {proc.stdout} {proc.stderr}")
            log("m7: interpreted console-policy capability PASS")

        if args.m7b:
            if not args.ssh_key:
                raise RuntimeError("--m7b requires --ssh-key")
            for event, action, op in [(10, 1, "start"), (11, 2, "stop"), (12, 3, "status")]:
                proc = ssh_run(args.ssh_key, args.ssh_user, args.ssh_port,
                               f"/usr/local/bin/jotnar-osd --event {event}")
                if proc.returncode != 0:
                    raise RuntimeError(f"emitted step failed: {proc.stdout} {proc.stderr}")
                out = json.loads(proc.stdout)
                if out.get("mode") != "emitted" or out.get("action") != action:
                    raise RuntimeError(f"emitted step mismatch for {event}: {out}")
                cmd = out.get("command", {})
                if cmd.get("capability") != "console" or cmd.get("op") != op:
                    raise RuntimeError(f"emitted command mismatch for {event}: {out}")
                if out.get("state", {}).get("ACTIONS", {}).get("data") != ["1"]:
                    raise RuntimeError(f"emitted state mismatch for {event}: {out}")
            log("m7b: emitted console-policy step PASS")

        if args.m7c:
            if not args.ssh_key:
                raise RuntimeError("--m7c requires --ssh-key")
            events = ["console.start", "console.stop", "console.status"]
            interpreted = []
            for ev in events:
                out = run_payload(args.payload_ctl, addr, token,
                                  ["module-step", "console-policy", ev])
                interpreted.append({
                    "event": ev,
                    "action": out.get("action"),
                    "state": out.get("state"),
                    "command": {
                        "capability": out.get("command", {}).get("capability"),
                        "op": out.get("command", {}).get("op"),
                        "args": out.get("command", {}).get("args"),
                    },
                })
            proc = ssh_run(args.ssh_key, args.ssh_user, args.ssh_port,
                           "/usr/local/bin/jotnar-osd --events 10,11,12")
            if proc.returncode != 0:
                raise RuntimeError(f"emitted run failed: {proc.stdout} {proc.stderr}")
            emitted = json.loads(proc.stdout).get("steps", [])
            if len(emitted) != len(interpreted):
                raise RuntimeError(f"parity length mismatch: {interpreted} vs {emitted}")
            for i, (a, b) in enumerate(zip(interpreted, emitted)):
                if a["action"] != b.get("action"):
                    raise RuntimeError(f"parity action mismatch step {i}: {a} vs {b}")
                if a["state"] != b.get("state"):
                    raise RuntimeError(f"parity state mismatch step {i}: {a} vs {b}")
                bc = b.get("command", {})
                if (a["command"]["capability"], a["command"]["op"], a["command"]["args"]) != (
                        bc.get("capability"), bc.get("op"), bc.get("args")):
                    raise RuntimeError(f"parity command mismatch step {i}: {a} vs {b}")
            trace = os.path.join(project, "artifacts", "m7c-parity.json")
            with open(trace, "w", encoding="utf-8") as f:
                json.dump({"interpreted": interpreted, "emitted": emitted}, f, indent=2)
            log(f"m7c: interpreted/emitted parity PASS ({trace})")

        if args.m7d:
            if not args.ssh_key:
                raise RuntimeError("--m7d requires --ssh-key")
            bundle = os.path.join(project, "artifacts",
                                  "console-alpine-minirootfs-3.24.4.tar.gz")
            proc = ssh_run(args.ssh_key, args.ssh_user, args.ssh_port,
                           "test -S /run/jotnar/osd.sock && "
                           "test -f /run/jotnar-osd.pid && echo osd-ok")
            if "osd-ok" not in proc.stdout:
                raise RuntimeError(f"emitted control plane not running: {proc.stdout} {proc.stderr}")

            run_payload(args.payload_ctl, addr, token, ["bundle-put", "console", bundle])
            st = run_payload(args.payload_ctl, addr, token, ["module-status", "console-policy"])
            mod = st.get("modules", [{}])[0]
            if mod.get("mode") != "emitted":
                raise RuntimeError(f"module status did not come from emitted daemon: {st}")

            out = run_payload(args.payload_ctl, addr, token,
                              ["module-event", "console-policy", "console.start"])
            if out.get("mode") != "emitted" or out.get("result", {}).get("ok") is not True:
                raise RuntimeError(f"emitted module start failed: {out}")
            pid = out.get("result", {}).get("result", {}).get("pid")
            if not pid:
                raise RuntimeError(f"emitted module returned no pid: {out}")
            proc = ssh_run(args.ssh_key, args.ssh_user, args.ssh_port,
                           f"readlink /proc/{pid}/root")
            if "/var/lib/jotnar/payloads/console/rootfs" not in proc.stdout:
                raise RuntimeError(f"emitted console root mismatch: {proc.stdout}")

            out = run_payload(args.payload_ctl, addr, token,
                              ["module-event", "console-policy", "console.status"])
            if out.get("result", {}).get("ok") is not True:
                raise RuntimeError(f"emitted module status failed: {out}")
            out = run_payload(args.payload_ctl, addr, token,
                              ["module-event", "console-policy", "console.stop"])
            if out.get("result", {}).get("ok") is not True:
                raise RuntimeError(f"emitted module stop failed: {out}")
            st = run_payload(args.payload_ctl, addr, token, ["console-status", "console"])
            if st.get("consoles", [{}])[0].get("state") != "stopped":
                raise RuntimeError(f"console did not stop via emitted daemon: {st}")
            proc = ssh_run(args.ssh_key, args.ssh_user, args.ssh_port,
                           "test -s /var/lib/jotnar/osd/console-policy/state.json && echo state-ok")
            if "state-ok" not in proc.stdout:
                raise RuntimeError(f"emitted control-plane state missing: {proc.stdout}")
            log("m7d: emitted control-plane console lifecycle PASS")

        if args.m7e:
            if not args.ssh_key:
                raise RuntimeError("--m7e requires --ssh-key")
            st = run_payload(args.payload_ctl, addr, token, ["module-status"])
            names = sorted(m.get("name") for m in st.get("modules", []))
            for need in ("console-policy", "system-log-policy", "denied-policy"):
                if need not in names:
                    raise RuntimeError(f"module {need} missing from composition: {st}")

            out = run_payload(args.payload_ctl, addr, token,
                              ["module-event", "system-log-policy", "system.log"])
            if out.get("result", {}).get("ok") is not True:
                raise RuntimeError(f"system-log-policy failed: {out}")
            st = run_payload(args.payload_ctl, addr, token,
                             ["module-status", "system-log-policy"])
            if st.get("modules", [{}])[0].get("actions", 0) < 1:
                raise RuntimeError(f"system-log-policy actions did not advance: {st}")
            log("m7e: second module composition PASS")

            interpreted = run_payload(args.payload_ctl, addr, token,
                                      ["module-step", "denied-policy", "console.start"])
            res = interpreted.get("result", {})
            if res.get("ok") is not False or res.get("error", {}).get("code") != "denied":
                raise RuntimeError(f"interpreted denial missing: {interpreted}")
            dispatched = run_payload(args.payload_ctl, addr, token,
                                     ["module-event", "denied-policy", "console.start"])
            res2 = dispatched.get("result", {})
            if res2.get("ok") is not False or res2.get("error", {}).get("code") != "denied":
                raise RuntimeError(f"interpreted denial dispatch missing: {dispatched}")

            proc = ssh_run(args.ssh_key, args.ssh_user, args.ssh_port,
                           "/usr/local/bin/jotnar-osd --manifest "
                           "/usr/local/share/jotnar/modules/denied-policy/manifest.json "
                           "--event 10")
            if proc.returncode != 0:
                raise RuntimeError(f"emitted denial run failed: {proc.stdout} {proc.stderr}")
            emitted = json.loads(proc.stdout)
            eres = emitted.get("result", {})
            if eres.get("ok") is not False or eres.get("error", {}).get("code") != "denied":
                raise RuntimeError(f"emitted denial missing: {emitted}")
            log("m7e: denied-capability parity PASS (interpreted + emitted)")

        if args.m8:
            if not args.ssh_key:
                raise RuntimeError("--m8 requires --ssh-key")
            st = run_payload(args.payload_ctl, addr, token, ["module-status"])
            mods = st.get("modules", [])
            if sorted(m.get("name") for m in mods) != ["console-policy", "denied-policy", "file-policy", "job-policy", "net-policy", "proc-policy", "system-log-policy", "timer-policy"]:
                raise RuntimeError(f"M8 module list mismatch: {st}")
            if any(m.get("mode") != "emitted" for m in mods):
                raise RuntimeError(f"M8 modules are not all emitted: {st}")
            log("m8: all system modules emitted")

            plan = [
                ("console-policy", "console.start", 10),
                ("console-policy", "console.stop", 11),
                ("console-policy", "console.status", 12),
                ("system-log-policy", "system.log", 30),
                ("denied-policy", "console.start", 10),
            ]
            trace = []
            for mod, ev, _code in plan:
                interp = run_payload(args.payload_ctl, addr, token,
                                     ["module-step", mod, ev])
                emitted = run_payload(args.payload_ctl, addr, token,
                                      ["module-step-emitted", mod, ev])
                for key in ("action", "state"):
                    if interp.get(key) != emitted.get(key):
                        raise RuntimeError(
                            f"M8 parity {mod}/{ev} {key}: {interp.get(key)} vs {emitted.get(key)}")
                ic = interp.get("command", {})
                ec = emitted.get("command", {})
                if (ic.get("capability"), ic.get("op"), ic.get("args")) != (
                        ec.get("capability"), ec.get("op"), ec.get("args")):
                    raise RuntimeError(f"M8 parity {mod}/{ev} command: {ic} vs {ec}")
                ir = interp.get("result", {})
                er = emitted.get("result", {})
                if ir.get("ok") != er.get("ok"):
                    raise RuntimeError(f"M8 parity {mod}/{ev} result: {ir} vs {er}")
                trace.append({"module": mod, "event": ev, "interpreted": interp, "emitted": emitted})
            log("m8: whole-system interpreted/emitted parity PASS")

            bundle = os.path.join(project, "artifacts",
                                  "console-alpine-minirootfs-3.24.4.tar.gz")
            run_payload(args.payload_ctl, addr, token, ["bundle-put", "console", bundle])
            out = run_payload(args.payload_ctl, addr, token,
                              ["module-event", "console-policy", "console.start"])
            if out.get("result", {}).get("ok") is not True:
                raise RuntimeError(f"M8 emitted console start failed: {out}")
            pid = out.get("result", {}).get("result", {}).get("pid")
            if not pid:
                raise RuntimeError(f"M8 emitted console returned no pid: {out}")
            proc = ssh_run(args.ssh_key, args.ssh_user, args.ssh_port,
                           f"readlink /proc/{pid}/root")
            if "/var/lib/jotnar/payloads/console/rootfs" not in proc.stdout:
                raise RuntimeError(f"M8 console root mismatch: {proc.stdout}")
            run_payload(args.payload_ctl, addr, token,
                        ["module-event", "console-policy", "console.stop"])
            out = run_payload(args.payload_ctl, addr, token,
                              ["module-event", "system-log-policy", "system.log"])
            if out.get("result", {}).get("ok") is not True:
                raise RuntimeError(f"M8 emitted system.log failed: {out}")
            out = run_payload(args.payload_ctl, addr, token,
                              ["module-event", "denied-policy", "console.start"])
            if out.get("result", {}).get("error", {}).get("code") != "denied":
                raise RuntimeError(f"M8 emitted denial failed: {out}")
            trace_path = os.path.join(project, "artifacts", "m8-parity.json")
            with open(trace_path, "w", encoding="utf-8") as f:
                json.dump(trace, f, indent=2)
            log(f"m8: PASS ({trace_path})")

        if args.m8b:
            if not args.ssh_key:
                raise RuntimeError("--m8b requires --ssh-key")
            st = run_payload(args.payload_ctl, addr, token, ["module-status"])
            names = sorted(m.get("name") for m in st.get("modules", []))
            if "file-policy" not in names:
                raise RuntimeError(f"file-policy missing: {st}")
            trace = []
            for ev in ("file.write", "file.read", "file.list"):
                interp = run_payload(args.payload_ctl, addr, token,
                                     ["module-step", "file-policy", ev])
                emitted = run_payload(args.payload_ctl, addr, token,
                                      ["module-step-emitted", "file-policy", ev])
                for key in ("action", "state"):
                    if interp.get(key) != emitted.get(key):
                        raise RuntimeError(f"M8b parity {ev} {key}: {interp} vs {emitted}")
                ic = interp.get("command", {})
                ec = emitted.get("command", {})
                if (ic.get("capability"), ic.get("op"), ic.get("args")) != (
                        ec.get("capability"), ec.get("op"), ec.get("args")):
                    raise RuntimeError(f"M8b parity {ev} command: {ic} vs {ec}")
                trace.append({"event": ev, "interpreted": interp, "emitted": emitted})
            log("m8b: file capability interpreted/emitted parity PASS")

            out = run_payload(args.payload_ctl, addr, token,
                              ["module-event", "file-policy", "file.write"])
            if out.get("result", {}).get("ok") is not True:
                raise RuntimeError(f"M8b emitted file.write failed: {out}")
            out = run_payload(args.payload_ctl, addr, token,
                              ["module-event", "file-policy", "file.read"])
            text = out.get("result", {}).get("result", {}).get("text")
            if text != "file-policy\n":
                raise RuntimeError(f"M8b emitted file.read mismatch: {out}")
            out = run_payload(args.payload_ctl, addr, token,
                              ["module-event", "file-policy", "file.list"])
            entries = out.get("result", {}).get("result", {}).get("entries", [])
            if "notes.txt" not in entries:
                raise RuntimeError(f"M8b emitted file.list missing notes.txt: {out}")

            out = run_payload(args.payload_ctl, addr, token,
                              ["module-event-interpreted", "file-policy", "file.write"])
            if out.get("result", {}).get("ok") is not True:
                raise RuntimeError(f"M8b interpreted file.write failed: {out}")
            out = run_payload(args.payload_ctl, addr, token,
                              ["module-event-interpreted", "file-policy", "file.read"])
            text = out.get("result", {}).get("result", {}).get("text")
            if text != "file-policy\n":
                raise RuntimeError(f"M8b interpreted file.read mismatch: {out}")
            proc = ssh_run(args.ssh_key, args.ssh_user, args.ssh_port,
                           "test -f /var/lib/jotnar/modules/file-policy/files/notes.txt && "
                           "test -f /var/lib/jotnar/osd/file-policy/files/notes.txt && echo files-ok")
            if "files-ok" not in proc.stdout:
                raise RuntimeError(f"M8b file roots missing: {proc.stdout} {proc.stderr}")
            trace_path = os.path.join(project, "artifacts", "m8b-file-parity.json")
            with open(trace_path, "w", encoding="utf-8") as f:
                json.dump(trace, f, indent=2)
            log(f"m8b: PASS ({trace_path})")

        if args.m8c:
            if not args.ssh_key:
                raise RuntimeError("--m8c requires --ssh-key")
            st = run_payload(args.payload_ctl, addr, token, ["module-status"])
            names = sorted(m.get("name") for m in st.get("modules", []))
            if "proc-policy" not in names:
                raise RuntimeError(f"proc-policy missing: {st}")
            trace = []
            for ev in ("proc.status", "proc.signal"):
                interp = run_payload(args.payload_ctl, addr, token,
                                     ["module-step", "proc-policy", ev])
                emitted = run_payload(args.payload_ctl, addr, token,
                                      ["module-step-emitted", "proc-policy", ev])
                for key in ("action", "state"):
                    if interp.get(key) != emitted.get(key):
                        raise RuntimeError(f"M8c parity {ev} {key}: {interp} vs {emitted}")
                ic = interp.get("command", {})
                ec = emitted.get("command", {})
                if (ic.get("capability"), ic.get("op"), ic.get("args")) != (
                        ec.get("capability"), ec.get("op"), ec.get("args")):
                    raise RuntimeError(f"M8c parity {ev} command: {ic} vs {ec}")
                trace.append({"event": ev, "interpreted": interp, "emitted": emitted})
            log("m8c: proc capability interpreted/emitted parity PASS")

            emitted = run_payload(args.payload_ctl, addr, token,
                                  ["module-event", "proc-policy", "proc.status"])
            interpreted = run_payload(args.payload_ctl, addr, token,
                                      ["module-event-interpreted", "proc-policy", "proc.status"])
            er = emitted.get("result", {}).get("result", {})
            ir = interpreted.get("result", {}).get("result", {})
            if er != ir or er.get("pid") != 1 or not er.get("comm"):
                raise RuntimeError(f"M8c proc.status mismatch: {er} vs {ir}")

            emitted = run_payload(args.payload_ctl, addr, token,
                                  ["module-event", "proc-policy", "proc.signal"])
            interpreted = run_payload(args.payload_ctl, addr, token,
                                      ["module-event-interpreted", "proc-policy", "proc.signal"])
            er = emitted.get("result", {}).get("result", {})
            ir = interpreted.get("result", {}).get("result", {})
            if er != ir or er.get("alive") is not True:
                raise RuntimeError(f"M8c proc.signal mismatch: {er} vs {ir}")
            trace_path = os.path.join(project, "artifacts", "m8c-proc-parity.json")
            with open(trace_path, "w", encoding="utf-8") as f:
                json.dump(trace, f, indent=2)
            log(f"m8c: PASS ({trace_path})")

        if args.m8d:
            if not args.ssh_key:
                raise RuntimeError("--m8d requires --ssh-key")
            st = run_payload(args.payload_ctl, addr, token, ["module-status"])
            names = sorted(m.get("name") for m in st.get("modules", []))
            if "timer-policy" not in names:
                raise RuntimeError(f"timer-policy missing: {st}")
            trace = []
            for ev in ("timer.arm", "timer.cancel", "timer.fired"):
                interp = run_payload(args.payload_ctl, addr, token,
                                     ["module-step", "timer-policy", ev])
                emitted = run_payload(args.payload_ctl, addr, token,
                                      ["module-step-emitted", "timer-policy", ev])
                for key in ("action", "state"):
                    if interp.get(key) != emitted.get(key):
                        raise RuntimeError(f"M8d parity {ev} {key}: {interp} vs {emitted}")
                ic = interp.get("command", {})
                ec = emitted.get("command", {})
                if (ic.get("capability"), ic.get("op"), ic.get("args")) != (
                        ec.get("capability"), ec.get("op"), ec.get("args")):
                    raise RuntimeError(f"M8d parity {ev} command: {ic} vs {ec}")
                trace.append({"event": ev, "interpreted": interp, "emitted": emitted})
            log("m8d: timer capability interpreted/emitted parity PASS")

            out = run_payload(args.payload_ctl, addr, token,
                              ["module-event", "timer-policy", "timer.arm"])
            if out.get("result", {}).get("ok") is not True:
                raise RuntimeError(f"M8d emitted timer.arm failed: {out}")
            out = run_payload(args.payload_ctl, addr, token,
                              ["module-event-interpreted", "timer-policy", "timer.arm"])
            if out.get("result", {}).get("ok") is not True:
                raise RuntimeError(f"M8d interpreted timer.arm failed: {out}")
            time.sleep(1.5)
            proc = ssh_run(args.ssh_key, args.ssh_user, args.ssh_port,
                           "echo OSD; tail -n 20 /var/log/jotnar-osd.log 2>&1; "
                           "echo SUP; tail -n 20 /var/lib/jotnar/log/supervisor.log 2>&1; "
                           "grep -q 'timer fired for timer-policy' /var/log/jotnar-osd.log && "
                           "grep -q 'module timer-policy log: timer.fired' /var/lib/jotnar/log/supervisor.log && "
                           "echo timer-ok")
            if "timer-ok" not in proc.stdout:
                raise RuntimeError(f"M8d timer fired evidence missing: {proc.stdout} {proc.stderr}")
            trace_path = os.path.join(project, "artifacts", "m8d-timer-parity.json")
            with open(trace_path, "w", encoding="utf-8") as f:
                json.dump(trace, f, indent=2)
            log(f"m8d: PASS ({trace_path})")

        if args.m9:
            if not args.ssh_key:
                raise RuntimeError("--m9 requires --ssh-key")
            st = run_payload(args.payload_ctl, addr, token, ["module-status"])
            names = sorted(m.get("name") for m in st.get("modules", []))
            if "net-policy" not in names:
                raise RuntimeError(f"net-policy missing: {st}")
            proc = ssh_run(args.ssh_key, args.ssh_user, args.ssh_port,
                           "nc -lk -p 19000 -e /bin/cat >/tmp/nc.log 2>&1 & echo $!")
            if proc.returncode != 0 or not proc.stdout.strip():
                raise RuntimeError(f"nc server failed: {proc.stdout} {proc.stderr}")
            nc_pid = proc.stdout.strip().splitlines()[-1]

            trace = []
            for ev in ("net.status", "net.connect", "net.send", "net.recv", "net.close"):
                interp = run_payload(args.payload_ctl, addr, token,
                                     ["module-step", "net-policy", ev])
                emitted = run_payload(args.payload_ctl, addr, token,
                                      ["module-step-emitted", "net-policy", ev])
                for key in ("action", "state"):
                    if interp.get(key) != emitted.get(key):
                        raise RuntimeError(f"M9 parity {ev} {key}: {interp} vs {emitted}")
                ic = interp.get("command", {})
                ec = emitted.get("command", {})
                if (ic.get("capability"), ic.get("op"), ic.get("args")) != (
                        ec.get("capability"), ec.get("op"), ec.get("args")):
                    raise RuntimeError(f"M9 parity {ev} command: {ic} vs {ec}")
                trace.append({"event": ev, "interpreted": interp, "emitted": emitted})
            log("m9: net capability interpreted/emitted parity PASS")

            out = run_payload(args.payload_ctl, addr, token,
                              ["module-event", "net-policy", "net.status"])
            ifaces = out.get("result", {}).get("result", {}).get("interfaces", [])
            if not any(i.get("name") == "eth0" for i in ifaces):
                raise RuntimeError(f"M9 net.status missing eth0: {out}")
            out = run_payload(args.payload_ctl, addr, token,
                              ["module-event", "net-policy", "net.connect"])
            handle = out.get("result", {}).get("result", {}).get("handle")
            if not handle:
                raise RuntimeError(f"M9 emitted net.connect failed: {out}")
            out = run_payload(args.payload_ctl, addr, token,
                              ["module-event", "net-policy", "net.send"])
            if out.get("result", {}).get("ok") is not True:
                raise RuntimeError(f"M9 emitted net.send failed: {out}")
            out = run_payload(args.payload_ctl, addr, token,
                              ["module-event", "net-policy", "net.recv"])
            text = out.get("result", {}).get("result", {}).get("text")
            if text != "ping\n":
                raise RuntimeError(f"M9 emitted net.recv mismatch: {out}")
            run_payload(args.payload_ctl, addr, token,
                        ["module-event", "net-policy", "net.close"])

            out = run_payload(args.payload_ctl, addr, token,
                              ["module-event-interpreted", "net-policy", "net.connect"])
            if out.get("result", {}).get("ok") is not True:
                raise RuntimeError(f"M9 interpreted net.connect failed: {out}")
            run_payload(args.payload_ctl, addr, token,
                        ["module-event-interpreted", "net-policy", "net.send"])
            out = run_payload(args.payload_ctl, addr, token,
                              ["module-event-interpreted", "net-policy", "net.recv"])
            text = out.get("result", {}).get("result", {}).get("text")
            if text != "ping\n":
                raise RuntimeError(f"M9 interpreted net.recv mismatch: {out}")
            run_payload(args.payload_ctl, addr, token,
                        ["module-event-interpreted", "net-policy", "net.close"])
            ssh_run(args.ssh_key, args.ssh_user, args.ssh_port,
                    f"kill {nc_pid} 2>/dev/null || true")
            trace_path = os.path.join(project, "artifacts", "m9-net-parity.json")
            with open(trace_path, "w", encoding="utf-8") as f:
                json.dump(trace, f, indent=2)
            log(f"m9: PASS ({trace_path})")

        if args.m10:
            if not args.ssh_key:
                raise RuntimeError("--m10 requires --ssh-key")
            bundle = os.path.join(project, "artifacts",
                                  "console-alpine-minirootfs-3.24.4.tar.gz")
            run_payload(args.payload_ctl, addr, token, ["bundle-put", "console", bundle])
            out = run_payload(args.payload_ctl, addr, token,
                              ["console-start", "console", "--vt", "2"])
            pid1 = out.get("pid")
            if not pid1:
                raise RuntimeError(f"M10 console start failed: {out}")
            time.sleep(1.2)
            out = run_payload(args.payload_ctl, addr, token,
                              ["console-write", "console", "--data", "touch /m10-ok\n"])
            if out.get("ok") is not True:
                raise RuntimeError(f"M10 console-write failed: {out}")
            time.sleep(0.8)
            proc = ssh_run(args.ssh_key, args.ssh_user, args.ssh_port,
                           "test -f /var/lib/jotnar/payloads/console/rootfs/m10-ok && echo write-ok")
            if "write-ok" not in proc.stdout:
                raise RuntimeError(f"M10 console input marker missing: {proc.stdout} {proc.stderr}")
            out = run_payload(args.payload_ctl, addr, token,
                              ["console-restart", "console", "--vt", "2"])
            pid2 = out.get("pid")
            if not pid2:
                raise RuntimeError(f"M10 console-restart failed: {out}")
            time.sleep(1.0)
            out = run_payload(args.payload_ctl, addr, token,
                              ["console-write", "console", "--data", "touch /m10-restart\n"])
            if out.get("ok") is not True:
                raise RuntimeError(f"M10 console-write after restart failed: {out}")
            time.sleep(0.8)
            proc = ssh_run(args.ssh_key, args.ssh_user, args.ssh_port,
                           "test -f /var/lib/jotnar/payloads/console/rootfs/m10-restart && echo restart-ok")
            if "restart-ok" not in proc.stdout:
                raise RuntimeError(f"M10 restart marker missing: {proc.stdout} {proc.stderr}")
            run_payload(args.payload_ctl, addr, token, ["console-stop", "console"])
            st = run_payload(args.payload_ctl, addr, token, ["console-status", "console"])
            if st.get("consoles", [{}])[0].get("state") != "stopped":
                raise RuntimeError(f"M10 console did not stop: {st}")
            log(f"m10: console write/restart PASS (pids {pid1}->{pid2})")

        if args.m11:
            import random
            rng = random.Random(1234)
            plan = {
                "console-policy": ["console.start", "console.stop", "console.status"],
                "system-log-policy": ["system.log"],
                "denied-policy": ["console.start"],
                "file-policy": ["file.write", "file.read", "file.list"],
                "proc-policy": ["proc.status", "proc.signal"],
                "timer-policy": ["timer.arm", "timer.cancel", "timer.fired"],
                "net-policy": ["net.status", "net.connect", "net.send", "net.recv", "net.close"],
                "job-policy": ["job.submit", "job.status", "job.cancel", "job.list"],
            }
            trace = []
            checked = 0
            for mod, events in plan.items():
                for _ in range(12):
                    ev = rng.choice(events)
                    interp = run_payload(args.payload_ctl, addr, token,
                                         ["module-step", mod, ev])
                    emitted = run_payload(args.payload_ctl, addr, token,
                                          ["module-step-emitted", mod, ev])
                    for key in ("action", "state"):
                        if interp.get(key) != emitted.get(key):
                            raise RuntimeError(f"M11 fuzz {mod}/{ev} {key}: {interp} vs {emitted}")
                    ic = interp.get("command", {})
                    ec = emitted.get("command", {})
                    if (ic.get("capability"), ic.get("op"), ic.get("args")) != (
                            ec.get("capability"), ec.get("op"), ec.get("args")):
                        raise RuntimeError(f"M11 fuzz {mod}/{ev} command: {ic} vs {ec}")
                    ir = interp.get("result", {})
                    er = emitted.get("result", {})
                    if ir.get("ok") != er.get("ok"):
                        raise RuntimeError(f"M11 fuzz {mod}/{ev} result: {ir} vs {er}")
                    checked += 1
                    trace.append({"module": mod, "event": ev,
                                  "interpreted": interp, "emitted": emitted})
            trace_path = os.path.join(project, "artifacts", "m11-fuzz-parity.json")
            with open(trace_path, "w", encoding="utf-8") as f:
                json.dump(trace, f, indent=2)
            log(f"m11: {checked} randomized parity steps PASS ({trace_path})")

        # Clean shutdown through the local console.
        if args.keyboard:
            log("typing 'poweroff -f' on the emulated local keyboard")
            time.sleep(2.0)
            send_keys(monitor_sock, list("poweroff") + ["spc", "minus", "f", "ret"])
            try:
                qemu.wait(timeout=30)
            except subprocess.TimeoutExpired:
                log("keyboard poweroff timed out; falling back to serial")
                sock.sendall(b"poweroff -f\n")
                try:
                    qemu.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    pass
                raise RuntimeError("keyboard input did not reach the tty0 shell")
        else:
            sock.sendall(b"poweroff -f\n")
            log("sent poweroff")
            try:
                qemu.wait(timeout=30)
            except subprocess.TimeoutExpired:
                raise RuntimeError("guest did not power off within 30s")
        success = qemu.returncode == 0
        log(f"QEMU exited ({qemu.returncode})")
    finally:
        if sock is not None:
            try:
                drain(sock, buf, time.time() + 1)
            except Exception:
                pass
            sock.close()
        if os.path.exists(serial_sock):
            os.unlink(serial_sock)
        if os.path.exists(monitor_sock):
            os.unlink(monitor_sock)
        if overlay and os.path.exists(overlay):
            os.unlink(overlay)
        if qemu.poll() is None:
            qemu.terminate()
            try:
                qemu.wait(timeout=5)
            except subprocess.TimeoutExpired:
                qemu.kill()
        with open(args.serial_log, "wb") as f:
            f.write(buf)
        log(f"serial log: {args.serial_log}")

    if not success:
        tail = bytes(buf)[-4000:].decode("utf-8", "replace")
        print(tail, file=sys.stderr)
        raise RuntimeError("smoke test failed")

    log("PASS")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(f"[smoke] FAIL: {exc}", file=sys.stderr)
        sys.exit(1)
