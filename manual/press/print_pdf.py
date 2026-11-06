#!/usr/bin/env python3
"""Print a Paged.js document to PDF through the Chrome DevTools Protocol.

The CLI flag --print-to-pdf fires at the end of a virtual-time budget, which
can happen while Paged.js is still chunking or while the page fonts are still
loading. This driver waits for both, then calls Page.printToPDF.
"""
import asyncio
import base64
import json
import os
import shlex
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request

import websockets


def find_browser():
    for name in ("chromium", "chromium-browser", "google-chrome", "chrome"):
        if shutil.which(name):
            return name
    raise SystemExit("no Chromium/Chrome found on PATH")


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


async def cdp(ws, method, params=None):
    await ws.send(json.dumps({"id": next_id(), "method": method, "params": params or {}}))
    while True:
        msg = json.loads(await ws.recv())
        if msg.get("id") == _last_id[0]:
            if "error" in msg:
                raise RuntimeError(f"{method}: {msg['error']}")
            return msg.get("result", {})


_id = 0


def next_id():
    global _id
    _id += 1
    _last_id[0] = _id
    return _id


_last_id = [0]


async def evaluate(ws, expression, await_promise=False):
    r = await cdp(
        ws,
        "Runtime.evaluate",
        {
            "expression": expression,
            "returnByValue": True,
            "awaitPromise": await_promise,
        },
    )
    return r.get("result", {}).get("value")


async def main(html_path, pdf_path, timeout=180):
    html_path = os.path.abspath(html_path)
    pdf_path = os.path.abspath(pdf_path)
    port = free_port()
    browser = find_browser()
    profile = tempfile.mkdtemp(prefix="jotnar-press-")
    # Linux Mint's /usr/bin/chromium is a wrapper script. Keep the quote
    # around the '*' value; quoting the whole option makes the wrapper see a
    # second target in headless mode.
    cmd = " ".join(
        [
            shlex.quote(browser),
            "--headless=new",
            "--no-sandbox",
            "--disable-gpu",
            "--disable-dev-shm-usage",
            "--no-first-run",
            "--no-default-browser-check",
            "--allow-file-access-from-files",
            "--user-data-dir",
            shlex.quote(profile),
            f"--remote-debugging-port={port}",
            "--remote-allow-origins='*'",
        ]
    )
    proc = subprocess.Popen(
        cmd,
        shell=True,
        executable="/bin/bash",
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        endpoint = None
        for _ in range(200):
            try:
                with urllib.request.urlopen(
                    f"http://127.0.0.1:{port}/json/version", timeout=0.5
                ) as r:
                    endpoint = json.load(r)["webSocketDebuggerUrl"]
                break
            except Exception:
                time.sleep(0.05)
        if not endpoint:
            err = ""
            try:
                _, err = proc.communicate(timeout=2)
            except Exception:
                pass
            raise RuntimeError(
                f"Chromium did not expose a debugging endpoint "
                f"(port {port}, rc {proc.poll()}, log {err[-800:]!r})"
            )
        async with websockets.connect(endpoint, max_size=64 * 1024 * 1024) as browser_ws:
            target = await cdp(browser_ws, "Target.createTarget", {"url": "about:blank"})
            target_id = target["targetId"]
            targets = await cdp(browser_ws, "Target.getTargets")
            page_ws_url = None
            for t in targets.get("targetInfos", []):
                if t.get("targetId") == target_id:
                    page_ws_url = f"ws://127.0.0.1:{port}/devtools/page/{target_id}"
                    break
            if not page_ws_url:
                raise RuntimeError("could not find page target")
            async with websockets.connect(page_ws_url, max_size=64 * 1024 * 1024) as ws:
                await cdp(ws, "Page.enable")
                await cdp(ws, "Runtime.enable")
                await cdp(ws, "Page.navigate", {"url": "file://" + html_path})
                # wait for load
                for _ in range(600):
                    state = await evaluate(ws, "document.readyState")
                    if state == "complete":
                        break
                    await asyncio.sleep(0.05)
                # wait for fonts and Paged.js
                await evaluate(ws, "document.fonts.ready.then(() => true)", True)
                stable = 0
                last = -1
                deadline = time.time() + timeout
                while time.time() < deadline:
                    count = await evaluate(
                        ws,
                        "document.querySelectorAll('.pagedjs_page').length",
                    )
                    ready = await evaluate(
                        ws, "document.body.getAttribute('data-press-ready') === '1'"
                    )
                    if count and count == last:
                        stable += 1
                    else:
                        stable = 0
                        last = count
                    if ready or stable >= 10:
                        break
                    await asyncio.sleep(0.1)
                count = await evaluate(
                    ws, "document.querySelectorAll('.pagedjs_page').length"
                )
                if not count:
                    raise RuntimeError("Paged.js produced no pages")
                # Fill the TOC from the final page boxes. The PagedConfig
                # `after` hook can fire before the last pages are laid out.
                await evaluate(
                    ws,
                    """
                    (() => {
                      const pages = Array.from(document.querySelectorAll('.pagedjs_page'));
                      const pageOf = new Map();
                      pages.forEach((page, idx) => {
                        page.querySelectorAll('[id]').forEach(el => {
                          if (!pageOf.has(el.id)) pageOf.set(el.id, idx + 1);
                        });
                      });
                      document.querySelectorAll('.toc-page[data-target]').forEach(el => {
                        const n = pageOf.get(el.getAttribute('data-target'));
                        if (n) el.textContent = String(n).padStart(2, '0');
                      });
                      return pages.length;
                    })()
                    """,
                )
                await asyncio.sleep(0.2)
                await cdp(ws, "Emulation.setEmulatedMedia", {"media": "print"})
                result = await cdp(
                    ws,
                    "Page.printToPDF",
                    {
                        "printBackground": True,
                        "preferCSSPageSize": True,
                        "marginTop": 0,
                        "marginRight": 0,
                        "marginBottom": 0,
                        "marginLeft": 0,
                        "displayHeaderFooter": False,
                    },
                )
                data = base64.b64decode(result["data"])
                os.makedirs(os.path.dirname(pdf_path) or ".", exist_ok=True)
                with open(pdf_path, "wb") as f:
                    f.write(data)
                print(f"pagedjs: {count} pages -> {pdf_path}")
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
        shutil.rmtree(profile, ignore_errors=True)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("usage: print_pdf.py <manual.html> <manual.pdf>")
    asyncio.run(main(sys.argv[1], sys.argv[2]))
