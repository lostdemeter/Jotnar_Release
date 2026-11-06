#!/usr/bin/env python3
"""Validate the Python Futhark reference against the actual browser JS.

Optional: needs `chromium` (headless). This is the oracle check behind
`parity_futhark.py`; it is kept separate because Chromium is not a build
dependency of the Jotnar program.
"""
from __future__ import annotations

import html
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import futhark_reference as R  # noqa: E402
import parity_futhark as P  # noqa: E402

ARTIFACT = Path("/home/balrog/Documents/OpenCode/futhark_language/ga_rules.json")
JS_DIR = Path("/home/balrog/Documents/OpenCode/futhark_language")
WORK = Path("/tmp/opencode/futhark_js")


def main() -> int:
    chromium = shutil.which("chromium") or shutil.which("chromium-browser")
    if not chromium:
        print("SKIP: chromium not found")
        return 0
    WORK.mkdir(parents=True, exist_ok=True)
    (WORK / "artifact.js").write_text("window.ARTIFACT = " + ARTIFACT.read_text() + ";")
    words = P.corpus()
    (WORK / "harness.html").write_text(
        f"""<!doctype html><meta charset="utf-8">
<script src="file://{JS_DIR / 'phonetic.js'}"></script>
<script src="file://{JS_DIR / 'translator.js'}"></script>
<script src="file://{WORK / 'artifact.js'}"></script>
<pre id="out"></pre>
<script>
FutharkPhonetics.setLexicon(ARTIFACT.lexicon);
var words = {json.dumps(words)};
var r = {{}};
for (var i=0;i<words.length;i++) r[words[i]] = FutharkPhonetics.wordToRunes(words[i], {{ingwaz:'stacked',ansuz:'standard',sowilo:'three-stroke',hagalaz:'standard',uruz:'standard'}});
document.getElementById('out').textContent = JSON.stringify(r);
</script>"""
    )
    dom = subprocess.run(
        [chromium, "--headless", "--disable-gpu", "--no-sandbox",
         "--allow-file-access-from-files", "--dump-dom", f"file://{WORK / 'harness.html'}"],
        capture_output=True, text=True,
    )
    m = re.search(r'<pre id="out">(.*?)</pre>', dom.stdout, re.S)
    if not m:
        print(dom.stderr[-500:])
        return 1
    js = json.loads(html.unescape(m.group(1)))
    ref = R.Translator(json.loads(ARTIFACT.read_text()))
    bad = 0
    for w, v in js.items():
        want = ref.word_to_runes(w)
        if v != want:
            bad += 1
            print(f"MISMATCH {w!r}: js={v!r} python={want!r}")
    print(f"js-vs-python: {len(js) - bad}/{len(js)}")
    print("RESULT:", "ALL OK" if bad == 0 else "FAIL")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
