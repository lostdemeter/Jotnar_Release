#!/usr/bin/env python3
"""Differential gate: Jotnar Futhark unit vs the Python/JS rule reference."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import futhark_reference as R  # noqa: E402

ARTIFACT = Path.home() / "Documents" / "OpenCode" / "futhark_language" / "ga_rules.json"
PAYLOAD = Path("/tmp/opencode/futhark_nc")
EXE = ROOT / "target" / "release" / "jotnar-demo-futhark"


def corpus() -> list[str]:
    data = json.loads(ARTIFACT.read_text())
    keys = sorted(data["lexicon"])
    words = [
        "the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog",
        "make", "global", "hello", "world", "futhark", "jotnar", "runes",
        "elder", "runic", "transliterate", "strengths", "chthonic",
        "qwerty", "florp", "blorptastic", "zib", "qatar", "nature",
        "nation", "thought", "through", "enough", "eigh", "y", "c", "g",
        "trump's", "isn't", "can't", "don't",
    ]
    words += keys[:25]
    # Deterministic unknown words for the fallback table.
    for stem in ("b", "br", "fl", "gr", "sn", "tr", "v", "z"):
        for tail in ("orp", "int", "astic", "une", "elly", "ow", "im"):
            words.append(stem + tail)
    out = []
    seen = set()
    for w in words:
        if w not in seen:
            seen.add(w)
            out.append(w)
    return out


def main() -> int:
    if not PAYLOAD.exists():
        print(f"building payload -> {PAYLOAD}")
        subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "build_futhark_payload.py"),
             "--no-contractions", "--out", str(PAYLOAD)],
            check=True,
        )
    if not EXE.exists():
        subprocess.run(["cargo", "build", "--release", "--bin", "jotnar-demo-futhark"],
                       cwd=ROOT, check=True)
    words = corpus()
    cpath = Path("/tmp/opencode/futhark_corpus.txt")
    cpath.write_text("\n".join(words) + "\n")
    r = subprocess.run([str(EXE), "--payload", str(PAYLOAD), "--corpus", str(cpath)],
                       cwd=ROOT, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stderr)
        return 1
    got = {}
    for line in r.stdout.splitlines():
        w, _, out = line.partition("\t")
        got[w] = out
    ref = R.Translator(json.loads(ARTIFACT.read_text()))
    bad = 0
    for w in words:
        want = ref.word_to_runes(w)
        if got.get(w) != want:
            bad += 1
            print(f"MISMATCH {w!r}: jotnar={got.get(w)!r} ref={want!r}")
    print(f"futhark parity: {len(words) - bad}/{len(words)}")
    print("RESULT:", "ALL OK" if bad == 0 else "FAIL")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
