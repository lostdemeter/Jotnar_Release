#!/usr/bin/env python3
"""Build the data payload for the Jotnar Futhark translator.

The Jotnar listing is the algorithm; this script supplies the lexicon,
phoneme/rune tables, and the compiled fallback rule table as ordinary I
streams. It is deliberately a build step, not runtime host logic.

Usage:
  python3 scripts/build_futhark_payload.py \
    --artifact /home/balrog/Documents/OpenCode/futhark_language/ga_rules.json \
    --out showcase/futhark/payload

For a small test fixture:
  python3 scripts/build_futhark_payload.py --limit 200 --no-contractions \
    --out /tmp/opencode/futhark_small
"""
from __future__ import annotations

import argparse
import json
import math
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import futhark_reference as R  # noqa: E402

L = 28
A = 28
STATE = 16  # 4 rem values * 2 has_out * 2 pos_zero


def i64_bytes(values: list[int]) -> bytes:
    return struct.pack("<" + "q" * len(values), *values)


def signed64(x: int) -> int:
    x &= (1 << 64) - 1
    return x - (1 << 64) if x >= (1 << 63) else x


def word_hash(chars: list[int], length: int) -> tuple[int, int]:
    h1 = 14695981039346656037
    h2 = 0
    for c in chars[:L]:
        h1 ^= c
        h1 = (h1 * 1099511628211) & ((1 << 64) - 1)
        h2 = (h2 * 131 + c + 1) & ((1 << 64) - 1)
    h1 ^= length
    h2 = (h2 * 131 + length + 1) & ((1 << 64) - 1)
    return signed64(h1), signed64(h2)


def context_index(c0: int, c1: int, c2: int, c3: int) -> int:
    return c0 + A * (c1 + A * (c2 + A * c3))


def build_entries(translator: R.Translator, extra_words: list[str]) -> list[tuple[str, list[str]]]:
    words = set(translator.lexicon) | set(R.EXCEPTIONS) | set(extra_words)
    # Derived contractions are a compilation of the JS contraction rule:
    # every base pronunciation plus its seven suffixes becomes a word.
    if not getattr(build_entries, "no_contractions", False):
        for w in list(translator.lexicon):
            if "'" in w or "’" in w:
                continue
            for suffix in ("s", "re", "ve", "ll", "d", "m", "t"):
                words.add(w + "'" + suffix)
    entries = []
    for w in sorted(words):
        if len(w) > L:
            continue
        phonemes = translator.word_to_phonemes(w)
        if not phonemes:
            continue
        entries.append((w, phonemes))
    return entries


def build_fallback_table() -> tuple[list[int], list[int]]:
    """Return packed action IDs and the action table.

    Key = context(base-28) * 16 + (rem-1)*4 + has_out*2 + pos_zero.
    Only contexts that can occur are filled; all others stay action 0.
    """
    actions: list[int] = [0]
    action_id: dict[tuple[tuple[str, ...], int], int] = {}
    packed_n = (A ** 4) * STATE // 8
    packed = [0] * packed_n

    def aid(phonemes: list[str], skip: int) -> int:
        key = (tuple(phonemes), skip)
        if key in action_id:
            return action_id[key]
        ids = [R.PHONEME_IDS.get(p, 0) for p in phonemes] + [0, 0, 0]
        if any(v >= 64 for v in ids):
            raise SystemExit(f"phoneme id overflow: {phonemes}")
        value = ids[0] | (ids[1] << 6) | (ids[2] << 12) | (skip << 18)
        idx = len(actions)
        if idx >= 256:
            raise SystemExit("more than 256 fallback actions")
        actions.append(value)
        action_id[key] = idx
        return idx

    count = 0
    for rem in (1, 2, 3, 4):
        ranges = [range(1, 28)] * rem + [range(1)] * (4 - rem)
        for c0 in ranges[0]:
            for c1 in ranges[1]:
                for c2 in ranges[2]:
                    for c3 in ranges[3]:
                        ctx = context_index(c0, c1, c2, c3)
                        for has_out in (0, 1):
                            for pos_zero in (0, 1):
                                phonemes, skip = R.fallback_action(
                                    c0, c1, c2, c3, rem, has_out, pos_zero
                                )
                                state = (rem - 1) * 4 + has_out * 2 + pos_zero
                                key = ctx * STATE + state
                                a = aid(phonemes, skip)
                                word = key >> 3
                                shift = (key & 7) * 8
                                packed[word] |= a << shift
                                count += 1
    print(f"fallback entries: {count}, actions: {len(actions)}")
    return packed, actions


def build(artifact: Path, out: Path, limit: int | None, contractions: bool) -> None:
    data = json.loads(artifact.read_text())
    tr = R.Translator(data)
    build_entries.no_contractions = not contractions
    extra: list[str] = []
    if limit is not None:
        # Keep the fixture deterministic: the first N lexicon words plus
        # the JS exceptions, with contractions derived only from that slice.
        keys = sorted(data["lexicon"])[:limit]
        tr.lexicon = {k: data["lexicon"][k] for k in keys}
        extra = keys
    entries = build_entries(tr, extra)
    print(f"words: {len(entries)}")

    all_phonemes = {p for _, phonemes in entries for p in phonemes}
    ids = R.phoneme_ids(all_phonemes)
    R.PHONEME_IDS = ids
    known_runes = R.rune_codepoints()
    max_id = max(ids.values())
    rune_slots = [[0] * 8 for _ in range(max_id + 1)]
    for name, pid in ids.items():
        if name in known_runes:
            r = known_runes[name]
        elif name == "NONE":
            r = []
        else:
            # JavaScript runeFor() returns unknown phonemes verbatim.
            r = [ord(ch) for ch in name]
        if len(r) > 8:
            raise SystemExit(f"phoneme {name!r} needs {len(r)} rune slots")
        for i, cp in enumerate(r):
            rune_slots[pid][i] = cp

    ph: list[int] = []
    offsets: list[int] = []
    lengths: list[int] = []
    hashes: list[tuple[int, int, int]] = []
    for idx, (w, phonemes) in enumerate(entries):
        offsets.append(len(ph))
        lengths.append(len(phonemes))
        for p in phonemes:
            ph.append(ids[p])
        chars = R.encode_word(w, L)
        h1, h2 = word_hash(chars, len(w))
        hashes.append((h1, h2, idx))

    hashes.sort()
    h1s = [x[0] for x in hashes]
    h2s = [x[1] for x in hashes]
    offs = [offsets[x[2]] for x in hashes]
    lens = [lengths[x[2]] for x in hashes]
    # PH must be safe for offset+MAX_PH lookups; pad it.
    ph.extend([0] * (4 * L))

    packed, actions = build_fallback_table()

    out.mkdir(parents=True, exist_ok=True)
    files = {
        "lex_h1.bin": h1s,
        "lex_h2.bin": h2s,
        "lex_off.bin": offs,
        "lex_len.bin": lens,
        "ph.bin": ph,
        "runes.bin": [x for row in rune_slots for x in row],
        "fallback.bin": packed,
        "actions.bin": actions,
        "z84.bin": [0] * (3 * L),
        "z672.bin": [0] * (24 * L),
    }
    for name, values in files.items():
        (out / name).write_bytes(i64_bytes(values))
    print(f"lexicon entries: {len(entries)}, hash entries: {len(hashes)}")
    print(f"fallback words: {len(packed)} i64 ({len(packed) * 8 / 1e6:.1f} MB)")
    print(f"ph: {len(ph)} i64, rune slots: {max_id + 1}x8")
    print(f"wrote {out}")

    # Print a few words for a quick eyeball and for the parity script.
    preview = {}
    for w in ("the", "quick", "brown", "fox", "make", "global", "qwerty", "trump's"):
        if w in tr.lexicon or w in R.EXCEPTIONS or w in {e[0] for e in entries}:
            preview[w] = tr.word_to_runes(w)
    (out / "preview.json").write_text(json.dumps(preview, ensure_ascii=False, indent=1))
    (out / "phonemes.json").write_text(json.dumps({str(v): k for k, v in ids.items()}, indent=1))
    (out / "manifest.json").write_text(json.dumps({
        "words": len(entries),
        "hash_entries": len(hashes),
        "ph": len(ph),
        "max_phoneme_id": max_id,
        "rune_slots": 8,
        "fallback_i64": len(packed),
    }, indent=1))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact", default="/home/balrog/Documents/OpenCode/futhark_language/ga_rules.json")
    ap.add_argument("--out", default="showcase/futhark/payload")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--no-contractions", action="store_true")
    args = ap.parse_args()
    build(Path(args.artifact), Path(args.out), args.limit, not args.no_contractions)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
