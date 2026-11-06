#!/usr/bin/env python3
"""Python reference for the Futhark Lens translator.

This mirrors `futhark_language/phonetic.js` and `translator.js` closely
enough to generate golden outputs for the Jotnar rewrite. It is a
reference, not the production extension.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

EXCEPTIONS = {
    "a": ["AH"], "an": ["AE", "N"], "are": ["AA", "R"],
    "come": ["K", "AH", "M"], "could": ["K", "UH", "D"],
    "done": ["D", "AH", "N"], "give": ["G", "IH", "V"],
    "have": ["HH", "AE", "V"], "he": ["HH", "IY"],
    "love": ["L", "AH", "V"], "of": ["AH", "V"],
    "one": ["W", "AH", "N"], "said": ["S", "EH", "D"],
    "says": ["S", "EH", "Z"], "some": ["S", "AH", "M"],
    "the": ["DH", "AH"], "to": ["T", "UW"], "two": ["T", "UW"],
    "was": ["W", "AH", "Z"], "were": ["W", "ER"],
    "you": ["Y", "UW"], "your": ["Y", "ER"],
    "can't": ["K", "AE", "N", "T"], "don't": ["D", "OW", "N", "T"],
    "isn't": ["IH", "Z", "AH", "N", "T"],
    "doesn't": ["D", "AH", "Z", "AH", "N", "T"],
    "didn't": ["D", "IH", "D", "AH", "N", "T"],
    "won't": ["W", "OW", "N", "T"],
    "wouldn't": ["W", "UH", "D", "AH", "N", "T"],
    "shouldn't": ["SH", "UH", "D", "AH", "N", "T"],
    "couldn't": ["K", "UH", "D", "AH", "N", "T"],
}

VOWEL_SOUNDS = {"a": "AE", "e": "EH", "i": "IH", "o": "AA", "u": "AH"}
LETTER_SOUNDS = {
    "a": "AE", "b": "B", "d": "D", "e": "EH", "f": "F", "h": "HH",
    "i": "IH", "j": "JH", "k": "K", "l": "L", "m": "M", "n": "N",
    "o": "AA", "p": "P", "q": "K", "r": "R", "s": "S", "t": "T",
    "u": "AH", "v": "V", "w": "W", "x": "KS", "y": "Y", "z": "Z",
    "c": "K", "g": "G",
}
DIGRAPHS = [
    ("tion", ["SH", "AH", "N"]), ("sion", ["ZH", "AH", "N"]),
    ("ture", ["CH", "ER"]), ("igh", ["AY"]), ("eigh", ["EY"]),
    ("ough", ["AO"]), ("augh", ["AE"]), ("ph", ["F"]),
    ("sh", ["SH"]), ("ch", ["CH"]), ("th", ["TH"]), ("ng", ["NG"]),
    ("qu", ["KW"]), ("ck", ["K"]), ("wh", ["W"]), ("wr", ["R"]),
    ("kn", ["N"]), ("gn", ["N"]), ("ee", ["IY"]), ("ea", ["IY"]),
    ("oo", ["UW"]), ("oi", ["OY"]), ("oy", ["OY"]), ("ai", ["EY"]),
    ("ay", ["EY"]), ("oa", ["OW"]), ("ou", ["AW"]), ("ow", ["OW"]),
    ("au", ["AO"]), ("aw", ["AO"]), ("er", ["ER"]), ("ir", ["ER"]),
    ("ur", ["ER"]), ("or", ["AO", "R"]), ("ar", ["AA", "R"]),
]
RUNES = {
    "P": "ᛈ", "B": "ᛒ", "T": "ᛏ", "D": "ᛞ", "K": "ᚲ", "G": "ᚷ",
    "M": "ᛗ", "N": "ᚾ", "L": "ᛚ", "R": "ᚱ", "F": "ᚠ", "V": "ᚹ",
    "S": "ᛋ", "Z": "ᛉ", "HH": "ᚺ", "W": "ᚹ", "Y": "ᛃ", "TH": "ᚦ",
    "DH": "ᚦ", "SH": "ᛋᚺ", "CH": "ᚲᚺ", "JH": "ᛃ", "NG": "ᛝ",
    "AE": "ᚨ", "EH": "ᛖ", "IH": "ᛁ", "IY": "ᛁ", "AA": "ᚨ",
    "AH": "ᚨ", "AO": "ᛟ", "UH": "ᚢ", "UW": "ᚢ", "ER": "ᛖᚱ",
    "AY": "ᚨᛁ", "EY": "ᛖᛁ", "OW": "ᛟᚢ", "AW": "ᚨᚢ",
    "OY": "ᛟᛃ", "YUW": "ᛃᚢ", "KW": "ᚲᚹ", "KS": "ᚲᛊ",
}
APOS = "APOS"


def is_vowel(ch: str) -> bool:
    return ch in "aeiou"


def is_final(word: str, index: int) -> bool:
    return index >= len(word) - 1


def has_magic_e(word: str, index: int) -> bool:
    return (
        index + 2 < len(word)
        and is_vowel(word[index])
        and not is_vowel(word[index + 1])
        and word[index + 2] == "e"
        and is_final(word, index + 2)
    )


def long_vowel(ch: str) -> str:
    return {"a": "EY", "e": "IY", "i": "AY", "o": "OW", "u": "YUW"}[ch]


def spelling_to_phonemes(word: str) -> list[str]:
    lower = word.lower()
    if lower in EXCEPTIONS:
        return list(EXCEPTIONS[lower])
    result: list[str] = []
    index = 0
    while index < len(lower):
        matched = False
        for pattern, phonemes in DIGRAPHS:
            if lower[index:index + len(pattern)] != pattern:
                continue
            if pattern == "eigh" and index > 0:
                result.append("EY")
            else:
                result.extend(phonemes)
            index += len(pattern)
            matched = True
            break
        if matched:
            continue
        ch = lower[index]
        if ch == "e" and is_final(lower, index) and result:
            index += 1
            continue
        if is_vowel(ch) and has_magic_e(lower, index):
            result.append(long_vowel(ch))
            index += 1
            continue
        nxt = lower[index + 1] if index + 1 < len(lower) else ""
        if ch == "c":
            result.append("S" if nxt in ("e", "i", "y") else "K")
        elif ch == "g":
            result.append("JH" if nxt in ("e", "i", "y") else "G")
        elif ch == "y":
            result.append("Y" if index == 0 else ("IY" if is_final(lower, index) else "IH"))
        else:
            result.append(LETTER_SOUNDS.get(ch, ch.upper()))
        index += 1
    return result


def contraction_phonemes(word: str, pron=None) -> tuple[list[str], list[str]] | None:
    lower = word.lower().replace("’", "'")
    match = re.fullmatch(r"([a-z]+)'(s|re|ve|ll|d|m|t)", lower)
    if not match:
        return None
    if lower in EXCEPTIONS:
        known = EXCEPTIONS[lower]
        suffix_len = {"re": 1, "ve": 1, "ll": 1, "d": 1, "m": 1, "t": 1}.get(match.group(2), 1)
        return known[:-suffix_len], known[-suffix_len:]
    base = pron(match.group(1)) if pron else spelling_to_phonemes(match.group(1))
    suffix = match.group(2)
    if suffix == "s":
        last = base[-1]
        if last in {"S", "Z", "SH", "CH", "JH"}:
            return base, ["IH", "Z"]
        if last in {"P", "T", "K", "F", "TH"}:
            return base, ["S"]
        return base, ["Z"]
    fixed = {"re": ["R"], "ve": ["V"], "ll": ["L"], "d": ["D"],
             "m": ["M"], "t": ["T"]}[suffix]
    return base, fixed


def rune_for(phoneme: str) -> str:
    if phoneme == APOS:
        return "'"
    return RUNES.get(phoneme, phoneme)


def runes_for(phonemes: list[str]) -> str:
    return "".join(rune_for(p) for p in phonemes)


class Translator:
    def __init__(self, artifact: dict[str, Any]):
        self.lexicon = artifact["lexicon"]

    def pronunciation_for_word(self, word: str) -> list[str]:
        entries = self.lexicon.get(word.lower())
        if entries:
            return list(entries[0])
        return spelling_to_phonemes(word)

    def word_to_phonemes(self, word: str) -> list[str]:
        contraction = contraction_phonemes(word, self.pronunciation_for_word)
        if contraction:
            base, suffix = contraction
            return base + [APOS] + suffix
        entries = self.lexicon.get(word.lower())
        if entries:
            return list(entries[0])
        if word.lower() in EXCEPTIONS:
            return list(EXCEPTIONS[word.lower()])
        return self.pronunciation_for_word(word)

    def word_to_runes(self, word: str) -> str:
        return runes_for(self.word_to_phonemes(word))

    def transliterate(self, text: str) -> str:
        translated = re.sub(
            r"[A-Za-z]+(?:['’][A-Za-z]+)*",
            lambda m: self.word_to_runes(m.group(0)),
            text,
        )
        return translated


def fallback_action(c0: int, c1: int, c2: int, c3: int,
                    rem: int, has_out: int, pos_zero: int) -> tuple[list[str], int]:
    """One greedy spelling step, matching `spelling_to_phonemes` locally."""
    if rem <= 0:
        return [], 0
    chars = "abcdefghijklmnopqrstuvwxyz"
    cs = [c0, c1, c2, c3]
    word = "".join(chars[c - 1] if 1 <= c <= 26 else "?" for c in cs[:rem])
    index = 0
    for pattern, phonemes in DIGRAPHS:
        if len(pattern) > rem:
            continue
        if word[index:index + len(pattern)] != pattern:
            continue
        if pattern == "eigh" and not pos_zero:
            return ["EY"], len(pattern) - 1
        return list(phonemes), len(pattern) - 1
    ch = word[index]
    if ch == "e" and is_final(word, index) and has_out:
        return [], 0
    if is_vowel(ch) and has_magic_e(word, index):
        return [long_vowel(ch)], 0
    nxt = word[index + 1] if index + 1 < len(word) else ""
    if ch == "c":
        return ["S" if nxt in ("e", "i", "y") else "K"], 0
    if ch == "g":
        return ["JH" if nxt in ("e", "i", "y") else "G"], 0
    if ch == "y":
        return ["Y" if pos_zero else ("IY" if is_final(word, index) else "IH")], 0
    return [LETTER_SOUNDS.get(ch, ch.upper())], 0


# Character encoding used by the Jotnar program.
def char_id(ch: str) -> int:
    if ch == "'" or ch == "’":
        return 27
    if "a" <= ch <= "z":
        return 1 + ord(ch) - ord("a")
    return 0


def encode_word(word: str, length: int = 28) -> list[int]:
    out = [0] * length
    for i, ch in enumerate(word[:length]):
        out[i] = char_id(ch)
    return out


def phoneme_ids(extra=()) -> dict[str, int]:
    names = set(RUNES) | {APOS} | set(extra)
    for phonemes in EXCEPTIONS.values():
        names.update(phonemes)
    for _, phonemes in DIGRAPHS:
        names.update(phonemes)
    names.update(LETTER_SOUNDS.values())
    # NONE is padding; APOS is the apostrophe; real phonemes start at 2.
    ordered = ["NONE", APOS] + sorted(n for n in names if n not in {"NONE", APOS})
    return {n: i for i, n in enumerate(ordered)}


def rune_codepoints() -> dict[str, list[int]]:
    out = {"NONE": [], APOS: [ord("'")]}
    for name, s in RUNES.items():
        out[name] = [ord(ch) for ch in s]
    return out

def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("artifact", nargs="?", default="/home/balrog/Documents/OpenCode/futhark_language/ga_rules.json")
    ap.add_argument("words", nargs="*", default=["the", "quick", "brown", "fox", "make", "global", "qwerty"])
    args = ap.parse_args()
    t = Translator(json.loads(Path(args.artifact).read_text()))
    for w in args.words:
        print(w, t.word_to_phonemes(w), t.word_to_runes(w))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
