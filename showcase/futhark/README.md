# Futhark as a Jotnar inference unit

This is the second-leg opener: one word is one request to one Jotnar
listing. The host only encodes characters and renders rune codepoints; the
translation decision lives in the program.

```text
text/word
  -> host: character IDs [32] + length
  -> Jotnar: lexicon fingerprint lookup + compiled spelling rules
  -> phoneme IDs -> rune codepoints
  -> host: UTF-8 rendering
```

## Build the data payload

The listing is code; the lexicon and rule tables are ordinary `I` payload
streams. Build them from the trained Futhark artifact:

```sh
python3 scripts/build_futhark_payload.py \
  --artifact /home/balrog/Documents/OpenCode/futhark_language/ga_rules.json \
  --no-contractions \
  --out showcase/futhark/payload
```

`--no-contractions` keeps the payload to the 117,505 trained entries
(about 20 MB). Common contractions already present in the lexicon or the
JS exception table work (`trump's`, `isn't`, `can't`, `don't`). Omitting
the flag expands all seven suffixes for every lexicon base
(939,951 entries, about 100 MB); that makes arbitrary known-base
contractions work, at a large load cost. Unknown-base contractions remain
a follow-up.

## Run one word

```sh
cargo run --release --bin jotnar-demo-futhark -- \
  --payload showcase/futhark/payload --word "the"
# the    ᚦᚨ
```

A corpus:

```sh
cargo run --release --bin jotnar-demo-futhark -- \
  --payload showcase/futhark/payload --corpus showcase/futhark/corpus.txt
```

Emit the actual listing:

```sh
cargo run --release --bin jotnar-demo-futhark -- \
  --payload showcase/futhark/payload --emit showcase/futhark/futhark.asm
```

## What is inside the listing

1. **Fingerprint index.** A 128-bit fingerprint (FNV-1a plus polynomial)
   over the fixed 28-character word. The table is sorted; the listing does
   a 17–20 step binary search. The test corpus is collision-free; the two
   independent 64-bit halves make an accidental match vanishingly unlikely.
2. **Fallback rule table.** The JavaScript greedy spelling rules are
   compiled into a dense action table over a four-character window, the
   remaining word length, `has_output`, and `position==0`. A 10 MB packed
   action-ID table covers every context that can occur; the listing reads
   one action per input position and emits up to three phonemes.
3. **Rune table.** Each phoneme maps to up to eight codepoints. This keeps
   multi-rune phonemes (`SH` -> `ᛋᚺ`, `ER` -> `ᛖᚱ`) and the unknown
   CMUdict annotations (`NAME,`, `ABBREV`, ...) byte-identical with the
   JavaScript `runeFor` fallback.

## Correctness gate

```sh
python3 scripts/parity_futhark.py
# optional oracle check (headless Chromium)
python3 scripts/check_futhark_js.py
```

Current result:

```text
futhark parity: 118/118
RESULT: ALL OK
```

The corpus covers trained words, exception contractions, and unknown words
that exercise the fallback table. The Python reference is
`scripts/futhark_reference.py`, a direct port of `phonetic.js` and
`translator.js`; it was itself checked **118/118** against the actual
JavaScript running in headless Chromium before it was used as the oracle.

## Boundary

This is **Jotnar-LAN/DIG** as a word-level inference unit. It is not the
browser extension: DOM traversal, text splitting, settings, fonts, and the
final separator stay host-side. Those are exactly the parts we said we did
not need.
