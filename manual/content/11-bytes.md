---
title: "Bytes, Containers, and Masks"
number: "10"
kicker: REFERENCE / THE INSTRUCTION SET
lede: A byte stream is an integer stream with a small range. Everything else is indexing, checking, and refusing to read past the end.
columns: 2
---

The byte family treats a 1D integer stream as a sequence of bytes. There is no separate `bytes` value at the language level; a byte is an `I64` in `[0,255]` (or a `U8` at the payload boundary). The operations are explicit about bounds, because a decoder that silently reads past the end is a security bug with a nicer name.

## The byte operations

| Mnemonic | Signature | Behavior |
|---|---|---|
| `IWINDOW` | `I:*, *, F:SCALAR -> I:*` | Slice `[start, start+len)`. |
| `IFIND` | `I:*, I:* -> I:*` | First index of a pattern; `-1` if absent. |
| `IGATHER1` | `I:*, I:* -> I:*` | Gather from a 1D table by indices. |
| `IASSEMBLE` | `I:*, F:SCALAR, F:SCALAR -> I:*` | Assemble 1..8 little-endian bytes into an integer. |
| `ISUM` | `I:*, F:SCALAR, F:SCALAR -> I:*` | Sum a slice. |
| `ISCATTER1` | `I:*, I:*, I:* -> I:*` | Scatter values into a 1D copy at indices. |
| `ICRC32` | `I:* -> I:*` | CRC-32 (IEEE) over bytes. |
| `ICRC32C` | `I:*, I:* -> I:*` | CRC-32C with an explicit seed. |
| `IVARINT` | `I:*, * -> I:*, I:*` | Decode a varint; returns value and bytes consumed. |
| `IINFLATE` | `I:*, F:SCALAR -> I:*` | Raw DEFLATE decode with an expected size. |

Every slice checks `start >= 0`, `len >= 0`, and `start + len <= len(x)`. Every gather checks each index against the table length. Every scatter checks each index against the destination. `IVARINT` reports truncation; `IINFLATE` reports a bad stream; `IFIND` returns `-1` rather than throwing, because “not found” is a normal result.

::: warning [BOUNDS ARE NOT OPTIONAL]
`IWINDOW(x, -1, 4)` is an error. `IGATHER1(table, [999])` is an error. `ISCATTER1` with an out-of-range index is an error. The runtime does not clamp, wrap, or skip. If you need a safer index, compute it with `IAND` or `IMOD` and check it first.
:::

## Checksums

`ICRC32` computes the IEEE CRC-32 of a byte stream, with the usual initial value and final XOR:

```asm
IN bytes
OUT = ICRC32(bytes)
```

`ICRC32C` takes an explicit seed, so a protocol can continue a checksum across chunks:

```asm
IN chunk
IN seed
OUT = ICRC32C(chunk, seed)
```

The polynomial is part of the mnemonic, not a configuration knob. If you need a different CRC, write it with the integer family; the byte family stays small on purpose.

## Varints and inflate

`IVARINT` decodes a base-128 varint at a start index and returns two streams: the value and the number of bytes consumed. That second output is what makes it composable: you can advance the cursor without guessing how long the encoding was.

```asm
IN data
value, used = IVARINT(data, 0)
next = IADD(used, 0)
```

`IINFLATE` decodes a raw DEFLATE stream with an expected output size. The expected size is a scalar, not a buffer size hint; the decoder allocates exactly that many bytes and fails if the stream would produce a different count. That keeps the operation deterministic and avoids a class of decompression bombs.

## Containers

The container operations are shape moves, not byte moves:

| Mnemonic | Signature | Behavior |
|---|---|---|
| `GATHER` | `*, I:* -> *` | Gather rows by index. |
| `GATHER3` | `*, I:*, I:* -> *` | Gather from a 3D table by `i` and `j`. |
| `SCATTER` | `$A, $A, I:* -> $A` | Scatter values into a copy. |
| `SLICE` | `$A, F:SCALAR, F:SCALAR, F:SCALAR -> $A` | Slice along an axis. |
| `CONCAT` | `$A, $A, F:SCALAR -> $A` | Concatenate along an axis. |
| `SELECT` | `I:*, $A, $A -> $A` | Masked select. |
| `GT` | `$A, $A -> I:*` | Greater-than mask. |

`GATHER` checks that the index vector is in range for the first axis. `GATHER3` checks that `i` and `j` are 1D, the same length, and in range for the table’s first two axes. `SCATTER` writes into a copy; it does not mutate the input stream.

::: note [WHY A COPY?]
The language has no mutation. A “scatter” is a new stream with some positions overwritten. That is what makes it safe to reorder independent work: no two operations can race on the same underlying buffer because there is no shared mutable buffer in the graph.
:::

## Two real consumers

The Futhark demo uses `IWINDOW` to slide a four-character window over a word, `IXOR`/`IMUL`/`IADD` to compute a fingerprint, and `IGATHER1` to read a sorted lexicon table. The DOOM runner uses the same byte family for a compiled RV32IM image: fetch bytes, assemble instructions, decode fields, and checksum frames.

Neither demo needed a new byte type. The integer family plus bounds checks was enough.

::: pullquote [BYTE RULE]
A byte is an integer with a small range. A decoder is a graph with a bounds check. Never confuse the two.
:::

::: end
:::
