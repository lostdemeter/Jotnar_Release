---
title: "Payloads, CLI, and the Witness"
number: "14"
kicker: FIELD NOTE / THE MACHINE
lede: The command line is the boundary between the graph and the world. It should be small enough to memorize and strict enough to trust.
columns: 2
---

The reference CLI has seven commands. Five of them are about a single listing; two are about the system.

| Command | What it does |
|---|---|
| `jotnar check <file.asm>` | Parse, expand, bind, and check arity. |
| `jotnar verify <file.asm>` | Replay layout unification; print coverage and the DEF interface. |
| `jotnar run <file.asm> --payload FILE` | Execute and write streams. |
| `jotnar emit-c <file.asm>` | Emit portable C99. |
| `jotnar build <file.asm> --target c\|nonfpu\|cuda` | Emit and build a native target. |
| `jotnar repeat <file.asm> --payload FILE` | Thread `STATE` streams across host iterations. |
| `jotnar ops` | Print the registry: name, arity, and whether a pattern exists. |

Common flags: `--stdlib DIR` overrides the `IMPORT` search path; `--outputs a,b` selects streams; `--out FILE` writes to a file; `--json` makes `verify` machine-readable; `--fp16` switches CUDA inputs to half precision.

## Run

```sh
jotnar run showcase/language/saxpy.asm \
  --payload /tmp/saxpy.json --outputs OUT
```

The output is a JSON stream map. Without `--outputs`, every stream is written. Without `--out`, the JSON goes to stdout.

The payload is the only input surface. It has a `streams` object; each entry has a `kind`, a `shape`, and data. Here is the full vocabulary:

```json
{
  "streams": {
    "f64": {"kind":"F64","shape":[2],"data":[0.5,-1.25]},
    "f32": {"kind":"F32","shape":[2],"data":[0.5,-1.25]},
    "i64": {"kind":"I64","shape":[2],"data":["9007199254740993","-4"]},
    "u8":  {"kind":"U8","shape":[3],"data":[0,127,255]},
    "b":   {"kind":"Bool","shape":[2],"data":[true,false]},
    "t":   {"kind":"T","shape":[2],
            "s":[1,-1],"e":[32000,33000],"z":[0,0]},
    "none":{"kind":"None","shape":[0],"data":[]}
  }
}
```

`F32` and `F64` are distinct. Integers may be numbers or strings; strings preserve int64 values above 2^53. `Bool` data must be `true`/`false`. `T` needs all three planes and they must match the shape.

::: warning [THE PAYLOAD IS NOT A SUGGESTION]
If a declared input is missing, the run fails and names the missing streams. If a stream’s kind does not match the declared layout, the run fails. If a shape does not match an operation’s rule, the run fails. The CLI does not fill defaults.
:::

## Repeat

`jotnar repeat` is the host-driven loop. The payload is an iteration list:

```json
{"iterations":[
  {"streams":{"x":{"kind":"I64","shape":[1],"data":["0"]}}},
  {"streams":{"x":{"kind":"I64","shape":[1],"data":["10"]}}}
]}
```

Non-`STATE` inputs are read from each iteration. `STATE` streams seed from iteration 0 and carry forward. `--grow a,b` allows append-only growth along axis 0. This is how a resident pipeline is driven from outside without giving the language a socket.

## Verify

```sh
jotnar verify tests/fixtures/cpu8.asm --json
```

The JSON report includes the resolved layout map, the error list, the coverage fraction, and the DEF interface table. `verify` only replays declared layouts; it does not execute the program. A clean verify is a static claim, not a runtime witness.

::: note [THE TRUST SPINE]
The parity harness is one kind of witness. The trust spine in Appendix A generalizes it: a certificate records the gate, the scope, the falsifiers, the nulls, and the bench labels. `jotnar-lab verify` is the command that checks the schema and the artifact hashes.
:::

::: note [OPS]
`jotnar ops` is the quickest way to answer “does this mnemonic exist?” It prints all 86 names, their arity, and a marker when a target has no reference pattern. If a name is missing, the assembler will reject it. There is no aliasing and no case-insensitive fallback.
:::

## The parity harness

The witness lives in `scripts/`. It needs the Python oracle checkout and `phi-core`:

```sh
export PHI_CORE_DIR=/path/to/phi-core
export JOTNAR_PY=/path/to/jotnar

python3 scripts/parity_ops.py      --bin target/release/jotnar
python3 scripts/parity_programs.py --bin target/release/jotnar
python3 scripts/parity_emit.py     --bin target/release/jotnar
python3 scripts/parity_cuda.py     --bin target/release/jotnar
```

`parity_ops.py` constructs a small listing per mnemonic, runs both implementations, and compares triples by plane, integers exactly, and floats with a tight tolerance. `parity_programs.py` runs the model programs. `parity_emit.py` compares emitted binaries. `parity_cuda.py` needs `nvcc` and a GPU.

::: pullquote [CLI RULE]
The command line is not the language. It is the smallest honest boundary between the graph and the world.
:::

::: end
:::
