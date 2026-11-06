---
title: "Composition: DEF, CALL, IMPORT, REPEAT"
number: "05"
kicker: FIELD NOTE / THE LANGUAGE
lede: A language of graphs still needs a way to name a subgraph, reuse it, and loop over it. Jotnar keeps all three explicit.
columns: 2
---

Composition in Jotnar is textual, not runtime. `DEF` defines a subgraph. `CALL` expands it at the call site, with fresh names for every internal stream. `IMPORT` splices another file. `REPEAT` duplicates a body or emits a runtime loop. None of these features is a function call in the C sense. They are ways to build a larger graph from smaller graphs.

## DEF and CALL

A definition names its formals, its outputs, and a body:

```asm
DEF twice(x) -> (y)
  t = ADD(x, x)
  y = ADD(t, x)
END

IN a
OUT = CALL twice(a)
```

The body is ordinary Jotnar. The formals are inputs to the subgraph. The declared outputs are the names the caller receives. The `END` closes the definition.

At the call site, every internal name is namespaced:

```text
twice#1.t
twice#1.y
```

The suffix `#1` is the call-site number. A second call gets `#2`, and so on. This is how two calls to the same definition can coexist in one flat program without colliding. The namespacing is visible in diagnostics, which is useful when a macro expands into a dozen instructions.

::: note [CALL FORMS]
All of these are accepted:

```asm
OUT = CALL twice(a)          # one output, named
x, y = CALL split(a)         # multiple outputs
x = CALL twice(a)            # one output, assigned
CALL twice(a)                # statement form; outputs are still bound
```
:::

A definition cannot call itself, directly or indirectly. Recursion is a textual expansion loop, not a runtime stack, so the assembler rejects it:

```text
recursive CALL: twice -> twice
```

## Layout contracts on formals

A formal may declare a layout:

```asm
DEF f(x AS F:SEQ) -> (y)
  y = x
END
```

The call site must satisfy it. If the actual argument is `T:SEQ`, the assembler reports a call-site layout violation. This is how a library can document the geometry it expects without relying on the caller’s payload.

::: warning [NO HIDDEN CONVERSIONS]
A layout contract is not a cast. If the formal says `F:SEQ`, the caller must pass an `F:SEQ` stream. `T:SEQ` is not “close enough.” Use `SQRT` or an explicit conversion if that is what you mean.
:::

## IMPORT

`IMPORT path` splices another listing into the current expansion. The path is resolved relative to the listing directory, with a `stdlib/` search path for bare names. `--stdlib DIR` overrides the search path.

```asm
IMPORT math.asm
```

Imports are textual and recursive. Cycles are detected and reported. Imported `IN` declarations stay `UNKNOWN` for layout purposes in the caller, matching the Python frontend. If you need a layout contract, put it on the `DEF` formal.

## REPEAT and BREAK

`REPEAT N` opens a block; `END` closes it. There are two behaviors.

**Static unrolling.** If `N` is an integer literal and the body contains no `BREAK`, the assembler copies the body `N` times. Names are not renamed between copies, so a reassignment in copy 2 replaces the name from copy 1. That is deliberate: it is how a sequence of updates is written.

```asm
IN x
REPEAT 3
  x = IADD(x, 1)
END
OUT = IADD(x, 0)
```

**Runtime loop.** If the body contains a `BREAK`, or if `N` is a stream name rather than a literal, the assembler keeps a loop marker and the interpreter iterates. `BREAK cond` exits the innermost loop when the first element of `cond` is nonzero.

```asm
IN x
IN limit
REPEAT limit
  x = IADD(x, 1)
  BREAK ISUB(x, limit)
END
```

`BREAK` outside a `REPEAT` is an error. `END` without `REPEAT` is an error. A nested loop must be closed before its parent.

::: note [STATIC VS. RUNTIME]
Static unrolling is a compile-time convenience. A runtime loop is a real loop in the interpreter and in emitters that support it. If you need a deterministic instruction count, prefer a literal `REPEAT` without `BREAK`.
:::

## STATE and RANGE

`STATE` marks a stream as carried across host iterations. It must also be declared `IN` (the seed) and assigned as an output (the carry). The CLI’s `jotnar repeat` command threads the state from one iteration to the next.

```asm
IN x
STATE x
REPEAT 3
  x = IADD(x, 1)
END
OUT = IADD(x, 0)
```

`jotnar repeat` expects a payload of the form `{"iterations":[{...}, {...}]}`. Non-state inputs are read from each iteration; state streams seed from iteration 0 and carry forward.

`RANGE name lo hi` is parsed and bounds-checked. It is an annotation for host planners and readers; the reference interpreter does not emit an instruction for it.

## CONFIG

`CONFIG key value` sets a program-wide constant. The registry reads these through typed helpers. The important ones:

| Key | Used by | Default |
|---|---|---|
| `m_acc` | accumulation scale | `31344` |
| `m_cov` | coverage/output scale | `33280` |
| `beta` | `BETA`, `BETA_V5`, attention mask | `0.5` |
| `beta_b` | `TBETA` | falls back to `beta` |
| `eps_rms` / `eps_rms_c` | `RMSNORM` | `4514` counts |
| `eps_ln` / `eps_ln_c` | `LAYERNORM` | `4514` counts |
| `rope_base` | `ROTARY`, `ROPE` | `10000.0` |

`CONFIG` is program-global and must appear outside `DEF`. It is not a runtime knob that changes per invocation; it is part of the program’s arithmetic contract.

## The directive table

| Directive | Scope | Effect |
|---|---|---|
| `IN name [AS layout]` | global | Declares an input stream. |
| `OUT = ...` | any | Names an output stream. |
| `DEF name(args) -> (outs)` | global | Defines a reusable subgraph. |
| `CALL name(args)` | any | Expands a definition at the call site. |
| `END` | block | Closes `DEF` or `REPEAT`. |
| `IMPORT path` | global | Splices another listing. |
| `REPEAT N` | any | Unrolls or loops a body. |
| `BREAK cond` | loop | Exits the innermost loop. |
| `STATE name` | global | Marks a stream as host-carried. |
| `RANGE name lo hi` | global | Annotates a numeric range. |
| `CONFIG key value` | global | Sets a program-wide constant. |
| `# comment` | any | Ignored to end of line. |

::: pullquote [COMPOSITION RULE]
`DEF` names a graph. `CALL` pastes a fresh copy. `REPEAT` copies or loops. Nothing is hidden, and nothing is recursive.
:::

::: end
:::
