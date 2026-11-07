---
title: The Complete Register
number: "11"
kicker: REFERENCE / ALL 90 MNEMONICS
lede: The registry is the authority. Every mnemonic has an arity, a signature, a family, and a reference implementation. If it is not here, it is not in the language.
columns: 1
---

This is the full register. It is generated from the same data the manual uses for the per-family chapters, and it is checked against the Rust registry’s 90 entries.

Read the columns as follows:

- **Mnemonic** — the exact name the assembler binds. Names are uppercase.
- **In→Out** — the arity. A `*` in a signature means “any layout.”
- **Signature** — the layout pattern. `$A` is a unified variable; `*` binds nothing; `KIND:GEOM` is a literal.
- **What it does** — the contract in one line. The source and the parity harness are the long version.

::: warning [THE REGISTER IS NOT A SUGGESTION]
An unknown mnemonic is an error. A wrong arity is an error. A layout conflict is an error. A registered mnemonic with no pattern for a target is a loud `NoPattern`, not a silent fallback. The table is a map, not a permission slip.
:::

::: op-table family="all"
:::

::: end
:::
