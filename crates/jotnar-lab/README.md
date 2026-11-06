# jotnar-lab

Trust-spine harness: certificate schema, gate/outcome separation, falsifiers,
nulls, benches, and verification.

It is an appendix/tool, not a language feature.

```sh
cargo run -p jotnar-lab -- --help
```

First slice: `certify doom`, `falsify doom`, `null doom`, `bench doom`,
`verify certificates/...`.

`verify` checks artifact hashes by default; `--no-hashes` skips only the
artifact check. Missing artifacts produce `BLOCKED`, never `PASS`.
