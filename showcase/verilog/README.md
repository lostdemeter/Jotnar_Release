# Verilog + CPU8

The integer subset emitted as structural Verilog and simulated against the
independent CPU8 ISA model.

```sh
cargo run --release --bin jotnar-verilog -- \
  tests/fixtures/cpu8.asm --config showcase/verilog/cpu8.json --module cpu8
```

`cpu8.json` supplies the state and memory layout. The Rust simulator's
90-cycle CPU8 trace matches the Python `chain.verilog`/`cpu8_model`
reference exactly; see `tests/verilog.rs`.

Proves: the language can describe a sequential machine, and the emitted
structure matches an independent ISA model.
Does not prove: a full RTL toolchain or every opcode.
