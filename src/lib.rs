//! Jotnar — geometric assembly for typed streams.
//!
//! This crate implements **Jotnar-LAN/DIG**: the assembly language, its
//! reference runtime (Jotnar-WIT/DIG for the digital interpreter), and the
//! digital target emitters (Jotnar-MAC/DIG). The Python implementation is the
//! oracle: the Rust side follows the same fail-loud doctrine — unknown
//! mnemonics, arity mismatches, use-before-def, layout conflicts, and shape
//! mismatches are errors, never silent guesses.
//!
//! The phi lattice substrate lives in the `phi-core` workspace crate and is
//! re-exported through [`lattice`] and [`Triple`].
//!
//! Layout of the crate:
//!
//! * [`error`]   — the single error type.
//! * [`value`]   — the runtime value model (F/T/I/U8 streams).
//! * [`layout`]  — `KIND:GEOM` layouts and signature unification.
//! * [`ir`]      — parsed/bound program representation.
//! * [`asm`]     — IMPORT/DEF/CALL expansion, parsing, assembling, verify.
//! * [`lattice`] — the integer lattice substrate (encode/decode/fixed point).
//! * [`ops`]     — the mnemonic registry and reference semantics.
//! * [`runtime`] — the interpreter (`run`/`run_text`/`repeat`).
//! * [`payload`] — portable JSON payloads for the CLI and differential tests.

pub mod asm;
pub mod builder;
pub mod codegen;
pub mod error;
pub mod ir;
pub mod lattice;
pub mod layout;
pub mod ops;
pub mod payload;
pub mod profile;
pub mod protocol;
pub mod runtime;
pub mod rv32;
pub mod showcase;
pub mod value;
pub mod verilog;

pub use error::{Error, Result};
pub use ir::{BoundProgram, Config, InDecl, Instr, Program};
pub use phi_core::t2;
pub use runtime::run_text;
pub use value::{Triple, Value};
