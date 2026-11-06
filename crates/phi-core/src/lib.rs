//! phi-core (Rust): the shared geometric lattice substrate.
//!
//! This is a Rust port of the `phi_core` Python package. It is deliberately
//! independent of the Jotnar frontend: Jotnar-LAN programs consume these
//! primitives through `jotnar::lattice` / `jotnar::ops`.
//!
//! Layers:
//! * [`lattice`] — encode/decode, fixed-point bridge, frozen LUTs.
//! * [`triple`]  — the `(s,e,z)` stream value.
//! * [`ops`]     — exact compare/clip/bin helpers.
//! * [`numpy_ops`] — numpy-style substrate ops (matmul, norms, softmax, …).

pub mod calibrate;
pub mod error;
pub mod lattice;
pub mod numpy_ops;
pub mod ops;
pub mod score;
pub mod triple;

pub use calibrate::m_of;
pub use error::{Error, Result};
pub use score::psnr;
pub use triple::Triple;
