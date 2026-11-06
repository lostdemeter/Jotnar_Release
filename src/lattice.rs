//! Rust port of the phi-core lattice substrate.
//!
//! The implementation lives in the standalone `phi-core` crate; this module
//! is the Jotnar-facing re-export so language code keeps using
//! `crate::lattice::*` while the substrate remains reusable.

pub use phi_core::lattice::*;
