//! Typed intent for the listing-level circuit.
//!
//! An intent says what is given and what is wanted. Missing data is resolved
//! from the library or refused; it is never guessed silently.

use std::collections::BTreeMap;

use crate::gate::Example;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Invariant {
    /// The candidate must produce the same outputs as the target on every
    /// supplied example.
    BitExact,
    /// The candidate must preserve the structural signature (inputs, outputs,
    /// op multiset) and the interpreter must still run it.
    Structural,
}

impl Invariant {
    pub fn name(&self) -> &'static str {
        match self {
            Invariant::BitExact => "bit_exact_outputs",
            Invariant::Structural => "degree_sequence_and_labels",
        }
    }

    pub fn level(&self) -> &'static str {
        match self {
            Invariant::BitExact => "semantic",
            Invariant::Structural => "structural",
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct InputSpace {
    /// A named state in the library.
    pub label: Option<String>,
    /// A literal listing.
    pub program: Option<String>,
    /// Optional input-name contract.
    pub names: Vec<String>,
}

#[derive(Clone, Debug, Default)]
pub struct OutputSpace {
    /// A named target in the library.
    pub label: Option<String>,
    /// A literal target listing.
    pub program: Option<String>,
    /// A family to instantiate from `params`.
    pub family: Option<String>,
    pub params: BTreeMap<String, String>,
    /// Tags are hints, not invariants.
    pub tags: Vec<String>,
    /// The stream names the gate compares. Empty means the target's outputs.
    pub outputs: Vec<String>,
}

#[derive(Clone, Debug)]
pub struct Intent {
    pub id: String,
    pub input: InputSpace,
    pub output: OutputSpace,
    pub invariant: Invariant,
    pub depth: usize,
}

impl Intent {
    pub fn new(id: impl Into<String>, input: InputSpace, output: OutputSpace) -> Self {
        Self {
            id: id.into(),
            input,
            output,
            invariant: Invariant::BitExact,
            depth: 2,
        }
    }

    pub fn with_invariant(mut self, invariant: Invariant) -> Self {
        self.invariant = invariant;
        self
    }

    pub fn with_depth(mut self, depth: usize) -> Self {
        self.depth = depth;
        self
    }

    pub fn with_tags(mut self, tags: &[&str]) -> Self {
        self.output.tags = tags.iter().map(|s| s.to_string()).collect();
        self
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CaseExpect {
    Solve,
    Refuse,
}

/// One task in the circuit stream: an intent, its visible examples, a frozen
/// held-out variant, and the expected outcome.
#[derive(Clone, Debug)]
pub struct Case {
    pub intent: Intent,
    pub examples: Vec<Example>,
    pub heldout: Vec<Example>,
    pub expect: CaseExpect,
    /// Capability family: elementwise, reduction, matrix, select, refine, control.
    pub family: String,
    /// Number of outputs on the start state. This is what a compose consumer
    /// actually binds; it is not the target's output count.
    pub start_outputs: usize,
    /// Op count on the start state.
    pub start_ops: usize,
}
