//! Listing-level autonomous circuit for Jotnar.
//!
//! The trust-spine loop is:
//!
//! ```text
//! construct a small space -> propose -> resolve exactly -> gate independently
//! -> keep the certificate -> refuse loudly when you cannot
//! ```
//!
//! This crate makes that loop *editable*. The editable surface is the circuit
//! harness: the family set, the bounded step set, the search depth, the
//! refinement policy, and the experience memory. The model/substrate is fixed:
//! candidate programs are ordinary Jotnar listings, resolved by the reference
//! interpreter and certified by `jotnar-lab`.
//!
//! The first slice is deliberately listing-level. It does not port the Python
//! spatial-map layer. A state is an assembled [`jotnar::BoundProgram`] plus the
//! provenance the circuit needs to reason about it.

pub mod archive;
pub mod backbone;
pub mod budget_predictor;
pub mod certificate;
pub mod corpus;
pub mod cost_model;
pub mod demo;
pub mod driver;
pub mod effect_model;
pub mod effects;
pub mod family;
pub mod gate;
pub mod harness;
pub mod intent;
pub mod joint;
pub mod library;
pub mod looped;
pub mod looped_encoder;
pub mod proposer;
pub mod solve;
pub mod state;
pub mod state_effect_model;
pub mod step;
pub mod task_encoder;
pub mod teacher;
pub mod trace;
pub mod view;
pub mod wall_breaking;

pub use archive::{Archive, ArchiveEntry, ArchiveProposer, Insert};
pub use backbone::{Backbone, BackboneKind};
pub use budget_predictor::{BudgetPredictor, BudgetWeights};
pub use certificate::{build_refused, build_solved};
pub use corpus::{
    all_cases, corpus, generated_corpus, generated_split_cases, hard_cases, hard_corpus,
    split_cases, task_specs, CorpusCase, Split, TaskSpec,
};
pub use cost_model::CostModel;
pub use driver::{
    evaluate, load_experience, run_round, run_round_with, run_round_with_memory, CaseOutcome,
    DeterministicProposer, EvalReport, Experience, OutcomeKind, Proposer, RoundReport,
};
pub use effect_model::{EffectModel, WorldModelProposer};
pub use effects::{
    action_effects, reachable_states, state_effects, ActionEffect, CaseEffects, StateEffectRow,
};
pub use family::{families, family, Family};
pub use gate::{bit_exact, structural, Example, GateReport, TestVec};
pub use harness::{Harness, Mutation, STATE_FEATURE_NAMES};
pub use intent::{Case, CaseExpect, InputSpace, Intent, Invariant, OutputSpace};
pub use joint::{fit_joint, JointProposer};
pub use library::{calibration_cases, cases, deep_calibration_cases, example_set, library};
pub use looped::{LoopWeights, LoopedProposer};
pub use looped_encoder::{LoopEncoderWeights, LoopedTaskEncoder};
pub use proposer::{
    action_applies, apply_action, apply_action_set, corpus_prior_weights, JotnarProposer,
    ProposerWeights, ACTIONS, FEATURE_NAMES,
};
pub use solve::{solve, SolveError, SolveResult};
pub use state::{State, StateKind};
pub use state_effect_model::{
    BeamProposer, DiscriminatingEval, ProposalOptions, StateEffectModel, STATE_MODEL_INPUTS,
};
pub use step::{Step, StepRecord};
pub use task_encoder::{
    calibration_dataset, EncoderSample, EncoderWeights, FitReport, TaskEncoder, LATENT_NAMES,
    TASK_FEATURE_NAMES,
};
pub use teacher::{read_teacher_actions, TeacherAction, TeacherActionProposer};
pub use trace::{read_teacher_records, write_round_traces, TeacherRecord, TraceRecord};
pub use view::{case_views, CaseView};
