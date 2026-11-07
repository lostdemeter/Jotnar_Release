//! Per-case task views for a case-aware proposer.
//!
//! The aggregate `EvalReport` answers "how many failed". A `CaseView` answers
//! "which task failed, and what shape does it have". The task encoder consumes
//! these views; the exact gate still decides.

use std::collections::BTreeMap;

use crate::driver::{EvalReport, OutcomeKind};
use crate::intent::{Case, CaseExpect};
use crate::library::library;
use crate::state::State;

#[derive(Clone, Debug)]
pub struct CaseView {
    pub case_id: String,
    pub family: String,
    pub expect_solve: bool,
    pub outcome: OutcomeKind,
    pub start_inputs: usize,
    pub start_outputs: usize,
    pub start_ops: usize,
    pub target_inputs: usize,
    pub target_outputs: usize,
    pub target_ops: usize,
    pub steps: usize,
    pub refined: bool,
    pub tags: Vec<String>,
}

fn meta(label: Option<&str>, lib: &BTreeMap<String, String>) -> (usize, usize, usize) {
    let Some(label) = label else {
        return (0, 0, 0);
    };
    let Some(text) = lib.get(label) else {
        return (0, 0, 0);
    };
    State::new(label, text.clone())
        .map(|s| (s.input_names().len(), s.outputs().len(), s.ops().len()))
        .unwrap_or((0, 0, 0))
}

pub fn case_views(cases: &[Case], report: &EvalReport) -> Vec<CaseView> {
    let lib = library();
    cases
        .iter()
        .map(|case| {
            let outcome = report.outcomes.iter().find(|o| o.case == case.intent.id);
            let kind = outcome.map(|o| o.kind).unwrap_or(OutcomeKind::WrongRefusal);
            let (si, so, sp) = meta(case.intent.input.label.as_deref(), &lib);
            let (ti, to, tp) = meta(case.intent.output.label.as_deref(), &lib);
            let start_outputs = if so == 0 {
                outcome.map(|o| o.start_outputs).unwrap_or(0)
            } else {
                so
            };
            let start_ops = if sp == 0 {
                outcome.map(|o| o.start_ops).unwrap_or(0)
            } else {
                sp
            };
            let target_outputs = if to == 0 {
                outcome.map(|o| o.target_outputs).unwrap_or(0)
            } else {
                to
            };
            let target_ops = if tp == 0 {
                outcome.map(|o| o.target_ops).unwrap_or(0)
            } else {
                tp
            };
            CaseView {
                case_id: case.intent.id.clone(),
                family: case.family.clone(),
                expect_solve: case.expect == CaseExpect::Solve,
                outcome: kind,
                start_inputs: si,
                start_outputs,
                start_ops,
                target_inputs: ti,
                target_outputs,
                target_ops,
                steps: outcome.map(|o| o.steps).unwrap_or(0),
                refined: outcome.map(|o| o.refined).unwrap_or(false),
                tags: case.intent.output.tags.clone(),
            }
        })
        .collect()
}
