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
    /// Geometric summary of the start/target output streams.
    pub start_rank: usize,
    pub target_rank: usize,
    pub start_numel: usize,
    pub target_numel: usize,
    pub start_is_field: bool,
    pub target_is_scalar: bool,
    pub shape_match: bool,
    /// Start-listing op family, so two fields with the same shape but a
    /// different mechanism (matrix vs select vs elementwise) are distinct.
    pub start_imatmul: bool,
    pub start_iselect: bool,
    pub start_iadd: bool,
    pub start_ireduce_sum: bool,
    /// Target-listing op family and operand semantics. This is what lets the
    /// model connect a held-out target mechanism to the family action.
    pub target_imatmul: bool,
    pub target_iselect: bool,
    pub target_iadd: bool,
    pub target_ireduce_sum: bool,
    pub target_imul: bool,
    pub target_select_min: bool,
    pub target_select_max: bool,
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

fn output_shapes(
    label: Option<&str>,
    example: Option<&crate::gate::Example>,
    lib: &BTreeMap<String, String>,
) -> Vec<Vec<usize>> {
    let (Some(label), Some(example)) = (label, example) else {
        return Vec::new();
    };
    let Some(text) = lib.get(label) else {
        return Vec::new();
    };
    let Ok(state) = State::new(label, text.clone()) else {
        return Vec::new();
    };
    crate::gate::run_output_shapes(&state, example, &state.outputs()).unwrap_or_default()
}

fn shape_summary(shapes: &[Vec<usize>]) -> (usize, usize, bool) {
    let rank = shapes.iter().map(|s| s.len()).max().unwrap_or(0);
    let numel = shapes
        .iter()
        .map(|s| s.iter().product::<usize>())
        .max()
        .unwrap_or(0);
    let is_field = shapes.iter().any(|s| !s.is_empty());
    (rank, numel, is_field)
}

fn op_flags(label: Option<&str>, lib: &BTreeMap<String, String>) -> (bool, bool, bool, bool) {
    let Some(label) = label else {
        return (false, false, false, false);
    };
    let Some(text) = lib.get(label) else {
        return (false, false, false, false);
    };
    let Ok(state) = State::new(label, text.clone()) else {
        return (false, false, false, false);
    };
    let ops = state.ops();
    let has = |name: &str| ops.iter().any(|op| op == name);
    (
        has("IMATMUL"),
        has("ISELECT"),
        has("IADD"),
        has("IREDUCE_SUM"),
    )
}

fn target_op_flags(
    label: Option<&str>,
    lib: &BTreeMap<String, String>,
) -> (bool, bool, bool, bool, bool) {
    let Some(label) = label else {
        return (false, false, false, false, false);
    };
    let Some(text) = lib.get(label) else {
        return (false, false, false, false, false);
    };
    let Ok(state) = State::new(label, text.clone()) else {
        return (false, false, false, false, false);
    };
    let ops = state.ops();
    let has = |name: &str| ops.iter().any(|op| op == name);
    (
        has("IMATMUL"),
        has("ISELECT"),
        has("IADD"),
        has("IREDUCE_SUM"),
        has("IMUL"),
    )
}

/// `ISELECT(C, A, B)` is min-like; `ISELECT(C, B, A)` is max-like.
fn select_orientation(label: Option<&str>, lib: &BTreeMap<String, String>) -> (bool, bool) {
    let Some(label) = label else {
        return (false, false);
    };
    let Some(text) = lib.get(label) else {
        return (false, false);
    };
    let Ok(state) = State::new(label, text.clone()) else {
        return (false, false);
    };
    let mut min = false;
    let mut max = false;
    for ins in &state.bound.instrs {
        if ins.mn == "ISELECT" && ins.args.len() == 3 {
            if ins.args[1] == "A" && ins.args[2] == "B" {
                min = true;
            }
            if ins.args[1] == "B" && ins.args[2] == "A" {
                max = true;
            }
        }
    }
    (min, max)
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
            let example = case.examples.first();
            let start_shapes = output_shapes(case.intent.input.label.as_deref(), example, &lib);
            let target_shapes = output_shapes(case.intent.output.label.as_deref(), example, &lib);
            let (start_rank, start_numel, start_is_field) = shape_summary(&start_shapes);
            let (target_rank, target_numel, _) = shape_summary(&target_shapes);
            let target_is_scalar =
                !target_shapes.is_empty() && target_shapes.iter().all(|s| s.is_empty());
            let (start_imatmul, start_iselect, start_iadd, start_ireduce_sum) =
                op_flags(case.intent.input.label.as_deref(), &lib);
            let (target_imatmul, target_iselect, target_iadd, target_ireduce_sum, target_imul) =
                target_op_flags(case.intent.output.label.as_deref(), &lib);
            let (target_select_min, target_select_max) =
                select_orientation(case.intent.output.label.as_deref(), &lib);
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
                start_rank,
                target_rank,
                start_numel,
                target_numel,
                start_is_field,
                target_is_scalar,
                shape_match: start_shapes == target_shapes,
                start_imatmul,
                start_iselect,
                start_iadd,
                start_ireduce_sum,
                target_imatmul,
                target_iselect,
                target_iadd,
                target_ireduce_sum,
                target_imul,
                target_select_min,
                target_select_max,
            }
        })
        .collect()
}
