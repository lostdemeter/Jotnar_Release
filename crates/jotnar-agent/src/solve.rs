//! `specify_or_solve -> plan -> gate -> certificate`, with bounded refinement.
//!
//! The refinement only widens the search (drop tag hints, then depth). It never
//! weakens the invariant.

use std::collections::{BTreeMap, BTreeSet};

use jotnar_lab::model::Certificate;

use crate::certificate::build_solved;
use crate::family;
use crate::gate::{bit_exact, structural, Example, GateReport};
use crate::harness::Harness;
use crate::intent::{Intent, Invariant};
use crate::library::library;
use crate::state::{State, StateKind};
use crate::step::{Step, StepRecord};

#[derive(Clone, Debug)]
pub struct SolveResult {
    pub intent_id: String,
    pub target: State,
    pub state: State,
    pub steps: Vec<StepRecord>,
    pub certificate: Certificate,
    pub refined: bool,
    pub gate: GateReport,
}

#[derive(Clone, Debug)]
pub enum SolveError {
    UnderSpecified {
        message: String,
        target: State,
        examples: Vec<Example>,
    },
    Unsolved {
        message: String,
        target: State,
        examples: Vec<Example>,
        attempts: Vec<String>,
    },
}

impl SolveError {
    pub fn target(&self) -> &State {
        match self {
            SolveError::UnderSpecified { target, .. } | SolveError::Unsolved { target, .. } => {
                target
            }
        }
    }

    pub fn examples(&self) -> &[Example] {
        match self {
            SolveError::UnderSpecified { examples, .. } | SolveError::Unsolved { examples, .. } => {
                examples
            }
        }
    }

    pub fn code(&self) -> &'static str {
        match self {
            SolveError::UnderSpecified { .. } => "under_specified",
            SolveError::Unsolved { .. } => "unsolved",
        }
    }

    pub fn message(&self) -> &str {
        match self {
            SolveError::UnderSpecified { message, .. } | SolveError::Unsolved { message, .. } => {
                message
            }
        }
    }

    pub fn missing(&self) -> Vec<String> {
        match self {
            SolveError::UnderSpecified { message, .. } => vec![message.clone()],
            SolveError::Unsolved { attempts, .. } => attempts.clone(),
        }
    }
}

#[allow(clippy::result_large_err)]
pub fn solve(
    intent: &Intent,
    examples: &[Example],
    harness: &Harness,
) -> Result<SolveResult, SolveError> {
    let lib = library();
    let start = match resolve_start(intent, &lib) {
        Ok(s) => s,
        Err(message) => {
            let target = resolve_target(intent, &lib).unwrap_or_else(|_| {
                State::new("missing-target", "IN A\nOUT = IADD(A, A)\n").unwrap()
            });
            return Err(SolveError::UnderSpecified {
                message,
                target,
                examples: examples.to_vec(),
            });
        }
    };
    let target = match resolve_target(intent, &lib) {
        Ok(t) => t,
        Err(message) => {
            return Err(SolveError::UnderSpecified {
                message,
                target: start.clone(),
                examples: examples.to_vec(),
            })
        }
    };

    let outputs = if intent.output.outputs.is_empty() {
        target.outputs()
    } else {
        intent.output.outputs.clone()
    };
    let mut attempts = Vec::new();
    let mut depth = intent.depth.max(harness.max_depth).max(1);
    let mut tags = if harness.use_tags {
        intent.output.tags.clone()
    } else {
        Vec::new()
    };
    let mut refined = false;

    for round in 0..=harness.max_refinements {
        let mut visited = BTreeSet::new();
        let mut path = Vec::new();
        if let Some((state, steps, gate)) = dfs(
            &start,
            &target,
            examples,
            &outputs,
            intent,
            harness,
            depth,
            &tags,
            &mut visited,
            &mut path,
            &mut attempts,
        ) {
            let recipe = steps
                .iter()
                .map(|s| format!("{}:{}", s.kind, s.name))
                .collect::<Vec<_>>()
                .join(" -> ");
            let certificate = build_solved(intent, &target, examples, &steps, &recipe);
            return Ok(SolveResult {
                intent_id: intent.id.clone(),
                target,
                state,
                steps,
                certificate,
                refined,
                gate,
            });
        }
        if round == harness.max_refinements {
            break;
        }
        if !tags.is_empty() {
            tags.clear();
            attempts.push("refine: dropped tag hints".to_string());
            refined = true;
        } else if depth < 6 {
            depth += 1;
            attempts.push(format!("refine: depth -> {depth}"));
            refined = true;
        } else {
            break;
        }
    }

    Err(SolveError::Unsolved {
        message: format!(
            "no verified path for {} within depth {} ({} attempt(s))",
            intent.id,
            depth,
            attempts.len()
        ),
        target,
        examples: examples.to_vec(),
        attempts,
    })
}

#[allow(clippy::too_many_arguments)]
fn dfs(
    state: &State,
    target: &State,
    examples: &[Example],
    outputs: &[String],
    intent: &Intent,
    harness: &Harness,
    remaining: usize,
    tags: &[String],
    visited: &mut BTreeSet<String>,
    path: &mut Vec<StepRecord>,
    attempts: &mut Vec<String>,
) -> Option<(State, Vec<StepRecord>, GateReport)> {
    if !visited.insert(state.key()) {
        return None;
    }
    if let Some(gate) = goal(state, target, examples, outputs, &intent.invariant) {
        return Some((state.clone(), path.clone(), gate));
    }
    if remaining == 0 {
        return None;
    }

    let mut candidates = candidate_steps(state, target, intent, harness, tags);
    candidates.sort_by_key(|s| s.rank());
    let mut best: Option<(usize, State, Vec<StepRecord>, GateReport)> = None;
    for step in candidates {
        let next = match step.apply(state) {
            Ok(n) => n,
            Err(e) => {
                attempts.push(format!("{}: {e}", step.label()));
                continue;
            }
        };
        // A no-op step is not evidence. Drop it instead of padding the path.
        if next.text == state.text {
            continue;
        }
        if matches!(step, Step::Refine { .. }) {
            let ok = bit_exact(&next, state, examples, outputs);
            if !ok.passed {
                attempts.push(format!("{}: refine changed outputs", step.label()));
                continue;
            }
        }
        path.push(step.record("applied"));
        if let Some((final_state, mut steps, gate)) = dfs(
            &next,
            target,
            examples,
            outputs,
            intent,
            harness,
            remaining - 1,
            tags,
            visited,
            path,
            attempts,
        ) {
            let cost = steps.len();
            if best.as_ref().map(|(c, _, _, _)| cost < *c).unwrap_or(true) {
                best = Some((cost, final_state, std::mem::take(&mut steps), gate));
            }
        }
        path.pop();
    }
    best.map(|(_, s, steps, gate)| (s, steps, gate))
}

fn goal(
    state: &State,
    target: &State,
    examples: &[Example],
    outputs: &[String],
    invariant: &Invariant,
) -> Option<GateReport> {
    match invariant {
        Invariant::BitExact => {
            let report = bit_exact(state, target, examples, outputs);
            report.passed.then_some(report)
        }
        Invariant::Structural => {
            // A structural candidate must still run on the examples; otherwise
            // "same op labels" is not a usable claim.
            let run = bit_exact(state, state, examples, outputs);
            if !run.passed {
                return None;
            }
            let report = structural(state, target);
            report.passed.then_some(report)
        }
    }
}

fn candidate_steps(
    state: &State,
    target: &State,
    intent: &Intent,
    harness: &Harness,
    tags: &[String],
) -> Vec<Step> {
    let mut out = Vec::new();

    // Prefer the start's own family. If that fit does not match the target,
    // fall back to the editable harness family set. The fallback is what lets
    // an explicit `add family ...` action cross families; the gate still has
    // to accept the resulting listing on the examples.
    if let Some(start_family) = &state.family {
        if let Some(fam) = family::family(start_family) {
            if let Some(params) = (fam.fit)(target) {
                out.push(Step::Fit {
                    family: start_family.clone(),
                    params,
                    rank: 0,
                });
            }
        }
    }
    let defaults = Harness::default();
    for name in &harness.families {
        if state.family.as_deref() == Some(name.as_str()) {
            continue;
        }
        // The fallback is for families explicitly added by an action. The
        // default families keep the "fit stays in the start family" rule so
        // the original corpus is not solved by copying a target family.
        if defaults.families.iter().any(|f| f == name) {
            continue;
        }
        let Some(fam) = family::family(name) else {
            continue;
        };
        if !tags.is_empty() && !tags.iter().any(|t| fam.tags.contains(&t.as_str())) {
            continue;
        }
        if let Some(arity) = fam.arity {
            if arity != state.input_names().len() {
                continue;
            }
        }
        if let Some(params) = (fam.fit)(target) {
            out.push(Step::Fit {
                family: name.clone(),
                params,
                rank: 10,
            });
        }
    }

    let n_out = state.outputs().len();
    for consumer in &harness.compose_consumers {
        let applies = match consumer.as_str() {
            "add2" => n_out == 1,
            "sum4" => n_out == 4,
            "sum9" => n_out == 9,
            _ => false,
        };
        if applies {
            out.push(Step::Compose {
                consumer: consumer.clone(),
                rank: 20,
            });
        }
    }

    if harness.refine_enabled {
        out.push(Step::Refine { rank: 30 });
    }

    // A target-label hint that the start already is the target is not a step;
    // the goal check handles identity.
    let _ = target;
    let _ = intent;
    out
}

fn resolve_start(intent: &Intent, lib: &BTreeMap<String, String>) -> Result<State, String> {
    if let Some(text) = &intent.input.program {
        return State::new(
            intent.input.label.clone().unwrap_or_else(|| "start".into()),
            text,
        );
    }
    let Some(label) = &intent.input.label else {
        return Err("input space has no program and no library label".to_string());
    };
    let text = lib
        .get(label)
        .ok_or_else(|| format!("input label {label:?} is not in the library"))?
        .clone();
    let mut state = State::new(label.clone(), text)?;
    if let Some((family, params)) = family_of_label(label) {
        state = state.with_kind(StateKind::Map).with_family(family, params);
    }
    Ok(state)
}

fn resolve_target(intent: &Intent, lib: &BTreeMap<String, String>) -> Result<State, String> {
    if let Some(text) = &intent.output.program {
        return State::new(
            intent
                .output
                .label
                .clone()
                .unwrap_or_else(|| "target".into()),
            text,
        );
    }
    if let Some(label) = &intent.output.label {
        let text = lib
            .get(label)
            .ok_or_else(|| format!("output label {label:?} is not in the library"))?
            .clone();
        let mut state = State::new(label.clone(), text)?;
        if let Some((family, params)) = family_of_label(label) {
            state = state.with_kind(StateKind::Map).with_family(family, params);
        }
        return Ok(state);
    }
    if let Some(name) = &intent.output.family {
        let fam = family::family(name).ok_or_else(|| format!("unknown output family {name:?}"))?;
        let text = (fam.build)(&intent.output.params)?;
        let mut state = State::new(name.clone(), text)?
            .with_kind(StateKind::Fitted)
            .with_family(name.clone(), intent.output.params.clone());
        state.name = name.clone();
        return Ok(state);
    }
    Err("output space has no program, label, or family".to_string())
}

fn family_of_label(label: &str) -> Option<(String, BTreeMap<String, String>)> {
    if let Some(rest) = label.strip_prefix("reduce") {
        if rest.chars().all(|c| c.is_ascii_digit()) && !rest.is_empty() {
            let mut p = BTreeMap::new();
            p.insert("n".to_string(), rest.to_string());
            return Some(("reduce".to_string(), p));
        }
    }
    for name in ["saxpy", "matmul2", "select_max", "select", "add2"] {
        if label == name {
            return Some((name.to_string(), BTreeMap::new()));
        }
    }
    None
}
