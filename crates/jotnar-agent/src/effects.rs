//! Per-case action effects.
//!
//! For each case and each bounded harness action, record what the exact gate
//! did: solved, refused, wrong, or not applicable. This is the feedback the
//! teacher needs to choose an action, and the data the student needs to learn
//! action effects instead of imitating blind guesses.

use serde::{Deserialize, Serialize};

use crate::corpus::{corpus, generated_corpus, hard_corpus};
use crate::driver::{evaluate, CaseOutcome};
use crate::harness::Harness;
use crate::intent::Case;
use crate::proposer::{apply_action, ACTIONS};
use crate::task_encoder::task_vector;
use crate::view::case_views;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ActionEffect {
    pub index: usize,
    pub label: String,
    pub applicable: bool,
    pub outcome: String,
    pub correct: bool,
    pub wrong: bool,
    pub steps: usize,
    pub note: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CaseEffects {
    pub case: String,
    pub task: Vec<i64>,
    pub split: String,
    pub start: String,
    pub target: String,
    pub base_outcome: String,
    pub base_correct: bool,
    pub actions: Vec<ActionEffect>,
}

fn split_map() -> std::collections::BTreeMap<String, String> {
    let mut map = std::collections::BTreeMap::new();
    for c in corpus() {
        map.insert(c.case.intent.id.clone(), c.split.as_str().to_string());
    }
    for c in generated_corpus() {
        map.insert(c.case.intent.id.clone(), c.split.as_str().to_string());
    }
    for c in hard_corpus() {
        map.insert(c.case.intent.id.clone(), c.split.as_str().to_string());
    }
    map
}

fn outcome_of(case: &Case, harness: &Harness) -> Option<CaseOutcome> {
    evaluate(harness, std::slice::from_ref(case), false)
        .outcomes
        .into_iter()
        .find(|o| o.case == case.intent.id)
}

pub fn action_effects(cases: &[Case], base: &Harness) -> Vec<CaseEffects> {
    let base_eval = evaluate(base, cases, false);
    let views = case_views(cases, &base_eval);
    let splits = split_map();
    let mut out = Vec::new();
    for view in views {
        let Some(case) = cases.iter().find(|c| c.intent.id == view.case_id) else {
            continue;
        };
        let Some(base_outcome) = base_eval.outcomes.iter().find(|o| o.case == view.case_id) else {
            continue;
        };
        let mut actions = Vec::new();
        for (index, action_label) in ACTIONS.iter().enumerate() {
            let Some(mutation) = apply_action(base, index) else {
                actions.push(ActionEffect {
                    index,
                    label: (*action_label).to_string(),
                    applicable: false,
                    outcome: "not-applicable".to_string(),
                    correct: false,
                    wrong: false,
                    steps: 0,
                    note: "action does not apply to the base harness".to_string(),
                });
                continue;
            };
            let Some(outcome) = outcome_of(case, &mutation.harness) else {
                continue;
            };
            actions.push(ActionEffect {
                index,
                label: mutation.label,
                applicable: true,
                outcome: outcome.kind.as_str().to_string(),
                correct: outcome.kind.correct(),
                wrong: !outcome.kind.correct(),
                steps: outcome.steps,
                note: outcome.note.clone(),
            });
        }
        out.push(CaseEffects {
            case: view.case_id.clone(),
            task: task_vector(&view, base),
            split: splits
                .get(&view.case_id)
                .cloned()
                .unwrap_or_else(|| "stream".to_string()),
            start: case.intent.input.label.clone().unwrap_or_default(),
            target: case.intent.output.label.clone().unwrap_or_default(),
            base_outcome: base_outcome.kind.as_str().to_string(),
            base_correct: base_outcome.kind.correct(),
            actions,
        });
    }
    out
}

/// One transition-table row: from a harness state, applying one action.
///
/// Unlike [`ActionEffect`], the parent harness is part of the row. The exact
/// gate measured the child harness, so the row is a causal label
/// `(task, state, action) -> outcome`.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StateEffectRow {
    pub case: String,
    pub split: String,
    pub task: Vec<i64>,
    pub state_key: String,
    pub state: Vec<i64>,
    pub action: usize,
    pub label: String,
    pub parent_outcome: String,
    pub outcome: String,
    pub parent_correct: bool,
    pub correct: bool,
    pub wrong: bool,
    /// `child_correct - parent_correct`: +1 fix, 0 no change, -1 break.
    pub delta: i64,
    pub steps: usize,
}

/// Every reachable harness state up to `max_depth` **steps**.
///
/// Unlike the old distinct-subset enumeration, this is a bounded BFS over
/// action sequences. Idempotent actions stop naturally because `apply_action`
/// returns `None` once the family/consumer is present; `depth + 1` and
/// `refinements + 1` can repeat up to their caps. States are keyed by
/// [`Harness::fingerprint`], so different orders that reach the same harness
/// collapse to one node. The first (shortest) path is kept for reporting.
pub fn reachable_states(base: &Harness, max_depth: usize) -> Vec<(String, Vec<usize>, Harness)> {
    use std::collections::VecDeque;

    let mut seen = std::collections::BTreeSet::new();
    let mut out = Vec::new();
    let base_key = base.fingerprint();
    seen.insert(base_key.clone());
    out.push((base_key, Vec::new(), base.clone()));
    let mut queue = VecDeque::new();
    queue.push_back((base.clone(), Vec::new(), 0usize));

    while let Some((harness, path, steps)) = queue.pop_front() {
        if steps >= max_depth {
            continue;
        }
        for action in 0..ACTIONS.len() {
            let Some(mutation) = apply_action(&harness, action) else {
                continue;
            };
            let key = mutation.harness.fingerprint();
            if seen.insert(key.clone()) {
                let mut next_path = path.clone();
                next_path.push(action);
                out.push((key, next_path.clone(), mutation.harness.clone()));
                queue.push_back((mutation.harness, next_path, steps + 1));
            }
        }
    }
    out
}

/// Build the state-dependent transition table with the exact gate.
///
/// For each reachable harness state and each case, the child outcome of every
/// applicable action is recorded. This is the data for
/// `fit-state-effect-model`; it is never used to grade a candidate.
pub fn state_effects(cases: &[Case], base: &Harness, max_depth: usize) -> Vec<StateEffectRow> {
    let base_eval = evaluate(base, cases, false);
    let views = case_views(cases, &base_eval);
    let tasks: std::collections::BTreeMap<String, Vec<i64>> = views
        .iter()
        .map(|view| (view.case_id.clone(), task_vector(view, base)))
        .collect();
    let splits = split_map();
    let states = reachable_states(base, max_depth);

    // Measure every state once per case.
    let mut outcomes: std::collections::BTreeMap<(String, String), CaseOutcome> =
        std::collections::BTreeMap::new();
    let mut harnesses: std::collections::BTreeMap<String, Harness> =
        std::collections::BTreeMap::new();
    for (key, _subset, harness) in &states {
        harnesses.insert(key.clone(), harness.clone());
        for outcome in evaluate(harness, cases, false).outcomes {
            outcomes.insert((key.clone(), outcome.case.clone()), outcome);
        }
    }

    let mut rows = Vec::new();
    for case in cases {
        let case_id = &case.intent.id;
        let Some(task) = tasks.get(case_id).cloned() else {
            continue;
        };
        let split = splits
            .get(case_id)
            .cloned()
            .unwrap_or_else(|| "stream".to_string());
        for (key, _path, harness) in &states {
            let Some(parent) = outcomes.get(&(key.clone(), case_id.clone())) else {
                continue;
            };
            for (action, action_label) in ACTIONS.iter().enumerate() {
                let Some(mutation) = apply_action(harness, action) else {
                    continue;
                };
                let child_key = mutation.harness.fingerprint();
                if !harnesses.contains_key(&child_key) {
                    continue;
                }
                let Some(child) = outcomes.get(&(child_key, case_id.clone())) else {
                    continue;
                };
                rows.push(StateEffectRow {
                    case: case_id.clone(),
                    split: split.clone(),
                    task: task.clone(),
                    state_key: key.clone(),
                    state: harness.state_features(),
                    action,
                    label: (*action_label).to_string(),
                    parent_outcome: parent.kind.as_str().to_string(),
                    outcome: child.kind.as_str().to_string(),
                    parent_correct: parent.kind.correct(),
                    correct: child.kind.correct(),
                    wrong: !child.kind.correct(),
                    delta: i64::from(child.kind.correct()) - i64::from(parent.kind.correct()),
                    steps: child.steps,
                });
            }
        }
    }
    rows
}
