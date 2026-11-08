//! A teacher-action proposer.
//!
//! The teacher (Dory, Qwen, or a human) writes one action per case. The driver
//! applies those actions and the exact gate decides whether the union is
//! adopted. This is behavior-level distillation: no latent contract needed.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

use crate::driver::{EvalReport, Experience, OutcomeKind, Proposer};
use crate::harness::{Harness, Mutation};
use crate::proposer::{apply_action, ACTIONS};
use crate::view::CaseView;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TeacherAction {
    #[serde(default)]
    pub case: String,
    #[serde(default)]
    pub task: Vec<i64>,
    pub action: usize,
    #[serde(default)]
    pub split: String,
    #[serde(default)]
    pub source: String,
}

pub fn read_teacher_actions(path: &std::path::Path) -> Result<BTreeMap<String, usize>, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut map = BTreeMap::new();
    for line in text.lines().filter(|line| !line.trim().is_empty()) {
        let record: TeacherAction =
            serde_json::from_str(line).map_err(|e| format!("teacher action: {e}"))?;
        if record.action >= ACTIONS.len() {
            return Err(format!(
                "teacher action {} out of range for {}",
                record.action, record.case
            ));
        }
        if !record.case.is_empty() {
            map.insert(record.case, record.action);
        }
    }
    Ok(map)
}

#[derive(Clone, Debug)]
pub struct TeacherActionProposer {
    pub actions: BTreeMap<String, usize>,
}

impl TeacherActionProposer {
    pub fn load(path: &std::path::Path) -> Result<Self, String> {
        Ok(Self {
            actions: read_teacher_actions(path)?,
        })
    }
}

impl Proposer for TeacherActionProposer {
    fn propose(
        &self,
        base: &Harness,
        _train: &EvalReport,
        _experience: &[Experience],
    ) -> Vec<Mutation> {
        base.mutations()
    }

    fn propose_cases(
        &self,
        views: &[CaseView],
        base: &Harness,
        _experience: &[Experience],
    ) -> Option<Vec<Mutation>> {
        let mut selected = BTreeSet::new();
        for view in views {
            if view.outcome != OutcomeKind::WrongRefusal {
                continue;
            }
            if let Some(action) = self.actions.get(&view.case_id) {
                if apply_action(base, *action).is_some() {
                    selected.insert(*action);
                }
            }
        }
        if selected.is_empty() {
            return Some(Vec::new());
        }
        let mut harness = base.clone();
        for action in &selected {
            if let Some(mutation) = apply_action(&harness, *action) {
                harness = mutation.harness;
            }
        }
        let names = selected
            .iter()
            .map(|action| ACTIONS[*action])
            .collect::<Vec<_>>()
            .join(" + ");
        Some(vec![Mutation {
            label: format!("teacher: {names}"),
            harness,
        }])
    }
}
