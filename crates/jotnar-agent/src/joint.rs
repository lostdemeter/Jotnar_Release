//! A joint latent + causal proposer.
//!
//! The state-effect model answers *what an action does* from the task and the
//! current harness. The backbone answers *what the case is* by mapping the task
//! vector to the latent. `JointProposer` keeps both:
//!
//! ```text
//! score(action) = w_effect * effect(task, state, action)
//!               + w_latent * <latent(task), prototype(action)>
//! ```
//!
//! The prototypes are fitted from the exact transition table: for each action,
//! the mean latent of the cases where that action was a fix. The joint score
//! only orders proposals; the exact gate still chooses.

use serde::{Deserialize, Serialize};

use crate::backbone::Backbone;
use crate::driver::{EvalReport, Experience, Proposer};
use crate::effects::StateEffectRow;
use crate::harness::{Harness, Mutation};
use crate::proposer::ACTIONS;
use crate::state_effect_model::{propose_from_scores, ProposalOptions, StateEffectModel};
use crate::task_encoder::LATENT_NAMES;
use crate::view::CaseView;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct JointProposer {
    pub effect: StateEffectModel,
    pub backbone: Backbone,
    /// ACTIONS x LATENT_NAMES: mean latent of the cases each action fixes.
    pub prototypes: Vec<Vec<i64>>,
    pub effect_weight: i64,
    pub latent_weight: i64,
    pub beam: usize,
    pub max_depth: usize,
    #[serde(default = "default_true")]
    pub include_pairs: bool,
    #[serde(default = "default_true")]
    pub include_complete: bool,
    #[serde(default = "default_true")]
    pub include_singles: bool,
}

fn default_true() -> bool {
    true
}

fn dot(a: &[i64], b: &[i64]) -> i64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn norm(v: &[i64]) -> i64 {
    let sum: i64 = v.iter().map(|x| x * x).sum();
    (sum as f64).sqrt().round() as i64
}

impl JointProposer {
    pub fn new(
        effect: StateEffectModel,
        backbone: Backbone,
        beam: usize,
        max_depth: usize,
    ) -> Self {
        Self {
            effect,
            backbone,
            prototypes: vec![vec![0; LATENT_NAMES.len()]; ACTIONS.len()],
            effect_weight: 4,
            latent_weight: 1,
            beam: beam.max(1),
            max_depth: max_depth.max(1),
            include_pairs: true,
            include_complete: true,
            include_singles: true,
        }
    }

    /// Fit action prototypes from the exact transition table. A row counts as
    /// an action example when the action is a fix (`delta > 0`); if an action
    /// never fixes a training case, its prototype is the mean of the states
    /// where it stays correct.
    pub fn fit_prototypes(&mut self, rows: &[StateEffectRow]) {
        let mut sums = vec![vec![0i64; LATENT_NAMES.len()]; ACTIONS.len()];
        let mut counts = vec![0i64; ACTIONS.len()];
        let mut fallback_sums = vec![vec![0i64; LATENT_NAMES.len()]; ACTIONS.len()];
        let mut fallback_counts = vec![0i64; ACTIONS.len()];
        for row in rows {
            let latent = self.backbone.latent(&row.task);
            let target = if row.delta > 0 {
                &mut sums[row.action]
            } else if row.correct {
                &mut fallback_sums[row.action]
            } else {
                continue;
            };
            for (j, value) in latent.iter().enumerate().take(LATENT_NAMES.len()) {
                target[j] += value;
            }
            if row.delta > 0 {
                counts[row.action] += 1;
            } else {
                fallback_counts[row.action] += 1;
            }
        }
        for action in 0..ACTIONS.len() {
            let (sum, count) = if counts[action] > 0 {
                (&sums[action], counts[action])
            } else {
                (&fallback_sums[action], fallback_counts[action])
            };
            if count > 0 {
                self.prototypes[action] = sum.iter().map(|value| value / count).collect();
            }
        }
    }

    pub fn scores(&self, task: &[i64], state: &[i64]) -> Vec<i64> {
        let effect = self.effect.scores(task, state);
        let latent = self.backbone.latent(task);
        (0..ACTIONS.len())
            .map(|action| {
                let prototype = &self.prototypes[action];
                let similarity = dot(&latent, prototype) / norm(prototype).max(1);
                self.effect_weight * effect[action] + self.latent_weight * similarity
            })
            .collect()
    }

    /// The latent is the identity channel; expose it for traces.
    pub fn latent(&self, task: &[i64]) -> Vec<i64> {
        self.backbone.latent(task)
    }

    pub fn save(&self, path: &std::path::Path) -> Result<(), String> {
        let text = serde_json::to_string_pretty(self).map_err(|e| e.to_string())?;
        std::fs::write(path, text + "\n").map_err(|e| format!("{}: {e}", path.display()))
    }

    pub fn load(path: &std::path::Path) -> Result<Self, String> {
        let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
        let mut joint: Self = serde_json::from_str(&text).map_err(|e| e.to_string())?;
        joint.effect.normalize();
        joint.prototypes.truncate(ACTIONS.len());
        while joint.prototypes.len() < ACTIONS.len() {
            joint.prototypes.push(vec![0; LATENT_NAMES.len()]);
        }
        Ok(joint)
    }
}

impl Proposer for JointProposer {
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
        Some(propose_from_scores(
            base,
            views,
            self.beam,
            self.max_depth,
            ProposalOptions {
                include_pairs: self.include_pairs,
                include_complete: self.include_complete,
                include_singles: self.include_singles,
            },
            "joint",
            |task, state| self.scores(task, state),
        ))
    }
}

/// Build the joint prototype table from the training transitions. The caller
/// has already loaded the state-effect model and the backbone.
pub fn fit_joint(
    effect: StateEffectModel,
    backbone: Backbone,
    rows: &[StateEffectRow],
    beam: usize,
    max_depth: usize,
) -> JointProposer {
    let mut joint = JointProposer::new(effect, backbone, beam, max_depth);
    joint.fit_prototypes(rows);
    joint
}
