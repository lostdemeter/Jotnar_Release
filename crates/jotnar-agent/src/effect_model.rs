//! A Jotnar-native action-effect world model.
//!
//! The model is a direct linear map from the 20 task features to one score per
//! bounded action. It is trained on the exact effect table: for each case, the
//! actions that keep or make the case correct should outrank the actions that
//! break it. The listing is ordinary Jotnar; the model is a proposer, not a
//! gate.

use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};

use crate::driver::{EvalReport, Experience, OutcomeKind, Proposer};
use crate::effects::CaseEffects;
use crate::harness::{Harness, Mutation};
use crate::proposer::{action_applies, apply_action, ACTIONS};
use crate::task_encoder::{task_vector, TaskEncoder, LATENT_NAMES, TASK_FEATURE_NAMES};
use crate::view::CaseView;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EffectModel {
    /// ACTIONS.len() x TASK_FEATURE_NAMES.len()
    pub w: Vec<Vec<i64>>,
    pub b: Vec<i64>,
    #[serde(default)]
    pub listing: String,
}

impl EffectModel {
    pub fn prior() -> Self {
        let t = TASK_FEATURE_NAMES.len();
        let d = LATENT_NAMES.len();
        let enc = TaskEncoder::prior();
        let mut w = vec![vec![0i64; t]; ACTIONS.len()];
        let mut b = enc.b_out.clone();
        for action in 0..ACTIONS.len() {
            for latent in 0..d {
                let scale = enc.w_out[action][latent];
                if scale == 0 {
                    continue;
                }
                for (feature, cell) in w[action].iter_mut().enumerate() {
                    *cell += scale * enc.w_enc[latent][feature];
                }
                b[action] += scale * enc.b_enc[latent];
            }
        }
        let mut model = Self {
            w,
            b,
            listing: String::new(),
        };
        model.listing = build_listing(&model);
        model
    }

    pub fn scores(&self, task: &[i64]) -> Vec<i64> {
        self.w
            .iter()
            .enumerate()
            .map(|(action, row)| {
                let mut sum = self.b.get(action).copied().unwrap_or(0);
                for (feature, value) in task.iter().enumerate() {
                    sum += value * row[feature];
                }
                sum
            })
            .collect()
    }

    /// Fit a per-case ranking perceptron on the exact effect table.
    pub fn fit(&mut self, effects: &[CaseEffects], epochs: usize, lr: i64) -> f64 {
        let lr = lr.max(1);
        for _ in 0..epochs.max(1) {
            for case in effects {
                let scores = self.scores(&case.task);
                let positives: Vec<usize> = case
                    .actions
                    .iter()
                    .filter(|action| {
                        action.applicable
                            && action.correct
                            && action_applies(action.index, &case.task)
                    })
                    .map(|action| action.index)
                    .collect();
                let negatives: Vec<usize> = case
                    .actions
                    .iter()
                    .filter(|action| {
                        action.applicable
                            && !action.correct
                            && action_applies(action.index, &case.task)
                    })
                    .map(|action| action.index)
                    .collect();
                if positives.is_empty() || negatives.is_empty() {
                    continue;
                }
                let best_positive = *positives
                    .iter()
                    .max_by_key(|action| scores[**action])
                    .expect("positive");
                let best_negative = *negatives
                    .iter()
                    .max_by_key(|action| scores[**action])
                    .expect("negative");
                if scores[best_negative] >= scores[best_positive] {
                    for (feature, value) in case.task.iter().enumerate() {
                        self.w[best_positive][feature] =
                            (self.w[best_positive][feature] + lr * value).clamp(-128, 128);
                        self.w[best_negative][feature] =
                            (self.w[best_negative][feature] - lr * value).clamp(-128, 128);
                    }
                    self.b[best_positive] = (self.b[best_positive] + lr).clamp(-128, 128);
                    self.b[best_negative] = (self.b[best_negative] - lr).clamp(-128, 128);
                }
            }
        }
        self.listing = build_listing(self);
        let correct = effects
            .iter()
            .filter(|case| {
                let scores = self.scores(&case.task);
                let top = case
                    .actions
                    .iter()
                    .filter(|action| action.applicable && action_applies(action.index, &case.task))
                    .max_by_key(|action| scores[action.index])
                    .map(|action| action.index)
                    .unwrap_or(0);
                case.actions
                    .iter()
                    .find(|action| action.index == top)
                    .map(|action| action.correct)
                    .unwrap_or(false)
            })
            .count();
        correct as f64 / effects.len().max(1) as f64
    }

    pub fn save(&self, path: &std::path::Path) -> Result<(), String> {
        let text = serde_json::to_string_pretty(self).map_err(|e| e.to_string())?;
        std::fs::write(path, text + "\n").map_err(|e| format!("{}: {e}", path.display()))
    }

    pub fn load(path: &std::path::Path) -> Result<Self, String> {
        let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
        let mut model: Self = serde_json::from_str(&text).map_err(|e| e.to_string())?;
        model.normalize();
        Ok(model)
    }

    /// Pad or truncate to the current action space for old artifact files.
    pub fn normalize(&mut self) {
        self.w.truncate(ACTIONS.len());
        while self.w.len() < ACTIONS.len() {
            self.w.push(vec![0; TASK_FEATURE_NAMES.len()]);
        }
        for row in &mut self.w {
            row.resize(TASK_FEATURE_NAMES.len(), 0);
        }
        self.b.truncate(ACTIONS.len());
        while self.b.len() < ACTIONS.len() {
            self.b.push(0);
        }
        self.listing = build_listing(self);
    }
}

fn build_listing(model: &EffectModel) -> String {
    let mut lines = vec![
        "# jotnar-agent action-effect world model: task -> action scores".to_string(),
        "# outputs: s0..sM (one predicted correctness score per action)".to_string(),
    ];
    for (j, name) in TASK_FEATURE_NAMES.iter().enumerate() {
        lines.push(format!("IN t{j}  # {name}"));
    }
    lines.push(String::new());
    for (action, row) in model.w.iter().enumerate() {
        let mut products = Vec::new();
        for (feature, weight) in row.iter().enumerate() {
            if *weight != 0 {
                let name = format!("p{action}_{feature}");
                lines.push(format!("{name} = IMUL(t{feature}, {weight})"));
                products.push(name);
            }
        }
        let bias = model.b.get(action).copied().unwrap_or(0);
        if products.is_empty() {
            let zero = format!("z{action}");
            lines.push(format!("{zero} = IMUL(t0, 0)"));
            lines.push(format!("s{action} = IADD({zero}, {bias})"));
        } else {
            let mut acc = products[0].clone();
            for (k, term) in products.iter().enumerate().skip(1) {
                let name = format!("a{action}_{k}");
                lines.push(format!("{name} = IADD({acc}, {term})"));
                acc = name;
            }
            lines.push(format!("s{action} = IADD({acc}, {bias})"));
        }
    }
    lines.join("\n") + "\n"
}

#[derive(Clone, Debug)]
pub struct WorldModelProposer {
    pub model: EffectModel,
    pub max_variants: usize,
}

impl WorldModelProposer {
    pub fn new(model: EffectModel, max_variants: usize) -> Self {
        Self {
            model,
            max_variants: max_variants.max(1),
        }
    }
}

fn union_harness(base: &Harness, actions: &BTreeSet<usize>) -> Harness {
    let mut harness = base.clone();
    for action in actions {
        if let Some(mutation) = apply_action(&harness, *action) {
            harness = mutation.harness;
        }
    }
    harness
}

impl Proposer for WorldModelProposer {
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
        let mut choices: Vec<(String, Vec<usize>)> = Vec::new();
        for view in views {
            if view.outcome != OutcomeKind::WrongRefusal {
                continue;
            }
            let task = task_vector(view, base);
            let scores = self.model.scores(&task);
            let mut order: Vec<usize> = (0..ACTIONS.len()).collect();
            order.sort_by_key(|action| std::cmp::Reverse(scores[*action]));
            let applicable: Vec<usize> = order
                .into_iter()
                .filter(|action| {
                    apply_action(base, *action).is_some() && action_applies(*action, &task)
                })
                .collect();
            if let Some(top) = applicable.first().copied() {
                selected.insert(top);
                choices.push((view.case_id.clone(), applicable));
            }
        }
        if selected.is_empty() {
            return Some(Vec::new());
        }
        let names = selected
            .iter()
            .map(|action| ACTIONS[*action])
            .collect::<Vec<_>>()
            .join(" + ");
        let mut out = vec![Mutation {
            label: format!("world: {names}"),
            harness: union_harness(base, &selected),
        }];
        for (case, order) in &choices {
            if out.len() > self.max_variants {
                break;
            }
            let Some(second) = order.get(1).copied() else {
                continue;
            };
            let mut alt = selected.clone();
            alt.remove(&order[0]);
            alt.insert(second);
            let alt_names = alt
                .iter()
                .map(|action| ACTIONS[*action])
                .collect::<Vec<_>>()
                .join(" + ");
            out.push(Mutation {
                label: format!("world-alt[{case}]: {alt_names}"),
                harness: union_harness(base, &alt),
            });
        }

        // Composition search: the effect model ranks single actions, but the
        // harness is compositional. Propose every action pair and let the
        // exact gate choose. 45 candidates is small enough to probe.
        for (a, a_name) in ACTIONS.iter().enumerate() {
            for (b, b_name) in ACTIONS.iter().enumerate().skip(a + 1) {
                let mut pair = BTreeSet::new();
                pair.insert(a);
                pair.insert(b);
                let harness = union_harness(base, &pair);
                if harness != *base {
                    out.push(Mutation {
                        label: format!("world-pair: {a_name} + {b_name}"),
                        harness,
                    });
                }
            }
        }
        Some(out)
    }
}
