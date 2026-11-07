//! A Jotnar-native cost head over the exact transition table.
//!
//! `CostModel` predicts the bounded steps a child state costs, from the same
//! `(task, harness_state)` features as the state-effect model. It is fitted by
//! ridge regression on the transition rows whose child is correct, and it is
//! an ordinary listing with an `IDIV` scale.
//!
//! The planner uses it as a soft tie-break: `effect - lambda * cost`. The exact
//! gate and the driver's actual-step tie-break remain the judges.

use serde::{Deserialize, Serialize};

use crate::backbone::solve_linear;
use crate::effects::StateEffectRow;
use crate::harness::STATE_FEATURE_NAMES;
use crate::proposer::ACTIONS;
use crate::state_effect_model::STATE_MODEL_INPUTS;
use crate::task_encoder::TASK_FEATURE_NAMES;

/// Fixed-point scale for the listing. Predictions are `sum / SCALE`.
pub const COST_SCALE: i64 = 16;
const FEATURES: usize = STATE_MODEL_INPUTS;
const WITH_BIAS: usize = FEATURES + 1;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CostModel {
    /// ACTIONS.len() x FEATURES, scaled by COST_SCALE.
    pub w: Vec<Vec<i64>>,
    /// ACTIONS.len(), scaled by COST_SCALE.
    pub b: Vec<i64>,
    #[serde(default)]
    pub listing: String,
}

impl CostModel {
    pub fn prior() -> Self {
        let mut model = Self {
            w: vec![vec![0; FEATURES]; ACTIONS.len()],
            b: vec![COST_SCALE; ACTIONS.len()],
            listing: String::new(),
        };
        model.listing = build_listing(&model);
        model
    }

    pub fn features(task: &[i64], state: &[i64]) -> Vec<i64> {
        let mut out = Vec::with_capacity(FEATURES);
        out.extend_from_slice(task);
        out.extend_from_slice(state);
        out
    }

    /// Predicted bounded steps, clamped to a sane range.
    pub fn scores(&self, task: &[i64], state: &[i64]) -> Vec<i64> {
        let f = Self::features(task, state);
        self.w
            .iter()
            .enumerate()
            .map(|(action, row)| {
                let mut sum = self.b.get(action).copied().unwrap_or(COST_SCALE);
                for (j, value) in f.iter().enumerate() {
                    sum += value * row[j];
                }
                (sum / COST_SCALE).clamp(0, 8)
            })
            .collect()
    }

    /// Ridge fit on correct children. Returns the mean absolute error.
    pub fn fit(&mut self, rows: &[StateEffectRow], lambda: f64) -> f64 {
        let lambda = lambda.max(1e-6);
        for action in 0..ACTIONS.len() {
            let selected: Vec<&StateEffectRow> = rows
                .iter()
                .filter(|row| row.action == action && row.correct)
                .collect();
            if selected.is_empty() {
                continue;
            }
            let mut a = vec![vec![0.0f64; WITH_BIAS]; WITH_BIAS];
            let mut y = vec![vec![0.0f64; 1]; WITH_BIAS];
            for row in &selected {
                let mut f = Self::features(&row.task, &row.state)
                    .into_iter()
                    .map(|value| value as f64)
                    .collect::<Vec<_>>();
                f.push(1.0);
                let target = row.steps as f64;
                for (i, f_i) in f.iter().enumerate() {
                    y[i][0] += f_i * target;
                    for (j, f_j) in f.iter().enumerate() {
                        a[i][j] += f_i * f_j;
                    }
                }
            }
            for (i, row) in a.iter_mut().enumerate() {
                row[i] += lambda;
            }
            let solved = solve_linear(a, y);
            for (j, weight) in self.w[action].iter_mut().enumerate() {
                *weight = (solved[j][0] * COST_SCALE as f64).round() as i64;
            }
            self.b[action] = (solved[FEATURES][0] * COST_SCALE as f64).round() as i64;
        }
        self.listing = build_listing(self);

        let mut error = 0.0;
        let mut count = 0usize;
        for row in rows.iter().filter(|row| row.correct) {
            let predicted = self.scores(&row.task, &row.state)[row.action];
            error += (predicted - row.steps as i64).abs() as f64;
            count += 1;
        }
        if count == 0 {
            0.0
        } else {
            error / count as f64
        }
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
            self.w.push(vec![0; FEATURES]);
        }
        self.b.truncate(ACTIONS.len());
        while self.b.len() < ACTIONS.len() {
            self.b.push(COST_SCALE);
        }
        self.listing = build_listing(self);
    }
}

fn build_listing(model: &CostModel) -> String {
    let mut lines = vec![
        "# jotnar-agent cost head: (task, harness state) -> predicted steps".to_string(),
        format!("# outputs: s0..sK, each = IDIV(linear, {COST_SCALE})"),
    ];
    for (j, name) in TASK_FEATURE_NAMES.iter().enumerate() {
        lines.push(format!("IN t{j}  # {name}"));
    }
    for (j, name) in STATE_FEATURE_NAMES.iter().enumerate() {
        lines.push(format!("IN u{j}  # {name}"));
    }
    lines.push(String::new());
    let names: Vec<String> = (0..TASK_FEATURE_NAMES.len())
        .map(|j| format!("t{j}"))
        .chain((0..STATE_FEATURE_NAMES.len()).map(|j| format!("u{j}")))
        .collect();
    for (action, row) in model.w.iter().enumerate() {
        let mut products = Vec::new();
        for (j, weight) in row.iter().enumerate() {
            if *weight != 0 {
                let name = format!("p{action}_{j}");
                lines.push(format!("{name} = IMUL({}, {weight})", names[j]));
                products.push(name);
            }
        }
        let bias = model.b.get(action).copied().unwrap_or(0);
        let sum = if products.is_empty() {
            let zero = format!("z{action}");
            lines.push(format!("{zero} = IMUL(t0, 0)"));
            let name = format!("c{action}_b");
            lines.push(format!("{name} = IADD({zero}, {bias})"));
            name
        } else {
            let mut acc = products[0].clone();
            for (k, term) in products.iter().enumerate().skip(1) {
                let name = format!("c{action}_{k}");
                lines.push(format!("{name} = IADD({acc}, {term})"));
                acc = name;
            }
            if bias != 0 {
                let name = format!("c{action}_b");
                lines.push(format!("{name} = IADD({acc}, {bias})"));
                acc = name;
            }
            acc
        };
        lines.push(format!("s{action} = IDIV({sum}, {COST_SCALE})"));
    }
    lines.join("\n") + "\n"
}
