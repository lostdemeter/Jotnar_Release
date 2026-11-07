//! Learned per-case loop budget.
//!
//! The task vector predicts the smallest loop count that covers the case. The
//! predictor is an ordinary Jotnar listing: task inputs -> one score per
//! budget class -> argmax. It is fitted with the same ranking perceptron used
//! by the task encoder.

use jotnar::asm::AsmOptions;
use jotnar::payload::Feeds;
use jotnar::runtime::run_text;
use ndarray::{ArrayD, IxDyn};
use serde::{Deserialize, Serialize};

use crate::task_encoder::TASK_FEATURE_NAMES;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BudgetWeights {
    pub weights: Vec<Vec<i64>>,
    pub bias: Vec<i64>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct BudgetFile {
    max_loops: usize,
    weights: BudgetWeights,
}

#[derive(Clone, Debug)]
pub struct BudgetPredictor {
    pub max_loops: usize,
    pub weights: Vec<Vec<i64>>,
    pub bias: Vec<i64>,
    pub listing: String,
}

fn build_listing(p: &BudgetPredictor) -> String {
    let mut lines = vec![
        "# jotnar-agent budget predictor: task vector -> one score per loop budget".to_string(),
    ];
    for (j, name) in TASK_FEATURE_NAMES.iter().enumerate() {
        lines.push(format!("IN t{j}  # {name}"));
    }
    lines.push(String::new());
    for (i, row) in p.weights.iter().enumerate() {
        let mut terms = Vec::new();
        for (j, w) in row.iter().enumerate() {
            if *w != 0 {
                let term = format!("b_{i}_{j}");
                lines.push(format!("{term} = IMUL(t{j}, {w})"));
                terms.push(term);
            }
        }
        let sum = if terms.is_empty() {
            let z = format!("bz_{i}");
            lines.push(format!("{z} = IMUL(t0, 0)"));
            z
        } else {
            let mut acc = terms[0].clone();
            for (k, term) in terms.iter().enumerate().skip(1) {
                let name = format!("ba_{i}_{k}");
                lines.push(format!("{name} = IADD({acc}, {term})"));
                acc = name;
            }
            acc
        };
        let bias = p.bias.get(i).copied().unwrap_or(0);
        if bias == 0 {
            lines.push(format!("s{i} = IADD({sum}, 0)"));
        } else {
            lines.push(format!("s{i} = IADD({sum}, {bias})"));
        }
    }
    lines.join("\n") + "\n"
}

impl BudgetPredictor {
    pub fn prior(max_loops: usize) -> Self {
        let max_loops = max_loops.clamp(1, 6);
        let t = TASK_FEATURE_NAMES.len();
        let mut weights = vec![vec![0i64; t]; max_loops];
        // budget 1: prefer when there is no refusal.
        weights[0][10] = -2;
        if max_loops >= 2 {
            // budget 2: matrix and deep failures.
            weights[1][10] = 1;
            weights[1][15] = 1; // start_is_four
            weights[1][14] = 1; // start_is_one
        }
        if max_loops >= 3 {
            // budget 3: a fallback class; the fit can move mass here.
            weights[2][10] = 1;
            weights[2][15] = -1;
            weights[2][14] = -1;
        }
        let mut p = Self {
            max_loops,
            weights,
            bias: vec![0; max_loops],
            listing: String::new(),
        };
        p.listing = build_listing(&p);
        p
    }

    pub fn scores(&self, task: &[i64]) -> Result<Vec<i64>, String> {
        let mut feeds = Feeds::new();
        for (j, value) in task.iter().enumerate() {
            feeds.insert(
                format!("t{j}"),
                jotnar::Value::I64(ArrayD::from_elem(IxDyn(&[]), *value)),
            );
        }
        let got = run_text(&self.listing, &AsmOptions::new("."), &feeds)
            .map_err(|e| format!("budget predictor listing: {e}"))?;
        (0..self.max_loops)
            .map(|i| {
                got.get(&format!("s{i}"))
                    .ok_or_else(|| format!("budget predictor missing s{i}"))?
                    .first_i64("budget")
                    .map_err(|e| e.to_string())
            })
            .collect()
    }

    /// Predicted loop budget, in `1..=max_loops`.
    pub fn predict(&self, task: &[i64]) -> usize {
        match self.scores(task) {
            Ok(scores) => {
                let mut order: Vec<usize> = (0..scores.len()).collect();
                order.sort_by_key(|i| std::cmp::Reverse(scores[*i]));
                order.first().copied().unwrap_or(0) + 1
            }
            Err(_) => 1,
        }
    }

    pub fn fit(
        &mut self,
        samples: &[(Vec<i64>, usize)],
        epochs: usize,
        lr: i64,
    ) -> Result<usize, String> {
        for _ in 0..epochs {
            let mut changed = 0usize;
            for (task, label) in samples {
                let label = (*label).clamp(1, self.max_loops) - 1;
                let scores = self.scores(task)?;
                let mut order: Vec<usize> = (0..scores.len()).collect();
                order.sort_by_key(|i| std::cmp::Reverse(scores[*i]));
                let top = order[0];
                if top == label {
                    continue;
                }
                for (j, value) in task.iter().enumerate() {
                    self.weights[label][j] = (self.weights[label][j] + lr * value).clamp(-128, 128);
                    self.weights[top][j] = (self.weights[top][j] - lr * value).clamp(-128, 128);
                }
                self.bias[label] = (self.bias[label] + lr).clamp(-128, 128);
                self.bias[top] = (self.bias[top] - lr).clamp(-128, 128);
                self.listing = build_listing(self);
                changed += 1;
            }
            if changed == 0 {
                break;
            }
        }
        self.listing = build_listing(self);
        Ok(samples
            .iter()
            .filter(|(task, label)| self.predict(task) == *label)
            .count())
    }

    pub fn save(&self, path: &std::path::Path) -> Result<(), String> {
        let file = BudgetFile {
            max_loops: self.max_loops,
            weights: BudgetWeights {
                weights: self.weights.clone(),
                bias: self.bias.clone(),
            },
        };
        let text = serde_json::to_string_pretty(&file).map_err(|e| e.to_string())?;
        std::fs::write(path, text + "\n").map_err(|e| format!("{}: {e}", path.display()))
    }

    pub fn load(path: &std::path::Path) -> Result<Self, String> {
        let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
        let file: BudgetFile = serde_json::from_str(&text).map_err(|e| e.to_string())?;
        let t = TASK_FEATURE_NAMES.len();
        if file.weights.weights.len() != file.max_loops
            || file.weights.weights.iter().any(|r| r.len() != t)
            || file.weights.bias.len() != file.max_loops
        {
            return Err("budget predictor weights have the wrong shape".to_string());
        }
        let mut p = Self {
            max_loops: file.max_loops,
            weights: file.weights.weights,
            bias: file.weights.bias,
            listing: String::new(),
        };
        p.listing = build_listing(&p);
        Ok(p)
    }
}
