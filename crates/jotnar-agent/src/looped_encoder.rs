//! A looped task encoder.
//!
//! The fitted linear task encoder provides the input block (the named latent)
//! and the output block (action scores). A shared recurrent block is applied
//! `N` times with the task vector reinjected every loop. `N` is chosen per
//! case: one loop for correct/refused cases, two for matrix-shaped failures,
//! three for deep failures.
//!
//! This is the Dory shape applied to the task vector, not the aggregate stream.

use std::collections::{BTreeMap, BTreeSet};

use jotnar::asm::AsmOptions;
use jotnar::payload::Feeds;
use jotnar::runtime::run_text;
use ndarray::{ArrayD, IxDyn};
use serde::{Deserialize, Serialize};

use crate::backbone::Backbone;
use crate::budget_predictor::BudgetPredictor;
use crate::driver::{Experience, OutcomeKind, Proposer};
use crate::harness::{Harness, Mutation};
use crate::proposer::{apply_action, ACTIONS};
use crate::task_encoder::{
    task_vector, EncoderWeights, TaskEncoder, LATENT_NAMES, TASK_FEATURE_NAMES,
};
use crate::view::CaseView;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LoopEncoderWeights {
    pub base: EncoderWeights,
    pub w_rec: Vec<Vec<i64>>,
    pub w_in: Vec<Vec<i64>>,
    pub b_rec: Vec<i64>,
    pub scale: i64,
    /// Optional fused budget head: B x T over the task vector.
    #[serde(default)]
    pub budget_w: Vec<Vec<i64>>,
    #[serde(default)]
    pub budget_b: Vec<i64>,
    /// Optional backbone fragment; None uses the named-selection input block.
    #[serde(default)]
    pub backbone: Option<Backbone>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct LoopedEncoderFile {
    max_loops: usize,
    #[serde(default)]
    budgets: BTreeMap<String, usize>,
    weights: LoopEncoderWeights,
}

#[derive(Clone, Debug)]
pub struct LoopedTaskEncoder {
    pub weights: LoopEncoderWeights,
    pub max_loops: usize,
    /// Per-family loop budgets. Empty means use the depth heuristic.
    pub budgets: BTreeMap<String, usize>,
    listings: Vec<String>,
}

fn sum_terms(lines: &mut Vec<String>, prefix: &str, terms: Vec<String>, bias: i64) -> String {
    if terms.is_empty() {
        let zero = format!("{prefix}_z");
        lines.push(format!("{zero} = IMUL(t0, 0)"));
        if bias == 0 {
            return zero;
        }
        let name = format!("{prefix}_c");
        lines.push(format!("{name} = IADD({zero}, {bias})"));
        return name;
    }
    let mut acc = terms[0].clone();
    for (k, term) in terms.iter().enumerate().skip(1) {
        let name = format!("{prefix}_a{k}");
        lines.push(format!("{name} = IADD({acc}, {term})"));
        acc = name;
    }
    if bias != 0 {
        let name = format!("{prefix}_b");
        lines.push(format!("{name} = IADD({acc}, {bias})"));
        acc = name;
    }
    acc
}

fn build_listing(w: &LoopEncoderWeights, loops: usize) -> String {
    let d = LATENT_NAMES.len();
    let t = TASK_FEATURE_NAMES.len();
    let mut lines = Vec::new();
    if let Some(backbone) = &w.backbone {
        for (key, value) in backbone.config() {
            lines.push(format!("CONFIG {key} {value}"));
        }
    }
    lines.push(
        "# jotnar-agent looped task encoder: task -> latent -> N recurrent steps -> scores"
            .to_string(),
    );
    lines.push(format!("# loops={loops}, latent={d}, scale={}", w.scale));
    for (j, name) in TASK_FEATURE_NAMES.iter().enumerate() {
        lines.push(format!("IN t{j}  # {name}"));
    }
    if let Some(backbone) = &w.backbone {
        for decl in backbone.input_decls() {
            lines.push(format!("IN {decl}"));
        }
    }
    lines.push(String::new());

    // Input block: either the named-selection baseline or a backbone fragment.
    let mut h = if let Some(backbone) = &w.backbone {
        backbone.build_h0(&mut lines)
    } else {
        let mut h = Vec::new();
        for i in 0..d {
            let mut terms = Vec::new();
            for (j, weight) in w.base.w_enc[i].iter().enumerate() {
                if *weight != 0 {
                    let p = format!("i_{i}_{j}");
                    lines.push(format!("{p} = IMUL(t{j}, {weight})"));
                    terms.push(p);
                }
            }
            let sum = sum_terms(&mut lines, &format!("in_{i}"), terms, w.base.b_enc[i]);
            let name = format!("h0_{i}");
            lines.push(format!("{name} = IADD({sum}, 0)"));
            h.push(name);
        }
        h
    };
    lines.push(String::new());

    // Shared recurrent block, task reinjected every loop.
    for step in 0..loops {
        let mut next = Vec::new();
        for i in 0..d {
            let mut terms = Vec::new();
            for (j, weight) in w.w_rec[i].iter().enumerate() {
                if *weight != 0 {
                    let p = format!("r_{step}_{i}_{j}");
                    lines.push(format!("{p} = IMUL({}, {weight})", h[j]));
                    terms.push(p);
                }
            }
            for (j, weight) in w.w_in[i].iter().enumerate() {
                if *weight != 0 {
                    let p = format!("x_{step}_{i}_{j}");
                    lines.push(format!("{p} = IMUL(t{j}, {weight})"));
                    terms.push(p);
                }
            }
            let sum = sum_terms(&mut lines, &format!("rec_{step}_{i}"), terms, w.b_rec[i]);
            let delta = format!("d_{step}_{i}");
            lines.push(format!("{delta} = IDIV({sum}, {})", w.scale));
            let residual = format!("rr_{step}_{i}");
            lines.push(format!("{residual} = IADD({}, {delta})", h[i]));
            let relu = format!("hr_{step}_{i}");
            lines.push(format!("{relu} = IMAX({residual}, 0)"));
            let lt = format!("lt_{step}_{i}");
            lines.push(format!("{lt} = ILT({relu}, 64)"));
            let name = format!("h{}_{i}", step + 1);
            lines.push(format!("{name} = ISELECT({lt}, {relu}, 64)"));
            next.push(name);
        }
        h = next;
        lines.push(String::new());
    }

    // Output block: scores = W_out * h_N + b_out.
    for i in 0..ACTIONS.len() {
        let mut terms = Vec::new();
        for (j, weight) in w.base.w_out[i].iter().enumerate() {
            if *weight != 0 {
                let p = format!("o_{i}_{j}");
                lines.push(format!("{p} = IMUL({}, {weight})", h[j]));
                terms.push(p);
            }
        }
        let sum = sum_terms(&mut lines, &format!("out_{i}"), terms, w.base.b_out[i]);
        lines.push(format!("s{i} = IADD({sum}, 0)"));
    }

    // Fused budget head: same program, second output group. It reads the task
    // vector directly, so it is loop-independent; the host can read it at any
    // loop count.
    for i in 0..w.budget_w.len() {
        let mut terms = Vec::new();
        for (j, weight) in w.budget_w[i].iter().enumerate() {
            if *weight != 0 {
                let term = format!("pb_{i}_{j}");
                lines.push(format!("{term} = IMUL(t{j}, {weight})"));
                terms.push(term);
            }
        }
        let sum = sum_terms(
            &mut lines,
            &format!("pout_{i}"),
            terms,
            w.budget_b.get(i).copied().unwrap_or(0),
        );
        lines.push(format!("p{i} = IADD({sum}, 0)"));
    }
    let _ = t;
    lines.join("\n") + "\n"
}

impl LoopedTaskEncoder {
    pub fn from_linear(linear: &TaskEncoder, max_loops: usize) -> Self {
        Self::with_recurrence(linear, max_loops, 4, -2)
    }

    pub fn with_recurrence(linear: &TaskEncoder, max_loops: usize, diag: i64, cross: i64) -> Self {
        let max_loops = max_loops.clamp(1, 6);
        let d = LATENT_NAMES.len();
        let t = TASK_FEATURE_NAMES.len();
        let mut w_rec = vec![vec![0i64; d]; d];
        for (i, row) in w_rec.iter_mut().enumerate() {
            row[i] = diag;
            row[(i + 1) % d] = cross;
        }
        let mut w_in = vec![vec![0i64; t]; d];
        // Reinject the task features that drive the named channels. In
        // particular start_is_four (t15) grows across loops, while
        // target_outputs_1 (t16) stays stable; that is the crossover.
        for (i, row) in w_in.iter_mut().take(5).enumerate() {
            row[i] = 2;
        }
        w_in[5][15] = 4; // start_is_four
        w_in[6][14] = 8; // start_is_one: zero at h0, grows across loops
        w_in[7][17] = 4; // has_sum4
        w_in[8][16] = 4; // target_outputs_1
        Self::from_weights(
            max_loops,
            LoopEncoderWeights {
                base: linear.weights(),
                w_rec,
                w_in,
                b_rec: vec![0; d],
                scale: 8,
                budget_w: Vec::new(),
                budget_b: Vec::new(),
                backbone: None,
            },
        )
    }

    pub fn prior(max_loops: usize) -> Self {
        Self::from_linear(&TaskEncoder::prior(), max_loops)
    }

    pub fn from_weights(max_loops: usize, weights: LoopEncoderWeights) -> Self {
        let max_loops = max_loops.clamp(1, 6);
        let t = TASK_FEATURE_NAMES.len();
        assert!(
            weights.budget_w.is_empty()
                || (weights.budget_w.iter().all(|r| r.len() == t)
                    && weights.budget_b.len() == weights.budget_w.len()),
            "fused budget head has the wrong shape"
        );
        if let Some(backbone) = &weights.backbone {
            backbone
                .validate()
                .expect("looped encoder has a malformed backbone");
        }
        let mut listings = Vec::with_capacity(max_loops);
        for loops in 1..=max_loops {
            listings.push(build_listing(&weights, loops));
        }
        Self {
            weights,
            max_loops,
            budgets: BTreeMap::new(),
            listings,
        }
    }

    pub fn with_budgets(mut self, budgets: BTreeMap<String, usize>) -> Self {
        self.budgets = budgets;
        self
    }

    pub fn with_budget_predictor(mut self, predictor: BudgetPredictor) -> Self {
        self.weights.budget_w = predictor.weights;
        self.weights.budget_b = predictor.bias;
        self.listings = (1..=self.max_loops)
            .map(|loops| build_listing(&self.weights, loops))
            .collect();
        self
    }

    pub fn with_backbone(mut self, backbone: Backbone) -> Self {
        self.weights.backbone = Some(backbone);
        self.listings = (1..=self.max_loops)
            .map(|loops| build_listing(&self.weights, loops))
            .collect();
        self
    }

    pub fn listings_file(&self) -> &String {
        self.listings
            .last()
            .expect("a looped encoder always has at least one listing")
    }

    pub fn save(&self, path: &std::path::Path) -> Result<(), String> {
        let file = LoopedEncoderFile {
            max_loops: self.max_loops,
            budgets: self.budgets.clone(),
            weights: self.weights.clone(),
        };
        let text = serde_json::to_string_pretty(&file).map_err(|e| e.to_string())?;
        std::fs::write(path, text + "\n").map_err(|e| format!("{}: {e}", path.display()))
    }

    pub fn load(path: &std::path::Path) -> Result<Self, String> {
        let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
        let file: LoopedEncoderFile = serde_json::from_str(&text).map_err(|e| e.to_string())?;
        Ok(Self::from_weights(file.max_loops, file.weights).with_budgets(file.budgets))
    }

    pub fn loops_for(&self, view: &CaseView) -> usize {
        if view.outcome != OutcomeKind::WrongRefusal {
            return 1;
        }
        if let Some(budget) = self.budgets.get(&view.family) {
            return (*budget).clamp(1, self.max_loops);
        }
        let depth = view.target_ops as i64 - view.start_ops as i64;
        if depth >= 4 {
            self.max_loops.min(3)
        } else {
            self.max_loops.min(2)
        }
    }

    pub fn loops_for_task(&self, view: &CaseView, task: &[i64]) -> usize {
        if view.outcome != OutcomeKind::WrongRefusal {
            return 1;
        }
        if !self.weights.budget_w.is_empty() {
            // The learned head may raise the budget, but it may not drop
            // below the structural family/depth prior. This keeps the
            // per-case head from spending less compute than the backbone
            // needs after a calibration fit.
            let predicted = self.predict_budget(task).clamp(1, self.max_loops);
            let prior = self.loops_for(view);
            return predicted.max(prior);
        }
        self.loops_for(view)
    }

    /// The fused budget head, read from the same program as the action scores.
    pub fn budget_scores(&self, task: &[i64], loops: usize) -> Result<Vec<i64>, String> {
        let loops = loops.clamp(1, self.max_loops);
        let got = self.run(task, loops)?;
        (0..self.weights.budget_w.len())
            .map(|i| {
                got.get(&format!("p{i}"))
                    .ok_or_else(|| format!("looped task encoder missing p{i}"))?
                    .first_i64("budget")
                    .map_err(|e| e.to_string())
            })
            .collect()
    }

    pub fn predict_budget(&self, task: &[i64]) -> usize {
        match self.budget_scores(task, 1) {
            Ok(scores) if !scores.is_empty() => {
                let mut order: Vec<usize> = (0..scores.len()).collect();
                order.sort_by_key(|i| std::cmp::Reverse(scores[*i]));
                order[0] + 1
            }
            _ => 1,
        }
    }

    /// One run, two output groups: action scores and budget scores.
    pub fn scores_and_budget(
        &self,
        task: &[i64],
        loops: usize,
    ) -> Result<(Vec<i64>, Vec<i64>), String> {
        let loops = loops.clamp(1, self.max_loops);
        let got = self.run(task, loops)?;
        let actions = (0..ACTIONS.len())
            .map(|i| {
                got.get(&format!("s{i}"))
                    .ok_or_else(|| format!("looped task encoder missing s{i}"))?
                    .first_i64("score")
                    .map_err(|e| e.to_string())
            })
            .collect::<Result<Vec<_>, _>>()?;
        let budget = (0..self.weights.budget_w.len())
            .map(|i| {
                got.get(&format!("p{i}"))
                    .ok_or_else(|| format!("looped task encoder missing p{i}"))?
                    .first_i64("budget")
                    .map_err(|e| e.to_string())
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok((actions, budget))
    }

    /// The initial latent emitted by the backbone, before the recurrent block.
    pub fn latents(&self, task: &[i64], loops: usize) -> Result<Vec<i64>, String> {
        let loops = loops.clamp(1, self.max_loops);
        let got = self.run(task, loops)?;
        (0..LATENT_NAMES.len())
            .map(|i| {
                got.get(&format!("h0_{i}"))
                    .ok_or_else(|| format!("looped task encoder missing h0_{i}"))?
                    .first_i64("latent")
                    .map_err(|e| e.to_string())
            })
            .collect()
    }

    pub fn scores(&self, task: &[i64], loops: usize) -> Result<Vec<i64>, String> {
        let loops = loops.clamp(1, self.max_loops);
        let got = self.run(task, loops)?;
        (0..ACTIONS.len())
            .map(|i| {
                got.get(&format!("s{i}"))
                    .ok_or_else(|| format!("looped task encoder missing s{i}"))?
                    .first_i64("score")
                    .map_err(|e| e.to_string())
            })
            .collect()
    }

    pub fn latent(&self, task: &[i64], loops: usize) -> Result<Vec<i64>, String> {
        let loops = loops.clamp(1, self.max_loops);
        let got = self.run(task, loops)?;
        (0..LATENT_NAMES.len())
            .map(|i| {
                got.get(&format!("h{loops}_{i}"))
                    .ok_or_else(|| format!("looped task encoder missing h{loops}_{i}"))?
                    .first_i64("latent")
                    .map_err(|e| e.to_string())
            })
            .collect()
    }

    fn run(&self, task: &[i64], loops: usize) -> Result<Feeds, String> {
        let listing = &self.listings[loops.saturating_sub(1)];
        let mut feeds = Feeds::new();
        for (j, value) in task.iter().enumerate() {
            feeds.insert(
                format!("t{j}"),
                jotnar::Value::I64(ArrayD::from_elem(IxDyn(&[]), *value)),
            );
        }
        if let Some(backbone) = &self.weights.backbone {
            for (name, value) in backbone.feeds(task)? {
                feeds.insert(name, value);
            }
        }
        run_text(listing, &AsmOptions::new("."), &feeds)
            .map_err(|e| format!("looped task encoder listing: {e}"))
    }
}

impl Proposer for LoopedTaskEncoder {
    fn propose(
        &self,
        base: &Harness,
        _train: &crate::driver::EvalReport,
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
            let task = task_vector(view, base);
            let loops = self.loops_for_task(view, &task);
            let scores = self.scores(&task, loops).ok()?;
            let mut order: Vec<usize> = (0..scores.len()).collect();
            order.sort_by_key(|i| std::cmp::Reverse(scores[*i]));
            for idx in order {
                if scores[idx] <= 0 {
                    break;
                }
                if apply_action(base, idx).is_some() {
                    selected.insert(idx);
                    break;
                }
            }
        }
        if selected.is_empty() {
            return Some(Vec::new());
        }
        let mut harness = base.clone();
        for idx in &selected {
            if let Some(m) = apply_action(&harness, *idx) {
                harness = m.harness;
            }
        }
        let names = selected
            .iter()
            .map(|i| ACTIONS[*i])
            .collect::<Vec<_>>()
            .join(" + ");
        Some(vec![Mutation {
            label: format!("case-union: {names}"),
            harness,
        }])
    }
}
