//! A Jotnar-native task encoder.
//!
//! Each case becomes a small task vector. The encoder is an ordinary Jotnar
//! listing: task inputs -> a named 8-dim latent -> one score per bounded
//! harness action. The driver selects the top action for each failing case,
//! unions them, and evaluates the union with the exact gate.
//!
//! This is the first case-aware proposer. The latent names are intentionally
//! legible; a later model can replace the selection matrix with a learned or
//! looped encoder without changing the interface.

use std::collections::BTreeSet;

use jotnar::asm::AsmOptions;
use jotnar::payload::Feeds;
use jotnar::runtime::run_text;
use ndarray::{ArrayD, IxDyn};
use serde::{Deserialize, Serialize};

use crate::driver::{evaluate, EvalReport, Experience, OutcomeKind, Proposer};
use crate::harness::{Harness, Mutation};
use crate::intent::Case;
use crate::proposer::{action_applies, apply_action, ACTIONS};
use crate::view::{case_views, CaseView};

pub const TASK_FEATURE_NAMES: [&str; 39] = [
    "start_inputs",
    "start_outputs",
    "start_ops",
    "target_inputs",
    "target_outputs",
    "target_ops",
    "depth_delta",
    "output_delta",
    "input_delta",
    "expect_solve",
    "wrong_refusal",
    "wrong_solve",
    "steps",
    "refined",
    "start_outputs_1",
    "start_outputs_4",
    "target_outputs_1",
    "has_reduce_sum",
    "refinements",
    "tags_count",
    "start_rank",
    "target_rank",
    "start_numel",
    "target_numel",
    "start_is_field",
    "target_is_scalar",
    "shape_match",
    "reduce_applicable",
    "start_imatmul",
    "start_iselect",
    "start_iadd",
    "start_ireduce_sum",
    "target_imatmul",
    "target_iselect",
    "target_iadd",
    "target_ireduce_sum",
    "target_imul",
    "target_select_min",
    "target_select_max",
];

pub const LATENT_NAMES: [&str; 9] = [
    "start_outputs",
    "start_ops",
    "target_ops",
    "depth_delta",
    "wrong_refusal",
    "start_is_four",
    "start_is_one",
    "reduce_applicable",
    "target_outputs_1",
];

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EncoderWeights {
    pub w_enc: Vec<Vec<i64>>,
    pub b_enc: Vec<i64>,
    pub w_out: Vec<Vec<i64>>,
    pub b_out: Vec<i64>,
}

#[derive(Clone, Debug)]
pub struct EncoderSample {
    pub case_id: String,
    pub task: Vec<i64>,
    pub action: usize,
    /// +1 when the action turns a wrong case correct, -1 when it breaks a
    /// correct case, 0 when it changes nothing.
    pub delta: i64,
}

#[derive(Clone, Debug)]
pub struct FitReport {
    pub epochs: usize,
    pub samples: usize,
    pub positive: usize,
    pub negative: usize,
    pub positive_top1: usize,
    pub negative_safe: usize,
}

#[derive(Clone, Debug)]
pub struct TaskEncoder {
    pub w_enc: Vec<Vec<i64>>,
    pub b_enc: Vec<i64>,
    pub w_out: Vec<Vec<i64>>,
    pub b_out: Vec<i64>,
    pub listing: String,
}

pub fn task_vector(view: &CaseView, base: &Harness) -> Vec<i64> {
    vec![
        view.start_inputs as i64,
        view.start_outputs as i64,
        view.start_ops as i64,
        view.target_inputs as i64,
        view.target_outputs as i64,
        view.target_ops as i64,
        view.target_ops as i64 - view.start_ops as i64,
        view.target_outputs as i64 - view.start_outputs as i64,
        view.target_inputs as i64 - view.start_inputs as i64,
        i64::from(view.expect_solve),
        i64::from(view.outcome == OutcomeKind::WrongRefusal),
        i64::from(view.outcome == OutcomeKind::WrongSolve),
        view.steps as i64,
        i64::from(view.refined),
        i64::from(view.start_outputs == 1),
        i64::from(view.start_outputs == 4),
        i64::from(view.target_outputs == 1),
        i64::from(base.compose_consumers.iter().any(|c| c == "reduce_sum")),
        base.max_refinements as i64,
        view.tags.len() as i64,
        view.start_rank as i64,
        view.target_rank as i64,
        view.start_numel as i64,
        view.target_numel as i64,
        i64::from(view.start_is_field),
        i64::from(view.target_is_scalar),
        i64::from(view.shape_match),
        i64::from(view.start_is_field && view.target_is_scalar && view.start_outputs == 1),
        i64::from(view.start_imatmul),
        i64::from(view.start_iselect),
        i64::from(view.start_iadd),
        i64::from(view.start_ireduce_sum),
        i64::from(view.target_imatmul),
        i64::from(view.target_iselect),
        i64::from(view.target_iadd),
        i64::from(view.target_ireduce_sum),
        i64::from(view.target_imul),
        i64::from(view.target_select_min),
        i64::from(view.target_select_max),
    ]
}

fn sum_terms(lines: &mut Vec<String>, prefix: &str, terms: Vec<String>, bias: i64) -> String {
    if terms.is_empty() {
        // A typed integer zero; `IADD(0,0)` would infer a float scalar.
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

fn build_listing(e: &TaskEncoder) -> String {
    let mut lines = vec![
        "# jotnar-agent task encoder: task vector -> named latent -> action scores".to_string(),
    ];
    for (j, name) in TASK_FEATURE_NAMES.iter().enumerate() {
        lines.push(format!("IN t{j}  # {name}"));
    }
    lines.push(String::new());

    let mut latent = Vec::new();
    for (i, row) in e.w_enc.iter().enumerate() {
        let mut terms = Vec::new();
        for (j, w) in row.iter().enumerate() {
            if *w != 0 {
                let p = format!("e_{i}_{j}");
                lines.push(format!("{p} = IMUL(t{j}, {w})"));
                terms.push(p);
            }
        }
        let sum = sum_terms(&mut lines, &format!("enc_{i}"), terms, e.b_enc[i]);
        let name = format!("h{i}");
        // A bare `h = stream` is not an instruction; read the latent through an op.
        lines.push(format!("{name} = IADD({sum}, 0)"));
        latent.push(name);
    }
    lines.push(String::new());

    for (i, row) in e.w_out.iter().enumerate() {
        let mut terms = Vec::new();
        for (j, w) in row.iter().enumerate() {
            if *w != 0 {
                let p = format!("o_{i}_{j}");
                lines.push(format!("{p} = IMUL({}, {w})", latent[j]));
                terms.push(p);
            }
        }
        let sum = sum_terms(&mut lines, &format!("out_{i}"), terms, e.b_out[i]);
        lines.push(format!("s{i} = IADD({sum}, 0)"));
    }
    lines.join("\n") + "\n"
}

impl TaskEncoder {
    pub fn prior() -> Self {
        let t = TASK_FEATURE_NAMES.len();
        let d = LATENT_NAMES.len();
        let mut w_enc = vec![vec![0i64; t]; d];
        // Named selection channels.
        w_enc[0][1] = 1; // start_outputs
        w_enc[1][2] = 1; // start_ops
        w_enc[2][5] = 1; // target_ops
        w_enc[3][6] = 1; // depth_delta
        w_enc[4][10] = 1; // wrong_refusal
        w_enc[5][15] = 1; // start_outputs_4
                          // start_is_one is intentionally zero in the input block; it is
                          // reinjected in the looped encoder and must grow across loops.
        w_enc[7][27] = 1; // reduce_applicable
        w_enc[8][16] = 1; // target_outputs_1

        let mut w_out = vec![vec![0i64; d]; ACTIONS.len()];
        // reduce_sum: a shaped start with a scalar target. The field signal
        // grows with the geometric channel; the stable decoy does not.
        w_out[0] = vec![0, 0, 0, 0, 2, 0, 0, 12, 2];
        // depth: the one-output deep fix. It grows with start_is_one.
        w_out[1] = vec![0, 0, 0, 0, 2, 0, 6, 0, 0];
        // refine off: weak probe.
        w_out[2] = vec![0, 0, 0, 0, 1, 0, 0, 0, 0];
        // tags off: the stable decoy. target_outputs_1 does not grow with the
        // shared block, so both matrix and deep cases prefer it at one loop.
        w_out[3] = vec![0, 0, 0, 0, 1, 0, 0, 0, 10];
        // select_max: weak probe.
        w_out[4] = vec![0, 0, 0, 0, 1, 0, 0, -5, 0];
        // refinements: the one-output deep fix, same signal as depth.
        w_out[5] = vec![0, 0, 0, 0, 2, 0, 6, 0, 0];

        let mut e = Self {
            w_enc,
            b_enc: vec![0; d],
            w_out,
            b_out: vec![0; ACTIONS.len()],
            listing: String::new(),
        };
        e.listing = build_listing(&e);
        e
    }

    pub fn weights(&self) -> EncoderWeights {
        EncoderWeights {
            w_enc: self.w_enc.clone(),
            b_enc: self.b_enc.clone(),
            w_out: self.w_out.clone(),
            b_out: self.b_out.clone(),
        }
    }

    pub fn from_weights(w: EncoderWeights) -> Result<Self, String> {
        let t = TASK_FEATURE_NAMES.len();
        let d = LATENT_NAMES.len();
        if w.w_enc.len() != d
            || w.w_enc.iter().any(|r| r.len() != t)
            || w.b_enc.len() != d
            || w.w_out.len() != ACTIONS.len()
            || w.w_out.iter().any(|r| r.len() != d)
            || w.b_out.len() != ACTIONS.len()
        {
            return Err("task-encoder weights have the wrong shape".to_string());
        }
        let mut e = Self {
            w_enc: w.w_enc,
            b_enc: w.b_enc,
            w_out: w.w_out,
            b_out: w.b_out,
            listing: String::new(),
        };
        e.listing = build_listing(&e);
        Ok(e)
    }

    pub fn save(&self, path: &std::path::Path) -> Result<(), String> {
        let text = serde_json::to_string_pretty(&self.weights()).map_err(|e| e.to_string())?;
        std::fs::write(path, text + "\n").map_err(|e| format!("{}: {e}", path.display()))
    }

    pub fn load(path: &std::path::Path) -> Result<Self, String> {
        let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
        let w: EncoderWeights = serde_json::from_str(&text).map_err(|e| e.to_string())?;
        Self::from_weights(w)
    }

    /// Fit the output selection matrix with a one-step perceptron over the
    /// named latent. The input block (the named channels) stays fixed.
    pub fn fit(
        &mut self,
        samples: &[EncoderSample],
        epochs: usize,
        lr: i64,
    ) -> Result<FitReport, String> {
        use std::collections::{BTreeMap, BTreeSet};
        let mut groups: BTreeMap<String, Vec<&EncoderSample>> = BTreeMap::new();
        for sample in samples {
            groups
                .entry(sample.case_id.clone())
                .or_default()
                .push(sample);
        }

        for _ in 0..epochs {
            let mut changed = 0usize;
            for group in groups.values() {
                let positives: Vec<&&EncoderSample> =
                    group.iter().filter(|s| s.delta > 0).collect();
                if positives.is_empty() {
                    continue;
                }
                let task = &group[0].task;
                let latent = self.latent(task)?;
                let scores = self.scores(task)?;
                let best = positives
                    .iter()
                    .max_by_key(|s| scores[s.action])
                    .map(|s| s.action)
                    .unwrap_or(0);
                let positive_set: BTreeSet<usize> = positives.iter().map(|s| s.action).collect();
                let mut updates = Vec::new();
                for action in 0..scores.len() {
                    if positive_set.contains(&action) {
                        continue;
                    }
                    if scores[action] >= scores[best] {
                        updates.push(action);
                    }
                }
                if updates.is_empty() {
                    continue;
                }
                for top in updates {
                    for (d, h) in latent.iter().enumerate() {
                        self.w_out[best][d] = (self.w_out[best][d] + lr * h).clamp(-128, 128);
                        self.w_out[top][d] = (self.w_out[top][d] - lr * h).clamp(-128, 128);
                    }
                    self.b_out[best] = (self.b_out[best] + lr).clamp(-128, 128);
                    self.b_out[top] = (self.b_out[top] - lr).clamp(-128, 128);
                }
                self.listing = build_listing(self);
                changed += 1;
            }
            if changed == 0 {
                break;
            }
        }
        self.listing = build_listing(self);

        let positive = samples.iter().filter(|s| s.delta > 0).count();
        let negative = samples.iter().filter(|s| s.delta < 0).count();
        let mut positive_top1 = 0;
        let mut negative_safe = 0;
        for sample in samples {
            let scores = self.scores(&sample.task)?;
            let mut order: Vec<usize> = (0..scores.len()).collect();
            order.sort_by_key(|i| std::cmp::Reverse(scores[*i]));
            let top = order.first().copied().unwrap_or(0);
            if sample.delta > 0 && top == sample.action {
                positive_top1 += 1;
            }
            if sample.delta < 0 && top != sample.action {
                negative_safe += 1;
            }
        }
        Ok(FitReport {
            epochs,
            samples: samples.len(),
            positive,
            negative,
            positive_top1,
            negative_safe,
        })
    }

    pub fn latent(&self, task: &[i64]) -> Result<Vec<i64>, String> {
        let got = self.run(task)?;
        (0..LATENT_NAMES.len())
            .map(|i| {
                got.get(&format!("h{i}"))
                    .ok_or_else(|| format!("task encoder missing h{i}"))?
                    .first_i64("latent")
                    .map_err(|e| e.to_string())
            })
            .collect()
    }

    pub fn scores(&self, task: &[i64]) -> Result<Vec<i64>, String> {
        let got = self.run(task)?;
        (0..ACTIONS.len())
            .map(|i| {
                got.get(&format!("s{i}"))
                    .ok_or_else(|| format!("task encoder missing s{i}"))?
                    .first_i64("score")
                    .map_err(|e| e.to_string())
            })
            .collect()
    }

    fn run(&self, task: &[i64]) -> Result<Feeds, String> {
        let mut feeds = Feeds::new();
        for (j, value) in task.iter().enumerate() {
            feeds.insert(
                format!("t{j}"),
                jotnar::Value::I64(ArrayD::from_elem(IxDyn(&[]), *value)),
            );
        }
        run_text(&self.listing, &AsmOptions::new("."), &feeds)
            .map_err(|e| format!("task encoder listing: {e}"))
    }
}

fn case_score(report: &EvalReport, case_id: &str) -> i64 {
    report
        .outcomes
        .iter()
        .find(|o| o.case == case_id)
        .map(|o| i64::from(o.kind.correct()))
        .unwrap_or(0)
}

/// Per-case calibration: for every case and every bounded action, measure
/// whether the action fixes a wrong case, breaks a correct case, or does
/// nothing. The held-out cases are excluded by the caller.
pub fn calibration_dataset(cases: &[Case], base: &Harness) -> Vec<EncoderSample> {
    let base_report = evaluate(base, cases, false);
    let views = case_views(cases, &base_report);
    let mut out = Vec::new();
    for (case, view) in cases.iter().zip(views.iter()) {
        let before = case_score(&base_report, &case.intent.id);
        let task = task_vector(view, base);
        for idx in 0..ACTIONS.len() {
            let Some(m) = apply_action(base, idx) else {
                continue;
            };
            let after_report = evaluate(&m.harness, std::slice::from_ref(case), false);
            let after = case_score(&after_report, &case.intent.id);
            out.push(EncoderSample {
                case_id: case.intent.id.clone(),
                task: task.clone(),
                action: idx,
                delta: after - before,
            });
        }
    }
    out
}

impl Proposer for TaskEncoder {
    fn propose(
        &self,
        base: &Harness,
        _train: &crate::driver::EvalReport,
        _experience: &[Experience],
    ) -> Vec<Mutation> {
        // No per-case views here; keep the deterministic fallback.
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
            let scores = self.scores(&task).ok()?;
            let mut order: Vec<usize> = (0..scores.len()).collect();
            order.sort_by_key(|i| std::cmp::Reverse(scores[*i]));
            for idx in order {
                if scores[idx] <= 0 {
                    break;
                }
                if apply_action(base, idx).is_some() && action_applies(idx, &task) {
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
