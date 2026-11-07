//! The first Jotnar-native proposer: a tiny linear scoring unit.
//!
//! The proposer is an ordinary Jotnar listing. Its inputs are evidence features;
//! its outputs are one integer score per bounded harness action. The driver
//! reads the top applicable scores and probes those actions. The exact gate is
//! still the acceptance; the proposer only chooses what to try.
//!
//! The weights are a prior, not a trained model. The next step is to fit them
//! from `experience.jsonl` or distill them from a larger Jotnar model.

use std::collections::{BTreeMap, BTreeSet};

use jotnar::asm::AsmOptions;
use jotnar::payload::Feeds;
use jotnar::runtime::run_text;
use ndarray::{ArrayD, IxDyn};
use serde::{Deserialize, Serialize};

use crate::driver::{EvalReport, Experience, OutcomeKind, Proposer, RoundReport};
use crate::harness::{Harness, Mutation};

pub const FEATURE_NAMES: [&str; 21] = [
    "wrong_refusal",
    "wrong_refusal_4out",
    "wrong_solve",
    "correct",
    "steps",
    "has_sum4",
    "depth",
    "refine",
    "families",
    "experience",
    "has_select_max",
    "tags",
    "depth_cap",
    "wrong_refusal_deep",
    "refinements",
    "wrong_refusal_elementwise",
    "wrong_refusal_reduction",
    "wrong_refusal_matrix",
    "wrong_refusal_select",
    "wrong_refusal_refine",
    "wrong_refusal_control",
];

pub const ACTIONS: [&str; 10] = [
    "add compose[sum4]",
    "depth + 1",
    "refine off",
    "tags off",
    "add family select_max",
    "refinements + 1",
    "add family select_min",
    "add family saxpy_dup",
    "add family affine_dup",
    "add family matmul2_dup",
];

/// A corpus-shaped prior. The 24-case stream has three matrix failures and one
/// deep elementwise failure; the ordinary 5-case prior would let the matrix
/// decoy crowd out the deep fix. This prior uses the per-family counts.
pub fn corpus_prior_weights() -> Vec<Vec<i64>> {
    let mut w = prior_weights();
    // refinements row: elementwise is the deep-family signal; the family count
    // is amplified by the recurrent block.
    w[5] = vec![
        0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 20, 0, 0, 0, 0, 0,
    ];
    // tags off: a weak decoy that reads the matrix count but stays below the
    // two capability actions.
    w[3] = vec![
        1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
    ];
    w
}

pub fn prior_weights() -> Vec<Vec<i64>> {
    vec![
        // sum4: a matrix-family failure is the signal; not useful once the
        // consumer is already present.
        vec![
            2, 0, 0, 0, 0, -20, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 10, 0, 0, 0,
        ],
        // depth: useful when there are wrong refusals, bounded by the cap.
        vec![
            3, 0, 1, 0, 0, 0, -2, 0, 0, 0, 0, 0, -10, 0, 0, 2, 0, 0, 0, 0, 0,
        ],
        // refine off: a weak probe.
        vec![
            0, 0, 2, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        ],
        // tags off: a decoy at one loop; it reads the refusal counts, but
        // solves neither failure.
        vec![
            2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 0, 0, 0,
        ],
        // select_max: a weak probe.
        vec![
            1, 0, 0, 0, 0, 0, 0, 0, 0, 0, -5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        ],
        // refinements: the deep-path fix. It reads the family count, which the
        // looped block amplifies; at one loop it is below the decoy.
        vec![
            0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        ],
        // select_min, saxpy_dup, affine_dup, matmul2_dup: generated-family
        // actions. The prior is zero; the generated fit fills them.
        vec![0; 21],
        vec![0; 21],
        vec![0; 21],
        vec![0; 21],
    ]
}

pub fn features(base: &Harness, train: &EvalReport, experience: &[Experience]) -> Vec<i64> {
    let wrong_refusal_4out = train
        .outcomes
        .iter()
        .filter(|o| o.kind == OutcomeKind::WrongRefusal && o.start_outputs == 4)
        .count() as i64;
    let wrong_refusal_deep = train
        .outcomes
        .iter()
        .filter(|o| {
            o.kind == OutcomeKind::WrongRefusal
                && o.start_outputs == 1
                && o.target_ops > o.start_ops
        })
        .count() as i64;
    vec![
        train.wrong_refusal as i64,
        wrong_refusal_4out,
        train.wrong_solve as i64,
        train.correct() as i64,
        train.total_steps as i64,
        i64::from(base.compose_consumers.iter().any(|c| c == "sum4")),
        base.max_depth as i64,
        i64::from(base.refine_enabled),
        base.families.len() as i64,
        experience.len() as i64,
        i64::from(base.families.iter().any(|f| f == "select_max")),
        i64::from(base.use_tags),
        i64::from(base.max_depth >= 4),
        wrong_refusal_deep,
        base.max_refinements as i64,
        train.wrong_refusal_in("elementwise") as i64,
        train.wrong_refusal_in("reduction") as i64,
        train.wrong_refusal_in("matrix") as i64,
        train.wrong_refusal_in("select") as i64,
        train.wrong_refusal_in("refine") as i64,
        train.wrong_refusal_in("control") as i64,
    ]
}

/// Build the scoring listing. It is intentionally plain Jotnar: no hidden host
/// arithmetic, no special op. The driver runs it through the reference runtime.
pub fn build_listing(weights: &[Vec<i64>], bias: &[i64]) -> String {
    let mut lines = vec![
        "# jotnar-agent proposer: linear scores over evidence features".to_string(),
        "# inputs: f0..fN; outputs: s0..sM (one score per bounded action)".to_string(),
    ];
    for (j, name) in FEATURE_NAMES.iter().enumerate() {
        lines.push(format!("IN f{j}  # {name}"));
    }
    lines.push(String::new());
    for (i, row) in weights.iter().enumerate() {
        let mut products = Vec::new();
        for (j, w) in row.iter().enumerate() {
            if *w != 0 {
                let p = format!("p{i}_{j}");
                lines.push(format!("{p} = IMUL(f{j}, {w})"));
                products.push(p);
            }
        }
        let b = bias.get(i).copied().unwrap_or(0);
        if products.is_empty() {
            let z = format!("z{i}");
            lines.push(format!("{z} = IMUL(f0, 0)"));
            lines.push(format!("s{i} = IADD({z}, {b})"));
        } else {
            let mut acc = products[0].clone();
            for (k, p) in products.iter().enumerate().skip(1) {
                let next = format!("a{i}_{k}");
                lines.push(format!("{next} = IADD({acc}, {p})"));
                acc = next;
            }
            lines.push(format!("s{i} = IADD({acc}, {b})"));
        }
    }
    lines.join("\n") + "\n"
}

pub fn score_listing(listing: &str, features: &[i64]) -> Result<Vec<i64>, String> {
    let mut feeds = Feeds::new();
    for (j, value) in features.iter().enumerate() {
        let arr = ArrayD::from_elem(IxDyn(&[]), *value);
        feeds.insert(format!("f{j}"), jotnar::Value::I64(arr));
    }
    let got = run_text(listing, &AsmOptions::new("."), &feeds)
        .map_err(|e| format!("proposer listing: {e}"))?;
    let mut out = Vec::new();
    for i in 0..ACTIONS.len() {
        let name = format!("s{i}");
        let value = got
            .get(&name)
            .ok_or_else(|| format!("proposer listing missing {name}"))?;
        out.push(value.first_i64(&name).map_err(|e| e.to_string())?);
    }
    Ok(out)
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ProposerWeights {
    pub weights: Vec<Vec<i64>>,
    pub bias: Vec<i64>,
}

#[derive(Clone, Debug)]
pub struct JotnarProposer {
    pub weights: Vec<Vec<i64>>,
    pub bias: Vec<i64>,
    pub listing: String,
}

impl Default for JotnarProposer {
    fn default() -> Self {
        Self::prior()
    }
}

impl JotnarProposer {
    pub fn prior() -> Self {
        Self::from_rows(prior_weights())
    }

    pub fn corpus_prior() -> Self {
        Self::from_rows(corpus_prior_weights())
    }

    fn from_rows(weights: Vec<Vec<i64>>) -> Self {
        let bias = vec![0; ACTIONS.len()];
        let listing = build_listing(&weights, &bias);
        Self {
            weights,
            bias,
            listing,
        }
    }

    pub fn scores(&self, features: &[i64]) -> Result<Vec<i64>, String> {
        score_listing(&self.listing, features)
    }

    pub fn weights(&self) -> ProposerWeights {
        ProposerWeights {
            weights: self.weights.clone(),
            bias: self.bias.clone(),
        }
    }

    pub fn from_weights(w: ProposerWeights) -> Result<Self, String> {
        if w.weights.len() != ACTIONS.len()
            || w.weights.iter().any(|row| row.len() != FEATURE_NAMES.len())
            || w.bias.len() != ACTIONS.len()
        {
            return Err("proposer weights have the wrong shape".to_string());
        }
        let listing = build_listing(&w.weights, &w.bias);
        Ok(Self {
            weights: w.weights,
            bias: w.bias,
            listing,
        })
    }

    pub fn save(&self, path: &std::path::Path) -> Result<(), String> {
        let text = serde_json::to_string_pretty(&self.weights()).map_err(|e| e.to_string())?;
        std::fs::write(path, text + "\n").map_err(|e| format!("{}: {e}", path.display()))
    }

    pub fn load(path: &std::path::Path) -> Result<Self, String> {
        let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
        let w: ProposerWeights = serde_json::from_str(&text).map_err(|e| e.to_string())?;
        Self::from_weights(w)
    }

    /// One online perceptron step over a calibration round. The driver has
    /// already measured every bounded action; this only nudges the prior.
    pub fn fit_from_report(
        &mut self,
        base: &Harness,
        report: &RoundReport,
        experience: &[Experience],
        lr: i64,
    ) -> Vec<String> {
        let f = features(base, &report.base, experience);
        let mut changes = Vec::new();
        for cand in &report.candidates {
            if cand.index == 0 {
                continue; // incumbent, not an action
            }
            let Some(action) = action_index(&cand.label) else {
                continue;
            };
            let delta = cand.train.correct() as i64 - report.base.correct() as i64;
            if delta == 0 {
                continue;
            }
            for (j, fj) in f.iter().enumerate() {
                self.weights[action][j] =
                    (self.weights[action][j] + lr * delta * fj).clamp(-64, 64);
            }
            self.bias[action] = (self.bias[action] + lr * delta).clamp(-64, 64);
            changes.push(format!("{}: delta {delta:+}", ACTIONS[action]));
        }
        self.listing = build_listing(&self.weights, &self.bias);
        changes
    }
}

pub fn action_index(label: &str) -> Option<usize> {
    ACTIONS.iter().position(|a| *a == label)
}

impl Proposer for JotnarProposer {
    fn propose(
        &self,
        base: &Harness,
        train: &EvalReport,
        experience: &[Experience],
    ) -> Vec<Mutation> {
        let f = features(base, train, experience);
        let scores = match self.scores(&f) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("jotnar proposer: {e}; falling back to bounded order");
                return base.mutations();
            }
        };
        select_from_scores(base, &scores)
    }
}

/// The shared read-out: rank actions by score, keep the top two applicable.
pub(crate) fn select_from_scores(base: &Harness, scores: &[i64]) -> Vec<Mutation> {
    let mut order: Vec<usize> = (0..ACTIONS.len()).collect();
    order.sort_by_key(|i| std::cmp::Reverse(scores.get(*i).copied().unwrap_or(i64::MIN)));

    let mut out = Vec::new();
    for idx in order {
        if let Some(m) = apply_action(base, idx) {
            out.push(m);
            if out.len() >= 2 {
                break;
            }
        }
    }
    if out.is_empty() {
        base.mutations()
    } else {
        out
    }
}

pub fn apply_action(base: &Harness, idx: usize) -> Option<Mutation> {
    let mut h = base.clone();
    let label = match idx {
        0 => {
            if h.compose_consumers.iter().any(|c| c == "sum4") {
                return None;
            }
            h.compose_consumers.push("sum4".into());
            "add compose[sum4]"
        }
        1 => {
            if h.max_depth >= 6 {
                return None;
            }
            h.max_depth += 1;
            "depth + 1"
        }
        2 => {
            if !h.refine_enabled {
                return None;
            }
            h.refine_enabled = false;
            "refine off"
        }
        3 => {
            if !h.use_tags {
                return None;
            }
            h.use_tags = false;
            "tags off"
        }
        4 => {
            if h.families.iter().any(|f| f == "select_max") {
                return None;
            }
            h.families.push("select_max".into());
            "add family select_max"
        }
        5 => {
            if h.max_refinements >= 3 {
                return None;
            }
            h.max_refinements += 1;
            "refinements + 1"
        }
        6 => {
            if h.families.iter().any(|f| f == "select_min") {
                return None;
            }
            h.families.push("select_min".into());
            "add family select_min"
        }
        7 => {
            if h.families.iter().any(|f| f == "saxpy_dup") {
                return None;
            }
            h.families.push("saxpy_dup".into());
            "add family saxpy_dup"
        }
        8 => {
            if h.families.iter().any(|f| f == "affine_dup") {
                return None;
            }
            h.families.push("affine_dup".into());
            "add family affine_dup"
        }
        9 => {
            if h.families.iter().any(|f| f == "matmul2_dup") {
                return None;
            }
            h.families.push("matmul2_dup".into());
            "add family matmul2_dup"
        }
        _ => return None,
    };
    Some(Mutation {
        label: label.to_string(),
        harness: h,
    })
}

pub fn apply_action_set(base: &Harness, actions: &BTreeSet<usize>) -> Harness {
    let mut harness = base.clone();
    for action in actions {
        if let Some(mutation) = apply_action(&harness, *action) {
            harness = mutation.harness;
        }
    }
    harness
}

/// A compact trace the CLI can write beside the listing.
pub fn trace(
    proposer: &JotnarProposer,
    base: &Harness,
    train: &EvalReport,
    experience: &[Experience],
) -> BTreeMap<String, serde_json::Value> {
    let f = features(base, train, experience);
    let scores = proposer.scores(&f).unwrap_or_default();
    let mut out = BTreeMap::new();
    out.insert("features".into(), serde_json::json!(f));
    out.insert("scores".into(), serde_json::json!(scores));
    out.insert("feature_names".into(), serde_json::json!(FEATURE_NAMES));
    out.insert("actions".into(), serde_json::json!(ACTIONS));
    out
}
