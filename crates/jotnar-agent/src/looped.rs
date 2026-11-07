//! A tiny looped latent-recurrent proposer, in the spirit of Dory.
//!
//! ```text
//! features -> [input block] -> h0
//! d        = (Wr h_t + Wf features + br) / scale
//! h_{t+1}  = clip(relu(h_t + d), 0, 64)          # shared, applied N times
//! scores   = [output block] h_N
//! ```
//!
//! The recurrent block is shared across loops, the input is reinjected every
//! loop, and `num_loops` is the elastic compute knob. The residual + ReLU +
//! clip is the smallest honest stand-in for Dory's normalized recurrent block:
//! without a nonlinearity, a looped linear model is only a reparameterized
//! linear model.
//!
//! This is not a transformer yet. It is the three-block structure with a
//! nonlinear shared recurrence, small enough to inspect and gate.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::driver::{EvalReport, Experience, Proposer, RoundReport};
use crate::harness::{Harness, Mutation};
use crate::proposer::{
    action_index, features, prior_weights, score_listing, select_from_scores, ACTIONS,
    FEATURE_NAMES,
};

pub const LATENT_DIM: usize = FEATURE_NAMES.len();
pub const SCALE: i64 = 8;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LoopWeights {
    /// Input block: D x F.
    pub win: Vec<Vec<i64>>,
    /// Shared recurrent block: D x D.
    pub wr: Vec<Vec<i64>>,
    /// Input reinjection: D x F.
    pub wf: Vec<Vec<i64>>,
    /// Output block: A x D.
    pub wout: Vec<Vec<i64>>,
    pub bin: Vec<i64>,
    pub br: Vec<i64>,
    pub bout: Vec<i64>,
    pub scale: i64,
}

#[derive(Clone, Debug)]
pub struct LoopedProposer {
    pub loops: usize,
    pub weights: LoopWeights,
    pub listing: String,
}

impl LoopedProposer {
    pub fn prior(loops: usize) -> Self {
        Self::from_weights(loops, prior_weights_struct())
    }

    pub fn corpus_prior(loops: usize) -> Self {
        let mut w = prior_weights_struct();
        w.wout = crate::proposer::corpus_prior_weights();
        w.bout = vec![0; ACTIONS.len()];
        Self::from_weights(loops, w)
    }

    /// Build a looped proposer with an explicit recurrent diagonal and
    /// cross-lane coefficient. This is the two-parameter recurrent fit surface.
    pub fn with_recurrence(loops: usize, diag: i64, cross: i64) -> Self {
        Self::from_weights(loops, weights_with_recurrence(diag, cross))
    }

    pub fn recurrence(&self) -> (i64, i64) {
        let d = LATENT_DIM;
        (self.weights.wr[0][0], self.weights.wr[0][1 % d])
    }

    /// Use the fitted linear model as the output block, and keep the
    /// constructed input/recurrent blocks. This is the bridge from the simple
    /// model to the looped model.
    pub fn from_linear(linear: &crate::proposer::JotnarProposer, loops: usize) -> Self {
        let mut w = prior_weights_struct();
        w.wout = linear.weights.clone();
        w.bout = linear.bias.clone();
        Self::from_weights(loops, w)
    }

    pub fn from_weights(loops: usize, weights: LoopWeights) -> Self {
        let loops = loops.clamp(1, 8);
        let listing = build_listing(&weights, loops);
        Self {
            loops,
            weights,
            listing,
        }
    }

    pub fn scores(&self, features: &[i64]) -> Result<Vec<i64>, String> {
        score_listing(&self.listing, features)
    }

    /// The final latent vector, computed in Rust with the same integer
    /// arithmetic as the listing. Used by fitting and by inspection; the
    /// listing remains the runtime.
    pub fn latent(&self, features: &[i64]) -> Vec<i64> {
        let d = LATENT_DIM;
        let scale = self.weights.scale.max(1);
        let mut h = vec![0i64; d];
        for (i, hi) in h.iter_mut().enumerate() {
            let mut sum = self.weights.bin[i];
            for (j, f) in features.iter().enumerate() {
                sum += f * self.weights.win[i][j];
            }
            *hi = sum / scale;
        }
        for _ in 0..self.loops {
            let mut next = vec![0i64; d];
            for (i, ni) in next.iter_mut().enumerate() {
                let mut sum = self.weights.br[i];
                for (j, hv) in h.iter().enumerate() {
                    sum += hv * self.weights.wr[i][j];
                }
                for (j, f) in features.iter().enumerate() {
                    sum += f * self.weights.wf[i][j];
                }
                let delta = sum / scale;
                *ni = (h[i] + delta).clamp(0, 64);
            }
            h = next;
        }
        h
    }

    pub fn save(&self, path: &std::path::Path) -> Result<(), String> {
        let text = serde_json::to_string_pretty(&self.weights).map_err(|e| e.to_string())?;
        std::fs::write(path, text + "\n").map_err(|e| format!("{}: {e}", path.display()))
    }

    pub fn load(loops: usize, path: &std::path::Path) -> Result<Self, String> {
        let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
        let w: LoopWeights = serde_json::from_str(&text).map_err(|e| e.to_string())?;
        Ok(Self::from_weights(loops, w))
    }

    /// Fit the output block from a calibration round, using the looped latent
    /// vector rather than the raw features. The recurrent block is unchanged;
    /// fitting it needs a richer per-case dataset and is the next step.
    pub fn fit_from_report(
        &mut self,
        base: &Harness,
        report: &RoundReport,
        experience: &[Experience],
        lr: i64,
    ) -> Vec<String> {
        let f = features(base, &report.base, experience);
        let h = self.latent(&f);
        let mut changes = Vec::new();
        for cand in &report.candidates {
            if cand.index == 0 {
                continue;
            }
            let Some(action) = action_index(&cand.label) else {
                continue;
            };
            let delta = cand.train.correct() as i64 - report.base.correct() as i64;
            if delta == 0 {
                continue;
            }
            for (d, hv) in h.iter().enumerate() {
                self.weights.wout[action][d] =
                    (self.weights.wout[action][d] + lr * delta * hv).clamp(-128, 128);
            }
            self.weights.bout[action] = (self.weights.bout[action] + lr * delta).clamp(-128, 128);
            changes.push(format!("{}: delta {delta:+}", ACTIONS[action]));
        }
        self.listing = build_listing(&self.weights, self.loops);
        changes
    }

    pub fn trace(
        &self,
        base: &Harness,
        train: &EvalReport,
        experience: &[Experience],
    ) -> BTreeMap<String, serde_json::Value> {
        let f = features(base, train, experience);
        let scores = self.scores(&f).unwrap_or_default();
        let mut out = BTreeMap::new();
        out.insert("loops".into(), serde_json::json!(self.loops));
        out.insert("features".into(), serde_json::json!(f));
        out.insert("scores".into(), serde_json::json!(scores));
        out.insert("feature_names".into(), serde_json::json!(FEATURE_NAMES));
        out.insert("actions".into(), serde_json::json!(ACTIONS));
        out
    }
}

impl Proposer for LoopedProposer {
    fn propose(
        &self,
        base: &Harness,
        train: &EvalReport,
        experience: &[Experience],
    ) -> Vec<Mutation> {
        let f = features(base, train, experience);
        match self.scores(&f) {
            Ok(scores) => select_from_scores(base, &scores),
            Err(e) => {
                eprintln!("looped proposer: {e}; falling back to bounded order");
                base.mutations()
            }
        }
    }
}

fn prior_weights_struct() -> LoopWeights {
    weights_with_recurrence(SCALE / 2, -(SCALE / 4))
}

fn weights_with_recurrence(diag: i64, cross: i64) -> LoopWeights {
    let d = LATENT_DIM;
    let f = FEATURE_NAMES.len();
    let mut win = vec![vec![0; f]; d];
    for (i, row) in win.iter_mut().enumerate() {
        row[i] = SCALE; // input block is the identity, scaled
    }
    let mut wf = vec![vec![0; f]; d];
    for (i, row) in wf.iter_mut().enumerate() {
        row[i] = SCALE / 4; // reinject each feature into its own latent lane
    }
    let mut wr = vec![vec![0; d]; d];
    for (i, row) in wr.iter_mut().enumerate() {
        row[i] = diag;
        row[(i + 1) % d] = cross;
    }
    let wout = prior_weights();
    LoopWeights {
        win,
        wr,
        wf,
        wout,
        bin: vec![0; d],
        br: vec![0; d],
        bout: vec![0; ACTIONS.len()],
        scale: SCALE,
    }
}

fn build_listing(w: &LoopWeights, loops: usize) -> String {
    let d = LATENT_DIM;
    let mut lines = vec![
        "# jotnar-agent looped proposer: input -> shared recurrent block -> output".to_string(),
        format!("# loops={loops}, latent={d}, scale={}", w.scale),
    ];
    for (j, name) in FEATURE_NAMES.iter().enumerate() {
        lines.push(format!("IN f{j}  # {name}"));
    }
    lines.push(String::new());

    // Input block: h0 = (Win f + bin) / scale.
    let mut prev = Vec::new();
    for i in 0..d {
        let mut terms = Vec::new();
        for (j, weight) in w.win[i].iter().enumerate() {
            if *weight != 0 {
                let p = format!("i_{i}_{j}");
                lines.push(format!("{p} = IMUL(f{j}, {weight})"));
                terms.push(p);
            }
        }
        let sum = sum_terms(&mut lines, &format!("in_{i}"), terms, w.bin[i]);
        let h = format!("h0_{i}");
        lines.push(format!("{h} = IDIV({sum}, {})", w.scale));
        prev.push(h);
    }
    lines.push(String::new());

    // Shared recurrent block, applied `loops` times. Input is reinjected.
    for t in 0..loops {
        let mut next = Vec::new();
        for i in 0..d {
            let mut terms = Vec::new();
            for (j, weight) in w.wr[i].iter().enumerate() {
                if *weight != 0 {
                    let p = format!("r_{t}_{i}_{j}");
                    lines.push(format!("{p} = IMUL({}, {weight})", prev[j]));
                    terms.push(p);
                }
            }
            for (j, weight) in w.wf[i].iter().enumerate() {
                if *weight != 0 {
                    let p = format!("x_{t}_{i}_{j}");
                    lines.push(format!("{p} = IMUL(f{j}, {weight})"));
                    terms.push(p);
                }
            }
            let sum = sum_terms(&mut lines, &format!("rec_{t}_{i}"), terms, w.br[i]);
            // Residual + nonlinearity + bound. This is the smallest stand-in for
            // Dory's normalized recurrent block; a purely linear loop would only
            // reparameterize the input model.
            let d = format!("d_{t}_{i}");
            lines.push(format!("{d} = IDIV({sum}, {})", w.scale));
            let r = format!("r_{t}_{i}");
            lines.push(format!("{r} = IADD({}, {d})", prev[i]));
            let hr = format!("hr_{t}_{i}");
            lines.push(format!("{hr} = IMAX({r}, 0)"));
            let lt = format!("lt_{t}_{i}");
            lines.push(format!("{lt} = ILT({hr}, 64)"));
            let h = format!("h{}_{i}", t + 1);
            lines.push(format!("{h} = ISELECT({lt}, {hr}, 64)"));
            next.push(h);
        }
        prev = next;
        lines.push(String::new());
    }

    // Output block: scores = Wout h + bout.
    for i in 0..ACTIONS.len() {
        let mut terms = Vec::new();
        for (j, weight) in w.wout[i].iter().enumerate() {
            if *weight != 0 {
                let p = format!("o_{i}_{j}");
                lines.push(format!("{p} = IMUL({}, {weight})", prev[j]));
                terms.push(p);
            }
        }
        let sum = sum_terms(&mut lines, &format!("out_{i}"), terms, w.bout[i]);
        // A bare `s = stream` is not a Jotnar instruction; make the readout an op.
        lines.push(format!("s{i} = IADD({sum}, 0)"));
    }
    lines.join("\n") + "\n"
}

/// Sum a list of stream names, add a literal bias, and return the resulting
/// stream name. The emitted ops are only `IADD`; no host arithmetic.
fn sum_terms(lines: &mut Vec<String>, prefix: &str, terms: Vec<String>, bias: i64) -> String {
    if terms.is_empty() {
        let zero = format!("{prefix}_z");
        lines.push(format!("{zero} = IMUL(f0, 0)"));
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
