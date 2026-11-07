//! The autonomous driver: propose -> probe -> compose -> blind select ->
//! reflect -> adopt.
//!
//! The driver does not train weights. It edits the circuit harness, probes the
//! edits on the task stream, selects by exact certificates, and retains the
//! adopted harness in an experience log. Adoption requires held-out
//! non-regression and retention of every case the incumbent already got right.

use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};

use jotnar_lab::model::Certificate;
use serde::{Deserialize, Serialize};
use serde_json::json;

use crate::certificate::build_refused;
use crate::harness::{Harness, Mutation};
use crate::intent::{Case, CaseExpect};
use crate::solve::solve;

/// The proposer seam. The deterministic implementation is the first one; a
/// future Jotnar LLM implements the same trait and reads the same evidence.
pub trait Proposer {
    fn propose(
        &self,
        base: &Harness,
        train: &EvalReport,
        experience: &[Experience],
    ) -> Vec<Mutation>;

    /// Optional case-aware path. When it returns `Some`, the driver uses these
    /// proposals instead of the aggregate `propose`.
    fn propose_cases(
        &self,
        _views: &[crate::view::CaseView],
        _base: &Harness,
        _experience: &[Experience],
    ) -> Option<Vec<Mutation>> {
        None
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct DeterministicProposer;

impl Proposer for DeterministicProposer {
    fn propose(
        &self,
        base: &Harness,
        _train: &EvalReport,
        _experience: &[Experience],
    ) -> Vec<Mutation> {
        base.mutations()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OutcomeKind {
    Solved,
    Refused,
    WrongSolve,
    WrongRefusal,
}

impl OutcomeKind {
    pub fn as_str(self) -> &'static str {
        match self {
            OutcomeKind::Solved => "solved",
            OutcomeKind::Refused => "refused",
            OutcomeKind::WrongSolve => "wrong-solve",
            OutcomeKind::WrongRefusal => "wrong-refusal",
        }
    }

    pub fn correct(self) -> bool {
        matches!(self, OutcomeKind::Solved | OutcomeKind::Refused)
    }
}

#[derive(Clone, Debug)]
pub struct CaseOutcome {
    pub case: String,
    pub family: String,
    pub kind: OutcomeKind,
    pub steps: usize,
    pub refined: bool,
    pub start_outputs: usize,
    pub start_ops: usize,
    pub target_outputs: usize,
    pub target_ops: usize,
    pub certificate: Certificate,
    pub note: String,
}

#[derive(Clone, Debug)]
pub struct EvalReport {
    pub outcomes: Vec<CaseOutcome>,
    pub solved: usize,
    pub refused: usize,
    pub wrong_solve: usize,
    pub wrong_refusal: usize,
    pub total_steps: usize,
}

impl EvalReport {
    pub fn wrong_refusal_in(&self, family: &str) -> usize {
        self.outcomes
            .iter()
            .filter(|o| o.family == family && o.kind == OutcomeKind::WrongRefusal)
            .count()
    }

    pub fn correct_in(&self, family: &str) -> usize {
        self.outcomes
            .iter()
            .filter(|o| o.family == family && o.kind.correct())
            .count()
    }

    pub fn correct(&self) -> usize {
        self.solved + self.refused
    }

    pub fn wrong(&self) -> usize {
        self.wrong_solve + self.wrong_refusal
    }

    /// Lexicographic selection score: more correct, fewer wrong, fewer steps.
    pub fn score(&self) -> (usize, usize, usize, usize) {
        (
            self.correct(),
            usize::MAX - self.wrong(),
            usize::MAX - self.total_steps,
            0,
        )
    }

    pub fn better_than(&self, other: &EvalReport) -> bool {
        self.score() > other.score()
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Experience {
    pub cue: String,
    pub lesson: String,
    pub evidence: String,
    pub confidence: String,
    pub edits: Vec<String>,
    pub harness: String,
}

#[derive(Clone, Debug)]
pub struct CandidateReport {
    pub index: usize,
    pub label: String,
    pub train: EvalReport,
    pub heldout: EvalReport,
    pub selected: bool,
    pub harness: Harness,
}

#[derive(Clone, Debug)]
pub struct RoundReport {
    pub base: EvalReport,
    pub base_heldout: EvalReport,
    pub adopted: EvalReport,
    pub adopted_heldout: EvalReport,
    pub adopted_harness: Harness,
    pub adopted_flag: bool,
    pub chosen_index: Option<usize>,
    pub chosen_label: String,
    pub candidates: Vec<CandidateReport>,
    pub edits: Vec<String>,
    pub experience: Option<Experience>,
    pub out_dir: PathBuf,
}

pub fn run_round(base: &Harness, cases: &[Case], out_dir: &Path) -> RoundReport {
    run_round_with(base, cases, out_dir, &DeterministicProposer)
}

pub fn run_round_with(
    base: &Harness,
    cases: &[Case],
    out_dir: &Path,
    proposer: &dyn Proposer,
) -> RoundReport {
    let memory_path = out_dir.join("experience.jsonl");
    run_round_with_memory(base, cases, out_dir, &memory_path, proposer)
}

/// Same loop, with an explicit persistent experience path. Use this when
/// several round directories should share one memory.
pub fn run_round_with_memory(
    base: &Harness,
    cases: &[Case],
    out_dir: &Path,
    memory_path: &Path,
    proposer: &dyn Proposer,
) -> RoundReport {
    fs::create_dir_all(out_dir).expect("create round dir");
    if let Some(parent) = memory_path.parent() {
        fs::create_dir_all(parent).expect("create memory dir");
    }
    let base_train = evaluate(base, cases, false);
    let base_held = evaluate(base, cases, true);
    let memory = load_experience(memory_path);

    let views = crate::view::case_views(cases, &base_train);
    let mutations = match proposer.propose_cases(&views, base, &memory) {
        Some(m) => m,
        None => proposer.propose(base, &base_train, &memory),
    };
    let mut candidates: Vec<CandidateReport> = Vec::new();
    let mut pool: Vec<(usize, Harness)> = vec![(0, base.clone())];
    for mutation in &mutations {
        pool.push((pool.len(), mutation.harness.clone()));
    }
    let labels: Vec<String> = std::iter::once("incumbent".to_string())
        .chain(mutations.into_iter().map(|m| m.label))
        .collect();

    for (index, harness) in &pool {
        let train = evaluate(harness, cases, false);
        let held = evaluate(harness, cases, true);
        candidates.push(CandidateReport {
            index: *index,
            label: labels.get(*index).cloned().unwrap_or_default(),
            train,
            heldout: held,
            selected: false,
            harness: harness.clone(),
        });
    }

    // Compose the two best non-incumbent proposals. This is the part that can
    // solve two independent failures when neither single edit can.
    let mut top: Vec<usize> = (1..candidates.len()).collect();
    top.sort_by_key(|i| {
        let c = &candidates[*i];
        (usize::MAX - c.train.correct(), c.train.wrong(), c.index)
    });
    if top.len() >= 2 {
        let a = candidates[top[0]].harness.clone();
        let b = candidates[top[1]].harness.clone();
        let merged = merge_harnesses(base, &a, &b);
        if merged != *base {
            let train = evaluate(&merged, cases, false);
            let held = evaluate(&merged, cases, true);
            let index = candidates.len();
            candidates.push(CandidateReport {
                index,
                label: format!(
                    "compose: {} + {}",
                    candidates[top[0]].label, candidates[top[1]].label
                ),
                train,
                heldout: held,
                selected: false,
                harness: merged,
            });
        }
    }

    // Blind selection: only the metrics are read here, never the labels.
    // Among candidates the exact gate cannot distinguish (same correctness and
    // step count), prefer the harness with more capability coverage. That is
    // neutral drift: it keeps a second capability alive for a different
    // composition without changing exact correctness on the selection stream.
    let mut ranked: Vec<usize> = (0..candidates.len()).collect();
    ranked.sort_by_key(|i| {
        let c = &candidates[*i];
        (
            usize::MAX - c.train.correct(),
            c.train.wrong(),
            c.train.total_steps,
            usize::MAX - capability_coverage(&c.harness),
            c.index,
        )
    });
    let chosen = ranked[0];
    candidates[chosen].selected = true;
    let chosen_harness = candidates[chosen].harness.clone();
    let chosen_train = candidates[chosen].train.clone();
    let chosen_held = candidates[chosen].heldout.clone();

    // Retention: every case the incumbent got right must stay right.
    let retention = cases.iter().all(|case| {
        let before = base_train
            .outcomes
            .iter()
            .find(|o| o.case == case.intent.id)
            .map(|o| o.kind.correct())
            .unwrap_or(false);
        let after = chosen_train
            .outcomes
            .iter()
            .find(|o| o.case == case.intent.id)
            .map(|o| o.kind.correct())
            .unwrap_or(false);
        !before || after
    });
    let heldout_ok = chosen_held.correct() >= base_held.correct()
        && chosen_held.wrong_solve <= base_held.wrong_solve
        && chosen_held.wrong_refusal <= base_held.wrong_refusal;
    let improves = chosen_train.correct() > base_train.correct() && chosen_train.wrong() == 0;
    let adopt = improves && retention && heldout_ok;

    let (adopted_harness, adopted_train, adopted_held) = if adopt {
        (chosen_harness, chosen_train, chosen_held)
    } else {
        (base.clone(), base_train.clone(), base_held.clone())
    };
    let edits = base.diff(&adopted_harness);
    let experience = adopt.then(|| {
        let improved = adopted_train
            .outcomes
            .iter()
            .filter(|o| {
                base_train
                    .outcomes
                    .iter()
                    .find(|b| b.case == o.case)
                    .map(|b| !b.kind.correct() && o.kind.correct())
                    .unwrap_or(false)
            })
            .map(|o| o.case.clone())
            .collect::<Vec<_>>();
        Experience {
            cue: improved.join(", "),
            lesson: format!(
                "Adopt {}: it changes {} and keeps every incumbent-correct case correct.",
                candidates[chosen].label,
                if edits.is_empty() {
                    "nothing".to_string()
                } else {
                    edits.join("; ")
                }
            ),
            evidence: format!(
                "train {}/{} correct -> {}/{} correct; held-out {}/{} -> {}/{}; wrong {} -> {}",
                base_train.correct(),
                cases.len(),
                adopted_train.correct(),
                cases.len(),
                base_held.correct(),
                cases.len(),
                adopted_held.correct(),
                cases.len(),
                base_train.wrong(),
                adopted_train.wrong(),
            ),
            confidence: if adoption_is_clean(&base_train, &adopted_train) {
                "high for this bounded stream".to_string()
            } else {
                "medium".to_string()
            },
            edits: edits.clone(),
            harness: adopted_harness.fingerprint(),
        }
    });

    write_round(
        out_dir,
        memory_path,
        &base_train,
        &base_held,
        &adopted_train,
        &adopted_held,
        &candidates,
        &adopted_harness,
        adopt,
        &edits,
        experience.as_ref(),
    );

    RoundReport {
        base: base_train,
        base_heldout: base_held,
        adopted: adopted_train,
        adopted_heldout: adopted_held,
        adopted_harness,
        adopted_flag: adopt,
        chosen_index: Some(candidates[chosen].index),
        chosen_label: candidates[chosen].label.clone(),
        candidates,
        edits,
        experience,
        out_dir: out_dir.to_path_buf(),
    }
}

fn adoption_is_clean(base: &EvalReport, adopted: &EvalReport) -> bool {
    adopted.correct() >= base.correct() && adopted.wrong() <= base.wrong()
}

/// Number of additional capabilities beyond the base harness. Used only as a
/// tie-break after exact correctness and step count, so it never overrides the
/// gate.
fn capability_coverage(h: &Harness) -> usize {
    let mut count = h.compose_consumers.len().saturating_sub(1);
    if h.max_depth > 2 {
        count += 1;
    }
    if h.max_refinements > 1 {
        count += 1;
    }
    for family in ["select_min", "select_max"] {
        if h.families.iter().any(|f| f == family) {
            count += 1;
        }
    }
    count
}

/// Union two edited harnesses against the incumbent. Disables are applied only
/// when both candidates agree; unions and bounds take the stronger value.
fn merge_harnesses(base: &Harness, a: &Harness, b: &Harness) -> Harness {
    let mut h = base.clone();
    h.families = union3(&base.families, &a.families, &b.families);
    h.compose_consumers = union3(
        &base.compose_consumers,
        &a.compose_consumers,
        &b.compose_consumers,
    );
    h.max_depth = base.max_depth.max(a.max_depth).max(b.max_depth);
    h.max_refinements = base
        .max_refinements
        .max(a.max_refinements)
        .max(b.max_refinements);
    h.refine_enabled = if !a.refine_enabled && !b.refine_enabled {
        false
    } else {
        base.refine_enabled
    };
    h.use_tags = if !a.use_tags && !b.use_tags {
        false
    } else {
        base.use_tags
    };
    h
}

fn union3(a: &[String], b: &[String], c: &[String]) -> Vec<String> {
    let mut out = Vec::new();
    for item in a.iter().chain(b).chain(c) {
        if !out.contains(item) {
            out.push(item.clone());
        }
    }
    out
}

pub fn load_experience(path: &Path) -> Vec<Experience> {
    let Ok(text) = fs::read_to_string(path) else {
        return Vec::new();
    };
    text.lines()
        .filter_map(|line| serde_json::from_str::<Experience>(line).ok())
        .collect()
}

pub fn evaluate(harness: &Harness, cases: &[Case], heldout: bool) -> EvalReport {
    let mut outcomes = Vec::new();
    let (mut solved, mut refused, mut wrong_solve, mut wrong_refusal, mut total_steps) =
        (0, 0, 0, 0, 0);
    for case in cases {
        let examples = if heldout {
            &case.heldout
        } else {
            &case.examples
        };
        match solve(&case.intent, examples, harness) {
            Ok(res) => {
                let kind = match case.expect {
                    CaseExpect::Solve => OutcomeKind::Solved,
                    CaseExpect::Refuse => OutcomeKind::WrongSolve,
                };
                solved += usize::from(kind == OutcomeKind::Solved);
                wrong_solve += usize::from(kind == OutcomeKind::WrongSolve);
                total_steps += res.steps.len();
                outcomes.push(CaseOutcome {
                    case: case.intent.id.clone(),
                    family: case.family.clone(),
                    kind,
                    steps: res.steps.len(),
                    refined: res.refined,
                    start_outputs: case.start_outputs,
                    start_ops: case.start_ops,
                    target_outputs: res.target.outputs().len(),
                    target_ops: res.target.ops().len(),
                    certificate: res.certificate,
                    note: format!("gate: {}", res.gate.detail),
                });
            }
            Err(err) => {
                let kind = match case.expect {
                    CaseExpect::Refuse => OutcomeKind::Refused,
                    CaseExpect::Solve => OutcomeKind::WrongRefusal,
                };
                refused += usize::from(kind == OutcomeKind::Refused);
                wrong_refusal += usize::from(kind == OutcomeKind::WrongRefusal);
                let cert = build_refused(
                    &case.intent,
                    err.target(),
                    err.examples(),
                    err.code(),
                    err.message(),
                    err.missing(),
                );
                outcomes.push(CaseOutcome {
                    case: case.intent.id.clone(),
                    family: case.family.clone(),
                    kind,
                    steps: 0,
                    refined: false,
                    start_outputs: case.start_outputs,
                    start_ops: case.start_ops,
                    target_outputs: err.target().outputs().len(),
                    target_ops: err.target().ops().len(),
                    certificate: cert,
                    note: err.message().to_string(),
                });
            }
        }
    }
    EvalReport {
        outcomes,
        solved,
        refused,
        wrong_solve,
        wrong_refusal,
        total_steps,
    }
}

#[allow(clippy::too_many_arguments)]
fn write_round(
    out_dir: &Path,
    memory_path: &Path,
    base_train: &EvalReport,
    base_held: &EvalReport,
    adopted_train: &EvalReport,
    adopted_held: &EvalReport,
    candidates: &[CandidateReport],
    adopted_harness: &Harness,
    adopted: bool,
    edits: &[String],
    experience: Option<&Experience>,
) {
    for (sub, report) in [("base-train", base_train), ("base-heldout", base_held)] {
        write_certificates(out_dir, sub, report);
    }
    for (sub, report) in [
        ("adopted-train", adopted_train),
        ("adopted-heldout", adopted_held),
    ] {
        write_certificates(out_dir, sub, report);
    }

    let round = json!({
        "base": report_json(base_train),
        "base_heldout": report_json(base_held),
        "adopted": report_json(adopted_train),
        "adopted_heldout": report_json(adopted_held),
        "adopted_flag": adopted,
        "adopted_harness": adopted_harness.fingerprint(),
        "edits": edits,
        "candidates": candidates.iter().map(|c| json!({
            "index": c.index,
            "label": c.label,
            "train_correct": c.train.correct(),
            "train_wrong": c.train.wrong(),
            "train_steps": c.train.total_steps,
            "heldout_correct": c.heldout.correct(),
            "heldout_wrong": c.heldout.wrong(),
            "selected": c.selected,
        })).collect::<Vec<_>>(),
        "experience": experience,
    });
    fs::write(
        out_dir.join("round.json"),
        serde_json::to_string_pretty(&round).unwrap() + "\n",
    )
    .expect("write round.json");

    if let Some(exp) = experience {
        let mut f = OpenOptions::new()
            .create(true)
            .append(true)
            .open(memory_path)
            .expect("open experience.jsonl");
        writeln!(f, "{}", serde_json::to_string(exp).unwrap()).expect("append experience");
    }
}

fn write_certificates(out_dir: &Path, sub: &str, report: &EvalReport) {
    let dir = out_dir.join("certificates").join(sub);
    fs::create_dir_all(&dir).expect("create certificate dir");
    for outcome in &report.outcomes {
        let path = dir.join(format!("{}.json", outcome.case));
        let _ = outcome.certificate.write(&path);
    }
}

fn report_json(report: &EvalReport) -> serde_json::Value {
    json!({
        "solved": report.solved,
        "refused": report.refused,
        "wrong_solve": report.wrong_solve,
        "wrong_refusal": report.wrong_refusal,
        "correct": report.correct(),
        "wrong": report.wrong(),
        "total_steps": report.total_steps,
        "outcomes": report.outcomes.iter().map(|o| json!({
            "case": o.case,
            "family": o.family,
            "kind": o.kind.as_str(),
            "steps": o.steps,
            "refined": o.refined,
            "start_outputs": o.start_outputs,
            "start_ops": o.start_ops,
            "target_outputs": o.target_outputs,
            "target_ops": o.target_ops,
            "note": o.note,
        })).collect::<Vec<_>>(),
    })
}
