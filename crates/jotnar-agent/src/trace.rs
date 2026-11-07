//! Execution-grounded trace records.
//!
//! Every record is a task, a candidate, an exact outcome, and a certificate.
//! The schema is deliberately flat JSONL so the data flywheel can consume it
//! without importing the agent's internals.

use std::collections::BTreeMap;
use std::fs::OpenOptions;
use std::io::Write;

use serde::{Deserialize, Serialize};

use crate::corpus::{corpus, generated_corpus};
use crate::driver::{evaluate, CaseOutcome, RoundReport};
use crate::harness::Harness;
use crate::intent::{Case, CaseExpect};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HarnessTrace {
    pub families: Vec<String>,
    pub compose_consumers: Vec<String>,
    pub max_depth: usize,
    pub max_refinements: usize,
    pub refine_enabled: bool,
    pub use_tags: bool,
}

impl From<&Harness> for HarnessTrace {
    fn from(h: &Harness) -> Self {
        Self {
            families: h.families.clone(),
            compose_consumers: h.compose_consumers.clone(),
            max_depth: h.max_depth,
            max_refinements: h.max_refinements,
            refine_enabled: h.refine_enabled,
            use_tags: h.use_tags,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TaskTrace {
    pub case: String,
    pub family: String,
    pub expect: String,
    pub split: String,
    pub phase: String,
    pub depth: usize,
    pub start_outputs: usize,
    pub start_ops: usize,
    pub target_outputs: usize,
    pub target_ops: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CandidateTrace {
    pub index: usize,
    pub label: String,
    pub selected: bool,
    pub adopted: bool,
    pub harness: HarnessTrace,
    pub train_correct: usize,
    pub train_wrong: usize,
    pub heldout_correct: usize,
    pub heldout_wrong: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GateTrace {
    pub verdict: String,
    pub outcome: String,
    pub certificate: jotnar_lab::model::Certificate,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CostTrace {
    pub steps: usize,
    pub refined: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ProvenanceTrace {
    pub proposer: String,
    pub round: usize,
    pub source: String,
}

/// A teacher row for distillation. Dory (or any external teacher) writes the
/// same JSONL shape: task vector in, latent out, with a split and source.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TeacherRecord {
    #[serde(default)]
    pub case: String,
    pub task: Vec<i64>,
    pub latent: Vec<f64>,
    pub split: String,
    pub source: String,
}

pub fn read_teacher_records(path: &std::path::Path) -> Result<Vec<TeacherRecord>, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
    text.lines()
        .filter(|line| !line.trim().is_empty())
        .map(|line| serde_json::from_str(line).map_err(|e| e.to_string()))
        .collect()
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TraceRecord {
    pub task: TaskTrace,
    pub candidate: CandidateTrace,
    pub gate: GateTrace,
    pub cost: CostTrace,
    pub provenance: ProvenanceTrace,
}

fn expect_str(expect: CaseExpect) -> &'static str {
    match expect {
        CaseExpect::Solve => "solve",
        CaseExpect::Refuse => "refuse",
    }
}

fn split_map() -> BTreeMap<String, String> {
    let mut map: BTreeMap<String, String> = corpus()
        .into_iter()
        .map(|c| (c.case.intent.id.clone(), c.split.as_str().to_string()))
        .collect();
    for c in generated_corpus() {
        map.insert(c.case.intent.id.clone(), c.split.as_str().to_string());
    }
    map
}

fn find_case<'a>(cases: &'a [Case], id: &str) -> Option<&'a Case> {
    cases.iter().find(|c| c.intent.id == id)
}

#[allow(clippy::too_many_arguments)]
fn record(
    case: &Case,
    outcome: &CaseOutcome,
    split: &str,
    phase: &str,
    candidate: &CandidateTrace,
    proposer: &str,
    round: usize,
    source: &str,
) -> TraceRecord {
    TraceRecord {
        task: TaskTrace {
            case: case.intent.id.clone(),
            family: case.family.clone(),
            expect: expect_str(case.expect).to_string(),
            split: split.to_string(),
            phase: phase.to_string(),
            depth: case.intent.depth,
            start_outputs: case.start_outputs,
            start_ops: case.start_ops,
            target_outputs: outcome.target_outputs,
            target_ops: outcome.target_ops,
        },
        candidate: candidate.clone(),
        gate: GateTrace {
            verdict: format!("{:?}", outcome.certificate.verdict).to_lowercase(),
            outcome: outcome.kind.as_str().to_string(),
            certificate: outcome.certificate.clone(),
        },
        cost: CostTrace {
            steps: outcome.steps,
            refined: outcome.refined,
        },
        provenance: ProvenanceTrace {
            proposer: proposer.to_string(),
            round,
            source: source.to_string(),
        },
    }
}

/// Append one JSONL record per (candidate, case, phase) to `path`.
///
/// `cases` are the selection cases; `heldout_cases` are the frozen held-out
/// family members. The adopted harness is also evaluated on the held-out
/// cases, so the trace includes the real held-out outcome, not just the
/// held-out examples embedded in the selection cases.
pub fn write_round_traces(
    path: &std::path::Path,
    cases: &[Case],
    heldout_cases: &[Case],
    report: &RoundReport,
    proposer: &str,
    round: usize,
) -> Result<usize, String> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    }
    let splits = split_map();
    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .map_err(|e| format!("{}: {e}", path.display()))?;
    let mut count = 0usize;

    for candidate in &report.candidates {
        let adopted = report.adopted_flag && candidate.selected;
        let trace_candidate = CandidateTrace {
            index: candidate.index,
            label: candidate.label.clone(),
            selected: candidate.selected,
            adopted,
            harness: HarnessTrace::from(&candidate.harness),
            train_correct: candidate.train.correct(),
            train_wrong: candidate.train.wrong(),
            heldout_correct: candidate.heldout.correct(),
            heldout_wrong: candidate.heldout.wrong(),
        };
        for outcome in &candidate.train.outcomes {
            let Some(case) = find_case(cases, &outcome.case) else {
                continue;
            };
            let split = splits
                .get(&outcome.case)
                .cloned()
                .unwrap_or_else(|| "stream".to_string());
            let rec = record(
                case,
                outcome,
                &split,
                "selection",
                &trace_candidate,
                proposer,
                round,
                "candidate",
            );
            writeln!(
                file,
                "{}",
                serde_json::to_string(&rec).map_err(|e| e.to_string())?
            )
            .map_err(|e| e.to_string())?;
            count += 1;
        }
        for outcome in &candidate.heldout.outcomes {
            let Some(case) = find_case(cases, &outcome.case) else {
                continue;
            };
            let split = splits
                .get(&outcome.case)
                .cloned()
                .unwrap_or_else(|| "stream".to_string());
            let rec = record(
                case,
                outcome,
                &split,
                "heldout-examples",
                &trace_candidate,
                proposer,
                round,
                "candidate",
            );
            writeln!(
                file,
                "{}",
                serde_json::to_string(&rec).map_err(|e| e.to_string())?
            )
            .map_err(|e| e.to_string())?;
            count += 1;
        }
    }

    if !heldout_cases.is_empty() {
        let held = evaluate(&report.adopted_harness, heldout_cases, false);
        let adopted_candidate = CandidateTrace {
            index: report.candidates.len(),
            label: "adopted".to_string(),
            selected: true,
            adopted: report.adopted_flag,
            harness: HarnessTrace::from(&report.adopted_harness),
            train_correct: report.adopted.correct(),
            train_wrong: report.adopted.wrong(),
            heldout_correct: held.correct(),
            heldout_wrong: held.wrong(),
        };
        for outcome in &held.outcomes {
            let Some(case) = find_case(heldout_cases, &outcome.case) else {
                continue;
            };
            let rec = record(
                case,
                outcome,
                "heldout",
                "heldout-split",
                &adopted_candidate,
                proposer,
                round,
                "adopted",
            );
            writeln!(
                file,
                "{}",
                serde_json::to_string(&rec).map_err(|e| e.to_string())?
            )
            .map_err(|e| e.to_string())?;
            count += 1;
        }
    }

    Ok(count)
}
