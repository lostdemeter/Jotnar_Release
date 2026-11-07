//! Bounded circuit steps.
//!
//! Every step is a proposal. It has a precondition, an application, and a
//! postcondition. The planner never trusts a step's own claim: the target gate
//! is the acceptance.

use std::collections::BTreeMap;

use crate::family::{self, Params};
use crate::state::{State, StateKind};

#[derive(Clone, Debug)]
pub enum Step {
    Fit {
        family: String,
        params: Params,
        rank: u32,
    },
    Compose {
        consumer: String,
        rank: u32,
    },
    Refine {
        rank: u32,
    },
}

#[derive(Clone, Debug)]
pub struct StepRecord {
    pub kind: String,
    pub name: String,
    pub rank: u32,
    pub params: BTreeMap<String, String>,
    pub note: String,
}

impl Step {
    pub fn rank(&self) -> u32 {
        match self {
            Step::Fit { rank, .. } | Step::Compose { rank, .. } | Step::Refine { rank } => *rank,
        }
    }

    pub fn label(&self) -> String {
        match self {
            Step::Fit { family, params, .. } => format!("fit[{family}]{params:?}"),
            Step::Compose { consumer, .. } => format!("compose[{consumer}]"),
            Step::Refine { .. } => "refine".to_string(),
        }
    }

    pub fn record(&self, note: impl Into<String>) -> StepRecord {
        match self {
            Step::Fit {
                family,
                params,
                rank,
            } => StepRecord {
                kind: "fit".into(),
                name: format!("fit[{family}]"),
                rank: *rank,
                params: params.clone(),
                note: note.into(),
            },
            Step::Compose { consumer, rank } => StepRecord {
                kind: "compose".into(),
                name: format!("compose[{consumer}]"),
                rank: *rank,
                params: BTreeMap::new(),
                note: note.into(),
            },
            Step::Refine { rank } => StepRecord {
                kind: "refine".into(),
                name: "refine".into(),
                rank: *rank,
                params: BTreeMap::new(),
                note: note.into(),
            },
        }
    }

    pub fn apply(&self, state: &State) -> Result<State, String> {
        match self {
            Step::Fit {
                family: name,
                params,
                ..
            } => {
                let fam = family::family(name).ok_or_else(|| format!("unknown family {name:?}"))?;
                let text = (fam.build)(params)?;
                let s = State::new(format!("fit[{name}]"), text)?
                    .with_kind(StateKind::Fitted)
                    .with_family(name.clone(), params.clone());
                Ok(s)
            }
            Step::Compose { consumer, .. } => compose(state, consumer),
            Step::Refine { .. } => {
                let text = cse(&state.text);
                let s = State::new(format!("refine[{}]", state.name), text)?
                    .with_kind(StateKind::Refined);
                Ok(s)
            }
        }
    }
}

fn compose(state: &State, consumer: &str) -> Result<State, String> {
    let outs = state.outputs();
    match consumer {
        "add2" => {
            if outs.len() != 1 {
                return Err(format!(
                    "compose[add2] needs exactly one output, got {}",
                    outs.len()
                ));
            }
            let fresh = fresh_name(&state.text, "s");
            let mut text = rename_token(&state.text, &outs[0], &fresh);
            text.push_str(&format!("\nOUT = IADD({fresh}, {fresh})\n"));
            let s = State::new(format!("compose[add2]({})", state.name), text)?
                .with_kind(StateKind::Composed)
                .with_family("compose:add2".to_string(), BTreeMap::new());
            Ok(s)
        }
        "sum4" => {
            if outs.len() != 4 {
                return Err(format!(
                    "compose[sum4] needs exactly four outputs, got {}",
                    outs.len()
                ));
            }
            let mut names = outs;
            let mut text = state.text.clone();
            for name in names.iter_mut() {
                if name == "OUT" {
                    let fresh = fresh_name(&text, "c");
                    text = rename_token(&text, name, &fresh);
                    *name = fresh;
                }
            }
            text.push_str(&format!(
                "\nt0 = IADD({}, {})\nt1 = IADD({}, {})\nOUT = IADD(t0, t1)\n",
                names[0], names[1], names[2], names[3]
            ));
            let s = State::new(format!("compose[sum4]({})", state.name), text)?
                .with_kind(StateKind::Composed)
                .with_family("compose:sum4".to_string(), BTreeMap::new());
            Ok(s)
        }
        other => Err(format!("unknown compose consumer {other:?}")),
    }
}

fn contains_token(text: &str, token: &str) -> bool {
    let mut cur = String::new();
    for ch in text.chars() {
        if ch.is_alphanumeric() || ch == '_' {
            cur.push(ch);
        } else {
            if cur == token {
                return true;
            }
            cur.clear();
        }
    }
    cur == token
}

fn fresh_name(text: &str, prefix: &str) -> String {
    for i in 0.. {
        let name = format!("{prefix}{i}");
        if !contains_token(text, &name) {
            return name;
        }
    }
    unreachable!()
}

/// Whole-identifier replacement. Comments are rewritten too; that is harmless
/// and keeps the listing internally consistent.
pub fn rename_token(text: &str, old: &str, new: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut cur = String::new();
    for ch in text.chars() {
        if ch.is_alphanumeric() || ch == '_' {
            cur.push(ch);
        } else {
            if !cur.is_empty() {
                if cur == old {
                    out.push_str(new);
                } else {
                    out.push_str(&cur);
                }
                cur.clear();
            }
            out.push(ch);
        }
    }
    if !cur.is_empty() {
        if cur == old {
            out.push_str(new);
        } else {
            out.push_str(&cur);
        }
    }
    out
}

/// A conservative common-subexpression elimination. It only removes a later
/// line whose right-hand side is byte-identical to an earlier one, and rewrites
/// the later name to the earlier name. It never invents an op.
pub fn cse(text: &str) -> String {
    let mut seen: BTreeMap<String, String> = BTreeMap::new();
    let mut repl: Vec<(String, String)> = Vec::new();
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let Some((lhs, rhs)) = line.split_once('=') else {
            continue;
        };
        let lhs = lhs.trim();
        let rhs = rhs.trim();
        if lhs.is_empty() || rhs.is_empty() || lhs.split_whitespace().count() != 1 {
            continue;
        }
        if let Some(first) = seen.get(rhs) {
            if first != lhs {
                repl.push((lhs.to_string(), first.clone()));
            }
        } else {
            seen.insert(rhs.to_string(), lhs.to_string());
        }
    }
    if repl.is_empty() {
        return text.to_string();
    }
    let mut out = text.to_string();
    for (from, to) in &repl {
        out = rename_token(&out, from, to);
    }
    let mut lines = Vec::new();
    for line in out.lines() {
        let t = line.trim();
        if let Some((lhs, rhs)) = t.split_once('=') {
            if lhs.trim() == rhs.trim() {
                continue;
            }
        }
        lines.push(line.to_string());
    }
    lines.join("\n") + "\n"
}
