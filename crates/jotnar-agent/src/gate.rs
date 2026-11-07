//! Exact gates over listings.
//!
//! The reference interpreter is the oracle. A candidate is accepted only when
//! it matches the target on every supplied example, stream by stream. The gate
//! can say no; a missing or failed run is never a pass.

use std::collections::BTreeMap;

use serde::Serialize;

use jotnar::asm::AsmOptions;
use jotnar::payload::Feeds;
use jotnar::runtime::run_text;
use ndarray::{ArrayD, IxDyn};

use crate::state::State;

#[derive(Clone, Debug, Serialize)]
pub struct TestVec {
    pub shape: Vec<usize>,
    pub data: Vec<i64>,
}

impl TestVec {
    pub fn vector(data: Vec<i64>) -> Self {
        Self {
            shape: vec![data.len()],
            data,
        }
    }

    pub fn scalar(data: i64) -> Self {
        Self {
            shape: Vec::new(),
            data: vec![data],
        }
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct Example {
    pub inputs: BTreeMap<String, TestVec>,
}

impl Example {
    pub fn new(inputs: BTreeMap<String, TestVec>) -> Self {
        Self { inputs }
    }

    pub fn from_pairs(pairs: &[(&str, TestVec)]) -> Self {
        Self {
            inputs: pairs
                .iter()
                .map(|(k, v)| ((*k).to_string(), v.clone()))
                .collect(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Observed {
    Int(Vec<i64>),
    Float(Vec<String>),
    Bytes(Vec<u8>),
    Triple(Vec<(i8, i32, u8)>),
}

#[derive(Clone, Debug)]
pub struct CaseReport {
    pub case: usize,
    pub passed: bool,
    pub detail: String,
}

#[derive(Clone, Debug)]
pub struct GateReport {
    pub passed: bool,
    pub cases: Vec<CaseReport>,
    pub detail: String,
}

impl GateReport {
    pub fn fail(detail: impl Into<String>) -> Self {
        Self {
            passed: false,
            cases: Vec::new(),
            detail: detail.into(),
        }
    }
}

fn opts() -> AsmOptions {
    AsmOptions::new(".")
}

fn feeds(example: &Example) -> Result<Feeds, String> {
    let mut out = Feeds::new();
    for (name, tv) in &example.inputs {
        let arr = ArrayD::from_shape_vec(IxDyn(&tv.shape), tv.data.clone())
            .map_err(|e| format!("example {name}: bad shape {:?}: {e}", tv.shape))?;
        out.insert(name.clone(), jotnar::Value::I64(arr));
    }
    Ok(out)
}

fn observe(value: &jotnar::Value, what: &str) -> Result<Observed, String> {
    match value {
        jotnar::Value::I64(a) => Ok(Observed::Int(a.iter().copied().collect())),
        jotnar::Value::U8(a) => Ok(Observed::Bytes(a.iter().copied().collect())),
        jotnar::Value::Bool(a) => Ok(Observed::Int(
            a.iter().map(|v| if *v { 1 } else { 0 }).collect(),
        )),
        jotnar::Value::F64(a) => Ok(Observed::Float(
            a.iter().map(|v| format!("{v:.17e}")).collect(),
        )),
        jotnar::Value::F32(a) => Ok(Observed::Float(
            a.iter().map(|v| format!("{v:.9e}")).collect(),
        )),
        jotnar::Value::Triple(t) => Ok(Observed::Triple(
            t.s.iter()
                .copied()
                .zip(t.e.iter().copied())
                .zip(t.z.iter().copied())
                .map(|((s, e), z)| (s, e, z))
                .collect(),
        )),
        jotnar::Value::None => Err(format!("{what}: stream is None")),
    }
}

pub fn run_outputs(
    state: &State,
    example: &Example,
    outputs: &[String],
) -> Result<BTreeMap<String, Observed>, String> {
    let got = run_text(&state.text, &opts(), &feeds(example)?)
        .map_err(|e| format!("run {}: {e}", state.name))?;
    let mut out = BTreeMap::new();
    for name in outputs {
        let value = got
            .get(name)
            .ok_or_else(|| format!("run {}: missing output {name:?}", state.name))?;
        out.insert(name.clone(), observe(value, name)?);
    }
    Ok(out)
}

/// Run `state` on one example and return the concrete shape of each named
/// output stream. This is the bridge from a listing to its geometry; a stream
/// with an empty shape is a scalar.
pub fn run_output_shapes(
    state: &State,
    example: &Example,
    outputs: &[String],
) -> Result<Vec<Vec<usize>>, String> {
    let got = run_text(&state.text, &opts(), &feeds(example)?)
        .map_err(|e| format!("run {}: {e}", state.name))?;
    outputs
        .iter()
        .map(|name| {
            let value = got
                .get(name)
                .ok_or_else(|| format!("run {}: missing output {name:?}", state.name))?;
            Ok(value.shape().unwrap_or_default())
        })
        .collect()
}

/// Exact equality on every output and every example.
pub fn bit_exact(
    candidate: &State,
    target: &State,
    examples: &[Example],
    outputs: &[String],
) -> GateReport {
    if examples.is_empty() {
        return GateReport::fail("no examples: an empty sample is not a gate");
    }
    let mut cases = Vec::new();
    let mut all = true;
    for (i, ex) in examples.iter().enumerate() {
        let got = run_outputs(candidate, ex, outputs);
        let want = run_outputs(target, ex, outputs);
        match (got, want) {
            (Ok(got), Ok(want)) => {
                let mut bad = Vec::new();
                for name in outputs {
                    match (got.get(name), want.get(name)) {
                        (Some(a), Some(b)) if a == b => {}
                        (Some(a), Some(b)) => {
                            bad.push(format!("{name}: got {:?}, want {:?}", short(a), short(b)))
                        }
                        _ => bad.push(format!("{name}: missing on one side")),
                    }
                }
                let passed = bad.is_empty();
                all &= passed;
                cases.push(CaseReport {
                    case: i,
                    passed,
                    detail: if passed {
                        "exact".to_string()
                    } else {
                        bad.join("; ")
                    },
                });
            }
            (Err(e), _) | (_, Err(e)) => {
                all = false;
                cases.push(CaseReport {
                    case: i,
                    passed: false,
                    detail: e,
                });
            }
        }
    }
    GateReport {
        passed: all,
        cases,
        detail: if all {
            format!("{} example(s), every output exact", examples.len())
        } else {
            "at least one example differs".to_string()
        },
    }
}

pub fn structural(candidate: &State, target: &State) -> GateReport {
    let a = candidate.structural_signature();
    let b = target.structural_signature();
    let passed = a == b;
    GateReport {
        passed,
        cases: vec![CaseReport {
            case: 0,
            passed,
            detail: format!("candidate {a:?} vs target {b:?}"),
        }],
        detail: if passed {
            "same inputs, outputs, and op multiset".to_string()
        } else {
            "structural signature differs".to_string()
        },
    }
}

fn short(o: &Observed) -> String {
    match o {
        Observed::Int(v) => format!("int{:?}", &v[..v.len().min(8)]),
        Observed::Float(v) => format!("float{:?}", &v[..v.len().min(4)]),
        Observed::Bytes(v) => format!("bytes{:?}", &v[..v.len().min(8)]),
        Observed::Triple(v) => format!("triple{}", v.len()),
    }
}
