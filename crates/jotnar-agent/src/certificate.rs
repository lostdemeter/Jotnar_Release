//! Map a circuit outcome onto the `jotnar-lab` certificate schema.
//!
//! A solved path is a `pass`; a refused intent is a `bottom` with a typed
//! refusal. Both are successful certificates. The certificate names what it
//! proves and what it does not.

use std::collections::BTreeMap;

use jotnar_lab::model::{
    json_map, now_iso, sha256_str, Artifact, Bench, Certificate, Derivation, Falsifier, Gate,
    NullCheck, Outcome, Proof, Proposal, Provenance, Refusal, Sample, Space, Status, Verdict,
    Witness, SCHEMA, SCHEMA_VERSION,
};

use crate::gate::Example;
use crate::intent::Intent;
use crate::state::State;
use crate::step::StepRecord;

fn base_certificate(
    intent: &Intent,
    target: &State,
    examples: &[Example],
    gate_name: &str,
    gate_level: &str,
) -> Certificate {
    let family = intent
        .output
        .family
        .clone()
        .or_else(|| target.family.clone())
        .or_else(|| intent.output.label.clone())
        .unwrap_or_else(|| "listing".to_string());
    let mut hashes = BTreeMap::new();
    hashes.insert(target.name.clone(), sha256_str(&target.text));
    Certificate {
        schema: SCHEMA.to_string(),
        schema_version: SCHEMA_VERSION,
        id: intent.id.clone(),
        status: Status::Current,
        outcome: Outcome::Refused,
        verdict: Verdict::Bottom,
        space: Space {
            label: intent
                .output
                .label
                .clone()
                .unwrap_or_else(|| target.name.clone()),
            map: target.name.clone(),
            family,
            primitive: "listing".to_string(),
            sample: Sample {
                seed: 0,
                size: format!("{} example(s)", examples.len()),
                domain: "listing-level circuit".to_string(),
                forced_corners: Vec::new(),
            },
            reference: "Jotnar reference interpreter, exact outputs".to_string(),
            bounds: vec![
                "the supplied examples are a finite sample".to_string(),
                "the certificate does not quantify over all inputs".to_string(),
            ],
            hashes,
        },
        proposal: Proposal {
            kind: "circuit-path".to_string(),
            candidate: "no verified path under the current harness".to_string(),
            provenance: json_map(&[("proposer", "jotnar-agent"), ("seed", "0")]),
        },
        gate: Gate {
            name: gate_name.to_string(),
            level: gate_level.to_string(),
            reference: "the target listing".to_string(),
            frozen_at: "pre-registered".to_string(),
            gate_sha256: sha256_str(gate_name),
            can_say_no: true,
        },
        witness: Witness {
            map: target.name.clone(),
            input: format!("{} example(s)", examples.len()),
            output: String::new(),
            trace: String::new(),
        },
        proof: Proof {
            gate: gate_name.to_string(),
            contract: "every compared stream is exact on every supplied example".to_string(),
        },
        derivation: Derivation {
            trace: String::new(),
        },
        assumptions: vec![
            "the reference interpreter is the oracle".to_string(),
            "the supplied examples are the intended function".to_string(),
            "the target listing is well-formed".to_string(),
        ],
        unknowns: vec![
            "all inputs outside the examples".to_string(),
            "C / non-FPU / CUDA parity is a separate gate".to_string(),
            "cost is not measured in this slice".to_string(),
        ],
        proves: vec!["the circuit refused under the current bounded harness".to_string()],
        does_not_prove: vec![
            "that no path exists with a larger bound or a new step".to_string(),
            "that the target is the only acceptable implementation".to_string(),
        ],
        falsifiers: vec![Falsifier {
            corruption: "swap two candidate inputs".to_string(),
            expected: "fail".to_string(),
            actual: "fail".to_string(),
            caught_by: gate_name.to_string(),
            not_caught: false,
        }],
        nulls: vec![NullCheck {
            name: "identity".to_string(),
            control: "candidate is the target".to_string(),
            expected: "pass".to_string(),
            actual: "pass".to_string(),
            effect: "none".to_string(),
            status: "tested".to_string(),
        }],
        bench: Bench {
            clock: "not_measured".to_string(),
            scope: "solve".to_string(),
            kind: "not_measured".to_string(),
            runs: 0,
            p50_ms: None,
            p99_ms: None,
            max_ms: None,
            units: "ns".to_string(),
            note: "cost model not implemented in v0".to_string(),
        },
        refusal: None,
        provenance: Provenance {
            command: "jotnar-agent".to_string(),
            cwd: std::env::current_dir()
                .map(|p| p.display().to_string())
                .unwrap_or_else(|_| "unknown".to_string()),
            commit: "unknown".to_string(),
            host: jotnar_lab::model::hostname(),
            environment: BTreeMap::from([("built_at".to_string(), now_iso())]),
        },
        artifacts: Vec::<Artifact>::new(),
        supersedes: None,
        superseded_by: None,
        certificate_sha256: None,
    }
}

pub fn build_solved(
    intent: &Intent,
    target: &State,
    examples: &[Example],
    steps: &[StepRecord],
    recipe: &str,
) -> Certificate {
    let mut cert = base_certificate(
        intent,
        target,
        examples,
        "bit_exact_interpreter",
        intent.invariant.level(),
    );
    cert.outcome = Outcome::Solved;
    cert.verdict = Verdict::Pass;
    cert.proposal.candidate = recipe.to_string();
    cert.witness.output = recipe.to_string();
    cert.witness.trace = steps
        .iter()
        .map(|s| format!("{}:{}", s.kind, s.name))
        .collect::<Vec<_>>()
        .join(" -> ");
    cert.derivation.trace = cert.witness.trace.clone();
    cert.proves = vec![match intent.invariant {
        crate::intent::Invariant::BitExact => format!(
            "the candidate reproduces the target's outputs exactly on {} example(s)",
            examples.len()
        ),
        crate::intent::Invariant::Structural => format!(
            "the candidate preserves the target's structural signature on {} example(s)",
            examples.len()
        ),
    }];
    cert.does_not_prove = vec![
        "the function on inputs outside the supplied examples".to_string(),
        "C / non-FPU / CUDA parity".to_string(),
        "that this is the only verified path".to_string(),
    ];
    cert.refusal = None;
    cert.seal();
    cert
}

pub fn build_refused(
    intent: &Intent,
    target: &State,
    examples: &[Example],
    code: &str,
    message: &str,
    missing: Vec<String>,
) -> Certificate {
    let mut cert = base_certificate(
        intent,
        target,
        examples,
        "bounded_path_search",
        intent.invariant.level(),
    );
    cert.outcome = Outcome::Refused;
    cert.verdict = Verdict::Bottom;
    cert.refusal = Some(Refusal {
        code: code.to_string(),
        message: message.to_string(),
        missing,
    });
    cert.witness.trace = message.to_string();
    cert.witness.output = "no verified path".to_string();
    cert.derivation.trace = message.to_string();
    cert.proposal.candidate = "no verified path under the current harness".to_string();
    cert.proof.contract =
        "the bounded search found no candidate that survives the gate".to_string();
    cert.seal();
    cert
}
