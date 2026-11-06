use std::collections::BTreeMap;

use jotnar_lab::model::*;

fn minimal() -> Certificate {
    let mut c = Certificate {
        schema: SCHEMA.into(),
        schema_version: SCHEMA_VERSION,
        id: "test/one".into(),
        status: Status::Current,
        outcome: Outcome::Solved,
        verdict: Verdict::Pass,
        space: Space {
            label: "space".into(),
            map: "m".into(),
            family: "f".into(),
            primitive: "p".into(),
            sample: Sample {
                seed: 0,
                size: "1".into(),
                domain: "d".into(),
                forced_corners: vec![],
            },
            reference: "r".into(),
            bounds: vec!["b".into()],
            hashes: BTreeMap::new(),
        },
        proposal: Proposal {
            kind: "k".into(),
            candidate: "c".into(),
            provenance: BTreeMap::new(),
        },
        gate: Gate {
            name: "g".into(),
            level: "semantic".into(),
            reference: "r".into(),
            frozen_at: "pre".into(),
            gate_sha256: sha256_str("gate"),
            can_say_no: true,
        },
        witness: Witness {
            map: "m".into(),
            input: "i".into(),
            output: "o".into(),
            trace: "t".into(),
        },
        proof: Proof {
            gate: "g".into(),
            contract: "c".into(),
        },
        derivation: Derivation { trace: "t".into() },
        assumptions: vec!["a".into()],
        unknowns: vec!["u".into()],
        proves: vec!["p".into()],
        does_not_prove: vec!["n".into()],
        falsifiers: vec![Falsifier {
            corruption: "x".into(),
            expected: "fail".into(),
            actual: "fail".into(),
            caught_by: "g".into(),
            not_caught: false,
        }],
        nulls: vec![NullCheck {
            name: "n".into(),
            control: "c".into(),
            expected: "reject".into(),
            actual: "reject".into(),
            effect: "rejected".into(),
            status: "pass".into(),
        }],
        bench: Bench {
            clock: "host_wall".into(),
            scope: "end_to_end".into(),
            kind: "measured".into(),
            runs: 1,
            p50_ms: Some(1.0),
            p99_ms: Some(1.0),
            max_ms: Some(1.0),
            units: "ms".into(),
            note: "n".into(),
        },
        refusal: None,
        provenance: Provenance {
            command: "c".into(),
            cwd: "w".into(),
            commit: "n".into(),
            host: "h".into(),
            environment: BTreeMap::new(),
        },
        artifacts: vec![],
        supersedes: None,
        superseded_by: None,
        certificate_sha256: None,
    };
    c.seal();
    c
}

#[test]
fn sealed_certificate_validates() {
    let c = minimal();
    assert!(c.validate().is_empty(), "{:?}", c.validate());
}

#[test]
fn missing_does_not_prove_is_rejected() {
    let mut c = minimal();
    c.does_not_prove.clear();
    assert!(c.validate().iter().any(|e| e.contains("does_not_prove")));
}

#[test]
fn vacuous_gate_is_rejected() {
    let mut c = minimal();
    c.gate.can_say_no = false;
    assert!(c.validate().iter().any(|e| e.contains("can_say_no")));
}

#[test]
fn canonical_hash_changes_when_body_changes() {
    let c = minimal();
    let h = c.compute_sha256();
    let mut d = c.clone();
    d.proves.push("another".into());
    assert_ne!(h, d.compute_sha256());
}
