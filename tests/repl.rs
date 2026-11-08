use jotnar::asm::{self, AsmOptions};
use jotnar::payload::{feeds_from_json, feeds_to_json};
use serde_json::{json, Value as Json};

const SAXPY: &str = "IN A\nIN X\nIN Y\nax = IMUL(A, X)\nOUT = IADD(ax, Y)\n";

fn payload(a: &[i64], x: &[i64], y: &[i64]) -> Json {
    let mk = |v: &[i64]| json!({"kind": "I64", "shape": [v.len()], "data": v.iter().map(|d| d.to_string()).collect::<Vec<_>>()});
    json!({"streams": {"A": mk(a), "X": mk(x), "Y": mk(y)}})
}

/// One assemble, many runs: the same contract `jotnar-repl` relies on.
#[test]
fn warm_assemble_reuses_bound_program_across_payloads() {
    let opts = AsmOptions::new(".");
    let bound = asm::assemble(SAXPY, &opts).unwrap();
    for (a, x, y, expected) in [
        (vec![3, 3, 3, 3], vec![1, 2, 3, 4], vec![10, 20, 30, 40], vec![13, 26, 39, 52]),
        (vec![2, 2], vec![5, 6], vec![1, 1], vec![11, 13]),
    ] {
        let feeds = feeds_from_json(&payload(&a, &x, &y)).unwrap();
        let got = jotnar::runtime::run(&bound, &feeds).unwrap();
        let out = feeds_to_json(&got, Some(&["OUT".to_string()])).unwrap();
        let data: Vec<String> = out["streams"]["OUT"]["data"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_str().unwrap().to_string())
            .collect();
        assert_eq!(
            data,
            expected.iter().map(|v| v.to_string()).collect::<Vec<_>>()
        );
    }
}

/// A bad payload is a per-request error, not a process failure: the bound
/// program still serves the next good payload.
#[test]
fn bad_payload_does_not_poison_warm_program() {
    let opts = AsmOptions::new(".");
    let bound = asm::assemble(SAXPY, &opts).unwrap();
    let bad = json!({"streams": {}});
    assert!(feeds_from_json(&bad).is_ok());
    // feeds_from_json succeeds (empty map); the failure surfaces at run().
    let feeds = feeds_from_json(&bad).unwrap();
    assert!(jotnar::runtime::run(&bound, &feeds).is_err());

    let good = payload(&[3], &[4], &[5]);
    let feeds = feeds_from_json(&good).unwrap();
    let got = jotnar::runtime::run(&bound, &feeds).unwrap();
    assert!(got.get("OUT").is_some());
}
