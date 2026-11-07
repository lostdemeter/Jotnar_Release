use std::fs;

use jotnar::asm::{self, AsmOptions};
use jotnar_agent::{
    calibration_cases, cases, run_round, run_round_with, Harness, JotnarProposer, Proposer,
    ProposerWeights, ACTIONS,
};

fn temp_dir(tag: &str) -> std::path::PathBuf {
    let p = std::env::temp_dir().join(format!("jotnar-agent-prop-{tag}-{}", std::process::id()));
    let _ = fs::remove_dir_all(&p);
    fs::create_dir_all(&p).unwrap();
    p
}

#[test]
fn proposer_is_an_ordinary_listing() {
    let p = JotnarProposer::prior();
    let bound = asm::assemble(&p.listing, &AsmOptions::new("."))
        .expect("the proposer must assemble like any other listing");
    assert_eq!(bound.inputs.len(), jotnar_agent::FEATURE_NAMES.len());
    assert!(bound.instrs.len() > ACTIONS.len());
}

#[test]
fn proposer_scores_choose_reduce_sum_for_the_base_failure() {
    let base = Harness::default();
    let report = run_round(&base, &cases(), &temp_dir("score"));
    let p = JotnarProposer::prior();
    let proposals = p.propose(&base, &report.base, &[]);
    assert!(
        proposals.iter().any(|m| m.label.contains("reduce_sum")),
        "expected reduce_sum in {:?}",
        proposals.iter().map(|m| &m.label).collect::<Vec<_>>()
    );
    let f = jotnar_agent::proposer::features(&base, &report.base, &[]);
    let scores = p.scores(&f).unwrap();
    assert_eq!(scores.len(), ACTIONS.len());
    assert!(
        scores[0] > scores[1],
        "reduce_sum should outrank depth here, scores={scores:?}"
    );
}

#[test]
fn linear_proposer_cannot_solve_both_failures() {
    let out = temp_dir("adopt");
    let report = run_round_with(
        &Harness::default(),
        &cases(),
        &out,
        &JotnarProposer::prior(),
    );
    assert!(
        !report.adopted_flag,
        "the simple model proposes the decoy second and cannot reach a zero-wrong harness"
    );
    assert_eq!(report.adopted.correct(), report.base.correct());
}

#[test]
fn fit_from_report_reinforces_the_winning_action() {
    let base = Harness::default();
    let calibration = run_round(&base, &calibration_cases(), &temp_dir("calibration"));
    let mut p = JotnarProposer::prior();
    let before = p.weights[0][0];
    let changes = p.fit_from_report(&base, &calibration, &[], 1);
    assert!(
        changes.iter().any(|c| c.contains("reduce_sum")),
        "expected a reduce_sum update, got {changes:?}"
    );
    assert!(
        p.weights[0][0] > before,
        "the reduce_sum refusal weight should rise"
    );
    assert!(p.bias[0] > 0, "the reduce_sum bias should rise");
}

#[test]
fn weights_roundtrip_through_json() {
    let p = JotnarProposer::prior();
    let out = temp_dir("weights").join("proposer.weights.json");
    p.save(&out).unwrap();
    let back = JotnarProposer::load(&out).unwrap();
    assert_eq!(p.weights, back.weights);
    assert_eq!(p.bias, back.bias);
    assert_eq!(p.listing, back.listing);

    let bad = ProposerWeights {
        weights: vec![vec![0; 2]],
        bias: vec![0],
    };
    assert!(JotnarProposer::from_weights(bad).is_err());
}
