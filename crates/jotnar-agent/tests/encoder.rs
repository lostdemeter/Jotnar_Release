use std::fs;

use jotnar::asm::{self, AsmOptions};
use jotnar_agent::{
    all_cases, case_views, evaluate, run_round, run_round_with, Harness, OutcomeKind, Proposer,
    TaskEncoder, LATENT_NAMES, TASK_FEATURE_NAMES,
};

fn temp_dir(tag: &str) -> std::path::PathBuf {
    let p = std::env::temp_dir().join(format!("jotnar-agent-encoder-{tag}-{}", std::process::id()));
    let _ = fs::remove_dir_all(&p);
    fs::create_dir_all(&p).unwrap();
    p
}

#[test]
fn encoder_is_an_ordinary_listing() {
    let e = TaskEncoder::prior();
    let bound = asm::assemble(&e.listing, &AsmOptions::new("."))
        .expect("the task encoder must assemble like any other listing");
    assert_eq!(bound.inputs.len(), TASK_FEATURE_NAMES.len());
    assert_eq!(LATENT_NAMES.len(), 9);
}

#[test]
fn encoder_reads_a_named_latent_and_scores_actions() {
    let base = Harness::default();
    let all = all_cases();
    let report = run_round(&base, &all, &temp_dir("views"));
    let views = case_views(&all, &report.base);
    let e = TaskEncoder::prior();

    let matrix = views
        .iter()
        .find(|v| v.family == "matrix" && v.outcome == OutcomeKind::WrongRefusal)
        .expect("matrix failure");
    let task = jotnar_agent::task_encoder::task_vector(matrix, &base);
    assert_eq!(task.len(), TASK_FEATURE_NAMES.len());
    let latent = e.latent(&task).unwrap();
    assert_eq!(latent.len(), LATENT_NAMES.len());
    let scores = e.scores(&task).unwrap();
    assert!(
        scores[0] < *scores.iter().max().unwrap(),
        "the linear/one-loop encoder should prefer the decoy on a matrix case, scores={scores:?}"
    );

    let deep = views
        .iter()
        .find(|v| v.family == "elementwise" && v.outcome == OutcomeKind::WrongRefusal)
        .expect("deep failure");
    let deep_scores = e
        .scores(&jotnar_agent::task_encoder::task_vector(deep, &base))
        .unwrap();
    let deep_top = deep_scores
        .iter()
        .enumerate()
        .max_by_key(|(_, score)| *score)
        .map(|(i, _)| i)
        .unwrap();
    assert_eq!(
        deep_top, 3,
        "linear encoder should prefer the stable decoy on a deep case too, scores={deep_scores:?}"
    );
}

#[test]
fn linear_encoder_misses_the_matrix_crossover() {
    let base = Harness::default();
    let all = all_cases();
    let report = run_round(&base, &all, &temp_dir("union"));
    let views = case_views(&all, &report.base);
    let e = TaskEncoder::prior();
    let proposals = e.propose_cases(&views, &base, &[]).expect("case proposals");
    assert_eq!(proposals.len(), 1);
    assert!(proposals[0].label.contains("case-union"));
    let adopted = evaluate(&proposals[0].harness, &all, false);
    assert!(
        adopted.correct() < 24,
        "the linear encoder should not solve the matrix case without looping"
    );
}

#[test]
fn linear_encoder_needs_the_loop() {
    let out = temp_dir("adopt");
    let report = run_round_with(
        &Harness::default(),
        &all_cases(),
        &out,
        &TaskEncoder::prior(),
    );
    assert!(
        !report.adopted_flag,
        "the linear prior should stop short of a zero-wrong harness"
    );
}
