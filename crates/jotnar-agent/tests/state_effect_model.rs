use std::fs;

use jotnar::asm::{self, AsmOptions};
use jotnar_agent::{
    apply_action, evaluate, generated_corpus, generated_split_cases, hard_cases, run_round_with,
    state_effects, BeamProposer, Harness, Split, StateEffectModel, ACTIONS, STATE_FEATURE_NAMES,
    STATE_MODEL_INPUTS, TASK_FEATURE_NAMES,
};

fn temp_dir(tag: &str) -> std::path::PathBuf {
    let p = std::env::temp_dir().join(format!(
        "jotnar-agent-state-effect-{tag}-{}",
        std::process::id()
    ));
    let _ = fs::remove_dir_all(&p);
    fs::create_dir_all(&p).unwrap();
    p
}

fn generated_train() -> Vec<jotnar_agent::Case> {
    let mut train = generated_split_cases(Split::Train);
    train.extend(generated_split_cases(Split::Calibration));
    train
}

#[test]
fn state_features_distinguish_the_composed_harness() {
    let base = Harness::default();
    let sum4 = apply_action(&base, 0).unwrap().harness;
    let a = base.state_features();
    let b = sum4.state_features();
    assert_eq!(a.len(), STATE_FEATURE_NAMES.len());
    assert_eq!(b.len(), STATE_FEATURE_NAMES.len());
    assert_ne!(a, b);
    assert_eq!(a[0], 0, "base has no sum4 consumer");
    assert_eq!(b[0], 1, "the sum4 state has the consumer");
}

#[test]
fn transition_table_records_state_dependent_effects() {
    let train = generated_train();
    let base = Harness::default();
    // Depth 2 exposes both the base state and the states reached by one
    // action, so the row can carry a non-trivial harness state.
    let rows = state_effects(&train, &base, 2);
    assert!(!rows.is_empty());
    assert!(rows
        .iter()
        .all(|row| row.state.len() == STATE_FEATURE_NAMES.len()));
    assert!(rows
        .iter()
        .all(|row| row.task.len() == TASK_FEATURE_NAMES.len()));

    let matrix = rows
        .iter()
        .find(|row| row.case == "gen-matmul-colsum" && row.action == 0 && row.state[0] == 0)
        .expect("sum4 transition from the base state for the matrix case");
    assert!(
        matrix.correct,
        "sum4 should solve the generated matrix case"
    );
    assert!(!matrix.parent_correct);
    assert_eq!(matrix.delta, 1);

    // From the sum4 state, sum4 is already in the subset, so the transition
    // table does not offer it again.
    let sum4_state = apply_action(&base, 0).unwrap().harness.fingerprint();
    let from_sum4 = rows
        .iter()
        .find(|row| row.case == "gen-matmul-colsum" && row.state_key == sum4_state)
        .expect("a transition from the composed state");
    assert_eq!(from_sum4.state[0], 1, "the state says sum4 is present");
    let repeat = rows
        .iter()
        .find(|row| row.state_key == sum4_state && row.action == 0);
    assert!(repeat.is_none());
}

#[test]
fn state_model_is_an_ordinary_listing() {
    let model = StateEffectModel::prior();
    let bound = asm::assemble(&model.listing, &AsmOptions::new("."))
        .expect("the state-effect model must assemble like any other listing");
    assert_eq!(bound.inputs.len(), STATE_MODEL_INPUTS);
    assert_eq!(
        TASK_FEATURE_NAMES.len() + STATE_FEATURE_NAMES.len(),
        STATE_MODEL_INPUTS
    );
}

#[test]
fn state_model_closes_the_generated_heldout() {
    let base = Harness::default();
    let train = generated_train();
    let heldout = generated_split_cases(Split::Heldout);
    let rows = state_effects(&train, &base, 2);
    let mut model = StateEffectModel::prior();
    let accuracy = model.fit(&rows, 10, 1);
    assert!(accuracy >= 0.99, "discriminating accuracy {accuracy}");

    let proposer = BeamProposer::new(model, 4, 2);
    let report = run_round_with(&base, &train, &temp_dir("adopt"), &proposer);
    assert!(report.adopted_flag);
    assert_eq!(report.adopted.correct(), train.len());
    assert_eq!(report.adopted.wrong(), 0);

    let held = evaluate(&report.adopted_harness, &heldout, false);
    assert_eq!(
        held.correct(),
        heldout.len(),
        "held-out {}/{}",
        held.correct(),
        heldout.len()
    );
    assert_eq!(held.wrong(), 0);
}

#[test]
fn beam_coverage_generalizes_to_hard_compositions() {
    let base = Harness::default();
    let train = generated_train();
    let heldout = generated_split_cases(Split::Heldout);
    let hard = hard_cases();
    let rows = state_effects(&train, &base, 2);
    let mut model = StateEffectModel::prior();
    model.fit(&rows, 10, 1);

    let proposer = BeamProposer::new(model, 4, 3);
    let report = run_round_with(&base, &train, &temp_dir("hard-coverage"), &proposer);
    assert!(report.adopted_flag);
    assert_eq!(report.adopted.correct(), train.len());
    assert_eq!(report.adopted.wrong(), 0);

    let held = evaluate(&report.adopted_harness, &heldout, false);
    assert_eq!(held.correct(), heldout.len());

    // The hard set is audit-only; it is never used for selection. The coverage
    // candidate happens to cover the three-action composition it needs.
    let hard_report = evaluate(&report.adopted_harness, &hard, false);
    assert_eq!(
        hard_report.correct(),
        hard.len(),
        "hard audit {}/{} (wrong {})",
        hard_report.correct(),
        hard.len(),
        hard_report.wrong()
    );
}

#[test]
fn beam_proposer_covers_triples() {
    let base = Harness::default();
    let train = generated_train();
    let rows = state_effects(&train, &base, 2);
    let mut model = StateEffectModel::prior();
    model.fit(&rows, 10, 1);
    let proposer = BeamProposer::new(model, 4, 2);
    let report = run_round_with(&base, &train, &temp_dir("triples"), &proposer);
    let distinct = report
        .candidates
        .iter()
        .map(|candidate| {
            candidate.harness.families.len() + candidate.harness.compose_consumers.len()
        })
        .max()
        .unwrap_or(0);
    assert!(distinct > 0);
    assert!(report.chosen_label.contains("beam") || report.chosen_label.contains("compose"));
    // The chosen harness must be at least a three-capability composition on
    // this stream: sum4 (matrix), depth/refinements (deep), select_min
    // (select). The single actions cannot solve all three failures.
    assert!(
        report.candidates.iter().any(|candidate| {
            candidate
                .harness
                .compose_consumers
                .iter()
                .any(|c| c == "sum4")
                && (candidate.harness.max_depth > 2 || candidate.harness.max_refinements > 1)
                && candidate.harness.families.iter().any(|f| f == "select_min")
        }),
        "the beam should reach a sum4 + depth/refinements + select_min triple"
    );
    assert_eq!(ACTIONS.len(), 10);
    let _ = generated_corpus();
}
