use std::fs;

use jotnar::asm::{self, AsmOptions};
use jotnar_agent::{
    evaluate, generated_split_cases, run_round_with, state_effects, BeamProposer, CostModel,
    Harness, Split, StateEffectModel, StateEffectRow, STATE_FEATURE_NAMES, STATE_MODEL_INPUTS,
    TASK_FEATURE_NAMES,
};

fn temp_dir(tag: &str) -> std::path::PathBuf {
    let p = std::env::temp_dir().join(format!("jotnar-agent-cost-{tag}-{}", std::process::id()));
    let _ = fs::remove_dir_all(&p);
    fs::create_dir_all(&p).unwrap();
    p
}

fn row(action: usize, steps: usize) -> StateEffectRow {
    StateEffectRow {
        case: "synthetic".into(),
        split: "train".into(),
        task: vec![0; TASK_FEATURE_NAMES.len()],
        state_key: format!("state-{action}"),
        state: vec![0; STATE_FEATURE_NAMES.len()],
        action,
        label: format!("action-{action}"),
        parent_outcome: "wrong-refusal".into(),
        outcome: "solved".into(),
        parent_correct: false,
        correct: true,
        wrong: false,
        delta: 1,
        steps,
    }
}

#[test]
fn cost_model_is_an_ordinary_listing() {
    let model = CostModel::prior();
    let bound = asm::assemble(&model.listing, &AsmOptions::new("."))
        .expect("the cost head must assemble like any other listing");
    assert_eq!(bound.inputs.len(), STATE_MODEL_INPUTS);
    assert!(model.listing.contains("IDIV"));
}

#[test]
fn cost_model_ranks_cheaper_actions_lower() {
    let rows = vec![row(0, 1), row(1, 4)];
    let mut model = CostModel::prior();
    let mae = model.fit(&rows, 1e-6);
    assert!(mae < 1.0, "cost MAE {mae}");
    let scores = model.scores(
        &vec![0; TASK_FEATURE_NAMES.len()],
        &vec![0; STATE_FEATURE_NAMES.len()],
    );
    assert!(scores[0] < scores[1], "scores {scores:?}");
    assert_eq!(scores.len(), jotnar_agent::ACTIONS.len());
}

#[test]
fn cost_aware_beam_still_adopts_the_stream() {
    let base = Harness::default();
    let mut train = generated_split_cases(Split::Train);
    train.extend(generated_split_cases(Split::Calibration));
    let heldout = generated_split_cases(Split::Heldout);
    let rows = state_effects(&train, &base, 2);

    let mut effect = StateEffectModel::prior();
    effect.fit(&rows, 10, 1);
    let mut cost = CostModel::prior();
    let mae = cost.fit(&rows, 1.0);
    assert!(mae < 1.0, "cost MAE {mae}");

    let proposer = BeamProposer::new(effect, 4, 3).with_cost(cost, 4);
    let report = run_round_with(&base, &train, &temp_dir("adopt"), &proposer);
    assert!(report.adopted_flag);
    assert_eq!(report.adopted.correct(), train.len());
    assert_eq!(report.adopted.wrong(), 0);
    let held = evaluate(&report.adopted_harness, &heldout, false);
    assert_eq!(held.correct(), heldout.len());
}
