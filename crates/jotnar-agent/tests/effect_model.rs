use std::fs;

use jotnar_agent::{
    action_effects, evaluate, generated_split_cases, run_round_with, EffectModel, Harness, Split,
    WorldModelProposer,
};

/// The legacy effect model is a single-action table; the repeated-depth s64
/// training case is out of its scope. The state model + path union owns it.
fn single_action_train() -> Vec<jotnar_agent::Case> {
    let mut train = generated_split_cases(Split::Train);
    train.extend(generated_split_cases(Split::Calibration));
    train.retain(|case| {
        !matches!(
            case.intent.id.as_str(),
            "gen-elem-saxpy-s64-train"
                | "gen-select-min-cross-train"
                | "gen-select-max-cross-train"
        )
    });
    train
}

fn temp_dir(tag: &str) -> std::path::PathBuf {
    let p = std::env::temp_dir().join(format!(
        "jotnar-agent-effect-model-{tag}-{}",
        std::process::id()
    ));
    let _ = fs::remove_dir_all(&p);
    fs::create_dir_all(&p).unwrap();
    p
}

#[test]
fn effect_model_ranks_the_solving_actions() {
    let train = single_action_train();
    let effects = action_effects(&train, &Harness::default());
    let mut model = EffectModel::prior();
    let accuracy = model.fit(&effects, 10, 1);
    assert!(accuracy >= 0.99, "effect model accuracy {accuracy}");

    let matrix = effects
        .iter()
        .find(|case| case.case == "gen-matmul-sum-cal")
        .expect("matrix case");
    let scores = model.scores(&matrix.task);
    let top = matrix
        .actions
        .iter()
        .filter(|a| a.applicable && jotnar_agent::action_applies(a.index, &matrix.task))
        .max_by_key(|a| scores[a.index])
        .expect("an applicable action");
    assert!(
        top.label.contains("reduce_sum"),
        "top matrix action should be reduce_sum, got {}",
        top.label
    );

    let select = effects
        .iter()
        .find(|case| case.case == "gen-select-min")
        .expect("select case");
    let scores = model.scores(&select.task);
    let top = select
        .actions
        .iter()
        .filter(|a| a.applicable && jotnar_agent::action_applies(a.index, &select.task))
        .max_by_key(|a| scores[a.index])
        .expect("an applicable action");
    assert!(
        top.label.contains("select_min"),
        "top select action should be select_min, got {}",
        top.label
    );
}

#[test]
fn world_model_composes_the_generated_stream() {
    let base = Harness::default();
    let train = single_action_train();
    let heldout = generated_split_cases(Split::Heldout);
    let effects = action_effects(&train, &base);
    let mut model = EffectModel::prior();
    model.fit(&effects, 10, 1);
    let proposer = WorldModelProposer::new(model, 4);
    let report = run_round_with(&base, &train, &temp_dir("adopt"), &proposer);
    assert!(report.adopted_flag);
    assert_eq!(report.adopted.correct(), train.len());
    assert_eq!(report.adopted.wrong(), 0);
    let held = evaluate(&report.adopted_harness, &heldout, false);
    assert!(
        held.correct() == heldout.len(),
        "held-out {}/{}",
        held.correct(),
        heldout.len()
    );
    assert_eq!(held.wrong(), 0);
}
