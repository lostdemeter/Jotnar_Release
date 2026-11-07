use std::fs;

use jotnar_agent::{
    evaluate, fit_joint, generated_split_cases, run_round_with, state_effects, Backbone, Harness,
    JointProposer, Split, StateEffectModel, TaskEncoder, ACTIONS, LATENT_NAMES,
};

fn temp_dir(tag: &str) -> std::path::PathBuf {
    let p = std::env::temp_dir().join(format!("jotnar-agent-joint-{tag}-{}", std::process::id()));
    let _ = fs::remove_dir_all(&p);
    fs::create_dir_all(&p).unwrap();
    p
}

fn train_cases() -> Vec<jotnar_agent::Case> {
    let mut train = generated_split_cases(Split::Train);
    train.extend(generated_split_cases(Split::Calibration));
    train
}

#[test]
fn joint_model_fits_action_prototypes_and_closes_the_stream() {
    let base = Harness::default();
    let train = train_cases();
    let heldout = generated_split_cases(Split::Heldout);
    let rows = state_effects(&train, &base, 3);
    let mut effect = StateEffectModel::prior();
    effect.fit(&rows, 20, 1);

    let named = TaskEncoder::prior();
    let backbone = Backbone::named(&named.w_enc, &named.b_enc);
    let joint: JointProposer = fit_joint(effect, backbone, &rows, 8, 4);

    assert_eq!(joint.prototypes.len(), ACTIONS.len());
    assert!(joint
        .prototypes
        .iter()
        .all(|prototype| prototype.len() == LATENT_NAMES.len()));
    assert!(
        joint
            .prototypes
            .iter()
            .any(|prototype| prototype.iter().any(|v| *v != 0)),
        "at least one action prototype should be fitted"
    );

    let report = run_round_with(&base, &train, &temp_dir("adopt"), &joint);
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
fn joint_score_combines_effect_and_latent() {
    let rows = state_effects(&train_cases(), &Harness::default(), 1);
    let mut effect = StateEffectModel::prior();
    effect.fit(&rows, 5, 1);
    let named = TaskEncoder::prior();
    let backbone = Backbone::named(&named.w_enc, &named.b_enc);
    let mut joint = fit_joint(effect.clone(), backbone, &rows, 2, 1);

    let task = rows[0].task.clone();
    let state = rows[0].state.clone();
    let before = joint.scores(&task, &state);
    assert_eq!(before.len(), ACTIONS.len());

    // Changing only the latent prototype must change the joint score but not
    // the pure effect score.
    let effect_scores = joint.effect.scores(&task, &state);
    joint.prototypes[0] = vec![1; LATENT_NAMES.len()];
    let after = joint.scores(&task, &state);
    assert_eq!(joint.effect.scores(&task, &state), effect_scores);
    assert_ne!(before, after);
}
