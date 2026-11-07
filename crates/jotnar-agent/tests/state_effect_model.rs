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
    let reduced = apply_action(&base, 0).unwrap().harness;
    let a = base.state_features();
    let b = reduced.state_features();
    assert_eq!(a.len(), STATE_FEATURE_NAMES.len());
    assert_eq!(b.len(), STATE_FEATURE_NAMES.len());
    assert_ne!(a, b);
    assert_eq!(a[0], 0, "base has no reduce_sum consumer");
    assert_eq!(b[0], 1, "the reduce_sum state has the consumer");
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
        .expect("reduce_sum transition from the base state for the matrix case");
    assert!(
        matrix.correct,
        "reduce_sum should solve the generated matrix case"
    );
    assert!(!matrix.parent_correct);
    assert_eq!(matrix.delta, 1);

    // From the reduce_sum state, the consumer is already in the subset, so
    // the transition table does not offer it again.
    let reduced_state = apply_action(&base, 0).unwrap().harness.fingerprint();
    let from_reduced = rows
        .iter()
        .find(|row| row.case == "gen-matmul-colsum" && row.state_key == reduced_state)
        .expect("a transition from the composed state");
    assert_eq!(
        from_reduced.state[0], 1,
        "the state says reduce_sum is present"
    );
    let repeat = rows
        .iter()
        .find(|row| row.state_key == reduced_state && row.action == 0);
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
    let rows = state_effects(&train, &base, 3);
    let mut model = StateEffectModel::prior();
    let accuracy = model.fit(&rows, 10, 1);
    assert!(accuracy >= 0.99, "discriminating accuracy {accuracy}");

    let proposer = BeamProposer::new(model, 8, 4);
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
    let rows = state_effects(&train, &base, 3);
    let mut model = StateEffectModel::prior();
    model.fit(&rows, 10, 1);

    let proposer = BeamProposer::new(model, 8, 4);
    let report = run_round_with(&base, &train, &temp_dir("hard-coverage"), &proposer);
    assert!(report.adopted_flag);
    assert_eq!(report.adopted.correct(), train.len());
    assert_eq!(report.adopted.wrong(), 0);
    assert!(
        report.chosen_label.contains("path-union"),
        "the deep table should select the learned path-union, got {}",
        report.chosen_label
    );

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
fn learned_ranking_solves_distant_shapes_without_safety_nets() {
    let base = Harness::default();
    let train = generated_train();
    let heldout = generated_split_cases(Split::Heldout);
    let hard = hard_cases();
    let rows = state_effects(&train, &base, 3);
    let mut model = StateEffectModel::prior();
    let accuracy = model.fit(&rows, 10, 1);
    assert!(accuracy >= 0.99, "discriminating accuracy {accuracy}");

    // The learned model ranks the taught consumer first on every held-out
    // distant shape, from the task vector and the base harness state alone.
    let base_report = evaluate(&base, &hard, false);
    let views = jotnar_agent::case_views(&hard, &base_report);
    for (id, action) in [
        ("hard-matmul3-sum", 0usize),
        ("hard-matmul4-sum", 0),
        ("hard-select16-sum", 0),
    ] {
        let index = hard
            .iter()
            .position(|case| case.intent.id == id)
            .unwrap_or_else(|| panic!("missing hard case {id}"));
        let task = jotnar_agent::task_encoder::task_vector(&views[index], &base);
        let scores = model.scores(&task, &base.state_features());
        let top = (0..scores.len())
            .filter(|action| jotnar_agent::action_applies(*action, &task))
            .max_by_key(|action| scores[*action])
            .unwrap_or(0);
        assert_eq!(
            top, action,
            "{id} should rank {} first: {scores:?}",
            ACTIONS[action]
        );
        assert!(scores[action] > 0, "{id} score {}", scores[action]);
    }

    // Learned-only: disable the compose-completeness, exhaustive-singles, and
    // exhaustive-pair safety nets. The learned per-case union must still carry
    // the wide consumers into the adopted harness.
    let proposer = BeamProposer::new(model, 8, 4).learned_only();
    let report = run_round_with(&base, &train, &temp_dir("learned-only"), &proposer);
    assert!(report.adopted_flag, "learned-only adoption");
    assert_eq!(report.adopted.correct(), train.len());
    assert_eq!(report.adopted.wrong(), 0);
    assert!(
        report
            .adopted_harness
            .compose_consumers
            .iter()
            .any(|c| c == "reduce_sum"),
        "learned-only harness should keep reduce_sum"
    );
    assert!(
        report
            .adopted_harness
            .compose_consumers
            .iter()
            .any(|c| c == "reduce_sum"),
        "learned-only harness should keep reduce_sum"
    );

    let held = evaluate(&report.adopted_harness, &heldout, false);
    assert_eq!(
        held.correct(),
        heldout.len(),
        "held-out {}/{}",
        held.correct(),
        heldout.len()
    );
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
    let rows = state_effects(&train, &base, 3);
    let mut model = StateEffectModel::prior();
    model.fit(&rows, 10, 1);
    let proposer = BeamProposer::new(model, 8, 4);
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
    // this stream: reduce_sum (matrix), depth/refinements (deep), select_min
    // (select). The single actions cannot solve all three failures.
    assert!(
        report.candidates.iter().any(|candidate| {
            candidate
                .harness
                .compose_consumers
                .iter()
                .any(|c| c == "reduce_sum")
                && (candidate.harness.max_depth > 2 || candidate.harness.max_refinements > 1)
                && candidate.harness.families.iter().any(|f| f == "select_min")
        }),
        "the beam should reach a reduce_sum + depth/refinements + select_min triple"
    );
    assert_eq!(ACTIONS.len(), 10);
    let _ = generated_corpus();
}

#[test]
fn held_out_case_split_is_a_mechanism_wall() {
    let base = Harness::default();
    let all = generated_train();
    let held_ids = [
        "gen-select-max-cal",
        "gen-select-min",
        "gen-vec16-sum-train",
        "gen-vec9-sum-train",
    ];
    let rows = state_effects(&all, &base, 2);
    let train_rows: Vec<_> = rows
        .iter()
        .filter(|row| !held_ids.contains(&row.case.as_str()))
        .cloned()
        .collect();
    let test_rows: Vec<_> = rows
        .iter()
        .filter(|row| held_ids.contains(&row.case.as_str()))
        .cloned()
        .collect();

    let mut linear = StateEffectModel::prior();
    linear.fit(&train_rows, 10, 1);
    let held = linear.evaluate(&test_rows);
    assert!(held.groups >= 10, "held-out groups {}", held.groups);
    assert!(
        held.accuracy < 0.9,
        "the held-out select/vec mechanisms should expose a wall, got {:.3}",
        held.accuracy
    );

    // WBP lag 35 does not break this wall, and the residual is ergodic.
    let mut inter = StateEffectModel::prior().with_interactions(vec![35]);
    inter.fit(&train_rows, 10, 1);
    let inter_held = inter.evaluate(&test_rows);
    assert!(
        (inter_held.accuracy - held.accuracy).abs() < 0.05,
        "lag 35 should be neutral on the mechanism wall: {:.3} vs {:.3}",
        held.accuracy,
        inter_held.accuracy
    );
    let r = jotnar_agent::wall_breaking::resfrac(&held.residual, 2).unwrap();
    let mut sorted = held.residual.clone();
    sorted.sort_unstable();
    let rs = jotnar_agent::wall_breaking::resfrac(&sorted, 2).unwrap();
    eprintln!(
        "mechanism wall: rho(case-order)={:.3} rho(sorted)={:.3}",
        r.rho, rs.rho
    );
    assert!(r.rho.is_finite() && rs.rho.is_finite());
    // The WBP ordering caveat: this residual is ordered by (case, state), so the
    // rho is a case-block signal, not a feature-lag signal. Lag 35 is neutral.
    assert!(
        (r.rho - rs.rho).abs() > 0.05 || r.rho > 0.85,
        "the diagnosis should be sensitive to ordering or genuinely ergodic: {:.3} vs {:.3}",
        r.rho,
        rs.rho
    );
}

#[test]
fn repeated_depth_and_refinements_are_reachable() {
    let states = jotnar_agent::reachable_states(&Harness::default(), 3);
    assert!(
        states
            .iter()
            .any(|(_, path, _)| path.iter().filter(|a| **a == 1).count() >= 2),
        "depth + 1 should repeat in the bounded BFS"
    );
    assert!(
        states
            .iter()
            .any(|(_, path, _)| path.iter().filter(|a| **a == 5).count() >= 2),
        "refinements + 1 should repeat in the bounded BFS"
    );
    assert!(
        states.iter().any(|(_, _, harness)| harness.max_depth >= 4),
        "a repeated depth state should reach max_depth >= 4"
    );
    assert!(
        states
            .iter()
            .any(|(_, _, harness)| harness.max_refinements >= 3),
        "a repeated refinements state should reach max_refinements >= 3"
    );
}

#[test]
fn wbp_interaction_features_are_an_ordinary_listing() {
    let model = StateEffectModel::prior().with_interactions(vec![35]);
    assert_eq!(model.feature_len(), STATE_MODEL_INPUTS + 11);
    let bound = asm::assemble(&model.listing, &AsmOptions::new("."))
        .expect("the WBP interaction model must assemble");
    assert_eq!(bound.inputs.len(), STATE_MODEL_INPUTS);
    // lag 35 maps task[6] (depth_delta) onto state[9] (refinements) at i=41.
    assert!(
        model.listing.contains("x35_41"),
        "lag 35 should include the refinements x depth_delta product"
    );
    let scores = model.scores(
        &vec![0; TASK_FEATURE_NAMES.len()],
        &vec![0; STATE_FEATURE_NAMES.len()],
    );
    assert_eq!(scores.len(), ACTIONS.len());
}
