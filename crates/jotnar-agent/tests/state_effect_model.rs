use std::fs;

use jotnar::asm::{self, AsmOptions};
use jotnar_agent::{
    apply_action, case_views, evaluate, generated_corpus, generated_split_cases, hard_cases,
    run_round_with, state_effects, task_encoder::task_vector, BeamProposer, Harness, Split,
    StateEffectModel, StateEffectRow, ACTIONS, STATE_FEATURE_NAMES, STATE_MODEL_INPUTS,
    TASK_FEATURE_NAMES,
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
    let accuracy = model.fit(&rows, 20, 1);
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
    model.fit(&rows, 20, 1);

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
    let accuracy = model.fit(&rows, 20, 1);
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
    model.fit(&rows, 20, 1);
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
fn target_features_close_the_held_out_select_cases() {
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
    linear.fit(&train_rows, 20, 1);
    let held = linear.evaluate(&test_rows);
    assert!(held.groups >= 10, "held-out groups {}", held.groups);
    assert!(
        held.accuracy >= 0.95,
        "target features should close the held-out select/vec split, got {:.3}",
        held.accuracy
    );
}

fn without_target_features(rows: &[StateEffectRow]) -> Vec<StateEffectRow> {
    let start = TASK_FEATURE_NAMES.len() - 7;
    rows.iter()
        .cloned()
        .map(|mut row| {
            for feature in row.task.iter_mut().skip(start) {
                *feature = 0;
            }
            row
        })
        .collect()
}

#[test]
fn target_select_orientation_features_are_parsed() {
    let base = Harness::default();
    let cases = generated_corpus();
    for (id, min, max) in [
        ("gen-select-min", 1, 0),
        ("gen-select-max-cal", 0, 1),
        ("gen-select-min-cross-train", 1, 0),
    ] {
        let case = cases
            .iter()
            .find(|c| c.case.intent.id == id)
            .unwrap_or_else(|| panic!("missing {id}"))
            .case
            .clone();
        let report = evaluate(&base, std::slice::from_ref(&case), false);
        let views = case_views(std::slice::from_ref(&case), &report);
        let task = task_vector(&views[0], &base);
        assert_eq!(task[37], min, "{id} target_select_min");
        assert_eq!(task[38], max, "{id} target_select_max");
        assert_eq!(task[33], 1, "{id} target_iselect");
    }
}

#[test]
fn target_features_resolve_the_select_min_max_ambiguity() {
    // Two otherwise identical states: one target is select_min and one is
    // select_max. Only the target-program flags distinguish them, so the
    // matching family actions are unlearnable without those features.
    let mut rows = Vec::new();
    for (i, is_min) in [true, false].into_iter().enumerate() {
        let mut task = vec![0i64; TASK_FEATURE_NAMES.len()];
        task[33] = 1; // target_iselect
        if is_min {
            task[37] = 1; // target_select_min
        } else {
            task[38] = 1; // target_select_max
        }
        for action in [4usize, 6] {
            let correct = if is_min { action == 6 } else { action == 4 };
            rows.push(StateEffectRow {
                case: format!("select-{i}"),
                split: "train".to_string(),
                task: task.clone(),
                state_key: "base".to_string(),
                state: vec![0; STATE_FEATURE_NAMES.len()],
                action,
                label: ACTIONS[action].to_string(),
                parent_outcome: "wrong-refusal".to_string(),
                outcome: if correct { "solved" } else { "wrong-refusal" }.to_string(),
                parent_correct: false,
                correct,
                wrong: !correct,
                delta: i64::from(correct),
                steps: 1,
            });
        }
    }

    let mut with_target = StateEffectModel::prior();
    with_target.fit(&rows, 20, 1);
    let with_eval = with_target.evaluate(&rows);
    assert!(
        with_eval.accuracy >= 0.99,
        "target features should separate min/max, got {:.3}",
        with_eval.accuracy
    );

    let no_target_rows = without_target_features(&rows);
    let mut without_target = StateEffectModel::prior();
    without_target.fit(&no_target_rows, 20, 1);
    let without_eval = without_target.evaluate(&no_target_rows);
    assert!(
        with_eval.accuracy > without_eval.accuracy + 0.25,
        "target features should be the separator: {:.3} vs {:.3}",
        with_eval.accuracy,
        without_eval.accuracy
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
    assert_eq!(model.feature_len(), STATE_MODEL_INPUTS + 18);
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
