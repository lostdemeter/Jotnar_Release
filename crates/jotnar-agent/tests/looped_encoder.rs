use std::fs;

use jotnar::asm::{self, AsmOptions};
use jotnar_agent::{
    all_cases, calibration_dataset, case_views, run_round, run_round_with, split_cases, Harness,
    LoopedTaskEncoder, OutcomeKind, Split, TaskEncoder, LATENT_NAMES,
};

fn temp_dir(tag: &str) -> std::path::PathBuf {
    let p = std::env::temp_dir().join(format!(
        "jotnar-agent-looped-encoder-{tag}-{}",
        std::process::id()
    ));
    let _ = fs::remove_dir_all(&p);
    fs::create_dir_all(&p).unwrap();
    p
}

#[test]
fn looped_encoder_listings_assemble() {
    for loops in [1, 2, 3] {
        let e = LoopedTaskEncoder::prior(loops);
        asm::assemble(e.listings_file(), &AsmOptions::new("."))
            .unwrap_or_else(|err| panic!("loops={loops}: {err}"));
    }
}

#[test]
fn looped_encoder_uses_a_per_case_budget() {
    let base = Harness::default();
    let all = all_cases();
    let report = run_round(&base, &all, &temp_dir("budget"));
    let views = case_views(&all, &report.base);
    let e = LoopedTaskEncoder::prior(3);

    let matrix = views
        .iter()
        .find(|v| v.family == "matrix" && v.outcome == OutcomeKind::WrongRefusal)
        .expect("matrix failure");
    assert_eq!(e.loops_for(matrix), 2);

    let deep = views
        .iter()
        .find(|v| v.family == "elementwise" && v.outcome == OutcomeKind::WrongRefusal)
        .expect("deep failure");
    assert_eq!(e.loops_for(deep), 3);

    let correct = views
        .iter()
        .find(|v| v.outcome == OutcomeKind::Solved)
        .expect("solved control");
    assert_eq!(e.loops_for(correct), 1);
}

#[test]
fn looped_encoder_has_a_named_latent_and_scores() {
    let base = Harness::default();
    let all = all_cases();
    let report = run_round(&base, &all, &temp_dir("latent"));
    let views = case_views(&all, &report.base);
    let matrix = views
        .iter()
        .find(|v| v.family == "matrix" && v.outcome == OutcomeKind::WrongRefusal)
        .unwrap();
    let task = jotnar_agent::task_encoder::task_vector(matrix, &base);
    let e = LoopedTaskEncoder::prior(3);
    let latent = e.latent(&task, 2).unwrap();
    assert_eq!(latent.len(), LATENT_NAMES.len());
    let scores = e.scores(&task, 2).unwrap();
    assert_eq!(scores.len(), jotnar_agent::ACTIONS.len());
}

fn top_action(scores: &[i64]) -> usize {
    let mut order: Vec<usize> = (0..scores.len()).collect();
    order.sort_by_key(|i| std::cmp::Reverse(scores[*i]));
    order[0]
}

#[test]
fn per_case_crossover_one_loop_fails_two_loops_succeed() {
    let base = Harness::default();
    let all = all_cases();
    let report = run_round(&base, &all, &temp_dir("crossover"));
    let views = case_views(&all, &report.base);
    let matrix = views
        .iter()
        .find(|v| v.family == "matrix" && v.outcome == OutcomeKind::WrongRefusal)
        .expect("matrix failure");
    let task = jotnar_agent::task_encoder::task_vector(matrix, &base);

    let linear = TaskEncoder::prior();
    let linear_scores = linear.scores(&task).unwrap();
    assert_ne!(
        top_action(&linear_scores),
        0,
        "linear encoder should pick the decoy, scores={linear_scores:?}"
    );

    let looped = LoopedTaskEncoder::prior(3);
    let one = looped.scores(&task, 1).unwrap();
    assert_ne!(
        top_action(&one),
        0,
        "one loop should still pick the decoy, scores={one:?}"
    );
    let two = looped.scores(&task, 2).unwrap();
    assert_eq!(
        top_action(&two),
        0,
        "two loops should pick sum4, scores={two:?}"
    );
    let three = looped.scores(&task, 3).unwrap();
    assert_eq!(
        top_action(&three),
        0,
        "three loops should retain sum4, scores={three:?}"
    );
}

#[test]
fn deep_case_also_crosses_at_two_loops() {
    let base = Harness::default();
    let all = all_cases();
    let report = run_round(&base, &all, &temp_dir("deep-cross"));
    let views = case_views(&all, &report.base);
    let deep = views
        .iter()
        .find(|v| v.family == "elementwise" && v.outcome == OutcomeKind::WrongRefusal)
        .expect("deep failure");
    let task = jotnar_agent::task_encoder::task_vector(deep, &base);
    let e = LoopedTaskEncoder::prior(3);

    assert_eq!(
        top_action(&e.scores(&task, 1).unwrap()),
        3,
        "one loop should pick the stable decoy"
    );
    let two_top = top_action(&e.scores(&task, 2).unwrap());
    assert!(
        two_top == 1 || two_top == 5,
        "two loops should pick a deep action, got {two_top}"
    );
}

#[test]
fn family_budgets_override_the_heuristic() {
    let base = Harness::default();
    let all = all_cases();
    let report = run_round(&base, &all, &temp_dir("budgets"));
    let views = case_views(&all, &report.base);
    let deep = views
        .iter()
        .find(|v| v.family == "elementwise" && v.outcome == OutcomeKind::WrongRefusal)
        .unwrap();
    let mut budgets = std::collections::BTreeMap::new();
    budgets.insert("elementwise".to_string(), 2);
    budgets.insert("matrix".to_string(), 2);
    let e = LoopedTaskEncoder::prior(3).with_budgets(budgets);
    assert_eq!(e.loops_for(deep), 2);
}

#[test]
fn recurrence_region_preserves_the_crossover() {
    let base = Harness::default();
    let all = all_cases();
    let report = run_round(&base, &all, &temp_dir("region"));
    let views = case_views(&all, &report.base);
    let matrix = views
        .iter()
        .find(|v| v.family == "matrix" && v.outcome == OutcomeKind::WrongRefusal)
        .unwrap();
    let task = jotnar_agent::task_encoder::task_vector(matrix, &base);

    for (diag, cross) in [(4, -2), (5, -3), (6, -4)] {
        let e = LoopedTaskEncoder::with_recurrence(&TaskEncoder::prior(), 3, diag, cross);
        assert_ne!(
            top_action(&e.scores(&task, 1).unwrap()),
            0,
            "diag={diag} cross={cross} should miss at one loop"
        );
        assert_eq!(
            top_action(&e.scores(&task, 2).unwrap()),
            0,
            "diag={diag} cross={cross} should find sum4 at two loops"
        );
    }
}

#[test]
fn looped_encoder_adopts_the_corpus() {
    let out = temp_dir("adopt");
    let report = run_round_with(
        &Harness::default(),
        &all_cases(),
        &out,
        &LoopedTaskEncoder::prior(3),
    );
    assert!(report.adopted_flag);
    assert_eq!(report.adopted.correct(), 24);
    assert_eq!(report.adopted.wrong(), 0);
}

#[test]
fn fitted_linear_encoder_can_be_looped() {
    let base = Harness::default();
    let mut train = split_cases(Split::Train);
    train.extend(split_cases(Split::Calibration));
    let samples = calibration_dataset(&train, &base);
    let mut linear = TaskEncoder::prior();
    linear.fit(&samples, 10, 1).unwrap();
    let looped = LoopedTaskEncoder::from_linear(&linear, 3);
    let out = temp_dir("fitted");
    let report = run_round_with(&base, &all_cases(), &out, &looped);
    assert!(report.adopted_flag);
    assert_eq!(report.adopted.correct(), 24);
    assert_eq!(report.adopted.wrong(), 0);
}

#[test]
fn looped_encoder_weights_roundtrip() {
    let base = Harness::default();
    let all = all_cases();
    let report = run_round(&base, &all, &temp_dir("roundtrip"));
    let views = case_views(&all, &report.base);
    let matrix = views
        .iter()
        .find(|v| v.family == "matrix" && v.outcome == OutcomeKind::WrongRefusal)
        .unwrap();
    let task = jotnar_agent::task_encoder::task_vector(matrix, &base);

    let e = LoopedTaskEncoder::prior(3);
    let path = temp_dir("weights").join("looped-encoder.json");
    e.save(&path).unwrap();
    let back = LoopedTaskEncoder::load(&path).unwrap();
    assert_eq!(back.max_loops, 3);
    assert_eq!(e.scores(&task, 2).unwrap(), back.scores(&task, 2).unwrap());
}
