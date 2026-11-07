use std::collections::{BTreeMap, BTreeSet};
use std::fs;

use jotnar::asm::{self, AsmOptions};
use jotnar_agent::{
    all_cases, calibration_dataset, case_views, run_round, run_round_with, split_cases,
    BudgetPredictor, Harness, LoopedTaskEncoder, OutcomeKind, Split,
};

fn temp_dir(tag: &str) -> std::path::PathBuf {
    let p = std::env::temp_dir().join(format!("jotnar-agent-budget-{tag}-{}", std::process::id()));
    let _ = fs::remove_dir_all(&p);
    fs::create_dir_all(&p).unwrap();
    p
}

fn top_action(scores: &[i64]) -> usize {
    let mut order: Vec<usize> = (0..scores.len()).collect();
    order.sort_by_key(|i| std::cmp::Reverse(scores[*i]));
    order[0]
}

fn fit_predictor() -> (BudgetPredictor, Harness) {
    let base = Harness::default();
    let mut train = split_cases(Split::Train);
    train.extend(split_cases(Split::Calibration));
    let samples = calibration_dataset(&train, &base);
    let mut positives: BTreeMap<String, BTreeSet<usize>> = BTreeMap::new();
    for sample in &samples {
        if sample.delta > 0 {
            positives
                .entry(sample.case_id.clone())
                .or_default()
                .insert(sample.action);
        }
    }
    let report = run_round(&base, &train, &temp_dir("fit"));
    let views = case_views(&train, &report.base);
    let looped = LoopedTaskEncoder::prior(3);
    let mut data = Vec::new();
    for view in &views {
        let task = jotnar_agent::task_encoder::task_vector(view, &base);
        let label = if view.outcome != OutcomeKind::WrongRefusal {
            1
        } else {
            let pos = positives.get(&view.case_id);
            (1..=3)
                .find(|loops| {
                    let Some(pos) = pos else {
                        return false;
                    };
                    looped
                        .scores(&task, *loops)
                        .map(|s| pos.contains(&top_action(&s)))
                        .unwrap_or(false)
                })
                .unwrap_or(3)
        };
        data.push((task, label));
    }
    let mut predictor = BudgetPredictor::prior(3);
    let correct = predictor.fit(&data, 10, 1).unwrap();
    assert_eq!(
        correct,
        data.len(),
        "predictor should fit the calibration labels"
    );
    (predictor, base)
}

#[test]
fn budget_listing_assembles() {
    let p = BudgetPredictor::prior(3);
    asm::assemble(&p.listing, &AsmOptions::new(".")).expect("budget listing assembles");
}

#[test]
fn budget_predictor_learns_two_loops_for_the_failures() {
    let (predictor, base) = fit_predictor();
    let all = all_cases();
    let report = run_round(&base, &all, &temp_dir("predict"));
    let views = case_views(&all, &report.base);

    let matrix = views
        .iter()
        .find(|v| v.family == "matrix" && v.outcome == OutcomeKind::WrongRefusal)
        .unwrap();
    let deep = views
        .iter()
        .find(|v| v.family == "elementwise" && v.outcome == OutcomeKind::WrongRefusal)
        .unwrap();
    let matrix_task = jotnar_agent::task_encoder::task_vector(matrix, &base);
    let deep_task = jotnar_agent::task_encoder::task_vector(deep, &base);
    assert_eq!(predictor.predict(&matrix_task), 2);
    assert_eq!(predictor.predict(&deep_task), 2);

    let looped = LoopedTaskEncoder::prior(3).with_budget_predictor(predictor);
    let correct = views
        .iter()
        .find(|v| v.outcome == OutcomeKind::Solved)
        .unwrap();
    let correct_task = jotnar_agent::task_encoder::task_vector(correct, &base);
    assert_eq!(looped.loops_for_task(correct, &correct_task), 1);
}

#[test]
fn budget_predictor_drives_corpus_adoption() {
    let (predictor, base) = fit_predictor();
    let looped = LoopedTaskEncoder::prior(3).with_budget_predictor(predictor);
    let report = run_round_with(&base, &all_cases(), &temp_dir("adopt"), &looped);
    assert!(report.adopted_flag);
    assert_eq!(report.adopted.correct(), 24);
    assert_eq!(report.adopted.wrong(), 0);
}

#[test]
fn learned_budget_cannot_drop_below_the_structural_prior() {
    let base = Harness::default();
    let all = all_cases();
    let report = run_round(&base, &all, &temp_dir("prior-floor"));
    let views = case_views(&all, &report.base);
    let matrix = views
        .iter()
        .find(|v| v.family == "matrix" && v.outcome == OutcomeKind::WrongRefusal)
        .expect("matrix failure");
    let deep = views
        .iter()
        .find(|v| v.family == "elementwise" && v.outcome == OutcomeKind::WrongRefusal)
        .expect("deep failure");
    let looped = LoopedTaskEncoder::prior(3).with_budget_predictor(BudgetPredictor::prior(3));
    let matrix_task = jotnar_agent::task_encoder::task_vector(matrix, &base);
    let deep_task = jotnar_agent::task_encoder::task_vector(deep, &base);
    assert_eq!(looped.loops_for_task(matrix, &matrix_task), 2);
    assert_eq!(looped.loops_for_task(deep, &deep_task), 3);
}

#[test]
fn budget_weights_roundtrip() {
    let p = BudgetPredictor::prior(3);
    let path = temp_dir("weights").join("budget.json");
    p.save(&path).unwrap();
    let back = BudgetPredictor::load(&path).unwrap();
    assert_eq!(back.max_loops, 3);
    assert_eq!(p.listing, back.listing);
}
