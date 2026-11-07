use std::fs;

use jotnar::asm::{self, AsmOptions};
use jotnar_agent::{
    calibration_cases, cases, run_round, run_round_with, Harness, JotnarProposer, LoopedProposer,
};

fn temp_dir(tag: &str) -> std::path::PathBuf {
    let p = std::env::temp_dir().join(format!("jotnar-agent-looped-{tag}-{}", std::process::id()));
    let _ = fs::remove_dir_all(&p);
    fs::create_dir_all(&p).unwrap();
    p
}

#[test]
fn looped_listing_assembles_for_one_two_and_four_loops() {
    for loops in [1, 2, 4] {
        let p = LoopedProposer::prior(loops);
        let bound = asm::assemble(&p.listing, &AsmOptions::new("."))
            .unwrap_or_else(|e| panic!("loops={loops}: {e}"));
        assert_eq!(bound.inputs.len(), jotnar_agent::FEATURE_NAMES.len());
        assert!(bound.instrs.len() > 100);
    }
}

#[test]
fn looped_scores_keep_sum4_in_the_top_two_at_every_loop_count() {
    let base = Harness::default();
    let report = run_round(&base, &cases(), &temp_dir("score"));
    let f = jotnar_agent::proposer::features(&base, &report.base, &[]);
    let mut seen = Vec::new();
    for loops in [1, 2, 4] {
        let p = LoopedProposer::prior(loops);
        let scores = p.scores(&f).expect("looped listing runs");
        assert_eq!(scores.len(), jotnar_agent::ACTIONS.len());
        let mut ranked: Vec<usize> = (0..scores.len()).collect();
        ranked.sort_by_key(|i| std::cmp::Reverse(scores[*i]));
        assert!(
            ranked[..2].contains(&0),
            "loops={loops} should keep sum4 in the top two, scores={scores:?}"
        );
        seen.push(scores);
    }
    assert_ne!(seen[0], seen[2], "loops should change the effective scores");
}

fn top2_has(scores: &[i64], idx: usize) -> bool {
    let mut ranked: Vec<usize> = (0..scores.len()).collect();
    ranked.sort_by_key(|i| std::cmp::Reverse(scores[*i]));
    ranked[..2.min(ranked.len())].contains(&idx)
}

#[test]
fn recurrent_parameters_move_the_crossover() {
    let base = Harness::default();
    let report = run_round(&base, &cases(), &temp_dir("recurrence"));
    let f = jotnar_agent::proposer::features(&base, &report.base, &[]);
    let ref_idx = jotnar_agent::ACTIONS
        .iter()
        .position(|a| *a == "refinements + 1")
        .unwrap();

    let prior2 = LoopedProposer::with_recurrence(2, 4, -2);
    assert!(top2_has(&prior2.scores(&f).unwrap(), ref_idx));

    let fitted2 = LoopedProposer::with_recurrence(2, 6, -4);
    assert!(
        !top2_has(&fitted2.scores(&f).unwrap(), ref_idx),
        "the fitted recurrent block needs more than two loops on the held-out case"
    );
    let fitted4 = LoopedProposer::with_recurrence(4, 6, -4);
    assert!(top2_has(&fitted4.scores(&f).unwrap(), ref_idx));
}

#[test]
fn deep_training_case_is_present_but_held_out_case_is_not() {
    let deep = jotnar_agent::deep_calibration_cases();
    assert!(deep.iter().any(|c| c.intent.id == "refine-affine-s16"));
    assert!(!deep.iter().any(|c| c.intent.id == "refine-s16"));
}

#[test]
fn efficient_deep_fit_preserves_two_loops_and_four_loop_retention() {
    let base = Harness::default();
    let report = run_round(&base, &cases(), &temp_dir("efficient-fit"));
    let f = jotnar_agent::proposer::features(&base, &report.base, &[]);
    let ref_idx = jotnar_agent::ACTIONS
        .iter()
        .position(|a| *a == "refinements + 1")
        .unwrap();

    let p1 = LoopedProposer::with_recurrence(1, 5, -4);
    let p2 = LoopedProposer::with_recurrence(2, 5, -4);
    let p4 = LoopedProposer::with_recurrence(4, 5, -4);
    assert!(
        !top2_has(&p1.scores(&f).unwrap(), ref_idx),
        "one loop should miss the held-out deep fix"
    );
    assert!(
        top2_has(&p2.scores(&f).unwrap(), ref_idx),
        "two loops should find it"
    );
    assert!(
        top2_has(&p4.scores(&f).unwrap(), ref_idx),
        "four loops should retain it"
    );
}

#[test]
fn two_loops_buy_the_deep_fix() {
    let base = Harness::default();
    let report = run_round(&base, &cases(), &temp_dir("crossover"));
    let f = jotnar_agent::proposer::features(&base, &report.base, &[]);

    let top2 = |loops: usize| {
        let p = LoopedProposer::prior(loops);
        let scores = p.scores(&f).unwrap();
        let mut ranked: Vec<usize> = (0..scores.len()).collect();
        ranked.sort_by_key(|i| std::cmp::Reverse(scores[*i]));
        (scores, ranked[..2].to_vec())
    };

    let (s1, top1) = top2(1);
    let (s2, top2) = top2(2);
    assert!(
        !top1.contains(&5),
        "one loop should miss the refinements fix, scores={s1:?}"
    );
    assert!(
        top2.contains(&5),
        "two loops should include the refinements fix, scores={s2:?}"
    );
}

#[test]
fn one_loop_does_not_adopt_but_two_do() {
    let base = Harness::default();
    let all = cases();

    let one = run_round_with(
        &base,
        &all,
        &temp_dir("one-loop"),
        &LoopedProposer::prior(1),
    );
    assert!(
        !one.adopted_flag,
        "one loop should pick the decoy second proposal and stop"
    );

    let two = run_round_with(
        &base,
        &all,
        &temp_dir("two-loops"),
        &LoopedProposer::prior(2),
    );
    assert!(two.adopted_flag, "two loops should find the composed fix");
    assert_eq!(two.adopted.correct(), all.len());
    assert_eq!(two.adopted.wrong(), 0);
}

#[test]
fn looped_from_linear_drives_adoption() {
    let base = Harness::default();
    let calibration = run_round(&base, &calibration_cases(), &temp_dir("calibration"));
    let mut linear = JotnarProposer::prior();
    linear.fit_from_report(&base, &calibration, &[], 1);
    let looped = LoopedProposer::from_linear(&linear, 2);
    let out = temp_dir("adopt");
    let report = run_round_with(&base, &cases(), &out, &looped);
    assert!(report.adopted_flag);
    assert!(report.chosen_label.contains("sum4"));
    assert_eq!(report.adopted.correct(), cases().len());
}

#[test]
fn latent_matches_the_listing_scores() {
    let base = Harness::default();
    let report = run_round(&base, &cases(), &temp_dir("latent"));
    let f = jotnar_agent::proposer::features(&base, &report.base, &[]);
    let p = LoopedProposer::prior(2);
    let h = p.latent(&f);
    let manual: Vec<i64> = (0..jotnar_agent::ACTIONS.len())
        .map(|i| {
            h.iter()
                .zip(p.weights.wout[i].iter())
                .map(|(a, b)| a * b)
                .sum::<i64>()
                + p.weights.bout[i]
        })
        .collect();
    assert_eq!(manual, p.scores(&f).unwrap());
}

#[test]
fn looped_weights_roundtrip() {
    let p = LoopedProposer::prior(2);
    let path = temp_dir("weights").join("looped.weights.json");
    p.save(&path).unwrap();
    let back = LoopedProposer::load(2, &path).unwrap();
    assert_eq!(p.weights.wout, back.weights.wout);
    assert_eq!(p.listing, back.listing);
}

#[test]
fn one_loop_after_fit_still_misses_the_heldout_deep_case() {
    let base = Harness::default();
    let calibration = run_round(&base, &calibration_cases(), &temp_dir("one-fit-cal"));
    let mut p = LoopedProposer::prior(1);
    p.fit_from_report(&base, &calibration, &[], 1);
    let report = run_round_with(&base, &cases(), &temp_dir("one-fit-run"), &p);
    assert!(
        !report.adopted_flag,
        "the held-out deep case should still require a second loop"
    );
}

#[test]
fn looped_fit_preserves_two_loop_adoption() {
    let base = Harness::default();
    let calibration = run_round(&base, &calibration_cases(), &temp_dir("fit-calibration"));
    let mut p = LoopedProposer::prior(2);
    let changes = p.fit_from_report(&base, &calibration, &[], 1);
    assert!(!changes.is_empty());
    let out = temp_dir("fit-adopt");
    let report = run_round_with(&base, &cases(), &out, &p);
    assert!(report.adopted_flag);
    assert_eq!(report.adopted.correct(), cases().len());
}

#[test]
fn looped_trace_reports_the_loop_count_and_scores() {
    let base = Harness::default();
    let report = run_round(&base, &cases(), &temp_dir("trace"));
    let p = LoopedProposer::prior(4);
    let trace = p.trace(&base, &report.base, &[]);
    assert_eq!(trace["loops"], serde_json::json!(4));
    assert_eq!(
        trace["scores"].as_array().map(|a| a.len()),
        Some(jotnar_agent::ACTIONS.len())
    );
}
