use std::fs;

use jotnar_agent::{
    all_cases, case_views, corpus, evaluate, generated_corpus, generated_split_cases, hard_cases,
    run_round, run_round_with, split_cases, task_encoder::task_vector, Harness, JotnarProposer,
    LoopedProposer, Proposer, Split,
};

fn temp_dir(tag: &str) -> std::path::PathBuf {
    let p = std::env::temp_dir().join(format!("jotnar-agent-corpus-{tag}-{}", std::process::id()));
    let _ = fs::remove_dir_all(&p);
    fs::create_dir_all(&p).unwrap();
    p
}

#[test]
fn generated_corpus_has_a_deep_training_failure_and_frozen_heldout() {
    let all = jotnar_agent::generated_corpus();
    let train = all.iter().filter(|c| c.split == Split::Train).count();
    let calibration = all.iter().filter(|c| c.split == Split::Calibration).count();
    let heldout = all.iter().filter(|c| c.split == Split::Heldout).count();
    assert_eq!((train, calibration, heldout), (10, 7, 6));

    // The wide compose consumers are taught on their own shapes; the distant
    // matrix/select shapes stay audit-only.
    for id in ["gen-vec9-sum-train", "gen-vec16-sum-train"] {
        assert!(all
            .iter()
            .any(|c| c.split == Split::Train && c.case.intent.id == id));
    }

    // The deep training member is a saxpy target; the held-out deep member is
    // still the affine target. The held-out row is never moved.
    assert!(all.iter().any(|c| {
        c.split == Split::Train
            && c.case.intent.id == "gen-elem-saxpy-s16-train"
            && c.case.intent.input.label.as_deref() == Some("saxpy")
            && c.case.intent.output.label.as_deref() == Some("saxpy_s16")
    }));
    assert!(all.iter().any(|c| {
        c.split == Split::Heldout && c.case.intent.id == "gen-elem-affine-s16-holdout"
    }));
}

#[test]
fn hard_cases_are_audit_only_and_require_composition() {
    let hard = hard_cases();
    assert_eq!(hard.len(), 11);
    let base = Harness::default();
    let report = evaluate(&base, &hard, false);
    assert_eq!(
        report.wrong_refusal, 7,
        "four refine/deep + matmul3 + matmul4 + select16"
    );
    assert_eq!(report.solved, 3, "three high-distance retention cases");
    assert_eq!(report.refused, 1, "one high-distance refusal control");

    let known: std::collections::BTreeSet<String> = generated_corpus()
        .into_iter()
        .map(|c| c.case.intent.id)
        .chain(corpus().into_iter().map(|c| c.case.intent.id))
        .collect();
    for case in &hard {
        assert!(
            !known.contains(&case.intent.id),
            "{} must not appear in a training/acceptance split",
            case.intent.id
        );
    }

    // The four refine/deep cases need depth + refinements together.
    let states = jotnar_agent::reachable_states(&base, 3);
    for case in hard
        .iter()
        .filter(|c| c.intent.id.starts_with("hard-refine") || c.intent.id.starts_with("hard-cross"))
    {
        let min = states
            .iter()
            .filter_map(|(_, subset, harness)| {
                let r = evaluate(harness, std::slice::from_ref(case), false);
                r.outcomes[0].kind.correct().then_some(subset.len())
            })
            .min()
            .unwrap_or(0);
        assert_eq!(min, 2, "{} should need depth + refinements", case.intent.id);
    }

    // The feature-distant compose cases need the wide consumers. They are
    // audit-only and were never trained on directly.
    for (id, consumer) in [
        ("hard-matmul3-sum", 0usize),
        ("hard-matmul4-sum", 0),
        ("hard-select16-sum", 0),
    ] {
        let case = hard
            .iter()
            .find(|c| c.intent.id == id)
            .unwrap_or_else(|| panic!("missing hard case {id}"));
        let min_actions = states
            .iter()
            .filter_map(|(_, subset, harness)| {
                let r = evaluate(harness, std::slice::from_ref(case), false);
                r.outcomes[0].kind.correct().then_some(subset.clone())
            })
            .min_by_key(|subset| subset.len())
            .unwrap_or_else(|| panic!("{id} is solvable"));
        assert_eq!(
            min_actions,
            vec![consumer],
            "{id} should be fixed by {}",
            jotnar_agent::ACTIONS[consumer]
        );
    }
}

#[test]
fn feature_distance_audit_exposes_the_sibling_heldout() {
    let base = Harness::default();
    let mut train = generated_split_cases(Split::Train);
    train.extend(generated_split_cases(Split::Calibration));
    let train_report = evaluate(&base, &train, false);
    let train_tasks: Vec<Vec<i64>> = case_views(&train, &train_report)
        .iter()
        .map(|view| task_vector(view, &base))
        .collect();
    let min_dist = |task: &[i64]| {
        train_tasks
            .iter()
            .map(|train_task| {
                task.iter()
                    .zip(train_task)
                    .map(|(x, y)| (*x - *y).unsigned_abs() as usize)
                    .sum()
            })
            .min()
            .unwrap_or(0)
    };

    let heldout = generated_split_cases(Split::Heldout);
    let held_report = evaluate(&base, &heldout, false);
    let held_views = case_views(&heldout, &held_report);
    let exact = held_views
        .iter()
        .filter(|view| min_dist(&task_vector(view, &base)) == 0)
        .count();
    assert!(
        exact >= 1,
        "the audit should expose at least one feature-duplicate held-out case"
    );

    let hard = hard_cases();
    let hard_report = evaluate(&base, &hard, false);
    let hard_views = case_views(&hard, &hard_report);
    let reduce = hard
        .iter()
        .position(|case| case.intent.id == "hard-reduce-128-8")
        .expect("hard reduction case");
    assert!(
        min_dist(&task_vector(&hard_views[reduce], &base)) > 100,
        "the hard reduction case should be feature-distant"
    );

    // The old scalar-count L1 distance is no longer the right metric: the
    // shaped tasks have similar counts but different geometry/mechanism. The
    // meaningful held-out test is that the shape + op-family signature does
    // not appear in training.
    let signature = |view: &jotnar_agent::CaseView| {
        (
            view.start_rank,
            view.start_numel,
            view.start_imatmul,
            view.start_iselect,
            view.start_iadd,
            view.target_is_scalar,
        )
    };
    let train_signatures: std::collections::BTreeSet<_> = case_views(&train, &train_report)
        .iter()
        .map(signature)
        .collect();
    for id in ["hard-matmul3-sum", "hard-matmul4-sum", "hard-select16-sum"] {
        let idx = hard
            .iter()
            .position(|case| case.intent.id == id)
            .unwrap_or_else(|| panic!("missing distant case {id}"));
        let sig = signature(&hard_views[idx]);
        assert!(
            !train_signatures.contains(&sig),
            "{id} shape/mechanism signature {sig:?} appears in training"
        );
        assert!(
            hard_views[idx].start_is_field && hard_views[idx].target_is_scalar,
            "{id} must be a field-to-scalar reduction"
        );
    }
}

#[test]
fn corpus_has_24_cases_with_a_12_6_6_split() {
    let all = corpus();
    assert_eq!(all.len(), 24);
    assert_eq!(split_cases(Split::Train).len(), 12);
    assert_eq!(split_cases(Split::Calibration).len(), 6);
    assert_eq!(split_cases(Split::Heldout).len(), 6);

    for family in [
        "elementwise",
        "reduction",
        "matrix",
        "select",
        "refine",
        "control",
    ] {
        let n = all.iter().filter(|c| c.case.family == family).count();
        assert_eq!(n, 4, "{family} should have four cases");
    }
}

#[test]
fn corpus_base_profile_is_18_correct_and_6_wrong() {
    let out = temp_dir("base");
    let report = run_round(&Harness::default(), &all_cases(), &out);
    assert_eq!(report.base.correct(), 18);
    assert_eq!(report.base.wrong(), 6);
    assert_eq!(report.base.wrong_refusal_in("matrix"), 4);
    assert_eq!(report.base.wrong_refusal_in("elementwise"), 2);
    assert_eq!(report.base.wrong_refusal_in("reduction"), 0);
    assert_eq!(report.base.wrong_refusal_in("select"), 0);
    assert_eq!(report.base.wrong_refusal_in("refine"), 0);
    assert_eq!(report.base.wrong_refusal_in("control"), 0);
    assert_eq!(report.adopted.correct(), 24);
}

#[test]
fn corpus_prior_selects_both_capabilities() {
    let out = temp_dir("prior");
    let report = run_round(&Harness::default(), &all_cases(), &out);
    let p = JotnarProposer::corpus_prior();
    let proposals = p.propose(&Harness::default(), &report.base, &[]);
    let labels = proposals
        .iter()
        .map(|m| m.label.as_str())
        .collect::<Vec<_>>();
    assert!(
        labels.iter().any(|l| l.contains("reduce_sum")),
        "{labels:?}"
    );
    assert!(
        labels.iter().any(|l| l.contains("refinements")),
        "{labels:?}"
    );
}

#[test]
fn corpus_priors_adopt_all_24_cases() {
    let all = all_cases();
    let linear = run_round_with(
        &Harness::default(),
        &all,
        &temp_dir("linear"),
        &JotnarProposer::corpus_prior(),
    );
    assert!(linear.adopted_flag);
    assert_eq!(linear.adopted.correct(), 24);
    assert_eq!(linear.adopted.wrong(), 0);

    let looped = run_round_with(
        &Harness::default(),
        &all,
        &temp_dir("looped"),
        &LoopedProposer::corpus_prior(2),
    );
    assert!(looped.adopted_flag);
    assert_eq!(looped.adopted.correct(), 24);
    assert_eq!(looped.adopted.wrong(), 0);
}
