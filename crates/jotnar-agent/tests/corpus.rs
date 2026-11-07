use std::fs;

use jotnar_agent::{
    all_cases, corpus, run_round, run_round_with, split_cases, Harness, JotnarProposer,
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
    assert_eq!((train, calibration, heldout), (8, 7, 6));

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
    assert!(labels.iter().any(|l| l.contains("sum4")), "{labels:?}");
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
