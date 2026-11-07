use std::collections::BTreeSet;
use std::fs;

use jotnar_agent::{
    all_cases, apply_action_set, case_views, evaluate, run_round_with, Archive, ArchiveEntry,
    ArchiveProposer, Harness, Proposer,
};

fn temp_dir(tag: &str) -> std::path::PathBuf {
    let p = std::env::temp_dir().join(format!("jotnar-agent-qd-{tag}-{}", std::process::id()));
    let _ = fs::remove_dir_all(&p);
    fs::create_dir_all(&p).unwrap();
    p
}

fn harness(actions: &[usize]) -> Harness {
    let set: BTreeSet<usize> = actions.iter().copied().collect();
    apply_action_set(&Harness::default(), &set)
}

fn entry(label: &str, h: Harness, correct: usize, heldout: usize) -> ArchiveEntry {
    ArchiveEntry {
        label: label.to_string(),
        source: "test".to_string(),
        descriptor: h.state_features(),
        harness: h,
        correct,
        wrong: 0,
        heldout_correct: heldout,
        heldout_wrong: 0,
        novelty: 0,
    }
}

#[test]
fn archive_proposer_seeds_and_hops() {
    let base = Harness::default();
    let all = all_cases();
    let report = evaluate(&base, &all, false);
    let views = case_views(&all, &report);

    let mut archive = Archive::new(8, 1);
    archive.insert(entry("sum4+refinements", harness(&[0, 5]), 24, 6));
    let proposer = ArchiveProposer::new(archive, 1);
    let proposals = proposer.propose_cases(&views, &base, &[]).unwrap();

    assert!(proposals.iter().any(|m| m.label.starts_with("qd-seed")));
    assert!(proposals.iter().any(|m| m.label.starts_with("qd-hop")));
    assert!(proposals.len() > 2);
}

#[test]
fn basin_hopping_adopts_the_archived_harness() {
    let base = Harness::default();
    let all = all_cases();
    let mut archive = Archive::new(8, 1);
    archive.insert(entry("sum4+refinements", harness(&[0, 5]), 24, 6));
    let proposer = ArchiveProposer::new(archive, 1);

    let report = run_round_with(&base, &all, &temp_dir("adopt"), &proposer);
    assert!(report.adopted_flag);
    assert_eq!(report.adopted.correct(), all.len());
    assert_eq!(report.adopted.wrong(), 0);
    assert!(
        report.chosen_label.contains("qd"),
        "chosen {}",
        report.chosen_label
    );
}
