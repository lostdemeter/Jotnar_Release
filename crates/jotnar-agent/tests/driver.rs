use std::fs;
use std::path::PathBuf;

use jotnar_agent::{
    cases, run_round, run_round_with, run_round_with_memory, EvalReport, Experience, Harness,
    Mutation, Proposer,
};
use jotnar_lab::model::Certificate;

struct NoopProposer;

impl Proposer for NoopProposer {
    fn propose(
        &self,
        _base: &Harness,
        _train: &EvalReport,
        _experience: &[Experience],
    ) -> Vec<Mutation> {
        Vec::new()
    }
}

fn temp_dir(tag: &str) -> PathBuf {
    let p = std::env::temp_dir().join(format!("jotnar-agent-{tag}-{}", std::process::id()));
    let _ = fs::remove_dir_all(&p);
    fs::create_dir_all(&p).unwrap();
    p
}

#[test]
fn a_round_discovers_reduce_sum_and_retains_the_incumbent() {
    let out = temp_dir("round");
    let report = run_round(&Harness::default(), &cases(), &out);

    assert!(
        report.adopted_flag,
        "the round should adopt a better harness"
    );
    let n = cases().len();
    assert_eq!(report.base.correct(), 3);
    assert_eq!(report.adopted.correct(), n);
    assert_eq!(report.base.wrong(), 2);
    assert_eq!(report.adopted.wrong(), 0);
    assert_eq!(report.adopted_heldout.correct(), n);
    assert!(report.edits.iter().any(|e| e.contains("reduce_sum")));
    assert!(report.chosen_label.contains("compose:"));
    assert!(report.chosen_label.contains("reduce_sum"));
    assert!(report.experience.is_some());

    let round = fs::read_to_string(out.join("round.json")).unwrap();
    assert!(round.contains("\"adopted_flag\": true"));
    assert!(round.contains("compose[reduce_sum]"));

    let exp = fs::read_to_string(out.join("experience.jsonl")).unwrap();
    assert!(exp.contains("\"lesson\""));

    for sub in ["adopted-train", "adopted-heldout"] {
        let dir = out.join("certificates").join(sub);
        let mut n = 0;
        for entry in fs::read_dir(&dir).unwrap() {
            let p = entry.unwrap().path();
            let cert = Certificate::read(&p).unwrap();
            let errors = cert.validate();
            assert!(errors.is_empty(), "{}: {errors:?}", p.display());
            n += 1;
        }
        assert_eq!(
            n,
            cases().len(),
            "{sub} should have one certificate per case"
        );
    }
}

#[test]
fn an_empty_harness_does_not_adopt() {
    // A deliberately broken harness: no families and no compose consumers.
    let mut h = Harness::default();
    h.families.clear();
    h.compose_consumers.clear();
    let out = temp_dir("empty");
    let report = run_round(&h, &cases(), &out);
    assert!(!report.adopted_flag);
}

#[test]
fn a_custom_proposer_can_disable_mutations() {
    // The LLM seam: a proposer is allowed to return nothing, and the driver
    // must simply keep the incumbent.
    let out = temp_dir("noop");
    let report = run_round_with(&Harness::default(), &cases(), &out, &NoopProposer);
    assert!(!report.adopted_flag);
    assert_eq!(report.candidates.len(), 1);
}

#[test]
fn rounds_share_one_experience_library() {
    let out = temp_dir("memory");
    let memory = out.join("memory.jsonl");
    let all = cases();
    let first = run_round_with_memory(
        &Harness::default(),
        &all,
        &out.join("round-1"),
        &memory,
        &jotnar_agent::DeterministicProposer,
    );
    assert!(first.adopted_flag);

    let second = run_round_with_memory(
        &first.adopted_harness,
        &all,
        &out.join("round-2"),
        &memory,
        &jotnar_agent::DeterministicProposer,
    );
    assert!(
        !second.adopted_flag,
        "an already-complete harness should not churn"
    );
    let lines = fs::read_to_string(&memory).unwrap();
    assert_eq!(
        lines.lines().count(),
        1,
        "round 2 must not duplicate the lesson"
    );
}
