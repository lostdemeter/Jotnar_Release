use std::fs;

use jotnar_agent::{all_cases, run_round, Harness, TeacherRecord, TraceRecord};

#[test]
fn trace_exporter_writes_jsonl() {
    let out = std::env::temp_dir().join(format!("jotnar-agent-trace-{}", std::process::id()));
    let _ = fs::remove_dir_all(&out);
    fs::create_dir_all(&out).unwrap();
    let cases = all_cases();
    let report = run_round(&Harness::default(), &cases, &out);
    let path = out.join("traces.jsonl");
    let n =
        jotnar_agent::trace::write_round_traces(&path, &cases, &[], &report, "deterministic", 1)
            .expect("write traces");
    assert!(n > 0, "expected at least one trace record");
    let text = fs::read_to_string(&path).expect("read traces");
    let first = text.lines().next().expect("first trace");
    let record: TraceRecord = serde_json::from_str(first).expect("trace schema");
    assert!(!record.task.case.is_empty());
    assert_eq!(record.provenance.round, 1);
    assert!(record.candidate.train_correct + record.candidate.train_wrong > 0);
}

#[test]
fn teacher_record_roundtrip() {
    let record = TeacherRecord {
        case: "gen-elem-saxpy-quad".to_string(),
        task: vec![3, 1, 2, 3, 1, 4, 2, 0, 0, 1, 0, 0, 2, 0, 1, 0, 1, 0, 1, 0],
        latent: vec![1.0, 2.0, 4.0, 2.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        split: "train".to_string(),
        source: "dory".to_string(),
    };
    let line = serde_json::to_string(&record).expect("serialize teacher");
    let back: TeacherRecord = serde_json::from_str(&line).expect("deserialize teacher");
    assert_eq!(back.task, record.task);
    assert_eq!(back.latent, record.latent);
    assert_eq!(back.source, "dory");
}
