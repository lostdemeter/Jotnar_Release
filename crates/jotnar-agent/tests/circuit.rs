use jotnar_agent::{bit_exact, build_refused, cases, library, solve, Harness, SolveError, State};

fn case(id: &str) -> jotnar_agent::Case {
    cases().into_iter().find(|c| c.intent.id == id).unwrap()
}

#[test]
fn base_harness_solves_two_and_refuses_two() {
    let h = Harness::default();
    let fit =
        solve(&case("fit-reduce").intent, &case("fit-reduce").examples, &h).expect("reduce fit");
    assert!(
        fit.certificate.validate().is_empty(),
        "{:?}",
        fit.certificate.validate()
    );
    assert!(fit.steps.iter().any(|s| s.kind == "fit"));

    let compose = solve(
        &case("compose-saxpy").intent,
        &case("compose-saxpy").examples,
        &h,
    )
    .expect("saxpy compose");
    assert!(compose.steps.iter().any(|s| s.kind == "compose"));

    match solve(
        &case("compose-matmul").intent,
        &case("compose-matmul").examples,
        &h,
    ) {
        Err(SolveError::Unsolved { .. }) => {}
        other => panic!("expected unsolved, got {other:?}"),
    }

    match solve(
        &case("refuse-saxpy-matmul").intent,
        &case("refuse-saxpy-matmul").examples,
        &h,
    ) {
        Err(SolveError::Unsolved { .. }) => {}
        other => panic!("expected unsolved, got {other:?}"),
    }
}

#[test]
fn reduce_sum_solves_shaped_hard_cases() {
    let mut h = Harness::default();
    h.compose_consumers.push("reduce_sum".to_string());
    for case in jotnar_agent::hard_cases() {
        if !matches!(
            case.intent.id.as_str(),
            "hard-matmul3-sum" | "hard-matmul4-sum" | "hard-select16-sum"
        ) {
            continue;
        }
        let res = solve(&case.intent, &case.examples, &h);
        eprintln!(
            "{}: {:?}",
            case.intent.id,
            res.as_ref().map(|r| r.certificate.outcome.clone())
        );
        assert!(res.is_ok(), "{}: {:?}", case.intent.id, res.err());
    }
}

#[test]
fn affine_s64_step_chain_is_exact() {
    use jotnar_agent::gate::bit_exact;
    use jotnar_agent::library::library;
    use jotnar_agent::{State, Step};
    let lib = library();
    let mut s = State::new("affine", lib["affine"].clone()).unwrap();
    for _ in 0..6 {
        s = Step::Compose {
            consumer: "add2".into(),
            rank: 20,
        }
        .apply(&s)
        .unwrap();
    }
    let t = State::new("affine_s64", lib["affine_s64"].clone()).unwrap();
    let ex = jotnar_agent::example_set("affine_s64", 0);
    let gate = bit_exact(&s, &t, &ex, &["OUT".to_string()]);
    assert!(gate.passed, "{:?}", gate.detail);
}

#[test]
fn s64_needs_repeated_depth() {
    let base = Harness::default();
    let case = jotnar_agent::hard_cases()
        .into_iter()
        .find(|c| c.intent.id == "hard-affine-s64")
        .expect("hard affine s64 case");
    let weak = solve(&case.intent, &case.examples, &base);
    assert!(
        weak.is_err(),
        "the default depth 2 harness should miss the 64x target"
    );
    let mut deep = base.clone();
    deep.max_depth = 6;
    let strong = solve(&case.intent, &case.examples, &deep).expect("depth 6 solves s64");
    assert!(strong.certificate.validate().is_empty());
    assert_eq!(
        strong.certificate.outcome,
        jotnar_lab::model::Outcome::Solved
    );
}

#[test]
fn reduce_sum_is_the_bounded_extension_that_solves_matmul() {
    let mut h = Harness::default();
    h.compose_consumers.push("reduce_sum".to_string());
    let c = case("compose-matmul");
    let res = solve(&c.intent, &c.examples, &h).expect("reduce_sum compose");
    assert!(res.steps.iter().any(|s| s.name.contains("reduce_sum")));
    assert!(res.certificate.validate().is_empty());
    assert_eq!(res.certificate.outcome, jotnar_lab::model::Outcome::Solved);

    let held = solve(&c.intent, &c.heldout, &h).expect("held-out compose");
    assert!(held.certificate.validate().is_empty());
}

#[test]
fn the_gate_can_say_no() {
    let lib = library();
    let a = State::new("saxpy", lib["saxpy"].clone()).unwrap();
    let b = State::new("saxpy_double", lib["saxpy_double"].clone()).unwrap();
    let examples = jotnar_agent::example_set("saxpy_double", 0);
    let report = bit_exact(&a, &b, &examples, &["OUT".to_string()]);
    assert!(!report.passed, "a wrong candidate must fail the gate");
}

#[test]
fn a_refusal_is_a_valid_certificate() {
    let c = case("compose-matmul");
    let h = Harness::default();
    let err = solve(&c.intent, &c.examples, &h).unwrap_err();
    let cert = build_refused(
        &c.intent,
        err.target(),
        err.examples(),
        err.code(),
        err.message(),
        err.missing(),
    );
    assert!(cert.validate().is_empty(), "{:?}", cert.validate());
    assert_eq!(cert.outcome, jotnar_lab::model::Outcome::Refused);
    assert_eq!(cert.verdict, jotnar_lab::model::Verdict::Bottom);
    assert!(cert.refusal.is_some());
}
