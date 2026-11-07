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
fn sum4_is_the_bounded_extension_that_solves_matmul() {
    let mut h = Harness::default();
    h.compose_consumers.push("sum4".to_string());
    let c = case("compose-matmul");
    let res = solve(&c.intent, &c.examples, &h).expect("sum4 compose");
    assert!(res.steps.iter().any(|s| s.name.contains("sum4")));
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
