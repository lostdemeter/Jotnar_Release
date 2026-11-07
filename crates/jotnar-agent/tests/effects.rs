use jotnar_agent::{action_effects, generated_corpus, Harness, ACTIONS};

#[test]
fn action_effects_expose_the_solving_action() {
    let cases: Vec<_> = generated_corpus().into_iter().map(|c| c.case).collect();
    let effects = action_effects(&cases, &Harness::default());
    assert_eq!(effects.len(), cases.len());
    let matrix = effects
        .iter()
        .find(|effect| effect.case == "gen-matmul-sum-cal")
        .expect("generated matrix case");
    assert_eq!(matrix.actions.len(), ACTIONS.len());
    assert!(
        !matrix.base_correct,
        "the generated matrix case starts wrong"
    );
    let sum4 = matrix
        .actions
        .iter()
        .find(|action| action.label.contains("sum4"))
        .expect("sum4 action");
    assert!(sum4.correct, "sum4 should solve the generated matrix case");

    let saxpy = effects
        .iter()
        .find(|effect| effect.case == "gen-elem-saxpy-quad")
        .expect("generated elementwise case");
    assert!(saxpy.base_correct);
    assert_eq!(saxpy.actions.len(), ACTIONS.len());
}
