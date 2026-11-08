//! Wall-Breaking Protocol experiments.
//!
//! The first test is the scientific claim in miniature: an additive linear
//! scorer cannot represent a product interaction, but the WBP self-similar
//! feature at the residual lag can. The feature is computed by a Jotnar listing
//! (`crate::wall_breaking::self_similar_products`).

use jotnar_agent::wall_breaking::{peel, resfrac, self_similar_products};
use jotnar_agent::{StateEffectModel, StateEffectRow, STATE_FEATURE_NAMES, TASK_FEATURE_NAMES};

fn samples(n: usize, seed: u64) -> Vec<(i64, i64)> {
    let mut s = seed;
    let mut out = Vec::with_capacity(n);
    while out.len() < n {
        s = s.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
        let a = ((s >> 33) % 11) as i64 - 5;
        s = s.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
        let b = ((s >> 33) % 11) as i64 - 5;
        if a != 0 && b != 0 {
            out.push((a, b));
        }
    }
    out
}

/// The product feature is computed by the Jotnar listing, not by Rust.
fn wbp_product(x0: i64, x1: i64) -> i64 {
    let p = self_similar_products(&[x0, x1], 1).expect("jotnar self-similar listing");
    p[1]
}

fn rows(samples: &[(i64, i64)], with_product: bool) -> Vec<StateEffectRow> {
    let mut out = Vec::new();
    for (i, (x0, x1)) in samples.iter().enumerate() {
        let target = if x0 * x1 > 0 { 1 } else { 2 };
        let mut task = vec![0i64; TASK_FEATURE_NAMES.len()];
        task[0] = *x0;
        task[1] = *x1;
        if with_product {
            task[2] = wbp_product(*x0, *x1);
        }
        for action in [1usize, 2] {
            let correct = action == target;
            out.push(StateEffectRow {
                case: format!("wbp-s{i}"),
                split: "train".to_string(),
                task: task.clone(),
                state_key: "base".to_string(),
                state: vec![0; STATE_FEATURE_NAMES.len()],
                action,
                label: format!("action{action}"),
                parent_outcome: "wrong-refusal".to_string(),
                outcome: if correct { "solved" } else { "wrong-refusal" }.to_string(),
                parent_correct: false,
                correct,
                wrong: !correct,
                delta: i64::from(correct),
                steps: 1,
            });
        }
    }
    out
}

fn accuracy(model: &StateEffectModel, samples: &[(i64, i64)], with_product: bool) -> f64 {
    let mut hits = 0usize;
    for (x0, x1) in samples {
        let mut task = vec![0i64; TASK_FEATURE_NAMES.len()];
        task[0] = *x0;
        task[1] = *x1;
        if with_product {
            task[2] = wbp_product(*x0, *x1);
        }
        let scores = model.scores(&task, &vec![0; STATE_FEATURE_NAMES.len()]);
        let top = [1usize, 2]
            .into_iter()
            .max_by_key(|action| scores[*action])
            .unwrap();
        let target = if x0 * x1 > 0 { 1 } else { 2 };
        if top == target {
            hits += 1;
        }
    }
    hits as f64 / samples.len().max(1) as f64
}

#[test]
fn self_similar_product_breaks_a_synthetic_interaction() {
    let train = samples(240, 7);
    let test = samples(120, 11);

    let mut linear = StateEffectModel::prior();
    linear.fit(&rows(&train, false), 30, 1);
    let before = accuracy(&linear, &test, false);

    let mut augmented = StateEffectModel::prior();
    augmented.fit(&rows(&train, true), 30, 1);
    let after = accuracy(&augmented, &test, true);

    eprintln!("WBP synthetic wall: linear={before:.3} wbp-product={after:.3}");
    assert!(
        before < 0.8,
        "the additive model should hit the product wall, got {before:.3}"
    );
    assert!(
        after > 0.95,
        "the self-similar product should break the wall, got {after:.3}"
    );
}

#[test]
fn wbp_audit_runs_on_a_structured_residual() {
    // A slow self-similar signal plus a small residue. The audit must return a
    // finite rho trajectory and a peel; this is the diagnostic the agent will
    // use on a transition residual.
    let n = 96;
    let signal: Vec<i64> = (0..n)
        .map(|i| {
            let t = i as f64;
            ((0.05 * t).sin() * 700.0 + (0.17 * t).sin() * 90.0).round() as i64
        })
        .collect();
    let r = resfrac(&signal, 2).expect("resfrac");
    assert!(r.rho.is_finite());
    let p = peel(&signal, 2, 6, 0.85).expect("peel");
    assert!(!p.rhos.is_empty());
    assert!(p.residual.len() < signal.len());
}
