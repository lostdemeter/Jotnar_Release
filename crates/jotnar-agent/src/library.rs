//! The seed library and the first acceptance corpus.
//!
//! The library is data: named listings the circuit may start from. The corpus
//! is a small set of intents with visible examples and a frozen held-out
//! variant. The held-out examples are used for adoption, never for selection.

use std::collections::BTreeMap;

use crate::family;
use crate::gate::{Example, TestVec};
use crate::intent::{Case, CaseExpect, InputSpace, Intent, OutputSpace};
use crate::state::State;

pub fn library() -> BTreeMap<String, String> {
    let mut out = BTreeMap::new();
    let empty = BTreeMap::new();
    out.insert("saxpy".into(), family::build_saxpy(&empty).unwrap());
    out.insert(
        "saxpy_double".into(),
        family::build_saxpy_double(&empty).unwrap(),
    );
    out.insert(
        "saxpy_quad".into(),
        family::build_saxpy_quad(&empty).unwrap(),
    );
    out.insert("saxpy_oct".into(), family::build_saxpy_oct(&empty).unwrap());
    out.insert("saxpy_s16".into(), family::build_saxpy_s16(&empty).unwrap());
    out.insert("saxpy_dup".into(), family::build_saxpy_dup(&empty).unwrap());
    out.insert("affine".into(), family::build_affine(&empty).unwrap());
    out.insert(
        "affine_s16".into(),
        family::build_affine_s16(&empty).unwrap(),
    );
    out.insert(
        "affine_quad".into(),
        family::build_affine_quad(&empty).unwrap(),
    );
    out.insert(
        "affine_oct".into(),
        family::build_affine_oct(&empty).unwrap(),
    );
    out.insert(
        "affine_dup".into(),
        family::build_affine_dup(&empty).unwrap(),
    );
    out.insert("reduce8".into(), family::reduce_listing(8));
    out.insert("reduce16".into(), family::reduce_listing(16));
    out.insert("reduce32".into(), family::reduce_listing(32));
    out.insert("reduce64".into(), family::reduce_listing(64));
    out.insert("reduce128".into(), family::reduce_listing(128));
    out.insert(
        "reduce8_dup".into(),
        family::build_reduce8_dup(&empty).unwrap(),
    );
    out.insert("matmul2".into(), family::build_matmul2(&empty).unwrap());
    out.insert(
        "matmul2_sum".into(),
        family::build_matmul2_sum(&empty).unwrap(),
    );
    out.insert(
        "matmul2_colsum".into(),
        family::build_matmul2_colsum(&empty).unwrap(),
    );
    out.insert(
        "matmul2_dup".into(),
        family::build_matmul2_dup(&empty).unwrap(),
    );
    out.insert("matmul3".into(), family::build_matmul3(&empty).unwrap());
    out.insert(
        "matmul3_sum".into(),
        family::build_matmul3_sum(&empty).unwrap(),
    );
    out.insert(
        "matmul3_dup".into(),
        family::build_matmul3_dup(&empty).unwrap(),
    );
    out.insert("matmul4".into(), family::build_matmul4(&empty).unwrap());
    out.insert(
        "matmul4_sum".into(),
        family::build_matmul4_sum(&empty).unwrap(),
    );
    out.insert("vec9".into(), family::build_vec9(&empty).unwrap());
    out.insert("vec9_sum".into(), family::build_vec9_sum(&empty).unwrap());
    out.insert("vec16".into(), family::build_vec16(&empty).unwrap());
    out.insert("vec16_sum".into(), family::build_vec16_sum(&empty).unwrap());
    out.insert("select16".into(), family::build_select16(&empty).unwrap());
    out.insert(
        "select16_sum".into(),
        family::build_select16_sum(&empty).unwrap(),
    );
    out.insert("select".into(), family::build_select(&empty).unwrap());
    out.insert(
        "select_max".into(),
        family::build_select_max(&empty).unwrap(),
    );
    out.insert(
        "select_min".into(),
        family::build_select_min(&empty).unwrap(),
    );
    out.insert("add2".into(), family::build_add2(&empty).unwrap());
    out.insert("pair".into(), "IN A\nIN B\nOUT = IADD(A, B)\n".to_string());
    out
}

fn tensor(n: usize, start: i64) -> TestVec {
    TestVec {
        shape: vec![n, n],
        data: (0..n * n).map(|i| start + i as i64).collect(),
    }
}

pub fn example_set(label: &str, variant: u64) -> Vec<Example> {
    let label = label.strip_suffix("_dup").unwrap_or(label);
    match label {
        "saxpy" | "saxpy_double" | "saxpy_quad" | "saxpy_oct" | "saxpy_s16" | "affine"
        | "affine_quad" | "affine_oct" | "affine_s16" => {
            let (a, x, y) = if variant == 0 {
                (vec![1, 2, 3, 4], vec![5, 6, 7, 8], vec![9, 10, 11, 12])
            } else {
                (
                    vec![-2, 4, -6, 8],
                    vec![3, -5, 7, -9],
                    vec![11, 13, -15, 17],
                )
            };
            vec![Example::from_pairs(&[
                ("A", TestVec::vector(a)),
                ("X", TestVec::vector(x)),
                ("Y", TestVec::vector(y)),
            ])]
        }
        _ if label.starts_with("reduce") => {
            let n: usize = label.trim_start_matches("reduce").parse().unwrap_or(8);
            let base = if variant == 0 { 1 } else { 1001 };
            let inputs = (0..n)
                .map(|i| {
                    (
                        format!("A{i}"),
                        TestVec::vector(vec![base + i as i64, base + 2 * i as i64]),
                    )
                })
                .collect::<BTreeMap<_, _>>();
            vec![Example::new(inputs)]
        }
        "matmul2" | "matmul2_sum" | "matmul2_colsum" => {
            let base = if variant == 0 { 1 } else { 2 };
            vec![Example::from_pairs(&[
                ("A", tensor(2, base)),
                ("B", tensor(2, base + 4)),
            ])]
        }
        "matmul3" | "matmul3_sum" => {
            let base = if variant == 0 { 1 } else { 2 };
            vec![Example::from_pairs(&[
                ("A", tensor(3, base)),
                ("B", tensor(3, base + 9)),
            ])]
        }
        "matmul4" | "matmul4_sum" => {
            let base = if variant == 0 { 1 } else { 2 };
            vec![Example::from_pairs(&[
                ("A", tensor(4, base)),
                ("B", tensor(4, base + 16)),
            ])]
        }
        "vec9" | "vec9_sum" => {
            let base = if variant == 0 { 1 } else { 101 };
            vec![Example::from_pairs(&[(
                "A",
                TestVec::vector((0..9).map(|i| base + i as i64).collect()),
            )])]
        }
        "vec16" | "vec16_sum" => {
            let base = if variant == 0 { 1 } else { 101 };
            vec![Example::from_pairs(&[(
                "A",
                TestVec::vector((0..16).map(|i| base + i as i64).collect()),
            )])]
        }
        "select16" | "select16_sum" => {
            let base = if variant == 0 { 1 } else { 41 };
            vec![Example::from_pairs(&[
                (
                    "A",
                    TestVec::vector((0..16).map(|i| base + i as i64).collect()),
                ),
                (
                    "B",
                    TestVec::vector((0..16).map(|i| base + 20 - i as i64).collect()),
                ),
            ])]
        }
        "select" => {
            let (c, a, b) = if variant == 0 {
                (vec![0, 1, 0, 1], vec![1, 2, 3, 4], vec![5, 6, 7, 8])
            } else {
                (vec![1, 0, 1, 0], vec![9, 8, 7, 6], vec![2, 3, 4, 5])
            };
            vec![Example::from_pairs(&[
                ("C", TestVec::vector(c)),
                ("A", TestVec::vector(a)),
                ("B", TestVec::vector(b)),
            ])]
        }
        "select_max" | "select_min" => {
            let (a, b) = if variant == 0 {
                (vec![1, 9, 3, 7], vec![5, 2, 8, 4])
            } else {
                (vec![-1, 12, -3, 0], vec![4, -2, 8, 0])
            };
            vec![Example::from_pairs(&[
                ("A", TestVec::vector(a)),
                ("B", TestVec::vector(b)),
            ])]
        }
        "pair" | "add2" => {
            let (a, b) = if variant == 0 {
                (vec![1, 2, 3, 4], vec![5, 6, 7, 8])
            } else {
                (vec![-1, 0, 1, 2], vec![2, 3, 5, 7])
            };
            vec![Example::from_pairs(&[
                ("A", TestVec::vector(a)),
                ("B", TestVec::vector(b)),
            ])]
        }
        other => panic!("no examples for {other}"),
    }
}

fn start_outputs(label: &str) -> usize {
    start_state(label).map(|s| s.outputs().len()).unwrap_or(0)
}

fn start_ops(label: &str) -> usize {
    start_state(label).map(|s| s.ops().len()).unwrap_or(0)
}

fn start_state(label: &str) -> Option<State> {
    let lib = library();
    State::new(label, lib.get(label)?.clone()).ok()
}

/// Cases the proposer may calibrate on. The deep `refine-s16` case is held
/// out: fitting on it would teach the one-loop model the answer and erase the
/// loop-count effect.
pub fn calibration_cases() -> Vec<Case> {
    cases()
        .into_iter()
        .filter(|c| c.intent.id != "refine-s16")
        .collect()
}

/// Calibration plus a *different* deep case (`affine -> affine_s16`). The
/// held-out `refine-s16` is still excluded; this is the training-deep-case
/// experiment.
pub fn deep_calibration_cases() -> Vec<Case> {
    let mut out = calibration_cases();
    out.push(affine_s16_case());
    out
}

fn affine_s16_case() -> Case {
    Case {
        intent: Intent::new(
            "refine-affine-s16",
            InputSpace {
                label: Some("affine".into()),
                ..Default::default()
            },
            OutputSpace {
                label: Some("affine_s16".into()),
                ..Default::default()
            },
        )
        .with_depth(1),
        examples: example_set("affine_s16", 0),
        heldout: example_set("affine_s16", 1),
        expect: CaseExpect::Solve,
        family: "elementwise".into(),
        start_outputs: start_outputs("affine"),
        start_ops: start_ops("affine"),
    }
}

pub fn cases() -> Vec<Case> {
    vec![
        Case {
            intent: Intent::new(
                "fit-reduce",
                InputSpace {
                    label: Some("reduce8".into()),
                    ..Default::default()
                },
                OutputSpace {
                    label: Some("reduce32".into()),
                    ..Default::default()
                },
            )
            .with_tags(&["reduction"])
            .with_depth(2),
            examples: example_set("reduce32", 0),
            heldout: example_set("reduce32", 1),
            expect: CaseExpect::Solve,
            family: "reduction".into(),
            start_outputs: start_outputs("reduce8"),
            start_ops: start_ops("reduce8"),
        },
        Case {
            intent: Intent::new(
                "compose-saxpy",
                InputSpace {
                    label: Some("saxpy".into()),
                    ..Default::default()
                },
                OutputSpace {
                    label: Some("saxpy_double".into()),
                    ..Default::default()
                },
            )
            .with_tags(&["linear"])
            .with_depth(2),
            examples: example_set("saxpy_double", 0),
            heldout: example_set("saxpy_double", 1),
            expect: CaseExpect::Solve,
            family: "elementwise".into(),
            start_outputs: start_outputs("saxpy"),
            start_ops: start_ops("saxpy"),
        },
        Case {
            intent: Intent::new(
                "refine-s16",
                InputSpace {
                    label: Some("saxpy".into()),
                    ..Default::default()
                },
                OutputSpace {
                    label: Some("saxpy_s16".into()),
                    ..Default::default()
                },
            )
            .with_depth(1),
            examples: example_set("saxpy_s16", 0),
            heldout: example_set("saxpy_s16", 1),
            expect: CaseExpect::Solve,
            family: "elementwise".into(),
            start_outputs: start_outputs("saxpy"),
            start_ops: start_ops("saxpy"),
        },
        Case {
            intent: Intent::new(
                "compose-matmul",
                InputSpace {
                    label: Some("matmul2".into()),
                    ..Default::default()
                },
                OutputSpace {
                    label: Some("matmul2_sum".into()),
                    ..Default::default()
                },
            )
            .with_tags(&["matrix"])
            .with_depth(2),
            examples: example_set("matmul2_sum", 0),
            heldout: example_set("matmul2_sum", 1),
            expect: CaseExpect::Solve,
            family: "matrix".into(),
            start_outputs: start_outputs("matmul2"),
            start_ops: start_ops("matmul2"),
        },
        Case {
            intent: Intent::new(
                "refuse-saxpy-matmul",
                InputSpace {
                    label: Some("saxpy".into()),
                    ..Default::default()
                },
                OutputSpace {
                    label: Some("matmul2".into()),
                    ..Default::default()
                },
            )
            .with_tags(&["matrix"])
            .with_depth(2),
            examples: example_set("matmul2", 0),
            heldout: example_set("matmul2", 1),
            expect: CaseExpect::Refuse,
            family: "control".into(),
            start_outputs: start_outputs("saxpy"),
            start_ops: start_ops("saxpy"),
        },
    ]
}
