//! The 24-case stratified corpus.
//!
//! Six capability families, two train / one calibration / one held-out case
//! each. The held-out split is by parameter and by family, not by duplicate.

use serde::Serialize;

use crate::gate::Example;
use crate::intent::{Case, CaseExpect, InputSpace, Intent, OutputSpace};
use crate::library::{example_set, library};
use crate::state::State;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Split {
    Train,
    Calibration,
    Heldout,
    /// Audit-only cases. They are never used for fitting, selection, or
    /// adoption; they measure whether the planner generalizes to a different
    /// composition.
    Hard,
}

impl Split {
    pub fn as_str(self) -> &'static str {
        match self {
            Split::Train => "train",
            Split::Calibration => "calibration",
            Split::Heldout => "heldout",
            Split::Hard => "hard",
        }
    }
}

#[derive(Clone, Debug)]
pub struct CorpusCase {
    pub case: Case,
    pub split: Split,
}

fn meta(label: &str) -> (usize, usize) {
    let lib = library();
    State::new(label, lib[label].clone())
        .map(|s| (s.outputs().len(), s.ops().len()))
        .unwrap_or((0, 0))
}

#[allow(clippy::too_many_arguments)]
fn make(
    id: &str,
    start: &str,
    target: &str,
    family: &str,
    expect: CaseExpect,
    depth: usize,
    examples_label: &str,
    split: Split,
) -> CorpusCase {
    let (start_outputs, start_ops) = meta(start);
    let examples: Vec<Example> = example_set(examples_label, 0);
    let heldout: Vec<Example> = example_set(examples_label, 1);
    let intent = Intent::new(
        id,
        InputSpace {
            label: Some(start.into()),
            ..Default::default()
        },
        OutputSpace {
            label: Some(target.into()),
            ..Default::default()
        },
    )
    .with_depth(depth);
    CorpusCase {
        case: Case {
            intent,
            examples,
            heldout,
            expect,
            family: family.into(),
            start_outputs,
            start_ops,
        },
        split,
    }
}

pub fn corpus() -> Vec<CorpusCase> {
    use CaseExpect::{Refuse, Solve};
    use Split::{Calibration, Heldout, Train};

    vec![
        // Elementwise chains: composition depth and the deep fix.
        make(
            "elem-saxpy-double",
            "saxpy",
            "saxpy_double",
            "elementwise",
            Solve,
            2,
            "saxpy_double",
            Train,
        ),
        make(
            "elem-affine-s16-train",
            "affine",
            "affine_s16",
            "elementwise",
            Solve,
            1,
            "affine_s16",
            Train,
        ),
        make(
            "elem-saxpy-oct",
            "saxpy",
            "saxpy_oct",
            "elementwise",
            Solve,
            1,
            "saxpy_oct",
            Calibration,
        ),
        make(
            "elem-saxpy-s16-holdout",
            "saxpy",
            "saxpy_s16",
            "elementwise",
            Solve,
            1,
            "saxpy_s16",
            Heldout,
        ),
        // Reduction sizes: the fit family, held out by arity.
        make(
            "reduce-8-32",
            "reduce8",
            "reduce32",
            "reduction",
            Solve,
            2,
            "reduce32",
            Train,
        ),
        make(
            "reduce-8-64",
            "reduce8",
            "reduce64",
            "reduction",
            Solve,
            2,
            "reduce64",
            Train,
        ),
        make(
            "reduce-16-64",
            "reduce16",
            "reduce64",
            "reduction",
            Solve,
            2,
            "reduce64",
            Calibration,
        ),
        make(
            "reduce-16-128-holdout",
            "reduce16",
            "reduce128",
            "reduction",
            Solve,
            2,
            "reduce128",
            Heldout,
        ),
        // Matrix: the same geometric reduction over two graph shapes.
        make(
            "matmul-sum",
            "matmul2",
            "matmul2_sum",
            "matrix",
            Solve,
            2,
            "matmul2_sum",
            Train,
        ),
        make(
            "matmul-colsum",
            "matmul2",
            "matmul2_colsum",
            "matrix",
            Solve,
            2,
            "matmul2_colsum",
            Train,
        ),
        make(
            "matmul-sum-cal",
            "matmul2",
            "matmul2_sum",
            "matrix",
            Solve,
            2,
            "matmul2_sum",
            Calibration,
        ),
        make(
            "matmul-colsum-holdout",
            "matmul2",
            "matmul2_colsum",
            "matrix",
            Solve,
            2,
            "matmul2_colsum",
            Heldout,
        ),
        // Select: the current circuit cannot switch to a predicate family, so
        // these are honest refusals plus one identity control.
        make(
            "select-max-refuse",
            "select",
            "select_max",
            "select",
            Refuse,
            2,
            "select",
            Train,
        ),
        make(
            "select-min-refuse",
            "select",
            "select_min",
            "select",
            Refuse,
            2,
            "select",
            Train,
        ),
        make(
            "select-identity",
            "select",
            "select",
            "select",
            Solve,
            2,
            "select",
            Calibration,
        ),
        make(
            "select-matmul-refuse-holdout",
            "select",
            "matmul2",
            "select",
            Refuse,
            2,
            "matmul2",
            Heldout,
        ),
        // Refine: a duplicate line must not change the meaning.
        make(
            "refine-saxpy",
            "saxpy_dup",
            "saxpy",
            "refine",
            Solve,
            2,
            "saxpy",
            Train,
        ),
        make(
            "refine-affine",
            "affine_dup",
            "affine",
            "refine",
            Solve,
            2,
            "affine",
            Train,
        ),
        make(
            "refine-reduce",
            "reduce8_dup",
            "reduce8",
            "refine",
            Solve,
            2,
            "reduce8",
            Calibration,
        ),
        make(
            "refine-matmul-holdout",
            "matmul2_dup",
            "matmul2",
            "refine",
            Solve,
            2,
            "matmul2",
            Heldout,
        ),
        // Refusal and identity controls.
        make(
            "refuse-saxpy-matmul",
            "saxpy",
            "matmul2",
            "control",
            Refuse,
            2,
            "matmul2",
            Train,
        ),
        make(
            "refuse-reduce-matmul",
            "reduce8",
            "matmul2",
            "control",
            Refuse,
            2,
            "matmul2",
            Train,
        ),
        make(
            "identity-saxpy",
            "saxpy",
            "saxpy",
            "control",
            Solve,
            2,
            "saxpy",
            Calibration,
        ),
        make(
            "refuse-affine-matmul-holdout",
            "affine",
            "matmul2",
            "control",
            Refuse,
            2,
            "matmul2",
            Heldout,
        ),
    ]
}

/// A generated task manifest row. The generated corpus is the first task
/// generator: it recombines library start/target pairs and assigns splits by
/// parameter, not by duplicate.
#[derive(Clone, Debug, Serialize)]
pub struct TaskSpec {
    pub id: String,
    pub start: String,
    pub target: String,
    pub family: String,
    pub expect: String,
    pub depth: usize,
    pub examples: String,
    pub start_text: String,
    pub target_text: String,
    pub example_inputs: Vec<Example>,
    pub split: String,
    pub start_outputs: usize,
    pub start_ops: usize,
    pub target_outputs: usize,
    pub target_ops: usize,
}

/// Recombine library start/target pairs into a new 23-case stream. The held-out
/// members use different parameters or a different composition than the
/// training/calibration members.
pub fn generated_corpus() -> Vec<CorpusCase> {
    use CaseExpect::{Refuse, Solve};
    use Split::{Calibration, Heldout, Train};

    vec![
        // Elementwise compositions.
        make(
            "gen-elem-saxpy-quad",
            "saxpy",
            "saxpy_quad",
            "elementwise",
            Solve,
            2,
            "saxpy_quad",
            Train,
        ),
        make(
            "gen-elem-affine-quad",
            "affine",
            "affine_quad",
            "elementwise",
            Solve,
            2,
            "affine_quad",
            Train,
        ),
        // The generated stream needs at least one deep elementwise *failure* in
        // training. Without it the effect model never sees that a deeper target
        // needs `depth + 1`, and the affine deep held-out case is unranked.
        // This is a different start family (saxpy) and example set from the
        // held-out affine case, so the held-out task stays frozen.
        make(
            "gen-elem-saxpy-s16-train",
            "saxpy",
            "saxpy_s16",
            "elementwise",
            Solve,
            1,
            "saxpy_s16",
            Train,
        ),
        // Deeper than s16: this needs repeated `depth + 1` (or equivalent
        // refinement rounds), so the transition table must allow repeated
        // actions. The held-out deeper shape is a different start family.
        make(
            "gen-elem-saxpy-s64-train",
            "saxpy",
            "saxpy_s64",
            "elementwise",
            Solve,
            1,
            "saxpy_s64",
            Train,
        ),
        // The wide compose consumers are taught on their own shapes, not on the
        // distant shape that is held out. `vec9` and `vec16` are shaped vector
        // reductions; the audit-only cases
        // (`matmul3`, `matmul4`, `select16`) are a different shape each.
        make(
            "gen-vec9-sum-train",
            "vec9",
            "vec9_sum",
            "elementwise",
            Solve,
            1,
            "vec9_sum",
            Train,
        ),
        make(
            "gen-vec16-sum-train",
            "vec16",
            "vec16_sum",
            "elementwise",
            Solve,
            1,
            "vec16_sum",
            Train,
        ),
        make(
            "gen-elem-saxpy-oct-cal",
            "saxpy",
            "saxpy_oct",
            "elementwise",
            Solve,
            1,
            "saxpy_oct",
            Calibration,
        ),
        make(
            "gen-elem-affine-oct",
            "affine",
            "affine_oct",
            "elementwise",
            Solve,
            2,
            "affine_oct",
            Calibration,
        ),
        make(
            "gen-elem-affine-s16-holdout",
            "affine",
            "affine_s16",
            "elementwise",
            Solve,
            1,
            "affine_s16",
            Heldout,
        ),
        // Reduction parameters.
        make(
            "gen-reduce-8-128",
            "reduce8",
            "reduce128",
            "reduction",
            Solve,
            2,
            "reduce128",
            Train,
        ),
        make(
            "gen-reduce-32-128",
            "reduce32",
            "reduce128",
            "reduction",
            Solve,
            2,
            "reduce128",
            Calibration,
        ),
        make(
            "gen-reduce-16-32-holdout",
            "reduce16",
            "reduce32",
            "reduction",
            Solve,
            2,
            "reduce32",
            Heldout,
        ),
        // Matrix variants.
        make(
            "gen-matmul-colsum",
            "matmul2",
            "matmul2_colsum",
            "matrix",
            Solve,
            2,
            "matmul2_colsum",
            Train,
        ),
        make(
            "gen-matmul-sum-cal",
            "matmul2",
            "matmul2_sum",
            "matrix",
            Solve,
            2,
            "matmul2_sum",
            Calibration,
        ),
        make(
            "gen-matmul-sum-holdout",
            "matmul2",
            "matmul2_sum",
            "matrix",
            Solve,
            1,
            "matmul2_sum",
            Heldout,
        ),
        // Select variants.
        make(
            "gen-select-min",
            "pair",
            "select_min",
            "select",
            Solve,
            1,
            "select_min",
            Train,
        ),
        make(
            "gen-select-max-cal",
            "select_max",
            "select_max",
            "select",
            Solve,
            1,
            "select_max",
            Calibration,
        ),
        make(
            "gen-select-min-holdout",
            "pair",
            "select_min",
            "select",
            Solve,
            1,
            "select_min",
            Heldout,
        ),
        // Refine variants.
        make(
            "gen-refine-affine-s16",
            "affine_s16",
            "affine_s16",
            "refine",
            Solve,
            1,
            "affine_s16",
            Train,
        ),
        make(
            "gen-refine-saxpy-s16",
            "saxpy_s16",
            "saxpy_s16",
            "refine",
            Solve,
            1,
            "saxpy_s16",
            Calibration,
        ),
        make(
            "gen-refine-affine-oct-holdout",
            "affine_oct",
            "affine_oct",
            "refine",
            Solve,
            1,
            "affine_oct",
            Heldout,
        ),
        // Control / refusal variants.
        make(
            "gen-refuse-saxpy-matmul",
            "saxpy",
            "matmul2_sum",
            "control",
            Refuse,
            1,
            "matmul2_sum",
            Train,
        ),
        make(
            "gen-refuse-affine-matmul",
            "affine",
            "matmul2_colsum",
            "control",
            Refuse,
            1,
            "matmul2_colsum",
            Calibration,
        ),
        make(
            "gen-refuse-saxpy-select-holdout",
            "saxpy",
            "select_max",
            "control",
            Refuse,
            1,
            "select_max",
            Heldout,
        ),
    ]
}

/// Audit-only cases for generalization. These are intentionally **not** part
/// of any training or adoption split.
///
/// The first four are novel compositions: a duplicate-line start (`*_dup`)
/// and a deep `s16` target need a three-action solver
/// (`reduce_sum + depth + refinements`) that no training case requires. The next
/// three are high-feature-distance retention cases (large reduction resizes
/// and a matrix transpose) that the base already solves. The last is a
/// high-distance control that must stay refused.
pub fn hard_corpus() -> Vec<CorpusCase> {
    use CaseExpect::{Refuse, Solve};
    use Split::Hard;

    vec![
        make(
            "hard-refine-s16-saxpy",
            "saxpy_dup",
            "saxpy_s16",
            "elementwise",
            Solve,
            2,
            "saxpy_s16",
            Hard,
        ),
        make(
            "hard-refine-s16-affine",
            "affine_dup",
            "affine_s16",
            "elementwise",
            Solve,
            2,
            "affine_s16",
            Hard,
        ),
        make(
            "hard-cross-saxpy-affine-s16",
            "saxpy_dup",
            "affine_s16",
            "elementwise",
            Solve,
            2,
            "affine_s16",
            Hard,
        ),
        make(
            "hard-cross-affine-saxpy-s16",
            "affine_dup",
            "saxpy_s16",
            "elementwise",
            Solve,
            2,
            "saxpy_s16",
            Hard,
        ),
        // Deeper than s16: six doublings. The adopted harness must either
        // repeat `depth + 1` or use the refinement rounds.
        make(
            "hard-affine-s64",
            "affine",
            "affine_s64",
            "elementwise",
            Solve,
            2,
            "affine_s64",
            Hard,
        ),
        make(
            "hard-reduce-128-8",
            "reduce128",
            "reduce8",
            "reduction",
            Solve,
            2,
            "reduce8",
            Hard,
        ),
        make(
            "hard-reduce-32-64",
            "reduce32",
            "reduce64",
            "reduction",
            Solve,
            2,
            "reduce64",
            Hard,
        ),
        make(
            "hard-matmul-sum-colsum",
            "matmul2_sum",
            "matmul2_colsum",
            "matrix",
            Solve,
            2,
            "matmul2_colsum",
            Hard,
        ),
        // Feature-distant: 18 inputs, 9 outputs, and a 9-input compose
        // field. No training case has this shape signature.
        make(
            "hard-matmul3-sum",
            "matmul3",
            "matmul3_sum",
            "matrix",
            Solve,
            2,
            "matmul3_sum",
            Hard,
        ),
        // Second distant shape and consumer: a [4,4] matmul field. The geometric
        // consumer is taught on `gen-vec16-sum-train`, never on this shape.
        make(
            "hard-matmul4-sum",
            "matmul4",
            "matmul4_sum",
            "matrix",
            Solve,
            2,
            "matmul4_sum",
            Hard,
        ),
        // A different mechanism at the same width: 32 inputs, 16 outputs, and
        // an elementwise select rather than a matrix product.
        make(
            "hard-select16-sum",
            "select16",
            "select16_sum",
            "select",
            Solve,
            2,
            "select16_sum",
            Hard,
        ),
        make(
            "hard-refuse-reduce-matmul",
            "reduce128",
            "matmul2",
            "control",
            Refuse,
            2,
            "matmul2",
            Hard,
        ),
    ]
}

pub fn hard_cases() -> Vec<Case> {
    hard_corpus().into_iter().map(|c| c.case).collect()
}

pub fn generated_split_cases(split: Split) -> Vec<Case> {
    generated_corpus()
        .into_iter()
        .filter(|c| c.split == split)
        .map(|c| c.case)
        .collect()
}

pub fn task_specs() -> Vec<TaskSpec> {
    generated_corpus()
        .into_iter()
        .map(|c| {
            let start = c.case.intent.input.label.clone().unwrap_or_default();
            let target = c.case.intent.output.label.clone().unwrap_or_default();
            let (target_outputs, target_ops) = meta(&target);
            let lib = library();
            TaskSpec {
                id: c.case.intent.id.clone(),
                start_text: lib.get(&start).cloned().unwrap_or_default(),
                target_text: lib.get(&target).cloned().unwrap_or_default(),
                example_inputs: c.case.examples.clone(),
                start,
                target,
                family: c.case.family.clone(),
                expect: match c.case.expect {
                    CaseExpect::Solve => "solve".to_string(),
                    CaseExpect::Refuse => "refuse".to_string(),
                },
                depth: c.case.intent.depth,
                examples: c.case.intent.output.label.clone().unwrap_or_default(),
                split: c.split.as_str().to_string(),
                start_outputs: c.case.start_outputs,
                start_ops: c.case.start_ops,
                target_outputs,
                target_ops,
            }
        })
        .collect()
}

pub fn split_cases(split: Split) -> Vec<Case> {
    corpus()
        .into_iter()
        .filter(|c| c.split == split)
        .map(|c| c.case)
        .collect()
}

pub fn all_cases() -> Vec<Case> {
    corpus().into_iter().map(|c| c.case).collect()
}
