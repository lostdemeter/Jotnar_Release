//! A tiny, bounded family registry.
//!
//! Families are *builders*, not stored answers. A fit derives a family's
//! parameters from the target listing, then the builder instantiates a fresh
//! candidate. The gate still has to accept it.

use std::collections::BTreeMap;

use crate::state::State;

pub type Params = BTreeMap<String, String>;

#[derive(Clone, Copy)]
pub struct Family {
    pub name: &'static str,
    pub tags: &'static [&'static str],
    pub arity: Option<usize>,
    pub build: fn(&Params) -> Result<String, String>,
    pub fit: fn(&State) -> Option<Params>,
}

pub fn families() -> Vec<Family> {
    vec![
        Family {
            name: "saxpy",
            tags: &["linear", "elementwise"],
            arity: Some(3),
            build: build_saxpy,
            fit: fit_saxpy,
        },
        Family {
            name: "reduce",
            tags: &["reduction", "tree"],
            arity: None,
            build: build_reduce,
            fit: fit_reduce,
        },
        Family {
            name: "matmul2",
            tags: &["matrix", "dot"],
            arity: Some(8),
            build: build_matmul2,
            fit: fit_matmul2,
        },
        Family {
            name: "select",
            tags: &["select", "predicate"],
            arity: Some(3),
            build: build_select,
            fit: fit_select,
        },
        Family {
            name: "select_max",
            tags: &["select", "predicate"],
            arity: Some(2),
            build: build_select_max,
            fit: fit_select_max,
        },
        Family {
            name: "add2",
            tags: &["linear"],
            arity: Some(2),
            build: build_add2,
            fit: fit_add2,
        },
        Family {
            name: "select_min",
            tags: &["select", "predicate"],
            arity: Some(2),
            build: build_select_min,
            fit: fit_select_min,
        },
        Family {
            name: "saxpy_dup",
            tags: &["linear", "elementwise"],
            arity: Some(3),
            build: build_saxpy_dup,
            fit: fit_saxpy_dup,
        },
        Family {
            name: "affine_dup",
            tags: &["linear", "elementwise"],
            arity: None,
            build: build_affine_dup,
            fit: fit_affine_dup,
        },
        Family {
            name: "matmul2_dup",
            tags: &["matrix", "dot"],
            arity: Some(8),
            build: build_matmul2_dup,
            fit: fit_matmul2_dup,
        },
    ]
}

pub fn family(name: &str) -> Option<Family> {
    families().into_iter().find(|f| f.name == name)
}

pub fn tags_of(name: &str) -> &'static [&'static str] {
    family(name).map(|f| f.tags).unwrap_or(&[])
}

// ---------------------------------------------------------------------------
// builders
// ---------------------------------------------------------------------------

pub fn build_saxpy(_: &Params) -> Result<String, String> {
    Ok("\
# SAXPY: OUT = A*X + Y.
IN A
IN X
IN Y

ax = IMUL(A, X)
OUT = IADD(ax, Y)
"
    .to_string())
}

pub fn build_saxpy_double(_: &Params) -> Result<String, String> {
    Ok("\
# SAXPY, then double the result.
IN A
IN X
IN Y

ax = IMUL(A, X)
s = IADD(ax, Y)
OUT = IADD(s, s)
"
    .to_string())
}

pub fn build_saxpy_quad(_: &Params) -> Result<String, String> {
    Ok("\
IN A
IN X
IN Y

ax = IMUL(A, X)
s1 = IADD(ax, Y)
s2 = IADD(s1, s1)
OUT = IADD(s2, s2)
"
    .to_string())
}

pub fn build_saxpy_oct(_: &Params) -> Result<String, String> {
    Ok("\
IN A
IN X
IN Y

ax = IMUL(A, X)
s1 = IADD(ax, Y)
s2 = IADD(s1, s1)
s3 = IADD(s2, s2)
OUT = IADD(s3, s3)
"
    .to_string())
}

pub fn build_saxpy_dup(_: &Params) -> Result<String, String> {
    let mut text = build_saxpy(&Params::new())?;
    text.push_str("\ndup = IMUL(A, X)\n");
    Ok(text)
}

pub fn build_saxpy_s16(_: &Params) -> Result<String, String> {
    Ok("\
# SAXPY, then quadruple-double the result (16x).
IN A
IN X
IN Y

ax = IMUL(A, X)
s1 = IADD(ax, Y)
s2 = IADD(s1, s1)
s3 = IADD(s2, s2)
s4 = IADD(s3, s3)
OUT = IADD(s4, s4)
"
    .to_string())
}

pub fn build_affine(_: &Params) -> Result<String, String> {
    Ok("\
# Affine: OUT = A*X + Y.
IN A
IN X
IN Y

OUT = IMAD(A, X, Y)
"
    .to_string())
}

pub fn build_affine_s16(_: &Params) -> Result<String, String> {
    Ok("\
# Affine, then quadruple-double the result (16x).
IN A
IN X
IN Y

ax = IMAD(A, X, Y)
s1 = IADD(ax, ax)
s2 = IADD(s1, s1)
s3 = IADD(s2, s2)
OUT = IADD(s3, s3)
"
    .to_string())
}

pub fn build_affine_quad(_: &Params) -> Result<String, String> {
    Ok("\
IN A
IN X
IN Y

ax = IMAD(A, X, Y)
s1 = IADD(ax, ax)
OUT = IADD(s1, s1)
"
    .to_string())
}

pub fn build_affine_oct(_: &Params) -> Result<String, String> {
    Ok("\
IN A
IN X
IN Y

ax = IMAD(A, X, Y)
s1 = IADD(ax, ax)
s2 = IADD(s1, s1)
OUT = IADD(s2, s2)
"
    .to_string())
}

pub fn build_affine_dup(_: &Params) -> Result<String, String> {
    let mut text = build_affine(&Params::new())?;
    text.push_str("dup = IMAD(A, X, Y)\n");
    Ok(text)
}

pub fn build_add2(_: &Params) -> Result<String, String> {
    Ok("\
# Add two streams.
IN A
IN B

OUT = IADD(A, B)
"
    .to_string())
}

pub fn build_reduce(params: &Params) -> Result<String, String> {
    let n: usize = params
        .get("n")
        .ok_or_else(|| "reduce needs params.n".to_string())?
        .parse()
        .map_err(|_| "reduce params.n must be an integer".to_string())?;
    if n < 2 || !n.is_power_of_two() {
        return Err(format!("reduce needs a power of two >= 2, got {n}"));
    }
    Ok(reduce_listing(n))
}

pub fn build_reduce8_dup(_: &Params) -> Result<String, String> {
    let mut text = reduce_listing(8);
    text.push_str("\ndup = IADD(A0, A1)\n");
    Ok(text)
}

pub fn reduce_listing(n: usize) -> String {
    let mut lines: Vec<String> = (0..n).map(|i| format!("IN A{i}")).collect();
    lines.push(String::new());
    let mut level: Vec<String> = (0..n).map(|i| format!("A{i}")).collect();
    let mut depth = 0usize;
    while level.len() > 1 {
        let mut next = Vec::new();
        let mut i = 0usize;
        while i < level.len() {
            let name = if level.len() == 2 {
                "OUT".to_string()
            } else {
                format!("l{depth}_{}", next.len())
            };
            lines.push(format!("{name} = IADD({}, {})", level[i], level[i + 1]));
            next.push(name);
            i += 2;
        }
        level = next;
        depth += 1;
    }
    lines.join("\n") + "\n"
}

/// `n x n` integer matrix multiply as a shaped field: `I:HW, I:HW -> I:HW`.
pub fn build_matmul_n(n: usize) -> String {
    format!(
        "# {n}x{n} integer matrix multiply (shaped field).\n\
         IN A AS I:HW\n\
         IN B AS I:HW\n\
         \n\
         C = IMATMUL(A, B)\n"
    )
}

pub fn build_matmul_n_sum(n: usize) -> String {
    let mut text = build_matmul_n(n);
    text.push_str("\nOUT = IREDUCE_SUM(C)\n");
    text
}

pub fn build_matmul_n_dup(n: usize) -> String {
    let mut text = build_matmul_n(n);
    text.push_str("\ndup = IMATMUL(A, B)\n");
    text
}

pub fn build_matmul2(_: &Params) -> Result<String, String> {
    Ok(build_matmul_n(2))
}

pub fn build_matmul2_sum(_: &Params) -> Result<String, String> {
    Ok(build_matmul_n_sum(2))
}

pub fn build_matmul2_colsum(_: &Params) -> Result<String, String> {
    // Exact integer addition is associative, so the column-sum order is the
    // same field reduction. The label stays for the corpus; the geometry is
    // one all-axis reduction, not a second consumer.
    Ok(build_matmul_n_sum(2))
}

pub fn build_matmul2_dup(_: &Params) -> Result<String, String> {
    Ok(build_matmul_n_dup(2))
}

pub fn build_matmul3(_: &Params) -> Result<String, String> {
    Ok(build_matmul_n(3))
}

pub fn build_matmul3_sum(_: &Params) -> Result<String, String> {
    Ok(build_matmul_n_sum(3))
}

pub fn build_matmul3_dup(_: &Params) -> Result<String, String> {
    Ok(build_matmul_n_dup(3))
}

/// An `n`-lane integer vector as one shaped field: `I:SEQ -> I:SEQ`.
pub fn build_vec(n: usize) -> String {
    format!(
        "# {n}-lane integer vector (shaped field).\n\
         IN A AS I:SEQ\n\
         \n\
         C = IADD(A, A)\n"
    )
}

/// The vector plus the geometric all-element reduction.
pub fn build_vec_sum(n: usize) -> String {
    let mut text = build_vec(n);
    text.push_str("\nOUT = IREDUCE_SUM(C)\n");
    text
}

/// An `n`-lane elementwise maximum as one shaped field.
pub fn build_select_wide(n: usize) -> String {
    format!(
        "# {n}-lane elementwise maximum via explicit predicates (shaped field).\n\
         IN A AS I:SEQ\n\
         IN B AS I:SEQ\n\
         \n\
         C = ILT(A, B)\n\
         M = ISELECT(C, B, A)\n"
    )
}

/// The wide select plus the geometric all-element reduction.
pub fn build_select_wide_sum(n: usize) -> String {
    let mut text = build_select_wide(n);
    text.push_str("\nOUT = IREDUCE_SUM(M)\n");
    text
}

pub fn build_vec9(_: &Params) -> Result<String, String> {
    Ok(build_vec(9))
}

pub fn build_vec9_sum(_: &Params) -> Result<String, String> {
    Ok(build_vec_sum(9))
}

pub fn build_vec16(_: &Params) -> Result<String, String> {
    Ok(build_vec(16))
}

pub fn build_vec16_sum(_: &Params) -> Result<String, String> {
    Ok(build_vec_sum(16))
}

pub fn build_matmul4(_: &Params) -> Result<String, String> {
    Ok(build_matmul_n(4))
}

pub fn build_matmul4_sum(_: &Params) -> Result<String, String> {
    Ok(build_matmul_n_sum(4))
}

pub fn build_select16(_: &Params) -> Result<String, String> {
    Ok(build_select_wide(16))
}

pub fn build_select16_sum(_: &Params) -> Result<String, String> {
    Ok(build_select_wide_sum(16))
}

pub fn build_select(_: &Params) -> Result<String, String> {
    Ok("\
# Two-way select on an integer sequence.
IN C AS I:SEQ
IN A AS I:SEQ
IN B AS I:SEQ

OUT = ISELECT(C, A, B)
"
    .to_string())
}

pub fn build_select_max(_: &Params) -> Result<String, String> {
    Ok("\
# Elementwise maximum via an explicit integer predicate.
IN A AS I:SEQ
IN B AS I:SEQ

C = ILT(A, B)
OUT = ISELECT(C, B, A)
"
    .to_string())
}

pub fn build_select_min(_: &Params) -> Result<String, String> {
    Ok("\
# Elementwise minimum via an explicit integer predicate.
IN A AS I:SEQ
IN B AS I:SEQ

C = ILT(A, B)
OUT = ISELECT(C, A, B)
"
    .to_string())
}

// ---------------------------------------------------------------------------
// fits
// ---------------------------------------------------------------------------

fn same_shape_as(state: &State, text: &str) -> bool {
    match State::new("probe", text) {
        Ok(probe) => probe.structural_signature() == state.structural_signature(),
        Err(_) => false,
    }
}

fn named(state: &State, name: &str, text: &str) -> bool {
    state.name == name && same_shape_as(state, text)
}

pub fn fit_saxpy(state: &State) -> Option<Params> {
    named(state, "saxpy", &build_saxpy(&Params::new()).ok()?).then(Params::new)
}

pub fn fit_matmul2(state: &State) -> Option<Params> {
    named(state, "matmul2", &build_matmul2(&Params::new()).ok()?).then(Params::new)
}

pub fn fit_select(state: &State) -> Option<Params> {
    named(state, "select", &build_select(&Params::new()).ok()?).then(Params::new)
}

pub fn fit_select_max(state: &State) -> Option<Params> {
    named(state, "select_max", &build_select_max(&Params::new()).ok()?).then(Params::new)
}

pub fn fit_add2(state: &State) -> Option<Params> {
    named(state, "add2", &build_add2(&Params::new()).ok()?).then(Params::new)
}

pub fn fit_select_min(state: &State) -> Option<Params> {
    named(state, "select_min", &build_select_min(&Params::new()).ok()?).then(Params::new)
}

pub fn fit_saxpy_dup(state: &State) -> Option<Params> {
    named(state, "saxpy_dup", &build_saxpy_dup(&Params::new()).ok()?).then(Params::new)
}

pub fn fit_affine_dup(state: &State) -> Option<Params> {
    named(state, "affine_dup", &build_affine_dup(&Params::new()).ok()?).then(Params::new)
}

pub fn fit_matmul2_dup(state: &State) -> Option<Params> {
    named(
        state,
        "matmul2_dup",
        &build_matmul2_dup(&Params::new()).ok()?,
    )
    .then(Params::new)
}

pub fn fit_reduce(state: &State) -> Option<Params> {
    let n = state
        .input_names()
        .iter()
        .filter(|name| {
            name.strip_prefix('A')
                .map(|rest| !rest.is_empty() && rest.chars().all(|c| c.is_ascii_digit()))
                .unwrap_or(false)
        })
        .count();
    if n < 2 || !n.is_power_of_two() {
        return None;
    }
    let mut p = Params::new();
    p.insert("n".to_string(), n.to_string());
    Some(p)
}
