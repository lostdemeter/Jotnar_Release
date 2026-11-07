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

pub fn build_matmul2(_: &Params) -> Result<String, String> {
    Ok("\
# 2x2 matrix multiply.
IN a00
IN a01
IN a10
IN a11
IN b00
IN b01
IN b10
IN b11

t00 = IMUL(a00, b00)
t01 = IMUL(a01, b10)
c00 = IADD(t00, t01)
t10 = IMUL(a10, b00)
t11 = IMUL(a11, b10)
c10 = IADD(t10, t11)
t02 = IMUL(a00, b01)
t03 = IMUL(a01, b11)
c01 = IADD(t02, t03)
t12 = IMUL(a10, b01)
t13 = IMUL(a11, b11)
c11 = IADD(t12, t13)
"
    .to_string())
}

pub fn build_matmul2_sum(_: &Params) -> Result<String, String> {
    let mut text = build_matmul2(&Params::new())?;
    text.push_str("\nt0 = IADD(c00, c01)\nt1 = IADD(c10, c11)\nOUT = IADD(t0, t1)\n");
    Ok(text)
}

pub fn build_matmul3(_: &Params) -> Result<String, String> {
    let mut lines = vec!["# 3x3 matrix multiply.".to_string()];
    for i in 0..3 {
        for j in 0..3 {
            lines.push(format!("IN a{i}{j}"));
        }
    }
    for i in 0..3 {
        for j in 0..3 {
            lines.push(format!("IN b{i}{j}"));
        }
    }
    lines.push(String::new());
    for i in 0..3 {
        for j in 0..3 {
            let t0 = format!("t{i}{j}0");
            let t1 = format!("t{i}{j}1");
            let t2 = format!("t{i}{j}2");
            let s = format!("s{i}{j}");
            lines.push(format!("{t0} = IMUL(a{i}0, b0{j})"));
            lines.push(format!("{t1} = IMUL(a{i}1, b1{j})"));
            lines.push(format!("{s} = IADD({t0}, {t1})"));
            lines.push(format!("{t2} = IMUL(a{i}2, b2{j})"));
            lines.push(format!("c{i}{j} = IADD({s}, {t2})"));
        }
    }
    Ok(lines.join("\n") + "\n")
}

pub fn build_matmul3_sum(_: &Params) -> Result<String, String> {
    let mut text = build_matmul3(&Params::new())?;
    text.push_str(
        "\ns0 = IADD(c00, c01)\ns1 = IADD(c02, c10)\ns2 = IADD(c11, c12)\n\
         s3 = IADD(c20, c21)\ns4 = IADD(s0, s1)\ns5 = IADD(s2, s3)\n\
         s6 = IADD(s4, s5)\nOUT = IADD(s6, c22)\n",
    );
    Ok(text)
}

pub fn build_matmul3_dup(_: &Params) -> Result<String, String> {
    let mut text = build_matmul3(&Params::new())?;
    text.push_str("\ndup = IMUL(a00, b00)\n");
    Ok(text)
}

pub fn build_select(_: &Params) -> Result<String, String> {
    Ok("\
# Two-way select.
IN C
IN A
IN B

OUT = ISELECT(C, A, B)
"
    .to_string())
}

pub fn build_select_max(_: &Params) -> Result<String, String> {
    Ok("\
# Elementwise maximum via an explicit integer predicate.
IN A
IN B

C = ILT(A, B)
OUT = ISELECT(C, B, A)
"
    .to_string())
}

pub fn build_select_min(_: &Params) -> Result<String, String> {
    Ok("\
# Elementwise minimum via an explicit integer predicate.
IN A
IN B

C = ILT(A, B)
OUT = ISELECT(C, A, B)
"
    .to_string())
}

pub fn build_matmul2_colsum(_: &Params) -> Result<String, String> {
    Ok("\
# 2x2 product, summed by columns then combined.
IN a00
IN a01
IN a10
IN a11
IN b00
IN b01
IN b10
IN b11

t00 = IMUL(a00, b00)
t01 = IMUL(a01, b10)
c00 = IADD(t00, t01)
t10 = IMUL(a10, b00)
t11 = IMUL(a11, b10)
c10 = IADD(t10, t11)
t02 = IMUL(a00, b01)
t03 = IMUL(a01, b11)
c01 = IADD(t02, t03)
t12 = IMUL(a10, b01)
t13 = IMUL(a11, b11)
c11 = IADD(t12, t13)
c0 = IADD(c00, c10)
c1 = IADD(c01, c11)
OUT = IADD(c0, c1)
"
    .to_string())
}

pub fn build_matmul2_dup(_: &Params) -> Result<String, String> {
    let mut text = build_matmul2(&Params::new())?;
    text.push_str("\ndup = IMUL(a00, b00)\n");
    Ok(text)
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
