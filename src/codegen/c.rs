//! C99 code emission (the portable reference backend).
//!
//! The pattern text intentionally mirrors `chain/emit_c.py`'s lowering so
//! the numeric order (and therefore the float agreement class) is the same.
//! Unsupported ops fail loud; there is no silent fallback.

use std::collections::HashMap;

use super::infer::{CKind, StreamInfo};
use crate::error::{Error, Result};
use crate::ir::Config;

pub struct RopeCtx {
    pub p: usize,
    pub d: usize,
    pub cos: crate::value::Triple,
    pub sin: crate::value::Triple,
}

pub struct Ctx {
    pub streams: HashMap<String, StreamInfo>,
    pub config: Config,
    pub gathered: HashMap<String, (usize, String)>,
    pub theta: Vec<f64>,
    pub inputs: Vec<String>,
    pub outputs: Vec<String>,
    pub nonfpu: bool,
    /// `Some(M)` enables the T+mantissa (T2) value format with M-bit residual.
    pub t2_m: Option<i64>,
    pub mns: Vec<String>,
    pub rope: Option<RopeCtx>,
}

fn fill(s: &str, vars: &[(&str, String)]) -> String {
    let mut out = s.to_string();
    for (k, v) in vars {
        out = out.replace(&format!("{{{k}}}"), v);
    }
    out
}

fn c_i64(v: i64) -> String {
    if v == i64::MIN {
        "(-9223372036854775807LL - 1LL)".to_string()
    } else {
        format!("{v}LL")
    }
}

fn int_lit(a: &str, what: &str) -> Result<i64> {
    let f: f64 = a
        .parse()
        .map_err(|_| Error::asm(format!("{what}: need integral literal, got {a}")))?;
    if !f.is_finite() || f.fract() != 0.0 {
        return Err(Error::asm(format!(
            "{what}: need integral literal, got {a}"
        )));
    }
    Ok(f as i64)
}

fn float_lit(v: f64) -> String {
    format!("{v:?}")
}

impl Ctx {
    pub(crate) fn s(&self, name: &str) -> Result<&StreamInfo> {
        self.streams
            .get(name)
            .ok_or_else(|| Error::asm(format!("C backend: unknown stream '{name}'")))
    }

    pub(crate) fn width(&self, name: &str, where_: &str) -> Result<usize> {
        let s = self.s(name)?;
        s.shape
            .last()
            .copied()
            .flatten()
            .ok_or_else(|| Error::unimplemented(format!("C backend: dynamic width in {where_}")))
    }

    fn rows(&self, name: &str) -> Result<usize> {
        self.s(name)?
            .rows()
            .ok_or_else(|| Error::unimplemented(format!("C backend: dynamic rows in {name}")))
    }

    pub(crate) fn elems(&self, name: &str) -> Result<usize> {
        self.s(name)?
            .elems()
            .ok_or_else(|| Error::unimplemented(format!("C backend: dynamic shape in {name}")))
    }
}

pub(crate) type Pat = (String, Option<(String, usize, String)>);

/// Emit one instruction body.
pub fn pattern(mn: &str, outs: &[String], args: &[String], ctx: &mut Ctx) -> Result<Pat> {
    if ctx.nonfpu {
        if let Some(r) = super::nonfpu::pattern(mn, outs, args, ctx) {
            return r;
        }
    }
    pattern_c(mn, outs, args, ctx)
}

pub(crate) fn pattern_c(mn: &str, outs: &[String], args: &[String], ctx: &mut Ctx) -> Result<Pat> {
    match mn {
        "GATHER" => {
            let (t, ids) = (args[0].as_str(), args[1].as_str());
            let o = outs[0].as_str();
            let tk = ctx.s(t)?.kind;
            let w = ctx.width(t, &format!("GATHER {o}"))?;
            match tk {
                CKind::T => {
                    let body = r#"/* {o} = GATHER({t}, {ids}) materialized + row views */
for (int64_t g_i = 0; g_i < {ids}_N; ++g_i) {
  memcpy(&{o}_s[g_i * {w}], &{t}_s[{ids}[g_i] * {w}], {w} * sizeof({o}_s[0]));
  memcpy(&{o}_e[g_i * {w}], &{t}_e[{ids}[g_i] * {w}], {w} * sizeof({o}_e[0]));
  memcpy(&{o}_z[g_i * {w}], &{t}_z[{ids}[g_i] * {w}], {w} * sizeof({o}_z[0]));
}
const int8_t *gr_{o}_rows_s[{ids}_N];
const int32_t *gr_{o}_rows_e[{ids}_N];
const uint8_t *gr_{o}_rows_z[{ids}_N];
for (int64_t g_i = 0; g_i < {ids}_N; ++g_i) {
  gr_{o}_rows_s[g_i] = &{o}_s[g_i * {w}];
  gr_{o}_rows_e[g_i] = &{o}_e[g_i * {w}];
  gr_{o}_rows_z[g_i] = &{o}_z[g_i * {w}];
}"#;
                    Ok((
                        fill(
                            body,
                            &[
                                ("o", o.into()),
                                ("t", t.into()),
                                ("ids", ids.into()),
                                ("w", w.to_string()),
                            ],
                        ),
                        Some((o.to_string(), w, format!("gr_{o}"))),
                    ))
                }
                CKind::F => {
                    let body = r#"/* {o} = GATHER({t}, {ids}) materialized rows */
for (int64_t g_i = 0; g_i < {ids}_N; ++g_i)
  memcpy(&{o}[g_i * {w}], &{t}[{ids}[g_i] * {w}], {w} * sizeof(double));"#;
                    Ok((
                        fill(
                            body,
                            &[
                                ("o", o.into()),
                                ("t", t.into()),
                                ("ids", ids.into()),
                                ("w", w.to_string()),
                            ],
                        ),
                        None,
                    ))
                }
                CKind::I => Err(Error::unimplemented("GATHER table must be T/F (v0.2)")),
            }
        }
        "GATHER3" => {
            let (t, ii, jj) = (args[0].as_str(), args[1].as_str(), args[2].as_str());
            let o = outs[0].as_str();
            let tsh = ctx.s(t)?.shape.clone();
            if tsh.len() != 3 || tsh.iter().any(|d| d.is_none()) {
                return Err(Error::unimplemented(format!(
                    "GATHER3 needs baked (V1,V2,C), got {tsh:?}"
                )));
            }
            let (v1, v2, w) = (tsh[0].unwrap(), tsh[1].unwrap(), tsh[2].unwrap());
            let guard = fill(
                r#"if ({ii}_N != {jj}_N) return 20;
for (int64_t g_i = 0; g_i < {o}_N; ++g_i) {
  int64_t _a = {ii}[g_i], _b = {jj}[g_i];
  if (_a < 0 || _a >= {v1} || _b < 0 || _b >= {v2}) return 4;
"#,
                &[
                    ("ii", ii.into()),
                    ("jj", jj.into()),
                    ("o", o.into()),
                    ("v1", v1.to_string()),
                    ("v2", v2.to_string()),
                ],
            );
            match ctx.s(t)?.kind {
                CKind::T => {
                    let mut body = format!("/* {o} = GATHER3({t}, {ii}, {jj}) materialized + row views */\n{guard}");
                    for comp in ["s", "e", "z"] {
                        body.push_str(&format!(
                            "  memcpy(&{o}_{comp}[g_i * {w}], &{t}_{comp}[(_a * {v2} + _b) * {w}], {w} * sizeof({o}_{comp}[0]));\n"
                        ));
                    }
                    body.push_str("}\n");
                    body.push_str(&format!("const int8_t *g3_{o}_rows_s[{o}_N];\n"));
                    body.push_str(&format!("const int32_t *g3_{o}_rows_e[{o}_N];\n"));
                    body.push_str(&format!("const uint8_t *g3_{o}_rows_z[{o}_N];\n"));
                    body.push_str(&format!("for (int64_t g_i = 0; g_i < {o}_N; ++g_i) {{\n"));
                    for comp in ["s", "e", "z"] {
                        body.push_str(&format!(
                            "  g3_{o}_rows_{comp}[g_i] = &{o}_{comp}[g_i * {w}];\n"
                        ));
                    }
                    body.push('}');
                    Ok((body, Some((o.to_string(), w, format!("g3_{o}")))))
                }
                CKind::F => Ok((
                    format!(
                        "/* {o} = GATHER3({t}, {ii}, {jj}) materialized rows */\n{guard}  memcpy(&{o}[g_i * {w}], &{t}[(_a * {v2} + _b) * {w}], {w} * sizeof(double));\n}}"
                    ),
                    None,
                )),
                CKind::I => Err(Error::unimplemented("GATHER3 table must be T/F")),
            }
        }
        "IWINDOW" => {
            let (x, o) = (args[0].as_str(), outs[0].as_str());
            let ln = int_lit(&args[2], "IWINDOW len")?;
            let co = ctx.elems(o)?;
            if ctx.s(x)?.kind != CKind::I || ctx.s(x)?.shape.len() != 1 {
                return Err(Error::unimplemented("C backend IWINDOW needs 1D I"));
            }
            if ctx.streams.contains_key(&args[1]) {
                let s = args[1].as_str();
                if ctx.s(s)?.kind != CKind::I || ctx.s(s)?.shape.len() != 1 {
                    return Err(Error::unimplemented(
                        "C backend IWINDOW dynamic start needs 1-element I",
                    ));
                }
                let xn = ctx.s(x)?.shape[0].ok_or_else(|| {
                    Error::unimplemented("C backend IWINDOW dynamic: X length must be static")
                })?;
                return Ok((
                    fill(
                        r#"/* {o} = IWINDOW({x}, {s}[0], {ln}) dynamic, bounds-checked */
{ int64_t _s = {s}[0];
  if (_s < 0 || _s + {ln} > {xn}) return 4;
  for (int64_t e_i = 0; e_i < {co}; ++e_i)
    {o}[e_i] = {x}[_s + e_i]; }"#,
                        &[
                            ("o", o.into()),
                            ("x", x.into()),
                            ("s", s.into()),
                            ("ln", ln.to_string()),
                            ("xn", xn.to_string()),
                            ("co", co.to_string()),
                        ],
                    ),
                    None,
                ));
            }
            let lo = int_lit(&args[1], "IWINDOW start")?;
            Ok((
                fill(
                    r#"/* {o} = IWINDOW({x}, {lo}, {ln}) exact */
for (int64_t e_i = 0; e_i < {co}; ++e_i)
  {o}[e_i] = {x}[{lo} + e_i];"#,
                    &[
                        ("o", o.into()),
                        ("x", x.into()),
                        ("lo", lo.to_string()),
                        ("co", co.to_string()),
                    ],
                ),
                None,
            ))
        }
        "ICRC32" | "ICRC32C" => {
            let (x, o) = (args[0].as_str(), outs[0].as_str());
            let xn = ctx.s(x)?.shape[0].ok_or_else(|| {
                Error::unimplemented(format!("C backend {mn}: input length must be static"))
            })?;
            if mn == "ICRC32" {
                Ok((
                    fill(
                        r#"/* {o} = ICRC32({x}) exact (reflected, finalized) */
{ uint32_t _crc = 0xFFFFFFFFu;
  for (int64_t e_i = 0; e_i < {xn}; ++e_i) {
    _crc ^= (uint32_t)({x}[e_i] & 0xFF);
    for (int _b = 0; _b < 8; ++_b)
      _crc = (_crc >> 1) ^ (0xEDB88320u & ((uint32_t)-(int32_t)(_crc & 1u)));
  }
  {o}[0] = (int64_t)(_crc ^ 0xFFFFFFFFu); }"#,
                        &[("o", o.into()), ("x", x.into()), ("xn", xn.to_string())],
                    ),
                    None,
                ))
            } else {
                let seed = args[1].as_str();
                Ok((
                    fill(
                        r#"/* {o} = ICRC32C({x}, {seed}) exact (Castagnoli, seeded, raw) */
{ uint32_t _crc = (uint32_t)({seed}[0] & 0xFFFFFFFF);
  for (int64_t e_i = 0; e_i < {xn}; ++e_i) {
    _crc ^= (uint32_t)({x}[e_i] & 0xFF);
    for (int _b = 0; _b < 8; ++_b)
      _crc = (_crc >> 1) ^ (0x82F63B78u & ((uint32_t)-(int32_t)(_crc & 1u)));
  }
  {o}[0] = (int64_t)_crc; }"#,
                        &[
                            ("o", o.into()),
                            ("x", x.into()),
                            ("seed", seed.into()),
                            ("xn", xn.to_string()),
                        ],
                    ),
                    None,
                ))
            }
        }
        "IINFLATE" => {
            let (x, o) = (args[0].as_str(), outs[0].as_str());
            let n = int_lit(&args[1], "IINFLATE size")?;
            let xn = ctx.s(x)?.shape[0].ok_or_else(|| {
                Error::unimplemented("C backend IINFLATE: input length must be static")
            })?;
            Ok((
                fill(
                    r#"/* {o} = IINFLATE({x}, {n}) raw deflate, exact */
if (jn_inflate({x}, {xn}, {o}, {n}) != 0) return 31;"#,
                    &[
                        ("o", o.into()),
                        ("x", x.into()),
                        ("n", n.to_string()),
                        ("xn", xn.to_string()),
                    ],
                ),
                None,
            ))
        }
        "IVARINT" => {
            let (x, vo, lo) = (args[0].as_str(), outs[0].as_str(), outs[1].as_str());
            let xn = ctx.s(x)?.shape[0].ok_or_else(|| {
                Error::unimplemented("C backend IVARINT: input length must be static")
            })?;
            let p_expr = if ctx.streams.contains_key(&args[1]) {
                format!("{}[0]", args[1])
            } else {
                int_lit(&args[1], "IVARINT start")?.to_string()
            };
            Ok((
                fill(
                    r#"/* {vo},{lo} = IVARINT({x}, {st}) */
{ int64_t _p = {p_expr}; int64_t _v = 0, _n = 0;
  if (_p < 0 || _p >= {xn}) return 32;
  for (int _i = 0; _i < 8; _i++) {
    if (_p + _i >= {xn}) return 32;
    int64_t _b = {x}[_p + _i] & 0xFF;
    _v = (_v << 7) | (_b & 0x7F);
    if (!(_b & 0x80)) { _n = _i + 1; break; }
  }
  if (_n == 0) { if (_p + 8 >= {xn}) return 32;
    _v = (_v << 8) | ({x}[_p + 8] & 0xFF); _n = 9; }
  {vo}[0] = _v; {lo}[0] = _n; }"#,
                    &[
                        ("vo", vo.into()),
                        ("lo", lo.into()),
                        ("x", x.into()),
                        ("st", args[1].clone()),
                        ("p_expr", p_expr),
                        ("xn", xn.to_string()),
                    ],
                ),
                None,
            ))
        }
        "IFIND" => {
            let (x, pat, o) = (args[0].as_str(), args[1].as_str(), outs[0].as_str());
            let xn = ctx.s(x)?.shape[0]
                .ok_or_else(|| Error::unimplemented("C backend IFIND: static lengths required"))?;
            let m = ctx.s(pat)?.shape[0]
                .ok_or_else(|| Error::unimplemented("C backend IFIND: static lengths required"))?;
            Ok((
                fill(
                    r#"/* {o} = IFIND({x}, {pat}) exact */
{ int64_t _hit = -1;
  for (int64_t i = 0; i + {m} <= {xn}; ++i) {
    int64_t _k = 0;
    for (; _k < {m}; ++_k)
      if ({x}[i + _k] != {pat}[_k]) break;
    if (_k == {m}) { _hit = i; break; }
  }
  {o}[0] = _hit; }"#,
                    &[
                        ("o", o.into()),
                        ("x", x.into()),
                        ("pat", pat.into()),
                        ("m", m.to_string()),
                        ("xn", xn.to_string()),
                    ],
                ),
                None,
            ))
        }
        "IGATHER1" => {
            let (t, ids, o) = (args[0].as_str(), args[1].as_str(), outs[0].as_str());
            let vn = ctx.s(t)?.shape[0].ok_or_else(|| {
                Error::unimplemented("C backend IGATHER1: table length must be static")
            })?;
            let co = ctx.elems(o)?;
            Ok((
                fill(
                    r#"/* {o} = IGATHER1({t}, {ids}) exact, bounds-checked */
for (int64_t g_i = 0; g_i < {co}; ++g_i) {
  int64_t _b = {ids}[g_i];
  if (_b < 0 || _b >= {vn}) return 4;
  {o}[g_i] = {t}[_b];
}"#,
                    &[
                        ("o", o.into()),
                        ("t", t.into()),
                        ("ids", ids.into()),
                        ("vn", vn.to_string()),
                        ("co", co.to_string()),
                    ],
                ),
                None,
            ))
        }
        "IASSEMBLE" => {
            let (x, o) = (args[0].as_str(), outs[0].as_str());
            let lo = int_lit(&args[1], "IASSEMBLE start")?;
            let ln = int_lit(&args[2], "IASSEMBLE len")?;
            Ok((
                fill(
                    r#"/* {o} = IASSEMBLE({x}, {lo}, {ln}) little-endian, exact */
{ uint64_t _v = 0;
  for (int64_t e_i = 0; e_i < {ln}; ++e_i)
    _v |= ((uint64_t)({x}[{lo} + e_i] & 0xFF)) << (8 * e_i);
  {o}[0] = (int64_t)_v; }"#,
                    &[
                        ("o", o.into()),
                        ("x", x.into()),
                        ("lo", lo.to_string()),
                        ("ln", ln.to_string()),
                    ],
                ),
                None,
            ))
        }
        "ISUM" => {
            let (x, o) = (args[0].as_str(), outs[0].as_str());
            let lo = int_lit(&args[1], "ISUM start")?;
            let ln = int_lit(&args[2], "ISUM len")?;
            Ok((
                fill(
                    r#"/* {o} = ISUM({x}, {lo}, {ln}) exact */
{ int64_t _v = 0;
  for (int64_t e_i = 0; e_i < {ln}; ++e_i)
    _v += {x}[{lo} + e_i];
  {o}[0] = _v; }"#,
                    &[
                        ("o", o.into()),
                        ("x", x.into()),
                        ("lo", lo.to_string()),
                        ("ln", ln.to_string()),
                    ],
                ),
                None,
            ))
        }
        "ISCATTER1" => {
            let (base, ids, vals, o) = (
                args[0].as_str(),
                args[1].as_str(),
                args[2].as_str(),
                outs[0].as_str(),
            );
            let vn = ctx.s(base)?.shape[0].ok_or_else(|| {
                Error::unimplemented("C backend ISCATTER1: OUT length must be static")
            })?;
            let nids = ctx.s(ids)?.shape[0].ok_or_else(|| {
                Error::unimplemented("C backend ISCATTER1: IDS length must be static")
            })?;
            let co = ctx.elems(o)?;
            let copy = if base == o {
                String::new()
            } else {
                format!("for (int64_t s_i = 0; s_i < {co}; ++s_i) {o}[s_i] = {base}[s_i];\n")
            };
            Ok((
                fill(
                    r#"/* {o} = ISCATTER1({base}, {ids}, {vals}) exact, bounds-checked */
{copy}for (int64_t s_i = 0; s_i < {nids}; ++s_i) {
  int64_t _b = {ids}[s_i];
  if (_b < 0 || _b >= {vn}) return 4;
  {o}[_b] = {vals}[s_i];
}"#,
                    &[
                        ("o", o.into()),
                        ("base", base.into()),
                        ("ids", ids.into()),
                        ("vals", vals.into()),
                        ("vn", vn.to_string()),
                        ("nids", nids.to_string()),
                        ("copy", copy),
                    ],
                ),
                None,
            ))
        }
        "IADD" | "ISUB" | "IMUL" | "IMAD" | "IDIV" | "IMOD" | "IDIVU" | "IMODU" | "ISAR" => {
            let (a, o) = (args[0].as_str(), outs[0].as_str());
            let co = ctx.elems(o)?;
            let term = |x: &str, pos: usize| -> Result<String> {
                if ctx.streams.contains_key(x) {
                    Ok(format!("{x}[e_i]"))
                } else {
                    Ok(c_i64(int_lit(x, &format!("{mn} operand {pos}"))?))
                }
            };
            let tb = term(&args[1], 1)?;
            if mn == "ISAR" {
                if !ctx.streams.contains_key(&args[1]) {
                    let sv = int_lit(&args[1], "ISAR operand 1")?;
                    if !(0..=63).contains(&sv) {
                        return Err(Error::asm(format!("ISAR: shift must be 0..63, got {sv}")));
                    }
                }
                return Ok((
                    fill(
                        r#"/* {o} = ISAR({a}, {b}) arithmetic, count 0..63 checked */
for (int64_t e_i = 0; e_i < {co}; ++e_i) {
  int64_t _s = {tb};
  if (_s < 0 || _s > 63) return 30;
  {o}[e_i] = {a}[e_i] >> _s;
}"#,
                        &[
                            ("o", o.into()),
                            ("a", a.into()),
                            ("b", args[1].clone()),
                            ("tb", tb),
                            ("co", co.to_string()),
                        ],
                    ),
                    None,
                ));
            }
            if mn == "IDIVU" || mn == "IMODU" {
                if !ctx.streams.contains_key(&args[1])
                    && int_lit(&args[1], &format!("{mn} operand 1"))? == 0
                {
                    return Err(Error::asm(format!("{mn}: division by zero (literal)")));
                }
                let op = if mn == "IDIVU" { "/" } else { "%" };
                return Ok((
                    fill(
                        r#"/* {o} = {mn}({a}, {b}) unsigned, checked */
for (int64_t e_i = 0; e_i < {co}; ++e_i) {
  uint64_t _a = (uint64_t){a}[e_i];
  uint64_t _b = (uint64_t)({tb});
  if (_b == 0) return 30;
  {o}[e_i] = (int64_t)(_a {op} _b);
}"#,
                        &[
                            ("o", o.into()),
                            ("mn", mn.into()),
                            ("a", a.into()),
                            ("b", args[1].clone()),
                            ("tb", tb),
                            ("co", co.to_string()),
                            ("op", op.into()),
                        ],
                    ),
                    None,
                ));
            }
            if mn == "IDIV" || mn == "IMOD" {
                if !ctx.streams.contains_key(&args[1])
                    && int_lit(&args[1], &format!("{mn} operand 1"))? == 0
                {
                    return Err(Error::asm(format!("{mn}: division by zero (literal)")));
                }
                let op = if mn == "IDIV" { "/" } else { "%" };
                Ok((
                    fill(
                        r#"/* {o} = {mn}({a}, {b}) truncating, checked */
for (int64_t e_i = 0; e_i < {co}; ++e_i) {
  int64_t _a = {a}[e_i];
  int64_t _b = {tb};
  if (_b == 0 || (_a == INT64_MIN && _b == -1)) return 30;
  {o}[e_i] = _a {op} _b;
}"#,
                        &[
                            ("o", o.into()),
                            ("mn", mn.into()),
                            ("a", a.into()),
                            ("b", args[1].clone()),
                            ("tb", tb),
                            ("co", co.to_string()),
                            ("op", op.into()),
                        ],
                    ),
                    None,
                ))
            } else {
                let expr = match mn {
                    "IADD" => format!("((uint64_t){a}[e_i] + (uint64_t)({tb}))"),
                    "ISUB" => format!("((uint64_t){a}[e_i] - (uint64_t)({tb}))"),
                    "IMUL" => format!("((uint64_t){a}[e_i] * (uint64_t)({tb}))"),
                    _ => {
                        let tc = term(&args[2], 2)?;
                        format!("((uint64_t){a}[e_i] * (uint64_t)({tb}) + (uint64_t)({tc}))")
                    }
                };
                Ok((
                    fill(
                        r#"/* {o} = {mn}({a}, {b}) 64-bit wrap */
for (int64_t e_i = 0; e_i < {co}; ++e_i)
  {o}[e_i] = (int64_t){expr};"#,
                        &[
                            ("o", o.into()),
                            ("mn", mn.into()),
                            ("a", a.into()),
                            ("b", args[1..].join(", ")),
                            ("co", co.to_string()),
                            ("expr", expr),
                        ],
                    ),
                    None,
                ))
            }
        }
        "IMATMUL" => {
            let (a, b, o) = (args[0].as_str(), args[1].as_str(), outs[0].as_str());
            let (m, k, n) = (
                ctx.rows(a)?,
                ctx.width(a, "IMATMUL left")?,
                ctx.width(b, "IMATMUL right")?,
            );
            Ok((
                fill(
                    r#"/* {o} = IMATMUL({a}, {b}) exact 64-bit wrap, [{m}x{k}] * [{k}x{n}] */
for (int64_t i_i = 0; i_i < {m}; ++i_i) {
  for (int64_t j_i = 0; j_i < {n}; ++j_i) {
    int64_t _acc = 0;
    for (int64_t k_i = 0; k_i < {k}; ++k_i)
      _acc = (int64_t)((uint64_t)_acc + (uint64_t)((int64_t)((uint64_t){a}[i_i * {k} + k_i] * (uint64_t){b}[k_i * {n} + j_i])));
    {o}[i_i * {n} + j_i] = _acc;
  }
}"#,
                    &[
                        ("o", o.into()),
                        ("a", a.into()),
                        ("b", b.into()),
                        ("m", m.to_string()),
                        ("k", k.to_string()),
                        ("n", n.to_string()),
                    ],
                ),
                None,
            ))
        }
        "IREDUCE_SUM" => {
            let (x, o) = (args[0].as_str(), outs[0].as_str());
            let n = ctx.elems(x)?;
            Ok((
                fill(
                    r#"/* {o} = IREDUCE_SUM({x}) exact 64-bit wrap over {n} elements */
{ int64_t _acc = 0;
  for (int64_t e_i = 0; e_i < {n}; ++e_i)
    _acc = (int64_t)((uint64_t)_acc + (uint64_t){x}[e_i]);
  {o}[0] = _acc; }"#,
                    &[("o", o.into()), ("x", x.into()), ("n", n.to_string())],
                ),
                None,
            ))
        }
        "IXOR" | "IAND" | "IOR" => {
            let (a, o) = (args[0].as_str(), outs[0].as_str());
            let co = ctx.elems(o)?;
            let tb = if ctx.streams.contains_key(&args[1]) {
                format!("{}[e_i]", args[1])
            } else {
                c_i64(int_lit(&args[1], &format!("{mn} operand 1"))?)
            };
            let op = match mn {
                "IXOR" => "^",
                "IAND" => "&",
                _ => "|",
            };
            Ok((
                fill(
                    r#"/* {o} = {mn}({a}, {b}) 64-bit wrap */
for (int64_t e_i = 0; e_i < {co}; ++e_i)
  {o}[e_i] = (int64_t)(((uint64_t){a}[e_i]) {op} ((uint64_t)({tb})));"#,
                    &[
                        ("o", o.into()),
                        ("mn", mn.into()),
                        ("a", a.into()),
                        ("b", args[1].clone()),
                        ("tb", tb),
                        ("op", op.into()),
                        ("co", co.to_string()),
                    ],
                ),
                None,
            ))
        }
        "IEQ" | "ILT" | "IMAX" | "SADD" => {
            let (a, o) = (args[0].as_str(), outs[0].as_str());
            let co = ctx.elems(o)?;
            let tb = if ctx.streams.contains_key(&args[1]) {
                format!("{}[e_i]", args[1])
            } else {
                c_i64(int_lit(&args[1], &format!("{mn} operand 1"))?)
            };
            let body = match mn {
                "IEQ" => format!(
                    "  {o}[e_i] = (int64_t)(({a}[e_i]) == ({tb}));"
                ),
                "ILT" => format!(
                    "  {o}[e_i] = (int64_t)(({a}[e_i]) < ({tb}));"
                ),
                "IMAX" => format!(
                    "  {o}[e_i] = ({a}[e_i] > ({tb})) ? ({a}[e_i]) : ({tb});"
                ),
                _ => format!(
                    "  __int128 _s = (__int128){a}[e_i] + (__int128)({tb});\n  {o}[e_i] = _s > 9223372036854775807LL ? 9223372036854775807LL : (_s < (-9223372036854775807LL - 1) ? (-9223372036854775807LL - 1) : (int64_t)_s);"
                ),
            };
            let body = fill(&body, &[("o", o.into()), ("a", a.into()), ("tb", tb)]);
            Ok((
                fill(
                    "/* {o} = {mn}({a}, {b}) exact */\nfor (int64_t e_i = 0; e_i < {co}; ++e_i) {{\n{body}\n}}",
                    &[
                        ("o", o.into()),
                        ("mn", mn.into()),
                        ("a", a.into()),
                        ("b", args[1].clone()),
                        ("co", co.to_string()),
                        ("body", body),
                    ],
                ),
                None,
            ))
        }
        "ISELECT" => {
            let (c, a, b, o) = (
                args[0].as_str(),
                args[1].as_str(),
                args[2].as_str(),
                outs[0].as_str(),
            );
            let co = ctx.elems(o)?;
            let sel = |x: &str, pos: usize| -> Result<String> {
                if ctx.streams.contains_key(x) {
                    Ok(format!("{x}[e_i]"))
                } else {
                    Ok(c_i64(int_lit(x, &format!("ISELECT operand {pos}"))?))
                }
            };
            let ta = sel(a, 1)?;
            let tb = sel(b, 2)?;
            Ok((
                fill(
                    r#"/* {o} = ISELECT({c}, {a}, {b}) exact */
for (int64_t e_i = 0; e_i < {co}; ++e_i)
  {o}[e_i] = ({c}[e_i] != 0) ? ({ta}) : ({tb});"#,
                    &[
                        ("o", o.into()),
                        ("c", c.into()),
                        ("a", a.into()),
                        ("b", b.into()),
                        ("ta", ta),
                        ("tb", tb),
                        ("co", co.to_string()),
                    ],
                ),
                None,
            ))
        }
        "ISHL" | "ISHR" => {
            let (a, o) = (args[0].as_str(), outs[0].as_str());
            let co = ctx.elems(o)?;
            let s = int_lit(&args[1], &format!("{mn} shift"))?;
            let op = if mn == "ISHL" { "<<" } else { ">>" };
            Ok((
                fill(
                    r#"/* {o} = {mn}({a}, {s}) logical */
for (int64_t e_i = 0; e_i < {co}; ++e_i)
  {o}[e_i] = (int64_t)(((uint64_t){a}[e_i]) {op} {s});"#,
                    &[
                        ("o", o.into()),
                        ("mn", mn.into()),
                        ("a", a.into()),
                        ("s", s.to_string()),
                        ("op", op.into()),
                        ("co", co.to_string()),
                    ],
                ),
                None,
            ))
        }
        "T2I" => {
            let (x, o) = (args[0].as_str(), outs[0].as_str());
            let co = ctx.elems(o)?;
            Ok((
                fill(
                    r#"/* {o} = T2I({x}) phi-ladder -> int64, round half away from zero */
for (int64_t t_i = 0; t_i < {co}; ++t_i) {
  double _v = 0.0;
  if (!{x}_z[t_i]) {
    if ({x}_e[t_i] < 0 || {x}_e[t_i] > 65535) return 31;
    if ({x}_s[t_i] != 0)
      _v = ({x}_s[t_i] < 0 ? -1.0 : 1.0)
         * pow(1.618033988749895, ((double){x}_e[t_i] - 32768.0) / 512.0);
  }
  {o}[t_i] = (int64_t)(_v >= 0.0 ? _v + 0.5 : _v - 0.5);
}"#,
                    &[("o", o.into()), ("x", x.into()), ("co", co.to_string())],
                ),
                None,
            ))
        }
        "IGET" => {
            let (x, o) = (args[0].as_str(), outs[0].as_str());
            let idx = int_lit(&args[1], "IGET idx")?;
            let elems = ctx.elems(x)?;
            if idx < 0 || idx as usize >= elems {
                return Err(Error::asm(format!(
                    "IGET: index {idx} out of range {elems}"
                )));
            }
            Ok((
                fill(
                    "/* {o} = IGET({x}, {idx}) exact */\n{o}[0] = {x}[{idx}];",
                    &[("o", o.into()), ("x", x.into()), ("idx", idx.to_string())],
                ),
                None,
            ))
        }
        "F2I" | "I2F" => {
            let (x, o) = (args[0].as_str(), outs[0].as_str());
            let co = ctx.elems(o)?;
            if mn == "F2I" {
                Ok((
                    fill(
                        r#"/* {o} = F2I({x}) trunc toward zero, range-checked */
for (int64_t e_i = 0; e_i < {co}; ++e_i) {
  double _v = {x}[e_i];
  if (_v != _v || _v >= 9223372036854775808.0 || _v < -9223372036854775808.0) return 30;
  {o}[e_i] = (int64_t)_v;
}"#,
                        &[("o", o.into()), ("x", x.into()), ("co", co.to_string())],
                    ),
                    None,
                ))
            } else {
                Ok((
                    fill(
                        r#"/* {o} = I2F({x}) round-to-nearest-even */
for (int64_t e_i = 0; e_i < {co}; ++e_i)
  {o}[e_i] = (double){x}[e_i];"#,
                        &[("o", o.into()), ("x", x.into()), ("co", co.to_string())],
                    ),
                    None,
                ))
            }
        }
        "SCATTER" => {
            let (base, row, idx, o) = (
                args[0].as_str(),
                args[1].as_str(),
                args[2].as_str(),
                outs[0].as_str(),
            );
            let sh = ctx.s(base)?.shape.clone();
            if sh.len() != 2 || sh.iter().any(|d| d.is_none()) {
                return Err(Error::unimplemented("C SCATTER needs static 2D shapes"));
            }
            let (s, c) = (sh[0].unwrap(), sh[1].unwrap());
            Ok((
                fill(
                    r#"/* {o} = SCATTER({base},{row},{idx}) */
{ int64_t _n = {idx}[0];
  if (_n < 0 || _n >= {s}) return 4;
  memcpy({o}, {base}, ({s} * {c}) * sizeof(double));
  memcpy(&{o}[_n * {c}], {row}, {c} * sizeof(double)); }"#,
                    &[
                        ("o", o.into()),
                        ("base", base.into()),
                        ("row", row.into()),
                        ("idx", idx.into()),
                        ("s", s.to_string()),
                        ("c", c.to_string()),
                    ],
                ),
                None,
            ))
        }
        "ARGMAX" => {
            let (src, o) = (args[0].as_str(), outs[0].as_str());
            let kind = ctx.s(src)?.kind;
            match kind {
                CKind::T => {
                    let (w, pref) = ctx.gathered.get(src).cloned().ok_or_else(|| {
                        Error::asm(format!(
                            "ARGMAX C pattern v0.2 needs a gathered T row-stream, got '{src}'"
                        ))
                    })?;
                    Ok((
                        fill(
                            r#"/* {o} = ARGMAX({src}, 1) -- class rank + exponent, ties first */
for (int64_t a_i = 0; a_i < {o}_N; ++a_i) {
  int64_t best = 0; int bcls = -1; int64_t bkey = 0;
  for (int64_t a_j = 0; a_j < {w}; ++a_j) {
    int8_t s = {pref}_rows_s[a_i][a_j];
    int32_t e = {pref}_rows_e[a_i][a_j];
    uint8_t z = {pref}_rows_z[a_i][a_j];
    int cls = (!z && s > 0) ? 2 : (z ? 1 : 0);
    int64_t key = (!z && s > 0) ? (int64_t)e : (z ? 0 : -(int64_t)e);
    if (bcls < 0 || cls > bcls || (cls == bcls && key > bkey)) {
      best = a_j; bcls = cls; bkey = key;
    }
  }
  {o}[a_i] = best;
}"#,
                            &[
                                ("o", o.into()),
                                ("src", src.into()),
                                ("w", w.to_string()),
                                ("pref", pref),
                            ],
                        ),
                        None,
                    ))
                }
                CKind::F => {
                    if ctx.s(src)?.shape.len() != 2 {
                        return Err(Error::unimplemented("ARGMAX-F v0.2 needs 2D input"));
                    }
                    let w = ctx.width(src, &format!("ARGMAX {o}"))?;
                    Ok((
                        fill(
                            r#"/* {o} = ARGMAX({src}) float max-first */
for (int64_t a_i = 0; a_i < {o}_N; ++a_i) {
  int64_t best = 0; double bv = {src}[a_i * {w}];
  for (int64_t a_j = 1; a_j < {w}; ++a_j)
    if ({src}[a_i * {w} + a_j] > bv) { bv = {src}[a_i * {w} + a_j]; best = a_j; }
  {o}[a_i] = best;
}"#,
                            &[("o", o.into()), ("src", src.into()), ("w", w.to_string())],
                        ),
                        None,
                    ))
                }
                CKind::I => Err(Error::unimplemented("ARGMAX-I C pattern not ported")),
            }
        }
        "TOPK" => {
            let (x, vo, io) = (args[0].as_str(), outs[0].as_str(), outs[1].as_str());
            let w = ctx.width(x, "TOPK")?;
            let k = int_lit(&args[1], "TOPK k")?;
            Ok((
                fill(
                    r#"/* {vo},{io} = TOPK({x},{k}) single-pass, ties-first */
for (int64_t t_i = 0; t_i < {vo}_N; ++t_i) {
  int64_t t_n = 0;
  for (int64_t t_c = 0; t_c < {w}; ++t_c) {
    const double t_v = {x}[t_i * {w} + t_c];
    int64_t t_p = (t_n < {k}) ? t_n : {k};
    while (t_p > 0 && t_v > {vo}[t_i * {k} + t_p - 1]) --t_p;
    if (t_p >= {k}) continue;
    int64_t t_m = (t_n < {k}) ? t_n : {k} - 1;
    for (int64_t t_s = t_m; t_s > t_p; --t_s) {
      {vo}[t_i * {k} + t_s] = {vo}[t_i * {k} + t_s - 1];
      {io}[t_i * {k} + t_s] = {io}[t_i * {k} + t_s - 1];
    }
    {vo}[t_i * {k} + t_p] = t_v;
    {io}[t_i * {k} + t_p] = t_c;
    if (t_n < {k}) ++t_n;
  }
}"#,
                    &[
                        ("vo", vo.into()),
                        ("io", io.into()),
                        ("x", x.into()),
                        ("k", k.to_string()),
                        ("w", w.to_string()),
                    ],
                ),
                None,
            ))
        }
        "RMSNORM" | "LAYERNORM" => {
            let x = args[0].as_str();
            let o = outs[0].as_str();
            if ctx.s(x)?.kind != CKind::F {
                return Err(Error::unimplemented(format!(
                    "C backend {mn} needs F (T lives on nonfpu)"
                )));
            }
            let d = ctx.width(x, &format!("{mn} {o}"))?;
            if mn == "RMSNORM" {
                let w = args[1].as_str();
                let eps = ctx
                    .config
                    .get_f64("eps_rms")
                    .ok_or_else(|| Error::asm("RMSNORM C pattern needs CONFIG eps_rms"))?;
                Ok((
                    fill(
                        r#"/* {o} = RMSNORM({x}) eps={eps} */
for (int64_t r_i = 0; r_i < {o}_N; ++r_i) {
  double ss = 0;
  for (int64_t r_j = 0; r_j < {d}; ++r_j)
    ss += {x}[r_i * {d} + r_j] * {x}[r_i * {d} + r_j];
  double rs = 1.0 / sqrt(ss / {d} + {eps});
  for (int64_t r_j = 0; r_j < {d}; ++r_j)
    {o}[r_i * {d} + r_j] = {x}[r_i * {d} + r_j] * rs * {w}[r_j];
}"#,
                        &[
                            ("o", o.into()),
                            ("x", x.into()),
                            ("w", w.into()),
                            ("d", d.to_string()),
                            ("eps", float_lit(eps)),
                        ],
                    ),
                    None,
                ))
            } else {
                let (w, b) = (args[1].as_str(), args[2].as_str());
                let eps = ctx
                    .config
                    .get_f64("eps_ln")
                    .ok_or_else(|| Error::asm("LAYERNORM C pattern needs CONFIG eps_ln"))?;
                Ok((
                    fill(
                        r#"/* {o} = LAYERNORM({x}) eps={eps} */
for (int64_t r_i = 0; r_i < {o}_N; ++r_i) {
  double mu = 0;
  for (int64_t r_j = 0; r_j < {d}; ++r_j) mu += {x}[r_i * {d} + r_j];
  mu /= {d};
  double ss = 0;
  for (int64_t r_j = 0; r_j < {d}; ++r_j) {
    double dv = {x}[r_i * {d} + r_j] - mu;
    ss += dv * dv;
  }
  double rs = 1.0 / sqrt(ss / {d} + {eps});
  for (int64_t r_j = 0; r_j < {d}; ++r_j)
    {o}[r_i * {d} + r_j] = ({x}[r_i * {d} + r_j] - mu) * rs * {w}[r_j] + {b}[r_j];
}"#,
                        &[
                            ("o", o.into()),
                            ("x", x.into()),
                            ("w", w.into()),
                            ("b", b.into()),
                            ("d", d.to_string()),
                            ("eps", float_lit(eps)),
                        ],
                    ),
                    None,
                ))
            }
        }
        "DECONV2D" => {
            let (x, w, b, o) = (
                args[0].as_str(),
                args[1].as_str(),
                args[2].as_str(),
                outs[0].as_str(),
            );
            if ctx.s(x)?.kind != CKind::F {
                return Err(Error::unimplemented("C DECONV2D needs F"));
            }
            let xsh = ctx.s(x)?.shape.clone();
            let wsh = ctx.s(w)?.shape.clone();
            if xsh.len() != 3 || wsh.len() != 4 {
                return Err(Error::unimplemented(
                    "C DECONV2D needs HWC + (Cout,kh,kw,Cin)",
                ));
            }
            let (h, ww, cin) = (xsh[0].unwrap(), xsh[1].unwrap(), xsh[2].unwrap());
            let (cout, kh, kw) = (wsh[0].unwrap(), wsh[1].unwrap(), wsh[2].unwrap());
            let stride = int_lit(&args[3], "DECONV2D stride")?;
            let pad = int_lit(&args[4], "DECONV2D pad")?;
            let ho = (h as i64 - 1) * stride - 2 * pad + kh as i64;
            let wo = (ww as i64 - 1) * stride - 2 * pad + kw as i64;
            if ho <= 0 || wo <= 0 || stride <= 0 || pad < 0 {
                return Err(Error::asm("C DECONV2D bad geometry"));
            }
            let co = ctx.elems(o)?;
            Ok((
                fill(
                    r#"/* {o} = DECONV2D({x}) {cout}ch {kh}x{kw} s{stride} p{pad} */
memset({o}, 0, {co} * sizeof(double));
for (int64_t ch = 0; ch < {cout}; ++ch) {
  for (int64_t iy = 0; iy < {h}; ++iy)
    for (int64_t ix = 0; ix < {w}; ++ix)
      for (int64_t ky = 0; ky < {kh}; ++ky) {
        int64_t oy = iy * {stride} - {pad} + ky;
        if (oy < 0 || oy >= {ho}) continue;
        for (int64_t kx = 0; kx < {kw}; ++kx) {
          int64_t ox = ix * {stride} - {pad} + kx;
          if (ox < 0 || ox >= {wo}) continue;
          for (int64_t ci = 0; ci < {cin}; ++ci)
            {o}[(oy * {wo} + ox) * {cout} + ch] +=
              {x}[(iy * {w} + ix) * {cin} + ci]
              * {w}[((ch * {kh} + ky) * {kw} + kx) * {cin} + ci];
        }
      }
  for (int64_t i = 0; i < {ho} * {wo}; ++i)
    {o}[i * {cout} + ch] += {b}[ch];
}"#,
                    &[
                        ("o", o.into()),
                        ("x", x.into()),
                        ("w", w.into()),
                        ("b", b.into()),
                        ("h", h.to_string()),
                        ("w", ww.to_string()),
                        ("cin", cin.to_string()),
                        ("cout", cout.to_string()),
                        ("kh", kh.to_string()),
                        ("kw", kw.to_string()),
                        ("ho", ho.to_string()),
                        ("wo", wo.to_string()),
                        ("stride", stride.to_string()),
                        ("pad", pad.to_string()),
                        ("co", co.to_string()),
                    ],
                ),
                None,
            ))
        }
        "CONV2D" => {
            let (x, w, b, o) = (
                args[0].as_str(),
                args[1].as_str(),
                args[2].as_str(),
                outs[0].as_str(),
            );
            let kind = ctx.s(x)?.kind;
            if kind != CKind::F {
                return Err(Error::unimplemented("C CONV2D needs F"));
            }
            let xsh = ctx.s(x)?.shape.clone();
            let wsh = ctx.s(w)?.shape.clone();
            if xsh.len() != 3 || wsh.len() != 4 {
                return Err(Error::unimplemented(
                    "C CONV2D needs HWC + (Cout,kh,kw,Cin)",
                ));
            }
            let (h, ww, cin) = (xsh[0].unwrap(), xsh[1].unwrap(), xsh[2].unwrap());
            let (cout, kh, kw) = (wsh[0].unwrap(), wsh[1].unwrap(), wsh[2].unwrap());
            let stride = int_lit(&args[3], "CONV2D stride")?;
            let pad = int_lit(&args[4], "CONV2D pad")?;
            let ho = (h as i64 + 2 * pad - kh as i64) / stride + 1;
            let wo = (ww as i64 + 2 * pad - kw as i64) / stride + 1;
            if ho <= 0 || wo <= 0 || stride <= 0 || pad < 0 {
                return Err(Error::asm("C CONV2D bad geometry"));
            }
            Ok((
                fill(
                    r#"/* {o} = CONV2D({x}) {cout}ch {kh}x{kw} s{stride} p{pad} */
for (int64_t co = 0; co < {cout}; ++co)
  for (int64_t oy = 0; oy < {ho}; ++oy)
    for (int64_t ox = 0; ox < {wo}; ++ox) {
      double acc = {b}[co];
      for (int64_t ky = 0; ky < {kh}; ++ky) {
        int64_t iy = oy * {stride} + ky - {pad};
        if (iy < 0 || iy >= {h}) continue;
        for (int64_t kx = 0; kx < {kw}; ++kx) {
          int64_t ix = ox * {stride} + kx - {pad};
          if (ix < 0 || ix >= {w}) continue;
          for (int64_t ci = 0; ci < {cin}; ++ci)
            acc += {x}[(iy * {w} + ix) * {cin} + ci]
                 * {w}[((co * {kh} + ky) * {kw} + kx) * {cin} + ci];
        }
      }
      {o}[(oy * {wo} + ox) * {cout} + co] = acc;
    }"#,
                    &[
                        ("o", o.into()),
                        ("x", x.into()),
                        ("w", w.into()),
                        ("b", b.into()),
                        ("h", h.to_string()),
                        ("w", ww.to_string()),
                        ("cin", cin.to_string()),
                        ("cout", cout.to_string()),
                        ("kh", kh.to_string()),
                        ("kw", kw.to_string()),
                        ("ho", ho.to_string()),
                        ("wo", wo.to_string()),
                        ("stride", stride.to_string()),
                        ("pad", pad.to_string()),
                    ],
                ),
                None,
            ))
        }
        "RESIZE2D" => {
            let (x, o) = (args[0].as_str(), outs[0].as_str());
            if ctx.s(x)?.kind != CKind::F {
                return Err(Error::unimplemented("C RESIZE2D needs F"));
            }
            let sh = ctx.s(x)?.shape.clone();
            if sh.len() != 3 {
                return Err(Error::unimplemented("C RESIZE2D needs HWC"));
            }
            let (h, w, c) = (sh[0].unwrap(), sh[1].unwrap(), sh[2].unwrap());
            let ho = int_lit(&args[1], "RESIZE2D ho")? as usize;
            let wo = int_lit(&args[2], "RESIZE2D wo")? as usize;
            let align = int_lit(&args[3], "RESIZE2D align")?;
            if ho == 0 || wo == 0 || (align != 0 && align != 1) {
                return Err(Error::asm("C RESIZE2D bad size/align"));
            }
            Ok((
                fill(
                    r#"/* {o} = RESIZE2D({x}) {h}x{w}->{ho}x{wo} align={align} */
for (int64_t oy = 0; oy < {ho}; ++oy) {
  double sy = {align} ? ({ho} > 1 ? (double)oy * {h1} / {ho1} : 0.0)
                      : ((double)oy + 0.5) * {h} / {ho} - 0.5;
  sy = sy < 0 ? 0 : (sy > {h1} ? {h1} : sy);
  int64_t y0 = (int64_t)floor(sy); double wy = sy - y0;
  int64_t y1 = y0 + 1 >= {h} ? {h} - 1 : y0 + 1;
  for (int64_t ox = 0; ox < {wo}; ++ox) {
    double sx = {align} ? ({wo} > 1 ? (double)ox * {w1} / {wo1} : 0.0)
                        : ((double)ox + 0.5) * {w} / {wo} - 0.5;
    sx = sx < 0 ? 0 : (sx > {w1} ? {w1} : sx);
    int64_t x0 = (int64_t)floor(sx); double wx = sx - x0;
    int64_t x1 = x0 + 1 >= {w} ? {w} - 1 : x0 + 1;
    for (int64_t cc = 0; cc < {c}; ++cc)
      {o}[(oy * {wo} + ox) * {c} + cc] =
        (1 - wy) * (1 - wx) * {x}[(y0 * {w} + x0) * {c} + cc]
        + (1 - wy) * wx * {x}[(y0 * {w} + x1) * {c} + cc]
        + wy * (1 - wx) * {x}[(y1 * {w} + x0) * {c} + cc]
        + wy * wx * {x}[(y1 * {w} + x1) * {c} + cc];
  }
}"#,
                    &[
                        ("o", o.into()),
                        ("x", x.into()),
                        ("h", h.to_string()),
                        ("w", w.to_string()),
                        ("c", c.to_string()),
                        ("h1", (h - 1).to_string()),
                        ("w1", (w - 1).to_string()),
                        ("ho", ho.to_string()),
                        ("wo", wo.to_string()),
                        ("ho1", (ho as f64 - 1.0).to_string()),
                        ("wo1", (wo as f64 - 1.0).to_string()),
                        ("align", align.to_string()),
                    ],
                ),
                None,
            ))
        }
        "CONV" => {
            let (x, k, o) = (args[0].as_str(), args[1].as_str(), outs[0].as_str());
            let xsh = ctx.s(x)?.shape.clone();
            let ksh = ctx.s(k)?.shape.clone();
            let (h, w, kh, kw) = (
                xsh[0].unwrap(),
                xsh[1].unwrap(),
                ksh[0].unwrap(),
                ksh[1].unwrap(),
            );
            let (rh, rw) = (kh / 2, kw / 2);
            Ok((
                fill(
                    r#"/* {o} = CONV({x}) edge-replicate {kh}x{kw} */
for (int64_t oy = 0; oy < {h}; ++oy) {
  for (int64_t ox = 0; ox < {w}; ++ox) {
    double acc = 0;
    for (int64_t ky = 0; ky < {kh}; ++ky) {
      int64_t iy = oy + ky - {rh};
      iy = iy < 0 ? 0 : (iy >= {h} ? {h} - 1 : iy);
      for (int64_t kx = 0; kx < {kw}; ++kx) {
        int64_t ix = ox + kx - {rw};
        ix = ix < 0 ? 0 : (ix >= {w} ? {w} - 1 : ix);
        acc += {x}[iy * {w} + ix] * {k}[ky * {kw} + kx];
      }
    }
    {o}[oy * {w} + ox] = acc;
  }
}"#,
                    &[
                        ("o", o.into()),
                        ("x", x.into()),
                        ("k", k.into()),
                        ("h", h.to_string()),
                        ("w", w.to_string()),
                        ("kh", kh.to_string()),
                        ("kw", kw.to_string()),
                        ("rh", rh.to_string()),
                        ("rw", rw.to_string()),
                    ],
                ),
                None,
            ))
        }
        "INTERP" => {
            let (x, o) = (args[0].as_str(), outs[0].as_str());
            let sy = args[1].parse::<f64>().unwrap_or(0.0);
            let sx = args[2].parse::<f64>().unwrap_or(0.0);
            let sh = ctx.s(x)?.shape.clone();
            let (h, w) = (sh[0].unwrap(), sh[1].unwrap());
            let c = if sh.len() == 3 { sh[2].unwrap() } else { 1 };
            let ho = ((h as f64 * sy).round().max(1.0)) as usize;
            let wo = ((w as f64 * sx).round().max(1.0)) as usize;
            Ok((
                fill(
                    r#"/* {o} = INTERP({x}) bilinear {h}x{w}->{ho}x{wo} */
for (int64_t oy = 0; oy < {ho}; ++oy) {
  double fsy = ((double)oy + 0.5) * {hdiv} - 0.5;
  fsy = fsy < 0 ? 0 : (fsy > {h} - 1 ? {h} - 1 : fsy);
  int64_t y0 = (int64_t)floor(fsy); double wy = fsy - y0;
  int64_t y1 = y0 + 1 >= {h} ? {h} - 1 : y0 + 1;
  for (int64_t ox = 0; ox < {wo}; ++ox) {
    double fsx = ((double)ox + 0.5) * {wdiv} - 0.5;
    fsx = fsx < 0 ? 0 : (fsx > {w} - 1 ? {w} - 1 : fsx);
    int64_t x0 = (int64_t)floor(fsx); double wx = fsx - x0;
    int64_t x1 = x0 + 1 >= {w} ? {w} - 1 : x0 + 1;
    for (int64_t cc = 0; cc < {c}; ++cc)
      {o}[(oy * {wo} + ox) * {c} + cc] =
        (1 - wy) * (1 - wx) * {x}[(y0 * {w} + x0) * {c} + cc]
        + (1 - wy) * wx * {x}[(y0 * {w} + x1) * {c} + cc]
        + wy * (1 - wx) * {x}[(y1 * {w} + x0) * {c} + cc]
        + wy * wx * {x}[(y1 * {w} + x1) * {c} + cc];
  }
}"#,
                    &[
                        ("o", o.into()),
                        ("x", x.into()),
                        ("h", h.to_string()),
                        ("w", w.to_string()),
                        ("ho", ho.to_string()),
                        ("wo", wo.to_string()),
                        ("c", c.to_string()),
                        ("hdiv", format!("{:?}", h as f64 / ho as f64)),
                        ("wdiv", format!("{:?}", w as f64 / wo as f64)),
                    ],
                ),
                None,
            ))
        }
        "MATMUL" | "BATCH_MATMUL" => {
            let (a, b, o) = (args[0].as_str(), args[1].as_str(), outs[0].as_str());
            if ctx.s(a)?.kind != CKind::F || ctx.s(b)?.kind != CKind::F {
                return Err(Error::unimplemented(format!(
                    "C backend {mn} needs F (T matmul lives on nonfpu)"
                )));
            }
            let co = ctx.elems(o)?;
            let ash = ctx.s(a)?.shape.clone();
            let bsh = ctx.s(b)?.shape.clone();
            let zero = format!("memset({o}, 0, {co} * sizeof(double));");
            if ash.len() == 2 {
                let (m, k, n) = (ash[0].unwrap(), ash[1].unwrap(), bsh[1].unwrap());
                Ok((
                    fill(
                        r#"/* {o} = {mn} ikj, restrict */
{zero}
for (int64_t m_i = 0; m_i < {m}; ++m_i)
  for (int64_t m_k = 0; m_k < {k}; ++m_k) {
    double aik = {a}[m_i * {k} + m_k];
    double *restrict crow = &{o}[m_i * {n}];
    const double *restrict brow = &{b}[m_k * {n}];
    for (int64_t m_j = 0; m_j < {n}; ++m_j)
      crow[m_j] += aik * brow[m_j];
  }"#,
                        &[
                            ("o", o.into()),
                            ("mn", mn.into()),
                            ("a", a.into()),
                            ("b", b.into()),
                            ("zero", zero),
                            ("m", m.to_string()),
                            ("k", k.to_string()),
                            ("n", n.to_string()),
                        ],
                    ),
                    None,
                ))
            } else {
                let (batch, m, k, n) = (
                    ash[0].unwrap(),
                    ash[1].unwrap(),
                    ash[2].unwrap(),
                    bsh[2].unwrap(),
                );
                Ok((
                    fill(
                        r#"/* {o} = {mn} batched ikj */
{zero}
for (int64_t m_b = 0; m_b < {batch}; ++m_b)
  for (int64_t m_i = 0; m_i < {m}; ++m_i)
    for (int64_t m_k = 0; m_k < {k}; ++m_k) {
      double aik = {a}[(m_b * {m} + m_i) * {k} + m_k];
      double *restrict crow = &{o}[(m_b * {m} + m_i) * {n}];
      const double *restrict brow = &{b}[(m_b * {k} + m_k) * {n}];
      for (int64_t m_j = 0; m_j < {n}; ++m_j)
        crow[m_j] += aik * brow[m_j];
    }"#,
                        &[
                            ("o", o.into()),
                            ("mn", mn.into()),
                            ("a", a.into()),
                            ("b", b.into()),
                            ("zero", zero),
                            ("batch", batch.to_string()),
                            ("m", m.to_string()),
                            ("k", k.to_string()),
                            ("n", n.to_string()),
                        ],
                    ),
                    None,
                ))
            }
        }
        "BMMV" => {
            let (a, b, o) = (args[0].as_str(), args[1].as_str(), outs[0].as_str());
            let nums: Vec<i64> = (2..13)
                .map(|i| int_lit(&args[i], "BMMV"))
                .collect::<Result<_>>()?;
            let (bb, m, n, k, laa, sa, lab, sb, _lac, _sc, tb) = (
                nums[0], nums[1], nums[2], nums[3], nums[4], nums[5], nums[6], nums[7], nums[8],
                nums[9], nums[10],
            );
            let co = ctx.elems(o)?;
            let zero = format!("memset({o}, 0, {co} * sizeof(double));");
            let inner = if tb != 0 {
                format!(
                    "      const double *restrict bcol = &{b}[m_b * {sb} + m_k];\n      for (int64_t m_j = 0; m_j < {n}; ++m_j)\n        crow[m_j] += aik * bcol[m_j * {lab}];\n"
                )
            } else {
                format!(
                    "      const double *restrict brow = &{b}[m_b * {sb} + m_k * {lab}];\n      for (int64_t m_j = 0; m_j < {n}; ++m_j)\n        crow[m_j] += aik * brow[m_j];\n"
                )
            };
            Ok((
                fill(
                    r#"/* {o} = BMMV strided ikj, restrict */
{zero}
for (int64_t m_b = 0; m_b < {bb}; ++m_b)
  for (int64_t m_i = 0; m_i < {m}; ++m_i)
    for (int64_t m_k = 0; m_k < {k}; ++m_k) {
      double aik = {a}[m_b * {sa} + m_i * {laa} + m_k];
      double *restrict crow = &{o}[(m_b * {m} + m_i) * {n}];
{inner}  }"#,
                    &[
                        ("o", o.into()),
                        ("a", a.into()),
                        ("b", b.into()),
                        ("zero", zero),
                        ("bb", bb.to_string()),
                        ("m", m.to_string()),
                        ("n", n.to_string()),
                        ("k", k.to_string()),
                        ("laa", laa.to_string()),
                        ("sa", sa.to_string()),
                        ("lab", lab.to_string()),
                        ("sb", sb.to_string()),
                        ("inner", inner),
                    ],
                ),
                None,
            ))
        }
        "ATTN" => {
            let (qr, kf, vf, mask, o) = (
                args[0].as_str(),
                args[1].as_str(),
                args[2].as_str(),
                args[3].as_str(),
                outs[0].as_str(),
            );
            let nh = int_lit(&args[4], "ATTN nh")?;
            let nkv = int_lit(&args[5], "ATTN nkv")?;
            let dh = int_lit(&args[6], "ATTN dh")?;
            let s = int_lit(&args[7], "ATTN S")?;
            let per = nh / nkv;
            let neg = ctx.config.get_f64_or("beta", -1e9);
            Ok((
                fill(
                    r#"/* {o} = ATTN({qr}) fused GQA decode */
{ double *sc = (double *)malloc(sizeof(double) * ({per} * {s}));
  for (int64_t g = 0; g < {nkv}; ++g) {
    for (int64_t hh = 0; hh < {per}; ++hh) {
      int64_t head = g * {per} + hh;
      for (int64_t tt = 0; tt < {s}; ++tt) {
        double acc = 0;
        for (int64_t dd = 0; dd < {dh}; ++dd)
          acc += {qr}[head * {dh} + dd] * {kf}[tt * {nkv} * {dh} + g * {dh} + dd];
        sc[hh * {s} + tt] = ({mask}[hh * {s} + tt] != 0) ? acc : {neg};
      }
      double mx = sc[hh * {s}];
      for (int64_t tt = 0; tt < {s}; ++tt) if (sc[hh * {s} + tt] > mx) mx = sc[hh * {s} + tt];
      double se = 0;
      for (int64_t tt = 0; tt < {s}; ++tt) { sc[hh * {s} + tt] = exp(sc[hh * {s} + tt] - mx); se += sc[hh * {s} + tt]; }
      for (int64_t tt = 0; tt < {s}; ++tt) sc[hh * {s} + tt] /= se;
      for (int64_t dd = 0; dd < {dh}; ++dd) {
        double acc = 0;
        for (int64_t tt = 0; tt < {s}; ++tt)
          acc += sc[hh * {s} + tt] * {vf}[tt * {nkv} * {dh} + g * {dh} + dd];
        {o}[head * {dh} + dd] = acc;
      }
    }
  }
  free(sc); }"#,
                    &[
                        ("o", o.into()),
                        ("qr", qr.into()),
                        ("kf", kf.into()),
                        ("vf", vf.into()),
                        ("mask", mask.into()),
                        ("nh", nh.to_string()),
                        ("nkv", nkv.to_string()),
                        ("dh", dh.to_string()),
                        ("s", s.to_string()),
                        ("per", per.to_string()),
                        ("neg", float_lit(neg)),
                    ],
                ),
                None,
            ))
        }
        "RESHAPE2" | "RESHAPE3" => {
            let (x, o) = (args[0].as_str(), outs[0].as_str());
            let co = ctx.elems(o)?;
            let kind = ctx.s(x)?.kind;
            match kind {
                CKind::T => {
                    let comps: &[&str] = if ctx.t2_m.is_some() {
                        &["s", "e", "m", "z"]
                    } else {
                        &["s", "e", "z"]
                    };
                    let mut body = format!("/* {o} = {mn}({x}) triples, exact reshape */\n");
                    for comp in comps {
                        body.push_str(&format!(
                            "memcpy({o}_{comp}, {x}_{comp}, {co} * sizeof({o}_{comp}[0]));\n"
                        ));
                    }
                    Ok((body, None))
                }
                CKind::F => Ok((
                    format!("/* {o} = {mn}({x}) */\nmemcpy({o}, {x}, {co} * sizeof(double));"),
                    None,
                )),
                CKind::I => Ok((
                    format!("/* {o} = {mn}({x}) */\nmemcpy({o}, {x}, {co} * sizeof(int64_t));"),
                    None,
                )),
            }
        }
        "SLICE" => {
            let (x, o) = (args[0].as_str(), outs[0].as_str());
            let ax = int_lit(&args[1], "SLICE ax")? as usize;
            let lo = int_lit(&args[2], "SLICE lo")?;
            let hi = int_lit(&args[3], "SLICE hi")?;
            let in_d = ctx.s(x)?.shape[ax]
                .ok_or_else(|| Error::unimplemented("C backend: SLICE with dynamic stride"))?;
            let kind = ctx.s(x)?.kind;
            let co = ctx.elems(o)?;
            match kind {
                CKind::T => {
                    let comps: &[&str] = if ctx.t2_m.is_some() {
                        &["s", "e", "m", "z"]
                    } else {
                        &["s", "e", "z"]
                    };
                    if ax == 1 {
                        let mut body =
                            format!("/* {o} = SLICE({x},1,{lo},{hi}) triples, exact */\n");
                        for comp in comps {
                            body.push_str(&format!(
                                "for (int64_t s_i = 0; s_i < {o}_N; ++s_i)\n  memcpy(&{o}_{comp}[s_i * {}], &{x}_{comp}[s_i * {in_d} + {lo}], {} * sizeof({o}_{comp}[0]));\n",
                                hi - lo, hi - lo
                            ));
                        }
                        Ok((body, None))
                    } else {
                        let w1 = ctx.s(x)?.shape[1].ok_or_else(|| {
                            Error::unimplemented("C backend: SLICE-T ax0 dynamic width")
                        })?;
                        let mut body =
                            format!("/* {o} = SLICE({x},0,{lo},{hi}) triples, exact */\n");
                        for comp in comps {
                            body.push_str(&format!(
                                "for (int64_t s_i = 0; s_i < {o}_N; ++s_i)\n  memcpy(&{o}_{comp}[s_i * {w1}], &{x}_{comp}[(s_i + {lo}) * {w1}], {w1} * sizeof({o}_{comp}[0]));\n"
                            ));
                        }
                        Ok((body, None))
                    }
                }
                CKind::I => {
                    if ax == 1 {
                        let rows = ctx.rows(o)?;
                        Ok((
                            fill(
                                r#"/* {o} = SLICE({x},1,{lo},{hi}) int64 */
for (int64_t s_i = 0; s_i < {rows}; ++s_i)
  memcpy(&{o}[s_i * {hi1}], &{x}[s_i * {in_d} + {lo}], {hi1} * sizeof(int64_t));"#,
                                &[
                                    ("o", o.into()),
                                    ("x", x.into()),
                                    ("lo", lo.to_string()),
                                    ("hi1", (hi - lo).to_string()),
                                    ("in_d", in_d.to_string()),
                                    ("rows", rows.to_string()),
                                ],
                            ),
                            None,
                        ))
                    } else {
                        let w1 = ctx.s(x)?.shape[1].unwrap();
                        Ok((
                            fill(
                                r#"/* {o} = SLICE({x},0,{lo},{hi}) int64 rows */
memcpy({o}, &{x}[{lo} * {w1}], {co} * sizeof(int64_t));"#,
                                &[
                                    ("o", o.into()),
                                    ("x", x.into()),
                                    ("lo", lo.to_string()),
                                    ("w1", w1.to_string()),
                                    ("co", co.to_string()),
                                ],
                            ),
                            None,
                        ))
                    }
                }
                CKind::F => {
                    if ax == 1 {
                        Ok((
                            fill(
                                r#"/* {o} = SLICE({x},1,{lo},{hi}) */
for (int64_t s_i = 0; s_i < {o}_N; ++s_i)
  memcpy(&{o}[s_i * {hi1}], &{x}[s_i * {in_d} + {lo}], {hi1} * sizeof(double));"#,
                                &[
                                    ("o", o.into()),
                                    ("x", x.into()),
                                    ("lo", lo.to_string()),
                                    ("hi1", (hi - lo).to_string()),
                                    ("in_d", in_d.to_string()),
                                ],
                            ),
                            None,
                        ))
                    } else {
                        let w1 = ctx.width(x, &format!("SLICE {o} stride"))?;
                        Ok((
                            fill(
                                r#"/* {o} = SLICE({x},0,{lo},{hi}) rows */
memcpy({o}, &{x}[{lo} * {w1}], {co} * sizeof(double));"#,
                                &[
                                    ("o", o.into()),
                                    ("x", x.into()),
                                    ("lo", lo.to_string()),
                                    ("w1", w1.to_string()),
                                    ("co", co.to_string()),
                                ],
                            ),
                            None,
                        ))
                    }
                }
            }
        }
        "ROTARY" => {
            let (x, pos, o) = (args[0].as_str(), args[1].as_str(), outs[0].as_str());
            if ctx.s(x)?.kind != CKind::F {
                return Err(Error::unimplemented(
                    "C backend ROTARY needs F (T lives on nonfpu)",
                ));
            }
            let d = ctx.width(x, &format!("ROTARY {o}"))?;
            let nh = d / 2;
            Ok((
                fill(
                    r#"/* {o} = ROTARY({x}) runtime sincos */
for (int64_t t_i = 0; t_i < {o}_N; ++t_i) {
  int64_t pp = {pos}[t_i];
  for (int64_t t_h = 0; t_h < {nh}; ++t_h) {
    double ang = (double)pp * THETA[t_h];
    double c = cos(ang), s = sin(ang);
    double x0 = {x}[(t_i * {d}) + 2 * t_h];
    double x1 = {x}[(t_i * {d}) + 2 * t_h + 1];
    {o}[(t_i * {d}) + 2 * t_h] = x0 * c - x1 * s;
    {o}[(t_i * {d}) + 2 * t_h + 1] = x0 * s + x1 * c;
  }
}"#,
                    &[
                        ("o", o.into()),
                        ("x", x.into()),
                        ("pos", pos.into()),
                        ("d", d.to_string()),
                        ("nh", nh.to_string()),
                    ],
                ),
                None,
            ))
        }
        "ROPE" => {
            let (x, pos, o) = (args[0].as_str(), args[1].as_str(), outs[0].as_str());
            let nh = int_lit(&args[2], "ROPE nh")?;
            let dh = int_lit(&args[3], "ROPE dh")?;
            let rows = ctx.rows(x)?;
            let base = ctx.config.get_f64_or("rope_base", 10000.0);
            Ok((
                fill(
                    r#"/* {o} = ROPE({x}) per-head runtime sincos */
for (int64_t r = 0; r < {rows}; ++r) {
  for (int64_t h = 0; h < {nh}; ++h) {
    for (int64_t p = 0; p < {dh2}; ++p) {
      double ang = (double){pos}[r] * pow({base}, -2.0 * p / {dh});
      double sn = sin(ang), cs = cos(ang);
      int64_t i0 = r * {nh} * {dh} + h * {dh} + 2 * p;
      int64_t i1 = i0 + 1;
      double a = {x}[i0], b = {x}[i1];
      {o}[i0] = a * cs - b * sn;
      {o}[i1] = a * sn + b * cs;
    }
  }
}"#,
                    &[
                        ("o", o.into()),
                        ("x", x.into()),
                        ("pos", pos.into()),
                        ("rows", rows.to_string()),
                        ("nh", nh.to_string()),
                        ("dh", dh.to_string()),
                        ("dh2", (dh / 2).to_string()),
                        ("base", float_lit(base)),
                    ],
                ),
                None,
            ))
        }
        "SWIGLU" => {
            let (x, o) = (args[0].as_str(), outs[0].as_str());
            let inter = int_lit(&args[1], "SWIGLU inter")?;
            let rows = ctx.rows(x)?;
            Ok((
                fill(
                    r#"/* {o} = SWIGLU({x}) silu(gate)*up */
for (int64_t r = 0; r < {rows}; ++r) {
  for (int64_t j = 0; j < {inter}; ++j) {
    double up = {x}[r * {w2} + j];
    double gate = {x}[r * {w2} + {inter} + j];
    double gs = gate / (1.0 + exp(-gate));
    {o}[r * {inter} + j] = gs * up;
  }
}"#,
                    &[
                        ("o", o.into()),
                        ("x", x.into()),
                        ("rows", rows.to_string()),
                        ("inter", inter.to_string()),
                        ("w2", (2 * inter).to_string()),
                    ],
                ),
                None,
            ))
        }
        "TRANSPOSE" => {
            let (x, o) = (args[0].as_str(), outs[0].as_str());
            let kind = ctx.s(x)?.kind;
            let sh = ctx.s(x)?.shape.clone();
            let (r, c) = (sh[0], sh[1]);
            let re = r.map(|v| v.to_string()).unwrap_or_else(|| format!("{x}_N"));
            match kind {
                CKind::T => {
                    let mut body = format!("/* {o} = TRANSPOSE({x}) triples, exact move */\n");
                    let comps: &[&str] = if ctx.t2_m.is_some() {
                        &["s", "e", "m", "z"]
                    } else {
                        &["s", "e", "z"]
                    };
                    for comp in comps {
                        body.push_str(&format!(
                            "for (int64_t t_i = 0; t_i < {re}; ++t_i)\n  for (int64_t t_j = 0; t_j < {}; ++t_j)\n    {o}_{comp}[t_j * {re} + t_i] = {x}_{comp}[t_i * {} + t_j];\n",
                            c.unwrap(), c.unwrap()
                        ));
                    }
                    Ok((body, None))
                }
                CKind::F => {
                    let cw = c.ok_or_else(|| {
                        Error::unimplemented("C backend: TRANSPOSE dynamic width")
                    })?;
                    Ok((
                        fill(
                            r#"/* {o} = TRANSPOSE({x}) */
for (int64_t t_i = 0; t_i < {re}; ++t_i)
  for (int64_t t_j = 0; t_j < {c}; ++t_j)
    {o}[t_j * {re} + t_i] = {x}[t_i * {c} + t_j];"#,
                            &[
                                ("o", o.into()),
                                ("x", x.into()),
                                ("re", re),
                                ("c", cw.to_string()),
                            ],
                        ),
                        None,
                    ))
                }
                CKind::I => Err(Error::unimplemented("C backend TRANSPOSE-I not ported")),
            }
        }
        "TBETA" | "BETA" => {
            let (r, o) = (args[0].as_str(), outs[0].as_str());
            let val = if mn == "TBETA" {
                ctx.config
                    .get_f64_or("beta_b", ctx.config.get_f64_or("beta", 0.5))
            } else {
                ctx.config.get_f64_or("beta", 0.5)
            };
            let co = ctx.elems(o)?;
            Ok((
                fill(
                    r#"/* {o} = {mn} ({val}) over {r} shape */
for (int64_t k_i = 0; k_i < {co}; ++k_i)
  {o}[k_i] = {val};"#,
                    &[
                        ("o", o.into()),
                        ("mn", mn.into()),
                        ("r", r.into()),
                        ("co", co.to_string()),
                        ("val", float_lit(val)),
                    ],
                ),
                None,
            ))
        }
        "MUL" | "ADD" | "SUB" | "DIV" => {
            let (a, b, o) = (args[0].as_str(), args[1].as_str(), outs[0].as_str());
            if ctx.s(a)?.kind != CKind::F || ctx.s(b)?.kind != CKind::F {
                return Err(Error::unimplemented(format!(
                    "C backend {mn} needs F (T arithmetic lives on nonfpu)"
                )));
            }
            let co = ctx.elems(o)?;
            let op = match mn {
                "MUL" => "*",
                "ADD" => "+",
                "SUB" => "-",
                _ => "/",
            };
            Ok((
                fill(
                    r#"/* {o} = {a} {op} {b} */
for (int64_t e_i = 0; e_i < {co}; ++e_i)
  {o}[e_i] = {a}[e_i] {op} {b}[e_i];"#,
                    &[
                        ("o", o.into()),
                        ("a", a.into()),
                        ("b", b.into()),
                        ("op", op.into()),
                        ("co", co.to_string()),
                    ],
                ),
                None,
            ))
        }
        "RELU" => {
            let (x, o) = (args[0].as_str(), outs[0].as_str());
            if ctx.s(x)?.kind != CKind::F {
                return Err(Error::unimplemented("C RELU needs F"));
            }
            let co = ctx.elems(o)?;
            Ok((
                fill(
                    r#"/* {o} = RELU({x}) */
for (int64_t e_i = 0; e_i < {co}; ++e_i)
  {o}[e_i] = {x}[e_i] > 0.0 ? {x}[e_i] : 0.0;"#,
                    &[("o", o.into()), ("x", x.into()), ("co", co.to_string())],
                ),
                None,
            ))
        }
        "SILU" => {
            let (x, o) = (args[0].as_str(), outs[0].as_str());
            let co = ctx.elems(o)?;
            Ok((
                fill(
                    r#"/* {o} = SILU({x}) */
for (int64_t e_i = 0; e_i < {co}; ++e_i)
  {o}[e_i] = {x}[e_i] / (1.0 + exp(-{x}[e_i]));"#,
                    &[("o", o.into()), ("x", x.into()), ("co", co.to_string())],
                ),
                None,
            ))
        }
        "SELECT" => {
            let (m, a, b, o) = (
                args[0].as_str(),
                args[1].as_str(),
                args[2].as_str(),
                outs[0].as_str(),
            );
            let co = ctx.elems(o)?;
            let guard = format!("if ({m}_N != {co}) return 21;\n  ");
            match ctx.s(a)?.kind {
                CKind::T => Ok((
                    fill(
                        r#"/* {o} = SELECT({m}) triples, exact */
  {guard}for (int64_t v_i = 0; v_i < {co}; ++v_i) {
  {o}_s[v_i] = {m}[v_i] ? {a}_s[v_i] : {b}_s[v_i];
  {o}_e[v_i] = {m}[v_i] ? {a}_e[v_i] : {b}_e[v_i];
  {o}_z[v_i] = {m}[v_i] ? {a}_z[v_i] : {b}_z[v_i];
}"#,
                        &[
                            ("o", o.into()),
                            ("m", m.into()),
                            ("a", a.into()),
                            ("b", b.into()),
                            ("co", co.to_string()),
                            ("guard", guard),
                        ],
                    ),
                    None,
                )),
                CKind::F => Ok((
                    fill(
                        r#"/* {o} = SELECT({m}) nonzero picks A */
  {guard}for (int64_t v_i = 0; v_i < {co}; ++v_i)
  {o}[v_i] = {m}[v_i] ? {a}[v_i] : {b}[v_i];"#,
                        &[
                            ("o", o.into()),
                            ("m", m.into()),
                            ("a", a.into()),
                            ("b", b.into()),
                            ("co", co.to_string()),
                            ("guard", guard),
                        ],
                    ),
                    None,
                )),
                CKind::I => Err(Error::unimplemented("C backend SELECT-I not ported")),
            }
        }
        "TSHIFT" | "SOFTMAX_WIDE" | "SOFTMAX" => {
            let (x, o) = (args[0].as_str(), outs[0].as_str());
            if ctx.s(x)?.kind != CKind::F {
                return Err(Error::unimplemented(format!(
                    "C backend {mn} needs F (T lives on nonfpu)"
                )));
            }
            let d = ctx.width(x, &format!("{mn} {o}"))?;
            if mn == "TSHIFT" {
                Ok((
                    fill(
                        r#"/* {o} = TSHIFT({x}) row-max shift */
for (int64_t f_i = 0; f_i < {o}_N; ++f_i) {
  double mx = {x}[f_i * {d}];
  for (int64_t f_j = 1; f_j < {d}; ++f_j)
    if ({x}[f_i * {d} + f_j] > mx) mx = {x}[f_i * {d} + f_j];
  for (int64_t f_j = 0; f_j < {d}; ++f_j)
    {o}[f_i * {d} + f_j] = {x}[f_i * {d} + f_j] - mx;
}"#,
                        &[("o", o.into()), ("x", x.into()), ("d", d.to_string())],
                    ),
                    None,
                ))
            } else {
                Ok((
                    fill(
                        r#"/* {o} = {mn}({x}) stable rows */
for (int64_t w_i = 0; w_i < {o}_N; ++w_i) {
  double mx = {x}[w_i * {d}];
  for (int64_t w_j = 1; w_j < {d}; ++w_j)
    if ({x}[w_i * {d} + w_j] > mx) mx = {x}[w_i * {d} + w_j];
  double se = 0;
  for (int64_t w_j = 0; w_j < {d}; ++w_j) {
    {o}[w_i * {d} + w_j] = exp({x}[w_i * {d} + w_j] - mx);
    se += {o}[w_i * {d} + w_j];
  }
  for (int64_t w_j = 0; w_j < {d}; ++w_j)
    {o}[w_i * {d} + w_j] /= se;
}"#,
                        &[
                            ("o", o.into()),
                            ("mn", mn.into()),
                            ("x", x.into()),
                            ("d", d.to_string()),
                        ],
                    ),
                    None,
                ))
            }
        }
        "CONCAT" => {
            let (a, b, o) = (args[0].as_str(), args[1].as_str(), outs[0].as_str());
            let da = ctx.s(a)?.shape[1]
                .ok_or_else(|| Error::unimplemented("C backend CONCAT needs baked widths"))?;
            let db = ctx.s(b)?.shape[1]
                .ok_or_else(|| Error::unimplemented("C backend CONCAT needs baked widths"))?;
            match ctx.s(a)?.kind {
                CKind::T => {
                    let mut body = format!("/* {o} = CONCAT({a},{b},1) triples, exact */\n");
                    let comps: &[&str] = if ctx.t2_m.is_some() {
                        &["s", "e", "m", "z"]
                    } else {
                        &["s", "e", "z"]
                    };
                    for comp in comps {
                        body.push_str(&format!(
                            "for (int64_t n_i = 0; n_i < {o}_N; ++n_i) {{\n  memcpy(&{o}_{comp}[n_i * {}], &{a}_{comp}[n_i * {da}], {da} * sizeof({o}_{comp}[0]));\n  memcpy(&{o}_{comp}[n_i * {} + {da}], &{b}_{comp}[n_i * {db}], {db} * sizeof({o}_{comp}[0]));\n}}\n",
                            da + db, da + db
                        ));
                    }
                    Ok((body, None))
                }
                CKind::F => Ok((
                    fill(
                        r#"/* {o} = CONCAT({a},{b},1) */
for (int64_t n_i = 0; n_i < {o}_N; ++n_i) {
  memcpy(&{o}[n_i * {dab}], &{a}[n_i * {da}], {da} * sizeof(double));
  memcpy(&{o}[n_i * {dab} + {da}], &{b}[n_i * {db}], {db} * sizeof(double));
}"#,
                        &[
                            ("o", o.into()),
                            ("a", a.into()),
                            ("b", b.into()),
                            ("da", da.to_string()),
                            ("db", db.to_string()),
                            ("dab", (da + db).to_string()),
                        ],
                    ),
                    None,
                )),
                CKind::I => Err(Error::unimplemented("C backend CONCAT-I not ported")),
            }
        }
        "GT" | "SQUARE" => Err(Error::unimplemented(format!(
            "c: no pattern for {mn} (matching the reference backend)"
        ))),
        other => Err(Error::unimplemented(format!("c: no pattern for {other}"))),
    }
}
