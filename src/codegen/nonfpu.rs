//! Integer-only (non-FPU) C backend.
//!
//! Exact moves and integer arithmetic reuse the C patterns; T-kind
//! fixed-point arithmetic gets the bridge implementation below. F storage is
//! raw binary64 bits (`uint64_t`) and only F2I/I2F cross that boundary.
//! The emitted source must pass `float_trap` (no float/double, no libm).

use regex::Regex;

use crate::error::{Error, Result};
use crate::ir::Config;
use crate::lattice;
use crate::value::Triple;

use super::c::{Ctx, Pat};
use super::infer::CKind;

const INTEGER_MN: &[&str] = &[
    "GATHER",
    "GATHER3",
    "IGATHER1",
    "IASSEMBLE",
    "ISUM",
    "ISCATTER1",
    "ARGMAX",
    "TRANSPOSE",
    "SLICE",
    "CONCAT",
    "SELECT",
    "GT",
    "IADD",
    "ISUB",
    "IMUL",
    "IMAD",
    "IDIV",
    "IMOD",
    "IDIVU",
    "IMODU",
    "ISAR",
    "SADD",
    "IMAX",
    "IWINDOW",
    "IXOR",
    "IAND",
    "IOR",
    "IEQ",
    "ILT",
    "ISELECT",
    "ISHL",
    "ISHR",
    "ICRC32",
    "ICRC32C",
    "IFIND",
    "IINFLATE",
    "IVARINT",
    "F2I",
    "I2F",
    "MUL",
    "ADD",
    "SUB",
    "SQUARE",
    "MATMUL",
    "BATCH_MATMUL",
    "RMSNORM",
    "TSHIFT",
    "SOFTMAX_WIDE",
    "SOFTMAX",
    "TBETA",
    "BETA",
    "ROTARY",
    "SILU",
];

const FX_MN: &[&str] = &[
    "MUL",
    "ADD",
    "SUB",
    "SQUARE",
    "MATMUL",
    "BATCH_MATMUL",
    "RMSNORM",
    "TSHIFT",
    "SOFTMAX_WIDE",
    "SOFTMAX",
    "TBETA",
    "BETA",
    "ROTARY",
    "ARGMAX",
    "SILU",
    "GT",
];

const SM_MN: &[&str] = &["TSHIFT", "SOFTMAX_WIDE", "SOFTMAX"];

pub fn allowed(mn: &str) -> bool {
    INTEGER_MN.contains(&mn)
}

fn is_fx(mn: &str) -> bool {
    FX_MN.contains(&mn)
}

fn fill(s: &str, vars: &[(&str, String)]) -> String {
    let mut out = s.to_string();
    for (k, v) in vars {
        out = out.replace(&format!("{{{k}}}"), v);
    }
    out
}

fn bake_vec(name: &str, arr: &[i64], per_line: usize) -> String {
    let flat: Vec<String> = arr.iter().map(|v| v.to_string()).collect();
    let lines: Vec<String> = flat.chunks(per_line).map(|c| c.join(", ")).collect();
    format!(
        "static const int64_t {name}[{}] = {{\n {}\n}};",
        flat.len(),
        lines.join(",\n ")
    )
}

fn eps_counts(config: &Config, m_cov: i64) -> i64 {
    if let Some(v) = config.get_f64("eps_rms") {
        let um = lattice::PHI.powf((m_cov - lattice::BIAS) as f64 / lattice::K as f64);
        lattice::round_half_even(v * ((1i64 << 36) as f64) / (um * um)) as i64
    } else {
        config.get_i64_or("eps_rms_c", 4514)
    }
}

const NF_F2I_HELPER: &str = r#"
/* F2I: binary64 bits -> int64, truncating and range-checked. */
static int nf_f2i_bits(uint64_t b, int64_t *out) {
  uint64_t neg = b >> 63;
  uint64_t ex = (b >> 52) & 0x7ffULL;
  uint64_t frac = b & 0xfffffffffffffULL;
  if (ex == 0x7ffULL) return 0;
  if (ex == 0) { *out = 0; return 1; }
  int e = (int)ex - 1023;
  if (e < 0) { *out = 0; return 1; }
  uint64_t m = (1ULL << 52) | frac;
  if (e > 63) return 0;
  if (e == 63) {
    if (neg && frac == 0) {
      *out = (-9223372036854775807LL - 1LL);
      return 1;
    }
    return 0;
  }
  int sh = 52 - e;
  uint64_t q = (sh >= 0) ? (m >> sh) : (m << (-sh));
  *out = neg ? -(int64_t)q : (int64_t)q;
  return 1;
}
"#;

const NF_I2F_HELPER: &str = r#"
/* I2F: int64 -> binary64 bits, round-to-nearest-even. */
static uint64_t nf_i2f_bits(int64_t x) {
  uint64_t sign = 0;
  uint64_t u;
  if (x < 0) { sign = 1ULL << 63; u = 0ULL - (uint64_t)x; }
  else { u = (uint64_t)x; }
  if (u == 0) return 0;
  int k = 63;
  while (!(u & (1ULL << k))) --k;
  uint64_t q;
  int e;
  if (k <= 52) { q = u << (52 - k); e = k + 1023; }
  else {
    int r = k - 52;
    q = u >> r;
    uint64_t rem = u & ((1ULL << r) - 1ULL);
    uint64_t half = 1ULL << (r - 1);
    if (rem > half || (rem == half && (q & 1ULL))) {
      q += 1ULL;
      if (q == (1ULL << 53)) { q >>= 1; ++k; }
    }
    e = k + 1023;
  }
  return sign | ((uint64_t)e << 52) | (q & 0xfffffffffffffULL);
}
"#;

const NF_BRIDGE_HELPERS: &str = r#"
static int64_t nf_to_fixed(int8_t s, int32_t e, uint8_t z, int m) {
  if (z) return 0;
  int64_t d = (int64_t)m - (int64_t)e;
  if (d < 0) return (int64_t)s * NF_BASE;
  if (d > 13312) return 0;
  return (int64_t)s * NF_FRAC[d];
}
static void nf_from_fixed(int64_t q, int m, int8_t *s, int32_t *e, uint8_t *z) {
  *z = (q == 0);
  *s = (q > 0) ? 1 : -1;
  uint64_t a = (q >= 0) ? (uint64_t)q : (uint64_t)(-(q + 1)) + 1u;
  int bl = 0; uint64_t x = a;
  if (x >> 32) { bl += 32; x >>= 32; }
  if (x >> 16) { bl += 16; x >>= 16; }
  if (x >> 8) { bl += 8; x >>= 8; }
  if (x >> 4) { bl += 4; x >>= 4; }
  if (x >> 2) { bl += 2; x >>= 2; }
  if (x >> 1) { bl += 1; }
  bl += (a != 0);
  int shift = bl - 15;
  uint64_t mant = (shift >= 0) ? ((shift > 0) ? (a >> shift) : a)
                               : (a << (-shift));
  int t = shift + 14 - 18;
  int idx = t + 64;
  if (idx < 0) idx = 0;
  if (idx > 192) idx = 192;
  int64_t fi = (int64_t)mant - 16384;
  if (fi < 0) fi = 0;
  if (fi > 16383) fi = 16383;
  int64_t ev = (int64_t)m + NF_COARSE[idx] + NF_FINE[fi];
  if (ev < 0) ev = 0;
  if (ev > 65535) ev = 65535;
  *e = (int32_t)ev;
}
static void nf_binop(int8_t sa, int32_t ea, uint8_t za,
                     int8_t sb, int32_t eb, uint8_t zb, int is_sub,
                     int8_t *os, int32_t *oe, uint8_t *oz) {
  int64_t qa = nf_to_fixed(sa, ea, za, NF_M);
  int64_t qb = nf_to_fixed(sb, eb, zb, NF_M);
  int64_t q = is_sub ? (qa - qb) : (qa + qb);
  nf_from_fixed(q, NF_M, os, oe, oz);
}
static void nf_tmul(int8_t sa, int32_t ea, uint8_t za,
                    int8_t sb, int32_t eb, uint8_t zb,
                    int8_t *os, int32_t *oe, uint8_t *oz) {
  *os = (int8_t)((int)sa * (int)sb);
  int64_t pe = (int64_t)ea + (int64_t)eb - 32768;
  if (pe < 0) pe = 0;
  if (pe > 65535) pe = 65535;
  *oe = (int32_t)pe;
  *oz = (uint8_t)(za | zb);
}
"#;

const NF_SOFTMAX_HELPERS: &str = r#"
static uint64_t nf_isqrt(uint64_t a) {
  if (a < 2) return a;
  uint64_t x = a;
  uint64_t y = (x + 1) >> 1;
  while (y < x) { x = y; y = (x + a / x) >> 1; }
  return x;
}
static int64_t nf_to_fixed_wide(int8_t s, int32_t e, uint8_t z, int m) {
  if (z) return 0;
  int64_t d = (int64_t)m - (int64_t)e;
  if (d < 0) {
    if (d < -8192) return (int64_t)s * NF_BASE;
    return (int64_t)s * NF_FRACHI[(int)(-d) - 1];
  }
  if (d > 13312) return 0;
  return (int64_t)s * NF_FRAC[d];
}
static void nf_softmax_tail(const int64_t *num, int64_t den, int64_t n,
                            int8_t *os, int32_t *oe, uint8_t *oz) {
  for (int64_t j = 0; j < n; ++j) {
    int64_t dd = (den == 0) ? 1 : den;
    int64_t c = (num[j] * (int64_t)(1 << 18)) / dd;
    nf_from_fixed(c, 32768, &os[j], &oe[j], &oz[j]);
  }
}
static void nf_softmax_row(const int64_t *q, int64_t n,
                           int8_t *os, int32_t *oe, uint8_t *oz,
                           int64_t *num) {
  int64_t s0 = (q[0] >= 0) ? (q[0] / 16) : -(((-q[0]) + 15) / 16);
  int64_t vmax = s0;
  for (int64_t j = 1; j < n; ++j) {
    int64_t s14 = (q[j] >= 0) ? (q[j] / 16) : -(((-q[j]) + 15) / 16);
    if (s14 > vmax) vmax = s14;
  }
  int64_t den = 0;
  for (int64_t j = 0; j < n; ++j) {
    int64_t s14 = (q[j] >= 0) ? (q[j] / 16) : -(((-q[j]) + 15) / 16);
    int64_t dd = vmax - s14;
    if (dd < 0) dd = 0;
    if (dd > 262144) dd = 262144;
    num[j] = NF_EXP[dd];
    den += num[j];
  }
  if (den == 0) den = 1;
  nf_softmax_tail(num, den, n, os, oe, oz);
}
static void nf_rmsnorm_row(const int8_t *xs, const int32_t *xe,
                           const uint8_t *xz, const int8_t *ws,
                           const int32_t *we, const uint8_t *wz,
                           int64_t C, int m,
                           int8_t *os, int32_t *oe, uint8_t *oz,
                           int64_t *q, int64_t *wq) {
  for (int64_t j = 0; j < C; ++j) {
    q[j] = nf_to_fixed(xs[j], xe[j], xz[j], m);
    wq[j] = nf_to_fixed(ws[j], we[j], wz[j], m);
  }
  __int128 ss = 0;
  for (int64_t j = 0; j < C; ++j) ss += (__int128)q[j] * q[j];
  int64_t ms = (ss >= 0) ? (int64_t)(ss / C) : -(int64_t)((-ss) / C);
  ms += NF_EPSC;
  uint64_t rms = nf_isqrt((ms >= 0) ? (uint64_t)ms : 0);
  for (int64_t j = 0; j < C; ++j) {
    int64_t norm;
    if (rms == 0) norm = 0;
    else {
      __int128 scaled = (__int128)q[j] * (int64_t)(1 << 18);
      norm = (scaled >= 0) ? (int64_t)(scaled / (int64_t)rms)
                           : -(int64_t)((-scaled) / (int64_t)rms);
    }
    __int128 aff = (__int128)norm * wq[j];
    int64_t av = (aff >= 0) ? (int64_t)(aff / (int64_t)(1 << 18))
                            : -(int64_t)((-aff) / (int64_t)(1 << 18));
    nf_from_fixed(av, m, &os[j], &oe[j], &oz[j]);
  }
}
"#;

const NF_SILU_HELPERS: &str = r#"
static void nf_sigmoid(int8_t s, int32_t e, uint8_t z,
                       int8_t *os, int32_t *oe, uint8_t *oz) {
  int64_t ee = (int64_t)e;
  if (ee < 0) ee = 0;
  if (ee > 65535) ee = 65535;
  int64_t x14 = z ? 0 : (int64_t)s * NF_SIGX[ee];
  int64_t y14;
  if (x14 < -262144) y14 = 0;
  else if (x14 > 262144) y14 = 16384;
  else y14 = NF_SIG[x14 + 262144];
  nf_from_fixed(y14 * 16, 32768, os, oe, oz);
}
static void nf_silu(int8_t s, int32_t e, uint8_t z,
                    int8_t *os, int32_t *oe, uint8_t *oz) {
  int8_t gs; int32_t ge; uint8_t gz;
  nf_sigmoid(s, e, z, &gs, &ge, &gz);
  nf_tmul(gs, ge, gz, s, e, z, os, oe, oz);
}
"#;

pub fn prologue(ctx: &Ctx) -> Result<String> {
    let mut l = String::from(
        "#include <stdint.h>\n#include <stdio.h>\n#include <stdlib.h>\n#include <string.h>\n\n",
    );
    let mns = &ctx.mns;
    let has = |x: &str| mns.iter().any(|m| m == x);
    if has("F2I") {
        l.push_str(NF_F2I_HELPER);
    }
    if has("I2F") {
        l.push_str(NF_I2F_HELPER);
    }
    if has("IINFLATE") {
        l.push_str(super::INFLATE_C);
    }
    if let Some(rope) = &ctx.rope {
        l.push_str(&format!("static const int NF_RP = {};\n", rope.p));
        let bins = |name: &str, t: &Triple| -> Result<String> {
            Ok(format!(
                "{}\n{}\n{}\n",
                bake_vec(&format!("{name}S"), &flat_i8(&t.s), 16),
                bake_vec(&format!("{name}E"), &flat_i32(&t.e), 16),
                bake_vec(&format!("{name}Z"), &flat_u8(&t.z), 16)
            ))
        };
        l.push_str(&bins("NF_RC", &rope.cos)?);
        l.push_str(&bins("NF_RS", &rope.sin)?);
    }
    if mns.iter().any(|m| is_fx(m)) {
        let m_cov = ctx.config.get_i64_or("m_cov", 33280);
        let m_acc = ctx.config.get_i64_or("m_acc", 31344);
        l.push_str(&bake_vec("NF_FRAC", lattice::l_frac(), 16));
        l.push('\n');
        l.push_str(&bake_vec("NF_COARSE", lattice::l_coarse(), 16));
        l.push('\n');
        l.push_str(&bake_vec("NF_FINE", lattice::l_fine(), 16));
        l.push('\n');
        l.push_str(&format!("static const int NF_M = {m_cov};\n"));
        l.push_str(&format!("static const int NF_MACC = {m_acc};\n"));
        l.push_str(&format!(
            "static const int64_t NF_BASE = {};\n",
            lattice::l_frac()[0]
        ));
        l.push_str(NF_BRIDGE_HELPERS);
        if has("SQUARE") {
            let qlo_t = lattice::encode(&ndarray::ArrayD::from_elem(ndarray::IxDyn(&[1]), 0.0f64));
            let qhi_t = lattice::encode(&ndarray::ArrayD::from_elem(ndarray::IxDyn(&[1]), 1.0f64));
            let qlo = lattice::to_fixed(&qlo_t, m_cov)[0];
            let qhi = lattice::to_fixed(&qhi_t, m_cov)[0];
            l.push_str(&format!("static const int64_t NF_QLO = {qlo}LL;\n"));
            l.push_str(&format!("static const int64_t NF_QHI = {qhi}LL;\n"));
        }
        if mns
            .iter()
            .any(|m| SM_MN.contains(&m.as_str()) || m == "RMSNORM")
        {
            l.push_str(&bake_vec("NF_EXP", lattice::exp_lut(), 8));
            l.push('\n');
            l.push_str(&bake_vec("NF_FRACHI", lattice::frac_hi_lut(), 8));
            l.push('\n');
            l.push_str(&format!(
                "static const int64_t NF_EPSC = {}LL;\n",
                eps_counts(&ctx.config, m_cov)
            ));
            l.push_str(NF_SOFTMAX_HELPERS);
        }
        if has("SILU") {
            l.push_str(&bake_vec("NF_SIGX", lattice::sigx_lut(), 8));
            l.push('\n');
            l.push_str(&bake_vec("NF_SIG", lattice::sig_lut(), 8));
            l.push('\n');
            l.push_str(NF_SILU_HELPERS);
        }
    }
    l.push('\n');
    Ok(l)
}

fn flat_i8(a: &ndarray::ArrayD<i8>) -> Vec<i64> {
    a.iter().map(|v| *v as i64).collect()
}
fn flat_i32(a: &ndarray::ArrayD<i32>) -> Vec<i64> {
    a.iter().map(|v| *v as i64).collect()
}
fn flat_u8(a: &ndarray::ArrayD<u8>) -> Vec<i64> {
    a.iter().map(|v| *v as i64).collect()
}

fn tplane(ctx: &Ctx, stream: &str, comp: &str, idx: &str) -> String {
    format!("{stream}_{comp}[{idx}]")
}

/// Non-FPU override. `None` means "fall through to the shared C pattern".
pub fn pattern(mn: &str, outs: &[String], args: &[String], ctx: &mut Ctx) -> Option<Result<Pat>> {
    if !allowed(mn) {
        return Some(Err(Error::unimplemented(format!(
            "nonfpu: no integer pattern for {mn}"
        ))));
    }
    if mn == "F2I" || mn == "I2F" {
        return Some(ieee_pattern(mn, outs, args, ctx));
    }
    // No float streams on this target.
    for a in args {
        if let Some(s) = ctx.streams.get(a) {
            if s.kind == CKind::F {
                return Some(Err(Error::unimplemented(format!(
                    "nonfpu: float stream '{a}' refused (no FPU on this target)"
                ))));
            }
        }
    }
    if is_fx(mn) {
        return Some(fx_pattern(mn, outs, args, ctx));
    }
    None
}

fn ieee_pattern(mn: &str, outs: &[String], args: &[String], ctx: &Ctx) -> Result<Pat> {
    let x = &args[0];
    let o = &outs[0];
    let co = ctx.elems(o)?;
    if mn == "F2I" {
        Ok((
            fill(
                r#"/* {o} = F2I({x}) trunc toward zero, range-checked */
for (int64_t e_i = 0; e_i < {co}; ++e_i) {
  int64_t _v;
  if (!nf_f2i_bits({x}[e_i], &_v)) return 30;
  {o}[e_i] = _v;
}"#,
                &[("o", o.clone()), ("x", x.clone()), ("co", co.to_string())],
            ),
            None,
        ))
    } else {
        Ok((
            fill(
                r#"/* {o} = I2F({x}) round-to-nearest-even */
for (int64_t e_i = 0; e_i < {co}; ++e_i)
  {o}[e_i] = nf_i2f_bits({x}[e_i]);"#,
                &[("o", o.clone()), ("x", x.clone()), ("co", co.to_string())],
            ),
            None,
        ))
    }
}

fn fx_pattern(mn: &str, outs: &[String], args: &[String], ctx: &mut Ctx) -> Result<Pat> {
    let o = outs[0].as_str();
    if mn == "ARGMAX" {
        let src = args[0].as_str();
        if ctx.gathered.contains_key(src) {
            return super::c::pattern_c("ARGMAX", outs, args, ctx);
        }
        if ctx.s(src)?.kind != CKind::T || ctx.s(src)?.shape.len() != 2 {
            return Err(Error::unimplemented("nonfpu ARGMAX needs 2D T"));
        }
        let w = ctx.width(src, "nonfpu ARGMAX")?;
        return Ok((
            fill(
                r#"/* {o} = ARGMAX({src}) lattice class/exponent order, ties first */
for (int64_t a_i = 0; a_i < {o}_N; ++a_i) {
  int64_t best = 0; int bcls = -1; int64_t bkey = 0;
  for (int64_t a_j = 0; a_j < {w}; ++a_j) {
    int8_t s = {src}_s[a_i * {w} + a_j];
    int32_t e = {src}_e[a_i * {w} + a_j];
    uint8_t z = {src}_z[a_i * {w} + a_j];
    int cls = (!z && s > 0) ? 2 : (z ? 1 : 0);
    int64_t key = (!z && s > 0) ? (int64_t)e : (z ? 0 : -(int64_t)e);
    if (bcls < 0 || cls > bcls || (cls == bcls && key > bkey)) {
      best = a_j; bcls = cls; bkey = key;
    }
  }
  {o}[a_i] = best;
}"#,
                &[("o", o.into()), ("src", src.into()), ("w", w.to_string())],
            ),
            None,
        ));
    }
    if mn == "TBETA" || mn == "BETA" {
        let r = args[0].as_str();
        if ctx.s(r)?.kind != CKind::T {
            return Err(Error::unimplemented(format!("nonfpu {mn} needs T ref")));
        }
        let val = if mn == "TBETA" {
            ctx.config
                .get_f64_or("beta_b", ctx.config.get_f64_or("beta", 0.5))
        } else {
            ctx.config.get_f64_or("beta", 0.5)
        };
        let t = lattice::encode(&ndarray::ArrayD::from_elem(ndarray::IxDyn(&[1]), val));
        let (s, e, z) = (
            t.s.iter().next().copied().unwrap_or(0),
            t.e.iter().next().copied().unwrap_or(0),
            t.z.iter().next().copied().unwrap_or(0),
        );
        let co = ctx.elems(o)?;
        return Ok((
            fill(
                r#"/* {o} = {mn} ({val}) baked triples */
for (int64_t k_i = 0; k_i < {co}; ++k_i) {
  {o}_s[k_i] = (int8_t){s};
  {o}_e[k_i] = (int32_t){e};
  {o}_z[k_i] = (uint8_t){z};
}"#,
                &[
                    ("o", o.into()),
                    ("mn", mn.into()),
                    ("val", format!("{val:?}")),
                    ("co", co.to_string()),
                    ("s", s.to_string()),
                    ("e", e.to_string()),
                    ("z", z.to_string()),
                ],
            ),
            None,
        ));
    }
    if mn == "ROTARY" {
        let (x, pos) = (args[0].as_str(), args[1].as_str());
        if ctx.s(x)?.kind != CKind::T || ctx.s(pos)?.kind != CKind::I {
            return Err(Error::unimplemented("nonfpu ROTARY needs T + I pos"));
        }
        let d = ctx.width(x, "nonfpu ROTARY")?;
        let rope = ctx
            .rope
            .as_ref()
            .ok_or_else(|| Error::unimplemented("nonfpu ROTARY needs baked tables"))?;
        if rope.d != d {
            return Err(Error::unimplemented(
                "nonfpu ROTARY tables do not match head dim",
            ));
        }
        return Ok((
            fill(
                r#"/* {o} = ROTARY({x}) tmul/binop pairs, exact */
for (int64_t t_i = 0; t_i < {o}_N; ++t_i) {
  int64_t pp = {pos}[t_i];
  if (pp < 0 || pp >= NF_RP) return 31;
  for (int64_t t_h = 0; t_h < {half}; ++t_h) {
    int8_t a0s = {x}_s[(t_i * {d}) + 2 * t_h];
    int32_t a0e = {x}_e[(t_i * {d}) + 2 * t_h];
    uint8_t a0z = {x}_z[(t_i * {d}) + 2 * t_h];
    int8_t a1s = {x}_s[(t_i * {d}) + 2 * t_h + 1];
    int32_t a1e = {x}_e[(t_i * {d}) + 2 * t_h + 1];
    uint8_t a1z = {x}_z[(t_i * {d}) + 2 * t_h + 1];
    int64_t co = pp * {half} + t_h;
    int8_t y0s, y1s; int32_t y0e, y1e; uint8_t y0z, y1z;
    int8_t t0s, t1s; int32_t t0e, t1e; uint8_t t0z, t1z;
    nf_tmul(a0s, a0e, a0z, NF_RCS[co], NF_RCE[co], NF_RCZ[co], &t0s, &t0e, &t0z);
    nf_tmul(a1s, a1e, a1z, NF_RSS[co], NF_RSE[co], NF_RSZ[co], &t1s, &t1e, &t1z);
    nf_binop(t0s, t0e, t0z, t1s, t1e, t1z, 1, &y0s, &y0e, &y0z);
    nf_tmul(a0s, a0e, a0z, NF_RSS[co], NF_RSE[co], NF_RSZ[co], &t0s, &t0e, &t0z);
    nf_tmul(a1s, a1e, a1z, NF_RCS[co], NF_RCE[co], NF_RCZ[co], &t1s, &t1e, &t1z);
    nf_binop(t0s, t0e, t0z, t1s, t1e, t1z, 0, &y1s, &y1e, &y1z);
    {o}_s[(t_i * {d}) + 2 * t_h] = y0s;
    {o}_e[(t_i * {d}) + 2 * t_h] = y0e;
    {o}_z[(t_i * {d}) + 2 * t_h] = y0z;
    {o}_s[(t_i * {d}) + 2 * t_h + 1] = y1s;
    {o}_e[(t_i * {d}) + 2 * t_h + 1] = y1e;
    {o}_z[(t_i * {d}) + 2 * t_h + 1] = y1z;
  }
}"#,
                &[
                    ("o", o.into()),
                    ("x", x.into()),
                    ("pos", pos.into()),
                    ("d", d.to_string()),
                    ("half", (d / 2).to_string()),
                ],
            ),
            None,
        ));
    }
    let co = ctx.elems(o)?;
    match mn {
        "GT" => {
            let (a, b) = (args[0].as_str(), args[1].as_str());
            Ok((
                fill(
                    r#"/* {o} = GT({a},{b}) lattice class/exponent order */
for (int64_t f_i = 0; f_i < {co}; ++f_i) {
  int8_t _sa = {as}; int32_t _ea = {ae}; uint8_t _za = {az};
  int8_t _sb = {bs}; int32_t _eb = {be}; uint8_t _zb = {bz};
  int _ca = (!_za && _sa > 0) ? 2 : (_za ? 1 : 0);
  int64_t _ka = (!_za && _sa > 0) ? (int64_t)_ea : (_za ? 0 : -(int64_t)_ea);
  int _cb = (!_zb && _sb > 0) ? 2 : (_zb ? 1 : 0);
  int64_t _kb = (!_zb && _sb > 0) ? (int64_t)_eb : (_zb ? 0 : -(int64_t)_eb);
  {o}[f_i] = (_ca > _cb) || ((_ca == _cb) && (_ka > _kb));
}"#,
                    &[
                        ("o", o.into()),
                        ("a", a.into()),
                        ("b", b.into()),
                        ("as", tplane(ctx, a, "s", "f_i")),
                        ("ae", tplane(ctx, a, "e", "f_i")),
                        ("az", tplane(ctx, a, "z", "f_i")),
                        ("bs", tplane(ctx, b, "s", "f_i")),
                        ("be", tplane(ctx, b, "e", "f_i")),
                        ("bz", tplane(ctx, b, "z", "f_i")),
                        ("co", co.to_string()),
                    ],
                ),
                None,
            ))
        }
        "MUL" => {
            let (a, b) = (args[0].as_str(), args[1].as_str());
            Ok((
                fill(
                    r#"/* {o} = MUL({a},{b}) tmul exact */
for (int64_t f_i = 0; f_i < {co}; ++f_i)
  nf_tmul({as}, {ae}, {az}, {bs}, {be}, {bz}, &{o}_s[f_i], &{o}_e[f_i], &{o}_z[f_i]);"#,
                    &[
                        ("o", o.into()),
                        ("a", a.into()),
                        ("b", b.into()),
                        ("as", tplane(ctx, a, "s", "f_i")),
                        ("ae", tplane(ctx, a, "e", "f_i")),
                        ("az", tplane(ctx, a, "z", "f_i")),
                        ("bs", tplane(ctx, b, "s", "f_i")),
                        ("be", tplane(ctx, b, "e", "f_i")),
                        ("bz", tplane(ctx, b, "z", "f_i")),
                        ("co", co.to_string()),
                    ],
                ),
                None,
            ))
        }
        "ADD" | "SUB" => {
            let (a, b) = (args[0].as_str(), args[1].as_str());
            let sub = if mn == "SUB" { 1 } else { 0 };
            Ok((
                fill(
                    r#"/* {o} = {mn}({a},{b}) bridge @ NF_M */
for (int64_t f_i = 0; f_i < {co}; ++f_i)
  nf_binop({as}, {ae}, {az}, {bs}, {be}, {bz}, {sub}, &{o}_s[f_i], &{o}_e[f_i], &{o}_z[f_i]);"#,
                    &[
                        ("o", o.into()),
                        ("mn", mn.into()),
                        ("a", a.into()),
                        ("b", b.into()),
                        ("as", tplane(ctx, a, "s", "f_i")),
                        ("ae", tplane(ctx, a, "e", "f_i")),
                        ("az", tplane(ctx, a, "z", "f_i")),
                        ("bs", tplane(ctx, b, "s", "f_i")),
                        ("be", tplane(ctx, b, "e", "f_i")),
                        ("bz", tplane(ctx, b, "z", "f_i")),
                        ("sub", sub.to_string()),
                        ("co", co.to_string()),
                    ],
                ),
                None,
            ))
        }
        "SQUARE" => {
            let a = args[0].as_str();
            Ok((
                fill(
                    r#"/* {o} = SQUARE({a}) tmul-self + clip [NF_QLO,NF_QHI] */
for (int64_t f_i = 0; f_i < {co}; ++f_i) {
  int8_t _s; int32_t _e; uint8_t _z;
  nf_tmul({as}, {ae}, {az}, {as}, {ae}, {az}, &_s, &_e, &_z);
  int64_t _q = nf_to_fixed(_s, _e, _z, NF_M);
  if (_q < NF_QLO) _q = NF_QLO;
  if (_q > NF_QHI) _q = NF_QHI;
  nf_from_fixed(_q, NF_M, &{o}_s[f_i], &{o}_e[f_i], &{o}_z[f_i]);
}"#,
                    &[
                        ("o", o.into()),
                        ("a", a.into()),
                        ("as", tplane(ctx, a, "s", "f_i")),
                        ("ae", tplane(ctx, a, "e", "f_i")),
                        ("az", tplane(ctx, a, "z", "f_i")),
                        ("co", co.to_string()),
                    ],
                ),
                None,
            ))
        }
        "MATMUL" | "BATCH_MATMUL" => {
            let (a, b) = (args[0].as_str(), args[1].as_str());
            let ash = ctx.s(a)?.shape.clone();
            let bsh = ctx.s(b)?.shape.clone();
            if ash.len() == 2 {
                let (m, k, n) = (ash[0].unwrap(), ash[1].unwrap(), bsh[1].unwrap());
                Ok((
                    fill(
                        r#"/* {o} = {mn}({a},{b}) tmul+bridge @ NF_MACC */
for (int64_t w_i = 0; w_i < {m}; ++w_i)
  for (int64_t w_j = 0; w_j < {n}; ++w_j) {
    int64_t acc = 0;
    for (int64_t w_k = 0; w_k < {k}; ++w_k) {
      int8_t _s; int32_t _e; uint8_t _z;
      nf_tmul({a}_s[w_i * {k} + w_k], {a}_e[w_i * {k} + w_k], {a}_z[w_i * {k} + w_k],
              {b}_s[w_k * {n} + w_j], {b}_e[w_k * {n} + w_j], {b}_z[w_k * {n} + w_j],
              &_s, &_e, &_z);
      acc += nf_to_fixed(_s, _e, _z, NF_MACC);
    }
    nf_from_fixed(acc, NF_MACC, &{o}_s[w_i * {n} + w_j], &{o}_e[w_i * {n} + w_j], &{o}_z[w_i * {n} + w_j]);
  }"#,
                        &[
                            ("o", o.into()),
                            ("mn", mn.into()),
                            ("a", a.into()),
                            ("b", b.into()),
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
                let bb = bsh[0].unwrap();
                let bo = if bb == 1 {
                    "0".to_string()
                } else {
                    "w_b".to_string()
                };
                Ok((
                    fill(
                        r#"/* {o} = {mn}({a},{b}) batched */
for (int64_t w_b = 0; w_b < {batch}; ++w_b)
  for (int64_t w_i = 0; w_i < {m}; ++w_i)
    for (int64_t w_j = 0; w_j < {n}; ++w_j) {
      int64_t acc = 0;
      int64_t _bo = {bo};
      for (int64_t w_k = 0; w_k < {k}; ++w_k) {
        int8_t _s; int32_t _e; uint8_t _z;
        nf_tmul({a}_s[(w_b * {m} + w_i) * {k} + w_k],
                {a}_e[(w_b * {m} + w_i) * {k} + w_k],
                {a}_z[(w_b * {m} + w_i) * {k} + w_k],
                {b}_s[(_bo * {k} + w_k) * {n} + w_j],
                {b}_e[(_bo * {k} + w_k) * {n} + w_j],
                {b}_z[(_bo * {k} + w_k) * {n} + w_j],
                &_s, &_e, &_z);
        acc += nf_to_fixed(_s, _e, _z, NF_MACC);
      }
      nf_from_fixed(acc, NF_MACC,
        &{o}_s[(w_b * {m} + w_i) * {n} + w_j],
        &{o}_e[(w_b * {m} + w_i) * {n} + w_j],
        &{o}_z[(w_b * {m} + w_i) * {n} + w_j]);
    }"#,
                        &[
                            ("o", o.into()),
                            ("mn", mn.into()),
                            ("a", a.into()),
                            ("b", b.into()),
                            ("batch", batch.to_string()),
                            ("m", m.to_string()),
                            ("k", k.to_string()),
                            ("n", n.to_string()),
                            ("bo", bo),
                        ],
                    ),
                    None,
                ))
            }
        }
        "RMSNORM" => {
            let (x, w) = (args[0].as_str(), args[1].as_str());
            let d = ctx.width(x, "nonfpu RMSNORM")?;
            Ok((
                fill(
                    r#"/* {o} = RMSNORM({x},{w}) integer rows @ NF_M */
{ int64_t *_q = malloc({d} * 8);
  int64_t *_wq = malloc({d} * 8);
  for (int64_t r_i = 0; r_i < {o}_N; ++r_i)
    nf_rmsnorm_row(&{x}_s[r_i * {d}], &{x}_e[r_i * {d}], &{x}_z[r_i * {d}],
      {w}_s, {w}_e, {w}_z, {d}, NF_M,
      &{o}_s[r_i * {d}], &{o}_e[r_i * {d}], &{o}_z[r_i * {d}], _q, _wq);
  free(_q); free(_wq); }"#,
                    &[
                        ("o", o.into()),
                        ("x", x.into()),
                        ("w", w.into()),
                        ("d", d.to_string()),
                    ],
                ),
                None,
            ))
        }
        "TSHIFT" => {
            let x = args[0].as_str();
            let d = ctx.width(x, "nonfpu TSHIFT")?;
            Ok((
                fill(
                    r#"/* {o} = TSHIFT({x}) wide bridge @ NF_MACC */
{ int64_t *_q = malloc({d} * 8);
  for (int64_t f_i = 0; f_i < {o}_N; ++f_i) {
    for (int64_t f_j = 0; f_j < {d}; ++f_j)
      _q[f_j] = nf_to_fixed_wide({x}_s[f_i * {d} + f_j], {x}_e[f_i * {d} + f_j], {x}_z[f_i * {d} + f_j], NF_MACC);
    int64_t _mx = _q[0];
    for (int64_t f_j = 1; f_j < {d}; ++f_j) if (_q[f_j] > _mx) _mx = _q[f_j];
    for (int64_t f_j = 0; f_j < {d}; ++f_j)
      nf_from_fixed(_q[f_j] - _mx, NF_MACC, &{o}_s[f_i * {d} + f_j], &{o}_e[f_i * {d} + f_j], &{o}_z[f_i * {d} + f_j]);
  }
  free(_q); }"#,
                    &[("o", o.into()), ("x", x.into()), ("d", d.to_string())],
                ),
                None,
            ))
        }
        "SOFTMAX_WIDE" | "SOFTMAX" => {
            let x = args[0].as_str();
            let d = ctx.width(x, "nonfpu SOFTMAX")?;
            let bridge = if mn == "SOFTMAX_WIDE" {
                format!("nf_to_fixed_wide({x}_s[f_i * {d} + f_j], {x}_e[f_i * {d} + f_j], {x}_z[f_i * {d} + f_j], 32768)")
            } else {
                format!("nf_to_fixed({x}_s[f_i * {d} + f_j], {x}_e[f_i * {d} + f_j], {x}_z[f_i * {d} + f_j], 32768)")
            };
            Ok((
                fill(
                    r#"/* {o} = {mn}({x}) row softmax @ BIAS */
{ int64_t *_q = malloc({d} * 8);
  int64_t *_num = malloc({d} * 8);
  for (int64_t f_i = 0; f_i < {o}_N; ++f_i) {
    for (int64_t f_j = 0; f_j < {d}; ++f_j) _q[f_j] = {bridge};
    nf_softmax_row(_q, {d}, &{o}_s[f_i * {d}], &{o}_e[f_i * {d}], &{o}_z[f_i * {d}], _num);
  }
  free(_q); free(_num); }"#,
                    &[
                        ("o", o.into()),
                        ("mn", mn.into()),
                        ("x", x.into()),
                        ("d", d.to_string()),
                        ("bridge", bridge),
                    ],
                ),
                None,
            ))
        }
        "SILU" => {
            let a = args[0].as_str();
            Ok((
                fill(
                    r#"/* {o} = SILU({a}) sigmoid-LUT + tmul, exact */
for (int64_t f_i = 0; f_i < {co}; ++f_i)
  nf_silu({a}_s[f_i], {a}_e[f_i], {a}_z[f_i], &{o}_s[f_i], &{o}_e[f_i], &{o}_z[f_i]);"#,
                    &[("o", o.into()), ("a", a.into()), ("co", co.to_string())],
                ),
                None,
            ))
        }
        other => Err(Error::unimplemented(format!(
            "nonfpu: no fx pattern for {other}"
        ))),
    }
}

/// The non-FPU float trap: comments stripped, then no FPU types/symbols.
pub fn float_trap(source: &str) -> Result<()> {
    let block = Regex::new(r"/\*.*?\*/").expect("regex");
    let line = Regex::new(r"//.*").expect("regex");
    let stripped = block.replace_all(source, "");
    let code = line.replace_all(&stripped, "");
    let checks = [
        Regex::new(r"\b(float|double)\b").expect("regex"),
        Regex::new(r"\b(sqrt|exp|sin|cos|tan|pow|fmax|fmin|fabs|round)\s*\(").expect("regex"),
        Regex::new(r"#include\s*<math\.h>").expect("regex"),
    ];
    for rx in checks {
        if let Some(m) = rx.find(&code) {
            return Err(Error::asm(format!(
                "nonfpu float-trap: {:?} -- FPU use in integer-only emission",
                m.as_str()
            )));
        }
    }
    Ok(())
}
