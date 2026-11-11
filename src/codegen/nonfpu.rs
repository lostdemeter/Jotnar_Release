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
    "IMATMUL",
    "IREDUCE_SUM",
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
    "T2I",
    "IGET",
    "MUL",
    "ADD",
    "SUB",
    "SQUARE",
    "MATMUL",
    "BATCH_MATMUL",
    "CONV2D",
    "DECONV2D",
    "RESIZE2D",
    "RELU",
    "RMSNORM",
    "TSHIFT",
    "SOFTMAX_WIDE",
    "SOFTMAX",
    "LAYERNORM",
    "GELU",
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
    "CONV2D",
    "DECONV2D",
    "RESIZE2D",
    "RELU",
    "RMSNORM",
    "TSHIFT",
    "SOFTMAX_WIDE",
    "SOFTMAX",
    "LAYERNORM",
    "GELU",
    "TBETA",
    "BETA",
    "ROTARY",
    "ARGMAX",
    "SILU",
    "GT",
    "T2I",
];

const SM_MN: &[&str] = &["TSHIFT", "SOFTMAX_WIDE", "SOFTMAX"];

pub fn allowed(mn: &str) -> bool {
    INTEGER_MN.contains(&mn)
}

fn is_fx(mn: &str) -> bool {
    FX_MN.contains(&mn)
}

fn int_lit(a: &str, what: &str) -> Result<i64> {
    a.parse::<i64>()
        .map_err(|_| Error::asm(format!("{what}: need integer literal, got {a}")))
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

fn eps_counts_key(config: &Config, m_cov: i64, fkey: &str, ckey: &str, default: i64) -> i64 {
    if let Some(v) = config.get_f64(fkey) {
        let um = lattice::PHI.powf((m_cov - lattice::BIAS) as f64 / lattice::K as f64);
        lattice::round_half_even(v * ((1i64 << 36) as f64) / (um * um)) as i64
    } else {
        config.get_i64_or(ckey, default)
    }
}

fn eps_counts(config: &Config, m_cov: i64) -> i64 {
    eps_counts_key(config, m_cov, "eps_rms", "eps_rms_c", 4514)
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
  if (d > NF_FRAC_CAP) return 0;
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
  int t = shift + 14 - NF_FIXED_F;
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
  if (d > NF_FRAC_CAP) return 0;
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
  nf_from_fixed(y14 * (int64_t)(1 << (NF_FIXED_F - 14)), 32768, os, oe, oz);
}
static void nf_silu(int8_t s, int32_t e, uint8_t z,
                    int8_t *os, int32_t *oe, uint8_t *oz) {
  int8_t gs; int32_t ge; uint8_t gz;
  nf_sigmoid(s, e, z, &gs, &ge, &gz);
  nf_tmul(gs, ge, gz, s, e, z, os, oe, oz);
}
"#;

const NF_LAYERNORM_HELPERS: &str = r#"
static void nf_layernorm_row(const int8_t *xs, const int32_t *xe, const uint8_t *xz,
                             const int8_t *ws, const int32_t *we, const uint8_t *wz,
                             const int8_t *bs, const int32_t *be, const uint8_t *bz,
                             int64_t C, int m, int64_t epsc,
                             int8_t *os, int32_t *oe, uint8_t *oz,
                             int64_t *q, int64_t *wq, int64_t *bq) {
  for (int64_t j = 0; j < C; ++j) {
    q[j] = nf_to_fixed(xs[j], xe[j], xz[j], m);
    wq[j] = nf_to_fixed(ws[j], we[j], wz[j], m);
    bq[j] = nf_to_fixed(bs[j], be[j], bz[j], m);
  }
  __int128 sum = 0;
  for (int64_t j = 0; j < C; ++j) sum += q[j];
  int64_t mean = (int64_t)(sum / C);
  __int128 ss = 0;
  for (int64_t j = 0; j < C; ++j) {
    int64_t d = q[j] - mean;
    ss += (__int128)d * d;
  }
  int64_t var = (int64_t)(ss / C) + epsc;
  uint64_t is = nf_isqrt((var >= 0) ? (uint64_t)var : 0);
  for (int64_t j = 0; j < C; ++j) {
    int64_t d = q[j] - mean;
    int64_t norm;
    if (is == 0) norm = 0;
    else {
      __int128 sc = (__int128)d * (int64_t)(1 << NF_FIXED_F);
      norm = (sc >= 0) ? (int64_t)(sc / (int64_t)is)
                       : -(int64_t)((-sc) / (int64_t)is);
    }
    __int128 aw = (__int128)norm * wq[j];
    int64_t aq = (aw >= 0) ? (int64_t)(aw / (int64_t)(1 << NF_FIXED_F))
                           : -(int64_t)((-aw) / (int64_t)(1 << NF_FIXED_F));
    int64_t av = aq + bq[j];
    nf_from_fixed(av, m, &os[j], &oe[j], &oz[j]);
  }
}
"#;

const NF_GELU_HELPERS: &str = r#"
static void nf_gelu(int8_t s, int32_t e, uint8_t z,
                    int8_t *os, int32_t *oe, uint8_t *oz) {
  int64_t ee = (int64_t)e;
  if (ee < 0) ee = 0;
  if (ee > 65535) ee = 65535;
  int64_t x14 = z ? 0 : (int64_t)s * NF_GELUX[ee];
  int64_t p14;
  if (x14 < -NF_GSPAN) p14 = 0;
  else if (x14 > NF_GSPAN) p14 = 16384;
  else p14 = NF_GELUP[x14 + NF_GSPAN];
  int64_t prod = x14 * p14;
  int64_t y14 = (prod >= 0) ? (prod / 16384) : -((-prod) / 16384);
  nf_from_fixed(y14 * (int64_t)(1 << (NF_FIXED_F - 14)), 32768, os, oe, oz);
}
"#;

const NF_T2_HELPERS: &str = r#"
static int64_t nf_t2_base(int d) {
  if (d < 0) { int k = -d; if (k > 8192) return (int64_t)(1LL << NF_FIXED_F); return NF_FRACHI[k - 1]; }
  if (d > NF_T2_CAP) return 0;
  return NF_FRAC[d];
}
static int64_t nf_t2_tdiv(int64_t a, int64_t b) {
  if (b == 0) return 0;
  if ((a < 0) ^ (b < 0)) return -((a < 0 ? -a : a) / (b < 0 ? -b : b));
  return (a < 0 ? -a : a) / (b < 0 ? -b : b);
}
static int64_t nf_t2_round_div(int64_t a, int64_t b) {
  if (b == 0) return 0;
  if (a < 0) return -(((-a) + b / 2) / b);
  return (a + b / 2) / b;
}
static uint64_t nf_t2_isqrt(uint64_t a) {
  if (a < 2) return a;
  uint64_t x = a, y = (x + 1) >> 1;
  while (y < x) { x = y; y = (x + a / x) >> 1; }
  return x;
}
static int64_t nf_t2_to_fixed(int8_t s, int32_t e, int32_t m, uint8_t z, int m0) {
  if (z) return 0;
  int64_t base = nf_t2_base(m0 - (int64_t)e);
  __int128 q = ((__int128)base * ((__int128)NF_T2_SCALE + (int64_t)m) + (NF_T2_SCALE >> 1)) >> NF_T2_M;
  return (int64_t)((s < 0) ? -q : q);
}
static void nf_t2_from_fixed(int64_t q, int m0, int8_t *s, int32_t *e, int32_t *m, uint8_t *z) {
  if (q == 0) { *z = 1; *s = 1; *e = 32768; *m = 0; return; }
  *z = 0; *s = (q > 0) ? 1 : -1;
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
  uint64_t mant = (shift >= 0) ? ((shift > 0) ? (a >> shift) : a) : (a << (-shift));
  int t = shift + 14 - NF_FIXED_F;
  int idx = t + 64; if (idx < 0) idx = 0; if (idx > 192) idx = 192;
  int64_t fi = (int64_t)mant - 16384; if (fi < 0) fi = 0; if (fi > 16383) fi = 16383;
  int64_t ev = (int64_t)m0 + NF_COARSE[idx] + NF_FINE[fi];
  if (ev < 0) ev = 0; if (ev > 65535) ev = 65535;
  *e = (int32_t)ev;
  int64_t base = nf_t2_base((int)((int64_t)m0 - ev));
  if (base == 0) { *m = 0; return; }
  __int128 num = ((__int128)a) << NF_T2_M;
  __int128 ratio = (num + base / 2) / base;
  __int128 mm = ratio - NF_T2_SCALE;
  if (mm < -NF_T2_HALF) mm = -NF_T2_HALF;
  if (mm > NF_T2_HALF - 1) mm = NF_T2_HALF - 1;
  *m = (int32_t)mm;
}
static void nf_t2_mul(int8_t sa, int32_t ea, int32_t ma, uint8_t za,
                      int8_t sb, int32_t eb, int32_t mb, uint8_t zb,
                      int8_t *os, int32_t *oe, int32_t *om, uint8_t *oz) {
  if (za || zb) { *os = 1; *oe = 32768; *om = 0; *oz = 1; return; }
  int64_t cross = nf_t2_round_div((int64_t)ma * (int64_t)mb, NF_T2_SCALE);
  int64_t m = (int64_t)ma + (int64_t)mb + cross;
  int64_t e = (int64_t)ea + (int64_t)eb - 32768;
  if (e < 0) e = 0; if (e > 65535) e = 65535;
  *os = (int8_t)((int)sa * (int)sb); *oe = (int32_t)e; *om = (int32_t)m; *oz = 0;
}
static void nf_t2_add(int8_t sa, int32_t ea, int32_t ma, uint8_t za,
                      int8_t sb, int32_t eb, int32_t mb, uint8_t zb, int sub,
                      int8_t *os, int32_t *oe, int32_t *om, uint8_t *oz) {
  if (za && zb) { *os = 1; *oe = 32768; *om = 0; *oz = 1; return; }
  int m0 = za ? (int)eb : (zb ? (int)ea : ((ea > eb) ? (int)ea : (int)eb));
  int64_t qa = nf_t2_to_fixed(sa, ea, ma, za, m0);
  int64_t qb = nf_t2_to_fixed(sb, eb, mb, zb, m0);
  int64_t acc = sub ? (qa - qb) : (qa + qb);
  nf_t2_from_fixed(acc, m0, os, oe, om, oz);
}
static void nf_t2_layernorm_row(const int8_t *xs, const int32_t *xe, const int32_t *xm, const uint8_t *xz,
                                const int8_t *ws, const int32_t *we, const int32_t *wm, const uint8_t *wz,
                                const int8_t *bs, const int32_t *be, const int32_t *bm, const uint8_t *bz,
                                int64_t C, int64_t *q, int64_t *wq, int64_t *bq,
                                int8_t *os, int32_t *oe, int32_t *om, uint8_t *oz) {
  int m0 = 32768; int any = 0;
  for (int64_t j = 0; j < C; ++j) {
    if (!xz[j]) { if (!any || (int)xe[j] > m0) m0 = (int)xe[j]; any = 1; }
  }
  if (!any) m0 = 32768;
  for (int64_t j = 0; j < C; ++j) {
    q[j]  = nf_t2_to_fixed(xs[j], xe[j], xm[j], xz[j], m0);
    wq[j] = nf_t2_to_fixed(ws[j], we[j], wm[j], wz[j], m0);
    bq[j] = nf_t2_to_fixed(bs[j], be[j], bm[j], bz[j], m0);
  }
  __int128 sum = 0;
  for (int64_t j = 0; j < C; ++j) sum += q[j];
  int64_t mean = (int64_t)(sum / C);
  __int128 ss = 0;
  for (int64_t j = 0; j < C; ++j) { int64_t d = q[j] - mean; ss += (__int128)d * d; }
  int64_t var = (int64_t)(ss / C);
  int64_t q1 = nf_t2_to_fixed(1, 32768, 0, 0, m0);
  __int128 ev = ((__int128)NF_T2_EPSQ * q1 * q1 + (1LL << 29)) >> 30;
  var += (int64_t)ev;
  uint64_t is = nf_t2_isqrt((var < 0) ? 0 : (uint64_t)var);
  int64_t S = (int64_t)(1LL << NF_FIXED_F);
  for (int64_t j = 0; j < C; ++j) {
    int64_t d = q[j] - mean;
    int64_t norm = (is == 0) ? 0 : nf_t2_tdiv(d * S, (int64_t)is);
    int64_t aq = nf_t2_tdiv(norm * wq[j], S);
    int64_t av = aq + bq[j];
    nf_t2_from_fixed(av, m0, &os[j], &oe[j], &om[j], &oz[j]);
  }
}
static void nf_t2_gelu(int8_t s, int32_t e, int32_t m, uint8_t z,
                       int8_t *os, int32_t *oe, int32_t *om, uint8_t *oz) {
  int64_t q = nf_t2_to_fixed(s, e, m, z, 32768);
  int64_t x14 = nf_t2_tdiv(q, (int64_t)(1LL << (NF_FIXED_F - 14)));
  int64_t p14;
  if (x14 < -NF_GSPAN) p14 = 0;
  else if (x14 > NF_GSPAN) p14 = 16384;
  else p14 = NF_GELUP[x14 + NF_GSPAN];
  int64_t y14 = nf_t2_tdiv(x14 * p14, 16384);
  nf_t2_from_fixed(y14 * (int64_t)(1LL << (NF_FIXED_F - 14)), 32768, os, oe, om, oz);
}
static void nf_t2_softmax_row(const int8_t *xs, const int32_t *xe, const int32_t *xm, const uint8_t *xz,
                              int64_t n, int64_t *q, int64_t *num,
                              int8_t *os, int32_t *oe, int32_t *om, uint8_t *oz) {
  int64_t sh14 = (int64_t)(1LL << (NF_FIXED_F - 14));
  int64_t vmax = 0;
  for (int64_t j = 0; j < n; ++j) {
    q[j] = nf_t2_to_fixed(xs[j], xe[j], xm[j], xz[j], 32768);
    if (j == 0 || nf_t2_tdiv(q[j], sh14) > vmax) vmax = nf_t2_tdiv(q[j], sh14);
  }
  __int128 den = 0;
  for (int64_t j = 0; j < n; ++j) {
    int64_t s14 = nf_t2_tdiv(q[j], sh14);
    int64_t dd = vmax - s14; if (dd < 0) dd = 0; if (dd > 262144) dd = 262144;
    num[j] = NF_EXP[dd];
    den += num[j];
  }
  if (den == 0) den = 1;
  int64_t S = (int64_t)(1LL << NF_FIXED_F);
  for (int64_t j = 0; j < n; ++j) {
    int64_t c = (int64_t)(((__int128)num[j] * S) / den);
    nf_t2_from_fixed(c, 32768, &os[j], &oe[j], &om[j], &oz[j]);
  }
}

typedef struct { int8_t *s; int32_t *e; int32_t *m; uint8_t *z; } nf_t2_planes;

static __int128 nf_t2_term(int8_t as, int32_t ae, int32_t am, uint8_t az,
                           int8_t bs, int32_t be, int32_t bm, uint8_t bz,
                           int64_t emax) {
  if (az || bz) return 0;
  int64_t ep = (int64_t)ae + (int64_t)be - 32768;
  int64_t base = nf_t2_base((int)(emax - ep));
  if (base == 0) return 0;
  __int128 aa = (__int128)NF_T2_SCALE + (int64_t)am;
  __int128 bb = (__int128)NF_T2_SCALE + (int64_t)bm;
  __int128 p = ((__int128)base * aa * bb + NF_T2_HALF2) >> (2 * NF_T2_M);
  return (__int128)as * (__int128)bs * p;
}

static void nf_t2_relu(int8_t s, int32_t e, int32_t m, uint8_t z,
                       int8_t *os, int32_t *oe, int32_t *om, uint8_t *oz) {
  if (z == 0 && s < 0) { *os = 1; *oe = 32768; *om = 0; *oz = 1; }
  else { *os = s; *oe = e; *om = m; *oz = z; }
}

static void nf_t2_conv2d(nf_t2_planes x, int64_t H, int64_t W, int64_t Cin,
                         nf_t2_planes w, int64_t Cout, int64_t KH, int64_t KW,
                         nf_t2_planes b, int64_t stride, int64_t pad,
                         int64_t Ho, int64_t Wo, nf_t2_planes o) {
  for (int64_t co = 0; co < Cout; ++co)
    for (int64_t oy = 0; oy < Ho; ++oy)
      for (int64_t ox = 0; ox < Wo; ++ox) {
        int64_t emax = 0; int any = 0;
        for (int64_t ky = 0; ky < KH; ++ky) {
          int64_t iy = oy * stride + ky - pad;
          if (iy < 0 || iy >= H) continue;
          for (int64_t kx = 0; kx < KW; ++kx) {
            int64_t ix = ox * stride + kx - pad;
            if (ix < 0 || ix >= W) continue;
            for (int64_t ci = 0; ci < Cin; ++ci) {
              int64_t ai = (iy * W + ix) * Cin + ci;
              int64_t wi = ((co * KH + ky) * KW + kx) * Cin + ci;
              if (x.z[ai] || w.z[wi]) continue;
              int64_t ep = (int64_t)x.e[ai] + (int64_t)w.e[wi] - 32768;
              if (!any || ep > emax) { emax = ep; any = 1; }
            }
          }
        }
        if (b.z[co] == 0) {
          int64_t ep = (int64_t)b.e[co];
          if (!any || ep > emax) { emax = ep; any = 1; }
        }
        int64_t oi = (oy * Wo + ox) * Cout + co;
        if (!any) { o.s[oi] = 1; o.e[oi] = 32768; o.m[oi] = 0; o.z[oi] = 1; continue; }
        __int128 acc = 0;
        for (int64_t ky = 0; ky < KH; ++ky) {
          int64_t iy = oy * stride + ky - pad;
          if (iy < 0 || iy >= H) continue;
          for (int64_t kx = 0; kx < KW; ++kx) {
            int64_t ix = ox * stride + kx - pad;
            if (ix < 0 || ix >= W) continue;
            for (int64_t ci = 0; ci < Cin; ++ci) {
              int64_t ai = (iy * W + ix) * Cin + ci;
              int64_t wi = ((co * KH + ky) * KW + kx) * Cin + ci;
              acc += nf_t2_term(x.s[ai], x.e[ai], x.m[ai], x.z[ai],
                                 w.s[wi], w.e[wi], w.m[wi], w.z[wi], emax);
            }
          }
        }
        if (b.z[co] == 0) acc += nf_t2_term(1, 32768, 0, 0, b.s[co], b.e[co], b.m[co], b.z[co], emax);
        nf_t2_from_fixed((int64_t)acc, (int)emax, &o.s[oi], &o.e[oi], &o.m[oi], &o.z[oi]);
      }
}

static void nf_t2_deconv2d(nf_t2_planes x, int64_t H, int64_t W, int64_t Cin,
                           nf_t2_planes w, int64_t Cout, int64_t KH, int64_t KW,
                           nf_t2_planes b, int64_t stride, int64_t pad,
                           int64_t Ho, int64_t Wo, nf_t2_planes o) {
  for (int64_t co = 0; co < Cout; ++co)
    for (int64_t oy = 0; oy < Ho; ++oy)
      for (int64_t ox = 0; ox < Wo; ++ox) {
        int64_t emax = 0; int any = 0;
        for (int64_t ky = 0; ky < KH; ++ky) {
          int64_t n = oy + pad - ky;
          if (n < 0 || n % stride != 0) continue;
          int64_t iy = n / stride;
          if (iy >= H) continue;
          for (int64_t kx = 0; kx < KW; ++kx) {
            int64_t n2 = ox + pad - kx;
            if (n2 < 0 || n2 % stride != 0) continue;
            int64_t ix = n2 / stride;
            if (ix >= W) continue;
            for (int64_t ci = 0; ci < Cin; ++ci) {
              int64_t ai = (iy * W + ix) * Cin + ci;
              int64_t wi = ((co * KH + ky) * KW + kx) * Cin + ci;
              if (x.z[ai] || w.z[wi]) continue;
              int64_t ep = (int64_t)x.e[ai] + (int64_t)w.e[wi] - 32768;
              if (!any || ep > emax) { emax = ep; any = 1; }
            }
          }
        }
        if (b.z[co] == 0) {
          int64_t ep = (int64_t)b.e[co];
          if (!any || ep > emax) { emax = ep; any = 1; }
        }
        int64_t oi = (oy * Wo + ox) * Cout + co;
        if (!any) { o.s[oi] = 1; o.e[oi] = 32768; o.m[oi] = 0; o.z[oi] = 1; continue; }
        __int128 acc = 0;
        for (int64_t ky = 0; ky < KH; ++ky) {
          int64_t n = oy + pad - ky;
          if (n < 0 || n % stride != 0) continue;
          int64_t iy = n / stride;
          if (iy >= H) continue;
          for (int64_t kx = 0; kx < KW; ++kx) {
            int64_t n2 = ox + pad - kx;
            if (n2 < 0 || n2 % stride != 0) continue;
            int64_t ix = n2 / stride;
            if (ix >= W) continue;
            for (int64_t ci = 0; ci < Cin; ++ci) {
              int64_t ai = (iy * W + ix) * Cin + ci;
              int64_t wi = ((co * KH + ky) * KW + kx) * Cin + ci;
              acc += nf_t2_term(x.s[ai], x.e[ai], x.m[ai], x.z[ai],
                                 w.s[wi], w.e[wi], w.m[wi], w.z[wi], emax);
            }
          }
        }
        if (b.z[co] == 0) acc += nf_t2_term(1, 32768, 0, 0, b.s[co], b.e[co], b.m[co], b.z[co], emax);
        nf_t2_from_fixed((int64_t)acc, (int)emax, &o.s[oi], &o.e[oi], &o.m[oi], &o.z[oi]);
      }
}

static void nf_t2_resize_coord(int64_t o, int64_t n, int64_t no, int align,
                               int64_t *i0, int64_t *i1, int64_t *w0, int64_t *w1) {
  const int64_t Q = 1LL << 24;
  if (n <= 1 || no <= 1) { *i0 = 0; *i1 = 0; *w0 = Q; *w1 = 0; return; }
  int64_t i, rem, den;
  if (align) {
    int64_t num = o * (n - 1), d = no - 1;
    i = num / d; rem = num % d; den = d;
  } else {
    int64_t num = (2 * o + 1) * n - no, d = 2 * no;
    if (num <= 0) { *i0 = 0; *i1 = (1 < n ? 1 : n - 1); *w0 = Q; *w1 = 0; return; }
    if (num >= (n - 1) * d) { *i0 = n - 1; *i1 = n - 1; *w0 = Q; *w1 = 0; return; }
    i = num / d; rem = num % d; den = d;
  }
  *i0 = i; *i1 = (i + 1 < n ? i + 1 : n - 1);
  int64_t frac = (rem * Q + den / 2) / den;
  *w1 = frac; *w0 = Q - frac;
}

static void nf_t2_resize2d(nf_t2_planes x, int64_t H, int64_t W, int64_t C,
                           int64_t Ho, int64_t Wo, int align, nf_t2_planes o) {
  const int64_t Q = 1LL << 24;
  for (int64_t oy = 0; oy < Ho; ++oy) {
    int64_t y0, y1, wy0, wy1;
    nf_t2_resize_coord(oy, H, Ho, align, &y0, &y1, &wy0, &wy1);
    for (int64_t ox = 0; ox < Wo; ++ox) {
      int64_t x0, x1, wx0, wx1;
      nf_t2_resize_coord(ox, W, Wo, align, &x0, &x1, &wx0, &wx1);
      int64_t w00 = (int64_t)(((__int128)wy0 * wx0 + Q / 2) / Q);
      int64_t w01 = (int64_t)(((__int128)wy0 * wx1 + Q / 2) / Q);
      int64_t w10 = (int64_t)(((__int128)wy1 * wx0 + Q / 2) / Q);
      int64_t w11 = Q - w00 - w01 - w10;
      for (int64_t c = 0; c < C; ++c) {
        int64_t i00 = (y0 * W + x0) * C + c;
        int64_t i01 = (y0 * W + x1) * C + c;
        int64_t i10 = (y1 * W + x0) * C + c;
        int64_t i11 = (y1 * W + x1) * C + c;
        int64_t emax = 0; int any = 0;
        int64_t is[4] = {i00, i01, i10, i11};
        for (int t = 0; t < 4; ++t) {
          if (x.z[is[t]] == 0) {
            int64_t ep = (int64_t)x.e[is[t]];
            if (!any || ep > emax) { emax = ep; any = 1; }
          }
        }
        int64_t oi = (oy * Wo + ox) * C + c;
        if (!any) { o.s[oi] = 1; o.e[oi] = 32768; o.m[oi] = 0; o.z[oi] = 1; continue; }
        int64_t q00 = nf_t2_to_fixed(x.s[i00], x.e[i00], x.m[i00], x.z[i00], emax);
        int64_t q01 = nf_t2_to_fixed(x.s[i01], x.e[i01], x.m[i01], x.z[i01], emax);
        int64_t q10 = nf_t2_to_fixed(x.s[i10], x.e[i10], x.m[i10], x.z[i10], emax);
        int64_t q11 = nf_t2_to_fixed(x.s[i11], x.e[i11], x.m[i11], x.z[i11], emax);
        __int128 acc = ((__int128)q00 * w00 + (__int128)q01 * w01
                        + (__int128)q10 * w10 + (__int128)q11 * w11 + Q / 2) / Q;
        nf_t2_from_fixed((int64_t)acc, (int)emax, &o.s[oi], &o.e[oi], &o.m[oi], &o.z[oi]);
      }
    }
  }
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
        let (t2_m, bridge_f) =
            phi_core::t2::check_widths(ctx.t2_m, ctx.config.get_i64("bridge_f"))?;
        let frac = lattice::l_frac_for(bridge_f);
        l.push_str(&bake_vec("NF_FRAC", &frac, 16));
        l.push('\n');
        l.push_str(&bake_vec("NF_COARSE", lattice::l_coarse(), 16));
        l.push('\n');
        l.push_str(&bake_vec("NF_FINE", lattice::l_fine(), 16));
        l.push('\n');
        l.push_str(&format!("static const int NF_M = {m_cov};\n"));
        l.push_str(&format!("static const int NF_MACC = {m_acc};\n"));
        l.push_str(&format!("static const int NF_FIXED_F = {bridge_f};\n"));
        l.push_str(&format!(
            "static const int NF_FRAC_CAP = {};\n",
            frac.len() - 1
        ));
        l.push_str(&format!(
            "static const int64_t NF_BASE = {};\n",
            1i64 << bridge_f
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
            .any(|m| SM_MN.contains(&m.as_str()) || m == "RMSNORM" || m == "LAYERNORM")
            || t2_m.is_some()
        {
            l.push_str(&bake_vec("NF_EXP", lattice::exp_lut(), 8));
            l.push('\n');
            l.push_str(&bake_vec("NF_FRACHI", &lattice::frac_hi_for(bridge_f), 8));
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
        if has("LAYERNORM") {
            l.push_str(&format!(
                "static const int64_t NF_EPSL = {}LL;\n",
                eps_counts_key(&ctx.config, m_cov, "eps_ln", "eps_ln_c", 4514)
            ));
            l.push_str(NF_LAYERNORM_HELPERS);
        }
        if has("GELU") || t2_m.is_some() {
            l.push_str(&bake_vec("NF_GELUX", lattice::sigx_lut(), 8));
            l.push('\n');
            l.push_str(&bake_vec("NF_GELUP", lattice::phi_lut(), 8));
            l.push('\n');
            l.push_str(&format!(
                "static const int64_t NF_GSPAN = {}LL;\n",
                lattice::SIG_SPAN
            ));
            l.push_str(NF_GELU_HELPERS);
        }
        if has("T2I") {
            l.push_str(&bake_vec("NF_T2I", lattice::t2i_lut(), 8));
            l.push('\n');
        }
        if let Some(m) = t2_m {
            let eps_q = lattice::round_half_even(
                ctx.config.get_f64_or("eps_ln", 1e-6) * (1i64 << 30) as f64,
            ) as i64;
            l.push_str(&format!("static const int NF_T2_M = {m};\n"));
            l.push_str(&format!("static const int64_t NF_T2_SCALE = 1LL << {m};\n"));
            l.push_str(&format!(
                "static const int64_t NF_T2_HALF = 1LL << {};\n",
                m - 1
            ));
            l.push_str(&format!(
                "static const int64_t NF_T2_HALF2 = 1LL << {};\n",
                2 * m - 1
            ));
            l.push_str(&format!(
                "static const int NF_T2_CAP = {};\n",
                frac.len() - 1
            ));
            l.push_str(&format!("static const int64_t NF_T2_EPSQ = {eps_q}LL;\n"));
            l.push_str(NF_T2_HELPERS);
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

fn tplane(_ctx: &Ctx, stream: &str, comp: &str, idx: &str) -> String {
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
    if mn == "IGET" {
        let x = args[0].as_str();
        let o = outs[0].as_str();
        let idx: i64 = match args[1].parse() {
            Ok(v) => v,
            Err(_) => return Some(Err(Error::asm("IGET idx must be an integer literal"))),
        };
        let elems = match ctx.elems(x) {
            Ok(v) => v,
            Err(e) => return Some(Err(e)),
        };
        if idx < 0 || idx as usize >= elems {
            return Some(Err(Error::asm(format!(
                "IGET: index {idx} out of range {elems}"
            ))));
        }
        return Some(Ok((
            fill(
                "/* {o} = IGET({x}, {idx}) exact */\n{o}[0] = {x}[{idx}];",
                &[("o", o.into()), ("x", x.into()), ("idx", idx.to_string())],
            ),
            None,
        )));
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

fn t2_pattern(mn: &str, outs: &[String], args: &[String], ctx: &mut Ctx) -> Result<Pat> {
    let o = outs[0].as_str();
    let co = ctx.elems(o)?;
    match mn {
        "MUL" => {
            let (a, b) = (args[0].as_str(), args[1].as_str());
            Ok((
                fill(
                    r#"/* {o} = MUL({a},{b}) T2 mantissa product */
for (int64_t f_i = 0; f_i < {co}; ++f_i)
  nf_t2_mul({as}, {ae}, {am}, {az}, {bs}, {be}, {bm}, {bz},
            &{o}_s[f_i], &{o}_e[f_i], &{o}_m[f_i], &{o}_z[f_i]);"#,
                    &[
                        ("o", o.into()),
                        ("a", a.into()),
                        ("b", b.into()),
                        ("as", tplane(ctx, a, "s", "f_i")),
                        ("ae", tplane(ctx, a, "e", "f_i")),
                        ("am", tplane(ctx, a, "m", "f_i")),
                        ("az", tplane(ctx, a, "z", "f_i")),
                        ("bs", tplane(ctx, b, "s", "f_i")),
                        ("be", tplane(ctx, b, "e", "f_i")),
                        ("bm", tplane(ctx, b, "m", "f_i")),
                        ("bz", tplane(ctx, b, "z", "f_i")),
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
                    r#"/* {o} = SQUARE({a}) T2 mantissa product */
for (int64_t f_i = 0; f_i < {co}; ++f_i)
  nf_t2_mul({as}, {ae}, {am}, {az}, {as}, {ae}, {am}, {az},
            &{o}_s[f_i], &{o}_e[f_i], &{o}_m[f_i], &{o}_z[f_i]);"#,
                    &[
                        ("o", o.into()),
                        ("a", a.into()),
                        ("as", tplane(ctx, a, "s", "f_i")),
                        ("ae", tplane(ctx, a, "e", "f_i")),
                        ("am", tplane(ctx, a, "m", "f_i")),
                        ("az", tplane(ctx, a, "z", "f_i")),
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
                    r#"/* {o} = {mn}({a},{b}) T2 aligned add */
for (int64_t f_i = 0; f_i < {co}; ++f_i)
  nf_t2_add({as}, {ae}, {am}, {az}, {bs}, {be}, {bm}, {bz}, {sub},
            &{o}_s[f_i], &{o}_e[f_i], &{o}_m[f_i], &{o}_z[f_i]);"#,
                    &[
                        ("o", o.into()),
                        ("mn", mn.into()),
                        ("a", a.into()),
                        ("b", b.into()),
                        ("as", tplane(ctx, a, "s", "f_i")),
                        ("ae", tplane(ctx, a, "e", "f_i")),
                        ("am", tplane(ctx, a, "m", "f_i")),
                        ("az", tplane(ctx, a, "z", "f_i")),
                        ("bs", tplane(ctx, b, "s", "f_i")),
                        ("be", tplane(ctx, b, "e", "f_i")),
                        ("bm", tplane(ctx, b, "m", "f_i")),
                        ("bz", tplane(ctx, b, "z", "f_i")),
                        ("sub", sub.to_string()),
                        ("co", co.to_string()),
                    ],
                ),
                None,
            ))
        }
        "MATMUL" => {
            let (a, b) = (args[0].as_str(), args[1].as_str());
            let ash = ctx.s(a)?.shape.clone();
            let bsh = ctx.s(b)?.shape.clone();
            if ash.len() != 2 || bsh.len() != 2 {
                return Err(Error::unimplemented(
                    "nonfpu T2 MATMUL supports 2-D only (no BATCH_MATMUL)",
                ));
            }
            let (m, k, n) = (ash[0].unwrap(), ash[1].unwrap(), bsh[1].unwrap());
            Ok((
                fill(
                    r#"/* {o} = MATMUL({a},{b}) T2 block-floating dot */
for (int64_t w_i = 0; w_i < {m}; ++w_i)
  for (int64_t w_j = 0; w_j < {n}; ++w_j) {
    int64_t emax = 0; int any = 0;
    for (int64_t w_k = 0; w_k < {k}; ++w_k) {
      int64_t ai = w_i * {k} + w_k, bi = w_k * {n} + w_j;
      if ({a}_z[ai] || {b}_z[bi]) continue;
      int64_t ep = (int64_t){a}_e[ai] + (int64_t){b}_e[bi] - 32768;
      if (!any || ep > emax) { emax = ep; any = 1; }
    }
    if (!any) {
      int64_t oi = w_i * {n} + w_j;
      {o}_s[oi] = 1; {o}_e[oi] = 32768; {o}_m[oi] = 0; {o}_z[oi] = 1;
      continue;
    }
    __int128 acc = 0;
    for (int64_t w_k = 0; w_k < {k}; ++w_k) {
      int64_t ai = w_i * {k} + w_k, bi = w_k * {n} + w_j;
      if ({a}_z[ai] || {b}_z[bi]) continue;
      int64_t ep = (int64_t){a}_e[ai] + (int64_t){b}_e[bi] - 32768;
      int64_t base = nf_t2_base((int)(emax - ep));
      if (base == 0) continue;
      __int128 aa = (__int128)NF_T2_SCALE + {a}_m[ai];
      __int128 bb = (__int128)NF_T2_SCALE + {b}_m[bi];
      __int128 p = ((__int128)base * aa * bb + NF_T2_HALF2) >> (2 * NF_T2_M);
      int64_t sg = (int64_t){a}_s[ai] * (int64_t){b}_s[bi];
      acc += sg * p;
    }
    int64_t oi = w_i * {n} + w_j;
    nf_t2_from_fixed((int64_t)acc, (int)emax,
      &{o}_s[oi], &{o}_e[oi], &{o}_m[oi], &{o}_z[oi]);
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
        "LAYERNORM" => {
            let (x, w, b) = (args[0].as_str(), args[1].as_str(), args[2].as_str());
            let d = ctx.width(x, "nonfpu T2 LAYERNORM")?;
            let rows = ctx.elems(x)? / d;
            Ok((
                fill(
                    r#"/* {o} = LAYERNORM({x},{w},{b}) T2 row bridge */
{lbrace} int64_t *_q = malloc({d} * 8);
  int64_t *_wq = malloc({d} * 8);
  int64_t *_bq = malloc({d} * 8);
  for (int64_t r_i = 0; r_i < {rows}; ++r_i)
    nf_t2_layernorm_row(&{x}_s[r_i * {d}], &{x}_e[r_i * {d}], &{x}_m[r_i * {d}], &{x}_z[r_i * {d}],
      {w}_s, {w}_e, {w}_m, {w}_z, {b}_s, {b}_e, {b}_m, {b}_z,
      {d}, _q, _wq, _bq,
      &{o}_s[r_i * {d}], &{o}_e[r_i * {d}], &{o}_m[r_i * {d}], &{o}_z[r_i * {d}]);
  free(_q); free(_wq); free(_bq); {rbrace}"#,
                    &[
                        ("o", o.into()),
                        ("x", x.into()),
                        ("w", w.into()),
                        ("b", b.into()),
                        ("d", d.to_string()),
                        ("rows", rows.to_string()),
                        ("lbrace", "{".into()),
                        ("rbrace", "}".into()),
                    ],
                ),
                None,
            ))
        }
        "GELU" => {
            let a = args[0].as_str();
            Ok((
                fill(
                    r#"/* {o} = GELU({a}) T2 mantissa-aware exact-erf LUT */
for (int64_t f_i = 0; f_i < {co}; ++f_i)
  nf_t2_gelu({as}, {ae}, {am}, {az}, &{o}_s[f_i], &{o}_e[f_i], &{o}_m[f_i], &{o}_z[f_i]);"#,
                    &[
                        ("o", o.into()),
                        ("a", a.into()),
                        ("as", tplane(ctx, a, "s", "f_i")),
                        ("ae", tplane(ctx, a, "e", "f_i")),
                        ("am", tplane(ctx, a, "m", "f_i")),
                        ("az", tplane(ctx, a, "z", "f_i")),
                        ("co", co.to_string()),
                    ],
                ),
                None,
            ))
        }
        "SOFTMAX_WIDE" | "SOFTMAX" => {
            let x = args[0].as_str();
            let d = ctx.width(x, "nonfpu T2 SOFTMAX")?;
            Ok((
                fill(
                    r#"/* {o} = {mn}({x}) T2 row softmax @ BIAS */
{ int64_t *_q = malloc({d} * 8);
  int64_t *_num = malloc({d} * 8);
  for (int64_t f_i = 0; f_i < {o}_N; ++f_i)
    nf_t2_softmax_row(&{x}_s[f_i * {d}], &{x}_e[f_i * {d}], &{x}_m[f_i * {d}], &{x}_z[f_i * {d}],
      {d}, _q, _num, &{o}_s[f_i * {d}], &{o}_e[f_i * {d}], &{o}_m[f_i * {d}], &{o}_z[f_i * {d}]);
  free(_q); free(_num); }"#,
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
        "CONV2D" | "DECONV2D" => {
            if ctx.t2_m.is_none() {
                return Err(Error::unimplemented(format!(
                    "nonfpu {mn} requires T2 (CONFIG t2)"
                )));
            }
            let (x, w, b) = (args[0].as_str(), args[1].as_str(), args[2].as_str());
            let xsh = ctx.s(x)?.shape.clone();
            let wsh = ctx.s(w)?.shape.clone();
            if xsh.len() != 3 || wsh.len() != 4 {
                return Err(Error::unimplemented(format!(
                    "nonfpu {mn} needs HWC + (Cout,kh,kw,Cin)"
                )));
            }
            let (h, ww, cin) = (xsh[0].unwrap(), xsh[1].unwrap(), xsh[2].unwrap());
            let (cout, kh, kw) = (wsh[0].unwrap(), wsh[1].unwrap(), wsh[2].unwrap());
            let stride = int_lit(&args[3], "stride")?;
            let pad = int_lit(&args[4], "pad")?;
            let (ho, wo) = if mn == "CONV2D" {
                (
                    (h as i64 + 2 * pad - kh as i64) / stride + 1,
                    (ww as i64 + 2 * pad - kw as i64) / stride + 1,
                )
            } else {
                (
                    (h as i64 - 1) * stride - 2 * pad + kh as i64,
                    (ww as i64 - 1) * stride - 2 * pad + kw as i64,
                )
            };
            if ho <= 0 || wo <= 0 {
                return Err(Error::asm(format!("nonfpu {mn} bad output size")));
            }
            let fn_name = if mn == "CONV2D" {
                "nf_t2_conv2d"
            } else {
                "nf_t2_deconv2d"
            };
            Ok((
                fill(
                    r#"/* {o} = {mn}({x}) T2 block-floating */
{ nf_t2_planes _x = { {x}_s, {x}_e, {x}_m, {x}_z };
  nf_t2_planes _w = { {w}_s, {w}_e, {w}_m, {w}_z };
  nf_t2_planes _b = { {b}_s, {b}_e, {b}_m, {b}_z };
  nf_t2_planes _o = { {o}_s, {o}_e, {o}_m, {o}_z };
  {fn_name}(_x, {h}, {ww}, {cin}, _w, {cout}, {kh}, {kw}, _b,
             {stride}, {pad}, {ho}, {wo}, _o); {rbrace}"#,
                    &[
                        ("o", o.into()),
                        ("mn", mn.into()),
                        ("x", x.into()),
                        ("w", w.into()),
                        ("b", b.into()),
                        ("h", h.to_string()),
                        ("ww", ww.to_string()),
                        ("cin", cin.to_string()),
                        ("cout", cout.to_string()),
                        ("kh", kh.to_string()),
                        ("kw", kw.to_string()),
                        ("ho", ho.to_string()),
                        ("wo", wo.to_string()),
                        ("stride", stride.to_string()),
                        ("pad", pad.to_string()),
                        ("fn_name", fn_name.into()),
                        ("rbrace", "}".into()),
                    ],
                ),
                None,
            ))
        }
        "RESIZE2D" => {
            if ctx.t2_m.is_none() {
                return Err(Error::unimplemented("nonfpu RESIZE2D requires T2"));
            }
            let x = args[0].as_str();
            let xsh = ctx.s(x)?.shape.clone();
            if xsh.len() != 3 {
                return Err(Error::unimplemented("nonfpu RESIZE2D needs HWC"));
            }
            let (h, ww, c) = (xsh[0].unwrap(), xsh[1].unwrap(), xsh[2].unwrap());
            let ho = int_lit(&args[1], "ho")?;
            let wo = int_lit(&args[2], "wo")?;
            let align = int_lit(&args[3], "align")?;
            Ok((
                fill(
                    r#"/* {o} = RESIZE2D({x}) T2 weighted bilinear */
{ nf_t2_planes _x = { {x}_s, {x}_e, {x}_m, {x}_z };
  nf_t2_planes _o = { {o}_s, {o}_e, {o}_m, {o}_z };
  nf_t2_resize2d(_x, {h}, {ww}, {c}, {ho}, {wo}, {align}, _o); {rbrace}"#,
                    &[
                        ("o", o.into()),
                        ("x", x.into()),
                        ("h", h.to_string()),
                        ("ww", ww.to_string()),
                        ("c", c.to_string()),
                        ("ho", ho.to_string()),
                        ("wo", wo.to_string()),
                        ("align", align.to_string()),
                        ("rbrace", "}".into()),
                    ],
                ),
                None,
            ))
        }
        "RELU" => {
            let x = args[0].as_str();
            let co = ctx.elems(o)?;
            if ctx.t2_m.is_some() {
                Ok((
                    fill(
                        r#"/* {o} = RELU({x}) T2 sign/zero move */
for (int64_t f_i = 0; f_i < {co}; ++f_i)
  nf_t2_relu({xs}, {xe}, {xm}, {xz}, &{o}_s[f_i], &{o}_e[f_i], &{o}_m[f_i], &{o}_z[f_i]);"#,
                        &[
                            ("o", o.into()),
                            ("x", x.into()),
                            ("co", co.to_string()),
                            ("xs", tplane(ctx, x, "s", "f_i")),
                            ("xe", tplane(ctx, x, "e", "f_i")),
                            ("xm", tplane(ctx, x, "m", "f_i")),
                            ("xz", tplane(ctx, x, "z", "f_i")),
                        ],
                    ),
                    None,
                ))
            } else {
                Ok((
                    fill(
                        r#"/* {o} = RELU({x}) T sign/zero move */
for (int64_t f_i = 0; f_i < {co}; ++f_i) {
  if ({xz} == 0 && {xs} < 0) {{ {o}_s[f_i] = 1; {o}_e[f_i] = 32768; {o}_z[f_i] = 1; }}
  else {{ {o}_s[f_i] = {xs}; {o}_e[f_i] = {xe}; {o}_z[f_i] = {xz}; }}
}"#,
                        &[
                            ("o", o.into()),
                            ("x", x.into()),
                            ("co", co.to_string()),
                            ("xs", tplane(ctx, x, "s", "f_i")),
                            ("xe", tplane(ctx, x, "e", "f_i")),
                            ("xz", tplane(ctx, x, "z", "f_i")),
                        ],
                    ),
                    None,
                ))
            }
        }
        other => Err(Error::unimplemented(format!(
            "nonfpu T2: no pattern for {other} (supported: MUL, SQUARE, ADD, SUB, \
             MATMUL, LAYERNORM, GELU, SOFTMAX, SOFTMAX_WIDE, CONV2D, \
             DECONV2D, RESIZE2D, RELU)"
        ))),
    }
}

fn fx_pattern(mn: &str, outs: &[String], args: &[String], ctx: &mut Ctx) -> Result<Pat> {
    if ctx.t2_m.is_some() {
        return t2_pattern(mn, outs, args, ctx);
    }
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
        "T2I" => {
            let x = args[0].as_str();
            let co = ctx.elems(o)?;
            Ok((
                fill(
                    r#"/* {o} = T2I({x}) phi-ladder -> int64, exact baked table */
for (int64_t t_i = 0; t_i < {co}; ++t_i) {
  if ({x}_z[t_i]) { {o}[t_i] = 0; continue; }
  int32_t _e = {x}_e[t_i];
  if (_e < 0 || _e > 65535) return 31;
  int64_t _v = NF_T2I[_e];
  {o}[t_i] = ({x}_s[t_i] == 0) ? 0 : (({x}_s[t_i] < 0) ? -_v : _v);
}"#,
                    &[("o", o.into()), ("x", x.into()), ("co", co.to_string())],
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
        "LAYERNORM" => {
            let (x, w, b) = (args[0].as_str(), args[1].as_str(), args[2].as_str());
            let d = ctx.width(x, "nonfpu LAYERNORM")?;
            let rows = ctx.elems(x)? / d;
            Ok((
                fill(
                    r#"/* {o} = LAYERNORM({x},{w},{b}) integer rows @ NF_M */
{lbrace} int64_t *_q = malloc({d} * 8);
  int64_t *_wq = malloc({d} * 8);
  int64_t *_bq = malloc({d} * 8);
  for (int64_t r_i = 0; r_i < {rows}; ++r_i)
    nf_layernorm_row(&{x}_s[r_i * {d}], &{x}_e[r_i * {d}], &{x}_z[r_i * {d}],
      {w}_s, {w}_e, {w}_z, {b}_s, {b}_e, {b}_z,
      {d}, NF_M, NF_EPSL,
      &{o}_s[r_i * {d}], &{o}_e[r_i * {d}], &{o}_z[r_i * {d}], _q, _wq, _bq);
  free(_q); free(_wq); free(_bq); {rbrace}"#,
                    &[
                        ("o", o.into()),
                        ("x", x.into()),
                        ("w", w.into()),
                        ("b", b.into()),
                        ("d", d.to_string()),
                        ("rows", rows.to_string()),
                        ("lbrace", "{".into()),
                        ("rbrace", "}".into()),
                    ],
                ),
                None,
            ))
        }
        "GELU" => {
            let a = args[0].as_str();
            Ok((
                fill(
                    r#"/* {o} = GELU({a}) exact-erf LUT */
for (int64_t f_i = 0; f_i < {co}; ++f_i)
  nf_gelu({a}_s[f_i], {a}_e[f_i], {a}_z[f_i], &{o}_s[f_i], &{o}_e[f_i], &{o}_z[f_i]);"#,
                    &[("o", o.into()), ("a", a.into()), ("co", co.to_string())],
                ),
                None,
            ))
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
        "CONV2D" | "DECONV2D" => {
            if ctx.t2_m.is_none() {
                return Err(Error::unimplemented(format!(
                    "nonfpu {mn} requires T2 (CONFIG t2)"
                )));
            }
            let (x, w, b) = (args[0].as_str(), args[1].as_str(), args[2].as_str());
            let xsh = ctx.s(x)?.shape.clone();
            let wsh = ctx.s(w)?.shape.clone();
            if xsh.len() != 3 || wsh.len() != 4 {
                return Err(Error::unimplemented(format!(
                    "nonfpu {mn} needs HWC + (Cout,kh,kw,Cin)"
                )));
            }
            let (h, ww, cin) = (xsh[0].unwrap(), xsh[1].unwrap(), xsh[2].unwrap());
            let (cout, kh, kw) = (wsh[0].unwrap(), wsh[1].unwrap(), wsh[2].unwrap());
            let stride = int_lit(&args[3], "stride")?;
            let pad = int_lit(&args[4], "pad")?;
            let (ho, wo) = if mn == "CONV2D" {
                (
                    (h as i64 + 2 * pad - kh as i64) / stride + 1,
                    (ww as i64 + 2 * pad - kw as i64) / stride + 1,
                )
            } else {
                (
                    (h as i64 - 1) * stride - 2 * pad + kh as i64,
                    (ww as i64 - 1) * stride - 2 * pad + kw as i64,
                )
            };
            if ho <= 0 || wo <= 0 {
                return Err(Error::asm(format!("nonfpu {mn} bad output size")));
            }
            let fn_name = if mn == "CONV2D" {
                "nf_t2_conv2d"
            } else {
                "nf_t2_deconv2d"
            };
            Ok((
                fill(
                    r#"/* {o} = {mn}({x}) T2 block-floating */
{ nf_t2_planes _x = { {x}_s, {x}_e, {x}_m, {x}_z };
  nf_t2_planes _w = { {w}_s, {w}_e, {w}_m, {w}_z };
  nf_t2_planes _b = { {b}_s, {b}_e, {b}_m, {b}_z };
  nf_t2_planes _o = { {o}_s, {o}_e, {o}_m, {o}_z };
  {fn_name}(_x, {h}, {ww}, {cin}, _w, {cout}, {kh}, {kw}, _b,
             {stride}, {pad}, {ho}, {wo}, _o); {rbrace}"#,
                    &[
                        ("o", o.into()),
                        ("mn", mn.into()),
                        ("x", x.into()),
                        ("w", w.into()),
                        ("b", b.into()),
                        ("h", h.to_string()),
                        ("ww", ww.to_string()),
                        ("cin", cin.to_string()),
                        ("cout", cout.to_string()),
                        ("kh", kh.to_string()),
                        ("kw", kw.to_string()),
                        ("ho", ho.to_string()),
                        ("wo", wo.to_string()),
                        ("stride", stride.to_string()),
                        ("pad", pad.to_string()),
                        ("fn_name", fn_name.into()),
                        ("rbrace", "}".into()),
                    ],
                ),
                None,
            ))
        }
        "RESIZE2D" => {
            if ctx.t2_m.is_none() {
                return Err(Error::unimplemented("nonfpu RESIZE2D requires T2"));
            }
            let x = args[0].as_str();
            let xsh = ctx.s(x)?.shape.clone();
            if xsh.len() != 3 {
                return Err(Error::unimplemented("nonfpu RESIZE2D needs HWC"));
            }
            let (h, ww, c) = (xsh[0].unwrap(), xsh[1].unwrap(), xsh[2].unwrap());
            let ho = int_lit(&args[1], "ho")?;
            let wo = int_lit(&args[2], "wo")?;
            let align = int_lit(&args[3], "align")?;
            Ok((
                fill(
                    r#"/* {o} = RESIZE2D({x}) T2 weighted bilinear */
{ nf_t2_planes _x = { {x}_s, {x}_e, {x}_m, {x}_z };
  nf_t2_planes _o = { {o}_s, {o}_e, {o}_m, {o}_z };
  nf_t2_resize2d(_x, {h}, {ww}, {c}, {ho}, {wo}, {align}, _o); {rbrace}"#,
                    &[
                        ("o", o.into()),
                        ("x", x.into()),
                        ("h", h.to_string()),
                        ("ww", ww.to_string()),
                        ("c", c.to_string()),
                        ("ho", ho.to_string()),
                        ("wo", wo.to_string()),
                        ("align", align.to_string()),
                        ("rbrace", "}".into()),
                    ],
                ),
                None,
            ))
        }
        "RELU" => {
            let x = args[0].as_str();
            let co = ctx.elems(o)?;
            if ctx.t2_m.is_some() {
                Ok((
                    fill(
                        r#"/* {o} = RELU({x}) T2 sign/zero move */
for (int64_t f_i = 0; f_i < {co}; ++f_i)
  nf_t2_relu({xs}, {xe}, {xm}, {xz}, &{o}_s[f_i], &{o}_e[f_i], &{o}_m[f_i], &{o}_z[f_i]);"#,
                        &[
                            ("o", o.into()),
                            ("x", x.into()),
                            ("co", co.to_string()),
                            ("xs", tplane(ctx, x, "s", "f_i")),
                            ("xe", tplane(ctx, x, "e", "f_i")),
                            ("xm", tplane(ctx, x, "m", "f_i")),
                            ("xz", tplane(ctx, x, "z", "f_i")),
                        ],
                    ),
                    None,
                ))
            } else {
                Ok((
                    fill(
                        r#"/* {o} = RELU({x}) T sign/zero move */
for (int64_t f_i = 0; f_i < {co}; ++f_i) {
  if ({xz} == 0 && {xs} < 0) {{ {o}_s[f_i] = 1; {o}_e[f_i] = 32768; {o}_z[f_i] = 1; }}
  else {{ {o}_s[f_i] = {xs}; {o}_e[f_i] = {xe}; {o}_z[f_i] = {xz}; }}
}"#,
                        &[
                            ("o", o.into()),
                            ("x", x.into()),
                            ("co", co.to_string()),
                            ("xs", tplane(ctx, x, "s", "f_i")),
                            ("xe", tplane(ctx, x, "e", "f_i")),
                            ("xz", tplane(ctx, x, "z", "f_i")),
                        ],
                    ),
                    None,
                ))
            }
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
