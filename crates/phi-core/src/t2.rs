//! T+mantissa ("phi-float", T2) value format.
//!
//! The frozen T triple stores `sign * PHI^((e - BIAS)/K)`, i.e. a value on the
//! K=512 log-phi lattice. T2 adds an opt-in integer residual:
//!
//! ```text
//! v = z ? 0 : s * PHI^((e - BIAS)/K) * (1 + m / 2^M)
//! ```
//!
//! where `M` is the mantissa width (CONFIG `t2`, default 24), `s in {-1,+1}`,
//! `e in [0,65535]`, and `m` is a signed binary residual. Encoders and bridge
//! outputs produce `m in [-2^(M-1), 2^(M-1))`; a raw MUL may leave the two
//! guard bits (`[-3*2^(M-2), 5*2^(M-2))`) because a product of two factors in
//! `[0.5,1.5)` lies in `[0.25,2.25)`. All stored values are canonical after
//! [`from_fixed`].
//!
//! Arithmetic is integer-only: exponents add (the exact phi ladder), binary
//! mantissas multiply, and the fixed-point bridge carries the residual. The
//! `floor(log2)` / coarse+fine tables are the same ones the frozen T bridge
//! uses, widened to `f_bits` fractional bits (normally `f_bits >= M`).
//!
//! `f64` appears only in [`encode`]/[`decode`], the sensor/display boundary.

use ndarray::ArrayD;

use crate::error::{Error, Result};
use crate::lattice::{self, BIAS, K, WIDE_CAP};
use crate::triple::Triple;

pub const DEFAULT_M: i64 = 24;

/// Tables at one bridge width. Fractions are `round(PHI^... * 2^f)`.
pub struct Tables {
    pub f: i64,
    pub frac: Vec<i64>,
    pub frachi: Vec<i64>,
    pub cap: i64,
}

impl Tables {
    pub fn new(f: i64) -> Self {
        Tables {
            f,
            frac: lattice::l_frac_for(f),
            frachi: lattice::frac_hi_for(f),
            cap: lattice::frac_cap_for(f),
        }
    }

    #[inline]
    fn base(&self, d: i64) -> i64 {
        if d < 0 {
            let k = -d;
            if k > WIDE_CAP {
                self.frac[0]
            } else {
                self.frachi[(k - 1) as usize]
            }
        } else if d > self.cap {
            0
        } else {
            self.frac[d as usize]
        }
    }
}

fn bit_length(mut x: i64) -> i64 {
    if x <= 0 {
        return 0;
    }
    let mut bl = 0i64;
    for sh in [32, 16, 8, 4, 2, 1] {
        if (x >> sh) > 0 {
            bl += sh;
            x >>= sh;
        }
    }
    bl + 1
}

/// Truncating division toward zero.
#[inline]
pub fn trunc_div(a: i128, b: i128) -> i128 {
    if b == 0 {
        return 0;
    }
    if (a < 0) ^ (b < 0) {
        -((a.abs()) / b.abs())
    } else {
        a.abs() / b.abs()
    }
}

/// Round-to-nearest, half away from zero (magnitudes are non-negative here).
#[inline]
pub fn round_div_pos(a: i128, b: i128) -> i128 {
    if b == 0 {
        return 0;
    }
    (a + b / 2) / b
}

/// Round-to-nearest signed divide.
#[inline]
pub fn round_div_signed(a: i128, b: i128) -> i128 {
    if b == 0 {
        return 0;
    }
    if a < 0 {
        -round_div_pos(-a, b)
    } else {
        round_div_pos(a, b)
    }
}

fn isqrt_u64(a: u64) -> u64 {
    if a < 2 {
        return a;
    }
    let mut x = a;
    let mut y = (x + 1) >> 1;
    while y < x {
        x = y;
        y = (x + a / x) >> 1;
    }
    x
}

/// Scalar T2 value.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct T2 {
    pub s: i8,
    pub e: i32,
    pub m: i32,
    pub z: u8,
}

impl T2 {
    pub fn zero() -> Self {
        T2 {
            s: 1,
            e: BIAS as i32,
            m: 0,
            z: 1,
        }
    }

    pub fn is_zero(&self) -> bool {
        self.z != 0
    }
}

/// Sensor boundary: encode `v` into T2 at `m_bits`.
pub fn encode(v: f64, m_bits: i64) -> T2 {
    if v == 0.0 || !v.is_finite() {
        return T2::zero();
    }
    let s = if v < 0.0 { -1 } else { 1 };
    let raw = (K as f64) * v.abs().ln() / lattice::LN_PHI;
    let e = (lattice::round_half_even(raw) as i64 + BIAS).clamp(0, 65535);
    let unit = lattice::PHI.powf((e - BIAS) as f64 / K as f64);
    let scale = (1i64 << m_bits) as f64;
    let half = 1i64 << (m_bits - 1);
    let mut m = lattice::round_half_even((v.abs() / unit - 1.0) * scale) as i64;
    if m < -half {
        m = -half;
    }
    if m > half - 1 {
        m = half - 1;
    }
    T2 {
        s,
        e: e as i32,
        m: m as i32,
        z: 0,
    }
}

/// Display/scoring boundary: decode T2 to `f64`.
pub fn decode(t: T2, m_bits: i64) -> f64 {
    if t.is_zero() {
        return 0.0;
    }
    let unit = lattice::PHI.powf((t.e as i64 - BIAS) as f64 / K as f64);
    (t.s as f64) * unit * (1.0 + t.m as f64 / (1i64 << m_bits) as f64)
}

/// T2 -> fixed-point integer at scale `m0` (integer units of `2^-f`).
pub fn to_fixed(t: T2, m0: i64, tabs: &Tables) -> i64 {
    if t.is_zero() {
        return 0;
    }
    let base = tabs.base(m0 - t.e as i64) as i128;
    let scale = 1i128 << tabs_f(tabs);
    let q = (base * (scale + t.m as i128) + (scale >> 1)) >> tabs_f(tabs);
    (t.s as i128 * q) as i64
}

#[inline]
fn tabs_f(tabs: &Tables) -> i64 {
    tabs.f
}

/// Fixed-point integer at scale `m0` -> canonical T2.
pub fn from_fixed(q: i64, m0: i64, tabs: &Tables, m_bits: i64) -> T2 {
    if q == 0 {
        return T2::zero();
    }
    let s = if q > 0 { 1 } else { -1 };
    let a = (q as i128).unsigned_abs() as i128;
    let a64 = a as i64;
    let bl = bit_length(a64);
    let shift = bl - 15;
    let mant = if shift >= 0 {
        a64 >> shift
    } else {
        a64 << (-shift)
    };
    let t = shift + 14 - tabs.f;
    let idx = (t + 64).clamp(0, 192);
    let fi = (mant - 16384).clamp(0, 16383);
    let coarse = lattice::l_coarse();
    let fine = lattice::l_fine();
    let e = (m0 + coarse[idx as usize] + fine[fi as usize]).clamp(0, 65535);
    let base = tabs.base(m0 - e) as i128;
    let scale = 1i128 << m_bits;
    let half = 1i128 << (m_bits - 1);
    let m = if base == 0 {
        0
    } else {
        (round_div_pos(a << m_bits, base) - scale).clamp(-half, half - 1) as i64
    };
    T2 {
        s,
        e: e as i32,
        m: m as i32,
        z: 0,
    }
}

/// MUL: exponents add exactly; mantissa factors multiply with the binary
/// correction `ma*mb/2^M`.
pub fn mul(a: T2, b: T2, m_bits: i64) -> T2 {
    if a.is_zero() || b.is_zero() {
        return T2::zero();
    }
    let scale = 1i64 << m_bits;
    let cross = round_div_signed(a.m as i128 * b.m as i128, scale as i128) as i64;
    let m = (a.m as i64) + (b.m as i64) + cross;
    let e = ((a.e as i64) + (b.e as i64) - BIAS).clamp(0, 65535);
    T2 {
        s: (a.s as i32 * b.s as i32) as i8,
        e: e as i32,
        m: m as i32,
        z: 0,
    }
}

/// ADD/SUB: align to the larger phi exponent, add fixed integers, canonicalize.
pub fn add(a: T2, b: T2, sub: bool, tabs: &Tables, m_bits: i64) -> T2 {
    if a.is_zero() && b.is_zero() {
        return T2::zero();
    }
    let m0 = if a.is_zero() {
        b.e as i64
    } else if b.is_zero() {
        a.e as i64
    } else {
        (a.e as i64).max(b.e as i64)
    };
    let qa = to_fixed(a, m0, tabs);
    let qb = to_fixed(b, m0, tabs);
    let acc = if sub {
        qa as i128 - qb as i128
    } else {
        qa as i128 + qb as i128
    };
    let acc = acc.clamp(i64::MIN as i128, i64::MAX as i128) as i64;
    from_fixed(acc, m0, tabs, m_bits)
}

// ---------------------------------------------------------------------------
// Array-level reference semantics (used by the interpreter and tests).
// ---------------------------------------------------------------------------

fn planes(t: &Triple) -> Result<(&[i8], &[i32], &[i32], &[u8])> {
    let m = t
        .m
        .as_ref()
        .ok_or_else(|| Error::runtime("T2 op needs a mantissa plane"))?;
    Ok((
        t.s.as_slice().unwrap(),
        t.e.as_slice().unwrap(),
        m.as_slice().unwrap(),
        t.z.as_slice().unwrap(),
    ))
}

fn from_planes(shape: &[usize], s: Vec<i8>, e: Vec<i32>, m: Vec<i32>, z: Vec<u8>) -> Result<Triple> {
    Triple::from_vecs_opt(shape, s, e, z, Some(m))
}

pub fn mul_array(a: &Triple, b: &Triple, m_bits: i64) -> Result<Triple> {
    if a.shape() != b.shape() {
        return Err(Error::runtime("T2 MUL shape mismatch"));
    }
    let (as_, ae, am, az) = planes(a)?;
    let (bs, be, bm, bz) = planes(b)?;
    let n = as_.len();
    let (mut s, mut e, mut m, mut z) = (Vec::with_capacity(n), Vec::with_capacity(n), Vec::with_capacity(n), Vec::with_capacity(n));
    for i in 0..n {
        let r = mul(
            T2 { s: as_[i], e: ae[i], m: am[i], z: az[i] },
            T2 { s: bs[i], e: be[i], m: bm[i], z: bz[i] },
            m_bits,
        );
        s.push(r.s);
        e.push(r.e);
        m.push(r.m);
        z.push(r.z);
    }
    from_planes(&a.shape(), s, e, m, z)
}

pub fn add_array(a: &Triple, b: &Triple, sub: bool, tabs: &Tables, m_bits: i64) -> Result<Triple> {
    if a.shape() != b.shape() {
        return Err(Error::runtime("T2 ADD shape mismatch"));
    }
    let (as_, ae, am, az) = planes(a)?;
    let (bs, be, bm, bz) = planes(b)?;
    let n = as_.len();
    let (mut s, mut e, mut m, mut z) = (Vec::with_capacity(n), Vec::with_capacity(n), Vec::with_capacity(n), Vec::with_capacity(n));
    for i in 0..n {
        let r = add(
            T2 { s: as_[i], e: ae[i], m: am[i], z: az[i] },
            T2 { s: bs[i], e: be[i], m: bm[i], z: bz[i] },
            sub,
            tabs,
            m_bits,
        );
        s.push(r.s);
        e.push(r.e);
        m.push(r.m);
        z.push(r.z);
    }
    from_planes(&a.shape(), s, e, m, z)
}

/// 2-D MATMUL with a block-floating accumulator: each output dot product is
/// aligned to its largest term exponent, so small terms are not crushed by a
/// global fixed scale.
pub fn matmul2(a: &Triple, b: &Triple, tabs: &Tables, m_bits: i64) -> Result<Triple> {
    let ash = a.shape();
    let bsh = b.shape();
    if ash.len() != 2 || bsh.len() != 2 || ash[1] != bsh[0] {
        return Err(Error::runtime("T2 MATMUL needs 2-D (m,k)x(k,n)"));
    }
    let (m, k, n) = (ash[0], ash[1], bsh[1]);
    let (as_, ae, am, az) = planes(a)?;
    let (bs, be, bm, bz) = planes(b)?;
    let scale = 1i128 << m_bits;
    let half = 1i128 << (2 * m_bits - 1);
    let mut os = vec![1i8; m * n];
    let mut oe = vec![BIAS as i32; m * n];
    let mut om = vec![0i32; m * n];
    let mut oz = vec![1u8; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut emax: Option<i64> = None;
            for t in 0..k {
                let ai = i * k + t;
                let bi = t * n + j;
                if az[ai] != 0 || bz[bi] != 0 {
                    continue;
                }
                let ep = ae[ai] as i64 + be[bi] as i64 - BIAS;
                emax = Some(emax.map_or(ep, |x| x.max(ep)));
            }
            let Some(emax) = emax else { continue };
            let mut acc: i128 = 0;
            for t in 0..k {
                let ai = i * k + t;
                let bi = t * n + j;
                if az[ai] != 0 || bz[bi] != 0 {
                    continue;
                }
                let ep = ae[ai] as i64 + be[bi] as i64 - BIAS;
                let base = tabs.base(emax - ep) as i128;
                if base == 0 {
                    continue;
                }
                let aa = scale + am[ai] as i128;
                let bb = scale + bm[bi] as i128;
                let p = (base * aa * bb + half) >> (2 * m_bits);
                let sign = as_[ai] as i128 * bs[bi] as i128;
                acc += sign * p;
            }
            let acc = acc.clamp(i64::MIN as i128, i64::MAX as i128) as i64;
            let r = from_fixed(acc, emax, tabs, m_bits);
            let o = i * n + j;
            os[o] = r.s;
            oe[o] = r.e;
            om[o] = r.m;
            oz[o] = r.z;
        }
    }
    from_planes(&[m, n], os, oe, om, oz)
}

fn eps_counts(eps: f64) -> i128 {
    // eps_q is the integer bridge for the LayerNorm epsilon; CONFIG eps_ln is
    // a build-time scalar. 2^30 keeps 1e-6 exact enough for f <= 24.
    lattice::round_half_even(eps * (1i64 << 30) as f64) as i128
}

fn eps_scaled(eps_q: i128, q1: i64) -> i64 {
    let q = q1 as i128;
    ((eps_q * q * q + (1i128 << 29)) >> 30).clamp(0, i64::MAX as i128) as i64
}

/// LAYERNORM over the last axis. `w`/`b` have shape `[C]`.
pub fn layernorm(
    x: &Triple,
    w: &Triple,
    b: &Triple,
    tabs: &Tables,
    m_bits: i64,
    eps: f64,
) -> Result<Triple> {
    let xsh = x.shape();
    let c = *xsh
        .last()
        .ok_or_else(|| Error::runtime("T2 LAYERNORM needs a last axis"))?;
    let rows = x.numel() / c;
    let (xs, xe, xm, xz) = planes(x)?;
    let (ws, we, wm, wz) = planes(w)?;
    let (bs, be_, bm, bz) = planes(b)?;
    if ws.len() != c || bs.len() != c {
        return Err(Error::runtime("T2 LAYERNORM weight shape mismatch"));
    }
    let eps_q = eps_counts(eps);
    let mut os = vec![1i8; rows * c];
    let mut oe = vec![BIAS as i32; rows * c];
    let mut om = vec![0i32; rows * c];
    let mut oz = vec![1u8; rows * c];
    let mut q = vec![0i64; c];
    let mut wq = vec![0i64; c];
    let mut bq = vec![0i64; c];
    for r in 0..rows {
        let mut m0 = BIAS;
        let mut any = false;
        for j in 0..c {
            let i = r * c + j;
            if xz[i] == 0 {
                if !any || xe[i] as i64 > m0 {
                    m0 = xe[i] as i64;
                }
                any = true;
            }
        }
        if !any {
            m0 = BIAS;
        }
        for j in 0..c {
            let i = r * c + j;
            q[j] = to_fixed(
                T2 { s: xs[i], e: xe[i], m: xm[i], z: xz[i] },
                m0,
                tabs,
            );
            wq[j] = to_fixed(
                T2 { s: ws[j], e: we[j], m: wm[j], z: wz[j] },
                m0,
                tabs,
            );
            bq[j] = to_fixed(
                T2 { s: bs[j], e: be_[j], m: bm[j], z: bz[j] },
                m0,
                tabs,
            );
        }
        let sum: i128 = q.iter().map(|&v| v as i128).sum();
        let mean = trunc_div(sum, c as i128) as i64;
        let ss: i128 = q
            .iter()
            .map(|&v| {
                let d = v as i128 - mean as i128;
                d * d
            })
            .sum();
        let var = trunc_div(ss, c as i128) as i64 + eps_scaled(eps_q, to_fixed(
            T2 { s: 1, e: BIAS as i32, m: 0, z: 0 },
            m0,
            tabs,
        ));
        let is = isqrt_u64(var.max(0) as u64) as i128;
        let scale = 1i128 << tabs.f;
        for j in 0..c {
            let i = r * c + j;
            let d = q[j] as i128 - mean as i128;
            let norm = if is == 0 { 0 } else { trunc_div(d * scale, is) };
            let aq = trunc_div(norm * wq[j] as i128, scale);
            let av = (aq + bq[j] as i128).clamp(i64::MIN as i128, i64::MAX as i128) as i64;
            let out = from_fixed(av, m0, tabs, m_bits);
            os[i] = out.s;
            oe[i] = out.e;
            om[i] = out.m;
            oz[i] = out.z;
        }
    }
    from_planes(&xsh, os, oe, om, oz)
}

/// GELU (exact-erf CDF LUT), operating on T2.
pub fn gelu(a: &Triple, tabs: &Tables, m_bits: i64) -> Result<Triple> {
    let (as_, ae, am, az) = planes(a)?;
    let n = as_.len();
    let (mut s, mut e, mut m, mut z) = (Vec::with_capacity(n), Vec::with_capacity(n), Vec::with_capacity(n), Vec::with_capacity(n));
    let phi = lattice::phi_lut();
    let span = lattice::SIG_SPAN;
    let sh = tabs.f - 14;
    for i in 0..n {
        let q = to_fixed(T2 { s: as_[i], e: ae[i], m: am[i], z: az[i] }, BIAS, tabs);
        let x14 = trunc_div(q as i128, 1i128 << sh) as i64;
        let p14 = if x14 < -span {
            0
        } else if x14 > span {
            16384
        } else {
            phi[(x14 + span) as usize]
        };
        let y14 = trunc_div(x14 as i128 * p14 as i128, 16384) as i64;
        let out = from_fixed(y14 << sh, BIAS, tabs, m_bits);
        s.push(out.s);
        e.push(out.e);
        m.push(out.m);
        z.push(out.z);
    }
    from_planes(&a.shape(), s, e, m, z)
}

/// Row softmax at BIAS scale (`SOFTMAX`/`SOFTMAX_WIDE` share the T2 path).
pub fn softmax(a: &Triple, tabs: &Tables, m_bits: i64) -> Result<Triple> {
    let sh = a.shape();
    let d = *sh
        .last()
        .ok_or_else(|| Error::runtime("T2 SOFTMAX needs a last axis"))?;
    let rows = a.numel() / d;
    let (as_, ae, am, az) = planes(a)?;
    let exp = lattice::exp_lut();
    let (mut s, mut e, mut m, mut z) = (vec![1i8; rows * d], vec![BIAS as i32; rows * d], vec![0i32; rows * d], vec![1u8; rows * d]);
    let sh14 = tabs.f - 14;
    for r in 0..rows {
        let mut q = vec![0i64; d];
        let mut s14 = vec![0i64; d];
        let mut vmax = i64::MIN;
        for j in 0..d {
            let i = r * d + j;
            q[j] = to_fixed(T2 { s: as_[i], e: ae[i], m: am[i], z: az[i] }, BIAS, tabs);
            s14[j] = trunc_div(q[j] as i128, 1i128 << sh14) as i64;
            if s14[j] > vmax {
                vmax = s14[j];
            }
        }
        let mut num = vec![0i64; d];
        let mut den: i128 = 0;
        for j in 0..d {
            let dd = (vmax - s14[j]).clamp(0, 262144);
            num[j] = exp[dd as usize];
            den += num[j] as i128;
        }
        if den == 0 {
            den = 1;
        }
        for j in 0..d {
            let i = r * d + j;
            let c = trunc_div(num[j] as i128 * (1i128 << tabs.f), den) as i64;
            let out = from_fixed(c, BIAS, tabs, m_bits);
            s[i] = out.s;
            e[i] = out.e;
            m[i] = out.m;
            z[i] = out.z;
        }
    }
    from_planes(&sh, s, e, m, z)
}

/// Encode a whole array into T2 (sensor boundary).
pub fn encode_array(a: &ArrayD<f64>, m_bits: i64) -> Triple {
    let shape = a.shape().to_vec();
    let n = a.len();
    let mut s = Vec::with_capacity(n);
    let mut e = Vec::with_capacity(n);
    let mut m = Vec::with_capacity(n);
    let mut z = Vec::with_capacity(n);
    for &v in a.iter() {
        let t = encode(v, m_bits);
        s.push(t.s);
        e.push(t.e);
        m.push(t.m);
        z.push(t.z);
    }
    Triple::from_vecs_opt(&shape, s, e, z, Some(m)).expect("t2 encode shape")
}

/// Decode a whole T2 array (display boundary).
pub fn decode_array(t: &Triple, m_bits: i64) -> Result<ArrayD<f64>> {
    let (s, e, m, z) = planes(t)?;
    let n = s.len();
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        out.push(decode(
            T2 { s: s[i], e: e[i], m: m[i], z: z[i] },
            m_bits,
        ));
    }
    Ok(ArrayD::from_shape_vec(ndarray::IxDyn(&t.shape()), out).expect("t2 decode shape"))
}
