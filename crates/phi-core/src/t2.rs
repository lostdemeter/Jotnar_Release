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

/// Validate the T2/bridge configuration widths.
///
/// `t2` is the mantissa width M (1..=30), `bridge_f` is the fixed-point
/// fraction width (14..=62). For T2 the effective bridge is
/// `max(bridge_f, M, 24)`; a listing must not silently lower the validated
/// T2 bridge below 24 or below its mantissa width.
pub fn check_widths(t2: Option<i64>, bridge_f: Option<i64>) -> Result<(Option<i64>, i64)> {
    if let Some(m) = t2 {
        if !(1..=30).contains(&m) {
            return Err(Error::runtime(format!(
                "CONFIG t2 must be 1..=30 (got {m})"
            )));
        }
    }
    let bf = match bridge_f {
        Some(f) => {
            if !(14..=62).contains(&f) {
                return Err(Error::runtime(format!(
                    "CONFIG bridge_f must be 14..=62 (got {f})"
                )));
            }
            f
        }
        None => 18,
    };
    let f = match t2 {
        Some(m) => bf.max(m).max(24),
        None => bf,
    };
    Ok((t2, f))
}

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

fn bit_length(mut x: u128) -> i64 {
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
///
/// The residual is `1 + m/2^M`, so the mantissa shift is `m_bits`; the
/// bridge width `tabs.f` only sets the fixed-point fraction scale.
pub fn to_fixed(t: T2, m0: i64, tabs: &Tables, m_bits: i64) -> i64 {
    if t.is_zero() {
        return 0;
    }
    let base = tabs.base(m0 - t.e as i64) as i128;
    let scale = 1i128 << m_bits;
    let q = (base * (scale + t.m as i128) + (scale >> 1)) >> m_bits;
    (t.s as i128 * q) as i64
}

/// Fixed-point integer at scale `m0` -> canonical T2.
pub fn from_fixed(q: i64, m0: i64, tabs: &Tables, m_bits: i64) -> T2 {
    if q == 0 {
        return T2::zero();
    }
    let s = if q > 0 { 1 } else { -1 };
    // u128 magnitude avoids the i64::MIN wrap in `q.abs()`/`as i64`.
    let a: u128 = (q as i128).unsigned_abs();
    let bl = bit_length(a);
    let shift = bl - 15;
    let mant: u128 = if shift >= 0 {
        a >> shift
    } else {
        a << (-shift)
    };
    let t = shift + 14 - tabs.f;
    let idx = (t + 64).clamp(0, 192);
    let fi = (mant.saturating_sub(16384)).min(16383) as i64;
    let coarse = lattice::l_coarse();
    let fine = lattice::l_fine();
    let e = (m0 + coarse[idx as usize] + fine[fi as usize]).clamp(0, 65535);
    let base = tabs.base(m0 - e) as i128;
    let scale = 1i128 << m_bits;
    let half = 1i128 << (m_bits - 1);
    let m = if base == 0 {
        0
    } else {
        (round_div_pos((a << m_bits) as i128, base) - scale).clamp(-half, half - 1) as i64
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
    let qa = to_fixed(a, m0, tabs, m_bits);
    let qb = to_fixed(b, m0, tabs, m_bits);
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
    let m =
        t.m.as_ref()
            .ok_or_else(|| Error::runtime("T2 op needs a mantissa plane"))?;
    Ok((
        t.s.as_slice().unwrap(),
        t.e.as_slice().unwrap(),
        m.as_slice().unwrap(),
        t.z.as_slice().unwrap(),
    ))
}

fn from_planes(
    shape: &[usize],
    s: Vec<i8>,
    e: Vec<i32>,
    m: Vec<i32>,
    z: Vec<u8>,
) -> Result<Triple> {
    Triple::from_vecs_opt(shape, s, e, z, Some(m))
}

pub fn mul_array(a: &Triple, b: &Triple, m_bits: i64) -> Result<Triple> {
    if a.shape() != b.shape() {
        return Err(Error::runtime("T2 MUL shape mismatch"));
    }
    let (as_, ae, am, az) = planes(a)?;
    let (bs, be, bm, bz) = planes(b)?;
    let n = as_.len();
    let (mut s, mut e, mut m, mut z) = (
        Vec::with_capacity(n),
        Vec::with_capacity(n),
        Vec::with_capacity(n),
        Vec::with_capacity(n),
    );
    for i in 0..n {
        let r = mul(
            T2 {
                s: as_[i],
                e: ae[i],
                m: am[i],
                z: az[i],
            },
            T2 {
                s: bs[i],
                e: be[i],
                m: bm[i],
                z: bz[i],
            },
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
    let (mut s, mut e, mut m, mut z) = (
        Vec::with_capacity(n),
        Vec::with_capacity(n),
        Vec::with_capacity(n),
        Vec::with_capacity(n),
    );
    for i in 0..n {
        let r = add(
            T2 {
                s: as_[i],
                e: ae[i],
                m: am[i],
                z: az[i],
            },
            T2 {
                s: bs[i],
                e: be[i],
                m: bm[i],
                z: bz[i],
            },
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

/// Output-block kernel: rows `i0..i0+rows` of an `(m,k)x(k,n)` product,
/// written into the corresponding block-local output slices.
///
/// Each output element is a pure function of `(i, j)`: the exponent scan and
/// the accumulation run in ascending `t` order. That is the determinism
/// contract — partitioning rows changes which thread computes an element, not
/// how it is computed.
#[allow(clippy::too_many_arguments)]
fn matmul_block(
    as_: &[i8],
    ae: &[i32],
    am: &[i32],
    az: &[u8],
    bs: &[i8],
    be: &[i32],
    bm: &[i32],
    bz: &[u8],
    (k, n): (usize, usize),
    i0: usize,
    rows: usize,
    tabs: &Tables,
    m_bits: i64,
    os: &mut [i8],
    oe: &mut [i32],
    om: &mut [i32],
    oz: &mut [u8],
) {
    let scale = 1i128 << m_bits;
    let half = 1i128 << (2 * m_bits - 1);
    for ii in 0..rows {
        let i = i0 + ii;
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
            let o = ii * n + j;
            os[o] = r.s;
            oe[o] = r.e;
            om[o] = r.m;
            oz[o] = r.z;
        }
    }
}

fn matmul_shape(ash: &[usize], bsh: &[usize]) -> Result<(usize, usize, usize)> {
    if ash.len() != 2 || bsh.len() != 2 || ash[1] != bsh[0] {
        return Err(Error::runtime("T2 MATMUL needs 2-D (m,k)x(k,n)"));
    }
    Ok((ash[0], ash[1], bsh[1]))
}

/// 2-D MATMUL with a block-floating accumulator: each output dot product is
/// aligned to its largest term exponent, so small terms are not crushed by a
/// global fixed scale.
pub fn matmul2(a: &Triple, b: &Triple, tabs: &Tables, m_bits: i64) -> Result<Triple> {
    matmul2_mt(a, b, tabs, m_bits, 1)
}

/// Deterministic output-partitioned MATMUL.
///
/// Work is split into contiguous row blocks by a pure function of `m` and
/// `threads`; every element keeps the serial accumulation order. Any thread
/// count therefore produces bit-identical planes. `threads <= 1` takes the
/// plain serial path.
pub fn matmul2_mt(
    a: &Triple,
    b: &Triple,
    tabs: &Tables,
    m_bits: i64,
    threads: usize,
) -> Result<Triple> {
    let (m, k, n) = matmul_shape(&a.shape(), &b.shape())?;
    let (as_, ae, am, az) = planes(a)?;
    let (bs, be, bm, bz) = planes(b)?;
    let mut os = vec![1i8; m * n];
    let mut oe = vec![BIAS as i32; m * n];
    let mut om = vec![0i32; m * n];
    let mut oz = vec![1u8; m * n];

    let threads = threads.clamp(1, m.max(1));
    if threads == 1 {
        matmul_block(
            as_,
            ae,
            am,
            az,
            bs,
            be,
            bm,
            bz,
            (k, n),
            0,
            m,
            tabs,
            m_bits,
            &mut os,
            &mut oe,
            &mut om,
            &mut oz,
        );
        return from_planes(&[m, n], os, oe, om, oz);
    }

    // Stable row partition: exactly the same blocks for a given (m, threads).
    let rows_per = m.div_ceil(threads);
    let block_len = rows_per * n;
    std::thread::scope(|scope| {
        let os_chunks = os.chunks_mut(block_len);
        let oe_chunks = oe.chunks_mut(block_len);
        let om_chunks = om.chunks_mut(block_len);
        let oz_chunks = oz.chunks_mut(block_len);
        for (block, (((os_c, oe_c), om_c), oz_c)) in os_chunks
            .zip(oe_chunks)
            .zip(om_chunks)
            .zip(oz_chunks)
            .enumerate()
        {
            let i0 = block * rows_per;
            let rows = os_c.len() / n;
            scope.spawn(move || {
                matmul_block(
                    as_,
                    ae,
                    am,
                    az,
                    bs,
                    be,
                    bm,
                    bz,
                    (k, n),
                    i0,
                    rows,
                    tabs,
                    m_bits,
                    os_c,
                    oe_c,
                    om_c,
                    oz_c,
                );
            });
        }
    });
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
                T2 {
                    s: xs[i],
                    e: xe[i],
                    m: xm[i],
                    z: xz[i],
                },
                m0,
                tabs,
                m_bits,
            );
            wq[j] = to_fixed(
                T2 {
                    s: ws[j],
                    e: we[j],
                    m: wm[j],
                    z: wz[j],
                },
                m0,
                tabs,
                m_bits,
            );
            bq[j] = to_fixed(
                T2 {
                    s: bs[j],
                    e: be_[j],
                    m: bm[j],
                    z: bz[j],
                },
                m0,
                tabs,
                m_bits,
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
        let var = trunc_div(ss, c as i128) as i64
            + eps_scaled(
                eps_q,
                to_fixed(
                    T2 {
                        s: 1,
                        e: BIAS as i32,
                        m: 0,
                        z: 0,
                    },
                    m0,
                    tabs,
                    m_bits,
                ),
            );
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
    let (mut s, mut e, mut m, mut z) = (
        Vec::with_capacity(n),
        Vec::with_capacity(n),
        Vec::with_capacity(n),
        Vec::with_capacity(n),
    );
    let phi = lattice::phi_lut();
    let span = lattice::SIG_SPAN;
    let sh = tabs.f - 14;
    for i in 0..n {
        let q = to_fixed(
            T2 {
                s: as_[i],
                e: ae[i],
                m: am[i],
                z: az[i],
            },
            BIAS,
            tabs,
            m_bits,
        );
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
#[allow(clippy::too_many_arguments)]
fn softmax_rows(
    as_: &[i8],
    ae: &[i32],
    am: &[i32],
    az: &[u8],
    d: usize,
    r0: usize,
    nrows: usize,
    sh14: i64,
    exp: &[i64],
    tabs: &Tables,
    m_bits: i64,
    os: &mut [i8],
    oe: &mut [i32],
    om: &mut [i32],
    oz: &mut [u8],
) {
    let mut q = vec![0i64; d];
    let mut s14 = vec![0i64; d];
    let mut num = vec![0i64; d];
    for rr in 0..nrows {
        let r = r0 + rr;
        let mut vmax = i64::MIN;
        for j in 0..d {
            let i = r * d + j;
            q[j] = to_fixed(
                T2 {
                    s: as_[i],
                    e: ae[i],
                    m: am[i],
                    z: az[i],
                },
                BIAS,
                tabs,
                m_bits,
            );
            s14[j] = trunc_div(q[j] as i128, 1i128 << sh14) as i64;
            if s14[j] > vmax {
                vmax = s14[j];
            }
        }
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
            let k = rr * d + j;
            let c = trunc_div(num[j] as i128 * (1i128 << tabs.f), den) as i64;
            let out = from_fixed(c, BIAS, tabs, m_bits);
            os[k] = out.s;
            oe[k] = out.e;
            om[k] = out.m;
            oz[k] = out.z;
        }
    }
}

pub fn softmax(a: &Triple, tabs: &Tables, m_bits: i64) -> Result<Triple> {
    softmax_mt(a, tabs, m_bits, 1)
}

/// Deterministic row-partitioned softmax. Each row keeps its own reduction
/// order; rows are independent.
pub fn softmax_mt(a: &Triple, tabs: &Tables, m_bits: i64, threads: usize) -> Result<Triple> {
    let sh = a.shape();
    let d = *sh
        .last()
        .ok_or_else(|| Error::runtime("T2 SOFTMAX needs a last axis"))?;
    let rows = a.numel() / d;
    let (as_, ae, am, az) = planes(a)?;
    let exp = lattice::exp_lut();
    let (mut s, mut e, mut m, mut z) = (
        vec![1i8; rows * d],
        vec![BIAS as i32; rows * d],
        vec![0i32; rows * d],
        vec![1u8; rows * d],
    );
    let sh14 = tabs.f - 14;
    let threads = threads.clamp(1, rows.max(1));
    let block_rows = rows.div_ceil(threads);
    let block_len = block_rows * d;
    if threads == 1 {
        softmax_rows(
            as_, ae, am, az, d, 0, rows, sh14, exp, tabs, m_bits, &mut s, &mut e, &mut m, &mut z,
        );
    } else {
        std::thread::scope(|scope| {
            for (block, (((os_c, oe_c), om_c), oz_c)) in s
                .chunks_mut(block_len)
                .zip(e.chunks_mut(block_len))
                .zip(m.chunks_mut(block_len))
                .zip(z.chunks_mut(block_len))
                .enumerate()
            {
                let r0 = block * block_rows;
                let nrows = os_c.len() / d;
                scope.spawn(move || {
                    softmax_rows(
                        as_, ae, am, az, d, r0, nrows, sh14, exp, tabs, m_bits, os_c, oe_c, om_c,
                        oz_c,
                    );
                });
            }
        });
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
            T2 {
                s: s[i],
                e: e[i],
                m: m[i],
                z: z[i],
            },
            m_bits,
        ));
    }
    Ok(ArrayD::from_shape_vec(ndarray::IxDyn(&t.shape()), out).expect("t2 decode shape"))
}

// ---------------------------------------------------------------------------
// T2 image ops: block-floating convolution / transposed conv / resize / relu
// ---------------------------------------------------------------------------

#[inline]
fn t2_at(s: &[i8], e: &[i32], m: &[i32], z: &[u8], i: usize) -> T2 {
    T2 {
        s: s[i],
        e: e[i],
        m: m[i],
        z: z[i],
    }
}

#[inline]
fn t2_one() -> T2 {
    T2 {
        s: 1,
        e: BIAS as i32,
        m: 0,
        z: 0,
    }
}

/// Accumulate a block of T2 products, aligned to the largest term exponent.
/// This is the same arithmetic as `matmul2`, factored for the image ops.
fn accumulate(terms: &[(T2, T2)], bias: Option<T2>, tabs: &Tables, m_bits: i64) -> T2 {
    let mut emax: Option<i64> = None;
    for (a, b) in terms {
        if a.z != 0 || b.z != 0 {
            continue;
        }
        let ep = a.e as i64 + b.e as i64 - BIAS;
        emax = Some(emax.map_or(ep, |x| x.max(ep)));
    }
    if let Some(bi) = bias {
        if bi.z == 0 {
            emax = Some(emax.map_or(bi.e as i64, |x| x.max(bi.e as i64)));
        }
    }
    let Some(emax) = emax else { return T2::zero() };
    let scale = 1i128 << m_bits;
    let half = 1i128 << (2 * m_bits - 1);
    let mut acc: i128 = 0;
    let mut add_term = |a: T2, b: T2| {
        if a.z != 0 || b.z != 0 {
            return;
        }
        let ep = a.e as i64 + b.e as i64 - BIAS;
        let base = tabs.base(emax - ep) as i128;
        if base == 0 {
            return;
        }
        let aa = scale + a.m as i128;
        let bb = scale + b.m as i128;
        let p = (base * aa * bb + half) >> (2 * m_bits);
        acc += (a.s as i128) * (b.s as i128) * p;
    };
    for (a, b) in terms {
        add_term(*a, *b);
    }
    if let Some(bi) = bias {
        add_term(t2_one(), bi);
    }
    let q = acc.clamp(i64::MIN as i128, i64::MAX as i128) as i64;
    from_fixed(q, emax, tabs, m_bits)
}

/// T2 RELU: negative -> zero; positive/zero pass through.
pub fn relu_array(a: &Triple) -> Result<Triple> {
    let (as_, ae, am, az) = planes(a)?;
    let mut s = Vec::with_capacity(as_.len());
    let mut e = Vec::with_capacity(as_.len());
    let mut m = Vec::with_capacity(as_.len());
    let mut z = Vec::with_capacity(as_.len());
    for i in 0..as_.len() {
        let t = t2_at(as_, ae, am, az, i);
        if t.z == 0 && t.s < 0 {
            let q = T2::zero();
            s.push(q.s);
            e.push(q.e);
            m.push(q.m);
            z.push(q.z);
        } else {
            s.push(t.s);
            e.push(t.e);
            m.push(t.m);
            z.push(t.z);
        }
    }
    from_planes(&a.shape(), s, e, m, z)
}

/// Compute a contiguous block of flattened HWC outputs for CONV2D.
///
/// The scratch term list is reused across output elements (the old kernel
/// allocated one `Vec` per pixel), but the term order and the block-floating
/// accumulation are unchanged, so results stay bit-identical.
#[allow(clippy::too_many_arguments)]
fn conv2d_block(
    xs: &[i8],
    xe: &[i32],
    xm: &[i32],
    xz: &[u8],
    ws: &[i8],
    we: &[i32],
    wm: &[i32],
    wz: &[u8],
    bs: &[i8],
    be: &[i32],
    bm: &[i32],
    bz: &[u8],
    (h, ww, cin): (usize, usize, usize),
    (cout, kh, kw): (usize, usize, usize),
    (_ho, wo): (usize, usize),
    stride: i64,
    pad: i64,
    oi0: usize,
    tabs: &Tables,
    m_bits: i64,
    os: &mut [i8],
    oe: &mut [i32],
    om: &mut [i32],
    oz: &mut [u8],
) {
    let mut terms: Vec<(T2, T2)> = Vec::with_capacity(kh * kw * cin);
    for k in 0..os.len() {
        let oi = oi0 + k;
        let co = oi % cout;
        let p = oi / cout;
        let oy = p / wo;
        let ox = p % wo;
        terms.clear();
        for ky in 0..kh {
            let iy = oy as i64 * stride + ky as i64 - pad;
            if iy < 0 || iy >= h as i64 {
                continue;
            }
            for kx in 0..kw {
                let ix = ox as i64 * stride + kx as i64 - pad;
                if ix < 0 || ix >= ww as i64 {
                    continue;
                }
                for ci in 0..cin {
                    let ai = ((iy as usize) * ww + ix as usize) * cin + ci;
                    let wi = ((co * kh + ky) * kw + kx) * cin + ci;
                    terms.push((t2_at(xs, xe, xm, xz, ai), t2_at(ws, we, wm, wz, wi)));
                }
            }
        }
        let bias = t2_at(bs, be, bm, bz, co);
        let r = accumulate(
            &terms,
            if bias.z == 0 { Some(bias) } else { None },
            tabs,
            m_bits,
        );
        os[k] = r.s;
        oe[k] = r.e;
        om[k] = r.m;
        oz[k] = r.z;
    }
}

/// T2 CONV2D on HWC, weight (Cout,kh,kw,Cin), bias (Cout).
pub fn conv2d(
    x: &Triple,
    w: &Triple,
    b: &Triple,
    stride: i64,
    pad: i64,
    tabs: &Tables,
    m_bits: i64,
) -> Result<Triple> {
    conv2d_mt(x, w, b, stride, pad, tabs, m_bits, 1)
}

/// Deterministic output-partitioned CONV2D. Output pixels are independent and
/// the flattened index partition is a pure function of shape and threads.
#[allow(clippy::too_many_arguments)]
pub fn conv2d_mt(
    x: &Triple,
    w: &Triple,
    b: &Triple,
    stride: i64,
    pad: i64,
    tabs: &Tables,
    m_bits: i64,
    threads: usize,
) -> Result<Triple> {
    let xsh = x.shape();
    let wsh = w.shape();
    if xsh.len() != 3 || wsh.len() != 4 {
        return Err(Error::runtime("T2 CONV2D needs HWC + (Cout,kh,kw,Cin)"));
    }
    let (h, ww, cin) = (xsh[0], xsh[1], xsh[2]);
    let (cout, kh, kw, wcin) = (wsh[0], wsh[1], wsh[2], wsh[3]);
    if wcin != cin || b.s.len() != cout || stride <= 0 || pad < 0 {
        return Err(Error::runtime("T2 CONV2D channel/stride mismatch"));
    }
    let ho = (h as i64 + 2 * pad - kh as i64) / stride + 1;
    let wo = (ww as i64 + 2 * pad - kw as i64) / stride + 1;
    if ho <= 0 || wo <= 0 {
        return Err(Error::runtime("T2 CONV2D bad output size"));
    }
    let (ho, wo) = (ho as usize, wo as usize);
    let (xs, xe, xm, xz) = planes(x)?;
    let (ws, we, wm, wz) = planes(w)?;
    let (bs, be, bm, bz) = planes(b)?;
    let out_len = ho * wo * cout;
    let mut os = vec![1i8; out_len];
    let mut oe = vec![BIAS as i32; out_len];
    let mut om = vec![0i32; out_len];
    let mut oz = vec![1u8; out_len];
    let threads = threads.clamp(1, out_len.max(1));
    let block_len = out_len.div_ceil(threads);
    if threads == 1 {
        conv2d_block(
            xs,
            xe,
            xm,
            xz,
            ws,
            we,
            wm,
            wz,
            bs,
            be,
            bm,
            bz,
            (h, ww, cin),
            (cout, kh, kw),
            (ho, wo),
            stride,
            pad,
            0,
            tabs,
            m_bits,
            &mut os,
            &mut oe,
            &mut om,
            &mut oz,
        );
    } else {
        std::thread::scope(|scope| {
            for (block, (((os_c, oe_c), om_c), oz_c)) in os
                .chunks_mut(block_len)
                .zip(oe.chunks_mut(block_len))
                .zip(om.chunks_mut(block_len))
                .zip(oz.chunks_mut(block_len))
                .enumerate()
            {
                let oi0 = block * block_len;
                scope.spawn(move || {
                    conv2d_block(
                        xs,
                        xe,
                        xm,
                        xz,
                        ws,
                        we,
                        wm,
                        wz,
                        bs,
                        be,
                        bm,
                        bz,
                        (h, ww, cin),
                        (cout, kh, kw),
                        (ho, wo),
                        stride,
                        pad,
                        oi0,
                        tabs,
                        m_bits,
                        os_c,
                        oe_c,
                        om_c,
                        oz_c,
                    );
                });
            }
        });
    }
    from_planes(&[ho, wo, cout], os, oe, om, oz)
}

#[allow(clippy::too_many_arguments)]
fn deconv2d_block(
    xs: &[i8],
    xe: &[i32],
    xm: &[i32],
    xz: &[u8],
    ws: &[i8],
    we: &[i32],
    wm: &[i32],
    wz: &[u8],
    bs: &[i8],
    be: &[i32],
    bm: &[i32],
    bz: &[u8],
    (h, ww, cin): (usize, usize, usize),
    (cout, kh, kw): (usize, usize, usize),
    (_ho, wo): (usize, usize),
    stride: i64,
    pad: i64,
    oi0: usize,
    tabs: &Tables,
    m_bits: i64,
    os: &mut [i8],
    oe: &mut [i32],
    om: &mut [i32],
    oz: &mut [u8],
) {
    let mut terms: Vec<(T2, T2)> = Vec::with_capacity(kh * kw * cin);
    for k in 0..os.len() {
        let oi = oi0 + k;
        let co = oi % cout;
        let p = oi / cout;
        let oy = p / wo;
        let ox = p % wo;
        terms.clear();
        for ky in 0..kh {
            let n = oy as i64 + pad - ky as i64;
            if n < 0 || n % stride != 0 {
                continue;
            }
            let iy = n / stride;
            if iy >= h as i64 {
                continue;
            }
            for kx in 0..kw {
                let n2 = ox as i64 + pad - kx as i64;
                if n2 < 0 || n2 % stride != 0 {
                    continue;
                }
                let ix = n2 / stride;
                if ix >= ww as i64 {
                    continue;
                }
                for ci in 0..cin {
                    let ai = ((iy as usize) * ww + ix as usize) * cin + ci;
                    let wi = ((co * kh + ky) * kw + kx) * cin + ci;
                    terms.push((t2_at(xs, xe, xm, xz, ai), t2_at(ws, we, wm, wz, wi)));
                }
            }
        }
        let bias = t2_at(bs, be, bm, bz, co);
        let r = accumulate(
            &terms,
            if bias.z == 0 { Some(bias) } else { None },
            tabs,
            m_bits,
        );
        os[k] = r.s;
        oe[k] = r.e;
        om[k] = r.m;
        oz[k] = r.z;
    }
}

/// T2 DECONV2D on HWC, weight (Cout,kh,kw,Cin), bias (Cout).
pub fn deconv2d(
    x: &Triple,
    w: &Triple,
    b: &Triple,
    stride: i64,
    pad: i64,
    tabs: &Tables,
    m_bits: i64,
) -> Result<Triple> {
    deconv2d_mt(x, w, b, stride, pad, tabs, m_bits, 1)
}

/// Deterministic output-partitioned DECONV2D (same partition rule as CONV2D).
#[allow(clippy::too_many_arguments)]
pub fn deconv2d_mt(
    x: &Triple,
    w: &Triple,
    b: &Triple,
    stride: i64,
    pad: i64,
    tabs: &Tables,
    m_bits: i64,
    threads: usize,
) -> Result<Triple> {
    let xsh = x.shape();
    let wsh = w.shape();
    if xsh.len() != 3 || wsh.len() != 4 {
        return Err(Error::runtime("T2 DECONV2D needs HWC + (Cout,kh,kw,Cin)"));
    }
    let (h, ww, cin) = (xsh[0], xsh[1], xsh[2]);
    let (cout, kh, kw, wcin) = (wsh[0], wsh[1], wsh[2], wsh[3]);
    if wcin != cin || b.s.len() != cout || stride <= 0 || pad < 0 {
        return Err(Error::runtime("T2 DECONV2D channel/stride mismatch"));
    }
    let ho = (h as i64 - 1) * stride - 2 * pad + kh as i64;
    let wo = (ww as i64 - 1) * stride - 2 * pad + kw as i64;
    if ho <= 0 || wo <= 0 {
        return Err(Error::runtime("T2 DECONV2D bad output size"));
    }
    let (ho, wo) = (ho as usize, wo as usize);
    let (xs, xe, xm, xz) = planes(x)?;
    let (ws, we, wm, wz) = planes(w)?;
    let (bs, be, bm, bz) = planes(b)?;
    let out_len = ho * wo * cout;
    let mut os = vec![1i8; out_len];
    let mut oe = vec![BIAS as i32; out_len];
    let mut om = vec![0i32; out_len];
    let mut oz = vec![1u8; out_len];
    let threads = threads.clamp(1, out_len.max(1));
    let block_len = out_len.div_ceil(threads);
    if threads == 1 {
        deconv2d_block(
            xs,
            xe,
            xm,
            xz,
            ws,
            we,
            wm,
            wz,
            bs,
            be,
            bm,
            bz,
            (h, ww, cin),
            (cout, kh, kw),
            (ho, wo),
            stride,
            pad,
            0,
            tabs,
            m_bits,
            &mut os,
            &mut oe,
            &mut om,
            &mut oz,
        );
    } else {
        std::thread::scope(|scope| {
            for (block, (((os_c, oe_c), om_c), oz_c)) in os
                .chunks_mut(block_len)
                .zip(oe.chunks_mut(block_len))
                .zip(om.chunks_mut(block_len))
                .zip(oz.chunks_mut(block_len))
                .enumerate()
            {
                let oi0 = block * block_len;
                scope.spawn(move || {
                    deconv2d_block(
                        xs,
                        xe,
                        xm,
                        xz,
                        ws,
                        we,
                        wm,
                        wz,
                        bs,
                        be,
                        bm,
                        bz,
                        (h, ww, cin),
                        (cout, kh, kw),
                        (ho, wo),
                        stride,
                        pad,
                        oi0,
                        tabs,
                        m_bits,
                        os_c,
                        oe_c,
                        om_c,
                        oz_c,
                    );
                });
            }
        });
    }
    from_planes(&[ho, wo, cout], os, oe, om, oz)
}

const RESIZE_Q: i64 = 1 << 24;

fn resize_coord(o: usize, n: usize, no: usize, align: bool) -> (usize, usize, i64, i64) {
    if n <= 1 || no <= 1 {
        return (0, 0, RESIZE_Q, 0);
    }
    let (i0, rem, den);
    if align {
        let num = (o * (n - 1)) as i64;
        let d = (no - 1) as i64;
        i0 = (num / d) as usize;
        rem = num % d;
        den = d;
    } else {
        let num = ((2 * o + 1) * n) as i64 - no as i64;
        let d = (2 * no) as i64;
        if num <= 0 {
            return (0, (1).min(n - 1), RESIZE_Q, 0);
        }
        if num >= (n as i64 - 1) * d {
            return (n - 1, n - 1, RESIZE_Q, 0);
        }
        i0 = (num / d) as usize;
        rem = num % d;
        den = d;
    }
    let i1 = (i0 + 1).min(n - 1);
    let w1 = (rem * RESIZE_Q + den / 2) / den;
    (i0, i1, RESIZE_Q - w1, w1)
}

fn resize_weight(wy0: i64, wy1: i64, wx0: i64, wx1: i64) -> (i64, i64, i64, i64) {
    let q = RESIZE_Q as i128;
    let w00 = ((wy0 as i128 * wx0 as i128 + q / 2) / q) as i64;
    let w01 = ((wy0 as i128 * wx1 as i128 + q / 2) / q) as i64;
    let w10 = ((wy1 as i128 * wx0 as i128 + q / 2) / q) as i64;
    let w11 = RESIZE_Q - w00 - w01 - w10;
    (w00, w01, w10, w11)
}

/// T2 bilinear resize to explicit (ho,wo), with align_corners flag.
pub fn resize2d(
    x: &Triple,
    ho: usize,
    wo: usize,
    align: bool,
    tabs: &Tables,
    m_bits: i64,
) -> Result<Triple> {
    let xsh = x.shape();
    if xsh.len() != 3 || ho == 0 || wo == 0 {
        return Err(Error::runtime("T2 RESIZE2D needs HWC and positive sizes"));
    }
    let (h, w, c) = (xsh[0], xsh[1], xsh[2]);
    let (xs, xe, xm, xz) = planes(x)?;
    let mut os = vec![1i8; ho * wo * c];
    let mut oe = vec![BIAS as i32; ho * wo * c];
    let mut om = vec![0i32; ho * wo * c];
    let mut oz = vec![1u8; ho * wo * c];
    for oy in 0..ho {
        let (y0, y1, wy0, wy1) = resize_coord(oy, h, ho, align);
        for ox in 0..wo {
            let (x0, x1, wx0, wx1) = resize_coord(ox, w, wo, align);
            let (w00, w01, w10, w11) = resize_weight(wy0, wy1, wx0, wx1);
            for cc in 0..c {
                let i00 = (y0 * w + x0) * c + cc;
                let i01 = (y0 * w + x1) * c + cc;
                let i10 = (y1 * w + x0) * c + cc;
                let i11 = (y1 * w + x1) * c + cc;
                let v = [
                    t2_at(xs, xe, xm, xz, i00),
                    t2_at(xs, xe, xm, xz, i01),
                    t2_at(xs, xe, xm, xz, i10),
                    t2_at(xs, xe, xm, xz, i11),
                ];
                let mut emax: Option<i64> = None;
                for t in v {
                    if t.z == 0 {
                        emax = Some(emax.map_or(t.e as i64, |z| z.max(t.e as i64)));
                    }
                }
                let oi = (oy * wo + ox) * c + cc;
                let Some(emax) = emax else { continue };
                let q00 = to_fixed(v[0], emax, tabs, m_bits) as i128;
                let q01 = to_fixed(v[1], emax, tabs, m_bits) as i128;
                let q10 = to_fixed(v[2], emax, tabs, m_bits) as i128;
                let q11 = to_fixed(v[3], emax, tabs, m_bits) as i128;
                let acc = (q00 * w00 as i128
                    + q01 * w01 as i128
                    + q10 * w10 as i128
                    + q11 * w11 as i128
                    + (RESIZE_Q as i128) / 2)
                    / (RESIZE_Q as i128);
                let r = from_fixed(
                    acc.clamp(i64::MIN as i128, i64::MAX as i128) as i64,
                    emax,
                    tabs,
                    m_bits,
                );
                os[oi] = r.s;
                oe[oi] = r.e;
                om[oi] = r.m;
                oz[oi] = r.z;
            }
        }
    }
    from_planes(&[ho, wo, c], os, oe, om, oz)
}

/// T2 exact 2-D slice (axis 0 or 1).
pub fn slice2d(a: &Triple, axis: usize, lo: usize, hi: usize) -> Result<Triple> {
    let sh = a.shape();
    if sh.len() != 2 || axis > 1 || lo > hi || hi > sh[axis] {
        return Err(Error::runtime("T2 SLICE: bad 2-D bounds"));
    }
    let (r, c) = (sh[0], sh[1]);
    let (orows, ocols) = if axis == 0 {
        (hi - lo, c)
    } else {
        (r, hi - lo)
    };
    let (as_, ae, am, az) = planes(a)?;
    let mut s = Vec::with_capacity(orows * ocols);
    let mut e = Vec::with_capacity(orows * ocols);
    let mut m = Vec::with_capacity(orows * ocols);
    let mut z = Vec::with_capacity(orows * ocols);
    for i in 0..orows {
        for j in 0..ocols {
            let (si, sj) = if axis == 0 { (i + lo, j) } else { (i, j + lo) };
            let k = si * c + sj;
            s.push(as_[k]);
            e.push(ae[k]);
            m.push(am[k]);
            z.push(az[k]);
        }
    }
    from_planes(&[orows, ocols], s, e, m, z)
}

/// T2 exact reshape (flat move) to two or three dimensions.
pub fn reshape2(a: &Triple, d0: usize, d1: usize) -> Result<Triple> {
    if d0 * d1 != a.numel() {
        return Err(Error::runtime("T2 RESHAPE2: element count mismatch"));
    }
    let (as_, ae, am, az) = planes(a)?;
    from_planes(
        &[d0, d1],
        as_.to_vec(),
        ae.to_vec(),
        am.to_vec(),
        az.to_vec(),
    )
}

pub fn reshape3(a: &Triple, d0: usize, d1: usize, d2: usize) -> Result<Triple> {
    if d0 * d1 * d2 != a.numel() {
        return Err(Error::runtime("T2 RESHAPE3: element count mismatch"));
    }
    let (as_, ae, am, az) = planes(a)?;
    from_planes(
        &[d0, d1, d2],
        as_.to_vec(),
        ae.to_vec(),
        am.to_vec(),
        az.to_vec(),
    )
}
