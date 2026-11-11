//! The integer lattice substrate, ported from `phi_core/lattice.py`.
//!
//! A phi-value is `sign * PHI^((e - 32768) / 512)`. Runtime arithmetic is
//! integer-only; float appears at the sensor boundary (`encode`), in frozen
//! LUT construction, and in `decode` for display/scoring.

use std::f64::consts::LN_2;
use std::sync::OnceLock;

use ndarray::ArrayD;

use crate::error::{Error, Result};
use crate::triple::Triple;

#[allow(clippy::approx_constant)]
pub const PHI: f64 = 1.618_033_988_749_895;
pub const LN_PHI: f64 = 0.481_211_825_596_398_2;
pub const K: i64 = 512;
pub const BIAS: i64 = 32768;
pub const NLVL: i64 = 65536;
pub const DMAX: i64 = 4096;
pub const FRAC_CAP: i64 = 13312;
pub const FIXED_F: i64 = 18;
pub const SIG_SPAN: i64 = 16 * 16384;
pub const WIDE_CAP: i64 = 8192;
pub const BOUND: i128 = 1i128 << 62;

static L_FRAC: OnceLock<Vec<i64>> = OnceLock::new();
static L_COARSE: OnceLock<Vec<i64>> = OnceLock::new();
static L_FINE: OnceLock<Vec<i64>> = OnceLock::new();
static SIG_LUT: OnceLock<Vec<i64>> = OnceLock::new();
static SIGX_LUT: OnceLock<Vec<i64>> = OnceLock::new();
static FRAC_HI_LUT: OnceLock<Vec<i64>> = OnceLock::new();
static EXP_LUT: OnceLock<Vec<i64>> = OnceLock::new();
static PHI_LUT: OnceLock<Vec<i64>> = OnceLock::new();
static T2I_LUT: OnceLock<Vec<i64>> = OnceLock::new();

/// `np.round`: round half to even (banker's rounding).
pub fn round_half_even(x: f64) -> f64 {
    if !x.is_finite() {
        return x;
    }
    let f = x.floor();
    let diff = x - f;
    if diff > 0.5 {
        f + 1.0
    } else if diff < 0.5 {
        f
    } else if (f as i64) % 2 == 0 {
        f
    } else {
        f + 1.0
    }
}

fn round_i64(x: f64) -> i64 {
    round_half_even(x) as i64
}

/// Fraction LUT at an explicit fixed-point width. Used by opt-in
/// higher-precision NonFpu bridges; the frozen default is FIXED_F.
pub fn frac_cap_for(f: i64) -> i64 {
    if f == FIXED_F {
        return FRAC_CAP;
    }
    (((f as f64 + 1.0) * K as f64 * LN_2 / LN_PHI).ceil() as i64) + 16
}

pub fn l_frac_for(f: i64) -> Vec<i64> {
    if f == FIXED_F {
        return l_frac().to_vec();
    }
    let cap = frac_cap_for(f);
    (0..=cap)
        .map(|i| round_i64(PHI.powf(-(i as f64) / K as f64) * (1i64 << f) as f64))
        .collect()
}

pub fn frac_hi_for(f: i64) -> Vec<i64> {
    if f == FIXED_F {
        return frac_hi_lut().to_vec();
    }
    (1..=WIDE_CAP)
        .map(|k| round_i64(PHI.powf(k as f64 / K as f64) * (1i64 << f) as f64))
        .collect()
}

pub fn l_frac() -> &'static [i64] {
    L_FRAC.get_or_init(|| {
        (0..=FRAC_CAP)
            .map(|i| round_i64(PHI.powf(-(i as f64) / K as f64) * (1i64 << FIXED_F) as f64))
            .collect()
    })
}

pub fn l_coarse() -> &'static [i64] {
    L_COARSE.get_or_init(|| {
        (-64..=128)
            .map(|i| round_i64(i as f64 * K as f64 * LN_2 / LN_PHI))
            .collect()
    })
}

pub fn l_fine() -> &'static [i64] {
    L_FINE.get_or_init(|| {
        (0..16384)
            .map(|i| round_i64(K as f64 * ((16384 + i) as f64 / 16384.0).ln() / LN_PHI))
            .collect()
    })
}

pub fn sig_lut() -> &'static [i64] {
    SIG_LUT.get_or_init(|| {
        let n = 2 * 16 * 16384;
        (0..=n)
            .map(|k| {
                let x = (k as f64 - 16.0 * 16384.0) / 16384.0;
                round_i64((1.0 / (1.0 + (-x).exp())) * 16384.0)
            })
            .collect()
    })
}

pub fn sigx_lut() -> &'static [i64] {
    SIGX_LUT.get_or_init(|| {
        (0..65536)
            .map(|i| round_i64(PHI.powf((i as f64 - BIAS as f64) / K as f64) * 16384.0))
            .collect()
    })
}

pub fn frac_hi_lut() -> &'static [i64] {
    FRAC_HI_LUT.get_or_init(|| {
        (1..=WIDE_CAP)
            .map(|k| round_i64(PHI.powf(k as f64 / K as f64) * (1i64 << 18) as f64))
            .collect()
    })
}

pub fn exp_lut() -> &'static [i64] {
    EXP_LUT.get_or_init(|| {
        (0..=262144)
            .map(|d| round_i64((2f64.powi(24)) * (-(d as f64) / 16384.0).exp()))
            .collect()
    })
}

/// Exact decoded magnitudes for `T2I`, rounded half-away-from-zero.
///
/// The triple exponent is the whole `0..NLVL` lattice range. This table is the
/// non-FPU bridge: it is computed with the same `f64` expression as the
/// interpreter, then baked as integers. The sign is applied by the caller.
pub fn t2i_lut() -> &'static [i64] {
    T2I_LUT.get_or_init(|| {
        (0..NLVL)
            .map(|e| {
                let v = PHI.powf((e as f64 - BIAS as f64) / K as f64);
                v.round() as i64
            })
            .collect()
    })
}

pub fn phi_lut() -> &'static [i64] {
    PHI_LUT.get_or_init(|| {
        let n = 2 * SIG_SPAN;
        (0..=n)
            .map(|k| {
                let x = (k as f64 - SIG_SPAN as f64) / 16384.0;
                let phi = 0.5 * (1.0 + libm::erf(x / std::f64::consts::SQRT_2));
                round_i64(phi * 16384.0)
            })
            .collect()
    })
}

fn clamp_i64(v: i64, lo: i64, hi: i64) -> i64 {
    v.max(lo).min(hi)
}

/// Build an `ArrayD` from a shape and flat C-order data.
pub fn array_from_vec<T: Clone>(shape: &[usize], data: Vec<T>) -> Result<ArrayD<T>> {
    let expected: usize = shape.iter().product();
    if expected != data.len() {
        return Err(Error::runtime(format!(
            "shape {:?} needs {expected} elements, got {}",
            shape,
            data.len()
        )));
    }
    ArrayD::from_shape_vec(ndarray::IxDyn(shape), data)
        .map_err(|e| Error::runtime(format!("array shape {:?}: {e}", shape)))
}

/// Sensor boundary: floats -> triples. Mirrors `lattice.encode`.
pub fn encode(a: &ArrayD<f64>) -> Triple {
    let mut s = Vec::with_capacity(a.len());
    let mut e = Vec::with_capacity(a.len());
    let mut z = Vec::with_capacity(a.len());
    for &v in a.iter() {
        let sign = if v < 0.0 { -1i8 } else { 1i8 };
        let raw = (K as f64) * ((v.abs() + 1e-15).ln()) / LN_PHI;
        let ev = clamp_i64(round_i64(raw) + BIAS, 0, NLVL - 1);
        s.push(sign);
        e.push(ev as i32);
        z.push(0u8);
    }
    Triple::from_vecs(a.shape(), s, e, z).expect("encode shape")
}

/// Display/scoring boundary: triples -> floats.
pub fn decode(s: &ArrayD<i8>, e: &ArrayD<i32>) -> ArrayD<f64> {
    ndarray::Zip::from(s)
        .and(e)
        .map_collect(|&sv, &ev| (sv as f64) * PHI.powf((ev as f64 - BIAS as f64) / K as f64))
}

/// Fixed-point bridge: triples at scale `m` -> int64 in 2^-18 units.
pub fn to_fixed(t: &Triple, m: i64) -> ArrayD<i64> {
    let f = l_frac();
    let base = f[0];
    let mut out = Vec::with_capacity(t.s.len());
    for ((&sv, &ev), &zv) in t.s.iter().zip(t.e.iter()).zip(t.z.iter()) {
        let d = m - ev as i64;
        let q = if zv != 0 {
            0
        } else if d < 0 {
            sv as i64 * base
        } else if d > FRAC_CAP {
            0
        } else {
            sv as i64 * f[d as usize]
        };
        out.push(q);
    }
    array_from_vec(t.s.shape(), out).expect("to_fixed shape")
}

/// Fixed-point bridge with the wide above-range branch served exactly.
pub fn to_fixed_wide(t: &Triple, m: i64) -> ArrayD<i64> {
    let f = l_frac();
    let base = f[0];
    let g = frac_hi_lut();
    let mut out = Vec::with_capacity(t.s.len());
    for ((&sv, &ev), &zv) in t.s.iter().zip(t.e.iter()).zip(t.z.iter()) {
        let d = m - ev as i64;
        let q_lo = if d < -WIDE_CAP {
            sv as i64 * base
        } else {
            let idx = clamp_i64(-d, 1, WIDE_CAP) - 1;
            sv as i64 * g[idx as usize]
        };
        let q = if zv != 0 {
            0
        } else if d < 0 {
            q_lo
        } else if d > FRAC_CAP {
            0
        } else {
            sv as i64 * f[d as usize]
        };
        out.push(q);
    }
    array_from_vec(t.s.shape(), out).expect("to_fixed_wide shape")
}

fn bit_length_int(mut x: i64) -> i64 {
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

/// Fixed-point -> triples, with the same overflow assertions as Python.
pub fn from_fixed(q: &ArrayD<i64>, m: i64) -> Result<Triple> {
    let c = l_coarse();
    let f = l_fine();
    let mut ss = Vec::with_capacity(q.len());
    let mut ee = Vec::with_capacity(q.len());
    let mut zz = Vec::with_capacity(q.len());
    for &qv in q.iter() {
        let z = qv == 0;
        let a = (qv as i128).unsigned_abs() as i128;
        if a >= (1i128 << 53) {
            return Err(Error::runtime(
                "from_fixed: accumulated value exceeds frexp-exact range (2^53)",
            ));
        }
        let a = a as i64;
        let bl = bit_length_int(a);
        let shift = bl - 15;
        let mant = if shift >= 0 {
            a >> shift
        } else {
            a << (-shift)
        };
        let t = shift + 14 - FIXED_F;
        let idx = t + 64;
        if !z && !(0..=192).contains(&idx) {
            return Err(Error::runtime(format!(
                "COARSE index OOB (t+64 range violated): idx={idx}"
            )));
        }
        let ev =
            m + c[clamp_i64(idx, 0, 192) as usize] + f[clamp_i64(mant - 16384, 0, 16383) as usize];
        let ev = clamp_i64(ev, 0, NLVL - 1) as i32;
        ss.push(if qv > 0 { 1 } else { -1 });
        ee.push(ev);
        zz.push(z as u8);
    }
    Triple::from_vecs(q.shape(), ss, ee, zz)
}

/// Truncating division (toward zero), both signs. Numpy's `floor_divide`
/// after `abs()` is exactly this.
pub fn tdiv(a: i64, b: i64) -> i64 {
    if b == 0 {
        return 0;
    }
    let q = (a as i128).abs() / (b as i128).abs();
    let q = if (a < 0) ^ (b < 0) { -q } else { q };
    q as i64
}

pub fn tdiv_array(a: &ArrayD<i64>, b: &ArrayD<i64>) -> Result<ArrayD<i64>> {
    if a.shape() != b.shape() {
        // Numpy broadcasts; the call sites in this crate use equal shapes.
        return Err(Error::runtime(format!(
            "tdiv shape mismatch: {:?} vs {:?}",
            a.shape(),
            b.shape()
        )));
    }
    Ok(ndarray::Zip::from(a)
        .and(b)
        .map_collect(|&x, &y| tdiv(x, y)))
}

/// Integer bit length (floor(log2)+1, 0 for 0), matching lattice.py.
pub fn bit_length_int_array(a: &ArrayD<i64>) -> ArrayD<i64> {
    a.mapv(bit_length_int)
}

pub fn decode_triple(t: &Triple) -> ArrayD<f64> {
    decode(&t.s, &t.e)
}

/// Rescale a fixed array from `m1` to `m2` through triples.
pub fn rescale(q: &ArrayD<i64>, m1: i64, m2: i64) -> Result<ArrayD<i64>> {
    if m1 == m2 {
        return Ok(q.clone());
    }
    let mid = from_fixed(q, m1)?;
    Ok(to_fixed(&mid, m2))
}

/// `_assert_bound`: every non-zero magnitude must stay below 2^62.
pub fn assert_bound(tag: &str, arrays: &[&ArrayD<i64>]) -> Result<()> {
    for a in arrays {
        let mut mx: i128 = 0;
        for &v in a.iter() {
            let m = (v as i128).abs();
            if m > mx {
                mx = m;
            }
        }
        if !a.is_empty() && mx >= BOUND {
            return Err(Error::runtime(format!("{tag}: intermediate {mx} >= 2^62")));
        }
    }
    Ok(())
}
