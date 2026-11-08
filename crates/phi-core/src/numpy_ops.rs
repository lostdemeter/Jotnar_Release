//! Numpy-style substrate ops (port of the `phi_core/numpy_ops.py` surface).
//!
//! Signatures operate on [`Triple`] rather than raw numpy planes. The
//! semantics (including 2^-18 fixed units, BIAS counts, and LUT rounding)
//! match the Python reference.

use ndarray::{ArrayD, IxDyn};

use crate::error::{Error, Result};
use crate::lattice::{self, BIAS, NLVL};
use crate::triple::Triple;

fn flat<T: Copy>(a: &ArrayD<T>) -> Vec<T> {
    a.iter().copied().collect()
}

fn from_flat<T: Clone>(shape: &[usize], v: Vec<T>) -> Result<ArrayD<T>> {
    lattice::array_from_vec(shape, v)
}

fn last_dim(shape: &[usize]) -> usize {
    *shape.last().unwrap_or(&1)
}

fn num_rows(shape: &[usize]) -> usize {
    if shape.is_empty() {
        1
    } else {
        shape[..shape.len() - 1].iter().product()
    }
}

fn isqrt_u128(n: u128) -> u128 {
    if n < 2 {
        return n;
    }
    let mut lo = 1u128;
    let mut hi = n;
    while lo < hi {
        let mid = lo + (hi - lo + 1) / 2;
        if mid <= n / mid {
            lo = mid;
        } else {
            hi = mid - 1;
        }
    }
    lo
}

fn int_sum(v: &[i64]) -> i64 {
    let mut acc: i128 = 0;
    for &x in v {
        acc = acc.wrapping_add(x as i128);
    }
    acc as i64
}

/// Exact triple multiply (sign XOR + exponent add).
pub fn tmul(a: &Triple, b: &Triple) -> Result<Triple> {
    if a.shape() != b.shape() {
        return Err(Error::runtime("tmul shape mismatch"));
    }
    let mut s = Vec::with_capacity(a.numel());
    let mut e = Vec::with_capacity(a.numel());
    let mut z = Vec::with_capacity(a.numel());
    for (((&sa, &ea), &za), ((&sb, &eb), &zb)) in
        a.s.iter()
            .zip(a.e.iter())
            .zip(a.z.iter())
            .zip(b.s.iter().zip(b.e.iter()).zip(b.z.iter()))
    {
        s.push(((sa as i16) * (sb as i16)) as i8);
        e.push((ea as i64 + eb as i64 - BIAS).clamp(0, NLVL - 1) as i32);
        z.push(za | zb);
    }
    Triple::from_vecs(&a.shape(), s, e, z)
}

/// Constant triple of the given shape.
pub fn const_triple(v: f64, shape: &[usize]) -> Triple {
    let one = lattice::encode(&ArrayD::from_elem(IxDyn(&[1]), v));
    let s = one.s.iter().next().copied().unwrap_or(1);
    let e = one.e.iter().next().copied().unwrap_or(0);
    let z = one.z.iter().next().copied().unwrap_or(0);
    Triple {
        s: ArrayD::from_elem(IxDyn(shape), s),
        e: ArrayD::from_elem(IxDyn(shape), e),
        z: ArrayD::from_elem(IxDyn(shape), z),
    }
}

/// Sigmoid via EXPACT + frozen LUT (any range).
pub fn sigmoid_int(t: &Triple) -> Result<Triple> {
    let xlut = lattice::sigx_lut();
    let glut = lattice::sig_lut();
    let span = lattice::SIG_SPAN;
    let mut out = Vec::with_capacity(t.numel());
    for ((&sv, &ev), &zv) in t.s.iter().zip(t.e.iter()).zip(t.z.iter()) {
        let e = (ev as i64).clamp(0, 65535);
        let x14 = if zv != 0 {
            0
        } else {
            sv as i64 * xlut[e as usize]
        };
        let y14 = if x14 < -span {
            0
        } else if x14 > span {
            16384
        } else {
            glut[(x14 + span).clamp(0, glut.len() as i64 - 1) as usize]
        };
        out.push(y14 * 16);
    }
    let q = from_flat(&t.shape(), out)?;
    lattice::from_fixed(&q, BIAS)
}

/// SiLU = x * sigmoid(x).
pub fn silu_int(t: &Triple) -> Result<Triple> {
    tmul(&sigmoid_int(t)?, t)
}

/// PReLU: x where x >= 0, x*slope otherwise (last-axis/scalar slope).
pub fn prelu_int(t: &Triple, slope: &Triple) -> Result<Triple> {
    let shape = t.shape();
    let c = *shape.last().unwrap_or(&1);
    let n = t.numel();
    let slope_full = if slope.numel() == 1 {
        Triple {
            s: ArrayD::from_elem(IxDyn(&shape), *slope.s.iter().next().unwrap_or(&1)),
            e: ArrayD::from_elem(IxDyn(&shape), *slope.e.iter().next().unwrap_or(&0)),
            z: ArrayD::from_elem(IxDyn(&shape), *slope.z.iter().next().unwrap_or(&0)),
        }
    } else if slope.numel() == c {
        let ss = flat(&slope.s);
        let se = flat(&slope.e);
        let sz = flat(&slope.z);
        Triple::from_vecs(
            &shape,
            (0..n).map(|i| ss[i % c]).collect(),
            (0..n).map(|i| se[i % c]).collect(),
            (0..n).map(|i| sz[i % c]).collect(),
        )?
    } else {
        return Err(Error::runtime(format!(
            "prelu slope shape {:?} not broadcastable to {shape:?}",
            slope.shape()
        )));
    };
    let prod = tmul(t, &slope_full)?;
    let ts = flat(&t.s);
    let te = flat(&t.e);
    let tz = flat(&t.z);
    let ps = flat(&prod.s);
    let pe = flat(&prod.e);
    let pz = flat(&prod.z);
    let mut s = Vec::with_capacity(n);
    let mut e = Vec::with_capacity(n);
    let mut z = Vec::with_capacity(n);
    for i in 0..n {
        let neg = ts[i] < 0 && tz[i] == 0;
        if neg {
            s.push(ps[i]);
            e.push(pe[i]);
            z.push(pz[i]);
        } else {
            s.push(ts[i]);
            e.push(te[i]);
            z.push(tz[i]);
        }
    }
    Triple::from_vecs(&shape, s, e, z)
}

/// Exact row gather `table[ids]`.
pub fn gather_int(w: &Triple, ids: &ArrayD<i64>) -> Result<Triple> {
    let sh = w.shape();
    if sh.len() != 2 {
        return Err(Error::runtime("gather_int table must be (V,C)"));
    }
    let (v, c) = (sh[0], sh[1]);
    let n = ids.len();
    let mut osh = ids.shape().to_vec();
    osh.push(c);
    let wf_s = flat(&w.s);
    let wf_e = flat(&w.e);
    let wf_z = flat(&w.z);
    let mut s = Vec::with_capacity(n * c);
    let mut e = Vec::with_capacity(n * c);
    let mut z = Vec::with_capacity(n * c);
    for &id in ids.iter() {
        if id < 0 || id as usize >= v {
            return Err(Error::runtime(format!("gather ids out of range [0,{v})")));
        }
        let r = id as usize;
        for j in 0..c {
            s.push(wf_s[r * c + j]);
            e.push(wf_e[r * c + j]);
            z.push(wf_z[r * c + j]);
        }
    }
    Triple::from_vecs(&osh, s, e, z)
}

/// Fixed-point -> triples through the lattice bridge.
pub fn rescale_via_triples(q: &ArrayD<i64>, m_from: i64, m_to: i64) -> Result<ArrayD<i64>> {
    lattice::rescale(q, m_from, m_to)
}

/// Batched triples matmul: A (...,N,K) x B (...,K,M) -> (...,N,M).
pub fn matmul_int(a: &Triple, b: &Triple, m_acc: i64) -> Result<Triple> {
    let ash = a.shape();
    let bsh = b.shape();
    if ash.len() < 2 || bsh.len() < 2 {
        return Err(Error::runtime("matmul_int needs rank >= 2"));
    }
    let n = ash[ash.len() - 2];
    let k = ash[ash.len() - 1];
    if bsh[bsh.len() - 2] != k {
        return Err(Error::runtime(format!(
            "matmul inner dims {ash:?} vs {bsh:?}"
        )));
    }
    let m = *bsh.last().unwrap();
    let lead = &ash[..ash.len() - 2];
    let nb: usize = lead.iter().product();
    let nb_b: usize = bsh[..bsh.len() - 2].iter().product();
    if nb_b != 1 && nb_b != nb {
        return Err(Error::runtime("matmul batch mismatch"));
    }
    let af = flat(&a.s);
    let aef = flat(&a.e);
    let azf = flat(&a.z);
    let bf = flat(&b.s);
    let bef = flat(&b.e);
    let bzf = flat(&b.z);
    let f = lattice::l_frac();
    let tofixed = |s: i8, e: i32, z: u8| -> i64 {
        if z != 0 {
            return 0;
        }
        let d = m_acc - e as i64;
        if d < 0 {
            s as i64 * f[0]
        } else if d > lattice::FRAC_CAP {
            0
        } else {
            s as i64 * f[d as usize]
        }
    };
    let mut os = vec![0i8; nb * n * m];
    let mut oe = vec![0i32; nb * n * m];
    let mut oz = vec![0u8; nb * n * m];
    for bi in 0..nb {
        let bb = if nb_b == 1 { 0 } else { bi };
        let mut acc = vec![0i64; n * m];
        for i in 0..n {
            for kk in 0..k {
                let ai = bi * n * k + i * k + kk;
                for j in 0..m {
                    let bj = bb * k * m + kk * m + j;
                    let ps = ((af[ai] as i16) * (bf[bj] as i16)) as i8;
                    let pe = (aef[ai] as i64 + bef[bj] as i64 - BIAS).clamp(0, NLVL - 1) as i32;
                    acc[i * m + j] =
                        acc[i * m + j].wrapping_add(tofixed(ps, pe, azf[ai] | bzf[bj]));
                }
            }
        }
        let arr = from_flat(&[n, m], acc)?;
        lattice::assert_bound("matmul:acc", &[&arr])?;
        let got = lattice::from_fixed(&arr, m_acc)?;
        let gs = flat(&got.s);
        let ge = flat(&got.e);
        let gz = flat(&got.z);
        for t in 0..n * m {
            let o = bi * n * m + t;
            os[o] = gs[t];
            oe[o] = ge[t];
            oz[o] = gz[t];
        }
    }
    let mut osh = lead.to_vec();
    osh.extend_from_slice(&[n, m]);
    Triple::from_vecs(&osh, os, oe, oz)
}

fn softmax_n_fixed(q: &ArrayD<i64>, m: i64) -> Result<(Vec<i64>, Vec<i64>, Vec<usize>)> {
    if m != BIAS {
        return Err(Error::runtime(format!(
            "softmax needs BIAS counts, got {m}"
        )));
    }
    let sh = q.shape().to_vec();
    let c = last_dim(&sh);
    let rows = num_rows(&sh);
    let qf = flat(q);
    let exp = lattice::exp_lut();
    let mut num = vec![0i64; qf.len()];
    let mut den = vec![0i64; rows];
    for r in 0..rows {
        let s14: Vec<i64> = qf[r * c..(r + 1) * c]
            .iter()
            .map(|&v| lattice::tdiv(v, 16))
            .collect();
        let vmax = *s14.iter().max().unwrap_or(&0);
        let mut dsum: i128 = 0;
        for j in 0..c {
            let dd = (vmax - s14[j]).clamp(0, 262144) as usize;
            let e = exp[dd];
            num[r * c + j] = e;
            dsum = dsum.wrapping_add(e as i128);
        }
        den[r] = if dsum == 0 { 1 } else { dsum as i64 };
    }
    Ok((num, den, sh))
}

/// N-way row softmax from fixed scores @ BIAS.
pub fn softmaxN_fixed(q: &ArrayD<i64>, m: i64) -> Result<(ArrayD<i64>, ArrayD<i64>)> {
    let (num, den, sh) = softmax_n_fixed(q, m)?;
    let c = last_dim(&sh);
    let mut num2 = vec![0i64; num.len()];
    for (r, &_d) in den.iter().enumerate() {
        for j in 0..c {
            num2[r * c + j] = num[r * c + j];
        }
    }
    Ok((from_flat(&sh, num2)?, from_flat(&[den.len()], den)?))
}

/// N-way row softmax from triples: bridge at BIAS, then LUT.
/// Returns `(num, den)` like the Python reference.
pub fn softmaxN_triples(t: &Triple) -> Result<(ArrayD<i64>, ArrayD<i64>)> {
    let q = lattice::to_fixed(t, BIAS);
    softmaxN_fixed(&q, BIAS)
}

/// Per-row RMSNorm + weight.
pub fn rmsnorm_int(x: &Triple, w: &Triple, m: i64, eps_c: i64) -> Result<Triple> {
    let sh = x.shape();
    let c = last_dim(&sh);
    if w.numel() != c {
        return Err(Error::runtime("rmsnorm weight length mismatch"));
    }
    let q = lattice::to_fixed(x, m);
    let wq = lattice::to_fixed(w, m);
    let qf = flat(&q);
    let wf = flat(&wq);
    let rows = num_rows(&sh);
    let mut out = vec![0i64; qf.len()];
    for r in 0..rows {
        let row = &qf[r * c..(r + 1) * c];
        let sq: Vec<i64> = row.iter().map(|&v| v.wrapping_mul(v)).collect();
        let ms = lattice::tdiv(int_sum(&sq), c as i64).wrapping_add(eps_c);
        let rms = if ms > 0 {
            isqrt_u128(ms as u128) as i64
        } else {
            0
        };
        for j in 0..c {
            let norm = if rms == 0 {
                0
            } else {
                lattice::tdiv(row[j].wrapping_mul(1 << 18), rms)
            };
            out[r * c + j] = lattice::tdiv(norm.wrapping_mul(wf[j]), 1 << 18);
        }
    }
    let arr = from_flat(&sh, out)?;
    lattice::assert_bound("rms:affine", &[&arr])?;
    lattice::from_fixed(&arr, m)
}

/// Per-row LayerNorm + weight + bias.
pub fn int_layernorm_rows(
    x: &Triple,
    w: &Triple,
    b: &Triple,
    m: i64,
    eps_c: i64,
) -> Result<Triple> {
    let sh = x.shape();
    if sh.len() != 2 {
        return Err(Error::runtime("layernorm needs 2D (N,C)"));
    }
    let (nn, c) = (sh[0], sh[1]);
    let q = lattice::to_fixed(x, m);
    let wq = lattice::to_fixed(w, m);
    let bq = lattice::to_fixed(b, m);
    let qf = flat(&q);
    let wf = flat(&wq);
    let bf = flat(&bq);
    let mut out = vec![0i64; qf.len()];
    for r in 0..nn {
        let row = &qf[r * c..(r + 1) * c];
        let mean = lattice::tdiv(int_sum(row), c as i64);
        let dev: Vec<i64> = row.iter().map(|&v| v.wrapping_sub(mean)).collect();
        let sqsum = int_sum(&dev.iter().map(|&v| v.wrapping_mul(v)).collect::<Vec<_>>());
        let var = lattice::tdiv(sqsum, c as i64).wrapping_add(eps_c);
        let std = if var > 0 {
            isqrt_u128(var as u128) as i64
        } else {
            0
        };
        for j in 0..c {
            let norm = if std == 0 {
                0
            } else {
                lattice::tdiv(dev[j].wrapping_mul(1 << 18), std)
            };
            out[r * c + j] = lattice::tdiv(norm.wrapping_mul(wf[j]), 1 << 18).wrapping_add(bf[j]);
        }
    }
    let arr = from_flat(&sh, out)?;
    lattice::assert_bound("ln:affine", &[&arr])?;
    lattice::from_fixed(&arr, m)
}

/// GELU exact-form via EXPACT + PHI LUT.
pub fn gelu_erf_int(t: &Triple) -> Result<Triple> {
    let xlut = lattice::sigx_lut();
    let glut = lattice::phi_lut();
    let span = lattice::SIG_SPAN;
    let mut out = Vec::with_capacity(t.numel());
    for ((&sv, &ev), &zv) in t.s.iter().zip(t.e.iter()).zip(t.z.iter()) {
        let e = (ev as i64).clamp(0, 65535);
        let x14 = if zv != 0 {
            0
        } else {
            sv as i64 * xlut[e as usize]
        };
        let p14 = if x14 < -span {
            0
        } else if x14 > span {
            16384
        } else {
            glut[(x14 + span).clamp(0, glut.len() as i64 - 1) as usize]
        };
        let y14 = lattice::tdiv((x14 as i128 * p14 as i128) as i64, 16384);
        out.push(y14 * 16);
    }
    let arr = from_flat(&t.shape(), out)?;
    lattice::from_fixed(&arr, BIAS)
}

/// One SSM scan step: h' = Abar*h + Bx.
pub fn scan_step_int(
    h_q: &ArrayD<i64>,
    abar_q: &ArrayD<i64>,
    bx_q: &ArrayD<i64>,
) -> Result<ArrayD<i64>> {
    if h_q.shape() != abar_q.shape() || h_q.shape() != bx_q.shape() {
        return Err(Error::runtime("scan shape mismatch"));
    }
    let mut out = Vec::with_capacity(h_q.len());
    for ((&h, &a), &b) in h_q.iter().zip(abar_q.iter()).zip(bx_q.iter()) {
        out.push(lattice::tdiv((a as i128 * h as i128) as i64, 1 << 18).wrapping_add(b));
    }
    let arr = from_flat(&h_q.shape(), out)?;
    lattice::assert_bound("scan:out", &[&arr])?;
    Ok(arr)
}

/// Global average pool over HWC -> (C,).
pub fn avgpool_int(t: &Triple, m: i64) -> Result<Triple> {
    let sh = t.shape();
    if sh.len() != 3 {
        return Err(Error::runtime("avgpool needs HWC"));
    }
    let (h, w, c) = (sh[0], sh[1], sh[2]);
    let q = lattice::to_fixed(t, m);
    let qf = flat(&q);
    let mut mean = vec![0i64; c];
    for ch in 0..c {
        let mut tot: i128 = 0;
        for i in 0..h * w {
            tot = tot.wrapping_add(qf[i * c + ch] as i128);
        }
        mean[ch] = lattice::tdiv(tot as i64, (h * w) as i64);
    }
    let arr = from_flat(&[c], mean)?;
    lattice::from_fixed(&arr, m)
}

pub use lattice::{exp_lut, frac_hi_lut, phi_lut, to_fixed_wide};

fn fixed_scalar(s: i8, e: i32, z: u8, m: i64) -> i64 {
    if z != 0 {
        return 0;
    }
    let d = m - e as i64;
    let f = lattice::l_frac();
    if d < 0 {
        s as i64 * f[0]
    } else if d > lattice::FRAC_CAP {
        0
    } else {
        s as i64 * f[d as usize]
    }
}

/// Integer bilinear sample; `x`/`y` are 2^-14 pixel coords, same shape.
pub fn bilinear_sample_fixed(
    f: &ArrayD<i64>,
    x: &ArrayD<i64>,
    y: &ArrayD<i64>,
) -> Result<ArrayD<i64>> {
    let sh = f.shape();
    if sh.len() != 3 {
        return Err(Error::runtime("bilinear_sample_fixed needs (H,W,C)"));
    }
    if x.shape() != y.shape() {
        return Err(Error::runtime("bilinear_sample_fixed coord shape mismatch"));
    }
    let (h, w, c) = (sh[0], sh[1], sh[2]);
    let one = 1i64 << 14;
    let ff = flat(f);
    let xo = x.shape().to_vec();
    let n: usize = xo.iter().product();
    let mut out = vec![0i64; n * c];
    let xf = flat(x);
    let yf = flat(y);
    for idx in 0..n {
        let xc = xf[idx].clamp(0, ((w as i64) - 1) << 14);
        let yc = yf[idx].clamp(0, ((h as i64) - 1) << 14);
        let xi = xc >> 14;
        let yi = yc >> 14;
        let xf_ = xc & (one - 1);
        let yf_ = yc & (one - 1);
        let xi1 = (xi + 1).min(w as i64 - 1);
        let yi1 = (yi + 1).min(h as i64 - 1);
        for ch in 0..c {
            let g = |yy: i64, xx: i64| ff[(yy as usize * w + xx as usize) * c + ch];
            let tot = g(yi, xi)
                .wrapping_mul((one - xf_).wrapping_mul(one - yf_))
                .wrapping_add(g(yi, xi1).wrapping_mul(xf_.wrapping_mul(one - yf_)))
                .wrapping_add(g(yi1, xi).wrapping_mul((one - xf_).wrapping_mul(yf_)))
                .wrapping_add(g(yi1, xi1).wrapping_mul(xf_.wrapping_mul(yf_)));
            out[idx * c + ch] = tot >> 28;
        }
    }
    let mut osh = xo;
    osh.push(c);
    from_flat(&osh, out)
}

/// Warp fixed field by fixed flow (`flow` is (H,W,2) in 2^-14 units).
pub fn warp_fixed(feat_q: &ArrayD<i64>, flow_q: &ArrayD<i64>) -> Result<ArrayD<i64>> {
    let sh = feat_q.shape();
    if sh.len() != 3 {
        return Err(Error::runtime("warp_fixed needs (H,W,C)"));
    }
    let (h, w, _) = (sh[0], sh[1], sh[2]);
    if flow_q.shape() != [h, w, 2] {
        return Err(Error::runtime("warp_fixed flow must be (H,W,2)"));
    }
    let ff = flat(flow_q);
    let mut xs = vec![0i64; h * w];
    let mut ys = vec![0i64; h * w];
    for yy in 0..h {
        for xx in 0..w {
            let i = (yy * w + xx) * 2;
            xs[yy * w + xx] = ((xx as i64) << 14) + ff[i];
            ys[yy * w + xx] = ((yy as i64) << 14) + ff[i + 1];
        }
    }
    let xa = from_flat(&[h, w], xs)?;
    let ya = from_flat(&[h, w], ys)?;
    bilinear_sample_fixed(feat_q, &xa, &ya)
}

/// Warp triples by fixed flow.
pub fn warp_triples_int(t: &Triple, flow_q: &ArrayD<i64>, m: i64) -> Result<Triple> {
    let q = lattice::to_fixed(t, m);
    let o = warp_fixed(&q, flow_q)?;
    lattice::from_fixed(&o, m)
}

fn interp_coord(n_out: usize, s: f64) -> Vec<i64> {
    let mut out = Vec::with_capacity(n_out);
    for o in 0..n_out {
        let o = o as i64;
        let mut v = None;
        for k in 1..14 {
            if s == 2f64.powi(-k) {
                v = Some(((2 * o + 1) << (13 + k)) - 8192);
                break;
            }
            if s == 2f64.powi(k) {
                v = Some(((2 * o + 1) - (1 << k)) << (13 - k));
                break;
            }
        }
        let v = v.unwrap_or_else(|| {
            if s == 1.0 {
                o << 14
            } else {
                let num = (2 * o + 1) << 14;
                let den = lattice::round_half_even(2.0 * s * 16384.0) as i64;
                let q = num.abs() / den;
                (if num < 0 { -q } else { q }) - 8192
            }
        });
        out.push(v);
    }
    out
}

/// Bilinear resample of a fixed field (dyadic scales exact).
pub fn interp_fixed(feat_q: &ArrayD<i64>, sy: f64, sx: f64) -> Result<ArrayD<i64>> {
    let sh = feat_q.shape();
    if sh.len() != 3 {
        return Err(Error::runtime("interp_fixed needs (H,W,C)"));
    }
    let (h, w, _) = (sh[0], sh[1], sh[2]);
    let ho = ((h as f64 * sy).round().max(1.0)) as usize;
    let wo = ((w as f64 * sx).round().max(1.0)) as usize;
    let xs = from_flat(&[wo], interp_coord(wo, sx))?;
    let ys = from_flat(&[ho], interp_coord(ho, sy))?;
    let mut xg = vec![0i64; ho * wo];
    let mut yg = vec![0i64; ho * wo];
    for y in 0..ho {
        for x in 0..wo {
            xg[y * wo + x] = xs[x];
            yg[y * wo + x] = ys[y];
        }
    }
    let xa = from_flat(&[ho, wo], xg)?;
    let ya = from_flat(&[ho, wo], yg)?;
    bilinear_sample_fixed(feat_q, &xa, &ya)
}

/// Nearest x2 reindex (exact); repeats axes 0 and 1, preserving trailing dims.
pub fn nearest2(t: &Triple) -> Result<Triple> {
    let sh = t.shape();
    if sh.len() < 2 {
        return Err(Error::runtime("nearest2 needs rank >= 2"));
    }
    let (h, w) = (sh[0], sh[1]);
    let inner: usize = sh[2..].iter().product();
    let ss = flat(&t.s);
    let se = flat(&t.e);
    let sz = flat(&t.z);
    let mut s = Vec::with_capacity(ss.len() * 4);
    let mut e = Vec::with_capacity(se.len() * 4);
    let mut z = Vec::with_capacity(sz.len() * 4);
    for y in 0..2 * h {
        for x in 0..2 * w {
            let src = ((y / 2) * w + (x / 2)) * inner;
            for k in 0..inner {
                s.push(ss[src + k]);
                e.push(se[src + k]);
                z.push(sz[src + k]);
            }
        }
    }
    let mut osh = sh.clone();
    osh[0] *= 2;
    osh[1] *= 2;
    Triple::from_vecs(&osh, s, e, z)
}

/// Transpose-conv via scatter-add (optional bias).
pub fn deconv_int(
    x: &Triple,
    w: &Triple,
    m_out: i64,
    stride: i64,
    pad: i64,
    bias: Option<&Triple>,
) -> Result<Triple> {
    let sh = x.shape();
    let wsh = w.shape();
    if sh.len() != 3 || wsh.len() != 4 {
        return Err(Error::runtime("deconv needs (H,W,Cin) and (O,kh,kw,Cin)"));
    }
    let (h, ww, cin) = (sh[0], sh[1], sh[2]);
    let (cout, kh, kw, wcin) = (wsh[0], wsh[1], wsh[2], wsh[3]);
    if wcin != cin {
        return Err(Error::runtime("deconv weight Cin mismatch"));
    }
    let ho = (h as i64 - 1) * stride - 2 * pad + kh as i64;
    let wo = (ww as i64 - 1) * stride - 2 * pad + kw as i64;
    if ho <= 0 || wo <= 0 {
        return Err(Error::runtime("deconv bad output size"));
    }
    let (ho, wo) = (ho as usize, wo as usize);
    let sf = flat(&x.s);
    let ef = flat(&x.e);
    let zf = flat(&x.z);
    let wsf = flat(&w.s);
    let wef = flat(&w.e);
    let wzf = flat(&w.z);
    let mut acc = vec![0i64; ho * wo * cout];
    for co in 0..cout {
        let mut plane = vec![0i64; ho * wo];
        for ky in 0..kh {
            for kx in 0..kw {
                for iy in 0..h {
                    let oy = iy as i64 * stride - pad + ky as i64;
                    if oy < 0 || oy >= ho as i64 {
                        continue;
                    }
                    for ix in 0..ww {
                        let ox = ix as i64 * stride - pad + kx as i64;
                        if ox < 0 || ox >= wo as i64 {
                            continue;
                        }
                        for ci in 0..cin {
                            let wi = ((co * kh + ky) * kw + kx) * cin + ci;
                            let ai = (iy * ww + ix) * cin + ci;
                            let ps = ((sf[ai] as i16) * (wsf[wi] as i16)) as i8;
                            let pe =
                                (ef[ai] as i64 + wef[wi] as i64 - BIAS).clamp(0, NLVL - 1) as i32;
                            let pz = zf[ai] | wzf[wi];
                            let oi = oy as usize * wo + ox as usize;
                            plane[oi] = plane[oi].wrapping_add(fixed_scalar(ps, pe, pz, m_out));
                        }
                    }
                }
            }
        }
        for i in 0..ho * wo {
            acc[i * cout + co] = plane[i];
        }
    }
    if let Some(b) = bias {
        for co in 0..cout {
            let bq = fixed_scalar(
                *b.s.get(IxDyn(&[co])).unwrap_or(&0),
                *b.e.get(IxDyn(&[co])).unwrap_or(&0),
                *b.z.get(IxDyn(&[co])).unwrap_or(&0),
                m_out,
            );
            for i in 0..ho * wo {
                acc[i * cout + co] = acc[i * cout + co].wrapping_add(bq);
            }
        }
    }
    let arr = from_flat(&[ho, wo, cout], acc)?;
    lattice::from_fixed(&arr, m_out)
}

/// kh x kh phi-conv, same/stride/pad convention, optional bias.
pub fn phi_conv(
    x: &Triple,
    w: &Triple,
    m_out: i64,
    stride: i64,
    bias: Option<&Triple>,
) -> Result<Triple> {
    let sh = x.shape();
    let wsh = w.shape();
    if sh.len() != 3 || wsh.len() != 4 {
        return Err(Error::runtime("phi_conv needs (H,W,Cin) and (O,kh,kh,Cin)"));
    }
    let (h, ww, cin) = (sh[0], sh[1], sh[2]);
    let (cout, kh, _, wcin) = (wsh[0], wsh[1], wsh[2], wsh[3]);
    if wcin != cin || kh != wsh[2] {
        return Err(Error::runtime("phi_conv weight shape mismatch"));
    }
    let pad = if kh == 3 { kh / 2 } else { 0 };
    let ho = (h + 2 * pad - kh) / stride as usize + 1;
    let wo = (ww + 2 * pad - kh) / stride as usize + 1;
    let sf = flat(&x.s);
    let ef = flat(&x.e);
    let zf = flat(&x.z);
    let wsf = flat(&w.s);
    let wef = flat(&w.e);
    let wzf = flat(&w.z);
    let mut os = vec![0i8; ho * wo * cout];
    let mut oe = vec![0i32; ho * wo * cout];
    let mut oz = vec![0u8; ho * wo * cout];
    for co in 0..cout {
        let mut acc = vec![0i64; ho * wo];
        for ky in 0..kh {
            for kx in 0..kh {
                for ci in 0..cin {
                    let wi = ((co * kh + ky) * kh + kx) * cin + ci;
                    for oy in 0..ho {
                        let iy = oy as i64 * stride - pad as i64 + ky as i64;
                        if iy < 0 || iy >= h as i64 {
                            continue;
                        }
                        for ox in 0..wo {
                            let ix = ox as i64 * stride - pad as i64 + kx as i64;
                            if ix < 0 || ix >= ww as i64 {
                                continue;
                            }
                            let ai = (iy as usize * ww + ix as usize) * cin + ci;
                            let ps = ((sf[ai] as i16) * (wsf[wi] as i16)) as i8;
                            let pe =
                                (ef[ai] as i64 + wef[wi] as i64 - BIAS).clamp(0, NLVL - 1) as i32;
                            let pz = zf[ai] | wzf[wi];
                            acc[oy * wo + ox] =
                                acc[oy * wo + ox].wrapping_add(fixed_scalar(ps, pe, pz, m_out));
                        }
                    }
                }
            }
        }
        if let Some(b) = bias {
            let bq = fixed_scalar(
                *b.s.get(IxDyn(&[co])).unwrap_or(&0),
                *b.e.get(IxDyn(&[co])).unwrap_or(&0),
                *b.z.get(IxDyn(&[co])).unwrap_or(&0),
                m_out,
            );
            for v in &mut acc {
                *v = v.wrapping_add(bq);
            }
        }
        let got = lattice::from_fixed(&from_flat(&[ho * wo], acc)?, m_out)?;
        for i in 0..ho * wo {
            let dst = i * cout + co;
            os[dst] = got.s[IxDyn(&[i])];
            oe[dst] = got.e[IxDyn(&[i])];
            oz[dst] = got.z[IxDyn(&[i])];
        }
    }
    Triple::from_vecs(&[ho, wo, cout], os, oe, oz)
}

/// GroupNorm + affine over HWC triples.
pub fn groupnorm_int(
    x: &Triple,
    w: &Triple,
    b: &Triple,
    groups: usize,
    m: i64,
    eps_c: i64,
) -> Result<Triple> {
    let sh = x.shape();
    if sh.len() != 3 || sh[2] % groups != 0 {
        return Err(Error::runtime("groupnorm needs HWC with C%G==0"));
    }
    let (h, ww, c) = (sh[0], sh[1], sh[2]);
    let cg = c / groups;
    let q = lattice::to_fixed(x, m);
    let wq = lattice::to_fixed(w, m);
    let bq = lattice::to_fixed(b, m);
    let qf = flat(&q);
    let wf = flat(&wq);
    let bf = flat(&bq);
    let mut out = vec![0i64; qf.len()];
    for g in 0..groups {
        let n = h * ww * cg;
        let mut tot: i128 = 0;
        for y in 0..h {
            for xx in 0..ww {
                for ci in 0..cg {
                    tot += qf[(y * ww + xx) * c + g * cg + ci] as i128;
                }
            }
        }
        let mean = lattice::tdiv(tot as i64, n as i64);
        let mut sq: i128 = 0;
        for y in 0..h {
            for xx in 0..ww {
                for ci in 0..cg {
                    let dv = qf[(y * ww + xx) * c + g * cg + ci] - mean;
                    sq += (dv as i128) * (dv as i128);
                }
            }
        }
        let var = lattice::tdiv(sq as i64, n as i64).wrapping_add(eps_c);
        let std = if var > 0 {
            isqrt_u128(var as u128) as i64
        } else {
            0
        };
        for y in 0..h {
            for xx in 0..ww {
                for ci in 0..cg {
                    let idx = (y * ww + xx) * c + g * cg + ci;
                    let norm = if std == 0 {
                        0
                    } else {
                        lattice::tdiv((qf[idx] - mean).wrapping_mul(1 << 18), std)
                    };
                    out[idx] = lattice::tdiv(norm.wrapping_mul(wf[g * cg + ci]), 1 << 18)
                        .wrapping_add(bf[g * cg + ci]);
                }
            }
        }
    }
    let arr = from_flat(&sh, out)?;
    lattice::from_fixed(&arr, m)
}

/// Dense inner product over raveled triples (+ optional bias).
pub fn ip_int(x: &Triple, w: &Triple, m_out: i64, bias: Option<&Triple>) -> Result<Triple> {
    let wsh = w.shape();
    if wsh.len() != 2 {
        return Err(Error::runtime("ip_int weight must be (O,N)"));
    }
    let (cout, n) = (wsh[0], wsh[1]);
    if x.numel() != n {
        return Err(Error::runtime("ip_int input size mismatch"));
    }
    let xf_s = flat(&x.s);
    let xf_e = flat(&x.e);
    let xf_z = flat(&x.z);
    let wf_s = flat(&w.s);
    let wf_e = flat(&w.e);
    let wf_z = flat(&w.z);
    let mut acc = vec![0i64; cout];
    if let Some(b) = bias {
        let bq = lattice::to_fixed(b, m_out);
        for (i, &v) in bq.iter().enumerate() {
            acc[i] = acc[i].wrapping_add(v);
        }
    }
    for j in 0..cout {
        let mut sum: i128 = 0;
        for i in 0..n {
            let ps = ((xf_s[i] as i16) * (wf_s[j * n + i] as i16)) as i8;
            let pe = (xf_e[i] as i64 + wf_e[j * n + i] as i64 - BIAS).clamp(0, NLVL - 1) as i32;
            let pz = xf_z[i] | wf_z[j * n + i];
            sum = sum.wrapping_add(fixed_scalar(ps, pe, pz, m_out) as i128);
        }
        acc[j] = acc[j].wrapping_add(sum as i64);
    }
    lattice::from_fixed(&from_flat(&[cout], acc)?, m_out)
}

/// Two-level tiled scan (reference implementation, order-sensitive).
pub fn tiled_scan(abar_q: &ArrayD<i64>, bx_q: &ArrayD<i64>, tile: usize) -> Result<ArrayD<i64>> {
    let sh = abar_q.shape().to_vec();
    if sh.len() != 3 || bx_q.shape() != abar_q.shape() {
        return Err(Error::runtime("tiled_scan needs (N,D,S) matching"));
    }
    let (n, d, s) = (sh[0], sh[1], sh[2]);
    let one = 1i64 << 18;
    let af = flat(abar_q);
    let bf = flat(bx_q);
    let mut outs = vec![0i64; af.len()];
    let mut pb = vec![0i64; d * s];
    for k0 in (0..n).step_by(tile) {
        let k1 = (k0 + tile).min(n);
        let mut h = vec![0i64; d * s];
        let mut ap = vec![0i64; (k1 - k0) * d * s];
        let mut run_a = vec![one; d * s];
        let mut traj = vec![0i64; (k1 - k0) * d * s];
        for t in k0..k1 {
            for p in 0..d * s {
                h[p] = lattice::tdiv(af[t * d * s + p].wrapping_mul(h[p]), one)
                    .wrapping_add(bf[t * d * s + p]);
                traj[(t - k0) * d * s + p] = h[p];
                run_a[p] = lattice::tdiv(run_a[p].wrapping_mul(af[t * d * s + p]), one);
                ap[(t - k0) * d * s + p] = run_a[p];
            }
        }
        for t in k0..k1 {
            for p in 0..d * s {
                outs[t * d * s + p] =
                    lattice::tdiv(ap[(t - k0) * d * s + p].wrapping_mul(pb[p]), one)
                        .wrapping_add(traj[(t - k0) * d * s + p]);
            }
        }
        let mut ak = vec![one; d * s];
        let mut bk = vec![0i64; d * s];
        for t in k0..k1 {
            for p in 0..d * s {
                ak[p] = lattice::tdiv(ak[p].wrapping_mul(af[t * d * s + p]), one);
                bk[p] = lattice::tdiv(bk[p].wrapping_mul(af[t * d * s + p]), one)
                    .wrapping_add(bf[t * d * s + p]);
            }
        }
        for p in 0..d * s {
            pb[p] = lattice::tdiv(ak[p].wrapping_mul(pb[p]), one).wrapping_add(bk[p]);
        }
    }
    from_flat(&sh, outs)
}
