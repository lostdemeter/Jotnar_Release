//! Image / holo path: sRGB boundaries, structure-tensor blur, beta fields,
//! convolution, warping, and the integer resamplers.

use ndarray::{ArrayD, IxDyn};

use crate::error::{Error, Result};
use crate::ir::Config;
use crate::lattice::{self, BIAS, NLVL};
use crate::value::{Arg, Triple, Value};

use super::element::{
    abs_trip, binop_fixed, clip_fixed, enc_fill, neg_trip, relu_trip, sigmoid_trip, sqrt_trip,
    tdiv_pure, tdiv_trip, tmul,
};
use super::{arg_f64, arg_f64_array, arg_i64, arg_triple, beta, ctrl, scales};

fn flat<T: Copy>(a: &ArrayD<T>) -> Vec<T> {
    a.iter().copied().collect()
}

fn from_flat<T: Clone>(shape: &[usize], v: Vec<T>) -> Result<ArrayD<T>> {
    crate::value::array_from_vec(shape, v)
}

fn to_fixed_scalar(s: i8, e: i32, z: u8, m: i64) -> i64 {
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

fn from_fixed(arr: &ArrayD<i64>, m: i64) -> Result<Triple> {
    lattice::from_fixed(arr, m)
}

fn zero_trip(shape: &[usize], m: i64) -> Result<Triple> {
    let z = ArrayD::from_elem(IxDyn(shape), 0i64);
    from_fixed(&z, m)
}

fn where_mask(mask: &ArrayD<bool>, a: &Triple, b: &Triple) -> Result<Triple> {
    super::triple_where(mask, a, b, "where")
}

// ---------------------------------------------------------------------------
// sRGB / luma / gain
// ---------------------------------------------------------------------------

pub fn op_srgb_decode(args: &[Arg], _: &Config) -> Result<Vec<Value>> {
    let x = args
        .first()
        .ok_or_else(|| Error::runtime("SRGB_DECODE: missing input"))?
        .stream("SRGB_DECODE", "x")?
        .as_u8("SRGB_DECODE")?;
    let out = x.mapv(|v| ((v as f64 / 255.0).clamp(0.0, 1.0)).powf(2.2) as f32);
    Ok(vec![Value::F32(out)])
}

pub fn op_luma(args: &[Arg], _: &Config) -> Result<Vec<Value>> {
    let lin = arg_f64_array(args, 0, "LUMA")?;
    let sh = lin.shape();
    if sh.len() != 3 || sh[2] < 3 {
        return Err(Error::runtime(format!(
            "LUMA: expected HWC with C>=3, got {:?}",
            sh
        )));
    }
    let (h, w, c) = (sh[0], sh[1], sh[2]);
    let f = flat(&lin);
    let mut out = vec![0f64; h * w];
    for y in 0..h {
        for x in 0..w {
            let base = (y * w + x) * c;
            out[y * w + x] = 0.2126 * f[base] + 0.7152 * f[base + 1] + 0.0722 * f[base + 2];
        }
    }
    Ok(vec![Value::F64(from_flat(&[h, w], out)?)])
}

pub fn op_srgb_encode(args: &[Arg], _: &Config) -> Result<Vec<Value>> {
    let lin = arg_f64_array(args, 0, "SRGB_ENCODE")?;
    let out = lin.mapv(|v| (v.clamp(0.0, 1.0).powf(1.0 / 2.2) * 255.0).clamp(0.0, 255.0) as u8);
    Ok(vec![Value::U8(out)])
}

pub fn op_gain(args: &[Arg], config: &Config) -> Result<Vec<Value>> {
    let lin = args
        .first()
        .ok_or_else(|| Error::runtime("GAIN: missing lin"))?
        .stream("GAIN", "lin")?
        .as_f32("GAIN")?;
    let y = arg_f64_array(args, 1, "GAIN")?;
    let yenh = arg_triple(args, 2, "GAIN")?;
    let (_, m_cov) = scales(config);
    let m = m_cov;
    let y_t = lattice::encode(&y);
    let g_t = tdiv_trip(yenh, &y_t)?;
    let g_t = clip_fixed(&g_t, 0.5, 2.0, m)?;
    let sh = lin.shape().to_vec();
    if sh.len() != 3 || sh[2] < 3 {
        return Err(Error::runtime(format!("GAIN: expected HWC, got {:?}", sh)));
    }
    let (h, w, c) = (sh[0], sh[1], sh[2]);
    let lf = flat(&lin);
    let mut out = vec![0f32; h * w * c];
    for ch in 0..3 {
        let mut plane = Vec::with_capacity(h * w);
        for i in 0..h * w {
            plane.push(lf[i * c + ch] as f64);
        }
        let ct = lattice::encode(&from_flat(&[h, w], plane)?);
        let et = tmul(&ct, &g_t)?;
        let dec = lattice::decode(&et.s, &et.e);
        let decf = dec.iter().copied().collect::<Vec<_>>();
        let ezf = et.z.iter().copied().collect::<Vec<_>>();
        for i in 0..h * w {
            let v = decf[i] * (1.0 - ezf[i] as f64);
            out[i * c + ch] = v.clamp(0.0, 1.0) as f32;
        }
    }
    Ok(vec![Value::F32(from_flat(&sh, out)?)])
}

// ---------------------------------------------------------------------------
// Convolution
// ---------------------------------------------------------------------------

pub fn gaussian_kernel(radius: i64, sigma: f64) -> Result<ArrayD<f64>> {
    let r = radius.max(0) as usize;
    let n = 2 * r + 1;
    if sigma == 0.0 {
        return Err(Error::runtime("GAUSS: sigma must be non-zero"));
    }
    let ax: Vec<f64> = (0..n)
        .map(|i| {
            let x = (i as i64 - r as i64) as f64;
            (-0.5 * (x / sigma).powi(2)).exp()
        })
        .collect();
    let sum: f64 = ax.iter().sum();
    let mut k = vec![0f64; n * n];
    for y in 0..n {
        for x in 0..n {
            k[y * n + x] = ax[y] * ax[x] / (sum * sum);
        }
    }
    from_flat(&[n, n], k)
}

fn aniso_kernel(sx: f64, sy: f64, radius: i64) -> Result<ArrayD<f64>> {
    let r = radius.max(0) as usize;
    let n = 2 * r + 1;
    let ax: Vec<f64> = (0..n)
        .map(|i| {
            let x = (i as i64 - r as i64) as f64;
            (-0.5 * (x / sx).powi(2)).exp()
        })
        .collect();
    let ay: Vec<f64> = (0..n)
        .map(|i| {
            let y = (i as i64 - r as i64) as f64;
            (-0.5 * (y / sy).powi(2)).exp()
        })
        .collect();
    let mut k = vec![0f64; n * n];
    let mut sum = 0.0;
    for y in 0..n {
        for x in 0..n {
            k[y * n + x] = ay[y] * ax[x];
            sum += k[y * n + x];
        }
    }
    for v in &mut k {
        *v /= sum;
    }
    from_flat(&[n, n], k)
}

fn rotated_kernel(theta_deg: f64, s_long: f64, s_short: f64, radius: i64) -> Result<ArrayD<f64>> {
    let r = radius.max(0) as usize;
    let n = 2 * r + 1;
    let th = theta_deg.to_radians();
    let (ux, uy) = (th.cos(), th.sin());
    let mut k = vec![0f64; n * n];
    let mut sum = 0.0;
    for y in 0..n {
        for x in 0..n {
            let xx = (x as i64 - r as i64) as f64;
            let yy = (y as i64 - r as i64) as f64;
            let u = xx * ux + yy * uy;
            let v = -xx * uy + yy * ux;
            let val = (-0.5 * ((u / s_long).powi(2) + (v / s_short).powi(2))).exp();
            k[y * n + x] = val;
            sum += val;
        }
    }
    for v in &mut k {
        *v /= sum;
    }
    from_flat(&[n, n], k)
}

pub fn bank() -> Result<Vec<ArrayD<f64>>> {
    Ok(vec![
        rotated_kernel(90.0, 1.8, 0.5, 2)?,
        rotated_kernel(0.0, 1.8, 0.5, 2)?,
        rotated_kernel(45.0, 1.8, 0.5, 2)?,
        rotated_kernel(-45.0, 1.8, 0.5, 2)?,
        aniso_kernel(1.0, 1.0, 2)?,
    ])
}

fn conv_trip(img: &Triple, kernel: &ArrayD<f64>, m_acc: i64, m_out: i64) -> Result<Triple> {
    let sh = img.shape();
    if sh.len() != 2 {
        return Err(Error::runtime(format!(
            "conv_trip: expected HW triples, got {:?}",
            sh
        )));
    }
    let (h, w) = (sh[0], sh[1]);
    let ksh = kernel.shape().to_vec();
    let (kh, kw) = (ksh[0], ksh[1]);
    let (ry, rx) = (kh / 2, kw / 2);
    let sf = flat(&img.s);
    let ef = flat(&img.e);
    let zf = flat(&img.z);
    let kf = flat(kernel);
    let mut acc = vec![0i64; h * w];
    for ky in 0..kh {
        for kx in 0..kw {
            let wv = kf[ky * kw + kx];
            if wv == 0.0 {
                continue;
            }
            let kt = lattice::encode(&ArrayD::from_elem(IxDyn(&[1]), wv));
            let ks = *kt.s.iter().next().unwrap_or(&0);
            let ke = *kt.e.iter().next().unwrap_or(&0) as i64;
            let kz = *kt.z.iter().next().unwrap_or(&0);
            for y in 0..h {
                let sy = (y as i64 + ky as i64 - ry as i64).clamp(0, h as i64 - 1) as usize;
                for x in 0..w {
                    let sx = (x as i64 + kx as i64 - rx as i64).clamp(0, w as i64 - 1) as usize;
                    let idx = sy * w + sx;
                    let ps = ((sf[idx] as i16) * (ks as i16)) as i8;
                    let pe = (ef[idx] as i64 + ke - BIAS).clamp(0, NLVL - 1) as i32;
                    let pz = zf[idx] | kz;
                    let q = to_fixed_scalar(ps, pe, pz, m_acc);
                    acc[y * w + x] = acc[y * w + x].wrapping_add(q);
                }
            }
        }
    }
    let acc_arr = from_flat(&sh, acc)?;
    lattice::assert_bound("conv:acc", &[&acc_arr])?;
    let rescaled = lattice::rescale(&acc_arr, m_acc, m_out)?;
    from_fixed(&rescaled, m_out)
}

pub fn op_iso_blur(args: &[Arg], config: &Config) -> Result<Vec<Value>> {
    let a = arg_triple(args, 0, "ISO_BLUR")?;
    let (m_acc, m_cov) = scales(config);
    let k = gaussian_kernel(2, 1.0)?;
    Ok(vec![Value::Triple(conv_trip(a, &k, m_acc, m_cov)?)])
}

pub fn op_gauss(args: &[Arg], config: &Config) -> Result<Vec<Value>> {
    let a = arg_triple(args, 0, "GAUSS")?;
    let r = arg_i64(args, 1, "GAUSS", "radius")?;
    let s = arg_f64(args, 2, "GAUSS", "sigma")?;
    let (m_acc, m_cov) = scales(config);
    let k = gaussian_kernel(r, s)?;
    Ok(vec![Value::Triple(conv_trip(a, &k, m_acc, m_cov)?)])
}

pub fn op_resize2d(args: &[Arg], _: &Config) -> Result<Vec<Value>> {
    let x = arg_f64_array(args, 0, "RESIZE2D")?;
    let ho = arg_i64(args, 1, "RESIZE2D", "ho")?;
    let wo = arg_i64(args, 2, "RESIZE2D", "wo")?;
    let align = arg_i64(args, 3, "RESIZE2D", "align")?;
    if ho <= 0 || wo <= 0 || (align != 0 && align != 1) {
        return Err(Error::runtime("RESIZE2D: bad size/align"));
    }
    let (ho, wo) = (ho as usize, wo as usize);
    let sh = x.shape();
    if sh.len() != 3 {
        return Err(Error::runtime(format!(
            "RESIZE2D: expected HWC, got {:?}",
            sh
        )));
    }
    let (h, w, c) = (sh[0], sh[1], sh[2]);
    let xf = flat(&x);
    let ac = align != 0;
    let mut out = vec![0f64; ho * wo * c];
    for oy in 0..ho {
        let sy = if ac {
            if ho > 1 {
                oy as f64 * (h as f64 - 1.0) / (ho as f64 - 1.0)
            } else {
                0.0
            }
        } else {
            (oy as f64 + 0.5) * h as f64 / ho as f64 - 0.5
        }
        .clamp(0.0, h as f64 - 1.0);
        let y0 = sy.floor() as usize;
        let y1 = (y0 + 1).min(h - 1);
        let wy = sy - y0 as f64;
        for ox in 0..wo {
            let sx = if ac {
                if wo > 1 {
                    ox as f64 * (w as f64 - 1.0) / (wo as f64 - 1.0)
                } else {
                    0.0
                }
            } else {
                (ox as f64 + 0.5) * w as f64 / wo as f64 - 0.5
            }
            .clamp(0.0, w as f64 - 1.0);
            let x0 = sx.floor() as usize;
            let x1 = (x0 + 1).min(w - 1);
            let wx = sx - x0 as f64;
            for cc in 0..c {
                let v00 = xf[(y0 * w + x0) * c + cc];
                let v01 = xf[(y0 * w + x1) * c + cc];
                let v10 = xf[(y1 * w + x0) * c + cc];
                let v11 = xf[(y1 * w + x1) * c + cc];
                out[(oy * wo + ox) * c + cc] = (1.0 - wy) * (1.0 - wx) * v00
                    + (1.0 - wy) * wx * v01
                    + wy * (1.0 - wx) * v10
                    + wy * wx * v11;
            }
        }
    }
    Ok(vec![Value::F64(from_flat(&[ho, wo, c], out)?)])
}

pub fn op_conv(args: &[Arg], config: &Config) -> Result<Vec<Value>> {
    let t = arg_triple(args, 0, "CONV")?;
    let k = arg_f64_array(args, 1, "CONV")?;
    let (m_acc, m_cov) = scales(config);
    Ok(vec![Value::Triple(conv_trip(t, &k, m_acc, m_cov)?)])
}

// ---------------------------------------------------------------------------
// Structure tensor + splat blur
// ---------------------------------------------------------------------------

const SOBEL_X: [f64; 9] = [
    -1.0 / 8.0,
    0.0,
    1.0 / 8.0,
    -2.0 / 8.0,
    0.0,
    2.0 / 8.0,
    -1.0 / 8.0,
    0.0,
    1.0 / 8.0,
];

fn sobel_kernel(transposed: bool) -> Result<ArrayD<f64>> {
    // SOBEL_X row-major; SOBEL_Y = SOBEL_X.T.
    let mut k = vec![0f64; 9];
    for y in 0..3 {
        for x in 0..3 {
            k[y * 3 + x] = if transposed {
                SOBEL_X[x * 3 + y]
            } else {
                SOBEL_X[y * 3 + x]
            };
        }
    }
    from_flat(&[3, 3], k)
}

fn structure(
    a: &Triple,
    m_acc: i64,
    m_cov: i64,
) -> Result<(Triple, Triple, Triple, Triple, Triple)> {
    let gx = conv_trip(a, &sobel_kernel(false)?, m_acc, m_cov)?;
    let gy = conv_trip(a, &sobel_kernel(true)?, m_acc, m_cov)?;
    let jxx = tmul(&gx, &gx)?;
    let jyy = tmul(&gy, &gy)?;
    let jxy = tmul(&gx, &gy)?;
    let k = gaussian_kernel(1, 0.8)?;
    let jxx = conv_trip(&jxx, &k, m_acc, m_cov)?;
    let jyy = conv_trip(&jyy, &k, m_acc, m_cov)?;
    let jxy = conv_trip(&jxy, &k, m_acc, m_cov)?;
    Ok((jxx, jyy, jxy, gx, gy))
}

struct CoherenceAudit {
    d: Triple,
    s2: Triple,
    bucket: ArrayD<i8>,
}

fn coherence_bucket(
    jxx: &Triple,
    jyy: &Triple,
    jxy: &Triple,
    m_cov: i64,
    m_acc: i64,
    coh_thr: f64,
) -> Result<(Triple, ArrayD<i8>, CoherenceAudit)> {
    let shape = jxx.shape();
    let d = binop_fixed(jxx, jyy, m_cov, m_cov, "sub")?;
    let d2 = tmul(&d, &d)?;
    let xy2 = tmul(jxy, jxy)?;
    let four = enc_fill(&shape, 4.0);
    let t4 = tmul(&xy2, &four)?;
    let disc_acc = binop_fixed(&d2, &t4, m_acc, m_acc, "add")?;
    let dq_acc = lattice::to_fixed(&disc_acc, m_acc);
    let disc = from_fixed(&lattice::rescale(&dq_acc, m_acc, m_cov)?, m_cov)?;
    let aniso = sqrt_trip(&disc)?;
    let trace = binop_fixed(jxx, jyy, m_cov, m_cov, "add")?;
    let raw = tdiv_trip(&aniso, &trace)?;
    let tz = trace.z.mapv(|v| v != 0);
    let zt = zero_trip(&shape, m_cov)?;
    let coh0 = where_mask(&tz, &zt, &raw)?;
    let coh = clip_fixed(&coh0, 0.0, 1.0, m_cov)?;
    let dq = lattice::to_fixed(&d, m_cov);
    let s2 = binop_fixed(jxy, jxy, m_cov, m_cov, "add")?;
    let sq = lattice::to_fixed(&s2, m_cov);
    let coh_q = lattice::to_fixed(&coh, m_cov);
    let tq = lattice::to_fixed(&enc_fill(&[1], coh_thr), m_cov)[0];
    let gate = coh_q.mapv(|v| v >= tq);
    let diag_dom = ndarray::Zip::from(&sq)
        .and(&dq)
        .map_collect(|&a, &b| a.abs() > b.abs());
    let vert = dq.mapv(|v| v > 0);
    let diag_pos = sq.mapv(|v| v >= 0);
    let mut direction = vec![0i8; shape.iter().product()];
    let ddf = diag_dom.iter().copied().collect::<Vec<_>>();
    let vf = vert.iter().copied().collect::<Vec<_>>();
    let dpf = diag_pos.iter().copied().collect::<Vec<_>>();
    for i in 0..direction.len() {
        direction[i] = if !ddf[i] {
            if vf[i] {
                0
            } else {
                1
            }
        } else if dpf[i] {
            3
        } else {
            2
        };
    }
    let dir = from_flat(&shape, direction)?;
    let mut bucket = vec![0i8; dir.len()];
    let gf = gate.iter().copied().collect::<Vec<_>>();
    let dirf = dir.iter().copied().collect::<Vec<_>>();
    for i in 0..dirf.len() {
        bucket[i] = if !gf[i] { 4 } else { dirf[i] };
    }
    let bucket = from_flat(&shape, bucket)?;
    Ok((coh, bucket.clone(), CoherenceAudit { d, s2, bucket }))
}

fn soft_blend(outs: &[Triple], d: &Triple, s2: &Triple, m_cov: i64) -> Result<Triple> {
    let rv = relu_trip(d, m_cov)?;
    let rh = relu_trip(&neg_trip(d)?, m_cov)?;
    let rd1 = relu_trip(s2, m_cov)?;
    let rd2 = relu_trip(&neg_trip(s2)?, m_cov)?;
    let den = binop_fixed(
        &binop_fixed(&rv, &rh, m_cov, m_cov, "add")?,
        &binop_fixed(&rd1, &rd2, m_cov, m_cov, "add")?,
        m_cov,
        m_cov,
        "add",
    )?;
    let wv = tdiv_pure(&rv, &den)?;
    let wh = tdiv_pure(&rh, &den)?;
    let wd1 = tdiv_pure(&rd1, &den)?;
    let wd2 = tdiv_pure(&rd2, &den)?;
    let mut acc = binop_fixed(
        &tmul(&wv, &outs[0])?,
        &tmul(&wh, &outs[1])?,
        m_cov,
        m_cov,
        "add",
    )?;
    acc = binop_fixed(&acc, &tmul(&wd1, &outs[3])?, m_cov, m_cov, "add")?;
    acc = binop_fixed(&acc, &tmul(&wd2, &outs[2])?, m_cov, m_cov, "add")?;
    let dz = den.z.mapv(|z| z != 0);
    where_mask(&dz, &outs[4], &acc)
}

fn splat_blur(
    a: &Triple,
    m_acc: i64,
    m_cov: i64,
    coh_thr: f64,
    soft: bool,
) -> Result<(Triple, Triple)> {
    let (jxx, jyy, jxy, _gx, _gy) = structure(a, m_acc, m_cov)?;
    let (coh, _bucket, audit) = coherence_bucket(&jxx, &jyy, &jxy, m_cov, m_acc, coh_thr)?;
    let bank_k = bank()?;
    let out = if soft {
        let mut outs = Vec::with_capacity(bank_k.len());
        for k in &bank_k {
            outs.push(conv_trip(a, k, m_acc, m_cov)?);
        }
        soft_blend(&outs, &audit.d, &audit.s2, m_cov)?
    } else {
        let mut outs = Vec::with_capacity(bank_k.len());
        for k in &bank_k {
            outs.push(conv_trip(a, k, m_acc, m_cov)?);
        }
        super::element::select_mux(&outs, &audit.bucket.mapv(|v| v as i64))?
    };
    Ok((out, coh))
}

pub fn op_splat_blur(args: &[Arg], config: &Config) -> Result<Vec<Value>> {
    let a = arg_triple(args, 0, "SPLAT_BLUR")?;
    let (m_acc, m_cov) = scales(config);
    let (out, coh) = splat_blur(a, m_acc, m_cov, ctrl::COH_THR, true)?;
    Ok(vec![Value::Triple(out), Value::Triple(coh)])
}

// ---------------------------------------------------------------------------
// Beta fields
// ---------------------------------------------------------------------------

fn beta_field_soft(beta: f64, coh: &Triple, m_cov: i64) -> Result<Triple> {
    let shape = coh.shape();
    let att = ctrl::ISO_ATTEN;
    let md = ctrl::MID_ATTEN;
    let st = ctrl::STRONG_ATTEN;
    let thr = ctrl::COH_THR;
    let hi = ctrl::COH_HI;
    let k = 30.0f64;
    let kt = enc_fill(&shape, k);
    let sig_at = |level: f64| -> Result<Triple> {
        let lt = enc_fill(&shape, level);
        let d = binop_fixed(coh, &lt, m_cov, m_cov, "sub")?;
        sigmoid_trip(&tmul(&d, &kt)?)
    };
    let s1 = sig_at(thr)?;
    let s2 = sig_at(hi)?;
    let iso = enc_fill(&shape, att);
    let dm = enc_fill(&shape, md - att);
    let ds = enc_fill(&shape, st - md);
    let mut w = binop_fixed(&iso, &tmul(&dm, &s1)?, m_cov, m_cov, "add")?;
    w = binop_fixed(&w, &tmul(&ds, &s2)?, m_cov, m_cov, "add")?;
    let b = enc_fill(&shape, beta);
    tmul(&b, &w)
}

fn beta_field_v5(beta: f64, coh: &Triple, d: &Triple, m_cov: i64) -> Result<Triple> {
    let shape = coh.shape();
    let (a0, a1, a2) = (ctrl::V5_W0, ctrl::V5_W1, ctrl::V5_W2);
    let base = beta_field_soft(beta, coh, m_cov)?;
    let four = enc_fill(&shape, 4.0);
    let dhat = clip_fixed(&tmul(&abs_trip(d)?, &four)?, 0.0, 1.0, m_cov)?;
    let t0 = enc_fill(&shape, a0);
    let t1 = tmul(&enc_fill(&shape, a1), coh)?;
    let t2 = tmul(&enc_fill(&shape, a2), &dhat)?;
    let logit = binop_fixed(
        &binop_fixed(&t0, &t1, m_cov, m_cov, "add")?,
        &t2,
        m_cov,
        m_cov,
        "add",
    )?;
    let gate = sigmoid_trip(&logit)?;
    let half = enc_fill(&shape, 0.5);
    let scale = binop_fixed(&half, &gate, m_cov, m_cov, "add")?;
    tmul(&base, &scale)
}

pub fn op_beta_v5(args: &[Arg], config: &Config) -> Result<Vec<Value>> {
    let d = arg_triple(args, 0, "BETA_V5")?;
    let coh = arg_triple(args, 1, "BETA_V5")?;
    let (_, m_cov) = scales(config);
    Ok(vec![Value::Triple(beta_field_v5(
        beta(config),
        coh,
        d,
        m_cov,
    )?)])
}

// ---------------------------------------------------------------------------
// Warp / static / mixdyad / pooling / PReLU / interp
// ---------------------------------------------------------------------------

fn warp_trips(dprev: &Triple, flow: &ArrayD<f64>, m: i64) -> Result<Triple> {
    let shape = dprev.shape();
    if shape.len() != 2 {
        return Err(Error::runtime("WARP: expected HW triples"));
    }
    let (h, w) = (shape[0], shape[1]);
    if flow.shape() != [h, w, 2] {
        return Err(Error::runtime(format!(
            "WARP: feature {:?} vs flow {:?} spatial mismatch",
            shape,
            flow.shape()
        )));
    }
    let q = lattice::to_fixed(dprev, m);
    let qf = flat(&q);
    let ff = flat(flow);
    let one = 1i64 << 14;
    let mut out = vec![0i64; h * w];
    for y in 0..h {
        for x in 0..w {
            let fi = (y * w + x) * 2;
            let fx = lattice::round_half_even(ff[fi + 1] * one as f64) as i64;
            let fy = lattice::round_half_even(ff[fi] * one as f64) as i64;
            let xs = ((x as i64) << 14) + fx;
            let ys = ((y as i64) << 14) + fy;
            let x0 = xs.div_euclid(one);
            let y0 = ys.div_euclid(one);
            let xf = xs - x0 * one;
            let yf = ys - y0 * one;
            let x0c = x0.clamp(0, w as i64 - 1) as usize;
            let x1c = (x0 + 1).clamp(0, w as i64 - 1) as usize;
            let y0c = y0.clamp(0, h as i64 - 1) as usize;
            let y1c = (y0 + 1).clamp(0, h as i64 - 1) as usize;
            let w00 = (one - xf) * (one - yf);
            let w10 = xf * (one - yf);
            let w01 = (one - xf) * yf;
            let w11 = xf * yf;
            let tot = qf[y0c * w + x0c]
                .wrapping_mul(w00)
                .wrapping_add(qf[y0c * w + x1c].wrapping_mul(w10))
                .wrapping_add(qf[y1c * w + x0c].wrapping_mul(w01))
                .wrapping_add(qf[y1c * w + x1c].wrapping_mul(w11));
            out[y * w + x] = tot.div_euclid(one * one);
        }
    }
    let arr = from_flat(&shape, out)?;
    lattice::assert_bound("warp:acc", &[&arr])?;
    from_fixed(&arr, m)
}

pub fn op_warp(args: &[Arg], config: &Config) -> Result<Vec<Value>> {
    let dprev = args
        .first()
        .ok_or_else(|| Error::runtime("WARP: missing history"))?
        .stream("WARP", "dprev")?;
    if dprev.is_none() {
        return Ok(vec![Value::None]);
    }
    let t = dprev.triple("WARP")?;
    let flow = arg_f64_array(args, 1, "WARP")?;
    let (_, m_cov) = scales(config);
    Ok(vec![Value::Triple(warp_trips(t, &flow, m_cov)?)])
}

pub fn op_static(args: &[Arg], _: &Config) -> Result<Vec<Value>> {
    let flow = arg_f64_array(args, 0, "STATIC")?;
    let sh = flow.shape();
    if sh.len() != 3 || sh[2] < 2 {
        return Err(Error::runtime(format!(
            "STATIC: expected HW2 flow, got {:?}",
            sh
        )));
    }
    let (h, w) = (sh[0], sh[1]);
    let f = flat(&flow);
    let mut out = vec![false; h * w];
    for i in 0..h * w {
        let mag = (f[i * sh[2]].powi(2) + f[i * sh[2] + 1].powi(2)).sqrt();
        out[i] = mag == 0.0;
    }
    Ok(vec![Value::Bool(from_flat(&[h, w], out)?)])
}

pub fn op_mixdyad(args: &[Arg], config: &Config) -> Result<Vec<Value>> {
    let d = arg_triple(args, 0, "MIXDYAD")?;
    let wv = args
        .get(1)
        .ok_or_else(|| Error::runtime("MIXDYAD: missing history"))?
        .stream("MIXDYAD", "w")?;
    if wv.is_none() {
        return Ok(vec![Value::Triple(d.clone())]);
    }
    let w = wv.triple("MIXDYAD")?;
    let s = args
        .get(2)
        .ok_or_else(|| Error::runtime("MIXDYAD: missing mask"))?
        .stream("MIXDYAD", "s")?
        .as_bool("MIXDYAD")?;
    let (_, m_cov) = scales(config);
    let mut acc = binop_fixed(d, w, m_cov, m_cov, "add")?;
    acc = binop_fixed(&acc, w, m_cov, m_cov, "add")?;
    acc = binop_fixed(&acc, w, m_cov, m_cov, "add")?;
    let q = lattice::to_fixed(&acc, m_cov);
    let mixed = from_fixed(&q.mapv(|v| lattice::tdiv(v, 4)), m_cov)?;
    let shape = d.shape();
    if s.shape() != shape.as_slice() {
        return Err(Error::runtime(format!(
            "MIXDYAD: mask geometry {:?}",
            s.shape()
        )));
    }
    Ok(vec![Value::Triple(where_mask(&s, &mixed, d)?)])
}

pub fn op_poolavg(args: &[Arg], config: &Config) -> Result<Vec<Value>> {
    let t = arg_triple(args, 0, "POOLAVG")?;
    let sh = t.shape();
    if sh.len() != 3 {
        return Err(Error::runtime(format!(
            "POOLAVG: HWC triples only, got ndim={}",
            sh.len()
        )));
    }
    let (h, w, c) = (sh[0], sh[1], sh[2]);
    let (_, m_cov) = scales(config);
    let q = lattice::to_fixed(t, m_cov);
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
    Ok(vec![Value::Triple(from_fixed(&arr, m_cov)?)])
}

pub fn op_prelu(args: &[Arg], _: &Config) -> Result<Vec<Value>> {
    let t = arg_triple(args, 0, "PRELU")?;
    let slope = arg_triple(args, 1, "PRELU")?;
    let shape = t.shape();
    let c = *shape.last().unwrap_or(&1);
    // numpy broadcasting: scalar slope, or a per-channel (C,) slope.
    let n = t.numel();

    let (slope_s, slope_e, slope_z) = if slope.numel() == 1 {
        let s0 = *slope.s.iter().next().unwrap_or(&0);
        let e0 = *slope.e.iter().next().unwrap_or(&0);
        let z0 = *slope.z.iter().next().unwrap_or(&0);
        (vec![s0; n], vec![e0; n], vec![z0; n])
    } else if slope.numel() == c {
        let ss = slope.s.iter().copied().collect::<Vec<_>>();
        let se = slope.e.iter().copied().collect::<Vec<_>>();
        let sz = slope.z.iter().copied().collect::<Vec<_>>();
        (
            (0..n).map(|i| ss[i % c]).collect(),
            (0..n).map(|i| se[i % c]).collect(),
            (0..n).map(|i| sz[i % c]).collect(),
        )
    } else {
        return Err(Error::runtime(format!(
            "PRELU: slope shape {:?} not broadcastable to {shape:?}",
            slope.shape()
        )));
    };
    let slope_full = Triple::from_vecs(&shape, slope_s, slope_e, slope_z)?;
    let prod = tmul(t, &slope_full)?;
    let mut s = Vec::with_capacity(t.numel());
    let mut e = Vec::with_capacity(t.numel());
    let mut z = Vec::with_capacity(t.numel());
    let ts = t.s.iter().copied().collect::<Vec<_>>();
    let te = t.e.iter().copied().collect::<Vec<_>>();
    let tz = t.z.iter().copied().collect::<Vec<_>>();
    let ps = prod.s.iter().copied().collect::<Vec<_>>();
    let pe = prod.e.iter().copied().collect::<Vec<_>>();
    let pz = prod.z.iter().copied().collect::<Vec<_>>();
    for i in 0..t.numel() {
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
    Ok(vec![Value::Triple(Triple::from_vecs(&shape, s, e, z)?)])
}

fn is_dyadic(v: f64) -> bool {
    if v <= 0.0 {
        return false;
    }
    // frexp significand == 0.5
    let (mant, _) = frexp(v);
    (mant - 0.5).abs() < 1e-12
}

fn frexp(x: f64) -> (f64, i32) {
    if x == 0.0 || !x.is_finite() {
        return (x, 0);
    }
    let bits = x.to_bits();
    let exp = ((bits >> 52) & 0x7ff) as i32 - 1022;
    let mant_bits = (bits & !(0x7ffu64 << 52)) | (1022u64 << 52);
    (f64::from_bits(mant_bits), exp)
}

fn bilinear_sample(
    f: &ArrayD<i64>,
    xs: &[i64],
    ys: &[i64],
    ho: usize,
    wo: usize,
) -> Result<ArrayD<i64>> {
    let sh = f.shape().to_vec();
    let (h, w, c) = (sh[0], sh[1], sh[2]);
    let ff = flat(f);
    let one = 1i64 << 14;
    let mut out = vec![0i64; ho * wo * c];
    for oy in 0..ho {
        let yc = ys[oy].clamp(0, (h as i64 - 1) << 14);
        let yi = yc >> 14;
        let yf = yc & (one - 1);
        let yi1 = (yi + 1).min(h as i64 - 1);
        for ox in 0..wo {
            let xc = xs[ox].clamp(0, (w as i64 - 1) << 14);
            let xi = xc >> 14;
            let xf = xc & (one - 1);
            let xi1 = (xi + 1).min(w as i64 - 1);
            for ch in 0..c {
                let g = |yy: i64, xx: i64| ff[(yy as usize * w + xx as usize) * c + ch];
                let tot = g(yi, xi)
                    .wrapping_mul((one - xf).wrapping_mul(one - yf))
                    .wrapping_add(g(yi, xi1).wrapping_mul(xf.wrapping_mul(one - yf)))
                    .wrapping_add(g(yi1, xi).wrapping_mul((one - xf).wrapping_mul(yf)))
                    .wrapping_add(g(yi1, xi1).wrapping_mul(xf.wrapping_mul(yf)));
                out[(oy * wo + ox) * c + ch] = tot >> 28;
            }
        }
    }
    from_flat(&[ho, wo, c], out)
}

fn interp_coord(n_out: usize, s: f64) -> Vec<i64> {
    let mut out = Vec::with_capacity(n_out);
    for o in 0..n_out {
        let o = o as i64;
        let mut val = None;
        for k in 1..14 {
            if s == 2f64.powi(-k) {
                val = Some(((2 * o + 1) << (13 + k)) - 8192);
                break;
            }
            if s == 2f64.powi(k) {
                val = Some(((2 * o + 1) - (1 << k)) << (13 - k));
                break;
            }
        }
        let v = val.unwrap_or_else(|| {
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

pub fn op_interp(args: &[Arg], config: &Config) -> Result<Vec<Value>> {
    let t = arg_triple(args, 0, "INTERP")?;
    let sy = arg_f64(args, 1, "INTERP", "sy")?;
    let sx = arg_f64(args, 2, "INTERP", "sx")?;
    if !(is_dyadic(sy) && is_dyadic(sx)) {
        return Err(Error::runtime(format!(
            "INTERP: non-dyadic scale ({sy},{sx}) is a LOWERING ERROR"
        )));
    }
    let (_, m_cov) = scales(config);
    let q = lattice::to_fixed(t, m_cov);
    let sh = q.shape().to_vec();
    let (h, w) = (sh[0], sh[1]);
    let c = if sh.len() == 3 { sh[2] } else { 1 };
    let ho = ((h as f64 * sy).round().max(1.0)) as usize;
    let wo = ((w as f64 * sx).round().max(1.0)) as usize;
    let ys = interp_coord(ho, sy);
    let xs = interp_coord(wo, sx);
    let q3 = from_flat(&[h, w, c], flat(&q))?;
    let out = bilinear_sample(&q3, &xs, &ys, ho, wo)?;
    let tri = from_fixed(&out, m_cov)?;
    Ok(vec![Value::Triple(Triple::from_vecs(
        &[ho, wo, c],
        flat(&tri.s),
        flat(&tri.e),
        flat(&tri.z),
    )?)])
}

pub fn op_deconv(args: &[Arg], config: &Config) -> Result<Vec<Value>> {
    let t = arg_triple(args, 0, "DECONV")?;
    let w = arg_triple(args, 1, "DECONV")?;
    let stride = arg_i64(args, 2, "DECONV", "stride")?;
    let pad = arg_i64(args, 3, "DECONV", "pad")?;
    let (_, m_cov) = scales(config);
    let sh = t.shape();
    let wsh = w.shape();
    if sh.len() != 3 || wsh.len() != 4 {
        return Err(Error::runtime(format!(
            "DECONV: expected (H,W,Cin) and (O,kh,kw,Cin), got {sh:?} / {wsh:?}"
        )));
    }
    if stride <= 0 || pad < 0 {
        return Err(Error::runtime("DECONV: stride must be > 0, pad >= 0"));
    }
    let (h, ww, cin) = (sh[0], sh[1], sh[2]);
    let (cout, kh, kw, wcin) = (wsh[0], wsh[1], wsh[2], wsh[3]);
    if wcin != cin {
        return Err(Error::runtime(format!(
            "DECONV: weight Cin {wcin} != input Cin {cin}"
        )));
    }
    let ho = (h as i64 - 1) * stride - 2 * pad + kh as i64;
    let wo = (ww as i64 - 1) * stride - 2 * pad + kw as i64;
    if ho <= 0 || wo <= 0 {
        return Err(Error::runtime(format!("DECONV: bad output size {ho}x{wo}")));
    }
    let (ho, wo) = (ho as usize, wo as usize);
    let sf = flat(&t.s);
    let ef = flat(&t.e);
    let zf = flat(&t.z);
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
                            let q = to_fixed_scalar(ps, pe, pz, m_cov);
                            let oi = oy as usize * wo + ox as usize;
                            plane[oi] = plane[oi].wrapping_add(q);
                        }
                    }
                }
            }
        }
        for i in 0..ho * wo {
            acc[i * cout + co] = plane[i];
        }
    }
    let arr = from_flat(&[ho, wo, cout], acc)?;
    lattice::assert_bound("deconv:acc", &[&arr])?;
    Ok(vec![Value::Triple(from_fixed(&arr, m_cov)?)])
}
