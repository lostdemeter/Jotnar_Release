//! Linear algebra, norms, softmax, attention, and exact move ops.

use ndarray::{ArrayD, IxDyn};

use crate::error::{Error, Result};
use crate::ir::Config;
use crate::lattice::{self, BIAS, NLVL, PHI};
use crate::value::{Arg, Triple, Value};

use super::element::{binop_fixed, sigmoid_trip, tmul};
use super::{arg_f64, arg_f64_array, arg_i64, arg_i64_array, arg_triple, scales};

fn flat<T: Copy>(a: &ArrayD<T>) -> Vec<T> {
    a.iter().copied().collect()
}

fn from_flat<T: Clone>(shape: &[usize], v: Vec<T>) -> Result<ArrayD<T>> {
    crate::value::array_from_vec(shape, v)
}

fn num_rows(shape: &[usize]) -> usize {
    if shape.is_empty() {
        return 1;
    }
    shape[..shape.len() - 1].iter().product()
}

fn last_dim(shape: &[usize]) -> usize {
    *shape.last().unwrap_or(&1)
}

fn isqrt_u128(n: u128) -> u128 {
    if n < 2 {
        return n;
    }
    let mut lo = 1u128;
    let mut hi = n;
    while lo < hi {
        let mid = lo + (hi - lo).div_ceil(2);
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

// ---------------------------------------------------------------------------
// Matmul
// ---------------------------------------------------------------------------

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

fn matmul_int(a: &Triple, b: &Triple, m_acc: i64) -> Result<Triple> {
    let ash = a.shape();
    let bsh = b.shape();
    if ash.len() < 2 || bsh.len() < 2 {
        return Err(Error::runtime(format!(
            "MATMUL: need rank >= 2, got {:?} x {:?}",
            ash, bsh
        )));
    }
    let n = ash[ash.len() - 2];
    let k = ash[ash.len() - 1];
    if bsh[bsh.len() - 2] != k {
        return Err(Error::runtime(format!(
            "MATMUL: inner dims {:?} vs {:?} (did you forget TRANSPOSE?)",
            ash, bsh
        )));
    }
    let m = *bsh.last().unwrap();
    let lead = &ash[..ash.len() - 2];
    let nb: usize = lead.iter().product();
    let nb_b: usize = bsh[..bsh.len() - 2].iter().product();
    if nb_b != 1 && nb_b != nb {
        return Err(Error::runtime(format!(
            "MATMUL: batch dims {:?} vs {:?}",
            lead,
            &bsh[..bsh.len() - 2]
        )));
    }
    let af = flat(&a.s);
    let aef = flat(&a.e);
    let azf = flat(&a.z);
    let bf = flat(&b.s);
    let bef = flat(&b.e);
    let bzf = flat(&b.z);

    let mut out_s = vec![0i8; nb * n * m];
    let mut out_e = vec![0i32; nb * n * m];
    let mut out_z = vec![0u8; nb * n * m];
    for bi in 0..nb {
        let bb = if nb_b == 1 { 0 } else { bi };
        let mut acc = vec![0i64; n * m];
        for i in 0..n {
            for kk in 0..k {
                let ai = bi * n * k + i * k + kk;
                let (as_, ae, az) = (af[ai], aef[ai], azf[ai]);
                for j in 0..m {
                    let bj = bb * k * m + kk * m + j;
                    let (bs, be, bz) = (bf[bj], bef[bj], bzf[bj]);
                    let ps = ((as_ as i16) * (bs as i16)) as i8;
                    let pe = (ae as i64 + be as i64 - BIAS).clamp(0, NLVL - 1) as i32;
                    let pz = az | bz;
                    let q = to_fixed_scalar(ps, pe, pz, m_acc);
                    acc[i * m + j] = acc[i * m + j].wrapping_add(q);
                }
            }
        }
        let acc_arr = from_flat(&[n, m], acc)?;
        lattice::assert_bound("matmul:acc", &[&acc_arr])?;
        let got = lattice::from_fixed(&acc_arr, m_acc)?;
        let gs = flat(&got.s);
        let ge = flat(&got.e);
        let gz = flat(&got.z);
        for t in 0..n * m {
            let o = (bi * n * m) + t;
            out_s[o] = gs[t];
            out_e[o] = ge[t];
            out_z[o] = gz[t];
        }
    }
    let mut osh = lead.to_vec();
    osh.push(n);
    osh.push(m);
    Triple::from_vecs(&osh, out_s, out_e, out_z)
}

/// Decode a triple stream back to integer counts. This is the bridge from the
/// triple transformer path to the integer harness/agent path.
pub fn op_t2i(args: &[Arg], _: &Config) -> Result<Vec<Value>> {
    let t = arg_triple(args, 0, "T2I")?;
    let lut = lattice::t2i_lut();
    let mut out = ArrayD::<i64>::zeros(t.s.raw_dim());
    for (idx, &zero) in t.z.indexed_iter() {
        let value = if zero != 0 {
            0
        } else {
            let e = t.e[idx.clone()];
            if !(0..lattice::NLVL).contains(&(e as i64)) {
                return Err(Error::runtime(format!(
                    "T2I: exponent {e} outside lattice range 0..{}",
                    lattice::NLVL
                )));
            }
            let magnitude = lut[e as usize];
            match t.s[idx.clone()] {
                s if s < 0 => -magnitude,
                0 => 0,
                _ => magnitude,
            }
        };
        out[idx.clone()] = value;
    }
    Ok(vec![Value::I64(out)])
}

/// F matmul: the `ikj` accumulation order the C emitter uses, with the same
/// leading-dimension broadcast rule as the triple path.
fn matmul_f64(a: &ArrayD<f64>, b: &ArrayD<f64>, op: &str) -> Result<ArrayD<f64>> {
    let ash = a.shape();
    let bsh = b.shape();
    if ash.len() < 2 || bsh.len() < 2 {
        return Err(Error::runtime(format!(
            "{op}: need rank >= 2, got {ash:?} x {bsh:?}"
        )));
    }
    let rank = ash.len();
    let n = ash[rank - 2];
    let k = ash[rank - 1];
    if bsh[bsh.len() - 2] != k {
        return Err(Error::runtime(format!(
            "{op}: inner dims {ash:?} vs {bsh:?} (did you forget TRANSPOSE?)"
        )));
    }
    let m = *bsh.last().unwrap();
    let lead = &ash[..rank - 2];
    let nb: usize = lead.iter().product();
    let nb_b: usize = bsh[..bsh.len() - 2].iter().product();
    if nb_b != 1 && nb_b != nb {
        return Err(Error::runtime(format!(
            "{op}: batch dims {lead:?} vs {:?}",
            &bsh[..bsh.len() - 2]
        )));
    }
    let af = flat(a);
    let bf = flat(b);
    let mut out = vec![0f64; nb * n * m];
    for bi in 0..nb {
        let bb = if nb_b == 1 { 0 } else { bi };
        for i in 0..n {
            for kk in 0..k {
                let aik = af[(bi * n + i) * k + kk];
                for j in 0..m {
                    out[(bi * n + i) * m + j] += aik * bf[(bb * k + kk) * m + j];
                }
            }
        }
    }
    let mut osh = lead.to_vec();
    osh.push(n);
    osh.push(m);
    from_flat(&osh, out)
}

fn op_matmul_any(args: &[Arg], config: &Config, mn: &str) -> Result<Vec<Value>> {
    let v = args
        .first()
        .ok_or_else(|| Error::runtime(format!("{mn}: missing lhs")))?
        .stream(mn, "a")?;
    if matches!(v, Value::Triple(_)) {
        let a = arg_triple(args, 0, mn)?;
        let b = arg_triple(args, 1, mn)?;
        let (m_acc, _) = scales(config);
        return Ok(vec![Value::Triple(matmul_int(a, b, m_acc)?)]);
    }
    let a = arg_f64_array(args, 0, mn)?;
    let b = arg_f64_array(args, 1, mn)?;
    Ok(vec![Value::F64(matmul_f64(&a, &b, mn)?)])
}

pub fn op_matmul(args: &[Arg], config: &Config) -> Result<Vec<Value>> {
    op_matmul_any(args, config, "MATMUL")
}

pub fn op_batch_matmul(args: &[Arg], config: &Config) -> Result<Vec<Value>> {
    op_matmul_any(args, config, "BATCH_MATMUL")
}

/// `C = A @ B + bias + residual`, all rank-2 float data.
pub fn op_matmul_bias_residual(args: &[Arg], _config: &Config) -> Result<Vec<Value>> {
    let a = arg_f64_array(args, 0, "MATMUL_BIAS_RESIDUAL")?;
    let b = arg_f64_array(args, 1, "MATMUL_BIAS_RESIDUAL")?;
    let bias = arg_f64_array(args, 2, "MATMUL_BIAS_RESIDUAL")?;
    let residual = arg_f64_array(args, 3, "MATMUL_BIAS_RESIDUAL")?;
    let c = matmul_f64(&a, &b, "MATMUL_BIAS_RESIDUAL")?;
    if residual.shape() != c.shape() {
        return Err(Error::runtime(format!(
            "MATMUL_BIAS_RESIDUAL: residual shape {:?} != output {:?}",
            residual.shape(),
            c.shape()
        )));
    }
    let n = c.shape()[1];
    if bias.len() != n {
        return Err(Error::runtime(format!(
            "MATMUL_BIAS_RESIDUAL: bias length {} != output width {n}",
            bias.len()
        )));
    }
    let bc = bias.broadcast(c.raw_dim()).ok_or_else(|| {
        Error::runtime("MATMUL_BIAS_RESIDUAL: bias is not broadcastable")
    })?;
    Ok(vec![Value::F64(&(&c + &bc) + &residual)])
}

/// `C = GELU(A @ B + bias)`.
pub fn op_matmul_bias_gelu(args: &[Arg], _config: &Config) -> Result<Vec<Value>> {
    let a = arg_f64_array(args, 0, "MATMUL_BIAS_GELU")?;
    let b = arg_f64_array(args, 1, "MATMUL_BIAS_GELU")?;
    let bias = arg_f64_array(args, 2, "MATMUL_BIAS_GELU")?;
    let c = matmul_f64(&a, &b, "MATMUL_BIAS_GELU")?;
    let n = c.shape()[1];
    if bias.len() != n {
        return Err(Error::runtime(format!(
            "MATMUL_BIAS_GELU: bias length {} != output width {n}",
            bias.len()
        )));
    }
    let bc = bias.broadcast(c.raw_dim()).ok_or_else(|| {
        Error::runtime("MATMUL_BIAS_GELU: bias is not broadcastable")
    })?;
    let v = &c + &bc;
    Ok(vec![Value::F64(v.mapv(gelu_approx_f64))])
}

/// `C = A @ B^T` for two rank-2 float matrices.
pub fn op_matmul_t(args: &[Arg], _config: &Config) -> Result<Vec<Value>> {
    let a = arg_f64_array(args, 0, "MATMUL_T")?;
    let b = arg_f64_array(args, 1, "MATMUL_T")?;
    if a.ndim() != 2 || b.ndim() != 2 {
        return Err(Error::runtime(format!(
            "MATMUL_T: need rank-2 A/B, got {:?}/{:?}",
            a.shape(),
            b.shape()
        )));
    }
    if a.shape()[1] != b.shape()[1] {
        return Err(Error::runtime(format!(
            "MATMUL_T: inner dims {:?} vs {:?}",
            a.shape(),
            b.shape()
        )));
    }
    let bt = b.t().to_owned();
    Ok(vec![Value::F64(matmul_f64(&a, &bt, "MATMUL_T")?)])
}

/// `C = A @ B + bias`, with a per-column bias on the output.
pub fn op_matmul_bias(args: &[Arg], _config: &Config) -> Result<Vec<Value>> {
    let a = arg_f64_array(args, 0, "MATMUL_BIAS")?;
    let b = arg_f64_array(args, 1, "MATMUL_BIAS")?;
    let bias = arg_f64_array(args, 2, "MATMUL_BIAS")?;
    let c = matmul_f64(&a, &b, "MATMUL_BIAS")?;
    let n = c.shape()[1];
    if bias.len() != n {
        return Err(Error::runtime(format!(
            "MATMUL_BIAS: bias length {} != output width {n}",
            bias.len()
        )));
    }
    let bc = bias.broadcast(c.raw_dim()).ok_or_else(|| {
        Error::runtime("MATMUL_BIAS: bias is not broadcastable to output")
    })?;
    Ok(vec![Value::F64(&c + &bc)])
}

// ---------------------------------------------------------------------------
// Norms
// ---------------------------------------------------------------------------

fn eps_counts(config: &Config, key_true: &str, key_counts: &str, default: i64, m: i64) -> i64 {
    if let Some(v) = config.get_f64(key_true) {
        let um = PHI.powf((m - BIAS) as f64 / lattice::K as f64);
        let raw = v * ((1i64 << 36) as f64) / (um * um);
        lattice::round_half_even(raw) as i64
    } else {
        config.get_i64(key_counts).unwrap_or(default)
    }
}

fn norm_rows(x: &Triple, w: &Triple, m: i64, eps_c: i64) -> Result<Triple> {
    let sh = x.shape();
    if sh.is_empty() {
        return Err(Error::runtime("RMSNORM: need rank >= 1"));
    }
    let c = last_dim(&sh);
    if w.numel() != c {
        return Err(Error::runtime(format!(
            "RMSNORM: weight length {} != {c}",
            w.numel()
        )));
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
    let out_arr = from_flat(&sh, out)?;
    lattice::assert_bound("rms:affine", &[&out_arr])?;
    lattice::from_fixed(&out_arr, m)
}

/// Row-wise F norm over the last axis, in the same operation order as the C
/// emitter: sequential sum, sequential squared deviations, reciprocal sqrt.
fn norm_rows_f64(
    x: &ArrayD<f64>,
    w: &ArrayD<f64>,
    b: Option<&ArrayD<f64>>,
    eps: f64,
    op: &str,
) -> Result<ArrayD<f64>> {
    let sh = x.shape();
    if sh.is_empty() {
        return Err(Error::runtime(format!("{op}: need rank >= 1")));
    }
    let d = sh[sh.len() - 1];
    if d == 0 {
        return Err(Error::runtime(format!("{op}: empty last axis")));
    }
    let wf = flat(w);
    if wf.len() != d {
        return Err(Error::runtime(format!(
            "{op}: weight length {} != {d}",
            wf.len()
        )));
    }
    let bf = b.map(flat);
    if let Some(bf) = &bf {
        if bf.len() != d {
            return Err(Error::runtime(format!(
                "{op}: bias length {} != {d}",
                bf.len()
            )));
        }
    }
    let xf = flat(x);
    let rows = xf.len() / d;
    let mut out = vec![0f64; xf.len()];
    if let Some(bf) = &bf {
        for r in 0..rows {
            let row = &xf[r * d..(r + 1) * d];
            let mut mu = 0f64;
            for &v in row {
                mu += v;
            }
            mu /= d as f64;
            let mut ss = 0f64;
            for &v in row {
                let dv = v - mu;
                ss += dv * dv;
            }
            let rs = 1.0 / (ss / d as f64 + eps).sqrt();
            for j in 0..d {
                out[r * d + j] = (row[j] - mu) * rs * wf[j] + bf[j];
            }
        }
    } else {
        for r in 0..rows {
            let row = &xf[r * d..(r + 1) * d];
            let mut ss = 0f64;
            for &v in row {
                ss += v * v;
            }
            let rs = 1.0 / (ss / d as f64 + eps).sqrt();
            for j in 0..d {
                out[r * d + j] = row[j] * rs * wf[j];
            }
        }
    }
    from_flat(sh, out)
}

pub fn op_rmsnorm(args: &[Arg], config: &Config) -> Result<Vec<Value>> {
    let v = args
        .first()
        .ok_or_else(|| Error::runtime("RMSNORM: missing x"))?
        .stream("RMSNORM", "x")?;
    if matches!(v, Value::Triple(_)) {
        let x = arg_triple(args, 0, "RMSNORM")?;
        let w = arg_triple(args, 1, "RMSNORM")?;
        let (_, m_cov) = scales(config);
        let eps = eps_counts(config, "eps_rms", "eps_rms_c", 4514, m_cov);
        return Ok(vec![Value::Triple(norm_rows(x, w, m_cov, eps)?)]);
    }
    let x = arg_f64_array(args, 0, "RMSNORM")?;
    let w = arg_f64_array(args, 1, "RMSNORM")?;
    let eps = config
        .get_f64("eps_rms")
        .ok_or_else(|| Error::runtime("RMSNORM: needs CONFIG eps_rms"))?;
    Ok(vec![Value::F64(norm_rows_f64(
        &x, &w, None, eps, "RMSNORM",
    )?)])
}

fn layernorm_rows(x: &Triple, w: &Triple, b: &Triple, m: i64, eps_c: i64) -> Result<Triple> {
    let sh = x.shape();
    if sh.len() != 2 {
        return Err(Error::runtime(format!(
            "LAYERNORM: need 2D (N,C), got {:?}",
            sh
        )));
    }
    let (nn, c) = (sh[0], sh[1]);
    if w.numel() != c || b.numel() != c {
        return Err(Error::runtime("LAYERNORM: weight/bias length mismatch"));
    }
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
    let out_arr = from_flat(&sh, out)?;
    lattice::assert_bound("ln:affine", &[&out_arr])?;
    lattice::from_fixed(&out_arr, m)
}

pub fn op_layernorm(args: &[Arg], config: &Config) -> Result<Vec<Value>> {
    let v = args
        .first()
        .ok_or_else(|| Error::runtime("LAYERNORM: missing x"))?
        .stream("LAYERNORM", "x")?;
    if matches!(v, Value::Triple(_)) {
        let x = arg_triple(args, 0, "LAYERNORM")?;
        let w = arg_triple(args, 1, "LAYERNORM")?;
        let b = arg_triple(args, 2, "LAYERNORM")?;
        let (_, m_cov) = scales(config);
        let eps = eps_counts(config, "eps_ln", "eps_ln_c", 4514, m_cov);
        return Ok(vec![Value::Triple(layernorm_rows(x, w, b, m_cov, eps)?)]);
    }
    let x = arg_f64_array(args, 0, "LAYERNORM")?;
    let w = arg_f64_array(args, 1, "LAYERNORM")?;
    let b = arg_f64_array(args, 2, "LAYERNORM")?;
    let eps = config
        .get_f64("eps_ln")
        .ok_or_else(|| Error::runtime("LAYERNORM: needs CONFIG eps_ln"))?;
    Ok(vec![Value::F64(norm_rows_f64(
        &x,
        &w,
        Some(&b),
        eps,
        "LAYERNORM",
    )?)])
}

// ---------------------------------------------------------------------------
// Softmax family
// ---------------------------------------------------------------------------

fn softmax_n_fixed(q: &ArrayD<i64>, m: i64) -> Result<(Vec<i64>, Vec<i64>, Vec<usize>)> {
    if m != BIAS {
        return Err(Error::runtime(format!(
            "softmax needs BIAS-scale counts, got m={m}"
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
        let mut dv = dsum as i64;
        if dv == 0 {
            dv = 1;
        }
        den[r] = dv;
    }
    Ok((num, den, sh))
}

fn softmax_from_q(q: &ArrayD<i64>) -> Result<Triple> {
    let (num, den, sh) = softmax_n_fixed(q, BIAS)?;
    let c = last_dim(&sh);
    let mut c18 = vec![0i64; num.len()];
    for (r, &d) in den.iter().enumerate() {
        let dd = if d == 0 { 1 } else { d };
        for j in 0..c {
            let v = (num[r * c + j] as i128 * (1i128 << 18)) as i64;
            c18[r * c + j] = lattice::tdiv(v, dd);
        }
    }
    let arr = from_flat(&sh, c18)?;
    lattice::from_fixed(&arr, BIAS)
}

pub fn op_softmax(args: &[Arg], _: &Config) -> Result<Vec<Value>> {
    let t = arg_triple(args, 0, "SOFTMAX")?;
    let q = lattice::to_fixed(t, BIAS);
    Ok(vec![Value::Triple(softmax_from_q(&q)?)])
}

pub fn op_softmax_wide(args: &[Arg], _: &Config) -> Result<Vec<Value>> {
    let t = arg_triple(args, 0, "SOFTMAX_WIDE")?;
    let q = lattice::to_fixed_wide(t, BIAS);
    Ok(vec![Value::Triple(softmax_from_q(&q)?)])
}

pub fn op_tshift(args: &[Arg], config: &Config) -> Result<Vec<Value>> {
    let t = arg_triple(args, 0, "TSHIFT")?;
    let (m_acc, _) = scales(config);
    let q = lattice::to_fixed_wide(t, m_acc);
    let sh = q.shape().to_vec();
    let c = last_dim(&sh);
    let rows = num_rows(&sh);
    let qf = flat(&q);
    let mut shifted = vec![0i64; qf.len()];
    for r in 0..rows {
        let mx = *qf[r * c..(r + 1) * c].iter().max().unwrap_or(&0);
        for j in 0..c {
            shifted[r * c + j] = qf[r * c + j].wrapping_sub(mx);
        }
    }
    let arr = from_flat(&sh, shifted)?;
    Ok(vec![Value::Triple(lattice::from_fixed(&arr, m_acc)?)])
}

// ---------------------------------------------------------------------------
// Activations / recurrence
// ---------------------------------------------------------------------------

pub fn op_silu(args: &[Arg], _: &Config) -> Result<Vec<Value>> {
    let t = arg_triple(args, 0, "SILU")?;
    let s = sigmoid_trip(t)?;
    Ok(vec![Value::Triple(tmul(&s, t)?)])
}

pub fn op_gelu(args: &[Arg], _: &Config) -> Result<Vec<Value>> {
    let t = arg_triple(args, 0, "GELU")?;
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
        let idx = (x14 + span).clamp(0, glut.len() as i64 - 1);
        let p14 = if x14 < -span {
            0
        } else if x14 > span {
            16384
        } else {
            glut[idx as usize]
        };
        let y14 = lattice::tdiv((x14 as i128 * p14 as i128) as i64, 16384);
        out.push(y14 * 16);
    }
    let arr = from_flat(&t.shape(), out)?;
    Ok(vec![Value::Triple(lattice::from_fixed(&arr, BIAS)?)])
}

pub fn op_scan(args: &[Arg], config: &Config) -> Result<Vec<Value>> {
    let h = arg_triple(args, 0, "SCAN")?;
    let abar = arg_triple(args, 1, "SCAN")?;
    let bx = arg_triple(args, 2, "SCAN")?;
    let (_, m_cov) = scales(config);
    let qh = lattice::to_fixed(h, m_cov);
    let qa = lattice::to_fixed(abar, BIAS);
    let qbx = lattice::to_fixed(bx, m_cov);
    if qh.shape() != qa.shape() || qh.shape() != qbx.shape() {
        return Err(Error::runtime("SCAN: shape mismatch"));
    }
    let mut out = Vec::with_capacity(qh.len());
    for ((&hh, &aa), &bb) in qh.iter().zip(qa.iter()).zip(qbx.iter()) {
        let prod = (aa as i128 * hh as i128) as i64;
        let v = lattice::tdiv(prod, 1 << 18).wrapping_add(bb);
        out.push(v);
    }
    let arr = from_flat(qh.shape(), out)?;
    lattice::assert_bound("scan:out", &[&arr])?;
    Ok(vec![Value::Triple(lattice::from_fixed(&arr, m_cov)?)])
}

// ---------------------------------------------------------------------------
// RoPE
// ---------------------------------------------------------------------------

fn rope_tables(max_pos: usize, dim: usize, base: f64) -> (ArrayD<f64>, ArrayD<f64>) {
    let half = dim / 2;
    let mut cos = vec![0f64; max_pos * half];
    let mut sin = vec![0f64; max_pos * half];
    for p in 0..max_pos {
        for i in 0..half {
            let theta = base.powf(-2.0 * i as f64 / dim as f64);
            let ang = p as f64 * theta;
            cos[p * half + i] = ang.cos();
            sin[p * half + i] = ang.sin();
        }
    }
    (
        from_flat(&[max_pos, half], cos).expect("rope cos"),
        from_flat(&[max_pos, half], sin).expect("rope sin"),
    )
}

pub fn op_rotary(args: &[Arg], config: &Config) -> Result<Vec<Value>> {
    let v = args
        .first()
        .ok_or_else(|| Error::runtime("ROTARY: missing x"))?
        .stream("ROTARY", "x")?;
    if !matches!(v, Value::Triple(_)) {
        let x = v.as_f64("ROTARY")?;
        let pos = arg_i64_array(args, 1, "ROTARY")?;
        let sh = x.shape().to_vec();
        if sh.is_empty() {
            return Err(Error::runtime("ROTARY: need rank >= 1"));
        }
        let d = sh[sh.len() - 1];
        if d % 2 != 0 {
            return Err(Error::runtime(format!(
                "ROTARY: head dim must be even, got {d}"
            )));
        }
        let rows: usize = sh[..sh.len() - 1].iter().product();
        if pos.len() != 1 && pos.len() != rows {
            return Err(Error::runtime(format!(
                "ROTARY: pos length {} not broadcastable to {} rows",
                pos.len(),
                rows
            )));
        }
        let base = config.get_f64_or("rope_base", 10000.0);
        let half = d / 2;
        let theta: Vec<f64> = (0..half)
            .map(|i| base.powf(-2.0 * i as f64 / d as f64))
            .collect();
        let xf = flat(&x);
        let mut out = vec![0f64; xf.len()];
        for r in 0..rows {
            let p = pos[if pos.len() == 1 { 0 } else { r }];
            if p < 0 {
                return Err(Error::runtime("ROTARY: negative position"));
            }
            for j in 0..half {
                let ang = p as f64 * theta[j];
                let c = ang.cos();
                let s = ang.sin();
                let i0 = r * d + 2 * j;
                let i1 = i0 + 1;
                out[i0] = xf[i0] * c - xf[i1] * s;
                out[i1] = xf[i0] * s + xf[i1] * c;
            }
        }
        return Ok(vec![Value::F64(from_flat(&sh, out)?)]);
    }
    let x = arg_triple(args, 0, "ROTARY")?;
    let pos_v = args
        .get(1)
        .ok_or_else(|| Error::runtime("ROTARY: missing pos"))?
        .stream("ROTARY", "pos")?
        .as_i64("ROTARY")?;
    let (_, m_cov) = scales(config);
    let sh = x.shape();
    let d = *sh
        .last()
        .ok_or_else(|| Error::runtime("ROTARY: need rank >= 1"))?;
    if d % 2 != 0 {
        return Err(Error::runtime(format!(
            "ROTARY: head dim must be even, got {d}"
        )));
    }
    let pmax = pos_v.iter().copied().max().unwrap_or(0) + 1;
    let base = config.get_f64_or("rope_base", 10000.0);
    let (cos_t, sin_t) = rope_tables(pmax.max(0) as usize, d, base);
    let pid = flat(&pos_v);
    let half = d / 2;
    // Build broadcast cos/sin of shape x.shape() by indexing pos over the
    // leading dimensions.
    let lead = &sh[..sh.len() - 1];
    let lead_n: usize = lead.iter().product();
    let xf_s = flat(&x.s);
    let xf_e = flat(&x.e);
    let xf_z = flat(&x.z);
    let mut xs0 = vec![0i8; lead_n * half];
    let mut xe0 = vec![0i32; lead_n * half];
    let mut xz0 = vec![0u8; lead_n * half];
    let mut xs1 = vec![0i8; lead_n * half];
    let mut xe1 = vec![0i32; lead_n * half];
    let mut xz1 = vec![0u8; lead_n * half];
    for n in 0..lead_n {
        for j in 0..half {
            let src = n * d + 2 * j;
            xs0[n * half + j] = xf_s[src];
            xe0[n * half + j] = xf_e[src];
            xz0[n * half + j] = xf_z[src];
            xs1[n * half + j] = xf_s[src + 1];
            xe1[n * half + j] = xf_e[src + 1];
            xz1[n * half + j] = xf_z[src + 1];
        }
    }
    let mut cs = vec![0i8; lead_n * half];
    let mut ce = vec![0i32; lead_n * half];
    let mut cz = vec![0u8; lead_n * half];
    let mut ss = vec![0i8; lead_n * half];
    let mut se = vec![0i32; lead_n * half];
    let mut sz = vec![0u8; lead_n * half];
    let cosf = flat(&cos_t);
    let sinf = flat(&sin_t);
    let pid = if pid.len() == 1 {
        vec![pid[0]; lead_n]
    } else {
        pid
    };
    if pid.len() != lead_n {
        return Err(Error::runtime(format!(
            "ROTARY: pos length {} not broadcastable to {} rows",
            pid.len(),
            lead_n
        )));
    }
    for n in 0..lead_n {
        if pid[n] < 0 {
            return Err(Error::runtime("ROTARY: negative position"));
        }
        let p = pid[n] as usize;
        for j in 0..half {
            let cidx = p * half + j;
            let c = lattice::encode(&ArrayD::from_elem(IxDyn(&[1]), cosf[cidx]));
            let s = lattice::encode(&ArrayD::from_elem(IxDyn(&[1]), sinf[cidx]));
            cs[n * half + j] = *c.s.iter().next().unwrap_or(&0);
            ce[n * half + j] = *c.e.iter().next().unwrap_or(&0);
            cz[n * half + j] = *c.z.iter().next().unwrap_or(&0);
            ss[n * half + j] = *s.s.iter().next().unwrap_or(&0);
            se[n * half + j] = *s.e.iter().next().unwrap_or(&0);
            sz[n * half + j] = *s.z.iter().next().unwrap_or(&0);
        }
    }
    let x0 = Triple::from_vecs(&[lead_n, half], xs0, xe0, xz0)?;
    let x1 = Triple::from_vecs(&[lead_n, half], xs1, xe1, xz1)?;
    let c = Triple::from_vecs(&[lead_n, half], cs, ce, cz)?;
    let s = Triple::from_vecs(&[lead_n, half], ss, se, sz)?;
    let y0 = binop_fixed(&tmul(&x0, &c)?, &tmul(&x1, &s)?, m_cov, m_cov, "sub")?;
    let y1 = binop_fixed(&tmul(&x0, &s)?, &tmul(&x1, &c)?, m_cov, m_cov, "add")?;
    let mut os = vec![0i8; lead_n * d];
    let mut oe = vec![0i32; lead_n * d];
    let mut oz = vec![0u8; lead_n * d];
    let y0s = flat(&y0.s);
    let y0e = flat(&y0.e);
    let y0z = flat(&y0.z);
    let y1s = flat(&y1.s);
    let y1e = flat(&y1.e);
    let y1z = flat(&y1.z);
    for n in 0..lead_n {
        for j in 0..half {
            os[n * d + 2 * j] = y0s[n * half + j];
            oe[n * d + 2 * j] = y0e[n * half + j];
            oz[n * d + 2 * j] = y0z[n * half + j];
            os[n * d + 2 * j + 1] = y1s[n * half + j];
            oe[n * d + 2 * j + 1] = y1e[n * half + j];
            oz[n * d + 2 * j + 1] = y1z[n * half + j];
        }
    }
    Ok(vec![Value::Triple(Triple::from_vecs(&sh, os, oe, oz)?)])
}

pub fn op_rope(args: &[Arg], config: &Config) -> Result<Vec<Value>> {
    let v = args
        .first()
        .ok_or_else(|| Error::runtime("ROPE: missing input"))?
        .stream("ROPE", "x")?;
    if matches!(v, Value::Triple(_)) {
        return Err(Error::runtime("ROPE: T triples unbuilt (decode path is F)"));
    }
    let a = v.as_f64("ROPE")?;
    let pos = arg_i64_array(args, 1, "ROPE")?;
    let nh = arg_i64(args, 2, "ROPE", "nh")? as usize;
    let dh = arg_i64(args, 3, "ROPE", "dh")? as usize;
    if a.ndim() != 2 || a.shape()[1] != nh * dh || !dh.is_multiple_of(2) {
        return Err(Error::runtime(format!(
            "ROPE: needs 2D (R,{}) dh-even, got {:?} dh={dh}",
            nh * dh,
            a.shape()
        )));
    }
    let posf = flat(&pos);
    if posf.len() != a.shape()[0] {
        return Err(Error::runtime(format!(
            "ROPE: pos len {} vs rows {}",
            posf.len(),
            a.shape()[0]
        )));
    }
    let base = config.get_f64_or("rope_base", 10000.0);
    let th: Vec<f64> = (0..dh / 2)
        .map(|i| base.powf(-2.0 * i as f64 / dh as f64))
        .collect();
    let af = flat(&a);
    let mut out = vec![0f64; af.len()];
    let rows = a.shape()[0];
    for r in 0..rows {
        for h in 0..nh {
            for p in 0..dh / 2 {
                let ang = posf[r] as f64 * th[p];
                let (sn, cs) = ang.sin_cos();
                let i0 = r * nh * dh + h * dh + 2 * p;
                let i1 = i0 + 1;
                out[i0] = af[i0] * cs - af[i1] * sn;
                out[i1] = af[i0] * sn + af[i1] * cs;
            }
        }
    }
    Ok(vec![Value::F64(from_flat(a.shape(), out)?)])
}

// ---------------------------------------------------------------------------
// Moves / gather / masks
// ---------------------------------------------------------------------------

fn permute_last2<T: Copy>(a: &ArrayD<T>) -> Result<ArrayD<T>> {
    let sh = a.shape();
    if sh.len() < 2 {
        return Err(Error::runtime(format!(
            "TRANSPOSE: need rank >= 2, got {:?}",
            sh
        )));
    }
    let mut axes: Vec<usize> = (0..sh.len()).collect();
    let n = axes.len();
    axes.swap(n - 1, n - 2);
    Ok(a.view().permuted_axes(IxDyn(&axes)).to_owned())
}

pub fn op_transpose(args: &[Arg], _: &Config) -> Result<Vec<Value>> {
    let t = arg_triple(args, 0, "TRANSPOSE")?;
    Ok(vec![Value::Triple(Triple::new(
        permute_last2(&t.s)?,
        permute_last2(&t.e)?,
        permute_last2(&t.z)?,
    )?)])
}

fn gather_triple(t: &Triple, ids: &ArrayD<i64>) -> Result<Triple> {
    let sh = t.shape();
    if sh.len() != 2 {
        return Err(Error::runtime(format!(
            "GATHER: table must be (V,C), got {:?}",
            sh
        )));
    }
    let (v, c) = (sh[0], sh[1]);
    let mut ishape = ids.shape().to_vec();
    let n: usize = ishape.iter().product();
    let mut oshape = ishape.clone();
    oshape.push(c);
    let mut s = vec![0i8; n * c];
    let mut e = vec![0i32; n * c];
    let mut z = vec![0u8; n * c];
    let ts = flat(&t.s);
    let te = flat(&t.e);
    let tz = flat(&t.z);
    for (nidx, &id) in ids.iter().enumerate() {
        if id < 0 || id as usize >= v {
            return Err(Error::runtime(format!("GATHER: ids out of range [0,{v})")));
        }
        let row = id as usize;
        for j in 0..c {
            s[nidx * c + j] = ts[row * c + j];
            e[nidx * c + j] = te[row * c + j];
            z[nidx * c + j] = tz[row * c + j];
        }
    }
    ishape.clear();
    Triple::from_vecs(&oshape, s, e, z)
}

fn gather_rows<T: Copy>(t: &ArrayD<T>, ids: &ArrayD<i64>, op: &str) -> Result<ArrayD<T>> {
    let sh = t.shape();
    if sh.len() != 2 {
        return Err(Error::runtime(format!(
            "{op}: table must be (V,C), got {sh:?}"
        )));
    }
    let (v, c) = (sh[0], sh[1]);
    let tf = flat(t);
    let mut oshape = if ids.ndim() == 0 {
        vec![1]
    } else {
        ids.shape().to_vec()
    };
    oshape.push(c);
    let mut out = Vec::with_capacity(ids.len() * c);
    for &id in ids.iter() {
        if id < 0 || id as usize >= v {
            return Err(Error::runtime(format!("{op}: ids out of range [0,{v})")));
        }
        let row = id as usize;
        out.extend_from_slice(&tf[row * c..(row + 1) * c]);
    }
    from_flat(&oshape, out)
}

pub fn op_gather(args: &[Arg], _: &Config) -> Result<Vec<Value>> {
    let w = args
        .first()
        .ok_or_else(|| Error::runtime("GATHER: missing table"))?
        .stream("GATHER", "table")?;
    let ids = arg_i64_array(args, 1, "GATHER")?;
    match w {
        Value::Triple(t) => Ok(vec![Value::Triple(gather_triple(t, &ids)?)]),
        Value::F64(_) | Value::F32(_) => {
            let t = w.as_f64("GATHER")?;
            Ok(vec![Value::F64(gather_rows(&t, &ids, "GATHER")?)])
        }
        Value::I64(_) => {
            let t = w.as_i64("GATHER")?;
            Ok(vec![Value::I64(gather_rows(&t, &ids, "GATHER")?)])
        }
        other => Err(Error::runtime(format!(
            "GATHER: table must be T/F/I, got {}",
            other.kind()
        ))),
    }
}

pub fn op_gather3(args: &[Arg], _: &Config) -> Result<Vec<Value>> {
    let w = args
        .first()
        .ok_or_else(|| Error::runtime("GATHER3: missing table"))?
        .stream("GATHER3", "table")?;
    let ii = arg_i64_array(args, 1, "GATHER3")?;
    let jj = arg_i64_array(args, 2, "GATHER3")?;
    let shape = w
        .shape()
        .ok_or_else(|| Error::runtime("GATHER3: table is None"))?;
    if shape.len() != 3 {
        return Err(Error::runtime(format!(
            "GATHER3: table must be 3D (V1,V2,C), got {:?}",
            shape
        )));
    }
    if ii.ndim() != 1 || jj.ndim() != 1 || ii.shape() != jj.shape() {
        return Err(Error::runtime(format!(
            "GATHER3: i/j must be 1D same length, got {:?} vs {:?}",
            ii.shape(),
            jj.shape()
        )));
    }
    let (v1, v2, c) = (shape[0], shape[1], shape[2]);
    let n = ii.len();
    let mut s = vec![0i8; n * c];
    let mut e = vec![0i32; n * c];
    let mut z = vec![0u8; n * c];
    match w {
        Value::Triple(t) => {
            let ts = flat(&t.s);
            let te = flat(&t.e);
            let tz = flat(&t.z);
            for nidx in 0..n {
                let iv = ii[nidx];
                let jv = jj[nidx];
                if iv < 0 || iv as usize >= v1 {
                    return Err(Error::runtime(format!("GATHER3: i out of range [0,{v1})")));
                }
                if jv < 0 || jv as usize >= v2 {
                    return Err(Error::runtime(format!("GATHER3: j out of range [0,{v2})")));
                }
                let base = (iv as usize * v2 + jv as usize) * c;
                for k in 0..c {
                    s[nidx * c + k] = ts[base + k];
                    e[nidx * c + k] = te[base + k];
                    z[nidx * c + k] = tz[base + k];
                }
            }
            Ok(vec![Value::Triple(Triple::from_vecs(&[n, c], s, e, z)?)])
        }
        other => {
            let f = other.as_f64("GATHER3")?;
            let ff = flat(&f);
            let mut out = vec![0f64; n * c];
            for nidx in 0..n {
                let iv = ii[nidx];
                let jv = jj[nidx];
                if iv < 0 || iv as usize >= v1 || jv < 0 || jv as usize >= v2 {
                    return Err(Error::runtime("GATHER3: id out of range"));
                }
                let base = (iv as usize * v2 + jv as usize) * c;
                for k in 0..c {
                    out[nidx * c + k] = ff[base + k];
                }
            }
            Ok(vec![Value::F64(from_flat(&[n, c], out)?)])
        }
    }
}

fn concat_axis<T: Copy>(a: &ArrayD<T>, b: &ArrayD<T>, axis: usize) -> Result<ArrayD<T>> {
    let ash = a.shape();
    let bsh = b.shape();
    if ash.len() != bsh.len() || axis >= ash.len() {
        return Err(Error::runtime(format!(
            "CONCAT: bad axis {axis} for ndim {:?}/{:?}",
            ash, bsh
        )));
    }
    for i in 0..ash.len() {
        if i != axis && ash[i] != bsh[i] {
            return Err(Error::runtime(format!(
                "CONCAT: non-axis dims differ {:?} vs {:?}",
                ash, bsh
            )));
        }
    }
    let mut osh = ash.to_vec();
    osh[axis] = ash[axis] + bsh[axis];
    let inner: usize = osh[axis + 1..].iter().product();
    let outer: usize = osh[..axis].iter().product();
    let a_inner = ash[axis] * inner;
    let b_inner = bsh[axis] * inner;
    let af = flat(a);
    let bf = flat(b);
    let mut out = Vec::with_capacity(osh.iter().product());
    for o in 0..outer {
        out.extend_from_slice(&af[o * a_inner..(o + 1) * a_inner]);
        out.extend_from_slice(&bf[o * b_inner..(o + 1) * b_inner]);
    }
    from_flat(&osh, out)
}

pub fn op_concat(args: &[Arg], _: &Config) -> Result<Vec<Value>> {
    let axis = arg_i64(args, 2, "CONCAT", "axis")?;
    if axis < 0 {
        return Err(Error::runtime("CONCAT: negative axis"));
    }
    let a = args
        .first()
        .ok_or_else(|| Error::runtime("CONCAT: missing lhs"))?
        .stream("CONCAT", "a")?;
    let b = args
        .get(1)
        .ok_or_else(|| Error::runtime("CONCAT: missing rhs"))?
        .stream("CONCAT", "b")?;
    let axis = axis as usize;
    match (a, b) {
        (Value::Triple(x), Value::Triple(y)) => Ok(vec![Value::Triple(Triple::new(
            concat_axis(&x.s, &y.s, axis)?,
            concat_axis(&x.e, &y.e, axis)?,
            concat_axis(&x.z, &y.z, axis)?,
        )?)]),
        (Value::F64(x), Value::F64(y)) => Ok(vec![Value::F64(concat_axis(x, y, axis)?)]),
        (Value::F32(x), Value::F32(y)) => Ok(vec![Value::F32(concat_axis(x, y, axis)?)]),
        (Value::I64(x), Value::I64(y)) => Ok(vec![Value::I64(concat_axis(x, y, axis)?)]),
        (Value::U8(x), Value::U8(y)) => Ok(vec![Value::U8(concat_axis(x, y, axis)?)]),
        _ => Err(Error::runtime("CONCAT: operands must be the same kind")),
    }
}

fn reshape_value(v: &Value, dims: &[usize], op: &str) -> Result<Value> {
    let n: usize = dims.iter().product();
    match v {
        Value::Triple(t) => {
            if t.numel() != n {
                return Err(Error::runtime(format!(
                    "{op}: {dims:?} != {} elements",
                    t.numel()
                )));
            }
            Ok(Value::Triple(Triple::from_vecs(
                dims,
                flat(&t.s),
                flat(&t.e),
                flat(&t.z),
            )?))
        }
        Value::F64(a) => {
            if a.len() != n {
                return Err(Error::runtime(format!(
                    "{op}: {dims:?} != {} elements",
                    a.len()
                )));
            }
            Ok(Value::F64(from_flat(dims, flat(a))?))
        }
        Value::I64(a) => {
            if a.len() != n {
                return Err(Error::runtime(format!(
                    "{op}: {dims:?} != {} elements",
                    a.len()
                )));
            }
            Ok(Value::I64(from_flat(dims, flat(a))?))
        }
        other => Err(Error::runtime(format!(
            "{op}: needs T/F/I, got {}",
            other.kind()
        ))),
    }
}

pub fn op_reshape2(args: &[Arg], _: &Config) -> Result<Vec<Value>> {
    let v = args
        .first()
        .ok_or_else(|| Error::runtime("RESHAPE2: missing input"))?
        .stream("RESHAPE2", "x")?;
    let d0 = arg_i64(args, 1, "RESHAPE2", "d0")? as usize;
    let d1 = arg_i64(args, 2, "RESHAPE2", "d1")? as usize;
    Ok(vec![reshape_value(v, &[d0, d1], "RESHAPE2")?])
}

pub fn op_reshape3(args: &[Arg], _: &Config) -> Result<Vec<Value>> {
    let v = args
        .first()
        .ok_or_else(|| Error::runtime("RESHAPE3: missing input"))?
        .stream("RESHAPE3", "x")?;
    let d: Vec<usize> = (1..=3)
        .map(|i| arg_i64(args, i, "RESHAPE3", "d").map(|v| v as usize))
        .collect::<Result<_>>()?;
    Ok(vec![reshape_value(v, &d, "RESHAPE3")?])
}

pub fn op_permute3(args: &[Arg], _: &Config) -> Result<Vec<Value>> {
    let t = arg_triple(args, 0, "PERMUTE3")?;
    let o: Vec<usize> = (1..=3)
        .map(|i| arg_i64(args, i, "PERMUTE3", "o").map(|v| v as usize))
        .collect::<Result<_>>()?;
    if t.s.ndim() != 3 || {
        let mut s = o.clone();
        s.sort_unstable();
        s != [0, 1, 2]
    } {
        return Err(Error::runtime(format!(
            "PERMUTE3: need a permutation of (0,1,2) on 3D input, got {o:?} ndim={}",
            t.s.ndim()
        )));
    }
    Ok(vec![Value::Triple(Triple::new(
        t.s.view().permuted_axes(IxDyn(&o)).to_owned(),
        t.e.view().permuted_axes(IxDyn(&o)).to_owned(),
        t.z.view().permuted_axes(IxDyn(&o)).to_owned(),
    )?)])
}

fn slice_axis<T: Copy>(a: &ArrayD<T>, axis: usize, start: usize, end: usize) -> Result<ArrayD<T>> {
    let sh = a.shape();
    if axis >= sh.len() || start > end || end > sh[axis] {
        return Err(Error::runtime(format!(
            "SLICE: [{start},{end}) out of range dim {axis}={}",
            sh.get(axis).copied().unwrap_or(0)
        )));
    }
    let inner: usize = sh[axis + 1..].iter().product();
    let outer: usize = sh[..axis].iter().product();
    let af = flat(a);
    let mut osh = sh.to_vec();
    osh[axis] = end - start;
    let mut out = Vec::with_capacity(osh.iter().product());
    for o in 0..outer {
        let base = o * sh[axis] * inner;
        out.extend_from_slice(&af[base + start * inner..base + end * inner]);
    }
    from_flat(&osh, out)
}

pub fn op_slice(args: &[Arg], _: &Config) -> Result<Vec<Value>> {
    let v = args
        .first()
        .ok_or_else(|| Error::runtime("SLICE: missing input"))?
        .stream("SLICE", "x")?;
    let axis = arg_i64(args, 1, "SLICE", "axis")? as usize;
    let start = arg_i64(args, 2, "SLICE", "start")? as usize;
    let end = arg_i64(args, 3, "SLICE", "end")? as usize;
    if let Value::I64(a) = v {
        if a.ndim() == 1 {
            return Err(Error::runtime(
                "SLICE: 1D I is not supported; use IWINDOW for 1D I",
            ));
        }
    }
    match v {
        Value::Triple(t) => Ok(vec![Value::Triple(Triple::new(
            slice_axis(&t.s, axis, start, end)?,
            slice_axis(&t.e, axis, start, end)?,
            slice_axis(&t.z, axis, start, end)?,
        )?)]),
        Value::F64(a) => Ok(vec![Value::F64(slice_axis(a, axis, start, end)?)]),
        Value::F32(a) => Ok(vec![Value::F32(slice_axis(a, axis, start, end)?)]),
        Value::I64(a) => Ok(vec![Value::I64(slice_axis(a, axis, start, end)?)]),
        Value::U8(a) => Ok(vec![Value::U8(slice_axis(a, axis, start, end)?)]),
        Value::Bool(a) => Ok(vec![Value::Bool(slice_axis(a, axis, start, end)?)]),
        Value::None => Err(Error::runtime("SLICE: input is None")),
    }
}

pub fn op_scatter(args: &[Arg], _: &Config) -> Result<Vec<Value>> {
    let base = arg_f64_array(args, 0, "SCATTER")?;
    let row = arg_f64_array(args, 1, "SCATTER")?;
    let idx = arg_i64_array(args, 2, "SCATTER")?;
    if base.ndim() != 2 || row.shape() != [1, base.shape()[1]] {
        return Err(Error::runtime(format!(
            "SCATTER: base {:?} + row {:?} mismatch",
            base.shape(),
            row.shape()
        )));
    }
    if idx.len() != 1 {
        return Err(Error::runtime("SCATTER: idx must be scalar/1-elem"));
    }
    let n = idx[0];
    if n < 0 || n as usize >= base.shape()[0] {
        return Err(Error::runtime(format!(
            "SCATTER: idx {n} out of [0,{})",
            base.shape()[0]
        )));
    }
    let mut out = base.clone();
    for j in 0..base.shape()[1] {
        out[[n as usize, j]] = row[[0, j]];
    }
    Ok(vec![Value::F64(out)])
}

pub fn op_argmax(args: &[Arg], _: &Config) -> Result<Vec<Value>> {
    let v = args
        .first()
        .ok_or_else(|| Error::runtime("ARGMAX: missing input"))?
        .stream("ARGMAX", "x")?;
    let axis = if args.len() > 1 {
        arg_i64(args, 1, "ARGMAX", "axis")?
    } else {
        -1
    };
    if let Value::Triple(t) = v {
        let sh = t.shape();
        if sh.is_empty() {
            return Err(Error::runtime("ARGMAX: need rank >= 1"));
        }
        let ax = ((axis % sh.len() as i64) + sh.len() as i64) as usize % sh.len();
        let outer: usize = sh[..ax].iter().product();
        let inner: usize = sh[ax + 1..].iter().product();
        let n = sh[ax];
        let mut out = vec![0i64; outer * inner];
        let ts = flat(&t.s);
        let te = flat(&t.e);
        let tz = flat(&t.z);
        for o in 0..outer {
            for i in 0..inner {
                let mut best: Option<(i64, i64, usize)> = None;
                for j in 0..n {
                    let idx = (o * n + j) * inner + i;
                    let s = ts[idx];
                    let e = te[idx];
                    let z = tz[idx];
                    let (cls, key) = if z != 0 {
                        (1, 0)
                    } else if s > 0 {
                        (2, e as i64)
                    } else {
                        (0, -(e as i64))
                    };
                    let better = match best {
                        None => true,
                        Some((bc, bk, _)) => cls > bc || (cls == bc && key > bk),
                    };
                    if better {
                        best = Some((cls, key, j));
                    }
                }
                out[o * inner + i] = best.map(|b| b.2 as i64).unwrap_or(0);
            }
        }
        let mut oshape: Vec<usize> = sh.clone();
        oshape.remove(ax);
        return Ok(vec![Value::I64(from_flat(&oshape, out)?)]);
    }
    let a = v.as_f64("ARGMAX")?;
    let sh = a.shape().to_vec();
    if sh.is_empty() {
        return Err(Error::runtime("ARGMAX: need rank >= 1"));
    }
    let ax = ((axis % sh.len() as i64) + sh.len() as i64) as usize % sh.len();
    let outer: usize = sh[..ax].iter().product();
    let inner: usize = sh[ax + 1..].iter().product();
    let n = sh[ax];
    let af = flat(&a);
    let mut out = vec![0i64; outer * inner];
    for o in 0..outer {
        for i in 0..inner {
            let mut best = f64::NEG_INFINITY;
            let mut bi = 0usize;
            for j in 0..n {
                let val = af[(o * n + j) * inner + i];
                if j == 0 || val > best {
                    best = val;
                    bi = j;
                }
            }
            out[o * inner + i] = bi as i64;
        }
    }
    let mut oshape = sh.clone();
    oshape.remove(ax);
    Ok(vec![Value::I64(from_flat(&oshape, out)?)])
}

pub fn op_topk(args: &[Arg], _: &Config) -> Result<Vec<Value>> {
    let v = args
        .first()
        .ok_or_else(|| Error::runtime("TOPK: missing input"))?
        .stream("TOPK", "x")?;
    if matches!(v, Value::Triple(_)) {
        return Err(Error::runtime("TOPK: T triples unbuilt (scores are F)"));
    }
    let k = arg_i64(args, 1, "TOPK", "k")?;
    let a = v.as_f64("TOPK")?;
    if a.ndim() != 2 {
        return Err(Error::runtime(format!(
            "TOPK: needs 2D, got {:?}",
            a.shape()
        )));
    }
    let (r, s) = (a.shape()[0], a.shape()[1]);
    if k < 1 || k as usize > s {
        return Err(Error::runtime(format!(
            "TOPK: k={k} out of range (1,S={s})"
        )));
    }
    let k = k as usize;
    let af = flat(&a);
    let mut vals = vec![0f64; r * k];
    let mut idxs = vec![0i64; r * k];
    for row in 0..r {
        // The C emitter's single-pass insertion: strictly greater shifts left,
        // so equal scores keep the lower index first. The arena kernel uses the
        // same loop, which is what makes ties and non-finite scores identical
        // without sorting (and without an allocation).
        let mut n = 0usize;
        for c in 0..s {
            let v = af[row * s + c];
            let mut p = if n < k { n } else { k };
            while p > 0 && v > vals[row * k + p - 1] {
                p -= 1;
            }
            if p >= k {
                continue;
            }
            let m = if n < k { n } else { k - 1 };
            let mut t = m;
            while t > p {
                vals[row * k + t] = vals[row * k + t - 1];
                idxs[row * k + t] = idxs[row * k + t - 1];
                t -= 1;
            }
            vals[row * k + p] = v;
            idxs[row * k + p] = c as i64;
            if n < k {
                n += 1;
            }
        }
    }
    Ok(vec![
        Value::F64(from_flat(&[r, k], vals)?),
        Value::I64(from_flat(&[r, k], idxs)?),
    ])
}

// ---------------------------------------------------------------------------
// Fused decode-path ops (F-only)
// ---------------------------------------------------------------------------

pub fn op_swiglu(args: &[Arg], _: &Config) -> Result<Vec<Value>> {
    let v = args
        .first()
        .ok_or_else(|| Error::runtime("SWIGLU: missing input"))?
        .stream("SWIGLU", "x")?;
    if matches!(v, Value::Triple(_)) {
        return Err(Error::runtime(
            "SWIGLU: T triples unbuilt (decode path is F)",
        ));
    }
    let inter = arg_i64(args, 1, "SWIGLU", "inter")? as usize;
    let a = v.as_f64("SWIGLU")?;
    if a.ndim() != 2 || a.shape()[1] != 2 * inter {
        return Err(Error::runtime(format!(
            "SWIGLU: needs 2D (R,{}), got {:?}",
            2 * inter,
            a.shape()
        )));
    }
    let (r, _) = (a.shape()[0], a.shape()[1]);
    let af = flat(&a);
    let mut out = vec![0f64; r * inter];
    for row in 0..r {
        for j in 0..inter {
            let up = af[row * 2 * inter + j];
            let gate = af[row * 2 * inter + inter + j];
            let gs = gate / (1.0 + (-gate).exp());
            out[row * inter + j] = gs * up;
        }
    }
    Ok(vec![Value::F64(from_flat(&[r, inter], out)?)])
}

pub fn op_attn(args: &[Arg], config: &Config) -> Result<Vec<Value>> {
    let qr = arg_f64_array(args, 0, "ATTN")?;
    let kf = arg_f64_array(args, 1, "ATTN")?;
    let vf = arg_f64_array(args, 2, "ATTN")?;
    let mask = arg_i64_array(args, 3, "ATTN")?;
    let nh = arg_i64(args, 4, "ATTN", "nh")? as usize;
    let nkv = arg_i64(args, 5, "ATTN", "nkv")? as usize;
    let dh = arg_i64(args, 6, "ATTN", "dh")? as usize;
    let s = arg_i64(args, 7, "ATTN", "S")? as usize;
    if !nh.is_multiple_of(nkv) {
        return Err(Error::runtime(format!(
            "ATTN: nh={nh} not a multiple of nkv={nkv}"
        )));
    }
    let per = nh / nkv;
    let (rows, qcols) = (qr.shape()[0], qr.shape()[1]);
    if qcols != nh * dh {
        return Err(Error::runtime(format!(
            "ATTN: q shape {:?} vs (R,{})",
            qr.shape(),
            nh * dh
        )));
    }
    let mask_sh = mask.shape();
    if mask_sh.len() != 2 || mask_sh[0] != per || mask_sh[1] != s {
        return Err(Error::runtime(format!(
            "ATTN: mask shape {:?} vs ({per},{s})",
            mask_sh
        )));
    }
    let q = flat(&qr);
    let k = flat(&kf);
    let v = flat(&vf);
    let m = flat(&mask);
    if k.len() != s * nkv * dh || v.len() != s * nkv * dh {
        return Err(Error::runtime(format!(
            "ATTN: K/V must be ({s},{})",
            nkv * dh
        )));
    }
    let neg = config.get_f64_or("beta", -1e9);
    let mut out = vec![0f64; rows * nh * dh];
    for g in 0..nkv {
        let mut scores = vec![0f64; rows * per * s];
        for r in 0..rows {
            for hh in 0..per {
                let head = g * per + hh;
                for t in 0..s {
                    let mut acc = 0.0;
                    for d in 0..dh {
                        acc += q[r * nh * dh + head * dh + d] * k[t * nkv * dh + g * dh + d];
                    }
                    scores[(r * per + hh) * s + t] = acc;
                }
            }
        }
        for r in 0..rows {
            for hh in 0..per {
                let base = (r * per + hh) * s;
                for t in 0..s {
                    if m[hh * s + t] == 0 {
                        scores[base + t] = neg;
                    }
                }
                let mx = scores[base..base + s]
                    .iter()
                    .copied()
                    .fold(f64::NEG_INFINITY, f64::max);
                let mut sum = 0.0;
                for t in 0..s {
                    let e = (scores[base + t] - mx).exp();
                    scores[base + t] = e;
                    sum += e;
                }
                for t in 0..s {
                    scores[base + t] /= sum;
                }
                for d in 0..dh {
                    let mut acc = 0.0;
                    for t in 0..s {
                        acc += scores[base + t] * v[t * nkv * dh + g * dh + d];
                    }
                    out[r * nh * dh + (g * per + hh) * dh + d] = acc;
                }
            }
        }
    }
    Ok(vec![Value::F64(from_flat(&[rows, nh * dh], out)?)])
}

pub fn op_self_attn(args: &[Arg], _config: &Config) -> Result<Vec<Value>> {
    let q = arg_f64_array(args, 0, "SELF_ATTN")?;
    let k = arg_f64_array(args, 1, "SELF_ATTN")?;
    let v = arg_f64_array(args, 2, "SELF_ATTN")?;
    let nh = arg_i64(args, 3, "SELF_ATTN", "nh")?;
    let dh = arg_i64(args, 4, "SELF_ATTN", "dh")?;
    let scale = arg_f64(args, 5, "SELF_ATTN", "scale")?;
    if nh <= 0 || dh <= 0 || !scale.is_finite() {
        return Err(Error::runtime("SELF_ATTN: bad nh/dh/scale"));
    }
    let (nh, dh) = (nh as usize, dh as usize);
    let qsh = q.shape();
    if qsh.len() != 2 || k.shape() != qsh || v.shape() != qsh {
        return Err(Error::runtime(format!(
            "SELF_ATTN: need equal 2D Q/K/V, got {:?}/{:?}/{:?}",
            qsh,
            k.shape(),
            v.shape()
        )));
    }
    let (n, h) = (qsh[0], qsh[1]);
    if h != nh * dh {
        return Err(Error::runtime(format!(
            "SELF_ATTN: width {h} != nh*dh {}",
            nh * dh
        )));
    }
    let qf = flat(&q);
    let kf = flat(&k);
    let vf = flat(&v);
    let mut out = vec![0f64; n * h];
    let mut scores = vec![0f64; n];
    for head in 0..nh {
        let hb = head * dh;
        for i in 0..n {
            let mut mx = f64::NEG_INFINITY;
            for j in 0..n {
                let mut acc = 0.0;
                for d in 0..dh {
                    acc += qf[i * h + hb + d] * kf[j * h + hb + d];
                }
                let s = acc * scale;
                scores[j] = s;
                if s > mx {
                    mx = s;
                }
            }
            let mut se = 0.0;
            for s in scores.iter_mut() {
                *s = (*s - mx).exp();
                se += *s;
            }
            for d in 0..dh {
                let mut acc = 0.0;
                for j in 0..n {
                    acc += (scores[j] / se) * vf[j * h + hb + d];
                }
                out[i * h + hb + d] = acc;
            }
        }
    }
    Ok(vec![Value::F64(from_flat(&[n, h], out)?)])
}

fn gelu_approx_f64(x: f64) -> f64 {
    let s = if x < 0.0 { -1.0 } else { 1.0 };
    let ax = x.abs();
    let t = 1.0 / (1.0 + 0.3275911 * ax);
    let y = 1.0
        - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736)
            * t
            + 0.254829592)
            * t
            * (-ax * ax).exp();
    s * y
}

fn softplus_f64(x: f64) -> f64 {
    if x > 20.0 {
        x
    } else {
        x.exp().ln_1p()
    }
}

/// LINEAR_ATTN(Q,K,V, W1,B1,W2,B2, LNQ_W,LNQ_B,LNK_W,LNK_B,TEMP, NH, DH)
///
/// Per-head normalized-key linear attention with a shared positive feature
/// MLP. Reference implementation for parity with the CUDA target.
pub fn op_linear_attn(args: &[Arg], _config: &Config) -> Result<Vec<Value>> {
    let q = arg_f64_array(args, 0, "LINEAR_ATTN")?;
    let k = arg_f64_array(args, 1, "LINEAR_ATTN")?;
    let v = arg_f64_array(args, 2, "LINEAR_ATTN")?;
    let w1 = arg_f64_array(args, 3, "LINEAR_ATTN")?;
    let b1 = arg_f64_array(args, 4, "LINEAR_ATTN")?;
    let w2 = arg_f64_array(args, 5, "LINEAR_ATTN")?;
    let b2 = arg_f64_array(args, 6, "LINEAR_ATTN")?;
    let lnq_w = arg_f64_array(args, 7, "LINEAR_ATTN")?;
    let lnq_b = arg_f64_array(args, 8, "LINEAR_ATTN")?;
    let lnk_w = arg_f64_array(args, 9, "LINEAR_ATTN")?;
    let lnk_b = arg_f64_array(args, 10, "LINEAR_ATTN")?;
    let temp = arg_f64_array(args, 11, "LINEAR_ATTN")?;
    let nh = arg_i64(args, 12, "LINEAR_ATTN", "nh")?;
    let dh = arg_i64(args, 13, "LINEAR_ATTN", "dh")?;
    if nh <= 0 || dh <= 0 {
        return Err(Error::runtime("LINEAR_ATTN: bad nh/dh"));
    }
    let (nh, dh) = (nh as usize, dh as usize);
    let qsh = q.shape();
    if qsh.len() != 2 || k.shape() != qsh || v.shape() != qsh {
        return Err(Error::runtime("LINEAR_ATTN: need equal 2D Q/K/V"));
    }
    let (n, h) = (qsh[0], qsh[1]);
    if h != nh * dh {
        return Err(Error::runtime(format!(
            "LINEAR_ATTN: width {h} != nh*dh {}",
            nh * dh
        )));
    }
    if w1.shape().len() != 2 || w1.shape()[0] != dh {
        return Err(Error::runtime("LINEAR_ATTN: W1 must be (dh,hid)"));
    }
    let hid = w1.shape()[1];
    if w2.shape().len() != 2 || w2.shape()[0] != hid {
        return Err(Error::runtime("LINEAR_ATTN: W2 must be (hid,m)"));
    }
    let m = w2.shape()[1];
    if b1.shape() != [hid]
        || b2.shape() != [m]
        || lnq_w.shape() != [dh]
        || lnq_b.shape() != [dh]
        || lnk_w.shape() != [dh]
        || lnk_b.shape() != [dh]
        || temp.shape() != [nh]
    {
        return Err(Error::runtime(
            "LINEAR_ATTN: bias/norm/temp shapes do not match",
        ));
    }
    let qf = flat(&q);
    let kf = flat(&k);
    let vf = flat(&v);
    let w1f = flat(&w1);
    let b1f = flat(&b1);
    let w2f = flat(&w2);
    let b2f = flat(&b2);
    let lnqw = flat(&lnq_w);
    let lnqb = flat(&lnq_b);
    let lnkw = flat(&lnk_w);
    let lnkb = flat(&lnk_b);
    let tempf = flat(&temp);
    let mut out = vec![0f64; n * h];
    let mut fq = vec![0f64; n * m];
    let mut fk = vec![0f64; n * m];
    let mut qn = vec![0f64; dh];
    let mut kn = vec![0f64; dh];
    for head in 0..nh {
        let hb = head * dh;
        let qt = tempf[head].exp();
        for i in 0..n {
            let mut sq = 0.0;
            let mut sk = 0.0;
            for d in 0..dh {
                let a = qf[i * h + hb + d];
                let b = kf[i * h + hb + d];
                sq += a * a;
                sk += b * b;
            }
            let meanq = (0..dh).map(|d| qf[i * h + hb + d]).sum::<f64>() / dh as f64;
            let meank = (0..dh).map(|d| kf[i * h + hb + d]).sum::<f64>() / dh as f64;
            let rsq = 1.0 / ((sq / dh as f64 - meanq * meanq) + 1e-5).sqrt();
            let rsk = 1.0 / ((sk / dh as f64 - meank * meank) + 1e-5).sqrt();
            for d in 0..dh {
                qn[d] = ((qf[i * h + hb + d] - meanq) * rsq * lnqw[d] + lnqb[d]) * qt;
                kn[d] = (kf[i * h + hb + d] - meank) * rsk * lnkw[d] + lnkb[d];
            }
            for mm in 0..m {
                let mut aq = b2f[mm];
                let mut ak = b2f[mm];
                for j in 0..hid {
                    let mut hq = b1f[j];
                    let mut hk = b1f[j];
                    for d in 0..dh {
                        let w = w1f[d * hid + j];
                        hq += qn[d] * w;
                        hk += kn[d] * w;
                    }
                    aq += 0.5 * hq * (1.0 + gelu_approx_f64(hq / std::f64::consts::SQRT_2))
                        * w2f[j * m + mm];
                    ak += 0.5 * hk * (1.0 + gelu_approx_f64(hk / std::f64::consts::SQRT_2))
                        * w2f[j * m + mm];
                }
                fq[i * m + mm] = softplus_f64(aq) + 1e-3;
                fk[i * m + mm] = softplus_f64(ak) + 1e-3;
            }
        }
        let mut s = vec![0f64; m * dh];
        let mut z = vec![0f64; m];
        for i in 0..n {
            for mm in 0..m {
                let f = fk[i * m + mm];
                z[mm] += f;
                for d in 0..dh {
                    s[mm * dh + d] += f * vf[i * h + hb + d];
                }
            }
        }
        for i in 0..n {
            let mut den = 0.0;
            for mm in 0..m {
                den += fq[i * m + mm] * z[mm];
            }
            den = den.max(1e-6);
            for d in 0..dh {
                let mut acc = 0.0;
                for mm in 0..m {
                    acc += fq[i * m + mm] * s[mm * dh + d];
                }
                out[i * h + hb + d] = acc / den;
            }
        }
    }
    Ok(vec![Value::F64(from_flat(&[n, h], out)?)])
}

pub fn op_bmmv(args: &[Arg], config: &Config) -> Result<Vec<Value>> {
    let a = arg_triple(args, 0, "BMMV")?;
    let b = arg_triple(args, 1, "BMMV")?;
    let names = [
        "BATCH", "M", "N", "K", "LDA_A", "STRIDE_A", "LDA_B", "STRIDE_B", "LDC", "STRIDE_C",
        "TRANSB",
    ];
    let mut nums = [0i64; 11];
    for (i, name) in names.iter().enumerate() {
        nums[i] = arg_i64(args, i + 2, "BMMV", name)?;
    }
    let (bb, m, n, k, laa, sa, lab, sb, _lac, _sc, tb) = (
        nums[0], nums[1], nums[2], nums[3], nums[4], nums[5], nums[6], nums[7], nums[8], nums[9],
        nums[10],
    );
    if !(tb == 0 || tb == 1) {
        return Err(Error::runtime(format!(
            "BMMV: TRANSB must be 0/1, got {tb}"
        )));
    }
    if bb < 0 || m < 0 || n < 0 || k < 0 || laa < 0 || sa < 0 || lab < 0 || sb < 0 {
        return Err(Error::runtime("BMMV: negative view literal"));
    }
    let (bb, m, n, k, laa, sa, lab, sb) = (
        bb as usize,
        m as usize,
        n as usize,
        k as usize,
        laa as usize,
        sa as usize,
        lab as usize,
        sb as usize,
    );

    let mut ia = Vec::with_capacity(bb * m * k);
    for batch in 0..bb {
        for i in 0..m {
            for j in 0..k {
                ia.push(batch * sa + i * laa + j);
            }
        }
    }
    let mut ib = Vec::with_capacity(bb * k * n);
    for batch in 0..bb {
        for i in 0..k {
            for j in 0..n {
                let idx = if tb == 1 {
                    batch * sb + j * lab + i
                } else {
                    batch * sb + i * lab + j
                };
                ib.push(idx);
            }
        }
    }

    let gather_i8 = |src: &ArrayD<i8>, idxs: &[usize], what: &str| -> Result<Vec<i8>> {
        let f = flat(src);
        let mut out = Vec::with_capacity(idxs.len());
        for &idx in idxs {
            if idx >= f.len() {
                return Err(Error::runtime(format!(
                    "BMMV: view overruns base ({what} {idx} vs {})",
                    f.len()
                )));
            }
            out.push(f[idx]);
        }
        Ok(out)
    };
    let gather_i32 = |src: &ArrayD<i32>, idxs: &[usize], what: &str| -> Result<Vec<i32>> {
        let f = flat(src);
        let mut out = Vec::with_capacity(idxs.len());
        for &idx in idxs {
            if idx >= f.len() {
                return Err(Error::runtime(format!(
                    "BMMV: view overruns base ({what} {idx} vs {})",
                    f.len()
                )));
            }
            out.push(f[idx]);
        }
        Ok(out)
    };
    let gather_u8 = |src: &ArrayD<u8>, idxs: &[usize], what: &str| -> Result<Vec<u8>> {
        let f = flat(src);
        let mut out = Vec::with_capacity(idxs.len());
        for &idx in idxs {
            if idx >= f.len() {
                return Err(Error::runtime(format!(
                    "BMMV: view overruns base ({what} {idx} vs {})",
                    f.len()
                )));
            }
            out.push(f[idx]);
        }
        Ok(out)
    };

    let ta = Triple::new(
        from_flat(&[bb, m, k], gather_i8(&a.s, &ia, "A")?)?,
        from_flat(&[bb, m, k], gather_i32(&a.e, &ia, "A")?)?,
        from_flat(&[bb, m, k], gather_u8(&a.z, &ia, "A")?)?,
    )?;
    let tbt = Triple::new(
        from_flat(&[bb, k, n], gather_i8(&b.s, &ib, "B")?)?,
        from_flat(&[bb, k, n], gather_i32(&b.e, &ib, "B")?)?,
        from_flat(&[bb, k, n], gather_u8(&b.z, &ib, "B")?)?,
    )?;
    let (m_acc, _) = scales(config);
    let out = matmul_int(&ta, &tbt, m_acc)?;
    Ok(vec![Value::Triple(Triple::new(
        from_flat(&[bb * m, n], flat(&out.s))?,
        from_flat(&[bb * m, n], flat(&out.e))?,
        from_flat(&[bb * m, n], flat(&out.z))?,
    )?)])
}

#[cfg(test)]
mod linear_attn_tests {
    use super::*;

    fn f(v: Vec<f64>, shape: &[usize]) -> Arg {
        Arg::Stream(Value::F64(
            ArrayD::from_shape_vec(IxDyn(shape), v).unwrap(),
        ))
    }

    #[test]
    fn uniform_features_average_v() {
        let (n, nh, dh, hid, m) = (2usize, 1usize, 2usize, 2usize, 1usize);
        let q = f(vec![0.0; n * dh], &[n, dh]);
        let k = f(vec![0.0; n * dh], &[n, dh]);
        let v = f(vec![1.0, 2.0, 3.0, 4.0], &[n, dh]);
        let w1 = f(vec![0.0; dh * hid], &[dh, hid]);
        let b1 = f(vec![0.0; hid], &[hid]);
        let w2 = f(vec![0.0; hid * m], &[hid, m]);
        let b2 = f(vec![0.0; m], &[m]);
        let lnw = f(vec![1.0; dh], &[dh]);
        let lnb = f(vec![0.0; dh], &[dh]);
        let temp = f(vec![0.0; nh], &[nh]);
        let args = vec![
            q, k, v, w1, b1, w2, b2, lnw.clone(), lnb.clone(), lnw, lnb, temp,
            Arg::Literal(nh as f64), Arg::Literal(dh as f64),
        ];
        let out = op_linear_attn(&args, &Config::new()).unwrap();
        let Value::F64(a) = &out[0] else { panic!("not F64") };
        for row in 0..n {
            assert!((a[[row, 0]] - 2.0).abs() < 1e-9);
            assert!((a[[row, 1]] - 3.0).abs() < 1e-9);
        }
    }
}
