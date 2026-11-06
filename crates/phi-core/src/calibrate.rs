//! Offline range analysis -> frozen integer scales (port of `calibrate.py`).

use ndarray::ArrayD;

use crate::error::{Error, Result};
use crate::lattice::{self, BIAS, K, LN_PHI};

/// Lattice scale covering `pmax` with margin (~2.7x at 512 steps).
pub fn m_of(pmax: f64, margin: i64) -> i64 {
    BIAS + lattice::round_half_even(K as f64 * (pmax + 1e-12).ln() / LN_PHI) as i64 + margin
}

/// Fold inference BatchNorm into conv weights (exact, offline).
///
/// `conv_w` is OIHW, `conv_b`/`mean`/`var`/`bn_w`/`bn_b` are `(C,)`.
/// Returns `(W', b')` with `W' = W * bw/sqrt(var+eps)` per output channel.
pub fn fold_batchnorm(
    conv_w: &ArrayD<f64>,
    conv_b: &ArrayD<f64>,
    mean: &ArrayD<f64>,
    var: &ArrayD<f64>,
    bn_w: &ArrayD<f64>,
    bn_b: &ArrayD<f64>,
    eps: f64,
) -> Result<(ArrayD<f64>, ArrayD<f64>)> {
    let sh = conv_w.shape();
    if sh.len() != 4 {
        return Err(Error::runtime("fold_batchnorm conv_w must be OIHW"));
    }
    let out_c = sh[0];
    let per = sh[1] * sh[2] * sh[3];
    if conv_b.len() != out_c
        || mean.len() != out_c
        || var.len() != out_c
        || bn_w.len() != out_c
        || bn_b.len() != out_c
    {
        return Err(Error::runtime("fold_batchnorm channel count mismatch"));
    }
    let wf: Vec<f64> = conv_w.iter().copied().collect();
    let scale: Vec<f64> = (0..out_c)
        .map(|i| bn_w[i] / (var[i] + eps).sqrt())
        .collect();
    let mut w = vec![0f64; wf.len()];
    for c in 0..out_c {
        for i in 0..per {
            w[c * per + i] = wf[c * per + i] * scale[c];
        }
    }
    let mut b = vec![0f64; out_c];
    for c in 0..out_c {
        b[c] = (conv_b[c] - mean[c]) * scale[c] + bn_b[c];
    }
    Ok((
        lattice::array_from_vec(sh, w)?,
        lattice::array_from_vec(&[out_c], b)?,
    ))
}
