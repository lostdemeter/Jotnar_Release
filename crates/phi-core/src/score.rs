//! Declarative scoring (port of `score.py`): peak/units are explicit.

use ndarray::ArrayD;

use crate::error::{Error, Result};

/// PSNR with an explicitly declared peak (255 for uint8, 1.0 for floats).
pub fn psnr(a: &ArrayD<f64>, b: &ArrayD<f64>, peak: f64) -> Result<f64> {
    if a.shape() != b.shape() {
        return Err(Error::runtime("psnr shape mismatch"));
    }
    let mut mse = 0f64;
    for (&x, &y) in a.iter().zip(b.iter()) {
        let d = x - y;
        mse += d * d;
    }
    if a.is_empty() {
        return Ok(99.0);
    }
    mse /= a.len() as f64;
    Ok(if mse > 0.0 {
        10.0 * (peak * peak / mse).log10()
    } else {
        99.0
    })
}

/// Maximum absolute error relative to the reference signal maximum.
pub fn lattice_rel_error(got: &ArrayD<f64>, reference: &ArrayD<f64>) -> Result<f64> {
    if got.shape() != reference.shape() {
        return Err(Error::runtime("lattice_rel_error shape mismatch"));
    }
    let mut d = 0f64;
    let mut denom = 0f64;
    for (&g, &r) in got.iter().zip(reference.iter()) {
        d = d.max((g - r).abs());
        denom = denom.max(r.abs());
    }
    Ok(d / (denom + 1e-12))
}

/// The honest lattice bar: relative to the signal maximum.
pub fn lattice_ok(got: &ArrayD<f64>, reference: &ArrayD<f64>, rel_tol: f64) -> Result<bool> {
    Ok(lattice_rel_error(got, reference)? < rel_tol)
}
