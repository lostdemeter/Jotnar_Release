//! Wall-Breaking Protocol (WBP) core, expressed as Jotnar integer listings.
//!
//! This is the first Jotnar-native slice of the protocol:
//!
//! - `resfrac`: an AR(1)/AR(2) predictability score. `rho -> 0` means the
//!   residual is structured; `rho -> 1` means it looks ergodic.
//! - `peel`: the `FractalPeeler` loop. Each layer is computed by a Jotnar
//!   listing; the recursion is host orchestration.
//! - `ergodic_jump`: inject a baked irrational harmonic (1/sqrt(5), 1/phi,
//!   1/pi) and re-measure `rho`.
//! - `self_similar_products`: the residual-selected self-similar basis. Given a
//!   feature vector and a lag, a Jotnar listing emits `x_i * x_{i-lag}`.
//!
//! All arithmetic is scaled fixed-point integers (`SCALE = 1024`). The listings
//! are ordinary Jotnar programs: `SLICE`, `IMUL`, `ISUM`, `IADD`, `ISUB`,
//! `IDIV`, `IMAX`, `IEQ`, `ISELECT`, `IGET`. No new opcodes and no floats at
//! runtime.

use std::collections::BTreeMap;

use jotnar::asm::AsmOptions;
use jotnar::payload::Feeds;
use jotnar::runtime::run_text;
use ndarray::{ArrayD, IxDyn};

/// Fixed-point scale for the WBP listings.
pub const SCALE: i64 = 1024;

fn opts() -> AsmOptions {
    AsmOptions::new(".")
}

fn feeds(pairs: &[(&str, &[i64])]) -> Result<Feeds, String> {
    let mut out = Feeds::new();
    for (name, data) in pairs {
        let arr = ArrayD::from_shape_vec(IxDyn(&[data.len()]), data.to_vec())
            .map_err(|e| format!("wbp feed {name}: {e}"))?;
        out.insert((*name).to_string(), jotnar::Value::I64(arr));
    }
    Ok(out)
}

fn run_listing(
    listing: &str,
    pairs: &[(&str, &[i64])],
) -> Result<BTreeMap<String, Vec<i64>>, String> {
    let got =
        run_text(listing, &opts(), &feeds(pairs)?).map_err(|e| format!("wbp listing: {e}"))?;
    let mut out = BTreeMap::new();
    for (name, value) in got {
        if let jotnar::Value::I64(a) = value {
            out.insert(name, a.iter().copied().collect());
        }
    }
    Ok(out)
}

/// A Jotnar listing that computes the AR(1)/AR(2) residual and `rho^2`.
///
/// The signal is `I:SEQ`, scaled by [`SCALE`]. Outputs:
/// `RHO2`, `E` (residual), `AC0`, `A1`, `A2`, `R1`, `R2`, `VE`, `V0`.
pub fn resfrac_listing(n: usize, order: usize) -> Result<String, String> {
    if order == 0 || order > 2 {
        return Err(format!("wbp resfrac supports order 1 or 2, got {order}"));
    }
    if n < order + 5 {
        return Err(format!("wbp resfrac needs n >= {}, got {n}", order + 5));
    }
    let s = SCALE;
    let mut l = vec![
        format!("# WBP resfrac: order {order}, n {n}, scale {s}"),
        "IN X AS I:SEQ".to_string(),
        "Xsq = IMUL(X, X)".to_string(),
        format!("AC0 = ISUM(Xsq, 0, {n})"),
        // lag 1
        format!("XA1 = IWINDOW(X, 0, {})", n - 1),
        format!("XB1 = IWINDOW(X, 1, {})", n - 1),
        "P1 = IMUL(XA1, XB1)".to_string(),
        format!("A1 = ISUM(P1, 0, {})", n - 1),
        format!("A1s = IMUL(A1, {s})"),
        "AC0d = IMAX(AC0, 1)".to_string(),
        "R1 = IDIV(A1s, AC0d)".to_string(),
    ];
    if order == 2 {
        // lag 2
        l.push(format!("XA2 = IWINDOW(X, 0, {})", n - 2));
        l.push(format!("XB2 = IWINDOW(X, 2, {})", n - 2));
        l.push("P2 = IMUL(XA2, XB2)".to_string());
        l.push(format!("A2 = ISUM(P2, 0, {})", n - 2));
        l.push(format!("A2s = IMUL(A2, {s})"));
        l.push("R2 = IDIV(A2s, AC0d)".to_string());
        // predictor: (R1*x_{i-1} + R2*x_{i-2}) / (R1 + R2)
        l.push(format!("X0 = IWINDOW(X, 0, {})", n - 2));
        l.push(format!("X1 = IWINDOW(X, 1, {})", n - 2));
        l.push(format!("X2v = IWINDOW(X, 2, {})", n - 2));
        l.push("R1s = IGET(R1, 0)".to_string());
        l.push("R2s = IGET(R2, 0)".to_string());
        l.push("T1 = IMUL(X1, R1s)".to_string());
        l.push("T2 = IMUL(X0, R2s)".to_string());
        l.push("NUM = IADD(T1, T2)".to_string());
        l.push("DEN = IADD(R1, R2)".to_string());
        l.push("Z = IEQ(DEN, 0)".to_string());
        l.push("DENs = ISELECT(Z, 1, DEN)".to_string());
        l.push("DENv = IGET(DENs, 0)".to_string());
        l.push("PRED = IDIV(NUM, DENv)".to_string());
        l.push("E = ISUB(X2v, PRED)".to_string());
    } else {
        l.push("A2 = IMUL(A1, 0)".to_string());
        l.push("R2 = IMUL(A1, 0)".to_string());
        l.push(format!("XP = IWINDOW(X, 0, {})", n - 1));
        l.push(format!("XN = IWINDOW(X, 1, {})", n - 1));
        l.push("R1s = IGET(R1, 0)".to_string());
        l.push("PRED0 = IMUL(XP, R1s)".to_string());
        l.push(format!("PRED = IDIV(PRED0, {s})"));
        l.push("E = ISUB(XN, PRED)".to_string());
    }
    let en = n - order;
    l.push(format!("Es = ISUM(E, 0, {en})"));
    l.push(format!("EM = IDIV(Es, {en})"));
    l.push("EMs = IGET(EM, 0)".to_string());
    l.push("EC = ISUB(E, EMs)".to_string());
    l.push("EC2 = IMUL(EC, EC)".to_string());
    l.push(format!("VEs = ISUM(EC2, 0, {en})"));
    l.push(format!("VE = IDIV(VEs, {en})"));
    l.push(format!("Xs = ISUM(X, 0, {n})"));
    l.push(format!("XM = IDIV(Xs, {n})"));
    l.push("XMs = IGET(XM, 0)".to_string());
    l.push("XC = ISUB(X, XMs)".to_string());
    l.push("XC2 = IMUL(XC, XC)".to_string());
    l.push(format!("V0s = ISUM(XC2, 0, {n})"));
    l.push(format!("V0 = IDIV(V0s, {n})"));
    l.push(format!("VEs2 = IMUL(VE, {s})"));
    l.push("V0d = IMAX(V0, 1)".to_string());
    l.push("RHO2 = IDIV(VEs2, V0d)".to_string());
    Ok(l.join("\n") + "\n")
}

#[derive(Clone, Debug)]
pub struct ResfracResult {
    /// `rho^2` scaled by [`SCALE`].
    pub rho2: i64,
    /// `rho` in `[0, 1]`.
    pub rho: f64,
    /// AR residual, length `n - order`.
    pub residual: Vec<i64>,
    pub ac0: i64,
    pub ac1: i64,
    pub ac2: i64,
    pub r1: i64,
    pub r2: i64,
    pub ve: i64,
    pub v0: i64,
}

/// Run the Jotnar `resfrac` listing on an integer signal.
pub fn resfrac(signal: &[i64], order: usize) -> Result<ResfracResult, String> {
    let n = signal.len();
    let listing = resfrac_listing(n, order)?;
    let out = run_listing(&listing, &[("X", signal)])?;
    let one = |name: &str| -> Result<i64, String> {
        out.get(name)
            .and_then(|v| v.first().copied())
            .ok_or_else(|| format!("wbp resfrac missing {name}"))
    };
    let rho2 = one("RHO2")?;
    let residual = out
        .get("E")
        .cloned()
        .ok_or_else(|| "wbp resfrac missing E".to_string())?;
    let rho = ((rho2 as f64) / (SCALE as f64)).clamp(0.0, 1.0).sqrt();
    Ok(ResfracResult {
        rho2,
        rho,
        residual,
        ac0: one("AC0")?,
        ac1: one("A1")?,
        ac2: one("A2")?,
        r1: one("R1")?,
        r2: one("R2")?,
        ve: one("VE")?,
        v0: one("V0")?,
    })
}

#[derive(Clone, Debug)]
pub struct PeelResult {
    /// Structured layers, each padded to the original length with leading zeros.
    pub layers: Vec<Vec<i64>>,
    pub residual: Vec<i64>,
    pub rhos: Vec<f64>,
}

/// The `FractalPeeler` loop. Each iteration runs the Jotnar `resfrac` listing.
pub fn peel(
    signal: &[i64],
    order: usize,
    max_layers: usize,
    rho_threshold: f64,
) -> Result<PeelResult, String> {
    if signal.is_empty() {
        return Ok(PeelResult {
            layers: Vec::new(),
            residual: Vec::new(),
            rhos: Vec::new(),
        });
    }
    let n = signal.len();
    let mean = signal.iter().sum::<i64>() / n as i64;
    let mut cur: Vec<i64> = signal.iter().map(|v| v - mean).collect();
    let mut layers = Vec::new();
    let mut rhos = Vec::new();
    for _ in 0..max_layers.max(1) {
        if cur.len() < order + 5 {
            break;
        }
        let r = resfrac(&cur, order)?;
        rhos.push(r.rho);
        if r.rho >= rho_threshold {
            break;
        }
        let mut layer = vec![0i64; order];
        layer.extend(
            cur[order..]
                .iter()
                .zip(r.residual.iter())
                .map(|(a, b)| a - b),
        );
        layers.push(layer);
        cur = r.residual;
    }
    Ok(PeelResult {
        layers,
        residual: cur,
        rhos,
    })
}

/// Self-similar product features at one lag: `x_i * x_{i-lag}`.
///
/// The first `lag` values are zero-padded so the result aligns with `x`.
/// This is a Jotnar `IMUL` over two `SLICE`s.
pub fn self_similar_products(x: &[i64], lag: usize) -> Result<Vec<i64>, String> {
    let n = x.len();
    if lag == 0 {
        return Err("wbp self_similar_products needs lag >= 1".to_string());
    }
    if lag >= n {
        return Ok(vec![0; n]);
    }
    let listing = format!(
        "IN X AS I:SEQ\nA = IWINDOW(X, 0, {})\nB = IWINDOW(X, {lag}, {})\nOUT = IMUL(A, B)\n",
        n - lag,
        n - lag
    );
    let out = run_listing(&listing, &[("X", x)])?;
    let products = out
        .get("OUT")
        .cloned()
        .ok_or_else(|| "wbp self_similar_products missing OUT".to_string())?;
    let mut aligned = vec![0i64; lag];
    aligned.extend(products);
    Ok(aligned)
}

/// A baked irrational harmonic table, scaled by [`SCALE`].
pub fn harmonic_table(n: usize, freq: f64, amp: f64, scale: i64) -> Vec<i64> {
    (0..n)
        .map(|i| {
            let phase = 2.0 * std::f64::consts::PI * freq * i as f64;
            (amp * scale as f64 * phase.sin()).round() as i64
        })
        .collect()
}

fn add_harmonic(signal: &[i64], harmonic: &[i64]) -> Result<Vec<i64>, String> {
    if signal.len() != harmonic.len() {
        return Err("wbp harmonic length mismatch".to_string());
    }
    let out = run_listing(
        "IN X AS I:SEQ\nIN H AS I:SEQ\nOUT = IADD(X, H)\n",
        &[("X", signal), ("H", harmonic)],
    )?;
    out.get("OUT")
        .cloned()
        .ok_or_else(|| "wbp add_harmonic missing OUT".to_string())
}

#[derive(Clone, Debug)]
pub struct JumpResult {
    pub label: &'static str,
    pub freq: f64,
    pub rho_before: f64,
    pub rho_after: f64,
    /// `rho_after - rho_before`; negative means the injection exposed structure.
    pub delta: f64,
}

#[derive(Clone, Debug)]
pub struct WbpAudit {
    pub peel: PeelResult,
    pub jumps: Vec<JumpResult>,
}

/// Run the WBP core: peel, then (if the residual is ergodic) inject irrational
/// harmonics and re-measure.
pub fn audit(signal: &[i64], order: usize, max_layers: usize) -> Result<WbpAudit, String> {
    let peel = peel(signal, order, max_layers, 0.85)?;
    let mut jumps = Vec::new();
    let rho = peel.rhos.last().copied().unwrap_or(1.0);
    if rho > 0.85 && !peel.residual.is_empty() {
        for (label, freq) in [
            ("sqrt5", 1.0 / 5.0_f64.sqrt()),
            ("phi", 1.0 / std::f64::consts::GOLDEN_RATIO),
            ("pi", 1.0 / std::f64::consts::PI),
        ] {
            let h = harmonic_table(peel.residual.len(), freq, 0.15, SCALE);
            let aug = add_harmonic(&peel.residual, &h)?;
            let after = resfrac(&aug, order)?;
            jumps.push(JumpResult {
                label,
                freq,
                rho_before: rho,
                rho_after: after.rho,
                delta: after.rho - rho,
            });
        }
    }
    Ok(WbpAudit { peel, jumps })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn slow_signal(n: usize) -> Vec<i64> {
        (0..n)
            .map(|i| {
                let t = i as f64;
                ((0.05 * t).sin() * 800.0 + (0.17 * t).sin() * 120.0).round() as i64
            })
            .collect()
    }

    fn noise(n: usize, seed: u64) -> Vec<i64> {
        let mut s = seed;
        (0..n)
            .map(|_| {
                s = s.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
                ((s >> 33) as i64 % 2001) - 1000
            })
            .collect()
    }

    #[test]
    fn resfrac_sees_structure_that_noise_does_not() {
        let n = 96;
        let structured = resfrac(&slow_signal(n), 2).unwrap();
        let random = resfrac(&noise(n, 7), 2).unwrap();
        assert!(
            structured.rho < random.rho,
            "structured rho={} random rho={}",
            structured.rho,
            random.rho
        );
        assert!(structured.residual.len() == n - 2);
    }

    #[test]
    fn peel_peels_a_structured_signal() {
        let p = peel(&slow_signal(96), 2, 6, 0.85).unwrap();
        assert!(!p.rhos.is_empty());
        assert!(p.residual.len() < 96);
        assert!(
            !p.layers.is_empty(),
            "a slow signal should peel at least once"
        );
    }

    #[test]
    fn self_similar_products_are_exact() {
        let x = [1, 2, 3, 4, 5];
        let p = self_similar_products(&x, 1).unwrap();
        assert_eq!(p, vec![0, 2, 6, 12, 20]);
        let q = self_similar_products(&x, 2).unwrap();
        assert_eq!(q, vec![0, 0, 3, 8, 15]);
    }

    #[test]
    fn ergodic_jump_returns_a_delta() {
        let n = 96;
        let a = audit(&noise(n, 11), 2, 3).unwrap();
        assert!(!a.peel.rhos.is_empty());
        // The residual may or may not be ergodic at this small size; if it is,
        // the jump must still produce a finite delta.
        for j in &a.jumps {
            assert!(j.delta.is_finite(), "{j:?}");
        }
    }
}
