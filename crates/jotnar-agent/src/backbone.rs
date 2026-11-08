//! The task-encoder backbone.
//!
//! A backbone is a Jotnar listing fragment: task features -> initial latent.
//! The first slice replaces the named-selection matrix with a two-layer MLP
//! (linear -> ReLU -> linear), fitted to reproduce the named latent. The
//! attention/Qwen-style block slots into the same interface later.

use jotnar::lattice;
use jotnar::Value;
use ndarray::{ArrayD, IxDyn};
use serde::{Deserialize, Serialize};

use crate::task_encoder::{LATENT_NAMES, TASK_FEATURE_NAMES};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum BackboneKind {
    Named,
    Mlp,
    Qwen,
}

/// A small Qwen-style block over the task tokens.
///
/// The block is a real attention/SwiGLU graph: RMSNorm, Q/K/V, RoPE,
/// scaled dot-product attention, output projection, residual, RMSNorm, and a
/// gated MLP. The first slice initializes the attention output and the MLP
/// down-projection to zero, so the block is an exact linear pass-through of
/// the named latent. The gate decides whether training those branches earns
/// its place.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct QwenWeights {
    /// Token width. Must be even (RoPE) and at least LATENT_NAMES.len().
    pub dim: usize,
    /// SwiGLU hidden width.
    pub inter: usize,
    /// Shared token embedding: 1 x dim.
    pub emb: Vec<Vec<f64>>,
    /// Per-token codes: (T+1) x dim. The last row is the latent bias token.
    pub pos_code: Vec<Vec<f64>>,
    /// Q/K/V/output projections: dim x dim.
    pub wq: Vec<Vec<f64>>,
    pub wk: Vec<Vec<f64>>,
    pub wv: Vec<Vec<f64>>,
    pub wo: Vec<Vec<f64>>,
    /// SwiGLU: dim x (2*inter) (gate+up), inter x dim (down).
    pub w_gate_up: Vec<Vec<f64>>,
    pub w_down: Vec<Vec<f64>>,
    /// RMSNorm weights: dim.
    pub rms1: Vec<f64>,
    pub rms2: Vec<f64>,
    /// Pooling row: 1 x (T+1).
    pub pool: Vec<Vec<f64>>,
    /// Latent readout: dim x D.
    pub w_out: Vec<Vec<f64>>,
    /// Attention scale (1/sqrt(dim)).
    pub beta_b: f64,
    /// Lattice scales. 38000 covers the current corpus (counts up to ~128).
    #[serde(default = "default_lattice_scale")]
    pub m_acc: i64,
    #[serde(default = "default_lattice_scale")]
    pub m_cov: i64,
}

fn default_lattice_scale() -> i64 {
    38000
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Backbone {
    pub kind: BackboneKind,
    /// Hidden layer: H x T.
    pub w1: Vec<Vec<i64>>,
    pub b1: Vec<i64>,
    /// Latent layer: D x H. Stored scaled by `out_scale`.
    pub w2: Vec<Vec<i64>>,
    pub b2: Vec<i64>,
    pub scale: i64,
    /// Fixed-point scale for the latent readout. 1 means unscaled.
    #[serde(default = "default_out_scale")]
    pub out_scale: i64,
    pub relu: bool,
    /// Qwen-style weights; present only for `BackboneKind::Qwen`.
    #[serde(default)]
    pub qwen: Option<QwenWeights>,
}

fn default_out_scale() -> i64 {
    1
}

pub(crate) fn solve_linear(mut a: Vec<Vec<f64>>, mut b: Vec<Vec<f64>>) -> Vec<Vec<f64>> {
    let n = a.len();
    let d = b.first().map(|row| row.len()).unwrap_or(0);
    for i in 0..n {
        let mut pivot = i;
        for r in (i + 1)..n {
            if a[r][i].abs() > a[pivot][i].abs() {
                pivot = r;
            }
        }
        if pivot != i {
            a.swap(i, pivot);
            b.swap(i, pivot);
        }
        let value = a[i][i];
        if value.abs() < 1e-12 {
            continue;
        }
        for x in a[i].iter_mut().skip(i) {
            *x /= value;
        }
        for x in b[i].iter_mut().take(d) {
            *x /= value;
        }
        let pivot_row = a[i].clone();
        let pivot_rhs = b[i].clone();
        for r in 0..n {
            if r == i {
                continue;
            }
            let factor = a[r][i];
            if factor == 0.0 {
                continue;
            }
            for (j, x) in a[r].iter_mut().enumerate().skip(i) {
                *x -= factor * pivot_row[j];
            }
            for (j, x) in b[r].iter_mut().enumerate().take(d) {
                *x -= factor * pivot_rhs[j];
            }
        }
    }
    b
}

fn sum_terms(lines: &mut Vec<String>, prefix: &str, terms: Vec<String>, bias: i64) -> String {
    if terms.is_empty() {
        let zero = format!("{prefix}_z");
        lines.push(format!("{zero} = IMUL(t0, 0)"));
        if bias == 0 {
            return zero;
        }
        let name = format!("{prefix}_c");
        lines.push(format!("{name} = IADD({zero}, {bias})"));
        return name;
    }
    let mut acc = terms[0].clone();
    for (k, term) in terms.iter().enumerate().skip(1) {
        let name = format!("{prefix}_a{k}");
        lines.push(format!("{name} = IADD({acc}, {term})"));
        acc = name;
    }
    if bias != 0 {
        let name = format!("{prefix}_b");
        lines.push(format!("{name} = IADD({acc}, {bias})"));
        acc = name;
    }
    acc
}

fn triple_vec(values: &[f64], shape: &[usize]) -> Result<Value, String> {
    let arr = ArrayD::from_shape_vec(IxDyn(shape), values.to_vec()).map_err(|e| e.to_string())?;
    Ok(Value::Triple(lattice::encode(&arr)))
}

fn triple_matrix(rows: &[Vec<f64>]) -> Result<Value, String> {
    let r = rows.len();
    let c = rows.first().map(|row| row.len()).unwrap_or(0);
    let flat: Vec<f64> = rows.iter().flat_map(|row| row.iter().copied()).collect();
    triple_vec(&flat, &[r, c])
}

fn check_matrix(name: &str, rows: &[Vec<f64>], r: usize, c: usize) -> Result<(), String> {
    if rows.len() != r || rows.iter().any(|row| row.len() != c) {
        return Err(format!("qwen {name} must be {r}x{c}"));
    }
    Ok(())
}

fn qwen_matmul(a: &[Vec<f64>], b: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n = a.len();
    let k = a.first().map(|r| r.len()).unwrap_or(0);
    let m = b.first().map(|r| r.len()).unwrap_or(0);
    let mut out = vec![vec![0.0f64; m]; n];
    for (i, row) in a.iter().enumerate() {
        for (kk, &av) in row.iter().enumerate().take(k) {
            for j in 0..m {
                out[i][j] += av * b[kk][j];
            }
        }
    }
    out
}

fn qwen_transpose(a: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let r = a.len();
    let c = a.first().map(|row| row.len()).unwrap_or(0);
    let mut out = vec![vec![0.0f64; r]; c];
    for i in 0..r {
        for j in 0..c {
            out[j][i] = a[i][j];
        }
    }
    out
}

fn qwen_add(a: &[Vec<f64>], b: &[Vec<f64>]) -> Vec<Vec<f64>> {
    a.iter()
        .zip(b)
        .map(|(ra, rb)| ra.iter().zip(rb).map(|(x, y)| x + y).collect())
        .collect()
}

fn qwen_mul(a: &[Vec<f64>], b: &[Vec<f64>]) -> Vec<Vec<f64>> {
    a.iter()
        .zip(b)
        .map(|(ra, rb)| ra.iter().zip(rb).map(|(x, y)| x * y).collect())
        .collect()
}

fn qwen_rmsnorm(x: &[Vec<f64>], w: &[f64]) -> Vec<Vec<f64>> {
    x.iter()
        .map(|row| {
            let mean = row.iter().map(|v| v * v).sum::<f64>() / row.len().max(1) as f64;
            let rms = (mean + 1e-6).sqrt();
            row.iter()
                .zip(w)
                .map(|(v, weight)| v / rms * weight)
                .collect()
        })
        .collect()
}

fn qwen_softmax(x: &[Vec<f64>]) -> Vec<Vec<f64>> {
    x.iter()
        .map(|row| {
            let max = row.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let ex: Vec<f64> = row.iter().map(|v| (v - max).exp()).collect();
            let sum: f64 = ex.iter().sum();
            ex.iter().map(|v| v / sum.max(1e-30)).collect()
        })
        .collect()
}

fn qwen_rotary(x: &[Vec<f64>], pos: &[i64], dh: usize) -> Vec<Vec<f64>> {
    let rows = x.len();
    let half = dh / 2;
    let theta: Vec<f64> = (0..half)
        .map(|i| 10000f64.powf(-2.0 * i as f64 / dh as f64))
        .collect();
    let mut out = x.to_vec();
    for r in 0..rows {
        for (p, &theta_p) in theta.iter().enumerate() {
            let angle = pos.get(r).copied().unwrap_or(0) as f64 * theta_p;
            let (sin, cos) = angle.sin_cos();
            let i0 = 2 * p;
            let i1 = i0 + 1;
            let a = x[r][i0];
            let b = x[r][i1];
            out[r][i0] = a * cos - b * sin;
            out[r][i1] = a * sin + b * cos;
        }
    }
    out
}

fn qwen_silu(x: f64) -> f64 {
    x / (1.0 + (-x).exp())
}

fn qwen_solve(a: &[Vec<f64>], y: &[f64], lambda: f64) -> Vec<f64> {
    let p = a.first().map(|row| row.len()).unwrap_or(0);
    let mut g = vec![vec![0.0f64; p]; p];
    let mut b = vec![vec![0.0f64; 1]; p];
    for (row, &target) in a.iter().zip(y) {
        for (i, &row_i) in row.iter().enumerate() {
            if row_i == 0.0 {
                continue;
            }
            b[i][0] += row_i * target;
            for (gij, &row_j) in g[i].iter_mut().zip(row) {
                *gij += row_i * row_j;
            }
        }
    }
    for (i, row) in g.iter_mut().enumerate() {
        row[i] += lambda.max(1e-9);
    }
    solve_linear(g, b)
        .into_iter()
        .map(|row| row.first().copied().unwrap_or(0.0))
        .collect()
}

struct QwenFeatures {
    x: Vec<Vec<f64>>,
    attn: Vec<Vec<f64>>,
    res: Vec<Vec<f64>>,
    mlp: Vec<Vec<f64>>,
}

struct QwenCache {
    x: Vec<Vec<f64>>,
    n1: Vec<Vec<f64>>,
    v: Vec<Vec<f64>>,
    qr: Vec<Vec<f64>>,
    kt: Vec<Vec<f64>>,
    probs: Vec<Vec<f64>>,
    attn: Vec<Vec<f64>>,
    res: Vec<Vec<f64>>,
    n2: Vec<Vec<f64>>,
    gate: Vec<Vec<f64>>,
    up: Vec<Vec<f64>>,
    silu: Vec<Vec<f64>>,
    act: Vec<Vec<f64>>,
    latent: Vec<Vec<f64>>,
}

struct QwenGrads {
    wq: Vec<Vec<f64>>,
    wk: Vec<Vec<f64>>,
    wv: Vec<Vec<f64>>,
    wo: Vec<Vec<f64>>,
    w_gate_up: Vec<Vec<f64>>,
    w_down: Vec<Vec<f64>>,
}

impl QwenGrads {
    fn zeros(q: &QwenWeights) -> Self {
        let mat = |rows: usize, cols: usize| vec![vec![0.0f64; cols]; rows];
        Self {
            wq: mat(q.dim, q.dim),
            wk: mat(q.dim, q.dim),
            wv: mat(q.dim, q.dim),
            wo: mat(q.dim, q.dim),
            w_gate_up: mat(q.dim, 2 * q.inter),
            w_down: mat(q.inter, q.dim),
        }
    }
}

fn qwen_all_zero(rows: &[Vec<f64>]) -> bool {
    rows.iter().flatten().all(|v| v.abs() < 1e-12)
}

fn qwen_add_assign(a: &mut [Vec<f64>], b: &[Vec<f64>]) {
    for (ra, rb) in a.iter_mut().zip(b) {
        for (x, y) in ra.iter_mut().zip(rb) {
            *x += y;
        }
    }
}

fn qwen_rmsnorm_backward(x: &[Vec<f64>], w: &[f64], dn: &[Vec<f64>]) -> Vec<Vec<f64>> {
    x.iter()
        .zip(dn)
        .map(|(row, drow)| {
            let cols = row.len().max(1) as f64;
            let mean = row.iter().map(|v| v * v).sum::<f64>() / cols;
            let rms = (mean + 1e-6).sqrt();
            let dot: f64 = row
                .iter()
                .zip(w)
                .zip(drow)
                .map(|((x, w), d)| w * d * x)
                .sum();
            let coeff = dot / (cols * rms.powi(3));
            row.iter()
                .enumerate()
                .map(|(i, x)| w[i] * drow[i] / rms - x * coeff)
                .collect()
        })
        .collect()
}

fn qwen_softmax_backward(p: &[Vec<f64>], dp: &[Vec<f64>]) -> Vec<Vec<f64>> {
    p.iter()
        .zip(dp)
        .map(|(row, drow)| {
            let dot: f64 = row.iter().zip(drow).map(|(p, d)| p * d).sum();
            row.iter().zip(drow).map(|(p, d)| p * (d - dot)).collect()
        })
        .collect()
}

fn qwen_silu_deriv(x: f64) -> f64 {
    let s = 1.0 / (1.0 + (-x).exp());
    s + x * s * (1.0 - s)
}

fn qwen_rotary_backward(dx: &[Vec<f64>], pos: &[i64], dh: usize) -> Vec<Vec<f64>> {
    let rows = dx.len();
    let half = dh / 2;
    let theta: Vec<f64> = (0..half)
        .map(|i| 10000f64.powf(-2.0 * i as f64 / dh as f64))
        .collect();
    let mut out = dx.to_vec();
    for r in 0..rows {
        for (p, &theta_p) in theta.iter().enumerate() {
            let angle = pos.get(r).copied().unwrap_or(0) as f64 * theta_p;
            let (sin, cos) = angle.sin_cos();
            let i0 = 2 * p;
            let i1 = i0 + 1;
            let a = dx[r][i0];
            let b = dx[r][i1];
            out[r][i0] = a * cos + b * sin;
            out[r][i1] = -a * sin + b * cos;
        }
    }
    out
}

impl QwenWeights {
    fn forward_cached(&self, task: &[i64]) -> QwenCache {
        let t = TASK_FEATURE_NAMES.len();
        let mut tokens = vec![vec![0.0f64]; t + 1];
        for (i, value) in task.iter().enumerate().take(t) {
            tokens[i][0] = *value as f64;
        }
        tokens[t][0] = 1.0;
        let x0 = qwen_matmul(&tokens, &self.emb);
        let x = qwen_mul(&x0, &self.pos_code);
        let n1 = qwen_rmsnorm(&x, &self.rms1);
        let q = qwen_matmul(&n1, &self.wq);
        let k = qwen_matmul(&n1, &self.wk);
        let v = qwen_matmul(&n1, &self.wv);
        let pos: Vec<i64> = (0..=t as i64).collect();
        let qr = qwen_rotary(&q, &pos, self.dim);
        let kr = qwen_rotary(&k, &pos, self.dim);
        let kt = qwen_transpose(&kr);
        let mut scores = qwen_matmul(&qr, &kt);
        for row in &mut scores {
            for value in row.iter_mut() {
                *value *= self.beta_b;
            }
        }
        let probs = qwen_softmax(&scores);
        let attn = qwen_matmul(&probs, &v);
        let ao = qwen_matmul(&attn, &self.wo);
        let res = qwen_add(&x, &ao);
        let n2 = qwen_rmsnorm(&res, &self.rms2);
        let gu = qwen_matmul(&n2, &self.w_gate_up);
        let mut gate = vec![vec![0.0f64; self.inter]; t + 1];
        let mut up = vec![vec![0.0f64; self.inter]; t + 1];
        let mut silu = vec![vec![0.0f64; self.inter]; t + 1];
        let mut act = vec![vec![0.0f64; self.inter]; t + 1];
        for i in 0..=t {
            for j in 0..self.inter {
                gate[i][j] = gu[i][j];
                up[i][j] = gu[i][self.inter + j];
                silu[i][j] = qwen_silu(gate[i][j]);
                act[i][j] = silu[i][j] * up[i][j];
            }
        }
        let dn = qwen_matmul(&act, &self.w_down);
        let out = qwen_add(&res, &dn);
        let pooled = qwen_matmul(&self.pool, &out);
        let latent = qwen_matmul(&pooled, &self.w_out);
        QwenCache {
            x,
            n1,
            v,
            qr,
            kt,
            probs,
            attn,
            res,
            n2,
            gate,
            up,
            silu,
            act,
            latent,
        }
    }

    fn forward_features(&self, task: &[i64]) -> QwenFeatures {
        let c = self.forward_cached(task);
        QwenFeatures {
            x: c.x,
            attn: c.attn,
            res: c.res,
            mlp: c.act,
        }
    }

    pub fn forward(&self, task: &[i64]) -> Vec<f64> {
        self.forward_cached(task)
            .latent
            .first()
            .cloned()
            .unwrap_or_else(|| vec![0.0; LATENT_NAMES.len()])
    }

    fn backward(&self, c: &QwenCache, target: &[f64], g: &mut QwenGrads) -> f64 {
        let d = target.len();
        let mut loss = 0.0;
        let mut d_latent = vec![vec![0.0f64; d]];
        for l in 0..d {
            let e = c.latent[0][l] - target[l];
            loss += 0.5 * e * e;
            d_latent[0][l] = e / d.max(1) as f64;
        }
        let d_pooled = qwen_matmul(&d_latent, &qwen_transpose(&self.w_out));
        let d_out = qwen_matmul(&qwen_transpose(&self.pool), &d_pooled);
        let mut d_res = d_out.clone();
        let d_dn = d_out.clone();
        let d_act = qwen_matmul(&d_dn, &qwen_transpose(&self.w_down));
        g.w_down = qwen_matmul(&qwen_transpose(&c.act), &d_dn);

        let mut d_gu = vec![vec![0.0f64; 2 * self.inter]; c.act.len()];
        for i in 0..c.act.len() {
            for j in 0..self.inter {
                d_gu[i][j] = d_act[i][j] * c.up[i][j] * qwen_silu_deriv(c.gate[i][j]);
                d_gu[i][self.inter + j] = d_act[i][j] * c.silu[i][j];
            }
        }
        qwen_add_assign(
            &mut g.w_gate_up,
            &qwen_matmul(&qwen_transpose(&c.n2), &d_gu),
        );
        let d_n2 = qwen_matmul(&d_gu, &qwen_transpose(&self.w_gate_up));
        qwen_add_assign(
            &mut d_res,
            &qwen_rmsnorm_backward(&c.res, &self.rms2, &d_n2),
        );

        let mut d_x = d_res.clone();
        let d_ao = d_res;
        let d_attn = qwen_matmul(&d_ao, &qwen_transpose(&self.wo));
        g.wo = qwen_matmul(&qwen_transpose(&c.attn), &d_ao);
        let d_probs = qwen_matmul(&d_attn, &qwen_transpose(&c.v));
        let d_v = qwen_matmul(&qwen_transpose(&c.probs), &d_attn);
        let mut d_scores = qwen_softmax_backward(&c.probs, &d_probs);
        for row in &mut d_scores {
            for value in row.iter_mut() {
                *value *= self.beta_b;
            }
        }
        let d_qr = qwen_matmul(&d_scores, &qwen_transpose(&c.kt));
        let d_kt = qwen_matmul(&qwen_transpose(&c.qr), &d_scores);
        let d_kr = qwen_transpose(&d_kt);
        let pos: Vec<i64> = (0..c.act.len() as i64).collect();
        let d_q = qwen_rotary_backward(&d_qr, &pos, self.dim);
        let d_k = qwen_rotary_backward(&d_kr, &pos, self.dim);
        let mut d_n1 = vec![vec![0.0f64; self.dim]; c.act.len()];
        qwen_add_assign(&mut d_n1, &qwen_matmul(&d_q, &qwen_transpose(&self.wq)));
        qwen_add_assign(&mut d_n1, &qwen_matmul(&d_k, &qwen_transpose(&self.wk)));
        qwen_add_assign(&mut d_n1, &qwen_matmul(&d_v, &qwen_transpose(&self.wv)));
        qwen_add_assign(&mut g.wq, &qwen_matmul(&qwen_transpose(&c.n1), &d_q));
        qwen_add_assign(&mut g.wk, &qwen_matmul(&qwen_transpose(&c.n1), &d_k));
        qwen_add_assign(&mut g.wv, &qwen_matmul(&qwen_transpose(&c.n1), &d_v));
        qwen_add_assign(&mut d_x, &qwen_rmsnorm_backward(&c.x, &self.rms1, &d_n1));
        let _ = d_x;
        loss
    }

    fn apply_branch_grads(&mut self, g: &QwenGrads, lr: f64) {
        let update = |w: &mut [Vec<f64>], gw: &[Vec<f64>]| {
            for (row, grow) in w.iter_mut().zip(gw) {
                for (value, grad) in row.iter_mut().zip(grow) {
                    let clipped = grad.clamp(-0.5, 0.5);
                    *value = (*value - lr * clipped).clamp(-16.0, 16.0);
                }
            }
        };
        update(&mut self.wq, &g.wq);
        update(&mut self.wk, &g.wk);
        update(&mut self.wv, &g.wv);
        update(&mut self.w_gate_up, &g.w_gate_up);
    }

    /// End-to-end f64 backprop through the Qwen block. The readouts are
    /// initialized small if they are still exactly zero, so gradients can
    /// reach Q/K/V and gate/up. The listing is still the exact gate.
    pub fn fit_end_to_end(
        &mut self,
        samples: &[(Vec<i64>, Vec<f64>)],
        epochs: usize,
        lr: f64,
    ) -> f64 {
        if qwen_all_zero(&self.wo) || qwen_all_zero(&self.w_down) {
            let mut state = 0x9e37_79b9_7f4a_7c15u64;
            let mut next = || {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1);
                ((state >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
            };
            for row in &mut self.wo {
                for value in row {
                    *value = next() * 0.02;
                }
            }
            for row in &mut self.w_down {
                for value in row {
                    *value = next() * 0.02;
                }
            }
        }
        let lr = lr.max(1e-6);
        let n = samples.len().max(1) as f64;
        for _ in 0..epochs.max(1) {
            let mut grads = QwenGrads::zeros(self);
            for (task, target) in samples {
                let cache = self.forward_cached(task);
                self.backward(&cache, target, &mut grads);
            }
            for matrix in [
                &mut grads.wq,
                &mut grads.wk,
                &mut grads.wv,
                &mut grads.w_gate_up,
            ] {
                for row in matrix {
                    for value in row {
                        *value /= n;
                    }
                }
            }
            self.apply_branch_grads(&grads, lr);
        }
        samples
            .iter()
            .map(|(task, target)| {
                let got = self.forward(task);
                got.iter()
                    .zip(target)
                    .map(|(x, y)| (x - y).abs())
                    .fold(0.0f64, f64::max)
            })
            .fold(0.0f64, f64::max)
    }

    /// Deterministic small random branch init. The readouts stay zero; the
    /// calibration fit learns `wo` and `w_down`.
    pub fn randomize_branches(&mut self, seed: u64, scale: f64) {
        let mut state = seed | 1;
        let mut next = || {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1);
            ((state >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
        };
        for row in &mut self.wq {
            for value in row {
                *value = next() * scale;
            }
        }
        for row in &mut self.wk {
            for value in row {
                *value = next() * scale;
            }
        }
        for row in &mut self.wv {
            for value in row {
                *value = next() * scale;
            }
        }
        for row in &mut self.w_gate_up {
            for value in row {
                *value = next() * scale;
            }
        }
    }

    /// Fit the attention and MLP readouts on masked calibration tasks. The
    /// targets are the named latent of the unmasked task. The branch features
    /// are fixed during the fit; this is a ridge readout fit, not end-to-end
    /// backprop.
    pub fn fit_readouts(
        &mut self,
        samples: &[(Vec<i64>, Vec<f64>)],
        ridge: f64,
        rounds: usize,
    ) -> f64 {
        let d = LATENT_NAMES.len();
        let dim = self.dim;
        for _ in 0..rounds.max(1) {
            // Attention output projection.
            let features: Vec<QwenFeatures> = samples
                .iter()
                .map(|(task, _)| self.forward_features(task))
                .collect();
            let p = dim * dim;
            let mut design = Vec::new();
            let mut target = Vec::new();
            for (s, (_, want)) in samples.iter().enumerate() {
                let f = &features[s];
                let base = qwen_matmul(&self.pool, &f.x);
                let base = qwen_matmul(&base, &self.w_out);
                for l in 0..d {
                    let mut row = vec![0.0f64; p];
                    for k in 0..dim {
                        let feat_k: f64 = (0..f.attn.len())
                            .map(|t| self.pool[0][t] * f.attn[t][k])
                            .sum();
                        for dd in 0..dim {
                            row[k * dim + dd] = feat_k * self.w_out[dd][l];
                        }
                    }
                    design.push(row);
                    target.push(want[l] - base[0][l]);
                }
            }
            let theta = qwen_solve(&design, &target, ridge);
            for k in 0..dim {
                for dd in 0..dim {
                    self.wo[k][dd] = theta[k * dim + dd];
                }
            }

            // MLP down projection, with the attention readout fixed.
            let features: Vec<QwenFeatures> = samples
                .iter()
                .map(|(task, _)| self.forward_features(task))
                .collect();
            let p = self.inter * dim;
            let mut design = Vec::new();
            let mut target = Vec::new();
            for (s, (_, want)) in samples.iter().enumerate() {
                let f = &features[s];
                let base = qwen_matmul(&self.pool, &f.res);
                let base = qwen_matmul(&base, &self.w_out);
                for l in 0..d {
                    let mut row = vec![0.0f64; p];
                    for j in 0..self.inter {
                        let feat_j: f64 = (0..f.mlp.len())
                            .map(|t| self.pool[0][t] * f.mlp[t][j])
                            .sum();
                        for dd in 0..dim {
                            row[j * dim + dd] = feat_j * self.w_out[dd][l];
                        }
                    }
                    design.push(row);
                    target.push(want[l] - base[0][l]);
                }
            }
            let theta = qwen_solve(&design, &target, ridge);
            for j in 0..self.inter {
                for dd in 0..dim {
                    self.w_down[j][dd] = theta[j * dim + dd];
                }
            }
        }

        samples
            .iter()
            .map(|(task, want)| {
                let got = self.forward(task);
                got.iter()
                    .zip(want)
                    .map(|(x, y)| (x - y).abs())
                    .fold(0.0f64, f64::max)
            })
            .fold(0.0f64, f64::max)
    }
}

impl QwenWeights {
    fn validate(&self) -> Result<(), String> {
        let t = TASK_FEATURE_NAMES.len();
        let d = LATENT_NAMES.len();
        if self.dim < d || !self.dim.is_multiple_of(2) {
            return Err(format!("qwen dim {} must be even and >= {d}", self.dim));
        }
        if self.inter == 0 {
            return Err("qwen inter must be positive".to_string());
        }
        check_matrix("emb", &self.emb, 1, self.dim)?;
        check_matrix("pos_code", &self.pos_code, t + 1, self.dim)?;
        check_matrix("wq", &self.wq, self.dim, self.dim)?;
        check_matrix("wk", &self.wk, self.dim, self.dim)?;
        check_matrix("wv", &self.wv, self.dim, self.dim)?;
        check_matrix("wo", &self.wo, self.dim, self.dim)?;
        check_matrix("w_gate_up", &self.w_gate_up, self.dim, 2 * self.inter)?;
        check_matrix("w_down", &self.w_down, self.inter, self.dim)?;
        if self.rms1.len() != self.dim || self.rms2.len() != self.dim {
            return Err("qwen RMS weights have the wrong shape".to_string());
        }
        check_matrix("pool", &self.pool, 1, t + 1)?;
        check_matrix("w_out", &self.w_out, self.dim, d)?;
        if !self.beta_b.is_finite() || self.beta_b <= 0.0 {
            return Err("qwen beta_b must be positive".to_string());
        }
        if self.m_acc <= 0 || self.m_cov <= 0 {
            return Err("qwen lattice scales must be positive".to_string());
        }
        Ok(())
    }
}

impl Backbone {
    /// The exact named-selection baseline, expressed as a backbone fragment.
    pub fn named(w_enc: &[Vec<i64>], b_enc: &[i64]) -> Self {
        let t = TASK_FEATURE_NAMES.len();
        let mut w1 = vec![vec![0i64; t]; t];
        for (i, row) in w1.iter_mut().enumerate() {
            row[i] = 1;
        }
        Self {
            kind: BackboneKind::Named,
            w1,
            b1: vec![0; t],
            w2: w_enc.to_vec(),
            b2: b_enc.to_vec(),
            scale: 1,
            out_scale: 1,
            relu: false,
            qwen: None,
        }
    }

    /// A two-layer MLP. The hidden layer is a deterministic mixture of task
    /// features; `w2` is fitted to reproduce the named latent.
    pub fn mlp(w_enc: &[Vec<i64>], b_enc: &[i64], hidden: usize, mix: i64) -> Self {
        let t = TASK_FEATURE_NAMES.len();
        let d = LATENT_NAMES.len();
        let hidden = hidden.clamp(d, 64);
        let mut w1 = vec![vec![0i64; t]; hidden];
        for (j, row) in w1.iter_mut().enumerate() {
            let src = j % t;
            row[src] = 8;
            if mix != 0 {
                row[(src + 1) % t] = mix;
            }
        }
        // Seed only the first T hidden units with the named readout; the extra
        // units start unused and the fit decides whether to use them. This
        // keeps the mix=0 case an exact reparameterization. The readout is
        // stored in fixed point so the fit can express fractional weights.
        let out_scale = 64i64;
        let mut w2 = vec![vec![0i64; hidden]; d];
        for (l, row) in w2.iter_mut().enumerate() {
            for (j, value) in row.iter_mut().enumerate().take(t) {
                *value = w_enc[l][j] * out_scale;
            }
        }
        Self {
            kind: BackboneKind::Mlp,
            w1,
            b1: vec![0; hidden],
            w2,
            b2: b_enc.iter().map(|v| v * out_scale).collect(),
            scale: 8,
            out_scale,
            relu: true,
            qwen: None,
        }
    }

    /// A Qwen-style block initialized as an exact linear pass-through of the
    /// named latent. The attention output projection and the MLP down
    /// projection are zero, so the residual stream carries the token codes;
    /// the pool + readout reproduce the named selection. The branch weights
    /// are present and trainable in the listing.
    pub fn qwen_named(w_enc: &[Vec<i64>], b_enc: &[i64], dim: usize, inter: usize) -> Self {
        let t = TASK_FEATURE_NAMES.len();
        let d = LATENT_NAMES.len();
        let mut dim = dim.max(d);
        if !dim.is_multiple_of(2) {
            dim += 1;
        }
        let inter = inter.max(dim);
        let tokens = t + 1;
        let emb = vec![vec![1.0; dim]];
        let mut pos_code = vec![vec![0.0; dim]; tokens];
        for (j, row) in pos_code.iter_mut().enumerate().take(t) {
            for (l, value) in row.iter_mut().enumerate().take(d) {
                *value = w_enc[l][j] as f64;
            }
        }
        for (l, value) in pos_code[t].iter_mut().enumerate().take(d) {
            *value = b_enc[l] as f64;
        }
        let mut w_out = vec![vec![0.0; d]; dim];
        for (l, row) in w_out.iter_mut().enumerate().take(d) {
            row[l] = 1.0;
        }
        let qwen = QwenWeights {
            dim,
            inter,
            emb,
            pos_code,
            wq: vec![vec![0.0; dim]; dim],
            wk: vec![vec![0.0; dim]; dim],
            wv: vec![vec![0.0; dim]; dim],
            wo: vec![vec![0.0; dim]; dim],
            w_gate_up: vec![vec![0.0; 2 * inter]; dim],
            w_down: vec![vec![0.0; dim]; inter],
            rms1: vec![1.0; dim],
            rms2: vec![1.0; dim],
            pool: vec![vec![1.0; tokens]],
            w_out,
            beta_b: 1.0 / (dim as f64).sqrt(),
            m_acc: 38000,
            m_cov: 38000,
        };
        Self {
            kind: BackboneKind::Qwen,
            w1: Vec::new(),
            b1: Vec::new(),
            w2: Vec::new(),
            b2: Vec::new(),
            scale: 1,
            out_scale: 1,
            relu: false,
            qwen: Some(qwen),
        }
    }

    /// A Qwen block with deterministic small random Q/K/V/gate-up weights.
    /// The output projections stay zero; the calibration fit learns them.
    pub fn qwen_random(
        w_enc: &[Vec<i64>],
        b_enc: &[i64],
        dim: usize,
        inter: usize,
        seed: u64,
        scale: f64,
    ) -> Self {
        let mut backbone = Self::qwen_named(w_enc, b_enc, dim, inter);
        if let Some(qwen) = &mut backbone.qwen {
            qwen.randomize_branches(seed, scale);
        }
        backbone
    }

    pub fn hidden(&self, task: &[i64]) -> Vec<i64> {
        let scale = self.scale.max(1);
        self.w1
            .iter()
            .enumerate()
            .map(|(j, row)| {
                let mut sum = self.b1[j];
                for (i, value) in task.iter().enumerate() {
                    sum += value * row[i];
                }
                if self.relu {
                    sum = sum.max(0);
                }
                sum / scale
            })
            .collect()
    }

    pub fn latent(&self, task: &[i64]) -> Vec<i64> {
        if self.kind == BackboneKind::Qwen {
            return self.qwen_latent(task);
        }
        let hidden = self.hidden(task);
        let scale = self.out_scale.max(1);
        self.w2
            .iter()
            .enumerate()
            .map(|(l, row)| {
                let mut sum = self.b2[l];
                for (j, value) in hidden.iter().enumerate() {
                    sum += value * row[j];
                }
                sum / scale
            })
            .collect()
    }

    /// Structural pass-through used by the initial Qwen slice. The branch
    /// weights are zero, so the output is the pooled token code; if the
    /// branches are trained later, this must be replaced by a full forward.
    fn qwen_latent(&self, task: &[i64]) -> Vec<i64> {
        let Some(q) = &self.qwen else {
            return vec![0; LATENT_NAMES.len()];
        };
        q.forward(task)
            .iter()
            .map(|value| value.round() as i64)
            .collect()
    }

    /// Fit `w2`/`b2` so the MLP reproduces the named latent on the samples.
    /// The hidden layer stays fixed; this solves the ridge-regularized linear
    /// readout in the dual (samples x samples) and rounds back to integers.
    pub fn fit(&mut self, tasks: &[Vec<i64>], targets: &[Vec<i64>], lr: i64) -> i64 {
        if self.kind == BackboneKind::Qwen {
            return 0;
        }
        let n = tasks.len();
        if n == 0 || targets.is_empty() {
            return 0;
        }
        let d = targets[0].len();
        let h = self.w1.len();
        // A larger `lr` allows a larger correction away from the named readout.
        let lambda = 10_000.0 / lr.max(1) as f64;

        // A: N x (H+1), with a trailing bias column.
        let mut a = Vec::with_capacity(n);
        for task in tasks {
            let hidden = self.hidden(task);
            let mut row: Vec<f64> = hidden.iter().map(|&v| v as f64).collect();
            row.push(1.0);
            a.push(row);
        }
        // B: N x D.
        let b: Vec<Vec<f64>> = targets
            .iter()
            .map(|row| row.iter().map(|&v| v as f64).collect())
            .collect();

        // Ridge correction around the seeded named readout:
        //   min ||A W - B||^2 + lambda ||W - W0||^2
        // solved in the dual.
        let out_scale = self.out_scale.max(1) as f64;
        let w0: Vec<Vec<f64>> = self
            .w2
            .iter()
            .map(|row| row.iter().map(|&v| v as f64 / out_scale).collect())
            .collect();
        let b0: Vec<f64> = self.b2.iter().map(|&v| v as f64 / out_scale).collect();
        let mut residual = vec![vec![0.0f64; d]; n];
        for i in 0..n {
            for l in 0..d {
                let mut pred = b0[l];
                for j in 0..h {
                    pred += a[i][j] * w0[l][j];
                }
                residual[i][l] = b[i][l] - pred;
            }
        }

        let mut g = vec![vec![0.0f64; n]; n];
        for i in 0..n {
            for j in 0..n {
                g[i][j] = a[i].iter().zip(&a[j]).map(|(x, y)| x * y).sum();
            }
            g[i][i] += lambda;
        }
        let alpha = solve_linear(g, residual);
        let mut w = vec![vec![0.0f64; d]; h + 1];
        for (k, row) in w.iter_mut().enumerate() {
            for (l, value) in row.iter_mut().enumerate() {
                let correction: f64 = (0..n).map(|i| a[i][k] * alpha[i][l]).sum();
                *value = if k < h {
                    w0[l][k] + correction
                } else {
                    b0[l] + correction
                };
            }
        }

        self.w2 = (0..d)
            .map(|l| {
                (0..h)
                    .map(|j| {
                        (w[j][l] * out_scale)
                            .round()
                            .clamp(-4096.0 * out_scale, 4096.0 * out_scale)
                            as i64
                    })
                    .collect()
            })
            .collect();
        self.b2 = (0..d)
            .map(|l| {
                (w[h][l] * out_scale)
                    .round()
                    .clamp(-4096.0 * out_scale, 4096.0 * out_scale) as i64
            })
            .collect();

        tasks
            .iter()
            .zip(targets)
            .map(|(task, target)| {
                let got = self.latent(task);
                got.iter()
                    .zip(target)
                    .map(|(x, y)| (x - y).abs())
                    .max()
                    .unwrap_or(0)
            })
            .max()
            .unwrap_or(0)
    }

    /// Emit the backbone fragment and return the initial latent stream names.
    pub fn build_h0(&self, lines: &mut Vec<String>) -> Vec<String> {
        if self.kind == BackboneKind::Qwen {
            return self.build_qwen_h0(lines);
        }
        let hidden = self.w1.len();
        let mut acts = Vec::with_capacity(hidden);
        for j in 0..hidden {
            let mut terms = Vec::new();
            for (i, weight) in self.w1[j].iter().enumerate() {
                if *weight != 0 {
                    let term = format!("bb1_{j}_{i}");
                    lines.push(format!("{term} = IMUL(t{i}, {weight})"));
                    terms.push(term);
                }
            }
            let sum = sum_terms(lines, &format!("bbh_{j}"), terms, self.b1[j]);
            let act = if self.relu {
                let name = format!("bbr_{j}");
                lines.push(format!("{name} = IMAX({sum}, 0)"));
                name
            } else {
                sum
            };
            let act = if self.scale > 1 {
                let name = format!("bbs_{j}");
                lines.push(format!("{name} = IDIV({act}, {})", self.scale));
                name
            } else {
                act
            };
            let name = format!("bb_h{j}");
            lines.push(format!("{name} = IADD({act}, 0)"));
            acts.push(name);
        }

        let mut latent = Vec::with_capacity(self.w2.len());
        for l in 0..self.w2.len() {
            let mut terms = Vec::new();
            for (j, weight) in self.w2[l].iter().enumerate() {
                if *weight != 0 {
                    let term = format!("bb2_{l}_{j}");
                    lines.push(format!("{term} = IMUL({}, {weight})", acts[j]));
                    terms.push(term);
                }
            }
            let sum = sum_terms(lines, &format!("bbo_{l}"), terms, self.b2[l]);
            let name = format!("h0_{l}");
            if self.out_scale > 1 {
                let divided = format!("h0d_{l}");
                lines.push(format!("{divided} = IDIV({sum}, {})", self.out_scale));
                lines.push(format!("{name} = IADD({divided}, 0)"));
            } else {
                lines.push(format!("{name} = IADD({sum}, 0)"));
            }
            latent.push(name);
        }
        latent
    }

    fn build_qwen_h0(&self, lines: &mut Vec<String>) -> Vec<String> {
        let q = self.qwen.as_ref().expect("qwen weights");
        let d = LATENT_NAMES.len();
        lines.push("qwen_x0 = MATMUL(tokens, emb)".to_string());
        lines.push("qwen_x = MUL(qwen_x0, pos_code)".to_string());
        lines.push("qwen_n1 = RMSNORM(qwen_x, rms1)".to_string());
        lines.push("qwen_q = MATMUL(qwen_n1, wq)".to_string());
        lines.push("qwen_k = MATMUL(qwen_n1, wk)".to_string());
        lines.push("qwen_v = MATMUL(qwen_n1, wv)".to_string());
        lines.push("qwen_qr = ROTARY(qwen_q, pos)".to_string());
        lines.push("qwen_kr = ROTARY(qwen_k, pos)".to_string());
        lines.push("qwen_kt = TRANSPOSE(qwen_kr)".to_string());
        lines.push("qwen_sc = MATMUL(qwen_qr, qwen_kt)".to_string());
        lines.push("qwen_bt = TBETA(qwen_sc)".to_string());
        lines.push("qwen_sc2 = MUL(qwen_sc, qwen_bt)".to_string());
        lines.push("qwen_sh = TSHIFT(qwen_sc2)".to_string());
        lines.push("qwen_pr = SOFTMAX_WIDE(qwen_sh)".to_string());
        lines.push("qwen_at = MATMUL(qwen_pr, qwen_v)".to_string());
        lines.push("qwen_ao = MATMUL(qwen_at, wo)".to_string());
        lines.push("qwen_res = ADD(qwen_x, qwen_ao)".to_string());
        lines.push("qwen_n2 = RMSNORM(qwen_res, rms2)".to_string());
        lines.push("qwen_gu = MATMUL(qwen_n2, w_gate_up)".to_string());
        lines.push(format!("qwen_gate = SLICE(qwen_gu, 1, 0, {})", q.inter));
        lines.push(format!(
            "qwen_up = SLICE(qwen_gu, 1, {}, {})",
            q.inter,
            2 * q.inter
        ));
        lines.push("qwen_silu = SILU(qwen_gate)".to_string());
        lines.push("qwen_act = MUL(qwen_silu, qwen_up)".to_string());
        lines.push("qwen_dn = MATMUL(qwen_act, w_down)".to_string());
        lines.push("qwen_out = ADD(qwen_res, qwen_dn)".to_string());
        lines.push("qwen_pooled = MATMUL(pool, qwen_out)".to_string());
        lines.push("qwen_latent_t = MATMUL(qwen_pooled, w_out)".to_string());
        lines.push("qwen_latent = T2I(qwen_latent_t)".to_string());
        (0..d)
            .map(|i| {
                let name = format!("h0_{i}");
                lines.push(format!("{name} = IGET(qwen_latent, {i})"));
                name
            })
            .collect()
    }

    /// Extra `IN` declarations required by the backbone.
    pub fn input_decls(&self) -> Vec<&'static str> {
        if self.kind == BackboneKind::Qwen {
            vec![
                "tokens AS T:*",
                "pos AS I:SEQ",
                "emb AS T:*",
                "pos_code AS T:*",
                "wq AS T:*",
                "wk AS T:*",
                "wv AS T:*",
                "wo AS T:*",
                "w_gate_up AS T:*",
                "w_down AS T:*",
                "rms1 AS T:*",
                "rms2 AS T:*",
                "pool AS T:*",
                "w_out AS T:*",
            ]
        } else {
            Vec::new()
        }
    }

    pub fn config(&self) -> Vec<(String, String)> {
        match &self.qwen {
            Some(qwen) => vec![
                ("m_acc".to_string(), qwen.m_acc.to_string()),
                ("m_cov".to_string(), qwen.m_cov.to_string()),
                ("beta_b".to_string(), format!("{}", qwen.beta_b)),
            ],
            None => Vec::new(),
        }
    }

    /// Payload feeds for the backbone's extra inputs.
    pub fn feeds(&self, task: &[i64]) -> Result<Vec<(String, Value)>, String> {
        let Some(q) = &self.qwen else {
            return Ok(Vec::new());
        };
        let t = TASK_FEATURE_NAMES.len();
        let mut tokens = vec![0.0; t + 1];
        for (i, value) in task.iter().enumerate().take(t) {
            tokens[i] = *value as f64;
        }
        tokens[t] = 1.0;
        let pos: Vec<i64> = (0..=t as i64).collect();
        Ok(vec![
            ("tokens".to_string(), triple_vec(&tokens, &[t + 1, 1])?),
            (
                "pos".to_string(),
                Value::I64(
                    ArrayD::from_shape_vec(IxDyn(&[t + 1]), pos).map_err(|e| e.to_string())?,
                ),
            ),
            ("emb".to_string(), triple_matrix(&q.emb)?),
            ("pos_code".to_string(), triple_matrix(&q.pos_code)?),
            ("wq".to_string(), triple_matrix(&q.wq)?),
            ("wk".to_string(), triple_matrix(&q.wk)?),
            ("wv".to_string(), triple_matrix(&q.wv)?),
            ("wo".to_string(), triple_matrix(&q.wo)?),
            ("w_gate_up".to_string(), triple_matrix(&q.w_gate_up)?),
            ("w_down".to_string(), triple_matrix(&q.w_down)?),
            ("rms1".to_string(), triple_vec(&q.rms1, &[q.dim])?),
            ("rms2".to_string(), triple_vec(&q.rms2, &[q.dim])?),
            ("pool".to_string(), triple_matrix(&q.pool)?),
            ("w_out".to_string(), triple_matrix(&q.w_out)?),
        ])
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.kind == BackboneKind::Qwen {
            let qwen = self
                .qwen
                .as_ref()
                .ok_or_else(|| "qwen backbone is missing its weights".to_string())?;
            return qwen.validate();
        }
        let t = TASK_FEATURE_NAMES.len();
        let d = LATENT_NAMES.len();
        let h = self.w1.len();
        if self.b1.len() != h || self.w1.iter().any(|row| row.len() != t) {
            return Err("backbone hidden layer has the wrong shape".to_string());
        }
        if self.w2.len() != d || self.w2.iter().any(|row| row.len() != h) || self.b2.len() != d {
            return Err("backbone readout has the wrong shape".to_string());
        }
        if self.scale < 1 || self.out_scale < 1 {
            return Err("backbone scales must be positive".to_string());
        }
        Ok(())
    }

    pub fn save(&self, path: &std::path::Path) -> Result<(), String> {
        self.validate()?;
        let text = serde_json::to_string_pretty(self).map_err(|e| e.to_string())?;
        std::fs::write(path, text + "\n").map_err(|e| format!("{}: {e}", path.display()))
    }

    pub fn load(path: &std::path::Path) -> Result<Self, String> {
        let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
        let backbone: Self = serde_json::from_str(&text).map_err(|e| e.to_string())?;
        backbone.validate()?;
        Ok(backbone)
    }
}
