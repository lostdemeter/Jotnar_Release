//! Shared exact integer helpers (port of `phi_core/ops.py`).

use ndarray::{ArrayD, IxDyn, Zip};

use crate::error::{Error, Result};
use crate::lattice;
use crate::triple::Triple;

/// Scale-table convention: covering scales live under `cov:<blob>`.
pub fn m_cov<'a>(m: &'a std::collections::BTreeMap<String, u32>, blob: &str) -> u32 {
    *m.get(&format!("cov:{blob}")).unwrap_or(&0)
}

/// Constant triple filled with a scalar value.
pub fn const_like(shape: &[usize], v: f64) -> Triple {
    let one = lattice::encode(&ArrayD::from_elem(IxDyn(&[1]), v));
    let s = one.s.iter().next().copied().unwrap_or(1);
    let e = one.e.iter().next().copied().unwrap_or(0);
    Triple {
        s: ArrayD::from_elem(IxDyn(shape), s),
        e: ArrayD::from_elem(IxDyn(shape), e),
        z: ArrayD::from_elem(IxDyn(shape), 0u8),
        m: None,
    }
}

/// Broadcast a scalar or per-channel slope triple to `(H,W,C)`.
pub fn slope_triples(slope: &Triple, shape: &[usize]) -> Result<Triple> {
    let c = *shape
        .get(2)
        .ok_or_else(|| Error::runtime("slope_triples needs HWC"))?;
    let n = shape.iter().product::<usize>();
    if slope.numel() == 1 {
        return Ok(const_from_scalar(slope, shape));
    }
    if slope.numel() != c {
        return Err(Error::runtime(format!(
            "slope count {} vs C {c}",
            slope.numel()
        )));
    }
    let ss: Vec<i8> = slope.s.iter().copied().collect();
    let se: Vec<i32> = slope.e.iter().copied().collect();
    let sz: Vec<u8> = slope.z.iter().copied().collect();
    Ok(Triple::from_vecs(
        shape,
        (0..n).map(|i| ss[i % c]).collect(),
        (0..n).map(|i| se[i % c]).collect(),
        (0..n).map(|i| sz[i % c]).collect(),
    )?)
}

fn const_from_scalar(t: &Triple, shape: &[usize]) -> Triple {
    Triple {
        s: ArrayD::from_elem(IxDyn(shape), t.s.iter().next().copied().unwrap_or(1)),
        e: ArrayD::from_elem(IxDyn(shape), t.e.iter().next().copied().unwrap_or(0)),
        z: ArrayD::from_elem(IxDyn(shape), t.z.iter().next().copied().unwrap_or(0)),
        m: t.m
            .as_ref()
            .map(|m| ArrayD::from_elem(IxDyn(shape), m.iter().next().copied().unwrap_or(0))),
    }
}

/// Lattice greater-than, exact ordering, zero==zero false.
pub fn cmp_gt(a: &Triple, b: &Triple) -> Result<ArrayD<bool>> {
    if a.shape() != b.shape() {
        return Err(Error::runtime("cmp_gt shape mismatch"));
    }
    let af_s: Vec<i8> = a.s.iter().copied().collect();
    let af_e: Vec<i32> = a.e.iter().copied().collect();
    let af_z: Vec<u8> = a.z.iter().copied().collect();
    let bf_s: Vec<i8> = b.s.iter().copied().collect();
    let bf_e: Vec<i32> = b.e.iter().copied().collect();
    let bf_z: Vec<u8> = b.z.iter().copied().collect();
    let mut out = Vec::with_capacity(a.numel());
    for i in 0..a.numel() {
        let (sa, ea, za) = (af_s[i], af_e[i], af_z[i]);
        let (sb, eb, zb) = (bf_s[i], bf_e[i], bf_z[i]);
        let az = za != 0;
        let bz = zb != 0;
        let live = !az && !bz;
        let same = live && sa == sb;
        out.push(
            (same && ea > eb && sa > 0)
                || (same && ea < eb && sa < 0)
                || (live && sa > sb)
                || (!az && bz && sa > 0),
        );
    }
    Ok(ArrayD::from_shape_vec(IxDyn(&a.shape()), out)
        .map_err(|e| Error::runtime(format!("cmp_gt shape: {e}")))?)
}

/// Clip triples in the lattice domain.
pub fn clip_int(t: &Triple, lo: f64, hi: f64) -> Result<Triple> {
    let lo_t = const_like(&t.shape(), lo);
    let hi_t = const_like(&t.shape(), hi);
    let lo_gt = cmp_gt(&lo_t, t)?;
    let s = ndarray::Zip::from(&t.s)
        .and(&lo_t.s)
        .and(&lo_gt)
        .map_collect(|&x, &l, &m| if m { l } else { x });
    let e = ndarray::Zip::from(&t.e)
        .and(&lo_t.e)
        .and(&lo_gt)
        .map_collect(|&x, &l, &m| if m { l } else { x });
    let z = ndarray::Zip::from(&t.z)
        .and(&lo_t.z)
        .and(&lo_gt)
        .map_collect(|&x, &l, &m| if m { l } else { x });
    let mid = Triple::new(s, e, z)?;
    let hi_gt = cmp_gt(&mid, &hi_t)?;
    let s = ndarray::Zip::from(&mid.s)
        .and(&hi_t.s)
        .and(&hi_gt)
        .map_collect(|&x, &h, &m| if m { h } else { x });
    let e = ndarray::Zip::from(&mid.e)
        .and(&hi_t.e)
        .and(&hi_gt)
        .map_collect(|&x, &h, &m| if m { h } else { x });
    let z = ndarray::Zip::from(&mid.z)
        .and(&hi_t.z)
        .and(&hi_gt)
        .map_collect(|&x, &h, &m| if m { h } else { x });
    Triple::new(s, e, z)
}

/// Binary op: 0 add, 1 sub, 2 mul, 7 reversed sub.
pub fn bin2(op: u8, a: &Triple, _ma: i64, c: &Triple, _mc: i64, m_out: i64) -> Result<Triple> {
    if op == 2 {
        return crate::numpy_ops::tmul(a, c);
    }
    let fa = lattice::to_fixed(a, m_out);
    let fc = lattice::to_fixed(c, m_out);
    let q = match op {
        0 => Zip::from(&fa)
            .and(&fc)
            .map_collect(|&x, &y| x.wrapping_add(y)),
        1 => Zip::from(&fa)
            .and(&fc)
            .map_collect(|&x, &y| x.wrapping_sub(y)),
        7 => Zip::from(&fa)
            .and(&fc)
            .map_collect(|&x, &y| y.wrapping_sub(x)),
        _ => return Err(Error::runtime(format!("binary op {op}"))),
    };
    lattice::from_fixed(&q, m_out)
}
