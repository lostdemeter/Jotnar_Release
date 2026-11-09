//! Triples arithmetic and the shared fixed-point helpers

use ndarray::{ArrayD, Zip};

use crate::error::{Error, Result};
use crate::ir::Config;
use crate::lattice::{self, BIAS, NLVL};
use crate::value::{Arg, Triple, Value};

use super::{arg_f64, arg_i64, arg_triple, beta, scales, triple_binop, triple_map, triple_where};

pub fn tmul(a: &Triple, b: &Triple) -> Result<Triple> {
    triple_binop(a, b, "MUL", |as_, ae, az, bs, be, bz| {
        let s = ((as_ as i16) * (bs as i16)) as i8;
        let e = (ae as i64 + be as i64 - BIAS).clamp(0, NLVL - 1) as i32;
        (s, e, az | bz)
    })
}

pub fn tdiv_pure(a: &Triple, b: &Triple) -> Result<Triple> {
    triple_binop(a, b, "DIV", |as_, ae, az, bs, be, bz| {
        let s = ((as_ as i16) * (bs as i16)) as i8;
        let e = (ae as i64 - be as i64 + BIAS).clamp(0, NLVL - 1) as i32;
        (s, e, az | bz)
    })
}

pub fn tdiv_trip(a: &Triple, b: &Triple) -> Result<Triple> {
    let raw = tdiv_pure(a, b)?;
    let one = lattice::encode(&ArrayD::from_elem(ndarray::IxDyn(&a.shape()), 1.0f64));
    triple_where(&b.z.mapv(|z| z != 0), &one, &raw, "tdiv_trip")
}

pub fn sqrt_trip(t: &Triple) -> Result<Triple> {
    triple_map(t, |_, e, z| {
        let e2 = ((e as i64 - BIAS).div_euclid(2) + BIAS).clamp(0, NLVL - 1);
        (1, e2 as i32, z)
    })
}

pub fn abs_trip(t: &Triple) -> Result<Triple> {
    Ok(Triple {
        s: t.s.mapv(|_| 1i8),
        e: t.e.clone(),
        z: t.z.clone(),
    })
}

pub fn neg_trip(t: &Triple) -> Result<Triple> {
    Ok(Triple {
        s: t.s.mapv(|s| -s),
        e: t.e.clone(),
        z: t.z.clone(),
    })
}

pub fn binop_fixed(a: &Triple, b: &Triple, m_a: i64, m_b: i64, op: &str) -> Result<Triple> {
    if m_a != m_b {
        return Err(Error::runtime(format!(
            "unit confusion: {m_a} vs {m_b} -- rescale explicitly"
        )));
    }
    let qa = lattice::to_fixed(a, m_a);
    let qb = lattice::to_fixed(b, m_b);
    let q = if op == "add" {
        Zip::from(&qa)
            .and(&qb)
            .map_collect(|&x, &y| x.wrapping_add(y))
    } else {
        Zip::from(&qa)
            .and(&qb)
            .map_collect(|&x, &y| x.wrapping_sub(y))
    };
    lattice::assert_bound(&format!("{op}:bound"), &[&q])?;
    lattice::from_fixed(&q, m_a)
}

pub fn clip_fixed(t: &Triple, lo: f64, hi: f64, m: i64) -> Result<Triple> {
    let q = lattice::to_fixed(t, m);
    let lo_t = lattice::encode(&ArrayD::from_elem(ndarray::IxDyn(&[1]), lo));
    let hi_t = lattice::encode(&ArrayD::from_elem(ndarray::IxDyn(&[1]), hi));
    let qlo = *lattice::to_fixed(&lo_t, m).iter().next().unwrap_or(&0);
    let qhi = *lattice::to_fixed(&hi_t, m).iter().next().unwrap_or(&0);
    let clipped = q.mapv(|v| v.max(qlo).min(qhi));
    lattice::from_fixed(&clipped, m)
}

pub fn relu_trip(t: &Triple, m: i64) -> Result<Triple> {
    let q = lattice::to_fixed(t, m);
    let q = q.mapv(|v| v.max(0));
    lattice::from_fixed(&q, m)
}

pub fn sigmoid_trip(t: &Triple) -> Result<Triple> {
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
        let idx = x14 + span;
        let y14 = if x14 < -span {
            0
        } else if x14 > span {
            16384
        } else {
            glut[idx.clamp(0, glut.len() as i64 - 1) as usize]
        };
        out.push(y14 * 16);
    }
    let q = crate::value::array_from_vec(&t.shape(), out)?;
    lattice::from_fixed(&q, BIAS)
}

pub fn select_mux(trips: &[Triple], bucket: &ArrayD<i64>) -> Result<Triple> {
    if trips.len() < 2 {
        return Err(Error::runtime("mux needs >= 2 streams"));
    }
    let last = trips.last().expect("nonempty");
    let expected = last.shape();
    for t in trips {
        if t.shape() != expected {
            return Err(Error::runtime("select_mux: stream shape mismatch"));
        }
    }
    if bucket.shape() != expected.as_slice() {
        return Err(Error::runtime(format!(
            "select_mux: bucket shape {:?} vs {:?}",
            bucket.shape(),
            expected
        )));
    }
    let mut s = Vec::with_capacity(last.numel());
    let mut e = Vec::with_capacity(last.numel());
    let mut z = Vec::with_capacity(last.numel());
    let lasts = last.s.iter().copied().collect::<Vec<_>>();
    let laste = last.e.iter().copied().collect::<Vec<_>>();
    let lastz = last.z.iter().copied().collect::<Vec<_>>();
    let trip_planes: Vec<(Vec<i8>, Vec<i32>, Vec<u8>)> = trips
        .iter()
        .map(|t| {
            (
                t.s.iter().copied().collect(),
                t.e.iter().copied().collect(),
                t.z.iter().copied().collect(),
            )
        })
        .collect();
    for (n, &b) in bucket.iter().enumerate() {
        let mut chosen = (lasts[n], laste[n], lastz[n]);
        for (i, (ts, te, tz)) in trip_planes.iter().enumerate().take(trips.len() - 1) {
            if b == i as i64 {
                chosen = (ts[n], te[n], tz[n]);
            }
        }
        s.push(chosen.0);
        e.push(chosen.1);
        z.push(chosen.2);
    }
    Triple::from_vecs(&expected, s, e, z)
}

// ---------------------------------------------------------------------------
// Public mnemonics
// ---------------------------------------------------------------------------

pub fn op_add(args: &[Arg], config: &Config) -> Result<Vec<Value>> {
    let a = arg_triple(args, 0, "ADD")?;
    let b = arg_triple(args, 1, "ADD")?;
    let (_, m_cov) = scales(config);
    Ok(vec![Value::Triple(binop_fixed(a, b, m_cov, m_cov, "add")?)])
}

pub fn op_sub(args: &[Arg], config: &Config) -> Result<Vec<Value>> {
    let a = arg_triple(args, 0, "SUB")?;
    let b = arg_triple(args, 1, "SUB")?;
    let (_, m_cov) = scales(config);
    Ok(vec![Value::Triple(binop_fixed(a, b, m_cov, m_cov, "sub")?)])
}

pub fn op_mul(args: &[Arg], _: &Config) -> Result<Vec<Value>> {
    let a = arg_triple(args, 0, "MUL")?;
    let b = arg_triple(args, 1, "MUL")?;
    Ok(vec![Value::Triple(tmul(a, b)?)])
}

pub fn op_div(args: &[Arg], _: &Config) -> Result<Vec<Value>> {
    let a = arg_triple(args, 0, "DIV")?;
    let b = arg_triple(args, 1, "DIV")?;
    Ok(vec![Value::Triple(tdiv_pure(a, b)?)])
}

pub fn op_square(args: &[Arg], config: &Config) -> Result<Vec<Value>> {
    let a = arg_triple(args, 0, "SQUARE")?;
    let (_, m_cov) = scales(config);
    Ok(vec![Value::Triple(clip_fixed(
        &tmul(a, a)?,
        0.0,
        1.0,
        m_cov,
    )?)])
}

pub fn op_relu(args: &[Arg], config: &Config) -> Result<Vec<Value>> {
    let v = args
        .first()
        .ok_or_else(|| Error::runtime("RELU: missing input"))?
        .stream("RELU", "x")?;
    match v {
        Value::F64(a) => Ok(vec![Value::F64(a.mapv(|x| x.max(0.0)))]),
        Value::F32(a) => Ok(vec![Value::F32(a.mapv(|x| x.max(0.0)))]),
        Value::Triple(t) => {
            let (_, m_cov) = scales(config);
            Ok(vec![Value::Triple(relu_trip(t, m_cov)?)])
        }
        _ => Err(Error::runtime("RELU: needs F/T")),
    }
}

pub fn op_sqrt(args: &[Arg], _: &Config) -> Result<Vec<Value>> {
    let arg = args
        .first()
        .ok_or_else(|| Error::runtime("SQRT: missing input"))?;
    let y = arg.stream("SQRT", "y")?.as_f64("SQRT")?;
    let t = lattice::encode(&y);
    Ok(vec![Value::Triple(sqrt_trip(&t)?)])
}

pub fn op_beta(args: &[Arg], config: &Config) -> Result<Vec<Value>> {
    let ref_t = arg_triple(args, 0, "BETA")?;
    let b = beta(config);
    let fill = ArrayD::from_elem(ndarray::IxDyn(&ref_t.shape()), b);
    Ok(vec![Value::Triple(lattice::encode(&fill))])
}

pub fn op_tbeta(args: &[Arg], config: &Config) -> Result<Vec<Value>> {
    let ref_t = arg_triple(args, 0, "TBETA")?;
    let b = config.get_f64_or("beta_b", beta(config));
    let fill = ArrayD::from_elem(ndarray::IxDyn(&ref_t.shape()), b);
    Ok(vec![Value::Triple(lattice::encode(&fill))])
}

pub fn op_clip(args: &[Arg], config: &Config) -> Result<Vec<Value>> {
    let t = arg_triple(args, 0, "CLIP")?;
    let lo = arg_f64(args, 1, "CLIP", "lo")?;
    let hi = arg_f64(args, 2, "CLIP", "hi")?;
    let (_, m_cov) = scales(config);
    Ok(vec![Value::Triple(clip_fixed(t, lo, hi, m_cov)?)])
}

pub fn op_sigmoid(args: &[Arg], _: &Config) -> Result<Vec<Value>> {
    let t = arg_triple(args, 0, "SIGMOID")?;
    Ok(vec![Value::Triple(sigmoid_trip(t)?)])
}

pub fn op_rescale(args: &[Arg], config: &Config) -> Result<Vec<Value>> {
    let t = arg_triple(args, 0, "RESCALE")?;
    let m2 = arg_i64(args, 1, "RESCALE", "m2")?;
    if !(0..65536).contains(&m2) {
        return Err(Error::runtime(format!("RESCALE: scale out of range: {m2}")));
    }
    let (_, m_cov) = scales(config);
    let q = lattice::to_fixed(t, m_cov);
    let q2 = super::rescale_checked(&q, m_cov, m2)?;
    Ok(vec![Value::Triple(lattice::from_fixed(&q2, m2)?)])
}

fn ordkeys(t: &Triple) -> (ArrayD<i64>, ArrayD<i64>) {
    let mut cls = Vec::with_capacity(t.numel());
    let mut key = Vec::with_capacity(t.numel());
    for ((&s, &e), &z) in t.s.iter().zip(t.e.iter()).zip(t.z.iter()) {
        let pos = z == 0 && s > 0;
        let zero = z != 0;
        let c = if pos {
            2i64
        } else if zero {
            1
        } else {
            0
        };
        let k = if pos {
            e as i64
        } else if zero {
            0
        } else {
            -(e as i64)
        };
        cls.push(c);
        key.push(k);
    }
    (
        ArrayD::from_shape_vec(ndarray::IxDyn(&t.shape()), cls).expect("cls"),
        ArrayD::from_shape_vec(ndarray::IxDyn(&t.shape()), key).expect("key"),
    )
}

pub fn op_gt(args: &[Arg], _: &Config) -> Result<Vec<Value>> {
    let a = arg_triple(args, 0, "GT")?;
    let b = arg_triple(args, 1, "GT")?;
    if a.shape() != b.shape() {
        return Err(Error::runtime(format!(
            "GT: shape mismatch {:?} vs {:?}",
            a.shape(),
            b.shape()
        )));
    }
    let (ca, ka) = ordkeys(a);
    let (cb, kb) = ordkeys(b);
    let out = Zip::from(&ca)
        .and(&ka)
        .and(&cb)
        .and(&kb)
        .map_collect(|&c1, &k1, &c2, &k2| c1 > c2 || (c1 == c2 && k1 > k2));
    Ok(vec![Value::Bool(out)])
}

pub fn op_select(args: &[Arg], _: &Config) -> Result<Vec<Value>> {
    let mask_v = args
        .first()
        .ok_or_else(|| Error::runtime("SELECT: missing mask"))?
        .stream("SELECT", "mask")?;
    let a = arg_triple(args, 1, "SELECT")?;
    let b = arg_triple(args, 2, "SELECT")?;
    let mask =
        match mask_v {
            Value::Bool(m) => m.clone(),
            Value::I64(m) => m.mapv(|v| v != 0),
            Value::U8(m) => m.mapv(|v| v != 0),
            Value::F32(_) | Value::F64(_) => return Err(Error::runtime(
                "SELECT: mask must be bool/int, got floats (float truthiness is silent coercion)",
            )),
            _ => return Err(Error::runtime("SELECT: mask must be bool/int")),
        };
    Ok(vec![Value::Triple(triple_where(&mask, a, b, "SELECT")?)])
}

// `enc` is used by image ops.
pub(crate) fn enc_fill(shape: &[usize], v: f64) -> Triple {
    lattice::encode(&ArrayD::from_elem(ndarray::IxDyn(shape), v))
}
