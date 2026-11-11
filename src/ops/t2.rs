//! Interpreter dispatch for the opt-in T+mantissa (T2) value format.
//!
//! A stream is T2 when its [`Triple`] carries a mantissa plane. The frozen T
//! ops never see such a stream: this module intercepts the supported set and
//! runs the integer reference semantics in [`phi_core::t2`]. Unsupported ops
//! fail loud rather than silently dropping the residual.

use crate::error::{Error, Result};
use crate::ir::Config;
use crate::value::{Arg, Triple, Value};

use super::arg_i64;

fn m_bits(config: &Config) -> Option<i64> {
    config.get_i64("t2")
}

fn eps(config: &Config) -> f64 {
    config.get_f64_or("eps_ln", 1e-6)
}

fn triple<'a>(args: &'a [Arg], i: usize, op: &str) -> Result<&'a Triple> {
    match args.get(i).and_then(|a| a.value()) {
        Some(Value::Triple(t)) => Ok(t),
        _ => Err(Error::runtime(format!(
            "{op}: argument {i} must be a T2 stream"
        ))),
    }
}

/// Route an instruction when any argument is T2. `None` means "not T2".
pub fn dispatch(name: &str, args: &[Arg], config: &Config) -> Option<Result<Vec<Value>>> {
    let any_t2 = args
        .iter()
        .any(|a| matches!(a.value(), Some(Value::Triple(t)) if t.is_t2()));
    if !any_t2 {
        return None;
    }
    Some(run(name, args, config))
}

fn run(name: &str, args: &[Arg], config: &Config) -> Result<Vec<Value>> {
    let (m, f) = phi_core::t2::check_widths(m_bits(config), config.get_i64("bridge_f"))?;
    let m = match m {
        Some(m) => m,
        None => {
            return Err(Error::runtime(
                "T2 stream without CONFIG t2 <mantissa bits>",
            ))
        }
    };
    let tabs = phi_core::t2::Tables::new(f);
    let one = |v: Triple| Ok(vec![Value::Triple(v)]);
    match name {
        "MUL" => one(phi_core::t2::mul_array(
            triple(args, 0, name)?,
            triple(args, 1, name)?,
            m,
        )?),
        "SQUARE" => {
            let x = triple(args, 0, name)?;
            one(phi_core::t2::mul_array(x, x, m)?)
        }
        "ADD" | "SUB" => one(phi_core::t2::add_array(
            triple(args, 0, name)?,
            triple(args, 1, name)?,
            name == "SUB",
            &tabs,
            m,
        )?),
        "MATMUL" => one(phi_core::t2::matmul2(
            triple(args, 0, name)?,
            triple(args, 1, name)?,
            &tabs,
            m,
        )?),
        "LAYERNORM" => one(phi_core::t2::layernorm(
            triple(args, 0, name)?,
            triple(args, 1, name)?,
            triple(args, 2, name)?,
            &tabs,
            m,
            eps(config),
        )?),
        "GELU" => one(phi_core::t2::gelu(triple(args, 0, name)?, &tabs, m)?),
        "SOFTMAX" | "SOFTMAX_WIDE" => one(phi_core::t2::softmax(triple(args, 0, name)?, &tabs, m)?),
        "RELU" => one(phi_core::t2::relu_array(triple(args, 0, name)?)?),
        "CONV2D" => {
            let stride = arg_i64(args, 3, name, "stride")?;
            let pad = arg_i64(args, 4, name, "pad")?;
            one(phi_core::t2::conv2d(
                triple(args, 0, name)?,
                triple(args, 1, name)?,
                triple(args, 2, name)?,
                stride,
                pad,
                &tabs,
                m,
            )?)
        }
        "DECONV2D" => {
            let stride = arg_i64(args, 3, name, "stride")?;
            let pad = arg_i64(args, 4, name, "pad")?;
            one(phi_core::t2::deconv2d(
                triple(args, 0, name)?,
                triple(args, 1, name)?,
                triple(args, 2, name)?,
                stride,
                pad,
                &tabs,
                m,
            )?)
        }
        "RESIZE2D" => {
            let ho = arg_i64(args, 1, name, "ho")? as usize;
            let wo = arg_i64(args, 2, name, "wo")? as usize;
            let align = arg_i64(args, 3, name, "align")? != 0;
            one(phi_core::t2::resize2d(
                triple(args, 0, name)?,
                ho,
                wo,
                align,
                &tabs,
                m,
            )?)
        }
        "SLICE" => {
            let axis = arg_i64(args, 1, name, "axis")? as usize;
            let lo = arg_i64(args, 2, name, "lo")? as usize;
            let hi = arg_i64(args, 3, name, "hi")? as usize;
            one(phi_core::t2::slice2d(triple(args, 0, name)?, axis, lo, hi)?)
        }
        "RESHAPE2" => {
            let d0 = arg_i64(args, 1, name, "d0")? as usize;
            let d1 = arg_i64(args, 2, name, "d1")? as usize;
            one(phi_core::t2::reshape2(triple(args, 0, name)?, d0, d1)?)
        }
        "RESHAPE3" => {
            let d0 = arg_i64(args, 1, name, "d0")? as usize;
            let d1 = arg_i64(args, 2, name, "d1")? as usize;
            let d2 = arg_i64(args, 3, name, "d2")? as usize;
            one(phi_core::t2::reshape3(triple(args, 0, name)?, d0, d1, d2)?)
        }
        "TRANSPOSE" => one(transpose(triple(args, 0, name)?)?),
        "CONCAT" => one(concat(triple(args, 0, name)?, triple(args, 1, name)?)?),
        other => Err(Error::unimplemented(format!(
            "T2: no non-FPU reference for {other} (supported: MUL, SQUARE, ADD, SUB, \
             MATMUL, LAYERNORM, GELU, SOFTMAX, SOFTMAX_WIDE, TRANSPOSE, CONCAT)"
        ))),
    }
}

fn transpose(t: &Triple) -> Result<Triple> {
    let sh = t.shape();
    if sh.len() != 2 {
        return Err(Error::runtime("T2 TRANSPOSE needs 2-D"));
    }
    let (r, c) = (sh[0], sh[1]);
    let m = t.m.as_ref().expect("t2");
    let (ts, te, tm, tz) = (
        t.s.as_slice().unwrap(),
        t.e.as_slice().unwrap(),
        m.as_slice().unwrap(),
        t.z.as_slice().unwrap(),
    );
    let mut s = vec![1i8; r * c];
    let mut e = vec![0i32; r * c];
    let mut mv = vec![0i32; r * c];
    let mut z = vec![0u8; r * c];
    for i in 0..r {
        for j in 0..c {
            let src = i * c + j;
            let dst = j * r + i;
            s[dst] = ts[src];
            e[dst] = te[src];
            mv[dst] = tm[src];
            z[dst] = tz[src];
        }
    }
    Triple::from_vecs_opt(&[c, r], s, e, z, Some(mv))
}

fn concat(a: &Triple, b: &Triple) -> Result<Triple> {
    let ash = a.shape();
    let bsh = b.shape();
    if ash.len() != 2 || bsh.len() != 2 || ash[0] != bsh[0] {
        return Err(Error::runtime("T2 CONCAT needs 2-D same-row inputs"));
    }
    let (rows, da, db) = (ash[0], ash[1], bsh[1]);
    let am = a.m.as_ref().expect("t2");
    let bm = b.m.as_ref().expect("t2");
    let mut s = vec![1i8; rows * (da + db)];
    let mut e = vec![0i32; rows * (da + db)];
    let mut m = vec![0i32; rows * (da + db)];
    let mut z = vec![0u8; rows * (da + db)];
    for i in 0..rows {
        for j in 0..da {
            let src = i * da + j;
            let dst = i * (da + db) + j;
            s[dst] = a.s.as_slice().unwrap()[src];
            e[dst] = a.e.as_slice().unwrap()[src];
            m[dst] = am.as_slice().unwrap()[src];
            z[dst] = a.z.as_slice().unwrap()[src];
        }
        for j in 0..db {
            let src = i * db + j;
            let dst = i * (da + db) + da + j;
            s[dst] = b.s.as_slice().unwrap()[src];
            e[dst] = b.e.as_slice().unwrap()[src];
            m[dst] = bm.as_slice().unwrap()[src];
            z[dst] = b.z.as_slice().unwrap()[src];
        }
    }
    Triple::from_vecs_opt(&[rows, da + db], s, e, z, Some(m))
}
