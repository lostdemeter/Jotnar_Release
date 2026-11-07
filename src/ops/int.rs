//! Integer stream ops: 64-bit two's-complement wrap, scalar literals
//! broadcast, no implicit shape broadcasting between streams.

use ndarray::{ArrayD, IxDyn, Zip};

use crate::error::{Error, Result};
use crate::ir::Config;
use crate::value::{Arg, Value};

use super::{arg_i64, arg_i64_array, checked_div};

fn int_operand(
    args: &[Arg],
    i: usize,
    op: &str,
    name: &str,
    shape: &[usize],
) -> Result<ArrayD<i64>> {
    let arg = args
        .get(i)
        .ok_or_else(|| Error::runtime(format!("{op}: missing '{name}'")))?;
    match arg {
        Arg::Literal(v) => {
            let iv = crate::value::int_literal(*v, op, name)?;
            Ok(ArrayD::from_elem(ndarray::IxDyn(shape), iv))
        }
        Arg::Stream(v) => {
            let a = v.as_i64(&format!("{op}:{name}"))?;
            if a.ndim() == 0 {
                let scalar = *a.iter().next().unwrap_or(&0);
                return Ok(ArrayD::from_elem(ndarray::IxDyn(shape), scalar));
            }
            if a.shape() != shape {
                return Err(Error::runtime(format!(
                    "{op}: '{name}' shape {:?} != {:?} (no broadcast)",
                    a.shape(),
                    shape
                )));
            }
            Ok(a)
        }
    }
}

fn int_stream(args: &[Arg], i: usize, op: &str, name: &str) -> Result<ArrayD<i64>> {
    arg_i64_array(args, i, op)
        .map_err(|_| Error::runtime(format!("{op}: stream '{name}' must be integer")))
}

fn wrap(v: i128) -> i64 {
    v as i64
}

/// Extract one element of an integer stream as a scalar.
pub fn op_iget(args: &[Arg], _: &Config) -> Result<Vec<Value>> {
    let x = int_stream(args, 0, "IGET", "X")?;
    let idx = arg_i64(args, 1, "IGET", "idx")?;
    if idx < 0 || idx as usize >= x.len() {
        return Err(Error::runtime(format!(
            "IGET: index {idx} out of range {}",
            x.len()
        )));
    }
    let v = *x.iter().nth(idx as usize).unwrap_or(&0);
    Ok(vec![Value::I64(ArrayD::from_elem(ndarray::IxDyn(&[]), v))])
}

pub fn op_iadd(args: &[Arg], _: &Config) -> Result<Vec<Value>> {
    let a = int_stream(args, 0, "IADD", "a")?;
    let b = int_operand(args, 1, "IADD", "b", a.shape())?;
    Ok(vec![Value::I64(
        Zip::from(&a)
            .and(&b)
            .map_collect(|&x, &y| x.wrapping_add(y)),
    )])
}

pub fn op_isub(args: &[Arg], _: &Config) -> Result<Vec<Value>> {
    let a = int_stream(args, 0, "ISUB", "a")?;
    let b = int_operand(args, 1, "ISUB", "b", a.shape())?;
    Ok(vec![Value::I64(
        Zip::from(&a)
            .and(&b)
            .map_collect(|&x, &y| x.wrapping_sub(y)),
    )])
}

pub fn op_imul(args: &[Arg], _: &Config) -> Result<Vec<Value>> {
    let a = int_stream(args, 0, "IMUL", "a")?;
    let b = int_operand(args, 1, "IMUL", "b", a.shape())?;
    Ok(vec![Value::I64(
        Zip::from(&a)
            .and(&b)
            .map_collect(|&x, &y| x.wrapping_mul(y)),
    )])
}

pub fn op_imad(args: &[Arg], _: &Config) -> Result<Vec<Value>> {
    let a = int_stream(args, 0, "IMAD", "a")?;
    let b = int_operand(args, 1, "IMAD", "b", a.shape())?;
    let c = int_operand(args, 2, "IMAD", "c", a.shape())?;
    let out = Zip::from(&a)
        .and(&b)
        .and(&c)
        .map_collect(|&x, &y, &z| x.wrapping_mul(y).wrapping_add(z));
    Ok(vec![Value::I64(out)])
}

/// Exact 2D integer matrix multiply: `[M,K] * [K,N] -> [M,N]`.
///
/// This is the geometric contraction primitive for the integer path. It is
/// deliberately not a triple/ladder op: the integer corpus is exact, and the
/// φ-ladder is a separate contract. Arithmetic wraps at 64 bits at every
/// multiply/add, matching the scalar `IMUL`/`IADD` graph it replaces.
pub fn op_imatmul(args: &[Arg], _: &Config) -> Result<Vec<Value>> {
    let a = int_stream(args, 0, "IMATMUL", "a")?;
    let b = int_stream(args, 1, "IMATMUL", "b")?;
    if a.ndim() != 2 || b.ndim() != 2 {
        return Err(Error::runtime(format!(
            "IMATMUL: need 2D I operands, got {:?} x {:?}",
            a.shape(),
            b.shape()
        )));
    }
    let (m, k) = (a.shape()[0], a.shape()[1]);
    let (k2, n) = (b.shape()[0], b.shape()[1]);
    if k != k2 {
        return Err(Error::runtime(format!(
            "IMATMUL: inner dims {k} vs {k2} (left [M,K], right [K,N])"
        )));
    }
    let mut out = ArrayD::from_elem(IxDyn(&[m, n]), 0i64);
    for i in 0..m {
        for j in 0..n {
            let mut acc: i64 = 0;
            for t in 0..k {
                acc = acc.wrapping_add(a[[i, t]].wrapping_mul(b[[t, j]]));
            }
            out[[i, j]] = acc;
        }
    }
    Ok(vec![Value::I64(out)])
}

/// Exact sum of every element of an integer stream: `I:* -> I:SCALAR`.
///
/// One geometric reduction, not one opcode per arity. The shape contract is
/// the input shape; the output is the scalar stream (empty shape). Arithmetic
/// wraps at 64 bits, matching an `IADD` chain.
pub fn op_ireduce_sum(args: &[Arg], _: &Config) -> Result<Vec<Value>> {
    let x = int_stream(args, 0, "IREDUCE_SUM", "x")?;
    let mut acc: i64 = 0;
    for &v in x.iter() {
        acc = acc.wrapping_add(v);
    }
    Ok(vec![Value::I64(ArrayD::from_elem(IxDyn(&[]), acc))])
}

pub fn op_ixor(args: &[Arg], _: &Config) -> Result<Vec<Value>> {
    let a = int_stream(args, 0, "IXOR", "a")?;
    let b = int_operand(args, 1, "IXOR", "b", a.shape())?;
    Ok(vec![Value::I64(
        Zip::from(&a).and(&b).map_collect(|&x, &y| x ^ y),
    )])
}

pub fn op_iand(args: &[Arg], _: &Config) -> Result<Vec<Value>> {
    let a = int_stream(args, 0, "IAND", "a")?;
    let b = int_operand(args, 1, "IAND", "b", a.shape())?;
    Ok(vec![Value::I64(
        Zip::from(&a).and(&b).map_collect(|&x, &y| x & y),
    )])
}

pub fn op_ior(args: &[Arg], _: &Config) -> Result<Vec<Value>> {
    let a = int_stream(args, 0, "IOR", "a")?;
    let b = int_operand(args, 1, "IOR", "b", a.shape())?;
    Ok(vec![Value::I64(
        Zip::from(&a).and(&b).map_collect(|&x, &y| x | y),
    )])
}

pub fn op_ishl(args: &[Arg], _: &Config) -> Result<Vec<Value>> {
    let a = int_stream(args, 0, "ISHL", "a")?;
    let s = arg_i64(args, 1, "ISHL", "shift")?;
    if !(0..=63).contains(&s) {
        return Err(Error::runtime(format!(
            "ISHL: shift must be 0..63, got {s}"
        )));
    }
    Ok(vec![Value::I64(
        a.mapv(|v| ((v as u64) << (s as u32)) as i64),
    )])
}

pub fn op_ishr(args: &[Arg], _: &Config) -> Result<Vec<Value>> {
    let a = int_stream(args, 0, "ISHR", "a")?;
    let s = arg_i64(args, 1, "ISHR", "shift")?;
    if !(0..=63).contains(&s) {
        return Err(Error::runtime(format!(
            "ISHR: shift must be 0..63, got {s}"
        )));
    }
    Ok(vec![Value::I64(
        a.mapv(|v| ((v as u64) >> (s as u32)) as i64),
    )])
}

fn divmod(args: &[Arg], op: &str) -> Result<ArrayD<i64>> {
    let a = int_stream(args, 0, op, "a")?;
    let b = int_operand(args, 1, op, "b", a.shape())?;
    let mut out = Vec::with_capacity(a.len());
    for (&x, &y) in a.iter().zip(b.iter()) {
        let q = checked_div(x, y)?;
        out.push(if op == "IDIV" {
            q
        } else {
            x.wrapping_sub(q.wrapping_mul(y))
        });
    }
    crate::value::array_from_vec(a.shape(), out)
}

pub fn op_idiv(args: &[Arg], _: &Config) -> Result<Vec<Value>> {
    Ok(vec![Value::I64(divmod(args, "IDIV")?)])
}

pub fn op_imod(args: &[Arg], _: &Config) -> Result<Vec<Value>> {
    Ok(vec![Value::I64(divmod(args, "IMOD")?)])
}

pub fn op_ieq(args: &[Arg], _: &Config) -> Result<Vec<Value>> {
    let a = int_stream(args, 0, "IEQ", "a")?;
    let b = int_operand(args, 1, "IEQ", "b", a.shape())?;
    Ok(vec![Value::I64(
        Zip::from(&a).and(&b).map_collect(|&x, &y| (x == y) as i64),
    )])
}

pub fn op_ilt(args: &[Arg], _: &Config) -> Result<Vec<Value>> {
    let a = int_stream(args, 0, "ILT", "a")?;
    let b = int_operand(args, 1, "ILT", "b", a.shape())?;
    Ok(vec![Value::I64(
        Zip::from(&a).and(&b).map_collect(|&x, &y| (x < y) as i64),
    )])
}

pub fn op_sadd(args: &[Arg], _: &Config) -> Result<Vec<Value>> {
    let a = int_stream(args, 0, "SADD", "a")?;
    let b = int_operand(args, 1, "SADD", "b", a.shape())?;
    Ok(vec![Value::I64(
        Zip::from(&a)
            .and(&b)
            .map_collect(|&x, &y| x.saturating_add(y)),
    )])
}

pub fn op_imax(args: &[Arg], _: &Config) -> Result<Vec<Value>> {
    let a = int_stream(args, 0, "IMAX", "a")?;
    let b = int_operand(args, 1, "IMAX", "b", a.shape())?;
    Ok(vec![Value::I64(
        Zip::from(&a).and(&b).map_collect(|&x, &y| x.max(y)),
    )])
}

pub fn op_iselect(args: &[Arg], _: &Config) -> Result<Vec<Value>> {
    let c = int_stream(args, 0, "ISELECT", "c")?;
    let a = int_operand(args, 1, "ISELECT", "a", c.shape())?;
    let b = int_operand(args, 2, "ISELECT", "b", c.shape())?;
    Ok(vec![Value::I64(
        Zip::from(&c)
            .and(&a)
            .and(&b)
            .map_collect(|&cc, &x, &y| if cc != 0 { x } else { y }),
    )])
}

pub fn op_isar(args: &[Arg], _: &Config) -> Result<Vec<Value>> {
    let a = int_stream(args, 0, "ISAR", "a")?;
    let b = int_operand(args, 1, "ISAR", "b", a.shape())?;
    let mut out = Vec::with_capacity(a.len());
    for (&x, &s) in a.iter().zip(b.iter()) {
        if !(0..=63).contains(&s) {
            return Err(Error::runtime(format!(
                "ISAR: shift must be 0..63, got {s}"
            )));
        }
        out.push(x >> (s as u32));
    }
    Ok(vec![Value::I64(crate::value::array_from_vec(
        a.shape(),
        out,
    )?)])
}

fn unsigned_divmod(args: &[Arg], op: &str) -> Result<ArrayD<i64>> {
    let a = int_stream(args, 0, op, "a")?;
    let b = int_operand(args, 1, op, "b", a.shape())?;
    let mut out = Vec::with_capacity(a.len());
    for (&x, &y) in a.iter().zip(b.iter()) {
        let au = x as u64;
        let bu = y as u64;
        if bu == 0 {
            return Err(Error::runtime(format!("{op}: division by zero")));
        }
        let q = au / bu;
        out.push(if op == "IDIVU" {
            q as i64
        } else {
            (au - q * bu) as i64
        });
    }
    crate::value::array_from_vec(a.shape(), out)
}

pub fn op_idivu(args: &[Arg], _: &Config) -> Result<Vec<Value>> {
    Ok(vec![Value::I64(unsigned_divmod(args, "IDIVU")?)])
}

pub fn op_imodu(args: &[Arg], _: &Config) -> Result<Vec<Value>> {
    Ok(vec![Value::I64(unsigned_divmod(args, "IMODU")?)])
}

pub fn op_f2i(args: &[Arg], _: &Config) -> Result<Vec<Value>> {
    let v = args
        .first()
        .ok_or_else(|| Error::runtime("F2I: missing input"))?
        .stream("F2I", "x")?;
    if matches!(v, Value::Triple(_)) {
        return Err(Error::runtime(
            "F2I: triples unbuilt (decode/quantize first)",
        ));
    }
    if !matches!(v, Value::F32(_) | Value::F64(_)) {
        return Err(Error::runtime(format!(
            "F2I: stream must be float, got {}",
            v.kind()
        )));
    }
    let a = v.as_f64("F2I")?;
    let mut out = Vec::with_capacity(a.len());
    for &x in a.iter() {
        if !x.is_finite() {
            return Err(Error::runtime("F2I: NaN/Inf input"));
        }
        let t = x.trunc();
        if t < -(2f64.powi(63)) || t >= 2f64.powi(63) {
            return Err(Error::runtime("F2I: value outside int64 range"));
        }
        out.push(t as i64);
    }
    Ok(vec![Value::I64(crate::value::array_from_vec(
        a.shape(),
        out,
    )?)])
}

pub fn op_i2f(args: &[Arg], _: &Config) -> Result<Vec<Value>> {
    let a = int_stream(args, 0, "I2F", "x")?;
    Ok(vec![Value::F64(a.mapv(|v| v as f64))])
}

// Keep the wrapping helper referenced (used by future widenings).
#[allow(dead_code)]
fn _wrap(v: i128) -> i64 {
    wrap(v)
}
