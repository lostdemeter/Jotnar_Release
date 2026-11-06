use std::fmt;

use ndarray::{ArrayD, IxDyn};

use crate::error::{Error, Result};

/// Stream kinds, matching the layout vocabulary (`F`, `T`, `I`, `U8`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    F,
    T,
    I,
    U8,
    Unknown,
}

impl Kind {
    pub fn as_str(self) -> &'static str {
        match self {
            Kind::F => "F",
            Kind::T => "T",
            Kind::I => "I",
            Kind::U8 => "U8",
            Kind::Unknown => "UNKNOWN",
        }
    }
}

impl fmt::Display for Kind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

pub use phi_core::Triple;

/// A runtime stream value.
///
/// `F32`/`F64` preserve numpy's distinction between float32 and float64;
/// `I64` is the integer stream type (numpy bools are normalised to 0/1
/// only where Python explicitly converts them); `U8` is the byte stream.
#[derive(Clone, Debug)]
pub enum Value {
    F32(ArrayD<f32>),
    F64(ArrayD<f64>),
    I64(ArrayD<i64>),
    U8(ArrayD<u8>),
    Bool(ArrayD<bool>),
    Triple(Triple),
    None,
}

impl Value {
    pub fn kind(&self) -> Kind {
        match self {
            Value::F32(_) | Value::F64(_) => Kind::F,
            Value::I64(_) | Value::Bool(_) => Kind::I,
            Value::U8(_) => Kind::U8,
            Value::Triple(_) => Kind::T,
            Value::None => Kind::Unknown,
        }
    }

    pub fn shape(&self) -> Option<Vec<usize>> {
        match self {
            Value::F32(a) => Some(a.shape().to_vec()),
            Value::F64(a) => Some(a.shape().to_vec()),
            Value::I64(a) => Some(a.shape().to_vec()),
            Value::U8(a) => Some(a.shape().to_vec()),
            Value::Bool(a) => Some(a.shape().to_vec()),
            Value::Triple(t) => Some(t.shape()),
            Value::None => None,
        }
    }

    pub fn ndim(&self) -> Option<usize> {
        self.shape().map(|s| s.len())
    }

    pub fn numel(&self) -> Option<usize> {
        self.shape().map(|s| s.iter().product())
    }

    pub fn is_none(&self) -> bool {
        matches!(self, Value::None)
    }

    pub fn as_f64(&self, what: &str) -> Result<ArrayD<f64>> {
        match self {
            Value::F64(a) => Ok(a.clone()),
            Value::F32(a) => Ok(a.mapv(|v| v as f64)),
            Value::I64(a) => Ok(a.mapv(|v| v as f64)),
            Value::U8(a) => Ok(a.mapv(|v| v as f64)),
            Value::Bool(a) => Ok(a.mapv(|v| if v { 1.0 } else { 0.0 })),
            Value::Triple(_) => Err(Error::runtime(format!(
                "{what}: expected a float stream, got triples"
            ))),
            Value::None => Err(Error::runtime(format!("{what}: stream is None"))),
        }
    }

    pub fn as_f32(&self, what: &str) -> Result<ArrayD<f32>> {
        match self {
            Value::F32(a) => Ok(a.clone()),
            Value::F64(a) => Ok(a.mapv(|v| v as f32)),
            Value::I64(a) => Ok(a.mapv(|v| v as f32)),
            Value::U8(a) => Ok(a.mapv(|v| v as f32)),
            Value::Bool(a) => Ok(a.mapv(|v| if v { 1.0 } else { 0.0 })),
            Value::Triple(_) => Err(Error::runtime(format!(
                "{what}: expected a float stream, got triples"
            ))),
            Value::None => Err(Error::runtime(format!("{what}: stream is None"))),
        }
    }

    pub fn as_i64(&self, what: &str) -> Result<ArrayD<i64>> {
        match self {
            Value::I64(a) => Ok(a.clone()),
            Value::U8(a) => Ok(a.mapv(|v| v as i64)),
            Value::Bool(a) => Ok(a.mapv(|v| if v { 1 } else { 0 })),
            Value::F64(_) | Value::F32(_) => Err(Error::runtime(format!(
                "{what}: expected an integer stream, got floats"
            ))),
            Value::Triple(_) => Err(Error::runtime(format!(
                "{what}: expected an integer stream, got triples"
            ))),
            Value::None => Err(Error::runtime(format!("{what}: stream is None"))),
        }
    }

    pub fn as_u8(&self, what: &str) -> Result<ArrayD<u8>> {
        match self {
            Value::U8(a) => Ok(a.clone()),
            Value::I64(a) => Ok(a.mapv(|v| v as u8)),
            Value::Bool(a) => Ok(a.mapv(|v| if v { 1 } else { 0 })),
            Value::F32(_) | Value::F64(_) => Err(Error::runtime(format!(
                "{what}: expected a byte stream, got floats"
            ))),
            Value::Triple(_) => Err(Error::runtime(format!(
                "{what}: expected a byte stream, got triples"
            ))),
            Value::None => Err(Error::runtime(format!("{what}: stream is None"))),
        }
    }

    pub fn as_bool(&self, what: &str) -> Result<ArrayD<bool>> {
        match self {
            Value::Bool(a) => Ok(a.clone()),
            Value::I64(a) => Ok(a.mapv(|v| v != 0)),
            Value::U8(a) => Ok(a.mapv(|v| v != 0)),
            Value::F32(_) | Value::F64(_) => Err(Error::runtime(format!(
                "{what}: expected a bool/int mask, got floats"
            ))),
            Value::Triple(_) => Err(Error::runtime(format!(
                "{what}: expected a bool/int mask, got triples"
            ))),
            Value::None => Err(Error::runtime(format!("{what}: stream is None"))),
        }
    }

    pub fn triple(&self, what: &str) -> Result<&Triple> {
        match self {
            Value::Triple(t) => Ok(t),
            other => Err(Error::runtime(format!(
                "{what}: expected triples, got {}",
                other.kind()
            ))),
        }
    }

    pub fn triple_owned(self, what: &str) -> Result<Triple> {
        match self {
            Value::Triple(t) => Ok(t),
            other => Err(Error::runtime(format!(
                "{what}: expected triples, got {}",
                other.kind()
            ))),
        }
    }

    /// First element as an integer (used by REPEAT/BREAK and mask reads).
    /// Follows Python's `int(view[name].reshape(-1)[0])`.
    pub fn first_i64(&self, what: &str) -> Result<i64> {
        match self {
            Value::I64(a) => a
                .iter()
                .next()
                .copied()
                .ok_or_else(|| Error::runtime(format!("{what}: empty stream"))),
            Value::U8(a) => a
                .iter()
                .next()
                .map(|v| *v as i64)
                .ok_or_else(|| Error::runtime(format!("{what}: empty stream"))),
            Value::Bool(a) => a
                .iter()
                .next()
                .map(|v| if *v { 1 } else { 0 })
                .ok_or_else(|| Error::runtime(format!("{what}: empty stream"))),
            Value::F32(a) => a
                .iter()
                .next()
                .map(|v| *v as i64)
                .ok_or_else(|| Error::runtime(format!("{what}: empty stream"))),
            Value::F64(a) => a
                .iter()
                .next()
                .map(|v| *v as i64)
                .ok_or_else(|| Error::runtime(format!("{what}: empty stream"))),
            _ => Err(Error::runtime(format!("{what}: stream is not numeric"))),
        }
    }

    pub fn from_f64(v: ArrayD<f64>) -> Self {
        Value::F64(v)
    }

    pub fn from_f32(v: ArrayD<f32>) -> Self {
        Value::F32(v)
    }

    pub fn from_i64(v: ArrayD<i64>) -> Self {
        Value::I64(v)
    }

    pub fn from_u8(v: ArrayD<u8>) -> Self {
        Value::U8(v)
    }

    pub fn from_bool(v: ArrayD<bool>) -> Self {
        Value::Bool(v)
    }
}

/// A resolved instruction argument: a stream value or a numeric literal.
///
/// Python passes bare numbers as Python floats; every op decides how to
/// interpret them (`float(a)`, `int(a)`, …). Keeping literals distinct from
/// streams preserves that contract (a literal is `F:SCALAR` for layouts).
#[derive(Clone, Debug)]
pub enum Arg {
    Stream(Value),
    Literal(f64),
}

impl Arg {
    pub fn is_literal(&self) -> bool {
        matches!(self, Arg::Literal(_))
    }

    pub fn value(&self) -> Option<&Value> {
        match self {
            Arg::Stream(v) => Some(v),
            Arg::Literal(_) => None,
        }
    }

    pub fn stream(&self, op: &str, name: &str) -> Result<&Value> {
        match self {
            Arg::Stream(v) => Ok(v),
            Arg::Literal(l) => Err(Error::runtime(format!(
                "{op}: '{name}' must be a stream, got literal {l}"
            ))),
        }
    }

    pub fn f64(&self, op: &str, name: &str) -> Result<f64> {
        match self {
            Arg::Literal(v) => Ok(*v),
            Arg::Stream(v) => {
                let a = v.as_f64(&format!("{op}:{name}"))?;
                let first = a
                    .iter()
                    .next()
                    .copied()
                    .ok_or_else(|| Error::runtime(format!("{op}: '{name}' is empty")))?;
                Ok(first)
            }
        }
    }

    pub fn i64(&self, op: &str, name: &str) -> Result<i64> {
        match self {
            Arg::Literal(v) => int_literal(*v, op, name),
            Arg::Stream(v) => {
                let a = v.as_f64(&format!("{op}:{name}"))?;
                let first = a
                    .iter()
                    .next()
                    .copied()
                    .ok_or_else(|| Error::runtime(format!("{op}: '{name}' is empty")))?;
                int_literal(first, op, name)
            }
        }
    }
}

/// Python's `_int_scalar`: accept `256.0`, refuse `256.5`; int64 range.
pub fn int_literal(v: f64, op: &str, name: &str) -> Result<i64> {
    if !v.is_finite() || v.fract() != 0.0 {
        return Err(Error::runtime(format!(
            "{op}: '{name}' must be an integer literal, got {v}"
        )));
    }
    // f64 exactly represents integers in this range.
    if v < -(2f64.powi(63)) || v >= 2f64.powi(63) {
        return Err(Error::runtime(format!("{op}: '{name}' outside int64: {v}")));
    }
    Ok(v as i64)
}

/// Build an `ArrayD` from a shape and flat C-order data.
pub fn array_from_vec<T: Clone>(shape: &[usize], data: Vec<T>) -> Result<ArrayD<T>> {
    let expected: usize = shape.iter().product();
    if expected != data.len() {
        return Err(Error::runtime(format!(
            "shape {:?} needs {expected} elements, got {}",
            shape,
            data.len()
        )));
    }
    ArrayD::from_shape_vec(IxDyn(shape), data)
        .map_err(|e| Error::runtime(format!("array shape {:?}: {e}", shape)))
}

/// Scalar helpers for shape arithmetic.
pub fn shape_size(shape: &[usize]) -> usize {
    shape.iter().product()
}
