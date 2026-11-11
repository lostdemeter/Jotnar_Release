use ndarray::{ArrayD, IxDyn};

use crate::error::{Error, Result};

/// Lattice triples: sign byte, exponent (int32), zero flag (uint8).
///
/// Triples are scale-free. All three planes have the same shape.
///
/// `m` is the opt-in T+mantissa ("phi-float") residual plane. `None` is the
/// frozen T format; `Some` turns the value into a T2 value:
/// `sign * PHI^((e - BIAS)/K) * (1 + m / 2^M)`. See [`crate::t2`].
#[derive(Clone, Debug)]
pub struct Triple {
    pub s: ArrayD<i8>,
    pub e: ArrayD<i32>,
    pub z: ArrayD<u8>,
    pub m: Option<ArrayD<i32>>,
}

impl Triple {
    pub fn new(s: ArrayD<i8>, e: ArrayD<i32>, z: ArrayD<u8>) -> Result<Self> {
        Self::new_opt(s, e, z, None)
    }

    pub fn new_opt(
        s: ArrayD<i8>,
        e: ArrayD<i32>,
        z: ArrayD<u8>,
        m: Option<ArrayD<i32>>,
    ) -> Result<Self> {
        if s.shape() != e.shape() || s.shape() != z.shape() {
            return Err(Error::runtime(format!(
                "triple planes have different shapes: s={:?} e={:?} z={:?}",
                s.shape(),
                e.shape(),
                z.shape()
            )));
        }
        if let Some(m) = &m {
            if m.shape() != s.shape() {
                return Err(Error::runtime(format!(
                    "triple mantissa shape {:?} != value shape {:?}",
                    m.shape(),
                    s.shape()
                )));
            }
        }
        Ok(Triple { s, e, z, m })
    }

    /// True when this triple carries a T+mantissa residual plane.
    pub fn is_t2(&self) -> bool {
        self.m.is_some()
    }

    pub fn shape(&self) -> Vec<usize> {
        self.s.shape().to_vec()
    }

    pub fn numel(&self) -> usize {
        self.s.len()
    }

    pub fn from_vecs(shape: &[usize], s: Vec<i8>, e: Vec<i32>, z: Vec<u8>) -> Result<Self> {
        Self::from_vecs_opt(shape, s, e, z, None)
    }

    pub fn from_vecs_opt(
        shape: &[usize],
        s: Vec<i8>,
        e: Vec<i32>,
        z: Vec<u8>,
        m: Option<Vec<i32>>,
    ) -> Result<Self> {
        let shp = IxDyn(shape);
        let s = ArrayD::from_shape_vec(shp.clone(), s)
            .map_err(|err| Error::runtime(format!("triple s: {err}")))?;
        let e = ArrayD::from_shape_vec(shp.clone(), e)
            .map_err(|err| Error::runtime(format!("triple e: {err}")))?;
        let z = ArrayD::from_shape_vec(shp.clone(), z)
            .map_err(|err| Error::runtime(format!("triple z: {err}")))?;
        let m = match m {
            Some(v) => Some(
                ArrayD::from_shape_vec(shp, v)
                    .map_err(|err| Error::runtime(format!("triple m: {err}")))?,
            ),
            None => None,
        };
        Self::new_opt(s, e, z, m)
    }
}
