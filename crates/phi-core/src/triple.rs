use ndarray::{ArrayD, IxDyn};

use crate::error::{Error, Result};

/// Lattice triples: sign byte, exponent (int32), zero flag (uint8).
///
/// Triples are scale-free. All three planes have the same shape.
#[derive(Clone, Debug)]
pub struct Triple {
    pub s: ArrayD<i8>,
    pub e: ArrayD<i32>,
    pub z: ArrayD<u8>,
}

impl Triple {
    pub fn new(s: ArrayD<i8>, e: ArrayD<i32>, z: ArrayD<u8>) -> Result<Self> {
        if s.shape() != e.shape() || s.shape() != z.shape() {
            return Err(Error::runtime(format!(
                "triple planes have different shapes: s={:?} e={:?} z={:?}",
                s.shape(),
                e.shape(),
                z.shape()
            )));
        }
        Ok(Triple { s, e, z })
    }

    pub fn shape(&self) -> Vec<usize> {
        self.s.shape().to_vec()
    }

    pub fn numel(&self) -> usize {
        self.s.len()
    }

    pub fn from_vecs(shape: &[usize], s: Vec<i8>, e: Vec<i32>, z: Vec<u8>) -> Result<Self> {
        let shp = IxDyn(shape);
        Ok(Triple {
            s: ArrayD::from_shape_vec(shp.clone(), s)
                .map_err(|err| Error::runtime(format!("triple s: {err}")))?,
            e: ArrayD::from_shape_vec(shp.clone(), e)
                .map_err(|err| Error::runtime(format!("triple e: {err}")))?,
            z: ArrayD::from_shape_vec(shp, z)
                .map_err(|err| Error::runtime(format!("triple z: {err}")))?,
        })
    }
}
