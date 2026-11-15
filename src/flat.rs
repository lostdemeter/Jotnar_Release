//! Flat payloads: one codec, one size table, one admission pass.
//!
//! This module is the runtime-internal discipline: a stream that crosses a
//! boundary is a [`FlatPayload`] — a `StreamSpec` plus one contiguous byte
//! image — and it becomes a [`Value`] only after [`FlatPayload::admit`] has
//! checked it.
//!
//! There are exactly two byte conventions, and they are named:
//!
//! * [`Convention::Wire`] — native width, explicit `dtype`. What the node
//!   protocol carries (`protocol::encode_stream`).
//! * [`Convention::Target`] — everything widened to little-endian `f64`/`i64`.
//!   What generated C/Rust targets read and write (`codegen::rust` contract).
//!
//! They differ on purpose: `F32` is 4 bytes on the wire and 8 as a target
//! stream, and `U8`/`Bool` are 1/8 bytes on the wire and 8 as a target
//! stream. Naming the convention makes a mixed pair a typed error instead of
//! a silent misread. The JSON payload in [`crate::payload`] is a third
//! representation, but it is a *rendering* for humans and diffs and never a
//! job boundary.
//!
//! Admission is the certifiable part. [`Admission`] is a witness: it records
//! the layout, element count, byte count, convention, and the canonical
//! FNV-1a hash used by receipts. All arithmetic is checked, so a hostile
//! shape (one whose element count overflows) is rejected rather than
//! panicking in debug or wrapping in release.

use crate::error::{Error, Result};
use crate::protocol::StreamSpec;
use crate::value::{array_from_vec, hash_bytes, Triple, Value};

/// Which byte convention a payload uses.
///
/// The two conventions are not interchangeable, and that is the point: a
/// payload admitted as `Wire` cannot be decoded as `Target`, and vice versa.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Convention {
    /// Native width with an explicit `dtype` (`f32`, `u8`, `i64x3`, `t2`).
    Wire,
    /// Widened little-endian `f64`/`i64`, matching generated-target files.
    Target,
}

impl Convention {
    pub fn as_str(self) -> &'static str {
        match self {
            Convention::Wire => "wire",
            Convention::Target => "target",
        }
    }
}

/// How the bytes of one element are laid out.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Encoding {
    /// Fixed-width scalar elements, little-endian (or raw for `u8`).
    Scalar(usize),
    /// T2: plane-major `s:i8[n] | e:i32[n] | z:u8[n] | m:i32[n]`, 10 bytes/elem.
    T2,
    /// Frozen T: element-interleaved `s/e/z` as `i64`, 24 bytes/elem.
    FrozenT,
}

impl Encoding {
    fn width(self) -> usize {
        match self {
            Encoding::Scalar(w) => w,
            Encoding::T2 => 10,
            Encoding::FrozenT => 24,
        }
    }
}

/// The single size table. Encode and decode both resolve through this, so a
/// width can only be wrong in one place.
fn encoding(convention: Convention, kind: &str, dtype: &str) -> Option<Encoding> {
    match (convention, kind, dtype) {
        (_, "F", "f64") => Some(Encoding::Scalar(8)),
        (_, "I", "i64") => Some(Encoding::Scalar(8)),
        // Wire-only native widths. A target never writes these.
        (Convention::Wire, "F", "f32") => Some(Encoding::Scalar(4)),
        (Convention::Wire, "I", "u8") => Some(Encoding::Scalar(1)),
        (Convention::Wire, "T", "t2") => Some(Encoding::T2),
        // Frozen-T compatibility: any other T dtype is the 24-byte layout the
        // pre-T2 decoder accepted (including hand-written `dtype: ""`).
        (Convention::Wire, "T", _) => Some(Encoding::FrozenT),
        _ => None,
    }
}

/// A stream that has crossed (or will cross) a boundary: header + one image.
#[derive(Clone, Debug)]
pub struct FlatPayload {
    pub spec: StreamSpec,
    pub bytes: Vec<u8>,
}

/// The witness produced by [`FlatPayload::admit`].
///
/// Drop this straight into a receipt or certificate: it is what the machine
/// checked before the bytes became a value.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Admission {
    pub convention: Convention,
    pub kind: String,
    pub dtype: String,
    pub shape: Vec<usize>,
    /// Element count, computed with checked arithmetic.
    pub elems: usize,
    /// Byte count actually admitted.
    pub bytes: usize,
    /// Canonical FNV-1a over the admitted bytes (same hash receipts use).
    pub hash: u64,
}

impl Admission {
    /// `KIND:GEOM`, the vocabulary `layout.rs` already speaks.
    pub fn layout(&self) -> String {
        format!("{}:{}", self.kind, geom(&self.shape))
    }

    /// The receipt/certificate rendering of this witness.
    pub fn json(&self) -> serde_json::Value {
        serde_json::json!({
            "convention": self.convention.as_str(),
            "layout": self.layout(),
            "dtype": self.dtype,
            "elems": self.elems,
            "bytes": self.bytes,
            "hash": format!("{:016x}", self.hash),
        })
    }
}

fn geom(shape: &[usize]) -> String {
    if shape.is_empty() {
        "SCALAR".to_string()
    } else {
        shape
            .iter()
            .map(|d| d.to_string())
            .collect::<Vec<_>>()
            .join("x")
    }
}

impl FlatPayload {
    pub fn new(spec: StreamSpec, bytes: Vec<u8>) -> Self {
        Self { spec, bytes }
    }

    /// Element count of `shape`, rejecting overflow.
    fn elems(&self) -> Result<usize> {
        self.spec.shape.iter().try_fold(1usize, |acc, &d| {
            acc.checked_mul(d)
                .ok_or_else(|| {
                    Error::runtime(format!(
                        "flat: shape {:?} overflows an element count",
                        self.spec.shape
                    ))
                })
        })
    }

    /// Check the header against the bytes. This is the admission gate: after
    /// it returns `Ok`, every subsequent read is length-checked by
    /// construction.
    pub fn admit(self, convention: Convention) -> Result<(Admitted, Admission)> {
        let enc = encoding(convention, &self.spec.kind, &self.spec.dtype).ok_or_else(|| {
            Error::runtime(format!(
                "flat: unsupported {}/{} as a {} payload",
                self.spec.kind,
                if self.spec.dtype.is_empty() {
                    "<empty>"
                } else {
                    &self.spec.dtype
                },
                convention.as_str()
            ))
        })?;
        let elems = self.elems()?;
        let want = elems.checked_mul(enc.width()).ok_or_else(|| {
            Error::runtime(format!(
                "flat: {} elements of {} bytes overflow a byte count",
                elems,
                enc.width()
            ))
        })?;
        if self.bytes.len() != want {
            return Err(Error::runtime(format!(
                "flat: {}/{} payload is {} bytes, layout {} needs {want}",
                self.spec.kind,
                self.spec.dtype,
                self.bytes.len(),
                geom(&self.spec.shape)
            )));
        }
        let admission = Admission {
            convention,
            kind: self.spec.kind.clone(),
            dtype: self.spec.dtype.clone(),
            shape: self.spec.shape.clone(),
            elems,
            bytes: self.bytes.len(),
            hash: hash_bytes(&self.bytes),
        };
        Ok((
            Admitted {
                spec: self.spec,
                bytes: self.bytes,
                enc,
                admission: admission.clone(),
            },
            admission,
        ))
    }
}

/// A payload whose header and length have been checked.
///
/// Constructing one is only possible through [`FlatPayload::admit`], so a
/// `decode` cannot see a mismatched length.
#[derive(Clone, Debug)]
pub struct Admitted {
    spec: StreamSpec,
    bytes: Vec<u8>,
    enc: Encoding,
    admission: Admission,
}

impl Admitted {
    pub fn admission(&self) -> &Admission {
        &self.admission
    }

    pub fn spec(&self) -> &StreamSpec {
        &self.spec
    }

    pub fn decode(&self) -> Result<Value> {
        let n = self.admission.elems;
        let shape = &self.spec.shape;
        match (self.enc, self.spec.kind.as_str()) {
            (Encoding::Scalar(8), "F") => Ok(Value::F64(array_from_vec(shape, le_f64(&self.bytes))?)),
            (Encoding::Scalar(8), "I") => Ok(Value::I64(array_from_vec(shape, le_i64(&self.bytes))?)),
            (Encoding::Scalar(4), "F") => Ok(Value::F32(array_from_vec(
                shape,
                self.bytes
                    .chunks_exact(4)
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect(),
            )?)),
            (Encoding::Scalar(1), "I") => {
                Ok(Value::U8(array_from_vec(shape, self.bytes.clone())?))
            }
            (Encoding::T2, "T") => {
                let s: Vec<i8> = self.bytes[..n].iter().map(|b| *b as i8).collect();
                let e = le_i32(&self.bytes[n..n * 5]);
                let z = self.bytes[n * 5..n * 6].to_vec();
                let m = le_i32(&self.bytes[n * 6..n * 10]);
                Ok(Value::Triple(Triple::from_vecs_opt(
                    shape, s, e, z, Some(m),
                )?))
            }
            (Encoding::FrozenT, "T") => {
                let mut s = Vec::with_capacity(n);
                let mut e = Vec::with_capacity(n);
                let mut z = Vec::with_capacity(n);
                for w in self.bytes.chunks_exact(24) {
                    s.push(i64::from_le_bytes(w[0..8].try_into().unwrap()) as i8);
                    e.push(i64::from_le_bytes(w[8..16].try_into().unwrap()) as i32);
                    z.push(i64::from_le_bytes(w[16..24].try_into().unwrap()) as u8);
                }
                Ok(Value::Triple(Triple::from_vecs(shape, s, e, z)?))
            }
            _ => Err(Error::runtime(format!(
                "flat: admitted {}/{} has no decoder",
                self.spec.kind, self.spec.dtype
            ))),
        }
    }
}

/// How many bytes a value occupies under a convention.
///
/// Width comes from the single `encoding` table, never from a literal here.
pub fn body_len(v: &Value, convention: Convention) -> Result<usize> {
    plan(v, convention).map(|(_, n)| n)
}

/// The header and byte count for a value, without writing anything.
fn plan(v: &Value, convention: Convention) -> Result<(StreamSpec, usize)> {
    let shape = |a: &[usize]| a.to_vec();
    let (spec, elems) = match (v, convention) {
        (Value::F64(a), _) => (spec_of("F", "f64", shape(a.shape())), a.len()),
        (Value::F32(a), Convention::Wire) => (spec_of("F", "f32", shape(a.shape())), a.len()),
        (Value::F32(a), Convention::Target) => (spec_of("F", "f64", shape(a.shape())), a.len()),
        (Value::I64(a), _) => (spec_of("I", "i64", shape(a.shape())), a.len()),
        (Value::U8(a), Convention::Wire) => (spec_of("I", "u8", shape(a.shape())), a.len()),
        (Value::U8(a), Convention::Target) => (spec_of("I", "i64", shape(a.shape())), a.len()),
        (Value::Bool(a), _) => (spec_of("I", "i64", shape(a.shape())), a.len()),
        (Value::Triple(t), _) => {
            let (dtype, shape) = match &t.m {
                Some(_) => ("t2", t.shape()),
                None => ("i64x3", t.shape()),
            };
            (spec_of("T", dtype, shape), t.s.len())
        }
        (Value::None, _) => return Err(Error::runtime("flat: cannot encode None")),
    };
    let enc = encoding(convention, &spec.kind, &spec.dtype).ok_or_else(|| {
        Error::runtime(format!(
            "flat: {}/{} is not a {} payload",
            spec.kind,
            spec.dtype,
            convention.as_str()
        ))
    })?;
    let want = elems.checked_mul(enc.width()).ok_or_else(|| {
        Error::runtime(format!("flat: {elems} elements overflow a byte count"))
    })?;
    Ok((spec, want))
}

fn spec_of(kind: &str, dtype: &str, shape: Vec<usize>) -> StreamSpec {
    StreamSpec {
        kind: kind.into(),
        dtype: dtype.into(),
        shape,
    }
}

/// Encode a value under a named convention into caller-provided storage.
///
/// `dst.len()` must equal [`body_len`]; the arena sizes the region from the
/// same `plan`, so the two cannot disagree.
pub fn write_into(v: &Value, convention: Convention, dst: &mut [u8]) -> Result<StreamSpec> {
    let (spec, want) = plan(v, convention)?;
    if dst.len() != want {
        return Err(Error::runtime(format!(
            "flat: destination is {} bytes, {}/{} needs {want}",
            dst.len(),
            spec.kind,
            spec.dtype
        )));
    }
    match (v, convention) {
        (Value::F64(a), _) => put_f64(dst, a.iter().copied()),
        (Value::F32(a), Convention::Wire) => put_f32(dst, a.iter().copied()),
        (Value::F32(a), Convention::Target) => put_f64(dst, a.iter().map(|x| *x as f64)),
        (Value::I64(a), _) => put_i64(dst, a.iter().copied()),
        (Value::U8(a), Convention::Wire) => dst.copy_from_slice(a.as_slice().unwrap()),
        (Value::U8(a), Convention::Target) => put_i64(dst, a.iter().map(|x| *x as i64)),
        (Value::Bool(a), _) => put_i64(dst, a.iter().map(|x| *x as i64)),
        (Value::Triple(t), _) => {
            let n = t.s.len();
            if t.m.is_some() {
                // Plane-major, 10 bytes/element: s:i8 | e:i32 | z:u8 | m:i32.
                let (mut p, ss, ee, zz) = (
                    0,
                    t.s.as_slice().unwrap(),
                    t.e.as_slice().unwrap(),
                    t.z.as_slice().unwrap(),
                );
                let mm = t.m.as_ref().unwrap().as_slice().unwrap();
                dst[p..p + n].copy_from_slice(bytemuck_i8(ss));
                p += n;
                for x in ee {
                    dst[p..p + 4].copy_from_slice(&x.to_le_bytes());
                    p += 4;
                }
                dst[p..p + n].copy_from_slice(zz);
                p += n;
                for x in mm {
                    dst[p..p + 4].copy_from_slice(&x.to_le_bytes());
                    p += 4;
                }
            } else {
                // Frozen T: element-interleaved s/e/z as i64, 24 bytes each.
                let (ss, ee, zz) = (
                    t.s.as_slice().unwrap(),
                    t.e.as_slice().unwrap(),
                    t.z.as_slice().unwrap(),
                );
                let mut p = 0;
                for i in 0..n {
                    dst[p..p + 8].copy_from_slice(&(ss[i] as i64).to_le_bytes());
                    dst[p + 8..p + 16].copy_from_slice(&(ee[i] as i64).to_le_bytes());
                    dst[p + 16..p + 24].copy_from_slice(&(zz[i] as i64).to_le_bytes());
                    p += 24;
                }
            }
        }
        (Value::None, _) => return Err(Error::runtime("flat: cannot encode None")),
    }
    Ok(spec)
}

/// Encode a value under a named convention into a fresh image.
pub fn encode(v: &Value, convention: Convention) -> Result<FlatPayload> {
    let mut bytes = vec![0u8; body_len(v, convention)?];
    let spec = write_into(v, convention, &mut bytes)?;
    Ok(FlatPayload { spec, bytes })
}

fn bytemuck_i8(s: &[i8]) -> &[u8] {
    // i8 and u8 are the same width and representation; this is the same
    // reinterpretation the old encoder did with `*x as u8`.
    unsafe { std::slice::from_raw_parts(s.as_ptr() as *const u8, s.len()) }
}

fn put_f64(dst: &mut [u8], it: impl Iterator<Item = f64>) {
    for (i, x) in it.enumerate() {
        dst[i * 8..i * 8 + 8].copy_from_slice(&x.to_le_bytes());
    }
}

fn put_f32(dst: &mut [u8], it: impl Iterator<Item = f32>) {
    for (i, x) in it.enumerate() {
        dst[i * 4..i * 4 + 4].copy_from_slice(&x.to_le_bytes());
    }
}

fn put_i64(dst: &mut [u8], it: impl Iterator<Item = i64>) {
    for (i, x) in it.enumerate() {
        dst[i * 8..i * 8 + 8].copy_from_slice(&x.to_le_bytes());
    }
}

fn le_f64(b: &[u8]) -> Vec<f64> {
    b.chunks_exact(8)
        .map(|c| f64::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

fn le_i64(b: &[u8]) -> Vec<i64> {
    b.chunks_exact(8)
        .map(|c| i64::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

fn le_i32(b: &[u8]) -> Vec<i32> {
    b.chunks_exact(4)
        .map(|c| i32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spec(kind: &str, dtype: &str, shape: &[usize]) -> StreamSpec {        StreamSpec {
            kind: kind.into(),
            dtype: dtype.into(),
            shape: shape.to_vec(),
        }
    }

    fn width_of(a: &Admission) -> usize {
        a.bytes / a.elems.max(1)
    }

    fn roundtrip(v: Value, c: Convention) {        let p = encode(&v, c).unwrap();
        let (adm, witness) = p.admit(c).unwrap();
        let back = adm.decode().unwrap();
        assert_eq!(
            crate::payload::value_to_json(&v).unwrap(),
            crate::payload::value_to_json(&back).unwrap(),
            "roundtrip {}",
            c.as_str()
        );
        assert_eq!(witness.bytes, adm.spec().shape.iter().product::<usize>() * width_of(&witness));
    }

    #[test]
    fn f32_diverges_between_conventions() {
        let v = Value::F32(array_from_vec(&[2], vec![1.5f32, -2.0]).unwrap());
        let wire = encode(&v, Convention::Wire).unwrap();
        let target = encode(&v, Convention::Target).unwrap();
        assert_eq!(wire.spec.dtype, "f32");
        assert_eq!(wire.bytes.len(), 8);
        assert_eq!(target.spec.dtype, "f64");
        assert_eq!(target.bytes.len(), 16);
        // The bytes are a different convention, so admitting one as the other
        // is a typed error, not a silent misread.
        assert!(wire.clone().admit(Convention::Target).is_err());
        assert_eq!(
            wire.admit(Convention::Wire).unwrap().1.layout(),
            "F:2"
        );
        assert_eq!(
            target.admit(Convention::Target).unwrap().1.layout(),
            "F:2"
        );
    }

    #[test]
    fn u8_and_bool_widen_on_target() {
        let u8s = Value::U8(array_from_vec(&[4], vec![7u8, 8, 9, 10]).unwrap());
        assert_eq!(encode(&u8s, Convention::Wire).unwrap().bytes.len(), 4);
        assert_eq!(encode(&u8s, Convention::Target).unwrap().bytes.len(), 32);

        let b = Value::Bool(array_from_vec(&[3], vec![true, false, true]).unwrap());
        let t = encode(&b, Convention::Target).unwrap();
        assert_eq!(t.spec.dtype, "i64");
        assert_eq!(t.bytes.len(), 24);
    }

    #[test]
    fn roundtrips_all_kinds() {
        roundtrip(
            Value::F64(array_from_vec(&[2, 2], vec![0.0, 1.25, -3.5, 1e300]).unwrap()),
            Convention::Wire,
        );
        roundtrip(
            Value::F64(array_from_vec(&[2, 2], vec![0.0, 1.25, -3.5, 1e300]).unwrap()),
            Convention::Target,
        );
        roundtrip(
            Value::I64(array_from_vec(&[3], vec![i64::MIN, 0, i64::MAX]).unwrap()),
            Convention::Wire,
        );
        roundtrip(
            Value::U8(array_from_vec(&[2], vec![0u8, 255]).unwrap()),
            Convention::Wire,
        );
    }

    #[test]
    fn truncated_payload_is_rejected() {
        let p = FlatPayload::new(spec("F", "f64", &[4]), vec![0u8; 31]);
        let e = p.admit(Convention::Wire).unwrap_err().to_string();
        assert!(e.contains("needs 32"), "{e}");
    }

    #[test]
    fn hostile_shape_is_rejected_not_panicked() {
        // 2^32 * 2^32 overflows usize: the old `product()` panics in debug
        // and wraps in release. Admission refuses it in both.
        let p = FlatPayload::new(spec("F", "f64", &[1 << 32, 1 << 32]), Vec::new());
        let e = p.admit(Convention::Wire).unwrap_err().to_string();
        assert!(e.contains("overflows"), "{e}");
    }

    #[test]
    fn empty_dtype_t_is_frozen_t() {
        let p = FlatPayload::new(spec("T", "", &[1]), vec![0u8; 24]);
        let (a, w) = p.admit(Convention::Wire).unwrap();
        assert_eq!(w.dtype, "");
        assert!(a.decode().is_ok());
    }

    #[test]
    fn witness_hash_matches_receipts() {
        let p = encode(
            &Value::I64(array_from_vec(&[2], vec![1i64, 2]).unwrap()),
            Convention::Wire,
        )
        .unwrap();
        let (_, w) = p.clone().admit(Convention::Wire).unwrap();
        assert_eq!(w.hash, hash_bytes(&p.bytes));
        assert_eq!(w.elems, 2);
        assert_eq!(w.bytes, 16);
    }
}

#[cfg(test)]
mod parity_probe {
    use super::*;

    /// The pre-admission decoder panicked here in debug and wrapped in
    /// release. Admission makes it a typed error in both.
    #[test]
    fn decoder_refuses_overflowing_shape() {
        let s = StreamSpec {
            kind: "F".into(),
            dtype: "f64".into(),
            shape: vec![1usize << 32, 1usize << 32],
        };
        let e = crate::protocol::decode_stream(&s, &[]).unwrap_err().to_string();
        assert!(e.contains("overflows"), "{e}");
    }
}

#[cfg(test)]
mod wire_frozen {
    use super::*;

    /// The refactor must not move a single byte on the wire. These are
    /// hand-computed expectations, independent of the encoder.
    #[test]
    fn t2_plane_major_bytes() {
        let t = Triple::from_vecs_opt(&[2], vec![-1i8, 2], vec![3i32, -4], vec![5u8, 6],
                                      Some(vec![7i32, -8])).unwrap();
        let p = encode(&Value::Triple(t), Convention::Wire).unwrap();
        assert_eq!(p.spec.dtype, "t2");
        assert_eq!(p.bytes, vec![
            0xFF, 0x02,                                     // s: i8 plane
            0x03, 0x00, 0x00, 0x00, 0xFC, 0xFF, 0xFF, 0xFF, // e: i32 LE plane
            0x05, 0x06,                                     // z: u8 plane
            0x07, 0x00, 0x00, 0x00, 0xF8, 0xFF, 0xFF, 0xFF, // m: i32 LE plane
        ]);
    }

    #[test]
    fn frozen_t_interleaved_i64_bytes() {
        let t = Triple::from_vecs(&[1], vec![-1i8], vec![3i32], vec![5u8]).unwrap();
        let p = encode(&Value::Triple(t), Convention::Wire).unwrap();
        assert_eq!(p.spec.dtype, "i64x3");
        let mut want = Vec::new();
        want.extend_from_slice(&(-1i64).to_le_bytes());
        want.extend_from_slice(&3i64.to_le_bytes());
        want.extend_from_slice(&5i64.to_le_bytes());
        assert_eq!(p.bytes, want);
        assert_eq!(p.bytes.len(), 24);
    }

    #[test]
    fn f32_wire_is_four_bytes() {
        let p = encode(&Value::F32(array_from_vec(&[2], vec![1.5f32, -2.0]).unwrap()),
                       Convention::Wire).unwrap();
        assert_eq!(p.bytes, vec![0x00, 0x00, 0xC0, 0x3F, 0x00, 0x00, 0x00, 0xC0]);
    }
}

#[cfg(test)]
mod receipt_shape {
    use super::*;

    #[test]
    fn witness_renders_as_a_receipt_field() {
        let p = encode(
            &Value::F64(array_from_vec(&[2, 3], vec![0.0; 6]).unwrap()),
            Convention::Target,
        ).unwrap();
        let (adm, w) = p.admit(Convention::Target).unwrap();
        println!("admitted = {}", serde_json::to_string_pretty(&w.json()).unwrap());
        assert_eq!(adm.admission(), &w);
        assert_eq!(w.elems, 6);
        assert_eq!(w.bytes, 48);
    }
}
