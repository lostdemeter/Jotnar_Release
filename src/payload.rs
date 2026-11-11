//! Portable JSON payloads for the CLI and differential tests.
//!
//! Format:
//! ```json
//! {
//!   "streams": {
//!     "x": {"kind": "F64", "shape": [2, 3], "data": [...]},
//!     "t": {"kind": "T", "shape": [2, 3], "s": [...], "e": [...], "z": [...]}
//!   }
//! }
//! ```
//!
//! Kinds: `F64`, `F32`, `I64`, `U8`, `Bool`, `T`. Integer data is carried as
//! JSON numbers; values above 2^53 are not exactly representable in JSON, so
//! large int64 fixtures should use the Python helper's string form (accepted
//! here) or stay within the exact range.

use std::collections::BTreeMap;
use std::path::Path;

use ndarray::ArrayD;
use serde_json::{json, Value as Json};

use crate::error::{Error, Result};
use crate::value::{Triple, Value};

pub type Feeds = BTreeMap<String, Value>;

fn num_i64(v: &Json, what: &str) -> Result<i64> {
    if let Some(n) = v.as_i64() {
        return Ok(n);
    }
    if let Some(s) = v.as_str() {
        return s
            .parse::<i64>()
            .map_err(|e| Error::runtime(format!("{what}: bad int {s}: {e}")));
    }
    if let Some(f) = v.as_f64() {
        return Ok(f as i64);
    }
    Err(Error::runtime(format!("{what}: expected an integer")))
}

fn num_f64(v: &Json, what: &str) -> Result<f64> {
    v.as_f64()
        .or_else(|| v.as_str().and_then(|s| s.parse().ok()))
        .ok_or_else(|| Error::runtime(format!("{what}: expected a number")))
}

fn shape_of(v: &Json) -> Result<Vec<usize>> {
    let arr = v
        .get("shape")
        .and_then(|s| s.as_array())
        .ok_or_else(|| Error::runtime("payload stream needs a shape array"))?;
    arr.iter()
        .map(|d| {
            d.as_u64()
                .map(|x| x as usize)
                .ok_or_else(|| Error::runtime("shape dims must be non-negative integers"))
        })
        .collect()
}

fn data_of<'a>(v: &'a Json, key: &str) -> Result<&'a Vec<Json>> {
    v.get(key)
        .and_then(|d| d.as_array())
        .ok_or_else(|| Error::runtime(format!("payload stream needs '{key}' array")))
}

fn decode_stream(v: &Json, name: &str) -> Result<Value> {
    if v.is_null() {
        return Ok(Value::None);
    }
    let kind = v
        .get("kind")
        .and_then(|k| k.as_str())
        .ok_or_else(|| Error::runtime(format!("stream '{name}': missing kind")))?;
    let shape = shape_of(v)?;
    let expected: usize = shape.iter().product();
    match kind {
        "T" => {
            let s: Vec<i8> = data_of(v, "s")?
                .iter()
                .map(|x| num_i64(x, "T.s").map(|v| v as i8))
                .collect::<Result<_>>()?;
            let e: Vec<i32> = data_of(v, "e")?
                .iter()
                .map(|x| num_i64(x, "T.e").map(|v| v as i32))
                .collect::<Result<_>>()?;
            let z: Vec<u8> = data_of(v, "z")?
                .iter()
                .map(|x| num_i64(x, "T.z").map(|v| v as u8))
                .collect::<Result<_>>()?;
            if s.len() != expected || e.len() != expected || z.len() != expected {
                return Err(Error::runtime(format!(
                    "stream '{name}': T planes do not match shape {shape:?}"
                )));
            }
            // Optional T2 mantissa plane; its presence selects the phi-float
            // format for this stream. Kept under kind "T" so frozen T
            // payloads and listings are byte-compatible.
            let m = match v.get("m") {
                Some(arr) => {
                    let arr = arr.as_array().ok_or_else(|| {
                        Error::runtime(format!("stream '{name}': T.m must be an array"))
                    })?;
                    let mv: Vec<i32> = arr
                        .iter()
                        .map(|x| num_i64(x, "T.m").map(|v| v as i32))
                        .collect::<Result<_>>()?;
                    if mv.len() != expected {
                        return Err(Error::runtime(format!(
                            "stream '{name}': T.m does not match shape {shape:?}"
                        )));
                    }
                    Some(mv)
                }
                None => None,
            };
            Ok(Value::Triple(Triple::from_vecs_opt(&shape, s, e, z, m)?))
        }
        "F64" | "F" => {
            let d: Vec<f64> = data_of(v, "data")?
                .iter()
                .map(|x| num_f64(x, "F64.data"))
                .collect::<Result<_>>()?;
            Ok(Value::F64(crate::value::array_from_vec(&shape, d)?))
        }
        "F32" => {
            let d: Vec<f32> = data_of(v, "data")?
                .iter()
                .map(|x| num_f64(x, "F32.data").map(|v| v as f32))
                .collect::<Result<_>>()?;
            Ok(Value::F32(crate::value::array_from_vec(&shape, d)?))
        }
        "I64" | "I" => {
            let d: Vec<i64> = data_of(v, "data")?
                .iter()
                .map(|x| num_i64(x, "I64.data"))
                .collect::<Result<_>>()?;
            Ok(Value::I64(crate::value::array_from_vec(&shape, d)?))
        }
        "U8" => {
            let d: Vec<u8> = data_of(v, "data")?
                .iter()
                .map(|x| num_i64(x, "U8.data").map(|v| v as u8))
                .collect::<Result<_>>()?;
            Ok(Value::U8(crate::value::array_from_vec(&shape, d)?))
        }
        "Bool" => {
            let d: Vec<bool> = data_of(v, "data")?
                .iter()
                .map(|x| {
                    x.as_bool()
                        .ok_or_else(|| Error::runtime("Bool.data entries must be true/false"))
                })
                .collect::<Result<_>>()?;
            Ok(Value::Bool(crate::value::array_from_vec(&shape, d)?))
        }
        "None" => Ok(Value::None),
        other => Err(Error::runtime(format!(
            "stream '{name}': unknown kind '{other}'"
        ))),
    }
}

pub fn feeds_from_json(v: &Json) -> Result<Feeds> {
    let obj = v
        .get("streams")
        .and_then(|s| s.as_object())
        .ok_or_else(|| Error::runtime("payload needs a 'streams' object"))?;
    let mut feeds = BTreeMap::new();
    for (name, sv) in obj {
        feeds.insert(name.clone(), decode_stream(sv, name)?);
    }
    Ok(feeds)
}

pub fn read_payload(path: &Path) -> Result<Feeds> {
    let text = std::fs::read_to_string(path)?;
    let v: Json = serde_json::from_str(&text)?;
    feeds_from_json(&v)
}

/// Read a host-driven `repeat()` sequence:
/// `{"iterations": [{"streams": {...}}, ...]}`.
pub fn read_sequence(path: &Path) -> Result<Vec<Feeds>> {
    let text = std::fs::read_to_string(path)?;
    let v: Json = serde_json::from_str(&text)?;
    let its = v
        .get("iterations")
        .and_then(|x| x.as_array())
        .ok_or_else(|| Error::runtime("repeat payload needs an 'iterations' array"))?;
    its.iter().map(feeds_from_json).collect()
}

fn arr_json<T, F>(a: &ArrayD<T>, mut f: F) -> Json
where
    F: FnMut(&T) -> Json,
{
    Json::Array(a.iter().map(&mut f).collect())
}

pub fn value_to_json(v: &Value) -> Result<Json> {
    match v {
        Value::F64(a) => Ok(json!({
            "kind": "F64",
            "shape": a.shape(),
            "data": arr_json(a, |x| json!(x)),
        })),
        Value::F32(a) => Ok(json!({
            "kind": "F32",
            "shape": a.shape(),
            "data": arr_json(a, |x| json!(x)),
        })),
        Value::I64(a) => Ok(json!({
            "kind": "I64",
            "shape": a.shape(),
            "data": arr_json(a, |x| json!(x.to_string())),
        })),
        Value::U8(a) => Ok(json!({
            "kind": "U8",
            "shape": a.shape(),
            "data": arr_json(a, |x| json!(x)),
        })),
        Value::Bool(a) => Ok(json!({
            "kind": "Bool",
            "shape": a.shape(),
            "data": arr_json(a, |x| json!(x)),
        })),
        Value::Triple(t) => {
            let mut obj = serde_json::Map::new();
            obj.insert("kind".into(), json!("T"));
            obj.insert("shape".into(), json!(t.shape()));
            obj.insert("s".into(), arr_json(&t.s, |x| json!(x)));
            obj.insert("e".into(), arr_json(&t.e, |x| json!(x)));
            obj.insert("z".into(), arr_json(&t.z, |x| json!(x)));
            if let Some(m) = &t.m {
                obj.insert("m".into(), arr_json(m, |x| json!(x)));
            }
            Ok(Json::Object(obj))
        }
        Value::None => Ok(Json::Null),
    }
}

pub fn feeds_to_json(feeds: &Feeds, outputs: Option<&[String]>) -> Result<Json> {
    let mut map = serde_json::Map::new();
    match outputs {
        Some(names) => {
            for name in names {
                let v = feeds
                    .get(name)
                    .ok_or_else(|| Error::runtime(format!("missing output '{name}'")))?;
                map.insert(name.clone(), value_to_json(v)?);
            }
        }
        None => {
            for (k, v) in feeds {
                map.insert(k.clone(), value_to_json(v)?);
            }
        }
    }
    Ok(Json::Object(
        [("streams".to_string(), Json::Object(map))]
            .into_iter()
            .collect(),
    ))
}

/// A small helper used by tests and callers that build payloads in Rust.
pub fn value_from_parts(kind: &str, shape: &[usize], data: &[f64]) -> Result<Value> {
    match kind {
        "F64" => Ok(Value::F64(crate::value::array_from_vec(
            shape,
            data.to_vec(),
        )?)),
        "F32" => Ok(Value::F32(crate::value::array_from_vec(
            shape,
            data.iter().map(|v| *v as f32).collect(),
        )?)),
        "I64" => Ok(Value::I64(crate::value::array_from_vec(
            shape,
            data.iter().map(|v| *v as i64).collect(),
        )?)),
        other => Err(Error::runtime(format!("unknown helper kind {other}"))),
    }
}
