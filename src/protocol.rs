//! In-process port of the Jotnar node protocol.
//!
//! No sockets: the same frames, program registry, blob shelf, and EXEC
//! semantics are carried through channels. A node is deliberately
//! model-blind: it sees a listing, named blobs, and typed streams, never the
//! workload that produced them.

use std::collections::{BTreeMap, HashMap};
use std::sync::mpsc::{self, Receiver, Sender};
use std::thread;

use serde_json::{json, Value as Json};

use crate::asm;
use crate::error::{Error, Result};
use crate::payload::Feeds;
use crate::runtime;
use crate::value::{Triple, Value};

pub const PROTO_V: u64 = 1;

#[derive(Clone, Debug)]
pub struct StreamSpec {
    pub kind: String,
    pub dtype: String,
    pub shape: Vec<usize>,
}

impl StreamSpec {
    fn from_json(v: &Json) -> Result<Self> {
        let kind = v
            .get("kind")
            .and_then(|x| x.as_str())
            .ok_or_else(|| Error::runtime("stream needs kind"))?
            .to_string();
        let dtype = v
            .get("dtype")
            .and_then(|x| x.as_str())
            .unwrap_or("")
            .to_string();
        let shape = v
            .get("shape")
            .and_then(|x| x.as_array())
            .ok_or_else(|| Error::runtime("stream needs shape"))?
            .iter()
            .map(|d| {
                d.as_u64()
                    .map(|x| x as usize)
                    .ok_or_else(|| Error::runtime("bad shape"))
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(StreamSpec { kind, dtype, shape })
    }

    pub fn json(&self) -> Json {
        json!({"kind": self.kind, "dtype": self.dtype, "shape": self.shape})
    }
}

pub fn encode_frame(header: &Json, payload: &[u8]) -> Result<Vec<u8>> {
    let h = serde_json::to_vec(header)?;
    let mut body = h;
    body.push(b'\n');
    body.extend_from_slice(payload);
    let n = body.len() as u32;
    let mut out = Vec::with_capacity(4 + body.len());
    out.extend_from_slice(&n.to_be_bytes());
    out.extend_from_slice(&body);
    Ok(out)
}

pub fn decode_frame(frame: &[u8]) -> Result<(Json, Vec<u8>)> {
    if frame.len() < 4 {
        return Err(Error::runtime("protocol: frame too short"));
    }
    let n = u32::from_be_bytes([frame[0], frame[1], frame[2], frame[3]]) as usize;
    let body = &frame[4..];
    if body.len() != n {
        return Err(Error::runtime(format!(
            "protocol: length prefix {n} != body {}",
            body.len()
        )));
    }
    let pos = body
        .iter()
        .position(|b| *b == b'\n')
        .ok_or_else(|| Error::runtime("protocol: header has no newline"))?;
    let header: Json = serde_json::from_slice(&body[..pos])?;
    if header.get("v").and_then(|x| x.as_u64()) != Some(PROTO_V) {
        return Err(Error::runtime("protocol: bad version"));
    }
    if header.get("id").is_none() || header.get("type").is_none() {
        return Err(Error::runtime("protocol: header needs id + type"));
    }
    Ok((header, body[pos + 1..].to_vec()))
}

pub fn encode_stream(v: &Value) -> Result<(StreamSpec, Vec<u8>)> {
    match v {
        Value::F64(a) => Ok((
            StreamSpec {
                kind: "F".into(),
                dtype: "f64".into(),
                shape: a.shape().to_vec(),
            },
            a.iter().flat_map(|x| x.to_le_bytes()).collect(),
        )),
        Value::F32(a) => Ok((
            StreamSpec {
                kind: "F".into(),
                dtype: "f32".into(),
                shape: a.shape().to_vec(),
            },
            a.iter().flat_map(|x| x.to_le_bytes()).collect(),
        )),
        Value::I64(a) => Ok((
            StreamSpec {
                kind: "I".into(),
                dtype: "i64".into(),
                shape: a.shape().to_vec(),
            },
            a.iter().flat_map(|x| x.to_le_bytes()).collect(),
        )),
        Value::U8(a) => Ok((
            StreamSpec {
                kind: "I".into(),
                dtype: "u8".into(),
                shape: a.shape().to_vec(),
            },
            a.iter().copied().collect(),
        )),
        Value::Bool(a) => Ok((
            StreamSpec {
                kind: "I".into(),
                dtype: "i64".into(),
                shape: a.shape().to_vec(),
            },
            a.iter().flat_map(|x| (*x as i64).to_le_bytes()).collect(),
        )),
        Value::Triple(t) => {
            let n = t.s.len();
            let mut b = Vec::with_capacity(n * 24);
            for x in t.s.iter() {
                b.extend_from_slice(&(*x as i64).to_le_bytes());
            }
            for x in t.e.iter() {
                b.extend_from_slice(&(*x as i64).to_le_bytes());
            }
            for x in t.z.iter() {
                b.extend_from_slice(&(*x as i64).to_le_bytes());
            }
            Ok((
                StreamSpec {
                    kind: "T".into(),
                    dtype: "i64x3".into(),
                    shape: t.shape(),
                },
                b,
            ))
        }
        Value::None => Err(Error::runtime("protocol: cannot encode None")),
    }
}

pub fn decode_stream(spec: &StreamSpec, payload: &[u8]) -> Result<Value> {
    let n: usize = spec.shape.iter().product();
    match (spec.kind.as_str(), spec.dtype.as_str()) {
        ("F", "f64") => {
            if payload.len() != n * 8 {
                return Err(Error::runtime("protocol: f64 payload size"));
            }
            let mut v = Vec::with_capacity(n);
            for c in payload.chunks_exact(8) {
                v.push(f64::from_le_bytes([
                    c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7],
                ]));
            }
            Ok(Value::F64(crate::value::array_from_vec(&spec.shape, v)?))
        }
        ("F", "f32") => {
            if payload.len() != n * 4 {
                return Err(Error::runtime("protocol: f32 payload size"));
            }
            let mut v = Vec::with_capacity(n);
            for c in payload.chunks_exact(4) {
                v.push(f32::from_le_bytes([c[0], c[1], c[2], c[3]]));
            }
            Ok(Value::F32(crate::value::array_from_vec(&spec.shape, v)?))
        }
        ("I", "i64") => {
            if payload.len() != n * 8 {
                return Err(Error::runtime("protocol: i64 payload size"));
            }
            let mut v = Vec::with_capacity(n);
            for c in payload.chunks_exact(8) {
                v.push(i64::from_le_bytes([
                    c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7],
                ]));
            }
            Ok(Value::I64(crate::value::array_from_vec(&spec.shape, v)?))
        }
        ("I", "u8") => {
            if payload.len() != n {
                return Err(Error::runtime("protocol: u8 payload size"));
            }
            Ok(Value::U8(crate::value::array_from_vec(
                &spec.shape,
                payload.to_vec(),
            )?))
        }
        ("T", _) => {
            if payload.len() != n * 24 {
                return Err(Error::runtime("protocol: T payload size"));
            }
            let mut s = Vec::with_capacity(n);
            let mut e = Vec::with_capacity(n);
            let mut z = Vec::with_capacity(n);
            for i in 0..n {
                let b = &payload[i * 24..];
                s.push(i64::from_le_bytes(b[0..8].try_into().unwrap()) as i8);
                e.push(i64::from_le_bytes(b[8..16].try_into().unwrap()) as i32);
                z.push(i64::from_le_bytes(b[16..24].try_into().unwrap()) as u8);
            }
            Ok(Value::Triple(Triple::from_vecs(&spec.shape, s, e, z)?))
        }
        (k, d) => Err(Error::runtime(format!(
            "protocol: unsupported stream {k}/{d}"
        ))),
    }
}

#[derive(Clone, Debug)]
struct Program {
    bound: crate::ir::BoundProgram,
}

#[derive(Clone, Debug)]
struct Stored {
    spec: StreamSpec,
    payload: Vec<u8>,
}

/// A model-blind node.
pub struct Node {
    pub allowed: Vec<String>,
    pub programs: HashMap<String, Program>,
    pub stores: HashMap<String, Stored>,
}

impl Node {
    pub fn new(allowed: Vec<String>) -> Self {
        Node {
            allowed,
            programs: HashMap::new(),
            stores: HashMap::new(),
        }
    }

    fn err(id: &Json, code: &str, msg: impl Into<String>) -> Vec<u8> {
        let h = json!({"v": PROTO_V, "id": id, "type": "error", "ok": false,
                       "error": {"code": code, "msg": msg.into()}});
        encode_frame(&h, &[]).unwrap()
    }

    fn ok(id: &Json, typ: &str, fields: Json, payload: &[u8]) -> Vec<u8> {
        let mut h = fields;
        if let Some(o) = h.as_object_mut() {
            o.insert("v".into(), json!(PROTO_V));
            o.insert("id".into(), id.clone());
            o.insert("type".into(), json!(typ));
            o.insert("ok".into(), json!(true));
        }
        encode_frame(&h, payload).unwrap()
    }

    pub fn handle(&mut self, frame: &[u8]) -> Vec<u8> {
        let (header, payload) = match decode_frame(frame) {
            Ok(x) => x,
            Err(e) => return Self::err(&json!(0), "malformed", e.to_string()),
        };
        let id = header.get("id").cloned().unwrap_or(json!(0));
        let typ = header.get("type").and_then(|x| x.as_str()).unwrap_or("");
        let result = match typ {
            "ping" => Ok(Self::ok(
                &id,
                "pong",
                json!({"pong": {"impl": "jotnar-node", "v": PROTO_V,
                                "backends": self.allowed, "mnemonics": 86}}),
                &[],
            )),
            "program" => self.program(&id, &header, &payload),
            "store" => self.store(&id, &header, &payload),
            "drop" => self.drop(&id, &header),
            "list" => Ok(Self::ok(
                &id,
                "list-ok",
                json!({"programs": self.programs.keys().cloned().collect::<Vec<_>>(),
                       "stores": self.stores.keys().cloned().collect::<Vec<_>>()}),
                &[],
            )),
            "exec" => self.exec(&id, &header, &payload),
            _ => Err((format!("no such message {typ:?}"), "unknown-type")),
        };
        match result {
            Ok(f) => f,
            Err((msg, code)) => Self::err(&id, code, msg),
        }
    }

    fn program(
        &mut self,
        id: &Json,
        h: &Json,
        payload: &[u8],
    ) -> std::result::Result<Vec<u8>, (String, &'static str)> {
        let name = h
            .get("name")
            .and_then(|x| x.as_str())
            .ok_or(("missing name".to_string(), "malformed"))?;
        let backend = h
            .get("backend")
            .and_then(|x| x.as_str())
            .unwrap_or("lattice");
        if !self.allowed.contains(&backend.to_string()) {
            return Err((
                format!("backend {backend:?} unavailable"),
                "backend-unavailable",
            ));
        }
        if backend != "lattice" {
            return Err((
                "only the lattice backend is ported".to_string(),
                "backend-unavailable",
            ));
        }
        let listing = std::str::from_utf8(payload).map_err(|e| (e.to_string(), "malformed"))?;
        let bound = asm::assemble(listing, &asm::AsmOptions::new("."))
            .map_err(|e| (e.to_string(), "npattern"))?;
        self.programs.insert(name.to_string(), Program { bound });
        Ok(Self::ok(
            id,
            "program-ok",
            json!({"handle": name, "replaced": false}),
            &[],
        ))
    }

    fn store(
        &mut self,
        id: &Json,
        h: &Json,
        payload: &[u8],
    ) -> std::result::Result<Vec<u8>, (String, &'static str)> {
        let name = h
            .get("name")
            .and_then(|x| x.as_str())
            .ok_or(("missing name".to_string(), "malformed"))?;
        let spec = StreamSpec::from_json(
            h.get("stream")
                .ok_or(("missing stream".to_string(), "malformed"))?,
        )
        .map_err(|e| (e.to_string(), "malformed"))?;
        // Validate now, not at EXEC.
        decode_stream(&spec, payload).map_err(|e| (e.to_string(), "malformed"))?;
        self.stores.insert(
            name.to_string(),
            Stored {
                spec,
                payload: payload.to_vec(),
            },
        );
        Ok(Self::ok(
            id,
            "store-ok",
            json!({"name": name, "bytes": payload.len()}),
            &[],
        ))
    }

    fn drop(
        &mut self,
        id: &Json,
        h: &Json,
    ) -> std::result::Result<Vec<u8>, (String, &'static str)> {
        let name = h.get("name").and_then(|x| x.as_str()).unwrap_or("");
        let removed = self.stores.remove(name).is_some() | self.programs.remove(name).is_some();
        Ok(Self::ok(
            id,
            "drop-ok",
            json!({"name": name, "missing": !removed}),
            &[],
        ))
    }

    fn exec(
        &mut self,
        id: &Json,
        h: &Json,
        payload: &[u8],
    ) -> std::result::Result<Vec<u8>, (String, &'static str)> {
        let handle = h
            .get("handle")
            .and_then(|x| x.as_str())
            .ok_or(("missing handle".to_string(), "malformed"))?;
        let prog = self
            .programs
            .get(handle)
            .ok_or((format!("no program {handle:?}"), "unknown-handle"))?;
        let mut feeds = Feeds::new();
        let inputs = h
            .get("inputs")
            .and_then(|x| x.as_object())
            .ok_or(("missing inputs".to_string(), "malformed"))?;
        for (stream, refv) in inputs {
            let v = if let Some(store) = refv.get("store").and_then(|x| x.as_str()) {
                let s = self
                    .stores
                    .get(store)
                    .ok_or((format!("no store {store:?}"), "unknown-store"))?;
                decode_stream(&s.spec, &s.payload).map_err(|e| (e.to_string(), "malformed"))?
            } else {
                let off = refv.get("off").and_then(|x| x.as_u64()).unwrap_or(0) as usize;
                let len = refv.get("len").and_then(|x| x.as_u64()).unwrap_or(0) as usize;
                let spec = StreamSpec::from_json(refv).map_err(|e| (e.to_string(), "malformed"))?;
                let end = off
                    .checked_add(len)
                    .ok_or(("bad segment".to_string(), "malformed"))?;
                if end > payload.len() {
                    return Err((format!("inline segment {stream} truncated"), "malformed"));
                }
                decode_stream(&spec, &payload[off..end])
                    .map_err(|e| (e.to_string(), "malformed"))?
            };
            feeds.insert(stream.clone(), v);
        }
        let got = runtime::run(&prog.bound, &feeds).map_err(|e| (e.to_string(), "node"))?;
        let outs = h
            .get("outputs")
            .and_then(|x| x.as_array())
            .ok_or(("missing outputs".to_string(), "malformed"))?;
        let mut payload = Vec::new();
        let mut specs = serde_json::Map::new();
        for name in outs {
            let name = name
                .as_str()
                .ok_or(("bad output name".to_string(), "malformed"))?;
            let v = got
                .get(name)
                .ok_or((format!("missing output {name}"), "arity"))?;
            let (spec, bytes) = encode_stream(v).map_err(|e| (e.to_string(), "node"))?;
            let off = payload.len();
            payload.extend_from_slice(&bytes);
            let mut sj = spec.json();
            if let Some(o) = sj.as_object_mut() {
                o.insert("off".into(), json!(off));
                o.insert("len".into(), json!(bytes.len()));
            }
            specs.insert(name.to_string(), sj);
        }
        Ok(Self::ok(id, "exec-ok", json!({"outputs": specs}), &payload))
    }
}

/// A node running on its own thread. One request at a time models the
/// protocol's one-client-per-server rule; fan-out lives in the controller.
#[derive(Clone)]
pub struct NodeHandle {
    tx: Sender<(Vec<u8>, Sender<Vec<u8>>)>,
}

impl NodeHandle {
    pub fn spawn(id: usize, allowed: &[&str]) -> Self {
        let allowed: Vec<String> = allowed.iter().map(|s| s.to_string()).collect();
        let (tx, rx) = mpsc::channel::<(Vec<u8>, Sender<Vec<u8>>)>();
        thread::spawn(move || {
            let mut node = Node::new(allowed);
            let _ = id;
            while let Ok((frame, reply)) = rx.recv() {
                let out = node.handle(&frame);
                let _ = reply.send(out);
            }
        });
        NodeHandle { tx }
    }

    pub fn request(&self, frame: Vec<u8>) -> Result<Vec<u8>> {
        let (tx, rx) = mpsc::channel();
        self.tx
            .send((frame, tx))
            .map_err(|_| Error::runtime("node thread is gone"))?;
        rx.recv()
            .map_err(|_| Error::runtime("node thread dropped the response"))
    }
}

/// Decode an EXEC response into output bytes keyed by name.
pub fn exec_outputs(response: &[u8]) -> Result<BTreeMap<String, Vec<u8>>> {
    let (header, payload) = decode_frame(response)?;
    if header.get("ok").and_then(|x| x.as_bool()) != Some(true) {
        return Err(Error::runtime(format!("exec failed: {header}")));
    }
    let outputs = header
        .get("outputs")
        .and_then(|x| x.as_object())
        .ok_or_else(|| Error::runtime("exec response has no outputs"))?;
    let mut out = BTreeMap::new();
    for (name, s) in outputs {
        let off = s.get("off").and_then(|x| x.as_u64()).unwrap_or(0) as usize;
        let len = s.get("len").and_then(|x| x.as_u64()).unwrap_or(0) as usize;
        let end = off + len;
        if end > payload.len() {
            return Err(Error::runtime("exec output truncated"));
        }
        out.insert(name.clone(), payload[off..end].to_vec());
    }
    Ok(out)
}

/// Build a request frame with an id.
pub fn request(id: u64, typ: &str, fields: Json, payload: &[u8]) -> Result<Vec<u8>> {
    let mut h = fields;
    if let Some(o) = h.as_object_mut() {
        o.insert("v".into(), json!(PROTO_V));
        o.insert("id".into(), json!(id));
        o.insert("type".into(), json!(typ));
    }
    encode_frame(&h, payload)
}

#[allow(dead_code)]
fn _keep(_: Receiver<()>) {}
