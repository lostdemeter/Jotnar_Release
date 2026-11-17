//! In-process port of the Jotnar node protocol.
//!
//! No sockets: the same frames, program registry, blob shelf, and EXEC
//! semantics are carried through channels. A node is deliberately
//! model-blind: it sees a listing, named blobs, and typed streams, never the
//! workload that produced them.
//!
//! `PROTO_V` bumps only on breaking header-semantics changes. Optional
//! header fields (e.g. `"auth"`) are ignored by older nodes, so they do
//! not bump the version. Unknown message types are `error/unknown-type`.

use std::collections::{BTreeMap, HashMap};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::Arc;
use std::thread;

use serde_json::{json, Value as Json};

use crate::arena::JobArena;
use crate::asm;
use crate::error::{Error, Result};
use crate::flat::{Admission, Convention};
use crate::payload::Feeds;
use crate::profile::JobScope;
use crate::runtime::{self, ExecOptions, ThreadPolicy};
use crate::value::{hash_bytes, Value};
use crate::wire::FrameBuf;

pub const PROTO_V: u64 = 1;

/// Build identity for protocol responses. Deterministic for a given binary:
/// package version plus the short git HEAD resolved by `build.rs`.
/// Build identity, allocation-free: both halves are `'static`.
pub const RUNTIME_VERSION: &str = concat!(env!("CARGO_PKG_VERSION"), "+", env!("JOTNAR_GIT_HASH"));

pub fn runtime_version() -> String {
    RUNTIME_VERSION.to_string()
}

#[derive(Clone, Debug)]
pub struct StreamSpec {
    pub kind: String,
    pub dtype: String,
    pub shape: Vec<usize>,
}

impl StreamSpec {
    pub fn from_json(v: &Json) -> Result<Self> {
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

/// Serialize a frame into `out`, reusing its capacity.
///
/// The header is streamed with `serde_json::to_writer` (no intermediate
/// `to_vec`), the length prefix is patched last, and the payload is appended
/// in place. After the buffer has grown once, a steady-state reply allocates
/// nothing here.
pub fn encode_frame_into(out: &mut Vec<u8>, header: &Json, payload: &[u8]) -> Result<()> {
    let mut f = crate::wire::FrameBuf::begin(out);
    {
        let mut w = f.json();
        w.value(header)?;
    }
    f.end(payload)
}

pub fn encode_frame(header: &Json, payload: &[u8]) -> Result<Vec<u8>> {
    let mut out = Vec::new();
    encode_frame_into(&mut out, header, payload)?;
    Ok(out)
}

pub fn decode_frame<'a>(frame: &'a [u8]) -> Result<(Json, &'a [u8])> {
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
    // Borrowed: the payload is a slice of the input frame, not a copy.
    Ok((header, &body[pos + 1..]))
}

/// Build an error frame without a `Node` (for transport-level rejects
/// such as an oversized length prefix, before `Node::handle` can run).
pub fn error_frame(id: &Json, code: &str, msg: impl Into<String>) -> Vec<u8> {
    let h = json!({"v": PROTO_V, "id": id, "type": "error", "ok": false,
                   "error": {"code": code, "msg": msg.into()}});
    encode_frame(&h, &[]).unwrap()
}

pub fn encode_stream(v: &Value) -> Result<(StreamSpec, Vec<u8>)> {
    // The wire codec lives in `flat`, next to the target codec, so both use
    // one size table and one admission gate.
    let p = crate::flat::encode(v, crate::flat::Convention::Wire)?;
    Ok((p.spec, p.bytes))
}

pub fn decode_stream(spec: &StreamSpec, payload: &[u8]) -> Result<Value> {
    let (admitted, _witness) = crate::flat::FlatPayload::new(spec.clone(), payload.to_vec())
        .admit(crate::flat::Convention::Wire)?;
    admitted.decode()
}

#[derive(Clone, Debug)]
struct Program {
    /// Shared with snapshots, so submitting a job does not copy the binding.
    bound: Arc<crate::ir::BoundProgram>,
    /// Canonical hash of the listing bytes as published.
    hash: u64,
}

#[derive(Clone, Debug)]
struct Stored {
    /// What the node checked when the blob was written.
    admission: Admission,
    /// Shared with snapshots, so submitting a job does not copy the bytes.
    payload: Arc<Vec<u8>>,
}

impl Stored {
    fn spec(&self) -> StreamSpec {
        StreamSpec {
            kind: self.admission.kind.clone(),
            dtype: self.admission.dtype.clone(),
            shape: self.admission.shape.clone(),
        }
    }
}

/// A model-blind node.
pub struct Node {
    pub allowed: Vec<String>,
    pub token: Option<String>,
    pub programs: HashMap<String, Program>,
    pub stores: HashMap<String, Stored>,
    /// Boundary storage for one job at a time, reset per EXEC.
    arena: JobArena,
    /// Stream storage for the arena execution path.
    stream_arena: crate::stream_arena::StreamArena,
    /// Per-program compiled plans, keyed by handle.
    plans: HashMap<String, crate::stream_arena::PlanCache>,
}

impl Node {
    pub fn new(allowed: Vec<String>) -> Self {
        Node {
            allowed,
            token: None,
            programs: HashMap::new(),
            stores: HashMap::new(),
            arena: JobArena::new(),
            stream_arena: crate::stream_arena::StreamArena::new(),
            plans: HashMap::new(),
        }
    }

    /// Require a shared `--token` from every client. Requests must
    /// carry `"auth": "<token>"` in the header; otherwise the node
    /// replies `error/unauthorized` before dispatching.
    ///
    /// The token is a trusted-LAN convenience, not a cryptographic
    /// authenticator: it rides cleartext in the JSON header with no
    /// TLS and is compared with `!=`. Empty or whitespace-only tokens
    /// are treated as misconfiguration and never authorize a request,
    /// even if the caller sends no `auth` header.
    pub fn with_token(mut self, token: impl Into<String>) -> Self {
        self.token = Some(token.into());
        self
    }

    /// Snapshot the registries for one job.
    ///
    /// Programs and blob payloads live behind `Arc`, so this copies keys and
    /// shares the immutable bodies. Execution state is fresh: a job gets its
    /// own boundary arena, stream arena, and plan cache, while later
    /// `program`/`store`/`drop` updates cannot change what the job sees.
    pub fn snapshot(&self) -> Node {
        Node {
            allowed: self.allowed.clone(),
            token: self.token.clone(),
            programs: self.programs.clone(),
            stores: self.stores.clone(),
            arena: JobArena::new(),
            stream_arena: crate::stream_arena::StreamArena::new(),
            plans: HashMap::new(),
        }
    }

    /// Check a request header against the configured shared token.
    ///
    /// Empty or whitespace-only tokens are treated as misconfiguration and
    /// never authorize a request, even if the caller sends no `auth` header.
    pub fn authorize(&self, header: &Json) -> bool {
        match &self.token {
            None => true,
            Some(want) => {
                let got = header.get("auth").and_then(|x| x.as_str()).unwrap_or("");
                !want.trim().is_empty() && got == want.as_str()
            }
        }
    }

    fn err_into(out: &mut Vec<u8>, id: &Json, code: &str, msg: impl Into<String>) {
        let h = json!({"v": PROTO_V, "id": id, "type": "error", "ok": false,
                       "error": {"code": code, "msg": msg.into()}});
        encode_frame_into(out, &h, &[]).expect("frame writer");
    }

    fn ok_into(out: &mut Vec<u8>, id: &Json, typ: &'static str, fields: Json, payload: &[u8]) {
        let mut h = fields;
        if let Some(o) = h.as_object_mut() {
            o.insert("v".into(), json!(PROTO_V));
            o.insert("id".into(), id.clone());
            o.insert("type".into(), json!(typ));
            o.insert("ok".into(), json!(true));
        }
        encode_frame_into(out, &h, payload).expect("frame writer");
    }

    /// Write one reply into `out`, reusing its capacity.
    ///
    /// The steady state allocates nothing for framing: the header is streamed
    /// into the buffer and the payload is appended in place. What remains is
    /// the header `Value` tree itself.
    pub fn handle_into(&mut self, frame: &[u8], out: &mut Vec<u8>) {
        out.clear();
        let (header, payload) = match decode_frame(frame) {
            Ok(x) => x,
            Err(e) => {
                Self::err_into(out, &json!(0), "malformed", e.to_string());
                return;
            }
        };
        let zero = json!(0);
        let id = header.get("id").unwrap_or(&zero);
        if !self.authorize(&header) {
            Self::err_into(out, id, "unauthorized", "bad or missing auth token");
            return;
        }
        let typ = header.get("type").and_then(|x| x.as_str()).unwrap_or("");
        let result = match typ {
            "ping" => {
                Self::ok_into(
                    out,
                    id,
                    "pong",
                    json!({"pong": {"impl": "jotnar-node", "v": PROTO_V,
                                    "backends": self.allowed,
                                    "mnemonics": crate::ops::registry().len(),
                                    "runtime_version": RUNTIME_VERSION}}),
                    &[],
                );
                Ok(())
            }
            "program" => self.program(id, &header, payload, out),
            "store" => self.store(id, &header, payload, out),
            "drop" => self.drop(id, &header, out),
            "list" => {
                Self::ok_into(
                    out,
                    id,
                    "list-ok",
                    json!({"programs": self.programs.keys().cloned().collect::<Vec<_>>(),
                           "stores": self.stores.keys().cloned().collect::<Vec<_>>()}),
                    &[],
                );
                Ok(())
            }
            "exec" => self.exec(id, &header, payload, out),
            _ => Err((format!("no such message {typ:?}"), "unknown-type")),
        };
        if let Err((msg, code)) = result {
            Self::err_into(out, id, code, msg);
        }
    }

    /// Convenience wrapper that allocates the reply buffer.
    pub fn handle(&mut self, frame: &[u8]) -> Vec<u8> {
        let mut out = Vec::new();
        self.handle_into(frame, &mut out);
        out
    }

    fn program(
        &mut self,
        id: &Json,
        h: &Json,
        payload: &[u8],
        out: &mut Vec<u8>,
    ) -> std::result::Result<(), (String, &'static str)> {
        let name = h
            .get("name")
            .and_then(|x| x.as_str())
            .ok_or(("missing name".to_string(), "malformed"))?;
        let backend = h
            .get("backend")
            .and_then(|x| x.as_str())
            .unwrap_or("lattice");
        if !self.allowed.iter().any(|a| a == backend) {
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
        let hash = hash_bytes(listing.as_bytes());
        self.programs.insert(
            name.to_string(),
            Program {
                bound: Arc::new(bound),
                hash,
            },
        );
        Self::ok_into(
            out,
            id,
            "program-ok",
            json!({"handle": name, "replaced": false}),
            &[],
        );
        Ok(())
    }

    fn store(
        &mut self,
        id: &Json,
        h: &Json,
        payload: &[u8],
        out: &mut Vec<u8>,
    ) -> std::result::Result<(), (String, &'static str)> {
        let name = h
            .get("name")
            .and_then(|x| x.as_str())
            .ok_or(("missing name".to_string(), "malformed"))?;
        let spec = StreamSpec::from_json(
            h.get("stream")
                .ok_or(("missing stream".to_string(), "malformed"))?,
        )
        .map_err(|e| (e.to_string(), "malformed"))?;
        // Admit now, not at EXEC: the witness is kept beside the bytes.
        let bytes = payload.to_vec();
        let (_, admission) = crate::flat::FlatPayload::new(spec, bytes.clone())
            .admit(Convention::Wire)
            .map_err(|e| (e.to_string(), "malformed"))?;
        let layout = admission.layout();
        let hash = format!("{:016x}", admission.hash);
        self.stores.insert(
            name.to_string(),
            Stored {
                admission,
                payload: Arc::new(bytes),
            },
        );
        Self::ok_into(
            out,
            id,
            "store-ok",
            json!({"name": name, "bytes": payload.len(), "layout": layout, "hash": hash}),
            &[],
        );
        Ok(())
    }

    fn drop(
        &mut self,
        id: &Json,
        h: &Json,
        out: &mut Vec<u8>,
    ) -> std::result::Result<(), (String, &'static str)> {
        let name = h.get("name").and_then(|x| x.as_str()).unwrap_or("");
        let removed = self.stores.remove(name).is_some() | self.programs.remove(name).is_some();
        self.plans.remove(name);
        Self::ok_into(
            out,
            id,
            "drop-ok",
            json!({"name": name, "missing": !removed}),
            &[],
        );
        Ok(())
    }

    fn exec(
        &mut self,
        id: &Json,
        h: &Json,
        payload: &[u8],
        out: &mut Vec<u8>,
    ) -> std::result::Result<(), (String, &'static str)> {
        let handle = h
            .get("handle")
            .and_then(|x| x.as_str())
            .ok_or(("missing handle".to_string(), "malformed"))?;
        let prog = self
            .programs
            .get(handle)
            .ok_or((format!("no program {handle:?}"), "unknown-handle"))?;
        // A job is a scope: one store to reuse boundary storage, and one
        // allocation measurement around the whole job.
        self.arena.reset();
        let scope = JobScope::begin();
        let mut feeds = Feeds::new();
        let mut input_hashes: Vec<(&str, u64)> = Vec::new();
        let mut blob_hashes: Vec<(&str, u64)> = Vec::new();
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
                let v = decode_stream(&s.spec(), &s.payload)
                    .map_err(|e| (e.to_string(), "malformed"))?;
                // Semantic content hash: independent of the wire dtype.
                if !blob_hashes.iter().any(|(n, _)| *n == store) {
                    blob_hashes.push((store, v.exact_hash()));
                }
                v
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
            input_hashes.push((stream.as_str(), v.exact_hash()));
            feeds.insert(stream.clone(), v);
        }

        // Additive execution policy. The default stays the 1-thread reference
        // order; auto-threading must be requested explicitly. `threads`
        // without a policy is a literal job budget.
        let policy_name = h.get("thread_policy").and_then(|x| x.as_str());
        let job_budget = h.get("threads").and_then(|x| x.as_u64()).unwrap_or(0) as usize;
        let exec = match policy_name {
            Some(s) => ThreadPolicy::parse(s)
                .ok_or((format!("bad thread_policy {s:?}"), "malformed"))?
                .resolve(job_budget),
            None if job_budget > 0 => ExecOptions::new(job_budget),
            None => ExecOptions::default(),
        };

        let profile_ops = h
            .get("profile_ops")
            .and_then(|x| x.as_bool())
            .unwrap_or(false);
        let mut prof = crate::profile::Profile::new();
        let got = if profile_ops {
            let was = crate::profile::set_track_alloc(true);
            let r = runtime::run_profiled(&prog.bound, &feeds, &exec, &mut prof);
            crate::profile::set_track_alloc(was);
            r
        } else if exec.threads == 1 && crate::stream_arena::enabled() {
            // Warm path: reuse the compiled plan and the stream arena.
            if !self.plans.contains_key(handle) {
                self.plans
                    .insert(handle.to_string(), crate::stream_arena::PlanCache::new());
            }
            let cache = self.plans.get_mut(handle).expect("just inserted");
            match cache.get(&prog.bound, &feeds) {
                Some(plan) => {
                    let mut out = Feeds::new();
                    plan.run_feeds(&mut self.stream_arena, &feeds, &mut out)
                        .map(|()| out)
                }
                None => runtime::run_with_options(&prog.bound, &feeds, &exec),
            }
        } else {
            runtime::run_with_options(&prog.bound, &feeds, &exec)
        };
        let got = got.map_err(|e| (e.to_string(), "node"))?;

        let outs = h
            .get("outputs")
            .and_then(|x| x.as_array())
            .ok_or(("missing outputs".to_string(), "malformed"))?;
        // Collected as plain tuples, not a `Json` tree: the header is streamed
        // below, so nothing here needs to allocate a map node or a hex String.
        let mut specs: Vec<(&str, StreamSpec, usize, usize)> = Vec::new();
        let mut output_hashes: Vec<(&str, u64)> = Vec::new();
        let mut kinds: Vec<crate::value::Kind> = Vec::new();
        for name in outs {
            let name = name
                .as_str()
                .ok_or(("bad output name".to_string(), "malformed"))?;
            let v = got
                .get(name)
                .ok_or((format!("missing output {name}"), "arity"))?;
            output_hashes.push((name, v.exact_hash()));
            kinds.push(v.kind());
            // Write the image straight into the job arena: no transient Vec.
            let n =
                crate::flat::body_len(v, Convention::Wire).map_err(|e| (e.to_string(), "node"))?;
            let off = self.arena.alloc(n).map_err(|e| (e.to_string(), "node"))?;
            let spec = crate::flat::write_into(v, Convention::Wire, self.arena.slice_mut(off, n))
                .map_err(|e| (e.to_string(), "node"))?;
            specs.push((name, spec, off, n));
        }
        // `serde_json::Map` is a `BTreeMap`, so canonical frames sort keys.
        specs.sort_by_key(|(n, ..)| *n);
        output_hashes.sort_by_key(|(n, _)| *n);
        input_hashes.sort_by_key(|(n, _)| *n);
        blob_hashes.sort_by_key(|(n, _)| *n);

        // Close the job scope after the last boundary write, then hold the
        // job to its declared ceiling (if the controller set one).
        let alloc_witness = scope.finish();
        if let Some(max) = h.get("alloc_max").and_then(|x| x.as_u64()) {
            alloc_witness
                .check(max)
                .map_err(|e| (e.to_string(), "alloc-budget"))?;
        }

        // Derive a conservative determinism label; the controller may
        // declare one explicitly. T2 (Triple with a mantissa) is exact-int.
        let derived_profile = if kinds.iter().all(|k| {
            matches!(
                k,
                crate::value::Kind::I | crate::value::Kind::U8 | crate::value::Kind::T
            )
        }) {
            "exact-int"
        } else if kinds.iter().all(|k| matches!(k, crate::value::Kind::F)) {
            "float"
        } else {
            "mixed"
        };
        let determinism = h
            .get("determinism")
            .and_then(|x| x.as_str())
            .unwrap_or(derived_profile);
        let verdict = match determinism {
            "exact-int" | "exact-float" => "exact",
            "tolerance-float" => "declared",
            _ => "unspecified",
        };
        let threads = exec.threads;
        // Diagnostic-only extensions: built as `Json` because they are opt-in
        // and are not on the steady-state path.
        let diagnostic = if profile_ops {
            let mut aw = alloc_witness.json();
            aw["arena_detail"] = self.arena.json();
            let ops: Vec<Json> = prof
                .ops
                .iter()
                .map(|(name, p)| {
                    json!({"op": name,
                           "calls": p.calls,
                           "wall_ms": p.wall.as_secs_f64() * 1e3,
                           "alloc_bytes": p.alloc_bytes,
                           "in_bytes": p.in_bytes,
                           "out_bytes": p.out_bytes})
                })
                .collect();
            Some((aw, ops))
        } else {
            None
        };

        // Canonical receipt and reply, streamed straight into the frame. Keys
        // are emitted in sorted order to match `serde_json::Map`.
        let mut f = FrameBuf::begin(out);
        {
            let mut w = f.json();
            w.begin_object();
            if let Some((aw, _)) = &diagnostic {
                w.kv_value("alloc_witness", aw)
                    .map_err(|e| (e.to_string(), "node"))?;
            }
            w.key("id");
            w.value(id).map_err(|e| (e.to_string(), "node"))?;
            w.kv_bool("ok", true);
            w.key("outputs");
            w.begin_object();
            for (name, spec, off, n) in &specs {
                w.key(name);
                w.begin_object();
                w.kv_str("dtype", &spec.dtype);
                w.kv_str("kind", &spec.kind);
                w.kv_uint("len", *n as u64);
                w.kv_uint("off", *off as u64);
                w.key("shape");
                w.begin_array();
                for d in &spec.shape {
                    w.uint(*d as u64);
                }
                w.end_array();
                w.end_object();
            }
            w.end_object();
            if let Some((_, ops)) = &diagnostic {
                w.key("profile_ops");
                w.value(&Json::Array(ops.clone()))
                    .map_err(|e| (e.to_string(), "node"))?;
            }
            w.key("receipt");
            w.begin_object();
            {
                w.key("arena");
                w.begin_object();
                w.kv_uint("used", self.arena.used() as u64);
                w.end_object();
                w.kv_str("backend", "lattice");
                w.key("blob_hashes");
                w.begin_object();
                for (n, h) in &blob_hashes {
                    w.kv_hex16(n, *h);
                }
                w.end_object();
                w.key("input_hashes");
                w.begin_object();
                for (n, h) in &input_hashes {
                    w.kv_hex16(n, *h);
                }
                w.end_object();
                w.kv_str(
                    "kernel",
                    if threads > 1 {
                        "lattice-threaded"
                    } else {
                        "lattice-scalar"
                    },
                );
                w.key("output_hashes");
                w.begin_object();
                for (n, h) in &output_hashes {
                    w.kv_hex16(n, *h);
                }
                w.end_object();
                w.key("partition");
                w.begin_object();
                w.kv_str("kind", "contiguous-output-blocks");
                w.kv_uint("sched_version", crate::runtime::SCHED_VERSION);
                w.kv_uint("threads", threads as u64);
                w.end_object();
                w.kv_str("profile", determinism);
                w.kv_hex16("program_hash", prog.hash);
                if let Some(r) = h.get("reference") {
                    w.kv_value("reference", r)
                        .map_err(|e| (e.to_string(), "node"))?;
                }
                w.kv_str("runtime_version", RUNTIME_VERSION);
                w.kv_uint("sched_version", crate::runtime::SCHED_VERSION);
                w.kv_uint("threads", threads as u64);
                if let Some(t) = h.get("tolerance") {
                    w.kv_value("tolerance", t)
                        .map_err(|e| (e.to_string(), "node"))?;
                }
                w.kv_str("verdict", verdict);
            }
            w.end_object();
            w.kv_str("runtime_version", RUNTIME_VERSION);
            w.kv_uint("threads", threads as u64);
            w.kv_str("type", "exec-ok");
            w.kv_uint("v", PROTO_V);
            w.end_object();
        }
        f.end(self.arena.used_slice())
            .map_err(|e| (e.to_string(), "node"))?;
        Ok(())
    }
}

/// A node running on its own thread. One request at a time models the
/// protocol's one-client-per-server rule; fan-out lives in the controller.
///
/// The request queue is bounded: a full queue applies backpressure to the
/// caller instead of growing without limit. Program/store requests still
/// share the single worker and therefore wait behind a long EXEC; the job
/// server needs immutable `Arc` snapshots for those (next step).
/// One unit of work: the request frame, the reply channel, and the caller's
/// reply buffer (moved, filled by the worker, moved back).
type Job = (Vec<u8>, Sender<Vec<u8>>, Vec<u8>);

#[derive(Default)]
struct WorkerStats {
    allocs: AtomicU64,
    replies: AtomicU64,
}

#[derive(Clone)]
pub struct NodeHandle {
    tx: mpsc::SyncSender<Job>,
    stats: Arc<WorkerStats>,
}

impl NodeHandle {
    /// Default bound for the in-process request queue.
    pub const DEFAULT_QUEUE: usize = 64;

    pub fn spawn(id: usize, allowed: &[&str]) -> Self {
        Self::spawn_bounded(id, allowed, Self::DEFAULT_QUEUE)
    }

    pub fn spawn_bounded(id: usize, allowed: &[&str], capacity: usize) -> Self {
        let allowed: Vec<String> = allowed.iter().map(|s| s.to_string()).collect();
        let (tx, rx) = mpsc::sync_channel::<Job>(capacity.max(1));
        let stats = Arc::new(WorkerStats::default());
        let worker = stats.clone();
        thread::spawn(move || {
            let mut node = Node::new(allowed);
            let _ = id;
            // The worker writes into the buffer the caller sent, so a caller
            // that reuses its buffer gives the worker zero allocations per
            // reply. No pool and no copy: ownership travels with the job.
            while let Ok((frame, reply, mut buf)) = rx.recv() {
                let scope = JobScope::begin();
                node.handle_into(&frame, &mut buf);
                let w = scope.finish();
                worker.allocs.fetch_add(w.allocs, Ordering::Relaxed);
                worker.replies.fetch_add(1, Ordering::Relaxed);
                let _ = reply.send(buf);
            }
        });
        NodeHandle { tx, stats }
    }

    /// Convenience: the reply buffer is allocated here and owned by the
    /// caller, so the worker still does not allocate one.
    pub fn request(&self, frame: Vec<u8>) -> Result<Vec<u8>> {
        let mut out = Vec::new();
        self.request_into(frame, &mut out)?;
        Ok(out)
    }

    /// Send a job and write its reply into `out`, reusing `out`'s capacity.
    ///
    /// The buffer is moved into the worker and back, so a reused `out` makes
    /// the whole round trip allocation-free after the first request.
    pub fn request_into(&self, frame: Vec<u8>, out: &mut Vec<u8>) -> Result<()> {
        let (tx, rx) = mpsc::channel();
        let buf = std::mem::take(out);
        self.tx
            .send((frame, tx, buf))
            .map_err(|_| Error::runtime("node thread is gone"))?;
        *out = rx
            .recv()
            .map_err(|_| Error::runtime("node thread dropped the response"))?;
        Ok(())
    }

    /// Allocations observed on the worker thread, cumulative.
    pub fn worker_allocs(&self) -> u64 {
        self.stats.allocs.load(Ordering::Relaxed)
    }

    /// Replies produced by the worker, cumulative.
    pub fn worker_replies(&self) -> u64 {
        self.stats.replies.load(Ordering::Relaxed)
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
