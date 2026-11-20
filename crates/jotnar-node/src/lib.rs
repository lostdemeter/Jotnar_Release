//! Shared pieces for the JotnarOS node daemon and payload supervisor.

pub mod capability;
pub mod doom_pipeline;
pub mod executor;
pub mod fleet;
pub mod hardware;
pub mod hud;
pub mod job;
pub mod stream;
/// Controller-side bundle/stage/ledger logic. `jotnar-noded` does not call it.
pub mod workload;

use std::collections::VecDeque;
use std::io::Write;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Mutex;
use std::time::{SystemTime, UNIX_EPOCH};

use serde_json::json;

pub const EVENT_RING: usize = 4096;

pub fn now_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or(0)
}

#[derive(Clone, Debug)]
pub struct Event {
    pub seq: u64,
    pub ts_ms: u128,
    pub kind: String,
    pub stream: Option<String>,
    pub data: Vec<u8>,
}

pub struct EventBus {
    inner: Mutex<Inner>,
    next_seq: AtomicU64,
    dropped: AtomicU64,
}

struct Inner {
    events: VecDeque<Event>,
}

impl Default for EventBus {
    fn default() -> Self {
        Self {
            inner: Mutex::new(Inner {
                events: VecDeque::with_capacity(EVENT_RING),
            }),
            next_seq: AtomicU64::new(1),
            dropped: AtomicU64::new(0),
        }
    }
}

impl EventBus {
    pub fn push(&self, kind: &str, stream: Option<&str>, data: Vec<u8>) -> u64 {
        let seq = self.next_seq.fetch_add(1, Ordering::SeqCst);
        let ev = Event {
            seq,
            ts_ms: now_ms(),
            kind: kind.to_string(),
            stream: stream.map(|s| s.to_string()),
            data,
        };
        let mut inner = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        if inner.events.len() >= EVENT_RING {
            inner.events.pop_front();
            self.dropped.fetch_add(1, Ordering::SeqCst);
        }
        inner.events.push_back(ev);
        seq
    }

    /// Sequence number that the next pushed event will use.
    pub fn head(&self) -> u64 {
        self.next_seq.load(Ordering::SeqCst)
    }

    /// Return events with `seq > after`, filtered by kind, up to `max`.
    pub fn poll(&self, after: u64, max: usize, kinds: &[String]) -> (Vec<Event>, u64, u64) {
        let inner = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        let mut out = Vec::new();
        for ev in inner.events.iter() {
            if ev.seq <= after {
                continue;
            }
            if !kinds.is_empty() && !kinds.iter().any(|k| k == &ev.kind) {
                continue;
            }
            out.push(ev.clone());
            if out.len() >= max {
                break;
            }
        }
        let next = out.last().map(|e| e.seq + 1).unwrap_or(after.max(1));
        let dropped = self.dropped.load(Ordering::SeqCst);
        (out, next, dropped)
    }
}

/// Write payload output to the selected local console(s). `both` is tty0 and
/// ttyS0; `local` is tty0; `serial` is ttyS0; `none` drops.
pub fn write_console(target: &str, data: &[u8]) {
    let paths: &[&str] = match target {
        "both" => &["/dev/tty0", "/dev/ttyS0"],
        "local" => &["/dev/tty0"],
        "serial" => &["/dev/ttyS0"],
        _ => &[],
    };
    if paths.is_empty() || data.is_empty() {
        return;
    }
    // VT consoles need CR before LF; serial terminals generally accept either.
    let mut buf = Vec::with_capacity(data.len() + 8);
    for &b in data {
        if b == b'\n' {
            buf.push(b'\r');
        }
        buf.push(b);
    }
    for p in paths {
        if let Ok(mut f) = std::fs::OpenOptions::new().write(true).append(true).open(p) {
            let _ = f.write_all(&buf);
        }
    }
}

pub fn error_frame(id: &serde_json::Value, code: &str, msg: impl Into<String>) -> Vec<u8> {
    jotnar::protocol::error_frame(id, code, msg)
}

pub fn ok_frame(
    id: &serde_json::Value,
    typ: &str,
    fields: serde_json::Value,
    payload: &[u8],
) -> Vec<u8> {
    let mut h = fields;
    if let Some(o) = h.as_object_mut() {
        o.insert("v".into(), json!(jotnar::protocol::PROTO_V));
        o.insert("id".into(), id.clone());
        o.insert("type".into(), json!(typ));
        o.insert("ok".into(), json!(true));
    }
    jotnar::protocol::encode_frame(&h, payload).unwrap_or_else(|_| {
        jotnar::protocol::error_frame(id, "encode", "could not encode response")
    })
}

/// A small shared liveness flag for a payload thread.
#[derive(Clone, Default)]
pub struct Alive(pub std::sync::Arc<AtomicBool>);

impl Alive {
    pub fn set(&self, v: bool) {
        self.0.store(v, Ordering::SeqCst);
    }
    pub fn get(&self) -> bool {
        self.0.load(Ordering::SeqCst)
    }
}
