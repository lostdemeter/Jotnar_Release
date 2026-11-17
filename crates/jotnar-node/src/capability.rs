//! Capability ABI v1 reference types and grant checks.
//!
//! The interpreted module host and the future emitted daemon share this shape.
//! Effects are JSON commands; results/events are JSON envelopes with monotonic
//! sequence numbers assigned by the scheduler.

use serde::{Deserialize, Serialize};
use serde_json::{json, Value as Json};

pub const CAP_ABI: &str = "jotnar.cap/1";

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Grant {
    pub cap: String,
    #[serde(default)]
    pub ops: Vec<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ModuleManifest {
    pub name: String,
    pub program: String,
    #[serde(default)]
    pub seed: Option<String>,
    #[serde(default = "one")]
    pub state_version: u64,
    #[serde(default)]
    pub autostart: bool,
    #[serde(default)]
    pub capabilities: Vec<Grant>,
}

fn one() -> u64 {
    1
}

impl ModuleManifest {
    pub fn granted(&self, cap: &str, op: &str) -> bool {
        self.capabilities
            .iter()
            .any(|g| g.cap == cap && (g.ops.is_empty() || g.ops.iter().any(|o| o == op)))
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Command {
    pub v: u64,
    pub abi: String,
    pub id: u64,
    #[serde(rename = "type")]
    pub typ: String,
    pub module: String,
    pub capability: String,
    pub op: String,
    #[serde(default)]
    pub args: Json,
}

impl Command {
    pub fn new(id: u64, module: &str, capability: &str, op: &str, args: Json) -> Self {
        Self {
            v: 1,
            abi: CAP_ABI.to_string(),
            id,
            typ: "command".to_string(),
            module: module.to_string(),
            capability: capability.to_string(),
            op: op.to_string(),
            args,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CapResult {
    pub v: u64,
    #[serde(rename = "type")]
    pub typ: String,
    pub id: u64,
    pub ok: bool,
    #[serde(default)]
    pub result: Json,
    #[serde(default)]
    pub error: Json,
}

impl CapResult {
    pub fn ok(id: u64, result: Json) -> Self {
        Self {
            v: 1,
            typ: "result".to_string(),
            id,
            ok: true,
            result,
            error: Json::Null,
        }
    }

    pub fn err(id: u64, code: &str, msg: impl Into<String>) -> Self {
        Self {
            v: 1,
            typ: "result".to_string(),
            id,
            ok: false,
            result: Json::Null,
            error: json!({"code": code, "msg": msg.into()}),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CapEvent {
    pub v: u64,
    #[serde(rename = "type")]
    pub typ: String,
    pub seq: u64,
    pub kind: String,
    #[serde(default)]
    pub data: Json,
}

impl CapEvent {
    pub fn new(seq: u64, kind: &str, data: Json) -> Self {
        Self {
            v: 1,
            typ: "event".to_string(),
            seq,
            kind: kind.to_string(),
            data,
        }
    }
}
