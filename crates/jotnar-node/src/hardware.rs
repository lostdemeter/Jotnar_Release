//! Normalized hardware inventory for the workload controller.
//!
//! Discovery records answer "who is out there"; this module turns them into a
//! stable, typed view of what each node offers. It is deliberately not a VM or
//! container abstraction: a node is a set of typed resources (CPU, GPU,
//! programs, queue limits, links) that a placement policy can reason about.
//!
//! The first slice normalizes the existing CPU, GPU, and native advertisements
//! and can probe TCP reachability/RTT between nodes. Measured throughput and
//! live GPU utilization are filled in by the executors later; the schema keeps
//! room for them.

use std::collections::BTreeMap;
use std::net::{TcpStream, ToSocketAddrs};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};
use serde_json::Value as Json;

pub const HARDWARE_SCHEMA: &str = "jotnar.hardware/1";

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HardwareInventory {
    pub schema: String,
    pub generated_ms: u64,
    pub nodes: Vec<NodeProfile>,
    pub links: Vec<LinkProfile>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NodeProfile {
    pub node_id: String,
    pub kind: String,
    pub addrs: Vec<String>,
    pub targets: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cpu: Option<CpuProfile>,
    #[serde(default)]
    pub gpus: Vec<GpuProfile>,
    #[serde(default)]
    pub programs: Vec<String>,
    pub limits: Limits,
    pub telemetry: Telemetry,
    pub runtime_version: String,
    pub auth: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CpuProfile {
    pub arch: String,
    pub workers: Option<u64>,
    pub queue: Option<u64>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GpuProfile {
    pub target: String,
    pub arch: String,
    pub toolchain: String,
    pub pid: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, Default)]
pub struct Limits {
    pub workers: Option<u64>,
    pub queue: Option<u64>,
    pub max_frame: Option<u64>,
}

#[derive(Clone, Debug, Serialize, Deserialize, Default)]
pub struct Telemetry {
    pub queued: Option<u64>,
    pub running: Option<u64>,
    pub warm_artifacts: Option<u64>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LinkProfile {
    pub from_node: String,
    pub to_node: String,
    pub from: String,
    pub to: String,
    pub reachable: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rtt_us: Option<u64>,
}

impl NodeProfile {
    pub fn endpoint(&self) -> Option<&str> {
        self.addrs
            .iter()
            .find(|a| !a.starts_with("127."))
            .or_else(|| self.addrs.first())
            .map(|s| s.as_str())
    }
}

impl HardwareInventory {
    pub fn from_capabilities(caps: &[Json]) -> Self {
        let mut nodes = caps
            .iter()
            .filter_map(node_from_capability)
            .collect::<Vec<_>>();
        nodes.sort_by(|a, b| {
            (a.kind.as_str(), a.node_id.as_str()).cmp(&(b.kind.as_str(), b.node_id.as_str()))
        });
        Self {
            schema: HARDWARE_SCHEMA.to_string(),
            generated_ms: now_ms(),
            nodes,
            links: Vec::new(),
        }
    }

    /// Probe each node pair once over TCP. RTT is connect time, not a
    /// protocol ping; it is a placement hint, not a performance claim.
    pub fn probe_links(&mut self, timeout: Duration) {
        let mut links = Vec::new();
        for i in 0..self.nodes.len() {
            for j in (i + 1)..self.nodes.len() {
                let Some(a) = self.nodes[i].endpoint() else {
                    continue;
                };
                let Some(b) = self.nodes[j].endpoint() else {
                    continue;
                };
                let (rtt, reachable) = match tcp_rtt(a, timeout) {
                    Some(ms) => (Some(ms), true),
                    None => (None, false),
                };
                links.push(LinkProfile {
                    from_node: self.nodes[i].node_id.clone(),
                    to_node: self.nodes[j].node_id.clone(),
                    from: a.to_string(),
                    to: b.to_string(),
                    reachable,
                    rtt_us: rtt,
                });
            }
        }
        links.sort_by(|a, b| {
            (a.from.as_str(), a.to.as_str()).cmp(&(b.from.as_str(), b.to.as_str()))
        });
        self.links = links;
    }

    pub fn by_kind(&self, kind: &str) -> Vec<&NodeProfile> {
        self.nodes.iter().filter(|n| n.kind == kind).collect()
    }

    pub fn summary(&self) -> Json {
        let mut counts = BTreeMap::new();
        for node in &self.nodes {
            *counts.entry(node.kind.clone()).or_insert(0u64) += 1;
        }
        serde_json::json!({
            "schema": self.schema,
            "nodes": self.nodes.len(),
            "kinds": counts,
            "links": self.links.len(),
            "reachable_links": self.links.iter().filter(|l| l.reachable).count(),
        })
    }
}

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

fn string(v: &Json, key: &str) -> String {
    v.get(key)
        .and_then(|x| x.as_str())
        .unwrap_or("")
        .to_string()
}

fn number(v: &Json, key: &str) -> Option<u64> {
    v.get(key).and_then(|x| {
        x.as_u64()
            .or_else(|| x.as_str().and_then(|s| s.parse::<u64>().ok()))
    })
}

fn strings(v: &Json, key: &str) -> Vec<String> {
    v.get(key)
        .and_then(|x| x.as_array())
        .map(|a| {
            a.iter()
                .filter_map(|x| x.as_str())
                .map(|s| s.to_string())
                .collect()
        })
        .unwrap_or_default()
}

fn csv(v: &Json, key: &str) -> Vec<String> {
    string(v, key)
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(|s| s.to_string())
        .collect()
}

fn node_from_capability(cap: &Json) -> Option<NodeProfile> {
    let kind = string(cap, "kind");
    if kind.is_empty() {
        return None;
    }
    let addrs = strings(cap, "addrs");
    let instance = string(cap, "instance");
    let node_id = {
        let id = string(cap, "node_id");
        if !id.is_empty() {
            id
        } else if !instance.is_empty() {
            instance
        } else {
            addrs.first().cloned().unwrap_or_else(|| kind.clone())
        }
    };
    let target = string(cap, "target");
    let mut targets = Vec::new();
    if !target.is_empty() {
        targets.push(target.clone());
    }
    let workers = number(cap, "workers");
    let queue = number(cap, "queue");
    let cpu = (kind == "cpu").then(|| CpuProfile {
        arch: if target.is_empty() {
            "cpu".to_string()
        } else {
            target.clone()
        },
        workers,
        queue,
    });
    let gpus = if kind == "gpu" {
        vec![GpuProfile {
            target,
            arch: string(cap, "arch"),
            toolchain: string(cap, "toolchain"),
            pid: string(cap, "pid"),
        }]
    } else {
        Vec::new()
    };
    Some(NodeProfile {
        node_id,
        kind,
        addrs,
        targets,
        cpu,
        gpus,
        programs: csv(cap, "programs"),
        limits: Limits {
            workers,
            queue,
            max_frame: number(cap, "max_frame"),
        },
        telemetry: Telemetry {
            queued: number(cap, "queued"),
            running: number(cap, "running"),
            warm_artifacts: number(cap, "warm_artifacts"),
        },
        runtime_version: string(cap, "runtime_version"),
        auth: string(cap, "auth"),
    })
}

fn tcp_rtt(addr: &str, timeout: Duration) -> Option<u64> {
    let sock = addr.to_socket_addrs().ok()?.next()?;
    let t = Instant::now();
    TcpStream::connect_timeout(&sock, timeout).ok()?;
    Some(t.elapsed().as_micros() as u64)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn caps() -> Vec<Json> {
        vec![
            json!({
                "kind": "cpu", "target": "cpu", "node_id": "cpu-1",
                "addrs": ["127.0.0.1:9999"], "workers": "8", "queue": "64",
                "auth": "required", "runtime_version": "0.1.0+x"
            }),
            json!({
                "kind": "gpu", "target": "cuda", "instance": "Jotnar GPU cuda 1",
                "addrs": ["192.168.1.116:9997"], "arch": "sm_120",
                "toolchain": "nvcc", "pid": "1", "auth": "required",
                "runtime_version": "0.1.0+x"
            }),
            json!({
                "kind": "native", "target": "process", "instance": "Jotnar Native 2",
                "addrs": ["127.0.0.1:9998"], "programs": "jotnar-demo-doom",
                "platform": "linux-x86_64", "auth": "required",
                "runtime_version": "0.1.0+x"
            }),
        ]
    }

    #[test]
    fn capabilities_become_typed_nodes() {
        let inv = HardwareInventory::from_capabilities(&caps());
        assert_eq!(inv.nodes.len(), 3);
        let cpu = &inv.by_kind("cpu")[0];
        assert_eq!(cpu.node_id, "cpu-1");
        assert_eq!(cpu.cpu.as_ref().unwrap().workers, Some(8));
        let gpu = &inv.by_kind("gpu")[0];
        assert_eq!(gpu.gpus[0].target, "cuda");
        assert_eq!(gpu.gpus[0].arch, "sm_120");
        let native = &inv.by_kind("native")[0];
        assert_eq!(native.programs, vec!["jotnar-demo-doom"]);
        assert_eq!(native.targets, vec!["process"]);
    }

    #[test]
    fn summary_counts_kinds() {
        let inv = HardwareInventory::from_capabilities(&caps());
        let s = inv.summary();
        assert_eq!(s["nodes"], 3);
        assert_eq!(s["kinds"]["cpu"], 1);
        assert_eq!(s["kinds"]["gpu"], 1);
        assert_eq!(s["kinds"]["native"], 1);
    }

    #[test]
    fn missing_kind_is_skipped() {
        let inv = HardwareInventory::from_capabilities(&[json!({"addrs": ["x:1"]})]);
        assert!(inv.nodes.is_empty());
    }
}
