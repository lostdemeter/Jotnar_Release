//! N3 scheduler core: health, placement, submission, retries, aggregation.
//!
//! The scheduler is a controller-side component. It owns the workload name,
//! placement, retry ledger, and result aggregation; nodes remain model-blind.
//! This is an N3 v0: static node list, `ping` health, least-loaded placement
//! with a deterministic tie-break, per-job idempotency keys, and retry on
//! connection/terminal failure. Background heartbeats and discovery follow.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::io::{Read, Write};
use std::net::TcpStream;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use jotnar::payload::Feeds;
use jotnar::protocol::{decode_frame, encode_stream, request};
use jotnar::value::hash_bytes;
use mdns_sd::{ServiceDaemon, ServiceEvent};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value as Json};
use sha2::{Digest, Sha256};

const MAX_FRAME: usize = 512 * 1024 * 1024;

#[derive(Clone, Debug)]
pub struct NodeSpec {
    pub addr: String,
    pub token: Option<String>,
}

#[derive(Clone, Debug)]
pub struct BatchSpec {
    pub handle: String,
    pub feeds: Feeds,
    pub outputs: Vec<String>,
    pub count: usize,
    pub threads: Option<usize>,
    pub timeout_ms: Option<u64>,
    pub wait_ms: u64,
    pub poll_ms: u64,
    pub max_attempts: usize,
    pub batch_id: String,
    pub heartbeat_ms: u64,
    pub down_after: u32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NodeSummary {
    pub addr: String,
    pub healthy: bool,
    pub workers: usize,
    pub queue_capacity: usize,
    pub queued: usize,
    pub draining: bool,
    pub failure_streak: u32,
    pub last_good_age_ms: Option<u64>,
    pub done: usize,
    pub bytes: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RunSummary {
    pub nodes: Vec<NodeSummary>,
    pub submitted: usize,
    pub done: usize,
    pub failed: usize,
    pub retried: usize,
    pub replayed: usize,
    pub output_bytes: usize,
    pub output_hash: Option<String>,
    pub identical_outputs: bool,
    pub wall_ms: u64,
    pub jobs_per_s: f64,
    #[serde(default)]
    pub cached: bool,
    #[serde(default)]
    pub resumed: bool,
    #[serde(skip)]
    pub receipts: Vec<Json>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
struct LedgerSpec {
    handle: String,
    outputs: Vec<String>,
    count: usize,
    threads: Option<usize>,
    timeout_ms: Option<u64>,
    batch_id: String,
    payload_hash: String,
    program_hash: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct LedgerJob {
    index: usize,
    attempts: usize,
    tried: Vec<String>,
    node: Option<String>,
    job_id: Option<String>,
    state: String,
    bytes: usize,
    hash: Option<String>,
    error: Option<String>,
    receipt: Option<Json>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct Ledger {
    v: u32,
    batch_id: String,
    spec: LedgerSpec,
    created_ms: u64,
    updated_ms: u64,
    complete: bool,
    jobs: Vec<LedgerJob>,
    summary: Option<RunSummary>,
}

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

/// Stable ledger file name for a batch id.
pub fn batch_ledger_path(dir: &Path, batch_id: &str) -> PathBuf {
    let safe: String = batch_id
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '-' || c == '_' {
                c
            } else {
                '_'
            }
        })
        .collect();
    dir.join(format!("{safe}.json"))
}

fn ledger_spec(spec: &BatchSpec, payload_hash: &str, program_hash: &str) -> LedgerSpec {
    LedgerSpec {
        handle: spec.handle.clone(),
        outputs: spec.outputs.clone(),
        count: spec.count,
        threads: spec.threads,
        timeout_ms: spec.timeout_ms,
        batch_id: spec.batch_id.clone(),
        payload_hash: payload_hash.to_string(),
        program_hash: program_hash.to_string(),
    }
}

fn ledger_load(path: &Path, spec: &LedgerSpec) -> Result<Option<Ledger>, String> {
    if !path.exists() {
        return Ok(None);
    }
    let text = std::fs::read(path).map_err(|e| format!("read {}: {e}", path.display()))?;
    let ledger: Ledger =
        serde_json::from_slice(&text).map_err(|e| format!("parse {}: {e}", path.display()))?;
    if ledger.spec != *spec {
        return Err(format!(
            "batch-id-conflict: {} already exists with a different payload/program/spec",
            path.display()
        ));
    }
    Ok(Some(ledger))
}

fn ledger_save(
    path: &Path,
    spec: &LedgerSpec,
    created_ms: u64,
    jobs: &[Job],
    nodes: &[Node],
    complete: bool,
    summary: Option<&RunSummary>,
) -> Result<(), String> {
    let ledger_jobs: Vec<LedgerJob> = jobs
        .iter()
        .map(|job| LedgerJob {
            index: job.index,
            attempts: job.attempts,
            tried: job
                .tried
                .iter()
                .filter_map(|i| nodes.get(*i).map(|n| n.spec.addr.clone()))
                .collect(),
            node: job
                .node
                .and_then(|i| nodes.get(i).map(|n| n.spec.addr.clone())),
            job_id: job.job_id.clone(),
            state: job.state.clone(),
            bytes: job.bytes,
            hash: job.hash.clone(),
            error: job.error.clone(),
            receipt: job.receipt.clone(),
        })
        .collect();
    let ledger = Ledger {
        v: 1,
        batch_id: spec.batch_id.clone(),
        spec: spec.clone(),
        created_ms,
        updated_ms: now_ms(),
        complete,
        jobs: ledger_jobs,
        summary: summary.cloned(),
    };
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|e| format!("create {}: {e}", parent.display()))?;
    }
    let tmp = path.with_extension("json.tmp");
    let bytes = serde_json::to_vec_pretty(&ledger).map_err(|e| format!("serialize ledger: {e}"))?;
    std::fs::write(&tmp, bytes).map_err(|e| format!("write {}: {e}", tmp.display()))?;
    std::fs::rename(&tmp, path).map_err(|e| format!("rename {}: {e}", path.display()))?;
    Ok(())
}

fn jobs_from_ledger(ledger: &Ledger, nodes: &[Node], spec: &BatchSpec) -> Vec<Job> {
    ledger
        .jobs
        .iter()
        .map(|lj| {
            let node = lj
                .node
                .as_ref()
                .and_then(|addr| nodes.iter().position(|n| &n.spec.addr == addr));
            let tried: HashSet<usize> = lj
                .tried
                .iter()
                .filter_map(|addr| nodes.iter().position(|n| &n.spec.addr == addr))
                .collect();
            let mut state = lj.state.clone();
            if matches!(state.as_str(), "failed" | "canceled" | "timed_out")
                && lj.attempts < spec.max_attempts
            {
                state = "pending".to_string();
            }
            if state != "done" && node.is_none() && lj.hash.is_none() {
                state = "pending".to_string();
            }
            let job_id = if state == "pending" {
                None
            } else {
                lj.job_id.clone()
            };
            Job {
                index: lj.index,
                attempts: lj.attempts,
                tried,
                node: if state == "pending" { None } else { node },
                job_id,
                state,
                bytes: lj.bytes,
                hash: lj.hash.clone(),
                receipt: lj.receipt.clone(),
                error: lj.error.clone(),
            }
        })
        .collect()
}

struct Client {
    spec: NodeSpec,
    stream: TcpStream,
    next_id: u64,
}

impl Client {
    fn connect(spec: &NodeSpec) -> Result<Self, String> {
        let stream =
            TcpStream::connect(&spec.addr).map_err(|e| format!("connect {}: {e}", spec.addr))?;
        let _ = stream.set_nodelay(true);
        let _ = stream.set_read_timeout(Some(Duration::from_secs(20)));
        let _ = stream.set_write_timeout(Some(Duration::from_secs(20)));
        Ok(Self {
            spec: spec.clone(),
            stream,
            next_id: 1,
        })
    }

    fn req(&mut self, typ: &str, fields: Json, payload: &[u8]) -> Result<(Json, Vec<u8>), String> {
        let id = self.next_id;
        self.next_id += 1;
        let mut fields = fields;
        if let Some(token) = &self.spec.token {
            if let Some(o) = fields.as_object_mut() {
                o.insert("auth".into(), json!(token));
            }
        }
        let frame = request(id, typ, fields, payload).map_err(|e| e.to_string())?;
        self.stream
            .write_all(&frame)
            .map_err(|e| format!("write {}: {e}", self.spec.addr))?;
        let mut hdr = [0u8; 4];
        self.stream
            .read_exact(&mut hdr)
            .map_err(|e| format!("read {}: {e}", self.spec.addr))?;
        let n = u32::from_be_bytes(hdr) as usize;
        if n > MAX_FRAME {
            return Err(format!("response {n} exceeds {MAX_FRAME}"));
        }
        let mut body = vec![0u8; n];
        self.stream
            .read_exact(&mut body)
            .map_err(|e| format!("read {}: {e}", self.spec.addr))?;
        let mut full = Vec::with_capacity(4 + n);
        full.extend_from_slice(&hdr);
        full.extend_from_slice(&body);
        let (header, payload) = decode_frame(&full).map_err(|e| e.to_string())?;
        if header.get("id").and_then(|x| x.as_u64()) != Some(id) {
            return Err(format!("response id mismatch on {}", self.spec.addr));
        }
        Ok((header, payload.to_vec()))
    }
}

struct Node {
    spec: NodeSpec,
    client: Option<Client>,
    healthy: bool,
    workers: usize,
    queue_capacity: usize,
    queued: usize,
    draining: bool,
    last_error: Option<String>,
    active: usize,
    done: usize,
    bytes: usize,
    program_uploaded: bool,
}

impl Node {
    fn mark_down(&mut self, err: impl Into<String>) {
        self.healthy = false;
        self.client = None;
        self.last_error = Some(err.into());
    }
}

#[derive(Clone, Debug, Default)]
struct HealthState {
    healthy: bool,
    workers: usize,
    queue_capacity: usize,
    queued: usize,
    running: usize,
    draining: bool,
    failure_streak: u32,
    last_ok: Option<Instant>,
    last_error: Option<String>,
}

impl HealthState {
    fn new() -> Self {
        Self {
            workers: 1,
            draining: true,
            ..Self::default()
        }
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct NodeHealth {
    pub addr: String,
    pub healthy: bool,
    pub workers: usize,
    pub queue_capacity: usize,
    pub queued: usize,
    pub running: usize,
    pub draining: bool,
    pub failure_streak: u32,
    pub last_good_age_ms: Option<u64>,
    pub last_error: Option<String>,
}

fn ping_node(spec: &NodeSpec) -> Result<(usize, usize, usize, usize), String> {
    let mut client = Client::connect(spec)?;
    let (h, _) = client.req("ping", json!({}), &[])?;
    if h.get("ok").and_then(|x| x.as_bool()) != Some(true) {
        return Err(format!("ping failed: {h}"));
    }
    let jobs = h
        .get("pong")
        .and_then(|p| p.get("jobs"))
        .ok_or_else(|| "node has no job API".to_string())?;
    let n = |k: &str| jobs.get(k).and_then(|x| x.as_u64()).unwrap_or(0) as usize;
    let queued = match jobs.get("queued").and_then(|x| x.as_u64()) {
        Some(q) => q as usize,
        None => {
            // Older N1/N2a images do not advertise live queue depth; fall back
            // to the job-list stats so the scheduler can still drain.
            let (list, _) = client.req("job-list", json!({}), &[])?;
            list.get("stats")
                .and_then(|s| s.get("queued"))
                .and_then(|x| x.as_u64())
                .unwrap_or(0) as usize
        }
    };
    Ok((
        n("workers").max(1),
        n("queue_capacity"),
        queued,
        n("running"),
    ))
}

fn record_health(
    h: &mut HealthState,
    result: Result<(usize, usize, usize, usize), String>,
    down_after: u32,
) {
    match result {
        Ok((workers, queue_capacity, queued, running)) => {
            h.healthy = true;
            h.workers = workers;
            h.queue_capacity = queue_capacity;
            h.queued = queued;
            h.running = running;
            h.draining = queue_capacity > 0 && queued >= queue_capacity;
            h.failure_streak = 0;
            h.last_ok = Some(Instant::now());
            h.last_error = None;
        }
        Err(e) => {
            h.failure_streak = h.failure_streak.saturating_add(1);
            h.healthy = h.failure_streak < down_after.max(1);
            if !h.healthy {
                h.draining = true;
            }
            h.last_error = Some(e);
        }
    }
}

fn ping_into(state: &Arc<Mutex<HashMap<String, HealthState>>>, spec: &NodeSpec, down_after: u32) {
    let result = ping_node(spec);
    let mut map = state.lock().unwrap();
    let h = map
        .entry(spec.addr.clone())
        .or_insert_with(HealthState::new);
    record_health(h, result, down_after);
}

/// Background heartbeat registry: periodic `ping`, failure streak, last-good,
/// queue depth, and draining state.
pub struct Registry {
    specs: Vec<NodeSpec>,
    state: Arc<Mutex<HashMap<String, HealthState>>>,
    stop: Arc<AtomicBool>,
    thread: Option<JoinHandle<()>>,
    down_after: u32,
}

impl Registry {
    pub fn start(specs: &[NodeSpec], interval: Duration, down_after: u32) -> Self {
        let mut map = HashMap::new();
        for spec in specs {
            map.insert(spec.addr.clone(), HealthState::new());
        }
        let state = Arc::new(Mutex::new(map));
        let stop = Arc::new(AtomicBool::new(false));
        let mut registry = Registry {
            specs: specs.to_vec(),
            state: state.clone(),
            stop: stop.clone(),
            thread: None,
            down_after: down_after.max(1),
        };
        registry.refresh_once();
        let specs = registry.specs.clone();
        let state2 = state.clone();
        let stop2 = stop.clone();
        let interval = interval.max(Duration::from_millis(10));
        let down_after = registry.down_after;
        registry.thread = thread::Builder::new()
            .name("jotnar-heartbeat".to_string())
            .spawn(move || {
                while !stop2.load(Ordering::SeqCst) {
                    for spec in &specs {
                        ping_into(&state2, spec, down_after);
                    }
                    let end = Instant::now() + interval;
                    while !stop2.load(Ordering::SeqCst) {
                        let remaining = end.saturating_duration_since(Instant::now());
                        if remaining.is_zero() {
                            break;
                        }
                        thread::sleep(remaining.min(Duration::from_millis(50)));
                    }
                }
            })
            .ok();
        registry
    }

    pub fn refresh_once(&self) {
        for spec in &self.specs {
            ping_into(&self.state, spec, self.down_after);
        }
    }

    pub fn snapshot(&self) -> Vec<NodeHealth> {
        let map = self.state.lock().unwrap();
        self.specs
            .iter()
            .map(|spec| {
                let h = map
                    .get(&spec.addr)
                    .cloned()
                    .unwrap_or_else(HealthState::new);
                NodeHealth {
                    addr: spec.addr.clone(),
                    healthy: h.healthy,
                    workers: h.workers,
                    queue_capacity: h.queue_capacity,
                    queued: h.queued,
                    running: h.running,
                    draining: h.draining,
                    failure_streak: h.failure_streak,
                    last_good_age_ms: h.last_ok.map(|t| t.elapsed().as_millis() as u64),
                    last_error: h.last_error,
                }
            })
            .collect()
    }

    pub fn stop(&mut self) {
        self.stop.store(true, Ordering::SeqCst);
        if let Some(handle) = self.thread.take() {
            let _ = handle.join();
        }
    }
}

impl Drop for Registry {
    fn drop(&mut self) {
        self.stop();
    }
}

struct Job {
    index: usize,
    attempts: usize,
    tried: HashSet<usize>,
    node: Option<usize>,
    job_id: Option<String>,
    state: String,
    bytes: usize,
    hash: Option<String>,
    receipt: Option<Json>,
    error: Option<String>,
}

fn sha256_hex(bytes: &[u8]) -> String {
    let mut h = Sha256::new();
    h.update(bytes);
    format!("{:x}", h.finalize())
}

fn body_from_feeds(feeds: &Feeds) -> Result<(Json, Vec<u8>), String> {
    let mut inputs = serde_json::Map::new();
    let mut body = Vec::new();
    for (name, v) in feeds {
        let (spec, bytes) = encode_stream(v).map_err(|e| e.to_string())?;
        let off = body.len();
        body.extend_from_slice(&bytes);
        let mut sj = spec.json();
        if let Some(o) = sj.as_object_mut() {
            o.insert("off".into(), json!(off));
            o.insert("len".into(), json!(bytes.len()));
        }
        inputs.insert(name.clone(), sj);
    }
    Ok((Json::Object(inputs), body))
}

fn pick_node(nodes: &[Node], job: &Job, batch_id: &str) -> Option<usize> {
    let healthy: Vec<usize> = nodes
        .iter()
        .enumerate()
        .filter(|(i, n)| n.healthy && !n.draining && !job.tried.contains(i))
        .map(|(i, _)| i)
        .collect();
    let pool = if healthy.is_empty() {
        (0..nodes.len())
            .filter(|i| nodes[*i].healthy && !nodes[*i].draining)
            .collect::<Vec<_>>()
    } else {
        healthy
    };
    pool.into_iter().min_by_key(|i| {
        let n = &nodes[*i];
        let load = (n.active as u128 * 1000) / n.workers.max(1) as u128;
        let tie = hash_bytes(format!("{batch_id}:{}:{}", job.index, n.spec.addr).as_bytes());
        (load, tie)
    })
}

fn requeue_node(nodes: &mut [Node], jobs: &mut [Job], n: usize, err: impl Into<String>) -> usize {
    nodes[n].mark_down(err);
    nodes[n].active = 0;
    let mut requeued = 0;
    for job in jobs
        .iter_mut()
        .filter(|j| j.node == Some(n) && j.hash.is_none())
    {
        job.state = "pending".to_string();
        job.node = None;
        job.job_id = None;
        requeued += 1;
    }
    requeued
}

fn stored_meta(h: &Json) -> Option<(String, usize, usize, usize, String, Vec<String>)> {
    let result = h.get("result")?;
    if result.get("transport").and_then(|x| x.as_str()) != Some("store") {
        return None;
    }
    let s = |k: &str| result.get(k).and_then(|x| x.as_str()).map(str::to_string);
    let n = |k: &str| result.get(k).and_then(|x| x.as_u64()).map(|v| v as usize);
    let chunk_hashes = result
        .get("chunk_hashes")?
        .as_array()?
        .iter()
        .filter_map(|v| v.as_str().map(str::to_string))
        .collect();
    Some((
        s("id")?,
        n("bytes")?,
        n("chunk_bytes")?,
        n("chunks")?,
        s("hash")?,
        chunk_hashes,
    ))
}

fn fetch_result(node: &mut Node, job_id: &str) -> Result<(usize, String, Option<Json>), String> {
    let client = node.client.as_mut().ok_or("node has no connection")?;
    let (h, payload) = client.req("job-result", json!({"job_id": job_id}), &[])?;
    if h.get("ok").and_then(|x| x.as_bool()) != Some(true) {
        return Err(format!(
            "job-result failed: {}",
            h.get("error").cloned().unwrap_or(json!("?"))
        ));
    }
    let receipt = h.get("receipt").cloned();
    if let Some((id, bytes, chunk_bytes, chunks, hash, chunk_hashes)) = stored_meta(&h) {
        let mut all = Vec::with_capacity(bytes);
        for i in 0..chunks {
            let off = i * chunk_bytes;
            let len = (bytes - off).min(chunk_bytes);
            let (ch, chunk) = client.req(
                "result-read",
                json!({"result_id": id, "off": off, "len": len}),
                &[],
            )?;
            if ch.get("ok").and_then(|x| x.as_bool()) != Some(true) {
                return Err(format!(
                    "result-read failed: {}",
                    ch.get("error").cloned().unwrap_or(json!("?"))
                ));
            }
            let got = sha256_hex(&chunk);
            if ch["hash"].as_str() != Some(got.as_str()) {
                return Err(format!("chunk {i} hash mismatch"));
            }
            if chunk_hashes.get(i) != Some(&got) {
                return Err(format!("chunk {i} manifest mismatch"));
            }
            all.extend_from_slice(&chunk);
        }
        if sha256_hex(&all) != hash {
            return Err("result hash mismatch".to_string());
        }
        Ok((bytes, hash, receipt))
    } else {
        let hash = h
            .get("result")
            .and_then(|r| r.get("hash"))
            .and_then(|x| x.as_str())
            .map(str::to_string)
            .unwrap_or_else(|| sha256_hex(&payload));
        if sha256_hex(&payload) != hash {
            return Err("inline result hash mismatch".to_string());
        }
        Ok((payload.len(), hash, receipt))
    }
}

fn connect_nodes(specs: &[NodeSpec], health: &[NodeHealth]) -> Vec<Node> {
    let mut nodes = Vec::with_capacity(specs.len());
    for spec in specs {
        let h = health.iter().find(|h| h.addr == spec.addr);
        let mut node = Node {
            spec: spec.clone(),
            client: None,
            healthy: false,
            workers: h.map(|h| h.workers).unwrap_or(1),
            queue_capacity: h.map(|h| h.queue_capacity).unwrap_or(0),
            queued: h.map(|h| h.queued).unwrap_or(0),
            draining: h.map(|h| h.draining).unwrap_or(true),
            last_error: h.and_then(|h| h.last_error.clone()),
            active: 0,
            done: 0,
            bytes: 0,
            program_uploaded: false,
        };
        if h.map(|h| h.healthy).unwrap_or(false) {
            match Client::connect(spec) {
                Ok(client) => {
                    node.client = Some(client);
                    node.healthy = true;
                }
                Err(e) => node.last_error = Some(e),
            }
        }
        nodes.push(node);
    }
    nodes
}

pub fn health(specs: &[NodeSpec], heartbeat_ms: u64, down_after: u32) -> Vec<Json> {
    let registry = Registry::start(
        specs,
        Duration::from_millis(heartbeat_ms.max(1)),
        down_after.max(1),
    );
    registry
        .snapshot()
        .into_iter()
        .map(|h| {
            json!({
                "addr": h.addr,
                "healthy": h.healthy,
                "workers": h.workers,
                "queue_capacity": h.queue_capacity,
                "queued": h.queued,
                "running": h.running,
                "draining": h.draining,
                "failure_streak": h.failure_streak,
                "last_good_age_ms": h.last_good_age_ms,
                "error": h.last_error,
            })
        })
        .collect()
}

/// Browse `_jotnar-node._tcp.local.` for a bounded time. Discovery is
/// informational: the controller still needs a token or an explicit trust
/// decision before submitting work.
pub fn discover(timeout: Duration) -> Result<Vec<Json>, String> {
    let daemon = ServiceDaemon::new().map_err(|e| e.to_string())?;
    let receiver = daemon
        .browse("_jotnar-node._tcp.local.")
        .map_err(|e| e.to_string())?;
    let deadline = Instant::now() + timeout;
    let mut found: BTreeMap<String, Json> = BTreeMap::new();
    loop {
        let remaining = deadline.saturating_duration_since(Instant::now());
        if remaining.is_zero() {
            break;
        }
        let event = match receiver.recv_timeout(remaining) {
            Ok(event) => event,
            Err(_) => break,
        };
        if let ServiceEvent::ServiceResolved(info) = event {
            let port = info.get_port();
            let addrs: Vec<String> = info
                .get_addresses_v4()
                .iter()
                .map(|ip| format!("{ip}:{port}"))
                .collect();
            let txt = |key: &str| {
                info.get_property_val_str(key)
                    .map(|s| s.to_string())
                    .unwrap_or_default()
            };
            let entry = json!({
                "fullname": info.get_fullname(),
                "instance": info.get_fullname().split('.').next().unwrap_or(""),
                "node_id": txt("node_id"),
                "addrs": addrs,
                "port": port,
                "caps": txt("caps"),
                "workers": txt("workers"),
                "queue": txt("queue"),
                "auth": txt("auth"),
                "result_transport": txt("result_transport"),
                "runtime_version": txt("runtime_version"),
            });
            found.insert(info.get_fullname().to_string(), entry);
        }
    }
    let _ = daemon.shutdown();
    Ok(found.into_values().collect())
}

fn sync_health(nodes: &mut [Node], jobs: &mut [Job], health: &[NodeHealth]) -> usize {
    let mut retried = 0usize;
    for n in 0..nodes.len() {
        let Some(h) = health
            .iter()
            .find(|h| h.addr == nodes[n].spec.addr)
            .cloned()
        else {
            continue;
        };
        nodes[n].workers = h.workers;
        nodes[n].queue_capacity = h.queue_capacity;
        nodes[n].queued = h.queued;
        if !h.healthy {
            if nodes[n].healthy {
                retried += requeue_node(
                    nodes,
                    jobs,
                    n,
                    h.last_error
                        .clone()
                        .unwrap_or_else(|| "heartbeat lost".to_string()),
                );
            }
            nodes[n].healthy = false;
            nodes[n].draining = true;
        } else {
            nodes[n].draining = h.draining;
            if !nodes[n].healthy || nodes[n].client.is_none() {
                let spec = nodes[n].spec.clone();
                match Client::connect(&spec) {
                    Ok(client) => {
                        nodes[n].client = Some(client);
                        nodes[n].healthy = true;
                        nodes[n].last_error = None;
                        nodes[n].program_uploaded = false;
                    }
                    Err(e) => {
                        nodes[n].healthy = false;
                        nodes[n].last_error = Some(e);
                    }
                }
            }
        }
    }
    retried
}

fn submit_job(
    nodes: &mut [Node],
    job: &mut Job,
    inputs: &Json,
    body: &[u8],
    spec: &BatchSpec,
) -> Result<(Option<usize>, bool), String> {
    if job.attempts >= spec.max_attempts {
        job.state = "failed".to_string();
        job.error = Some("retry budget exhausted".to_string());
        return Ok((None, false));
    }
    let Some(n) = pick_node(nodes, job, &spec.batch_id) else {
        if nodes.iter().any(|n| n.healthy) {
            // All healthy nodes are draining; wait for heartbeat to clear.
            job.state = "pending".to_string();
        } else {
            job.state = "failed".to_string();
            job.error = Some("no healthy node".to_string());
        }
        return Ok((None, false));
    };
    let mut fields = json!({
        "handle": spec.handle,
        "inputs": inputs,
        "outputs": spec.outputs,
        "idempotency_key": format!("{}/{}", spec.batch_id, job.index),
    });
    if let Some(o) = fields.as_object_mut() {
        if let Some(t) = spec.threads {
            o.insert("threads".into(), json!(t));
        }
        if let Some(ms) = spec.timeout_ms {
            o.insert("timeout_ms".into(), json!(ms));
        }
    }
    job.attempts += 1;
    job.tried.insert(n);
    let client = match nodes[n].client.as_mut() {
        Some(client) => client,
        None => {
            nodes[n].mark_down("selected node has no connection");
            job.state = "pending".to_string();
            return Ok((Some(n), false));
        }
    };
    match client.req("submit", fields, body) {
        Ok((h, _)) if h.get("ok").and_then(|x| x.as_bool()) == Some(true) => {
            job.job_id = h.get("job_id").and_then(|x| x.as_str()).map(str::to_string);
            job.node = Some(n);
            job.state = h
                .get("state")
                .and_then(|x| x.as_str())
                .unwrap_or("queued")
                .to_string();
            nodes[n].active += 1;
            let replayed = h.get("replayed").and_then(|x| x.as_bool()).unwrap_or(false);
            Ok((None, replayed))
        }
        Ok((h, _)) => {
            let err = h.get("error").cloned().unwrap_or(json!({}));
            let code = err.get("code").and_then(|x| x.as_str()).unwrap_or("");
            if code == "queue-full" {
                job.state = "pending".to_string();
                job.tried.remove(&n);
                return Ok((None, false));
            }
            job.error = Some(format!("submit rejected: {err}"));
            job.state = "failed".to_string();
            Ok((None, false))
        }
        Err(e) => {
            nodes[n].mark_down(e);
            job.state = "pending".to_string();
            Ok((Some(n), false))
        }
    }
}

fn poll_node(nodes: &mut [Node], jobs: &mut [Job], n: usize) -> usize {
    let ids: HashSet<String> = jobs
        .iter()
        .filter(|j| j.node == Some(n) && j.hash.is_none() && j.state != "failed")
        .filter_map(|j| j.job_id.clone())
        .collect();
    if ids.is_empty() {
        return 0;
    }
    let Some(client) = nodes[n].client.as_mut() else {
        nodes[n].healthy = false;
        return 0;
    };
    match client.req("job-list", json!({}), &[]) {
        Ok((h, _)) if h.get("ok").and_then(|x| x.as_bool()) == Some(true) => {
            let states: Vec<(String, String)> = h["jobs"]
                .as_array()
                .map(|a| {
                    a.iter()
                        .filter_map(|j| {
                            let id = j.get("job_id")?.as_str()?.to_string();
                            let state = j.get("state")?.as_str()?.to_string();
                            Some((id, state))
                        })
                        .collect()
                })
                .unwrap_or_default();
            for job in jobs.iter_mut().filter(|j| j.node == Some(n)) {
                if let Some((_, state)) = states
                    .iter()
                    .find(|(id, _)| Some(id) == job.job_id.as_ref())
                {
                    if job.state != "done" {
                        job.state = state.clone();
                    }
                }
            }
            0
        }
        Ok(_) => 0,
        Err(e) => requeue_node(nodes, jobs, n, e),
    }
}

fn upload_program(nodes: &mut [Node], program: &Option<(String, Vec<u8>)>) {
    let Some((name, listing)) = program.as_ref() else {
        return;
    };
    for n in nodes
        .iter_mut()
        .filter(|n| n.healthy && !n.program_uploaded)
    {
        let Some(client) = n.client.as_mut() else {
            n.healthy = false;
            continue;
        };
        match client.req(
            "program",
            json!({"name": name, "backend": "lattice"}),
            listing,
        ) {
            Ok((h, _)) if h.get("ok").and_then(|x| x.as_bool()) == Some(true) => {
                n.program_uploaded = true;
            }
            Ok((h, _)) => n.mark_down(format!("program failed: {h}")),
            Err(e) => n.mark_down(e),
        }
    }
}

pub fn run_batch(
    specs: &[NodeSpec],
    spec: &BatchSpec,
    program: Option<(String, Vec<u8>)>,
    ledger_path: Option<PathBuf>,
) -> Result<RunSummary, String> {
    if spec.count == 0 {
        return Err("count must be >= 1".to_string());
    }
    let (inputs, body) = body_from_feeds(&spec.feeds)?;
    let payload_hash = sha256_hex(&body);
    let program_hash = program
        .as_ref()
        .map(|(_, listing)| sha256_hex(listing))
        .unwrap_or_default();
    let ledger_spec = ledger_spec(spec, &payload_hash, &program_hash);
    let existing = match &ledger_path {
        Some(path) => ledger_load(path, &ledger_spec)?,
        None => None,
    };
    let created_ms = existing
        .as_ref()
        .map(|l| l.created_ms)
        .unwrap_or_else(now_ms);
    let resumed = existing.is_some();
    if let Some(ledger) = existing.as_ref().filter(|l| l.complete) {
        let mut summary = ledger
            .summary
            .clone()
            .ok_or_else(|| "complete ledger has no summary".to_string())?;
        summary.cached = true;
        summary.resumed = true;
        summary.receipts = ledger
            .jobs
            .iter()
            .filter_map(|j| j.receipt.clone())
            .collect();
        return Ok(summary);
    }

    let registry = Registry::start(
        specs,
        Duration::from_millis(spec.heartbeat_ms.max(1)),
        spec.down_after.max(1),
    );
    let mut nodes = connect_nodes(specs, &registry.snapshot());
    upload_program(&mut nodes, &program);
    if !nodes.iter().any(|n| n.healthy) {
        let errors: Vec<String> = nodes
            .iter()
            .map(|n| {
                format!(
                    "{}: {}",
                    n.spec.addr,
                    n.last_error.clone().unwrap_or_default()
                )
            })
            .collect();
        return Err(format!("no healthy nodes: {}", errors.join("; ")));
    }

    let start = Instant::now();
    let deadline = start + Duration::from_millis(spec.wait_ms.max(1));
    let mut jobs: Vec<Job> = match &existing {
        Some(ledger) => jobs_from_ledger(ledger, &nodes, spec),
        None => (0..spec.count)
            .map(|index| Job {
                index,
                attempts: 0,
                tried: HashSet::new(),
                node: None,
                job_id: None,
                state: "pending".to_string(),
                bytes: 0,
                hash: None,
                receipt: None,
                error: None,
            })
            .collect(),
    };
    for n in 0..nodes.len() {
        nodes[n].active = jobs
            .iter()
            .filter(|j| j.node == Some(n) && matches!(j.state.as_str(), "queued" | "running"))
            .count();
    }
    if let Some(path) = &ledger_path {
        ledger_save(path, &ledger_spec, created_ms, &jobs, &nodes, false, None)?;
    }
    let mut retried = 0usize;
    let mut replayed = 0usize;
    let mut next_save = Instant::now() + Duration::from_secs(1);

    loop {
        let health = registry.snapshot();
        retried += sync_health(&mut nodes, &mut jobs, &health);
        upload_program(&mut nodes, &program);
        for i in 0..jobs.len() {
            if jobs[i].state != "pending" {
                continue;
            }
            let (lost, was_replayed) = submit_job(&mut nodes, &mut jobs[i], &inputs, &body, spec)?;
            if was_replayed {
                replayed += 1;
            }
            if let Some(n) = lost {
                retried += requeue_node(&mut nodes, &mut jobs, n, "node lost during submission");
            }
        }
        for n in 0..nodes.len() {
            if nodes[n].healthy {
                retried += poll_node(&mut nodes, &mut jobs, n);
            }
        }
        for i in 0..jobs.len() {
            if jobs[i].state == "done" && jobs[i].hash.is_none() {
                let n = match jobs[i].node {
                    Some(n) => n,
                    None => continue,
                };
                if !nodes[n].healthy {
                    continue;
                }
                let job_id = match jobs[i].job_id.clone() {
                    Some(id) => id,
                    None => continue,
                };
                match fetch_result(&mut nodes[n], &job_id) {
                    Ok((bytes, hash, receipt)) => {
                        jobs[i].bytes = bytes;
                        jobs[i].hash = Some(hash);
                        jobs[i].receipt = receipt;
                        nodes[n].active = nodes[n].active.saturating_sub(1);
                        nodes[n].done += 1;
                        nodes[n].bytes += bytes;
                    }
                    Err(e) => {
                        retried += requeue_node(&mut nodes, &mut jobs, n, e);
                    }
                }
            }
        }

        for i in 0..jobs.len() {
            if matches!(jobs[i].state.as_str(), "failed" | "canceled" | "timed_out") {
                if let Some(n) = jobs[i].node.take() {
                    nodes[n].active = nodes[n].active.saturating_sub(1);
                }
                if jobs[i].attempts < spec.max_attempts {
                    jobs[i].state = "pending".to_string();
                    jobs[i].job_id = None;
                    retried += 1;
                }
            }
        }

        if let Some(path) = &ledger_path {
            if Instant::now() >= next_save {
                ledger_save(path, &ledger_spec, created_ms, &jobs, &nodes, false, None)?;
                next_save = Instant::now() + Duration::from_secs(1);
            }
        }

        let all_done = jobs.iter().all(|j| j.state == "done");
        let all_terminal = jobs
            .iter()
            .all(|j| matches!(j.state.as_str(), "done" | "failed"));
        if all_done || all_terminal {
            break;
        }
        if Instant::now() >= deadline {
            for job in jobs.iter_mut() {
                if job.state != "done" {
                    job.state = "failed".to_string();
                    job.error = Some("batch deadline exceeded".to_string());
                }
            }
            break;
        }
        std::thread::sleep(Duration::from_millis(spec.poll_ms.max(1)));
    }

    let done = jobs.iter().filter(|j| j.state == "done").count();
    let failed = jobs.len() - done;
    let output_bytes: usize = jobs.iter().map(|j| j.bytes).sum();
    let mut output_hash: Option<String> = None;
    let mut identical = true;
    for job in jobs.iter().filter(|j| j.hash.is_some()) {
        match &output_hash {
            None => output_hash = job.hash.clone(),
            Some(prev) if Some(prev) != job.hash.as_ref() => identical = false,
            _ => {}
        }
    }
    let wall = start.elapsed();
    let receipts: Vec<Json> = jobs.iter().filter_map(|j| j.receipt.clone()).collect();
    let health = registry.snapshot();

    let summary = RunSummary {
        nodes: nodes
            .iter()
            .map(|n| {
                let h = health.iter().find(|h| h.addr == n.spec.addr);
                NodeSummary {
                    addr: n.spec.addr.clone(),
                    healthy: n.healthy,
                    workers: n.workers,
                    queue_capacity: n.queue_capacity,
                    queued: n.queued,
                    draining: n.draining,
                    failure_streak: h.map(|h| h.failure_streak).unwrap_or(0),
                    last_good_age_ms: h.and_then(|h| h.last_good_age_ms),
                    done: n.done,
                    bytes: n.bytes,
                }
            })
            .collect(),
        submitted: jobs.len(),
        done,
        failed,
        retried,
        replayed,
        output_bytes,
        output_hash,
        identical_outputs: identical,
        wall_ms: wall.as_millis() as u64,
        jobs_per_s: jobs.len() as f64 / wall.as_secs_f64().max(1e-9),
        cached: false,
        resumed,
        receipts,
    };
    if let Some(path) = &ledger_path {
        ledger_save(
            path,
            &ledger_spec,
            created_ms,
            &jobs,
            &nodes,
            true,
            Some(&summary),
        )?;
    }
    Ok(summary)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn node(addr: &str, workers: usize) -> Node {
        Node {
            spec: NodeSpec {
                addr: addr.to_string(),
                token: None,
            },
            client: None,
            healthy: true,
            workers,
            queue_capacity: 64,
            queued: 0,
            draining: false,
            last_error: None,
            active: 0,
            done: 0,
            bytes: 0,
            program_uploaded: false,
        }
    }

    fn job(index: usize) -> Job {
        Job {
            index,
            attempts: 0,
            tried: HashSet::new(),
            node: None,
            job_id: None,
            state: "pending".to_string(),
            bytes: 0,
            hash: None,
            receipt: None,
            error: None,
        }
    }

    #[test]
    fn placement_prefers_least_loaded_and_skips_tried() {
        let mut a = node("a:1", 2);
        let mut b = node("b:1", 2);
        a.active = 2;
        b.active = 0;
        let nodes = vec![a, b];
        let j = job(0);
        assert_eq!(pick_node(&nodes, &j, "batch"), Some(1));

        let mut j = job(0);
        j.tried.insert(1);
        assert_eq!(pick_node(&nodes, &j, "batch"), Some(0));
    }

    #[test]
    fn placement_requires_a_healthy_node() {
        let mut a = node("a:1", 1);
        a.healthy = false;
        let nodes = vec![a];
        assert_eq!(pick_node(&nodes, &job(0), "batch"), None);
    }

    #[test]
    fn deterministic_tie_break_is_stable() {
        let nodes = vec![node("a:1", 1), node("b:1", 1)];
        let first = pick_node(&nodes, &job(7), "batch-1");
        let second = pick_node(&nodes, &job(7), "batch-1");
        assert_eq!(first, second);
    }

    #[test]
    fn node_loss_requeues_every_unfetched_job() {
        let mut nodes = vec![node("a:1", 1), node("b:1", 1)];
        nodes[1].active = 3;
        let mut jobs = vec![job(0), job(1), job(2)];
        jobs[0].node = Some(1);
        jobs[0].state = "running".to_string();
        jobs[1].node = Some(1);
        jobs[1].state = "done".to_string();
        jobs[2].node = Some(1);
        jobs[2].state = "done".to_string();
        jobs[2].hash = Some("already-fetched".to_string());

        let requeued = requeue_node(&mut nodes, &mut jobs, 1, "node lost");
        assert_eq!(requeued, 2);
        assert!(!nodes[1].healthy);
        assert_eq!(nodes[1].active, 0);
        assert_eq!(jobs[0].state, "pending");
        assert_eq!(jobs[1].state, "pending");
        assert_eq!(jobs[2].state, "done");
        assert_eq!(jobs[2].hash.as_deref(), Some("already-fetched"));
    }

    #[test]
    fn health_tracks_draining_and_failure_streak() {
        let mut h = HealthState::new();
        record_health(&mut h, Ok((2, 8, 8, 2)), 2);
        assert!(h.healthy);
        assert!(h.draining);
        assert_eq!(h.queued, 8);

        record_health(&mut h, Ok((2, 8, 1, 2)), 2);
        assert!(!h.draining);
        assert_eq!(h.failure_streak, 0);

        record_health(&mut h, Err("down".to_string()), 2);
        assert!(h.healthy);
        assert_eq!(h.failure_streak, 1);
        record_health(&mut h, Err("down".to_string()), 2);
        assert!(!h.healthy);
        assert!(h.draining);
        assert_eq!(h.failure_streak, 2);
    }

    #[test]
    fn body_offsets_are_contiguous() {
        use jotnar::value::{array_from_vec, Value};
        let mut feeds = Feeds::new();
        feeds.insert(
            "X".to_string(),
            Value::I64(array_from_vec(&[2], vec![3, 3]).unwrap()),
        );
        let (inputs, body) = body_from_feeds(&feeds).unwrap();
        assert!(!body.is_empty());
        assert_eq!(inputs["X"]["off"], 0);
        assert_eq!(inputs["X"]["len"], 16);
    }
}
