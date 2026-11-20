//! Host-side GPU runner for Jotnar.
//!
//! This is deliberately not `jotnar-noded`: the model-blind CPU node stays
//! unchanged. The runner is target-aware, compiles a listing with the existing
//! HIP or CUDA codegen path, runs it on the local GPU, and returns typed stream
//! outputs over the same frame codec.
//!
//! Request:
//!   header: v, id, type=run, target=hip|cuda, outputs=[...], auth?
//!   body:   {"listing": "...", "payload": {"streams": {...}}, "use_fp16": false}
//!
//! Response:
//!   header: type=run-ok, target, runtime_version, compile_ms, build_ms, run_ms
//!   body:   {"outputs": {name: <payload stream>, ...}, "timing": {...}}
//!
//! Messages are one frame per connection. `ping` is available for capability
//! checks.

use std::collections::{HashMap, VecDeque};
use std::fs;
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream, ToSocketAddrs};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use jotnar::asm::AsmOptions;
use jotnar::codegen::infer::CKind;
use jotnar::codegen::{build, compile_program, Artifact, CompileOptions, Target};
use jotnar::payload::{feeds_from_binary, feeds_from_json, value_to_json, Feeds};
use jotnar::protocol::{decode_frame, encode_frame, PROTO_V, RUNTIME_VERSION};
use jotnar::value::{array_from_vec, hash_bytes, Value};
use mdns_sd::{ServiceDaemon, ServiceInfo};
use serde_json::{json, Value as Json};

mod warm;

const DEFAULT_BIND: &str = "127.0.0.1:9997";
const DEFAULT_MAX_FRAME: usize = 64 * 1024 * 1024;
const MAX_BATCH_FRAMES: usize = 64;

#[derive(Clone)]
struct Config {
    bind: String,
    token: Option<String>,
    target: Target,
    max_frame: usize,
    cache_dir: PathBuf,
    build_lock: Arc<Mutex<()>>,
    mdns: bool,
    warm: bool,
    warm_idle: Option<Duration>,
    warm_cache: Arc<Mutex<warm::WarmCache>>,
    payload_cache: Arc<Mutex<PayloadCache>>,
    artifact_cache: Arc<Mutex<ArtifactCache>>,
    jobs: Arc<Mutex<JobStore>>,
    self_addr: String,
    started_ms: u64,
}

struct CachedPayload {
    feeds: Arc<Feeds>,
    bytes: usize,
}

struct PayloadCache {
    map: HashMap<String, CachedPayload>,
    order: VecDeque<String>,
    bytes: usize,
    limit: usize,
}

impl PayloadCache {
    fn new(limit: usize) -> Self {
        Self {
            map: HashMap::new(),
            order: VecDeque::new(),
            bytes: 0,
            limit,
        }
    }

    fn put(&mut self, id: String, feeds: Feeds, bytes: usize) -> String {
        if self.map.contains_key(&id) {
            return id;
        }
        self.bytes = self.bytes.saturating_add(bytes);
        self.map.insert(
            id.clone(),
            CachedPayload {
                feeds: Arc::new(feeds),
                bytes,
            },
        );
        self.order.push_back(id.clone());
        while self.bytes > self.limit && self.order.len() > 1 {
            if let Some(old) = self.order.pop_front() {
                if let Some(p) = self.map.remove(&old) {
                    self.bytes = self.bytes.saturating_sub(p.bytes);
                }
            }
        }
        id
    }

    fn get(&self, id: &str) -> Option<Arc<Feeds>> {
        self.map.get(id).map(|p| p.feeds.clone())
    }
}

struct ArtifactCache {
    map: HashMap<String, Arc<Artifact>>,
    order: VecDeque<String>,
    limit: usize,
}

impl ArtifactCache {
    fn new(limit: usize) -> Self {
        Self {
            map: HashMap::new(),
            order: VecDeque::new(),
            limit: limit.max(1),
        }
    }

    fn get(&mut self, key: &str) -> Option<Arc<Artifact>> {
        let value = self.map.get(key).cloned();
        if value.is_some() {
            if let Some(pos) = self.order.iter().position(|k| k == key) {
                self.order.remove(pos);
            }
            self.order.push_back(key.to_string());
        }
        value
    }

    fn put(&mut self, key: String, value: Arc<Artifact>) {
        if self.map.insert(key.clone(), value).is_none() {
            self.order.push_back(key);
        }
        while self.map.len() > self.limit {
            if let Some(old) = self.order.pop_front() {
                self.map.remove(&old);
            }
        }
    }
}

const RETAIN_JOBS: usize = 1024;

struct AsyncJob {
    id: String,
    state: String,
    error: Option<String>,
    created_ms: u128,
    started_ms: Option<u128>,
    finished_ms: Option<u128>,
    result_header: Option<Json>,
    result_body: Vec<u8>,
    cancel: Arc<AtomicBool>,
}

struct JobStore {
    jobs: HashMap<String, AsyncJob>,
    order: VecDeque<String>,
    next_id: u64,
    limit: usize,
}

impl JobStore {
    fn new(limit: usize) -> Self {
        Self {
            jobs: HashMap::new(),
            order: VecDeque::new(),
            next_id: 1,
            limit: limit.max(1),
        }
    }

    fn next_id(&mut self) -> String {
        let id = format!("gpu-{}-{}", now_ms(), self.next_id);
        self.next_id = self.next_id.saturating_add(1);
        id
    }

    fn insert(&mut self, job: AsyncJob) {
        let id = job.id.clone();
        self.jobs.insert(id.clone(), job);
        self.order.push_back(id);
        while self.jobs.len() > self.limit {
            let mut removed = false;
            for idx in 0..self.order.len() {
                let key = self.order[idx].clone();
                let evict = self
                    .jobs
                    .get(&key)
                    .map(|j| !matches!(j.state.as_str(), "queued" | "running"))
                    .unwrap_or(true);
                if evict {
                    self.order.remove(idx);
                    self.jobs.remove(&key);
                    removed = true;
                    break;
                }
            }
            if !removed {
                break;
            }
        }
    }
}

fn now_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or(0)
}

fn usage() -> &'static str {
    "jotnar-gpu-runner — host-side HIP/CUDA job runner\n\
     \n\
     usage:\n\
       jotnar-gpu-runner --target hip|cuda [--bind ADDR] [--token SECRET]\n\
                         [--max-frame BYTES] [--cache-dir DIR] [--mdns]\n\
     \n\
     The runner compiles and runs Jotnar listings on the local GPU host. Set\n\
     HIPCC / HIP_ARCH for HIP, or NVCC / CUDA_ARCH for CUDA."
}

fn main() -> std::process::ExitCode {
    match real_main() {
        Ok(()) => std::process::ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("{e}");
            std::process::ExitCode::FAILURE
        }
    }
}

fn is_loopback_bind(bind: &str) -> bool {
    if bind == "localhost" {
        return true;
    }
    if let Ok(ip) = bind.parse::<std::net::IpAddr>() {
        return ip.is_loopback();
    }
    if let Ok(addrs) = std::net::ToSocketAddrs::to_socket_addrs(&bind) {
        let ips: Vec<std::net::IpAddr> = addrs.map(|a| a.ip()).collect();
        if !ips.is_empty() {
            return ips.iter().all(|ip| ip.is_loopback());
        }
    }
    false
}

fn real_main() -> Result<(), String> {
    let mut bind = DEFAULT_BIND.to_string();
    let mut token = None;
    let mut target = None;
    let mut max_frame = DEFAULT_MAX_FRAME;
    let mut cache_dir = std::env::temp_dir().join("jotnar-gpu-runner-cache");
    let mut mdns = false;
    let mut warm = true;
    let mut warm_cache_workers = 8usize;
    let mut warm_idle_ms = 300_000u64;
    let mut payload_cache_bytes = 256 * 1024 * 1024usize;
    let mut artifact_cache_entries = 64usize;
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--bind" => {
                i += 1;
                bind = args.get(i).ok_or("--bind needs an address")?.clone();
            }
            "--token" => {
                i += 1;
                token = Some(args.get(i).ok_or("--token needs a value")?.clone());
            }
            "--max-frame" => {
                i += 1;
                max_frame = args
                    .get(i)
                    .ok_or("--max-frame needs a value")?
                    .parse()
                    .map_err(|e| format!("bad --max-frame: {e}"))?;
            }
            "--target" => {
                i += 1;
                let raw = args.get(i).ok_or("--target needs hip or cuda")?;
                target = Some(match Target::parse(raw) {
                    Ok(Target::Hip) | Ok(Target::Cuda) => Target::parse(raw).unwrap(),
                    _ => return Err("runner target must be hip or cuda".to_string()),
                });
            }
            "--cache-dir" => {
                i += 1;
                cache_dir = PathBuf::from(args.get(i).ok_or("--cache-dir needs a path")?);
            }
            "--mdns" => {
                mdns = true;
            }
            "--no-warm" => {
                warm = false;
            }
            "--warm-cache-workers" => {
                i += 1;
                warm_cache_workers = args
                    .get(i)
                    .ok_or("--warm-cache-workers needs a value")?
                    .parse()
                    .map_err(|e| format!("bad --warm-cache-workers: {e}"))?;
            }
            "--warm-idle-ms" => {
                i += 1;
                warm_idle_ms = args
                    .get(i)
                    .ok_or("--warm-idle-ms needs a value")?
                    .parse()
                    .map_err(|e| format!("bad --warm-idle-ms: {e}"))?;
            }
            "--payload-cache-bytes" => {
                i += 1;
                payload_cache_bytes = args
                    .get(i)
                    .ok_or("--payload-cache-bytes needs a value")?
                    .parse()
                    .map_err(|e| format!("bad --payload-cache-bytes: {e}"))?;
            }
            "--artifact-cache-entries" => {
                i += 1;
                artifact_cache_entries = args
                    .get(i)
                    .ok_or("--artifact-cache-entries needs a value")?
                    .parse()
                    .map_err(|e| format!("bad --artifact-cache-entries: {e}"))?;
            }
            "-h" | "--help" => return Err(usage().to_string()),
            other => return Err(format!("unknown argument '{other}'\n\n{}", usage())),
        }
        i += 1;
    }
    let target = target.ok_or("--target hip|cuda is required")?;
    if token.as_deref().map(str::trim).is_some_and(str::is_empty) {
        return Err("--token must not be empty".to_string());
    }
    if !is_loopback_bind(&bind) && token.is_none() {
        return Err(format!(
            "refusing non-loopback bind {bind} without --token\n\n{}",
            usage()
        ));
    }
    let mut cfg = Config {
        bind,
        token,
        target,
        max_frame: max_frame.max(1024),
        cache_dir,
        build_lock: Arc::new(Mutex::new(())),
        mdns,
        warm,
        warm_idle: (warm_idle_ms > 0).then(|| Duration::from_millis(warm_idle_ms)),
        warm_cache: Arc::new(Mutex::new(warm::WarmCache::new(warm_cache_workers))),
        payload_cache: Arc::new(Mutex::new(PayloadCache::new(payload_cache_bytes.max(1)))),
        artifact_cache: Arc::new(Mutex::new(ArtifactCache::new(artifact_cache_entries))),
        jobs: Arc::new(Mutex::new(JobStore::new(RETAIN_JOBS))),
        self_addr: String::new(),
        started_ms: now_ms() as u64,
    };
    if cfg.warm {
        if let Some(idle) = cfg.warm_idle {
            let cache = Arc::clone(&cfg.warm_cache);
            let tick = idle
                .min(Duration::from_secs(30))
                .max(Duration::from_millis(100));
            let _ = thread::Builder::new()
                .name("jotnar-warm-reaper".to_string())
                .spawn(move || loop {
                    thread::sleep(tick);
                    let dropped = {
                        let mut cache = cache.lock().unwrap_or_else(|e| e.into_inner());
                        cache.remove_idle(idle, Instant::now())
                    };
                    drop(dropped);
                });
        }
    }
    let listener = TcpListener::bind(&cfg.bind).map_err(|e| format!("bind {}: {e}", cfg.bind))?;
    let port = listener
        .local_addr()
        .map_err(|e| format!("bound address: {e}"))?
        .port();
    let host = cfg
        .bind
        .rsplit_once(':')
        .map(|(h, _)| h)
        .unwrap_or("127.0.0.1");
    let host = match host {
        "0.0.0.0" => "127.0.0.1",
        "[::]" => "[::1]",
        other => other,
    };
    cfg.self_addr = format!("{host}:{port}");
    let _mdns = if cfg.mdns {
        match start_mdns(&cfg, port) {
            Ok(daemon) => {
                eprintln!(
                    "mdns: advertising _jotnar-gpu._tcp on port {port} as {}",
                    cfg.target.as_str()
                );
                Some(daemon)
            }
            Err(e) => {
                eprintln!("mdns unavailable: {e}");
                None
            }
        }
    } else {
        None
    };
    eprintln!(
        "gpu-runner target={} bind={} runtime={}",
        cfg.target.as_str(),
        cfg.bind,
        RUNTIME_VERSION
    );
    for conn in listener.incoming() {
        match conn {
            Ok(stream) => {
                let cfg = cfg.clone();
                thread::spawn(move || {
                    if let Err(e) = handle(stream, &cfg) {
                        eprintln!("gpu-runner request: {e}");
                    }
                });
            }
            Err(e) => eprintln!("gpu-runner accept: {e}"),
        }
    }
    Ok(())
}

fn start_mdns(cfg: &Config, port: u16) -> Result<ServiceDaemon, String> {
    let daemon = ServiceDaemon::new().map_err(|e| e.to_string())?;
    let mut props = HashMap::new();
    props.insert("target".to_string(), cfg.target.as_str().to_string());
    props.insert(
        "arch".to_string(),
        if cfg.target == Target::Hip {
            std::env::var("HIP_ARCH").unwrap_or_default()
        } else {
            std::env::var("CUDA_ARCH").unwrap_or_default()
        },
    );
    props.insert(
        "toolchain".to_string(),
        if cfg.target == Target::Hip {
            "hipcc".to_string()
        } else {
            "nvcc".to_string()
        },
    );
    props.insert(
        "auth".to_string(),
        if cfg.token.is_some() {
            "required".to_string()
        } else {
            "none".to_string()
        },
    );
    props.insert("runtime_version".to_string(), RUNTIME_VERSION.to_string());
    props.insert("pid".to_string(), std::process::id().to_string());
    let host = format!(
        "jotnar-gpu-{}-{}.local.",
        cfg.target.as_str(),
        std::process::id()
    );
    let instance = format!("Jotnar GPU {} {}", cfg.target.as_str(), std::process::id());
    let info = ServiceInfo::new("_jotnar-gpu._tcp.local.", &instance, &host, "", port, props)
        .map_err(|e| e.to_string())?
        .enable_addr_auto();
    daemon.register(info).map_err(|e| e.to_string())?;
    Ok(daemon)
}

fn handle(mut stream: TcpStream, cfg: &Config) -> Result<(), String> {
    let _ = stream.set_nodelay(true);
    let mut len = [0u8; 4];
    stream
        .read_exact(&mut len)
        .map_err(|e| format!("read frame header: {e}"))?;
    let n = u32::from_be_bytes(len) as usize;
    if n > cfg.max_frame {
        return Err(format!("frame {n} exceeds max {}", cfg.max_frame));
    }
    let mut body = vec![0u8; n];
    stream
        .read_exact(&mut body)
        .map_err(|e| format!("read frame body: {e}"))?;
    let mut frame = Vec::with_capacity(4 + n);
    frame.extend_from_slice(&len);
    frame.extend_from_slice(&body);
    let (header, payload) = decode_frame(&frame).map_err(|e| format!("bad frame: {e}"))?;
    let id = header.get("id").cloned().unwrap_or(json!(0));
    if let Some(want) = &cfg.token {
        let got = header.get("auth").and_then(|x| x.as_str()).unwrap_or("");
        if want.trim().is_empty() || got != want.as_str() {
            return send_error(
                &mut stream,
                &id,
                "unauthorized",
                "bad or missing auth token",
            );
        }
    }
    match header.get("type").and_then(|x| x.as_str()).unwrap_or("") {
        "ping" => {
            let arch = if cfg.target == Target::Hip {
                std::env::var("HIP_ARCH").unwrap_or_default()
            } else {
                std::env::var("CUDA_ARCH").unwrap_or_default()
            };
            let toolchain = if cfg.target == Target::Hip {
                "hipcc"
            } else {
                "nvcc"
            };
            send(
                &mut stream,
                &json!({
                    "v": PROTO_V,
                    "id": id,
                    "type": "pong",
                    "ok": true,
                    "target": cfg.target.as_str(),
                    "toolchain": toolchain,
                    "arch": arch,
                    "runtime_version": RUNTIME_VERSION,
                }),
                &[],
            )
        }
        "health" => {
            if let Err(e) = send_health(&mut stream, cfg, &id) {
                send_error(&mut stream, &id, "health-failed", &e)?;
            }
            Ok(())
        }
        "payload-put" => {
            let pid = format!("{:016x}", hash_bytes(payload));
            let cached = {
                let cache = cfg.payload_cache.lock().unwrap_or_else(|e| e.into_inner());
                cache.get(&pid).is_some()
            };
            if !cached {
                let parsed: Json = match serde_json::from_slice(payload) {
                    Ok(v) => v,
                    Err(e) => return send_error(&mut stream, &id, "bad-payload", &format!("{e}")),
                };
                let feeds = match feeds_from_json(&parsed) {
                    Ok(f) => f,
                    Err(e) => return send_error(&mut stream, &id, "bad-payload", &e.to_string()),
                };
                let mut cache = cfg.payload_cache.lock().unwrap_or_else(|e| e.into_inner());
                cache.put(pid.clone(), feeds, payload.len());
            }
            send(
                &mut stream,
                &json!({
                    "v": PROTO_V,
                    "id": id,
                    "type": "payload-ok",
                    "ok": true,
                    "payload_id": pid,
                    "bytes": payload.len(),
                }),
                &[],
            )
        }
        "submit" => {
            if let Err(e) = submit_job(&mut stream, cfg, &id, &header, payload) {
                send_error(&mut stream, &id, "submit-failed", &e)?;
            }
            Ok(())
        }
        "job-status" => {
            if let Err(e) = job_status(&mut stream, cfg, &id, &header) {
                let code = if e.starts_with("unknown job") {
                    "unknown-job"
                } else {
                    "job-status-failed"
                };
                send_error(&mut stream, &id, code, &e)?;
            }
            Ok(())
        }
        "job-result" => {
            if let Err(e) = job_result(&mut stream, cfg, &id, &header) {
                let code = if e.starts_with("unknown job") {
                    "unknown-job"
                } else {
                    "job-result-failed"
                };
                send_error(&mut stream, &id, code, &e)?;
            }
            Ok(())
        }
        "job-cancel" => {
            if let Err(e) = job_cancel(&mut stream, cfg, &id, &header) {
                let code = if e.starts_with("unknown job") {
                    "unknown-job"
                } else {
                    "job-cancel-failed"
                };
                send_error(&mut stream, &id, code, &e)?;
            }
            Ok(())
        }
        "run" => {
            if let Err(e) = run(&mut stream, cfg, &id, &header, payload) {
                send_error(&mut stream, &id, "run-failed", &e)?;
            }
            Ok(())
        }
        "run-batch" => {
            if let Err(e) = run_batch(&mut stream, cfg, &id, &header, payload) {
                send_error(&mut stream, &id, "run-batch-failed", &e)?;
            }
            Ok(())
        }
        "warm-prepare" => {
            if let Err(e) = warm_prepare(&mut stream, cfg, &id, &header, payload) {
                send_error(&mut stream, &id, "warm-prepare-failed", &e)?;
            }
            Ok(())
        }
        other => send_error(
            &mut stream,
            &id,
            "unknown-type",
            &format!("no such message {other:?}"),
        ),
    }
}

fn send_health(stream: &mut TcpStream, cfg: &Config, id: &Json) -> Result<(), String> {
    let now = now_ms();
    let (queued, running, retained, jobs) = {
        let store = cfg.jobs.lock().unwrap_or_else(|e| e.into_inner());
        let mut jobs = Vec::new();
        for job_id in store.order.iter().rev().take(32) {
            let Some(job) = store.jobs.get(job_id) else {
                continue;
            };
            let elapsed_ms = match (job.started_ms, job.finished_ms) {
                (Some(started), Some(finished)) => Some(finished.saturating_sub(started)),
                (Some(started), None) => Some(now.saturating_sub(started)),
                _ => None,
            };
            jobs.push(json!({
                "id": job.id,
                "state": job.state,
                "created_ms": job.created_ms as u64,
                "started_ms": job.started_ms.map(|v| v as u64),
                "finished_ms": job.finished_ms.map(|v| v as u64),
                "elapsed_ms": elapsed_ms.map(|v| v as u64),
            }));
        }
        jobs.reverse();
        let queued = store.jobs.values().filter(|j| j.state == "queued").count();
        let running = store.jobs.values().filter(|j| j.state == "running").count();
        (queued, running, store.jobs.len(), jobs)
    };
    let warm_workers = cfg
        .warm_cache
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .len();
    let arch = if cfg.target == Target::Hip {
        std::env::var("HIP_ARCH").unwrap_or_default()
    } else {
        std::env::var("CUDA_ARCH").unwrap_or_default()
    };
    let toolchain = if cfg.target == Target::Hip {
        "hipcc"
    } else {
        "nvcc"
    };
    send(
        stream,
        &json!({
            "v": PROTO_V,
            "id": id,
            "type": "health-ok",
            "ok": true,
            "schema": "jotnar.executor.health/1",
            "kind": "gpu",
            "node_id": format!("Jotnar GPU {} {}", cfg.target.as_str(), std::process::id()),
            "target": cfg.target.as_str(),
            "arch": arch,
            "toolchain": toolchain,
            "runtime_version": RUNTIME_VERSION,
            "ts_ms": now as u64,
            "uptime_ms": (now as u64).saturating_sub(cfg.started_ms),
            "limits": {
                "workers": Json::Null,
                "queue": Json::Null,
                "max_frame": cfg.max_frame,
            },
            "activity": {
                "queued": queued,
                "running": running,
                "retained": retained,
                "warm_workers": warm_workers,
                "warm_artifacts": warm_workers,
            },
            "jobs": jobs,
        }),
        &[],
    )
}

fn body_head(payload: &[u8], binary: bool) -> Result<Json, String> {
    if binary {
        let pos = payload
            .iter()
            .position(|b| *b == b'\n')
            .ok_or("binary body needs a JSON metadata line")?;
        serde_json::from_slice(&payload[..pos]).map_err(|e| format!("bad metadata line: {e}"))
    } else {
        serde_json::from_slice(payload).map_err(|e| format!("bad body: {e}"))
    }
}

fn submit_job(
    stream: &mut TcpStream,
    cfg: &Config,
    id: &Json,
    header: &Json,
    payload: &[u8],
) -> Result<(), String> {
    let outputs: Vec<String> = header
        .get("outputs")
        .and_then(|x| x.as_array())
        .ok_or("submit needs outputs[]")?
        .iter()
        .map(|x| {
            x.as_str()
                .map(|s| s.to_string())
                .ok_or("outputs must be strings")
        })
        .collect::<Result<_, _>>()?;
    if outputs.is_empty() {
        return send_error(
            stream,
            id,
            "bad-request",
            "submit needs at least one output",
        );
    }
    let frames = header.get("frames").and_then(|x| x.as_u64()).unwrap_or(1) as usize;
    if frames == 0 || frames > MAX_BATCH_FRAMES {
        return send_error(
            stream,
            id,
            "bad-request",
            &format!("submit frames must be 1..={MAX_BATCH_FRAMES}"),
        );
    }
    let binary_payload = header.get("payload_format").and_then(|x| x.as_str()) == Some("binary");
    let body = body_head(payload, binary_payload)?;
    let listing = body
        .get("listing")
        .and_then(|x| x.as_str())
        .ok_or("submit body needs listing")?;
    let _ = listing;
    let typ = if frames > 1 { "run-batch" } else { "run" };
    let mut inner = json!({
        "v": PROTO_V,
        "id": 1,
        "type": typ,
        "target": cfg.target.as_str(),
        "outputs": outputs,
    });
    if binary_payload {
        inner["payload_format"] = json!("binary");
    }
    if frames > 1 {
        inner["frames"] = json!(frames);
    }
    if let Some(pid) = header
        .get("payload_id")
        .or_else(|| body.get("payload_id"))
        .and_then(|x| x.as_str())
    {
        inner["payload_id"] = json!(pid);
    }
    if let Some(fmt) = header.get("output_format").and_then(|x| x.as_str()) {
        inner["output_format"] = json!(fmt);
    }
    if let Some(u8_output) = header.get("output_u8").and_then(|x| x.as_bool()) {
        inner["output_u8"] = json!(u8_output);
    }
    if let Some(token) = &cfg.token {
        inner["auth"] = json!(token);
    }
    let inner_body = payload.to_vec();
    let job_id = {
        let mut store = cfg.jobs.lock().unwrap_or_else(|e| e.into_inner());
        let job_id = store.next_id();
        let cancel = Arc::new(AtomicBool::new(false));
        store.insert(AsyncJob {
            id: job_id.clone(),
            state: "queued".to_string(),
            error: None,
            created_ms: now_ms(),
            started_ms: None,
            finished_ms: None,
            result_header: None,
            result_body: Vec::new(),
            cancel: cancel.clone(),
        });
        (job_id, cancel)
    };
    let (job_id, cancel) = job_id;
    let thread_cfg = cfg.clone();
    let thread_inner = inner.clone();
    let thread_id = job_id.clone();
    thread::spawn(move || {
        if cancel.load(Ordering::SeqCst) {
            finish_job(&thread_cfg, &thread_id, "canceled", None, None, Vec::new());
            return;
        }
        mark_running(&thread_cfg, &thread_id);
        let result = self_request(&thread_cfg, &thread_inner, &inner_body);
        match result {
            Ok((reply, body)) if reply.get("ok").and_then(|x| x.as_bool()) == Some(true) => {
                finish_job(&thread_cfg, &thread_id, "done", None, Some(reply), body);
            }
            Ok((reply, _)) => {
                let err = reply
                    .get("error")
                    .cloned()
                    .map(|e| e.to_string())
                    .unwrap_or_else(|| reply.to_string());
                finish_job(
                    &thread_cfg,
                    &thread_id,
                    "failed",
                    Some(err),
                    None,
                    Vec::new(),
                );
            }
            Err(e) => finish_job(&thread_cfg, &thread_id, "failed", Some(e), None, Vec::new()),
        }
    });

    send(
        stream,
        &json!({
            "v": PROTO_V,
            "id": id,
            "type": "submit-ok",
            "ok": true,
            "job_id": job_id,
            "state": "queued",
        }),
        &[],
    )
}

fn mark_running(cfg: &Config, job_id: &str) {
    let mut store = cfg.jobs.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(job) = store.jobs.get_mut(job_id) {
        job.state = "running".to_string();
        job.started_ms = Some(now_ms());
    }
}

fn finish_job(
    cfg: &Config,
    job_id: &str,
    state: &str,
    error: Option<String>,
    header: Option<Json>,
    body: Vec<u8>,
) {
    let mut store = cfg.jobs.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(job) = store.jobs.get_mut(job_id) {
        job.state = state.to_string();
        job.error = error;
        job.finished_ms = Some(now_ms());
        job.result_header = header;
        job.result_body = body;
    }
}

fn self_request(cfg: &Config, header: &Json, body: &[u8]) -> Result<(Json, Vec<u8>), String> {
    let addrs = cfg
        .self_addr
        .to_socket_addrs()
        .map_err(|e| format!("resolve {}: {e}", cfg.self_addr))?;
    let mut last = None;
    let mut stream = None;
    for addr in addrs {
        match TcpStream::connect_timeout(&addr, Duration::from_secs(3)) {
            Ok(s) => {
                stream = Some(s);
                break;
            }
            Err(e) => last = Some(e),
        }
    }
    let mut stream = stream.ok_or_else(|| {
        format!(
            "connect {}: {}",
            cfg.self_addr,
            last.map(|e| e.to_string()).unwrap_or_default()
        )
    })?;
    let _ = stream.set_read_timeout(Some(Duration::from_secs(600)));
    let _ = stream.set_write_timeout(Some(Duration::from_secs(600)));
    let frame = encode_frame(header, body).map_err(|e| e.to_string())?;
    stream
        .write_all(&frame)
        .map_err(|e| format!("write: {e}"))?;
    let mut hdr = [0u8; 4];
    stream
        .read_exact(&mut hdr)
        .map_err(|e| format!("read header: {e}"))?;
    let n = u32::from_be_bytes(hdr) as usize;
    if n > 2 * 1024 * 1024 * 1024 {
        return Err("response too large".to_string());
    }
    let mut buf = vec![0u8; n];
    stream
        .read_exact(&mut buf)
        .map_err(|e| format!("read body: {e}"))?;
    let mut full = Vec::with_capacity(4 + n);
    full.extend_from_slice(&hdr);
    full.extend_from_slice(&buf);
    decode_frame(&full)
        .map(|(header, payload)| (header, payload.to_vec()))
        .map_err(|e| e.to_string())
}

fn job_status(
    stream: &mut TcpStream,
    cfg: &Config,
    id: &Json,
    header: &Json,
) -> Result<(), String> {
    let job_id = header
        .get("job_id")
        .and_then(|x| x.as_str())
        .ok_or("job-status needs job_id")?;
    let store = cfg.jobs.lock().unwrap_or_else(|e| e.into_inner());
    let job = store
        .jobs
        .get(job_id)
        .ok_or_else(|| format!("unknown job {job_id:?}"))?;
    send(
        stream,
        &json!({
            "v": PROTO_V,
            "id": id,
            "type": "job-status-ok",
            "ok": true,
            "job": {
                "id": job.id,
                "state": job.state,
                "error": job.error,
                "created_ms": job.created_ms,
                "started_ms": job.started_ms,
                "finished_ms": job.finished_ms,
                "bytes": job.result_body.len(),
            },
        }),
        &[],
    )
}

fn job_result(
    stream: &mut TcpStream,
    cfg: &Config,
    id: &Json,
    header: &Json,
) -> Result<(), String> {
    let job_id = header
        .get("job_id")
        .and_then(|x| x.as_str())
        .ok_or("job-result needs job_id")?;
    let (state, error, reply, body) = {
        let store = cfg.jobs.lock().unwrap_or_else(|e| e.into_inner());
        let job = store
            .jobs
            .get(job_id)
            .ok_or_else(|| format!("unknown job {job_id:?}"))?;
        (
            job.state.clone(),
            job.error.clone(),
            job.result_header.clone(),
            job.result_body.clone(),
        )
    };
    if state != "done" {
        return send_error(
            stream,
            id,
            if state == "failed" {
                "job-failed"
            } else {
                "not-ready"
            },
            &error.unwrap_or_else(|| format!("job is {state}")),
        );
    }
    let mut reply = reply.ok_or("job has no result")?;
    if let Some(o) = reply.as_object_mut() {
        o.insert("id".into(), id.clone());
        o.insert("type".into(), json!("job-result-ok"));
        o.insert("ok".into(), json!(true));
        o.insert("job_id".into(), json!(job_id));
    }
    send(stream, &reply, &body)
}

fn job_cancel(
    stream: &mut TcpStream,
    cfg: &Config,
    id: &Json,
    header: &Json,
) -> Result<(), String> {
    let job_id = header
        .get("job_id")
        .and_then(|x| x.as_str())
        .ok_or("job-cancel needs job_id")?;
    let mut store = cfg.jobs.lock().unwrap_or_else(|e| e.into_inner());
    let job = store
        .jobs
        .get_mut(job_id)
        .ok_or_else(|| format!("unknown job {job_id:?}"))?;
    let previous = job.state.clone();
    let accepted = if previous == "queued" {
        job.cancel.store(true, Ordering::SeqCst);
        job.state = "canceled".to_string();
        job.finished_ms = Some(now_ms());
        true
    } else if previous == "running" {
        job.cancel.store(true, Ordering::SeqCst);
        false
    } else {
        false
    };
    send(
        stream,
        &json!({
            "v": PROTO_V,
            "id": id,
            "type": "job-cancel-ok",
            "ok": true,
            "job_id": job_id,
            "accepted": accepted,
            "previous": previous,
            "state": job.state,
        }),
        &[],
    )
}

fn run(
    stream: &mut TcpStream,
    cfg: &Config,
    id: &Json,
    header: &Json,
    payload: &[u8],
) -> Result<(), String> {
    run_impl(stream, cfg, id, header, payload, 1, false, false, false)
}

fn run_batch(
    stream: &mut TcpStream,
    cfg: &Config,
    id: &Json,
    header: &Json,
    payload: &[u8],
) -> Result<(), String> {
    let frames = header
        .get("frames")
        .and_then(|x| x.as_u64())
        .ok_or("run-batch needs frames")? as usize;
    if frames == 0 || frames > MAX_BATCH_FRAMES {
        return send_error(
            stream,
            id,
            "bad-request",
            &format!("run-batch frames must be 1..={MAX_BATCH_FRAMES}"),
        );
    }
    run_impl(stream, cfg, id, header, payload, frames, true, false, false)
}

fn warm_prepare(
    stream: &mut TcpStream,
    cfg: &Config,
    id: &Json,
    header: &Json,
    payload: &[u8],
) -> Result<(), String> {
    let warmup = header
        .get("warmup")
        .and_then(|x| x.as_bool())
        .unwrap_or(true);
    run_impl(stream, cfg, id, header, payload, 1, false, true, warmup)
}

fn run_impl(
    stream: &mut TcpStream,
    cfg: &Config,
    id: &Json,
    header: &Json,
    payload: &[u8],
    frames: usize,
    batch: bool,
    prepare: bool,
    warmup: bool,
) -> Result<(), String> {
    let requested_target = header
        .get("target")
        .and_then(|x| x.as_str())
        .unwrap_or(cfg.target.as_str());
    if requested_target != cfg.target.as_str() {
        return send_error(
            stream,
            id,
            "target-mismatch",
            &format!(
                "runner is {}, request asked for {requested_target}",
                cfg.target.as_str()
            ),
        );
    }
    let outputs: Vec<String> = header
        .get("outputs")
        .and_then(|x| x.as_array())
        .ok_or("run needs outputs[]")?
        .iter()
        .map(|x| {
            x.as_str()
                .map(|s| s.to_string())
                .ok_or("outputs must be strings")
        })
        .collect::<Result<_, _>>()?;
    if outputs.is_empty() {
        return send_error(stream, id, "bad-request", "run needs at least one output");
    }
    let binary_output = header.get("output_format").and_then(|x| x.as_str()) == Some("binary");
    let output_u8 = header
        .get("output_u8")
        .and_then(|x| x.as_bool())
        .unwrap_or(false);
    let binary_payload = header.get("payload_format").and_then(|x| x.as_str()) == Some("binary");
    let body = body_head(payload, binary_payload)?;
    let listing = body
        .get("listing")
        .and_then(|x| x.as_str())
        .ok_or("run body needs listing")?;
    let payload_id = header
        .get("payload_id")
        .or_else(|| body.get("payload_id"))
        .and_then(|x| x.as_str());
    let feeds: Arc<Feeds> = if let Some(pid) = payload_id {
        let cache = cfg.payload_cache.lock().unwrap_or_else(|e| e.into_inner());
        cache
            .get(pid)
            .ok_or_else(|| format!("unknown-payload {pid}"))?
    } else if binary_payload {
        let payload_meta = body
            .get("payload")
            .ok_or("binary run body needs payload metadata")?;
        let pos = payload
            .iter()
            .position(|b| *b == b'\n')
            .ok_or("binary run body needs a metadata line")?;
        Arc::new(feeds_from_binary(payload_meta, &payload[pos + 1..]).map_err(|e| e.to_string())?)
    } else {
        let payload_json = body
            .get("payload")
            .ok_or("run body needs payload or payload_id")?;
        Arc::new(feeds_from_json(payload_json).map_err(|e| e.to_string())?)
    };
    let use_fp16 = body
        .get("use_fp16")
        .and_then(|x| x.as_bool())
        .unwrap_or(false);
    let opts = CompileOptions {
        target: cfg.target,
        outputs: outputs.clone(),
        asm: AsmOptions::new("."),
        use_fp16,
    };
    let artifact_key = artifact_key(listing, &outputs, &feeds, use_fp16, cfg.target);
    let cached_art = {
        let mut cache = cfg.artifact_cache.lock().unwrap_or_else(|e| e.into_inner());
        cache.get(&artifact_key)
    };
    let (artifact, compile_ms) = if let Some(artifact) = cached_art {
        (artifact, 0)
    } else {
        let t0 = Instant::now();
        let artifact = Arc::new(
            compile_program(listing, &opts, (*feeds).clone())
                .map_err(|e| format!("compile: {e}"))?,
        );
        let compile_ms = t0.elapsed().as_millis();
        let mut cache = cfg.artifact_cache.lock().unwrap_or_else(|e| e.into_inner());
        cache.put(artifact_key, artifact.clone());
        (artifact, compile_ms)
    };
    let dir = tempfile::Builder::new()
        .prefix("jotnar-gpu-runner-")
        .tempdir()
        .map_err(|e| format!("tempdir: {e}"))?;
    let key = cache_key(&artifact.source, cfg.target, use_fp16);
    let cache = cfg.cache_dir.join(&key);
    let exe = cache.join("job");
    let t_build = Instant::now();
    let mut cache_hit = exe.is_file();
    if !cache_hit {
        let _guard = cfg.build_lock.lock().unwrap_or_else(|e| e.into_inner());
        if !exe.is_file() {
            fs::create_dir_all(&cache)
                .map_err(|e| format!("cache dir {}: {e}", cache.display()))?;
            build(&artifact.source, &cache, "job", cfg.target)
                .map_err(|e| format!("build: {e}"))?;
        } else {
            cache_hit = true;
        }
    }
    let build_ms = t_build.elapsed().as_millis();

    if prepare && !(cfg.warm && artifact.warm_frame) {
        let timing = json!({
            "compile_ms": compile_ms,
            "build_ms": build_ms,
            "worker_start_ms": 0,
            "warmup_ms": 0,
            "warm": false,
            "warmup": false,
            "worker_reused": false,
            "cache_hit": cache_hit,
        });
        return send_prepare(stream, id, cfg, &artifact, timing);
    }

    if cfg.warm && artifact.warm_frame {
        let spec = warm::spec_from_artifact(&artifact, &exe, &key, cfg.target)?;
        let cache_key = spec.key.clone();
        let (worker, worker_started) = {
            let mut cache = cfg.warm_cache.lock().unwrap_or_else(|e| e.into_inner());
            cache.get_or_create(&spec)?
        };
        if prepare {
            let mut worker = worker.lock().unwrap_or_else(|e| e.into_inner());
            let worker_start_ms = if worker_started {
                worker.startup_ms()
            } else {
                0
            };
            let mut warmup_ms = 0;
            if warmup {
                let t = Instant::now();
                if let Err(e) = worker.run_frame(&feeds) {
                    drop(worker);
                    let mut cache = cfg.warm_cache.lock().unwrap_or_else(|e| e.into_inner());
                    cache.remove(&cache_key);
                    return Err(e);
                }
                warmup_ms = t.elapsed().as_millis();
            }
            drop(worker);
            let timing = json!({
                "compile_ms": compile_ms,
                "build_ms": build_ms,
                "worker_start_ms": worker_start_ms,
                "warmup_ms": warmup_ms,
                "warm": true,
                "warmup": warmup,
                "worker_reused": !worker_started,
                "cache_hit": cache_hit,
            });
            return send_prepare(stream, id, cfg, &artifact, timing);
        }
        let mut outputs = Vec::with_capacity(frames);
        let mut frame_ms = Vec::with_capacity(frames);
        let mut worker_start_ms = 0;
        {
            let mut worker = worker.lock().unwrap_or_else(|e| e.into_inner());
            for _ in 0..frames {
                let run = match worker.run_frame(&feeds) {
                    Ok(run) => run,
                    Err(e) => {
                        drop(worker);
                        let mut cache = cfg.warm_cache.lock().unwrap_or_else(|e| e.into_inner());
                        cache.remove(&cache_key);
                        return Err(e);
                    }
                };
                worker_start_ms = run.worker_start_ms;
                frame_ms.push(run.frame_ms);
                outputs.push(run.outputs);
            }
        }
        let run_ms: u128 = frame_ms.iter().sum();
        let timing = json!({
            "compile_ms": compile_ms,
            "build_ms": build_ms,
            "worker_start_ms": worker_start_ms,
            "frame_ms": if batch { json!(frame_ms) } else { json!(frame_ms[0]) },
            "run_ms": run_ms,
            "warm": true,
            "cache_hit": cache_hit,
        });
        if batch {
            return send_run_batch_outputs(
                stream,
                id,
                cfg,
                &artifact,
                outputs,
                timing,
                binary_output,
                output_u8,
            );
        }
        return send_run_outputs(
            stream,
            id,
            cfg,
            &artifact,
            outputs.pop().unwrap(),
            timing,
            binary_output,
            output_u8,
        );
    }

    let mut argv = Vec::new();
    for name in &artifact.inputs {
        let value = feeds
            .get(name)
            .ok_or_else(|| format!("payload missing input {name:?}"))?;
        let info = artifact
            .streams
            .get(name)
            .ok_or_else(|| format!("no stream info for input {name:?}"))?;
        let path = dir.path().join(format!("in_{name}.bin"));
        write_value(&path, value, info.kind, cfg.target)?;
        argv.push(path);
    }
    let mut output_paths = Vec::new();
    for name in &artifact.outputs {
        let path = dir.path().join(format!("out_{name}.bin"));
        output_paths.push((name.clone(), path.clone()));
        argv.push(path);
    }
    let mut outputs = Vec::with_capacity(frames);
    let mut frame_ms = Vec::with_capacity(frames);
    for _ in 0..frames {
        let t1 = Instant::now();
        let run = Command::new(&exe)
            .args(&argv)
            .output()
            .map_err(|e| format!("run: {e}"))?;
        let run_ms = t1.elapsed().as_millis();
        if !run.status.success() {
            return send_error(
                stream,
                id,
                "run-failed",
                &format!(
                    "exit {:?}: {}",
                    run.status.code(),
                    String::from_utf8_lossy(&run.stderr)
                ),
            );
        }
        let mut values = HashMap::new();
        for (name, path) in &output_paths {
            let info = artifact
                .streams
                .get(name)
                .ok_or_else(|| format!("no stream info for output {name:?}"))?;
            let value = read_value(path, info.kind, cfg.target, &info.shape)?;
            values.insert(name.clone(), value);
        }
        frame_ms.push(run_ms);
        outputs.push(values);
    }
    let run_ms: u128 = frame_ms.iter().sum();
    let timing = json!({
        "compile_ms": compile_ms,
        "build_ms": build_ms,
        "frame_ms": if batch { json!(frame_ms) } else { json!(frame_ms[0]) },
        "run_ms": run_ms,
        "cache_hit": cache_hit,
    });
    if batch {
        return send_run_batch_outputs(
            stream,
            id,
            cfg,
            &artifact,
            outputs,
            timing,
            binary_output,
            output_u8,
        );
    }
    send_run_outputs(
        stream,
        id,
        cfg,
        &artifact,
        outputs.pop().unwrap(),
        timing,
        binary_output,
        output_u8,
    )
}

fn artifact_key(
    listing: &str,
    outputs: &[String],
    feeds: &Feeds,
    use_fp16: bool,
    target: Target,
) -> String {
    let mut key = format!(
        "{:016x}|{}|{}",
        hash_bytes(listing.as_bytes()),
        use_fp16,
        target.as_str()
    );
    for o in outputs {
        key.push('|');
        key.push_str(o);
    }
    for (name, value) in feeds {
        key.push_str(&format!("|{}:{:?}:{:?}", name, value.kind(), value.shape()));
    }
    key
}

fn cache_key(source: &str, target: Target, use_fp16: bool) -> String {
    let arch = if target == Target::Hip {
        std::env::var("HIP_ARCH").unwrap_or_default()
    } else {
        std::env::var("CUDA_ARCH").unwrap_or_default()
    };
    format!(
        "{:016x}-{}-{}-{}",
        jotnar::value::hash_bytes(source.as_bytes()),
        target.as_str(),
        arch,
        if use_fp16 { 1 } else { 0 }
    )
}

fn send_prepare(
    stream: &mut TcpStream,
    id: &Json,
    cfg: &Config,
    artifact: &Artifact,
    timing: Json,
) -> Result<(), String> {
    let field = |k: &str| timing.get(k).cloned().unwrap_or(Json::Null);
    let header = json!({
        "v": PROTO_V,
        "id": id,
        "type": "warm-prepare-ok",
        "ok": true,
        "target": cfg.target.as_str(),
        "runtime_version": RUNTIME_VERSION,
        "outputs": artifact.outputs,
        "warm": field("warm"),
        "worker_start_ms": field("worker_start_ms"),
        "warmup_ms": field("warmup_ms"),
        "warmup": field("warmup"),
        "worker_reused": field("worker_reused"),
        "compile_ms": field("compile_ms"),
        "build_ms": field("build_ms"),
        "cache_hit": field("cache_hit"),
    });
    send(stream, &header, &[])
}

fn send_run_batch_outputs(
    stream: &mut TcpStream,
    id: &Json,
    cfg: &Config,
    artifact: &Artifact,
    frames: Vec<HashMap<String, Value>>,
    timing: Json,
    binary: bool,
    u8_output: bool,
) -> Result<(), String> {
    let field = |k: &str| timing.get(k).cloned().unwrap_or(Json::Null);
    let static_shapes = artifact.outputs.iter().all(|n| {
        artifact
            .streams
            .get(n)
            .map(|s| s.shape.iter().all(|d| d.is_some()))
            .unwrap_or(false)
    });
    if binary && static_shapes {
        let mut descriptors = Vec::new();
        let mut body = Vec::new();
        for (frame, values) in frames.iter().enumerate() {
            for name in &artifact.outputs {
                let value = values
                    .get(name)
                    .ok_or_else(|| format!("missing output {name:?} in frame {frame}"))?;
                let info = artifact
                    .streams
                    .get(name)
                    .ok_or_else(|| format!("no stream info for output {name:?}"))?;
                let (bytes, kind) = if u8_output {
                    (encode_value_u8(value)?, "U8")
                } else {
                    let bytes = encode_value(value, info.kind, cfg.target)?;
                    let kind = match info.kind {
                        CKind::F if matches!(cfg.target, Target::Cuda | Target::Hip) => "F32",
                        CKind::F => "F64",
                        CKind::I => "I64",
                        CKind::T => "T",
                    };
                    (bytes, kind)
                };
                let off = body.len();
                body.extend_from_slice(&bytes);
                let shape: Vec<usize> = info.shape.iter().map(|d| d.unwrap()).collect();
                descriptors.push(json!({
                    "frame": frame,
                    "name": name,
                    "kind": kind,
                    "shape": shape,
                    "off": off,
                    "bytes": bytes.len(),
                }));
            }
        }
        let header = json!({
            "v": PROTO_V,
            "id": id,
            "type": "run-batch-ok",
            "ok": true,
            "target": cfg.target.as_str(),
            "runtime_version": RUNTIME_VERSION,
            "output_format": "binary",
            "frames": frames.len(),
            "outputs": descriptors,
            "compile_ms": field("compile_ms"),
            "build_ms": field("build_ms"),
            "worker_start_ms": field("worker_start_ms"),
            "frame_ms": field("frame_ms"),
            "run_ms": field("run_ms"),
            "warm": field("warm"),
            "cache_hit": field("cache_hit"),
        });
        return send(stream, &header, &body);
    }

    let mut out = Vec::with_capacity(frames.len());
    for values in &frames {
        let mut frame = serde_json::Map::new();
        for name in &artifact.outputs {
            let value = values
                .get(name)
                .ok_or_else(|| format!("missing output {name:?}"))?;
            frame.insert(
                name.clone(),
                value_to_json(value).map_err(|e| e.to_string())?,
            );
        }
        out.push(json!({"outputs": frame}));
    }
    let body = json!({"frames": out, "timing": timing});
    let header = json!({
        "v": PROTO_V,
        "id": id,
        "type": "run-batch-ok",
        "ok": true,
        "target": cfg.target.as_str(),
        "runtime_version": RUNTIME_VERSION,
        "frames": frames.len(),
        "compile_ms": field("compile_ms"),
        "build_ms": field("build_ms"),
        "worker_start_ms": field("worker_start_ms"),
        "frame_ms": field("frame_ms"),
        "run_ms": field("run_ms"),
        "warm": field("warm"),
        "cache_hit": field("cache_hit"),
    });
    send(stream, &header, body.to_string().as_bytes())
}

fn send_run_outputs(
    stream: &mut TcpStream,
    id: &Json,
    cfg: &Config,
    artifact: &Artifact,
    values: HashMap<String, Value>,
    timing: Json,
    binary: bool,
    u8_output: bool,
) -> Result<(), String> {
    let field = |k: &str| timing.get(k).cloned().unwrap_or(Json::Null);
    let static_shapes = artifact.outputs.iter().all(|n| {
        artifact
            .streams
            .get(n)
            .map(|s| s.shape.iter().all(|d| d.is_some()))
            .unwrap_or(false)
    });
    if binary && static_shapes {
        let mut descriptors = Vec::new();
        let mut body = Vec::new();
        for name in &artifact.outputs {
            let value = values
                .get(name)
                .ok_or_else(|| format!("missing output {name:?}"))?;
            let info = artifact
                .streams
                .get(name)
                .ok_or_else(|| format!("no stream info for output {name:?}"))?;
            let (bytes, kind) = if u8_output {
                (encode_value_u8(value)?, "U8")
            } else {
                let bytes = encode_value(value, info.kind, cfg.target)?;
                let kind = match info.kind {
                    CKind::F if matches!(cfg.target, Target::Cuda | Target::Hip) => "F32",
                    CKind::F => "F64",
                    CKind::I => "I64",
                    CKind::T => "T",
                };
                (bytes, kind)
            };
            let off = body.len();
            body.extend_from_slice(&bytes);
            let shape: Vec<usize> = info.shape.iter().map(|d| d.unwrap()).collect();
            descriptors.push(json!({
                "name": name,
                "kind": kind,
                "shape": shape,
                "off": off,
                "bytes": bytes.len(),
            }));
        }
        let header = json!({
            "v": PROTO_V,
            "id": id,
            "type": "run-ok",
            "ok": true,
            "target": cfg.target.as_str(),
            "runtime_version": RUNTIME_VERSION,
            "output_format": "binary",
            "outputs": descriptors,
            "compile_ms": field("compile_ms"),
            "build_ms": field("build_ms"),
            "worker_start_ms": field("worker_start_ms"),
            "frame_ms": field("frame_ms"),
            "run_ms": field("run_ms"),
            "warm": field("warm"),
            "cache_hit": field("cache_hit"),
        });
        return send(stream, &header, &body);
    }

    let mut out = serde_json::Map::new();
    for name in &artifact.outputs {
        let value = values
            .get(name)
            .ok_or_else(|| format!("missing output {name:?}"))?;
        out.insert(
            name.clone(),
            value_to_json(value).map_err(|e| e.to_string())?,
        );
    }
    let body = json!({"outputs": out, "timing": timing});
    let header = json!({
        "v": PROTO_V,
        "id": id,
        "type": "run-ok",
        "ok": true,
        "target": cfg.target.as_str(),
        "runtime_version": RUNTIME_VERSION,
        "compile_ms": field("compile_ms"),
        "build_ms": field("build_ms"),
        "worker_start_ms": field("worker_start_ms"),
        "frame_ms": field("frame_ms"),
        "run_ms": field("run_ms"),
        "warm": field("warm"),
        "cache_hit": field("cache_hit"),
    });
    send(stream, &header, body.to_string().as_bytes())
}

fn encode_value(value: &Value, kind: CKind, target: Target) -> Result<Vec<u8>, String> {
    let mut bytes = Vec::new();
    match kind {
        CKind::F => {
            if matches!(target, Target::Cuda | Target::Hip) {
                let a = value.as_f32("runner input").map_err(|e| e.to_string())?;
                for x in a.iter() {
                    bytes.extend_from_slice(&x.to_le_bytes());
                }
            } else {
                let a = value.as_f64("runner input").map_err(|e| e.to_string())?;
                for x in a.iter() {
                    bytes.extend_from_slice(&x.to_le_bytes());
                }
            }
        }
        CKind::I => {
            let a = value.as_i64("runner input").map_err(|e| e.to_string())?;
            for x in a.iter() {
                bytes.extend_from_slice(&x.to_le_bytes());
            }
        }
        CKind::T => return Err("T streams are not supported by the GPU runner".to_string()),
    }
    Ok(bytes)
}

fn encode_value_u8(value: &Value) -> Result<Vec<u8>, String> {
    let bytes = match value {
        Value::F32(a) => a
            .iter()
            .map(|v| v.round().clamp(0.0, 255.0) as u8)
            .collect::<Vec<_>>(),
        Value::F64(a) => a
            .iter()
            .map(|v| v.round().clamp(0.0, 255.0) as u8)
            .collect::<Vec<_>>(),
        Value::I64(a) => a
            .iter()
            .map(|v| (*v).clamp(0, 255) as u8)
            .collect::<Vec<_>>(),
        Value::U8(a) => a.iter().copied().collect::<Vec<_>>(),
        Value::Bool(a) => a.iter().map(|v| u8::from(*v)).collect::<Vec<_>>(),
        _ => return Err("u8 output conversion needs a numeric stream".to_string()),
    };
    Ok(bytes)
}

fn write_value(path: &Path, value: &Value, kind: CKind, target: Target) -> Result<(), String> {
    let bytes = encode_value(value, kind, target)?;
    fs::write(path, bytes).map_err(|e| format!("write {}: {e}", path.display()))
}

fn decode_value(
    bytes: &[u8],
    kind: CKind,
    target: Target,
    shape: &[Option<usize>],
) -> Result<Value, String> {
    let shape: Vec<usize> = shape
        .iter()
        .map(|x| x.ok_or_else(|| "dynamic output shape is not supported".to_string()))
        .collect::<Result<_, _>>()?;
    let n = shape.iter().product::<usize>();
    match kind {
        CKind::F => {
            if matches!(target, Target::Cuda | Target::Hip) {
                let data = bytes
                    .chunks_exact(4)
                    .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                    .collect::<Vec<_>>();
                if data.len() != n {
                    return Err(format!("output has {} floats, expected {n}", data.len()));
                }
                let arr = array_from_vec(&shape, data).map_err(|e| e.to_string())?;
                Ok(Value::F32(arr))
            } else {
                let data = bytes
                    .chunks_exact(8)
                    .map(|b| f64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]]))
                    .collect::<Vec<_>>();
                if data.len() != n {
                    return Err(format!("output has {} doubles, expected {n}", data.len()));
                }
                let arr = array_from_vec(&shape, data).map_err(|e| e.to_string())?;
                Ok(Value::F64(arr))
            }
        }
        CKind::I => {
            let data = bytes
                .chunks_exact(8)
                .map(|b| i64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]]))
                .collect::<Vec<_>>();
            if data.len() != n {
                return Err(format!("output has {} ints, expected {n}", data.len()));
            }
            let arr = array_from_vec(&shape, data).map_err(|e| e.to_string())?;
            Ok(Value::I64(arr))
        }
        CKind::T => Err("T streams are not supported by the GPU runner".to_string()),
    }
}

fn read_value(
    path: &Path,
    kind: CKind,
    target: Target,
    shape: &[Option<usize>],
) -> Result<Value, String> {
    let bytes = fs::read(path).map_err(|e| format!("read {}: {e}", path.display()))?;
    decode_value(&bytes, kind, target, shape)
}

fn send(stream: &mut TcpStream, header: &Json, payload: &[u8]) -> Result<(), String> {
    let frame = encode_frame(header, payload).map_err(|e| e.to_string())?;
    stream
        .write_all(&frame)
        .map_err(|e| format!("write response: {e}"))
}

fn send_error(stream: &mut TcpStream, id: &Json, code: &str, msg: &str) -> Result<(), String> {
    send(
        stream,
        &json!({
            "v": PROTO_V,
            "id": id,
            "type": "error",
            "ok": false,
            "error": {"code": code, "msg": msg},
        }),
        &[],
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn value_file_roundtrip() {
        let dir = tempfile::tempdir().unwrap();

        let p = dir.path().join("ints.bin");
        let v = Value::I64(array_from_vec(&[2, 2], vec![1, 2, 3, 4]).unwrap());
        write_value(&p, &v, CKind::I, Target::Cuda).unwrap();
        let r = read_value(&p, CKind::I, Target::Cuda, &[Some(2), Some(2)]).unwrap();
        assert_eq!(
            r.as_i64("test")
                .unwrap()
                .iter()
                .copied()
                .collect::<Vec<_>>(),
            vec![1, 2, 3, 4]
        );

        let p = dir.path().join("floats.bin");
        let v = Value::F64(array_from_vec(&[2], vec![1.5, 2.5]).unwrap());
        write_value(&p, &v, CKind::F, Target::Cuda).unwrap();
        let r = read_value(&p, CKind::F, Target::Cuda, &[Some(2)]).unwrap();
        assert_eq!(
            r.as_f64("test")
                .unwrap()
                .iter()
                .copied()
                .collect::<Vec<_>>(),
            vec![1.5, 2.5]
        );
    }
}

#[cfg(test)]
mod bind_tests {
    use super::is_loopback_bind;

    #[test]
    fn loopback_bind_guard() {
        assert!(is_loopback_bind("127.0.0.1:9997"));
        assert!(is_loopback_bind("localhost:9997"));
        assert!(is_loopback_bind("[::1]:9997"));
        assert!(!is_loopback_bind("0.0.0.0:9997"));
        assert!(!is_loopback_bind("[::]:9997"));
    }
}

#[test]
fn payload_cache_reuses_and_evicts() {
    let mut cache = PayloadCache::new(8);
    let feeds = Feeds::new();
    assert!(cache.get("a").is_none());
    cache.put("a".to_string(), feeds.clone(), 4);
    assert!(cache.get("a").is_some());
    cache.put("a".to_string(), feeds.clone(), 4);
    cache.put("b".to_string(), feeds, 8);
    assert!(cache.get("a").is_none());
    assert!(cache.get("b").is_some());
}

#[cfg(test)]
mod artifact_cache_tests {
    use super::ArtifactCache;
    use std::sync::Arc;

    fn artifact(name: &str) -> Arc<jotnar::codegen::Artifact> {
        Arc::new(jotnar::codegen::Artifact {
            source: name.to_string(),
            streams: Default::default(),
            inputs: Vec::new(),
            outputs: Vec::new(),
            source_inputs: Vec::new(),
            source_outputs: Vec::new(),
            config: jotnar::ir::Config::default(),
            n_argv: 0,
            warm_frame: false,
        })
    }

    #[test]
    fn artifact_cache_evicts_lru() {
        let mut cache = ArtifactCache::new(2);
        cache.put("a".into(), artifact("a"));
        cache.put("b".into(), artifact("b"));
        assert!(cache.get("a").is_some());
        cache.put("c".into(), artifact("c"));
        assert!(cache.get("b").is_none());
        assert!(cache.get("a").is_some());
        assert!(cache.get("c").is_some());
    }
}

#[cfg(test)]
mod job_store_tests {
    use super::*;

    fn job(id: &str, state: &str) -> AsyncJob {
        AsyncJob {
            id: id.to_string(),
            state: state.to_string(),
            error: None,
            created_ms: 0,
            started_ms: None,
            finished_ms: None,
            result_header: None,
            result_body: Vec::new(),
            cancel: Arc::new(AtomicBool::new(false)),
        }
    }

    #[test]
    fn job_store_evicts_completed_but_keeps_active() {
        let mut store = JobStore::new(2);
        store.insert(job("run", "running"));
        store.insert(job("done-a", "done"));
        store.insert(job("done-b", "done"));
        assert!(store.jobs.contains_key("run"));
        assert!(!store.jobs.contains_key("done-a"));
        assert!(store.jobs.contains_key("done-b"));
    }
}

#[cfg(test)]
mod async_job_tests {
    use super::*;
    use std::io::{Read, Write};
    use std::net::{TcpListener, TcpStream};

    fn test_config(dir: &Path) -> Config {
        Config {
            bind: "127.0.0.1:0".to_string(),
            token: None,
            target: Target::Cuda,
            max_frame: DEFAULT_MAX_FRAME,
            cache_dir: dir.to_path_buf(),
            build_lock: Arc::new(Mutex::new(())),
            mdns: false,
            warm: false,
            warm_idle: None,
            warm_cache: Arc::new(Mutex::new(warm::WarmCache::new(1))),
            payload_cache: Arc::new(Mutex::new(PayloadCache::new(1024))),
            artifact_cache: Arc::new(Mutex::new(ArtifactCache::new(2))),
            jobs: Arc::new(Mutex::new(JobStore::new(4))),
            self_addr: "127.0.0.1:0".to_string(),
            started_ms: now_ms() as u64,
        }
    }

    fn read_frame(stream: &mut TcpStream) -> (Json, Vec<u8>) {
        let mut hdr = [0u8; 4];
        stream.read_exact(&mut hdr).unwrap();
        let n = u32::from_be_bytes(hdr) as usize;
        let mut body = vec![0u8; n];
        stream.read_exact(&mut body).unwrap();
        let mut full = Vec::with_capacity(4 + n);
        full.extend_from_slice(&hdr);
        full.extend_from_slice(&body);
        let (header, payload) = decode_frame(&full).unwrap();
        (header, payload.to_vec())
    }

    #[test]
    fn missing_async_jobs_return_unknown_job_frames() {
        let dir = tempfile::tempdir().unwrap();
        let cfg = test_config(dir.path());
        for typ in ["job-status", "job-result", "job-cancel"] {
            let listener = TcpListener::bind("127.0.0.1:0").unwrap();
            let addr = listener.local_addr().unwrap();
            let mut client = TcpStream::connect(addr).unwrap();
            let (server, _) = listener.accept().unwrap();
            let thread_cfg = cfg.clone();
            let handler = std::thread::spawn(move || handle(server, &thread_cfg));
            let frame = encode_frame(
                &json!({"v": PROTO_V, "id": 1, "type": typ, "job_id": "missing"}),
                &[],
            )
            .unwrap();
            client.write_all(&frame).unwrap();
            let (reply, _) = read_frame(&mut client);
            handler.join().unwrap().unwrap();
            assert_eq!(reply["type"], "error", "{typ}: {reply}");
            assert_eq!(reply["error"]["code"], "unknown-job", "{typ}: {reply}");
        }
    }
}
