//! Model-blind native/process executor.
//!
//! The CPU node executes Jotnar listings; the GPU runner compiles listings for
//! a GPU. This runner fills the missing native lane: it stages files named by
//! the controller, runs an allow-listed local program with a relative argv, and
//! returns the requested output files through the same async job contract.
//!
//! It knows nothing about DOOM, RV32IM, model manifests, or stage graphs. The
//! workload controller owns that plan; this process only runs one command.
//!
//! ```text
//! submit      header: command[], inputs[], outputs[], timeout_ms?
//!             body:   concatenated input bytes
//! job-status  -> queued/running/done/failed/canceled
//! job-result  -> output_format=files + descriptors + concatenated bytes
//! job-cancel  -> queued jobs cancel immediately; running jobs are killed
//! ```

use std::collections::{HashMap, VecDeque};
use std::fs;
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::{Component, Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use jotnar::protocol::{decode_frame, encode_frame, PROTO_V, RUNTIME_VERSION};
use mdns_sd::{ServiceDaemon, ServiceInfo};
use serde_json::{json, Value as Json};
use sha2::{Digest, Sha256};

const DEFAULT_BIND: &str = "127.0.0.1:9998";
const DEFAULT_MAX_FRAME: usize = 512 * 1024 * 1024;
const DEFAULT_TIMEOUT_MS: u64 = 600_000;
const MAX_TIMEOUT_MS: u64 = 24 * 60 * 60 * 1000;
const RETAIN_JOBS: usize = 16;
const MAX_LOG: usize = 128 * 1024;

#[derive(Clone)]
struct Config {
    bind: String,
    token: Option<String>,
    max_frame: usize,
    work_root: PathBuf,
    allow: Vec<PathBuf>,
    keep_work: bool,
    mdns: bool,
    jobs: Arc<Mutex<JobStore>>,
    started_ms: u64,
}

struct AsyncJob {
    id: String,
    state: String,
    error: Option<String>,
    created_ms: u128,
    started_ms: Option<u128>,
    finished_ms: Option<u128>,
    result_header: Option<Json>,
    result_body: Arc<Vec<u8>>,
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
        let id = format!("native-{}-{}", now_ms(), self.next_id);
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

fn sha256_hex(bytes: &[u8]) -> String {
    let mut h = Sha256::new();
    h.update(bytes);
    format!("{:x}", h.finalize())
}

fn usage() -> &'static str {
    "jotnar-native-runner — model-blind native/process executor\n\
     \n\
     usage:\n\
       jotnar-native-runner --allow PROGRAM [--allow PROGRAM]... \\\n\
         [--bind ADDR] [--token SECRET] [--work-root DIR] [--max-frame BYTES] \\\n\
         [--timeout-ms N] [--keep-work] [--mdns] [--port-file PATH]\n\
     \n\
     A job stages named input files, runs an allow-listed program with a\n\
     relative argv in a private work directory, and returns named output files.\n\
     It does not know about models, stage graphs, or DOOM."
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

fn main() -> std::process::ExitCode {
    match real_main() {
        Ok(()) => std::process::ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("{e}");
            std::process::ExitCode::FAILURE
        }
    }
}

fn real_main() -> Result<(), String> {
    let mut bind = DEFAULT_BIND.to_string();
    let mut token = None;
    let mut max_frame = DEFAULT_MAX_FRAME;
    let mut work_root = std::env::temp_dir().join("jotnar-native-runner");
    let mut allow = Vec::new();
    let mut keep_work = false;
    let mut mdns = false;
    let mut port_file = None;
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
            "--work-root" => {
                i += 1;
                work_root = PathBuf::from(args.get(i).ok_or("--work-root needs a path")?);
            }
            "--allow" => {
                i += 1;
                let p = args.get(i).ok_or("--allow needs a program path")?;
                allow.push(PathBuf::from(p));
            }
            "--keep-work" => keep_work = true,
            "--mdns" => mdns = true,
            "--port-file" => {
                i += 1;
                port_file = Some(PathBuf::from(
                    args.get(i).ok_or("--port-file needs a path")?,
                ));
            }
            "-h" | "--help" => {
                println!("{}", usage());
                return Ok(());
            }
            other => return Err(format!("unknown argument '{other}'\n\n{}", usage())),
        }
        i += 1;
    }
    if allow.is_empty() {
        return Err(format!(
            "at least one --allow PROGRAM is required\n\n{}",
            usage()
        ));
    }
    let allow: Vec<PathBuf> = allow
        .into_iter()
        .map(|p| fs::canonicalize(&p).map_err(|e| format!("allow {}: {e}", p.display())))
        .collect::<Result<_, _>>()?;
    if token.as_deref().map(str::trim).is_some_and(str::is_empty) {
        return Err("--token must not be empty".to_string());
    }
    if !is_loopback_bind(&bind) && token.is_none() {
        return Err(format!(
            "refusing non-loopback bind {bind} without --token\n\n{}",
            usage()
        ));
    }
    fs::create_dir_all(&work_root)
        .map_err(|e| format!("create work root {}: {e}", work_root.display()))?;
    let cfg = Config {
        bind,
        token,
        max_frame: max_frame.max(1024),
        work_root,
        allow,
        keep_work,
        mdns,
        jobs: Arc::new(Mutex::new(JobStore::new(RETAIN_JOBS))),
        started_ms: now_ms() as u64,
    };
    let listener = TcpListener::bind(&cfg.bind).map_err(|e| format!("bind {}: {e}", cfg.bind))?;
    let port = listener
        .local_addr()
        .map_err(|e| format!("bound address: {e}"))?
        .port();
    if let Some(path) = &port_file {
        fs::write(path, format!("{port}\n"))
            .map_err(|e| format!("write port file {}: {e}", path.display()))?;
    }
    let _mdns = if cfg.mdns {
        match start_mdns(&cfg, port) {
            Ok(daemon) => {
                eprintln!("mdns: advertising _jotnar-native._tcp on port {port}");
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
        "native-runner bind={} programs={} runtime={}",
        cfg.bind,
        cfg.allow
            .iter()
            .filter_map(|p| p.file_name().and_then(|x| x.to_str()))
            .collect::<Vec<_>>()
            .join(","),
        RUNTIME_VERSION
    );
    for conn in listener.incoming() {
        match conn {
            Ok(stream) => {
                let cfg = cfg.clone();
                thread::spawn(move || {
                    if let Err(e) = handle(stream, &cfg) {
                        eprintln!("native-runner request: {e}");
                    }
                });
            }
            Err(e) => eprintln!("native-runner accept: {e}"),
        }
    }
    Ok(())
}

fn start_mdns(cfg: &Config, port: u16) -> Result<ServiceDaemon, String> {
    let daemon = ServiceDaemon::new().map_err(|e| e.to_string())?;
    let mut props = HashMap::new();
    props.insert("target".to_string(), "process".to_string());
    props.insert(
        "programs".to_string(),
        cfg.allow
            .iter()
            .filter_map(|p| p.file_name().and_then(|x| x.to_str()))
            .collect::<Vec<_>>()
            .join(","),
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
    props.insert(
        "platform".to_string(),
        format!("{}-{}", std::env::consts::OS, std::env::consts::ARCH),
    );
    let host = format!("jotnar-native-{}.local.", std::process::id());
    let instance = format!("Jotnar Native {}", std::process::id());
    let info = ServiceInfo::new(
        "_jotnar-native._tcp.local.",
        &instance,
        &host,
        "",
        port,
        props,
    )
    .map_err(|e| e.to_string())?
    .enable_addr_auto();
    daemon.register(info).map_err(|e| e.to_string())?;
    Ok(daemon)
}

fn send(stream: &mut TcpStream, header: &Json, body: &[u8]) -> Result<(), String> {
    let frame = encode_frame(header, body).map_err(|e| e.to_string())?;
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
    send(
        stream,
        &json!({
            "v": PROTO_V,
            "id": id,
            "type": "health-ok",
            "ok": true,
            "schema": "jotnar.executor.health/1",
            "kind": "native",
            "node_id": format!("Jotnar Native {}", std::process::id()),
            "target": "process",
            "platform": format!("{}-{}", std::env::consts::OS, std::env::consts::ARCH),
            "programs": cfg.allow.iter()
                .filter_map(|p| p.file_name().and_then(|x| x.to_str()))
                .collect::<Vec<_>>(),
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
                "warm_workers": 0,
                "warm_artifacts": 0,
            },
            "jobs": jobs,
        }),
        &[],
    )
}

fn handle(mut stream: TcpStream, cfg: &Config) -> Result<(), String> {
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
        "ping" => send(
            &mut stream,
            &json!({
                "v": PROTO_V,
                "id": id,
                "type": "pong",
                "ok": true,
                "target": "process",
                "platform": format!("{}-{}", std::env::consts::OS, std::env::consts::ARCH),
                "programs": cfg.allow.iter()
                    .filter_map(|p| p.file_name().and_then(|x| x.to_str()))
                    .collect::<Vec<_>>(),
                "runtime_version": RUNTIME_VERSION,
            }),
            &[],
        ),
        "health" => {
            if let Err(e) = send_health(&mut stream, cfg, &id) {
                send_error(&mut stream, &id, "health-failed", &e)?;
            }
            Ok(())
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
        other => send_error(
            &mut stream,
            &id,
            "unknown-type",
            &format!("no such message {other:?}"),
        ),
    }
}

fn submit_job(
    stream: &mut TcpStream,
    cfg: &Config,
    id: &Json,
    header: &Json,
    payload: &[u8],
) -> Result<(), String> {
    let command: Vec<String> = header
        .get("command")
        .and_then(|x| x.as_array())
        .ok_or("submit needs command[]")?
        .iter()
        .map(|x| {
            x.as_str()
                .map(|s| s.to_string())
                .ok_or("command must be strings")
        })
        .collect::<Result<_, _>>()?;
    if command.is_empty() {
        return send_error(stream, id, "bad-request", "submit needs a command");
    }
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
    for name in &outputs {
        if !is_safe_relative(name) {
            return send_error(
                stream,
                id,
                "bad-request",
                &format!("unsafe output {name:?}"),
            );
        }
    }
    let inputs = match parse_inputs(header, payload) {
        Ok(x) => x,
        Err(e) => return send_error(stream, id, "bad-request", &e),
    };
    if let Err(e) = prepare_job(cfg, &command, &inputs) {
        return send_error(stream, id, "bad-request", &e);
    }
    let timeout_ms = header
        .get("timeout_ms")
        .and_then(|x| x.as_u64())
        .unwrap_or(DEFAULT_TIMEOUT_MS)
        .clamp(1, MAX_TIMEOUT_MS);
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
            result_body: Arc::new(Vec::new()),
            cancel: cancel.clone(),
        });
        (job_id, cancel)
    };
    let (job_id, cancel) = job_id;
    let thread_cfg = cfg.clone();
    let thread_header = header.clone();
    let thread_payload = payload.to_vec();
    let thread_id = job_id.clone();
    thread::spawn(move || {
        if cancel.load(Ordering::SeqCst) {
            finish_job(&thread_cfg, &thread_id, "canceled", None, None, Vec::new());
            return;
        }
        mark_running(&thread_cfg, &thread_id);
        let result = run_process(
            &thread_cfg,
            &thread_id,
            &thread_header,
            &thread_payload,
            timeout_ms,
            &cancel,
        );
        match result {
            Ok(result) => finish_job(
                &thread_cfg,
                &thread_id,
                "done",
                None,
                Some(result.header),
                result.body,
            ),
            Err(e) if cancel.load(Ordering::SeqCst) => finish_job(
                &thread_cfg,
                &thread_id,
                "canceled",
                Some(e),
                None,
                Vec::new(),
            ),
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

fn parse_inputs<'a>(
    header: &'a Json,
    payload: &'a [u8],
) -> Result<Vec<(&'a str, &'a [u8])>, String> {
    let mut out = Vec::new();
    let Some(inputs) = header.get("inputs").and_then(|x| x.as_array()) else {
        return Ok(out);
    };
    let mut seen = std::collections::BTreeSet::new();
    for input in inputs {
        let name = input
            .get("name")
            .and_then(|x| x.as_str())
            .ok_or("input descriptor needs name")?;
        if !is_safe_relative(name) {
            return Err(format!("unsafe input {name:?}"));
        }
        if !seen.insert(name.to_string()) {
            return Err(format!("duplicate input {name:?}"));
        }
        let off = input.get("off").and_then(|x| x.as_u64()).unwrap_or(0) as usize;
        let len = input
            .get("bytes")
            .and_then(|x| x.as_u64())
            .ok_or_else(|| format!("input {name:?} has no bytes"))? as usize;
        let end = off
            .checked_add(len)
            .ok_or_else(|| format!("input {name:?} range overflows"))?;
        if end > payload.len() {
            return Err(format!(
                "input {name:?} range {off}..{end} exceeds body {}",
                payload.len()
            ));
        }
        let bytes = &payload[off..end];
        if let Some(want) = input.get("sha256").and_then(|x| x.as_str()) {
            let got = sha256_hex(bytes);
            if got != want {
                return Err(format!("input {name:?} hash {got} != {want}"));
            }
        }
        out.push((name, bytes));
    }
    Ok(out)
}

fn prepare_job(cfg: &Config, command: &[String], inputs: &[(&str, &[u8])]) -> Result<(), String> {
    let program =
        fs::canonicalize(&command[0]).map_err(|e| format!("program {}: {e}", command[0]))?;
    if !cfg.allow.iter().any(|p| p == &program) {
        return Err(format!(
            "program {} is not in the allowlist ({})",
            program.display(),
            cfg.allow
                .iter()
                .map(|p| p.display().to_string())
                .collect::<Vec<_>>()
                .join(", ")
        ));
    }
    let _ = inputs;
    Ok(())
}

struct ProcessResult {
    header: Json,
    body: Vec<u8>,
}

fn run_process(
    cfg: &Config,
    job_id: &str,
    header: &Json,
    payload: &[u8],
    timeout_ms: u64,
    cancel: &AtomicBool,
) -> Result<ProcessResult, String> {
    let command: Vec<String> = header
        .get("command")
        .and_then(|x| x.as_array())
        .ok_or("submit needs command[]")?
        .iter()
        .map(|x| {
            x.as_str()
                .map(|s| s.to_string())
                .ok_or("command must be strings")
        })
        .collect::<Result<_, _>>()?;
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
    let inputs = parse_inputs(header, payload)?;
    prepare_job(cfg, &command, &inputs)?;
    let program =
        fs::canonicalize(&command[0]).map_err(|e| format!("program {}: {e}", command[0]))?;

    let work = cfg.work_root.join(format!("job-{job_id}"));
    if work.exists() {
        fs::remove_dir_all(&work).map_err(|e| format!("clear {}: {e}", work.display()))?;
    }
    fs::create_dir_all(&work).map_err(|e| format!("create {}: {e}", work.display()))?;
    let cleanup = || {
        if !cfg.keep_work {
            let _ = fs::remove_dir_all(&work);
        }
    };
    for (name, bytes) in &inputs {
        let path = safe_join(&work, name)?;
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).map_err(|e| format!("create {}: {e}", parent.display()))?;
        }
        fs::write(&path, bytes).map_err(|e| format!("write {}: {e}", path.display()))?;
    }
    let mut child = match Command::new(&program)
        .args(&command[1..])
        .current_dir(&work)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
    {
        Ok(c) => c,
        Err(e) => {
            cleanup();
            return Err(format!("spawn {}: {e}", program.display()));
        }
    };
    let stdout_pipe = child.stdout.take();
    let stderr_pipe = child.stderr.take();
    let out_thread = stdout_pipe.map(|p| thread::spawn(move || drain_capped(p)));
    let err_thread = stderr_pipe.map(|p| thread::spawn(move || drain_capped(p)));
    let deadline = Instant::now() + Duration::from_millis(timeout_ms);
    let status = loop {
        if cancel.load(Ordering::SeqCst) {
            let _ = child.kill();
            let _ = child.wait();
            cleanup();
            return Err("job canceled".to_string());
        }
        match child.try_wait() {
            Ok(Some(status)) => break status,
            Ok(None) => {}
            Err(e) => {
                let _ = child.kill();
                let _ = child.wait();
                cleanup();
                return Err(format!("wait: {e}"));
            }
        }
        if Instant::now() >= deadline {
            let _ = child.kill();
            let _ = child.wait();
            cleanup();
            return Err(format!("timed out after {timeout_ms} ms"));
        }
        thread::sleep(Duration::from_millis(25));
    };
    let stdout = out_thread.and_then(|t| t.join().ok()).unwrap_or_default();
    let stderr = err_thread.and_then(|t| t.join().ok()).unwrap_or_default();
    let exit_code = status.code().unwrap_or(-1);
    if !status.success() {
        cleanup();
        let tail = String::from_utf8_lossy(&stderr);
        return Err(format!(
            "process exited {exit_code}: {}",
            tail.chars().take(2048).collect::<String>()
        ));
    }
    let mut body = Vec::new();
    let mut descs = Vec::new();
    for name in &outputs {
        let path = safe_join(&work, name)?;
        let bytes = match fs::read(&path) {
            Ok(b) => b,
            Err(e) => {
                cleanup();
                return Err(format!("read output {}: {e}", path.display()));
            }
        };
        if bytes.len() > cfg.max_frame {
            cleanup();
            return Err(format!(
                "output {name:?} is {} bytes, over max {}",
                bytes.len(),
                cfg.max_frame
            ));
        }
        let off = body.len();
        let sha = sha256_hex(&bytes);
        body.extend_from_slice(&bytes);
        descs.push(json!({
            "frame": 0,
            "name": name,
            "kind": "U8",
            "shape": [bytes.len()],
            "off": off,
            "bytes": bytes.len(),
            "sha256": sha,
        }));
    }
    let header = json!({
        "v": PROTO_V,
        "output_format": "files",
        "frames": 1,
        "exit_code": exit_code,
        "stdout": String::from_utf8_lossy(&stdout),
        "stderr": String::from_utf8_lossy(&stderr),
        "outputs": descs,
        "runtime_version": RUNTIME_VERSION,
    });
    cleanup();
    Ok(ProcessResult { header, body })
}

fn drain_capped(mut r: impl Read) -> Vec<u8> {
    let mut out = Vec::new();
    let mut buf = [0u8; 8192];
    loop {
        match r.read(&mut buf) {
            Ok(0) => break,
            Ok(n) => {
                if out.len() < MAX_LOG {
                    let take = (MAX_LOG - out.len()).min(n);
                    out.extend_from_slice(&buf[..take]);
                }
            }
            Err(_) => break,
        }
    }
    out
}

fn is_safe_relative(name: &str) -> bool {
    if name.is_empty() {
        return false;
    }
    let p = Path::new(name);
    !p.is_absolute() && p.components().all(|c| matches!(c, Component::Normal(_)))
}

fn safe_join(root: &Path, rel: &str) -> Result<PathBuf, String> {
    if !is_safe_relative(rel) {
        return Err(format!("unsafe path {rel:?}"));
    }
    Ok(root.join(rel))
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
        job.result_body = Arc::new(body);
    }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn relative_paths_are_checked() {
        assert!(is_safe_relative("frame1.mem"));
        assert!(is_safe_relative("stage/frame.mem"));
        assert!(!is_safe_relative(""));
        assert!(!is_safe_relative("../escape"));
        assert!(!is_safe_relative("/absolute"));
        assert!(!is_safe_relative("./x"));
    }

    #[test]
    fn input_descriptors_are_bounded_and_hashed() {
        let payload = b"hello world";
        let sha = sha256_hex(payload);
        let header = json!({
            "inputs": [{"name": "in.txt", "off": 0, "bytes": 11, "sha256": sha}]
        });
        let inputs = parse_inputs(&header, payload).unwrap();
        assert_eq!(inputs.len(), 1);
        assert_eq!(inputs[0].0, "in.txt");
        assert_eq!(inputs[0].1, payload);
        let bad = json!({
            "inputs": [{"name": "in.txt", "off": 1, "bytes": 11}]
        });
        assert!(parse_inputs(&bad, payload).is_err());
        let bad_hash = json!({
            "inputs": [{"name": "in.txt", "off": 0, "bytes": 11, "sha256": "00"}]
        });
        assert!(parse_inputs(&bad_hash, payload).is_err());
    }

    #[test]
    fn process_stages_inputs_and_returns_outputs() {
        let root = std::env::temp_dir().join(format!(
            "jotnar-native-runner-test-{}-{}",
            std::process::id(),
            now_ms()
        ));
        fs::create_dir_all(&root).unwrap();
        let mk = |name: &str, body: &str| {
            let p = root.join(name);
            fs::write(&p, body).unwrap();
            fs::canonicalize(p).unwrap()
        };
        let cfg = Config {
            bind: "127.0.0.1:0".to_string(),
            token: None,
            max_frame: 1 << 20,
            work_root: root.join("work"),
            allow: vec![fs::canonicalize("/bin/sh").unwrap()],
            keep_work: false,
            mdns: false,
            jobs: Arc::new(Mutex::new(JobStore::new(8))),
            started_ms: now_ms() as u64,
        };
        fs::create_dir_all(&cfg.work_root).unwrap();
        let input = mk("in.txt", "native runner");
        let input_bytes = fs::read(&input).unwrap();
        let header = json!({
            "command": ["/bin/sh", "-c", "cat in.txt > out.txt"],
            "inputs": [{"name": "in.txt", "off": 0, "bytes": input_bytes.len()}],
            "outputs": ["out.txt"],
        });
        let payload = input_bytes;
        let cancel = AtomicBool::new(false);
        let result = run_process(&cfg, "test", &header, &payload, 60_000, &cancel).unwrap();
        assert_eq!(result.header["exit_code"], 0);
        let desc = &result.header["outputs"][0];
        assert_eq!(desc["name"], "out.txt");
        assert_eq!(desc["sha256"], sha256_hex(b"native runner"));
        assert_eq!(result.body, b"native runner");
        let _ = fs::remove_dir_all(&root);
    }
}
