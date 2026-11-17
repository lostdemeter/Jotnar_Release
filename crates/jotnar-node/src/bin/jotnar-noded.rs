//! TCP front door for the Jotnar node, with supervisor-backed payload and event
//! commands.
//!
//! Core `ping`/`program`/`store`/`list`/`drop`/`exec` are handled by
//! `jotnar::protocol::Node` exactly as before. `payload-*` and `event-*` frames
//! are forwarded unchanged over a root-only Unix socket to
//! `jotnar-supervisord`. `ping` gains capability fields when the supervisor is
//! reachable.
//!
//! N1 adds an async job surface (`submit`, `job-status`, `job-cancel`,
//! `job-result`, `job-list`) backed by a bounded worker pool. Jobs execute the
//! same `Node::handle("exec")` path over an immutable registry snapshot, so
//! receipts and output bytes are unchanged. Connections are served
//! concurrently; legacy `exec` also runs over a snapshot so it no longer blocks
//! `program`/`store` for the whole process.

use std::collections::HashMap;
use std::fs;
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::os::unix::net::UnixStream;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, RwLock};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use jotnar::protocol::{decode_frame, encode_frame, error_frame, Node, PROTO_V};
use mdns_sd::{ServiceDaemon, ServiceInfo};
use serde_json::{json, Value as Json};

#[path = "../jobs.rs"]
mod jobs;

use jobs::{JobConfig, JobEngine};

fn usage() -> &'static str {
    "jotnar-noded - TCP server for the Jotnar node protocol + payload supervisor\n\
     \n\
     usage:\n\
       jotnar-noded [--bind ADDR] [--allow lattice,...] [--max-frame N] [--token SECRET]\n\
                    [--supervisor PATH] [--module-socket PATH] [--stats]\n\
                    [--workers N] [--queue N] [--job-timeout-ms N] [--max-timeout-ms N]\n\
                    [--max-job-bytes N] [--max-result-bytes N] [--retain-jobs N]\n\
                    [--inline-result-bytes N] [--result-store-bytes N]\n\
                    [--result-chunk-bytes N] [--result-lease-ms N] [--max-connections N]\n\
                    [--node-id-file PATH] [--mdns]\n\
     \n\
     ADDR defaults to 127.0.0.1:9999. Port 0 binds an ephemeral port. --allow\n\
     defaults to lattice. --max-frame defaults to 67108864 (64 MiB). Non-loopback\n\
     binds require --token. --supervisor defaults to /run/jotnar/supervisor.sock.\n\
     Async jobs: --workers defaults to min(available CPUs, 8); --queue 64;\n\
     --job-timeout-ms 30000 (0 disables); --max-job-bytes 64 MiB;\n\
     --max-result-bytes/--result-store-bytes 256 MiB; --inline-result-bytes 4 MiB;\n\
     --result-chunk-bytes 1 MiB; --result-lease-ms 60000.\n"
}

struct Config {
    bind: String,
    allowed: Vec<String>,
    max_frame: usize,
    stats: bool,
    token: Option<String>,
    supervisor: PathBuf,
    module_socket: Option<PathBuf>,
    workers: usize,
    queue: usize,
    job_timeout_ms: u64,
    max_timeout_ms: u64,
    max_job_bytes: usize,
    max_result_bytes: usize,
    inline_result_bytes: usize,
    result_store_bytes: usize,
    result_chunk_bytes: usize,
    result_lease_ms: u64,
    retain_jobs: usize,
    max_connections: usize,
    node_id_file: PathBuf,
    node_id: String,
    mdns: bool,
}

fn default_workers() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4)
        .clamp(1, 8)
}

fn load_node_id(path: &Path) -> String {
    if let Ok(text) = fs::read_to_string(path) {
        let id = text.trim();
        if !id.is_empty() {
            return id.to_string();
        }
    }
    let mut seed = String::new();
    if let Ok(mid) = fs::read_to_string("/etc/machine-id") {
        seed.push_str(mid.trim());
    }
    if let Ok(host) = fs::read_to_string("/etc/hostname") {
        seed.push_str(host.trim());
    }
    if let Ok(now) = SystemTime::now().duration_since(UNIX_EPOCH) {
        seed.push_str(&now.as_nanos().to_string());
    }
    seed.push_str(&std::process::id().to_string());
    let id = format!("node-{:016x}", jotnar::value::hash_bytes(seed.as_bytes()));
    if let Some(parent) = path.parent() {
        let _ = fs::create_dir_all(parent);
    }
    let _ = fs::write(path, format!("{id}\n"));
    id
}

fn start_mdns(cfg: &Config, port: u16) -> Result<ServiceDaemon, String> {
    let daemon = ServiceDaemon::new().map_err(|e| e.to_string())?;
    let mut props = HashMap::new();
    props.insert("proto".to_string(), PROTO_V.to_string());
    props.insert("node_id".to_string(), cfg.node_id.clone());
    props.insert("caps".to_string(), "jobs.v1".to_string());
    props.insert("workers".to_string(), cfg.workers.to_string());
    props.insert("queue".to_string(), cfg.queue.to_string());
    props.insert(
        "auth".to_string(),
        if cfg.token.is_some() {
            "required".to_string()
        } else {
            "none".to_string()
        },
    );
    props.insert("result_transport".to_string(), "inline+chunked".to_string());
    props.insert(
        "runtime_version".to_string(),
        jotnar::protocol::runtime_version(),
    );
    let host = format!("{}.local.", cfg.node_id);
    let instance = format!("Jotnar {}", cfg.node_id);
    let info = ServiceInfo::new(
        "_jotnar-node._tcp.local.",
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

fn parse_args() -> Result<Config, String> {
    let mut cfg = Config {
        bind: "127.0.0.1:9999".to_string(),
        allowed: vec!["lattice".to_string()],
        max_frame: 64 * 1024 * 1024,
        stats: false,
        token: None,
        supervisor: PathBuf::from("/run/jotnar/supervisor.sock"),
        module_socket: None,
        workers: default_workers(),
        queue: 64,
        job_timeout_ms: 30_000,
        max_timeout_ms: 600_000,
        max_job_bytes: 64 * 1024 * 1024,
        max_result_bytes: 256 * 1024 * 1024,
        inline_result_bytes: 4 * 1024 * 1024,
        result_store_bytes: 256 * 1024 * 1024,
        result_chunk_bytes: 1024 * 1024,
        result_lease_ms: 60_000,
        retain_jobs: 1024,
        max_connections: 64,
        node_id_file: PathBuf::from("/var/lib/jotnar/node-id"),
        node_id: String::new(),
        mdns: false,
    };
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--bind" => cfg.bind = it.next().ok_or("--bind needs an address")?,
            "--allow" => {
                let s = it.next().ok_or("--allow needs a comma list")?;
                cfg.allowed = s.split(',').map(|x| x.trim().to_string()).collect();
            }
            "--max-frame" => {
                cfg.max_frame = it
                    .next()
                    .ok_or("--max-frame needs a byte count")?
                    .parse()
                    .map_err(|_| "--max-frame must be an integer")?;
            }
            "--token" => {
                let t = it.next().ok_or("--token needs a secret")?;
                if t.trim().is_empty() {
                    return Err("--token must be a non-empty secret".to_string());
                }
                cfg.token = Some(t);
            }
            "--supervisor" => {
                cfg.supervisor = PathBuf::from(it.next().ok_or("--supervisor needs a path")?)
            }
            "--module-socket" => {
                cfg.module_socket = Some(PathBuf::from(
                    it.next().ok_or("--module-socket needs a path")?,
                ))
            }
            "--workers" => {
                cfg.workers = it
                    .next()
                    .ok_or("--workers needs a count")?
                    .parse()
                    .map_err(|_| "--workers must be an integer")?;
                if cfg.workers == 0 {
                    return Err("--workers must be >= 1".to_string());
                }
            }
            "--queue" => {
                cfg.queue = it
                    .next()
                    .ok_or("--queue needs a count")?
                    .parse()
                    .map_err(|_| "--queue must be an integer")?;
            }
            "--job-timeout-ms" => {
                cfg.job_timeout_ms = it
                    .next()
                    .ok_or("--job-timeout-ms needs milliseconds")?
                    .parse()
                    .map_err(|_| "--job-timeout-ms must be an integer")?;
            }
            "--max-timeout-ms" => {
                cfg.max_timeout_ms = it
                    .next()
                    .ok_or("--max-timeout-ms needs milliseconds")?
                    .parse()
                    .map_err(|_| "--max-timeout-ms must be an integer")?;
            }
            "--max-job-bytes" => {
                cfg.max_job_bytes = it
                    .next()
                    .ok_or("--max-job-bytes needs a byte count")?
                    .parse()
                    .map_err(|_| "--max-job-bytes must be an integer")?;
            }
            "--max-result-bytes" => {
                cfg.max_result_bytes = it
                    .next()
                    .ok_or("--max-result-bytes needs a byte count")?
                    .parse()
                    .map_err(|_| "--max-result-bytes must be an integer")?;
            }
            "--inline-result-bytes" => {
                cfg.inline_result_bytes = it
                    .next()
                    .ok_or("--inline-result-bytes needs a byte count")?
                    .parse()
                    .map_err(|_| "--inline-result-bytes must be an integer")?;
            }
            "--result-store-bytes" => {
                cfg.result_store_bytes = it
                    .next()
                    .ok_or("--result-store-bytes needs a byte count")?
                    .parse()
                    .map_err(|_| "--result-store-bytes must be an integer")?;
            }
            "--result-chunk-bytes" => {
                cfg.result_chunk_bytes = it
                    .next()
                    .ok_or("--result-chunk-bytes needs a byte count")?
                    .parse()
                    .map_err(|_| "--result-chunk-bytes must be an integer")?;
            }
            "--result-lease-ms" => {
                cfg.result_lease_ms = it
                    .next()
                    .ok_or("--result-lease-ms needs milliseconds")?
                    .parse()
                    .map_err(|_| "--result-lease-ms must be an integer")?;
            }
            "--retain-jobs" => {
                cfg.retain_jobs = it
                    .next()
                    .ok_or("--retain-jobs needs a count")?
                    .parse()
                    .map_err(|_| "--retain-jobs must be an integer")?;
            }
            "--max-connections" => {
                cfg.max_connections = it
                    .next()
                    .ok_or("--max-connections needs a count")?
                    .parse()
                    .map_err(|_| "--max-connections must be an integer")?;
            }
            "--node-id-file" => {
                cfg.node_id_file = PathBuf::from(it.next().ok_or("--node-id-file needs a path")?);
            }
            "--mdns" => cfg.mdns = true,
            "--stats" => cfg.stats = true,
            "-h" | "--help" | "help" => return Err(usage().to_string()),
            other => return Err(format!("unknown argument '{other}'\n\n{}", usage())),
        }
    }
    if cfg.queue == 0 {
        return Err("--queue must be >= 1".to_string());
    }
    if cfg.max_connections == 0 {
        return Err("--max-connections must be >= 1".to_string());
    }
    if cfg.max_timeout_ms == 0 {
        return Err("--max-timeout-ms must be >= 1 (0 disables all timeouts)".to_string());
    }
    if cfg.max_job_bytes == 0 || cfg.max_result_bytes == 0 || cfg.retain_jobs == 0 {
        return Err("--max-job-bytes, --max-result-bytes, --retain-jobs must be >= 1".to_string());
    }
    if cfg.inline_result_bytes == 0 || cfg.inline_result_bytes > cfg.max_result_bytes {
        return Err("--inline-result-bytes must be in 1..=--max-result-bytes".to_string());
    }
    if cfg.result_chunk_bytes == 0 || cfg.result_store_bytes < cfg.result_chunk_bytes {
        return Err(
            "--result-chunk-bytes must be >= 1 and --result-store-bytes must be >= it".to_string(),
        );
    }
    cfg.node_id = load_node_id(&cfg.node_id_file);
    Ok(cfg)
}

fn job_config(cfg: &Config) -> JobConfig {
    JobConfig {
        workers: cfg.workers,
        queue_capacity: cfg.queue,
        default_timeout: (cfg.job_timeout_ms > 0)
            .then(|| Duration::from_millis(cfg.job_timeout_ms)),
        max_timeout: Duration::from_millis(cfg.max_timeout_ms),
        max_input_bytes: cfg.max_job_bytes,
        max_result_bytes: cfg.max_result_bytes,
        inline_result_bytes: cfg.inline_result_bytes,
        result_store_bytes: cfg.result_store_bytes,
        result_chunk_bytes: cfg.result_chunk_bytes,
        result_lease: Duration::from_millis(cfg.result_lease_ms),
        retain_jobs: cfg.retain_jobs,
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

fn read_frame(stream: &mut TcpStream, max_frame: usize) -> Option<Result<Vec<u8>, ()>> {
    let mut hdr = [0u8; 4];
    if stream.read_exact(&mut hdr).is_err() {
        return None;
    }
    let n = u32::from_be_bytes(hdr) as usize;
    if n > max_frame {
        let _ = stream.write_all(&error_frame(
            &json!(0),
            "too-large",
            "frame exceeds max-frame",
        ));
        return Some(Err(()));
    }
    let mut body = vec![0u8; n];
    if stream.read_exact(&mut body).is_err() {
        return None;
    }
    let mut frame = Vec::with_capacity(4 + n);
    frame.extend_from_slice(&hdr);
    frame.extend_from_slice(&body);
    Some(Ok(frame))
}

fn forward(frame: &[u8], path: &PathBuf) -> Result<Vec<u8>, String> {
    let mut s = UnixStream::connect(path).map_err(|e| format!("supervisor: {e}"))?;
    let _ = s.set_read_timeout(Some(Duration::from_secs(30)));
    let _ = s.set_write_timeout(Some(Duration::from_secs(30)));
    s.write_all(frame)
        .map_err(|e| format!("supervisor write: {e}"))?;
    let mut hdr = [0u8; 4];
    s.read_exact(&mut hdr)
        .map_err(|e| format!("supervisor read: {e}"))?;
    let n = u32::from_be_bytes(hdr) as usize;
    if n > 64 * 1024 * 1024 {
        return Err("supervisor response too large".to_string());
    }
    let mut body = vec![0u8; n];
    s.read_exact(&mut body)
        .map_err(|e| format!("supervisor read: {e}"))?;
    let mut out = Vec::with_capacity(4 + n);
    out.extend_from_slice(&hdr);
    out.extend_from_slice(&body);
    Ok(out)
}

fn supervisor_ping(path: &PathBuf) -> Option<u64> {
    let frame = encode_frame(
        &json!({"v": PROTO_V, "id": 0, "type": "supervisor-ping"}),
        &[],
    )
    .ok()?;
    let out = forward(&frame, path).ok()?;
    let (h, _) = decode_frame(&out).ok()?;
    h.get("payloads").and_then(|x| x.as_u64())
}

fn reply(id: &Json, typ: &str, fields: Json) -> Vec<u8> {
    let mut h = fields;
    if let Some(o) = h.as_object_mut() {
        o.insert("v".into(), json!(PROTO_V));
        o.insert("id".into(), id.clone());
        o.insert("type".into(), json!(typ));
        o.insert("ok".into(), json!(true));
    }
    encode_frame(&h, &[]).unwrap()
}

fn job_field<'a>(header: &'a Json, name: &str, id: &Json) -> Result<&'a str, Vec<u8>> {
    header
        .get(name)
        .and_then(|x| x.as_str())
        .ok_or_else(|| error_frame(id, "malformed", format!("missing {name:?}")))
}

fn handle_job_frame(
    engine: &JobEngine,
    node: &Arc<RwLock<Node>>,
    header: &Json,
    payload: &[u8],
    id: &Json,
    typ: &str,
) -> Vec<u8> {
    // Authenticate from a cheap registry snapshot. The snapshot is also what a
    // submitted job will execute against, so `program`/`store` cannot change
    // the job after submission.
    let snapshot = node.read().unwrap().snapshot();
    if !snapshot.authorize(header) {
        return error_frame(id, "unauthorized", "bad or missing auth token");
    }

    match typ {
        "submit" => match engine.submit(snapshot, header, payload) {
            Ok(out) => reply(
                id,
                "submit-ok",
                json!({
                    "job_id": out.job_id,
                    "state": out.state.as_str(),
                    "replayed": out.replayed,
                    "queue_position": out.queue_position,
                }),
            ),
            Err(e) => error_frame(id, &e.code, e.msg),
        },
        "job-status" => {
            let job_id = match job_field(header, "job_id", id) {
                Ok(v) => v,
                Err(f) => return f,
            };
            match engine.status(job_id) {
                Some(status) => reply(id, "job-status-ok", json!({"job": status.to_json()})),
                None => error_frame(id, "unknown-job", format!("no job {job_id:?}")),
            }
        }
        "job-result" => {
            let job_id = match job_field(header, "job_id", id) {
                Ok(v) => v,
                Err(f) => return f,
            };
            match engine.result(job_id) {
                Ok(jobs::JobResultReply::Inline(result)) => {
                    let mut h = result.header;
                    if let Some(o) = h.as_object_mut() {
                        o.insert("v".into(), json!(PROTO_V));
                        o.insert("id".into(), id.clone());
                        o.insert("type".into(), json!("job-result-ok"));
                        o.insert("ok".into(), json!(true));
                        o.insert("job_id".into(), json!(job_id));
                    }
                    encode_frame(&h, &result.payload).unwrap()
                }
                Ok(jobs::JobResultReply::Stored(result)) => {
                    let meta = json!({
                        "transport": "store",
                        "id": result.meta.id,
                        "bytes": result.meta.bytes,
                        "chunk_bytes": result.meta.chunk_bytes,
                        "chunks": result.meta.chunks,
                        "hash": result.meta.hash,
                        "chunk_hashes": result.meta.chunk_hashes,
                    });
                    let mut h = result.header;
                    if let Some(o) = h.as_object_mut() {
                        o.insert("v".into(), json!(PROTO_V));
                        o.insert("id".into(), id.clone());
                        o.insert("type".into(), json!("job-result-ok"));
                        o.insert("ok".into(), json!(true));
                        o.insert("job_id".into(), json!(job_id));
                        o.insert("result".into(), meta);
                    }
                    encode_frame(&h, &[]).unwrap()
                }
                Err(e) => {
                    let mut err = json!({"code": e.code, "msg": e.msg});
                    if let Some(status) = engine.status(job_id) {
                        if let Some(o) = err.as_object_mut() {
                            o.insert("state".into(), json!(status.state.as_str()));
                            o.insert(
                                "retryable".into(),
                                json!(matches!(
                                    status.state,
                                    jobs::JobState::Queued | jobs::JobState::Running
                                )),
                            );
                        }
                    }
                    let h = json!({"v": PROTO_V, "id": id, "type": "error", "ok": false,
                                   "error": err});
                    encode_frame(&h, &[]).unwrap()
                }
            }
        }
        "job-cancel" => {
            let job_id = match job_field(header, "job_id", id) {
                Ok(v) => v,
                Err(f) => return f,
            };
            match engine.cancel(job_id) {
                Ok(out) => reply(
                    id,
                    "job-cancel-ok",
                    json!({
                        "job_id": out.job_id,
                        "accepted": out.accepted,
                        "previous": out.previous.as_str(),
                        "state": out.state.as_str(),
                    }),
                ),
                Err(e) => error_frame(id, &e.code, e.msg),
            }
        }
        "result-read" => {
            let result_id = match job_field(header, "result_id", id) {
                Ok(v) => v,
                Err(f) => return f,
            };
            let off = header.get("off").and_then(|x| x.as_u64()).unwrap_or(0) as usize;
            let len = match header.get("len").and_then(|x| x.as_u64()) {
                Some(v) => v as usize,
                None => return error_frame(id, "malformed", "missing \"len\""),
            };
            match engine.read_result(result_id, off, len) {
                Ok(chunk) => {
                    let h = json!({
                        "v": PROTO_V, "id": id, "type": "result-read-ok", "ok": true,
                        "result_id": chunk.result_id,
                        "off": chunk.off,
                        "len": chunk.len,
                        "total": chunk.total,
                        "hash": chunk.hash,
                        "eof": chunk.eof,
                    });
                    encode_frame(&h, &chunk.payload).unwrap()
                }
                Err(e) => error_frame(id, &e.code, e.msg),
            }
        }
        "result-drop" => {
            let result_id = match job_field(header, "result_id", id) {
                Ok(v) => v,
                Err(f) => return f,
            };
            match engine.drop_result(result_id) {
                Ok(released) => reply(
                    id,
                    "result-drop-ok",
                    json!({"result_id": result_id, "released_bytes": released}),
                ),
                Err(e) => error_frame(id, &e.code, e.msg),
            }
        }
        "job-list" => {
            let jobs: Vec<Json> = engine.list().iter().map(|s| s.to_json()).collect();
            let stats = engine.stats();
            reply(
                id,
                "job-list-ok",
                json!({
                    "jobs": jobs,
                    "stats": {
                        "running": stats.running,
                        "peak_running": stats.peak_running,
                        "queued": stats.queued,
                        "retained": stats.retained,
                        "result_count": stats.result_count,
                        "result_bytes": stats.result_bytes,
                    },
                }),
            )
        }
        _ => error_frame(id, "unknown-type", format!("no such job message {typ:?}")),
    }
}

fn handle_frame(
    node: &Arc<RwLock<Node>>,
    engine: &JobEngine,
    frame: &[u8],
    cfg: &Config,
) -> Vec<u8> {
    let (header, payload) = match decode_frame(frame) {
        Ok(x) => x,
        Err(e) => return error_frame(&json!(0), "malformed", e.to_string()),
    };
    let id = header.get("id").cloned().unwrap_or(json!(0));
    let typ = header.get("type").and_then(|x| x.as_str()).unwrap_or("");

    if typ == "ping" {
        let out = node.write().unwrap().handle(frame);
        let (mut h, payload) = match decode_frame(&out) {
            Ok(x) => x,
            Err(_) => return out,
        };
        let ready = supervisor_ping(&cfg.supervisor);
        let mut features = vec!["jobs.v1"];
        if ready.is_some() {
            features.push("payload.v1");
            features.push("events.v1");
        }
        if let Some(pong) = h.get_mut("pong").and_then(|v| v.as_object_mut()) {
            pong.insert("node_id".into(), json!(cfg.node_id));
            pong.insert("features".into(), json!(features));
            pong.insert("jobs".into(), engine.capabilities());
            match ready {
                Some(n) => {
                    pong.insert(
                        "supervisor".into(),
                        json!({"state": "ready", "payloads": n}),
                    );
                }
                None => {
                    pong.insert(
                        "supervisor".into(),
                        json!({"state": "unavailable", "payloads": 0}),
                    );
                }
            }
        }
        return encode_frame(&h, &payload).unwrap_or(out);
    }

    if matches!(
        typ,
        "submit"
            | "job-status"
            | "job-result"
            | "job-cancel"
            | "job-list"
            | "result-read"
            | "result-drop"
    ) {
        return handle_job_frame(engine, node, &header, &payload, &id, typ);
    }

    if typ.starts_with("payload-")
        || typ.starts_with("event-")
        || typ.starts_with("file-")
        || typ.starts_with("bundle-")
        || typ.starts_with("console-")
        || typ.starts_with("module-")
    {
        // Production module events go to the emitted daemon when configured.
        // `module-step` remains an interpreted parity hook and stays on the
        // supervisor.
        let target = if typ.starts_with("module-")
            && typ != "module-step"
            && typ != "module-event-interpreted"
        {
            cfg.module_socket.as_ref().unwrap_or(&cfg.supervisor)
        } else {
            &cfg.supervisor
        };
        return match forward(frame, target) {
            Ok(out) => out,
            Err(e) => error_frame(&id, "supervisor-unavailable", e),
        };
    }

    // Legacy `exec` runs on a snapshot too, so a long synchronous job does not
    // hold the registry lock. Auth and semantics are unchanged (`Node::handle`
    // still validates the token and builds the same receipt).
    if typ == "exec" {
        let mut snapshot = node.read().unwrap().snapshot();
        return snapshot.handle(frame);
    }

    node.write().unwrap().handle(frame)
}

fn serve(
    mut stream: TcpStream,
    node: Arc<RwLock<Node>>,
    engine: JobEngine,
    cfg: Arc<Config>,
) -> std::io::Result<()> {
    loop {
        let frame = match read_frame(&mut stream, cfg.max_frame) {
            Some(Ok(f)) => f,
            Some(Err(())) => return Ok(()),
            None => return Ok(()),
        };
        let t0 = std::time::Instant::now();
        let out = handle_frame(&node, &engine, &frame, &cfg);
        if cfg.stats {
            eprintln!(
                "req {:.3}ms ({} B -> {} B)",
                t0.elapsed().as_secs_f64() * 1e3,
                frame.len(),
                out.len()
            );
        }
        stream.write_all(&out)?;
    }
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("{e}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<(), String> {
    let cfg = parse_args()?;
    if !is_loopback_bind(&cfg.bind) && cfg.token.is_none() {
        return Err(format!(
            "refusing non-loopback bind {} without --token\n\n{}",
            cfg.bind,
            usage()
        ));
    }
    let listener =
        TcpListener::bind(&cfg.bind).map_err(|e| format!("cannot bind {}: {e}", cfg.bind))?;
    let local = listener
        .local_addr()
        .map_err(|e| format!("cannot read bound address: {e}"))?;
    eprintln!(
        "jotnar-noded v{} listening on {} (allowed {:?}, supervisor {}, workers {}, queue {})",
        PROTO_V,
        local,
        cfg.allowed,
        cfg.supervisor.display(),
        cfg.workers,
        cfg.queue
    );
    let mut node = Node::new(cfg.allowed.clone());
    if let Some(t) = cfg.token.clone() {
        node = node.with_token(t);
    }
    let node = Arc::new(RwLock::new(node));
    let cfg = Arc::new(cfg);
    let engine = JobEngine::new(job_config(&cfg)).map_err(|e| format!("{}: {}", e.code, e.msg))?;
    let connections = Arc::new(AtomicUsize::new(0));
    let _mdns = if cfg.mdns {
        match start_mdns(&cfg, local.port()) {
            Ok(daemon) => {
                eprintln!(
                    "mdns: advertising _jotnar-node._tcp on port {} as {}",
                    local.port(),
                    cfg.node_id
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

    for conn in listener.incoming() {
        match conn {
            Ok(mut stream) => {
                let live = connections.fetch_add(1, Ordering::SeqCst) + 1;
                if live > cfg.max_connections {
                    connections.fetch_sub(1, Ordering::SeqCst);
                    let _ = stream.set_nodelay(true);
                    let _ = stream.write_all(&error_frame(
                        &json!(0),
                        "busy",
                        format!("connection limit {} reached", cfg.max_connections),
                    ));
                    continue;
                }
                let _ = stream.set_nodelay(true);
                let _ = stream.set_read_timeout(Some(Duration::from_secs(60)));
                let _ = stream.set_write_timeout(Some(Duration::from_secs(60)));
                let node = node.clone();
                let engine = engine.clone();
                let cfg = cfg.clone();
                let connections = connections.clone();
                // One thread per connection: a stalled client or a long legacy
                // `exec` no longer blocks unrelated clients. Jobs still run on
                // the bounded worker pool, not on connection threads.
                let conns_for_thread = connections.clone();
                if let Err(e) = std::thread::Builder::new()
                    .name("jotnar-conn".to_string())
                    .spawn(move || {
                        if let Err(e) = serve(stream, node, engine, cfg) {
                            eprintln!("connection error: {e}");
                        }
                        conns_for_thread.fetch_sub(1, Ordering::SeqCst);
                    })
                {
                    connections.fetch_sub(1, Ordering::SeqCst);
                    eprintln!("spawn connection thread: {e}");
                }
            }
            Err(e) => eprintln!("accept error: {e}"),
        }
    }
    Ok(())
}
