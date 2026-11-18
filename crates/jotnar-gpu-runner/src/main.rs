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
use std::net::{TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Instant;

use jotnar::asm::AsmOptions;
use jotnar::codegen::infer::CKind;
use jotnar::codegen::{build, compile_program, Artifact, CompileOptions, Target};
use jotnar::payload::{feeds_from_json, value_to_json, Feeds};
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
    warm_cache: Arc<Mutex<warm::WarmCache>>,
    payload_cache: Arc<Mutex<PayloadCache>>,
    artifact_cache: Arc<Mutex<ArtifactCache>>,
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
    let cfg = Config {
        bind,
        token,
        target,
        max_frame: max_frame.max(1024),
        cache_dir,
        build_lock: Arc::new(Mutex::new(())),
        mdns,
        warm,
        warm_cache: Arc::new(Mutex::new(warm::WarmCache::new(warm_cache_workers))),
        payload_cache: Arc::new(Mutex::new(PayloadCache::new(payload_cache_bytes.max(1)))),
        artifact_cache: Arc::new(Mutex::new(ArtifactCache::new(artifact_cache_entries))),
    };
    let listener = TcpListener::bind(&cfg.bind).map_err(|e| format!("bind {}: {e}", cfg.bind))?;
    let port = listener
        .local_addr()
        .map_err(|e| format!("bound address: {e}"))?
        .port();
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
    let body: Json = serde_json::from_slice(payload).map_err(|e| format!("bad run body: {e}"))?;
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
        return send_run_batch_outputs(stream, id, cfg, &artifact, outputs, timing, binary_output);
    }
    send_run_outputs(
        stream,
        id,
        cfg,
        &artifact,
        outputs.pop().unwrap(),
        timing,
        binary_output,
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
                let bytes = encode_value(value, info.kind, cfg.target)?;
                let off = body.len();
                body.extend_from_slice(&bytes);
                let kind = match info.kind {
                    CKind::F if matches!(cfg.target, Target::Cuda | Target::Hip) => "F32",
                    CKind::F => "F64",
                    CKind::I => "I64",
                    CKind::T => "T",
                };
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
            let bytes = encode_value(value, info.kind, cfg.target)?;
            let off = body.len();
            body.extend_from_slice(&bytes);
            let kind = match info.kind {
                CKind::F if matches!(cfg.target, Target::Cuda | Target::Hip) => "F32",
                CKind::F => "F64",
                CKind::I => "I64",
                CKind::T => "T",
            };
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
