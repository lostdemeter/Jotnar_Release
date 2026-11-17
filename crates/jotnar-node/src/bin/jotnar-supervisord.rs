//! JotnarOS persistent payload supervisor.
//!
//! Long-lived daemon that owns local Jotnar payload processes, injects TICK and
//! STDIN streams, applies STDOUT to the local console, and keeps a bounded event
//! ring for the remote node. It listens on a root-only Unix socket; the TCP node
//! daemon forwards `payload-*` and `event-*` frames to it unchanged.

use std::collections::HashMap;
use std::io::{Read, Write};
use std::net::TcpStream;
use std::os::unix::fs::PermissionsExt;
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::Duration;

use jotnar::asm::{self, AsmOptions};
use jotnar::ir::BoundProgram;
use jotnar::payload::{self, Feeds};
use jotnar::protocol::decode_frame;
use jotnar::runtime::{self, ExecOptions};
use jotnar::value::{array_from_vec, Value};
use jotnar_node_tools::capability::{CapResult, Command as CapCommand, ModuleManifest};
use jotnar_node_tools::job::JobClient;
use jotnar_node_tools::{error_frame, now_ms, ok_frame, write_console, Alive, EventBus};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value as Json};

#[derive(Clone, Debug, Serialize, Deserialize)]
struct Manifest {
    name: String,
    #[serde(default)]
    autostart: bool,
    #[serde(default = "default_restart")]
    restart: String,
    #[serde(default)]
    tick_ms: u64,
    #[serde(default = "default_console")]
    console: String,
    #[serde(default = "default_program")]
    program: String,
    #[serde(default)]
    seed: Option<String>,
    #[serde(default)]
    files: Vec<String>,
}

fn default_restart() -> String {
    "never".to_string()
}
fn default_console() -> String {
    "both".to_string()
}
fn default_program() -> String {
    "program.asm".to_string()
}

struct Payload {
    manifest: Manifest,
    bound: BoundProgram,
    seed: Feeds,
    dir: PathBuf,
    generation: u64,
    started_at_ms: u128,
    alive: Alive,
    last_error: Arc<Mutex<Option<String>>>,
    tx: Option<Sender<Msg>>,
    join: Option<JoinHandle<()>>,
}

enum Msg {
    Input(Vec<u8>),
    Stop,
}

struct Console {
    name: String,
    bundle: PathBuf,
    rootfs: PathBuf,
    bundle_sha256: String,
    vt: u32,
    started_at_ms: u128,
    child: Option<Child>,
}

type Registry = Arc<Mutex<HashMap<String, Payload>>>;
type Consoles = Arc<Mutex<HashMap<String, Console>>>;
type Modules = HashMap<String, ModuleHandle>;

#[derive(Clone)]
struct ModuleHandle {
    name: String,
    tx: Sender<ModuleMsg>,
}

enum ModuleMsg {
    Event {
        code: i64,
        event: String,
        dispatch: bool,
        args: Json,
        reply: Sender<Json>,
    },
    Status {
        reply: Sender<Json>,
    },
}

struct ModuleHost {
    manifest: ModuleManifest,
    bound: BoundProgram,
    seed: Feeds,
    dir: PathBuf,
    carry: Feeds,
    actions: u64,
    next_command: u64,
    consoles: Consoles,
    self_tx: Sender<ModuleMsg>,
    timers: Arc<Mutex<HashMap<String, Arc<AtomicBool>>>>,
    net_handles: Arc<Mutex<HashMap<u64, TcpStream>>>,
    next_net: Arc<AtomicU64>,
    job: JobClient,
    log: Arc<String>,
}

fn main() {
    let mut socket = "/run/jotnar/supervisor.sock".to_string();
    let mut data_dir = "/var/lib/jotnar".to_string();
    let mut console = "both".to_string();
    let mut log_path = "/var/lib/jotnar/log/supervisor.log".to_string();
    let mut job_endpoint = "127.0.0.1:9999".to_string();
    let mut job_token_file = "/etc/jotnar/node.token".to_string();

    let mut it = std::env::args().skip(1);
    while let Some(a) = it.next() {
        match a.as_str() {
            "--socket" => socket = it.next().unwrap_or(socket),
            "--data-dir" => data_dir = it.next().unwrap_or(data_dir),
            "--console" => console = it.next().unwrap_or(console),
            "--log" => log_path = it.next().unwrap_or(log_path),
            "--job-endpoint" => job_endpoint = it.next().unwrap_or(job_endpoint),
            "--job-token-file" => job_token_file = it.next().unwrap_or(job_token_file),
            "-h" | "--help" => {
                println!("jotnar-supervisord [--socket PATH] [--data-dir DIR] [--console both|local|serial|none] [--log FILE] [--job-endpoint ADDR] [--job-token-file FILE]");
                return;
            }
            other => {
                eprintln!("unknown argument {other:?}");
                std::process::exit(2);
            }
        }
    }

    let data = PathBuf::from(&data_dir);
    if let Some(parent) = Path::new(&socket).parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    if let Some(parent) = Path::new(&log_path).parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    let _ = std::fs::remove_file(&socket);

    let bus = Arc::new(EventBus::default());
    let registry: Registry = Arc::new(Mutex::new(HashMap::new()));
    let consoles: Consoles = Arc::new(Mutex::new(load_consoles(&data)));
    let log = Arc::new(log_path.clone());

    log_line(&log, "supervisord starting");
    load_existing(&data, &registry, &bus, &console, &log);
    let job_token = std::fs::read_to_string(&job_token_file)
        .ok()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty());
    let job = JobClient::new(job_endpoint, job_token);
    let modules = load_module_hosts(&data, &consoles, &log, &job);

    let listener = match UnixListener::bind(&socket) {
        Ok(l) => l,
        Err(e) => {
            log_line(&log, &format!("bind {socket}: {e}"));
            eprintln!("cannot bind {socket}: {e}");
            std::process::exit(1);
        }
    };
    let _ = std::fs::set_permissions(&socket, std::fs::Permissions::from_mode(0o600));
    log_line(&log, &format!("listening on {socket}"));

    for conn in listener.incoming() {
        match conn {
            Ok(stream) => {
                if let Err(e) = serve(
                    stream, &registry, &consoles, &modules, &bus, &data, &console, &log, &job,
                ) {
                    log_line(&log, &format!("connection: {e}"));
                }
            }
            Err(e) => log_line(&log, &format!("accept: {e}")),
        }
    }
}

fn log_line(path: &Arc<String>, msg: &str) {
    eprintln!("jotnar-supervisord: {msg}");
    if let Ok(mut f) = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path.as_str())
    {
        let _ = writeln!(f, "{} {msg}", now_ms());
    }
}

fn load_existing(
    data: &Path,
    registry: &Registry,
    bus: &Arc<EventBus>,
    console: &str,
    log: &Arc<String>,
) {
    let dir = data.join("payloads");
    let Ok(entries) = std::fs::read_dir(&dir) else {
        return;
    };
    for ent in entries.flatten() {
        let pdir = ent.path();
        if !pdir.is_dir() {
            continue;
        }
        match load_payload(&pdir) {
            Ok((manifest, bound, seed)) => {
                let name = manifest.name.clone();
                let autostart = manifest.autostart;
                let is_new = {
                    let mut map = registry.lock().unwrap_or_else(|e| e.into_inner());
                    if map.contains_key(&name) {
                        false
                    } else {
                        map.insert(
                            name.clone(),
                            Payload {
                                manifest,
                                bound,
                                seed,
                                dir: pdir.clone(),
                                generation: 0,
                                started_at_ms: 0,
                                alive: Alive::default(),
                                last_error: Arc::new(Mutex::new(None)),
                                tx: None,
                                join: None,
                            },
                        );
                        true
                    }
                };
                if is_new && autostart {
                    let mut map = registry.lock().unwrap_or_else(|e| e.into_inner());
                    let _ = start_locked(&mut map, &name, bus, console, log);
                }
            }
            Err(e) => log_line(log, &format!("skip {}: {e}", pdir.display())),
        }
    }
}

fn load_payload(dir: &Path) -> Result<(Manifest, BoundProgram, Feeds), String> {
    let mtext = std::fs::read(dir.join("manifest.json")).map_err(|e| e.to_string())?;
    let manifest: Manifest = serde_json::from_slice(&mtext).map_err(|e| e.to_string())?;
    let listing =
        std::fs::read_to_string(dir.join(&manifest.program)).map_err(|e| e.to_string())?;
    let base = dir.to_string_lossy().to_string();
    let bound = asm::assemble(&listing, &AsmOptions::new(&base)).map_err(|e| e.to_string())?;
    let seed = match &manifest.seed {
        Some(f) => {
            let p = dir.join(f);
            if p.exists() {
                payload::read_payload(&p).map_err(|e| e.to_string())?
            } else {
                Feeds::new()
            }
        }
        None => Feeds::new(),
    };
    Ok((manifest, bound, seed))
}

fn serve(
    mut stream: UnixStream,
    registry: &Registry,
    consoles: &Consoles,
    modules: &Modules,
    bus: &Arc<EventBus>,
    data: &Path,
    console: &str,
    log: &Arc<String>,
    job: &JobClient,
) -> std::io::Result<()> {
    loop {
        let mut hdr = [0u8; 4];
        if stream.read_exact(&mut hdr).is_err() {
            return Ok(());
        }
        let n = u32::from_be_bytes(hdr) as usize;
        if n > 64 * 1024 * 1024 {
            let out = error_frame(&json!(0), "too-large", "supervisor frame too large");
            let _ = stream.write_all(&out);
            return Ok(());
        }
        let mut body = vec![0u8; n];
        if stream.read_exact(&mut body).is_err() {
            return Ok(());
        }
        let mut frame = Vec::with_capacity(4 + n);
        frame.extend_from_slice(&hdr);
        frame.extend_from_slice(&body);
        let out = dispatch(
            &frame, registry, consoles, modules, bus, data, console, log, &job,
        );
        stream.write_all(&out)?;
    }
}

fn dispatch(
    frame: &[u8],
    registry: &Registry,
    consoles: &Consoles,
    modules: &Modules,
    bus: &Arc<EventBus>,
    data: &Path,
    console: &str,
    log: &Arc<String>,
    job: &JobClient,
) -> Vec<u8> {
    let (header, body) = match decode_frame(frame) {
        Ok(x) => x,
        Err(e) => return error_frame(&json!(0), "malformed", e.to_string()),
    };
    let id = header.get("id").cloned().unwrap_or(json!(0));
    let typ = header.get("type").and_then(|x| x.as_str()).unwrap_or("");
    match typ {
        "supervisor-ping" => {
            let n = registry.lock().unwrap_or_else(|e| e.into_inner()).len();
            ok_frame(&id, "supervisor-pong", json!({"payloads": n}), &[])
        }
        "payload-put" => match put(&header, &body, registry, data) {
            Ok((name, replaced)) => ok_frame(
                &id,
                "payload-ok",
                json!({"name": name, "staged": true, "replaced": replaced}),
                &[],
            ),
            Err((code, msg)) => error_frame(&id, code, msg),
        },
        "payload-start" => {
            let name = header.get("name").and_then(|x| x.as_str()).unwrap_or("");
            let mut map = registry.lock().unwrap_or_else(|e| e.into_inner());
            match start_locked(&mut map, name, bus, console, log) {
                Ok(()) => ok_frame(
                    &id,
                    "payload-ok",
                    json!({"name": name, "state": "running"}),
                    &[],
                ),
                Err((code, msg)) => error_frame(&id, code, msg),
            }
        }
        "payload-stop" => {
            let name = header.get("name").and_then(|x| x.as_str()).unwrap_or("");
            let mut map = registry.lock().unwrap_or_else(|e| e.into_inner());
            match stop_locked(&mut map, name, log) {
                Ok(()) => ok_frame(
                    &id,
                    "payload-ok",
                    json!({"name": name, "state": "stopped"}),
                    &[],
                ),
                Err((code, msg)) => error_frame(&id, code, msg),
            }
        }
        "payload-reload" => {
            let name = header.get("name").and_then(|x| x.as_str()).unwrap_or("");
            let mut map = registry.lock().unwrap_or_else(|e| e.into_inner());
            if !map.contains_key(name) {
                return error_frame(&id, "unknown-payload", format!("no payload {name:?}"));
            }
            let _ = stop_locked(&mut map, name, log);
            match start_locked(&mut map, name, bus, console, log) {
                Ok(()) => ok_frame(
                    &id,
                    "payload-ok",
                    json!({"name": name, "state": "running"}),
                    &[],
                ),
                Err((code, msg)) => error_frame(&id, code, msg),
            }
        }
        "payload-drop" => {
            let name = header.get("name").and_then(|x| x.as_str()).unwrap_or("");
            let mut map = registry.lock().unwrap_or_else(|e| e.into_inner());
            let had = map.contains_key(name);
            let _ = stop_locked(&mut map, name, log);
            map.remove(name);
            let _ = std::fs::remove_dir_all(data.join("payloads").join(name));
            ok_frame(
                &id,
                "payload-ok",
                json!({"name": name, "missing": !had}),
                &[],
            )
        }
        "payload-status" => {
            let want = header.get("name").and_then(|x| x.as_str());
            let map = registry.lock().unwrap_or_else(|e| e.into_inner());
            let mut list = Vec::new();
            for (name, p) in map.iter() {
                if let Some(w) = want {
                    if w != name {
                        continue;
                    }
                }
                list.push(status_json(name, p));
            }
            if let Some(w) = want {
                if list.is_empty() {
                    return error_frame(&id, "unknown-payload", format!("no payload {w:?}"));
                }
            }
            ok_frame(&id, "payload-status", json!({"payloads": list}), &[])
        }
        "event-poll" => {
            let after = header
                .get("after_seq")
                .and_then(|x| x.as_u64())
                .unwrap_or(0);
            let max = header
                .get("max")
                .and_then(|x| x.as_u64())
                .unwrap_or(64)
                .min(256) as usize;
            let kinds: Vec<String> = header
                .get("kinds")
                .and_then(|x| x.as_array())
                .map(|a| {
                    a.iter()
                        .filter_map(|v| v.as_str().map(|s| s.to_string()))
                        .collect()
                })
                .unwrap_or_default();
            let (events, next, dropped) = bus.poll(after, max, &kinds);
            let mut payload = Vec::new();
            let mut ev_json = Vec::new();
            for e in events {
                let off = payload.len();
                payload.extend_from_slice(&e.data);
                ev_json.push(json!({
                    "seq": e.seq,
                    "ts_ms": e.ts_ms as u64,
                    "kind": e.kind,
                    "stream": e.stream,
                    "off": off,
                    "len": e.data.len()
                }));
            }
            ok_frame(
                &id,
                "event-batch",
                json!({"events": ev_json, "next_seq": next, "head_seq": bus.head(),
                       "dropped": dropped, "more": false}),
                &payload,
            )
        }
        "event-write" => {
            let name = header.get("name").and_then(|x| x.as_str()).unwrap_or("");
            let target = header
                .get("target")
                .and_then(|x| x.as_str())
                .unwrap_or("stdin");
            if target == "file" {
                let path = header.get("path").and_then(|x| x.as_str()).unwrap_or("");
                if !safe_rel(path) {
                    return error_frame(&id, "payload-invalid", format!("invalid path {path:?}"));
                }
                let mode = header
                    .get("mode")
                    .and_then(|x| x.as_str())
                    .unwrap_or("write");
                let map = registry.lock().unwrap_or_else(|e| e.into_inner());
                let Some(p) = map.get(name) else {
                    return error_frame(&id, "unknown-payload", format!("no payload {name:?}"));
                };
                if !p.manifest.files.iter().any(|f| f == path) {
                    return error_frame(
                        &id,
                        "payload-invalid",
                        format!("file {path:?} is not declared"),
                    );
                }
                let full = p.dir.join("files").join(path);
                if let Some(parent) = full.parent() {
                    let _ = std::fs::create_dir_all(parent);
                }
                let mut opts = std::fs::OpenOptions::new();
                opts.create(true).write(true);
                if mode == "append" {
                    opts.append(true);
                } else {
                    opts.truncate(true);
                }
                let result = opts.open(&full).and_then(|mut f| f.write_all(&body));
                match result {
                    Ok(()) => {
                        bus.push("file-write", Some(path), body.to_vec());
                        ok_frame(
                            &id,
                            "event-ok",
                            json!({"name": name, "path": path, "bytes": body.len()}),
                            &[],
                        )
                    }
                    Err(e) => error_frame(&id, "payload-invalid", e.to_string()),
                }
            } else if target == "stdin" {
                let map = registry.lock().unwrap_or_else(|e| e.into_inner());
                let Some(p) = map.get(name) else {
                    return error_frame(&id, "unknown-payload", format!("no payload {name:?}"));
                };
                let Some(tx) = &p.tx else {
                    return error_frame(
                        &id,
                        "payload-not-running",
                        format!("payload {name:?} is not running"),
                    );
                };
                let accepted = body.len();
                match tx.send(Msg::Input(body.to_vec())) {
                    Ok(()) => ok_frame(
                        &id,
                        "event-ok",
                        json!({"name": name, "accepted": accepted}),
                        &[],
                    ),
                    Err(_) => error_frame(&id, "payload-not-running", "payload channel closed"),
                }
            } else {
                error_frame(
                    &id,
                    "payload-invalid",
                    format!("unsupported target {target:?}"),
                )
            }
        }
        "file-read" => {
            let name = header.get("name").and_then(|x| x.as_str()).unwrap_or("");
            let path = header.get("path").and_then(|x| x.as_str()).unwrap_or("");
            if !safe_rel(path) {
                return error_frame(&id, "payload-invalid", format!("invalid path {path:?}"));
            }
            let map = registry.lock().unwrap_or_else(|e| e.into_inner());
            let Some(p) = map.get(name) else {
                return error_frame(&id, "unknown-payload", format!("no payload {name:?}"));
            };
            if !p.manifest.files.iter().any(|f| f == path) {
                return error_frame(
                    &id,
                    "payload-invalid",
                    format!("file {path:?} is not declared"),
                );
            }
            match std::fs::read(p.dir.join("files").join(path)) {
                Ok(bytes) => ok_frame(
                    &id,
                    "file-data",
                    json!({"name": name, "path": path, "len": bytes.len()}),
                    &bytes,
                ),
                Err(e) => error_frame(&id, "payload-invalid", e.to_string()),
            }
        }
        "job-submit" => match job.submit(&header) {
            Ok(h) => ok_frame(&id, "job-ok", h, &[]),
            Err((code, msg)) => error_frame(&id, &code, msg),
        },
        "job-status" => {
            let job_id = header.get("job_id").and_then(|x| x.as_str()).unwrap_or("");
            match job.status(job_id) {
                Ok(h) => ok_frame(&id, "job-ok", h.get("job").cloned().unwrap_or(h), &[]),
                Err((code, msg)) => error_frame(&id, &code, msg),
            }
        }
        "job-cancel" => {
            let job_id = header.get("job_id").and_then(|x| x.as_str()).unwrap_or("");
            match job.cancel(job_id) {
                Ok(h) => ok_frame(&id, "job-ok", h, &[]),
                Err((code, msg)) => error_frame(&id, &code, msg),
            }
        }
        "job-list" => match job.list() {
            Ok(h) => ok_frame(&id, "job-ok", h, &[]),
            Err((code, msg)) => error_frame(&id, &code, msg),
        },
        "module-event" | "module-step" | "module-event-interpreted" => {
            let name = header.get("module").and_then(|x| x.as_str()).unwrap_or("");
            let event = header.get("event").and_then(|x| x.as_str()).unwrap_or("");
            let dispatch = typ != "module-step";
            let args = header.get("args").cloned().unwrap_or_else(|| json!({}));
            let Some(handle) = modules.get(name) else {
                return error_frame(&id, "unknown-module", format!("no module {name:?}"));
            };
            let Some(code) = module_event_code(event) else {
                return error_frame(&id, "module-invalid", format!("unknown event {event:?}"));
            };
            let (tx, rx) = mpsc::channel();
            if handle
                .tx
                .send(ModuleMsg::Event {
                    code,
                    event: event.to_string(),
                    dispatch,
                    args,
                    reply: tx,
                })
                .is_err()
            {
                return error_frame(&id, "module-unavailable", "module host stopped");
            }
            match rx.recv_timeout(Duration::from_secs(10)) {
                Ok(v) => ok_frame(
                    &id,
                    if dispatch { "module-ok" } else { "module-step" },
                    v,
                    &[],
                ),
                Err(_) => error_frame(&id, "module-timeout", "module did not reply"),
            }
        }
        "module-status" | "module-list" => {
            let want = header
                .get("module")
                .and_then(|x| x.as_str())
                .or_else(|| header.get("name").and_then(|x| x.as_str()));
            let names: Vec<String> = match want {
                Some(n) => vec![n.to_string()],
                None => {
                    let mut v: Vec<String> = modules.keys().cloned().collect();
                    v.sort();
                    v
                }
            };
            let mut list = Vec::new();
            for n in names {
                let Some(handle) = modules.get(&n) else {
                    continue;
                };
                let (tx, rx) = mpsc::channel();
                if handle.tx.send(ModuleMsg::Status { reply: tx }).is_err() {
                    continue;
                }
                if let Ok(v) = rx.recv_timeout(Duration::from_secs(5)) {
                    list.push(v);
                }
            }
            if let Some(n) = want {
                if list.is_empty() {
                    return error_frame(&id, "unknown-module", format!("no module {n:?}"));
                }
            }
            ok_frame(&id, "module-status", json!({"modules": list}), &[])
        }
        "bundle-put" => {
            let name = header.get("name").and_then(|x| x.as_str()).unwrap_or("");
            match bundle_put(name, &body, data, consoles) {
                Ok((sha, bytes)) => ok_frame(
                    &id,
                    "bundle-ok",
                    json!({"name": name, "bytes": bytes, "sha256": sha}),
                    &[],
                ),
                Err((code, msg)) => error_frame(&id, code, msg),
            }
        }
        "console-start" => {
            let name = header.get("name").and_then(|x| x.as_str()).unwrap_or("");
            let vt = header.get("vt").and_then(|x| x.as_u64()).unwrap_or(2) as u32;
            match console_start(name, vt, consoles) {
                Ok((pid, vt)) => ok_frame(
                    &id,
                    "console-ok",
                    json!({"name": name, "state": "running", "pid": pid, "vt": vt}),
                    &[],
                ),
                Err((code, msg)) => error_frame(&id, code, msg),
            }
        }
        "console-stop" => {
            let name = header.get("name").and_then(|x| x.as_str()).unwrap_or("");
            match console_stop(name, consoles) {
                Ok(()) => ok_frame(
                    &id,
                    "console-ok",
                    json!({"name": name, "state": "stopped"}),
                    &[],
                ),
                Err((code, msg)) => error_frame(&id, code, msg),
            }
        }
        "console-write" => {
            let name = header.get("name").and_then(|x| x.as_str()).unwrap_or("");
            match console_write(name, &body, consoles) {
                Ok(n) => ok_frame(&id, "console-ok", json!({"name": name, "bytes": n}), &[]),
                Err((code, msg)) => error_frame(&id, code, msg),
            }
        }
        "console-restart" => {
            let name = header.get("name").and_then(|x| x.as_str()).unwrap_or("");
            let vt = header.get("vt").and_then(|x| x.as_u64()).unwrap_or(2) as u32;
            match console_restart(name, vt, consoles) {
                Ok((pid, vt)) => ok_frame(
                    &id,
                    "console-ok",
                    json!({"name": name, "state": "running", "pid": pid, "vt": vt}),
                    &[],
                ),
                Err((code, msg)) => error_frame(&id, code, msg),
            }
        }
        "console-status" | "console-list" => {
            let want = header.get("name").and_then(|x| x.as_str());
            let mut map = consoles.lock().unwrap_or_else(|e| e.into_inner());
            let mut list = Vec::new();
            for (name, c) in map.iter_mut() {
                if let Some(w) = want {
                    if w != name {
                        continue;
                    }
                }
                refresh_console(c);
                list.push(console_json(name, c));
            }
            if let Some(w) = want {
                if list.is_empty() {
                    return error_frame(&id, "unknown-console", format!("no console {w:?}"));
                }
            }
            ok_frame(&id, "console-status", json!({"consoles": list}), &[])
        }
        _ => error_frame(&id, "unknown-type", format!("no such message {typ:?}")),
    }
}

fn module_event_code(event: &str) -> Option<i64> {
    match event {
        "console.start" => Some(10),
        "console.stop" => Some(11),
        "console.status" => Some(12),
        "system.log" => Some(30),
        "file.write" => Some(40),
        "file.read" => Some(41),
        "file.list" => Some(42),
        "proc.status" => Some(60),
        "proc.signal" => Some(61),
        "timer.arm" => Some(70),
        "timer.cancel" => Some(71),
        "timer.fired" => Some(72),
        "net.status" => Some(80),
        "net.connect" => Some(81),
        "net.send" => Some(82),
        "net.recv" => Some(83),
        "net.close" => Some(84),
        "job.submit" => Some(90),
        "job.status" => Some(91),
        "job.cancel" => Some(92),
        "job.list" => Some(93),
        _ => None,
    }
}

fn load_module_hosts(
    data: &Path,
    consoles: &Consoles,
    log: &Arc<String>,
    job: &JobClient,
) -> Modules {
    let mut dirs: HashMap<String, PathBuf> = HashMap::new();
    for base in [
        PathBuf::from("/usr/local/share/jotnar/modules"),
        data.join("modules"),
    ] {
        let Ok(entries) = std::fs::read_dir(&base) else {
            continue;
        };
        for ent in entries.flatten() {
            let dir = ent.path();
            if !dir.join("manifest.json").exists() {
                continue;
            }
            if let Some(name) = dir.file_name().and_then(|s| s.to_str()) {
                // Later bases (data/modules) override built-ins.
                dirs.insert(name.to_string(), dir);
            }
        }
    }
    let mut out = HashMap::new();
    for (_, dir) in dirs {
        if let Some(handle) = load_one_module(&dir, data, consoles, log, job) {
            out.insert(handle.name.clone(), handle);
        }
    }
    out
}

fn load_one_module(
    dir: &Path,
    data: &Path,
    consoles: &Consoles,
    log: &Arc<String>,
    job: &JobClient,
) -> Option<ModuleHandle> {
    let manifest: ModuleManifest = std::fs::read(dir.join("manifest.json"))
        .ok()
        .and_then(|b| serde_json::from_slice(&b).ok())?;
    let listing = std::fs::read_to_string(dir.join(&manifest.program)).ok()?;
    let bound = match asm::assemble(&listing, &AsmOptions::new(dir.to_string_lossy().as_ref())) {
        Ok(b) => b,
        Err(e) => {
            log_line(
                log,
                &format!("module {} assemble failed: {e}", manifest.name),
            );
            return None;
        }
    };
    let seed = manifest
        .seed
        .as_ref()
        .map(|s| dir.join(s))
        .filter(|p| p.exists())
        .and_then(|p| payload::read_payload(&p).ok())
        .unwrap_or_default();
    // Built-in modules live in the image; their state belongs on jotnar-data.
    let state_dir = if dir.starts_with("/usr/local/share") {
        let d = data.join("modules").join(&manifest.name);
        let _ = std::fs::create_dir_all(&d);
        d
    } else {
        dir.to_path_buf()
    };
    let carry = load_state(&state_dir);
    let name = manifest.name.clone();
    let (tx, rx) = mpsc::channel();
    let self_tx = tx.clone();
    let host = ModuleHost {
        manifest,
        bound,
        seed,
        dir: state_dir,
        carry,
        actions: 0,
        next_command: 1,
        consoles: consoles.clone(),
        self_tx,
        timers: Arc::new(Mutex::new(HashMap::new())),
        net_handles: Arc::new(Mutex::new(HashMap::new())),
        next_net: Arc::new(AtomicU64::new(1)),
        job: job.clone(),
        log: log.clone(),
    };
    thread::spawn(move || host.run(rx));
    log_line(log, &format!("module host ready: {name}"));
    Some(ModuleHandle { name, tx })
}

impl ModuleHost {
    fn run(mut self, rx: Receiver<ModuleMsg>) {
        for msg in rx {
            match msg {
                ModuleMsg::Event {
                    code,
                    event,
                    dispatch,
                    args,
                    reply,
                } => {
                    let out = self.step(code, &event, dispatch, &args);
                    let _ = reply.send(out);
                }
                ModuleMsg::Status { reply } => {
                    let _ = reply.send(self.status());
                }
            }
        }
    }

    fn state_json(&self) -> Json {
        let mut streams = serde_json::Map::new();
        for (name, v) in &self.carry {
            if let Ok(j) = payload::value_to_json(v) {
                streams.insert(name.clone(), j);
            }
        }
        json!(streams)
    }

    fn status(&self) -> Json {
        json!({
            "name": self.manifest.name,
            "actions": self.actions,
            "state": self.state_json(),
            "capabilities": self.manifest.capabilities,
            "state_file": self.dir.join("state.ndjson"),
        })
    }

    fn step(&mut self, code: i64, event: &str, dispatch: bool, override_args: &Json) -> Json {
        let mut feeds = self.seed.clone();
        for name in &self.bound.state {
            if let Some(v) = self.carry.get(name) {
                feeds.insert(name.clone(), v.clone());
            }
        }
        feeds.insert(
            "EVENT".to_string(),
            Value::I64(array_from_vec(&[1], vec![code]).unwrap()),
        );
        let out = match runtime::run_with_options(&self.bound, &feeds, &ExecOptions::default()) {
            Ok(v) => v,
            Err(e) => {
                log_line(
                    &self.log,
                    &format!("module {} step failed: {e}", self.manifest.name),
                );
                return json!({"module": self.manifest.name, "event": event, "error": e.to_string()});
            }
        };
        let mut changed = false;
        for name in &self.bound.state {
            if let Some(v) = out.get(name) {
                self.carry.insert(name.clone(), v.clone());
                changed = true;
            }
        }
        if changed {
            save_state(&self.dir, &self.carry);
        }
        let action = out
            .get("ACTION")
            .and_then(|v| match v {
                Value::I64(a) => a.iter().next().copied(),
                _ => None,
            })
            .unwrap_or(0);
        self.actions += 1;
        let (cap, op, args) = match action {
            1 => ("console", "start", json!({"name": "console", "vt": 2})),
            2 => ("console", "stop", json!({"name": "console"})),
            3 => ("console", "status", json!({"name": "console"})),
            4 => ("system", "log", json!({"message": event})),
            5 => (
                "file",
                "write",
                json!({"root": "module", "path": "notes.txt", "mode": "write", "data": "file-policy\n"}),
            ),
            6 => (
                "file",
                "read",
                json!({"root": "module", "path": "notes.txt"}),
            ),
            7 => ("file", "list", json!({"root": "module", "path": ""})),
            9 => ("proc", "status", json!({"pid": 1})),
            10 => ("proc", "signal", json!({"pid": 1, "signal": 0})),
            11 => ("timer", "arm", json!({"delay_ms": 500, "key": "t1"})),
            12 => ("timer", "cancel", json!({"key": "t1"})),
            13 => ("net", "status", json!({})),
            14 => (
                "net",
                "connect",
                json!({"host": "127.0.0.1", "port": 19000}),
            ),
            15 => ("net", "send", json!({"handle": 1, "data": "ping\n"})),
            16 => ("net", "recv", json!({"handle": 1, "max": 64})),
            17 => ("net", "close", json!({"handle": 1})),
            18 => ("job", "submit", json!({})),
            19 => ("job", "status", json!({})),
            20 => ("job", "cancel", json!({})),
            21 => ("job", "list", json!({})),
            _ => ("none", "none", json!({})),
        };
        let mut args = args;
        if cap == "job" {
            if let (Some(dst), Some(src)) = (args.as_object_mut(), override_args.as_object()) {
                for (k, v) in src {
                    dst.insert(k.clone(), v.clone());
                }
            }
        }
        let cmd_id = self.next_command;
        self.next_command += 1;
        let command = CapCommand::new(cmd_id, &self.manifest.name, cap, op, args.clone());
        let result = if !self.manifest.granted(cap, op) {
            CapResult::err(cmd_id, "denied", format!("grant missing for {cap}.{op}"))
        } else if dispatch {
            self.dispatch(cmd_id, cap, op, &args)
        } else {
            CapResult::ok(cmd_id, json!({"dispatched": false}))
        };
        log_line(
            &self.log,
            &format!(
                "module {} event {} action {} -> {}.{} ok={}",
                self.manifest.name, event, action, cap, op, result.ok
            ),
        );
        json!({
            "module": self.manifest.name,
            "event": event,
            "action": action,
            "actions": self.actions,
            "command": command,
            "result": result,
            "state": self.state_json(),
        })
    }

    fn dispatch(&self, id: u64, cap: &str, op: &str, args: &Json) -> CapResult {
        match (cap, op) {
            ("console", "start") => {
                let vt = args.get("vt").and_then(|x| x.as_u64()).unwrap_or(2) as u32;
                match console_start("console", vt, &self.consoles) {
                    Ok((pid, vt)) => CapResult::ok(id, json!({"pid": pid, "vt": vt})),
                    Err((code, msg)) => CapResult::err(id, code, msg),
                }
            }
            ("console", "stop") => match console_stop("console", &self.consoles) {
                Ok(()) => CapResult::ok(id, json!({"state": "stopped"})),
                Err((code, msg)) => CapResult::err(id, code, msg),
            },
            ("console", "status") => {
                let mut map = self.consoles.lock().unwrap_or_else(|e| e.into_inner());
                match map.get_mut("console") {
                    Some(c) => {
                        refresh_console(c);
                        CapResult::ok(id, console_json("console", c))
                    }
                    None => CapResult::err(id, "unknown-console", "no console \"console\""),
                }
            }
            ("system", "log") => {
                let msg = args.get("message").and_then(|x| x.as_str()).unwrap_or("");
                log_line(
                    &self.log,
                    &format!("module {} log: {msg}", self.manifest.name),
                );
                CapResult::ok(id, json!({"logged": msg}))
            }
            ("file", "write") => {
                let path = args.get("path").and_then(|x| x.as_str()).unwrap_or("");
                if !safe_rel(path) {
                    return CapResult::err(id, "payload-invalid", format!("invalid path {path:?}"));
                }
                let mode = args.get("mode").and_then(|x| x.as_str()).unwrap_or("write");
                let data = args
                    .get("data")
                    .and_then(|x| x.as_str())
                    .unwrap_or("")
                    .as_bytes();
                let full = self.dir.join("files").join(path);
                if let Some(parent) = full.parent() {
                    let _ = std::fs::create_dir_all(parent);
                }
                let mut opts = std::fs::OpenOptions::new();
                opts.create(true).write(true);
                if mode == "append" {
                    opts.append(true);
                } else {
                    opts.truncate(true);
                }
                match opts.open(&full).and_then(|mut f| f.write_all(data)) {
                    Ok(()) => CapResult::ok(id, json!({"path": path, "bytes": data.len()})),
                    Err(e) => CapResult::err(id, "file-error", e.to_string()),
                }
            }
            ("file", "read") => {
                let path = args.get("path").and_then(|x| x.as_str()).unwrap_or("");
                if !safe_rel(path) {
                    return CapResult::err(id, "payload-invalid", format!("invalid path {path:?}"));
                }
                match std::fs::read(self.dir.join("files").join(path)) {
                    Ok(bytes) => {
                        let text = String::from_utf8_lossy(&bytes).to_string();
                        CapResult::ok(
                            id,
                            json!({"path": path, "bytes": bytes.len(), "text": text}),
                        )
                    }
                    Err(e) => CapResult::err(id, "file-error", e.to_string()),
                }
            }
            ("file", "list") => {
                let path = args.get("path").and_then(|x| x.as_str()).unwrap_or("");
                let full = if path.is_empty() {
                    self.dir.join("files")
                } else if safe_rel(path) {
                    self.dir.join("files").join(path)
                } else {
                    return CapResult::err(id, "payload-invalid", format!("invalid path {path:?}"));
                };
                let mut entries = Vec::new();
                if let Ok(rd) = std::fs::read_dir(&full) {
                    for e in rd.flatten() {
                        if let Some(n) = e.file_name().to_str() {
                            entries.push(n.to_string());
                        }
                    }
                }
                entries.sort();
                CapResult::ok(id, json!({"path": path, "entries": entries}))
            }
            ("proc", "status") => {
                let pid = args.get("pid").and_then(|x| x.as_u64()).unwrap_or(0) as u32;
                match proc_status(pid) {
                    Ok(v) => CapResult::ok(id, v),
                    Err(e) => CapResult::err(id, "proc-error", e),
                }
            }
            ("proc", "signal") => {
                let pid = args.get("pid").and_then(|x| x.as_u64()).unwrap_or(0) as u32;
                let sig = args.get("signal").and_then(|x| x.as_i64()).unwrap_or(0) as i32;
                // v1 deliberately allows only signal 0 (existence check).
                if sig != 0 {
                    return CapResult::err(id, "unsupported", "only signal 0 is supported in v1");
                }
                let rc = unsafe { libc::kill(pid as i32, 0) };
                if rc == 0 {
                    CapResult::ok(id, json!({"pid": pid, "signal": sig, "alive": true}))
                } else {
                    CapResult::err(id, "proc-error", format!("pid {pid} not alive"))
                }
            }
            ("timer", "arm") => {
                let delay = args.get("delay_ms").and_then(|x| x.as_u64()).unwrap_or(500);
                let key = args
                    .get("key")
                    .and_then(|x| x.as_str())
                    .unwrap_or("timer")
                    .to_string();
                let cancel = Arc::new(AtomicBool::new(false));
                self.timers
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .insert(key.clone(), cancel.clone());
                let tx = self.self_tx.clone();
                let name = self.manifest.name.clone();
                thread::spawn(move || {
                    thread::sleep(Duration::from_millis(delay));
                    if cancel.load(Ordering::SeqCst) {
                        return;
                    }
                    let (rtx, _rrx) = mpsc::channel();
                    let _ = tx.send(ModuleMsg::Event {
                        code: 72,
                        event: "timer.fired".to_string(),
                        dispatch: true,
                        args: json!({}),
                        reply: rtx,
                    });
                    let _ = name;
                });
                CapResult::ok(id, json!({"armed": true, "key": key, "delay_ms": delay}))
            }
            ("timer", "cancel") => {
                let key = args.get("key").and_then(|x| x.as_str()).unwrap_or("timer");
                if let Some(flag) = self
                    .timers
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .get(key)
                {
                    flag.store(true, Ordering::SeqCst);
                }
                CapResult::ok(id, json!({"cancelled": true, "key": key}))
            }
            ("net", "status") => match net_status() {
                Ok(v) => CapResult::ok(id, v),
                Err(e) => CapResult::err(id, "net-error", e),
            },
            ("net", "connect") => {
                let host = args
                    .get("host")
                    .and_then(|x| x.as_str())
                    .unwrap_or("127.0.0.1");
                let port = args.get("port").and_then(|x| x.as_u64()).unwrap_or(0) as u16;
                match TcpStream::connect((host, port)) {
                    Ok(s) => {
                        let _ = s.set_read_timeout(Some(Duration::from_secs(5)));
                        let _ = s.set_write_timeout(Some(Duration::from_secs(5)));
                        let handle = self.next_net.fetch_add(1, Ordering::SeqCst);
                        self.net_handles
                            .lock()
                            .unwrap_or_else(|e| e.into_inner())
                            .insert(handle, s);
                        CapResult::ok(id, json!({"handle": handle, "host": host, "port": port}))
                    }
                    Err(e) => CapResult::err(id, "net-error", e.to_string()),
                }
            }
            ("net", "send") => {
                let handle = args.get("handle").and_then(|x| x.as_u64()).unwrap_or(0);
                let data = args.get("data").and_then(|x| x.as_str()).unwrap_or("");
                let mut map = self.net_handles.lock().unwrap_or_else(|e| e.into_inner());
                match map.get_mut(&handle) {
                    Some(s) => match s.write_all(data.as_bytes()) {
                        Ok(()) => CapResult::ok(id, json!({"handle": handle, "bytes": data.len()})),
                        Err(e) => CapResult::err(id, "net-error", e.to_string()),
                    },
                    None => CapResult::err(id, "net-error", format!("no handle {handle}")),
                }
            }
            ("net", "recv") => {
                let handle = args.get("handle").and_then(|x| x.as_u64()).unwrap_or(0);
                let max = args.get("max").and_then(|x| x.as_u64()).unwrap_or(4096) as usize;
                let mut buf = vec![0u8; max.min(65536)];
                let mut map = self.net_handles.lock().unwrap_or_else(|e| e.into_inner());
                match map.get_mut(&handle) {
                    Some(s) => match s.read(&mut buf) {
                        Ok(n) => {
                            buf.truncate(n);
                            CapResult::ok(
                                id,
                                json!({
                                    "handle": handle, "bytes": n,
                                    "text": String::from_utf8_lossy(&buf)
                                }),
                            )
                        }
                        Err(e) => CapResult::err(id, "net-error", e.to_string()),
                    },
                    None => CapResult::err(id, "net-error", format!("no handle {handle}")),
                }
            }
            ("net", "close") => {
                let handle = args.get("handle").and_then(|x| x.as_u64()).unwrap_or(0);
                self.net_handles
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .remove(&handle);
                CapResult::ok(id, json!({"closed": handle}))
            }
            ("job", "submit") => match self.job.submit(args) {
                Ok(h) => CapResult::ok(id, h),
                Err((code, msg)) => CapResult::err(id, &code, msg),
            },
            ("job", "status") => {
                let job_id = args.get("job_id").and_then(|x| x.as_str()).unwrap_or("");
                if job_id.is_empty() {
                    return CapResult::err(id, "job-invalid", "missing job_id");
                }
                match self.job.status(job_id) {
                    Ok(h) => CapResult::ok(id, h.get("job").cloned().unwrap_or(h)),
                    Err((code, msg)) => CapResult::err(id, &code, msg),
                }
            }
            ("job", "cancel") => {
                let job_id = args.get("job_id").and_then(|x| x.as_str()).unwrap_or("");
                if job_id.is_empty() {
                    return CapResult::err(id, "job-invalid", "missing job_id");
                }
                match self.job.cancel(job_id) {
                    Ok(h) => CapResult::ok(id, h),
                    Err((code, msg)) => CapResult::err(id, &code, msg),
                }
            }
            ("job", "list") => match self.job.list() {
                Ok(h) => CapResult::ok(id, h),
                Err((code, msg)) => CapResult::err(id, &code, msg),
            },
            _ => CapResult::err(id, "unsupported", format!("unsupported {cap}.{op}")),
        }
    }
}

fn load_consoles(data: &Path) -> HashMap<String, Console> {
    let mut out = HashMap::new();
    let dir = data.join("bundles");
    let Ok(entries) = std::fs::read_dir(&dir) else {
        return out;
    };
    for ent in entries.flatten() {
        let p = ent.path();
        let Some(fname) = p.file_name().and_then(|s| s.to_str()) else {
            continue;
        };
        let Some(name) = fname.strip_suffix(".tar.gz") else {
            continue;
        };
        if !safe_name(name) {
            continue;
        }
        out.insert(
            name.to_string(),
            Console {
                name: name.to_string(),
                bundle: p.clone(),
                rootfs: data.join("payloads").join(name).join("rootfs"),
                bundle_sha256: String::new(),
                vt: 2,
                started_at_ms: 0,
                child: None,
            },
        );
    }
    out
}

fn sha256_file(path: &Path) -> String {
    Command::new("sha256sum")
        .arg(path)
        .output()
        .ok()
        .and_then(|o| {
            if o.status.success() {
                String::from_utf8(o.stdout)
                    .ok()
                    .and_then(|s| s.split_whitespace().next().map(|x| x.to_string()))
            } else {
                None
            }
        })
        .unwrap_or_default()
}

fn extract_bundle(bundle: &Path, rootfs: &Path) -> Result<(), String> {
    if rootfs.exists() {
        std::fs::remove_dir_all(rootfs).map_err(|e| e.to_string())?;
    }
    std::fs::create_dir_all(rootfs).map_err(|e| e.to_string())?;
    let out = Command::new("tar")
        .arg("-xzf")
        .arg(bundle)
        .arg("-C")
        .arg(rootfs)
        .output()
        .map_err(|e| e.to_string())?;
    if !out.status.success() {
        return Err(String::from_utf8_lossy(&out.stderr).to_string());
    }
    if !rootfs.join("bin/sh").exists() {
        return Err("bundle does not contain /bin/sh".to_string());
    }
    Ok(())
}

fn bundle_put(
    name: &str,
    body: &[u8],
    data: &Path,
    consoles: &Consoles,
) -> Result<(String, usize), (&'static str, String)> {
    if !safe_name(name) {
        return Err(("bundle-invalid", format!("invalid bundle name {name:?}")));
    }
    if body.len() > 32 * 1024 * 1024 {
        return Err(("bundle-too-large", "bundle exceeds 32 MiB".to_string()));
    }
    if body.is_empty() {
        return Err(("bundle-invalid", "empty bundle".to_string()));
    }
    {
        let mut map = consoles.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(c) = map.get_mut(name) {
            refresh_console(c);
            if c.child.is_some() {
                return Err(("console-running", format!("console {name:?} is running")));
            }
        }
    }
    let bdir = data.join("bundles");
    std::fs::create_dir_all(&bdir).map_err(|e| ("bundle-invalid", e.to_string()))?;
    let bundle = bdir.join(format!("{name}.tar.gz"));
    let tmp = bdir.join(format!(".{name}.tmp"));
    std::fs::write(&tmp, body).map_err(|e| ("bundle-invalid", e.to_string()))?;
    std::fs::rename(&tmp, &bundle).map_err(|e| ("bundle-invalid", e.to_string()))?;
    let rootfs = data.join("payloads").join(name).join("rootfs");
    extract_bundle(&bundle, &rootfs).map_err(|e| ("bundle-invalid", e))?;
    let sha = sha256_file(&bundle);
    let mut map = consoles.lock().unwrap_or_else(|e| e.into_inner());
    map.insert(
        name.to_string(),
        Console {
            name: name.to_string(),
            bundle,
            rootfs,
            bundle_sha256: sha.clone(),
            vt: 2,
            started_at_ms: 0,
            child: None,
        },
    );
    Ok((sha, body.len()))
}

fn refresh_console(c: &mut Console) {
    if let Some(child) = c.child.as_mut() {
        if let Ok(Some(_)) = child.try_wait() {
            c.child = None;
        }
    }
}

fn console_json(name: &str, c: &Console) -> Json {
    let pid = c.child.as_ref().map(|ch| ch.id());
    json!({
        "name": name,
        "state": if pid.is_some() { "running" } else { "stopped" },
        "pid": pid,
        "vt": c.vt,
        "rootfs": c.rootfs,
        "bundle": c.bundle,
        "bundle_sha256": c.bundle_sha256,
        "started_at_ms": c.started_at_ms as u64
    })
}

fn console_start(
    name: &str,
    vt: u32,
    consoles: &Consoles,
) -> Result<(u32, u32), (&'static str, String)> {
    if !(2..=6).contains(&vt) {
        return Err(("console-start-failed", format!("vt must be 2..6, got {vt}")));
    }
    let mut map = consoles.lock().unwrap_or_else(|e| e.into_inner());
    let c = map
        .get_mut(name)
        .ok_or(("unknown-console", format!("no console {name:?}")))?;
    refresh_console(c);
    if let Some(ch) = &c.child {
        return Ok((ch.id(), c.vt));
    }
    if !c.rootfs.join("bin/sh").exists() {
        extract_bundle(&c.bundle, &c.rootfs).map_err(|e| ("console-start-failed", e))?;
    }
    let tty = format!("/dev/tty{vt}");
    let f = std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .open(&tty)
        .map_err(|e| ("console-start-failed", format!("{tty}: {e}")))?;
    let out = f
        .try_clone()
        .map_err(|e| ("console-start-failed", e.to_string()))?;
    let err = f
        .try_clone()
        .map_err(|e| ("console-start-failed", e.to_string()))?;
    let _ = Command::new("chvt").arg(vt.to_string()).status();
    let mut cmd = Command::new("setsid");
    cmd.arg("-c")
        .arg("chroot")
        .arg(&c.rootfs)
        .arg("/bin/sh")
        .stdin(Stdio::from(f))
        .stdout(Stdio::from(out))
        .stderr(Stdio::from(err));
    let child = cmd
        .spawn()
        .map_err(|e| ("console-start-failed", format!("setsid/chroot: {e}")))?;
    let pid = child.id();
    c.child = Some(child);
    c.vt = vt;
    c.started_at_ms = now_ms();
    Ok((pid, vt))
}

fn console_write(
    name: &str,
    data: &[u8],
    consoles: &Consoles,
) -> Result<usize, (&'static str, String)> {
    let map = consoles.lock().unwrap_or_else(|e| e.into_inner());
    let c = map
        .get(name)
        .ok_or(("unknown-console", format!("no console {name:?}")))?;
    if c.child.is_none() {
        return Err((
            "console-not-running",
            format!("console {name:?} is stopped"),
        ));
    }
    let tty = format!("/dev/tty{}", c.vt);
    let f = std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .open(&tty)
        .map_err(|e| ("console-write", format!("{tty}: {e}")))?;
    let mut injected = 0usize;
    for &b in data {
        let ch = b as libc::c_char;
        let rc = unsafe {
            libc::ioctl(
                std::os::unix::io::AsRawFd::as_raw_fd(&f),
                libc::TIOCSTI,
                &ch,
            )
        };
        if rc != 0 {
            break;
        }
        injected += 1;
    }
    if injected == 0 {
        return Err((
            "console-write",
            format!("TIOCSTI input injection failed on {tty} (kernel policy?)"),
        ));
    }
    Ok(injected)
}

fn console_restart(
    name: &str,
    vt: u32,
    consoles: &Consoles,
) -> Result<(u32, u32), (&'static str, String)> {
    let old_vt = {
        let map = consoles.lock().unwrap_or_else(|e| e.into_inner());
        map.get(name).map(|c| c.vt)
    };
    console_stop(name, consoles)?;
    console_start(name, old_vt.unwrap_or(vt), consoles)
}

fn console_stop(name: &str, consoles: &Consoles) -> Result<(), (&'static str, String)> {
    let mut map = consoles.lock().unwrap_or_else(|e| e.into_inner());
    let c = map
        .get_mut(name)
        .ok_or(("unknown-console", format!("no console {name:?}")))?;
    refresh_console(c);
    if let Some(mut child) = c.child.take() {
        let pid = child.id() as i32;
        unsafe {
            libc::kill(-pid, libc::SIGTERM);
        }
        let mut reaped = false;
        for _ in 0..30 {
            match child.try_wait() {
                Ok(Some(_)) => {
                    reaped = true;
                    break;
                }
                Ok(None) => thread::sleep(Duration::from_millis(100)),
                Err(_) => break,
            }
        }
        if !reaped {
            unsafe {
                libc::kill(-pid, libc::SIGKILL);
            }
            let _ = child.wait();
        }
    }
    let _ = Command::new("chvt").arg("1").status();
    Ok(())
}

fn net_status() -> Result<Json, String> {
    let dev = std::fs::read_to_string("/proc/net/dev").map_err(|e| e.to_string())?;
    let mut interfaces = Vec::new();
    for line in dev.lines().skip(2) {
        let Some((name, rest)) = line.split_once(':') else {
            continue;
        };
        let name = name.trim();
        let fields: Vec<&str> = rest.split_whitespace().collect();
        interfaces.push(json!({
            "name": name,
            "rx_bytes": fields.first().and_then(|x| x.parse::<u64>().ok()).unwrap_or(0),
            "tx_bytes": fields.get(8).and_then(|x| x.parse::<u64>().ok()).unwrap_or(0)
        }));
    }
    Ok(json!({"interfaces": interfaces}))
}

fn proc_status(pid: u32) -> Result<Json, String> {
    if pid == 0 {
        return Err("pid 0 is not a process".to_string());
    }
    let comm = std::fs::read_to_string(format!("/proc/{pid}/comm"))
        .map_err(|e| e.to_string())?
        .trim()
        .to_string();
    let status =
        std::fs::read_to_string(format!("/proc/{pid}/status")).map_err(|e| e.to_string())?;
    let state = status
        .lines()
        .find(|l| l.starts_with("State:"))
        .map(|l| l.trim_start_matches("State:").trim().to_string())
        .unwrap_or_default();
    Ok(json!({"pid": pid, "comm": comm, "state": state}))
}

fn status_json(name: &str, p: &Payload) -> Json {
    let err = p
        .last_error
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .clone();
    let state = if p.alive.get() {
        "running"
    } else if err.is_some() {
        "failed"
    } else {
        "stopped"
    };
    json!({
        "name": name,
        "state": state,
        "generation": p.generation,
        "started_at_ms": p.started_at_ms as u64,
        "restarts": 0,
        "last_error": err,
        "console": p.manifest.console,
        "tick_ms": p.manifest.tick_ms,
        "autostart": p.manifest.autostart
    })
}

fn safe_name(name: &str) -> bool {
    !name.is_empty()
        && name.len() <= 64
        && name
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_' || c == '.')
        && name != "."
        && name != ".."
}

fn safe_rel(path: &str) -> bool {
    if path.is_empty() || path.len() > 256 || path.starts_with('/') {
        return false;
    }
    path.split('/')
        .all(|c| !c.is_empty() && c != "." && c != "..")
}

fn put(
    header: &Json,
    body: &[u8],
    registry: &Registry,
    data: &Path,
) -> Result<(String, bool), (&'static str, String)> {
    let name = header
        .get("name")
        .and_then(|x| x.as_str())
        .ok_or(("malformed", "missing name".to_string()))?
        .to_string();
    if !safe_name(&name) {
        return Err(("payload-invalid", format!("invalid payload name {name:?}")));
    }
    let listing = std::str::from_utf8(body)
        .map_err(|e| ("malformed", e.to_string()))?
        .to_string();
    let base = data.join("payloads").join(&name);
    let bound = asm::assemble(&listing, &AsmOptions::new(base.to_string_lossy().as_ref()))
        .map_err(|e| ("npattern", e.to_string()))?;

    let mut map = registry.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(p) = map.get(&name) {
        if p.alive.get() {
            return Err((
                "payload-running",
                format!("payload {name:?} is running; stop it first"),
            ));
        }
    }
    let _ = stop_locked(&mut map, &name, &Arc::new(String::new()));

    let seed_json = header.get("seed").cloned();
    let seed = match &seed_json {
        Some(v) if !v.is_null() => {
            let p = data.join("payloads").join(&name).join("seed.json");
            if let Some(parent) = p.parent() {
                std::fs::create_dir_all(parent).map_err(|e| ("payload-invalid", e.to_string()))?;
            }
            std::fs::write(&p, serde_json::to_vec_pretty(v).unwrap_or_default())
                .map_err(|e| ("payload-invalid", e.to_string()))?;
            payload::read_payload(&p).map_err(|e| ("payload-invalid", e.to_string()))?
        }
        _ => Feeds::new(),
    };

    std::fs::create_dir_all(&base).map_err(|e| ("payload-invalid", e.to_string()))?;
    std::fs::write(base.join("program.asm"), &listing)
        .map_err(|e| ("payload-invalid", e.to_string()))?;
    let files: Vec<String> = header
        .get("files")
        .and_then(|x| x.as_array())
        .map(|a| {
            a.iter()
                .filter_map(|v| v.as_str().map(|s| s.to_string()))
                .collect()
        })
        .unwrap_or_default();
    for f in &files {
        if !safe_rel(f) {
            return Err(("payload-invalid", format!("invalid file capability {f:?}")));
        }
    }
    let manifest = Manifest {
        name: name.clone(),
        autostart: header
            .get("autostart")
            .and_then(|x| x.as_bool())
            .unwrap_or(false),
        restart: header
            .get("restart")
            .and_then(|x| x.as_str())
            .unwrap_or("never")
            .to_string(),
        tick_ms: header.get("tick_ms").and_then(|x| x.as_u64()).unwrap_or(0),
        console: header
            .get("console")
            .and_then(|x| x.as_str())
            .unwrap_or("both")
            .to_string(),
        program: "program.asm".to_string(),
        seed: if seed.is_empty() {
            None
        } else {
            Some("seed.json".to_string())
        },
        files,
    };
    std::fs::write(
        base.join("manifest.json"),
        serde_json::to_vec_pretty(&manifest).unwrap_or_default(),
    )
    .map_err(|e| ("payload-invalid", e.to_string()))?;

    let replaced = map.contains_key(&name);
    map.insert(
        name.clone(),
        Payload {
            manifest,
            bound,
            seed,
            dir: base.clone(),
            generation: 0,
            started_at_ms: 0,
            alive: Alive::default(),
            last_error: Arc::new(Mutex::new(None)),
            tx: None,
            join: None,
        },
    );
    Ok((name, replaced))
}

fn start_locked(
    map: &mut HashMap<String, Payload>,
    name: &str,
    bus: &Arc<EventBus>,
    console: &str,
    log: &Arc<String>,
) -> Result<(), (&'static str, String)> {
    let p = map
        .get_mut(name)
        .ok_or(("unknown-payload", format!("no payload {name:?}")))?;
    if p.alive.get() {
        return Ok(());
    }
    if let Some(j) = p.join.take() {
        let _ = j.join();
    }
    let (tx, rx) = mpsc::channel();
    let alive = Alive::default();
    alive.set(true);
    let last_error = Arc::new(Mutex::new(None));
    let m = p.manifest.clone();
    let bound = p.bound.clone();
    let seed = p.seed.clone();
    let dir = p.dir.clone();
    let bus = bus.clone();
    let console = if m.console == "both" || m.console == "local" || m.console == "serial" {
        m.console.clone()
    } else {
        console.to_string()
    };
    let err_slot = last_error.clone();
    let alive_thread = alive.clone();
    let log_thread = log.clone();
    let join = thread::spawn(move || {
        run_payload(
            m,
            bound,
            seed,
            dir,
            bus,
            &console,
            rx,
            alive_thread,
            err_slot,
            log_thread,
        );
    });
    p.tx = Some(tx);
    p.alive = alive;
    p.last_error = last_error;
    p.join = Some(join);
    p.generation += 1;
    p.started_at_ms = now_ms();
    log_line(log, &format!("started payload {name}"));
    Ok(())
}

fn stop_locked(
    map: &mut HashMap<String, Payload>,
    name: &str,
    log: &Arc<String>,
) -> Result<(), (&'static str, String)> {
    let Some(p) = map.get_mut(name) else {
        return Ok(());
    };
    p.alive.set(false);
    if let Some(tx) = p.tx.take() {
        let _ = tx.send(Msg::Stop);
    }
    if let Some(j) = p.join.take() {
        let _ = j.join();
    }
    log_line(log, &format!("stopped payload {name}"));
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn run_payload(
    manifest: Manifest,
    bound: BoundProgram,
    seed: Feeds,
    dir: PathBuf,
    bus: Arc<EventBus>,
    console: &str,
    rx: Receiver<Msg>,
    alive: Alive,
    last_error: Arc<Mutex<Option<String>>>,
    log: Arc<String>,
) {
    let mut stdin: Vec<u8> = Vec::new();
    let mut tick: u64 = 0;
    let mut carry = load_state(&dir);
    loop {
        if !alive.get() {
            break;
        }
        if manifest.tick_ms > 0 {
            match rx.recv_timeout(Duration::from_millis(manifest.tick_ms)) {
                Ok(Msg::Stop) | Err(mpsc::RecvTimeoutError::Disconnected) => break,
                Ok(Msg::Input(b)) => stdin = b,
                Err(mpsc::RecvTimeoutError::Timeout) => {}
            }
        } else {
            match rx.recv() {
                Ok(Msg::Stop) | Err(_) => break,
                Ok(Msg::Input(b)) => stdin = b,
            }
        }
        tick += 1;
        let had_input = !stdin.is_empty();
        let mut feeds = seed.clone();
        // STATE streams are seeded once, then carried from the previous event.
        for name in &bound.state {
            if let Some(v) = carry.get(name) {
                feeds.insert(name.clone(), v.clone());
            }
        }
        for decl in &bound.inputs {
            match decl.name.as_str() {
                "TICK" => {
                    let v = Value::I64(array_from_vec(&[1], vec![now_ms() as i64]).unwrap());
                    feeds.insert("TICK".to_string(), v);
                }
                "STDIN" => {
                    if !stdin.is_empty() || !feeds.contains_key("STDIN") {
                        let v = Value::U8(
                            array_from_vec(&[stdin.len()], stdin.clone())
                                .unwrap_or_else(|_| array_from_vec(&[0], Vec::new()).unwrap()),
                        );
                        feeds.insert("STDIN".to_string(), v);
                    }
                }
                _ => {}
            }
        }
        match runtime::run_with_options(&bound, &feeds, &ExecOptions::default()) {
            Ok(out) => {
                let mut state_changed = false;
                for name in &bound.state {
                    if let Some(v) = out.get(name) {
                        carry.insert(name.clone(), v.clone());
                        state_changed = true;
                    }
                }
                if state_changed {
                    save_state(&dir, &carry);
                }
                let bytes = out.get("STDOUT").or_else(|| out.get("OUT"));
                let data: Option<Vec<u8>> = match bytes {
                    Some(Value::U8(a)) => Some(a.iter().copied().collect()),
                    // Integer ops preserve values but may widen U8 to I64.
                    // Treat I64 STDOUT elements as bytes (0..=255).
                    Some(Value::I64(a)) => Some(a.iter().map(|v| *v as u8).collect()),
                    _ => None,
                };
                if let Some(data) = data {
                    if !data.is_empty() {
                        write_console(console, &data);
                        bus.push("console-out", Some("STDOUT"), data);
                    }
                }
                bus.push("tick", Some("TICK"), Vec::new());
                let _ = tick;
            }
            Err(e) => {
                let msg = e.to_string();
                *last_error.lock().unwrap_or_else(|x| x.into_inner()) = Some(msg.clone());
                log_line(&log, &format!("payload {} error: {msg}", manifest.name));
                bus.push("exit", None, msg.into_bytes());
                break;
            }
        }
        if had_input {
            stdin.clear();
        }
    }
    alive.set(false);
}

fn load_state(dir: &Path) -> Feeds {
    let path = dir.join("state.ndjson");
    if !path.exists() {
        return Feeds::new();
    }
    payload::read_payload(&path).unwrap_or_default()
}

fn save_state(dir: &Path, carry: &Feeds) {
    if carry.is_empty() {
        return;
    }
    let mut streams = serde_json::Map::new();
    for (name, value) in carry {
        if let Ok(j) = payload::value_to_json(value) {
            streams.insert(name.clone(), j);
        }
    }
    let doc = json!({"streams": streams});
    let tmp = dir.join(".state.tmp");
    if std::fs::write(&tmp, serde_json::to_vec_pretty(&doc).unwrap_or_default()).is_ok() {
        let _ = std::fs::rename(&tmp, dir.join("state.ndjson"));
    }
}
