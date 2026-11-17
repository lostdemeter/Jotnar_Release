//! M7/M8 emitted control-plane harness.
//!
//! - `--event/--events`: deterministic emitted-step self test.
//! - `--serve`: boot control daemon; runs the emitted system modules, accepts
//!   `module-*` frames on a Unix socket, and dispatches capabilities through the
//!   native supervisor shim.
//!
//! `module-step` (interpreted) remains the parity oracle; `module-step-emitted`
//! runs the emitted module without side effects.

mod generated {
    include!(concat!(env!("OUT_DIR"), "/system_modules.rs"));
}

use std::collections::HashMap;
use std::io::{Read, Write};
use std::net::TcpStream;
use std::os::unix::fs::PermissionsExt;
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use jotnar::protocol::{decode_frame, encode_frame, PROTO_V};
use jotnar_node_tools::capability::{CapResult, Command, ModuleManifest};
use jotnar_node_tools::{error_frame, ok_frame};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value as Json};

#[derive(Serialize, Deserialize, Default)]
struct Persist {
    actions: i64,
}

struct NetState {
    handles: Mutex<HashMap<u64, TcpStream>>,
    next: AtomicU64,
}

impl NetState {
    fn new() -> Self {
        Self {
            handles: Mutex::new(HashMap::new()),
            next: AtomicU64::new(1),
        }
    }
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.iter().any(|a| a == "--serve") {
        let mut socket = "/run/jotnar/osd.sock".to_string();
        let mut supervisor = "/run/jotnar/supervisor.sock".to_string();
        let mut state_root = "/var/lib/jotnar/osd".to_string();
        let mut manifest_dir = "/usr/local/share/jotnar/modules".to_string();
        let mut it = args.iter();
        while let Some(a) = it.next() {
            match a.as_str() {
                "--serve" => {}
                "--socket" => socket = it.next().cloned().unwrap_or(socket),
                "--supervisor" => supervisor = it.next().cloned().unwrap_or(supervisor),
                "--state-root" => state_root = it.next().cloned().unwrap_or(state_root),
                "--manifest-dir" => manifest_dir = it.next().cloned().unwrap_or(manifest_dir),
                "-h" | "--help" => {
                    println!("jotnar-osd --serve [--socket PATH] [--supervisor PATH] [--state-root DIR] [--manifest-dir DIR]");
                    return;
                }
                _ => {}
            }
        }
        if let Err(e) = serve(&socket, &supervisor, &state_root, &manifest_dir) {
            eprintln!("jotnar-osd serve: {e}");
            std::process::exit(1);
        }
        return;
    }

    let mut event: Option<i64> = None;
    let mut events: Option<Vec<i64>> = None;
    let mut manifest_path =
        "/usr/local/share/jotnar/modules/console-policy/manifest.json".to_string();
    let mut it = std::env::args().skip(1);
    while let Some(a) = it.next() {
        match a.as_str() {
            "--event" => event = it.next().and_then(|v| v.parse().ok()),
            "--events" => {
                events = it.next().map(|v| {
                    v.split(',')
                        .filter_map(|x| x.trim().parse().ok())
                        .collect::<Vec<i64>>()
                });
            }
            "--manifest" => manifest_path = it.next().unwrap_or(manifest_path),
            "-h" | "--help" => {
                println!("jotnar-osd --event N | --events N,N,N | --serve");
                return;
            }
            other => {
                eprintln!("unknown argument {other:?}");
                std::process::exit(2);
            }
        }
    }
    let manifest = load_manifest(&manifest_path);
    let mut module =
        generated::SystemModule::new("console-policy").expect("console-policy generated");

    if let Some(events) = events {
        let mut steps = Vec::new();
        for (i, event) in events.iter().copied().enumerate() {
            let action = module.step(event);
            let (cap, op, args) = action_to_command(action);
            let command = Command::new((i + 1) as u64, module.name(), cap, op, args);
            let result = if !manifest.granted(cap, op) {
                CapResult::err(
                    (i + 1) as u64,
                    "denied",
                    format!("grant missing for {cap}.{op}"),
                )
            } else {
                CapResult::ok((i + 1) as u64, json!({"dispatched": false}))
            };
            steps.push(json!({
                "event": event,
                "action": action,
                "command": command,
                "result": result,
                "state": module.state_json()
            }));
        }
        println!(
            "{}",
            serde_json::to_string_pretty(&json!({
                "mode": "emitted", "module": "console-policy", "steps": steps
            }))
            .unwrap()
        );
        return;
    }

    let Some(event) = event else {
        eprintln!("usage: jotnar-osd --event N | --events N,N,N | --serve");
        std::process::exit(2);
    };
    let action = module.step(event);
    let (cap, op, args) = action_to_command(action);
    let command = Command::new(1, module.name(), cap, op, args);
    let result = if !manifest.granted(cap, op) {
        CapResult::err(1, "denied", format!("grant missing for {cap}.{op}"))
    } else {
        CapResult::ok(1, json!({"dispatched": false}))
    };
    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "mode": "emitted", "module": "console-policy", "event": event,
            "action": action, "command": command, "result": result,
            "state": module.state_json()
        }))
        .unwrap()
    );
}

fn action_to_command(action: i64) -> (&'static str, &'static str, Json) {
    match action {
        1 => ("console", "start", json!({"name": "console", "vt": 2})),
        2 => ("console", "stop", json!({"name": "console"})),
        3 => ("console", "status", json!({"name": "console"})),
        4 => ("system", "log", json!({"message": "system.log"})),
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
    }
}

fn load_manifest(path: &str) -> ModuleManifest {
    std::fs::read(path)
        .ok()
        .and_then(|b| serde_json::from_slice(&b).ok())
        .unwrap_or_else(|| {
            eprintln!("cannot load/parse manifest {path}");
            std::process::exit(2);
        })
}

fn save_state(path: &Path, actions: i64) {
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    let tmp = path.with_extension("tmp");
    let doc = json!({"actions": actions});
    if std::fs::write(&tmp, serde_json::to_vec_pretty(&doc).unwrap_or_default()).is_ok() {
        let _ = std::fs::rename(&tmp, path);
    }
}

fn load_state(path: &Path) -> i64 {
    std::fs::read(path)
        .ok()
        .and_then(|b| serde_json::from_slice::<Persist>(&b).ok())
        .map(|p| p.actions)
        .unwrap_or(0)
}

fn supervisor_call(
    path: &str,
    typ: &str,
    fields: Json,
    payload: &[u8],
) -> Result<Json, (String, String)> {
    let mut s =
        UnixStream::connect(path).map_err(|e| ("supervisor-unavailable".into(), e.to_string()))?;
    let _ = s.set_read_timeout(Some(Duration::from_secs(15)));
    let _ = s.set_write_timeout(Some(Duration::from_secs(15)));
    let mut h = fields;
    if let Some(o) = h.as_object_mut() {
        o.insert("v".into(), json!(PROTO_V));
        o.insert("id".into(), json!(1));
        o.insert("type".into(), json!(typ));
    }
    let frame = encode_frame(&h, payload).map_err(|e| ("encode".into(), e.to_string()))?;
    s.write_all(&frame)
        .map_err(|e| ("supervisor-write".into(), e.to_string()))?;
    let mut hdr = [0u8; 4];
    s.read_exact(&mut hdr)
        .map_err(|e| ("supervisor-read".into(), e.to_string()))?;
    let n = u32::from_be_bytes(hdr) as usize;
    let mut body = vec![0u8; n];
    s.read_exact(&mut body)
        .map_err(|e| ("supervisor-read".into(), e.to_string()))?;
    let mut full = Vec::with_capacity(4 + n);
    full.extend_from_slice(&hdr);
    full.extend_from_slice(&body);
    let (rh, _) = decode_frame(&full).map_err(|e| ("supervisor-decode".into(), e.to_string()))?;
    if rh.get("ok").and_then(|x| x.as_bool()) != Some(true) {
        let code = rh
            .get("error")
            .and_then(|e| e.get("code"))
            .and_then(|x| x.as_str())
            .unwrap_or("capability-error");
        let msg = rh
            .get("error")
            .and_then(|e| e.get("msg"))
            .and_then(|x| x.as_str())
            .unwrap_or("capability failed");
        return Err((code.to_string(), msg.to_string()));
    }
    Ok(rh)
}

fn run_step(
    module: &mut generated::SystemModule,
    manifest: &ModuleManifest,
    module_dir: &Path,
    net: &NetState,
    event: &str,
    dispatch: bool,
    supervisor: &str,
    override_args: &Json,
) -> Json {
    let code = match event {
        "console.start" => 10,
        "console.stop" => 11,
        "console.status" => 12,
        "system.log" => 30,
        "file.write" => 40,
        "file.read" => 41,
        "file.list" => 42,
        "proc.status" => 60,
        "proc.signal" => 61,
        "timer.arm" => 70,
        "timer.cancel" => 71,
        "timer.fired" => 72,
        "net.status" => 80,
        "net.connect" => 81,
        "net.send" => 82,
        "net.recv" => 83,
        "net.close" => 84,
        "job.submit" => 90,
        "job.status" => 91,
        "job.cancel" => 92,
        "job.list" => 93,
        _ => 0,
    };
    if code == 0 {
        return json!({"module": module.name(), "event": event,
                      "error": {"code": "module-invalid", "msg": "unknown event"}});
    }
    let action = module.step(code);
    let (cap, op, mut args) = action_to_command(action);
    if cap == "system" && op == "log" {
        args = json!({"message": event});
    }
    if cap == "job" {
        if let (Some(dst), Some(src)) = (args.as_object_mut(), override_args.as_object()) {
            for (k, v) in src {
                dst.insert(k.clone(), v.clone());
            }
        }
    }
    let command = Command::new(1, module.name(), cap, op, args.clone());
    let result = if !manifest.granted(cap, op) {
        CapResult::err(1, "denied", format!("grant missing for {cap}.{op}"))
    } else if !dispatch {
        CapResult::ok(1, json!({"dispatched": false}))
    } else {
        match (cap, op) {
            ("console", "start") => {
                let vt = args.get("vt").and_then(|x| x.as_u64()).unwrap_or(2);
                match supervisor_call(
                    supervisor,
                    "console-start",
                    json!({"name": "console", "vt": vt}),
                    &[],
                ) {
                    Ok(h) => CapResult::ok(
                        1,
                        json!({
                            "pid": h.get("pid").cloned().unwrap_or(Json::Null),
                            "vt": h.get("vt").cloned().unwrap_or(Json::Null)
                        }),
                    ),
                    Err((code, msg)) => CapResult::err(1, &code, msg),
                }
            }
            ("console", "stop") => {
                match supervisor_call(supervisor, "console-stop", json!({"name": "console"}), &[]) {
                    Ok(_) => CapResult::ok(1, json!({"state": "stopped"})),
                    Err((code, msg)) => CapResult::err(1, &code, msg),
                }
            }
            ("console", "status") => match supervisor_call(
                supervisor,
                "console-status",
                json!({"name": "console"}),
                &[],
            ) {
                Ok(h) => CapResult::ok(1, h),
                Err((code, msg)) => CapResult::err(1, &code, msg),
            },
            ("system", "log") => CapResult::ok(1, json!({"logged": "system.log"})),
            ("file", "write") => file_write(module_dir, &args),
            ("file", "read") => file_read(module_dir, &args),
            ("file", "list") => file_list(module_dir, &args),
            ("proc", "status") => {
                let pid = args.get("pid").and_then(|x| x.as_u64()).unwrap_or(0) as u32;
                match proc_status(pid) {
                    Ok(v) => CapResult::ok(1, v),
                    Err(e) => CapResult::err(1, "proc-error", e),
                }
            }
            ("proc", "signal") => {
                let pid = args.get("pid").and_then(|x| x.as_u64()).unwrap_or(0) as u32;
                let sig = args.get("signal").and_then(|x| x.as_i64()).unwrap_or(0) as i32;
                if sig != 0 {
                    CapResult::err(1, "unsupported", "only signal 0 is supported in v1")
                } else if unsafe { libc::kill(pid as i32, 0) } == 0 {
                    CapResult::ok(1, json!({"pid": pid, "signal": sig, "alive": true}))
                } else {
                    CapResult::err(1, "proc-error", format!("pid {pid} not alive"))
                }
            }
            ("timer", "arm") => {
                let delay = args.get("delay_ms").and_then(|x| x.as_u64()).unwrap_or(500);
                let key = args.get("key").and_then(|x| x.as_str()).unwrap_or("timer");
                CapResult::ok(1, json!({"armed": true, "key": key, "delay_ms": delay}))
            }
            ("timer", "cancel") => {
                let key = args.get("key").and_then(|x| x.as_str()).unwrap_or("timer");
                CapResult::ok(1, json!({"cancelled": true, "key": key}))
            }
            ("net", "status") => match net_status() {
                Ok(v) => CapResult::ok(1, v),
                Err(e) => CapResult::err(1, "net-error", e),
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
                        let handle = net.next.fetch_add(1, Ordering::SeqCst);
                        net.handles
                            .lock()
                            .unwrap_or_else(|e| e.into_inner())
                            .insert(handle, s);
                        CapResult::ok(1, json!({"handle": handle, "host": host, "port": port}))
                    }
                    Err(e) => CapResult::err(1, "net-error", e.to_string()),
                }
            }
            ("net", "send") => {
                let handle = args.get("handle").and_then(|x| x.as_u64()).unwrap_or(0);
                let data = args.get("data").and_then(|x| x.as_str()).unwrap_or("");
                let mut map = net.handles.lock().unwrap_or_else(|e| e.into_inner());
                match map.get_mut(&handle) {
                    Some(s) => match s.write_all(data.as_bytes()) {
                        Ok(()) => CapResult::ok(1, json!({"handle": handle, "bytes": data.len()})),
                        Err(e) => CapResult::err(1, "net-error", e.to_string()),
                    },
                    None => CapResult::err(1, "net-error", format!("no handle {handle}")),
                }
            }
            ("net", "recv") => {
                let handle = args.get("handle").and_then(|x| x.as_u64()).unwrap_or(0);
                let max = args.get("max").and_then(|x| x.as_u64()).unwrap_or(4096) as usize;
                let mut buf = vec![0u8; max.min(65536)];
                let mut map = net.handles.lock().unwrap_or_else(|e| e.into_inner());
                match map.get_mut(&handle) {
                    Some(s) => match s.read(&mut buf) {
                        Ok(n) => {
                            buf.truncate(n);
                            CapResult::ok(
                                1,
                                json!({
                                    "handle": handle, "bytes": n,
                                    "text": String::from_utf8_lossy(&buf)
                                }),
                            )
                        }
                        Err(e) => CapResult::err(1, "net-error", e.to_string()),
                    },
                    None => CapResult::err(1, "net-error", format!("no handle {handle}")),
                }
            }
            ("net", "close") => {
                let handle = args.get("handle").and_then(|x| x.as_u64()).unwrap_or(0);
                net.handles
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .remove(&handle);
                CapResult::ok(1, json!({"closed": handle}))
            }
            ("job", "submit") => match supervisor_call(supervisor, "job-submit", args.clone(), &[])
            {
                Ok(h) => CapResult::ok(1, h),
                Err((code, msg)) => CapResult::err(1, &code, msg),
            },
            ("job", "status") => {
                let job_id = args.get("job_id").and_then(|x| x.as_str()).unwrap_or("");
                match supervisor_call(supervisor, "job-status", json!({"job_id": job_id}), &[]) {
                    Ok(h) => CapResult::ok(1, h),
                    Err((code, msg)) => CapResult::err(1, &code, msg),
                }
            }
            ("job", "cancel") => {
                let job_id = args.get("job_id").and_then(|x| x.as_str()).unwrap_or("");
                match supervisor_call(supervisor, "job-cancel", json!({"job_id": job_id}), &[]) {
                    Ok(h) => CapResult::ok(1, h),
                    Err((code, msg)) => CapResult::err(1, &code, msg),
                }
            }
            ("job", "list") => match supervisor_call(supervisor, "job-list", json!({}), &[]) {
                Ok(h) => CapResult::ok(1, h),
                Err((code, msg)) => CapResult::err(1, &code, msg),
            },
            _ => CapResult::err(1, "unsupported", format!("unsupported {cap}.{op}")),
        }
    };
    json!({
        "mode": "emitted",
        "module": module.name(),
        "event": event,
        "action": action,
        "command": command,
        "result": result,
        "state": module.state_json()
    })
}

fn file_write(module_dir: &Path, args: &Json) -> CapResult {
    let path = args.get("path").and_then(|x| x.as_str()).unwrap_or("");
    if !safe_rel(path) {
        return CapResult::err(1, "payload-invalid", "invalid path");
    }
    let data = args
        .get("data")
        .and_then(|x| x.as_str())
        .unwrap_or("")
        .as_bytes();
    let full = module_dir.join("files").join(path);
    if let Some(parent) = full.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    match std::fs::write(&full, data) {
        Ok(()) => CapResult::ok(1, json!({"path": path, "bytes": data.len()})),
        Err(e) => CapResult::err(1, "file-error", e.to_string()),
    }
}

fn file_read(module_dir: &Path, args: &Json) -> CapResult {
    let path = args.get("path").and_then(|x| x.as_str()).unwrap_or("");
    if !safe_rel(path) {
        return CapResult::err(1, "payload-invalid", "invalid path");
    }
    match std::fs::read(module_dir.join("files").join(path)) {
        Ok(bytes) => CapResult::ok(
            1,
            json!({
                "path": path, "bytes": bytes.len(),
                "text": String::from_utf8_lossy(&bytes)
            }),
        ),
        Err(e) => CapResult::err(1, "file-error", e.to_string()),
    }
}

fn file_list(module_dir: &Path, args: &Json) -> CapResult {
    let path = args.get("path").and_then(|x| x.as_str()).unwrap_or("");
    let full = if path.is_empty() {
        module_dir.join("files")
    } else if safe_rel(path) {
        module_dir.join("files").join(path)
    } else {
        return CapResult::err(1, "payload-invalid", "invalid path");
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
    CapResult::ok(1, json!({"path": path, "entries": entries}))
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

fn module_state_path(root: &str, name: &str) -> PathBuf {
    Path::new(root).join(name).join("state.json")
}

fn safe_rel(path: &str) -> bool {
    if path.is_empty() || path.len() > 256 || path.starts_with('/') {
        return false;
    }
    path.split('/')
        .all(|c| !c.is_empty() && c != "." && c != "..")
}

fn serve(
    socket: &str,
    supervisor: &str,
    state_root: &str,
    manifest_dir: &str,
) -> std::io::Result<()> {
    if let Some(parent) = Path::new(socket).parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    let _ = std::fs::remove_file(socket);
    let listener = UnixListener::bind(socket)?;
    let _ = std::fs::set_permissions(socket, std::fs::Permissions::from_mode(0o600));

    let mut modules: HashMap<String, generated::SystemModule> = HashMap::new();
    let mut manifests: HashMap<String, ModuleManifest> = HashMap::new();
    for name in generated::NAMES {
        let mpath = Path::new(manifest_dir).join(name).join("manifest.json");
        let manifest: ModuleManifest = match std::fs::read(&mpath)
            .ok()
            .and_then(|b| serde_json::from_slice(&b).ok())
        {
            Some(m) => m,
            None => {
                eprintln!("jotnar-osd: cannot load manifest {}", mpath.display());
                continue;
            }
        };
        let mut module = generated::SystemModule::new(name).expect("generated module");
        let sp = module_state_path(state_root, name);
        module.set_actions(load_state(&sp));
        modules.insert(name.to_string(), module);
        manifests.insert(name.to_string(), manifest);
    }
    eprintln!(
        "jotnar-osd listening on {socket} (mode=emitted, {} modules, supervisor={supervisor})",
        modules.len()
    );

    listener.set_nonblocking(true)?;
    let (timer_tx, timer_rx) = mpsc::channel::<String>();
    let timers: Arc<Mutex<HashMap<String, Arc<AtomicBool>>>> = Arc::new(Mutex::new(HashMap::new()));
    let net = NetState::new();

    loop {
        while let Ok(name) = timer_rx.try_recv() {
            if let (Some(module), Some(manifest)) = (modules.get_mut(&name), manifests.get(&name)) {
                let module_dir = Path::new(state_root).join(&name);
                let _ = run_step(
                    module,
                    manifest,
                    &module_dir,
                    &net,
                    "timer.fired",
                    true,
                    supervisor,
                    &json!({}),
                );
                save_state(&module_state_path(state_root, &name), module.actions());
                eprintln!("jotnar-osd: timer fired for {name}");
            }
        }
        let conn = match listener.accept() {
            Ok(c) => c,
            Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                std::thread::sleep(Duration::from_millis(50));
                continue;
            }
            Err(e) => {
                eprintln!("jotnar-osd accept: {e}");
                continue;
            }
        };
        let mut stream = conn.0;
        let _ = stream.set_nonblocking(false);
        let _ = stream.set_read_timeout(Some(Duration::from_secs(15)));
        let _ = stream.set_write_timeout(Some(Duration::from_secs(15)));
        loop {
            let mut hdr = [0u8; 4];
            if stream.read_exact(&mut hdr).is_err() {
                break;
            }
            let n = u32::from_be_bytes(hdr) as usize;
            let mut body = vec![0u8; n];
            if stream.read_exact(&mut body).is_err() {
                break;
            }
            let mut frame = Vec::with_capacity(4 + n);
            frame.extend_from_slice(&hdr);
            frame.extend_from_slice(&body);
            let (header, _) = match decode_frame(&frame) {
                Ok(x) => x,
                Err(e) => {
                    let _ = stream.write_all(&error_frame(&json!(0), "malformed", e.to_string()));
                    continue;
                }
            };
            let id = header.get("id").cloned().unwrap_or(json!(0));
            let typ = header.get("type").and_then(|x| x.as_str()).unwrap_or("");
            let name = header
                .get("module")
                .and_then(|x| x.as_str())
                .or_else(|| header.get("name").and_then(|x| x.as_str()))
                .unwrap_or("");
            let out = match typ {
                "module-event" | "module-step-emitted" => {
                    let Some(module) = modules.get_mut(name) else {
                        let _ = stream.write_all(&error_frame(
                            &id,
                            "unknown-module",
                            format!("no module {name:?}"),
                        ));
                        continue;
                    };
                    let Some(manifest) = manifests.get(name) else {
                        let _ = stream.write_all(&error_frame(
                            &id,
                            "unknown-module",
                            format!("no manifest {name:?}"),
                        ));
                        continue;
                    };
                    let event = header.get("event").and_then(|x| x.as_str()).unwrap_or("");
                    let module_args = header.get("args").cloned().unwrap_or_else(|| json!({}));
                    let dispatch = typ == "module-event";
                    let module_dir = Path::new(state_root).join(name);
                    let v = run_step(
                        module,
                        manifest,
                        &module_dir,
                        &net,
                        event,
                        dispatch,
                        supervisor,
                        &module_args,
                    );
                    if dispatch {
                        let action = v.get("action").and_then(|x| x.as_i64()).unwrap_or(0);
                        if action == 11 {
                            let args = v.get("command").and_then(|c| c.get("args"));
                            let key = args
                                .and_then(|a| a.get("key"))
                                .and_then(|x| x.as_str())
                                .unwrap_or("timer")
                                .to_string();
                            let delay = args
                                .and_then(|a| a.get("delay_ms"))
                                .and_then(|x| x.as_u64())
                                .unwrap_or(500);
                            let cancel = Arc::new(AtomicBool::new(false));
                            timers
                                .lock()
                                .unwrap_or_else(|e| e.into_inner())
                                .insert(key.clone(), cancel.clone());
                            let tx = timer_tx.clone();
                            let n = name.to_string();
                            std::thread::spawn(move || {
                                std::thread::sleep(Duration::from_millis(delay));
                                if !cancel.load(Ordering::SeqCst) {
                                    let _ = tx.send(n);
                                }
                            });
                        } else if action == 12 {
                            let key = v
                                .get("command")
                                .and_then(|c| c.get("args"))
                                .and_then(|a| a.get("key"))
                                .and_then(|x| x.as_str())
                                .unwrap_or("timer");
                            if let Some(flag) =
                                timers.lock().unwrap_or_else(|e| e.into_inner()).get(key)
                            {
                                flag.store(true, Ordering::SeqCst);
                            }
                        }
                        save_state(&module_state_path(state_root, name), module.actions());
                    }
                    let rtyp = if dispatch {
                        "module-ok"
                    } else {
                        "module-step-emitted"
                    };
                    ok_frame(&id, rtyp, v, &[])
                }
                "module-status" | "module-list" => {
                    let want = header
                        .get("module")
                        .and_then(|x| x.as_str())
                        .or_else(|| header.get("name").and_then(|x| x.as_str()));
                    let mut list = Vec::new();
                    let mut names: Vec<&String> = modules.keys().collect();
                    names.sort();
                    for n in names {
                        if let Some(w) = want {
                            if w != n {
                                continue;
                            }
                        }
                        let m = &modules[n];
                        let caps = manifests
                            .get(n)
                            .map(|x| serde_json::to_value(&x.capabilities).unwrap_or(Json::Null))
                            .unwrap_or(Json::Null);
                        list.push(json!({
                            "name": n,
                            "mode": "emitted",
                            "actions": m.actions(),
                            "state": m.state_json(),
                            "capabilities": caps,
                            "state_file": module_state_path(state_root, n)
                        }));
                    }
                    if let Some(w) = want {
                        if list.is_empty() {
                            let _ = stream.write_all(&error_frame(
                                &id,
                                "unknown-module",
                                format!("no module {w:?}"),
                            ));
                            continue;
                        }
                    }
                    ok_frame(&id, "module-status", json!({"modules": list}), &[])
                }
                _ => error_frame(&id, "unknown-type", format!("no such message {typ:?}")),
            };
            if stream.write_all(&out).is_err() {
                break;
            }
        }
    }
    Ok(())
}
