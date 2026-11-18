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

use std::collections::HashMap;
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
use jotnar::codegen::{build, compile_program, CompileOptions, Target};
use jotnar::payload::{feeds_from_json, value_to_json, Feeds};
use jotnar::protocol::{decode_frame, encode_frame, PROTO_V, RUNTIME_VERSION};
use jotnar::value::{array_from_vec, Value};
use mdns_sd::{ServiceDaemon, ServiceInfo};
use serde_json::{json, Value as Json};

const DEFAULT_BIND: &str = "127.0.0.1:9997";
const DEFAULT_MAX_FRAME: usize = 64 * 1024 * 1024;

#[derive(Clone)]
struct Config {
    bind: String,
    token: Option<String>,
    target: Target,
    max_frame: usize,
    cache_dir: PathBuf,
    build_lock: Arc<Mutex<()>>,
    mdns: bool,
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

fn real_main() -> Result<(), String> {
    let mut bind = DEFAULT_BIND.to_string();
    let mut token = None;
    let mut target = None;
    let mut max_frame = DEFAULT_MAX_FRAME;
    let mut cache_dir = std::env::temp_dir().join("jotnar-gpu-runner-cache");
    let mut mdns = false;
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
            "-h" | "--help" => return Err(usage().to_string()),
            other => return Err(format!("unknown argument '{other}'\n\n{}", usage())),
        }
        i += 1;
    }
    let target = target.ok_or("--target hip|cuda is required")?;
    let cfg = Config {
        bind,
        token,
        target,
        max_frame: max_frame.max(1024),
        cache_dir,
        build_lock: Arc::new(Mutex::new(())),
        mdns,
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
        "run" => {
            if let Err(e) = run(&mut stream, cfg, &id, &header, payload) {
                send_error(&mut stream, &id, "run-failed", &e)?;
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
    let body: Json = serde_json::from_slice(payload).map_err(|e| format!("bad run body: {e}"))?;
    let listing = body
        .get("listing")
        .and_then(|x| x.as_str())
        .ok_or("run body needs listing")?;
    let payload_json = body.get("payload").ok_or("run body needs payload")?;
    let use_fp16 = body
        .get("use_fp16")
        .and_then(|x| x.as_bool())
        .unwrap_or(false);
    let feeds: Feeds = feeds_from_json(payload_json).map_err(|e| e.to_string())?;
    let opts = CompileOptions {
        target: cfg.target,
        outputs: outputs.clone(),
        asm: AsmOptions::new("."),
        use_fp16,
    };
    let t0 = Instant::now();
    let artifact =
        compile_program(listing, &opts, feeds.clone()).map_err(|e| format!("compile: {e}"))?;
    let compile_ms = t0.elapsed().as_millis();
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
    let t1 = Instant::now();
    let mut cmd = Command::new(&exe);
    cmd.args(&argv);
    let run = cmd.output().map_err(|e| format!("run: {e}"))?;
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
    let mut out = serde_json::Map::new();
    for (name, path) in output_paths {
        let info = artifact
            .streams
            .get(&name)
            .ok_or_else(|| format!("no stream info for output {name:?}"))?;
        let value = read_value(&path, info.kind, cfg.target, &info.shape)?;
        out.insert(name, value_to_json(&value).map_err(|e| e.to_string())?);
    }
    let body = json!({
        "outputs": out,
        "timing": {"compile_ms": compile_ms, "build_ms": build_ms, "run_ms": run_ms, "cache_hit": cache_hit},
    });
    send(
        stream,
        &json!({
            "v": PROTO_V,
            "id": id,
            "type": "run-ok",
            "ok": true,
            "target": cfg.target.as_str(),
            "runtime_version": RUNTIME_VERSION,
            "compile_ms": compile_ms,
            "build_ms": build_ms,
            "run_ms": run_ms,
            "cache_hit": cache_hit,
        }),
        body.to_string().as_bytes(),
    )
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

fn write_value(path: &Path, value: &Value, kind: CKind, target: Target) -> Result<(), String> {
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
    fs::write(path, bytes).map_err(|e| format!("write {}: {e}", path.display()))
}

fn read_value(
    path: &Path,
    kind: CKind,
    target: Target,
    shape: &[Option<usize>],
) -> Result<Value, String> {
    let shape: Vec<usize> = shape
        .iter()
        .map(|x| x.ok_or_else(|| "dynamic output shape is not supported".to_string()))
        .collect::<Result<_, _>>()?;
    let n = shape.iter().product::<usize>();
    let bytes = fs::read(path).map_err(|e| format!("read {}: {e}", path.display()))?;
    match kind {
        CKind::F => {
            if matches!(target, Target::Cuda | Target::Hip) {
                let data = bytes
                    .chunks_exact(4)
                    .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                    .collect::<Vec<_>>();
                if data.len() != n {
                    return Err(format!(
                        "{} has {} floats, expected {n}",
                        path.display(),
                        data.len()
                    ));
                }
                let arr = array_from_vec(&shape, data).map_err(|e| e.to_string())?;
                Ok(Value::F32(arr))
            } else {
                let data = bytes
                    .chunks_exact(8)
                    .map(|b| f64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]]))
                    .collect::<Vec<_>>();
                if data.len() != n {
                    return Err(format!(
                        "{} has {} doubles, expected {n}",
                        path.display(),
                        data.len()
                    ));
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
                return Err(format!(
                    "{} has {} ints, expected {n}",
                    path.display(),
                    data.len()
                ));
            }
            let arr = array_from_vec(&shape, data).map_err(|e| e.to_string())?;
            Ok(Value::I64(arr))
        }
        CKind::T => Err("T streams are not supported by the GPU runner".to_string()),
    }
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
