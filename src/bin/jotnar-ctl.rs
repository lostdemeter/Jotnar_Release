//! Framed control client for `jotnar-node`.
//!
//! Talks to a node over TCP using the same frame codec as the server
//! (`u32BE || JSON header || payload`). Drive a listing, blob shelf,
//! and `EXEC` requests from the shell without writing Rust.
//!
//! ```sh
//! cargo run --release --bin jotnar-ctl -- --addr 127.0.0.1:9999 ping
//! cargo run --release --bin jotnar-ctl -- --addr 127.0.0.1:9999 program tile showcase/language/matmul2.asm
//! cargo run --release --bin jotnar-ctl -- --addr 127.0.0.1:9999 exec tile --payload /tmp/p.json --outputs c00
//! ```
//!
//! `exec --payload` takes the same JSON shape as `jotnar run --payload`.
//! Results are printed as JSON on stdout (streams decoded where possible).

use std::io::{Read, Write};
use std::net::TcpStream;
use std::path::PathBuf;
use std::process::ExitCode;

use jotnar::payload::{self, Feeds};
use jotnar::protocol::{decode_frame, decode_stream, encode_stream, StreamSpec};
use serde_json::{json, Value as Json};

fn usage() -> &'static str {
    "jotnar-ctl - framed client for jotnar-node\n\
     \n\
     usage:\n\
       jotnar-ctl [--addr ADDR] [--token SECRET] ping\n\
       jotnar-ctl [--addr ADDR] [--token SECRET] list\n\
       jotnar-ctl [--addr ADDR] [--token SECRET] program <name> <file.asm> [--backend lattice]\n\
       jotnar-ctl [--addr ADDR] [--token SECRET] store <name> --payload <payload.json> --stream <NAME>\n\
       jotnar-ctl [--addr ADDR] [--token SECRET] exec <handle> --payload <payload.json> --outputs a,b [--threads N] [--policy reference|auto] [--receipt]\n\
       jotnar-ctl [--addr ADDR] [--token SECRET] drop <name>\n\
     \n\
     --addr defaults to 127.0.0.1:9999. --payload is the JSON shape from\n\
     `jotnar run --payload` (`{\"streams\": {...}}`).\n"
}

fn connect(addr: &str) -> Result<TcpStream, String> {
    let stream = TcpStream::connect(addr).map_err(|e| format!("cannot connect to {addr}: {e}"))?;
    let _ = stream.set_nodelay(true);
    let _ = stream.set_read_timeout(Some(std::time::Duration::from_secs(30)));
    let _ = stream.set_write_timeout(Some(std::time::Duration::from_secs(30)));
    Ok(stream)
}

/// Send one framed request, return `(header, payload)` of the reply.
fn roundtrip(
    stream: &mut TcpStream,
    id: u64,
    typ: &str,
    fields: Json,
    payload: &[u8],
    token: &Option<String>,
) -> Result<(Json, Vec<u8>), String> {
    let mut fields = fields;
    if let Some(t) = token {
        if let Some(o) = fields.as_object_mut() {
            o.insert("auth".into(), json!(t));
        }
    }
    let frame = jotnar::protocol::request(id, typ, fields, payload).map_err(|e| e.to_string())?;
    stream
        .write_all(&frame)
        .map_err(|e| format!("write: {e}"))?;
    let mut hdr = [0u8; 4];
    stream
        .read_exact(&mut hdr)
        .map_err(|e| format!("read header: {e}"))?;
    let n = u32::from_be_bytes(hdr) as usize;
    const MAX_CLIENT_FRAME: usize = 256 * 1024 * 1024;
    if n > MAX_CLIENT_FRAME {
        return Err(format!(
            "response frame too large ({n} > {MAX_CLIENT_FRAME})"
        ));
    }
    let mut body = vec![0u8; n];
    stream
        .read_exact(&mut body)
        .map_err(|e| format!("read body: {e}"))?;
    let mut full = Vec::with_capacity(4 + n);
    full.extend_from_slice(&hdr);
    full.extend_from_slice(&body);
    let (header, payload) = decode_frame(&full).map_err(|e| e.to_string())?;
    if header.get("id").and_then(|x| x.as_u64()) != Some(id) {
        return Err(format!("response id mismatch: expected {id}, got {header}"));
    }
    if header.get("v").and_then(|x| x.as_u64()) != Some(jotnar::protocol::PROTO_V) {
        return Err(format!(
            "response version mismatch: {}",
            header.get("v").cloned().unwrap_or(json!("?"))
        ));
    }
    Ok((header, payload))
}

fn ok_or_show(header: &Json) -> Result<(), String> {
    if header.get("ok").and_then(|x| x.as_bool()) == Some(true) {
        Ok(())
    } else {
        Err(format!(
            "node error: {}",
            header.get("error").cloned().unwrap_or(json!("?"))
        ))
    }
}

fn main() -> ExitCode {
    match real_main() {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("{e}");
            ExitCode::FAILURE
        }
    }
}

enum Plan {
    Ping,
    List,
    Program {
        name: String,
        file: PathBuf,
        backend: String,
    },
    Store {
        name: String,
        payload: PathBuf,
        stream: String,
    },
    Exec {
        handle: String,
        payload: PathBuf,
        outputs: Vec<String>,
        threads: Option<usize>,
        policy: Option<String>,
        receipt: bool,
    },
    Drop {
        name: String,
    },
}

fn parse_plan(cmd: &str, mut args: Vec<String>) -> Result<Plan, String> {
    match cmd {
        "ping" => Ok(Plan::Ping),
        "list" => Ok(Plan::List),
        "program" => {
            if args.len() < 2 {
                return Err(format!("program needs <name> <file.asm>\n\n{}", usage()));
            }
            let name = args.remove(0);
            let file = PathBuf::from(args.remove(0));
            let mut backend = "lattice".to_string();
            while let Some(a) = args.first() {
                let a = a.clone();
                args.remove(0);
                match a.as_str() {
                    "--backend" => {
                        backend = args
                            .first()
                            .cloned()
                            .ok_or("--backend needs a value".to_string())?;
                        args.remove(0);
                    }
                    "-h" | "--help" => return Err(usage().to_string()),
                    other => return Err(format!("unknown argument '{other}'")),
                }
            }
            Ok(Plan::Program {
                name,
                file,
                backend,
            })
        }
        "store" => {
            if args.is_empty() {
                return Err(format!("store needs <name>\n\n{}", usage()));
            }
            let name = args.remove(0);
            let (payload, stream) = parse_payload_args(&args)?;
            Ok(Plan::Store {
                name,
                payload,
                stream,
            })
        }
        "exec" => {
            if args.is_empty() {
                return Err(format!("exec needs <handle>\n\n{}", usage()));
            }
            let handle = args.remove(0);
            let (payload, outputs, threads, policy, receipt) = parse_exec_args(&args)?;
            Ok(Plan::Exec {
                handle,
                payload,
                outputs,
                threads,
                policy,
                receipt,
            })
        }
        "drop" => {
            if args.is_empty() {
                return Err(format!("drop needs <name>\n\n{}", usage()));
            }
            Ok(Plan::Drop {
                name: args.remove(0),
            })
        }
        "-h" | "--help" | "help" => Err(usage().to_string()),
        _ => Err(format!("unknown command '{cmd}'\n\n{}", usage())),
    }
}

fn real_main() -> Result<(), String> {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    let mut addr = "127.0.0.1:9999".to_string();
    let mut token: Option<String> = None;
    loop {
        match args.first().map(|s| s.as_str()) {
            Some("--addr") => {
                args.remove(0);
                addr = args
                    .first()
                    .cloned()
                    .ok_or("--addr needs a value".to_string())?;
                args.remove(0);
            }
            Some("--token") => {
                args.remove(0);
                token = Some(
                    args.first()
                        .cloned()
                        .ok_or("--token needs a value".to_string())?,
                );
                args.remove(0);
            }
            _ => break,
        }
    }
    if args.is_empty() {
        return Err(usage().to_string());
    }
    let cmd = args.remove(0);
    // Validate all subcommand arguments before connecting so malformed
    // commands fail with usage, not a connection error.
    let plan = parse_plan(&cmd, args)?;
    let mut stream = connect(&addr)?;
    match plan {
        Plan::Ping => {
            let (h, _) = roundtrip(&mut stream, 1, "ping", json!({}), &[], &token)?;
            ok_or_show(&h)?;
            println!("{}", serde_json::to_string_pretty(&h).unwrap());
        }
        Plan::List => {
            let (h, _) = roundtrip(&mut stream, 2, "list", json!({}), &[], &token)?;
            ok_or_show(&h)?;
            println!("{}", serde_json::to_string_pretty(&h).unwrap());
        }
        Plan::Program {
            name,
            file,
            backend,
        } => {
            let listing =
                std::fs::read(&file).map_err(|e| format!("cannot read {}: {e}", file.display()))?;
            let (h, _) = roundtrip(
                &mut stream,
                3,
                "program",
                json!({"name": name, "backend": backend}),
                &listing,
                &token,
            )?;
            ok_or_show(&h)?;
            println!("{}", serde_json::to_string_pretty(&h).unwrap());
        }
        Plan::Store {
            name,
            payload: payload_file,
            stream: stream_name,
        } => {
            let feeds = read_feeds(&payload_file)?;
            let v = feeds
                .get(&stream_name)
                .ok_or_else(|| format!("stream '{stream_name}' not in {}", payload_file.display()))?
                .clone();
            let (spec, bytes) = encode_stream(&v).map_err(|e| e.to_string())?;
            let fields = json!({"name": name, "stream": spec.json()});
            let (h, _) = roundtrip(&mut stream, 4, "store", fields, &bytes, &token)?;
            ok_or_show(&h)?;
            println!("{}", serde_json::to_string_pretty(&h).unwrap());
        }
        Plan::Exec {
            handle,
            payload: payload_file,
            outputs,
            threads,
            policy,
            receipt,
        } => {
            let feeds = read_feeds(&payload_file)?;
            // Inline all inputs as segments.
            let mut inputs = serde_json::Map::new();
            let mut body = Vec::new();
            for (name, v) in &feeds {
                let (spec, bytes) = encode_stream(v).map_err(|e| e.to_string())?;
                let off = body.len();
                body.extend_from_slice(&bytes);
                let mut sj = spec.json();
                su(sj.as_object_mut(), "off", json!(off));
                su(sj.as_object_mut(), "len", json!(bytes.len()));
                inputs.insert(name.clone(), sj);
            }
            let mut fields =
                json!({"handle": handle, "inputs": Json::Object(inputs), "outputs": outputs});
            if let Some(o) = fields.as_object_mut() {
                if let Some(t) = threads {
                    o.insert("threads".into(), json!(t));
                }
                if let Some(p) = &policy {
                    o.insert("thread_policy".into(), json!(p));
                }
            }
            let (h, resp) = roundtrip(&mut stream, 5, "exec", fields, &body, &token)?;
            ok_or_show(&h)?;
            if receipt {
                if let Some(r) = h.get("receipt") {
                    eprintln!("{}", serde_json::to_string_pretty(r).unwrap());
                }
            }
            // Decode outputs to JSON streams where specs are present.
            let mut out_streams = serde_json::Map::new();
            if let Some(specs) = h.get("outputs").and_then(|x| x.as_object()) {
                for (name, s) in specs {
                    let spec = StreamSpec::from_json(s).map_err(|e| e.to_string())?;
                    let off = s.get("off").and_then(|x| x.as_u64()).unwrap_or(0) as usize;
                    let len = s.get("len").and_then(|x| x.as_u64()).unwrap_or(0) as usize;
                    let end = off.checked_add(len).ok_or("exec output offset overflow")?;
                    if end > resp.len() {
                        return Err(format!("exec output {name} truncated"));
                    }
                    let v = decode_stream(&spec, &resp[off..end]).map_err(|e| e.to_string())?;
                    out_streams.insert(
                        name.clone(),
                        payload::value_to_json(&v).map_err(|e| e.to_string())?,
                    );
                }
            }
            println!(
                "{}",
                serde_json::to_string_pretty(&json!({"streams": Json::Object(out_streams)}))
                    .unwrap()
            );
        }
        Plan::Drop { name } => {
            let (h, _) = roundtrip(&mut stream, 6, "drop", json!({"name": name}), &[], &token)?;
            ok_or_show(&h)?;
            println!("{}", serde_json::to_string_pretty(&h).unwrap());
        }
    }
    Ok(())
}

fn su(m: Option<&mut serde_json::Map<String, Json>>, k: &str, v: Json) {
    if let Some(o) = m {
        o.insert(k.to_string(), v);
    }
}

fn read_feeds(path: &std::path::Path) -> Result<Feeds, String> {
    payload::read_payload(path).map_err(|e| e.to_string())
}

fn parse_payload_args(args: &[String]) -> Result<(PathBuf, String), String> {
    let mut file: Option<PathBuf> = None;
    let mut stream_name: Option<String> = None;
    let mut it = args.iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--payload" => file = Some(PathBuf::from(it.next().ok_or("--payload needs a file")?)),
            "--stream" => stream_name = Some(it.next().ok_or("--stream needs a name")?.clone()),
            "-h" | "--help" => return Err(usage().to_string()),
            other => return Err(format!("unknown argument '{other}'")),
        }
    }
    Ok((
        file.ok_or("--payload is required")?,
        stream_name.ok_or("--stream is required")?,
    ))
}

fn parse_exec_args(
    args: &[String],
) -> Result<(PathBuf, Vec<String>, Option<usize>, Option<String>, bool), String> {
    let mut file: Option<PathBuf> = None;
    let mut outputs: Option<Vec<String>> = None;
    let mut threads: Option<usize> = None;
    let mut policy: Option<String> = None;
    let mut receipt = false;
    let mut it = args.iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--payload" => file = Some(PathBuf::from(it.next().ok_or("--payload needs a file")?)),
            "--outputs" => {
                let s = it.next().ok_or("--outputs needs a comma list")?;
                outputs = Some(s.split(',').map(|s| s.trim().to_string()).collect());
            }
            "--threads" => {
                let s = it.next().ok_or("--threads needs a count")?;
                let n = s
                    .parse::<usize>()
                    .map_err(|e| format!("bad --threads: {e}"))?;
                if n == 0 {
                    return Err("--threads must be >= 1".to_string());
                }
                threads = Some(n);
            }
            "--policy" => {
                let s = it.next().ok_or("--policy needs a value")?;
                if !matches!(
                    s.as_str(),
                    "reference" | "serial" | "auto" | "deterministic-auto"
                ) {
                    return Err(format!("bad --policy {s:?}"));
                }
                policy = Some(s.to_string());
            }
            "--receipt" => receipt = true,
            "-h" | "--help" => return Err(usage().to_string()),
            other => return Err(format!("unknown argument '{other}'")),
        }
    }
    Ok((
        file.ok_or("--payload is required")?,
        outputs.ok_or("--outputs is required")?,
        threads,
        policy,
        receipt,
    ))
}
