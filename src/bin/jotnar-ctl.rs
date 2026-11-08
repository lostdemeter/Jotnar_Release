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
use jotnar::protocol::{decode_frame, decode_stream, encode_stream, StreamSpec, PROTO_V};
use serde_json::{json, Value as Json};

fn usage() -> &'static str {
    "jotnar-ctl - framed client for jotnar-node\n\
     \n\
     usage:\n\
       jotnar-ctl [--addr ADDR] ping\n\
       jotnar-ctl [--addr ADDR] list\n\
       jotnar-ctl [--addr ADDR] program <name> <file.asm> [--backend lattice]\n\
       jotnar-ctl [--addr ADDR] store <name> --payload <payload.json> --stream <NAME>\n\
       jotnar-ctl [--addr ADDR] exec <handle> --payload <payload.json> --outputs a,b\n\
       jotnar-ctl [--addr ADDR] drop <name>\n\
     \n\
     --addr defaults to 127.0.0.1:9999. --payload is the JSON shape from\n\
     `jotnar run --payload` (`{\"streams\": {...}}`).\n"
}

fn connect(addr: &str) -> Result<TcpStream, String> {
    let stream = TcpStream::connect(addr).map_err(|e| format!("cannot connect to {addr}: {e}"))?;
    let _ = stream.set_nodelay(true);
    Ok(stream)
}

/// Send one framed request, return `(header, payload)` of the reply.
fn roundtrip(stream: &mut TcpStream, id: u64, typ: &str, fields: Json, payload: &[u8]) -> Result<(Json, Vec<u8>), String> {
    let frame = jotnar::protocol::request(id, typ, fields, payload).map_err(|e| e.to_string())?;
    stream.write_all(&frame).map_err(|e| format!("write: {e}"))?;
    let mut hdr = [0u8; 4];
    stream.read_exact(&mut hdr).map_err(|e| format!("read header: {e}"))?;
    let n = u32::from_be_bytes(hdr) as usize;
    let mut body = vec![0u8; n];
    stream.read_exact(&mut body).map_err(|e| format!("read body: {e}"))?;
    let mut full = Vec::with_capacity(4 + n);
    full.extend_from_slice(&hdr);
    full.extend_from_slice(&body);
    decode_frame(&full).map_err(|e| e.to_string())
}

fn ok_or_show(header: &Json) -> Result<(), String> {
    if header.get("ok").and_then(|x| x.as_bool()) == Some(true) {
        Ok(())
    } else {
        Err(format!("node error: {}", header.get("error").cloned().unwrap_or(json!("?"))))
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

fn real_main() -> Result<(), String> {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    let mut addr = "127.0.0.1:9999".to_string();
    while args.first().map(|s| s.as_str()) == Some("--addr") {
        args.remove(0);
        addr = args.remove(0);
    }
    if args.is_empty() {
        return Err(usage().to_string());
    }
    let cmd = args.remove(0);
    let mut stream = connect(&addr)?;
    match cmd.as_str() {
        "ping" => {
            let (h, _) = roundtrip(&mut stream, 1, "ping", json!({}), &[])?;
            ok_or_show(&h)?;
            println!("{}", serde_json::to_string_pretty(&h).unwrap());
        }
        "list" => {
            let (h, _) = roundtrip(&mut stream, 2, "list", json!({}), &[])?;
            ok_or_show(&h)?;
            println!("{}", serde_json::to_string_pretty(&h).unwrap());
        }
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
                    "--backend" => backend = args.remove(0),
                    "-h" | "--help" => return Err(usage().to_string()),
                    other => return Err(format!("unknown argument '{other}'")),
                }
            }
            let listing = std::fs::read(&file)
                .map_err(|e| format!("cannot read {}: {e}", file.display()))?;
            let (h, _) = roundtrip(&mut stream, 3, "program", json!({"name": name, "backend": backend}), &listing)?;
            ok_or_show(&h)?;
            println!("{}", serde_json::to_string_pretty(&h).unwrap());
        }
        "store" => {
            if args.len() < 1 {
                return Err(format!("store needs <name>\n\n{}", usage()));
            }
            let name = args.remove(0);
            let (payload_file, stream_name) = parse_payload_args(&args)?;
            let feeds = read_feeds(&payload_file)?;
            let v = feeds
                .get(&stream_name)
                .ok_or_else(|| format!("stream '{stream_name}' not in {}", payload_file.display()))?
                .clone();
            let (spec, bytes) = encode_stream(&v).map_err(|e| e.to_string())?;
            let fields = json!({"name": name, "stream": spec.json()});
            let (h, _) = roundtrip(&mut stream, 4, "store", fields, &bytes)?;
            ok_or_show(&h)?;
            println!("{}", serde_json::to_string_pretty(&h).unwrap());
        }
        "exec" => {
            if args.len() < 1 {
                return Err(format!("exec needs <handle>\n\n{}", usage()));
            }
            let handle = args.remove(0);
            let (payload_file, outputs) = parse_exec_args(&args)?;
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
            let fields = json!({"handle": handle, "inputs": Json::Object(inputs), "outputs": outputs});
            let (h, resp) = roundtrip(&mut stream, 5, "exec", fields, &body)?;
            ok_or_show(&h)?;
            // Decode outputs to JSON streams where specs are present.
            let mut out_streams = serde_json::Map::new();
            if let Some(specs) = h.get("outputs").and_then(|x| x.as_object()) {
                for (name, s) in specs {
                    let spec = StreamSpec::from_json(s).map_err(|e| e.to_string())?;
                    let off = s.get("off").and_then(|x| x.as_u64()).unwrap_or(0) as usize;
                    let len = s.get("len").and_then(|x| x.as_u64()).unwrap_or(0) as usize;
                    let v = decode_stream(&spec, &resp[off..off + len]).map_err(|e| e.to_string())?;
                    out_streams.insert(name.clone(), payload::value_to_json(&v).map_err(|e| e.to_string())?);
                }
            }
            println!("{}", serde_json::to_string_pretty(&json!({"streams": Json::Object(out_streams)})).unwrap());
        }
        "drop" => {
            if args.is_empty() {
                return Err(format!("drop needs <name>\n\n{}", usage()));
            }
            let name = args.remove(0);
            let (h, _) = roundtrip(&mut stream, 6, "drop", json!({"name": name}), &[])?;
            ok_or_show(&h)?;
            println!("{}", serde_json::to_string_pretty(&h).unwrap());
        }
        "-h" | "--help" | "help" => return Err(usage().to_string()),
        other => return Err(format!("unknown command '{other}'\n\n{}", usage())),
    }
    Ok(())
}

fn su(m: Option<&mut serde_json::Map<String, Json>>, k: &str, v: Json) {
    if let Some(o) = m { o.insert(k.to_string(), v); }
}

fn read_feeds(path: &PathBuf) -> Result<Feeds, String> {
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
    Ok((file.ok_or("--payload is required")?, stream_name.ok_or("--stream is required")?))
}

fn parse_exec_args(args: &[String]) -> Result<(PathBuf, Vec<String>), String> {
    let mut file: Option<PathBuf> = None;
    let mut outputs: Option<Vec<String>> = None;
    let mut it = args.iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--payload" => file = Some(PathBuf::from(it.next().ok_or("--payload needs a file")?)),
            "--outputs" => {
                let s = it.next().ok_or("--outputs needs a comma list")?;
                outputs = Some(s.split(',').map(|s| s.trim().to_string()).collect());
            }
            "-h" | "--help" => return Err(usage().to_string()),
            other => return Err(format!("unknown argument '{other}'")),
        }
    }
    Ok((file.ok_or("--payload is required")?, outputs.ok_or("--outputs is required")?))
}

#[allow(dead_code)]
fn _v() -> u64 { PROTO_V }
