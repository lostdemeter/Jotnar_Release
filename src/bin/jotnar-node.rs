//! TCP server for the Jotnar node protocol (`jotnar::protocol`).
//!
//! Wraps `Node::handle` over TCP with the existing frame codec:
//! `u32BE || JSON header || payload`. The server is model-blind: it
//! receives a listing, named blobs, and typed streams, never the
//! workload that produced them.
//!
//! One Node per process, one client connection served at a time
//! (fan-out lives in the controller, matching the protocol's
//! one-client-per-server rule). Frames larger than `--max-frame`
//! are rejected with an error frame and the connection is closed,
//! since the length prefix cannot be trusted.
//!
//! ```sh
//! cargo run --release --bin jotnar-node -- --bind 127.0.0.1:9999
//! ```
//!
//! Drive it from another process (or another machine) with `jotnar-ctl`.

use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::process::ExitCode;
use std::time::Instant;

use jotnar::protocol::{error_frame, Node, PROTO_V};
use serde_json::json;

fn usage() -> &'static str {
    "jotnar-node - TCP server for the Jotnar node protocol\n\
     \n\
     usage:\n\
       jotnar-node [--bind ADDR] [--allow lattice,...] [--max-frame N] [--stats]\n\
     \n\
     ADDR defaults to 127.0.0.1:9999. --allow defaults to lattice.\n\
     --max-frame defaults to 67108864 (64 MiB). --stats logs per-request ms.\n"
}

struct Config {
    bind: String,
    allowed: Vec<String>,
    max_frame: usize,
    stats: bool,
}

fn parse_args() -> Result<Config, String> {
    let mut cfg = Config {
        bind: "127.0.0.1:9999".to_string(),
        allowed: vec!["lattice".to_string()],
        max_frame: 64 * 1024 * 1024,
        stats: false,
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
                let s = it.next().ok_or("--max-frame needs a byte count")?;
                cfg.max_frame = s.parse().map_err(|_| "--max-frame must be an integer")?;
            }
            "--stats" => cfg.stats = true,
            "-h" | "--help" | "help" => return Err(usage().to_string()),
            other => return Err(format!("unknown argument '{other}'\n\n{}", usage())),
        }
    }
    Ok(cfg)
}

/// Read one frame body: `u32BE` prefix already peeked at. Returns the
/// full framed bytes (`4 + body`) on success, or `None` on clean EOF.
/// On oversize, writes an error frame and returns `Err(())` so the
/// caller closes the connection (stream cannot be resynchronised).
fn read_frame(stream: &mut TcpStream, max_frame: usize) -> Option<Result<Vec<u8>, ()>> {
    let mut hdr = [0u8; 4];
    if stream.read_exact(&mut hdr).is_err() {
        return None;
    }
    let n = u32::from_be_bytes(hdr) as usize;
    if n > max_frame {
        let _ = stream.write_all(&error_frame(&json!(0), "too-large", "frame exceeds max-frame"));
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

fn serve(stream: &mut TcpStream, node: &mut Node, cfg: &Config) -> std::io::Result<()> {
    loop {
        let frame = match read_frame(stream, cfg.max_frame) {
            Some(Ok(f)) => f,
            Some(Err(())) => return Ok(()),
            None => return Ok(()),
        };
        let t0 = Instant::now();
        let out = node.handle(&frame);
        if cfg.stats {
            eprintln!("req {:.3}ms ({} B -> {} B)", t0.elapsed().as_secs_f64() * 1e3, frame.len(), out.len());
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
    let listener =
        TcpListener::bind(&cfg.bind).map_err(|e| format!("cannot bind {}: {e}", cfg.bind))?;
    eprintln!("jotnar-node v{} listening on {} (allowed {:?})", PROTO_V, cfg.bind, cfg.allowed);
    let mut node = Node::new(cfg.allowed.clone());
    for conn in listener.incoming() {
        match conn {
            Ok(mut stream) => {
                let _ = stream.set_nodelay(true);
                if let Err(e) = serve(&mut stream, &mut node, &cfg) {
                    eprintln!("connection error: {e}");
                }
            }
            Err(e) => eprintln!("accept error: {e}"),
        }
    }
    Ok(())
}
