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
       jotnar-node [--bind ADDR] [--allow lattice,...] [--max-frame N] [--token SECRET] [--stats]\n\
     \n\
     ADDR defaults to 127.0.0.1:9999. --allow defaults to lattice.\n\
     --max-frame defaults to 67108864 (64 MiB). --stats logs per-request ms.\n\
     Non-loopback binds require --token (trusted-LAN requirement).\n"
}

struct Config {
    bind: String,
    allowed: Vec<String>,
    max_frame: usize,
    stats: bool,
    token: Option<String>,
}

fn parse_args() -> Result<Config, String> {
    let mut cfg = Config {
        bind: "127.0.0.1:9999".to_string(),
        allowed: vec!["lattice".to_string()],
        max_frame: 64 * 1024 * 1024,
        stats: false,
        token: None,
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
            "--token" => {
                let t = it.next().ok_or("--token needs a secret")?;
                if t.trim().is_empty() {
                    return Err("--token must be a non-empty secret (empty/whitespace tokens are rejected, not treated as disabled auth)".to_string());
                }
                cfg.token = Some(t);
            }
            "--stats" => cfg.stats = true,
            "-h" | "--help" | "help" => return Err(usage().to_string()),
            other => return Err(format!("unknown argument '{other}'\n\n{}", usage())),
        }
    }
    Ok(cfg)
}

fn is_loopback_bind(bind: &str) -> bool {
    // Bare host without a port, e.g. "localhost" or "127.0.0.1".
    if bind == "localhost" {
        return true;
    }
    if let Ok(ip) = bind.parse::<std::net::IpAddr>() {
        return ip.is_loopback();
    }
    // host:port or [ipv6]:port. Resolve via ToSocketAddrs so
    // "localhost:9999", "[::1]:9999", "127.0.0.1:9999" and
    // "0.0.0.0:9999" all classify by their IP, not the port.
    // 0.0.0.0 / :: are NOT loopback.
    if let Ok(addrs) = std::net::ToSocketAddrs::to_socket_addrs(&bind) {
        let ips: Vec<std::net::IpAddr> = addrs.map(|a| a.ip()).collect();
        if !ips.is_empty() {
            // Require every resolved address to be loopback. `TcpListener`
            // may bind a later address if the first is unavailable, so a
            // hostname resolving to both loopback and a LAN address must
            // not be treated as loopback-safe.
            return ips.iter().all(|ip| ip.is_loopback());
        }
    }
    // Bare hostname without a port (e.g. "localhost" already handled).
    // Anything we cannot classify is treated as non-loopback so the
    // caller must supply a token (fail safe, not fail open).
    false
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
            "refusing non-loopback bind {} without --token: the node protocol has no TLS; run on a trusted LAN with a shared token\n\n{}",
            cfg.bind,
            usage()
        ));
    }
    let listener =
        TcpListener::bind(&cfg.bind).map_err(|e| format!("cannot bind {}: {e}", cfg.bind))?;
    eprintln!(
        "jotnar-node v{} listening on {} (allowed {:?})",
        PROTO_V, cfg.bind, cfg.allowed
    );
    let mut node = Node::new(cfg.allowed.clone());
    if let Some(t) = cfg.token.clone() {
        node = node.with_token(t);
    }
    for conn in listener.incoming() {
        match conn {
            Ok(mut stream) => {
                let _ = stream.set_nodelay(true);
                // One-client, one-request-at-a-time policy: a stalled
                // client must not block the only slot forever.
                let _ = stream.set_read_timeout(Some(std::time::Duration::from_secs(60)));
                let _ = stream.set_write_timeout(Some(std::time::Duration::from_secs(60)));
                if let Err(e) = serve(&mut stream, &mut node, &cfg) {
                    eprintln!("connection error: {e}");
                }
            }
            Err(e) => eprintln!("accept error: {e}"),
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::is_loopback_bind;

    #[test]
    fn loopback_binds_are_classified_as_loopback() {
        assert!(is_loopback_bind("127.0.0.1:9999"));
        assert!(is_loopback_bind("localhost:9999"));
        assert!(is_loopback_bind("[::1]:9999"));
        assert!(is_loopback_bind("::1"));
        assert!(is_loopback_bind("localhost"));
        assert!(is_loopback_bind("127.0.0.1"));
    }

    #[test]
    fn non_loopback_binds_require_a_token() {
        assert!(!is_loopback_bind("0.0.0.0:9999"));
        assert!(!is_loopback_bind("192.168.1.2:9999"));
        assert!(!is_loopback_bind("[::]:9999"));
        assert!(!is_loopback_bind("::"));
        assert!(!is_loopback_bind("0.0.0.0"));
    }
}
