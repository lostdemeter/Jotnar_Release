use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::process::{Command, Stdio};
use std::time::Duration;

use serde_json::{json, Value as Json};

use jotnar::protocol::{decode_frame, request};

fn free_port() -> String {
    let l = TcpListener::bind("127.0.0.1:0").unwrap();
    format!("{}", l.local_addr().unwrap())
}

fn wait_for_port(addr: &str) -> bool {
    for _ in 0..50 {
        if TcpStream::connect(addr).is_ok() {
            return true;
        }
        std::thread::sleep(Duration::from_millis(20));
    }
    false
}

#[test]
fn repl_runs_actual_binary_and_survives_bad_lines() {
    let bin = env!("CARGO_BIN_EXE_jotnar-repl");
    let good = json!({"streams": {
        "A": {"kind": "I64", "shape": [4], "data": ["3","3","3","3"]},
        "X": {"kind": "I64", "shape": [4], "data": ["1","2","3","4"]},
        "Y": {"kind": "I64", "shape": [4], "data": ["10","20","30","40"]}
    }});
    let bad = "not json";
    let good2 = json!({"streams": {
        "A": {"kind": "I64", "shape": [4], "data": ["3","3","3","3"]},
        "X": {"kind": "I64", "shape": [4], "data": ["1","2","3","4"]},
        "Y": {"kind": "I64", "shape": [4], "data": ["10","20","30","40"]}
    }});
    let input = format!("{}\n{}\n{}\n", good, bad, good2);
    let mut child = Command::new(bin)
        .arg("showcase/language/saxpy.asm")
        .arg("--outputs")
        .arg("OUT")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn jotnar-repl");
    let mut stdin = child.stdin.take().unwrap();
    stdin.write_all(input.as_bytes()).unwrap();
    drop(stdin);
    let output = child.wait_with_output().unwrap();
    assert!(
        output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).unwrap();
    let lines: Vec<&str> = stdout.lines().collect();
    assert_eq!(lines.len(), 3, "got: {:?}", stdout);
    let first: Json = serde_json::from_str(lines[0]).unwrap();
    assert_eq!(
        first["streams"]["OUT"]["data"],
        json!(["13", "26", "39", "52"])
    );
    let err: Json = serde_json::from_str(lines[1]).unwrap();
    assert!(err.get("error").is_some(), "bad line should error");
    let third: Json = serde_json::from_str(lines[2]).unwrap();
    assert_eq!(
        third["streams"]["OUT"]["data"],
        json!(["13", "26", "39", "52"])
    );
}

#[test]
fn node_binary_loopback_no_token_and_token_gate() {
    let bin = env!("CARGO_BIN_EXE_jotnar-node");
    let addr = free_port();
    let mut child = Command::new(bin)
        .arg("--bind")
        .arg(&addr)
        .arg("--token")
        .arg("s3cret")
        .spawn()
        .expect("spawn jotnar-node");
    assert!(wait_for_port(&addr), "node did not listen on {addr}");
    // Bad token is rejected with unauthorized.
    let frame = request(1, "ping", json!({"auth": "wrong"}), &[]).unwrap();
    let (h, _) = node_roundtrip(&addr, &frame);
    assert_eq!(h["ok"], false);
    assert_eq!(h["error"]["code"], "unauthorized");
    // Good token works.
    let frame = request(2, "ping", json!({"auth": "s3cret"}), &[]).unwrap();
    let (h, _) = node_roundtrip(&addr, &frame);
    assert_eq!(h["ok"], true);
    let _ = child.kill();
}

#[test]
fn node_binary_refuses_non_loopback_and_empty_token() {
    let bin = env!("CARGO_BIN_EXE_jotnar-node");
    let out = Command::new(bin)
        .arg("--bind")
        .arg("0.0.0.0:19999")
        .output()
        .expect("node non-loopback should refuse");
    assert!(!out.status.success());
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("without --token"), "got: {}", stderr);

    let out = Command::new(bin)
        .arg("--bind")
        .arg("127.0.0.1:19999")
        .arg("--token")
        .arg("")
        .output()
        .expect("node empty token should refuse");
    assert!(!out.status.success());
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("non-empty"), "got: {}", stderr);
}

#[test]
fn node_binary_rejects_oversize_frame() {
    let bin = env!("CARGO_BIN_EXE_jotnar-node");
    let addr = free_port();
    let mut child = Command::new(bin)
        .arg("--bind")
        .arg(&addr)
        .arg("--max-frame")
        .arg("1024")
        .spawn()
        .expect("spawn jotnar-node");
    assert!(wait_for_port(&addr));
    // Server requires no token on loopback; send a frame whose claimed
    // length exceeds the cap. The server replies error/too-large and closes.
    let mut stream = TcpStream::connect(&addr).unwrap();
    stream.write_all(&u32::to_be_bytes(2048)).unwrap();
    let mut hdr = [0u8; 4];
    stream.read_exact(&mut hdr).unwrap();
    let n = u32::from_be_bytes(hdr) as usize;
    let mut body = vec![0u8; n];
    stream.read_exact(&mut body).unwrap();
    let mut full = hdr.to_vec();
    full.extend_from_slice(&body);
    let (h, _) = decode_frame(&full).unwrap();
    assert_eq!(h["ok"], false);
    assert_eq!(h["error"]["code"], "too-large");
    let _ = child.kill();
}

#[test]
fn ctl_missing_value_is_usage_error_not_panic() {
    let bin = env!("CARGO_BIN_EXE_jotnar-ctl");
    let out = Command::new(bin)
        .arg("--addr")
        .output()
        .expect("ctl --addr");
    assert!(!out.status.success());
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("--addr"), "got: {}", stderr);
    assert!(!stderr.contains("panicked"), "panicked: {}", stderr);
}

#[test]
fn ctl_backend_without_value_fails_before_connecting() {
    let bin = env!("CARGO_BIN_EXE_jotnar-ctl");
    // No server is running on this address; the flag validation must
    // fire before the client even attempts to connect.
    let out = Command::new(bin)
        .args([
            "--addr",
            "127.0.0.1:1",
            "program",
            "tile",
            "showcase/language/matmul2.asm",
            "--backend",
        ])
        .output()
        .expect("ctl backend missing value");
    assert!(!out.status.success());
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("--backend needs a value"),
        "got: {}",
        stderr
    );
    assert!(
        !stderr.contains("cannot connect"),
        "connected before validating: {}",
        stderr
    );
}

fn node_roundtrip(addr: &str, frame: &[u8]) -> (Json, Vec<u8>) {
    let mut stream = TcpStream::connect(addr).unwrap();
    // Node serves one client at a time; each call opens a fresh connection.
    // Because our client helper writes one request and the server keeps the
    // connection open, we must share the connection for the two shapes
    // used in this test (each test uses a fresh server, and each call is a
    // separate TCP connection, so state persists across connections).
    stream.write_all(frame).unwrap();
    let mut hdr = [0u8; 4];
    stream.read_exact(&mut hdr).unwrap();
    let n = u32::from_be_bytes(hdr) as usize;
    let mut body = vec![0u8; n];
    stream.read_exact(&mut body).unwrap();
    let mut full = hdr.to_vec();
    full.extend_from_slice(&body);
    decode_frame(&full).unwrap()
}

#[test]
fn node_loopback_no_token_ping_succeeds() {
    let bin = env!("CARGO_BIN_EXE_jotnar-node");
    let addr = free_port();
    let mut child = Command::new(bin)
        .arg("--bind")
        .arg(&addr)
        .spawn()
        .expect("spawn jotnar-node");
    assert!(wait_for_port(&addr), "node did not listen on {addr}");
    let frame = request(1, "ping", json!({}), &[]).unwrap();
    let (h, _) = node_roundtrip(&addr, &frame);
    assert_eq!(h["ok"], true);
    let _ = child.kill();
}
