//! Transport benchmark: in-process vs TCP frames vs TCP JSON.
//!
//! Same workload, same `Node::handle` execution underneath. Only the
//! wire changes, so the measurement is purely transport overhead.
//! Determinism is the gate: every mode must agree byte-for-byte with
//! the local `run_text` reference.
//!
//! ```sh
//! cargo run --release --bin jotnar-bench -- --jobs 200
//! cargo run --release --bin jotnar-bench -- --workload vec --n 16384 --jobs 100
//! cargo run --release --bin jotnar-bench -- --jobs 1000 --modes inproc,tcp-frames
//! ```
//!
//! Workloads:
//! - `matmul2` (default): 8 scalar `i64` inputs, 4 outputs (showcase/matmul2.asm).
//! - `vec`: `OUT = IADD(X, Y)` over `n` `i64` elements per stream. Use a
//!   large `n` (e.g. 16384 or 65536) to see bandwidth / JSON inflation.
//!
//! Modes are run sequentially over a single connection (one-client-
//! per-server rule; fan-out lives in the controller). For the JSON
//! transport the server accepts one NDJSON line per request and
//! translates internally to framed `Node` requests, so EXEC semantics
//! are identical and only encoding + bytes-on-wire differ.
//!
//! JSON request shapes (one line per request):
//! ```json
//! {"id":1,"type":"program","name":"tile","backend":"lattice","listing":"...","listing_bytes":123}
//! {"id":2,"type":"exec","handle":"tile","payload":{"streams":{...}},"outputs":["c00"]}
//! ```
//! `listing_bytes` is advisory; the server counts the bytes it moves.

use std::collections::BTreeMap;
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::process::ExitCode;
use std::thread;
use std::time::Instant;

use serde_json::{json, Value as Json};

use jotnar::payload::{self, Feeds};
use jotnar::protocol::{
    decode_frame, decode_stream, encode_stream, exec_outputs, request, Node, NodeHandle, StreamSpec,
};
use jotnar::value::Value;

const MATMUL_LISTING: &str = include_str!("../../showcase/language/matmul2.asm");
const VEC_LISTING: &str = "IN X\nIN Y\nOUT = IADD(X, Y)\n";
const U8_LISTING: &str = "IN RGB\nOUT = SRGB_DECODE(RGB)\n";
const F64_LISTING: &str = "IN X\nOUT = I2F(X)\n";
const SHELF_LISTING: &str = "IN B\nIN S\nOUT = IADD(B, S)\n";
const NAMES: [&str; 8] = ["a00", "a01", "a10", "a11", "b00", "b01", "b10", "b11"];
const MATMUL_OUTS: [&str; 4] = ["c00", "c01", "c10", "c11"];
const VEC_OUTS: [&str; 1] = ["OUT"];
const U8_OUTS: [&str; 1] = ["OUT"];
const F64_OUTS: [&str; 1] = ["OUT"];
const SHELF_OUTS: [&str; 1] = ["OUT"];

fn usage() -> &'static str {
    "jotnar-bench - transport benchmark (in-process vs TCP frames vs TCP JSON)\n\
     \n\
     usage: jotnar-bench [--jobs N] [--workload matmul2|vec|u8|f64|shelf] [--n N] [--side N] [--modes inproc,tcp-frames,tcp-json]\n\
     \n\
     --workload defaults to matmul2. --n sets vector length for vec/f64/shelf\n\
     (default 16384). --side sets u8 image side length (default 128).\n\
     shelf stores a large B once and references it by name per EXEC.\n"
}

#[derive(Default)]
struct Stats {
    lat: Vec<f64>,
    wire_req: usize,
    wire_resp: usize,
}

impl Stats {
    fn median(&self) -> f64 {
        let mut v = self.lat.clone();
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        v.get(v.len() / 2).copied().unwrap_or(0.0)
    }
    fn p99(&self) -> f64 {
        let mut v = self.lat.clone();
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let i = ((v.len() as f64 * 0.99).ceil() as usize).saturating_sub(1);
        v.get(i).copied().unwrap_or(0.0)
    }
    fn max(&self) -> f64 {
        self.lat.iter().copied().fold(0.0, f64::max)
    }
}

fn scalar(v: i64) -> Value {
    Value::I64(ndarray::ArrayD::from_elem(ndarray::IxDyn(&[1]), v))
}

fn matmul_feeds(vals: &[i64; 8]) -> Feeds {
    let mut f = Feeds::new();
    for (n, v) in NAMES.iter().zip(vals) {
        f.insert((*n).to_string(), scalar(*v));
    }
    f
}

fn vec_feeds(seed: i64, n: usize) -> Feeds {
    let xs: Vec<i64> = (0..n as i64).map(|i| seed + i).collect();
    let ys: Vec<i64> = (0..n as i64).map(|i| seed * 2 + i).collect();
    let mut f = Feeds::new();
    f.insert(
        "X".to_string(),
        Value::I64(jotnar::value::array_from_vec(&[n], xs).unwrap()),
    );
    f.insert(
        "Y".to_string(),
        Value::I64(jotnar::value::array_from_vec(&[n], ys).unwrap()),
    );
    f
}

fn u8_feeds(seed: usize, side: usize) -> Feeds {
    let total = side * side;
    let bytes: Vec<u8> = (0..total).map(|i| ((seed + i) % 256) as u8).collect();
    let mut f = Feeds::new();
    f.insert(
        "RGB".to_string(),
        Value::U8(jotnar::value::array_from_vec(&[side, side], bytes).unwrap()),
    );
    f
}

fn f64_feeds(seed: i64, n: usize) -> Feeds {
    let xs: Vec<i64> = (0..n as i64).map(|i| seed + i * 7).collect();
    let mut f = Feeds::new();
    f.insert(
        "X".to_string(),
        Value::I64(jotnar::value::array_from_vec(&[n], xs).unwrap()),
    );
    f
}

fn reference_for(listing: &str, feeds: &Feeds, outs: &[&str]) -> BTreeMap<String, Vec<u8>> {
    let got = jotnar::run_text(listing, &jotnar::asm::AsmOptions::new("."), feeds)
        .expect("reference run");
    let mut out = BTreeMap::new();
    for o in outs {
        let (_, bytes) = encode_stream(got.get(*o).unwrap()).unwrap();
        out.insert((*o).to_string(), bytes);
    }
    out
}

fn exec_frame_from_feeds(id: u64, handle: &str, feeds: &Feeds, outs: &[&str]) -> (Vec<u8>, usize) {
    let mut payload = Vec::new();
    let mut inputs = serde_json::Map::new();
    for (name, v) in feeds {
        let (spec, bytes) = encode_stream(v).unwrap();
        let off = payload.len();
        payload.extend_from_slice(&bytes);
        let mut sj = spec.json();
        sj["off"] = json!(off);
        sj["len"] = json!(bytes.len());
        inputs.insert(name.clone(), sj);
    }
    let frame = request(
        id,
        "exec",
        json!({"handle": handle, "inputs": Json::Object(inputs), "outputs": outs}),
        &payload,
    )
    .unwrap();
    let len = frame.len();
    (frame, len)
}

fn read_frame(stream: &mut TcpStream) -> Vec<u8> {
    let mut hdr = [0u8; 4];
    stream.read_exact(&mut hdr).unwrap();
    let n = u32::from_be_bytes(hdr) as usize;
    let mut body = vec![0u8; n];
    stream.read_exact(&mut body).unwrap();
    let mut full = hdr.to_vec();
    full.extend_from_slice(&body);
    full
}

fn bench_inproc(
    jobs: &[Feeds],
    refs: &[BTreeMap<String, Vec<u8>>],
    handle: &str,
    listing: &str,
    outs: &[&str],
) -> Stats {
    let node = NodeHandle::spawn(0, &["lattice"]);
    let f = request(
        1,
        "program",
        json!({"name": handle, "backend": "lattice"}),
        listing.as_bytes(),
    )
    .unwrap();
    let _ = node.request(f).unwrap();
    let mut st = Stats::default();
    for (i, feeds) in jobs.iter().enumerate() {
        let (frame, wlen) = exec_frame_from_feeds(100 + i as u64, handle, feeds, outs);
        let t0 = Instant::now();
        let resp = node.request(frame).unwrap();
        st.lat.push(t0.elapsed().as_secs_f64() * 1e3);
        st.wire_req += wlen;
        st.wire_resp += resp.len();
        assert_eq!(exec_outputs(&resp).unwrap(), refs[i]);
    }
    st
}

/// Framed TCP server for one connection, using the same loop as jotnar-node.
fn spawn_frames_server(listener: TcpListener) -> thread::JoinHandle<()> {
    thread::spawn(move || {
        if let Ok((mut stream, _)) = listener.accept() {
            let _ = stream.set_nodelay(true);
            let mut node = Node::new(vec!["lattice".into()]);
            loop {
                let frame = {
                    let mut hdr = [0u8; 4];
                    if stream.read_exact(&mut hdr).is_err() {
                        break;
                    }
                    let n = u32::from_be_bytes(hdr) as usize;
                    let mut body = vec![0u8; n];
                    if stream.read_exact(&mut body).is_err() {
                        break;
                    }
                    let mut f = hdr.to_vec();
                    f.extend_from_slice(&body);
                    f
                };
                let out = node.handle(&frame);
                if stream.write_all(&out).is_err() {
                    break;
                }
            }
        }
    })
}

fn bench_tcp_frames(
    jobs: &[Feeds],
    refs: &[BTreeMap<String, Vec<u8>>],
    handle: &str,
    listing: &str,
    outs: &[&str],
) -> Stats {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap().to_string();
    let h = spawn_frames_server(listener);
    let mut stream = TcpStream::connect(&addr).unwrap();
    let _ = stream.set_nodelay(true);
    let prog = request(
        1,
        "program",
        json!({"name": handle, "backend": "lattice"}),
        listing.as_bytes(),
    )
    .unwrap();
    stream.write_all(&prog).unwrap();
    let _ = read_frame(&mut stream);
    let mut st = Stats::default();
    for (i, feeds) in jobs.iter().enumerate() {
        let (frame, wlen) = exec_frame_from_feeds(100 + i as u64, handle, feeds, outs);
        let t0 = Instant::now();
        stream.write_all(&frame).unwrap();
        let resp = read_frame(&mut stream);
        st.lat.push(t0.elapsed().as_secs_f64() * 1e3);
        st.wire_req += wlen;
        st.wire_resp += resp.len();
        assert_eq!(exec_outputs(&resp).unwrap(), refs[i]);
    }
    drop(stream);
    let _ = h.join();
    st
}

/// NDJSON/JSON server: translates to framed Node requests internally.
fn spawn_json_server(listener: TcpListener) -> thread::JoinHandle<()> {
    thread::spawn(move || {
        if let Ok((stream, _)) = listener.accept() {
            let _ = stream.set_nodelay(true);
            let mut node = Node::new(vec!["lattice".into()]);
            let mut reader = BufReader::with_capacity(1 << 20, stream.try_clone().unwrap());
            let mut writer = stream;
            loop {
                let mut line = String::new();
                let n = match reader.read_line(&mut line) {
                    Ok(n) => n,
                    Err(_) => break,
                };
                if n == 0 {
                    break;
                }
                if line.trim().is_empty() {
                    continue;
                }
                let resp = json_handle(&mut node, &line);
                let mut out = resp.into_bytes();
                out.push(b'\n');
                if writer.write_all(&out).is_err() {
                    break;
                }
                let _ = writer.flush();
            }
        }
    })
}

fn json_handle(node: &mut Node, line: &str) -> String {
    let v: Json = match serde_json::from_str(line) {
        Ok(x) => x,
        Err(e) => {
            return serde_json::to_string(&json!({"ok": false, "error": e.to_string()})).unwrap()
        }
    };
    let id = v.get("id").cloned().unwrap_or(json!(0));
    match v.get("type").and_then(|x| x.as_str()).unwrap_or("") {
        "program" => {
            let name = v.get("name").and_then(|x| x.as_str()).unwrap_or("");
            let listing = v.get("listing").and_then(|x| x.as_str()).unwrap_or("");
            let f = request(
                id.as_u64().unwrap_or(0),
                "program",
                json!({"name": name, "backend": "lattice"}),
                listing.as_bytes(),
            )
            .unwrap();
            let (h, _) = decode_frame(&node.handle(&f)).unwrap();
            serde_json::to_string(&h).unwrap()
        }
        "exec" => {
            let payload = v.get("payload").cloned().unwrap_or(json!({"streams": {}}));
            let feeds = match payload::feeds_from_json(&payload) {
                Ok(f) => f,
                Err(e) => {
                    return serde_json::to_string(&json!({"ok": false, "error": e.to_string()}))
                        .unwrap()
                }
            };
            let outs: Vec<String> = v
                .get("outputs")
                .and_then(|x| x.as_array())
                .map(|a| {
                    a.iter()
                        .filter_map(|o| o.as_str().map(|s| s.to_string()))
                        .collect()
                })
                .unwrap_or_default();
            let mut inputs = serde_json::Map::new();
            let mut body = Vec::new();
            for (name, val) in &feeds {
                let (spec, bytes) = encode_stream(val).unwrap();
                let off = body.len();
                body.extend_from_slice(&bytes);
                let mut sj = spec.json();
                sj["off"] = json!(off);
                sj["len"] = json!(bytes.len());
                inputs.insert(name.clone(), sj);
            }
            let handle_name = v.get("handle").and_then(|x| x.as_str()).unwrap_or("tile");
            let f = request(
                id.as_u64().unwrap_or(0),
                "exec",
                json!({"handle": handle_name, "inputs": Json::Object(inputs), "outputs": outs}),
                &body,
            )
            .unwrap();
            let resp = node.handle(&f);
            let (h, p) = decode_frame(&resp).unwrap();
            if h.get("ok").and_then(|x| x.as_bool()) != Some(true) {
                return serde_json::to_string(&h).unwrap();
            }
            // Decode outputs back to JSON streams for the wire.
            let mut streams = serde_json::Map::new();
            for (name, s) in h
                .get("outputs")
                .and_then(|x| x.as_object())
                .cloned()
                .unwrap_or_default()
            {
                let spec = StreamSpec::from_json(&s).unwrap();
                let off = s.get("off").and_then(|x| x.as_u64()).unwrap_or(0) as usize;
                let len = s.get("len").and_then(|x| x.as_u64()).unwrap_or(0) as usize;
                let val = decode_stream(&spec, &p[off..off + len]).unwrap();
                streams.insert(name.clone(), payload::value_to_json(&val).unwrap());
            }
            serde_json::to_string(&json!({"id": id, "ok": true, "streams": Json::Object(streams)}))
                .unwrap()
        }
        "ping" => serde_json::to_string(&json!({"id": id, "ok": true, "type": "pong"})).unwrap(),
        other => serde_json::to_string(
            &json!({"id": id, "ok": false, "error": format!("no such message {other:?}")}),
        )
        .unwrap(),
    }
}

fn bench_tcp_json(
    jobs: &[Feeds],
    refs: &[BTreeMap<String, Vec<u8>>],
    handle: &str,
    listing: &str,
    outs: &[&str],
) -> Stats {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap().to_string();
    let h = spawn_json_server(listener);
    let mut stream = TcpStream::connect(&addr).unwrap();
    let _ = stream.set_nodelay(true);
    let prog = json!({"id": 1, "type": "program", "name": handle, "backend": "lattice", "listing": listing, "listing_bytes": listing.len()});
    let mut reader = BufReader::with_capacity(1 << 20, stream.try_clone().unwrap());
    let prog_line = format!("{}\n", serde_json::to_string(&prog).unwrap());
    stream.write_all(prog_line.as_bytes()).unwrap();
    let mut line = String::new();
    reader.read_line(&mut line).unwrap();
    let mut st = Stats::default();
    for (i, feeds) in jobs.iter().enumerate() {
        let payload = payload::feeds_to_json(feeds, None).unwrap();
        let req = json!({"id": 100 + i as u64, "type": "exec", "handle": handle, "payload": payload, "outputs": outs});
        let req_line = format!("{}\n", serde_json::to_string(&req).unwrap());
        let t0 = Instant::now();
        stream.write_all(req_line.as_bytes()).unwrap();
        let mut resp_line = String::new();
        reader.read_line(&mut resp_line).unwrap();
        st.lat.push(t0.elapsed().as_secs_f64() * 1e3);
        st.wire_req += req_line.len();
        st.wire_resp += resp_line.len();
        let r: Json = serde_json::from_str(&resp_line).unwrap();
        assert_eq!(
            r.get("ok").and_then(|x| x.as_bool()),
            Some(true),
            "json exec failed: {r}"
        );
        // Decode the returned streams and compare exact bytes, like
        // the in-process and framed arms (nothing is claimed without
        // this comparison).
        let roundtrip = payload::feeds_from_json(
            &json!({"streams": r.get("streams").cloned().unwrap_or(json!({}))}),
        )
        .expect("json response streams decode");
        let mut got = BTreeMap::new();
        for o in outs {
            let (_, bytes) = encode_stream(roundtrip.get(*o).unwrap()).unwrap();
            got.insert((*o).to_string(), bytes);
        }
        assert_eq!(got, refs[i]);
    }
    drop(reader);
    drop(stream);
    let _ = h.join();
    st
}

/// Shelf workload: large B stored once, tiny scalar S per job.
/// Steady-state EXEC references B by name (zero input bytes for B).
fn shelf_b(n: usize) -> Value {
    let data: Vec<i64> = (0..n as i64).map(|i| i * 3).collect();
    Value::I64(jotnar::value::array_from_vec(&[n], data).unwrap())
}

fn shelf_scalar(v: i64) -> Value {
    Value::I64(ndarray::ArrayD::from_elem(
        ndarray::IxDyn(&[] as &[usize]),
        v,
    ))
}

fn bench_shelf(jobs_n: usize, n: usize) {
    // Reference: run once with B+S inline, then per-scalar varying S.
    let b = shelf_b(n);
    let mut refs = Vec::with_capacity(jobs_n);
    for i in 0..jobs_n {
        let mut f = Feeds::new();
        f.insert("B".to_string(), b.clone());
        f.insert("S".to_string(), shelf_scalar(i as i64));
        refs.push(reference_for(SHELF_LISTING, &f, &SHELF_OUTS));
    }
    let (spec_b, bytes_b) = encode_stream(&b).unwrap();

    // --- in-process, shelf vs inline ---
    let inproc = |shelf: bool| -> Stats {
        let node = NodeHandle::spawn(0, &["lattice"]);
        let f = request(
            1,
            "program",
            json!({"name": "shelf", "backend": "lattice"}),
            SHELF_LISTING.as_bytes(),
        )
        .unwrap();
        let _ = node.request(f).unwrap();
        if shelf {
            let s = request(
                2,
                "store",
                json!({"name": "B", "stream": spec_b.json()}),
                &bytes_b,
            )
            .unwrap();
            let _ = node.request(s).unwrap();
        }
        let mut st = Stats::default();
        for (i, expected) in refs.iter().cloned().enumerate().take(jobs_n) {
            let s = shelf_scalar(i as i64);
            let (frame, wlen) = if shelf {
                let (ss, bs) = encode_stream(&s).unwrap();
                let fields = json!({"handle": "shelf", "inputs": {"B": {"store": "B"}, "S": {"kind": ss.kind, "dtype": ss.dtype, "shape": ss.shape, "off": 0, "len": bs.len()}}, "outputs": SHELF_OUTS});
                let f = request(100 + i as u64, "exec", fields, &bs).unwrap();
                let l = f.len();
                (f, l)
            } else {
                let mut f_ = Feeds::new();
                f_.insert("B".to_string(), b.clone());
                f_.insert("S".to_string(), s);
                // Inline both, scalars shape [] encode as 8 bytes.
                let mut payload = Vec::new();
                let mut inputs = serde_json::Map::new();
                for (name, v) in &f_ {
                    let (spec, bytes) = encode_stream(v).unwrap();
                    let off = payload.len();
                    payload.extend_from_slice(&bytes);
                    let mut sj = spec.json();
                    sj["off"] = json!(off);
                    sj["len"] = json!(bytes.len());
                    inputs.insert(name.clone(), sj);
                }
                let f = request(100 + i as u64, "exec", json!({"handle": "shelf", "inputs": Json::Object(inputs), "outputs": SHELF_OUTS}), &payload).unwrap();
                let l = f.len();
                (f, l)
            };
            let t0 = Instant::now();
            let resp = node.request(frame).unwrap();
            st.lat.push(t0.elapsed().as_secs_f64() * 1e3);
            st.wire_req += wlen;
            st.wire_resp += resp.len();
            assert_eq!(exec_outputs(&resp).unwrap(), expected);
        }
        st
    };

    // --- TCP frames: one process, two modes (fresh listener each) ---
    let tcp = |shelf: bool| -> Stats {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap().to_string();
        let h = spawn_frames_server(listener);
        let mut stream = TcpStream::connect(&addr).unwrap();
        let _ = stream.set_nodelay(true);
        let prog = request(
            1,
            "program",
            json!({"name": "shelf", "backend": "lattice"}),
            SHELF_LISTING.as_bytes(),
        )
        .unwrap();
        stream.write_all(&prog).unwrap();
        let _ = read_frame(&mut stream);
        if shelf {
            let s = request(
                2,
                "store",
                json!({"name": "B", "stream": spec_b.json()}),
                &bytes_b,
            )
            .unwrap();
            stream.write_all(&s).unwrap();
            let _ = read_frame(&mut stream);
        }
        let mut st = Stats::default();
        for (i, expected) in refs.iter().cloned().enumerate().take(jobs_n) {
            let (frame, wlen) = if shelf {
                let s = shelf_scalar(i as i64);
                let (ss, bs) = encode_stream(&s).unwrap();
                let fields = json!({"handle": "shelf", "inputs": {"B": {"store": "B"}, "S": {"kind": ss.kind, "dtype": ss.dtype, "shape": ss.shape, "off": 0, "len": bs.len()}}, "outputs": SHELF_OUTS});
                let f = request(100 + i as u64, "exec", fields, &bs).unwrap();
                let l = f.len();
                (f, l)
            } else {
                let mut f_ = Feeds::new();
                f_.insert("B".to_string(), b.clone());
                f_.insert("S".to_string(), shelf_scalar(i as i64));
                exec_frame_from_feeds(100 + i as u64, "shelf", &f_, &SHELF_OUTS)
            };
            let t0 = Instant::now();
            stream.write_all(&frame).unwrap();
            let resp = read_frame(&mut stream);
            st.lat.push(t0.elapsed().as_secs_f64() * 1e3);
            st.wire_req += wlen;
            st.wire_resp += resp.len();
            assert_eq!(exec_outputs(&resp).unwrap(), expected);
        }
        drop(stream);
        let _ = h.join();
        st
    };

    let a = inproc(false);
    let b_ = inproc(true);
    let c = tcp(false);
    let d = tcp(true);
    println!(
        "jotnar-bench: {} shelf jobs, blob B n={} ({} bytes stored once)",
        jobs_n,
        n,
        bytes_b.len()
    );
    println!(
        "labels: clock=host_wall, scope=end_to_end, kind=measured, runs={jobs_n}, machine=local"
    );
    println!("mode               median  p99     max     jobs/s   bytes/job");
    for (name, st) in [
        ("inproc-inline", &a),
        ("inproc-shelf", &b_),
        ("tcp-inline", &c),
        ("tcp-shelf", &d),
    ] {
        let total: f64 = st.lat.iter().sum();
        let jps = jobs_n as f64 / (total / 1e3).max(1e-9);
        let bytes_job = (st.wire_req + st.wire_resp) / jobs_n.max(1);
        println!(
            "{:<18}{:7.3} {:7.3} {:7.3} {:8.1}   {}",
            name,
            st.median(),
            st.p99(),
            st.max(),
            jps,
            bytes_job
        );
    }
    println!(
        "determinism: shelf == inline == reference ({} jobs)",
        jobs_n
    );
}

fn main() -> ExitCode {
    match run_bench() {
        ExitCode::SUCCESS => ExitCode::SUCCESS,
        other => other,
    }
}

fn run_bench() -> ExitCode {
    let mut jobs_n = 200usize;
    let mut workload = "matmul2".to_string();
    let mut n = 16384usize;
    let mut side = 128usize;
    let mut modes = vec![
        "inproc".to_string(),
        "tcp-frames".to_string(),
        "tcp-json".to_string(),
    ];
    let mut it = std::env::args().skip(1);
    while let Some(a) = it.next() {
        match a.as_str() {
            "--jobs" => match it.next() {
                Some(v) => {
                    jobs_n = match v.parse::<usize>() {
                        Ok(n) if n > 0 => n,
                        _ => {
                            eprintln!("--jobs must be a positive integer, got {v}");
                            return ExitCode::FAILURE;
                        }
                    };
                }
                None => {
                    eprintln!("--jobs needs a value");
                    return ExitCode::FAILURE;
                }
            },
            "--workload" => match it.next() {
                Some(v) => workload = v,
                None => {
                    eprintln!("--workload needs a value");
                    return ExitCode::FAILURE;
                }
            },
            "--n" => match it.next() {
                Some(v) => {
                    n = match v.parse::<usize>() {
                        Ok(n) if n > 0 => n,
                        _ => {
                            eprintln!("--n must be a positive integer, got {v}");
                            return ExitCode::FAILURE;
                        }
                    };
                }
                None => {
                    eprintln!("--n needs a value");
                    return ExitCode::FAILURE;
                }
            },
            "--side" => match it.next() {
                Some(v) => {
                    side = match v.parse::<usize>() {
                        Ok(s) if s > 0 => s,
                        _ => {
                            eprintln!("--side must be a positive integer, got {v}");
                            return ExitCode::FAILURE;
                        }
                    };
                }
                None => {
                    eprintln!("--side needs a value");
                    return ExitCode::FAILURE;
                }
            },
            "--modes" => {
                let s = match it.next() {
                    Some(v) => v,
                    None => {
                        eprintln!("--modes needs a value");
                        return ExitCode::FAILURE;
                    }
                };
                modes = s.split(',').map(|s| s.trim().to_string()).collect();
            }
            "-h" | "--help" | "help" => {
                println!("{}", usage());
                return ExitCode::SUCCESS;
            }
            other => {
                eprintln!("unknown argument {other}\n\n{}", usage());
                return ExitCode::FAILURE;
            }
        }
    }
    if modes.is_empty() || modes.iter().all(|m| m.is_empty()) {
        eprintln!("--modes must name at least one mode");
        return ExitCode::FAILURE;
    }
    for m in &modes {
        if !["inproc", "tcp-frames", "tcp-json"].contains(&m.as_str()) {
            eprintln!(
                "unknown mode {m} (try inproc,tcp-frames,tcp-json)\n\n{}",
                usage()
            );
            return ExitCode::FAILURE;
        }
    }

    if workload == "shelf" {
        println!("note: --workload shelf uses its own inproc/tcp arms; --modes is validated but not used");
        bench_shelf(jobs_n, n);
        return ExitCode::SUCCESS;
    }
    let (listing, handle, outs, jobs): (&str, &str, Vec<&str>, Vec<Feeds>) = match workload.as_str()
    {
        "matmul2" => {
            let jobs = (0..jobs_n)
                .map(|i| {
                    let b = i as i64;
                    matmul_feeds(&[b, b + 1, b + 2, b + 3, b + 4, b + 5, b + 6, b + 7])
                })
                .collect();
            (MATMUL_LISTING, "tile", MATMUL_OUTS.to_vec(), jobs)
        }
        "vec" => {
            let jobs = (0..jobs_n).map(|i| vec_feeds(i as i64, n)).collect();
            (VEC_LISTING, "vec", VEC_OUTS.to_vec(), jobs)
        }
        "u8" => {
            let jobs = (0..jobs_n).map(|i| u8_feeds(i, side)).collect();
            (U8_LISTING, "u8", U8_OUTS.to_vec(), jobs)
        }
        "f64" => {
            let jobs = (0..jobs_n).map(|i| f64_feeds(i as i64, n)).collect();
            (F64_LISTING, "f64", F64_OUTS.to_vec(), jobs)
        }
        other => {
            eprintln!("unknown workload {other} (try matmul2|vec|u8|f64|shelf)");
            return ExitCode::FAILURE;
        }
    };
    let refs: Vec<BTreeMap<String, Vec<u8>>> = jobs
        .iter()
        .map(|f| reference_for(listing, f, &outs))
        .collect();

    let desc = match workload.as_str() {
        "vec" => format!("vec (n={n}, {} bytes listing)", listing.len()),
        "u8" => format!(
            "u8 ({}x{} = {} bytes, {} bytes listing)",
            side,
            side,
            side * side,
            listing.len()
        ),
        "f64" => format!("f64 (n={n}, {} bytes listing)", listing.len()),
        _ => format!("matmul2 ({} bytes listing)", listing.len()),
    };
    println!("jotnar-bench: {} jobs, {}", jobs_n, desc);
    println!(
        "labels: clock=host_wall, scope=end_to_end, kind=measured, runs={jobs_n}, machine=local"
    );
    println!("mode        median  p99     max     jobs/s   req->resp bytes/job");
    let mut ran = 0usize;
    for mode in &modes {
        let st = match mode.as_str() {
            "inproc" => bench_inproc(&jobs, &refs, handle, listing, &outs),
            "tcp-frames" => bench_tcp_frames(&jobs, &refs, handle, listing, &outs),
            "tcp-json" => bench_tcp_json(&jobs, &refs, handle, listing, &outs),
            _ => unreachable!("modes validated above"),
        };
        ran += 1;
        let total: f64 = st.lat.iter().sum();
        let jps = jobs_n as f64 / (total / 1e3).max(1e-9);
        let bytes_job = (st.wire_req + st.wire_resp) / jobs_n.max(1);
        println!(
            "{:<12}{:7.3} {:7.3} {:7.3} {:8.1}   {}",
            mode,
            st.median(),
            st.p99(),
            st.max(),
            jps,
            bytes_job
        );
    }
    if ran == 0 {
        eprintln!("no modes ran");
        return ExitCode::FAILURE;
    }
    println!("determinism: all modes matched reference ({} jobs)", jobs_n);
    ExitCode::SUCCESS
}
