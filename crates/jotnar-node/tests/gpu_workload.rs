use std::collections::BTreeMap;
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{SystemTime, UNIX_EPOCH};

use jotnar::protocol::decode_frame;
use jotnar_node_tools::workload::{
    build_gpu_bundle, generated_payload, run_bundle, ExecutorEndpoint, GpuBundleSpec, Ledger,
    RunConfig,
};
use serde_json::{json, Value as Json};

fn free_port() -> u16 {
    let l = TcpListener::bind("127.0.0.1:0").unwrap();
    l.local_addr().unwrap().port()
}

fn temp_root() -> PathBuf {
    let id = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let root = std::env::temp_dir().join(format!(
        "jotnar-gpu-workload-test-{}-{id}",
        std::process::id()
    ));
    std::fs::create_dir_all(&root).unwrap();
    root
}

fn sha256_hex(bytes: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    let mut h = Sha256::new();
    h.update(bytes);
    format!("{:x}", h.finalize())
}

struct Job {
    state: String,
    body: Vec<u8>,
    shape: Vec<u64>,
    sha256: String,
}

struct FakeGpu {
    addr: String,
    stop: Arc<AtomicBool>,
    counter: Arc<AtomicUsize>,
    last_stream: Arc<Mutex<Option<Json>>>,
    thread: Option<thread::JoinHandle<()>>,
}

impl FakeGpu {
    fn start(target: &str) -> Self {
        let port = free_port();
        let addr = format!("127.0.0.1:{port}");
        let listener = TcpListener::bind(&addr).unwrap();
        let stop = Arc::new(AtomicBool::new(false));
        let jobs = Arc::new(Mutex::new(BTreeMap::new()));
        let counter = Arc::new(AtomicUsize::new(0));
        let last_stream = Arc::new(Mutex::new(None));
        let cfg = FakeConfig {
            target: target.to_string(),
            stop: stop.clone(),
            jobs: jobs.clone(),
            counter: counter.clone(),
            last_stream: last_stream.clone(),
        };
        let thread = thread::spawn(move || {
            for stream in listener.incoming() {
                if cfg.stop.load(Ordering::SeqCst) {
                    break;
                }
                if let Ok(mut stream) = stream {
                    let _ = handle(&mut stream, &cfg);
                }
            }
        });
        Self {
            addr,
            stop,
            counter,
            last_stream,
            thread: Some(thread),
        }
    }

    fn last_stream(&self) -> Option<Json> {
        self.last_stream.lock().unwrap().clone()
    }

    fn jobs(&self) -> usize {
        self.counter.load(Ordering::SeqCst)
    }
}

impl Drop for FakeGpu {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::SeqCst);
        let _ = TcpStream::connect(&self.addr);
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

struct FakeConfig {
    target: String,
    stop: Arc<AtomicBool>,
    jobs: Arc<Mutex<BTreeMap<String, Job>>>,
    counter: Arc<AtomicUsize>,
    last_stream: Arc<Mutex<Option<Json>>>,
}

fn read_frame(stream: &mut TcpStream) -> Result<(Json, Vec<u8>), String> {
    let mut hdr = [0u8; 4];
    stream.read_exact(&mut hdr).map_err(|e| e.to_string())?;
    let n = u32::from_be_bytes(hdr) as usize;
    let mut body = vec![0u8; n];
    stream.read_exact(&mut body).map_err(|e| e.to_string())?;
    let mut full = Vec::with_capacity(4 + n);
    full.extend_from_slice(&hdr);
    full.extend_from_slice(&body);
    let (header, payload) = decode_frame(&full).map_err(|e| e.to_string())?;
    Ok((header, payload.to_vec()))
}

fn write_frame(stream: &mut TcpStream, header: &Json, body: &[u8]) -> Result<(), String> {
    let frame = jotnar::protocol::encode_frame(header, body).map_err(|e| e.to_string())?;
    stream.write_all(&frame).map_err(|e| e.to_string())
}

fn handle(stream: &mut TcpStream, cfg: &FakeConfig) -> Result<(), String> {
    let (header, body) = read_frame(stream)?;
    let id = header.get("id").cloned().unwrap_or(json!(1));
    let typ = header
        .get("type")
        .and_then(|x| x.as_str())
        .unwrap_or("")
        .to_string();
    match typ.as_str() {
        "submit" => {
            let body_json: Json =
                serde_json::from_slice(&body).map_err(|e| format!("bad submit body: {e}"))?;
            let listing = body_json
                .get("listing")
                .and_then(|x| x.as_str())
                .ok_or("submit body has no listing")?;
            if !listing.contains("ADD") {
                return Err("fake GPU expected the doubling listing".to_string());
            }
            let payload_stream = body_json
                .get("payload")
                .and_then(|p| p.get("streams"))
                .and_then(|s| s.get("X"))
                .cloned()
                .ok_or("submit body has no X stream")?;
            let kind = payload_stream
                .get("kind")
                .and_then(|x| x.as_str())
                .ok_or("stream has no kind")?;
            if kind != "F32" {
                return Err(format!("fake GPU expected F32, got {kind}"));
            }
            let shape: Vec<u64> = payload_stream
                .get("shape")
                .and_then(|x| x.as_array())
                .unwrap()
                .iter()
                .map(|x| x.as_u64().unwrap())
                .collect();
            let input: Vec<f32> = payload_stream
                .get("data")
                .and_then(|x| x.as_array())
                .unwrap()
                .iter()
                .map(|x| x.as_f64().unwrap() as f32)
                .collect();
            let mut out = Vec::with_capacity(input.len() * 4);
            for v in &input {
                out.extend_from_slice(&(v + v).to_le_bytes());
            }
            let job_id = format!(
                "{}-{}",
                cfg.target,
                cfg.counter.fetch_add(1, Ordering::SeqCst)
            );
            let sha = sha256_hex(&out);
            cfg.jobs.lock().unwrap().insert(
                job_id.clone(),
                Job {
                    state: "done".to_string(),
                    body: out,
                    shape: shape.clone(),
                    sha256: sha,
                },
            );
            *cfg.last_stream.lock().unwrap() = Some(payload_stream);
            write_frame(
                stream,
                &json!({
                    "v": 1,
                    "id": id,
                    "type": "submit-ok",
                    "ok": true,
                    "job_id": job_id,
                    "state": "done",
                }),
                &[],
            )
        }
        "job-status" => {
            let job_id = header
                .get("job_id")
                .and_then(|x| x.as_str())
                .ok_or("job-status needs job_id")?;
            let jobs = cfg.jobs.lock().unwrap();
            let job = jobs
                .get(job_id)
                .ok_or_else(|| format!("unknown job {job_id}"))?;
            write_frame(
                stream,
                &json!({
                    "v": 1,
                    "id": id,
                    "type": "job-status-ok",
                    "ok": true,
                    "job": {
                        "job_id": job_id,
                        "state": job.state,
                        "started_ms": 1,
                        "finished_ms": 2,
                    },
                }),
                &[],
            )
        }
        "job-result" => {
            let job_id = header
                .get("job_id")
                .and_then(|x| x.as_str())
                .ok_or("job-result needs job_id")?;
            let jobs = cfg.jobs.lock().unwrap();
            let job = jobs
                .get(job_id)
                .ok_or_else(|| format!("unknown job {job_id}"))?;
            write_frame(
                stream,
                &json!({
                    "v": 1,
                    "id": id,
                    "type": "job-result-ok",
                    "ok": true,
                    "job_id": job_id,
                    "frames": 1,
                    "output_format": "binary",
                    "outputs": [{
                        "frame": 0,
                        "name": "OUT",
                        "kind": "F32",
                        "shape": job.shape,
                        "off": 0,
                        "bytes": job.body.len(),
                        "sha256": job.sha256,
                    }],
                }),
                &job.body,
            )
        }
        other => Err(format!("fake GPU got unsupported request {other:?}")),
    }
}

fn endpoint(addr: &str, target: &str) -> ExecutorEndpoint {
    ExecutorEndpoint {
        addr: addr.to_string(),
        capability: json!({"kind": "gpu", "target": target, "addr": addr}),
    }
}

#[test]
fn gpu_chain_hands_an_intermediate_binary_result_to_the_next_stage() {
    let root = temp_root();
    let listing = root.join("double.asm");
    std::fs::write(&listing, "CONFIG warm_frame 1\nIN X\nOUT = ADD(X, X)\n").unwrap();
    let bundle_dir = root.join("bundle");
    let loaded = build_gpu_bundle(&GpuBundleSpec {
        listing,
        input: None,
        generate: Some((2, 3, 1)),
        items: 2,
        out_dir: bundle_dir,
        name: "gpu-chain-test".to_string(),
        timeout_ms: 30_000,
    })
    .unwrap();
    assert_eq!(
        loaded.bundle.stage_order().unwrap(),
        vec!["gpu-a-00", "gpu-b-00", "gpu-a-01", "gpu-b-01"]
    );

    let cuda = FakeGpu::start("cuda");
    let hip = FakeGpu::start("hip");
    let executors = vec![endpoint(&cuda.addr, "cuda"), endpoint(&hip.addr, "hip")];

    // Single-node reference: every stage on the fake CUDA runner.
    let mut reference_placement = BTreeMap::new();
    for name in ["gpu-a-00", "gpu-b-00", "gpu-a-01", "gpu-b-01"] {
        reference_placement.insert(name.to_string(), cuda.addr.clone());
    }
    let reference = run_bundle(
        &loaded,
        &root.join("reference"),
        &RunConfig {
            program: None,
            executors: executors.clone(),
            placement: reference_placement,
            token: None,
            force: false,
            timeout_ms: Some(30_000),
            compare: Vec::new(),
        },
    )
    .unwrap();
    assert_eq!(reference["ok"], true);

    // Distributed run: stage A on CUDA, stage B on HIP, with an intermediate
    // binary result becoming stage B's JSON payload stream.
    let mut placement = BTreeMap::new();
    let mut compare = Vec::new();
    for item in 0..2 {
        placement.insert(format!("gpu-a-{item:02}"), cuda.addr.clone());
        placement.insert(format!("gpu-b-{item:02}"), hip.addr.clone());
        compare.push((
            format!("gpu-b-{item:02}:OUT"),
            root.join(format!("reference/stages/gpu-b-{item:02}/OUT")),
        ));
    }
    let cfg = RunConfig {
        program: None,
        executors,
        placement,
        token: None,
        force: false,
        timeout_ms: Some(30_000),
        compare,
    };
    let receipt = run_bundle(&loaded, &root.join("distributed"), &cfg).unwrap();

    assert_eq!(receipt["ok"], true);
    assert_eq!(receipt["stages"]["gpu-b-00"]["executor"], hip.addr);
    assert_eq!(receipt["stages"]["gpu-a-00"]["executor"], cuda.addr);
    for comparison in receipt["comparisons"].as_array().unwrap() {
        assert_eq!(comparison["match"], true);
    }
    // The reference run used four CUDA jobs; the distributed run added the two
    // CUDA A stages. The HIP runner saw only the two B stages.
    assert_eq!(cuda.jobs(), 6);
    assert_eq!(hip.jobs(), 2);

    // The HIP runner received 2 * the generated input, proving the controller
    // decoded stage A's binary OUT into the next stage's payload stream. The
    // last B stage is item 1.
    let expected = generated_payload(2, 3, 1, 1);
    let expected: Vec<f64> = expected["streams"]["X"]["data"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap())
        .collect();
    let got = hip.last_stream().unwrap();
    assert_eq!(got["shape"], json!([2, 3, 1]));
    let got: Vec<f64> = got["data"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap())
        .collect();
    assert_eq!(got, expected.iter().map(|v| v * 2.0).collect::<Vec<_>>());

    // A second run against the same ledger reuses every stage, including the
    // GPU outputs whose descriptor metadata is stored in the ledger.
    let reused = run_bundle(&loaded, &root.join("distributed"), &cfg).unwrap();
    for item in 0..2 {
        assert_eq!(reused["stages"][format!("gpu-a-{item:02}")]["reused"], true);
        assert_eq!(reused["stages"][format!("gpu-b-{item:02}")]["reused"], true);
    }
    assert_eq!(cuda.jobs(), 6);
    assert_eq!(hip.jobs(), 2);

    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn failed_submit_is_recorded_in_the_ledger() {
    let root = temp_root();
    let listing = root.join("double.asm");
    std::fs::write(&listing, "IN X\nOUT = ADD(X, X)\n").unwrap();
    let loaded = build_gpu_bundle(&GpuBundleSpec {
        listing,
        input: None,
        generate: Some((2, 2, 1)),
        items: 1,
        out_dir: root.join("bundle"),
        name: "fail-test".to_string(),
        timeout_ms: 2_000,
    })
    .unwrap();
    let addr = format!("127.0.0.1:{}", free_port());
    let cfg = RunConfig {
        program: None,
        executors: vec![endpoint(&addr, "cuda")],
        placement: BTreeMap::new(),
        token: None,
        force: false,
        timeout_ms: Some(2_000),
        compare: Vec::new(),
    };
    let err = run_bundle(&loaded, &root.join("ledger"), &cfg).unwrap_err();
    assert!(err.contains("submit"), "{err}");
    let ledger = Ledger::read(&root.join("ledger")).unwrap();
    assert_eq!(ledger.stages["gpu-a-00"].state, "submit-failed");
    assert_eq!(
        ledger.stages["gpu-a-00"].executor.as_deref(),
        Some(addr.as_str())
    );
    let _ = std::fs::remove_dir_all(&root);
}
