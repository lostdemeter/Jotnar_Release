//! Two-node DOOM coordinator.
//!
//! Node A runs the RV32IM machine in this process. Nodes B and C are the CUDA
//! and HIP GPU runners: each receives one half of the framebuffer and runs the
//! same Jotnar upscale program (`showcase/doom/upscale-half.asm`). The
//! coordinator stitches the two results into one frame and records the raw
//! frames plus latency metadata for the GIF assembler.
//!
//! This is the first cooperative, per-frame pipeline: the RV32 core and the
//! Jotnar program run on different nodes, and the coordinator measures the
//! cost of moving a frame between them.

use std::collections::BTreeMap;
use std::fs::{self, File};
use std::io::{BufWriter, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use serde_json::{json, Value as Json};

use jotnar::payload::feeds_to_json;
use jotnar::rv32::Rv32;
use jotnar::value::{array_from_vec, Value};
use jotnar_node_tools::doom_pipeline::{
    choose_split, f32_rgb_to_u8, latency_summary, rgb_window, sha256_hex, Split, FB_H, FB_W, OUT_H,
    OUT_W,
};
use jotnar_node_tools::executor::{validate_binary_result, Executor, RemoteExecutor};
use jotnar_node_tools::hud::{render as render_hud, HudInfo, HUD_BOTTOM, HUD_TOP};
use jotnar_node_tools::job::JobClient;

const KEY_ENTER: u8 = 13;
const KEY_UP: u8 = 0xAD;
const KEY_LEFT: u8 = 0xAC;
const KEY_RIGHT: u8 = 0xAE;
const KEY_FIRE: u8 = 0xA3;

fn usage() -> &'static str {
    "jotnar-doom-coord — cooperative two-node DOOM frame pipeline\n\
     \n\
     usage:\n\
       jotnar-doom-coord --image FILE --wad FILE \\\n\
         --cuda ADDR --hip ADDR [--token T] \\\n\
         [--listing FILE] [--start-frame N] [--frames N] [--fps N] \\\n\
         [--out DIR] [--max-instr N] [--no-warmup] [--no-auto-split] \\\n\
         [--probe-iters N] [--min-cols N] [--max-cols N] [--job-timeout-ms N] \\\n\
         [--viewer-listen ADDR] [--recover-every N]\n\
     \n\
     Node A runs the RV32IM core here. Nodes B and C run the Jotnar upscale\n\
     program on the CUDA and HIP runners. By default the coordinator probes\n\
     both cards and splits the frame by measured throughput. The output is raw\n\
     RGB frames plus metadata; scripts/doom_two_gpu_gate.py assembles the GIF."
}

struct Config {
    image: PathBuf,
    wad: PathBuf,
    cuda: String,
    hip: String,
    token: Option<String>,
    listing: PathBuf,
    start_frame: u32,
    frames: u32,
    fps: u32,
    out: PathBuf,
    max_instr: u64,
    warmup: bool,
    auto_split: bool,
    probe_iters: u32,
    min_cols: usize,
    max_cols: usize,
    job_timeout_ms: u64,
    viewer: Option<String>,
    recover_every: u32,
}

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

fn parse_args() -> Result<Config, String> {
    let mut image = None;
    let mut wad = None;
    let mut cuda = None;
    let mut hip = None;
    let mut token = None;
    let mut listing = PathBuf::from("showcase/doom/upscale-half.asm");
    let mut start_frame = 200u32;
    let mut frames = 72u32;
    let mut fps = 12u32;
    let mut out = PathBuf::from("/tmp/opencode/doom_two_nodes");
    let mut max_instr = 2_000_000_000u64;
    let mut warmup = true;
    let mut auto_split = true;
    let mut probe_iters = 3u32;
    let mut min_cols = 32usize;
    let mut max_cols = 288usize;
    let mut job_timeout_ms = 5000u64;
    let mut viewer = None;
    let mut recover_every = 20u32;
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--image" => {
                i += 1;
                image = Some(PathBuf::from(args.get(i).ok_or("--image needs a path")?));
            }
            "--wad" => {
                i += 1;
                wad = Some(PathBuf::from(args.get(i).ok_or("--wad needs a path")?));
            }
            "--cuda" => {
                i += 1;
                cuda = Some(args.get(i).ok_or("--cuda needs an addr")?.clone());
            }
            "--hip" => {
                i += 1;
                hip = Some(args.get(i).ok_or("--hip needs an addr")?.clone());
            }
            "--token" => {
                i += 1;
                token = Some(args.get(i).ok_or("--token needs a value")?.clone());
            }
            "--listing" => {
                i += 1;
                listing = PathBuf::from(args.get(i).ok_or("--listing needs a path")?);
            }
            "--start-frame" => {
                i += 1;
                start_frame = args
                    .get(i)
                    .ok_or("--start-frame needs a number")?
                    .parse()
                    .map_err(|_| "--start-frame must be an integer")?;
            }
            "--frames" => {
                i += 1;
                frames = args
                    .get(i)
                    .ok_or("--frames needs a number")?
                    .parse()
                    .map_err(|_| "--frames must be an integer")?;
            }
            "--fps" => {
                i += 1;
                fps = args
                    .get(i)
                    .ok_or("--fps needs a number")?
                    .parse()
                    .map_err(|_| "--fps must be an integer")?;
            }
            "--out" => {
                i += 1;
                out = PathBuf::from(args.get(i).ok_or("--out needs a dir")?);
            }
            "--max-instr" => {
                i += 1;
                max_instr = args
                    .get(i)
                    .ok_or("--max-instr needs a number")?
                    .parse()
                    .map_err(|_| "--max-instr must be an integer")?;
            }
            "--no-warmup" => warmup = false,
            "--no-auto-split" => auto_split = false,
            "--probe-iters" => {
                i += 1;
                probe_iters = args
                    .get(i)
                    .ok_or("--probe-iters needs a number")?
                    .parse()
                    .map_err(|_| "--probe-iters must be an integer")?;
            }
            "--min-cols" => {
                i += 1;
                min_cols = args
                    .get(i)
                    .ok_or("--min-cols needs a number")?
                    .parse()
                    .map_err(|_| "--min-cols must be an integer")?;
            }
            "--max-cols" => {
                i += 1;
                max_cols = args
                    .get(i)
                    .ok_or("--max-cols needs a number")?
                    .parse()
                    .map_err(|_| "--max-cols must be an integer")?;
            }
            "--job-timeout-ms" => {
                i += 1;
                job_timeout_ms = args
                    .get(i)
                    .ok_or("--job-timeout-ms needs a number")?
                    .parse()
                    .map_err(|_| "--job-timeout-ms must be an integer")?;
            }
            "--viewer-listen" => {
                i += 1;
                viewer = Some(args.get(i).ok_or("--viewer-listen needs an addr")?.clone());
            }
            "--recover-every" => {
                i += 1;
                recover_every = args
                    .get(i)
                    .ok_or("--recover-every needs a number")?
                    .parse()
                    .map_err(|_| "--recover-every must be an integer")?;
            }
            "-h" | "--help" => {
                println!("{}", usage());
                std::process::exit(0);
            }
            other => return Err(format!("unknown argument {other:?}\n\n{}", usage())),
        }
        i += 1;
    }
    Ok(Config {
        image: image.ok_or("need --image FILE")?,
        wad: wad.ok_or("need --wad FILE")?,
        cuda: cuda.ok_or("need --cuda ADDR")?,
        hip: hip.ok_or("need --hip ADDR")?,
        token,
        listing,
        start_frame: start_frame.max(1),
        frames: frames.max(1),
        fps: fps.max(1),
        out,
        max_instr: max_instr.max(1),
        warmup,
        auto_split: auto_split && warmup,
        probe_iters: probe_iters.max(1),
        min_cols: min_cols.max(1),
        max_cols: max_cols.max(1),
        job_timeout_ms: job_timeout_ms.max(1),
        viewer,
        recover_every,
    })
}

fn half_payload(data: Vec<f32>, width: usize) -> Result<Json, String> {
    let arr = array_from_vec(&[FB_H, width, 3], data).map_err(|e| e.to_string())?;
    let mut feeds = BTreeMap::new();
    feeds.insert("X".to_string(), Value::F32(arr));
    feeds_to_json(&feeds, Some(&["X".to_string()])).map_err(|e| e.to_string())
}

fn body_for(listing: &str, payload: Json) -> Result<Vec<u8>, String> {
    serde_json::to_vec(&json!({"listing": listing, "payload": payload})).map_err(|e| e.to_string())
}

struct HalfRun {
    job_id: String,
    bytes: Vec<u8>,
    elapsed_ms: u64,
}

fn run_job(
    addr: &str,
    target: &str,
    token: Option<String>,
    body: &[u8],
    timeout: Duration,
) -> Result<HalfRun, String> {
    let executor = RemoteExecutor::new(
        json!({"kind": "gpu", "target": target, "addr": addr}),
        addr,
        token,
    );
    let started = Instant::now();
    let spec = json!({
        "type": "submit",
        "outputs": ["OUT"],
        "frames": 1,
        "output_format": "binary",
    });
    let submitted = executor
        .submit(&spec, body)
        .map_err(|(code, msg)| format!("{addr} submit {code}: {msg}"))?;
    let job_id = submitted
        .get("job_id")
        .and_then(|x| x.as_str())
        .ok_or_else(|| format!("{addr} submit returned no job_id"))?
        .to_string();
    let state = loop {
        let status = executor
            .status(&job_id)
            .map_err(|(code, msg)| format!("{addr} job-status {code}: {msg}"))?;
        let state = status
            .get("job")
            .and_then(|j| j.get("state"))
            .and_then(|x| x.as_str())
            .unwrap_or("unknown")
            .to_string();
        if matches!(state.as_str(), "done" | "failed" | "canceled") {
            break state;
        }
        if started.elapsed() >= timeout {
            return Err(format!("{addr} job {job_id} timed out in state {state}"));
        }
        thread::sleep(Duration::from_millis(2));
    };
    if state != "done" {
        let status = executor
            .status(&job_id)
            .map_err(|(code, msg)| format!("{addr} job-status {code}: {msg}"))?;
        return Err(format!("{addr} job {job_id} ended {state}: {status}"));
    }
    let (header, body) = executor
        .result(&job_id)
        .map_err(|(code, msg)| format!("{addr} job-result {code}: {msg}"))?;
    validate_binary_result(&header, &body).map_err(|e| format!("{addr} invalid result: {e}"))?;
    let desc = header
        .get("outputs")
        .and_then(|x| x.as_array())
        .and_then(|a| {
            a.iter()
                .find(|d| d.get("name").and_then(|x| x.as_str()) == Some("OUT"))
        })
        .ok_or_else(|| format!("{addr} result has no OUT descriptor"))?;
    let off = desc.get("off").and_then(|x| x.as_u64()).unwrap_or(0) as usize;
    let len = desc
        .get("bytes")
        .and_then(|x| x.as_u64())
        .ok_or_else(|| format!("{addr} OUT has no byte count"))? as usize;
    let end = off
        .checked_add(len)
        .ok_or_else(|| format!("{addr} OUT range overflows"))?;
    if end > body.len() {
        return Err(format!("{addr} OUT is truncated"));
    }
    Ok(HalfRun {
        job_id,
        bytes: body[off..end].to_vec(),
        elapsed_ms: started.elapsed().as_millis() as u64,
    })
}

fn bodies_for_frame(listing: &str, split: Split, fb: &[u8]) -> Result<(Vec<u8>, Vec<u8>), String> {
    let ((lx, lw), (rx, rw)) = split.windows();
    let left = half_payload(rgb_window(fb, lx, lw)?, lw)?;
    let right = half_payload(rgb_window(fb, rx, rw)?, rw)?;
    Ok((body_for(listing, left)?, body_for(listing, right)?))
}

fn black_bodies(listing: &str, split: Split) -> Result<(Vec<u8>, Vec<u8>), String> {
    let ((_, lw), (_, rw)) = split.windows();
    let left = half_payload(vec![0.0f32; FB_H * lw * 3], lw)?;
    let right = half_payload(vec![0.0f32; FB_H * rw * 3], rw)?;
    Ok((body_for(listing, left)?, body_for(listing, right)?))
}

struct PairRun {
    cuda: Result<HalfRun, String>,
    hip: Result<HalfRun, String>,
    total_ms: u64,
}

fn run_bodies(cfg: &Config, cuda_body: &[u8], hip_body: &[u8], timeout: Duration) -> PairRun {
    let cuda_addr = cfg.cuda.clone();
    let hip_addr = cfg.hip.clone();
    let token = cfg.token.clone();
    let started = Instant::now();
    let (a, b) = thread::scope(|s| {
        let a = s.spawn(|| run_job(&cuda_addr, "cuda", token.clone(), cuda_body, timeout));
        let b = s.spawn(|| run_job(&hip_addr, "hip", token.clone(), hip_body, timeout));
        (a.join().unwrap(), b.join().unwrap())
    });
    PairRun {
        cuda: a.map_err(|e| format!("cuda: {e}")),
        hip: b.map_err(|e| format!("hip: {e}")),
        total_ms: started.elapsed().as_millis() as u64,
    }
}

fn full_body(listing: &str, fb: &[u8]) -> Result<Vec<u8>, String> {
    let data = rgb_window(fb, 0, FB_W)?;
    let payload = half_payload(data, FB_W)?;
    body_for(listing, payload)
}

fn black_full(listing: &str) -> Result<Vec<u8>, String> {
    let data = vec![0.0f32; FB_H * FB_W * 3];
    let payload = half_payload(data, FB_W)?;
    body_for(listing, payload)
}

fn run_one(cfg: &Config, node: Node, body: &[u8], timeout: Duration) -> Result<HalfRun, String> {
    match node {
        Node::Cuda => run_job(&cfg.cuda, "cuda", cfg.token.clone(), body, timeout),
        Node::Hip => run_job(&cfg.hip, "hip", cfg.token.clone(), body, timeout),
    }
}

fn ping_node(cfg: &Config, node: Node) -> Result<(), String> {
    let addr = match node {
        Node::Cuda => &cfg.cuda,
        Node::Hip => &cfg.hip,
    };
    let client = JobClient::new(addr, cfg.token.clone());
    client
        .ping()
        .map(|_| ())
        .map_err(|(code, msg)| format!("{code}: {msg}"))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Node {
    Cuda,
    Hip,
}

impl Node {
    fn name(self) -> &'static str {
        match self {
            Node::Cuda => "cuda",
            Node::Hip => "hip",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Mode {
    Split,
    CudaFull,
    HipFull,
}

impl Mode {
    fn name(self) -> &'static str {
        match self {
            Mode::Split => "split",
            Mode::CudaFull => "cuda-full",
            Mode::HipFull => "hip-full",
        }
    }
}

fn decode_split(split: Split, cuda: &HalfRun, hip: &HalfRun) -> Result<Vec<u8>, String> {
    let left = f32_rgb_to_u8(&cuda.bytes, split.left_out_width(), OUT_H)
        .map_err(|e| format!("cuda output: {e}"))?;
    let right = f32_rgb_to_u8(&hip.bytes, split.right_out_width(), OUT_H)
        .map_err(|e| format!("hip output: {e}"))?;
    split
        .stitch(&left, &right)
        .map_err(|e| format!("stitch: {e}"))
}

fn decode_full(run: &HalfRun) -> Result<Vec<u8>, String> {
    f32_rgb_to_u8(&run.bytes, OUT_W, OUT_H).map_err(|e| format!("full output: {e}"))
}

/// If the calibrated split is predicted to miss the frame budget, choose a
/// single-node full frame only when it is predicted to be faster.
fn deadline_choice(probe: &Json, split: Split, budget_ms: u64) -> Option<(Mode, Json)> {
    let p50 = |key: &str| -> Option<u64> {
        probe
            .get("rebalance")
            .and_then(|r| r.get(key))
            .or_else(|| probe.get(key))
            .and_then(|x| x.as_u64())
    };
    let cuda = p50("cuda_p50")?;
    let hip = p50("hip_p50")?;
    let split_est = cuda.max(hip);
    let cuda_full = cuda.saturating_mul(FB_W as u64) / split.left_cols.max(1) as u64;
    let hip_full = hip.saturating_mul(FB_W as u64) / split.right_cols.max(1) as u64;
    let best_full = cuda_full.min(hip_full);
    if split_est <= budget_ms || best_full >= split_est {
        return None;
    }
    let mode = if cuda_full <= hip_full {
        Mode::CudaFull
    } else {
        Mode::HipFull
    };
    Some((
        mode,
        json!({
            "split_est_ms": split_est,
            "cuda_full_est_ms": cuda_full,
            "hip_full_est_ms": hip_full,
            "budget_ms": budget_ms,
            "choice": mode.name(),
        }),
    ))
}

#[derive(Clone, Default)]
struct ViewerHub {
    writer: Arc<Mutex<Option<TcpStream>>>,
    keys: Arc<Mutex<Vec<(bool, u8)>>>,
    connected: Arc<AtomicBool>,
    quit: Arc<AtomicBool>,
}

fn start_viewer(addr: &str, hub: ViewerHub) -> Result<(), String> {
    let listener = TcpListener::bind(addr).map_err(|e| format!("bind viewer {addr}: {e}"))?;
    eprintln!("viewer: listening on {addr}");
    thread::spawn(move || {
        for conn in listener.incoming() {
            let Ok(stream) = conn else {
                continue;
            };
            let _ = stream.set_nodelay(true);
            let _ = stream.set_write_timeout(Some(Duration::from_millis(500)));
            let Ok(mut reader) = stream.try_clone() else {
                continue;
            };
            *hub.writer.lock().unwrap_or_else(|e| e.into_inner()) = Some(stream);
            hub.connected.store(true, Ordering::SeqCst);
            let hub2 = hub.clone();
            thread::spawn(move || {
                loop {
                    let mut len = [0u8; 4];
                    if reader.read_exact(&mut len).is_err() {
                        break;
                    }
                    let n = u32::from_be_bytes(len) as usize;
                    if n == 0 || n > 64 * 1024 {
                        break;
                    }
                    let mut buf = vec![0u8; n];
                    if reader.read_exact(&mut buf).is_err() {
                        break;
                    }
                    let Ok(v) = serde_json::from_slice::<Json>(&buf) else {
                        continue;
                    };
                    match v.get("type").and_then(|x| x.as_str()) {
                        Some("key") => {
                            let pressed =
                                v.get("pressed").and_then(|x| x.as_bool()).unwrap_or(false);
                            let key = v.get("key").and_then(|x| x.as_u64()).unwrap_or(0) as u8;
                            hub2.keys
                                .lock()
                                .unwrap_or_else(|e| e.into_inner())
                                .push((pressed, key));
                        }
                        Some("quit") => {
                            hub2.quit.store(true, Ordering::SeqCst);
                            break;
                        }
                        _ => {}
                    }
                }
                hub2.connected.store(false, Ordering::SeqCst);
                *hub2.writer.lock().unwrap_or_else(|e| e.into_inner()) = None;
            });
        }
    });
    Ok(())
}

fn drain_viewer_keys(hub: &ViewerHub, cpu: &mut Rv32) -> Vec<Json> {
    let keys = {
        let mut q = hub.keys.lock().unwrap_or_else(|e| e.into_inner());
        std::mem::take(&mut *q)
    };
    let mut out = Vec::new();
    for (pressed, key) in keys {
        cpu.key(pressed, key);
        out.push(json!({"pressed": pressed, "key": key}));
    }
    out
}

fn send_viewer_frame(hub: &ViewerHub, header: &Json, rgb: &[u8]) {
    if !hub.connected.load(Ordering::SeqCst) {
        return;
    }
    let hbytes = match serde_json::to_vec(header) {
        Ok(v) => v,
        Err(_) => return,
    };
    let mut guard = hub.writer.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(stream) = guard.as_mut() {
        let ok = stream
            .write_all(&(hbytes.len() as u32).to_be_bytes())
            .is_ok()
            && stream.write_all(&hbytes).is_ok()
            && stream.write_all(rgb).is_ok();
        if !ok {
            *guard = None;
            hub.connected.store(false, Ordering::SeqCst);
        }
    }
}

fn press(cpu: &mut Rv32, pressed: bool, key: u8) {
    cpu.key(pressed, key);
}

fn start_keys(cpu: &mut Rv32, frame: u32) {
    match frame {
        20 => press(cpu, true, KEY_ENTER),
        30 => press(cpu, false, KEY_ENTER),
        100 => press(cpu, true, KEY_ENTER),
        110 => press(cpu, false, KEY_ENTER),
        130 => press(cpu, true, KEY_ENTER),
        140 => press(cpu, false, KEY_ENTER),
        f if (160..=200).contains(&f) => press(cpu, f % 2 == 0, KEY_ENTER),
        _ => {}
    }
}

fn game_keys(cpu: &mut Rv32, index: u32) -> Vec<&'static str> {
    let mut out = Vec::new();
    match index {
        5 => {
            press(cpu, true, KEY_UP);
            out.push("up");
        }
        30 => {
            press(cpu, false, KEY_UP);
            out.push("up_release");
        }
        35 => {
            press(cpu, true, KEY_FIRE);
            out.push("fire");
        }
        41 => {
            press(cpu, false, KEY_FIRE);
            out.push("fire_release");
        }
        45 => {
            press(cpu, true, KEY_LEFT);
            out.push("left");
        }
        60 => {
            press(cpu, false, KEY_LEFT);
            out.push("left_release");
        }
        63 => {
            press(cpu, true, KEY_FIRE);
            out.push("fire");
        }
        69 => {
            press(cpu, false, KEY_FIRE);
            out.push("fire_release");
        }
        _ => {}
    }
    out
}

fn frame_budget_ms(fps: u32) -> u64 {
    (1000 / fps.max(1)) as u64
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
    let image = fs::read(&cfg.image).map_err(|e| format!("read {}: {e}", cfg.image.display()))?;
    let wad = fs::read(&cfg.wad).map_err(|e| format!("read {}: {e}", cfg.wad.display()))?;
    let listing = fs::read_to_string(&cfg.listing)
        .map_err(|e| format!("read {}: {e}", cfg.listing.display()))?;
    let listing_sha = sha256_hex(listing.as_bytes());
    let image_sha = sha256_hex(&image);
    let wad_sha = sha256_hex(&wad);
    fs::create_dir_all(cfg.out.join("frames"))
        .map_err(|e| format!("create {}: {e}", cfg.out.display()))?;
    let viewer_hub = ViewerHub::default();
    if let Some(addr) = &cfg.viewer {
        start_viewer(addr, viewer_hub.clone())?;
    }

    let mut cpu = Rv32::new(64 << 20);
    cpu.load_doom(&image, &wad).map_err(|e| e.to_string())?;
    eprintln!(
        "node A: RV32IM loading image {} wad {}",
        &image_sha[..12],
        &wad_sha[..12]
    );

    for f in 1..=cfg.start_frame {
        start_keys(&mut cpu, f);
        cpu.run_to_frame(f, cfg.max_instr);
        if cpu.frame() < f {
            return Err(format!(
                "RV32IM stopped at frame {} while advancing to {f}",
                cpu.frame()
            ));
        }
    }
    eprintln!(
        "node A: frame {} reached (pc=0x{:08x}, cycles={})",
        cpu.frame(),
        cpu.pc,
        cpu.cycles
    );

    let timeout = Duration::from_millis(cfg.job_timeout_ms);
    let startup_timeout = Duration::from_secs(60);
    let mut cuda_ok = true;
    let mut hip_ok = true;
    let mut failover_events: Vec<Json> = Vec::new();
    let mut recovery_events: Vec<Json> = Vec::new();
    let mut calibration = json!({});
    let mut split = Split::EVEN;

    // Warm the full-frame fallback on both nodes before calibrating the split.
    // A failover reuses this shape, so the surviving node should not pay
    // artifact build or worker startup in the middle of the run.
    let mut full_warm = json!({});
    if cfg.warmup {
        let full = black_full(&listing)?;
        let (c, h) = thread::scope(|s| {
            let c = s.spawn(|| run_one(&cfg, Node::Cuda, &full, startup_timeout));
            let h = s.spawn(|| run_one(&cfg, Node::Hip, &full, startup_timeout));
            (c.join().unwrap(), h.join().unwrap())
        });
        match c {
            Ok(r) => full_warm["cuda_ms"] = json!(r.elapsed_ms),
            Err(e) => {
                cuda_ok = false;
                full_warm["cuda_error"] = json!(e);
            }
        }
        match h {
            Ok(r) => full_warm["hip_ms"] = json!(r.elapsed_ms),
            Err(e) => {
                hip_ok = false;
                full_warm["hip_error"] = json!(e);
            }
        }
        eprintln!("full-frame fallback warmup: {full_warm}");
    }
    if !cuda_ok && !hip_ok {
        return Err("both GPU nodes failed during fallback warmup".to_string());
    }

    let mut probe = Json::Null;
    if cfg.auto_split && cuda_ok && hip_ok {
        let even = Split::EVEN;
        let (cb, hb) = black_bodies(&listing, even)?;
        let pair = run_bodies(&cfg, &cb, &hb, startup_timeout);
        let mut even_ok = true;
        if let Err(e) = pair.cuda {
            cuda_ok = false;
            even_ok = false;
            eprintln!("even warmup: cuda failed: {e}");
        }
        if let Err(e) = pair.hip {
            hip_ok = false;
            even_ok = false;
            eprintln!("even warmup: hip failed: {e}");
        }
        if even_ok {
            let fb = cpu.framebuffer();
            let (cb, hb) = bodies_for_frame(&listing, even, fb)?;
            let mut c = Vec::new();
            let mut h = Vec::new();
            let mut probe_ok = true;
            for _ in 0..cfg.probe_iters {
                let pair = run_bodies(&cfg, &cb, &hb, startup_timeout);
                match (pair.cuda, pair.hip) {
                    (Ok(a), Ok(b)) => {
                        c.push(a.elapsed_ms);
                        h.push(b.elapsed_ms);
                    }
                    (Err(e), _) => {
                        cuda_ok = false;
                        probe_ok = false;
                        eprintln!("calibration: cuda failed: {e}");
                        break;
                    }
                    (_, Err(e)) => {
                        hip_ok = false;
                        probe_ok = false;
                        eprintln!("calibration: hip failed: {e}");
                        break;
                    }
                }
            }
            if probe_ok {
                let (cp, _, _) = latency_summary(&c);
                let (hp, _, _) = latency_summary(&h);
                split = choose_split(cp, hp, cfg.min_cols, cfg.max_cols);
                let (cb, hb) = black_bodies(&listing, split)?;
                let pair = run_bodies(&cfg, &cb, &hb, startup_timeout);
                let mut split_warm = json!({});
                if let Ok(r) = pair.cuda {
                    split_warm["cuda_ms"] = json!(r.elapsed_ms);
                } else if let Err(e) = pair.cuda {
                    cuda_ok = false;
                    split_warm["cuda_error"] = json!(e);
                }
                if let Ok(r) = pair.hip {
                    split_warm["hip_ms"] = json!(r.elapsed_ms);
                } else if let Err(e) = pair.hip {
                    hip_ok = false;
                    split_warm["hip_error"] = json!(e);
                }

                // Iterative rebalance: measure at the chosen split and adjust
                // once if the per-column balance is materially different.
                let mut rebalance = json!({});
                if cuda_ok && hip_ok {
                    let (cb, hb) = bodies_for_frame(&listing, split, fb)?;
                    let mut rc = Vec::new();
                    let mut rh = Vec::new();
                    let mut rb_ok = true;
                    for _ in 0..cfg.probe_iters {
                        let pair = run_bodies(&cfg, &cb, &hb, startup_timeout);
                        match (pair.cuda, pair.hip) {
                            (Ok(a), Ok(b)) => {
                                rc.push(a.elapsed_ms);
                                rh.push(b.elapsed_ms);
                            }
                            (Err(e), _) => {
                                cuda_ok = false;
                                rb_ok = false;
                                eprintln!("rebalance: cuda failed: {e}");
                                break;
                            }
                            (_, Err(e)) => {
                                hip_ok = false;
                                rb_ok = false;
                                eprintln!("rebalance: hip failed: {e}");
                                break;
                            }
                        }
                    }
                    if rb_ok {
                        let (rcp, _, _) = latency_summary(&rc);
                        let (rhp, _, _) = latency_summary(&rh);
                        let c_norm = rcp * 160 / split.left_cols.max(1) as u64;
                        let h_norm = rhp * 160 / split.right_cols.max(1) as u64;
                        let next = choose_split(c_norm, h_norm, cfg.min_cols, cfg.max_cols);
                        let before = (split.left_cols, split.right_cols);
                        let mut changed = next.left_cols.abs_diff(split.left_cols) >= 4;
                        if changed {
                            let (cb, hb) = black_bodies(&listing, next)?;
                            let pair = run_bodies(&cfg, &cb, &hb, startup_timeout);
                            let mut warm = json!({});
                            if let Ok(r) = pair.cuda {
                                warm["cuda_ms"] = json!(r.elapsed_ms);
                            } else if let Err(e) = pair.cuda {
                                cuda_ok = false;
                                warm["cuda_error"] = json!(e);
                            }
                            if let Ok(r) = pair.hip {
                                warm["hip_ms"] = json!(r.elapsed_ms);
                            } else if let Err(e) = pair.hip {
                                hip_ok = false;
                                warm["hip_error"] = json!(e);
                            }
                            if cuda_ok && hip_ok {
                                split = next;
                            } else {
                                changed = false;
                            }
                            rebalance = json!({
                                "before": {
                                    "left_cols": before.0,
                                    "right_cols": before.1,
                                },
                                "warmup": warm,
                            });
                        }
                        rebalance["cuda_ms"] = json!(rc);
                        rebalance["hip_ms"] = json!(rh);
                        rebalance["cuda_p50"] = json!(rcp);
                        rebalance["hip_p50"] = json!(rhp);
                        rebalance["changed"] = json!(changed);
                        rebalance["after"] = json!({
                            "left_cols": split.left_cols,
                            "right_cols": split.right_cols,
                        });
                        eprintln!(
                            "rebalance: cuda {} ms, hip {} ms -> split {}/{} (changed {changed})",
                            rcp, rhp, split.left_cols, split.right_cols
                        );
                    }
                }

                probe = json!({
                    "cuda_ms": c,
                    "hip_ms": h,
                    "cuda_p50": cp,
                    "hip_p50": hp,
                    "chosen": {
                        "left_cols": split.left_cols,
                        "right_cols": split.right_cols,
                        "left_share": split.left_share(),
                    },
                    "split_warmup": split_warm,
                    "rebalance": rebalance,
                });
                eprintln!(
                    "calibration: cuda p50 {cp} ms, hip p50 {hp} ms -> split {}/{}",
                    split.left_cols, split.right_cols
                );
            }
        }
    }
    calibration = json!({"full_warmup": full_warm, "probe": probe});
    if !cuda_ok && !hip_ok {
        return Err("both GPU nodes failed during calibration".to_string());
    }

    let meta_path = cfg.out.join("frames.jsonl");
    let mut meta = BufWriter::new(
        File::create(&meta_path).map_err(|e| format!("create {}: {e}", meta_path.display()))?,
    );
    let mut totals = Vec::new();
    let mut cuda_samples = Vec::new();
    let mut hip_samples = Vec::new();
    let mut core_samples = Vec::new();
    let started = Instant::now();
    let budget = frame_budget_ms(cfg.fps);
    let mut mode = if cuda_ok && hip_ok {
        Mode::Split
    } else if hip_ok {
        Mode::HipFull
    } else {
        Mode::CudaFull
    };
    if mode == Mode::Split {
        if let Some((choice, info)) = deadline_choice(&probe, split, budget) {
            mode = choice;
            calibration["deadline"] = info;
            eprintln!(
                "deadline: split predicted over budget; choosing {}",
                mode.name()
            );
        }
    }
    eprintln!("placement mode: {}", mode.name());
    let mut recorded = 0u32;
    let mut recent_totals: Vec<u64> = Vec::new();

    for n in 0..cfg.frames {
        let frame = cfg.start_frame + n + 1;
        if mode != Mode::Split && cfg.recover_every > 0 && n > 0 && n % cfg.recover_every == 0 {
            let mut recovered = Vec::new();
            if !cuda_ok && ping_node(&cfg, Node::Cuda).is_ok() {
                cuda_ok = true;
                recovered.push("cuda");
            }
            if !hip_ok && ping_node(&cfg, Node::Hip).is_ok() {
                hip_ok = true;
                recovered.push("hip");
            }
            if cuda_ok && hip_ok {
                // Warm both split shapes before switching back; the healthy
                // node may have evicted its split worker while running full
                // frames.
                let (cb, hb) = black_bodies(&listing, split)?;
                let pair = run_bodies(&cfg, &cb, &hb, startup_timeout);
                let mut warmed = true;
                if let Err(e) = pair.cuda {
                    cuda_ok = false;
                    warmed = false;
                    eprintln!("recovery: cuda warm failed: {e}");
                }
                if let Err(e) = pair.hip {
                    hip_ok = false;
                    warmed = false;
                    eprintln!("recovery: hip warm failed: {e}");
                }
                if warmed {
                    mode = Mode::Split;
                    recovery_events.push(json!({
                        "frame": frame,
                        "nodes": recovered,
                        "action": "rejoin-split",
                    }));
                    eprintln!("recovery at frame {frame}: split resumed");
                } else {
                    recovery_events.push(json!({
                        "frame": frame,
                        "nodes": recovered,
                        "action": "warm-failed",
                    }));
                }
            } else if !recovered.is_empty() {
                recovery_events.push(json!({
                    "frame": frame,
                    "nodes": recovered,
                    "action": "ping-only",
                }));
                eprintln!("recovery at frame {frame}: {recovered:?} reachable, waiting");
            }
        }
        let actions: Vec<Json> = if cfg.viewer.is_some() {
            drain_viewer_keys(&viewer_hub, &mut cpu)
        } else {
            game_keys(&mut cpu, n)
                .into_iter()
                .map(|s| json!(s))
                .collect()
        };
        let t_core = Instant::now();
        cpu.run_to_frame(frame, cfg.max_instr);
        let core_ms = t_core.elapsed().as_millis() as u64;
        if cpu.frame() < frame {
            return Err(format!(
                "RV32IM stopped at frame {} while advancing to {frame}",
                cpu.frame()
            ));
        }
        let fb = cpu.framebuffer();
        let t_frame = Instant::now();
        let mut used = mode;
        let mut cuda_ms = 0u64;
        let mut hip_ms = 0u64;
        let mut cuda_job = String::new();
        let mut hip_job = String::new();
        let mut cuda_sha = String::new();
        let mut hip_sha = String::new();
        let frame_rgb = match mode {
            Mode::Split => {
                let (cuda_body, hip_body) = bodies_for_frame(&listing, split, fb)?;
                let pair = run_bodies(&cfg, &cuda_body, &hip_body, timeout);
                match (pair.cuda, pair.hip) {
                    (Ok(cuda), Ok(hip)) => {
                        cuda_ms = cuda.elapsed_ms;
                        hip_ms = hip.elapsed_ms;
                        cuda_job = cuda.job_id.clone();
                        hip_job = hip.job_id.clone();
                        cuda_sha = sha256_hex(&cuda.bytes);
                        hip_sha = sha256_hex(&hip.bytes);
                        decode_split(split, &cuda, &hip)
                            .map_err(|e| format!("frame {frame} split output: {e}"))?
                    }
                    (Ok(_), Err(e)) => {
                        hip_ok = false;
                        mode = Mode::CudaFull;
                        used = Mode::CudaFull;
                        failover_events.push(json!({
                            "frame": frame,
                            "node": "hip",
                            "error": e,
                            "action": "cuda-full",
                        }));
                        eprintln!(
                            "failover at frame {frame}: hip lost ({e}); CUDA takes the full frame"
                        );
                        let full = full_body(&listing, fb)?;
                        let r = run_one(&cfg, Node::Cuda, &full, timeout)
                            .map_err(|e| format!("frame {frame} cuda fallback: {e}"))?;
                        cuda_ms = r.elapsed_ms;
                        cuda_job = r.job_id.clone();
                        cuda_sha = sha256_hex(&r.bytes);
                        decode_full(&r)
                            .map_err(|e| format!("frame {frame} cuda full output: {e}"))?
                    }
                    (Err(e), Ok(_)) => {
                        cuda_ok = false;
                        mode = Mode::HipFull;
                        used = Mode::HipFull;
                        failover_events.push(json!({
                            "frame": frame,
                            "node": "cuda",
                            "error": e,
                            "action": "hip-full",
                        }));
                        eprintln!(
                            "failover at frame {frame}: cuda lost ({e}); HIP takes the full frame"
                        );
                        let full = full_body(&listing, fb)?;
                        let r = run_one(&cfg, Node::Hip, &full, timeout)
                            .map_err(|e| format!("frame {frame} hip fallback: {e}"))?;
                        hip_ms = r.elapsed_ms;
                        hip_job = r.job_id.clone();
                        hip_sha = sha256_hex(&r.bytes);
                        decode_full(&r)
                            .map_err(|e| format!("frame {frame} hip full output: {e}"))?
                    }
                    (Err(e1), Err(e2)) => {
                        return Err(format!("frame {frame}: both nodes failed: {e1}; {e2}"))
                    }
                }
            }
            Mode::CudaFull => {
                let full = full_body(&listing, fb)?;
                let r = run_one(&cfg, Node::Cuda, &full, timeout)
                    .map_err(|e| format!("frame {frame} cuda full: {e}"))?;
                cuda_ms = r.elapsed_ms;
                cuda_job = r.job_id.clone();
                cuda_sha = sha256_hex(&r.bytes);
                decode_full(&r).map_err(|e| format!("frame {frame} cuda output: {e}"))?
            }
            Mode::HipFull => {
                let full = full_body(&listing, fb)?;
                let r = run_one(&cfg, Node::Hip, &full, timeout)
                    .map_err(|e| format!("frame {frame} hip full: {e}"))?;
                hip_ms = r.elapsed_ms;
                hip_job = r.job_id.clone();
                hip_sha = sha256_hex(&r.bytes);
                decode_full(&r).map_err(|e| format!("frame {frame} hip output: {e}"))?
            }
        };
        let total_ms = t_frame.elapsed().as_millis() as u64;
        let file = format!("frames/frame_{n:04}.rgb");
        fs::write(cfg.out.join(&file), &frame_rgb).map_err(|e| format!("write {file}: {e}"))?;
        let sha = sha256_hex(&frame_rgb);
        let record = json!({
            "index": n,
            "frame": frame,
            "file": file,
            "width": OUT_W,
            "height": OUT_H,
            "mode": used.name(),
            "split": {
                "left_cols": split.left_cols,
                "right_cols": split.right_cols,
                "left_share": split.left_share(),
            },
            "core_ms": core_ms,
            "cuda_ms": cuda_ms,
            "hip_ms": hip_ms,
            "total_ms": total_ms,
            "cuda_job": cuda_job,
            "hip_job": hip_job,
            "cuda_sha256": cuda_sha,
            "hip_sha256": hip_sha,
            "sha256": sha,
            "inputs": actions,
        });
        writeln!(meta, "{}", serde_json::to_string(&record).unwrap())
            .map_err(|e| format!("write metadata: {e}"))?;
        totals.push(total_ms);
        recent_totals.push(total_ms);
        if recent_totals.len() > 30 {
            recent_totals.remove(0);
        }
        let (rp50, rp95, _) = latency_summary(&recent_totals);
        if cuda_ms > 0 {
            cuda_samples.push(cuda_ms);
        }
        if hip_ms > 0 {
            hip_samples.push(hip_ms);
        }
        core_samples.push(core_ms);
        recorded += 1;
        if cfg.viewer.is_some() {
            let note = if let Some(last) = recovery_events.last() {
                format!(
                    "recovered {} -> split",
                    last.get("nodes")
                        .and_then(|x| x.as_array())
                        .map(|a| a
                            .iter()
                            .filter_map(|v| v.as_str())
                            .collect::<Vec<_>>()
                            .join("+"))
                        .unwrap_or_else(|| "node".to_string())
                )
            } else if let Some(last) = failover_events.last() {
                format!(
                    "failover {} -> {}",
                    last.get("node").and_then(|x| x.as_str()).unwrap_or("?"),
                    last.get("action").and_then(|x| x.as_str()).unwrap_or("?")
                )
            } else if cuda_ok && hip_ok {
                "all nodes healthy".to_string()
            } else {
                "degraded".to_string()
            };
            let info = HudInfo {
                frame,
                mode: used.name().to_string(),
                cuda_addr: cfg.cuda.clone(),
                hip_addr: cfg.hip.clone(),
                cuda_cols: if used == Mode::Split {
                    split.left_cols
                } else {
                    0
                },
                hip_cols: if used == Mode::Split {
                    split.right_cols
                } else {
                    0
                },
                cuda_ms,
                hip_ms,
                total_ms,
                budget_ms: budget,
                p50: rp50,
                p95: rp95,
                cuda_ok,
                hip_ok,
                note,
            };
            let hud = render_hud(&frame_rgb, OUT_W, OUT_H, &info);
            let (width, height, body) = match hud {
                Ok(hud) => (OUT_W, OUT_H + HUD_TOP + HUD_BOTTOM, hud),
                Err(_) => (OUT_W, OUT_H, frame_rgb.clone()),
            };
            let header = json!({
                "type": "frame",
                "frame": frame,
                "width": width,
                "height": height,
                "bytes": body.len(),
                "mode": used.name(),
                "split": {
                    "left_cols": split.left_cols,
                    "right_cols": split.right_cols,
                    "left_share": split.left_share(),
                },
                "cuda_ms": cuda_ms,
                "hip_ms": hip_ms,
                "total_ms": total_ms,
                "failover": failover_events.last().cloned(),
            });
            send_viewer_frame(&viewer_hub, &header, &body);
        }
        if viewer_hub.quit.load(Ordering::SeqCst) {
            eprintln!("viewer quit at frame {frame}");
            break;
        }
        if n % 10 == 0 || n + 1 == cfg.frames || used != Mode::Split {
            eprintln!(
                "frame {frame} ({}/{}) mode {} core {} ms, cuda {} ms, hip {} ms, total {} ms",
                n + 1,
                cfg.frames,
                used.name(),
                core_ms,
                cuda_ms,
                hip_ms,
                total_ms
            );
        }
        if total_ms < budget {
            thread::sleep(Duration::from_millis(budget - total_ms));
        }
    }
    meta.flush().map_err(|e| format!("flush metadata: {e}"))?;

    let (p50, p95, max) = latency_summary(&totals);
    let (c50, _, _) = latency_summary(&cuda_samples);
    let (h50, _, _) = latency_summary(&hip_samples);
    let (k50, _, _) = latency_summary(&core_samples);
    let summary = json!({
        "schema": "jotnar.doom.two_nodes/1",
        "image": cfg.image.display().to_string(),
        "wad": cfg.wad.display().to_string(),
        "image_sha256": image_sha,
        "wad_sha256": wad_sha,
        "listing": cfg.listing.display().to_string(),
        "listing_sha256": listing_sha,
        "node_a": "rv32im-native",
        "node_b": {"target": "cuda", "addr": cfg.cuda},
        "node_c": {"target": "hip", "addr": cfg.hip},
        "start_frame": cfg.start_frame,
        "frames": recorded,
        "requested_frames": cfg.frames,
        "fps": cfg.fps,
        "frame_budget_ms": budget,
        "auto_split": cfg.auto_split,
        "viewer": cfg.viewer,
        "split": {
            "left_cols": split.left_cols,
            "right_cols": split.right_cols,
            "left_share": split.left_share(),
        },
        "calibration": calibration,
        "mode": mode.name(),
        "failover": {
            "events": failover_events,
            "recoveries": recovery_events,
            "final_mode": mode.name(),
            "cuda_ok": cuda_ok,
            "hip_ok": hip_ok,
        },
        "latency_ms": {
            "total_p50": p50,
            "total_p95": p95,
            "total_max": max,
            "cuda_p50": c50,
            "hip_p50": h50,
            "core_p50": k50,
        },
        "real_time_at_target": p95 <= budget,
        "wall_ms": started.elapsed().as_millis() as u64,
        "frames_dir": cfg.out.join("frames").display().to_string(),
        "metadata": meta_path.display().to_string(),
        "finished_unix_ms": now_ms(),
    });
    let summary_path = cfg.out.join("run.json");
    fs::write(
        &summary_path,
        serde_json::to_vec_pretty(&summary).map_err(|e| e.to_string())?,
    )
    .map_err(|e| format!("write {}: {e}", summary_path.display()))?;
    println!("{}", serde_json::to_string_pretty(&summary).unwrap());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frame_budget_is_positive() {
        assert_eq!(frame_budget_ms(12), 83);
        assert_eq!(frame_budget_ms(0), 1000);
    }

    #[test]
    fn deadline_choice_keeps_split_when_inside_budget() {
        let probe = json!({"cuda_p50": 40, "hip_p50": 20});
        assert!(deadline_choice(&probe, Split::new(80, 240).unwrap(), 66).is_none());
    }

    #[test]
    fn deadline_choice_prefers_a_faster_full_frame() {
        let probe = json!({
            "rebalance": {"cuda_p50": 40, "hip_p50": 20}
        });
        let (mode, info) = deadline_choice(&probe, Split::new(40, 280).unwrap(), 30).unwrap();
        assert_eq!(mode, Mode::HipFull);
        assert_eq!(info["choice"], "hip-full");
    }
}
