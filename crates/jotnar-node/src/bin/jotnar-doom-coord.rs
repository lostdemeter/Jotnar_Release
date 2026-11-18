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
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use serde_json::{json, Value as Json};

use jotnar::payload::feeds_to_json;
use jotnar::rv32::Rv32;
use jotnar::value::{array_from_vec, Value};
use jotnar_node_tools::doom_pipeline::{
    f32_rgb_to_u8, latency_summary, rgb_half, sha256_hex, stitch_overlap, FB_H, OUT_H, OUT_W,
    SEND_OUT_W, SEND_W,
};
use jotnar_node_tools::executor::{validate_binary_result, Executor, RemoteExecutor};

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
         [--out DIR] [--max-instr N] [--no-warmup]\n\
     \n\
     Node A runs the RV32IM core here. Nodes B and C run the Jotnar upscale\n\
     program on the CUDA and HIP runners. The output is raw RGB frames plus\n\
     metadata; scripts/doom_two_node_gate.py assembles the GIF."
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
    })
}

fn half_payload(data: Vec<f32>) -> Result<Json, String> {
    let arr = array_from_vec(&[FB_H, SEND_W, 3], data).map_err(|e| e.to_string())?;
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

    let timeout = Duration::from_secs(120);
    let mut warmup = json!({});
    if cfg.warmup {
        let black = half_payload(vec![0.0f32; FB_H * SEND_W * 3])?;
        let cuda_body = body_for(&listing, black.clone())?;
        let hip_body = body_for(&listing, black)?;
        let cuda_addr = cfg.cuda.clone();
        let hip_addr = cfg.hip.clone();
        let token = cfg.token.clone();
        let (a, b) = thread::scope(|s| {
            let a = s.spawn(|| run_job(&cuda_addr, "cuda", token.clone(), &cuda_body, timeout));
            let b = s.spawn(|| run_job(&hip_addr, "hip", token.clone(), &hip_body, timeout));
            (a.join().unwrap(), b.join().unwrap())
        });
        let a = a.map_err(|e| format!("cuda warmup: {e}"))?;
        let b = b.map_err(|e| format!("hip warmup: {e}"))?;
        warmup = json!({"cuda_ms": a.elapsed_ms, "hip_ms": b.elapsed_ms});
        eprintln!("warmup: cuda {} ms, hip {} ms", a.elapsed_ms, b.elapsed_ms);
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

    for n in 0..cfg.frames {
        let frame = cfg.start_frame + n + 1;
        let actions = game_keys(&mut cpu, n);
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
        let left = half_payload(rgb_half(fb, 0)?)?;
        let right = half_payload(rgb_half(fb, 1)?)?;
        let cuda_body = body_for(&listing, left)?;
        let hip_body = body_for(&listing, right)?;
        let cuda_addr = cfg.cuda.clone();
        let hip_addr = cfg.hip.clone();
        let token = cfg.token.clone();
        let t_frame = Instant::now();
        let (cuda, hip) = thread::scope(|s| {
            let a = s.spawn(|| run_job(&cuda_addr, "cuda", token.clone(), &cuda_body, timeout));
            let b = s.spawn(|| run_job(&hip_addr, "hip", token.clone(), &hip_body, timeout));
            (a.join().unwrap(), b.join().unwrap())
        });
        let total_ms = t_frame.elapsed().as_millis() as u64;
        let cuda = cuda.map_err(|e| format!("frame {frame} cuda: {e}"))?;
        let hip = hip.map_err(|e| format!("frame {frame} hip: {e}"))?;
        let left8 = f32_rgb_to_u8(&cuda.bytes, SEND_OUT_W, OUT_H)
            .map_err(|e| format!("frame {frame} cuda output: {e}"))?;
        let right8 = f32_rgb_to_u8(&hip.bytes, SEND_OUT_W, OUT_H)
            .map_err(|e| format!("frame {frame} hip output: {e}"))?;
        let frame_rgb =
            stitch_overlap(&left8, &right8).map_err(|e| format!("frame {frame} stitch: {e}"))?;
        let file = format!("frames/frame_{n:04}.rgb");
        fs::write(cfg.out.join(&file), &frame_rgb).map_err(|e| format!("write {file}: {e}"))?;
        let sha = sha256_hex(&frame_rgb);
        let record = json!({
            "index": n,
            "frame": frame,
            "file": file,
            "width": OUT_W,
            "height": OUT_H,
            "core_ms": core_ms,
            "cuda_ms": cuda.elapsed_ms,
            "hip_ms": hip.elapsed_ms,
            "total_ms": total_ms,
            "cuda_job": cuda.job_id,
            "hip_job": hip.job_id,
            "cuda_sha256": sha256_hex(&cuda.bytes),
            "hip_sha256": sha256_hex(&hip.bytes),
            "sha256": sha,
            "inputs": actions,
        });
        writeln!(meta, "{}", serde_json::to_string(&record).unwrap())
            .map_err(|e| format!("write metadata: {e}"))?;
        totals.push(total_ms);
        cuda_samples.push(cuda.elapsed_ms);
        hip_samples.push(hip.elapsed_ms);
        core_samples.push(core_ms);
        if n % 10 == 0 || n + 1 == cfg.frames {
            eprintln!(
                "frame {frame} ({}/{}) core {} ms, cuda {} ms, hip {} ms, total {} ms",
                n + 1,
                cfg.frames,
                core_ms,
                cuda.elapsed_ms,
                hip.elapsed_ms,
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
        "frames": cfg.frames,
        "fps": cfg.fps,
        "frame_budget_ms": budget,
        "warmup": warmup,
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
}
