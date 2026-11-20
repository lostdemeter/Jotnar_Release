//! Live DOOM source/sink adapter for the generic streaming controller.
//!
//! The stream controller owns the clock, column split, GPU placement, stitch,
//! failover, leases, and receipts. This binary owns only the DOOM-specific
//! source (RV32IM core + framebuffer) and the DOOM viewer sink/key forwarding.
//! It replaces the streaming half of `jotnar-doom-coord`; the coordinator is
//! kept until the HUD/calibration policy is ported.

use std::fs;
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::PathBuf;
use std::process::{Command, ExitCode};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

use serde_json::{json, Value as Json};

use jotnar::rv32::Rv32;
use jotnar_node_tools::doom_pipeline::{rgb_window, sha256_hex, FB_H, FB_W};
use jotnar_node_tools::hud::{render as render_hud, HudInfo, HUD_BOTTOM, HUD_TOP};
use jotnar_node_tools::stream::{
    run_stream_with, CalibrationSpec, ColumnSplit, Frame, FrameDirSink, FrameMeta, FrameSink,
    FrameSource, GpuEndpoint, ReceiptSpec, StreamClock, StreamDescriptor, StreamRunConfig,
    StreamSinkSpec, StreamSourceSpec, StreamStageSpec, TileSpec,
};

const KEY_ENTER: u8 = 13;
const KEY_UP: u8 = 0xAD;
const KEY_LEFT: u8 = 0xAC;
const KEY_RIGHT: u8 = 0xAE;
const KEY_FIRE: u8 = 0xA3;

type KeyQueue = Arc<Mutex<Vec<(bool, u8)>>>;

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

fn game_keys(cpu: &mut Rv32, index: u32) {
    match index {
        5 => press(cpu, true, KEY_UP),
        30 => press(cpu, false, KEY_UP),
        35 => press(cpu, true, KEY_FIRE),
        41 => press(cpu, false, KEY_FIRE),
        45 => press(cpu, true, KEY_LEFT),
        60 => press(cpu, false, KEY_LEFT),
        63 => press(cpu, true, KEY_FIRE),
        69 => press(cpu, false, KEY_FIRE),
        _ => {}
    }
}

struct DoomSource {
    cpu: Rv32,
    start_frame: u32,
    frames: Option<u32>,
    max_instr: u64,
    keys: KeyQueue,
    image_sha256: String,
    wad_sha256: String,
}

impl DoomSource {
    fn new(
        image: &PathBuf,
        wad: &PathBuf,
        start_frame: u32,
        frames: Option<u32>,
        max_instr: u64,
        keys: KeyQueue,
    ) -> Result<Self, String> {
        let image_bytes = fs::read(image).map_err(|e| format!("read {}: {e}", image.display()))?;
        let wad_bytes = fs::read(wad).map_err(|e| format!("read {}: {e}", wad.display()))?;
        let image_sha256 = sha256_hex(&image_bytes);
        let wad_sha256 = sha256_hex(&wad_bytes);
        let mut cpu = Rv32::new(64 << 20);
        cpu.load_doom(&image_bytes, &wad_bytes)
            .map_err(|e| e.to_string())?;
        for f in 1..=start_frame {
            start_keys(&mut cpu, f);
            cpu.run_to_frame(f, max_instr);
            if cpu.frame() < f {
                return Err(format!(
                    "RV32IM stopped at frame {} while advancing to {f}",
                    cpu.frame()
                ));
            }
        }
        Ok(Self {
            cpu,
            start_frame,
            frames,
            max_instr,
            keys,
            image_sha256,
            wad_sha256,
        })
    }
}

impl FrameSource for DoomSource {
    fn next_frame(&mut self, index: u64) -> Result<Option<Frame>, String> {
        if let Some(limit) = self.frames {
            if index >= limit as u64 {
                return Ok(None);
            }
        }
        let queued = {
            let mut q = self.keys.lock().unwrap_or_else(|e| e.into_inner());
            std::mem::take(&mut *q)
        };
        for (pressed, key) in queued {
            self.cpu.key(pressed, key);
        }
        game_keys(&mut self.cpu, index as u32);
        let target = self.start_frame + index as u32 + 1;
        self.cpu.run_to_frame(target, self.max_instr);
        let fb = self.cpu.framebuffer();
        let src = rgb_window(&fb, 0, FB_W)?;
        let bytes: Vec<u8> = src
            .iter()
            .map(|v| v.round().clamp(0.0, 255.0) as u8)
            .collect();
        Frame::rgb(index, FB_W, FB_H, bytes).map(Some)
    }

    fn describe(&self) -> Json {
        json!({
            "kind": "rv32im",
            "start_frame": self.start_frame,
            "frames": self.frames,
            "image_sha256": self.image_sha256,
            "wad_sha256": self.wad_sha256,
        })
    }
}

struct DoomSink {
    listener: TcpListener,
    stream: Option<TcpStream>,
    keys: KeyQueue,
    width: usize,
    height: usize,
    addr: String,
    frame: u64,
}

impl DoomSink {
    fn new(addr: &str, width: usize, height: usize, keys: KeyQueue) -> Result<Self, String> {
        let listener = TcpListener::bind(addr).map_err(|e| format!("bind viewer {addr}: {e}"))?;
        listener
            .set_nonblocking(true)
            .map_err(|e| format!("set viewer nonblocking: {e}"))?;
        Ok(Self {
            listener,
            stream: None,
            keys,
            width,
            height,
            addr: addr.to_string(),
            frame: 0,
        })
    }

    fn accept(&mut self) -> Result<(), String> {
        match self.listener.accept() {
            Ok((stream, _)) => {
                let _ = stream.set_nodelay(true);
                let _ = stream.set_write_timeout(Some(Duration::from_millis(500)));
                let mut reader = stream
                    .try_clone()
                    .map_err(|e| format!("clone viewer stream: {e}"))?;
                let keys = self.keys.clone();
                thread::spawn(move || loop {
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
                    let Ok(value) = serde_json::from_slice::<Json>(&buf) else {
                        continue;
                    };
                    match value.get("type").and_then(|x| x.as_str()) {
                        Some("key") => {
                            let pressed = value
                                .get("pressed")
                                .and_then(|x| x.as_bool())
                                .unwrap_or(false);
                            let key = value.get("key").and_then(|x| x.as_u64()).unwrap_or(0) as u8;
                            keys.lock()
                                .unwrap_or_else(|e| e.into_inner())
                                .push((pressed, key));
                        }
                        Some("quit") => break,
                        _ => {}
                    }
                });
                self.stream = Some(stream);
                Ok(())
            }
            Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => Ok(()),
            Err(e) => Err(format!("accept viewer: {e}")),
        }
    }

    fn send(&mut self, header: &Json, body: &[u8]) -> Result<(), String> {
        if self.stream.is_none() {
            self.accept()?;
        }
        let header = serde_json::to_vec(header).map_err(|e| e.to_string())?;
        let ok = self
            .stream
            .as_mut()
            .map(|stream| {
                stream
                    .write_all(&(header.len() as u32).to_be_bytes())
                    .is_ok()
                    && stream.write_all(&header).is_ok()
                    && stream.write_all(body).is_ok()
            })
            .unwrap_or(false);
        if !ok {
            self.stream = None;
        }
        Ok(())
    }
}

/// Render the DOOM HUD over an RGB frame and return `(width, height, body)`.
///
/// The controller passes [`FrameMeta`] with the measured split, per-target
/// latency, running percentiles, and failover state. The game pixels are copied
/// through unchanged; the HUD is only the top and bottom bands.
fn hud_frame(frame: &Frame, meta: &FrameMeta) -> Result<(usize, usize, Vec<u8>), String> {
    let target_addr = |target: &str| -> String {
        meta.tiles
            .iter()
            .find(|t| t.planned_target == target)
            .map(|t| t.addr.clone())
            .or_else(|| {
                meta.tiles
                    .iter()
                    .find(|t| t.target == target)
                    .map(|t| t.addr.clone())
            })
            .unwrap_or_else(|| "-".to_string())
    };
    let target_cols = |target: &str| -> usize {
        meta.tiles
            .iter()
            .filter(|t| t.planned_target == target)
            .map(|t| t.cols)
            .sum()
    };
    let target_ms = |target: &str| -> u64 {
        meta.tiles
            .iter()
            .filter(|t| t.planned_target == target)
            .map(|t| t.worker_ms.filter(|ms| *ms > 0).unwrap_or(t.latency_ms))
            .max()
            .unwrap_or(0)
    };
    let info = HudInfo {
        frame: meta.index as u32,
        mode: if meta.split.is_some() {
            "split".to_string()
        } else {
            meta.mode.clone()
        },
        cuda_addr: target_addr("cuda"),
        hip_addr: target_addr("hip"),
        cuda_cols: target_cols("cuda"),
        hip_cols: target_cols("hip"),
        cuda_ms: target_ms("cuda"),
        hip_ms: target_ms("hip"),
        total_ms: meta.work_ms,
        budget_ms: meta.budget_ms,
        p50: meta.p50_ms,
        p95: meta.p95_ms,
        cuda_ok: !meta.failed_targets.iter().any(|t| t == "cuda"),
        hip_ok: !meta.failed_targets.iter().any(|t| t == "hip"),
        note: meta.note.clone(),
    };
    let body = render_hud(&frame.data, frame.width, frame.height, &info)?;
    Ok((
        frame.width,
        frame.height + HUD_TOP + HUD_BOTTOM,
        body,
    ))
}

impl FrameSink for DoomSink {
    fn write(&mut self, frame: &Frame) -> Result<(), String> {
        self.frame = frame.index;
        let header = json!({
            "v": 1,
            "width": self.width,
            "height": self.height,
            "frame": frame.index,
            "total_ms": 0,
            "mode": "split",
        });
        self.send(&header, &frame.data)
    }

    fn write_meta(&mut self, frame: &Frame, meta: &FrameMeta) -> Result<(), String> {
        self.frame = frame.index;
        let (width, height, body) = hud_frame(frame, meta)?;
        let header = json!({
            "v": 1,
            "width": width,
            "height": height,
            "frame": frame.index,
            "total_ms": meta.work_ms,
            "mode": meta.mode,
            "split": meta.split,
            "failovers": meta.failovers,
            "recoveries": meta.recoveries,
            "failed_targets": meta.failed_targets,
            "note": meta.note,
        });
        self.send(&header, &body)
    }

    fn describe(&self) -> Json {
        json!({"kind": "doom-viewer", "listen": self.addr, "width": self.width, "height": self.height})
    }
}

/// Frames sink that writes the HUD-composited viewer frame.
///
/// Used by the headless live gate so the HUD can be verified without SDL. The
/// game region is byte-identical to the plain sink; the bands are the overlay.
struct HudDirSink {
    dir: PathBuf,
    index: u64,
}

impl HudDirSink {
    fn new(dir: &PathBuf) -> Result<Self, String> {
        fs::create_dir_all(dir).map_err(|e| format!("create {}: {e}", dir.display()))?;
        Ok(Self {
            dir: dir.clone(),
            index: 0,
        })
    }

    fn write_body(&mut self, body: &[u8]) -> Result<(), String> {
        let path = self.dir.join(format!("frame_{:06}.rgb", self.index));
        self.index += 1;
        fs::write(&path, body).map_err(|e| format!("write {}: {e}", path.display()))
    }
}

impl FrameSink for HudDirSink {
    fn write(&mut self, frame: &Frame) -> Result<(), String> {
        self.write_body(&frame.data)
    }

    fn write_meta(&mut self, frame: &Frame, meta: &FrameMeta) -> Result<(), String> {
        let (_, _, body) = hud_frame(frame, meta)?;
        self.write_body(&body)
    }

    fn describe(&self) -> Json {
        json!({
            "kind": "frames-hud",
            "dir": self.dir.display().to_string(),
            "width": FB_W * 2,
            "height": FB_H * 2 + HUD_TOP + HUD_BOTTOM,
        })
    }
}

struct Args {
    image: PathBuf,
    wad: PathBuf,
    listing: PathBuf,
    stage_descriptor: Option<PathBuf>,
    cuda: String,
    hip: String,
    token: Option<String>,
    frames: Option<u32>,
    start_frame: u32,
    max_instr: u64,
    fps: u32,
    left_cols: usize,
    sink: String,
    out: Option<PathBuf>,
    listen: String,
    receipt: Option<PathBuf>,
    fleet_dir: Option<PathBuf>,
    warmup_frames: u64,
    viewer: bool,
    calibrate: bool,
    probe_iters: u32,
    min_cols: usize,
    max_cols: usize,
    hud: bool,
    request_timeout_ms: u64,
    quarantine_frames: u64,
    warm_fallbacks: bool,
}

fn usage() -> &'static str {
    "jotnar-doom-stream — live DOOM source/sink on the generic stream controller\n\
     \n\
     usage:\n\
       jotnar-doom-stream --image FILE --wad FILE --cuda ADDR --hip ADDR \\\n\
         --listing FILE [--stage-descriptor FILE] [--token T] [--frames N] \\\n\
         [--start-frame N] [--fps N] \\\n\
         [--left-cols N] [--no-calibrate] [--probe-iters N] \\\n\
         [--min-cols N] [--max-cols N] \\\n\
         [--sink tcp|frames] [--out DIR] [--hud] [--viewer-listen ADDR] \\\n\
         [--receipt DIR] [--fleet-dir DIR] [--warmup-frames N] [--viewer] \\\n\
         [--request-timeout-ms N] [--quarantine-frames N] [--no-warm-fallbacks]\n\
     \n\
     The RV32IM core is the source; the generic controller owns the column\n\
     split, GPU placement, stitch, failover, leases, and receipt. The sink is\n\
     the existing SDL viewer protocol, a frames directory, or a HUD frames\n\
     directory (--hud) for headless verification. Split calibration is on by\n\
     default; --no-calibrate keeps the declared --left-cols split."
}

fn parse() -> Result<Args, String> {
    let raw: Vec<String> = std::env::args().skip(1).collect();
    let mut it = raw.into_iter();
    let mut image = None;
    let mut wad = None;
    let mut listing = PathBuf::from("demos/doom-two-nodes/upscale-half.asm");
    let mut stage_descriptor = None;
    let mut cuda = None;
    let mut hip = None;
    let mut token = None;
    let mut frames = None;
    let mut start_frame = 200u32;
    let mut max_instr = 2_000_000_000u64;
    let mut fps = 15u32;
    let mut left_cols = 72usize;
    let mut sink = "tcp".to_string();
    let mut out = None;
    let mut listen = "127.0.0.1:9990".to_string();
    let mut receipt = None;
    let mut fleet_dir = None;
    let mut warmup_frames = 0u64;
    let mut viewer = false;
    let mut calibrate = true;
    let mut probe_iters = 7u32;
    let mut min_cols = 32usize;
    let mut max_cols = 288usize;
    let mut hud = false;
    let mut request_timeout_ms = 1_000u64;
    let mut quarantine_frames = 4u64;
    let mut warm_fallbacks = true;
    while let Some(arg) = it.next() {
        let mut value = |flag: &str| it.next().ok_or_else(|| format!("{flag} needs a value"));
        match arg.as_str() {
            "--image" => image = Some(PathBuf::from(value("--image")?)),
            "--wad" => wad = Some(PathBuf::from(value("--wad")?)),
            "--listing" => listing = PathBuf::from(value("--listing")?),
            "--stage-descriptor" => {
                stage_descriptor = Some(PathBuf::from(value("--stage-descriptor")?))
            }
            "--cuda" => cuda = Some(value("--cuda")?),
            "--hip" => hip = Some(value("--hip")?),
            "--token" => token = Some(value("--token")?),
            "--frames" => {
                let n: u32 = value("--frames")?
                    .parse()
                    .map_err(|_| "--frames must be an integer")?;
                frames = (n > 0).then_some(n);
            }
            "--start-frame" => {
                start_frame = value("--start-frame")?
                    .parse()
                    .map_err(|_| "--start-frame must be an integer")?
            }
            "--max-instr" => {
                max_instr = value("--max-instr")?
                    .parse()
                    .map_err(|_| "--max-instr must be an integer")?
            }
            "--fps" => {
                fps = value("--fps")?
                    .parse()
                    .map_err(|_| "--fps must be an integer")?
            }
            "--left-cols" => {
                left_cols = value("--left-cols")?
                    .parse()
                    .map_err(|_| "--left-cols must be an integer")?
            }
            "--sink" => sink = value("--sink")?,
            "--out" => out = Some(PathBuf::from(value("--out")?)),
            "--viewer-listen" => listen = value("--viewer-listen")?,
            "--receipt" => receipt = Some(PathBuf::from(value("--receipt")?)),
            "--fleet-dir" => fleet_dir = Some(PathBuf::from(value("--fleet-dir")?)),
            "--warmup-frames" => {
                warmup_frames = value("--warmup-frames")?
                    .parse()
                    .map_err(|_| "--warmup-frames must be an integer")?
            }
            "--viewer" => viewer = true,
            "--no-calibrate" => calibrate = false,
            "--probe-iters" => {
                probe_iters = value("--probe-iters")?
                    .parse()
                    .map_err(|_| "--probe-iters must be an integer")?
            }
            "--min-cols" => {
                min_cols = value("--min-cols")?
                    .parse()
                    .map_err(|_| "--min-cols must be an integer")?
            }
            "--max-cols" => {
                max_cols = value("--max-cols")?
                    .parse()
                    .map_err(|_| "--max-cols must be an integer")?
            }
            "--hud" => hud = true,
            "--request-timeout-ms" => {
                request_timeout_ms = value("--request-timeout-ms")?
                    .parse()
                    .map_err(|_| "--request-timeout-ms must be an integer")?
            }
            "--quarantine-frames" => {
                quarantine_frames = value("--quarantine-frames")?
                    .parse()
                    .map_err(|_| "--quarantine-frames must be an integer")?
            }
            "--no-warm-fallbacks" => warm_fallbacks = false,
            "-h" | "--help" => return Err(usage().to_string()),
            other => return Err(format!("unknown argument {other:?}\n\n{}", usage())),
        }
    }
    Ok(Args {
        image: image.ok_or("need --image FILE")?,
        wad: wad.ok_or("need --wad FILE")?,
        listing,
        stage_descriptor,
        cuda: cuda.ok_or("need --cuda ADDR")?,
        hip: hip.ok_or("need --hip ADDR")?,
        token,
        frames,
        start_frame: start_frame.max(1),
        max_instr: max_instr.max(1),
        fps: fps.max(1),
        left_cols: left_cols.clamp(1, FB_W - 1),
        sink,
        out,
        listen,
        receipt,
        fleet_dir,
        warmup_frames,
        viewer,
        calibrate,
        probe_iters: probe_iters.max(1),
        min_cols: min_cols.max(1),
        max_cols: max_cols.max(1),
        hud,
        request_timeout_ms: request_timeout_ms.max(1),
        quarantine_frames,
        warm_fallbacks,
    })
}

fn real_main() -> Result<(), String> {
    let args = parse()?;
    if !matches!(args.sink.as_str(), "tcp" | "frames") {
        return Err("--sink must be tcp or frames".to_string());
    }
    let keys: KeyQueue = Arc::new(Mutex::new(Vec::new()));
    let source_frames = args
        .frames
        .map(|n| n.saturating_add(args.warmup_frames as u32));
    let source = Box::new(DoomSource::new(
        &args.image,
        &args.wad,
        args.start_frame,
        source_frames,
        args.max_instr,
        keys.clone(),
    )?);
    let wall_w = FB_W * 2;
    let wall_h = FB_H * 2;
    let sink_spec = match args.sink.as_str() {
        "tcp" => StreamSinkSpec {
            kind: "tcp".to_string(),
            dir: None,
            listen: Some(args.listen.clone()),
            title: Some("Jotnar DOOM".to_string()),
            hud: false,
        },
        "frames" => StreamSinkSpec {
            kind: "frames".to_string(),
            dir: Some(args.out.clone().ok_or("--sink frames needs --out DIR")?),
            listen: None,
            title: None,
            hud: false,
        },
        _ => unreachable!(),
    };
    let mut viewer = None;
    let sink: Box<dyn FrameSink> = match args.sink.as_str() {
        "tcp" => {
            if args.viewer {
                let local = PathBuf::from("target/release/jotnar-doom-view");
                let program = if local.exists() {
                    local
                } else {
                    PathBuf::from("jotnar-doom-view")
                };
                eprintln!(
                    "viewer: starting {} --addr {}",
                    program.display(),
                    args.listen
                );
                viewer = Some(
                    Command::new(program)
                        .args(["--addr", &args.listen])
                        .spawn()
                        .map_err(|e| format!("start viewer: {e}"))?,
                );
            }
            Box::new(DoomSink::new(&args.listen, wall_w, wall_h, keys)?)
        }
        "frames" => {
            if args.hud {
                Box::new(HudDirSink::new(
                    args.out.as_ref().ok_or("--out needed")?,
                )?)
            } else {
                Box::new(FrameDirSink::new(
                    args.out.as_deref().ok_or("--out needed")?,
                    wall_w,
                    wall_h,
                )?)
            }
        }
        _ => unreachable!(),
    };
    let descriptor = StreamDescriptor {
        schema: "jotnar.stream/1".to_string(),
        name: "doom-live".to_string(),
        clock: StreamClock {
            fps: args.fps,
            deadline_ms: (1000 / args.fps) as u64,
            max_in_flight: 1,
        },
        source: StreamSourceSpec {
            kind: "doom".to_string(),
            dir: None,
            device: None,
            width: FB_W,
            height: FB_H,
            channels: 3,
            format: "rgb24".to_string(),
            capture_width: None,
            capture_height: None,
            capture_fps: None,
        },
        stage: StreamStageSpec {
            listing: args.listing,
            descriptor: args.stage_descriptor,
            targets: vec!["cuda".to_string(), "hip".to_string()],
        },
        tiles: TileSpec {
            rows: 1,
            cols: 2,
            halo: 1,
            bezel: 0,
        },
        split: Some(ColumnSplit {
            left_cols: args.left_cols,
            right_cols: FB_W - args.left_cols,
            overlap: 1,
        }),
        calibration: args.calibrate.then(|| CalibrationSpec {
            probe_iters: args.probe_iters,
            min_cols: args.min_cols,
            max_cols: args.max_cols,
            rebalance: true,
        }),
        sink: sink_spec,
        merge: "columns".to_string(),
        placement: [("cuda".to_string(), 1), ("hip".to_string(), 1)]
            .into_iter()
            .collect(),
        backpressure: "drop-oldest".to_string(),
        determinism: "exact".to_string(),
        receipt: Some(ReceiptSpec {
            kind: "every-frame".to_string(),
            dir: args.receipt.or_else(|| args.out.clone()),
        }),
    };
    let endpoints = vec![
        GpuEndpoint::new("cuda", &args.cuda, args.token.clone()),
        GpuEndpoint::new("hip", &args.hip, args.token.clone()),
    ];
    let cfg = StreamRunConfig {
        frame_limit: args.frames.map(|n| n as u64),
        warmup_frames: args.warmup_frames,
        all_first_target: false,
        reference_dir: None,
        fleet_dir: args.fleet_dir,
        lease_ms: 3_000,
        timeout_ms: 10_000,
        request_timeout_ms: args.request_timeout_ms,
        quarantine_frames: args.quarantine_frames,
        warm_fallbacks: args.warm_fallbacks,
    };
    let result = run_stream_with(&descriptor, "doom-live", &endpoints, &cfg, source, sink);
    if let Some(mut child) = viewer {
        let _ = child.kill();
        let _ = child.wait();
    }
    let summary = result?;
    println!(
        "{}",
        serde_json::to_string_pretty(&summary).map_err(|e| e.to_string())?
    );
    Ok(())
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

#[cfg(test)]
mod tests {
    use super::*;
    use jotnar_node_tools::stream::{SplitMeta, TileMeta};

    fn meta() -> FrameMeta {
        let tile = |col: usize, target: &str, cols: usize| TileMeta {
            row: 0,
            col,
            target: target.to_string(),
            addr: format!("{target}:9997"),
            planned_target: target.to_string(),
            cols,
            latency_ms: 40,
            worker_ms: Some(35),
            submit_call_ms: 1,
            status_ms: 2,
            result_ms: 1,
            attempts: 1,
            failover_from: None,
            job_id: format!("job-{col}"),
            ok: true,
        };
        FrameMeta {
            index: 3,
            work_ms: 45,
            budget_ms: 66,
            deadline_ok: true,
            mode: "columns".to_string(),
            p50_ms: 44,
            p95_ms: 50,
            p99_ms: 52,
            split: Some(SplitMeta {
                left_cols: 72,
                right_cols: 248,
                left_share: 72.0 / 320.0,
            }),
            failovers: 0,
            recoveries: 0,
            failed_targets: Vec::new(),
            recovered_targets: Vec::new(),
            note: "all targets healthy".to_string(),
            tiles: vec![tile(0, "cuda", 72), tile(1, "hip", 248)],
        }
    }

    #[test]
    fn hud_frame_overlays_bands_and_keeps_game_pixels() {
        let w = 64;
        let h = 32;
        let frame = Frame::rgb(3, w, h, vec![42u8; w * h * 3]).unwrap();
        let (ow, oh, body) = hud_frame(&frame, &meta()).unwrap();
        assert_eq!((ow, oh), (w, h + HUD_TOP + HUD_BOTTOM));
        let game = HUD_TOP * w * 3;
        assert_eq!(&body[game..game + w * h * 3], &frame.data[..]);
        assert!(body[..game].iter().any(|b| *b > 20));
        let bottom = (HUD_TOP + h) * w * 3;
        assert!(body[bottom..].iter().any(|b| *b > 20));
    }

    #[test]
    fn hud_frame_records_a_down_target() {
        let w = 64;
        let h = 32;
        let frame = Frame::rgb(3, w, h, vec![0u8; w * h * 3]).unwrap();
        let mut meta = meta();
        meta.failed_targets = vec!["hip".to_string()];
        meta.tiles[1].target = "cuda".to_string();
        meta.tiles[1].failover_from = Some("hip".to_string());
        meta.tiles[1].attempts = 2;
        meta.note = "failover hip".to_string();
        let (_, _, body) = hud_frame(&frame, &meta).unwrap();
        assert!(body.iter().any(|b| *b > 20));
    }
}
