//! General streaming layer above the executor contract.
//!
//! This is the extraction of the DOOM-specific coordinator: a stream has a
//! clock, a source adapter, a tile/partition plan, one or more typed stages, a
//! fan-in/composite, and a sink adapter. The controller owns the frame clock,
//! splitting, deadlines, leases, and receipts. The executor still runs one
//! listing plus one payload per job; it never learns about streams.
//!
//! The first customer is the live camera wall. The second is DOOM, which will
//! keep only its RV32IM source and display/HUD sink once the port is complete.

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdout, Command, Stdio};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};
use serde_json::{json, Value as Json};
use sha2::{Digest, Sha256};

use crate::doom_pipeline::{choose_columns, percentile};
use crate::executor::{validate_binary_result, Executor, RemoteExecutor};
use crate::fleet::{Fleet, LeaseGuard};
use crate::hud::{render_wall, WallHudInfo, WallTileInfo, HUD_BOTTOM, HUD_TOP};

pub const STREAM_SCHEMA: &str = "jotnar.stream/1";
pub const STREAM_RECEIPT_SCHEMA: &str = "jotnar.stream.receipt/1";

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

fn three() -> usize {
    3
}

fn two() -> usize {
    2
}

fn one() -> usize {
    1
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StreamClock {
    pub fps: u32,
    pub deadline_ms: u64,
    #[serde(default = "one_u32")]
    pub max_in_flight: u32,
}

fn one_u32() -> u32 {
    1
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StreamSourceSpec {
    pub kind: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dir: Option<PathBuf>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub device: Option<String>,
    pub width: usize,
    pub height: usize,
    #[serde(default = "three")]
    pub channels: usize,
    #[serde(default = "rgb24")]
    pub format: String,
    /// Camera capture size before the stream downscale.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub capture_width: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub capture_height: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub capture_fps: Option<u32>,
}

fn rgb24() -> String {
    "rgb24".to_string()
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StreamStageSpec {
    pub listing: PathBuf,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub descriptor: Option<PathBuf>,
    #[serde(default)]
    pub targets: Vec<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TileSpec {
    #[serde(default = "two")]
    pub rows: usize,
    #[serde(default = "two")]
    pub cols: usize,
    /// Pixels of overlap on each side of a tile before the 2x upscale.
    #[serde(default = "one")]
    pub halo: usize,
    /// Display/export bezel between tiles. 0 keeps the composite seamless.
    #[serde(default)]
    pub bezel: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ColumnSplit {
    pub left_cols: usize,
    pub right_cols: usize,
    #[serde(default = "one")]
    pub overlap: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StreamSinkSpec {
    pub kind: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dir: Option<PathBuf>,
    /// Address for a TCP frame sink (the existing SDL viewer protocol).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub listen: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
    /// Render the controller metadata as a wall HUD in the TCP viewer frame.
    #[serde(default)]
    pub hud: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ReceiptSpec {
    #[serde(default = "every_frame")]
    pub kind: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dir: Option<PathBuf>,
}

fn every_frame() -> String {
    "every-frame".to_string()
}

/// Controller-side split calibration policy for a `merge=columns` stream.
///
/// The controller probes both column sides on the first real source frame,
/// chooses a split from the measured times, warms it, and (optionally) does one
/// rebalance pass. The executor never sees this policy: it only runs the tiles
/// the controller submits.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CalibrationSpec {
    /// Probe iterations per measurement point.
    #[serde(default = "seven_u32")]
    pub probe_iters: u32,
    /// Smallest number of columns either side may receive.
    #[serde(default = "thirty_two")]
    pub min_cols: usize,
    /// Largest number of columns either side may receive.
    #[serde(default = "two_eighty_eight")]
    pub max_cols: usize,
    /// Measure the chosen split and adjust once if the normalized balance
    /// differs by at least four columns.
    #[serde(default = "default_true")]
    pub rebalance: bool,
}

fn seven_u32() -> u32 {
    7
}

fn thirty_two() -> usize {
    32
}

fn two_eighty_eight() -> usize {
    288
}

fn default_true() -> bool {
    true
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StreamDescriptor {
    pub schema: String,
    pub name: String,
    pub clock: StreamClock,
    pub source: StreamSourceSpec,
    pub stage: StreamStageSpec,
    pub tiles: TileSpec,
    /// Optional 1D column split for the DOOM shape. Requires merge="columns".
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub split: Option<ColumnSplit>,
    /// Optional controller-side split calibration. Requires a column split.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub calibration: Option<CalibrationSpec>,
    pub sink: StreamSinkSpec,
    #[serde(default = "tile_grid")]
    pub merge: String,
    /// Optional target weights for tile placement, e.g. {"cuda":1,"hip":3}.
    /// Empty means equal round-robin.
    #[serde(default)]
    pub placement: BTreeMap<String, u32>,
    #[serde(default = "drop_oldest")]
    pub backpressure: String,
    #[serde(default = "exact")]
    pub determinism: String,
    #[serde(default)]
    pub receipt: Option<ReceiptSpec>,
}

fn tile_grid() -> String {
    "tile-grid".to_string()
}

fn drop_oldest() -> String {
    "drop-oldest".to_string()
}

fn exact() -> String {
    "exact".to_string()
}

impl StreamDescriptor {
    pub fn load(path: &Path) -> Result<(Self, String), String> {
        let bytes = fs::read(path)
            .map_err(|e| format!("read stream descriptor {}: {e}", path.display()))?;
        let descriptor: Self = serde_json::from_slice(&bytes)
            .map_err(|e| format!("parse stream descriptor {}: {e}", path.display()))?;
        descriptor.validate()?;
        Ok((descriptor, sha256_hex(&bytes)))
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.schema != STREAM_SCHEMA {
            return Err(format!(
                "stream schema {:?} != {STREAM_SCHEMA:?}",
                self.schema
            ));
        }
        if self.name.trim().is_empty() {
            return Err("stream needs a name".to_string());
        }
        if self.clock.fps == 0 || self.clock.deadline_ms == 0 {
            return Err("stream clock needs fps > 0 and deadline_ms > 0".to_string());
        }
        let expected = self.clock.deadline_ms;
        let derived = (1000 / self.clock.fps) as u64;
        if expected > derived {
            return Err(format!(
                "stream deadline {} ms exceeds one frame at {} fps ({} ms)",
                expected, self.clock.fps, derived
            ));
        }
        if self.source.width == 0 || self.source.height == 0 || self.source.channels != 3 {
            return Err("stream source needs width/height and 3 channels".to_string());
        }
        if self.source.format != "rgb24" {
            return Err(format!(
                "stream source format {:?} is not supported; use rgb24",
                self.source.format
            ));
        }
        match self.source.kind.as_str() {
            "frames" => {
                if self.source.dir.is_none() {
                    return Err("frames source needs dir".to_string());
                }
            }
            "camera" => {
                if self.source.device.is_none() {
                    return Err("camera source needs device".to_string());
                }
            }
            // Supplied by the caller (the DOOM RV32 source adapter); the
            // built-in source builder rejects it.
            "doom" => {}
            other => return Err(format!("unknown source kind {other:?}")),
        }
        if self.tiles.rows == 0 || self.tiles.cols == 0 || self.tiles.halo == 0 {
            return Err("tile grid needs rows, cols, halo > 0".to_string());
        }
        if self.source.width % self.tiles.cols != 0 || self.source.height % self.tiles.rows != 0 {
            return Err(format!(
                "source {}x{} is not divisible into {}x{} tiles",
                self.source.width, self.source.height, self.tiles.cols, self.tiles.rows
            ));
        }
        if self.tiles.bezel > 0 && self.merge != "tile-grid" {
            return Err("bezel is only supported for tile-grid".to_string());
        }
        if let Some(split) = &self.split {
            if self.merge != "columns" {
                return Err("a column split requires merge=columns".to_string());
            }
            if split.overlap != 1 {
                return Err("column split overlap must be 1 in this slice".to_string());
            }
            if split.left_cols == 0 || split.right_cols == 0 {
                return Err("column split needs positive left/right widths".to_string());
            }
            if split.left_cols + split.right_cols != self.source.width {
                return Err(format!(
                    "column split {} + {} != source width {}",
                    split.left_cols, split.right_cols, self.source.width
                ));
            }
        }
        if let Some(calibration) = &self.calibration {
            if self.split.is_none() || self.merge != "columns" {
                return Err("calibration requires a column split with merge=columns".to_string());
            }
            if calibration.probe_iters == 0 {
                return Err("calibration probe_iters must be > 0".to_string());
            }
            if calibration.min_cols == 0
                || calibration.max_cols < calibration.min_cols
                || calibration.max_cols >= self.source.width
            {
                return Err(format!(
                    "calibration bounds {}..={} are invalid for source width {}",
                    calibration.min_cols, calibration.max_cols, self.source.width
                ));
            }
        }
        if self.stage.targets.is_empty() {
            return Err("stream stage needs at least one target".to_string());
        }
        for (target, weight) in &self.placement {
            if !self.stage.targets.iter().any(|t| t == target) {
                return Err(format!(
                    "placement names target {target:?}, which is not in the stage targets"
                ));
            }
            if *weight == 0 {
                return Err(format!("placement weight for {target:?} must be > 0"));
            }
        }
        match self.sink.kind.as_str() {
            "frames" => {
                if self.sink.dir.is_none() {
                    return Err("frames sink needs dir".to_string());
                }
            }
            "tcp" => {
                if self.sink.listen.is_none() {
                    return Err("tcp sink needs listen".to_string());
                }
            }
            "null" => {}
            other => return Err(format!("unknown sink kind {other:?}")),
        }
        Ok(())
    }

    pub fn frame_budget_ms(&self) -> u64 {
        self.clock.deadline_ms
    }
}

pub fn sha256_hex(bytes: &[u8]) -> String {
    let mut hash = Sha256::new();
    hash.update(bytes);
    format!("{:x}", hash.finalize())
}

#[derive(Clone, Debug)]
pub struct Frame {
    pub index: u64,
    pub ts_ms: u64,
    pub width: usize,
    pub height: usize,
    pub channels: usize,
    pub data: Vec<u8>,
}

impl Frame {
    pub fn rgb(index: u64, width: usize, height: usize, data: Vec<u8>) -> Result<Self, String> {
        let expected = width
            .checked_mul(height)
            .and_then(|n| n.checked_mul(3))
            .ok_or("frame size overflows")?;
        if data.len() != expected {
            return Err(format!(
                "frame is {} bytes, expected {expected} for {width}x{height} RGB",
                data.len()
            ));
        }
        Ok(Self {
            index,
            ts_ms: now_ms(),
            width,
            height,
            channels: 3,
            data,
        })
    }

    pub fn sha256(&self) -> String {
        sha256_hex(&self.data)
    }
}

pub trait FrameSource {
    fn next_frame(&mut self, index: u64) -> Result<Option<Frame>, String>;
    fn describe(&self) -> Json;
}

/// Per-frame controller metadata handed to a sink.
///
/// This is reporting, not planning: it says what the controller measured, where
/// each tile ran, and why. A sink may render it (a HUD) or ignore it. The
/// default [`FrameSink::write_meta`] falls back to [`FrameSink::write`], so
/// frames-only and null sinks keep their byte-identical behavior.
#[derive(Clone, Debug, Serialize)]
pub struct FrameMeta {
    pub index: u64,
    pub work_ms: u64,
    pub budget_ms: u64,
    pub deadline_ok: bool,
    pub mode: String,
    pub p50_ms: u64,
    pub p95_ms: u64,
    pub p99_ms: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub split: Option<SplitMeta>,
    pub failovers: u64,
    pub recoveries: u64,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub failed_targets: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub recovered_targets: Vec<String>,
    pub note: String,
    pub tiles: Vec<TileMeta>,
}

#[derive(Clone, Debug, Serialize)]
pub struct SplitMeta {
    pub left_cols: usize,
    pub right_cols: usize,
    pub left_share: f64,
}

impl SplitMeta {
    fn from_split(split: &ColumnSplit) -> Self {
        let total = (split.left_cols + split.right_cols).max(1);
        Self {
            left_cols: split.left_cols,
            right_cols: split.right_cols,
            left_share: split.left_cols as f64 / total as f64,
        }
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct TileMeta {
    pub row: usize,
    pub col: usize,
    pub target: String,
    pub addr: String,
    pub planned_target: String,
    pub cols: usize,
    pub latency_ms: u64,
    pub worker_ms: Option<u64>,
    pub submit_call_ms: u64,
    pub status_ms: u64,
    pub result_ms: u64,
    pub attempts: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub failover_from: Option<String>,
    pub job_id: String,
    pub ok: bool,
}

pub trait FrameSink {
    fn write(&mut self, frame: &Frame) -> Result<(), String>;

    /// Write a frame together with its controller metadata. Sinks that do not
    /// care about the metadata keep the default, which is exactly `write`.
    fn write_meta(&mut self, frame: &Frame, meta: &FrameMeta) -> Result<(), String> {
        let _ = meta;
        self.write(frame)
    }

    fn describe(&self) -> Json;
}

pub struct FrameDirSource {
    files: Vec<PathBuf>,
    width: usize,
    height: usize,
    next: usize,
}

impl FrameDirSource {
    pub fn new(dir: &Path, width: usize, height: usize) -> Result<Self, String> {
        let entries =
            fs::read_dir(dir).map_err(|e| format!("read frame dir {}: {e}", dir.display()))?;
        let mut files = Vec::new();
        for entry in entries {
            let path = entry
                .map_err(|e| format!("read frame dir {}: {e}", dir.display()))?
                .path();
            if path.extension().and_then(|x| x.to_str()) == Some("rgb") {
                files.push(path);
            }
        }
        files.sort();
        if files.is_empty() {
            return Err(format!("frame dir {} has no .rgb frames", dir.display()));
        }
        let expected = width * height * 3;
        for file in &files {
            let len = fs::metadata(file)
                .map_err(|e| format!("stat {}: {e}", file.display()))?
                .len() as usize;
            if len != expected {
                return Err(format!(
                    "frame {} is {len} bytes, expected {expected}",
                    file.display()
                ));
            }
        }
        Ok(Self {
            files,
            width,
            height,
            next: 0,
        })
    }
}

impl FrameSource for FrameDirSource {
    fn next_frame(&mut self, index: u64) -> Result<Option<Frame>, String> {
        let Some(path) = self.files.get(self.next) else {
            return Ok(None);
        };
        self.next += 1;
        let data = fs::read(path).map_err(|e| format!("read {}: {e}", path.display()))?;
        Frame::rgb(index, self.width, self.height, data).map(Some)
    }

    fn describe(&self) -> Json {
        json!({"kind": "frames", "files": self.files.len(), "width": self.width, "height": self.height})
    }
}

pub struct CameraSource {
    child: Child,
    stdout: ChildStdout,
    width: usize,
    height: usize,
    device: String,
}

impl CameraSource {
    pub fn new(
        device: &str,
        width: usize,
        height: usize,
        fps: u32,
        capture_width: usize,
        capture_height: usize,
        capture_fps: u32,
    ) -> Result<Self, String> {
        let filter = format!("fps={fps},scale={width}:{height}");
        let mut child = Command::new("ffmpeg")
            .args([
                "-hide_banner",
                "-loglevel",
                "error",
                "-f",
                "v4l2",
                "-input_format",
                "mjpeg",
                "-video_size",
                &format!("{capture_width}x{capture_height}"),
                "-framerate",
                &capture_fps.to_string(),
                "-i",
                device,
                "-vf",
                &filter,
                "-pix_fmt",
                "rgb24",
                "-f",
                "rawvideo",
                "-",
            ])
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .map_err(|e| format!("spawn ffmpeg camera source: {e}"))?;
        let stdout = child.stdout.take().ok_or("camera source has no stdout")?;
        Ok(Self {
            child,
            stdout,
            width,
            height,
            device: device.to_string(),
        })
    }
}

impl FrameSource for CameraSource {
    fn next_frame(&mut self, index: u64) -> Result<Option<Frame>, String> {
        let len = self.width * self.height * 3;
        let mut data = vec![0u8; len];
        match self.stdout.read_exact(&mut data) {
            Ok(()) => Frame::rgb(index, self.width, self.height, data).map(Some),
            Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => Ok(None),
            Err(e) => Err(format!("read camera frame: {e}")),
        }
    }

    fn describe(&self) -> Json {
        json!({"kind": "camera", "device": self.device, "width": self.width, "height": self.height})
    }
}

impl Drop for CameraSource {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

pub struct FrameDirSink {
    dir: PathBuf,
    width: usize,
    height: usize,
    index: u64,
}

impl FrameDirSink {
    pub fn new(dir: &Path, width: usize, height: usize) -> Result<Self, String> {
        fs::create_dir_all(dir).map_err(|e| format!("create {}: {e}", dir.display()))?;
        Ok(Self {
            dir: dir.to_path_buf(),
            width,
            height,
            index: 0,
        })
    }
}

impl FrameSink for FrameDirSink {
    fn write(&mut self, frame: &Frame) -> Result<(), String> {
        if frame.width != self.width || frame.height != self.height {
            return Err(format!(
                "sink is {}x{}, frame is {}x{}",
                self.width, self.height, frame.width, frame.height
            ));
        }
        let path = self.dir.join(format!("frame_{:06}.rgb", self.index));
        self.index += 1;
        fs::write(&path, &frame.data).map_err(|e| format!("write {}: {e}", path.display()))
    }

    fn describe(&self) -> Json {
        json!({"kind": "frames", "dir": self.dir.display().to_string(), "width": self.width, "height": self.height})
    }
}

pub struct NullSink;

impl FrameSink for NullSink {
    fn write(&mut self, _frame: &Frame) -> Result<(), String> {
        Ok(())
    }

    fn describe(&self) -> Json {
        json!({"kind": "null"})
    }
}

pub struct TcpFrameSink {
    listener: TcpListener,
    stream: Option<TcpStream>,
    width: usize,
    height: usize,
    addr: String,
    hud: bool,
    fps: u32,
}

impl TcpFrameSink {
    pub fn new(
        addr: &str,
        width: usize,
        height: usize,
        hud: bool,
        fps: u32,
    ) -> Result<Self, String> {
        let listener = TcpListener::bind(addr).map_err(|e| format!("bind viewer {addr}: {e}"))?;
        listener
            .set_nonblocking(true)
            .map_err(|e| format!("set viewer nonblocking: {e}"))?;
        Ok(Self {
            listener,
            stream: None,
            width,
            height,
            addr: addr.to_string(),
            hud,
            fps,
        })
    }

    fn send(&mut self, header: &Json, body: &[u8]) -> Result<(), String> {
        if self.stream.is_none() {
            match self.listener.accept() {
                Ok((stream, _)) => {
                    let _ = stream.set_nodelay(true);
                    let _ = stream.set_write_timeout(Some(Duration::from_millis(500)));
                    self.stream = Some(stream);
                }
                Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => return Ok(()),
                Err(e) => return Err(format!("accept viewer: {e}")),
            }
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

impl FrameSink for TcpFrameSink {
    fn write(&mut self, frame: &Frame) -> Result<(), String> {
        let header = json!({
            "v": 1,
            "width": self.width,
            "height": self.height,
            "frame": frame.index,
            "total_ms": 0,
            "mode": "wall",
        });
        self.send(&header, &frame.data)
    }

    fn write_meta(&mut self, frame: &Frame, meta: &FrameMeta) -> Result<(), String> {
        if !self.hud {
            return self.write(frame);
        }
        let info = WallHudInfo {
            frame: meta.index,
            fps: self.fps,
            work_ms: meta.work_ms,
            budget_ms: meta.budget_ms,
            p50: meta.p50_ms,
            p95: meta.p95_ms,
            p99: meta.p99_ms,
            tiles: meta
                .tiles
                .iter()
                .map(|tile| WallTileInfo {
                    row: tile.row,
                    col: tile.col,
                    target: tile.target.clone(),
                    cols: tile.cols,
                    ms: tile.worker_ms.filter(|ms| *ms > 0).unwrap_or(tile.latency_ms),
                    attempts: tile.attempts,
                    failover_from: tile.failover_from.clone(),
                })
                .collect(),
            note: meta.note.clone(),
        };
        let body = render_wall(&frame.data, frame.width, frame.height, &info)?;
        let header = json!({
            "v": 1,
            "width": frame.width,
            "height": frame.height + HUD_TOP + HUD_BOTTOM,
            "frame": frame.index,
            "total_ms": meta.work_ms,
            "mode": meta.mode,
            "split": meta.split,
            "failovers": meta.failovers,
            "recoveries": meta.recoveries,
            "note": meta.note,
        });
        self.send(&header, &body)
    }

    fn describe(&self) -> Json {
        json!({"kind": "tcp", "listen": self.addr, "width": self.width, "height": self.height, "hud": self.hud})
    }
}

#[derive(Clone, Debug)]
pub struct TileInput {
    pub row: usize,
    pub col: usize,
    /// Padded input dimensions, including halo on both sides.
    pub width: usize,
    pub height: usize,
    pub data: Vec<u8>,
}

impl TileInput {
    pub fn output_width(&self) -> usize {
        self.width * 2
    }

    pub fn output_height(&self) -> usize {
        self.height * 2
    }
}

pub fn tile_frame(frame: &Frame, spec: &TileSpec) -> Result<Vec<TileInput>, String> {
    if frame.width % spec.cols != 0 || frame.height % spec.rows != 0 {
        return Err(format!(
            "frame {}x{} is not divisible into {}x{} tiles",
            frame.width, frame.height, spec.cols, spec.rows
        ));
    }
    let base_w = frame.width / spec.cols;
    let base_h = frame.height / spec.rows;
    let halo = spec.halo;
    let tile_w = base_w + halo * 2;
    let tile_h = base_h + halo * 2;
    let mut tiles = Vec::with_capacity(spec.rows * spec.cols);
    for row in 0..spec.rows {
        for col in 0..spec.cols {
            let mut data = Vec::with_capacity(tile_w * tile_h * 3);
            let x0 = (col * base_w) as isize;
            let y0 = (row * base_h) as isize;
            for y in 0..tile_h {
                let sy =
                    (y0 + y as isize - halo as isize).clamp(0, frame.height as isize - 1) as usize;
                for x in 0..tile_w {
                    let sx = (x0 + x as isize - halo as isize).clamp(0, frame.width as isize - 1)
                        as usize;
                    let base = (sy * frame.width + sx) * 3;
                    data.extend_from_slice(&frame.data[base..base + 3]);
                }
            }
            tiles.push(TileInput {
                row,
                col,
                width: tile_w,
                height: tile_h,
                data,
            });
        }
    }
    Ok(tiles)
}

pub fn column_tiles(frame: &Frame, split: &ColumnSplit) -> Result<Vec<TileInput>, String> {
    let left_w = split.left_cols + split.overlap;
    let right_w = split.right_cols + split.overlap;
    if left_w > frame.width || right_w > frame.width {
        return Err("column windows exceed frame width".to_string());
    }
    let mut tiles = Vec::with_capacity(2);
    for (col, x0, w) in [
        (0usize, 0usize, left_w),
        (1usize, frame.width - right_w, right_w),
    ] {
        let mut data = Vec::with_capacity(w * frame.height * 3);
        for y in 0..frame.height {
            let start = (y * frame.width + x0) * 3;
            data.extend_from_slice(&frame.data[start..start + w * 3]);
        }
        tiles.push(TileInput {
            row: 0,
            col,
            width: w,
            height: frame.height,
            data,
        });
    }
    Ok(tiles)
}

fn stitch_columns(split: &ColumnSplit, results: &[TileResult]) -> Result<Vec<u8>, String> {
    let left = results
        .iter()
        .find(|r| r.col == 0)
        .ok_or("column stream has no left tile")?;
    let right = results
        .iter()
        .find(|r| r.col == 1)
        .ok_or("column stream has no right tile")?;
    let split = crate::doom_pipeline::Split::new(split.left_cols, split.right_cols)?;
    split.stitch(&left.data, &right.data)
}

#[derive(Clone, Debug)]
pub struct GpuEndpoint {
    pub target: String,
    pub addr: String,
    pub capability: Json,
    pub token: Option<String>,
}

impl GpuEndpoint {
    pub fn new(target: &str, addr: &str, token: Option<String>) -> Self {
        Self {
            target: target.to_string(),
            addr: addr.to_string(),
            capability: json!({"kind": "gpu", "target": target, "addr": addr}),
            token,
        }
    }
}

pub struct StreamRunConfig {
    pub frame_limit: Option<u64>,
    /// Frames processed through the full tile path before the measured run.
    /// They are not written to the sink, receipt, or latency samples.
    pub warmup_frames: u64,
    pub all_first_target: bool,
    pub reference_dir: Option<PathBuf>,
    pub fleet_dir: Option<PathBuf>,
    pub lease_ms: u64,
    pub timeout_ms: u64,
    /// Per-request read/write deadline for submit/status/result. Keeps a
    /// blackholed target from holding a frame for the 60 s client default.
    pub request_timeout_ms: u64,
    /// After a target fails over, skip it for this many measured frames before
    /// probing it again. 0 disables quarantine.
    pub quarantine_frames: u64,
    /// Warm each tile shape on its fallback target before the measured loop.
    /// This pays compilation once, so a blackholed target fails over without a
    /// multi-second first-use compile.
    pub warm_fallbacks: bool,
}

impl Default for StreamRunConfig {
    fn default() -> Self {
        Self {
            frame_limit: None,
            warmup_frames: 0,
            all_first_target: false,
            reference_dir: None,
            fleet_dir: None,
            lease_ms: 3_000,
            timeout_ms: 10_000,
            request_timeout_ms: 1_000,
            quarantine_frames: 4,
            warm_fallbacks: false,
        }
    }
}

struct TileResult {
    row: usize,
    col: usize,
    target: String,
    addr: String,
    job_id: String,
    submit_ms: u64,
    submit_call_ms: u64,
    status_ms: u64,
    result_ms: u64,
    worker_ms: Option<u64>,
    done_ms: u64,
    sha256: String,
    bytes: usize,
    data: Vec<u8>,
}

fn run_tile_with_failover(
    planned: Arc<RemoteExecutor>,
    planned_target: String,
    planned_addr: String,
    fallbacks: Vec<(GpuEndpoint, Arc<RemoteExecutor>)>,
    listing: String,
    tile: TileInput,
    timeout_ms: u64,
    request_timeout_ms: u64,
    skip_planned: bool,
) -> Result<(TileResult, u32, Option<String>, Vec<Json>), String> {
    let mut errors = Vec::new();
    let mut attempts = 0u32;
    let request_timeout = Duration::from_millis(request_timeout_ms.max(1));
    if skip_planned {
        errors.push(json!({
            "target": planned_target,
            "addr": planned_addr,
            "error": "quarantined after an earlier failure",
        }));
    } else {
        attempts += 1;
        match run_tile(
            planned,
            planned_target.clone(),
            planned_addr.clone(),
            listing.clone(),
            tile.clone(),
            timeout_ms,
            request_timeout,
        ) {
            Ok(result) => return Ok((result, attempts, None, errors)),
            Err(e) => errors.push(json!({
                "target": planned_target,
                "addr": planned_addr,
                "error": e,
            })),
        }
    }
    for (fallback, executor) in fallbacks {
        attempts += 1;
        match run_tile(
            executor,
            fallback.target.clone(),
            fallback.addr.clone(),
            listing.clone(),
            tile.clone(),
            timeout_ms,
            request_timeout,
        ) {
            Ok(result) => {
                return Ok((result, attempts, Some(planned_target), errors));
            }
            Err(e) => errors.push(json!({
                "target": fallback.target,
                "addr": fallback.addr,
                "error": e,
            })),
        }
    }
    Err(format!(
        "all targets failed for tile {}:{}: {}",
        tile.row,
        tile.col,
        serde_json::to_string(&errors).unwrap_or_default()
    ))
}

fn build_binary_body(listing: &str, tile: &TileInput) -> Result<Vec<u8>, String> {
    let bytes = tile
        .width
        .checked_mul(tile.height)
        .and_then(|n| n.checked_mul(3))
        .and_then(|n| n.checked_mul(4))
        .ok_or("tile payload size overflows")?;
    let meta = json!({
        "listing": listing,
        "payload": {
            "streams": {
                "X": {
                    "kind": "F32",
                    "shape": [tile.height, tile.width, 3],
                    "off": 0,
                    "bytes": bytes,
                }
            }
        }
    });
    let mut body = serde_json::to_vec(&meta).map_err(|e| e.to_string())?;
    body.push(b'\n');
    body.reserve(bytes);
    for value in &tile.data {
        body.extend_from_slice(&(*value as f32).to_le_bytes());
    }
    Ok(body)
}

fn run_tile(
    executor: Arc<RemoteExecutor>,
    target: String,
    addr: String,
    listing: String,
    tile: TileInput,
    timeout_ms: u64,
    request_timeout: Duration,
) -> Result<TileResult, String> {
    let submit_ms = now_ms();
    let body = build_binary_body(&listing, &tile)?;
    let header = json!({
        "type": "submit",
        "class": "interactive",
        "outputs": ["OUT"],
        "frames": 1,
        "output_format": "binary",
        "payload_format": "binary",
        "output_u8": true,
    });
    let submit_started = Instant::now();
    let mut submitted = None;
    for attempt in 0..2 {
        match executor.submit_with_timeout(&header, &body, request_timeout) {
            Ok(value) => {
                submitted = Some(value);
                break;
            }
            Err((code, msg))
                if attempt == 0
                    && (code == "job-unavailable" || code == "job-invalid")
                    && !msg.contains("timed out")
                    && !msg.contains("os error 11")
                    && !msg.contains("WouldBlock") =>
            {
                // A persistent session can be reset between frames. Retrying a
                // submit can duplicate a job whose ack was lost; the first
                // result is ignored, so correctness is unchanged. A request
                // timeout is already the bounded blackhole signal: fail over
                // instead of paying the deadline twice.
                continue;
            }
            Err((code, msg)) => {
                return Err(format!("submit {addr}: {code}: {msg}"));
            }
        }
    }
    let submitted = submitted.ok_or_else(|| format!("submit {addr} failed after retry"))?;
    let submit_call_ms = submit_started.elapsed().as_millis() as u64;
    let job_id = submitted
        .get("job_id")
        .and_then(|x| x.as_str())
        .ok_or_else(|| format!("submit returned no job_id: {submitted}"))?
        .to_string();
    let deadline = Instant::now() + Duration::from_millis(timeout_ms);
    let status_started = Instant::now();
    loop {
        let status = executor
            .status_with_timeout(&job_id, request_timeout)
            .map_err(|(code, msg)| format!("status {addr}: {code}: {msg}"))?;
        let state = status
            .get("job")
            .and_then(|j| j.get("state"))
            .and_then(|x| x.as_str())
            .unwrap_or("");
        match state {
            "done" => break,
            "failed" | "canceled" => {
                return Err(format!("job {job_id} ended in {state}: {status}"));
            }
            _ if Instant::now() >= deadline => {
                let _ = executor.cancel(&job_id);
                return Err(format!("job {job_id} timed out after {timeout_ms} ms"));
            }
            _ => thread::sleep(Duration::from_millis(1)),
        }
    }
    let status_ms = status_started.elapsed().as_millis() as u64;
    let result_started = Instant::now();
    let (header, body) = executor
        .result_with_timeout(&job_id, request_timeout)
        .map_err(|(code, msg)| format!("result {addr}: {code}: {msg}"))?;
    let result_ms = result_started.elapsed().as_millis() as u64;
    validate_binary_result(&header, &body)?;
    let out = header
        .get("outputs")
        .and_then(|x| x.as_array())
        .and_then(|a| a.first())
        .ok_or("result has no output descriptor")?;
    let shape = out
        .get("shape")
        .and_then(|x| x.as_array())
        .ok_or("result output has no shape")?;
    let got: Vec<u64> = shape
        .iter()
        .map(|x| x.as_u64())
        .collect::<Option<Vec<_>>>()
        .ok_or("result output shape is not integer")?;
    let expected = [tile.output_height() as u64, tile.output_width() as u64, 3];
    if got != expected {
        return Err(format!("tile output shape {got:?}, expected {expected:?}"));
    }
    let bytes = out.get("bytes").and_then(|x| x.as_u64()).unwrap_or(0) as usize;
    let worker_ms = header
        .get("frame_ms")
        .and_then(|x| x.as_u64())
        .or_else(|| header.get("run_ms").and_then(|x| x.as_u64()));
    Ok(TileResult {
        row: tile.row,
        col: tile.col,
        target,
        addr,
        job_id,
        submit_ms,
        submit_call_ms,
        status_ms,
        result_ms,
        worker_ms,
        done_ms: now_ms(),
        sha256: sha256_hex(&body),
        bytes,
        data: body,
    })
}

/// Controller-side column split calibration.
///
/// The policy is deliberately outside the executor: it only measures the tiles
/// the controller already knows how to submit. It runs on the first real source
/// frame, before the measured loop, and never changes the frame sequence.
struct SplitPolicy {
    probe_iters: u32,
    min_cols: usize,
    max_cols: usize,
    rebalance: bool,
}

struct CalibrationOutcome {
    split: ColumnSplit,
    report: Json,
    ok: bool,
}

impl SplitPolicy {
    fn new(spec: &CalibrationSpec) -> Self {
        Self {
            probe_iters: spec.probe_iters.max(1),
            min_cols: spec.min_cols.max(1),
            max_cols: spec.max_cols.max(1),
            rebalance: spec.rebalance,
        }
    }

    /// One probe iteration: run both column sides concurrently and return
    /// `(left_elapsed_ms, right_elapsed_ms, left_worker_ms, right_worker_ms)`.
    fn probe_once(
        &self,
        frame: &Frame,
        split: &ColumnSplit,
        planned_targets: &[usize],
        pools: &[Vec<Arc<RemoteExecutor>>],
        endpoints: &[GpuEndpoint],
        listing: &str,
        timeout_ms: u64,
        request_timeout: Duration,
    ) -> Result<(u64, u64, Option<u64>, Option<u64>), String> {
        let tiles = column_tiles(frame, split)?;
        if tiles.len() != 2 {
            return Err("column calibration needs exactly two tiles".to_string());
        }
        let mut used = vec![0usize; endpoints.len()];
        let mut sides = Vec::with_capacity(2);
        for (i, tile) in tiles.into_iter().enumerate() {
            let planned = *planned_targets
                .get(i)
                .ok_or_else(|| format!("calibration side {i} has no planned target"))?;
            let slot = used[planned];
            used[planned] += 1;
            let executor = pools
                .get(planned)
                .and_then(|pool| pool.get(slot))
                .cloned()
                .ok_or_else(|| format!("calibration has no executor slot {slot} for target {planned}"))?;
            let endpoint = endpoints
                .get(planned)
                .ok_or_else(|| format!("calibration target {planned} has no endpoint"))?;
            sides.push((
                executor,
                endpoint.target.clone(),
                endpoint.addr.clone(),
                tile,
            ));
        }
        let run = |(executor, target, addr, tile): (
            Arc<RemoteExecutor>,
            String,
            String,
            TileInput,
        )| {
            let started = Instant::now();
            run_tile(
                executor,
                target,
                addr,
                listing.to_string(),
                tile,
                timeout_ms,
                request_timeout,
            )
            .map(|result| (started.elapsed().as_millis() as u64, result.worker_ms))
        };
        let left_side = sides.remove(0);
        let right_side = sides.pop().expect("two sides");
        let (left, right) = thread::scope(|scope| {
            let left = scope.spawn(|| run(left_side));
            let right = scope.spawn(|| run(right_side));
            (left.join(), right.join())
        });
        let left = left.map_err(|_| "calibration left probe panicked".to_string())??;
        let right = right.map_err(|_| "calibration right probe panicked".to_string())??;
        Ok((left.0, right.0, left.1, right.1))
    }

    fn probe_n(
        &self,
        frame: &Frame,
        split: &ColumnSplit,
        planned_targets: &[usize],
        pools: &[Vec<Arc<RemoteExecutor>>],
        endpoints: &[GpuEndpoint],
        listing: &str,
        timeout_ms: u64,
        request_timeout: Duration,
    ) -> Result<(Vec<u64>, Vec<u64>), String> {
        let mut left = Vec::with_capacity(self.probe_iters as usize);
        let mut right = Vec::with_capacity(self.probe_iters as usize);
        for _ in 0..self.probe_iters {
            let (l, r, _, _) = self.probe_once(
                frame,
                split,
                planned_targets,
                pools,
                endpoints,
                listing,
                timeout_ms,
                request_timeout,
            )?;
            left.push(l);
            right.push(r);
        }
        Ok((left, right))
    }

    fn warm_once(
        &self,
        frame: &Frame,
        split: &ColumnSplit,
        planned_targets: &[usize],
        pools: &[Vec<Arc<RemoteExecutor>>],
        endpoints: &[GpuEndpoint],
        listing: &str,
        timeout_ms: u64,
        request_timeout: Duration,
    ) -> Result<Json, String> {
        let (l, r, lw, rw) = self.probe_once(
            frame,
            split,
            planned_targets,
            pools,
            endpoints,
            listing,
            timeout_ms,
            request_timeout,
        )?;
        Ok(json!({
            "left_ms": l,
            "right_ms": r,
            "left_worker_ms": lw,
            "right_worker_ms": rw,
        }))
    }

    /// Calibrate on `frame`. A probe failure aborts calibration and keeps the
    /// declared split; per-frame failover still runs.
    fn calibrate(
        &self,
        frame: &Frame,
        declared: &ColumnSplit,
        planned_targets: &[usize],
        pools: &[Vec<Arc<RemoteExecutor>>],
        endpoints: &[GpuEndpoint],
        listing: &str,
        timeout_ms: u64,
        request_timeout_ms: u64,
    ) -> CalibrationOutcome {
        let request_timeout = Duration::from_millis(request_timeout_ms.max(1));
        let width = frame.width;
        let overlap = declared.overlap;
        let even = ColumnSplit {
            left_cols: width / 2,
            right_cols: width - width / 2,
            overlap,
        };
        let targets: Vec<String> = planned_targets
            .iter()
            .filter_map(|idx| endpoints.get(*idx).map(|e| e.target.clone()))
            .collect();
        let mut report = json!({
            "ok": false,
            "policy": {
                "probe_iters": self.probe_iters,
                "min_cols": self.min_cols,
                "max_cols": self.max_cols,
                "rebalance": self.rebalance,
            },
            "declared": SplitMeta::from_split(declared),
            "source_width": width,
            "targets": targets,
        });
        let abort = |report: &mut Json, stage: &str, error: String| CalibrationOutcome {
            split: declared.clone(),
            report: {
                report["stage"] = json!(stage);
                report["error"] = json!(error);
                report["ok"] = json!(false);
                report.clone()
            },
            ok: false,
        };

        if let Err(e) = self.warm_once(
            frame,
            &even,
            planned_targets,
            pools,
            endpoints,
            listing,
            timeout_ms,
            request_timeout,
        ) {
            return abort(&mut report, "warm-even", e);
        }

        let (left_samples, right_samples) = match self.probe_n(
            frame,
            &even,
            planned_targets,
            pools,
            endpoints,
            listing,
            timeout_ms,
            request_timeout,
        ) {
            Ok(samples) => samples,
            Err(e) => return abort(&mut report, "probe", e),
        };
        let left_p50 = percentile(&left_samples, 50);
        let right_p50 = percentile(&right_samples, 50);
        let (left_cols, right_cols) = choose_columns(
            width,
            left_p50,
            right_p50,
            self.min_cols,
            self.max_cols,
        );
        let mut split = ColumnSplit {
            left_cols,
            right_cols,
            overlap,
        };
        report["probe"] = json!({
            "split": SplitMeta::from_split(&even),
            "samples": {"left": left_samples, "right": right_samples},
            "p50_ms": {"left": left_p50, "right": right_p50},
        });

        let split_warmup = match self.warm_once(
            frame,
            &split,
            planned_targets,
            pools,
            endpoints,
            listing,
            timeout_ms,
            request_timeout,
        ) {
            Ok(warm) => warm,
            Err(e) => return abort(&mut report, "warm-chosen", e),
        };
        report["split_warmup"] = split_warmup;

        let mut rebalance = Json::Null;
        if self.rebalance {
            match self.probe_n(
                frame,
                &split,
                planned_targets,
                pools,
                endpoints,
                listing,
                timeout_ms,
                request_timeout,
            ) {
                Ok((left_samples, right_samples)) => {
                    let left_p50 = percentile(&left_samples, 50);
                    let right_p50 = percentile(&right_samples, 50);
                    let half = (width / 2) as u64;
                    let left_norm = left_p50 * half / split.left_cols.max(1) as u64;
                    let right_norm = right_p50 * half / split.right_cols.max(1) as u64;
                    let (next_left, next_right) = choose_columns(
                        width,
                        left_norm,
                        right_norm,
                        self.min_cols,
                        self.max_cols,
                    );
                    let before = SplitMeta::from_split(&split);
                    let mut changed = next_left.abs_diff(split.left_cols) >= 4;
                    let mut warmup = Json::Null;
                    if changed {
                        let next = ColumnSplit {
                            left_cols: next_left,
                            right_cols: next_right,
                            overlap,
                        };
                        match self.warm_once(
                            frame,
                            &next,
                            planned_targets,
                            pools,
                            endpoints,
                            listing,
                            timeout_ms,
                            request_timeout,
                        ) {
                            Ok(warm) => {
                                warmup = warm;
                                split = next;
                            }
                            Err(e) => {
                                changed = false;
                                warmup = json!({"error": e});
                            }
                        }
                    }
                    rebalance = json!({
                        "before": before,
                        "after": SplitMeta::from_split(&split),
                        "samples": {"left": left_samples, "right": right_samples},
                        "p50_ms": {"left": left_p50, "right": right_p50},
                        "normalized_ms": {"left": left_norm, "right": right_norm},
                        "changed": changed,
                        "warmup": warmup,
                    });
                }
                Err(e) => {
                    rebalance = json!({"error": e});
                }
            }
        }
        report["rebalance"] = rebalance;
        report["chosen"] = json!(SplitMeta::from_split(&split));
        report["ok"] = json!(true);
        CalibrationOutcome {
            split,
            report: report.clone(),
            ok: true,
        }
    }
}

fn composite(
    frame_width: usize,
    frame_height: usize,
    spec: &TileSpec,
    results: &[TileResult],
) -> Result<Vec<u8>, String> {
    let base_w = frame_width / spec.cols;
    let base_h = frame_height / spec.rows;
    let out_w = base_w * 2;
    let out_h = base_h * 2;
    let halo_out = spec.halo * 2;
    let tile_w = out_w + halo_out * 2;
    let mut canvas = vec![0u8; out_w * spec.cols * out_h * spec.rows * 3];
    for result in results {
        let expected = tile_w * (out_h + halo_out * 2) * 3;
        if result.data.len() != expected {
            return Err(format!(
                "tile body is {} bytes, expected {expected}",
                result.data.len()
            ));
        }
        for y in 0..out_h {
            for x in 0..out_w {
                for c in 0..3 {
                    let src = ((y + halo_out) * tile_w + (x + halo_out)) * 3 + c;
                    let dst_x = result.col * out_w + x;
                    let dst_y = result.row * out_h + y;
                    canvas[(dst_y * out_w * spec.cols + dst_x) * 3 + c] = result.data[src];
                }
            }
        }
    }
    Ok(canvas)
}

pub struct StreamSummary {
    pub frames: u64,
    pub budget_ms: u64,
    pub p50: u64,
    pub p95: u64,
    pub p99: u64,
    pub max: u64,
    pub over_budget: u64,
    pub matches: Option<u64>,
    pub mismatches: Option<u64>,
}

fn validate_partition_claim(claim: &Json, descriptor: &StreamDescriptor) -> Result<(), String> {
    if claim.is_null() {
        return Ok(());
    }
    let axes = claim
        .get("axes")
        .and_then(|x| x.as_array())
        .ok_or("stage descriptor partitionable needs axes[]")?;
    if let Some(split) = &descriptor.split {
        for axis in axes {
            let name = axis
                .get("axis")
                .and_then(|x| x.as_str())
                .ok_or("partition axis needs a name")?;
            let min = axis
                .get("min")
                .and_then(|x| x.as_u64())
                .ok_or("partition axis needs min")?;
            let max = axis
                .get("max")
                .and_then(|x| x.as_u64())
                .ok_or("partition axis needs max")?;
            let overlap = axis.get("overlap").and_then(|x| x.as_u64()).unwrap_or(0);
            let values: &[usize] = match name {
                "rows" => &[descriptor.source.height],
                "columns" => &[split.left_cols, split.right_cols],
                _ => continue,
            };
            for value in values {
                if (*value as u64) < min || (*value as u64) > max {
                    return Err(format!(
                        "partition {name} size {value} is outside the compiler claim {min}..={max}"
                    ));
                }
            }
            if (split.overlap as u64) < overlap {
                return Err(format!(
                    "partition {name} needs overlap {overlap}, but the split overlap is {}",
                    split.overlap
                ));
            }
        }
        return Ok(());
    }
    for axis in axes {
        let name = axis
            .get("axis")
            .and_then(|x| x.as_str())
            .ok_or("partition axis needs a name")?;
        let min = axis
            .get("min")
            .and_then(|x| x.as_u64())
            .ok_or("partition axis needs min")?;
        let max = axis
            .get("max")
            .and_then(|x| x.as_u64())
            .ok_or("partition axis needs max")?;
        let overlap = axis.get("overlap").and_then(|x| x.as_u64()).unwrap_or(0);
        let base = match name {
            "rows" => descriptor.source.height / descriptor.tiles.rows,
            "columns" => descriptor.source.width / descriptor.tiles.cols,
            _ => continue,
        } as u64;
        if base < min || base > max {
            return Err(format!(
                "partition {name} size {base} is outside the compiler claim {min}..={max}"
            ));
        }
        if (descriptor.tiles.halo as u64) < overlap {
            return Err(format!(
                "partition {name} needs overlap {overlap}, but the tile halo is {}",
                descriptor.tiles.halo
            ));
        }
    }
    Ok(())
}

fn stage_claim(
    path: &Path,
    listing_sha256: &str,
    descriptor: &StreamDescriptor,
) -> Result<Json, String> {
    let bytes =
        fs::read(path).map_err(|e| format!("read stage descriptor {}: {e}", path.display()))?;
    let value: Json = serde_json::from_slice(&bytes)
        .map_err(|e| format!("parse stage descriptor {}: {e}", path.display()))?;
    if value.get("schema").and_then(|x| x.as_str()) != Some("jotnar.stage/1") {
        return Err(format!(
            "stage descriptor {} is not jotnar.stage/1",
            path.display()
        ));
    }
    let declared = value
        .get("listing_sha256")
        .and_then(|x| x.as_str())
        .unwrap_or("");
    if declared != listing_sha256 {
        return Err(format!(
            "stage descriptor was compiled from listing {declared}, stream uses {listing_sha256}"
        ));
    }
    let targets = value
        .get("targets")
        .and_then(|x| x.as_array())
        .map(|a| {
            a.iter()
                .filter_map(|t| t.as_str().map(str::to_string))
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    if !targets.is_empty()
        && !targets
            .iter()
            .all(|t| descriptor.stage.targets.iter().any(|s| s == t))
    {
        return Err(format!(
            "stage descriptor targets {targets:?} are not a subset of the stream targets {:?}",
            descriptor.stage.targets
        ));
    }
    let partitionable = value.get("partitionable").cloned().unwrap_or(Json::Null);
    validate_partition_claim(&partitionable, descriptor)?;
    Ok(json!({
        "path": path.display().to_string(),
        "sha256": sha256_hex(&bytes),
        "targets": targets,
        "partitionable": partitionable,
    }))
}

pub fn run_stream(
    descriptor: &StreamDescriptor,
    descriptor_sha256: &str,
    endpoints: &[GpuEndpoint],
    cfg: &StreamRunConfig,
) -> Result<Json, String> {
    let (source, sink) = default_adapters(descriptor)?;
    run_stream_with(descriptor, descriptor_sha256, endpoints, cfg, source, sink)
}

fn default_adapters(
    descriptor: &StreamDescriptor,
) -> Result<(Box<dyn FrameSource>, Box<dyn FrameSink>), String> {
    let wall_w = descriptor.source.width * 2;
    let wall_h = descriptor.source.height * 2;
    let source: Box<dyn FrameSource> = match descriptor.source.kind.as_str() {
        "frames" => Box::new(FrameDirSource::new(
            descriptor
                .source
                .dir
                .as_deref()
                .ok_or("frames source needs dir")?,
            descriptor.source.width,
            descriptor.source.height,
        )?),
        "camera" => Box::new(CameraSource::new(
            descriptor
                .source
                .device
                .as_deref()
                .ok_or("camera source needs device")?,
            descriptor.source.width,
            descriptor.source.height,
            descriptor.clock.fps,
            descriptor.source.capture_width.unwrap_or(1280),
            descriptor.source.capture_height.unwrap_or(720),
            descriptor.source.capture_fps.unwrap_or(30),
        )?),
        other => return Err(format!("unknown source kind {other:?}")),
    };
    let sink: Box<dyn FrameSink> = match descriptor.sink.kind.as_str() {
        "frames" => Box::new(FrameDirSink::new(
            descriptor
                .sink
                .dir
                .as_deref()
                .ok_or("frames sink needs dir")?,
            wall_w,
            wall_h,
        )?),
        "tcp" => Box::new(TcpFrameSink::new(
            descriptor
                .sink
                .listen
                .as_deref()
                .ok_or("tcp sink needs listen")?,
            wall_w,
            wall_h,
            descriptor.sink.hud,
            descriptor.clock.fps,
        )?),
        "null" => Box::new(NullSink),
        other => return Err(format!("unknown sink kind {other:?}")),
    };
    Ok((source, sink))
}

pub fn run_stream_with(
    descriptor: &StreamDescriptor,
    descriptor_sha256: &str,
    endpoints: &[GpuEndpoint],
    cfg: &StreamRunConfig,
    mut source: Box<dyn FrameSource>,
    mut sink: Box<dyn FrameSink>,
) -> Result<Json, String> {
    descriptor.validate()?;
    if endpoints.is_empty() {
        return Err("stream needs at least one GPU endpoint".to_string());
    }
    let mut placement = Vec::new();
    if descriptor.placement.is_empty() {
        for i in 0..endpoints.len() {
            placement.push(i);
        }
    } else {
        for (target, weight) in &descriptor.placement {
            let idx = endpoints
                .iter()
                .position(|e| &e.target == target)
                .ok_or_else(|| format!("placement target {target:?} has no endpoint"))?;
            for _ in 0..*weight {
                placement.push(idx);
            }
        }
    }
    if placement.is_empty() {
        return Err("stream placement is empty".to_string());
    }
    let tile_count = if descriptor.split.is_some() {
        2
    } else {
        descriptor.tiles.rows * descriptor.tiles.cols
    };
    let planned_targets: Vec<usize> = if cfg.all_first_target {
        vec![0; tile_count]
    } else {
        (0..tile_count)
            .map(|i| placement[i % placement.len()])
            .collect()
    };
    let mut pools: Vec<Vec<Arc<RemoteExecutor>>> = Vec::with_capacity(endpoints.len());
    for (idx, endpoint) in endpoints.iter().enumerate() {
        let count = planned_targets
            .iter()
            .filter(|target| **target == idx)
            .count()
            .max(1);
        pools.push(
            (0..count)
                .map(|_| {
                    Arc::new(RemoteExecutor::new(
                        endpoint.capability.clone(),
                        endpoint.addr.clone(),
                        endpoint.token.clone(),
                    ))
                })
                .collect(),
        );
    }
    // One persistent session per endpoint for fallback attempts. Reusing it
    // keeps the first failover from paying a new handshake every frame.
    let fallback_pools: Vec<Arc<RemoteExecutor>> = endpoints
        .iter()
        .map(|endpoint| {
            Arc::new(RemoteExecutor::new(
                endpoint.capability.clone(),
                endpoint.addr.clone(),
                endpoint.token.clone(),
            ))
        })
        .collect();
    let listing = fs::read_to_string(&descriptor.stage.listing)
        .map_err(|e| format!("read listing {}: {e}", descriptor.stage.listing.display()))?;
    let listing_sha256 = sha256_hex(listing.as_bytes());
    let claim = match &descriptor.stage.descriptor {
        Some(path) => stage_claim(path, &listing_sha256, descriptor)?,
        None => Json::Null,
    };
    // The tiled stage is a 2x upscale per tile, so the wall is 2x the source
    // in each dimension.
    let wall_w = descriptor.source.width * 2;
    let wall_h = descriptor.source.height * 2;
    let receipt_dir = descriptor
        .receipt
        .as_ref()
        .and_then(|r| r.dir.clone())
        .unwrap_or_else(|| {
            descriptor
                .sink
                .dir
                .clone()
                .unwrap_or_else(|| PathBuf::from("."))
        });
    fs::create_dir_all(&receipt_dir)
        .map_err(|e| format!("create receipt dir {}: {e}", receipt_dir.display()))?;
    let receipt_path = receipt_dir.join("frames.jsonl");
    let mut receipt = fs::File::create(&receipt_path)
        .map_err(|e| format!("create {}: {e}", receipt_path.display()))?;

    let mut leases: Vec<LeaseGuard> = Vec::new();
    if let Some(dir) = &cfg.fleet_dir {
        let fleet = Fleet::new(
            dir,
            format!("jotnar-stream:{}", std::process::id()),
            1,
            1,
            cfg.lease_ms,
        );
        for endpoint in endpoints {
            leases.push(fleet.register_lease(
                "interactive",
                &endpoint.target,
                1,
                cfg.lease_ms,
                Some("jotnar-stream interactive wall"),
            )?);
        }
    }

    let frame_budget = descriptor.clock.deadline_ms;
    let period = Duration::from_millis((1000 / descriptor.clock.fps.max(1)) as u64);
    let limit = cfg.frame_limit.unwrap_or(u64::MAX);
    let mut work_ms = Vec::new();
    let mut matches = 0u64;
    let mut mismatches = 0u64;
    let mut failover_count = 0u64;
    let mut recovery_count = 0u64;
    let mut source_index = 0u64;
    let mut measured = 0u64;
    let started = Instant::now();
    let mut paused = Duration::ZERO;
    let split_policy = descriptor.calibration.as_ref().map(SplitPolicy::new);
    let mut active_split = descriptor.split.clone();
    let mut calibration_pending = split_policy.is_some();
    let mut calibration = Json::Null;
    let mut prev_failed: BTreeSet<String> = BTreeSet::new();
    let mut down_until: BTreeMap<String, u64> = BTreeMap::new();
    let mut warm_fallbacks_pending =
        cfg.warm_fallbacks && descriptor.split.is_some() && endpoints.len() > 1;
    let tile_cols = |split: &Option<ColumnSplit>, col: usize| -> usize {
        match split {
            Some(split) if col == 0 => split.left_cols,
            Some(split) => split.right_cols,
            None => descriptor.source.width / descriptor.tiles.cols,
        }
    };
    let result_index = |split: &Option<ColumnSplit>, row: usize, col: usize| -> usize {
        if split.is_some() {
            col
        } else {
            row * descriptor.tiles.cols + col
        }
    };
    loop {
        let mut frame_start = Instant::now();
        for lease in &mut leases {
            lease.renew(cfg.lease_ms)?;
        }
        let Some(frame) = source.next_frame(source_index)? else {
            break;
        };
        source_index += 1;
        if calibration_pending {
            if let Some(policy) = split_policy.as_ref() {
                if let Some(declared) = active_split.clone() {
                    let outcome = policy.calibrate(
                        &frame,
                        &declared,
                        &planned_targets,
                        &pools,
                        endpoints,
                        &listing,
                        cfg.timeout_ms,
                        cfg.request_timeout_ms,
                    );
                    calibration = outcome.report;
                    if outcome.ok {
                        // The calibrated split is still a partition claim; a
                        // policy must not pick a shape the compiler rejected.
                        let mut candidate = descriptor.clone();
                        candidate.split = Some(outcome.split.clone());
                        let partitionable = claim
                            .get("partitionable")
                            .cloned()
                            .unwrap_or(Json::Null);
                        match validate_partition_claim(&partitionable, &candidate) {
                            Ok(()) => active_split = Some(outcome.split),
                            Err(e) => {
                                calibration["ok"] = json!(false);
                                calibration["stage"] = json!("claim");
                                calibration["error"] = json!(e);
                            }
                        }
                    }
                }
            }
            calibration_pending = false;
            // Calibration is setup, not part of this frame's pipeline work.
            frame_start = Instant::now();
            for lease in &mut leases {
                lease.renew(cfg.lease_ms)?;
            }
        }
        if warm_fallbacks_pending {
            let warm_tiles = if let Some(split) = &active_split {
                column_tiles(&frame, split)?
            } else {
                tile_frame(&frame, &descriptor.tiles)?
            };
            let request_timeout =
                Duration::from_millis(cfg.request_timeout_ms.max(1));
            for (i, tile) in warm_tiles.into_iter().enumerate() {
                let planned = *planned_targets
                    .get(i)
                    .ok_or_else(|| format!("tile {i} has no planned target"))?;
                for (idx, endpoint) in endpoints.iter().enumerate() {
                    if idx == planned {
                        continue;
                    }
                    let executor = fallback_pools[idx].clone();
                    // A cold fallback compiles here, before the measured loop.
                    let _ = run_tile(
                        executor,
                        endpoint.target.clone(),
                        endpoint.addr.clone(),
                        listing.clone(),
                        tile.clone(),
                        cfg.timeout_ms,
                        request_timeout,
                    );
                }
            }
            warm_fallbacks_pending = false;
            frame_start = Instant::now();
            for lease in &mut leases {
                lease.renew(cfg.lease_ms)?;
            }
        }
        let tiles = if let Some(split) = &active_split {
            column_tiles(&frame, split)?
        } else {
            tile_frame(&frame, &descriptor.tiles)?
        };
        let mut results = Vec::with_capacity(tiles.len());
        let mut tile_meta: BTreeMap<(usize, usize), (u32, Option<String>, Vec<Json>)> =
            BTreeMap::new();
        let mut handles = Vec::with_capacity(tiles.len());
        let mut used = vec![0usize; endpoints.len()];
        for (i, tile) in tiles.into_iter().enumerate() {
            let planned = *planned_targets
                .get(i)
                .ok_or_else(|| format!("tile {i} has no planned target"))?;
            let slot = used[planned];
            used[planned] += 1;
            let executor = pools
                .get(planned)
                .and_then(|pool| pool.get(slot))
                .cloned()
                .ok_or_else(|| format!("no executor slot {slot} for target {planned}"))?;
            let target = endpoints[planned].target.clone();
            let addr = endpoints[planned].addr.clone();
            let skip_planned = cfg.quarantine_frames > 0
                && down_until
                    .get(&target)
                    .is_some_and(|until| measured < *until);
            let fallbacks: Vec<(GpuEndpoint, Arc<RemoteExecutor>)> = endpoints
                .iter()
                .enumerate()
                .filter(|(idx, _)| *idx != planned)
                .map(|(idx, endpoint)| (endpoint.clone(), fallback_pools[idx].clone()))
                .collect();
            let listing = listing.clone();
            let timeout = cfg.timeout_ms;
            let request_timeout = cfg.request_timeout_ms;
            handles.push(thread::spawn(move || {
                run_tile_with_failover(
                    executor,
                    target,
                    addr,
                    fallbacks,
                    listing,
                    tile,
                    timeout,
                    request_timeout,
                    skip_planned,
                )
            }));
        }
        for handle in handles {
            match handle.join() {
                Ok(Ok((result, attempts, failover_from, errors))) => {
                    tile_meta.insert((result.row, result.col), (attempts, failover_from, errors));
                    results.push(result);
                }
                Ok(Err(e)) => return Err(e),
                Err(_) => return Err("tile thread panicked".to_string()),
            }
        }
        results.sort_by_key(|r| (r.row, r.col));
        let composited = if let Some(split) = &active_split {
            stitch_columns(split, &results)?
        } else {
            composite(
                descriptor.source.width,
                descriptor.source.height,
                &descriptor.tiles,
                &results,
            )?
        };
        if source_index <= cfg.warmup_frames {
            continue;
        }
        // A planned target that failed over this frame; a target that failed
        // last frame but not now has rejoined.
        let failed: BTreeSet<String> = tile_meta
            .values()
            .filter_map(|meta| meta.1.clone())
            .collect();
        let recovered: Vec<String> = prev_failed.difference(&failed).cloned().collect();
        prev_failed = failed.clone();
        if cfg.quarantine_frames > 0 {
            // Only a real failed attempt extends the quarantine. A skipped
            // frame also carries `failover_from`, so keying off that alone
            // would keep the target quarantined forever.
            for (attempts, failover_from, _) in tile_meta.values() {
                if *attempts > 1 {
                    if let Some(target) = failover_from {
                        down_until.insert(target.clone(), measured + cfg.quarantine_frames);
                    }
                }
            }
            for target in &recovered {
                down_until.remove(target);
            }
        }
        let frame_failovers = tile_meta
            .values()
            .filter(|m| m.1.is_some())
            .count() as u64;
        failover_count += frame_failovers;
        recovery_count += recovered.len() as u64;

        let work = frame_start.elapsed().as_millis() as u64;
        work_ms.push(work);
        let p50_ms = percentile(&work_ms, 50);
        let p95_ms = percentile(&work_ms, 95);
        let p99_ms = percentile(&work_ms, 99);
        let split_meta = active_split.as_ref().map(SplitMeta::from_split);
        let failed_targets: Vec<String> = failed.iter().cloned().collect();
        let mut note_parts = Vec::new();
        if !recovered.is_empty() {
            note_parts.push(format!("recovered {}", recovered.join("+")));
        }
        if !failed_targets.is_empty() {
            note_parts.push(format!("failover {}", failed_targets.join("+")));
        }
        if note_parts.is_empty() {
            note_parts.push("all targets healthy".to_string());
        }
        let note = note_parts.join("; ");

        let tile_metas: Vec<TileMeta> = results
            .iter()
            .map(|r| {
                let (attempts, failover_from, _) = tile_meta
                    .get(&(r.row, r.col))
                    .cloned()
                    .unwrap_or((1, None, Vec::new()));
                let idx = result_index(&active_split, r.row, r.col);
                let planned_target = planned_targets
                    .get(idx)
                    .and_then(|target| endpoints.get(*target))
                    .map(|endpoint| endpoint.target.clone())
                    .unwrap_or_default();
                TileMeta {
                    row: r.row,
                    col: r.col,
                    target: r.target.clone(),
                    addr: r.addr.clone(),
                    planned_target,
                    cols: tile_cols(&active_split, r.col),
                    latency_ms: r.done_ms.saturating_sub(r.submit_ms),
                    worker_ms: r.worker_ms,
                    submit_call_ms: r.submit_call_ms,
                    status_ms: r.status_ms,
                    result_ms: r.result_ms,
                    attempts,
                    failover_from,
                    job_id: r.job_id.clone(),
                    ok: true,
                }
            })
            .collect();
        let mut placement: BTreeMap<String, (usize, usize)> = BTreeMap::new();
        for tile in &tile_metas {
            let entry = placement
                .entry(tile.planned_target.clone())
                .or_insert((0, 0));
            entry.0 += 1;
            entry.1 += tile.cols;
        }
        let placement: Vec<Json> = placement
            .into_iter()
            .map(|(target, (tiles, cols))| json!({"target": target, "tiles": tiles, "cols": cols}))
            .collect();

        let meta = FrameMeta {
            index: measured,
            work_ms: work,
            budget_ms: frame_budget,
            deadline_ok: work <= frame_budget,
            mode: if active_split.is_some() {
                "columns".to_string()
            } else {
                "tile-grid".to_string()
            },
            p50_ms,
            p95_ms,
            p99_ms,
            split: split_meta.clone(),
            failovers: failover_count,
            recoveries: recovery_count,
            failed_targets: failed_targets.clone(),
            recovered_targets: recovered.clone(),
            note: note.clone(),
            tiles: tile_metas.clone(),
        };
        let wall = Frame::rgb(measured, wall_w, wall_h, composited)?;
        sink.write_meta(&wall, &meta)?;

        let expected_path = cfg
            .reference_dir
            .as_ref()
            .map(|dir| dir.join(format!("frame_{:06}.rgb", measured)));
        let mut verdict = Json::Null;
        if let Some(path) = expected_path {
            let want =
                fs::read(&path).map_err(|e| format!("read reference {}: {e}", path.display()))?;
            let matched = want == wall.data;
            if matched {
                matches += 1;
            } else {
                mismatches += 1;
            }
            verdict = json!({
                "reference": path.display().to_string(),
                "reference_sha256": sha256_hex(&want),
                "match": matched,
            });
        }

        let line = json!({
            "index": measured,
            "ts_ms": frame.ts_ms,
            "started_ms": now_ms().saturating_sub(work),
            "work_ms": work,
            "deadline_ms": frame_budget,
            "deadline_ok": work <= frame_budget,
            "mode": meta.mode,
            "split": split_meta,
            "placement": placement,
            "p50_ms": p50_ms,
            "p95_ms": p95_ms,
            "p99_ms": p99_ms,
            "failovers": failover_count,
            "recoveries": recovery_count,
            "failed_targets": failed_targets,
            "recovered_targets": recovered,
            "note": note,
            "wall_sha256": wall.sha256(),
            "wall_bytes": wall.data.len(),
            "verdict": verdict,
            "tiles": results.iter().map(|r| json!({
                "row": r.row,
                "col": r.col,
                "target": r.target,
                "addr": r.addr,
                "job_id": r.job_id,
                "submit_ms": r.submit_ms,
                "submit_call_ms": r.submit_call_ms,
                "status_ms": r.status_ms,
                "result_ms": r.result_ms,
                "worker_ms": r.worker_ms,
                "done_ms": r.done_ms,
                "sha256": r.sha256,
                "bytes": r.bytes,
                "cols": tile_cols(&active_split, r.col),
                "attempts": tile_meta.get(&(r.row, r.col)).map(|m| m.0).unwrap_or(1),
                "failover_from": tile_meta.get(&(r.row, r.col)).and_then(|m| m.1.clone()),
                "errors": tile_meta.get(&(r.row, r.col)).map(|m| m.2.clone()).unwrap_or_default(),
            })).collect::<Vec<_>>(),
        });
        use std::io::Write;
        writeln!(receipt, "{}", line).map_err(|e| format!("write receipt: {e}"))?;
        measured += 1;
        if measured >= limit {
            break;
        }

        // Pace to the stream clock. The measured `work_ms` excludes this sleep.
        let elapsed = frame_start.elapsed();
        if elapsed < period {
            let sleep = period - elapsed;
            thread::sleep(sleep);
            paused += sleep;
        }
    }
    drop(receipt);

    let over_budget = work_ms.iter().filter(|w| **w > frame_budget).count() as u64;
    let summary = json!({
        "schema": STREAM_RECEIPT_SCHEMA,
        "ok": mismatches == 0 && over_budget == 0,
        "stream": descriptor.name,
        "descriptor_sha256": descriptor_sha256,
        "listing": descriptor.stage.listing.display().to_string(),
        "listing_sha256": listing_sha256,
        "source": source.describe(),
        "sink": sink.describe(),
        "clock": descriptor.clock,
        "tiles": descriptor.tiles,
        "mode": if active_split.is_some() {
            "columns"
        } else {
            "tile-grid"
        },
        "split": active_split.as_ref().map(SplitMeta::from_split),
        "calibration": calibration,
        "placement": descriptor.placement,
        "determinism": descriptor.determinism,
        "stage_descriptor": claim,
        "controller": {
            "process": format!("jotnar-stream:{}", std::process::id()),
            "source_owner": "controller",
            "state_owner": "controller",
        },
        "frames": measured,
        "wall_ms": started.elapsed().as_millis() as u64,
        "paused_ms": paused.as_millis() as u64,
        "latency_ms": {
            "p50": percentile(&work_ms, 50),
            "p95": percentile(&work_ms, 95),
            "p99": percentile(&work_ms, 99),
            "max": work_ms.iter().copied().max().unwrap_or(0),
            "over_budget": over_budget,
        },
        "failovers": failover_count,
        "recoveries": recovery_count,
        "exactness": match cfg.reference_dir {
            Some(_) => json!({"matches": matches, "mismatches": mismatches}),
            None => Json::Null,
        },
        "leases": leases.iter().map(|l| l.lease()).collect::<Vec<_>>(),
        "receipt": receipt_path.display().to_string(),
    });
    let summary_path = receipt_dir.join("summary.json");
    fs::write(
        &summary_path,
        serde_json::to_vec_pretty(&summary).map_err(|e| e.to_string())?,
    )
    .map_err(|e| format!("write {}: {e}", summary_path.display()))?;
    Ok(summary)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn frame() -> Frame {
        let data = (0..4 * 4 * 3).map(|i| (i % 251) as u8).collect();
        Frame::rgb(0, 4, 4, data).unwrap()
    }

    #[test]
    fn tiling_pads_halo_and_replicates_edges() {
        let tiles = tile_frame(
            &frame(),
            &TileSpec {
                rows: 2,
                cols: 2,
                halo: 1,
                bezel: 0,
            },
        )
        .unwrap();
        assert_eq!(tiles.len(), 4);
        for tile in &tiles {
            assert_eq!(tile.width, 4);
            assert_eq!(tile.height, 4);
            assert_eq!(tile.output_width(), 8);
            assert_eq!(tile.output_height(), 8);
        }
        // Top-left tile's top-left pixel is the replicated source corner.
        assert_eq!(&tiles[0].data[0..3], &frame().data[0..3]);
    }

    #[test]
    fn composite_crops_the_halo() {
        let spec = TileSpec {
            rows: 1,
            cols: 1,
            halo: 1,
            bezel: 0,
        };
        let tile_w = 8;
        let tile_h = 8;
        let data = vec![7u8; tile_w * tile_h * 3];
        let result = TileResult {
            row: 0,
            col: 0,
            target: "cuda".to_string(),
            addr: "x".to_string(),
            job_id: "j".to_string(),
            submit_ms: 0,
            submit_call_ms: 0,
            status_ms: 0,
            result_ms: 0,
            worker_ms: None,
            done_ms: 0,
            sha256: String::new(),
            bytes: data.len(),
            data,
        };
        let out = composite(2, 2, &spec, &[result]).unwrap();
        assert_eq!(out.len(), 4 * 4 * 3);
        assert!(out.iter().all(|v| *v == 7));
    }

    fn meta() -> FrameMeta {
        FrameMeta {
            index: 0,
            work_ms: 1,
            budget_ms: 66,
            deadline_ok: true,
            mode: "columns".to_string(),
            p50_ms: 1,
            p95_ms: 1,
            p99_ms: 1,
            split: None,
            failovers: 0,
            recoveries: 0,
            failed_targets: Vec::new(),
            recovered_targets: Vec::new(),
            note: "ok".to_string(),
            tiles: Vec::new(),
        }
    }

    #[test]
    fn default_write_meta_delegates_to_write() {
        struct Recorder {
            writes: u32,
        }
        impl FrameSink for Recorder {
            fn write(&mut self, _frame: &Frame) -> Result<(), String> {
                self.writes += 1;
                Ok(())
            }
            fn describe(&self) -> Json {
                json!({"kind": "recorder"})
            }
        }
        let mut sink = Recorder { writes: 0 };
        sink.write_meta(&frame(), &meta()).unwrap();
        assert_eq!(sink.writes, 1);
    }

    fn descriptor() -> StreamDescriptor {
        StreamDescriptor {
            schema: STREAM_SCHEMA.to_string(),
            name: "test".to_string(),
            clock: StreamClock {
                fps: 15,
                deadline_ms: 66,
                max_in_flight: 1,
            },
            source: StreamSourceSpec {
                kind: "frames".to_string(),
                dir: Some(PathBuf::from("/tmp")),
                device: None,
                width: 640,
                height: 360,
                channels: 3,
                format: "rgb24".to_string(),
                capture_width: None,
                capture_height: None,
                capture_fps: None,
            },
            stage: StreamStageSpec {
                listing: PathBuf::from("x.asm"),
                descriptor: None,
                targets: vec!["cuda".to_string()],
            },
            tiles: TileSpec {
                rows: 2,
                cols: 2,
                halo: 1,
                bezel: 0,
            },
            split: None,
            calibration: None,
            sink: StreamSinkSpec {
                kind: "null".to_string(),
                dir: None,
                listen: None,
                title: None,
                hud: false,
            },
            merge: "tile-grid".to_string(),
            placement: BTreeMap::new(),
            backpressure: "drop-oldest".to_string(),
            determinism: "exact".to_string(),
            receipt: None,
        }
    }

    #[test]
    fn descriptor_rejects_bad_geometry() {
        let mut descriptor = descriptor();
        descriptor.tiles.rows = 7;
        assert!(descriptor.validate().is_err());
    }

    #[test]
    fn calibration_requires_a_column_split_and_sane_bounds() {
        let mut descriptor = descriptor();
        descriptor.calibration = Some(CalibrationSpec {
            probe_iters: 7,
            min_cols: 32,
            max_cols: 288,
            rebalance: true,
        });
        // No split: calibration has nothing to balance.
        assert!(descriptor.validate().is_err());

        descriptor.merge = "columns".to_string();
        descriptor.split = Some(ColumnSplit {
            left_cols: 320,
            right_cols: 320,
            overlap: 1,
        });
        assert!(descriptor.validate().is_ok());

        // Bounds must stay inside the source width.
        descriptor.calibration = Some(CalibrationSpec {
            probe_iters: 7,
            min_cols: 32,
            max_cols: 640,
            rebalance: true,
        });
        assert!(descriptor.validate().is_err());
        descriptor.calibration = Some(CalibrationSpec {
            probe_iters: 0,
            min_cols: 32,
            max_cols: 288,
            rebalance: true,
        });
        assert!(descriptor.validate().is_err());
    }
}
