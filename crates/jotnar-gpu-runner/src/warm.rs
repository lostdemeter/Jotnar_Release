//! Runner-side warm worker pool for CUDA/HIP artifacts.

use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStderr, ChildStdin, ChildStdout, Command, Stdio};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use jotnar::codegen::infer::CKind;
use jotnar::codegen::{Artifact, Target};
use jotnar::payload::Feeds;
use jotnar::value::Value;
use memmap2::MmapMut;

use super::{decode_value, encode_value};

#[derive(Clone, Debug)]
pub struct WarmStream {
    pub name: String,
    pub kind: CKind,
    pub bytes: usize,
    pub shape: Vec<Option<usize>>,
}

#[derive(Clone, Debug)]
pub struct WarmSpec {
    pub key: String,
    pub exe: PathBuf,
    pub target: Target,
    pub inputs: Vec<WarmStream>,
    pub outputs: Vec<WarmStream>,
}

pub struct WarmRun {
    pub outputs: HashMap<String, Value>,
    pub worker_start_ms: u128,
    pub frame_ms: u128,
}

pub struct WarmCache {
    workers: HashMap<String, Arc<Mutex<WarmWorker>>>,
}

impl WarmCache {
    pub fn new() -> Self {
        Self {
            workers: HashMap::new(),
        }
    }

    pub fn get_or_create(&mut self, spec: &WarmSpec) -> Result<Arc<Mutex<WarmWorker>>, String> {
        if let Some(w) = self.workers.get(&spec.key) {
            return Ok(w.clone());
        }
        let w = Arc::new(Mutex::new(WarmWorker::start(spec.clone())?));
        self.workers.insert(spec.key.clone(), w.clone());
        Ok(w)
    }

    pub fn remove(&mut self, key: &str) {
        self.workers.remove(key);
    }
}

pub struct WarmWorker {
    child: Child,
    stdin: ChildStdin,
    stdout: ChildStdout,
    stderr: ChildStderr,
    _dir: tempfile::TempDir,
    spec: WarmSpec,
    input_maps: Vec<MmapMut>,
    output_maps: Vec<MmapMut>,
    worker_start_ms: u128,
}

fn stream_bytes(kind: CKind, shape: &[Option<usize>]) -> Result<usize, String> {
    let n = shape
        .iter()
        .map(|x| x.ok_or_else(|| "warm frame needs static shapes".to_string()))
        .collect::<Result<Vec<_>, _>>()?
        .iter()
        .product::<usize>();
    match kind {
        CKind::F => Ok(n * 4),
        CKind::I => Ok(n * 8),
        CKind::T => Err("warm frame does not support T streams".to_string()),
    }
}

fn stream(name: &str, artifact: &Artifact) -> Result<WarmStream, String> {
    let info = artifact
        .streams
        .get(name)
        .ok_or_else(|| format!("no stream info for {name:?}"))?;
    Ok(WarmStream {
        name: name.to_string(),
        kind: info.kind,
        bytes: stream_bytes(info.kind, &info.shape)?,
        shape: info.shape.clone(),
    })
}

pub fn spec_from_artifact(
    artifact: &Artifact,
    exe: &Path,
    base_key: &str,
    target: Target,
) -> Result<WarmSpec, String> {
    if !artifact.warm_frame {
        return Err("artifact has no warm-frame mode".to_string());
    }
    let inputs = artifact
        .inputs
        .iter()
        .map(|n| stream(n, artifact))
        .collect::<Result<Vec<_>, _>>()?;
    let outputs = artifact
        .outputs
        .iter()
        .map(|n| stream(n, artifact))
        .collect::<Result<Vec<_>, _>>()?;
    let mut key = format!("{base_key}|{}|", target.as_str());
    for s in inputs.iter().chain(outputs.iter()) {
        key.push_str(&format!(
            "{}:{}:{:?}:{};",
            s.name,
            match s.kind {
                CKind::F => "f",
                CKind::I => "i",
                CKind::T => "t",
            },
            s.shape,
            s.bytes
        ));
    }
    Ok(WarmSpec {
        key,
        exe: exe.to_path_buf(),
        target,
        inputs,
        outputs,
    })
}

impl WarmWorker {
    fn start(spec: WarmSpec) -> Result<Self, String> {
        let base = if Path::new("/dev/shm").is_dir() {
            PathBuf::from("/dev/shm")
        } else {
            std::env::temp_dir()
        };
        let dir = tempfile::Builder::new()
            .prefix("jotnar-warm-")
            .tempdir_in(&base)
            .map_err(|e| format!("warm tempdir: {e}"))?;
        let mut input_maps = Vec::with_capacity(spec.inputs.len());
        let mut output_maps = Vec::with_capacity(spec.outputs.len());
        let mut cfg = format!("jotnar-warm 1\ntarget {}\n", spec.target.as_str());
        for (i, s) in spec.inputs.iter().enumerate() {
            let path = dir.path().join(format!("in_{i}"));
            let file = OpenOptions::new()
                .create(true)
                .read(true)
                .write(true)
                .truncate(true)
                .open(&path)
                .map_err(|e| format!("warm input {}: {e}", path.display()))?;
            file.set_len(s.bytes as u64)
                .map_err(|e| format!("warm input size {}: {e}", path.display()))?;
            cfg.push_str(&format!("input {}\n", path.display()));
            input_maps.push(map(&file)?);
        }
        for (i, s) in spec.outputs.iter().enumerate() {
            let path = dir.path().join(format!("out_{i}"));
            let file = OpenOptions::new()
                .create(true)
                .read(true)
                .write(true)
                .truncate(true)
                .open(&path)
                .map_err(|e| format!("warm output {}: {e}", path.display()))?;
            file.set_len(s.bytes as u64)
                .map_err(|e| format!("warm output size {}: {e}", path.display()))?;
            cfg.push_str(&format!("output {}\n", path.display()));
            output_maps.push(map(&file)?);
        }
        let cfg_path = dir.path().join("warm.cfg");
        std::fs::write(&cfg_path, cfg).map_err(|e| format!("warm config: {e}"))?;
        let t0 = Instant::now();
        let mut child = Command::new(&spec.exe)
            .arg("--warm-frame")
            .arg(&cfg_path)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|e| format!("spawn warm worker {}: {e}", spec.exe.display()))?;
        let stdin = child.stdin.take().ok_or("warm worker has no stdin")?;
        let mut stdout = child.stdout.take().ok_or("warm worker has no stdout")?;
        let stderr = child.stderr.take().ok_or("warm worker has no stderr")?;
        let mut ready = [0u8; 1];
        stdout
            .read_exact(&mut ready)
            .map_err(|e| format!("warm worker startup: {e}"))?;
        if ready[0] != b'R' {
            let _ = child.kill();
            let _ = child.wait();
            return Err("warm worker did not become ready".to_string());
        }
        let worker_start_ms = t0.elapsed().as_millis();
        Ok(Self {
            child,
            stdin,
            stdout,
            stderr,
            _dir: dir,
            spec,
            input_maps,
            output_maps,
            worker_start_ms,
        })
    }

    pub fn run_frame(&mut self, feeds: &Feeds) -> Result<WarmRun, String> {
        for (i, s) in self.spec.inputs.iter().enumerate() {
            let value = feeds
                .get(&s.name)
                .ok_or_else(|| format!("payload missing input {:?}", s.name))?;
            let bytes = encode_value(value, s.kind, self.spec.target)?;
            if bytes.len() != s.bytes {
                return Err(format!(
                    "input {:?} encoded {} bytes, expected {}",
                    s.name,
                    bytes.len(),
                    s.bytes
                ));
            }
            self.input_maps[i][..s.bytes].copy_from_slice(&bytes);
        }
        let t0 = Instant::now();
        self.stdin
            .write_all(&[1])
            .map_err(|e| format!("warm request: {e}"))?;
        self.stdin.flush().map_err(|e| format!("warm flush: {e}"))?;
        let mut status = [0u8; 1];
        self.stdout
            .read_exact(&mut status)
            .map_err(|e| format!("warm response: {e}"))?;
        if status[0] != 0 {
            let mut err = String::new();
            let _ = self.stderr.read_to_string(&mut err);
            let _ = self.child.kill();
            let _ = self.child.wait();
            return Err(format!("warm worker error: {err}"));
        }
        let frame_ms = t0.elapsed().as_millis();
        let mut outputs = HashMap::new();
        for (i, s) in self.spec.outputs.iter().enumerate() {
            let value = decode_value(
                &self.output_maps[i][..s.bytes],
                s.kind,
                self.spec.target,
                &s.shape,
            )?;
            outputs.insert(s.name.clone(), value);
        }
        Ok(WarmRun {
            outputs,
            worker_start_ms: self.worker_start_ms,
            frame_ms,
        })
    }
}

fn map(file: &File) -> Result<MmapMut, String> {
    unsafe { MmapMut::map_mut(file) }.map_err(|e| format!("warm mmap: {e}"))
}

impl Drop for WarmWorker {
    fn drop(&mut self) {
        let _ = self.stdin.write_all(&[0]);
        let _ = self.stdin.flush();
        let _ = self.child.wait();
    }
}
