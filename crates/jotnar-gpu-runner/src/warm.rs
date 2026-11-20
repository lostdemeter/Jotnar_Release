//! Runner-side warm worker pool for CUDA/HIP artifacts.

use std::collections::{HashMap, VecDeque};
use std::fs::{File, OpenOptions};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStderr, ChildStdin, ChildStdout, Command, Stdio};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

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

struct WarmEntry {
    worker: Arc<Mutex<WarmWorker>>,
    last_used: Instant,
}

pub struct WarmCache {
    workers: HashMap<String, WarmEntry>,
    order: VecDeque<String>,
    limit: usize,
}

impl WarmCache {
    pub fn new(limit: usize) -> Self {
        sweep_stale_workers();
        Self {
            workers: HashMap::new(),
            order: VecDeque::new(),
            limit: limit.max(1),
        }
    }

    pub fn get_or_create(
        &mut self,
        spec: &WarmSpec,
    ) -> Result<(Arc<Mutex<WarmWorker>>, bool), String> {
        if let Some(entry) = self.workers.get_mut(&spec.key) {
            entry.last_used = Instant::now();
            touch(&mut self.order, &spec.key);
            return Ok((entry.worker.clone(), false));
        }
        while self.workers.len() >= self.limit {
            let Some(old) = self.order.pop_front() else {
                break;
            };
            self.workers.remove(&old);
        }
        let w = Arc::new(Mutex::new(WarmWorker::start(spec.clone())?));
        self.workers.insert(
            spec.key.clone(),
            WarmEntry {
                worker: w.clone(),
                last_used: Instant::now(),
            },
        );
        self.order.push_back(spec.key.clone());
        Ok((w, true))
    }

    pub fn remove(&mut self, key: &str) {
        self.workers.remove(key);
        self.order.retain(|k| k != key);
    }

    /// Number of live warm workers (one per compiled artifact key).
    pub fn len(&self) -> usize {
        self.workers.len()
    }

    pub fn is_empty(&self) -> bool {
        self.workers.is_empty()
    }

    /// Remove workers idle for at least `idle` and not currently used.
    ///
    /// Returns the dropped workers so the caller can release them outside the
    /// cache lock; dropping a wedged worker can block for the shutdown grace
    /// period.
    pub fn remove_idle(&mut self, idle: Duration, now: Instant) -> Vec<Arc<Mutex<WarmWorker>>> {
        if idle.is_zero() {
            return Vec::new();
        }
        let stale: Vec<String> = self
            .workers
            .iter()
            .filter(|(_, entry)| {
                idle_expired(now, entry.last_used, idle, Arc::strong_count(&entry.worker))
            })
            .map(|(key, _)| key.clone())
            .collect();
        let mut dropped = Vec::with_capacity(stale.len());
        for key in stale {
            if let Some(entry) = self.workers.remove(&key) {
                dropped.push(entry.worker);
            }
            self.order.retain(|k| k != &key);
        }
        dropped
    }
}

fn idle_expired(now: Instant, last_used: Instant, idle: Duration, strong: usize) -> bool {
    strong == 1 && now.saturating_duration_since(last_used) >= idle
}

fn touch(order: &mut VecDeque<String>, key: &str) {
    if let Some(pos) = order.iter().position(|k| k == key) {
        order.remove(pos);
    }
    order.push_back(key.to_string());
}

fn warm_base() -> PathBuf {
    if Path::new("/dev/shm").is_dir() {
        PathBuf::from("/dev/shm")
    } else {
        std::env::temp_dir()
    }
}

fn sweep_stale_workers() {
    sweep_stale_workers_in(&warm_base(), Duration::from_secs(600));
}

/// Remove worker directories left by dead runners.
///
/// A live worker always has an `owner` file (startup fails if it cannot write
/// one), so a directory with no owner marker can only come from a runner that
/// died before the marker existed, or from a pre-owner build. Those are
/// removed once they are older than `stale_after`, so a worker that is still
/// starting up is not mistaken for debris.
fn sweep_stale_workers_in(base: &Path, stale_after: Duration) {
    if !cfg!(target_os = "linux") {
        return;
    }
    let Ok(entries) = std::fs::read_dir(base) else {
        return;
    };
    for entry in entries.flatten() {
        let name = entry.file_name();
        if !name.to_string_lossy().starts_with("jotnar-warm-") {
            continue;
        }
        let path = entry.path();
        match std::fs::read_to_string(path.join("owner")) {
            Ok(owner) => {
                if let Ok(pid) = owner.trim().parse::<u32>() {
                    if !Path::new("/proc").join(pid.to_string()).exists() {
                        let _ = std::fs::remove_dir_all(&path);
                    }
                }
            }
            // Only an absent marker is debris; never delete a directory whose
            // marker exists but cannot be read or parsed.
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                sweep_if_old(&path, &entry, stale_after);
            }
            Err(_) => {}
        }
    }
}

fn sweep_if_old(path: &Path, entry: &std::fs::DirEntry, stale_after: Duration) {
    let old = entry
        .metadata()
        .ok()
        .and_then(|m| m.modified().ok())
        .and_then(|t| std::time::SystemTime::now().duration_since(t).ok())
        .map(|age| age >= stale_after)
        .unwrap_or(false);
    if old {
        let _ = std::fs::remove_dir_all(path);
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
        let base = warm_base();
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
        std::fs::write(dir.path().join("owner"), std::process::id().to_string())
            .map_err(|e| format!("warm owner: {e}"))?;
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

    pub fn startup_ms(&self) -> u128 {
        self.worker_start_ms
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
        let started = Instant::now();
        loop {
            match self.child.try_wait() {
                Ok(Some(_)) => return,
                Ok(None) if started.elapsed() < Duration::from_secs(2) => {
                    std::thread::sleep(Duration::from_millis(20));
                }
                Ok(None) | Err(_) => {
                    let _ = self.child.kill();
                    let _ = self.child.wait();
                    return;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn base() -> tempfile::TempDir {
        tempfile::tempdir().unwrap()
    }

    fn worker_dir(base: &Path, name: &str) -> PathBuf {
        let p = base.join(name);
        std::fs::create_dir(&p).unwrap();
        p
    }

    #[test]
    fn idle_expired_needs_timeout_and_exclusive_use() {
        let now = Instant::now();
        let last = now - Duration::from_secs(10);
        assert!(idle_expired(now, last, Duration::from_secs(5), 1));
        assert!(!idle_expired(now, last, Duration::from_secs(20), 1));
        assert!(!idle_expired(now, last, Duration::from_secs(5), 2));
    }

    #[test]
    fn sweep_removes_dead_owner() {
        let b = base();
        let d = worker_dir(b.path(), "jotnar-warm-dead");
        std::fs::write(d.join("owner"), "4294967295").unwrap();
        sweep_stale_workers_in(b.path(), Duration::from_secs(3600));
        assert!(!d.exists());
    }

    #[test]
    fn sweep_keeps_live_owner() {
        let b = base();
        let d = worker_dir(b.path(), "jotnar-warm-live");
        std::fs::write(d.join("owner"), std::process::id().to_string()).unwrap();
        sweep_stale_workers_in(b.path(), Duration::ZERO);
        assert!(d.exists());
    }

    #[test]
    fn sweep_keeps_recent_unowned_then_removes_it() {
        let b = base();
        let d = worker_dir(b.path(), "jotnar-warm-unowned");
        // A fresh directory could still be starting up.
        sweep_stale_workers_in(b.path(), Duration::from_secs(3600));
        assert!(d.exists());
        // Once the grace period has elapsed it is debris.
        sweep_stale_workers_in(b.path(), Duration::ZERO);
        assert!(!d.exists());
    }

    #[test]
    fn sweep_ignores_other_directories() {
        let b = base();
        let d = worker_dir(b.path(), "not-a-worker");
        sweep_stale_workers_in(b.path(), Duration::ZERO);
        assert!(d.exists());
    }

    #[test]
    fn sweep_keeps_malformed_owner() {
        let b = base();
        let d = worker_dir(b.path(), "jotnar-warm-malformed");
        std::fs::write(d.join("owner"), "not-a-pid").unwrap();
        sweep_stale_workers_in(b.path(), Duration::ZERO);
        assert!(d.exists());
    }
}
