//! Async job engine for `jotnar-noded`.
//!
//! This is the N1 slice of the networking phase: a bounded worker pool around
//! the unchanged deterministic `Node::handle("exec")` path. The node stays
//! model-blind. A job is just an already-authenticated `exec` frame plus an
//! immutable registry snapshot taken at submission time.
//!
//! Design invariants:
//!
//! - `exec` semantics and receipts are unchanged: a job is executed by
//!   `Node::handle`, not by a parallel implementation.
//! - A job sees the program/blob registry as of submission. `program`,
//!   `store`, and `drop` update the live registry without disturbing running
//!   jobs.
//! - The queue is bounded and rejects with `queue-full` instead of growing
//!   without limit. Backpressure is the scheduler's problem.
//! - Cancellation and timeouts are cooperative. The interpreter checks a
//!   thread-local flag between instructions and between `REPEAT` iterations.
//!   A single long op finishes before cancellation takes effect.
//! - Job timings are not part of the deterministic receipt. Once a terminal
//!   result exists, its bytes match a synchronous `exec` exactly.

use std::collections::{BTreeMap, HashMap, VecDeque};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use jotnar::protocol::{decode_frame, request, Node};
use jotnar::runtime::CancelGuard;
use serde_json::{json, Value as Json};
use sha2::{Digest, Sha256};

/// Engine limits and defaults. All byte limits are admission limits, not
/// process RSS limits.
#[derive(Clone, Debug)]
pub struct JobConfig {
    /// Number of jobs that may execute at the same time.
    pub workers: usize,
    /// Number of jobs that may wait in the queue (running jobs excluded).
    pub queue_capacity: usize,
    /// Default per-job timeout. `None` means no timeout.
    pub default_timeout: Option<Duration>,
    /// Largest timeout a client may request.
    pub max_timeout: Duration,
    /// Largest accepted `submit` payload (input bytes).
    pub max_input_bytes: usize,
    /// Largest result payload the node will retain (inline or stored).
    pub max_result_bytes: usize,
    /// Results at or below this payload size return in one `job-result` frame.
    pub inline_result_bytes: usize,
    /// Total byte budget for stored (chunked) results.
    pub result_store_bytes: usize,
    /// Maximum bytes in one `result-read` chunk.
    pub result_chunk_bytes: usize,
    /// A result read within this window is not evicted.
    pub result_lease: Duration,
    /// How many completed jobs to keep for status/result lookups.
    pub retain_jobs: usize,
}

impl Default for JobConfig {
    fn default() -> Self {
        Self {
            workers: thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(4)
                .clamp(1, 8),
            queue_capacity: 64,
            default_timeout: Some(Duration::from_secs(30)),
            max_timeout: Duration::from_secs(600),
            max_input_bytes: 64 * 1024 * 1024,
            max_result_bytes: 256 * 1024 * 1024,
            inline_result_bytes: 4 * 1024 * 1024,
            result_store_bytes: 256 * 1024 * 1024,
            result_chunk_bytes: 1024 * 1024,
            result_lease: Duration::from_secs(60),
            retain_jobs: 1024,
        }
    }
}

/// Job lifecycle states. `Canceled` and `TimedOut` are terminal and carry no
/// result, even if the runtime had already computed one when the flag landed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum JobState {
    Queued,
    Running,
    Done,
    Failed,
    Canceled,
    TimedOut,
}

impl JobState {
    pub fn as_str(self) -> &'static str {
        match self {
            JobState::Queued => "queued",
            JobState::Running => "running",
            JobState::Done => "done",
            JobState::Failed => "failed",
            JobState::Canceled => "canceled",
            JobState::TimedOut => "timed_out",
        }
    }

    pub fn is_terminal(self) -> bool {
        matches!(
            self,
            JobState::Done | JobState::Failed | JobState::Canceled | JobState::TimedOut
        )
    }
}

#[derive(Clone, Debug)]
pub struct JobError {
    pub code: String,
    pub msg: String,
}

impl JobError {
    fn new(code: &str, msg: impl Into<String>) -> Self {
        Self {
            code: code.to_string(),
            msg: msg.into(),
        }
    }
}

/// Public status snapshot. Timings are advisory and deliberately excluded
/// from execution receipts.
#[derive(Clone, Debug)]
pub struct JobStatus {
    pub id: String,
    pub state: JobState,
    pub queue_ms: u64,
    pub run_ms: u64,
    pub age_ms: u64,
    pub timeout_ms: Option<u64>,
    pub idempotency_key: Option<String>,
    pub result_ready: bool,
    pub error: Option<JobError>,
}

impl JobStatus {
    pub fn to_json(&self) -> Json {
        json!({
            "job_id": self.id,
            "state": self.state.as_str(),
            "queue_ms": self.queue_ms,
            "run_ms": self.run_ms,
            "age_ms": self.age_ms,
            "timeout_ms": self.timeout_ms,
            "idempotency_key": self.idempotency_key,
            "result_ready": self.result_ready,
            "error": self.error.as_ref().map(|e| json!({"code": e.code, "msg": e.msg})),
        })
    }
}

#[derive(Clone, Debug)]
pub struct SubmitOutcome {
    pub job_id: String,
    pub state: JobState,
    pub replayed: bool,
    pub queue_position: Option<usize>,
}

#[derive(Clone, Debug)]
pub struct CancelOutcome {
    pub job_id: String,
    pub accepted: bool,
    pub previous: JobState,
    pub state: JobState,
}

#[derive(Clone, Debug)]
pub struct JobResult {
    /// The original `exec-ok` header (outputs, receipt, runtime metadata).
    pub header: Json,
    pub payload: Vec<u8>,
}

/// Metadata for a result that is too large for one `job-result` frame.
#[derive(Clone, Debug)]
pub struct StoredResultMeta {
    pub id: String,
    pub bytes: usize,
    pub chunk_bytes: usize,
    pub chunks: usize,
    pub hash: String,
    pub chunk_hashes: Vec<String>,
}

#[derive(Clone, Debug)]
pub struct StoredResultReply {
    pub header: Json,
    pub meta: StoredResultMeta,
}

#[derive(Clone, Debug)]
pub enum JobResultReply {
    Inline(JobResult),
    Stored(StoredResultReply),
}

#[derive(Clone, Debug)]
pub struct ResultChunk {
    pub result_id: String,
    pub off: usize,
    pub len: usize,
    pub total: usize,
    pub hash: String,
    pub eof: bool,
    pub payload: Vec<u8>,
}

#[derive(Clone, Debug, Default)]
pub struct EngineStats {
    pub running: usize,
    pub peak_running: usize,
    pub queued: usize,
    pub retained: usize,
    pub result_count: usize,
    pub result_bytes: usize,
}

struct Work {
    node: Node,
    frame: Vec<u8>,
}

/// Per-job watchdog rendezvous. The worker notifies on completion so the
/// timeout thread joins immediately instead of sleeping out its poll interval.
struct Watchdog {
    done: Mutex<bool>,
    cv: Condvar,
}

impl Watchdog {
    fn new() -> Self {
        Self {
            done: Mutex::new(false),
            cv: Condvar::new(),
        }
    }

    fn finish(&self) {
        *self.done.lock().unwrap() = true;
        self.cv.notify_all();
    }
}

struct JobData {
    state: JobState,
    started: Option<Instant>,
    finished: Option<Instant>,
    error: Option<JobError>,
    result: Option<JobResultState>,
}

#[derive(Clone)]
enum JobResultState {
    Inline(JobResult),
    Stored(StoredResultMeta),
}

struct StoredEntry {
    job_id: String,
    header: Json,
    payload: Arc<Vec<u8>>,
    meta: StoredResultMeta,
    /// Set on the first `result-read`; used as a soft reader lease.
    last_read: Mutex<Option<Instant>>,
    readers: AtomicUsize,
}

struct ResultStoreState {
    entries: BTreeMap<String, Arc<StoredEntry>>,
    bytes: usize,
}

struct ResultStore {
    max_bytes: usize,
    chunk_bytes: usize,
    lease: Duration,
    state: Mutex<ResultStoreState>,
    next_id: AtomicU64,
}

fn sha256_hex(bytes: &[u8]) -> String {
    let mut h = Sha256::new();
    h.update(bytes);
    format!("{:x}", h.finalize())
}

impl ResultStore {
    fn new(max_bytes: usize, chunk_bytes: usize, lease: Duration) -> Self {
        Self {
            max_bytes,
            chunk_bytes,
            lease,
            state: Mutex::new(ResultStoreState {
                entries: BTreeMap::new(),
                bytes: 0,
            }),
            next_id: AtomicU64::new(1),
        }
    }

    fn evict_for(&self, state: &mut ResultStoreState, needed: usize) -> bool {
        if needed > self.max_bytes {
            return false;
        }
        while state.bytes + needed > self.max_bytes {
            let now = Instant::now();
            let mut victim = None;
            for (id, entry) in state.entries.iter() {
                if entry.readers.load(Ordering::SeqCst) > 0 {
                    continue;
                }
                let leased = entry
                    .last_read
                    .lock()
                    .unwrap()
                    .map(|t| now.saturating_duration_since(t) < self.lease)
                    .unwrap_or(false);
                if leased {
                    continue;
                }
                victim = Some(id.clone());
                break;
            }
            let Some(id) = victim else { return false };
            if let Some(entry) = state.entries.remove(&id) {
                state.bytes = state.bytes.saturating_sub(entry.payload.len());
            }
        }
        true
    }

    fn store(
        &self,
        job_id: &str,
        header: Json,
        payload: Vec<u8>,
    ) -> Result<StoredResultMeta, JobError> {
        let bytes = payload.len();
        if bytes == 0 {
            return Err(JobError::new("bad-result", "empty result"));
        }
        if bytes > self.max_bytes {
            return Err(JobError::new(
                "result-store-full",
                format!(
                    "result {bytes} bytes exceeds result store budget {}",
                    self.max_bytes
                ),
            ));
        }
        let chunk_bytes = self.chunk_bytes.max(1);
        let chunks = bytes.div_ceil(chunk_bytes);
        let hash = sha256_hex(&payload);
        let chunk_hashes: Vec<String> = payload.chunks(chunk_bytes).map(sha256_hex).collect();
        let id = format!("res-{:016x}", self.next_id.fetch_add(1, Ordering::Relaxed));
        let meta = StoredResultMeta {
            id: id.clone(),
            bytes,
            chunk_bytes,
            chunks,
            hash,
            chunk_hashes,
        };
        let payload = Arc::new(payload);
        let entry = Arc::new(StoredEntry {
            job_id: job_id.to_string(),
            header,
            payload,
            meta: meta.clone(),
            last_read: Mutex::new(None),
            readers: AtomicUsize::new(0),
        });
        let mut state = self.state.lock().unwrap();
        if !self.evict_for(&mut state, bytes) {
            return Err(JobError::new(
                "result-store-full",
                format!(
                    "result {bytes} bytes does not fit in the result store ({} bytes used)",
                    state.bytes
                ),
            ));
        }
        state.bytes += bytes;
        state.entries.insert(id, entry);
        Ok(meta)
    }

    fn get(&self, id: &str) -> Result<StoredResultReply, JobError> {
        let state = self.state.lock().unwrap();
        let entry = state
            .entries
            .get(id)
            .ok_or_else(|| JobError::new("result-gone", format!("no stored result {id:?}")))?;
        Ok(StoredResultReply {
            header: entry.header.clone(),
            meta: entry.meta.clone(),
        })
    }

    fn read(&self, id: &str, off: usize, len: usize) -> Result<ResultChunk, JobError> {
        let entry =
            {
                let state = self.state.lock().unwrap();
                state.entries.get(id).cloned().ok_or_else(|| {
                    JobError::new("result-gone", format!("no stored result {id:?}"))
                })?
            };
        let total = entry.payload.len();
        if len == 0 {
            return Err(JobError::new("bad-range", "len must be > 0"));
        }
        if len > self.chunk_bytes {
            return Err(JobError::new(
                "bad-range",
                format!("len {len} exceeds chunk_bytes {}", self.chunk_bytes),
            ));
        }
        let end = off
            .checked_add(len)
            .ok_or_else(|| JobError::new("bad-range", "offset overflow"))?;
        if end > total {
            return Err(JobError::new(
                "bad-range",
                format!("range {off}..{end} exceeds result {total}"),
            ));
        }
        entry.readers.fetch_add(1, Ordering::SeqCst);
        *entry.last_read.lock().unwrap() = Some(Instant::now());
        let payload = entry.payload[off..end].to_vec();
        entry.readers.fetch_sub(1, Ordering::SeqCst);
        let hash = sha256_hex(&payload);
        Ok(ResultChunk {
            result_id: id.to_string(),
            off,
            len: payload.len(),
            total,
            hash,
            eof: end == total,
            payload,
        })
    }

    fn drop_result(&self, id: &str) -> Result<usize, JobError> {
        let mut state = self.state.lock().unwrap();
        let (released, readers) = {
            let entry = state
                .entries
                .get(id)
                .ok_or_else(|| JobError::new("result-gone", format!("no stored result {id:?}")))?;
            (entry.payload.len(), entry.readers.load(Ordering::SeqCst))
        };
        if readers > 0 {
            return Err(JobError::new("result-busy", "result has an active reader"));
        }
        state.entries.remove(id);
        state.bytes = state.bytes.saturating_sub(released);
        Ok(released)
    }

    fn remove_for_job(&self, job_id: &str) {
        let mut state = self.state.lock().unwrap();
        let ids: Vec<String> = state
            .entries
            .iter()
            .filter(|(_, e)| e.job_id == job_id)
            .map(|(id, _)| id.clone())
            .collect();
        for id in ids {
            if let Some(entry) = state.entries.remove(&id) {
                state.bytes = state.bytes.saturating_sub(entry.payload.len());
            }
        }
    }

    fn stats(&self) -> (usize, usize) {
        let state = self.state.lock().unwrap();
        (state.entries.len(), state.bytes)
    }
}

struct Job {
    id: String,
    submitted: Instant,
    timeout: Option<Duration>,
    cancel: Arc<AtomicBool>,
    timed_out: Arc<AtomicBool>,
    idempotency_key: Option<String>,
    data: Mutex<JobData>,
    work: Mutex<Option<Work>>,
}

impl Job {
    fn status(&self) -> JobStatus {
        let data = self.data.lock().unwrap();
        let end = data.finished.unwrap_or_else(Instant::now);
        let queue_ms = data
            .started
            .map(|s| s.saturating_duration_since(self.submitted))
            .unwrap_or_else(|| end.saturating_duration_since(self.submitted));
        let run_ms = data
            .started
            .map(|s| end.saturating_duration_since(s))
            .unwrap_or_default();
        JobStatus {
            id: self.id.clone(),
            state: data.state,
            queue_ms: queue_ms.as_millis() as u64,
            run_ms: run_ms.as_millis() as u64,
            age_ms: end.saturating_duration_since(self.submitted).as_millis() as u64,
            timeout_ms: self.timeout.map(|d| d.as_millis() as u64),
            idempotency_key: self.idempotency_key.clone(),
            result_ready: data.result.is_some(),
            error: data.error.clone(),
        }
    }
}

struct Queue {
    jobs: VecDeque<Arc<Job>>,
    closed: bool,
}

struct Inner {
    cfg: JobConfig,
    jobs: Mutex<BTreeMap<String, Arc<Job>>>,
    queue: Mutex<Queue>,
    cv: Condvar,
    next_id: AtomicU64,
    idem: Mutex<HashMap<String, String>>,
    running: AtomicU64,
    peak_running: AtomicU64,
    results: ResultStore,
}

/// Cloneable handle to a running pool of job workers.
#[derive(Clone)]
pub struct JobEngine {
    inner: Arc<Inner>,
}

impl JobEngine {
    pub fn new(mut cfg: JobConfig) -> Result<Self, JobError> {
        if cfg.workers == 0 {
            return Err(JobError::new("bad-config", "workers must be >= 1"));
        }
        cfg.queue_capacity = cfg.queue_capacity.max(1);
        cfg.retain_jobs = cfg.retain_jobs.max(1);
        cfg.result_chunk_bytes = cfg.result_chunk_bytes.max(1);
        if cfg.max_input_bytes == 0 || cfg.max_result_bytes == 0 {
            return Err(JobError::new(
                "bad-config",
                "max_input_bytes and max_result_bytes must be > 0",
            ));
        }
        if cfg.inline_result_bytes == 0 || cfg.inline_result_bytes > cfg.max_result_bytes {
            return Err(JobError::new(
                "bad-config",
                "inline_result_bytes must be in 1..=max_result_bytes",
            ));
        }
        if cfg.result_store_bytes < cfg.result_chunk_bytes {
            return Err(JobError::new(
                "bad-config",
                "result_store_bytes must be >= result_chunk_bytes",
            ));
        }
        let results = ResultStore::new(
            cfg.result_store_bytes,
            cfg.result_chunk_bytes,
            cfg.result_lease,
        );
        let inner = Arc::new(Inner {
            cfg,
            jobs: Mutex::new(BTreeMap::new()),
            queue: Mutex::new(Queue {
                jobs: VecDeque::new(),
                closed: false,
            }),
            cv: Condvar::new(),
            next_id: AtomicU64::new(1),
            idem: Mutex::new(HashMap::new()),
            running: AtomicU64::new(0),
            peak_running: AtomicU64::new(0),
            results,
        });
        for n in 0..inner.cfg.workers {
            let inner = inner.clone();
            thread::Builder::new()
                .name(format!("jotnar-job-{n}"))
                .spawn(move || worker_loop(inner))
                .map_err(|e| JobError::new("bad-config", format!("spawn worker: {e}")))?;
        }
        Ok(JobEngine { inner })
    }

    /// Submit an `exec` request as an async job.
    ///
    /// `node` must be a snapshot (see `Node::snapshot`) taken before this
    /// call; the job will not observe later registry mutations. The header is
    /// the same shape as `exec`, plus optional `timeout_ms` and
    /// `idempotency_key`.
    pub fn submit(
        &self,
        node: Node,
        header: &Json,
        payload: &[u8],
    ) -> Result<SubmitOutcome, JobError> {
        let obj = header
            .as_object()
            .ok_or_else(|| JobError::new("malformed", "submit header must be an object"))?;
        for key in ["handle", "inputs", "outputs"] {
            if !obj.contains_key(key) {
                return Err(JobError::new(
                    "malformed",
                    format!("submit needs {key:?} (same fields as exec)"),
                ));
            }
        }
        if payload.len() > self.inner.cfg.max_input_bytes {
            return Err(JobError::new(
                "job-too-large",
                format!(
                    "input {} bytes exceeds max_input_bytes {}",
                    payload.len(),
                    self.inner.cfg.max_input_bytes
                ),
            ));
        }

        let timeout = match obj.get("timeout_ms") {
            None => self.inner.cfg.default_timeout,
            Some(v) => {
                let ms = v
                    .as_u64()
                    .ok_or_else(|| JobError::new("malformed", "timeout_ms must be an integer"))?;
                if ms == 0 {
                    None
                } else {
                    let d = Duration::from_millis(ms);
                    if d > self.inner.cfg.max_timeout {
                        return Err(JobError::new(
                            "timeout-too-large",
                            format!(
                                "timeout {}ms exceeds max_timeout {}ms",
                                ms,
                                self.inner.cfg.max_timeout.as_millis()
                            ),
                        ));
                    }
                    Some(d)
                }
            }
        };

        let idem = match obj.get("idempotency_key") {
            None => None,
            Some(v) => {
                let s = v.as_str().ok_or_else(|| {
                    JobError::new("malformed", "idempotency_key must be a string")
                })?;
                if s.is_empty() || s.len() > 128 {
                    return Err(JobError::new(
                        "malformed",
                        "idempotency_key must be 1..=128 bytes",
                    ));
                }
                Some(s.to_string())
            }
        };

        // Optional replay: a repeated key returns the existing job, even if it
        // is still running. Stale entries (evicted results) are discarded.
        if let Some(key) = &idem {
            let existing = self.inner.idem.lock().unwrap().get(key).cloned();
            if let Some(existing) = existing {
                let job = self.inner.jobs.lock().unwrap().get(&existing).cloned();
                if let Some(job) = job {
                    let status = job.status();
                    return Ok(SubmitOutcome {
                        job_id: job.id.clone(),
                        state: status.state,
                        replayed: true,
                        queue_position: None,
                    });
                }
                self.inner.idem.lock().unwrap().remove(key);
            }
        }

        // Build the exact `exec` frame once, at submission. The worker runs
        // this frame through the same `Node::handle` path as synchronous exec.
        let mut exec_header = header.clone();
        if let Some(o) = exec_header.as_object_mut() {
            o.remove("timeout_ms");
            o.remove("idempotency_key");
            o.remove("wait_ms");
            o.insert("v".into(), json!(1));
            o.insert("id".into(), json!(0));
            o.insert("type".into(), json!("exec"));
            if let Some(token) = &node.token {
                o.insert("auth".into(), json!(token));
            }
        }
        let frame = request(0, "exec", exec_header, payload)
            .map_err(|e| JobError::new("malformed", e.to_string()))?;

        let id = format!(
            "job-{:016x}",
            self.inner.next_id.fetch_add(1, Ordering::Relaxed)
        );
        let job = Arc::new(Job {
            id: id.clone(),
            submitted: Instant::now(),
            timeout,
            cancel: Arc::new(AtomicBool::new(false)),
            timed_out: Arc::new(AtomicBool::new(false)),
            idempotency_key: idem.clone(),
            data: Mutex::new(JobData {
                state: JobState::Queued,
                started: None,
                finished: None,
                error: None,
                result: None,
            }),
            work: Mutex::new(Some(Work { node, frame })),
        });

        let queue_position = {
            let mut q = self.inner.queue.lock().unwrap();
            if q.jobs.len() >= self.inner.cfg.queue_capacity {
                return Err(JobError::new(
                    "queue-full",
                    format!(
                        "queue is full ({} waiting jobs); retry after a job completes",
                        q.jobs.len()
                    ),
                ));
            }
            // Publish the job before it becomes visible to a worker, so a
            // fast completion can always find and evict its own record.
            self.inner
                .jobs
                .lock()
                .unwrap()
                .insert(id.clone(), job.clone());
            if let Some(key) = &idem {
                self.inner
                    .idem
                    .lock()
                    .unwrap()
                    .insert(key.clone(), id.clone());
            }
            q.jobs.push_back(job.clone());
            // Zero-based position: jobs already waiting ahead of this one.
            let pos = q.jobs.len() - 1;
            self.inner.cv.notify_one();
            Some(pos)
        };

        let status = job.status();
        Ok(SubmitOutcome {
            job_id: id,
            state: status.state,
            replayed: false,
            queue_position,
        })
    }

    pub fn status(&self, id: &str) -> Option<JobStatus> {
        self.inner
            .jobs
            .lock()
            .unwrap()
            .get(id)
            .map(|job| job.status())
    }

    pub fn list(&self) -> Vec<JobStatus> {
        let jobs = self.inner.jobs.lock().unwrap();
        jobs.values().map(|job| job.status()).collect()
    }

    /// Fetch a completed result. Non-terminal jobs return `job-pending`.
    pub fn result(&self, id: &str) -> Result<JobResultReply, JobError> {
        let job = self
            .inner
            .jobs
            .lock()
            .unwrap()
            .get(id)
            .cloned()
            .ok_or_else(|| JobError::new("unknown-job", format!("no job {id:?}")))?;
        let data = job.data.lock().unwrap();
        match data.state {
            JobState::Done => match data.result.clone() {
                Some(JobResultState::Inline(result)) => Ok(JobResultReply::Inline(result)),
                Some(JobResultState::Stored(meta)) => {
                    self.inner.results.get(&meta.id).map(JobResultReply::Stored)
                }
                None => Err(JobError::new("job-lost", "done job has no retained result")),
            },
            JobState::Failed => Err(JobError::new(
                "job-failed",
                data.error
                    .as_ref()
                    .map(|e| e.msg.clone())
                    .unwrap_or_else(|| "job failed".to_string()),
            )),
            JobState::Canceled => Err(JobError::new("job-canceled", "job was canceled")),
            JobState::TimedOut => Err(JobError::new("job-timed-out", "job exceeded its deadline")),
            state => Err(JobError::new(
                "job-pending",
                format!("job is {}", state.as_str()),
            )),
        }
    }

    /// Read one chunk of a stored result. `result-read` clients should use
    /// `meta.chunk_bytes` and compare each returned `hash` to
    /// `meta.chunk_hashes[index]`.
    pub fn read_result(
        &self,
        result_id: &str,
        off: usize,
        len: usize,
    ) -> Result<ResultChunk, JobError> {
        self.inner.results.read(result_id, off, len)
    }

    pub fn drop_result(&self, result_id: &str) -> Result<usize, JobError> {
        self.inner.results.drop_result(result_id)
    }

    /// Cancel a queued job immediately, or ask a running job to stop at its
    /// next cooperative check. Running jobs report `accepted=true` while their
    /// state remains `running` until the worker observes the flag.
    pub fn cancel(&self, id: &str) -> Result<CancelOutcome, JobError> {
        let job = self
            .inner
            .jobs
            .lock()
            .unwrap()
            .get(id)
            .cloned()
            .ok_or_else(|| JobError::new("unknown-job", format!("no job {id:?}")))?;
        let mut data = job.data.lock().unwrap();
        let previous = data.state;
        match previous {
            JobState::Queued => {
                data.state = JobState::Canceled;
                data.finished = Some(Instant::now());
                drop(data);
                // Release the registry snapshot immediately; the queued entry
                // will be skipped by a worker and retained only for status.
                let _ = job.work.lock().unwrap().take();
                Ok(CancelOutcome {
                    job_id: job.id.clone(),
                    accepted: true,
                    previous,
                    state: JobState::Canceled,
                })
            }
            JobState::Running => {
                job.cancel.store(true, Ordering::SeqCst);
                Ok(CancelOutcome {
                    job_id: job.id.clone(),
                    accepted: true,
                    previous,
                    state: data.state,
                })
            }
            _ => Ok(CancelOutcome {
                job_id: job.id.clone(),
                accepted: false,
                previous,
                state: previous,
            }),
        }
    }

    pub fn stats(&self) -> EngineStats {
        let (result_count, result_bytes) = self.inner.results.stats();
        EngineStats {
            running: self.inner.running.load(Ordering::Relaxed) as usize,
            peak_running: self.inner.peak_running.load(Ordering::Relaxed) as usize,
            queued: self.inner.queue.lock().unwrap().jobs.len(),
            retained: self.inner.jobs.lock().unwrap().len(),
            result_count,
            result_bytes,
        }
    }

    pub fn capabilities(&self) -> Json {
        let cfg = &self.inner.cfg;
        json!({
            "protocol": 1,
            "features": ["submit", "status", "cancel", "result", "result-read",
                         "result-drop", "list", "idempotency",
                         "cooperative-cancel", "timeouts", "chunked-results"],
            "workers": cfg.workers,
            "queue_capacity": cfg.queue_capacity,
            "queued": self.inner.queue.lock().unwrap().jobs.len(),
            "running": self.inner.running.load(Ordering::Relaxed),
            "max_input_bytes": cfg.max_input_bytes,
            "max_result_bytes": cfg.max_result_bytes,
            "inline_result_bytes": cfg.inline_result_bytes,
            "result_store_bytes": cfg.result_store_bytes,
            "result_chunk_bytes": cfg.result_chunk_bytes,
            "result_lease_ms": cfg.result_lease.as_millis() as u64,
            "default_timeout_ms": cfg.default_timeout.map(|d| d.as_millis() as u64),
            "max_timeout_ms": cfg.max_timeout.as_millis() as u64,
            "retain_jobs": cfg.retain_jobs,
            "result_transport": "inline+chunked",
        })
    }

    fn note_running(&self) {
        let now = self.inner.running.fetch_add(1, Ordering::SeqCst) + 1;
        let mut peak = self.inner.peak_running.load(Ordering::SeqCst);
        while now > peak {
            match self.inner.peak_running.compare_exchange(
                peak,
                now,
                Ordering::SeqCst,
                Ordering::SeqCst,
            ) {
                Ok(_) => break,
                Err(actual) => peak = actual,
            }
        }
    }

    fn finish_job(&self) {
        // Keep a bounded ring of completed jobs. IDs are monotonic, so BTreeMap
        // order is submission order; evict the oldest terminal entry first.
        let mut jobs = self.inner.jobs.lock().unwrap();
        while jobs.len() > self.inner.cfg.retain_jobs {
            let victim = jobs.iter().find_map(|(id, job)| {
                let terminal = job.data.lock().unwrap().state.is_terminal();
                terminal.then(|| id.clone())
            });
            let Some(victim) = victim else { break };
            jobs.remove(&victim);
            self.inner.results.remove_for_job(&victim);
            self.inner
                .idem
                .lock()
                .unwrap()
                .retain(|_, target| target != &victim);
        }
    }
}

fn worker_loop(inner: Arc<Inner>) {
    let engine = JobEngine {
        inner: inner.clone(),
    };
    loop {
        let job = {
            let mut queue = inner.queue.lock().unwrap();
            loop {
                if let Some(job) = queue.jobs.pop_front() {
                    break job;
                }
                if queue.closed {
                    return;
                }
                queue = inner.cv.wait(queue).unwrap();
            }
        };

        // A queued job may have been canceled in place. Leave terminal jobs
        // untouched; the entry drains here and is retained for status.
        {
            let data = job.data.lock().unwrap();
            if data.state.is_terminal() {
                continue;
            }
        }

        let mut work = match job.work.lock().unwrap().take() {
            Some(work) => work,
            None => continue,
        };

        {
            let mut data = job.data.lock().unwrap();
            if data.state != JobState::Queued {
                continue;
            }
            data.state = JobState::Running;
            data.started = Some(Instant::now());
        }
        engine.note_running();

        let finished = Arc::new(Watchdog::new());
        let watchdog = job.timeout.map(|timeout| {
            let cancel = job.cancel.clone();
            let timed_out = job.timed_out.clone();
            let finished = finished.clone();
            thread::spawn(move || {
                let deadline = Instant::now() + timeout;
                let mut done = finished.done.lock().unwrap();
                loop {
                    if *done {
                        return;
                    }
                    let now = Instant::now();
                    if now >= deadline {
                        timed_out.store(true, Ordering::SeqCst);
                        cancel.store(true, Ordering::SeqCst);
                        return;
                    }
                    let (guard, result) = finished.cv.wait_timeout(done, deadline - now).unwrap();
                    done = guard;
                    if result.timed_out() && !*done {
                        timed_out.store(true, Ordering::SeqCst);
                        cancel.store(true, Ordering::SeqCst);
                        return;
                    }
                }
            })
        });

        let guard = CancelGuard::install(job.cancel.clone());
        let outcome = catch_unwind(AssertUnwindSafe(|| work.node.handle(&work.frame)));
        drop(guard);
        finished.finish();
        if let Some(handle) = watchdog {
            let _ = handle.join();
        }

        let canceled = job.cancel.load(Ordering::SeqCst);
        let timed_out = job.timed_out.load(Ordering::SeqCst);
        let end = Instant::now();

        let mut data = job.data.lock().unwrap();
        if timed_out || canceled {
            // A deadline or cancel wins over a result that raced to completion:
            // the terminal state carries no output bytes.
            data.result = None;
            data.error = Some(if timed_out {
                JobError::new("job-timed-out", "job exceeded its deadline")
            } else {
                JobError::new("job-canceled", "job was canceled")
            });
            data.state = if timed_out {
                JobState::TimedOut
            } else {
                JobState::Canceled
            };
        } else {
            match outcome {
                Ok(frame) => match decode_frame(&frame) {
                    Ok((mut header, payload)) => {
                        if header.get("ok").and_then(|x| x.as_bool()) != Some(true) {
                            let err = header.get("error").cloned().unwrap_or(json!({}));
                            data.error = Some(JobError::new(
                                err.get("code").and_then(|x| x.as_str()).unwrap_or("node"),
                                err.get("msg")
                                    .and_then(|x| x.as_str())
                                    .unwrap_or("node rejected the job"),
                            ));
                            data.state = JobState::Failed;
                        } else if payload.len() > inner.cfg.max_result_bytes {
                            data.error = Some(JobError::new(
                                "result-too-large",
                                format!(
                                    "result {} bytes exceeds max_result_bytes {}",
                                    payload.len(),
                                    inner.cfg.max_result_bytes
                                ),
                            ));
                            data.state = JobState::Failed;
                        } else if payload.len() <= inner.cfg.inline_result_bytes {
                            let hash = sha256_hex(&payload);
                            if let Some(o) = header.as_object_mut() {
                                o.insert(
                                    "result".into(),
                                    json!({
                                        "transport": "inline",
                                        "bytes": payload.len(),
                                        "hash": hash,
                                    }),
                                );
                            }
                            data.result = Some(JobResultState::Inline(JobResult {
                                header,
                                payload: payload.to_vec(),
                            }));
                            data.state = JobState::Done;
                        } else {
                            match inner.results.store(&job.id, header, payload.to_vec()) {
                                Ok(meta) => {
                                    data.result = Some(JobResultState::Stored(meta));
                                    data.state = JobState::Done;
                                }
                                Err(e) => {
                                    data.error = Some(e);
                                    data.state = JobState::Failed;
                                }
                            }
                        }
                    }
                    Err(e) => {
                        data.error = Some(JobError::new("node", format!("bad node frame: {e}")));
                        data.state = JobState::Failed;
                    }
                },
                Err(_) => {
                    data.error = Some(JobError::new("panic", "job panicked; worker survived"));
                    data.state = JobState::Failed;
                }
            }
        }
        if data.state.is_terminal() {
            data.finished = Some(end);
        }
        drop(data);

        inner.running.fetch_sub(1, Ordering::SeqCst);
        engine.finish_job();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use jotnar::protocol::{decode_frame, encode_frame, exec_outputs, request, StreamSpec};

    const TILE: &str = "IN X\nIN S\nT = IMUL(X, S)\nOUT = IADD(T, S)\n";
    const LONG: &str = "IN X\nIN N\nREPEAT N\nOUT = IADD(X, 1)\nEND\n";

    fn node_with(listing: &str, name: &str) -> Node {
        let mut node = Node::new(vec!["lattice".to_string()]);
        let frame = request(
            1,
            "program",
            json!({"name": name, "backend": "lattice"}),
            listing.as_bytes(),
        )
        .unwrap();
        let (header, _) = decode_frame(&node.handle(&frame)).unwrap();
        assert_eq!(header["ok"], true, "{header}");
        node
    }

    fn scalar_spec(off: usize) -> Json {
        json!({"kind": "I", "dtype": "i64", "shape": [1], "off": off, "len": 8})
    }

    fn tile_submit(engine: &JobEngine, node: &Node, x: i64, s: i64) -> SubmitOutcome {
        let mut payload = Vec::new();
        payload.extend_from_slice(&x.to_le_bytes());
        payload.extend_from_slice(&s.to_le_bytes());
        let fields = json!({
            "handle": "tile",
            "inputs": {"X": scalar_spec(0), "S": scalar_spec(8)},
            "outputs": ["OUT"],
        });
        engine.submit(node.snapshot(), &fields, &payload).unwrap()
    }

    fn wait_terminal(engine: &JobEngine, id: &str) -> JobStatus {
        let deadline = Instant::now() + Duration::from_secs(5);
        loop {
            let status = engine.status(id).expect("job retained");
            if status.state.is_terminal() {
                return status;
            }
            assert!(
                Instant::now() < deadline,
                "job {id} stuck in {:?}",
                status.state
            );
            thread::sleep(Duration::from_millis(5));
        }
    }

    fn inline_result(engine: &JobEngine, id: &str) -> JobResult {
        match engine.result(id).unwrap() {
            JobResultReply::Inline(result) => result,
            JobResultReply::Stored(_) => panic!("expected an inline result"),
        }
    }

    fn result_value(result: &JobResult, name: &str) -> i64 {
        let frame = encode_frame(&result.header, &result.payload).unwrap();
        let (header, payload) = decode_frame(&frame).unwrap();
        let spec = StreamSpec::from_json(&header["outputs"][name]).unwrap();
        let off = header["outputs"][name]["off"].as_u64().unwrap() as usize;
        let len = header["outputs"][name]["len"].as_u64().unwrap() as usize;
        let v = jotnar::protocol::decode_stream(&spec, &payload[off..off + len]).unwrap();
        v.first_i64("test").unwrap()
    }

    #[test]
    fn submit_runs_the_same_exec_path_and_result() {
        let engine = JobEngine::new(JobConfig {
            workers: 2,
            default_timeout: None,
            ..JobConfig::default()
        })
        .unwrap();
        let node = node_with(TILE, "tile");
        let submitted = tile_submit(&engine, &node, 7, 3);
        assert_eq!(submitted.state, JobState::Queued);
        let status = wait_terminal(&engine, &submitted.job_id);
        assert_eq!(status.state, JobState::Done, "{status:?}");
        let result = inline_result(&engine, &submitted.job_id);
        assert_eq!(result_value(&result, "OUT"), 24);
        assert!(result.header["receipt"]["program_hash"].is_string());
        assert_eq!(result.header["receipt"]["verdict"], "exact");
    }

    #[test]
    fn concurrent_jobs_complete_and_respect_the_pool() {
        let engine = JobEngine::new(JobConfig {
            workers: 2,
            default_timeout: None,
            ..JobConfig::default()
        })
        .unwrap();
        let node = node_with(LONG, "long");
        let n = 200_000i64;
        let jobs: Vec<SubmitOutcome> = (0..2)
            .map(|i| tile_submit_named(&engine, &node, "long", i, n))
            .collect();
        for job in &jobs {
            let status = wait_terminal(&engine, &job.job_id);
            assert_eq!(status.state, JobState::Done);
        }
        for (i, job) in jobs.iter().enumerate() {
            let result = inline_result(&engine, &job.job_id);
            assert_eq!(result_value(&result, "OUT"), i as i64 + 1);
        }
        assert_eq!(engine.stats().peak_running, 2);
    }

    #[test]
    fn queue_is_bounded_and_rejects() {
        let engine = JobEngine::new(JobConfig {
            workers: 1,
            queue_capacity: 1,
            default_timeout: None,
            ..JobConfig::default()
        })
        .unwrap();
        let node = node_with(LONG, "long");
        let first = tile_submit_named(&engine, &node, "long", 0, 2_000_000_000);
        let deadline = Instant::now() + Duration::from_secs(2);
        while engine.status(&first.job_id).unwrap().state != JobState::Running {
            assert!(Instant::now() < deadline, "first job never started");
            thread::sleep(Duration::from_millis(5));
        }
        let second = tile_submit_named(&engine, &node, "long", 0, 2_000_000_000);
        assert!(!second.job_id.is_empty());
        let err = engine
            .submit(
                node.snapshot(),
                &long_fields(0, 2_000_000_000),
                &0i64.to_le_bytes(),
            )
            .unwrap_err();
        assert_eq!(err.code, "queue-full");
        engine.cancel(&first.job_id).unwrap();
        engine.cancel(&second.job_id).unwrap();
        wait_terminal(&engine, &first.job_id);
        wait_terminal(&engine, &second.job_id);
    }

    fn long_fields(_x: i64, _n: i64) -> Json {
        json!({
            "handle": "long",
            "inputs": {"X": scalar_spec(0), "N": scalar_spec(8)},
            "outputs": ["OUT"],
        })
    }

    fn tile_submit_named(
        engine: &JobEngine,
        node: &Node,
        handle: &str,
        x: i64,
        n: i64,
    ) -> SubmitOutcome {
        let mut payload = Vec::new();
        payload.extend_from_slice(&x.to_le_bytes());
        payload.extend_from_slice(&n.to_le_bytes());
        let fields = json!({
            "handle": handle,
            "inputs": {"X": scalar_spec(0), "N": scalar_spec(8)},
            "outputs": ["OUT"],
        });
        engine.submit(node.snapshot(), &fields, &payload).unwrap()
    }

    #[test]
    fn cancel_queued_and_running() {
        let engine = JobEngine::new(JobConfig {
            workers: 1,
            queue_capacity: 4,
            default_timeout: None,
            ..JobConfig::default()
        })
        .unwrap();
        let node = node_with(LONG, "long");
        let running = tile_submit_named(&engine, &node, "long", 0, 2_000_000_000);
        let queued = tile_submit_named(&engine, &node, "long", 0, 2_000_000_000);
        let deadline = Instant::now() + Duration::from_secs(2);
        while engine.status(&running.job_id).unwrap().state != JobState::Running {
            assert!(Instant::now() < deadline);
            thread::sleep(Duration::from_millis(5));
        }
        let qcancel = engine.cancel(&queued.job_id).unwrap();
        assert!(qcancel.accepted);
        assert_eq!(qcancel.state, JobState::Canceled);
        assert_eq!(
            engine.status(&queued.job_id).unwrap().state,
            JobState::Canceled
        );

        let rcancel = engine.cancel(&running.job_id).unwrap();
        assert!(rcancel.accepted);
        let status = wait_terminal(&engine, &running.job_id);
        assert_eq!(status.state, JobState::Canceled);
        assert!(engine.result(&running.job_id).is_err());
    }

    #[test]
    fn timeout_is_terminal_and_discards_the_result() {
        let engine = JobEngine::new(JobConfig {
            workers: 1,
            default_timeout: Some(Duration::from_millis(50)),
            max_timeout: Duration::from_secs(1),
            ..JobConfig::default()
        })
        .unwrap();
        let node = node_with(LONG, "long");
        let job = tile_submit_named(&engine, &node, "long", 0, 2_000_000_000);
        let status = wait_terminal(&engine, &job.job_id);
        assert_eq!(status.state, JobState::TimedOut, "{status:?}");
        assert_eq!(
            engine.result(&job.job_id).unwrap_err().code,
            "job-timed-out"
        );
    }

    #[test]
    fn idempotency_key_replays_the_same_job() {
        let engine = JobEngine::new(JobConfig {
            workers: 1,
            default_timeout: None,
            ..JobConfig::default()
        })
        .unwrap();
        let node = node_with(TILE, "tile");
        let mut payload = Vec::new();
        payload.extend_from_slice(&4i64.to_le_bytes());
        payload.extend_from_slice(&5i64.to_le_bytes());
        let fields = json!({
            "handle": "tile",
            "inputs": {"X": scalar_spec(0), "S": scalar_spec(8)},
            "outputs": ["OUT"],
            "idempotency_key": "batch-1/tile/0",
        });
        let a = engine.submit(node.snapshot(), &fields, &payload).unwrap();
        let b = engine.submit(node.snapshot(), &fields, &payload).unwrap();
        assert_eq!(a.job_id, b.job_id);
        assert!(!a.replayed);
        assert!(b.replayed);
        let status = wait_terminal(&engine, &a.job_id);
        assert_eq!(status.state, JobState::Done);
        assert_eq!(engine.list().len(), 1);
    }

    const ADD: &str = "IN X\nOUT = IADD(X, X)\n";

    fn vector_submit(engine: &JobEngine, node: &Node, n: usize) -> SubmitOutcome {
        let mut payload = Vec::new();
        for i in 0..n as i64 {
            payload.extend_from_slice(&i.to_le_bytes());
        }
        let fields = json!({
            "handle": "add",
            "inputs": {"X": {"kind": "I", "dtype": "i64", "shape": [n],
                             "off": 0, "len": n * 8}},
            "outputs": ["OUT"],
        });
        engine.submit(node.snapshot(), &fields, &payload).unwrap()
    }

    fn direct_vector_output(node: &mut Node, n: usize) -> Vec<u8> {
        let mut payload = Vec::new();
        for i in 0..n as i64 {
            payload.extend_from_slice(&i.to_le_bytes());
        }
        let fields = json!({
            "handle": "add",
            "inputs": {"X": {"kind": "I", "dtype": "i64", "shape": [n],
                             "off": 0, "len": n * 8}},
            "outputs": ["OUT"],
        });
        let exec = request(99, "exec", fields, &payload).unwrap();
        exec_outputs(&node.handle(&exec)).unwrap()["OUT"].clone()
    }

    fn wait_done(engine: &JobEngine, id: &str) -> JobStatus {
        let status = wait_terminal(engine, id);
        assert_eq!(status.state, JobState::Done, "{status:?}");
        status
    }

    #[test]
    fn stored_result_roundtrips_in_chunks() {
        let engine = JobEngine::new(JobConfig {
            workers: 1,
            default_timeout: None,
            max_result_bytes: 4096,
            inline_result_bytes: 64,
            result_store_bytes: 4096,
            result_chunk_bytes: 256,
            result_lease: Duration::ZERO,
            ..JobConfig::default()
        })
        .unwrap();
        let mut node = node_with(ADD, "add");
        let n = 100;
        let direct = direct_vector_output(&mut node, n);
        let job = vector_submit(&engine, &node, n);
        wait_done(&engine, &job.job_id);
        let reply = engine.result(&job.job_id).unwrap();
        let stored = match reply {
            JobResultReply::Stored(stored) => stored,
            JobResultReply::Inline(_) => panic!("expected a stored result"),
        };
        assert_eq!(stored.meta.bytes, 800);
        assert_eq!(stored.meta.chunks, 4);
        assert_eq!(stored.meta.hash, sha256_hex(&direct));

        let mut got = Vec::new();
        for i in 0..stored.meta.chunks {
            let off = i * stored.meta.chunk_bytes;
            let len = (stored.meta.bytes - off).min(stored.meta.chunk_bytes);
            let chunk = engine.read_result(&stored.meta.id, off, len).unwrap();
            assert_eq!(chunk.hash, stored.meta.chunk_hashes[i]);
            assert_eq!(chunk.eof, i + 1 == stored.meta.chunks);
            assert_eq!(chunk.total, stored.meta.bytes);
            got.extend_from_slice(&chunk.payload);
        }
        assert_eq!(got, direct);
        assert_eq!(sha256_hex(&got), stored.meta.hash);

        let err = engine
            .read_result(&stored.meta.id, 0, stored.meta.chunk_bytes + 1)
            .unwrap_err();
        assert_eq!(err.code, "bad-range");
        let err = engine
            .read_result(&stored.meta.id, stored.meta.bytes - 1, 2)
            .unwrap_err();
        assert_eq!(err.code, "bad-range");

        let released = engine.drop_result(&stored.meta.id).unwrap();
        assert_eq!(released, 800);
        assert_eq!(engine.result(&job.job_id).unwrap_err().code, "result-gone");
        assert_eq!(engine.stats().result_count, 0);
    }

    #[test]
    fn result_store_evicts_unread_results() {
        let engine = JobEngine::new(JobConfig {
            workers: 1,
            default_timeout: None,
            max_result_bytes: 4096,
            inline_result_bytes: 64,
            result_store_bytes: 1000,
            result_chunk_bytes: 512,
            result_lease: Duration::ZERO,
            ..JobConfig::default()
        })
        .unwrap();
        let node = node_with(ADD, "add");
        let first = vector_submit(&engine, &node, 100);
        wait_done(&engine, &first.job_id);
        let second = vector_submit(&engine, &node, 100);
        wait_done(&engine, &second.job_id);
        assert_eq!(
            engine.result(&first.job_id).unwrap_err().code,
            "result-gone"
        );
        assert!(matches!(
            engine.result(&second.job_id).unwrap(),
            JobResultReply::Stored(_)
        ));
        let stats = engine.stats();
        assert_eq!(stats.result_count, 1);
        assert_eq!(stats.result_bytes, 800);
    }

    #[test]
    fn active_result_lease_blocks_eviction() {
        let engine = JobEngine::new(JobConfig {
            workers: 1,
            default_timeout: None,
            max_result_bytes: 4096,
            inline_result_bytes: 64,
            result_store_bytes: 1000,
            result_chunk_bytes: 512,
            result_lease: Duration::from_secs(60),
            ..JobConfig::default()
        })
        .unwrap();
        let node = node_with(ADD, "add");
        let first = vector_submit(&engine, &node, 100);
        wait_done(&engine, &first.job_id);
        let meta = match engine.result(&first.job_id).unwrap() {
            JobResultReply::Stored(stored) => stored.meta,
            JobResultReply::Inline(_) => panic!("expected stored"),
        };
        engine.read_result(&meta.id, 0, 256).unwrap();

        let second = vector_submit(&engine, &node, 100);
        let status = wait_terminal(&engine, &second.job_id);
        assert_eq!(status.state, JobState::Failed, "{status:?}");
        assert_eq!(status.error.unwrap().code, "result-store-full");
        // The first result is still present and readable.
        assert!(matches!(
            engine.result(&first.job_id).unwrap(),
            JobResultReply::Stored(_)
        ));
    }

    #[test]
    fn inline_result_carries_hash() {
        let engine = JobEngine::new(JobConfig {
            workers: 1,
            default_timeout: None,
            ..JobConfig::default()
        })
        .unwrap();
        let node = node_with(TILE, "tile");
        let job = tile_submit(&engine, &node, 7, 3);
        wait_done(&engine, &job.job_id);
        let result = inline_result(&engine, &job.job_id);
        assert_eq!(result.header["result"]["transport"], "inline");
        assert_eq!(result.header["result"]["bytes"], 8);
        assert_eq!(result.header["result"]["hash"], sha256_hex(&result.payload));
    }

    #[test]
    fn oversize_results_fail_loudly() {
        let engine = JobEngine::new(JobConfig {
            workers: 1,
            default_timeout: None,
            max_result_bytes: 1,
            inline_result_bytes: 1,
            ..JobConfig::default()
        })
        .unwrap();
        let node = node_with(TILE, "tile");
        let job = tile_submit(&engine, &node, 2, 3);
        let status = wait_terminal(&engine, &job.job_id);
        assert_eq!(status.state, JobState::Failed);
        assert_eq!(status.error.unwrap().code, "result-too-large");
    }

    #[test]
    fn exec_compatibility_is_byte_identical() {
        let mut node = node_with(TILE, "tile");
        let payload = {
            let mut p = Vec::new();
            p.extend_from_slice(&9i64.to_le_bytes());
            p.extend_from_slice(&2i64.to_le_bytes());
            p
        };
        let exec = request(
            55,
            "exec",
            json!({
                "handle": "tile",
                "inputs": {"X": scalar_spec(0), "S": scalar_spec(8)},
                "outputs": ["OUT"],
            }),
            &payload,
        )
        .unwrap();
        let direct = exec_outputs(&node.handle(&exec)).unwrap();

        let engine = JobEngine::new(JobConfig {
            workers: 1,
            default_timeout: None,
            ..JobConfig::default()
        })
        .unwrap();
        let fields = json!({
            "handle": "tile",
            "inputs": {"X": scalar_spec(0), "S": scalar_spec(8)},
            "outputs": ["OUT"],
        });
        let job = engine.submit(node.snapshot(), &fields, &payload).unwrap();
        wait_terminal(&engine, &job.job_id);
        let result = inline_result(&engine, &job.job_id);
        let frame = encode_frame(&result.header, &result.payload).unwrap();
        let async_out = exec_outputs(&frame).unwrap();
        assert_eq!(direct, async_out);
    }
}
