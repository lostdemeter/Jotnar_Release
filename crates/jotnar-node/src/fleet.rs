//! Controller-side fleet leases and admission.
//!
//! This module is the smallest shared object that lets independent controller
//! processes and the DOOM coordinator agree on who has reserved capacity. It
//! is deliberately not a scheduler, a daemon, or a VM abstraction:
//!
//! - a **lease** is a JSON file with an owner, class, target, slot count, and
//!   an expiry; the owner must renew it;
//! - a **batch admission** is a short expiring reservation taken for one stage,
//!   so a killed client cannot hold a slot forever;
//! - a **waiter** is a queued request ordered by its original request time.
//!
//! Enforcement is advisory: the controller refuses to over-admit; the executor
//! queue still accepts whatever it is given. The receipt records that fact.
//!
//! All timestamps are Unix milliseconds. The fleet directory must be on shared
//! storage visible to every participant; the trusted-LAN assumption is the
//! same one the token already makes.

use std::collections::BTreeMap;
use std::fs::{self, File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};
use serde_json::{json, Value as Json};

pub const LEASE_SCHEMA: &str = "jotnar.lease/1";
pub const ADMISSION_SCHEMA: &str = "jotnar.fleet.admission/1";
pub const WAITER_SCHEMA: &str = "jotnar.fleet.waiter/1";
pub const ADMISSION_RECORD_SCHEMA: &str = "jotnar.fleet.admission_record/1";

/// An interactive or batch reservation on one executor target.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Lease {
    pub schema: String,
    pub id: String,
    pub owner: String,
    pub class: String,
    pub target: String,
    pub slots: u32,
    pub created_ms: u64,
    pub renewed_ms: u64,
    pub expires_ms: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pid: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub note: Option<String>,
}

/// The evidence for one stage's admission decision.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AdmissionRecord {
    pub schema: String,
    pub target: String,
    pub class: String,
    pub decision: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub admission_id: Option<String>,
    pub waited_ms: u64,
    pub attempts: u32,
    pub deferrals: u32,
    pub reason: String,
    pub capacity: u32,
    pub interactive_slots: u32,
    pub in_flight_batch: u32,
    #[serde(default)]
    pub interactive_leases: Vec<Lease>,
}

impl AdmissionRecord {
    pub fn json(&self) -> Json {
        serde_json::to_value(self).unwrap_or_else(|_| json!({}))
    }
}

/// A held batch admission. Dropping it releases the slot.
#[derive(Debug)]
pub struct AdmissionGuard {
    record: AdmissionRecord,
    path: PathBuf,
}

impl AdmissionGuard {
    pub fn record(&self) -> &AdmissionRecord {
        &self.record
    }
}

impl Drop for AdmissionGuard {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.path);
    }
}

/// A renewal-capable lease held by this process.
pub struct LeaseGuard {
    dir: PathBuf,
    lease: Lease,
    revoked: bool,
}

impl LeaseGuard {
    pub fn lease(&self) -> &Lease {
        &self.lease
    }

    /// Renew the lease and push its expiry out. Returns an error instead of
    /// panicking so a coordinator can record a renewal failure.
    pub fn renew(&mut self, lifetime_ms: u64) -> Result<(), String> {
        let now = now_ms();
        self.lease.renewed_ms = now;
        self.lease.expires_ms = now.saturating_add(lifetime_ms.max(1));
        write_json_atomic(&lease_path(&self.dir, &self.lease.id), &self.lease)
    }

    pub fn revoke(&mut self) {
        if !self.revoked {
            let _ = fs::remove_file(lease_path(&self.dir, &self.lease.id));
            self.revoked = true;
        }
    }
}

impl Drop for LeaseGuard {
    fn drop(&mut self) {
        self.revoke();
    }
}

#[derive(Debug)]
pub enum AdmissionError {
    Rejected(AdmissionRecord),
    Io(String),
}

impl AdmissionError {
    pub fn record(&self) -> Option<&AdmissionRecord> {
        match self {
            AdmissionError::Rejected(record) => Some(record),
            AdmissionError::Io(_) => None,
        }
    }

    pub fn message(&self) -> String {
        match self {
            AdmissionError::Rejected(record) => format!(
                "admission rejected for target {}: {}",
                record.target, record.reason
            ),
            AdmissionError::Io(e) => e.clone(),
        }
    }
}

/// Controller-side handle on the shared fleet directory.
#[derive(Clone, Debug)]
pub struct Fleet {
    dir: PathBuf,
    owner: String,
    capacity: u32,
    max_wait: Duration,
    hold_ms: u64,
}

impl Fleet {
    /// `capacity` is the controller's declared batch+interactive slot count per
    /// target. `max_wait_ms` turns indefinite deferral into a recorded
    /// rejection. `hold_ms` bounds an admission if the process dies.
    pub fn new(
        dir: impl Into<PathBuf>,
        owner: impl Into<String>,
        capacity: u32,
        max_wait_ms: u64,
        hold_ms: u64,
    ) -> Self {
        Self {
            dir: dir.into(),
            owner: owner.into(),
            capacity: capacity.max(1),
            max_wait: Duration::from_millis(max_wait_ms.max(1)),
            hold_ms: hold_ms.max(1),
        }
    }

    pub fn dir(&self) -> &Path {
        &self.dir
    }

    pub fn owner(&self) -> &str {
        &self.owner
    }

    pub fn capacity(&self) -> u32 {
        self.capacity
    }

    pub fn max_wait_ms(&self) -> u64 {
        self.max_wait.as_millis() as u64
    }

    pub fn ensure_dirs(&self) -> Result<(), String> {
        for sub in ["leases", "waiters", "admissions"] {
            fs::create_dir_all(self.dir.join(sub))
                .map_err(|e| format!("create fleet dir {}: {e}", self.dir.display()))?;
        }
        Ok(())
    }

    /// Register an interactive or batch lease and keep it until revoked or
    /// dropped. The caller must renew it before it expires.
    pub fn register_lease(
        &self,
        class: &str,
        target: &str,
        slots: u32,
        lifetime_ms: u64,
        note: Option<&str>,
    ) -> Result<LeaseGuard, String> {
        self.ensure_dirs()?;
        let now = now_ms();
        let lease = Lease {
            schema: LEASE_SCHEMA.to_string(),
            id: format!(
                "{}-{}-{}-{}",
                safe_component(&self.owner),
                target,
                std::process::id(),
                now
            ),
            owner: self.owner.clone(),
            class: class.to_string(),
            target: target.to_string(),
            slots: slots.max(1),
            created_ms: now,
            renewed_ms: now,
            expires_ms: now.saturating_add(lifetime_ms.max(1)),
            pid: Some(std::process::id()),
            note: note.map(str::to_string),
        };
        write_json_atomic(&lease_path(&self.dir, &lease.id), &lease)?;
        Ok(LeaseGuard {
            dir: self.dir.clone(),
            lease,
            revoked: false,
        })
    }

    /// Wait for a batch admission on `target`, or reject after the configured
    /// wait. `stage` is evidence only; it does not affect ordering.
    pub fn acquire_batch(
        &self,
        target: &str,
        stage: &str,
    ) -> Result<AdmissionGuard, AdmissionError> {
        self.ensure_dirs().map_err(AdmissionError::Io)?;
        let started = Instant::now();
        let waiter = Waiter::new(&self.owner, target, stage);
        let waiter_path = waiter_path(&self.dir, &waiter.id);
        write_json_atomic(&waiter_path, &waiter).map_err(AdmissionError::Io)?;

        let mut attempts = 0u32;
        let mut deferrals = 0u32;
        let mut last_reason = "no batch slot available".to_string();
        let mut seen_leases: BTreeMap<String, Lease> = BTreeMap::new();
        let mut max_interactive_slots = 0u32;

        loop {
            attempts += 1;
            let locked = FileLock::acquire(&self.dir.join("fleet.lock"), Duration::from_secs(10));
            let held = match locked {
                Ok(lock) => lock,
                Err(e) => {
                    let _ = fs::remove_file(&waiter_path);
                    return Err(AdmissionError::Io(e));
                }
            };
            let view = match self.inspect(target) {
                Ok(view) => view,
                Err(e) => {
                    drop(held);
                    let _ = fs::remove_file(&waiter_path);
                    return Err(AdmissionError::Io(e));
                }
            };
            for lease in &view.interactive_leases {
                seen_leases.insert(lease.id.clone(), lease.clone());
            }
            max_interactive_slots = max_interactive_slots.max(view.interactive_slots);
            if view.in_flight_batch < view.available_batch() && view.oldest_waiter(&waiter.id) {
                let admission = Admission {
                    schema: ADMISSION_SCHEMA.to_string(),
                    id: waiter.id.clone(),
                    owner: self.owner.clone(),
                    class: "batch".to_string(),
                    target: target.to_string(),
                    stage: stage.to_string(),
                    granted_ms: now_ms(),
                    expires_ms: now_ms().saturating_add(self.hold_ms),
                };
                let path = admission_path(&self.dir, &admission.id);
                if let Err(e) = write_json_atomic(&path, &admission) {
                    drop(held);
                    let _ = fs::remove_file(&waiter_path);
                    return Err(AdmissionError::Io(e));
                }
                let _ = fs::remove_file(&waiter_path);
                drop(held);
                let record = AdmissionRecord {
                    schema: ADMISSION_RECORD_SCHEMA.to_string(),
                    target: target.to_string(),
                    class: "batch".to_string(),
                    decision: "admitted".to_string(),
                    admission_id: Some(admission.id.clone()),
                    waited_ms: started.elapsed().as_millis() as u64,
                    attempts,
                    deferrals,
                    reason: if deferrals == 0 {
                        "batch slot available".to_string()
                    } else {
                        format!("batch slot available after {deferrals} deferral(s): {last_reason}")
                    },
                    capacity: self.capacity,
                    interactive_slots: max_interactive_slots.max(view.interactive_slots),
                    in_flight_batch: view.in_flight_batch,
                    interactive_leases: seen_leases.values().cloned().collect(),
                };
                self.sweep();
                return Ok(AdmissionGuard { record, path });
            }

            last_reason = view.reason(target, &waiter.id);
            // Keep the waiter alive with a fresh seen_ms so a dead peer is
            // reaped quickly without expiring a live one.
            let mut alive = waiter.clone();
            alive.seen_ms = now_ms();
            alive.expires_ms = alive.seen_ms.saturating_add(WAITER_TTL_MS);
            let _ = write_json_atomic(&waiter_path, &alive);
            deferrals += 1;
            drop(held);

            if started.elapsed() >= self.max_wait {
                let _ = fs::remove_file(&waiter_path);
                let record = AdmissionRecord {
                    schema: ADMISSION_RECORD_SCHEMA.to_string(),
                    target: target.to_string(),
                    class: "batch".to_string(),
                    decision: "rejected".to_string(),
                    admission_id: None,
                    waited_ms: started.elapsed().as_millis() as u64,
                    attempts,
                    deferrals,
                    reason: format!(
                        "admission wait exceeded {} ms: {last_reason}",
                        self.max_wait.as_millis()
                    ),
                    capacity: self.capacity,
                    interactive_slots: max_interactive_slots.max(view.interactive_slots),
                    in_flight_batch: view.in_flight_batch,
                    interactive_leases: seen_leases.values().cloned().collect(),
                };
                self.sweep();
                return Err(AdmissionError::Rejected(record));
            }

            // A short, slightly increasing backoff: enough to let one stage
            // finish without spinning a core in the controller.
            let delay = 20u64.saturating_add(u64::from(attempts.min(8)) * 5);
            thread::sleep(Duration::from_millis(delay));
        }
    }

    fn inspect(&self, target: &str) -> Result<Inspect, String> {
        let now = now_ms();
        let mut interactive_slots = 0u32;
        let mut interactive_leases = Vec::new();
        for path in json_files(&self.dir.join("leases"))? {
            match read_json::<Lease>(&path) {
                Ok(lease) => {
                    if lease.expires_ms <= now {
                        let _ = fs::remove_file(&path);
                        continue;
                    }
                    if lease.class == "interactive" && lease.target == target {
                        interactive_slots = interactive_slots.saturating_add(lease.slots);
                        interactive_leases.push(lease);
                    }
                }
                Err(_) => {
                    // A half-written or foreign file must not wedge admission.
                    let _ = fs::remove_file(&path);
                }
            }
        }

        let mut in_flight_batch = 0u32;
        for path in json_files(&self.dir.join("admissions"))? {
            match read_json::<Admission>(&path) {
                Ok(admission) => {
                    if admission.expires_ms <= now {
                        let _ = fs::remove_file(&path);
                        continue;
                    }
                    if admission.class == "batch" && admission.target == target {
                        in_flight_batch = in_flight_batch.saturating_add(1);
                    }
                }
                Err(_) => {
                    let _ = fs::remove_file(&path);
                }
            }
        }

        let mut waiters = Vec::new();
        for path in json_files(&self.dir.join("waiters"))? {
            match read_json::<Waiter>(&path) {
                Ok(waiter) => {
                    if waiter.expires_ms <= now {
                        let _ = fs::remove_file(&path);
                        continue;
                    }
                    if waiter.target == target {
                        waiters.push(waiter);
                    }
                }
                Err(_) => {
                    let _ = fs::remove_file(&path);
                }
            }
        }
        waiters
            .sort_by(|a, b| (a.requested_ms, a.id.as_str()).cmp(&(b.requested_ms, b.id.as_str())));
        Ok(Inspect {
            capacity: self.capacity,
            interactive_slots,
            in_flight_batch,
            interactive_leases,
            waiters,
        })
    }

    /// Best-effort cleanup of expired artifacts outside the lock.
    pub fn sweep(&self) {
        let now = now_ms();
        for sub in ["leases", "admissions", "waiters"] {
            let Ok(paths) = json_files(&self.dir.join(sub)) else {
                continue;
            };
            for path in paths {
                let expired = match sub {
                    "leases" => read_json::<Lease>(&path)
                        .map(|v| v.expires_ms <= now)
                        .unwrap_or(true),
                    "admissions" => read_json::<Admission>(&path)
                        .map(|v| v.expires_ms <= now)
                        .unwrap_or(true),
                    _ => read_json::<Waiter>(&path)
                        .map(|v| v.expires_ms <= now)
                        .unwrap_or(true),
                };
                if expired {
                    let _ = fs::remove_file(&path);
                }
            }
        }
    }
}

struct Inspect {
    capacity: u32,
    interactive_slots: u32,
    in_flight_batch: u32,
    interactive_leases: Vec<Lease>,
    waiters: Vec<Waiter>,
}

impl Inspect {
    fn available_batch(&self) -> u32 {
        self.capacity.saturating_sub(self.interactive_slots)
    }

    fn oldest_waiter(&self, id: &str) -> bool {
        self.waiters.first().map(|w| w.id == id).unwrap_or(false)
    }

    fn reason(&self, target: &str, id: &str) -> String {
        if self.interactive_slots >= self.capacity {
            let owners = self
                .interactive_leases
                .iter()
                .map(|l| format!("{}:{}", l.owner, l.slots))
                .collect::<Vec<_>>()
                .join(",");
            return format!(
                "target {target} has {} interactive slot(s) reserved by {owners}",
                self.interactive_slots
            );
        }
        if self.in_flight_batch >= self.available_batch() {
            return format!(
                "target {target} batch slots full ({}/{} in flight, {} interactive)",
                self.in_flight_batch,
                self.available_batch(),
                self.interactive_slots
            );
        }
        if !self.oldest_waiter(id) {
            return format!("target {target} is waiting for an older queued stage");
        }
        "batch slot not granted".to_string()
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct Waiter {
    schema: String,
    id: String,
    owner: String,
    target: String,
    stage: String,
    requested_ms: u64,
    seen_ms: u64,
    expires_ms: u64,
}

impl Waiter {
    fn new(owner: &str, target: &str, stage: &str) -> Self {
        static SEQ: AtomicU64 = AtomicU64::new(1);
        let seq = SEQ.fetch_add(1, Ordering::Relaxed);
        let now = now_ms();
        Self {
            schema: WAITER_SCHEMA.to_string(),
            id: format!(
                "{}-{}-{}-{}-{}",
                safe_component(owner),
                target,
                std::process::id(),
                seq,
                now
            ),
            owner: owner.to_string(),
            target: target.to_string(),
            stage: stage.to_string(),
            requested_ms: now,
            seen_ms: now,
            expires_ms: now.saturating_add(WAITER_TTL_MS),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct Admission {
    schema: String,
    id: String,
    owner: String,
    class: String,
    target: String,
    stage: String,
    granted_ms: u64,
    expires_ms: u64,
}

const WAITER_TTL_MS: u64 = 2_000;

struct FileLock {
    path: PathBuf,
}

impl FileLock {
    fn acquire(path: &Path, timeout: Duration) -> Result<Self, String> {
        let start = Instant::now();
        loop {
            match OpenOptions::new().write(true).create_new(true).open(path) {
                Ok(mut file) => {
                    let _ = writeln!(file, "{}", now_ms());
                    return Ok(Self {
                        path: path.to_path_buf(),
                    });
                }
                Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {
                    if let Ok(meta) = fs::metadata(path) {
                        if let Ok(modified) = meta.modified() {
                            if modified.elapsed().unwrap_or_default() > Duration::from_secs(10) {
                                let _ = fs::remove_file(path);
                                continue;
                            }
                        }
                    }
                    if start.elapsed() > timeout {
                        return Err(format!("fleet lock {} is busy", path.display()));
                    }
                    thread::sleep(Duration::from_millis(2));
                }
                Err(e) => return Err(format!("fleet lock {}: {e}", path.display())),
            }
        }
    }
}

impl Drop for FileLock {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.path);
    }
}

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

fn safe_component(value: &str) -> String {
    value
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '-' || c == '_' || c == '.' {
                c
            } else {
                '_'
            }
        })
        .collect()
}

fn lease_path(dir: &Path, id: &str) -> PathBuf {
    dir.join("leases").join(format!("{id}.json"))
}

fn admission_path(dir: &Path, id: &str) -> PathBuf {
    dir.join("admissions").join(format!("{id}.json"))
}

fn waiter_path(dir: &Path, id: &str) -> PathBuf {
    dir.join("waiters").join(format!("{id}.json"))
}

fn json_files(dir: &Path) -> Result<Vec<PathBuf>, String> {
    let mut out = Vec::new();
    let entries = match fs::read_dir(dir) {
        Ok(entries) => entries,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(out),
        Err(e) => return Err(format!("read {}: {e}", dir.display())),
    };
    for entry in entries {
        let entry = entry.map_err(|e| format!("read {}: {e}", dir.display()))?;
        let path = entry.path();
        if path.extension().and_then(|x| x.to_str()) == Some("json") {
            out.push(path);
        }
    }
    out.sort();
    Ok(out)
}

fn read_json<T: for<'de> Deserialize<'de>>(path: &Path) -> Result<T, String> {
    let bytes = fs::read(path).map_err(|e| format!("read {}: {e}", path.display()))?;
    serde_json::from_slice(&bytes).map_err(|e| format!("parse {}: {e}", path.display()))
}

fn write_json_atomic<T: Serialize>(path: &Path, value: &T) -> Result<(), String> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|e| format!("create {}: {e}", parent.display()))?;
    }
    let bytes = serde_json::to_vec_pretty(value).map_err(|e| e.to_string())?;
    let tmp = path.with_extension(format!("tmp-{}-{}", std::process::id(), now_ms()));
    {
        let mut file = File::create(&tmp).map_err(|e| format!("write {}: {e}", tmp.display()))?;
        file.write_all(&bytes)
            .map_err(|e| format!("write {}: {e}", tmp.display()))?;
    }
    fs::rename(&tmp, path).map_err(|e| format!("rename {}: {e}", path.display()))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_fleet(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "jotnar-fleet-test-{name}-{}-{}",
            std::process::id(),
            now_ms()
        ));
        let _ = fs::remove_dir_all(&dir);
        dir
    }

    #[test]
    fn lease_registration_and_expiry() {
        let dir = temp_fleet("lease");
        let fleet = Fleet::new(&dir, "test:1", 2, 500, 5_000);
        let mut guard = fleet
            .register_lease("interactive", "cuda", 1, 5_000, Some("unit"))
            .unwrap();
        let path = lease_path(&dir, &guard.lease().id);
        assert!(path.is_file());
        guard.renew(5_000).unwrap();
        assert!(read_json::<Lease>(&path).unwrap().renewed_ms >= guard.lease().created_ms);
        guard.revoke();
        assert!(!path.is_file());
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn interactive_lease_reserves_batch_capacity() {
        let dir = temp_fleet("reserve");
        let fleet = Fleet::new(&dir, "batch:1", 2, 100, 5_000);
        let _lease = fleet
            .register_lease("interactive", "cuda", 1, 30_000, None)
            .unwrap();
        // One batch slot remains. The first admission is immediate.
        let first = fleet.acquire_batch("cuda", "gpu-a-00").unwrap();
        let record = first.record().clone();
        assert_eq!(record.decision, "admitted");
        assert_eq!(record.interactive_slots, 1);
        assert_eq!(record.capacity, 2);
        assert_eq!(record.interactive_leases.len(), 1);
        drop(first);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_reserved_target_rejects_after_the_wait_budget() {
        let dir = temp_fleet("reject");
        let fleet = Fleet::new(&dir, "batch:2", 1, 120, 5_000);
        let _lease = fleet
            .register_lease("interactive", "cuda", 1, 30_000, None)
            .unwrap();
        match fleet.acquire_batch("cuda", "gpu-a-00") {
            Err(AdmissionError::Rejected(record)) => {
                assert_eq!(record.decision, "rejected");
                assert_eq!(record.interactive_slots, 1);
                assert!(record.reason.contains("interactive"));
            }
            other => panic!("expected rejection, got {other:?}"),
        }
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn waiters_are_granted_in_request_order() {
        let dir = temp_fleet("fifo");
        let a = Fleet::new(&dir, "batch:first", 1, 2_000, 5_000);
        let b = Fleet::new(&dir, "batch:second", 1, 2_000, 5_000);
        let held = a.acquire_batch("hip", "first-stage").unwrap();
        let waiter = std::thread::spawn(move || {
            let guard = b.acquire_batch("hip", "second-stage").unwrap();
            guard.record().waited_ms
        });
        // Let the second controller queue behind the held admission.
        thread::sleep(Duration::from_millis(80));
        drop(held);
        let waited = waiter.join().unwrap();
        assert!(
            waited >= 50,
            "second waiter should have waited, got {waited}"
        );
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn expired_admissions_do_not_block_forever() {
        let dir = temp_fleet("expire");
        let fleet = Fleet::new(&dir, "batch:3", 1, 2_000, 5_000);
        fleet.ensure_dirs().unwrap();
        let stale = Admission {
            schema: ADMISSION_SCHEMA.to_string(),
            id: "stale".to_string(),
            owner: "dead".to_string(),
            class: "batch".to_string(),
            target: "cuda".to_string(),
            stage: "old".to_string(),
            granted_ms: now_ms().saturating_sub(10_000),
            expires_ms: now_ms().saturating_sub(5_000),
        };
        write_json_atomic(&admission_path(&dir, "stale"), &stale).unwrap();
        let guard = fleet.acquire_batch("cuda", "new").unwrap();
        assert_eq!(guard.record().decision, "admitted");
        assert!(!admission_path(&dir, "stale").is_file());
        let _ = fs::remove_dir_all(&dir);
    }
}
