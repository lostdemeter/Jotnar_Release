//! Per-op profiling for the reference interpreter.
//!
//! The runtime is the measuring instrument, not the model. `Profile` records
//! wall time, allocated bytes, input bytes, and output bytes per mnemonic so
//! performance work starts from a measured budget. It is opt-in: `run` pays
//! only a boolean check per instruction.
//!
//! Allocation volume is tracked with [`CountingAlloc`], a thin wrapper over
//! the system allocator. Benchmark binaries opt in with
//!
//! ```ignore
//! #[global_allocator]
//! static ALLOC: jotnar::profile::CountingAlloc = jotnar::profile::CountingAlloc;
//! ```
//!
//! while libraries and tests keep the default allocator. Profiling is a
//! diagnostic; it never changes arithmetic or the determinism verdict.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::Duration;

/// Bytes requested from the allocator while [`TRACK_ALLOC`] is set.
pub static ALLOCATED_BYTES: AtomicU64 = AtomicU64::new(0);
/// Number of allocation requests while tracking is on.
pub static ALLOC_CALLS: AtomicU64 = AtomicU64::new(0);
/// Number of deallocations while tracking is on.
pub static DEALLOC_CALLS: AtomicU64 = AtomicU64::new(0);
/// Bytes requested but not yet released since tracking started.
pub static LIVE_BYTES: AtomicU64 = AtomicU64::new(0);
/// Enables allocation counting. Off by default so normal runs are untouched.
pub static TRACK_ALLOC: AtomicBool = AtomicBool::new(false);

/// System allocator wrapper that totals requested bytes and counts.
pub struct CountingAlloc;

impl CountingAlloc {
    #[inline]
    fn count(size: usize) {
        if tracking() {
            ALLOCATED_BYTES.fetch_add(size as u64, Ordering::Relaxed);
            ALLOC_CALLS.fetch_add(1, Ordering::Relaxed);
            LIVE_BYTES.fetch_add(size as u64, Ordering::Relaxed);
            let _ = LOCAL.try_with(|l| {
                l.allocs.set(l.allocs.get() + 1);
                l.bytes.set(l.bytes.get() + size as u64);
                l.live.set(l.live.get() + size as i64);
            });
        }
    }

    #[inline]
    fn release(size: usize) {
        if tracking() {
            DEALLOC_CALLS.fetch_add(1, Ordering::Relaxed);
            // Saturate: tracking may start after the block was allocated.
            let _ = LIVE_BYTES.try_update(Ordering::Relaxed, Ordering::Relaxed, |v| {
                Some(v.saturating_sub(size as u64))
            });
            let _ = LOCAL.try_with(|l| {
                l.deallocs.set(l.deallocs.get() + 1);
                l.live.set(l.live.get() - size as i64);
            });
        }
    }
}

unsafe impl GlobalAlloc for CountingAlloc {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        Self::count(layout.size());
        System.alloc(layout)
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        Self::count(layout.size());
        System.alloc_zeroed(layout)
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        Self::release(layout.size());
        System.dealloc(ptr, layout)
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        Self::count(new_size);
        Self::release(layout.size());
        System.realloc(ptr, layout, new_size)
    }
}

/// Per-thread counters. A scope's witness must not include another thread's
/// allocations: the harness runs tests in parallel and a node may run workers,
/// so a process-global delta would smear both.
///
/// Const-initialised so reading it from inside the allocator cannot allocate.
#[derive(Clone)]
struct Local {
    allocs: Cell<u64>,
    deallocs: Cell<u64>,
    bytes: Cell<u64>,
    live: Cell<i64>,
}

impl Local {
    const ZERO: Local = Local {
        allocs: Cell::new(0),
        deallocs: Cell::new(0),
        bytes: Cell::new(0),
        live: Cell::new(0),
    };
}

thread_local! {
    static LOCAL: Local = const { Local::ZERO };
    /// Tracking is per-thread. A process-global switch lets one thread's scope
    /// disarm another's (tests run in parallel, and a node runs workers).
    static TRACK_LOCAL: Cell<bool> = const { Cell::new(false) };
}

#[inline]
fn tracking() -> bool {
    TRACK_LOCAL.try_with(|c| c.get()).unwrap_or(false)
}

/// A point-in-time reading of this thread's allocation counters.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Snapshot {
    pub allocs: u64,
    pub deallocs: u64,
    pub bytes: u64,
    pub live: i64,
}

impl Snapshot {
    pub fn now() -> Self {
        LOCAL
            .try_with(|l| Self {
                allocs: l.allocs.get(),
                deallocs: l.deallocs.get(),
                bytes: l.bytes.get(),
                live: l.live.get(),
            })
            .unwrap_or_default()
    }

    /// What happened between `self` and `later`.
    pub fn delta(self, later: Snapshot) -> AllocWitness {
        AllocWitness {
            allocs: later.allocs.saturating_sub(self.allocs),
            deallocs: later.deallocs.saturating_sub(self.deallocs),
            bytes: later.bytes.saturating_sub(self.bytes),
            live_delta: later.live - self.live,
        }
    }
}

/// Allocation activity observed inside one [`JobScope`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct AllocWitness {
    pub allocs: u64,
    pub deallocs: u64,
    pub bytes: u64,
    /// Net change in live bytes over the scope (may be negative).
    pub live_delta: i64,
}

impl AllocWitness {
    pub fn is_zero_allocation(&self) -> bool {
        self.allocs == 0
    }

    /// Fail loud when a job exceeds its declared allocation ceiling.
    ///
    /// This is the enforceable half of the discipline: a number that is only
    /// printed can rot, a number that can fail a run cannot.
    pub fn check(&self, max: u64) -> crate::error::Result<()> {
        if self.allocs > max {
            return Err(crate::error::Error::runtime(format!(
                "alloc budget: {} allocations exceeds ceiling {max} ({} bytes)",
                self.allocs, self.bytes
            )));
        }
        Ok(())
    }

    pub fn json(&self) -> serde_json::Value {
        serde_json::json!({
            "allocs": self.allocs,
            "deallocs": self.deallocs,
            "bytes": self.bytes,
            "live_delta": self.live_delta,
        })
    }
}

/// A job-bounded allocation measurement.
///
/// `begin` turns tracking on (restoring the previous state on `finish`), takes
/// a snapshot, and `finish` turns that into an [`AllocWitness`]. Tracking is
/// process-global, so the witness describes the whole process for the scope's
/// duration: exact when jobs are serial (which a [`crate::protocol::Node`] is),
/// an upper bound when another thread runs concurrently.
pub struct JobScope {
    prior: Option<bool>,
    start: Snapshot,
}

impl JobScope {
    pub fn begin() -> Self {
        let prior = set_track_alloc(true);
        Self {
            prior: Some(prior),
            start: Snapshot::now(),
        }
    }

    pub fn finish(mut self) -> AllocWitness {
        let end = Snapshot::now();
        if let Some(prior) = self.prior.take() {
            set_track_alloc(prior);
        }
        self.start.delta(end)
    }
}

impl Drop for JobScope {
    fn drop(&mut self) {
        // An early return must not leave tracking armed.
        if let Some(prior) = self.prior.take() {
            set_track_alloc(prior);
        }
    }
}


/// Total bytes requested since the process started (0 when tracking is off).
pub fn allocated() -> u64 {
    ALLOCATED_BYTES.load(Ordering::Relaxed)
}

/// Start/stop allocation counting on this thread. Returns the previous state.
///
/// The authoritative switch is thread-local; [`TRACK_ALLOC`] is a mirror for
/// callers that only want to know whether anything is tracking.
pub fn set_track_alloc(on: bool) -> bool {
    let prev = TRACK_LOCAL
        .try_with(|c| c.replace(on))
        .unwrap_or(false);
    TRACK_ALLOC.store(on, Ordering::Relaxed);
    prev
}

#[derive(Clone, Copy, Debug, Default)]
pub struct OpProfile {
    pub calls: u64,
    pub wall: Duration,
    /// Requested allocation bytes observed while the op was executing.
    pub alloc_bytes: u64,
    /// Exact arithmetic bytes in the op's arguments.
    pub in_bytes: u64,
    /// Exact arithmetic bytes materialised by the op.
    pub out_bytes: u64,
}

#[derive(Clone, Debug, Default)]
pub struct Profile {
    pub ops: BTreeMap<String, OpProfile>,
    pub steps: u64,
}

impl Profile {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn record(
        &mut self,
        name: &str,
        wall: Duration,
        alloc_bytes: u64,
        in_bytes: u64,
        out_bytes: u64,
    ) {
        self.steps += 1;
        let e = self.ops.entry(name.to_string()).or_default();
        e.calls += 1;
        e.wall += wall;
        e.alloc_bytes += alloc_bytes;
        e.in_bytes += in_bytes;
        e.out_bytes += out_bytes;
    }

    pub fn total_wall(&self) -> Duration {
        self.ops.values().map(|v| v.wall).sum()
    }

    /// One report line per op, slowest first. Percentages are of measured
    /// op time (not process wall time).
    pub fn report(&self) -> String {
        let total = self.total_wall().as_secs_f64();
        let mut rows: Vec<(&String, &OpProfile)> = self.ops.iter().collect();
        rows.sort_by(|a, b| b.1.wall.cmp(&a.1.wall));
        let mut out = String::new();
        out.push_str(&format!(
            "profile: {} instrs, {:.3}s in ops\n",
            self.steps, total
        ));
        out.push_str(&format!(
            "  {:<14} {:>6} {:>11} {:>7} {:>11} {:>11}\n",
            "op", "calls", "wall", "share", "allocated", "output"
        ));
        for (name, p) in rows {
            let share = if total > 0.0 {
                p.wall.as_secs_f64() / total * 100.0
            } else {
                0.0
            };
            out.push_str(&format!(
                "  {:<14} {:>6} {:>10.3}s {:>6.1}% {:>11} {:>11}\n",
                name,
                p.calls,
                p.wall.as_secs_f64(),
                share,
                human_bytes(p.alloc_bytes),
                human_bytes(p.out_bytes)
            ));
        }
        out
    }
}

pub fn human_bytes(n: u64) -> String {
    const UNITS: [&str; 5] = ["B", "KiB", "MiB", "GiB", "TiB"];
    let mut v = n as f64;
    let mut i = 0;
    while v >= 1024.0 && i + 1 < UNITS.len() {
        v /= 1024.0;
        i += 1;
    }
    if i == 0 {
        format!("{n} B")
    } else {
        format!("{v:.2} {}", UNITS[i])
    }
}
