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
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::Duration;

/// Bytes requested from the allocator while [`TRACK_ALLOC`] is set.
pub static ALLOCATED_BYTES: AtomicU64 = AtomicU64::new(0);
/// Enables allocation counting. Off by default so normal runs are untouched.
pub static TRACK_ALLOC: AtomicBool = AtomicBool::new(false);

/// System allocator wrapper that totals requested bytes.
pub struct CountingAlloc;

impl CountingAlloc {
    #[inline]
    fn count(size: usize) {
        if TRACK_ALLOC.load(Ordering::Relaxed) {
            ALLOCATED_BYTES.fetch_add(size as u64, Ordering::Relaxed);
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
        System.dealloc(ptr, layout)
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        Self::count(new_size);
        System.realloc(ptr, layout, new_size)
    }
}

/// Total bytes requested since the process started (0 when tracking is off).
pub fn allocated() -> u64 {
    ALLOCATED_BYTES.load(Ordering::Relaxed)
}

/// Start/stop allocation counting. Returns the previous state.
pub fn set_track_alloc(on: bool) -> bool {
    TRACK_ALLOC.swap(on, Ordering::Relaxed)
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
