//! A job-scoped bump arena for boundary images.
//!
//! This is the runtime-internal half of the "a job is a scope" discipline.
//! One [`JobArena`] lives for the lifetime of a [`crate::protocol::Node`];
//! each job calls [`JobArena::reset`], which is a single store, and allocates
//! from the top. Steady-state allocation is zero: the buffer grows only until
//! the largest job seen so far has run, and never moves afterwards.
//!
//! Regions are handed out as **offsets, not pointers**, so a later `alloc`
//! that grows the buffer cannot invalidate an earlier region. That is the
//! property a data-stack model buys, and here it is nearly free because a
//! `Vec<u8>` index is already stable across reallocation.
//!
//! What this does *not* do is make the interpreter allocation-free. The
//! interpreter still materialises an `ndarray` per op; the arena covers the
//! byte images at the boundary. The witness tells you which is which.

use crate::error::{Error, Result};

/// A bump arena reset once per job.
#[derive(Clone, Debug, Default)]
pub struct JobArena {
    buf: Vec<u8>,
    used: usize,
    peak: usize,
    resets: u64,
}

impl JobArena {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_capacity(n: usize) -> Self {
        Self {
            buf: Vec::with_capacity(n),
            ..Self::default()
        }
    }

    /// Start a new job: one store. Returns the bytes reclaimed.
    pub fn reset(&mut self) -> usize {
        let reclaimed = self.used;
        self.used = 0;
        self.resets += 1;
        reclaimed
    }

    /// Reserve `len` bytes and return the region's offset.
    ///
    /// Growth only happens when a job exceeds every previous job; after that
    /// this is a pointer bump.
    pub fn alloc(&mut self, len: usize) -> Result<usize> {
        let off = self.used;
        let end = off
            .checked_add(len)
            .ok_or_else(|| Error::runtime(format!("arena: {off} + {len} overflows")))?;
        if end > self.buf.len() {
            self.buf.resize(end, 0);
        }
        self.used = end;
        self.peak = self.peak.max(end);
        Ok(off)
    }

    pub fn slice(&self, off: usize, len: usize) -> &[u8] {
        &self.buf[off..off + len]
    }

    pub fn slice_mut(&mut self, off: usize, len: usize) -> &mut [u8] {
        &mut self.buf[off..off + len]
    }

    /// Every byte allocated in this job, in allocation order.
    pub fn used_slice(&self) -> &[u8] {
        &self.buf[..self.used]
    }

    pub fn used(&self) -> usize {
        self.used
    }

    pub fn peak(&self) -> usize {
        self.peak
    }

    pub fn capacity(&self) -> usize {
        self.buf.capacity()
    }

    pub fn resets(&self) -> u64 {
        self.resets
    }

    /// The arena half of a job witness.
    pub fn json(&self) -> serde_json::Value {
        serde_json::json!({
            "used": self.used,
            "peak": self.peak,
            "capacity": self.capacity(),
            "resets": self.resets,
            // True once capacity has stopped growing: the next job cannot
            // allocate regardless of its size, up to `capacity`.
            "warm": self.capacity() >= self.peak && self.resets > 0,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn offsets_survive_growth() {
        let mut a = JobArena::new();
        let first = a.alloc(4).unwrap();
        a.slice_mut(first, 4).copy_from_slice(b"keep");
        // Force a reallocation by exceeding capacity.
        let big = a.alloc(1 << 20).unwrap();
        assert!(big > first);
        // `first` is an offset, so it still reads the same bytes.
        assert_eq!(a.slice(first, 4), b"keep");
    }

    #[test]
    fn reset_is_one_store_and_reuses_capacity() {
        let mut a = JobArena::new();
        let off = a.alloc(1024).unwrap();
        assert_eq!(off, 0);
        let cap = a.capacity();
        assert_eq!(a.reset(), 1024);
        assert_eq!(a.used(), 0);
        assert_eq!(a.capacity(), cap, "reset must not free the buffer");
        let off2 = a.alloc(1024).unwrap();
        assert_eq!(off2, 0, "the same job shape reuses the same offset");
        assert_eq!(a.resets(), 1);
    }

    #[test]
    fn peak_tracks_high_water_mark() {
        let mut a = JobArena::new();
        a.alloc(10).unwrap();
        a.reset();
        a.alloc(4).unwrap();
        assert_eq!(a.peak(), 10);
        assert_eq!(a.used(), 4);
        assert!(a.json()["warm"].as_bool().unwrap());
    }

    #[test]
    fn overflow_is_rejected() {
        let mut a = JobArena::new();
        a.alloc(8).unwrap();
        assert!(a.alloc(usize::MAX).is_err());
    }
}
