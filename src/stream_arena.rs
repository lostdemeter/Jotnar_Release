//! Stream arena: reuse one buffer for a whole listing's working set.
//!
//! The arena is an opt-in execution path (`JOTNAR_ARENA=1`), not a second
//! language. It covers I and F streams and 53 kernels; anything else falls
//! back to the interpreter before the first instruction runs.
//!
//! The shape is the one the design note argued for:
//!
//! * [`Plan::compile`] uses the real assembler and [`infer_shapes`], so every
//!   stream's element count comes from the listing, not from a guess. It
//!   allocates one slot per stream and records the offsets.
//! * [`StreamArena`] is one `Vec<i64>`. Each job resets the watermark and
//!   re-bumps to the same total, so a warm job allocates nothing.
//! * Slots are allocated in program order, so an instruction's destination is
//!   always at a higher offset than both of its operands. That lets `run`
//!   split one buffer into `&mut dst` and two immutable operand slices with no
//!   `unsafe` and no copy.
//!
//! The measurement this exists for is in `tests/discipline.rs`: the same chain
//! listing that costs ~24 allocations per mnemonic through `Value`/`ArrayD`
//! costs **zero** here after the first job.
//!
//! # Dispatch
//!
//! Porting a mnemonic is two lines: write a `Kernel` and add it to
//! [`kernel_for`]. Arity is **not** duplicated here — it comes from
//! [`crate::ops::registry`], so a mnemonic that exists in the language but has
//! no kernel is reported as `Unimplemented` rather than silently missing.
//! Kernels receive a [`Ctx`] split into an immutable operand prefix and a
//! contiguous mutable output block, so no kernel needs `unsafe` or a copy.

use std::cell::Cell;
use std::collections::HashMap;
use std::sync::OnceLock;

use crate::codegen::infer::{infer_shapes, CKind};
use crate::error::{Error, Result};
use crate::ir::BoundProgram;
use crate::payload::Feeds;
use crate::value::Value;

thread_local! {
    /// Per-thread override; `None` falls back to the process default.
    static ENABLED: Cell<Option<bool>> = const { Cell::new(None) };
    /// Per-thread arena size cap in bytes; `None` uses the process default.
    static MAX_BYTES: Cell<Option<usize>> = const { Cell::new(None) };
}

fn default_max_bytes() -> usize {
    static DEFAULT: OnceLock<usize> = OnceLock::new();
    *DEFAULT.get_or_init(|| {
        std::env::var("JOTNAR_ARENA_MAX_MB")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .filter(|mb| *mb > 0)
            .map(|mb| mb.saturating_mul(1024 * 1024))
            // A listing whose whole working set does not fit falls back to the
            // interpreter. Sized for a full image pipeline; override for larger.
            .unwrap_or(64 * 1024 * 1024)
    })
}

/// Largest working set the arena will take on, in bytes.
///
/// The arena grows to the largest job it has seen and never shrinks, so a
/// listing with a big image per stream would hold that memory for the life of
/// the node. Compilation refuses anything above this cap and the caller falls
/// back to the interpreter.
pub fn max_bytes() -> usize {
    MAX_BYTES
        .with(|c| c.get())
        .unwrap_or_else(default_max_bytes)
}

/// Override the arena size cap for this thread. Returns the previous override.
pub fn set_max_bytes(v: Option<usize>) -> Option<usize> {
    MAX_BYTES.with(|c| c.replace(v))
}

fn default_enabled() -> bool {
    static DEFAULT: OnceLock<bool> = OnceLock::new();
    *DEFAULT.get_or_init(|| matches!(std::env::var("JOTNAR_ARENA").as_deref(), Ok("1")))
}

/// Whether the runtime should try the arena path first.
///
/// Off by default: `JOTNAR_ARENA=1` (or `set_enabled`) turns it on. The
/// interpreter stays the reference and the fallback.
pub fn enabled() -> bool {
    ENABLED.with(|c| c.get()).unwrap_or_else(default_enabled)
}

/// Override the arena switch for this thread. Returns the previous override.
pub fn set_enabled(v: Option<bool>) -> Option<bool> {
    ENABLED.with(|c| c.replace(v))
}

/// One bump arena of `i64` words, reset per job.
#[derive(Clone, Debug, Default)]
pub struct StreamArena {
    words: Vec<i64>,
    used: usize,
    peak: usize,
    resets: u64,
}

impl StreamArena {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_capacity_words(n: usize) -> Self {
        Self {
            words: vec![0; n],
            ..Self::default()
        }
    }

    /// Start a new job. One store, exactly like the byte arena.
    pub fn reset(&mut self) -> usize {
        let reclaimed = std::mem::take(&mut self.used);
        self.resets += 1;
        reclaimed
    }

    /// Bump `len` words and return the word offset.
    pub fn alloc(&mut self, len: usize) -> Result<usize> {
        let off = self.used;
        let end = off
            .checked_add(len)
            .ok_or_else(|| Error::runtime(format!("stream arena: {off} + {len} overflows")))?;
        if end > self.words.len() {
            self.words.resize(end, 0);
        }
        self.used = end;
        self.peak = self.peak.max(end);
        Ok(off)
    }

    /// Reserve capacity so the first job does not allocate.
    pub fn reserve_words(&mut self, n: usize) {
        if self.bytes_len() < n {
            self.words.resize(n, 0);
        }
    }

    fn bytes_len(&self) -> usize {
        self.words.len()
    }

    pub fn words(&self) -> &[i64] {
        &self.words[..self.used]
    }

    fn words_mut(&mut self) -> &mut [i64] {
        let end = self.used;
        &mut self.words[..end]
    }

    pub fn used_words(&self) -> usize {
        self.used
    }

    pub fn used_bytes(&self) -> usize {
        self.used * 8
    }

    pub fn capacity_bytes(&self) -> usize {
        self.words.len() * 8
    }

    pub fn peak_bytes(&self) -> usize {
        self.peak * 8
    }

    pub fn resets(&self) -> u64 {
        self.resets
    }

    pub fn json(&self) -> serde_json::Value {
        serde_json::json!({
            "used": self.used_bytes(),
            "peak": self.peak_bytes(),
            "capacity": self.capacity_bytes(),
            "resets": self.resets,
        })
    }
}

/// A compiled operand: a slot into the arena, or a literal.
#[derive(Clone, Copy, Debug)]
pub enum Arg {
    /// `dims` is `(start, rank)` into `Plan::shapes`.
    Slot {
        off: usize,
        len: usize,
        dims: (usize, usize),
        /// True for an F stream (words are `f64` bits).
        is_f: bool,
    },
    Const(i64),
    /// A non-integer literal such as `INTERP`'s `0.5` scale.
    ConstF(f64),
}

/// One stream's slot: offset, element count, and where its dims live.
#[derive(Clone, Copy, Debug)]
struct Slot {
    off: usize,
    len: usize,
    dims: (usize, usize),
    is_f: bool,
}

/// What a kernel sees.
///
/// `head` is everything below this instruction's destination block, so every
/// operand is an immutable view. `out` is the contiguous destination block.
/// The two never overlap, which is why a kernel needs no `unsafe` and no copy.
pub struct Ctx<'a> {
    head: &'a [i64],
    out: &'a mut [i64],
    /// Flattened dims: each operand's `dims` span indexes into this.
    shapes: &'a [usize],
}

impl<'a> Ctx<'a> {
    pub fn out_len(&self) -> usize {
        self.out.len()
    }

    /// Split into the immutable prefix, the mutable output block, and the
    /// shape table. All three are disjoint fields.
    pub fn parts(&mut self) -> (&[i64], &mut [i64], &[usize]) {
        (self.head, &mut *self.out, self.shapes)
    }
}

fn dims_of(a: Arg, shapes: &[usize]) -> &[usize] {
    match a {
        Arg::Slot { dims: (s, r), .. } => &shapes[s..s + r],
        Arg::Const(_) | Arg::ConstF(_) => &[],
    }
}

fn off_len(a: Arg) -> Result<(usize, usize)> {
    match a {
        Arg::Slot { off, len, .. } => Ok((off, len)),
        Arg::Const(_) | Arg::ConstF(_) => Err(Error::runtime(
            "stream arena: expected a stream operand, got a literal",
        )),
    }
}

/// Arena words are raw 64-bit patterns: `i64` for I streams, `f64` bits for F.
#[inline]
fn f64_of(word: i64) -> f64 {
    f64::from_bits(word as u64)
}

#[inline]
fn word_of(x: f64) -> i64 {
    x.to_bits() as i64
}

#[inline]
fn arg_f64(head: &[i64], a: Arg, i: usize) -> f64 {
    f64_of(arg(head, a, i))
}

#[inline]
fn arg(head: &[i64], a: Arg, i: usize) -> i64 {
    match a {
        Arg::Const(c) => c,
        Arg::ConstF(x) => x as i64,
        Arg::Slot { off, .. } => head[off + i],
    }
}

fn slot<'a>(head: &'a [i64], a: Arg) -> Result<&'a [i64]> {
    match a {
        Arg::Const(_) | Arg::ConstF(_) => Err(Error::runtime(
            "stream arena: expected a stream operand, got a literal",
        )),
        Arg::Slot { off, len, .. } => Ok(&head[off..off + len]),
    }
}

fn literal(a: Arg) -> Result<i64> {
    match a {
        Arg::Const(c) => Ok(c),
        Arg::ConstF(_) | Arg::Slot { .. } => Err(Error::runtime(
            "stream arena: expected a literal operand, got a stream",
        )),
    }
}

fn literal_f64(a: Arg) -> Result<f64> {
    match a {
        Arg::Const(c) => Ok(c as f64),
        Arg::ConstF(x) => Ok(x),
        Arg::Slot { .. } => Err(Error::runtime(
            "stream arena: expected a literal operand, got a stream",
        )),
    }
}

/// One mnemonic's kernel. This is the unit the grind ports: a small function
/// that reads operands out of `head` and writes into `out`.
pub type Kernel = fn(&mut Ctx<'_>, &[Arg], &crate::ir::Config) -> Result<()>;

/// The dispatch table. Arity is taken from `ops::registry()`, so a new mnemonic
/// is reported as "not ported" here rather than silently missing.
///
/// The `usize` is how many leading operands must have the output's element
/// count. `2` for binary elementwise, `1` for shift-by-literal, `3` for
/// three-operand elementwise, `0` for reductions and encoders.
fn kernel_for(mn: &str) -> Option<(Kernel, usize)> {
    Some(match mn {
        // --- binary elementwise -------------------------------------------
        "IADD" => (k_add, 2),
        "ISUB" => (k_sub, 2),
        "IMUL" => (k_mul, 2),
        "IDIV" => (k_div, 2),
        "IMOD" => (k_mod, 2),
        "IDIVU" => (k_divu, 2),
        "IMODU" => (k_modu, 2),
        "IAND" => (k_and, 2),
        "IOR" => (k_or, 2),
        "IXOR" => (k_xor, 2),
        "IEQ" => (k_eq, 2),
        "ILT" => (k_lt, 2),
        "IMAX" => (k_max, 2),
        "SADD" => (k_sadd, 2),
        "ISAR" => (k_sar, 2),
        // --- shift by literal ---------------------------------------------
        "ISHL" => (k_shl, 1),
        "ISHR" => (k_shr, 1),
        // --- three-operand elementwise ------------------------------------
        "IMAD" => (k_mad, 3),
        "IMATMUL" => (k_imatmul, 0),
        "SLICE" => (k_slice, 0),
        // --- F linear algebra / norms / rotary -----------------------------
        "MATMUL" => (k_fmatmul, 0),
        "BATCH_MATMUL" => (k_fmatmul, 0),
        "LAYERNORM" => (k_layernorm, 1),
        "RMSNORM" => (k_rmsnorm, 1),
        "ROTARY" => (k_rotary, 1),
        // --- shape-changing moves ------------------------------------------
        "RESHAPE2" => (k_reshape, 1),
        "RESHAPE3" => (k_reshape, 1),
        "CONV" => (k_conv, 1),
        "INTERP" => (k_interp, 0),
        "GATHER" => (k_gather, 0),
        "GATHER3" => (k_gather3, 0),
        "SCATTER" => (k_scatter, 1),
        "ARGMAX" => (k_argmax, 0),
        "TOPK" => (k_topk, 0),
        // --- indexing, windows, encoders -----------------------------------
        "IGET" => (k_iget, 0),
        "IWINDOW" => (k_iwindow, 0),
        "IFIND" => (k_ifind, 0),
        "IGATHER1" => (k_igather1, 0),
        "ISCATTER1" => (k_iscatter1, 0),
        "IASSEMBLE" => (k_iassemble, 0),
        "ICRC32" => (k_icrc32, 0),
        "ICRC32C" => (k_icrc32c, 0),
        "ISELECT" => (k_select, 3),
        // --- shape-changing / encoding ------------------------------------
        "IREDUCE_SUM" => (k_reduce_sum, 0),
        "ISUM" => (k_window_sum, 0),
        "IVARINT" => (k_varint, 0),
        "CONCAT" => (k_concat, 0),
        // F streams: the language has no F elemental arithmetic; these are the
        // conversion/activation paths that reach F.
        "RELU" => (k_relu, 1),
        "RESIZE2D" => (k_resize2d, 0),
        "CONV2D" => (k_conv2d, 0),
        "DECONV2D" => (k_deconv2d, 0),
        "F2I" => (k_f2i, 1),
        "I2F" => (k_i2f, 1),
        _ => return None,
    })
}

#[inline]
fn binop(c: &mut Ctx, a: &[Arg], f: impl Fn(i64, i64) -> i64) -> Result<()> {
    let n = c.out_len();
    let (head, out, _) = c.parts();
    for i in 0..n {
        out[i] = f(arg(head, a[0], i), arg(head, a[1], i));
    }
    Ok(())
}

/// A scalar operand: a literal, or a one-element stream.
fn scalar(head: &[i64], a: Arg) -> Result<i64> {
    match a {
        Arg::Const(c) => Ok(c),
        Arg::ConstF(x) => Ok(x as i64),
        Arg::Slot { off, len, .. } => {
            if len != 1 {
                return Err(Error::runtime(format!(
                    "stream arena: expected a scalar operand, got {len} elements"
                )));
            }
            Ok(head[off])
        }
    }
}

fn shift(s: i64, op: &str) -> Result<u32> {
    if !(0..=63).contains(&s) {
        return Err(Error::runtime(format!(
            "{op}: shift must be 0..63, got {s}"
        )));
    }
    Ok(s as u32)
}

fn k_add(c: &mut Ctx, a: &[Arg], _: &crate::ir::Config) -> Result<()> {
    binop(c, a, |x, y| x.wrapping_add(y))
}

fn k_sub(c: &mut Ctx, a: &[Arg], _: &crate::ir::Config) -> Result<()> {
    binop(c, a, |x, y| x.wrapping_sub(y))
}

fn k_mul(c: &mut Ctx, a: &[Arg], _: &crate::ir::Config) -> Result<()> {
    binop(c, a, |x, y| x.wrapping_mul(y))
}

fn k_and(c: &mut Ctx, a: &[Arg], _: &crate::ir::Config) -> Result<()> {
    binop(c, a, |x, y| x & y)
}

fn k_or(c: &mut Ctx, a: &[Arg], _: &crate::ir::Config) -> Result<()> {
    binop(c, a, |x, y| x | y)
}

fn k_xor(c: &mut Ctx, a: &[Arg], _: &crate::ir::Config) -> Result<()> {
    binop(c, a, |x, y| x ^ y)
}

fn k_eq(c: &mut Ctx, a: &[Arg], _: &crate::ir::Config) -> Result<()> {
    binop(c, a, |x, y| (x == y) as i64)
}

fn k_lt(c: &mut Ctx, a: &[Arg], _: &crate::ir::Config) -> Result<()> {
    binop(c, a, |x, y| (x < y) as i64)
}

fn k_max(c: &mut Ctx, a: &[Arg], _: &crate::ir::Config) -> Result<()> {
    binop(c, a, |x, y| x.max(y))
}

fn k_sadd(c: &mut Ctx, a: &[Arg], _: &crate::ir::Config) -> Result<()> {
    binop(c, a, |x, y| x.saturating_add(y))
}

/// Mirrors `checked_div`: zero and `MIN / -1` are errors, not traps.
fn checked_div(x: i64, y: i64, op: &str) -> Result<i64> {
    if y == 0 {
        return Err(Error::runtime(format!("{op}: division by zero")));
    }
    if x == i64::MIN && y == -1 {
        return Err(Error::runtime(format!(
            "{op}: INT64_MIN / -1 is unrepresentable"
        )));
    }
    Ok(x / y)
}

fn k_div(c: &mut Ctx, a: &[Arg], _: &crate::ir::Config) -> Result<()> {
    let n = c.out_len();
    let (head, out, _) = c.parts();
    for i in 0..n {
        out[i] = checked_div(arg(head, a[0], i), arg(head, a[1], i), "IDIV")?;
    }
    Ok(())
}

fn k_mod(c: &mut Ctx, a: &[Arg], _: &crate::ir::Config) -> Result<()> {
    let n = c.out_len();
    let (head, out, _) = c.parts();
    for i in 0..n {
        let (x, y) = (arg(head, a[0], i), arg(head, a[1], i));
        let q = checked_div(x, y, "IMOD")?;
        out[i] = x.wrapping_sub(q.wrapping_mul(y));
    }
    Ok(())
}

fn k_divu(c: &mut Ctx, a: &[Arg], _: &crate::ir::Config) -> Result<()> {
    let n = c.out_len();
    let (head, out, _) = c.parts();
    for i in 0..n {
        let (x, y) = (arg(head, a[0], i) as u64, arg(head, a[1], i) as u64);
        if y == 0 {
            return Err(Error::runtime("IDIVU: division by zero"));
        }
        out[i] = (x / y) as i64;
    }
    Ok(())
}

fn k_modu(c: &mut Ctx, a: &[Arg], _: &crate::ir::Config) -> Result<()> {
    let n = c.out_len();
    let (head, out, _) = c.parts();
    for i in 0..n {
        let (x, y) = (arg(head, a[0], i) as u64, arg(head, a[1], i) as u64);
        if y == 0 {
            return Err(Error::runtime("IMODU: division by zero"));
        }
        out[i] = (x % y) as i64;
    }
    Ok(())
}

fn k_sar(c: &mut Ctx, a: &[Arg], _: &crate::ir::Config) -> Result<()> {
    let n = c.out_len();
    let (head, out, _) = c.parts();
    for i in 0..n {
        let s = shift(arg(head, a[1], i), "ISAR")?;
        out[i] = arg(head, a[0], i) >> s;
    }
    Ok(())
}

fn k_shl(c: &mut Ctx, a: &[Arg], _: &crate::ir::Config) -> Result<()> {
    let n = c.out_len();
    let (head, out, _) = c.parts();
    let s = shift(scalar(head, a[1])?, "ISHL")?;
    for i in 0..n {
        out[i] = ((arg(head, a[0], i) as u64) << s) as i64;
    }
    Ok(())
}

fn k_shr(c: &mut Ctx, a: &[Arg], _: &crate::ir::Config) -> Result<()> {
    let n = c.out_len();
    let (head, out, _) = c.parts();
    let s = shift(scalar(head, a[1])?, "ISHR")?;
    for i in 0..n {
        out[i] = ((arg(head, a[0], i) as u64) >> s) as i64;
    }
    Ok(())
}

fn k_mad(c: &mut Ctx, a: &[Arg], _: &crate::ir::Config) -> Result<()> {
    let n = c.out_len();
    let (head, out, _) = c.parts();
    for i in 0..n {
        out[i] = arg(head, a[0], i)
            .wrapping_mul(arg(head, a[1], i))
            .wrapping_add(arg(head, a[2], i));
    }
    Ok(())
}

fn k_select(c: &mut Ctx, a: &[Arg], _: &crate::ir::Config) -> Result<()> {
    let n = c.out_len();
    let (head, out, _) = c.parts();
    for i in 0..n {
        out[i] = if arg(head, a[0], i) != 0 {
            arg(head, a[1], i)
        } else {
            arg(head, a[2], i)
        };
    }
    Ok(())
}

fn k_imatmul(c: &mut Ctx, a: &[Arg], _: &crate::ir::Config) -> Result<()> {
    let (head, out, shapes) = c.parts();
    let ad = dims_of(a[0], shapes);
    let bd = dims_of(a[1], shapes);
    if ad.len() != 2 || bd.len() != 2 {
        return Err(Error::runtime(format!(
            "IMATMUL: need 2D operands, got {ad:?} x {bd:?}"
        )));
    }
    let (m, k) = (ad[0], ad[1]);
    let (k2, n) = (bd[0], bd[1]);
    if k != k2 {
        return Err(Error::runtime(format!("IMATMUL: inner dims {k} vs {k2}")));
    }
    if out.len() != m * n {
        return Err(Error::runtime(format!(
            "IMATMUL: output is {} elements, want {m}x{n}",
            out.len()
        )));
    }
    let (ao, _) = off_len(a[0])?;
    let (bo, _) = off_len(a[1])?;
    for i in 0..m {
        for j in 0..n {
            let mut acc: i64 = 0;
            for t in 0..k {
                acc = acc.wrapping_add(head[ao + i * k + t].wrapping_mul(head[bo + t * n + j]));
            }
            out[i * n + j] = acc;
        }
    }
    Ok(())
}

fn k_iget(c: &mut Ctx, a: &[Arg], _: &crate::ir::Config) -> Result<()> {
    let (head, out, _) = c.parts();
    let src = slot(head, a[0])?;
    let idx = literal(a[1])?;
    if idx < 0 || idx as usize >= src.len() {
        return Err(Error::runtime(format!(
            "IGET: index {idx} out of range {}",
            src.len()
        )));
    }
    out[0] = src[idx as usize];
    Ok(())
}

fn k_iwindow(c: &mut Ctx, a: &[Arg], _: &crate::ir::Config) -> Result<()> {
    let (head, out, _) = c.parts();
    let src = slot(head, a[0])?;
    let start = scalar(head, a[1])?;
    let len = literal(a[2])?;
    if start < 0 || len < 0 || (start + len) as usize > src.len() {
        return Err(Error::runtime(format!(
            "IWINDOW: [{start},{}) out of range {}",
            start + len,
            src.len()
        )));
    }
    if out.len() != len as usize {
        return Err(Error::runtime(format!(
            "IWINDOW: output is {} elements, window is {len}",
            out.len()
        )));
    }
    out.copy_from_slice(&src[start as usize..(start + len) as usize]);
    Ok(())
}

fn k_ifind(c: &mut Ctx, a: &[Arg], _: &crate::ir::Config) -> Result<()> {
    let (head, out, _) = c.parts();
    let x = slot(head, a[0])?;
    let pat = slot(head, a[1])?;
    if pat.is_empty() {
        return Err(Error::runtime("IFIND: empty pattern"));
    }
    let mut found: i64 = -1;
    if x.len() >= pat.len() {
        'outer: for i in 0..=x.len() - pat.len() {
            for j in 0..pat.len() {
                if x[i + j] != pat[j] {
                    continue 'outer;
                }
            }
            found = i as i64;
            break;
        }
    }
    out[0] = found;
    Ok(())
}

fn k_igather1(c: &mut Ctx, a: &[Arg], _: &crate::ir::Config) -> Result<()> {
    let (head, out, _) = c.parts();
    let table = slot(head, a[0])?;
    let ids = slot(head, a[1])?;
    if out.len() != ids.len() {
        return Err(Error::runtime(format!(
            "IGATHER1: output is {} elements, ids are {}",
            out.len(),
            ids.len()
        )));
    }
    for i in 0..ids.len() {
        let id = ids[i];
        if id < 0 || id as usize >= table.len() {
            return Err(Error::runtime(format!(
                "IGATHER1: ids out of range [0,{})",
                table.len()
            )));
        }
        out[i] = table[id as usize];
    }
    Ok(())
}

fn k_iscatter1(c: &mut Ctx, a: &[Arg], _: &crate::ir::Config) -> Result<()> {
    let (head, out, _) = c.parts();
    let base = slot(head, a[0])?;
    let ids = slot(head, a[1])?;
    let vals = slot(head, a[2])?;
    if ids.len() != vals.len() {
        return Err(Error::runtime(format!(
            "ISCATTER1: IDS {} vs VALS {}",
            ids.len(),
            vals.len()
        )));
    }
    if out.len() != base.len() {
        return Err(Error::runtime(format!(
            "ISCATTER1: output is {} elements, base is {}",
            out.len(),
            base.len()
        )));
    }
    out.copy_from_slice(base);
    for n in 0..ids.len() {
        let id = ids[n];
        if id < 0 || id as usize >= out.len() {
            return Err(Error::runtime(format!(
                "ISCATTER1: ids out of range [0,{})",
                out.len()
            )));
        }
        out[id as usize] = vals[n];
    }
    Ok(())
}

fn k_iassemble(c: &mut Ctx, a: &[Arg], _: &crate::ir::Config) -> Result<()> {
    let (head, out, _) = c.parts();
    let src = slot(head, a[0])?;
    let start = literal(a[1])?;
    let len = literal(a[2])?;
    if !(1..=8).contains(&len) {
        return Err(Error::runtime(format!(
            "IASSEMBLE: len must be 1..8, got {len}"
        )));
    }
    if start < 0 || start + len > src.len() as i64 {
        return Err(Error::runtime(format!(
            "IASSEMBLE: [{start},{}) out of range {}",
            start + len,
            src.len()
        )));
    }
    let mut v: u128 = 0;
    for i in 0..len as usize {
        v |= ((src[start as usize + i] as u8 as u128) & 0xFF) << (8 * i);
    }
    // The interpreter writes `v - 2^64` here, which underflows `u128` in a
    // debug build for any 8-byte value with the top bit set. Truncating the
    // low 64 bits is the same value without the underflow.
    out[0] = v as i64;
    Ok(())
}

fn crc32_bytes(x: &[i64], mut crc: u32, poly: u32) -> i64 {
    for &b in x {
        crc ^= (b as u8) as u32;
        for _ in 0..8 {
            crc = (crc >> 1) ^ if crc & 1 != 0 { poly } else { 0 };
        }
    }
    crc as i64
}

fn k_icrc32(c: &mut Ctx, a: &[Arg], _: &crate::ir::Config) -> Result<()> {
    let (head, out, _) = c.parts();
    let src = slot(head, a[0])?;
    out[0] = crc32_bytes(src, 0xFFFF_FFFF, 0xEDB8_8320) ^ 0xFFFF_FFFF;
    Ok(())
}

fn k_icrc32c(c: &mut Ctx, a: &[Arg], _: &crate::ir::Config) -> Result<()> {
    let (head, out, _) = c.parts();
    let src = slot(head, a[0])?;
    let seed = slot(head, a[1])?;
    if seed.len() != 1 {
        return Err(Error::runtime("ICRC32C: SEED must be a 1-element I"));
    }
    out[0] = crc32_bytes(src, seed[0] as u32, 0x82F6_3B78);
    Ok(())
}

fn k_slice(c: &mut Ctx, a: &[Arg], _: &crate::ir::Config) -> Result<()> {
    let (head, out, shapes) = c.parts();
    let xd = dims_of(a[0], shapes);
    if xd.len() == 1 {
        // The interpreter refuses 1-D I slices in favour of IWINDOW; match it.
        return Err(Error::runtime(
            "SLICE: 1D I is not supported; use IWINDOW for 1D I",
        ));
    }
    let axis = literal(a[1])?;
    let start = literal(a[2])?;
    let end = literal(a[3])?;
    if axis < 0 || axis as usize >= xd.len() || start < 0 || end < start {
        return Err(Error::runtime("SLICE: bad axis or range"));
    }
    let ax = axis as usize;
    let (start, end) = (start as usize, end as usize);
    if end > xd[ax] {
        return Err(Error::runtime(format!(
            "SLICE: [{start},{end}) out of range {}",
            xd[ax]
        )));
    }
    let outer: usize = xd[..ax].iter().product();
    let inner: usize = xd[ax + 1..].iter().product();
    let n = xd[ax];
    let m = end - start;
    if out.len() != outer * m * inner {
        return Err(Error::runtime("SLICE: output size does not match shapes"));
    }
    let (xo, _) = off_len(a[0])?;
    for o in 0..outer {
        let src = xo + o * n * inner + start * inner;
        let dst = o * m * inner;
        out[dst..dst + m * inner].copy_from_slice(&head[src..src + m * inner]);
    }
    Ok(())
}

fn k_relu(c: &mut Ctx, a: &[Arg], _: &crate::ir::Config) -> Result<()> {
    let (head, out, _) = c.parts();
    if !matches!(a[0], Arg::Slot { is_f: true, .. }) {
        return Err(Error::runtime("RELU: needs F/T"));
    }
    for i in 0..out.len() {
        out[i] = word_of(arg_f64(head, a[0], i).max(0.0));
    }
    Ok(())
}

/// Bilinear HWC resize, mirroring `ops::image::op_resize2d` term for term.
fn k_resize2d(c: &mut Ctx, a: &[Arg], _: &crate::ir::Config) -> Result<()> {
    let (head, out, shapes) = c.parts();
    if !matches!(a[0], Arg::Slot { is_f: true, .. }) {
        return Err(Error::runtime("RESIZE2D: needs F HWC"));
    }
    let sh = dims_of(a[0], shapes);
    if sh.len() != 3 {
        return Err(Error::runtime(format!(
            "RESIZE2D: expected HWC, got {sh:?}"
        )));
    }
    let (h, w, ch) = (sh[0], sh[1], sh[2]);
    if h == 0 || w == 0 {
        return Err(Error::runtime("RESIZE2D: empty input"));
    }
    let ho = literal(a[1])?;
    let wo = literal(a[2])?;
    let align = literal(a[3])?;
    if ho <= 0 || wo <= 0 || (align != 0 && align != 1) {
        return Err(Error::runtime("RESIZE2D: bad size/align"));
    }
    let (ho, wo) = (ho as usize, wo as usize);
    if out.len() != ho * wo * ch {
        return Err(Error::runtime(
            "RESIZE2D: output size does not match shapes",
        ));
    }
    let (xo, _) = off_len(a[0])?;
    let xf = &head[xo..];
    let ac = align != 0;
    for oy in 0..ho {
        let sy = if ac {
            if ho > 1 {
                oy as f64 * (h as f64 - 1.0) / (ho as f64 - 1.0)
            } else {
                0.0
            }
        } else {
            (oy as f64 + 0.5) * h as f64 / ho as f64 - 0.5
        }
        .clamp(0.0, h as f64 - 1.0);
        let y0 = sy.floor() as usize;
        let y1 = (y0 + 1).min(h - 1);
        let wy = sy - y0 as f64;
        for ox in 0..wo {
            let sx = if ac {
                if wo > 1 {
                    ox as f64 * (w as f64 - 1.0) / (wo as f64 - 1.0)
                } else {
                    0.0
                }
            } else {
                (ox as f64 + 0.5) * w as f64 / wo as f64 - 0.5
            }
            .clamp(0.0, w as f64 - 1.0);
            let x0 = sx.floor() as usize;
            let x1 = (x0 + 1).min(w - 1);
            let wx = sx - x0 as f64;
            for cc in 0..ch {
                let v00 = f64_of(xf[(y0 * w + x0) * ch + cc]);
                let v01 = f64_of(xf[(y0 * w + x1) * ch + cc]);
                let v10 = f64_of(xf[(y1 * w + x0) * ch + cc]);
                let v11 = f64_of(xf[(y1 * w + x1) * ch + cc]);
                let v = (1.0 - wy) * (1.0 - wx) * v00
                    + (1.0 - wy) * wx * v01
                    + wy * (1.0 - wx) * v10
                    + wy * wx * v11;
                out[(oy * wo + ox) * ch + cc] = word_of(v);
            }
        }
    }
    Ok(())
}

/// Shared argument unpacking for the two convolution kernels.
struct ConvArgs<'a> {
    h: usize,
    w: usize,
    cout: usize,
    kh: usize,
    kw: usize,
    cin: usize,
    stride: i64,
    pad: i64,
    xf: &'a [i64],
    wf: &'a [i64],
    bf: &'a [i64],
}

fn conv_args<'a>(
    head: &'a [i64],
    out: &[i64],
    a: &[Arg],
    shapes: &[usize],
    op: &str,
) -> Result<ConvArgs<'a>> {
    for arg in &a[..3] {
        if !matches!(arg, Arg::Slot { is_f: true, .. }) {
            return Err(Error::runtime(format!("{op}: needs F operands")));
        }
    }
    let xs = dims_of(a[0], shapes);
    let ws = dims_of(a[1], shapes);
    let bs = dims_of(a[2], shapes);
    if xs.len() != 3 || ws.len() != 4 || bs.len() != 1 {
        return Err(Error::runtime(format!(
            "{op}: expected (H,W,Cin), (Cout,kh,kw,Cin), (Cout); got {xs:?}, {ws:?}, {bs:?}"
        )));
    }
    let (h, w, cin) = (xs[0], xs[1], xs[2]);
    let (cout, kh, kw, wcin) = (ws[0], ws[1], ws[2], ws[3]);
    if wcin != cin || bs[0] != cout {
        return Err(Error::runtime(format!("{op}: channel/bias mismatch")));
    }
    let stride = literal(a[3])?;
    let pad = literal(a[4])?;
    if stride <= 0 || pad < 0 {
        return Err(Error::runtime(format!("{op}: bad stride/pad")));
    }
    let (xo, _) = off_len(a[0])?;
    let (w_off, _) = off_len(a[1])?;
    let (bo, _) = off_len(a[2])?;
    if out.len() % cout != 0 {
        return Err(Error::runtime(format!(
            "{op}: output size does not match Cout"
        )));
    }
    Ok(ConvArgs {
        h,
        w,
        cout,
        kh,
        kw,
        cin,
        stride,
        pad,
        xf: &head[xo..],
        wf: &head[w_off..],
        bf: &head[bo..],
    })
}

fn k_conv2d(c: &mut Ctx, a: &[Arg], _: &crate::ir::Config) -> Result<()> {
    let (head, out, shapes) = c.parts();
    let g = conv_args(head, out, a, shapes, "CONV2D")?;
    let ho = (g.h as i64 + 2 * g.pad - g.kh as i64) / g.stride + 1;
    let wo = (g.w as i64 + 2 * g.pad - g.kw as i64) / g.stride + 1;
    if ho <= 0 || wo <= 0 {
        return Err(Error::runtime(format!("CONV2D: bad output size {ho}x{wo}")));
    }
    let (ho, wo) = (ho as usize, wo as usize);
    if out.len() != ho * wo * g.cout {
        return Err(Error::runtime("CONV2D: output size does not match shapes"));
    }
    for co in 0..g.cout {
        for oy in 0..ho {
            for ox in 0..wo {
                let mut acc = f64_of(g.bf[co]);
                for ky in 0..g.kh {
                    let iy = oy as i64 * g.stride + ky as i64 - g.pad;
                    if iy < 0 || iy >= g.h as i64 {
                        continue;
                    }
                    for kx in 0..g.kw {
                        let ix = ox as i64 * g.stride + kx as i64 - g.pad;
                        if ix < 0 || ix >= g.w as i64 {
                            continue;
                        }
                        for ci in 0..g.cin {
                            acc += f64_of(g.xf[(iy as usize * g.w + ix as usize) * g.cin + ci])
                                * f64_of(g.wf[((co * g.kh + ky) * g.kw + kx) * g.cin + ci]);
                        }
                    }
                }
                out[(oy * wo + ox) * g.cout + co] = word_of(acc);
            }
        }
    }
    Ok(())
}

fn k_deconv2d(c: &mut Ctx, a: &[Arg], _: &crate::ir::Config) -> Result<()> {
    let (head, out, shapes) = c.parts();
    let g = conv_args(head, out, a, shapes, "DECONV2D")?;
    let ho = (g.h as i64 - 1) * g.stride - 2 * g.pad + g.kh as i64;
    let wo = (g.w as i64 - 1) * g.stride - 2 * g.pad + g.kw as i64;
    if ho <= 0 || wo <= 0 {
        return Err(Error::runtime(format!(
            "DECONV2D: bad output size {ho}x{wo}"
        )));
    }
    let (ho, wo) = (ho as usize, wo as usize);
    if out.len() != ho * wo * g.cout {
        return Err(Error::runtime(
            "DECONV2D: output size does not match shapes",
        ));
    }
    // Arena memory is reused, not fresh: the interpreter starts from zeros.
    out.fill(0);
    for co in 0..g.cout {
        for iy in 0..g.h {
            for ix in 0..g.w {
                for ky in 0..g.kh {
                    let oy = iy as i64 * g.stride - g.pad + ky as i64;
                    if oy < 0 || oy >= ho as i64 {
                        continue;
                    }
                    for kx in 0..g.kw {
                        let ox = ix as i64 * g.stride - g.pad + kx as i64;
                        if ox < 0 || ox >= wo as i64 {
                            continue;
                        }
                        for ci in 0..g.cin {
                            let idx = (oy as usize * wo + ox as usize) * g.cout + co;
                            out[idx] = word_of(
                                f64_of(out[idx])
                                    + f64_of(g.xf[(iy * g.w + ix) * g.cin + ci])
                                        * f64_of(g.wf[((co * g.kh + ky) * g.kw + kx) * g.cin + ci]),
                            );
                        }
                    }
                }
            }
        }
        for i in 0..ho * wo {
            let idx = i * g.cout + co;
            out[idx] = word_of(f64_of(out[idx]) + f64_of(g.bf[co]));
        }
    }
    Ok(())
}

fn k_f2i(c: &mut Ctx, a: &[Arg], _: &crate::ir::Config) -> Result<()> {
    let (head, out, _) = c.parts();
    if !matches!(a[0], Arg::Slot { is_f: true, .. }) {
        return Err(Error::runtime("F2I: stream must be float"));
    }
    for i in 0..out.len() {
        let x = arg_f64(head, a[0], i);
        if !x.is_finite() {
            return Err(Error::runtime("F2I: NaN/Inf input"));
        }
        let t = x.trunc();
        if t < -(2f64.powi(63)) || t >= 2f64.powi(63) {
            return Err(Error::runtime("F2I: value outside int64 range"));
        }
        out[i] = t as i64;
    }
    Ok(())
}

fn k_i2f(c: &mut Ctx, a: &[Arg], _: &crate::ir::Config) -> Result<()> {
    let (head, out, _) = c.parts();
    for i in 0..out.len() {
        out[i] = word_of(arg(head, a[0], i) as f64);
    }
    Ok(())
}

fn k_concat(c: &mut Ctx, a: &[Arg], _: &crate::ir::Config) -> Result<()> {
    let (head, out, shapes) = c.parts();
    let ad = dims_of(a[0], shapes);
    let bd = dims_of(a[1], shapes);
    let axis = literal(a[2])?;
    if axis < 0 || ad.len() != bd.len() || axis as usize >= ad.len() {
        return Err(Error::runtime("CONCAT: bad axis or rank"));
    }
    let ax = axis as usize;
    for i in 0..ad.len() {
        if i != ax && ad[i] != bd[i] {
            return Err(Error::runtime(format!(
                "CONCAT: dim {i} differs ({} vs {})",
                ad[i], bd[i]
            )));
        }
    }
    let outer: usize = ad[..ax].iter().product();
    let inner: usize = ad[ax + 1..].iter().product();
    let (na, nb) = (ad[ax], bd[ax]);
    if out.len() != outer * (na + nb) * inner {
        return Err(Error::runtime("CONCAT: output size does not match shapes"));
    }
    let (ao, _) = off_len(a[0])?;
    let (bo, _) = off_len(a[1])?;
    for o in 0..outer {
        let a_len = na * inner;
        let b_len = nb * inner;
        let a_from = ao + o * a_len;
        let b_from = bo + o * b_len;
        let dst = o * (na + nb) * inner;
        out[dst..dst + a_len].copy_from_slice(&head[a_from..a_from + a_len]);
        out[dst + a_len..dst + a_len + b_len].copy_from_slice(&head[b_from..b_from + b_len]);
    }
    Ok(())
}

fn k_reshape(c: &mut Ctx, a: &[Arg], _: &crate::ir::Config) -> Result<()> {
    let (head, out, _) = c.parts();
    let src = slot(head, a[0])?;
    if src.len() != out.len() {
        return Err(Error::runtime(format!(
            "RESHAPE: input is {} elements, output is {}",
            src.len(),
            out.len()
        )));
    }
    out.copy_from_slice(src);
    Ok(())
}

/// F matmul, same `ikj` order as the interpreter and the C emitter.
fn k_fmatmul(c: &mut Ctx, a: &[Arg], _: &crate::ir::Config) -> Result<()> {
    let (head, out, shapes) = c.parts();
    let ad = dims_of(a[0], shapes);
    let bd = dims_of(a[1], shapes);
    if ad.len() < 2 || bd.len() < 2 {
        return Err(Error::runtime(format!(
            "MATMUL: need rank >= 2, got {ad:?} x {bd:?}"
        )));
    }
    let rank = ad.len();
    let m = ad[rank - 2];
    let k = ad[rank - 1];
    if bd[bd.len() - 2] != k {
        return Err(Error::runtime(format!(
            "MATMUL: inner dims {ad:?} vs {bd:?}"
        )));
    }
    let n = *bd.last().unwrap();
    let nb: usize = ad[..rank - 2].iter().product();
    let nb_b: usize = bd[..bd.len() - 2].iter().product();
    if nb_b != 1 && nb_b != nb {
        return Err(Error::runtime(format!(
            "MATMUL: batch dims {:?} vs {:?}",
            &ad[..rank - 2],
            &bd[..bd.len() - 2]
        )));
    }
    if out.len() != nb * m * n {
        return Err(Error::runtime("MATMUL: output size does not match shapes"));
    }
    let (ao, _) = off_len(a[0])?;
    let (bo, _) = off_len(a[1])?;
    out.fill(0);
    for bi in 0..nb {
        let bb = if nb_b == 1 { 0 } else { bi };
        for i in 0..m {
            for kk in 0..k {
                let aik = f64_of(head[ao + (bi * m + i) * k + kk]);
                for j in 0..n {
                    let idx = (bi * m + i) * n + j;
                    out[idx] =
                        word_of(f64_of(out[idx]) + aik * f64_of(head[bo + (bb * k + kk) * n + j]));
                }
            }
        }
    }
    Ok(())
}

fn k_layernorm(c: &mut Ctx, a: &[Arg], cfg: &crate::ir::Config) -> Result<()> {
    let (head, out, shapes) = c.parts();
    if !matches!(a[0], Arg::Slot { is_f: true, .. }) {
        return Err(Error::runtime("LAYERNORM: needs F"));
    }
    let xd = dims_of(a[0], shapes);
    if xd.is_empty() {
        return Err(Error::runtime("LAYERNORM: need rank >= 1"));
    }
    let d = *xd.last().unwrap();
    if d == 0 || out.len() % d != 0 {
        return Err(Error::runtime(
            "LAYERNORM: output size does not match the last axis",
        ));
    }
    let x = slot(head, a[0])?;
    let w = slot(head, a[1])?;
    let b = slot(head, a[2])?;
    if w.len() != d || b.len() != d {
        return Err(Error::runtime(format!(
            "LAYERNORM: weight/bias length mismatch (want {d}, got {}/{})",
            w.len(),
            b.len()
        )));
    }
    let eps = cfg
        .get_f64("eps_ln")
        .ok_or_else(|| Error::runtime("LAYERNORM: needs CONFIG eps_ln"))?;
    let rows = out.len() / d;
    for r in 0..rows {
        let row = &x[r * d..(r + 1) * d];
        let mut mu = 0f64;
        for &v in row {
            mu += f64_of(v);
        }
        mu /= d as f64;
        let mut ss = 0f64;
        for &v in row {
            let dv = f64_of(v) - mu;
            ss += dv * dv;
        }
        let rs = 1.0 / (ss / d as f64 + eps).sqrt();
        for j in 0..d {
            out[r * d + j] = word_of((f64_of(row[j]) - mu) * rs * f64_of(w[j]) + f64_of(b[j]));
        }
    }
    Ok(())
}

fn k_rmsnorm(c: &mut Ctx, a: &[Arg], cfg: &crate::ir::Config) -> Result<()> {
    let (head, out, shapes) = c.parts();
    if !matches!(a[0], Arg::Slot { is_f: true, .. }) {
        return Err(Error::runtime("RMSNORM: needs F"));
    }
    let xd = dims_of(a[0], shapes);
    if xd.is_empty() {
        return Err(Error::runtime("RMSNORM: need rank >= 1"));
    }
    let d = *xd.last().unwrap();
    if d == 0 || out.len() % d != 0 {
        return Err(Error::runtime(
            "RMSNORM: output size does not match the last axis",
        ));
    }
    let x = slot(head, a[0])?;
    let w = slot(head, a[1])?;
    if w.len() != d {
        return Err(Error::runtime(format!(
            "RMSNORM: weight length {} != {d}",
            w.len()
        )));
    }
    let eps = cfg
        .get_f64("eps_rms")
        .ok_or_else(|| Error::runtime("RMSNORM: needs CONFIG eps_rms"))?;
    let rows = out.len() / d;
    for r in 0..rows {
        let row = &x[r * d..(r + 1) * d];
        let mut ss = 0f64;
        for &v in row {
            let xv = f64_of(v);
            ss += xv * xv;
        }
        let rs = 1.0 / (ss / d as f64 + eps).sqrt();
        for j in 0..d {
            out[r * d + j] = word_of(f64_of(row[j]) * rs * f64_of(w[j]));
        }
    }
    Ok(())
}

fn k_rotary(c: &mut Ctx, a: &[Arg], cfg: &crate::ir::Config) -> Result<()> {
    let (head, out, shapes) = c.parts();
    if !matches!(a[0], Arg::Slot { is_f: true, .. }) {
        return Err(Error::runtime("ROTARY: needs F"));
    }
    let xd = dims_of(a[0], shapes);
    if xd.is_empty() {
        return Err(Error::runtime("ROTARY: need rank >= 1"));
    }
    let d = *xd.last().unwrap();
    if d % 2 != 0 {
        return Err(Error::runtime(format!(
            "ROTARY: head dim must be even, got {d}"
        )));
    }
    if d == 0 || out.len() % d != 0 {
        return Err(Error::runtime(
            "ROTARY: output size does not match the head dim",
        ));
    }
    let x = slot(head, a[0])?;
    let pos = slot(head, a[1])?;
    let rows = out.len() / d;
    if pos.len() != 1 && pos.len() != rows {
        return Err(Error::runtime(format!(
            "ROTARY: pos length {} not broadcastable to {} rows",
            pos.len(),
            rows
        )));
    }
    let base = cfg.get_f64_or("rope_base", 10000.0);
    let half = d / 2;
    for r in 0..rows {
        let p = pos[if pos.len() == 1 { 0 } else { r }];
        if p < 0 {
            return Err(Error::runtime("ROTARY: negative position"));
        }
        for j in 0..half {
            let ang = p as f64 * base.powf(-2.0 * j as f64 / d as f64);
            let c = ang.cos();
            let s = ang.sin();
            let i0 = r * d + 2 * j;
            let i1 = i0 + 1;
            out[i0] = word_of(f64_of(x[i0]) * c - f64_of(x[i1]) * s);
            out[i1] = word_of(f64_of(x[i0]) * s + f64_of(x[i1]) * c);
        }
    }
    Ok(())
}

fn k_conv(c: &mut Ctx, a: &[Arg], _: &crate::ir::Config) -> Result<()> {
    let (head, out, shapes) = c.parts();
    let xd = dims_of(a[0], shapes);
    let kd = dims_of(a[1], shapes);
    if xd.len() != 2 || kd.len() != 2 {
        return Err(Error::runtime(format!(
            "CONV: expected HW image and kernel, got {xd:?} / {kd:?}"
        )));
    }
    if !matches!(a[0], Arg::Slot { is_f: true, .. })
        || !matches!(a[1], Arg::Slot { is_f: true, .. })
    {
        return Err(Error::runtime("CONV: needs F operands"));
    }
    let (h, w) = (xd[0], xd[1]);
    let (kh, kw) = (kd[0], kd[1]);
    if kh == 0 || kw == 0 {
        return Err(Error::runtime("CONV: empty kernel"));
    }
    if out.len() != h * w {
        return Err(Error::runtime("CONV: output size does not match shapes"));
    }
    let x = slot(head, a[0])?;
    let k = slot(head, a[1])?;
    let (rh, rw) = (kh / 2, kw / 2);
    for oy in 0..h {
        for ox in 0..w {
            let mut acc = 0f64;
            for ky in 0..kh {
                let iy = (oy as i64 + ky as i64 - rh as i64).clamp(0, h as i64 - 1) as usize;
                for kx in 0..kw {
                    let ix = (ox as i64 + kx as i64 - rw as i64).clamp(0, w as i64 - 1) as usize;
                    acc += f64_of(x[iy * w + ix]) * f64_of(k[ky * kw + kx]);
                }
            }
            out[oy * w + ox] = word_of(acc);
        }
    }
    Ok(())
}

fn frexp(x: f64) -> (f64, i32) {
    if x == 0.0 || !x.is_finite() {
        return (x, 0);
    }
    let bits = x.to_bits();
    let exp = ((bits >> 52) & 0x7ff) as i32 - 1022;
    let mant_bits = (bits & !(0x7ffu64 << 52)) | (1022u64 << 52);
    (f64::from_bits(mant_bits), exp)
}

fn is_dyadic(v: f64) -> bool {
    if v <= 0.0 {
        return false;
    }
    let (mant, _) = frexp(v);
    (mant - 0.5).abs() < 1e-12
}

fn k_interp(c: &mut Ctx, a: &[Arg], _: &crate::ir::Config) -> Result<()> {
    let (head, out, shapes) = c.parts();
    if !matches!(a[0], Arg::Slot { is_f: true, .. }) {
        return Err(Error::runtime("INTERP: needs F"));
    }
    let xd = dims_of(a[0], shapes);
    if xd.len() != 2 && xd.len() != 3 {
        return Err(Error::runtime(format!(
            "INTERP: needs F x(H,W[,C]), got {xd:?}"
        )));
    }
    let sy = literal_f64(a[1])?;
    let sx = literal_f64(a[2])?;
    if !(is_dyadic(sy) && is_dyadic(sx)) {
        return Err(Error::runtime(format!(
            "INTERP: non-dyadic scale ({sy},{sx}) is a LOWERING ERROR"
        )));
    }
    let (h, w) = (xd[0], xd[1]);
    let c = if xd.len() == 3 { xd[2] } else { 1 };
    let ho = ((h as f64 * sy).round().max(1.0)) as usize;
    let wo = ((w as f64 * sx).round().max(1.0)) as usize;
    if out.len() != ho * wo * c {
        return Err(Error::runtime("INTERP: output size does not match shapes"));
    }
    let x = slot(head, a[0])?;
    let hdiv = h as f64 / ho as f64;
    let wdiv = w as f64 / wo as f64;
    for oy in 0..ho {
        let fsy = ((oy as f64 + 0.5) * hdiv - 0.5).clamp(0.0, h as f64 - 1.0);
        let y0 = fsy.floor() as usize;
        let wy = fsy - y0 as f64;
        let y1 = (y0 + 1).min(h - 1);
        for ox in 0..wo {
            let fsx = ((ox as f64 + 0.5) * wdiv - 0.5).clamp(0.0, w as f64 - 1.0);
            let x0 = fsx.floor() as usize;
            let wx = fsx - x0 as f64;
            let x1 = (x0 + 1).min(w - 1);
            for cc in 0..c {
                let v00 = f64_of(x[(y0 * w + x0) * c + cc]);
                let v01 = f64_of(x[(y0 * w + x1) * c + cc]);
                let v10 = f64_of(x[(y1 * w + x0) * c + cc]);
                let v11 = f64_of(x[(y1 * w + x1) * c + cc]);
                out[(oy * wo + ox) * c + cc] = word_of(
                    (1.0 - wy) * (1.0 - wx) * v00
                        + (1.0 - wy) * wx * v01
                        + wy * (1.0 - wx) * v10
                        + wy * wx * v11,
                );
            }
        }
    }
    Ok(())
}

fn k_gather(c: &mut Ctx, a: &[Arg], _: &crate::ir::Config) -> Result<()> {
    let (head, out, shapes) = c.parts();
    let td = dims_of(a[0], shapes);
    if td.len() != 2 {
        return Err(Error::runtime(format!(
            "GATHER: table must be (V,C), got {td:?}"
        )));
    }
    let (v, c) = (td[0], td[1]);
    let table = slot(head, a[0])?;
    let ids = slot(head, a[1])?;
    if out.len() != ids.len() * c {
        return Err(Error::runtime("GATHER: output size does not match ids"));
    }
    for (n, &id) in ids.iter().enumerate() {
        if id < 0 || id as usize >= v {
            return Err(Error::runtime(format!("GATHER: ids out of range [0,{v})")));
        }
        let row = id as usize;
        out[n * c..(n + 1) * c].copy_from_slice(&table[row * c..(row + 1) * c]);
    }
    Ok(())
}

fn k_gather3(c: &mut Ctx, a: &[Arg], _: &crate::ir::Config) -> Result<()> {
    let (head, out, shapes) = c.parts();
    let td = dims_of(a[0], shapes);
    if td.len() != 3 {
        return Err(Error::runtime(format!(
            "GATHER3: table must be (V1,V2,C), got {td:?}"
        )));
    }
    let (v1, v2, c) = (td[0], td[1], td[2]);
    let table = slot(head, a[0])?;
    let ii = slot(head, a[1])?;
    let jj = slot(head, a[2])?;
    if ii.len() != jj.len() {
        return Err(Error::runtime(format!(
            "GATHER3: i/j length mismatch {} vs {}",
            ii.len(),
            jj.len()
        )));
    }
    let n = ii.len();
    if out.len() != n * c {
        return Err(Error::runtime("GATHER3: output size does not match ids"));
    }
    for k in 0..n {
        let iv = ii[k];
        let jv = jj[k];
        if iv < 0 || iv as usize >= v1 {
            return Err(Error::runtime(format!("GATHER3: i out of range [0,{v1})")));
        }
        if jv < 0 || jv as usize >= v2 {
            return Err(Error::runtime(format!("GATHER3: j out of range [0,{v2})")));
        }
        let base = (iv as usize * v2 + jv as usize) * c;
        out[k * c..(k + 1) * c].copy_from_slice(&table[base..base + c]);
    }
    Ok(())
}

fn k_scatter(c: &mut Ctx, a: &[Arg], _: &crate::ir::Config) -> Result<()> {
    let (head, out, shapes) = c.parts();
    let bd = dims_of(a[0], shapes);
    let rd = dims_of(a[1], shapes);
    if bd.len() != 2 || rd != [1, bd[1]] {
        return Err(Error::runtime(format!(
            "SCATTER: base {bd:?} + row {rd:?} mismatch (row must be (1,C))"
        )));
    }
    let base = slot(head, a[0])?;
    let row = slot(head, a[1])?;
    let idx = slot(head, a[2])?;
    if idx.len() != 1 {
        return Err(Error::runtime("SCATTER: idx must be scalar/1-elem"));
    }
    if out.len() != base.len() {
        return Err(Error::runtime("SCATTER: output size does not match base"));
    }
    let n = idx[0];
    let c = bd[1];
    if n < 0 || n as usize >= bd[0] {
        return Err(Error::runtime(format!(
            "SCATTER: idx {n} out of [0,{})",
            bd[0]
        )));
    }
    out.copy_from_slice(base);
    out[n as usize * c..(n as usize + 1) * c].copy_from_slice(&row[..c]);
    Ok(())
}

fn k_argmax(c: &mut Ctx, a: &[Arg], _: &crate::ir::Config) -> Result<()> {
    let (head, out, shapes) = c.parts();
    let xd = dims_of(a[0], shapes);
    if xd.is_empty() {
        return Err(Error::runtime("ARGMAX: need rank >= 1"));
    }
    let axis = literal(a[1])?;
    let rank = xd.len() as i64;
    let ax = (((axis % rank) + rank) % rank) as usize;
    let outer: usize = xd[..ax].iter().product();
    let inner: usize = xd[ax + 1..].iter().product();
    let n = xd[ax];
    if out.len() != outer * inner {
        return Err(Error::runtime("ARGMAX: output size does not match shapes"));
    }
    let (xo, _) = off_len(a[0])?;
    let is_f = matches!(a[0], Arg::Slot { is_f: true, .. });
    for o in 0..outer {
        for i in 0..inner {
            let mut best = f64::NEG_INFINITY;
            let mut bi = 0usize;
            for j in 0..n {
                let idx = (o * n + j) * inner + i;
                let val = if is_f {
                    f64_of(head[xo + idx])
                } else {
                    head[xo + idx] as f64
                };
                if j == 0 || val > best {
                    best = val;
                    bi = j;
                }
            }
            out[o * inner + i] = bi as i64;
        }
    }
    Ok(())
}

/// The C emitter's single-pass insertion, so ties and non-finite scores match.
fn k_topk(c: &mut Ctx, a: &[Arg], _: &crate::ir::Config) -> Result<()> {
    let (head, out, shapes) = c.parts();
    if !matches!(a[0], Arg::Slot { is_f: true, .. }) {
        return Err(Error::runtime("TOPK: needs F scores"));
    }
    let xd = dims_of(a[0], shapes);
    if xd.len() != 2 {
        return Err(Error::runtime(format!("TOPK: needs 2D, got {xd:?}")));
    }
    let (r, s) = (xd[0], xd[1]);
    let k = literal(a[1])?;
    if k < 1 || k as usize > s {
        return Err(Error::runtime(format!(
            "TOPK: k={k} out of range (1,S={s})"
        )));
    }
    let k = k as usize;
    let n = r * k;
    if out.len() != 2 * n {
        return Err(Error::runtime("TOPK: output size does not match shapes"));
    }
    let (xo, _) = off_len(a[0])?;
    let (vals, idxs) = out.split_at_mut(n);
    for row in 0..r {
        let mut seen = 0usize;
        for cc in 0..s {
            let v = f64_of(head[xo + row * s + cc]);
            let mut p = if seen < k { seen } else { k };
            while p > 0 && v > f64_of(vals[row * k + p - 1]) {
                p -= 1;
            }
            if p >= k {
                continue;
            }
            let m = if seen < k { seen } else { k - 1 };
            let mut t = m;
            while t > p {
                vals[row * k + t] = vals[row * k + t - 1];
                idxs[row * k + t] = idxs[row * k + t - 1];
                t -= 1;
            }
            vals[row * k + p] = word_of(v);
            idxs[row * k + p] = cc as i64;
            if seen < k {
                seen += 1;
            }
        }
    }
    Ok(())
}

fn k_reduce_sum(c: &mut Ctx, a: &[Arg], _: &crate::ir::Config) -> Result<()> {
    let (head, out, _) = c.parts();
    let src = slot(head, a[0])?;
    let mut acc: i64 = 0;
    for v in src {
        acc = acc.wrapping_add(*v);
    }
    out[0] = acc;
    Ok(())
}

fn k_window_sum(c: &mut Ctx, a: &[Arg], _: &crate::ir::Config) -> Result<()> {
    let (head, out, _) = c.parts();
    let src = slot(head, a[0])?;
    let start = literal(a[1])?;
    let len = literal(a[2])?;
    if start < 0 || len < 0 || (start + len) as usize > src.len() {
        return Err(Error::runtime(format!(
            "stream arena: ISUM [{start},{}) of dim {}",
            start + len,
            src.len()
        )));
    }
    let mut acc: i64 = 0;
    for i in 0..len as usize {
        acc = acc.wrapping_add(src[start as usize + i]);
    }
    out[0] = acc;
    Ok(())
}

/// Mirrors `ops::bytes::op_ivarint`: up to eight 7-bit groups, then one full
/// byte, most-significant group first.
fn decode_varint(src: &[i64], start: usize) -> Result<(i64, i64)> {
    if start >= src.len() {
        return Err(Error::runtime(format!(
            "stream arena: IVARINT start {start} out of range {}",
            src.len()
        )));
    }
    let mut v: u128 = 0;
    for i in 0..8 {
        if start + i >= src.len() {
            return Err(Error::runtime("stream arena: IVARINT truncated"));
        }
        let b = src[start + i] as u8 as u128;
        v = (v << 7) | (b & 0x7F);
        if b & 0x80 == 0 {
            return Ok((v as i64, (i + 1) as i64));
        }
    }
    if start + 8 >= src.len() {
        return Err(Error::runtime("stream arena: IVARINT truncated"));
    }
    v = (v << 8) | (src[start + 8] as u8 as u128);
    Ok((v as i64, 9))
}

fn k_varint(c: &mut Ctx, a: &[Arg], _: &crate::ir::Config) -> Result<()> {
    let (head, out, _) = c.parts();
    let src = slot(head, a[0])?;
    let start = match a[1] {
        Arg::Const(c) => {
            if c < 0 {
                return Err(Error::runtime("stream arena: IVARINT negative start"));
            }
            c as usize
        }
        Arg::ConstF(_) => {
            return Err(Error::runtime(
                "stream arena: IVARINT start is not an integer",
            ))
        }
        Arg::Slot { off, .. } => head[off] as usize,
    };
    let (v, n) = decode_varint(src, start)?;
    out[0] = v;
    out[1] = n;
    Ok(())
}

/// A cached plan for one program, recompiled when the input shapes change.
#[derive(Default)]
pub struct PlanCache {
    key: Vec<(String, Vec<usize>)>,
    plan: Option<Plan>,
    /// True once compilation has failed for `key`, so we do not retry it.
    failed: bool,
    valid: bool,
}

/// Every instruction output, unique and in program order.
pub fn outputs_of(bound: &BoundProgram) -> Vec<String> {
    let mut names: Vec<String> = Vec::new();
    for ins in &bound.instrs {
        for o in &ins.outs {
            if !names.iter().any(|n| n == o) {
                names.push(o.clone());
            }
        }
    }
    names
}

impl PlanCache {
    pub fn new() -> Self {
        Self::default()
    }

    fn key_for(bound: &BoundProgram, feeds: &Feeds) -> Vec<(String, Vec<usize>)> {
        bound
            .inputs
            .iter()
            .map(|d| {
                let shape = feeds
                    .get(&d.name)
                    .and_then(|v| v.shape())
                    .unwrap_or_default();
                (d.name.clone(), shape)
            })
            .collect()
    }

    /// The plan for this program and these input shapes, compiling once.
    ///
    /// `None` means the listing is outside the arena's scope; callers fall
    /// back to the interpreter.
    pub fn get(&mut self, bound: &BoundProgram, feeds: &Feeds) -> Option<&Plan> {
        let key = Self::key_for(bound, feeds);
        if !self.valid || self.key != key {
            self.key = key;
            self.valid = true;
            let outputs = outputs_of(bound);
            match Plan::compile(bound, feeds, &outputs) {
                Ok(p) => {
                    self.plan = Some(p);
                    self.failed = false;
                }
                Err(_) => {
                    self.plan = None;
                    self.failed = true;
                }
            }
        }
        if self.failed {
            None
        } else {
            self.plan.as_ref()
        }
    }

    pub fn plan(&self) -> Option<&Plan> {
        if self.failed {
            None
        } else {
            self.plan.as_ref()
        }
    }
}

#[derive(Clone, Debug)]
struct Step {
    kernel: Kernel,
    dst: usize,
    dst_len: usize,
    args: std::ops::Range<usize>,
}

fn slot_of(slots: &HashMap<String, Slot>, name: &str) -> Result<Slot> {
    slots
        .get(name)
        .copied()
        .ok_or_else(|| Error::asm(format!("stream arena: use-before-def {name:?}")))
}

fn operand(slots: &HashMap<String, Slot>, name: &str) -> Result<Arg> {
    if let Some(s) = slots.get(name) {
        Ok(Arg::Slot {
            off: s.off,
            len: s.len,
            dims: s.dims,
            is_f: s.is_f,
        })
    } else if crate::layout::is_literal(name) {
        if let Ok(v) = name.trim().parse::<i64>() {
            Ok(Arg::Const(v))
        } else {
            let x = name
                .trim()
                .parse::<f64>()
                .map_err(|e| Error::asm(format!("stream arena: bad literal {name:?}: {e}")))?;
            Ok(Arg::ConstF(x))
        }
    } else {
        Err(Error::asm(format!("stream arena: use-before-def {name:?}")))
    }
}

/// A compiled, allocation-free execution plan for one listing.
#[derive(Clone, Debug)]
pub struct Plan {
    inputs: Vec<(String, Slot)>,
    steps: Vec<Step>,
    /// All step arguments, flattened; `Step::args` indexes into this.
    args: Vec<Arg>,
    /// Flattened dims; `Arg::Slot::dims` indexes into this.
    shapes: Vec<usize>,
    /// Program config, so eps/rope settings match the interpreter.
    config: crate::ir::Config,
    /// Every instruction output, in program order.
    streams: Vec<(String, Slot)>,
    outputs: Vec<(String, Slot)>,
    words: usize,
}

impl Plan {
    /// Size every stream from the listing's inferred shapes.
    ///
    /// The loop is uniform across mnemonics: arity comes from
    /// [`crate::ops::registry`], the kernel comes from [`kernel_for`], and the
    /// only structural rules are that outputs are contiguous and sit above
    /// every operand. Shapes travel with the slots, so a kernel like
    /// `IMATMUL` gets its strides without the plan knowing what a matrix is.
    pub fn compile(bound: &BoundProgram, sample: &Feeds, outputs: &[String]) -> Result<Plan> {
        let streams = infer_shapes(bound, sample)?;
        let registry = crate::ops::registry();
        let mut slots: HashMap<String, Slot> = HashMap::new();
        let mut words = 0usize;
        let mut shape_data: Vec<usize> = Vec::new();

        let take = |name: &str, words: &mut usize, shape_data: &mut Vec<usize>| -> Result<Slot> {
            let s = streams
                .get(name)
                .ok_or_else(|| Error::asm(format!("stream arena: no shape for {name:?}")))?;
            let is_f = match s.kind {
                CKind::I => false,
                CKind::F => true,
                CKind::T => {
                    return Err(Error::unimplemented(format!(
                        "stream arena: stream {name:?} is T, not ported"
                    )))
                }
            };
            let mut dims = Vec::with_capacity(s.shape.len());
            for d in &s.shape {
                dims.push(d.ok_or_else(|| {
                    Error::asm(format!("stream arena: {name:?} has a dynamic shape"))
                })?);
            }
            let len = s
                .elems()
                .ok_or_else(|| Error::asm(format!("stream arena: {name:?} has a dynamic shape")))?;
            let dim_start = shape_data.len();
            shape_data.extend_from_slice(&dims);
            let off = *words;
            *words = words
                .checked_add(len)
                .ok_or_else(|| Error::asm("stream arena: word count overflows"))?;
            Ok(Slot {
                off,
                len,
                dims: (dim_start, dims.len()),
                is_f,
            })
        };

        let mut inputs = Vec::new();
        for decl in &bound.inputs {
            let slot = take(&decl.name, &mut words, &mut shape_data)?;
            inputs.push((decl.name.clone(), slot));
            slots.insert(decl.name.clone(), slot);
        }

        let mut streams: Vec<(String, Slot)> = Vec::new();

        // Only I64/F64 payloads are representable as arena words. Anything
        // else (F32, U8, Bool, T2) falls back to the interpreter.
        for (name, slot) in &inputs {
            match sample.get(name) {
                Some(Value::I64(_)) if !slot.is_f => {}
                Some(Value::F64(_)) if slot.is_f => {}
                Some(_) => {
                    return Err(Error::unimplemented(format!(
                        "stream arena: input {name:?} is not I64/F64"
                    )))
                }
                None => {}
            }
        }

        let mut steps = Vec::new();
        let mut args: Vec<Arg> = Vec::new();
        for ins in &bound.instrs {
            if ins.is_repeat() || ins.is_end() || ins.is_break() {
                return Err(Error::unimplemented(format!(
                    "stream arena: {} not ported",
                    ins.mn
                )));
            }
            let def = registry
                .get(&ins.mn)
                .ok_or_else(|| Error::asm(format!("stream arena: unknown mnemonic {}", ins.mn)))?;
            if ins.args.len() != def.arity_in || ins.outs.len() != def.arity_out {
                return Err(Error::asm(format!(
                    "stream arena: {} arity mismatch (listing {} in / {} out, registry {} / {})",
                    ins.mn,
                    ins.args.len(),
                    ins.outs.len(),
                    def.arity_in,
                    def.arity_out
                )));
            }
            let (kernel, matching) = kernel_for(&ins.mn).ok_or_else(|| {
                Error::unimplemented(format!("stream arena: {} not ported", ins.mn))
            })?;

            // Allocate every output of this instruction before recording the
            // step: they are therefore contiguous, and `run` can split one
            // buffer into N destinations with `split_at_mut` and no allocation.
            let dsts: Vec<Slot> = ins
                .outs
                .iter()
                .map(|o| take(o, &mut words, &mut shape_data))
                .collect::<Result<_>>()?;
            let dst = dsts[0].off;
            let mut dst_len = 0usize;
            for s in &dsts {
                if s.off != dst + dst_len {
                    return Err(Error::asm(format!(
                        "stream arena: {} outputs are not contiguous",
                        ins.mn
                    )));
                }
                dst_len += s.len;
            }

            let arg_start = args.len();
            for (i, a) in ins.args.iter().enumerate() {
                let compiled = operand(&slots, a)?;
                if let Arg::Slot { off, len, .. } = compiled {
                    if off + len > dst {
                        return Err(Error::asm(format!(
                            "stream arena: {} operand overlaps the destination block",
                            ins.mn
                        )));
                    }
                    if i < matching && len != dst_len {
                        return Err(Error::asm(format!(
                            "stream arena: {} operand {i} is {len} elements, output is {dst_len}",
                            ins.mn
                        )));
                    }
                }
                args.push(compiled);
            }
            steps.push(Step {
                kernel,
                dst,
                dst_len,
                args: arg_start..args.len(),
            });
            for (i, name) in ins.outs.iter().enumerate() {
                slots.insert(name.clone(), dsts[i]);
                streams.push((name.clone(), dsts[i]));
            }
        }

        let mut outs = Vec::new();
        for name in outputs {
            let s = slot_of(&slots, name)?;
            outs.push((name.clone(), s));
        }

        let bytes = words * 8;
        if bytes > max_bytes() {
            return Err(Error::unimplemented(format!(
                "stream arena: working set {bytes} bytes exceeds cap {} (JOTNAR_ARENA_MAX_MB)",
                max_bytes()
            )));
        }

        Ok(Plan {
            inputs,
            steps,
            args,
            shapes: shape_data,
            config: bound.config.clone(),
            streams,
            outputs: outs,
            words,
        })
    }

    /// Words the arena needs for one job.
    pub fn words(&self) -> usize {
        self.words
    }

    pub fn bytes(&self) -> usize {
        self.words * 8
    }

    /// Run the plan. After the first call on a reserved arena this allocates
    /// nothing: the only writes are into `arena` and `out`.
    fn dims(&self, slot: Slot) -> &[usize] {
        &self.shapes[slot.dims.0..slot.dims.0 + slot.dims.1]
    }

    /// Materialize every resulting stream as a `Value`, which is the shape the
    /// interpreter returns. Only the inputs and the final outputs allocate;
    /// every intermediate lives in the arena.
    pub fn run_feeds(&self, arena: &mut StreamArena, feeds: &Feeds, out: &mut Feeds) -> Result<()> {
        self.execute(arena, feeds)?;
        out.clear();
        let words = arena.words();
        for (name, slot) in self.inputs.iter().chain(self.streams.iter()) {
            let raw = &words[slot.off..slot.off + slot.len];
            let value = if slot.is_f {
                let vals: Vec<f64> = raw.iter().map(|w| f64_of(*w)).collect();
                Value::F64(crate::value::array_from_vec(self.dims(*slot), vals)?)
            } else {
                let vals: Vec<i64> = raw.to_vec();
                Value::I64(crate::value::array_from_vec(self.dims(*slot), vals)?)
            };
            out.insert(name.clone(), value);
        }
        Ok(())
    }

    fn execute(&self, arena: &mut StreamArena, feeds: &Feeds) -> Result<()> {
        arena.reset();
        let start = arena.alloc(self.words)?;
        if start != 0 || arena.used_words() != self.words {
            return Err(Error::runtime("stream arena: offset layout drifted"));
        }
        for (name, slot) in &self.inputs {
            let (off, len) = (slot.off, slot.len);
            let v = feeds
                .get(name)
                .ok_or_else(|| Error::runtime(format!("stream arena: missing input {name:?}")))?;
            let words = arena.words_mut();
            let dst = &mut words[off..off + len];
            match (v, slot.is_f) {
                (Value::I64(a), false) => {
                    let src = a.as_slice().ok_or_else(|| {
                        Error::runtime(format!("stream arena: input {name:?} is not contiguous"))
                    })?;
                    if src.len() != len {
                        return Err(Error::runtime(format!(
                            "stream arena: input {name:?} has {} elements, plan wants {len}",
                            src.len()
                        )));
                    }
                    dst.copy_from_slice(src);
                }
                (Value::F64(a), true) => {
                    let src = a.as_slice().ok_or_else(|| {
                        Error::runtime(format!("stream arena: input {name:?} is not contiguous"))
                    })?;
                    if src.len() != len {
                        return Err(Error::runtime(format!(
                            "stream arena: input {name:?} has {} elements, plan wants {len}",
                            src.len()
                        )));
                    }
                    for (d, x) in dst.iter_mut().zip(src.iter()) {
                        *d = word_of(*x);
                    }
                }
                _ => {
                    return Err(Error::runtime(format!(
                        "stream arena: input {name:?} kind does not match the plan"
                    )))
                }
            }
        }

        let cfg = &self.config;
        for step in &self.steps {
            let (head, tail) = arena.words_mut().split_at_mut(step.dst);
            let head: &[i64] = head;
            let out = &mut tail[..step.dst_len];
            let mut ctx = Ctx {
                head,
                out,
                shapes: &self.shapes,
            };
            (step.kernel)(&mut ctx, &self.args[step.args.clone()], cfg)?;
        }
        Ok(())
    }

    /// Encode the requested outputs as little-endian `i64` bytes.
    pub fn run(&self, arena: &mut StreamArena, feeds: &Feeds, out: &mut Vec<u8>) -> Result<()> {
        self.execute(arena, feeds)?;
        let words = arena.words();
        out.clear();
        for (_name, slot) in &self.outputs {
            for v in &words[slot.off..slot.off + slot.len] {
                out.extend_from_slice(&v.to_le_bytes());
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::asm::{assemble, AsmOptions};
    use crate::value::array_from_vec;

    fn int_feeds(vals: &[i64]) -> Feeds {
        let mut f = Feeds::new();
        f.insert(
            "X".to_string(),
            Value::I64(array_from_vec(&[vals.len()], vals.to_vec()).unwrap()),
        );
        f
    }

    fn parity(text: &str) {
        let vals = [-7i64, -1, 0, 3, 9, 64, -64, 12345];
        let f = int_feeds(&vals);
        assert_eq!(
            arena_bytes(text, &f, "OUT"),
            interp_bytes(text, &f, "OUT"),
            "byte mismatch for {text:?}"
        );
    }

    /// The whole elementwise integer family, against the reference interpreter.
    #[test]
    fn elementwise_family_matches_interpreter() {
        for text in [
            "IN X\nOUT = IADD(X, X)\n",
            "IN X\nOUT = ISUB(X, X)\n",
            "IN X\nOUT = IMUL(X, X)\n",
            "IN X\nOUT = IAND(X, X)\n",
            "IN X\nOUT = IOR(X, X)\n",
            "IN X\nOUT = IXOR(X, X)\n",
            "IN X\nOUT = IEQ(X, X)\n",
            "IN X\nOUT = ILT(X, X)\n",
            "IN X\nOUT = IMAX(X, X)\n",
            "IN X\nOUT = SADD(X, X)\n",
            "IN X\nOUT = IADD(X, 5)\n",
            "IN X\nOUT = IMUL(X, -3)\n",
            "IN X\nOUT = IAND(X, 12)\n",
            "IN X\nOUT = IDIV(X, 3)\n",
            "IN X\nOUT = IMOD(X, 3)\n",
            "IN X\nOUT = IDIVU(X, 3)\n",
            "IN X\nOUT = IMODU(X, 3)\n",
            "IN X\nOUT = ISHL(X, 5)\n",
            "IN X\nOUT = ISHR(X, 5)\n",
            "IN X\nOUT = ISAR(X, 2)\n",
            "IN X\nOUT = IMAD(X, X, 3)\n",
            "IN X\nOUT = ISELECT(X, X, 7)\n",
        ] {
            parity(text);
        }
    }

    /// Division by zero and out-of-range shifts fail on both paths.
    #[test]
    fn division_errors_match_the_interpreter() {
        for text in [
            "IN X\nOUT = IDIV(X, 0)\n",
            "IN X\nOUT = IMOD(X, 0)\n",
            "IN X\nOUT = IDIVU(X, 0)\n",
            "IN X\nOUT = IMODU(X, 0)\n",
            "IN X\nOUT = ISAR(X, 64)\n",
        ] {
            let f = int_feeds(&[1, 2, 3]);
            let bound = assemble(text, &AsmOptions::default()).unwrap();
            let plan = Plan::compile(&bound, &f, &["OUT".to_string()]).unwrap();
            let mut arena = StreamArena::new();
            arena.reserve_words(plan.words());
            let mut out = Vec::new();
            assert!(
                plan.run(&mut arena, &f, &mut out).is_err(),
                "{text} should fail"
            );
            assert!(
                crate::runtime::run_with_options(
                    &bound,
                    &f,
                    &crate::runtime::ExecOptions::default()
                )
                .is_err(),
                "{text} should fail in the interpreter too"
            );
        }
    }

    fn feeds_n(items: &[(&str, &[i64], &[usize])]) -> Feeds {
        let mut f = Feeds::new();
        for (name, vals, shape) in items {
            f.insert(
                name.to_string(),
                Value::I64(array_from_vec(shape, vals.to_vec()).unwrap()),
            );
        }
        f
    }

    fn parity_with(text: &str, f: &Feeds) {
        assert_eq!(
            arena_bytes(text, f, "OUT"),
            interp_bytes(text, f, "OUT"),
            "byte mismatch for {text:?}"
        );
    }

    /// Indexing, windows, gather/scatter, packing, checksums.
    #[test]
    fn byte_and_index_ops_match_interpreter() {
        let x: Vec<i64> = (0..8).collect();
        parity_with("IN X\nOUT = IGET(X, 3)\n", &feeds_n(&[("X", &x, &[8])]));
        parity_with(
            "IN X\nOUT = IWINDOW(X, 2, 4)\n",
            &feeds_n(&[("X", &x, &[8])]),
        );
        // Dynamic start: a one-element stream.
        parity_with(
            "IN X\nIN S\nOUT = IWINDOW(X, S, 4)\n",
            &feeds_n(&[("X", &x, &[8]), ("S", &[1], &[1])]),
        );
        parity_with(
            "IN X\nIN P\nOUT = IFIND(X, P)\n",
            &feeds_n(&[("X", &[1, 2, 3, 4, 2, 3, 5], &[7]), ("P", &[2, 3], &[2])]),
        );
        parity_with(
            "IN X\nIN P\nOUT = IFIND(X, P)\n",
            &feeds_n(&[("X", &[1, 2, 3], &[3]), ("P", &[9], &[1])]),
        );
        parity_with(
            "IN T\nIN I\nOUT = IGATHER1(T, I)\n",
            &feeds_n(&[("T", &[10, 20, 30, 40], &[4]), ("I", &[3, 0, 2], &[3])]),
        );
        parity_with(
            "IN B\nIN I\nIN V\nOUT = ISCATTER1(B, I, V)\n",
            &feeds_n(&[
                ("B", &[1, 2, 3, 4], &[4]),
                ("I", &[0, 2], &[2]),
                ("V", &[9, 8], &[2]),
            ]),
        );
        parity_with(
            "IN X\nOUT = IASSEMBLE(X, 0, 4)\n",
            &feeds_n(&[("X", &[0x78, 0x56, 0x34, 0x12], &[4])]),
        );
        parity_with("IN X\nOUT = ICRC32(X)\n", &feeds_n(&[("X", &x, &[8])]));
        parity_with(
            "IN X\nIN S\nOUT = ICRC32C(X, S)\n",
            &feeds_n(&[("X", &x, &[8]), ("S", &[0], &[1])]),
        );
    }

    /// The index-taking ops fail the same way as the interpreter.
    #[test]
    fn index_errors_match_the_interpreter() {
        let cases: [(&str, Feeds); 3] = [
            (
                "IN X\nOUT = IGET(X, 99)\n",
                feeds_n(&[("X", &[1, 2, 3], &[3])]),
            ),
            (
                "IN T\nIN I\nOUT = IGATHER1(T, I)\n",
                feeds_n(&[("T", &[1, 2], &[2]), ("I", &[5], &[1])]),
            ),
            (
                "IN X\nOUT = IWINDOW(X, 1, 9)\n",
                feeds_n(&[("X", &[1, 2, 3], &[3])]),
            ),
        ];
        for (text, f) in cases {
            let bound = match assemble(text, &AsmOptions::default()) {
                Ok(b) => b,
                Err(_) => continue, // inference can refuse it first
            };
            let plan = match Plan::compile(&bound, &f, &["OUT".to_string()]) {
                Ok(p) => p,
                Err(_) => {
                    // Inference refused it; the listing is not runnable at all.
                    continue;
                }
            };
            let mut arena = StreamArena::new();
            arena.reserve_words(plan.words());
            let mut out = Vec::new();
            assert!(plan.run(&mut arena, &f, &mut out).is_err(), "{text}");
            assert!(
                crate::runtime::run_with_options(
                    &bound,
                    &f,
                    &crate::runtime::ExecOptions::default()
                )
                .is_err(),
                "{text} should fail in the interpreter too"
            );
        }
    }

    /// `IASSEMBLE` with a full 8-byte window and the top bit set.
    ///
    /// The old interpreter computed `v - 2^64` in `u128`, which panicked in
    /// debug. Both paths now truncate the low 64 bits, so this is parity, not
    /// an arena-only exception.
    #[test]
    fn iassemble_sign_bit_matches_interpreter() {
        let f = feeds_n(&[("X", &[0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF], &[8])]);
        let text = "IN X\nOUT = IASSEMBLE(X, 0, 8)\n";
        let got = arena_bytes(text, &f, "OUT");
        assert_eq!(got, (-1i64).to_le_bytes().to_vec());
        assert_eq!(got, interp_bytes(text, &f, "OUT"));
    }

    #[test]
    fn slice_matches_interpreter() {
        // `infer_shapes` allows 2-D axis 0/1 SLICE on I; 1-D uses IWINDOW.
        let m = [1i64, 2, 3, 4, 5, 6];
        parity_with(
            "IN X\nOUT = SLICE(X, 0, 0, 2)\n",
            &feeds_n(&[("X", &m, &[2, 3])]),
        );
        parity_with(
            "IN X\nOUT = SLICE(X, 1, 1, 3)\n",
            &feeds_n(&[("X", &m, &[2, 3])]),
        );
    }

    #[test]
    fn size_cap_refuses_an_oversized_listing() {
        let f = feeds_n(&[("X", &vec![1i64; 1024], &[1024])]);
        let bound = assemble("IN X\nOUT = IADD(X, X)\n", &AsmOptions::default()).unwrap();
        let prev = set_max_bytes(Some(4096));
        let e = Plan::compile(&bound, &f, &["OUT".to_string()]).unwrap_err();
        assert!(e.to_string().contains("exceeds cap"), "{e}");
        set_max_bytes(prev);
        assert!(Plan::compile(&bound, &f, &["OUT".to_string()]).is_ok());
    }

    fn f_feeds(vals: &[f64]) -> Feeds {
        let mut f = Feeds::new();
        f.insert(
            "X".to_string(),
            Value::F64(array_from_vec(&[vals.len()], vals.to_vec()).unwrap()),
        );
        f
    }

    fn feeds_f(items: &[(&str, &[f64], &[usize])]) -> Feeds {
        let mut f = Feeds::new();
        for (name, vals, shape) in items {
            f.insert(
                name.to_string(),
                Value::F64(array_from_vec(shape, vals.to_vec()).unwrap()),
            );
        }
        f
    }

    /// Add an I stream to an existing feed set (mixed-kind fixtures).
    fn add_i(mut f: Feeds, name: &str, vals: &[i64], shape: &[usize]) -> Feeds {
        f.insert(
            name.to_string(),
            Value::I64(array_from_vec(shape, vals.to_vec()).unwrap()),
        );
        f
    }

    #[test]
    fn resize2d_matches_interpreter() {
        // 1x1 -> 2x2 is pure replication.
        parity_with(
            "IN X\nOUT = RESIZE2D(X, 2, 2, 0)\n",
            &feeds_f(&[("X", &[5.0], &[1, 1, 1])]),
        );
        // Upsample HWC with both align conventions.
        let img: Vec<f64> = (0..8).map(|i| i as f64 * 0.25).collect();
        for align in [0, 1] {
            let text = format!("IN X\nOUT = RESIZE2D(X, 3, 3, {align})\n");
            parity_with(&text, &feeds_f(&[("X", &img, &[2, 2, 2])]));
        }
        // Downsample, non-square target.
        let img: Vec<f64> = (0..9).map(|i| i as f64 / 8.0).collect();
        parity_with(
            "IN X\nOUT = RESIZE2D(X, 2, 3, 0)\n",
            &feeds_f(&[("X", &img, &[3, 3, 1])]),
        );
    }

    #[test]
    fn conv2d_matches_interpreter() {
        // 3x3x1 through a 2x2 single-channel kernel.
        let x: Vec<f64> = (0..9).map(|i| i as f64).collect();
        let f = feeds_f(&[
            ("X", &x, &[3, 3, 1]),
            ("W", &[1.0, 0.0, 0.0, 1.0], &[1, 2, 2, 1]),
            ("B", &[0.5], &[1]),
        ]);
        parity_with("IN X\nIN W\nIN B\nOUT = CONV2D(X, W, B, 1, 0)\n", &f);

        // Stride 2, padding 1, two output channels.
        let x: Vec<f64> = (0..16).map(|i| (i as f64) * 0.125).collect();
        let w: Vec<f64> = (0..8).map(|i| (i as f64) * 0.25 - 0.5).collect();
        let f = feeds_f(&[
            ("X", &x, &[4, 4, 1]),
            ("W", &w, &[2, 2, 2, 1]),
            ("B", &[0.25, -0.75], &[2]),
        ]);
        parity_with("IN X\nIN W\nIN B\nOUT = CONV2D(X, W, B, 2, 1)\n", &f);
    }

    #[test]
    fn deconv2d_matches_interpreter() {
        // 2x2x1 through a 2x2 kernel at stride 1 -> 3x3.
        let f = feeds_f(&[
            ("X", &[1.0, 2.0, 3.0, 4.0], &[2, 2, 1]),
            ("W", &[1.0, -1.0, 0.5, 2.0], &[1, 2, 2, 1]),
            ("B", &[0.25], &[1]),
        ]);
        parity_with("IN X\nIN W\nIN B\nOUT = DECONV2D(X, W, B, 1, 0)\n", &f);

        // Stride 2 with padding: the accumulate-and-zero path.
        let f = feeds_f(&[
            ("X", &[1.0, -2.0, 0.5, 3.0], &[2, 2, 1]),
            (
                "W",
                &[0.5, 1.0, -1.0, 2.0, 0.25, 0.75, 1.5, -0.5, 1.0],
                &[1, 3, 3, 1],
            ),
            ("B", &[-0.5], &[1]),
        ]);
        parity_with("IN X\nIN W\nIN B\nOUT = DECONV2D(X, W, B, 2, 1)\n", &f);
    }

    #[test]
    fn float_paths_match_interpreter() {
        let vals = [
            -2.5f64,
            -0.0,
            0.0,
            1.5,
            3.25,
            f64::INFINITY,
            f64::NEG_INFINITY,
        ];
        let f = f_feeds(&vals);
        assert_eq!(
            arena_bytes("IN X\nOUT = RELU(X)\n", &f, "OUT"),
            interp_bytes("IN X\nOUT = RELU(X)\n", &f, "OUT"),
        );

        // F -> I -> F round trip through two kind changes.
        let text = "IN X\nI = F2I(X)\nOUT = I2F(I)\n";
        let g = f_feeds(&[-2.9, -0.0, 0.0, 1.9, 3.0]);
        assert_eq!(arena_bytes(text, &g, "OUT"), interp_bytes(text, &g, "OUT"));

        // F concatenation (infer allows F for 2-D axis 1).
        let mut h = Feeds::new();
        h.insert(
            "A".to_string(),
            Value::F64(array_from_vec(&[2, 1], vec![1.0, 2.0]).unwrap()),
        );
        h.insert(
            "B".to_string(),
            Value::F64(array_from_vec(&[2, 2], vec![3.0, 4.0, 5.0, 6.0]).unwrap()),
        );
        assert_eq!(
            arena_bytes("IN A\nIN B\nOUT = CONCAT(A, B, 1)\n", &h, "OUT"),
            interp_bytes("IN A\nIN B\nOUT = CONCAT(A, B, 1)\n", &h, "OUT"),
        );
    }

    #[test]
    fn f2i_rejects_non_finite_like_the_interpreter() {
        let f = f_feeds(&[1.0, f64::NAN]);
        let text = "IN X\nOUT = F2I(X)\n";
        let bound = assemble(text, &AsmOptions::default()).unwrap();
        let plan = Plan::compile(&bound, &f, &["OUT".to_string()]).unwrap();
        let mut arena = StreamArena::new();
        arena.reserve_words(plan.words());
        let mut out = Vec::new();
        assert!(plan.run(&mut arena, &f, &mut out).is_err());
        assert!(crate::runtime::run_with_options(
            &bound,
            &f,
            &crate::runtime::ExecOptions::default()
        )
        .is_err());
    }

    fn mat_feeds(a: &[i64], ash: &[usize], b: &[i64], bsh: &[usize]) -> Feeds {
        let mut f = Feeds::new();
        f.insert(
            "A".to_string(),
            Value::I64(array_from_vec(ash, a.to_vec()).unwrap()),
        );
        f.insert(
            "B".to_string(),
            Value::I64(array_from_vec(bsh, b.to_vec()).unwrap()),
        );
        f
    }

    fn mat_parity(a: &[i64], ash: &[usize], b: &[i64], bsh: &[usize]) -> Vec<u8> {
        let text = "IN A\nIN B\nOUT = IMATMUL(A, B)\n";
        let f = mat_feeds(a, ash, b, bsh);
        let got = arena_bytes(text, &f, "OUT");
        assert_eq!(got, interp_bytes(text, &f, "OUT"), "matmul mismatch");
        got
    }

    #[test]
    fn matmul_matches_interpreter() {
        // 2x2 * 2x2
        let got = mat_parity(&[1, 2, 3, 4], &[2, 2], &[5, 6, 7, 8], &[2, 2]);
        let want: Vec<i64> = vec![19, 22, 43, 50];
        assert_eq!(
            got,
            want.iter()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<_>>()
        );

        // 2x3 * 3x2 — the output shape differs from both operands.
        let got = mat_parity(
            &[1, 2, 3, 4, 5, 6],
            &[2, 3],
            &[7, 8, 9, 10, 11, 12],
            &[3, 2],
        );
        let want: Vec<i64> = vec![58, 64, 139, 154];
        assert_eq!(
            got,
            want.iter()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<_>>()
        );

        // Wrapping at every multiply and add, like the scalar graph it replaces.
        let big = i64::MAX / 2 + 1;
        mat_parity(&[big, -big, big, 1], &[2, 2], &[big, 1, -big, 2], &[2, 2]);
    }

    #[test]
    fn matmul_plan_sizes_the_output_shape() {
        let text = "IN A\nIN B\nOUT = IMATMUL(A, B)\n";
        let f = mat_feeds(&[1, 2, 3, 4], &[2, 2], &[5, 6, 7, 8], &[2, 2]);
        let bound = assemble(text, &AsmOptions::default()).unwrap();
        let plan = Plan::compile(&bound, &f, &["OUT".to_string()]).unwrap();
        // A (4) + B (4) + OUT (4)
        assert_eq!(plan.words(), 12);
    }

    fn feeds(n: usize) -> Feeds {
        let mut f = Feeds::new();
        f.insert(
            "X".to_string(),
            Value::I64(array_from_vec(&[n], (0..n as i64).collect()).unwrap()),
        );
        f
    }

    fn chain(n: usize) -> String {
        let mut s = String::from("IN X\n");
        s.push_str("t0 = IADD(X, X)\n");
        for i in 1..n {
            s.push_str(&format!("t{i} = IMUL(t{}, X)\n", i - 1));
        }
        s.push_str(&format!("OUT = IADD(t{}, X)\n", n - 1));
        s
    }

    #[test]
    fn plan_sizes_every_stream_from_inferred_shapes() {
        let text = chain(3);
        let bound = assemble(&text, &AsmOptions::default()).unwrap();
        let plan = Plan::compile(&bound, &feeds(4), &["OUT".to_string()]).unwrap();
        // IN X, t0, t1, t2, OUT = 5 streams of 4 words.
        assert_eq!(plan.words(), 20);
        assert_eq!(plan.bytes(), 160);
        let mut arena = StreamArena::new();
        arena.reserve_words(plan.words());
        let mut out = Vec::new();
        plan.run(&mut arena, &feeds(4), &mut out).unwrap();
        assert_eq!(out.len(), 4 * 8);
    }

    #[test]
    fn reset_reuses_the_same_offsets() {
        let bound = assemble(&chain(2), &AsmOptions::default()).unwrap();
        let plan = Plan::compile(&bound, &feeds(4), &["OUT".to_string()]).unwrap();
        let mut arena = StreamArena::new();
        arena.reserve_words(plan.words());
        let mut a = Vec::new();
        let mut b = Vec::new();
        plan.run(&mut arena, &feeds(4), &mut a).unwrap();
        plan.run(&mut arena, &feeds(4), &mut b).unwrap();
        assert_eq!(a, b);
        assert_eq!(arena.used_words(), plan.words());
        assert_eq!(arena.resets(), 2);
    }

    fn arena_bytes(text: &str, feeds: &Feeds, name: &str) -> Vec<u8> {
        let bound = assemble(text, &AsmOptions::default()).unwrap();
        let plan = Plan::compile(&bound, &feeds, &[name.to_string()]).unwrap();
        let mut arena = StreamArena::new();
        arena.reserve_words(plan.words());
        let mut out = Vec::new();
        plan.run(&mut arena, &feeds, &mut out).unwrap();
        out
    }

    fn interp_bytes(text: &str, feeds: &Feeds, name: &str) -> Vec<u8> {
        let bound = assemble(text, &AsmOptions::default()).unwrap();
        let got = crate::runtime::run_with_options(
            &bound,
            feeds,
            &crate::runtime::ExecOptions::default(),
        )
        .unwrap();
        match &got[name] {
            Value::I64(a) => a.iter().flat_map(|v| v.to_le_bytes()).collect(),
            Value::F64(a) => a.iter().flat_map(|v| v.to_le_bytes()).collect(),
            other => panic!("expected I64/F64, got {:?}", other.kind()),
        }
    }

    #[test]
    fn reduce_sum_matches_interpreter() {
        let text = "IN X\nOUT = IREDUCE_SUM(X)\n";
        let f = feeds(16);
        let got = arena_bytes(text, &f, "OUT");
        assert_eq!(got, interp_bytes(text, &f, "OUT"));
        let sum: i64 = (0..16).sum();
        assert_eq!(got, sum.to_le_bytes().to_vec());
    }

    #[test]
    fn reduction_changes_shape_mid_graph() {
        // X:[N] -> S:[] -> OUT:[]  — the destination is one word where the
        // source was N, and the slot ordering still lets `run` split safely.
        let text = "IN X\nS = IREDUCE_SUM(X)\nOUT = IADD(S, S)\n";
        let f = feeds(8);
        let bound = assemble(text, &AsmOptions::default()).unwrap();
        let plan = Plan::compile(&bound, &f, &["OUT".to_string()]).unwrap();
        // X (8) + S (1) + OUT (1)
        assert_eq!(plan.words(), 10);
        let mut arena = StreamArena::new();
        arena.reserve_words(plan.words());
        let mut out = Vec::new();
        plan.run(&mut arena, &f, &mut out).unwrap();
        assert_eq!(out, interp_bytes(text, &f, "OUT"));
        let sum: i64 = (0..8).sum();
        assert_eq!(out, (sum * 2).to_le_bytes().to_vec());
    }

    #[test]
    fn windowed_sum_matches_interpreter() {
        let text = "IN X\nOUT = ISUM(X, 1, 3)\n";
        let mut f = Feeds::new();
        f.insert(
            "X".to_string(),
            Value::I64(array_from_vec(&[5], vec![10, 20, 30, 40, 50]).unwrap()),
        );
        let got = arena_bytes(text, &f, "OUT");
        assert_eq!(got, interp_bytes(text, &f, "OUT"));
        assert_eq!(got, 90i64.to_le_bytes().to_vec());
    }

    #[test]
    fn windowed_sum_validates_its_window() {
        let text = "IN X\nOUT = ISUM(X, 3, 9)\n";
        let bound = match assemble(text, &AsmOptions::default()) {
            Ok(b) => b,
            // Inference may reject the window first; that is also fail-loud.
            Err(_) => return,
        };
        let e = Plan::compile(&bound, &feeds(5), &["OUT".to_string()]).unwrap_err();
        assert!(e.to_string().contains("ISUM"), "{e}");
    }

    fn varint_feeds(bytes: &[i64]) -> Feeds {
        let mut f = Feeds::new();
        f.insert(
            "X".to_string(),
            Value::I64(array_from_vec(&[bytes.len()], bytes.to_vec()).unwrap()),
        );
        f
    }

    #[test]
    fn multi_output_varint_matches_interpreter() {
        let text = "IN X\nV, N = IVARINT(X, 0)\nOUT = IADD(V, N)\n";
        // Jotnar's groups are most-significant first: (44 << 7) | 2 = 5634.
        // (This is not LEB128 order; the probe mirrors `op_ivarint` exactly.)
        let f = varint_feeds(&[0xAC, 0x02, 0, 0, 0, 0, 0, 0]);
        assert_eq!(arena_bytes(text, &f, "V"), 5634i64.to_le_bytes().to_vec());
        assert_eq!(arena_bytes(text, &f, "N"), 2i64.to_le_bytes().to_vec());
        assert_eq!(arena_bytes(text, &f, "OUT"), 5636i64.to_le_bytes().to_vec());
        assert_eq!(arena_bytes(text, &f, "OUT"), interp_bytes(text, &f, "OUT"));
    }

    #[test]
    fn multi_output_slots_are_adjacent() {
        let text = "IN X\nV, N = IVARINT(X, 0)\nOUT = IADD(V, N)\n";
        let f = varint_feeds(&[0xAC, 0x02, 0, 0, 0, 0, 0, 0]);
        let bound = assemble(text, &AsmOptions::default()).unwrap();
        let plan = Plan::compile(&bound, &f, &["OUT".to_string()]).unwrap();
        // X (8) + V (1) + N (1) + OUT (1)
        assert_eq!(plan.words(), 11);
        let mut arena = StreamArena::new();
        arena.reserve_words(plan.words());
        let mut out = Vec::new();
        plan.run(&mut arena, &f, &mut out).unwrap();
        assert_eq!(out, 5636i64.to_le_bytes().to_vec());
    }

    #[test]
    fn varint_ninth_byte_matches_interpreter() {
        let text = "IN X\nV, N = IVARINT(X, 0)\nOUT = IADD(V, N)\n";
        let f = varint_feeds(&[0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x7F]);
        assert_eq!(arena_bytes(text, &f, "V"), 127i64.to_le_bytes().to_vec());
        assert_eq!(arena_bytes(text, &f, "N"), 9i64.to_le_bytes().to_vec());
        assert_eq!(arena_bytes(text, &f, "OUT"), interp_bytes(text, &f, "OUT"));
    }

    #[test]
    fn varint_truncated_is_a_typed_error() {
        let text = "IN X\nV, N = IVARINT(X, 0)\nOUT = IADD(V, N)\n";
        let f = varint_feeds(&[0x80, 0x80]);
        let bound = assemble(text, &AsmOptions::default()).unwrap();
        let plan = Plan::compile(&bound, &f, &["OUT".to_string()]).unwrap();
        let mut arena = StreamArena::new();
        arena.reserve_words(plan.words());
        let mut out = Vec::new();
        let e = plan.run(&mut arena, &f, &mut out).unwrap_err();
        assert!(e.to_string().contains("truncated"), "{e}");
    }

    /// The table is keyed by mnemonic and borrows the registry's arity, so it
    /// must never name something the language does not have.
    #[test]
    fn reshape_matches_interpreter() {
        let xi = [0i64, 1, 2, 3, 4, 5];
        parity_with(
            "IN X\nOUT = RESHAPE2(X, 3, 2)\n",
            &feeds_n(&[("X", &xi, &[2, 3])]),
        );
        let xf: Vec<f64> = (0..6).map(|i| i as f64 * 0.5).collect();
        parity_with(
            "IN X\nOUT = RESHAPE3(X, 1, 2, 3)\n",
            &feeds_f(&[("X", &xf, &[2, 3])]),
        );
    }

    #[test]
    fn norms_match_interpreter() {
        let x = [-1.5, 0.25, 2.0, -0.75, 3.5, -2.0, 0.0, 1.25];
        let w = [1.0, 0.5, -1.0, 2.0];
        let b = [0.25, -0.25, 0.0, 1.0];
        let f = feeds_f(&[("X", &x, &[2, 4]), ("W", &w, &[4]), ("B", &b, &[4])]);
        parity_with(
            "CONFIG eps_ln 1e-6\nIN X\nIN W\nIN B\nOUT = LAYERNORM(X, W, B)\n",
            &f,
        );
        parity_with("CONFIG eps_rms 1e-6\nIN X\nIN W\nOUT = RMSNORM(X, W)\n", &f);
        // Rank > 2: the norm is over the last axis only.
        let x3: Vec<f64> = (0..16).map(|i| (i as f64 - 8.0) * 0.25).collect();
        parity_with(
            "CONFIG eps_ln 1e-5\nIN X\nIN W\nIN B\nOUT = LAYERNORM(X, W, B)\n",
            &feeds_f(&[("X", &x3, &[2, 2, 4]), ("W", &w, &[4]), ("B", &b, &[4])]),
        );
    }

    #[test]
    fn rotary_matches_interpreter() {
        let x: Vec<f64> = (0..8).map(|i| (i as f64 - 3.0) * 0.5).collect();
        // Per-row positions.
        parity_with(
            "CONFIG rope_base 10000\nIN X\nIN P\nOUT = ROTARY(X, P)\n",
            &add_i(feeds_f(&[("X", &x, &[2, 4])]), "P", &[1, 3], &[2]),
        );
    }

    fn f_mat(a: &[f64], ash: &[usize], b: &[f64], bsh: &[usize]) -> Feeds {
        feeds_f(&[("A", a, ash), ("B", b, bsh)])
    }

    #[test]
    fn float_matmul_matches_interpreter() {
        let f = f_mat(
            &[1.0, 2.0, 3.0, 4.0],
            &[2, 2],
            &[5.0, 6.0, 7.0, 8.0],
            &[2, 2],
        );
        parity_with("IN A\nIN B\nOUT = MATMUL(A, B)\n", &f);

        // 3-D batched.
        let a: Vec<f64> = (0..12).map(|i| i as f64 * 0.25 - 1.0).collect();
        let b: Vec<f64> = (0..12).map(|i| (i as f64 - 6.0) * 0.125).collect();
        let f = f_mat(&a, &[2, 2, 3], &b, &[2, 3, 2]);
        parity_with("IN A\nIN B\nOUT = BATCH_MATMUL(A, B)\n", &f);

        // Batch broadcast: B has one leading batch.
        let f = f_mat(&a, &[2, 2, 3], &b[..6], &[1, 3, 2]);
        parity_with("IN A\nIN B\nOUT = BATCH_MATMUL(A, B)\n", &f);
    }

    #[test]
    fn conv_and_interp_match_interpreter() {
        let x: Vec<f64> = (0..12).map(|i| i as f64 * 0.5).collect();
        let k = [0.0, 1.0, 0.0, 1.0, -4.0, 1.0, 0.0, 1.0, 0.0];
        parity_with(
            "IN X\nIN K\nOUT = CONV(X, K)\n",
            &feeds_f(&[("X", &x, &[3, 4]), ("K", &k, &[3, 3])]),
        );

        let img: Vec<f64> = (0..16).map(|i| i as f64 * 0.125).collect();
        parity_with(
            "IN X\nOUT = INTERP(X, 0.5, 0.5)\n",
            &feeds_f(&[("X", &img, &[4, 4])]),
        );
        let img3: Vec<f64> = (0..32).map(|i| (i as f64 - 16.0) * 0.25).collect();
        parity_with(
            "IN X\nOUT = INTERP(X, 2.0, 1.0)\n",
            &feeds_f(&[("X", &img3, &[4, 4, 2])]),
        );
    }

    #[test]
    fn gather_and_scatter_match_interpreter() {
        let ti = [10i64, 11, 20, 21, 30, 31, 40, 41];
        let tf: Vec<f64> = ti.iter().map(|v| *v as f64 * 0.5).collect();
        let ids = [3i64, 0, 2];

        parity_with(
            "IN T\nIN I\nOUT = GATHER(T, I)\n",
            &feeds_n(&[("T", &ti, &[4, 2]), ("I", &ids, &[3])]),
        );
        let mut f = Feeds::new();
        f.insert(
            "T".to_string(),
            Value::F64(array_from_vec(&[4, 2], tf).unwrap()),
        );
        f.insert(
            "I".to_string(),
            Value::I64(array_from_vec(&[3], ids.to_vec()).unwrap()),
        );
        parity_with("IN T\nIN I\nOUT = GATHER(T, I)\n", &f);

        // GATHER3: table (V1,V2,C) with paired i/j.
        let t3: Vec<f64> = (0..24).map(|i| i as f64 * 0.25).collect();
        let mut g = Feeds::new();
        g.insert(
            "T".to_string(),
            Value::F64(array_from_vec(&[3, 4, 2], t3).unwrap()),
        );
        g.insert(
            "II".to_string(),
            Value::I64(array_from_vec(&[2], vec![0, 2]).unwrap()),
        );
        g.insert(
            "JJ".to_string(),
            Value::I64(array_from_vec(&[2], vec![1, 3]).unwrap()),
        );
        parity_with("IN T\nIN II\nIN JJ\nOUT = GATHER3(T, II, JJ)\n", &g);

        // SCATTER: copy base, replace one row.
        let base: Vec<f64> = (0..12).map(|i| i as f64).collect();
        let row = [9.5, -1.0, 4.25];
        let mut s = Feeds::new();
        s.insert(
            "B".to_string(),
            Value::F64(array_from_vec(&[4, 3], base).unwrap()),
        );
        s.insert(
            "R".to_string(),
            Value::F64(array_from_vec(&[1, 3], row.to_vec()).unwrap()),
        );
        s.insert(
            "I".to_string(),
            Value::I64(array_from_vec(&[1], vec![2]).unwrap()),
        );
        parity_with("IN B\nIN R\nIN I\nOUT = SCATTER(B, R, I)\n", &s);
    }

    #[test]
    fn argmax_matches_interpreter() {
        let x = [1.0, 7.0, 3.0, 7.0, 2.0, -1.0, 0.0, 5.0];
        parity_with(
            "IN X\nOUT = ARGMAX(X, 1)\n",
            &feeds_f(&[("X", &x, &[2, 4])]),
        );
        // I input goes through the same f64 comparison, as the interpreter does.
        let xi = [1i64, 7, 3, 7, 2, -1, 0, 5];
        parity_with(
            "IN X\nOUT = ARGMAX(X, 1)\n",
            &feeds_n(&[("X", &xi, &[2, 4])]),
        );
        // 1-D input yields a scalar index.
        parity_with(
            "IN X\nOUT = ARGMAX(X, 0)\n",
            &feeds_f(&[("X", &[3.0, -2.0, 9.0, 9.0], &[4])]),
        );
    }

    #[test]
    fn topk_matches_interpreter() {
        let x = [1.0, 3.0, 3.0, 2.0, 3.0, -1.0, 5.0, 0.0, 5.0, 4.0];
        let f = feeds_f(&[("X", &x, &[2, 5])]);
        let text = "IN X\nV, I = TOPK(X, 3)\n";
        assert_eq!(
            arena_bytes(text, &f, "V"),
            interp_bytes(text, &f, "V"),
            "TOPK values"
        );
        assert_eq!(
            arena_bytes(text, &f, "I"),
            interp_bytes(text, &f, "I"),
            "TOPK indices"
        );
        // Ties keep the lower index first.
        assert_eq!(
            interp_bytes(text, &f, "I"),
            [1i64, 2, 4, 1, 3, 4]
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<_>>()
        );
        // Non-finite scores follow the same insertion rule on both paths.
        let g = feeds_f(&[("X", &[f64::NAN, 2.0, 3.0, 1.0], &[1, 4])]);
        let t2 = "IN X\nV, I = TOPK(X, 2)\n";
        assert_eq!(arena_bytes(t2, &g, "V"), interp_bytes(t2, &g, "V"));
        assert_eq!(arena_bytes(t2, &g, "I"), interp_bytes(t2, &g, "I"));
    }

    fn fails_both(text: &str, f: &Feeds) {
        let bound = assemble(text, &AsmOptions::default()).unwrap();
        assert!(
            crate::runtime::run_with_options(&bound, f, &crate::runtime::ExecOptions::default())
                .is_err(),
            "interpreter should reject {text:?}"
        );
        if let Ok(plan) = Plan::compile(&bound, f, &["OUT".to_string()]) {
            let mut arena = StreamArena::new();
            arena.reserve_words(plan.words());
            let mut out = Vec::new();
            assert!(
                plan.run(&mut arena, f, &mut out).is_err(),
                "arena should reject {text:?}"
            );
        }
    }

    #[test]
    fn new_family_errors_match_the_interpreter() {
        // Norm weight length disagrees with the last axis.
        fails_both(
            "CONFIG eps_ln 1e-6\nIN X\nIN W\nIN B\nOUT = LAYERNORM(X, W, B)\n",
            &feeds_f(&[
                ("X", &[1.0, 2.0, 3.0, 4.0], &[2, 2]),
                ("W", &[1.0, 1.0, 1.0], &[3]),
                ("B", &[0.0, 0.0, 0.0], &[3]),
            ]),
        );
        // Gather id past the table.
        fails_both(
            "IN T\nIN I\nOUT = GATHER(T, I)\n",
            &feeds_n(&[("T", &[1, 2], &[1, 2]), ("I", &[5], &[1])]),
        );
        // Gather3 j past the middle axis.
        let mut g = Feeds::new();
        g.insert(
            "T".to_string(),
            Value::F64(array_from_vec(&[2, 2, 1], vec![0.0; 4]).unwrap()),
        );
        g.insert(
            "II".to_string(),
            Value::I64(array_from_vec(&[1], vec![0]).unwrap()),
        );
        g.insert(
            "JJ".to_string(),
            Value::I64(array_from_vec(&[1], vec![7]).unwrap()),
        );
        fails_both("IN T\nIN II\nIN JJ\nOUT = GATHER3(T, II, JJ)\n", &g);
        // Scatter idx outside the base rows.
        let mut s = Feeds::new();
        s.insert(
            "B".to_string(),
            Value::F64(array_from_vec(&[2, 2], vec![0.0; 4]).unwrap()),
        );
        s.insert(
            "R".to_string(),
            Value::F64(array_from_vec(&[1, 2], vec![1.0; 2]).unwrap()),
        );
        s.insert(
            "I".to_string(),
            Value::I64(array_from_vec(&[1], vec![9]).unwrap()),
        );
        fails_both("IN B\nIN R\nIN I\nOUT = SCATTER(B, R, I)\n", &s);
        // Negative rotary position.
        fails_both(
            "IN X\nIN P\nOUT = ROTARY(X, P)\n",
            &add_i(feeds_f(&[("X", &[1.0, 2.0], &[1, 2])]), "P", &[-1], &[1]),
        );
        // Non-dyadic INTERP scale is a lowering error on both paths.
        fails_both(
            "IN X\nOUT = INTERP(X, 1.5, 1.0)\n",
            &feeds_f(&[("X", &[1.0, 2.0, 3.0, 4.0], &[2, 2])]),
        );
        // TOPK k outside the row.
        fails_both(
            "IN X\nOUT, IDX = TOPK(X, 3)\n",
            &feeds_f(&[("X", &[1.0, 2.0], &[1, 2])]),
        );
    }

    #[test]
    fn every_kernel_names_a_registered_mnemonic() {
        let reg = crate::ops::registry();
        for mn in [
            "IADD",
            "ISUB",
            "IMUL",
            "IDIV",
            "IMOD",
            "IDIVU",
            "IMODU",
            "IAND",
            "IOR",
            "IXOR",
            "IEQ",
            "ILT",
            "IMAX",
            "SADD",
            "ISAR",
            "ISHL",
            "ISHR",
            "IMAD",
            "ISELECT",
            "IREDUCE_SUM",
            "ISUM",
            "IVARINT",
            "IMATMUL",
            "IGET",
            "IWINDOW",
            "IFIND",
            "IGATHER1",
            "ISCATTER1",
            "IASSEMBLE",
            "ICRC32",
            "ICRC32C",
            "SLICE",
            "RELU",
            "F2I",
            "I2F",
            "CONCAT",
            "RESIZE2D",
            "CONV2D",
            "DECONV2D",
            "RESHAPE2",
            "RESHAPE3",
            "LAYERNORM",
            "RMSNORM",
            "ROTARY",
            "MATMUL",
            "BATCH_MATMUL",
            "CONV",
            "INTERP",
            "GATHER",
            "GATHER3",
            "SCATTER",
            "ARGMAX",
            "TOPK",
        ] {
            assert!(reg.get(mn).is_some(), "{mn} is not in the registry");
            assert!(kernel_for(mn).is_some(), "{mn} has no kernel");
        }
        // Registered, not yet ported: the arena must say so, not panic.
        assert!(reg.get("IINFLATE").is_some());
        assert!(kernel_for("IINFLATE").is_none());
    }

    #[test]
    fn refuses_unsupported_mnemonics() {
        let bound = assemble("IN X\nOUT = IINFLATE(X, 4)\n", &AsmOptions::default()).unwrap();
        let e = Plan::compile(&bound, &feeds(4), &["OUT".to_string()]).unwrap_err();
        assert!(e.to_string().contains("not ported"), "{e}");
    }
}
