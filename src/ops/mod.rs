//! Mnemonic registry and reference semantics.
//!
//! Every op follows the Python contract: fail loud on the wrong kind,
//! shape, or range; never silently coerce. Ops that are registered but not
//! ported yet return [`crate::Error::Unimplemented`] (the Rust `NoPattern`).

use std::collections::HashMap;
use std::sync::OnceLock;

use ndarray::ArrayD;

use crate::error::{Error, Result};
use crate::ir::Config;
use crate::layout::LayoutSig;
use crate::value::{Arg, Triple, Value};

pub mod bytes;
pub mod element;
pub mod image;
pub mod int;
pub mod linalg;

pub type OpFn = fn(&[Arg], &Config) -> Result<Vec<Value>>;

#[derive(Clone)]
pub struct OpDef {
    pub name: &'static str,
    pub arity_in: usize,
    pub arity_out: usize,
    pub sig: Option<LayoutSig>,
    pub run: Option<OpFn>,
}

pub struct Registry {
    map: HashMap<&'static str, OpDef>,
}

impl Registry {
    pub fn get(&self, name: &str) -> Option<&OpDef> {
        self.map.get(name)
    }

    pub fn len(&self) -> usize {
        self.map.len()
    }

    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }

    pub fn names(&self) -> Vec<&'static str> {
        let mut v: Vec<_> = self.map.keys().copied().collect();
        v.sort_unstable();
        v
    }

    pub fn iter(&self) -> impl Iterator<Item = &OpDef> {
        self.map.values()
    }
}

fn sig(inputs: &[&str], outputs: &[&str]) -> Option<LayoutSig> {
    Some(LayoutSig::new(inputs, outputs))
}

static REGISTRY: OnceLock<Registry> = OnceLock::new();

pub fn registry() -> &'static Registry {
    REGISTRY.get_or_init(build_registry)
}

fn build_registry() -> Registry {
    let mut m = HashMap::new();
    let mut add =
        |name: &'static str, ai: usize, ao: usize, sg: Option<LayoutSig>, run: Option<OpFn>| {
            m.insert(
                name,
                OpDef {
                    name,
                    arity_in: ai,
                    arity_out: ao,
                    sig: sg,
                    run,
                },
            );
        };

    use bytes::*;
    use element::*;
    use image::*;
    use int::*;
    use linalg::*;

    // --- image / holo path ---
    add(
        "SRGB_DECODE",
        1,
        1,
        sig(&["U8:HWC"], &["F:HWC"]),
        Some(op_srgb_decode),
    );
    add("LUMA", 1, 1, sig(&["F:HWC"], &["F:HW"]), Some(op_luma));
    add("SQRT", 1, 1, sig(&["F:HW"], &["T:HW"]), Some(op_sqrt));
    add(
        "SPLAT_BLUR",
        1,
        2,
        sig(&["T:HW"], &["T:HW", "T:HW"]),
        Some(op_splat_blur),
    );
    add("SUB", 2, 1, sig(&["$A", "$A"], &["$A"]), Some(op_sub));
    add(
        "BETA_V5",
        2,
        1,
        sig(&["T:HW", "T:HW"], &["T:HW"]),
        Some(op_beta_v5),
    );
    add("MUL", 2, 1, sig(&["$A", "$A"], &["$A"]), Some(op_mul));
    add("ADD", 2, 1, sig(&["$A", "$A"], &["$A"]), Some(op_add));
    add("SQUARE", 1, 1, sig(&["$A"], &["$A"]), Some(op_square));
    add(
        "GAIN",
        3,
        1,
        sig(&["F:HWC", "F:HW", "T:HW"], &["F:HWC"]),
        Some(op_gain),
    );
    add(
        "SRGB_ENCODE",
        1,
        1,
        sig(&["F:HWC"], &["U8:HWC"]),
        Some(op_srgb_encode),
    );
    add("BETA", 1, 1, sig(&["$A"], &["$A"]), Some(op_beta));
    add("TBETA", 1, 1, sig(&["$A"], &["$A"]), Some(op_tbeta));
    add(
        "ISO_BLUR",
        1,
        1,
        sig(&["T:HW"], &["T:HW"]),
        Some(op_iso_blur),
    );
    add(
        "WARP",
        2,
        1,
        sig(&["T:HW", "F:HW2"], &["T:HW"]),
        Some(op_warp),
    );
    add("STATIC", 1, 1, sig(&["F:HW2"], &["I:HW"]), Some(op_static));
    add(
        "MIXDYAD",
        3,
        1,
        sig(&["T:HW", "T:HW", "I:HW"], &["T:HW"]),
        Some(op_mixdyad),
    );

    // --- integer path ---
    add("IADD", 2, 1, sig(&["I:*", "*"], &["I:*"]), Some(op_iadd));
    add("SADD", 2, 1, sig(&["I:*", "*"], &["I:*"]), Some(op_sadd));
    add("IMAX", 2, 1, sig(&["I:*", "*"], &["I:*"]), Some(op_imax));
    add("ISUB", 2, 1, sig(&["I:*", "*"], &["I:*"]), Some(op_isub));
    add("IMUL", 2, 1, sig(&["I:*", "*"], &["I:*"]), Some(op_imul));
    add(
        "IMAD",
        3,
        1,
        sig(&["I:*", "*", "*"], &["I:*"]),
        Some(op_imad),
    );
    add(
        "IMATMUL",
        2,
        1,
        sig(&["I:*", "I:*"], &["I:*"]),
        Some(op_imatmul),
    );
    add(
        "IREDUCE_SUM",
        1,
        1,
        sig(&["I:*"], &["I:SCALAR"]),
        Some(op_ireduce_sum),
    );
    add("IDIV", 2, 1, sig(&["I:*", "*"], &["I:*"]), Some(op_idiv));
    add("IMOD", 2, 1, sig(&["I:*", "*"], &["I:*"]), Some(op_imod));
    add("IDIVU", 2, 1, sig(&["I:*", "*"], &["I:*"]), Some(op_idivu));
    add("IMODU", 2, 1, sig(&["I:*", "*"], &["I:*"]), Some(op_imodu));
    add("IXOR", 2, 1, sig(&["I:*", "*"], &["I:*"]), Some(op_ixor));
    add("IAND", 2, 1, sig(&["I:*", "*"], &["I:*"]), Some(op_iand));
    add("IOR", 2, 1, sig(&["I:*", "*"], &["I:*"]), Some(op_ior));
    add("ISAR", 2, 1, sig(&["I:*", "*"], &["I:*"]), Some(op_isar));
    add("IEQ", 2, 1, sig(&["I:*", "*"], &["I:*"]), Some(op_ieq));
    add("ILT", 2, 1, sig(&["I:*", "*"], &["I:*"]), Some(op_ilt));
    add(
        "ISELECT",
        3,
        1,
        sig(&["I:*", "*", "*"], &["I:*"]),
        Some(op_iselect),
    );
    add(
        "ISHL",
        2,
        1,
        sig(&["I:*", "F:SCALAR"], &["I:*"]),
        Some(op_ishl),
    );
    add(
        "ISHR",
        2,
        1,
        sig(&["I:*", "F:SCALAR"], &["I:*"]),
        Some(op_ishr),
    );
    add("F2I", 1, 1, sig(&["F:*"], &["I:*"]), Some(op_f2i));
    add("I2F", 1, 1, sig(&["I:*"], &["F:*"]), Some(op_i2f));
    add("T2I", 1, 1, sig(&["T:*"], &["I:*"]), Some(op_t2i));
    add(
        "IGET",
        2,
        1,
        sig(&["I:*", "F:SCALAR"], &["I:SCALAR"]),
        Some(op_iget),
    );

    // --- triples arithmetic / transformer ---
    add("MATMUL", 2, 1, sig(&["*", "*"], &["*"]), Some(op_matmul));
    add(
        "BATCH_MATMUL",
        2,
        1,
        sig(&["*", "*"], &["*"]),
        Some(op_batch_matmul),
    );
    add("SOFTMAX", 1, 1, sig(&["$A"], &["$A"]), Some(op_softmax));
    add(
        "SOFTMAX_WIDE",
        1,
        1,
        sig(&["$A"], &["$A"]),
        Some(op_softmax_wide),
    );
    add("TSHIFT", 1, 1, sig(&["$A"], &["$A"]), Some(op_tshift));
    add(
        "RMSNORM",
        2,
        1,
        sig(&["$X", "$W"], &["$X"]),
        Some(op_rmsnorm),
    );
    add(
        "LAYERNORM",
        3,
        1,
        sig(&["$X", "$W", "$W"], &["$X"]),
        Some(op_layernorm),
    );
    add("SILU", 1, 1, sig(&["$A"], &["$A"]), Some(op_silu));
    add("GELU", 1, 1, sig(&["$A"], &["$A"]), Some(op_gelu));
    add("RELU", 1, 1, sig(&["$A"], &["$A"]), Some(op_relu));
    add(
        "SCAN",
        3,
        1,
        sig(&["$X", "$X", "$X"], &["$X"]),
        Some(op_scan),
    );
    add("ROTARY", 2, 1, sig(&["$X", "*"], &["$X"]), Some(op_rotary));
    add(
        "ROPE",
        4,
        1,
        sig(&["$A", "*", "F:SCALAR", "F:SCALAR"], &["$A"]),
        Some(op_rope),
    );
    add(
        "SWIGLU",
        2,
        1,
        sig(&["$A", "F:SCALAR"], &["$A"]),
        Some(op_swiglu),
    );
    add(
        "ATTN",
        8,
        1,
        sig(
            &[
                "$A", "$A", "$A", "I:*", "F:SCALAR", "F:SCALAR", "F:SCALAR", "F:SCALAR",
            ],
            &["$A"],
        ),
        Some(op_attn),
    );
    add("BMMV", 13, 1, None, Some(op_bmmv));

    // --- exact moves / gather / masks ---
    add(
        "TRANSPOSE",
        1,
        1,
        sig(&["$A"], &["$A^T"]),
        Some(op_transpose),
    );
    add(
        "GAUSS",
        3,
        1,
        sig(&["$A", "F:SCALAR", "F:SCALAR"], &["$A"]),
        Some(op_gauss),
    );
    add(
        "RESHAPE2",
        3,
        1,
        sig(&["*", "F:SCALAR", "F:SCALAR"], &["*"]),
        Some(op_reshape2),
    );
    add(
        "RESHAPE3",
        4,
        1,
        sig(&["*", "F:SCALAR", "F:SCALAR", "F:SCALAR"], &["*"]),
        Some(op_reshape3),
    );
    add(
        "PERMUTE3",
        4,
        1,
        sig(&["*", "F:SCALAR", "F:SCALAR", "F:SCALAR"], &["*"]),
        Some(op_permute3),
    );
    add(
        "SELECT",
        3,
        1,
        sig(&["I:*", "$A", "$A"], &["$A"]),
        Some(op_select),
    );
    add(
        "CONCAT",
        3,
        1,
        sig(&["$A", "$A", "F:SCALAR"], &["$A"]),
        Some(op_concat),
    );
    add(
        "ARGMAX",
        2,
        1,
        sig(&["$A", "F:SCALAR"], &["I:*"]),
        Some(op_argmax),
    );
    add(
        "TOPK",
        2,
        2,
        sig(&["$A", "F:SCALAR"], &["$A", "I:*"]),
        Some(op_topk),
    );
    add("GT", 2, 1, sig(&["$A", "$A"], &["I:*"]), Some(op_gt));
    add(
        "SLICE",
        4,
        1,
        sig(&["$A", "F:SCALAR", "F:SCALAR", "F:SCALAR"], &["$A"]),
        Some(op_slice),
    );
    add(
        "CLIP",
        3,
        1,
        sig(&["$A", "F:SCALAR", "F:SCALAR"], &["$A"]),
        Some(op_clip),
    );
    add("DIV", 2, 1, sig(&["$A", "$A"], &["$A"]), Some(op_div));
    add("SIGMOID", 1, 1, sig(&["$A"], &["$A"]), Some(op_sigmoid));
    add(
        "RESCALE",
        2,
        1,
        sig(&["$A", "F:SCALAR"], &["$A"]),
        Some(op_rescale),
    );
    add("GATHER", 2, 1, sig(&["*", "I:*"], &["*"]), Some(op_gather));
    add(
        "GATHER3",
        3,
        1,
        sig(&["*", "I:*", "I:*"], &["*"]),
        Some(op_gather3),
    );
    add(
        "SCATTER",
        3,
        1,
        sig(&["$A", "$A", "I:*"], &["$A"]),
        Some(op_scatter),
    );
    add("PRELU", 2, 1, sig(&["$A", "$A"], &["$A"]), Some(op_prelu));
    add("POOLAVG", 1, 1, sig(&["*"], &["*"]), Some(op_poolavg));
    add(
        "DECONV",
        4,
        1,
        sig(&["*", "*", "F:SCALAR", "F:SCALAR"], &["*"]),
        Some(op_deconv),
    );
    add(
        "INTERP",
        3,
        1,
        sig(&["*", "F:SCALAR", "F:SCALAR"], &["*"]),
        Some(op_interp),
    );
    add("CONV", 2, 1, sig(&["*", "*"], &["*"]), Some(op_conv));

    // --- byte-index family ---
    add(
        "IWINDOW",
        3,
        1,
        sig(&["I:*", "*", "F:SCALAR"], &["I:*"]),
        Some(op_iwindow),
    );
    add("ICRC32", 1, 1, sig(&["I:*"], &["I:*"]), Some(op_icrc32));
    add(
        "ICRC32C",
        2,
        1,
        sig(&["I:*", "I:*"], &["I:*"]),
        Some(op_icrc32c),
    );
    add(
        "IINFLATE",
        2,
        1,
        sig(&["I:*", "F:SCALAR"], &["I:*"]),
        Some(op_iinflate),
    );
    add(
        "IVARINT",
        2,
        2,
        sig(&["I:*", "*"], &["I:*", "I:*"]),
        Some(op_ivarint),
    );
    add(
        "IFIND",
        2,
        1,
        sig(&["I:*", "I:*"], &["I:*"]),
        Some(op_ifind),
    );
    add(
        "IGATHER1",
        2,
        1,
        sig(&["I:*", "I:*"], &["I:*"]),
        Some(op_igather1),
    );
    add(
        "IASSEMBLE",
        3,
        1,
        sig(&["I:*", "F:SCALAR", "F:SCALAR"], &["I:*"]),
        Some(op_iassemble),
    );
    add(
        "ISUM",
        3,
        1,
        sig(&["I:*", "F:SCALAR", "F:SCALAR"], &["I:*"]),
        Some(op_isum),
    );
    add(
        "ISCATTER1",
        3,
        1,
        sig(&["I:*", "I:*", "I:*"], &["I:*"]),
        Some(op_iscatter1),
    );

    Registry { map: m }
}

/// Dispatch one instruction. Arity/layout are the assembler's job; this is
/// the runtime call boundary.
pub fn dispatch(name: &str, args: &[Arg], config: &Config) -> Result<Vec<Value>> {
    let def = registry()
        .get(name)
        .ok_or_else(|| Error::asm(format!("unknown mnemonic: {name}")))?;
    match def.run {
        Some(f) => {
            let out = f(args, config)?;
            if out.len() != def.arity_out {
                return Err(Error::runtime(format!(
                    "{name}: runtime arity break: expected {} outs, got {}",
                    def.arity_out,
                    out.len()
                )));
            }
            Ok(out)
        }
        None => Err(Error::unimplemented(format!(
            "{name}: no Rust pattern yet (loud NoPattern)"
        ))),
    }
}

// ---------------------------------------------------------------------------
// Shared argument/array helpers
// ---------------------------------------------------------------------------

pub fn arg_triple<'a>(args: &'a [Arg], i: usize, op: &str) -> Result<&'a Triple> {
    args.get(i)
        .ok_or_else(|| Error::runtime(format!("{op}: missing argument {i}")))?
        .stream(op, "arg")?
        .triple(op)
}

pub fn arg_f64_array(args: &[Arg], i: usize, op: &str) -> Result<ArrayD<f64>> {
    args.get(i)
        .ok_or_else(|| Error::runtime(format!("{op}: missing argument {i}")))?
        .stream(op, "arg")?
        .as_f64(op)
}

pub fn arg_i64_array(args: &[Arg], i: usize, op: &str) -> Result<ArrayD<i64>> {
    args.get(i)
        .ok_or_else(|| Error::runtime(format!("{op}: missing argument {i}")))?
        .stream(op, "arg")?
        .as_i64(op)
}

pub fn arg_i64(args: &[Arg], i: usize, op: &str, what: &str) -> Result<i64> {
    args.get(i)
        .ok_or_else(|| Error::runtime(format!("{op}: missing argument {what}")))?
        .i64(op, what)
}

pub fn arg_f64(args: &[Arg], i: usize, op: &str, what: &str) -> Result<f64> {
    args.get(i)
        .ok_or_else(|| Error::runtime(format!("{op}: missing argument {what}")))?
        .f64(op, what)
}

pub fn shape_of(args: &[Arg], i: usize) -> Option<Vec<usize>> {
    args.get(i).and_then(|a| a.value()).and_then(|v| v.shape())
}

pub fn same_shape(a: &[usize], b: &[usize]) -> bool {
    a == b
}

/// Two's-complement wrap into i64 (numpy int64 arithmetic).
pub fn wrap_i64(v: i128) -> i64 {
    v as i64
}

/// Rescale a fixed array through triples (the only scale changer).
pub fn rescale_checked(q: &ArrayD<i64>, m1: i64, m2: i64) -> Result<ArrayD<i64>> {
    crate::lattice::rescale(q, m1, m2)
}

/// numpy's truncating integer division. Raises on zero (matching IDIV).
pub fn checked_div(a: i64, b: i64) -> Result<i64> {
    if b == 0 {
        return Err(Error::runtime("IDIV/IMOD: division by zero"));
    }
    if a == i64::MIN && b == -1 {
        return Err(Error::runtime(
            "IDIV/IMOD: INT64_MIN / -1 is unrepresentable",
        ));
    }
    Ok(a / b)
}

pub fn triple_binop<F>(a: &Triple, b: &Triple, op_name: &str, mut f: F) -> Result<Triple>
where
    F: FnMut(i8, i32, u8, i8, i32, u8) -> (i8, i32, u8),
{
    if a.shape() != b.shape() {
        return Err(Error::runtime(format!(
            "{op_name}: shape mismatch {:?} vs {:?}",
            a.shape(),
            b.shape()
        )));
    }
    let mut s = Vec::with_capacity(a.numel());
    let mut e = Vec::with_capacity(a.numel());
    let mut z = Vec::with_capacity(a.numel());
    for (((&as_, &ae), &az), ((&bs, &be), &bz)) in
        a.s.iter()
            .zip(a.e.iter())
            .zip(a.z.iter())
            .zip(b.s.iter().zip(b.e.iter()).zip(b.z.iter()))
    {
        let (os, oe, oz) = f(as_, ae, az, bs, be, bz);
        s.push(os);
        e.push(oe);
        z.push(oz);
    }
    Triple::from_vecs(&a.shape(), s, e, z)
}

pub fn triple_ternary<F>(
    a: &Triple,
    b: &Triple,
    c: &Triple,
    op_name: &str,
    mut f: F,
) -> Result<Triple>
where
    F: FnMut((i8, i32, u8), (i8, i32, u8), (i8, i32, u8)) -> (i8, i32, u8),
{
    if a.shape() != b.shape() || a.shape() != c.shape() {
        return Err(Error::runtime(format!(
            "{op_name}: shape mismatch {:?} vs {:?} vs {:?}",
            a.shape(),
            b.shape(),
            c.shape()
        )));
    }
    let mut s = Vec::with_capacity(a.numel());
    let mut e = Vec::with_capacity(a.numel());
    let mut z = Vec::with_capacity(a.numel());
    for ((((&as_, &ae), &az), ((&bs, &be), &bz)), ((&cs, &ce), &cz)) in
        a.s.iter()
            .zip(a.e.iter())
            .zip(a.z.iter())
            .zip(b.s.iter().zip(b.e.iter()).zip(b.z.iter()))
            .zip(c.s.iter().zip(c.e.iter()).zip(c.z.iter()))
    {
        let (os, oe, oz) = f((as_, ae, az), (bs, be, bz), (cs, ce, cz));
        s.push(os);
        e.push(oe);
        z.push(oz);
    }
    Triple::from_vecs(&a.shape(), s, e, z)
}

/// Elementwise map on a triple.
pub fn triple_map<F>(a: &Triple, mut f: F) -> Result<Triple>
where
    F: FnMut(i8, i32, u8) -> (i8, i32, u8),
{
    let mut s = Vec::with_capacity(a.numel());
    let mut e = Vec::with_capacity(a.numel());
    let mut z = Vec::with_capacity(a.numel());
    for ((&sv, &ev), &zv) in a.s.iter().zip(a.e.iter()).zip(a.z.iter()) {
        let (os, oe, oz) = f(sv, ev, zv);
        s.push(os);
        e.push(oe);
        z.push(oz);
    }
    Triple::from_vecs(&a.shape(), s, e, z)
}

/// Select between two triples by a boolean mask (numpy `np.where`).
pub fn triple_where(mask: &ArrayD<bool>, a: &Triple, b: &Triple, op_name: &str) -> Result<Triple> {
    if mask.shape() != a.s.shape() || mask.shape() != b.s.shape() {
        return Err(Error::runtime(format!(
            "{op_name}: mask shape {:?} vs {:?} vs {:?}",
            mask.shape(),
            a.s.shape(),
            b.s.shape()
        )));
    }
    let mut s = Vec::with_capacity(a.numel());
    let mut e = Vec::with_capacity(a.numel());
    let mut z = Vec::with_capacity(a.numel());
    for ((&m, ((&as_, &ae), &az)), ((&bs, &be), &bz)) in mask
        .iter()
        .zip(a.s.iter().zip(a.e.iter()).zip(a.z.iter()))
        .zip(b.s.iter().zip(b.e.iter()).zip(b.z.iter()))
    {
        let (sv, ev, zv) = if m { (as_, ae, az) } else { (bs, be, bz) };
        s.push(sv);
        e.push(ev);
        z.push(zv);
    }
    Triple::from_vecs(&a.shape(), s, e, z)
}

/// Load the frozen scales: `(m_acc, m_cov)`.
pub fn scales(config: &Config) -> (i64, i64) {
    let mut m_acc = 31344i64;
    let mut m_cov = 33280i64;
    if let Some(v) = config.get_i64("m_acc") {
        m_acc = v;
    }
    if let Some(v) = config.get_i64("m_cov") {
        m_cov = v;
    }
    (m_acc, m_cov)
}

pub fn beta(config: &Config) -> f64 {
    config.get_f64_or("beta", 0.5)
}

/// Frozen controller values (`chain/CTRL.json`, embedded doctrine).
pub mod ctrl {
    pub const ISO_ATTEN: f64 = 0.5;
    pub const MID_ATTEN: f64 = 0.4;
    pub const STRONG_ATTEN: f64 = 1.0;
    pub const COH_HI: f64 = 0.4;
    pub const COH_THR: f64 = 0.15;
    pub const V5_W0: f64 = 0.0;
    pub const V5_W1: f64 = 4.0;
    pub const V5_W2: f64 = 8.0;
}
