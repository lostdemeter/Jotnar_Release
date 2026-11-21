//! Vulkan compute backend (correctness-first).
//!
//! Primary numeric path is the **φ-ladder (T)**: the same integer arithmetic
//! the non-FPU target emits, so results can be gated bit-identically instead
//! of by float tolerance. Streams are multi-plane:
//!
//!   T   -> 3 planes (s, e, z); T2 adds m (not yet)
//!   I   -> 1 plane  (i64)
//!   F   -> 1 plane  (f32; Adreno has no shaderFloat64)
//!
//! Ladder values are `s * φ^((e - 32768)/512)`. MUL is the φ-spoof identity:
//! sign multiply, rung add with bias, zero OR — no float hardware involved.
//! ADD/SUB need the fixed-point bridge and its baked tables; they are the
//! next milestone and refuse loudly until then.
//!
//! `glslc` is required at emit time: the emitted C embeds the resulting
//! SPIR-V, so the runtime needs only libvulkan.

use std::collections::HashMap;

use crate::error::{Error, Result};
use crate::ir::BoundInstr;
use crate::payload::Feeds;

use super::infer::{infer_shapes, CKind, StreamInfo};
use super::{sanitize_cnames, Artifact, CompileOptions};

const LOCAL_SIZE: u32 = 64;

/// Ladder constants, mirrored from phi_core::lattice.
const BIAS: i64 = 32768;
const NLVL: i64 = 65536;
const FRAC_CAP: i64 = 13312;
const FIXED_F: i64 = 18;

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
enum Comp {
    V,
    S,
    E,
    Z,
}

/// How a plane is laid out on the host side. GPU-side every plane is 32-bit
/// except I64, which keeps shaders free of int8/int16 features.
#[derive(Clone, Copy, PartialEq)]
enum Elem {
    I8,
    U8,
    I32,
    I64,
    F32,
}

fn gpu_width(e: Elem) -> usize {
    match e {
        Elem::I64 => 8,
        _ => 4,
    }
}

fn file_width(e: Elem) -> usize {
    match e {
        Elem::I8 | Elem::U8 => 1,
        Elem::I32 | Elem::F32 => 4,
        Elem::I64 => 8,
    }
}

fn comps_for(kind: CKind) -> &'static [Comp] {
    match kind {
        CKind::T => &[Comp::S, Comp::E, Comp::Z],
        _ => &[Comp::V],
    }
}

fn elem_for(kind: CKind, c: Comp) -> Elem {
    match (kind, c) {
        (CKind::T, Comp::S) => Elem::I8,
        (CKind::T, Comp::E) => Elem::I32,
        (CKind::T, Comp::Z) => Elem::U8,
        (CKind::I, _) => Elem::I64,
        (CKind::F, _) => Elem::F32,
        _ => Elem::F32,
    }
}

const LADDER_GLSL: &str = r#"
/// Shared ladder helpers. Shaders must declare bindings 9/10/11 as FRAC,
/// COARSE and FINE (int64_t) before including this text.
const int FRAC_CAP = 13312;
const int FIXED_F = 18;
const int BIAS = 32768;
const int NLVL = 65536;

struct Ladder { int s; int e; uint z; };

int64_t to_fixed(int sgn, int ev, uint z, int64_t m0)
{
  if (z != 0u) return 0;
  int64_t d = m0 - int64_t(ev);
  if (d < 0) return int64_t(sgn) * frac[0];
  if (d > int64_t(FRAC_CAP)) return 0;
  return int64_t(sgn) * frac[int(d)];
}

int bitlen64(uint64_t a)
{
  int bl = 0;
  while (a > 0ul) { bl += 1; a >>= 1; }
  return bl;
}

Ladder from_fixed(int64_t q, int64_t m0)
{
  Ladder r;
  int sgn = q > 0 ? 1 : -1;
  uint64_t uq = q < 0 ? (~uint64_t(q)) + 1ul : uint64_t(q);
  if (uq >= (1ul << 53)) uq = (1ul << 53) - 1ul;
  int bl = bitlen64(uq);
  int shift = bl - 15;
  uint64_t mant = shift >= 0 ? (uq >> uint(shift)) : (uq << uint(-shift));
  int t = shift + 14 - FIXED_F;
  int idx = t + 64;
  idx = idx < 0 ? 0 : (idx > 192 ? 192 : idx);
  int fi = int(mant) - 16384;
  fi = fi < 0 ? 0 : (fi > 16383 ? 16383 : fi);
  int64_t ev = m0 + coarse[idx] + fine[fi];
  ev = ev < 0 ? 0 : (ev > int64_t(NLVL - 1) ? int64_t(NLVL - 1) : ev);
  r.s = sgn; r.e = int(ev); r.z = (q == 0) ? 1u : 0u;
  return r;
}
"#;

enum Operand {
    Stream(usize),
    Lit(f64),
}

struct Dispatch {
    shader: usize,
    buffers: Vec<usize>,
    params: [u32; 4],
    groups: [u32; 3],
    local: [u32; 3],
}

type Lowered = (String, Vec<usize>, [u32; 4], [u32; 3], [u32; 3]);

/// Ladder tables are appended to the buffer table after the stream planes,
/// so a dispatch can bind them like any other storage buffer.
struct Tables {
    n_plane: usize,
    names: Vec<&'static str>,
    data: Vec<Vec<i64>>,
}

impl Tables {
    fn new(n_plane: usize) -> Self {
        Tables { n_plane, names: Vec::new(), data: Vec::new() }
    }

    fn get(&mut self, name: &'static str, make: impl FnOnce() -> Vec<i64>) -> usize {
        if let Some(i) = self.names.iter().position(|n| *n == name) {
            return self.n_plane + i;
        }
        let d = make();
        self.names.push(name);
        self.data.push(d);
        self.n_plane + self.data.len() - 1
    }
}

struct Ctx<'a> {
    streams: &'a HashMap<String, StreamInfo>,
    index: &'a HashMap<String, usize>,
    plane: &'a HashMap<(usize, Comp), usize>,
    /// Declared fixed-point scales: the scale is part of the contract.
    m_acc: i64,
    m_cov: i64,
}

impl<'a> Ctx<'a> {
    fn ordinal(&self, name: &str) -> Result<usize> {
        self.index
            .get(name)
            .copied()
            .ok_or_else(|| Error::asm(format!("Vulkan: '{name}' is not a stream")))
    }

    fn plane_of(&self, ord: usize, c: Comp) -> Result<usize> {
        self.plane
            .get(&(ord, c))
            .copied()
            .ok_or_else(|| Error::asm("Vulkan: missing plane".to_string()))
    }

    fn info(&self, ord: usize) -> Result<&StreamInfo> {
        let name = self
            .index
            .iter()
            .find(|(_, v)| **v == ord)
            .map(|(k, _)| k.as_str())
            .ok_or_else(|| Error::asm("Vulkan: stream not in table".to_string()))?;
        self.streams
            .get(name)
            .ok_or_else(|| Error::asm(format!("Vulkan: unknown stream '{name}'")))
    }

    fn elems(&self, ord: usize) -> Result<usize> {
        self.info(ord)?.elems().ok_or_else(|| {
            Error::unimplemented("Vulkan: dynamic shape (static geometry required)".to_string())
        })
    }

    fn name(&self, ord: usize) -> String {
        self.index
            .iter()
            .find(|(_, v)| **v == ord)
            .map(|(k, _)| k.clone())
            .unwrap_or_default()
    }
}

pub fn compile_vulkan(text: &str, opts: &CompileOptions, sample: Feeds) -> Result<Artifact> {
    if opts.outputs.is_empty() {
        return Err(Error::asm(
            "compile_program needs outputs=[...] (OUT streams)",
        ));
    }
    let mut bound = crate::asm::assemble(text, &opts.asm)?;
    let source_inputs: Vec<String> = bound.inputs.iter().map(|d| d.name.clone()).collect();
    let source_outputs = opts.outputs.clone();
    let mut sample = sample;
    let mut outputs = opts.outputs.clone();
    sanitize_cnames(&mut bound, &mut outputs, &mut sample)?;

    let streams = infer_shapes(&bound, &sample)?;
    for o in &outputs {
        if !streams.contains_key(o) {
            return Err(Error::asm(format!("Vulkan: output '{o}' is not a stream")));
        }
    }
    let in_names: Vec<String> = bound.inputs.iter().map(|d| d.name.clone()).collect();

    let serve_in = bound.config.get("serve_input").map(str::to_string);
    let serve_out = bound.config.get("serve_output").map(str::to_string);
    if serve_in.is_some() != serve_out.is_some() {
        return Err(Error::asm(
            "Vulkan serve mode needs both CONFIG serve_input and CONFIG serve_output",
        ));
    }
    if let Some(n) = &serve_in {
        if !in_names.contains(n) {
            return Err(Error::asm("Vulkan serve_input must be an IN stream"));
        }
    }
    if let Some(n) = &serve_out {
        if !outputs.contains(n) {
            return Err(Error::asm("Vulkan serve_output must be an output"));
        }
    }

    // Streams (ordinals), then planes (GPU buffers).
    let mut order: Vec<String> = Vec::new();
    let mut index: HashMap<String, usize> = HashMap::new();
    {
        let mut add = |name: &str, order: &mut Vec<String>, index: &mut HashMap<String, usize>| {
            if !index.contains_key(name) {
                index.insert(name.to_string(), order.len());
                order.push(name.to_string());
            }
        };
        for n in &in_names {
            add(n, &mut order, &mut index);
        }
        for ins in &bound.instrs {
            if ins.mn.starts_with("__") {
                return Err(Error::unimplemented(format!(
                    "Vulkan: control pseudo-op {} not supported yet",
                    ins.mn
                )));
            }
            for o in &ins.outs {
                add(o, &mut order, &mut index);
            }
        }
    }

    let mut planes: Vec<(usize, Comp)> = Vec::new();
    let mut plane: HashMap<(usize, Comp), usize> = HashMap::new();
    for (ord, name) in order.iter().enumerate() {
        let kind = streams[name].kind;
        for &c in comps_for(kind) {
            plane.insert((ord, c), planes.len());
            planes.push((ord, c));
        }
    }

    let ctx = Ctx {
        streams: &streams,
        index: &index,
        plane: &plane,
        m_acc: bound.config.get_i64("m_acc").unwrap_or(31344),
        m_cov: bound.config.get_i64("m_cov").unwrap_or(33280),
    };

    let mut dispatches: Vec<Dispatch> = Vec::new();
    let mut shader_srcs: Vec<String> = Vec::new();
    let mut shader_ids: HashMap<String, usize> = HashMap::new();
    let mut tabs = Tables::new(planes.len());

    for ins in &bound.instrs {
        let (glsl, buffers, params, groups, local) = lower(ins, &ctx, &mut tabs)?;
        if buffers.len() > 16 {
            return Err(Error::unimplemented(format!(
                "Vulkan: {} needs {} bindings (max 16)",
                ins.mn,
                buffers.len()
            )));
        }
        let sid = match shader_ids.get(&glsl) {
            Some(&id) => id,
            None => {
                let id = shader_srcs.len();
                shader_ids.insert(glsl.clone(), id);
                shader_srcs.push(glsl);
                id
            }
        };
        dispatches.push(Dispatch {
            shader: sid,
            buffers,
            params,
            groups,
            local,
        });
    }

    let spirv: Vec<Vec<u8>> = shader_srcs
        .iter()
        .enumerate()
        .map(|(i, src)| glslc(src, i))
        .collect::<Result<_>>()?;

    let source = emit_c(
        &order, &planes, &ctx, &tabs, &dispatches, &spirv,
        &in_names, &outputs,
        serve_in.as_deref(), serve_out.as_deref(),
    )?;

    // argv = one file per plane, inputs then outputs
    let n_argv = in_names
        .iter()
        .map(|n| comps_for(streams[n].kind).len())
        .sum::<usize>()
        + outputs
            .iter()
            .map(|n| comps_for(streams[n].kind).len())
            .sum::<usize>();

    Ok(Artifact {
        source,
        streams,
        inputs: in_names,
        outputs,
        source_inputs,
        source_outputs,
        config: bound.config.clone(),
        n_argv,
        warm_frame: false,
    })
}

fn lower(ins: &BoundInstr, ctx: &Ctx, tabs: &mut Tables) -> Result<Lowered> {
    let out = ins
        .outs
        .first()
        .ok_or_else(|| Error::asm(format!("Vulkan: {} has no output", ins.mn)))?;
    let out_ord = ctx.ordinal(out)?;
    let out_n = ctx.elems(out_ord)?;
    let out_kind = ctx.info(out_ord)?.kind;

    match (ins.mn.as_str(), out_kind) {
        // φ-ladder: MUL is the spoof identity, pure integer.
        ("MUL", CKind::T) => tmul(ins, ctx, out_ord, out_n, false),
        ("SQUARE", CKind::T) => tmul(ins, ctx, out_ord, out_n, true),
        ("ADD", CKind::T) => tadd(ins, ctx, tabs, out_ord, out_n, false),
        ("SUB", CKind::T) => tadd(ins, ctx, tabs, out_ord, out_n, true),
        ("MATMUL", CKind::T) => tmatmul(ins, ctx, tabs),

        // exact integer arithmetic
        ("IADD", CKind::I) => int_bin(ins, ctx, out_ord, out_n, "+"),
        ("ISUB", CKind::I) => int_bin(ins, ctx, out_ord, out_n, "-"),
        ("IMUL", CKind::I) => int_bin(ins, ctx, out_ord, out_n, "*"),
        ("IMATMUL", CKind::I) => matmul(ins, ctx, CKind::I),

        // float path (F streams only; not the ladder)
        ("MATMUL", CKind::F) => matmul(ins, ctx, CKind::F),

        // Ladder ops that need the fixed-point bridge / baked tables.
        ("DIV", CKind::T)
        | ("RMSNORM", CKind::T) | ("SOFTMAX", CKind::T)
        | ("SOFTMAX_WIDE", CKind::T) | ("SILU", CKind::T) | ("GELU", CKind::T)
        | ("ROTARY", CKind::T) | ("TSHIFT", CKind::T) | ("TBETA", CKind::T)
        | ("TRANSPOSE", CKind::T) | ("SLICE", CKind::T) | ("CONCAT", CKind::T)
        | ("T2I", CKind::I) | ("IGET", CKind::I) | ("IREDUCE_SUM", CKind::I) => {
            Err(Error::unimplemented(format!(
                "Vulkan backend: {} on {out_kind:?} needs the ladder bridge/tables (next milestone)",
                ins.mn
            )))
        }

        (other, kind) => Err(Error::unimplemented(format!(
            "Vulkan backend: no lowering for {other} on {kind:?} yet"
        ))),
    }
}

/// T = A*B: s multiplies, rung adds with bias, zero propagates.
fn tmul(
    ins: &BoundInstr,
    ctx: &Ctx,
    out_ord: usize,
    out_n: usize,
    square: bool,
) -> Result<Lowered> {
    let (a_ord, b_ord) = if square {
        if ins.args.len() != 1 {
            return Err(Error::asm("Vulkan: SQUARE expects 1 operand"));
        }
        let a = ctx.ordinal(&ins.args[0])?;
        (a, a)
    } else {
        if ins.args.len() != 2 {
            return Err(Error::asm("Vulkan: MUL expects 2 operands"));
        }
        (ctx.ordinal(&ins.args[0])?, ctx.ordinal(&ins.args[1])?)
    };
    for ord in [a_ord, b_ord] {
        let k = ctx.info(ord)?.kind;
        if k != CKind::T {
            return Err(Error::asm(format!(
                "Vulkan: MUL mixes ladder with {k:?} at {}",
                ctx.name(ord)
            )));
        }
        if ctx.elems(ord)? != out_n {
            return Err(Error::unimplemented(format!(
                "Vulkan: MUL shape mismatch at {} (broadcasting not implemented)",
                ctx.name(ord)
            )));
        }
    }

    let (a_s, a_e, a_z) = (
        ctx.plane_of(a_ord, Comp::S)?,
        ctx.plane_of(a_ord, Comp::E)?,
        ctx.plane_of(a_ord, Comp::Z)?,
    );
    let (b_s, b_e, b_z) = (
        ctx.plane_of(b_ord, Comp::S)?,
        ctx.plane_of(b_ord, Comp::E)?,
        ctx.plane_of(b_ord, Comp::Z)?,
    );
    let (o_s, o_e, o_z) = (
        ctx.plane_of(out_ord, Comp::S)?,
        ctx.plane_of(out_ord, Comp::E)?,
        ctx.plane_of(out_ord, Comp::Z)?,
    );

    // Bindings are positional; identical operands (SQUARE) still bind twice,
    // which is legal and keeps the shader uniform.
    let bufs = vec![a_s, a_e, a_z, b_s, b_e, b_z, o_s, o_e, o_z];
    let src = format!(
        r#"#version 450
layout(local_size_x = {LOCAL_SIZE}) in;
layout(std430, binding = 0) readonly buffer AS {{ int s[]; }};
layout(std430, binding = 1) readonly buffer AE {{ int e[]; }};
layout(std430, binding = 2) readonly buffer AZ {{ uint z[]; }};
layout(std430, binding = 3) readonly buffer BS {{ int s2[]; }};
layout(std430, binding = 4) readonly buffer BE {{ int e2[]; }};
layout(std430, binding = 5) readonly buffer BZ {{ uint z2[]; }};
layout(std430, binding = 6) buffer OS {{ int os[]; }};
layout(std430, binding = 7) buffer OE {{ int oe[]; }};
layout(std430, binding = 8) buffer OZ {{ uint oz[]; }};
layout(push_constant) uniform PC {{ uint n; uint p0; uint p1; uint p2; }} pc;
void main() {{
  uint i = gl_GlobalInvocationID.x;
  if (i >= pc.n) return;
  os[i] = s[i] * s2[i];
  int e = e[i] + e2[i] - {BIAS};
  oe[i] = clamp(e, 0, {NLVL} - 1);
  oz[i] = z[i] | z2[i];
}}
"#
    );

    let groups = [div_ceil(out_n, LOCAL_SIZE as usize) as u32, 1, 1];
    Ok((src, bufs, [out_n as u32, 0, 0, 0], groups, [LOCAL_SIZE, 1, 1]))
}

/// T = A ± B through the fixed-point bridge: align to the larger rung,
/// convert both to integers at that rung, add, convert back.
fn tadd(
    ins: &BoundInstr,
    ctx: &Ctx,
    tabs: &mut Tables,
    out_ord: usize,
    out_n: usize,
    sub: bool,
) -> Result<Lowered> {
    if ins.args.len() != 2 {
        return Err(Error::asm(format!("Vulkan: {} expects 2 operands", ins.mn)));
    }
    let a = match operand(&ins.args[0], ctx)? {
        Operand::Stream(i) => i,
        Operand::Lit(_) => {
            return Err(Error::unimplemented(format!(
                "Vulkan: {} with a literal ladder operand is not implemented",
                ins.mn
            )))
        }
    };
    let b = match operand(&ins.args[1], ctx)? {
        Operand::Stream(i) => i,
        Operand::Lit(_) => {
            return Err(Error::unimplemented(format!(
                "Vulkan: {} with a literal ladder operand is not implemented",
                ins.mn
            )))
        }
    };
    for ord in [a, b] {
        if ctx.info(ord)?.kind != CKind::T {
            return Err(Error::asm(format!(
                "Vulkan: {} mixes ladder with another kind at {}",
                ins.mn,
                ctx.name(ord)
            )));
        }
        if ctx.elems(ord)? != out_n {
            return Err(Error::unimplemented(format!(
                "Vulkan: {} shape mismatch at {}; broadcasting not implemented",
                ins.mn,
                ctx.name(ord)
            )));
        }
    }

    let (a_s, a_e, a_z) = (
        ctx.plane_of(a, Comp::S)?,
        ctx.plane_of(a, Comp::E)?,
        ctx.plane_of(a, Comp::Z)?,
    );
    let (b_s, b_e, b_z) = (
        ctx.plane_of(b, Comp::S)?,
        ctx.plane_of(b, Comp::E)?,
        ctx.plane_of(b, Comp::Z)?,
    );
    let (o_s, o_e, o_z) = (
        ctx.plane_of(out_ord, Comp::S)?,
        ctx.plane_of(out_ord, Comp::E)?,
        ctx.plane_of(out_ord, Comp::Z)?,
    );

    let frac = tabs.get("frac", || phi_core::lattice::l_frac().to_vec());
    let coarse = tabs.get("coarse", || phi_core::lattice::l_coarse().to_vec());
    let fine = tabs.get("fine", || phi_core::lattice::l_fine().to_vec());

    let op = if sub { "-" } else { "+" };
    let ladder = LADDER_GLSL;
    let bufs = vec![a_s, a_e, a_z, b_s, b_e, b_z, o_s, o_e, o_z, frac, coarse, fine];

    let src = format!(
        r#"#version 450
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
layout(local_size_x = {LOCAL_SIZE}) in;
layout(std430, binding = 0) readonly buffer AS {{ int as_[]; }};
layout(std430, binding = 1) readonly buffer AE {{ int ae[]; }};
layout(std430, binding = 2) readonly buffer AZ {{ uint az[]; }};
layout(std430, binding = 3) readonly buffer BS {{ int bs[]; }};
layout(std430, binding = 4) readonly buffer BE {{ int be[]; }};
layout(std430, binding = 5) readonly buffer BZ {{ uint bz[]; }};
layout(std430, binding = 6) buffer OS {{ int os_[]; }};
layout(std430, binding = 7) buffer OE {{ int oe[]; }};
layout(std430, binding = 8) buffer OZ {{ uint oz[]; }};
layout(std430, binding = 9) readonly buffer FRAC {{ int64_t frac[]; }};
layout(std430, binding = 10) readonly buffer COARSE {{ int64_t coarse[]; }};
layout(std430, binding = 11) readonly buffer FINE {{ int64_t fine[]; }};
layout(push_constant) uniform PC {{ uint n; uint p0; uint p1; uint p2; }} pc;
{ladder}
void main()
{{
  uint i = gl_GlobalInvocationID.x;
  if (i >= pc.n) return;

  /* The frozen T bridge uses the declared m_cov scale, not the max rung. */
  int64_t m0 = int64_t(pc.p0);

  int64_t qa = to_fixed(as_[i], ae[i], az[i], m0);
  int64_t qb = to_fixed(bs[i], be[i], bz[i], m0);
  int64_t q = qa {op} qb;

  Ladder r = from_fixed(q, m0);
  os_[i] = r.s;
  oe[i] = r.e;
  oz[i] = r.z;
}}
"#
    );

    let groups = [div_ceil(out_n, LOCAL_SIZE as usize) as u32, 1, 1];
    Ok((src, bufs, [out_n as u32, ctx.m_cov as u32, 0, 0], groups, [LOCAL_SIZE, 1, 1]))
}

/// T = A @ B at the declared m_acc scale.
///
/// The reference sums `to_fixed(ladder_product_term, m_acc)` per term and
/// converts the accumulated integer once, which is not the same as
/// `to_fixed(a) * to_fixed(b)`.
fn tmatmul(ins: &BoundInstr, ctx: &Ctx, tabs: &mut Tables) -> Result<Lowered> {
    if ins.args.len() != 2 {
        return Err(Error::asm(format!("Vulkan: {} expects 2 operands", ins.mn)));
    }
    let a = ctx.ordinal(&ins.args[0])?;
    let b = ctx.ordinal(&ins.args[1])?;
    let out = ins
        .outs
        .first()
        .ok_or_else(|| Error::asm(format!("Vulkan: {} has no output", ins.mn)))?;
    let out_ord = ctx.ordinal(out)?;

    for ord in [a, b] {
        if ctx.info(ord)?.kind != CKind::T {
            return Err(Error::asm(format!(
                "Vulkan: {} operand {} is not a ladder stream",
                ins.mn,
                ctx.name(ord)
            )));
        }
    }
    let ash = ctx.info(a)?.static_shape(&ctx.name(a))?;
    let bsh = ctx.info(b)?.static_shape(&ctx.name(b))?;
    if ash.len() != 2 || bsh.len() != 2 {
        return Err(Error::unimplemented(format!(
            "Vulkan: {} supports rank-2 only (got {:?} x {:?}); \
             batched matmul is a later milestone",
            ins.mn, ash, bsh
        )));
    }
    let (m, k) = (ash[0], ash[1]);
    if bsh[0] != k {
        return Err(Error::asm(format!(
            "Vulkan: {} inner dimensions differ ({}x{} @ {}x{})",
            ins.mn, m, k, bsh[0], bsh[1]
        )));
    }
    let n = bsh[1];
    if ctx.elems(out_ord)? != m * n {
        return Err(Error::asm(format!(
            "Vulkan: {} output shape mismatch",
            ins.mn
        )));
    }

    let (a_s, a_e, a_z) = (
        ctx.plane_of(a, Comp::S)?,
        ctx.plane_of(a, Comp::E)?,
        ctx.plane_of(a, Comp::Z)?,
    );
    let (b_s, b_e, b_z) = (
        ctx.plane_of(b, Comp::S)?,
        ctx.plane_of(b, Comp::E)?,
        ctx.plane_of(b, Comp::Z)?,
    );
    let (o_s, o_e, o_z) = (
        ctx.plane_of(out_ord, Comp::S)?,
        ctx.plane_of(out_ord, Comp::E)?,
        ctx.plane_of(out_ord, Comp::Z)?,
    );

    let frac = tabs.get("frac", || phi_core::lattice::l_frac().to_vec());
    let coarse = tabs.get("coarse", || phi_core::lattice::l_coarse().to_vec());
    let fine = tabs.get("fine", || phi_core::lattice::l_fine().to_vec());
    let ladder = LADDER_GLSL;

    let bufs = vec![a_s, a_e, a_z, b_s, b_e, b_z, o_s, o_e, o_z, frac, coarse, fine];
    let src = format!(
        r#"#version 450
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
layout(local_size_x = 16, local_size_y = 16) in;
layout(std430, binding = 0) readonly buffer AS {{ int as_[]; }};
layout(std430, binding = 1) readonly buffer AE {{ int ae[]; }};
layout(std430, binding = 2) readonly buffer AZ {{ uint az[]; }};
layout(std430, binding = 3) readonly buffer BS {{ int bs[]; }};
layout(std430, binding = 4) readonly buffer BE {{ int be[]; }};
layout(std430, binding = 5) readonly buffer BZ {{ uint bz[]; }};
layout(std430, binding = 6) buffer OS {{ int os_[]; }};
layout(std430, binding = 7) buffer OE {{ int oe[]; }};
layout(std430, binding = 8) buffer OZ {{ uint oz[]; }};
layout(std430, binding = 9) readonly buffer FRAC {{ int64_t frac[]; }};
layout(std430, binding = 10) readonly buffer COARSE {{ int64_t coarse[]; }};
layout(std430, binding = 11) readonly buffer FINE {{ int64_t fine[]; }};
layout(push_constant) uniform PC {{ uint n; uint p0; uint p1; uint p2; }} pc;
{ladder}
void main()
{{
  uint M = pc.n;
  uint N = pc.p0;
  uint K = pc.p1;
  int64_t m0 = int64_t(pc.p2);

  uint r = gl_GlobalInvocationID.y;
  uint col = gl_GlobalInvocationID.x;
  if (r >= M || col >= N) return;

  int64_t acc = int64_t(0);
  for (uint kk = 0; kk < K; ++kk) {{
    uint ai = r * K + kk;
    uint bj = kk * N + col;
    int ps = as_[ai] * bs[bj];
    int pe = ae[ai] + be[bj] - BIAS;
    pe = pe < 0 ? 0 : (pe > NLVL - 1 ? NLVL - 1 : pe);
    uint z = az[ai] | bz[bj];
    acc += to_fixed(ps, pe, z, m0);
  }}

  Ladder res = from_fixed(acc, m0);
  os_[r * N + col] = res.s;
  oe[r * N + col] = res.e;
  oz[r * N + col] = res.z;
}}
"#
    );

    let groups = [div_ceil(n, 16) as u32, div_ceil(m, 16) as u32, 1];
    Ok((
        src,
        bufs,
        [m as u32, n as u32, k as u32, ctx.m_acc as u32],
        groups,
        [16, 16, 1],
    ))
}

fn int_bin(
    ins: &BoundInstr,
    ctx: &Ctx,
    out_ord: usize,
    out_n: usize,
    op: &str,
) -> Result<Lowered> {
    if ins.args.len() != 2 {
        return Err(Error::asm(format!("Vulkan: {} expects 2 operands", ins.mn)));
    }
    let a = match operand(&ins.args[0], ctx)? {
        Operand::Stream(i) => i,
        Operand::Lit(_) => {
            return Err(Error::unimplemented(format!(
                "Vulkan: {} with a literal integer operand is not implemented",
                ins.mn
            )))
        }
    };
    let b = match operand(&ins.args[1], ctx)? {
        Operand::Stream(i) => i,
        Operand::Lit(_) => {
            return Err(Error::unimplemented(format!(
                "Vulkan: {} with a literal integer operand is not implemented",
                ins.mn
            )))
        }
    };
    for ord in [a, b] {
        if ctx.info(ord)?.kind != CKind::I {
            return Err(Error::asm(format!(
                "Vulkan: {} mixes kinds at {}",
                ins.mn,
                ctx.name(ord)
            )));
        }
        if ctx.elems(ord)? != out_n {
            return Err(Error::unimplemented(format!(
                "Vulkan: {} shape mismatch at {}; broadcasting not implemented",
                ins.mn,
                ctx.name(ord)
            )));
        }
    }
    let a_p = ctx.plane_of(a, Comp::V)?;
    let b_p = ctx.plane_of(b, Comp::V)?;
    let o_p = ctx.plane_of(out_ord, Comp::V)?;
    let src = format!(
        r#"#version 450
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
layout(local_size_x = {LOCAL_SIZE}) in;
layout(std430, binding = 0) readonly buffer A {{ int64_t a[]; }};
layout(std430, binding = 1) readonly buffer B {{ int64_t b[]; }};
layout(std430, binding = 2) buffer O {{ int64_t c[]; }};
layout(push_constant) uniform PC {{ uint n; uint p0; uint p1; uint p2; }} pc;
void main() {{
  uint i = gl_GlobalInvocationID.x;
  if (i >= pc.n) return;
  c[i] = a[i] {op} b[i];
}}
"#
    );
    let groups = [div_ceil(out_n, LOCAL_SIZE as usize) as u32, 1, 1];
    Ok((src, vec![a_p, b_p, o_p], [out_n as u32, 0, 0, 0], groups, [LOCAL_SIZE, 1, 1]))
}

fn operand(arg: &str, ctx: &Ctx) -> Result<Operand> {
    if let Some(&i) = ctx.index.get(arg) {
        if ctx.streams.contains_key(arg) {
            return Ok(Operand::Stream(i));
        }
    }
    let v: f64 = arg.parse().map_err(|_| {
        Error::asm(format!(
            "Vulkan: operand '{arg}' is neither a stream nor a numeric literal"
        ))
    })?;
    Ok(Operand::Lit(v))
}

fn matmul(ins: &BoundInstr, ctx: &Ctx, kind: CKind) -> Result<Lowered> {
    if ins.args.len() != 2 {
        return Err(Error::asm(format!("Vulkan: {} expects 2 operands", ins.mn)));
    }
    let a_ord = ctx.ordinal(&ins.args[0])?;
    let b_ord = ctx.ordinal(&ins.args[1])?;
    let out = ins
        .outs
        .first()
        .ok_or_else(|| Error::asm(format!("Vulkan: {} has no output", ins.mn)))?;
    let out_ord = ctx.ordinal(out)?;

    for ord in [a_ord, b_ord] {
        if ctx.info(ord)?.kind != kind {
            return Err(Error::asm(format!(
                "Vulkan: {} operand {} has the wrong kind",
                ins.mn,
                ctx.name(ord)
            )));
        }
    }
    let a_sh = ctx.info(a_ord)?.static_shape(&ctx.name(a_ord))?;
    let b_sh = ctx.info(b_ord)?.static_shape(&ctx.name(b_ord))?;
    if a_sh.len() != 2 || b_sh.len() != 2 {
        return Err(Error::unimplemented(format!(
            "Vulkan: {} supports rank-2 only (got {:?} x {:?})",
            ins.mn, a_sh, b_sh
        )));
    }
    let (m, k) = (a_sh[0], a_sh[1]);
    if b_sh[0] != k {
        return Err(Error::asm(format!(
            "Vulkan: {} inner dimensions differ ({}x{} @ {}x{})",
            ins.mn, m, k, b_sh[0], b_sh[1]
        )));
    }
    let n = b_sh[1];

    let (ty, ext) = if kind == CKind::I {
        ("int64_t", "#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require\n")
    } else {
        ("float", "")
    };
    let a_p = ctx.plane_of(a_ord, Comp::V)?;
    let b_p = ctx.plane_of(b_ord, Comp::V)?;
    let o_p = ctx.plane_of(out_ord, Comp::V)?;

    let src = format!(
        r#"#version 450
{ext}layout(local_size_x = 16, local_size_y = 16) in;
layout(std430, binding = 0) readonly buffer A {{ {ty} a[]; }};
layout(std430, binding = 1) readonly buffer B {{ {ty} b[]; }};
layout(std430, binding = 2) buffer C {{ {ty} c[]; }};
layout(push_constant) uniform PC {{ uint n; uint p0; uint p1; uint p2; }} pc;
void main() {{
  uint M = pc.n; uint N = pc.p0; uint K = pc.p1;
  uint r = gl_GlobalInvocationID.y;
  uint col = gl_GlobalInvocationID.x;
  if (r >= M || col >= N) return;
  {ty} acc = {ty}(0);
  for (uint kk = 0; kk < K; ++kk)
    acc = acc + a[r * K + kk] * b[kk * N + col];
  c[r * N + col] = acc;
}}
"#
    );

    let groups = [div_ceil(n, 16) as u32, div_ceil(m, 16) as u32, 1];
    Ok((
        src,
        vec![a_p, b_p, o_p],
        [m as u32, n as u32, k as u32, 0],
        groups,
        [16, 16, 1],
    ))
}

fn div_ceil(a: usize, b: usize) -> usize {
    (a + b - 1) / b
}

fn glslc(src: &str, idx: usize) -> Result<Vec<u8>> {
    let dir = std::env::temp_dir().join(format!("jotnar-vk-{}", std::process::id()));
    std::fs::create_dir_all(&dir)?;
    let comp = dir.join(format!("s{idx}.comp"));
    let spv = dir.join(format!("s{idx}.spv"));
    std::fs::write(&comp, src)?;
    let out = std::process::Command::new("glslc")
        .arg(&comp)
        .arg("-o")
        .arg(&spv)
        .output()
        .map_err(|e| {
            Error::unimplemented(format!("Vulkan target needs `glslc` on PATH (shaderc): {e}"))
        })?;
    if !out.status.success() {
        return Err(Error::asm(format!(
            "glslc failed:\n{}\n--- shader ---\n{}",
            String::from_utf8_lossy(&out.stderr),
            src
        )));
    }
    Ok(std::fs::read(&spv)?)
}

fn emit_c(
    order: &[String],
    planes: &[(usize, Comp)],
    ctx: &Ctx,
    tabs: &Tables,
    dispatches: &[Dispatch],
    spirv: &[Vec<u8>],
    inputs: &[String],
    outputs: &[String],
    serve_in: Option<&str>,
    serve_out: Option<&str>,
) -> Result<String> {
    let mut c = String::new();
    c.push_str(HOST_PRELUDE);

    // plane table
    c.push_str(&format!("\n#define N_PLANE {}\n", planes.len()));
    c.push_str("static const char *kPlaneName[N_PLANE] = {\n");
    for (ord, comp) in planes {
        let suffix = match comp {
            Comp::V => "",
            Comp::S => ".s",
            Comp::E => ".e",
            Comp::Z => ".z",
        };
        c.push_str(&format!("  \"{}{suffix}\",\n", order[*ord]));
    }
    c.push_str("};\nstatic const uint64_t kPlaneCount[N_PLANE] = {\n");
    for (ord, _) in planes {
        c.push_str(&format!("  {},\n", ctx.elems(*ord)?));
    }
    c.push_str("};\nstatic const int kPlaneElem[N_PLANE] = {\n");
    for (ord, comp) in planes {
        let e = match elem_for(ctx.info(*ord)?.kind, *comp) {
            Elem::I8 => 0,
            Elem::U8 => 1,
            Elem::I32 => 2,
            Elem::I64 => 3,
            Elem::F32 => 4,
        };
        c.push_str(&format!("  {e},\n"));
    }
    c.push_str("};\n");

    c.push_str(&format!("\n#define N_TABLE {}\n", tabs.data.len()));
    c.push_str("#define N_BUF (N_PLANE + N_TABLE)\n");
    for (i, t) in tabs.data.iter().enumerate() {
        let vals: Vec<String> = t.iter().map(|v| format!("{v}LL")).collect();
        c.push_str(&format!(
            "static const int64_t kTable{i}[] = {{\n  {}\n}};\n",
            vals.join(", ")
        ));
    }
    let nt = tabs.data.len().max(1);
    c.push_str(&format!("static const int64_t *kTablePtr[{nt}] = {{"));
    if tabs.data.is_empty() {
        c.push_str(" 0,");
    } else {
        for i in 0..tabs.data.len() {
            c.push_str(&format!(" kTable{i},"));
        }
    }
    c.push_str(" };\n");
    c.push_str(&format!("static const uint32_t kTableN[{nt}] = {{"));
    if tabs.data.is_empty() {
        c.push_str(" 0,");
    } else {
        for t in &tabs.data {
            c.push_str(&format!(" {},", t.len()));
        }
    }
    c.push_str(" };\n");

    for (i, bytes) in spirv.iter().enumerate() {
        let words: Vec<String> = bytes
            .chunks_exact(4)
            .map(|w| format!("0x{:08x}u", u32::from_le_bytes([w[0], w[1], w[2], w[3]])))
            .collect();
        c.push_str(&format!(
            "static const uint32_t kSpv{i}[] = {{\n  {}\n}};\n",
            words.join(", ")
        ));
    }
    c.push_str(&format!("\n#define N_SHADER {}\n", spirv.len()));
    c.push_str("static const uint32_t kSpvWords[N_SHADER] = {");
    for bytes in spirv {
        c.push_str(&format!(" {},", bytes.len() / 4));
    }
    c.push_str(" };\n");

    c.push_str(&format!("\n#define N_DISPATCH {}\n", dispatches.len()));
    c.push_str("static const Dispatch kDisp[N_DISPATCH] = {\n");
    for d in dispatches {
        let bufs: String = d
            .buffers
            .iter()
            .map(|b| format!("{b}"))
            .collect::<Vec<_>>()
            .join(", ");
        c.push_str(&format!(
            "  {{ kSpv{}, {}, {}, {{ {bufs} }}, {{ {}, {}, {}, {} }}, {{ {}, {}, {} }}, {{ {}, {}, {} }} }},\n",
            d.shader,
            spirv[d.shader].len() / 4,
            d.buffers.len(),
            d.params[0], d.params[1], d.params[2], d.params[3],
            d.groups[0], d.groups[1], d.groups[2],
            d.local[0], d.local[1], d.local[2],
        ));
    }
    c.push_str("};\n");

    // flattened input/output plane + argv maps
    let mut in_pairs: Vec<(usize, usize)> = Vec::new();
    let mut argv = 1usize;
    for name in inputs {
        let ord = ctx.ordinal(name)?;
        for &comp in comps_for(ctx.info(ord)?.kind) {
            in_pairs.push((ctx.plane_of(ord, comp)?, argv));
            argv += 1;
        }
    }
    let mut out_pairs: Vec<(usize, usize)> = Vec::new();
    for name in outputs {
        let ord = ctx.ordinal(name)?;
        for &comp in comps_for(ctx.info(ord)?.kind) {
            out_pairs.push((ctx.plane_of(ord, comp)?, argv));
            argv += 1;
        }
    }

    c.push_str(&format!("\n#define N_IN {}\n", in_pairs.len()));
    c.push_str("static const int kInPlane[N_IN] = {");
    for (p, _) in &in_pairs {
        c.push_str(&format!(" {p},"));
    }
    c.push_str(" };\nstatic const int kInArgv[N_IN] = {");
    for (_, a) in &in_pairs {
        c.push_str(&format!(" {a},"));
    }
    c.push_str(" };\n");

    c.push_str(&format!("\n#define N_OUT {}\n", out_pairs.len()));
    c.push_str("static const int kOutPlane[N_OUT] = {");
    for (p, _) in &out_pairs {
        c.push_str(&format!(" {p},"));
    }
    c.push_str(" };\nstatic const int kOutArgv[N_OUT] = {");
    for (_, a) in &out_pairs {
        c.push_str(&format!(" {a},"));
    }
    c.push_str(" };\n");

    let serve_in_planes: Vec<usize> = match serve_in {
        Some(n) => {
            let ord = ctx.ordinal(n)?;
            comps_for(ctx.info(ord)?.kind)
                .iter()
                .map(|&c| ctx.plane_of(ord, c))
                .collect::<Result<_>>()?
        }
        None => Vec::new(),
    };
    let serve_out_planes: Vec<usize> = match serve_out {
        Some(n) => {
            let ord = ctx.ordinal(n)?;
            comps_for(ctx.info(ord)?.kind)
                .iter()
                .map(|&c| ctx.plane_of(ord, c))
                .collect::<Result<_>>()?
        }
        None => Vec::new(),
    };
    c.push_str(&format!("\n#define SERVE_MODE {}\n", serve_in.is_some() as u8));
    c.push_str(&format!("#define N_SERVE_IN {}\n", serve_in_planes.len()));
    c.push_str("static const int kServeIn[N_SERVE_IN > 0 ? N_SERVE_IN : 1] = {");
    for p in &serve_in_planes { c.push_str(&format!(" {p},")); }
    if serve_in_planes.is_empty() { c.push_str(" 0,"); }
    c.push_str(" };\n");
    c.push_str(&format!("#define N_SERVE_OUT {}\n", serve_out_planes.len()));
    c.push_str("static const int kServeOut[N_SERVE_OUT > 0 ? N_SERVE_OUT : 1] = {");
    for p in &serve_out_planes { c.push_str(&format!(" {p},")); }
    if serve_out_planes.is_empty() { c.push_str(" 0,"); }
    c.push_str(" };\n");

    c.push_str(HOST_MAIN);
    Ok(c)
}

pub fn build_vulkan(
    source: &str,
    workdir: &std::path::Path,
    name: &str,
) -> Result<std::path::PathBuf> {
    std::fs::create_dir_all(workdir)?;
    let src = workdir.join(format!("{name}.c"));
    let exe = workdir.join(name);
    std::fs::write(&src, source)?;
    let out = std::process::Command::new("cc")
        .args(["-O2", "-std=c99", "-o"])
        .arg(&exe)
        .arg(&src)
        .args(["-lvulkan", "-lm"])
        .output()?;
    if !out.status.success() {
        return Err(Error::asm(format!(
            "Vulkan C build failed:\n{}",
            String::from_utf8_lossy(&out.stderr)
        )));
    }
    Ok(exe)
}

const HOST_PRELUDE: &str = r#"/* generated by jotnar: Vulkan compute backend */
#define _POSIX_C_SOURCE 200809L
#include <vulkan/vulkan.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now_ms(void)
{
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

#define CK(x) do { VkResult _r = (x); if (_r != VK_SUCCESS) { \
  fprintf(stderr, "vk error %d at line %d\n", (int)_r, __LINE__); return 7; } } while (0)

enum { E_I8 = 0, E_U8 = 1, E_I32 = 2, E_I64 = 3, E_F32 = 4 };

typedef struct {
  const uint32_t *code;
  uint32_t words;
  int nbuf;
  int buf[16];
  uint32_t p[4];
  uint32_t g[3];
  uint32_t l[3];
} Dispatch;

"#;

const HOST_MAIN: &str = r#"
static VkInstance g_inst;
static VkPhysicalDevice g_phys;
static VkDevice g_dev;
static VkQueue g_queue;
static uint32_t g_qfam;
static VkCommandPool g_cpool;
static VkDescriptorPool g_dpool;
static VkBuffer g_buf[N_BUF];
static VkDeviceMemory g_mem[N_BUF];
static void *g_map[N_BUF];
static VkDeviceSize g_size[N_BUF];
static VkDescriptorSetLayout g_dsl[N_DISPATCH];
static VkPipelineLayout g_pl[N_DISPATCH];
static VkPipeline g_pipe[N_DISPATCH];
static VkDescriptorSet g_ds[N_DISPATCH];
static const char *g_argv_in[N_IN];
static const char *g_argv_out[N_OUT];

static uint32_t mem_type(uint32_t bits, VkMemoryPropertyFlags want)
{
  VkPhysicalDeviceMemoryProperties mp;
  vkGetPhysicalDeviceMemoryProperties(g_phys, &mp);
  for (uint32_t i = 0; i < mp.memoryTypeCount; ++i)
    if ((bits & (1u << i)) && (mp.memoryTypes[i].propertyFlags & want) == want)
      return i;
  return UINT32_MAX;
}

static VkDeviceSize buf_bytes(int p)
{
  if (p >= N_PLANE)
    return (VkDeviceSize)kTableN[p - N_PLANE] * 8;
  int elem = kPlaneElem[p];
  VkDeviceSize w = (elem == E_I64) ? 8 : 4;
  return kPlaneCount[p] * w;
}

static int init_vk(void)
{
  VkApplicationInfo app = { .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO };
  app.pApplicationName = "jotnar-vulkan";
  app.apiVersion = VK_API_VERSION_1_1;

  VkInstanceCreateInfo ci = { .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO };
  ci.pApplicationInfo = &app;
  CK(vkCreateInstance(&ci, NULL, &g_inst));

  uint32_t n = 0;
  CK(vkEnumeratePhysicalDevices(g_inst, &n, NULL));
  if (n == 0) { fprintf(stderr, "no Vulkan device\n"); return 3; }
  VkPhysicalDevice *phys = calloc(n, sizeof(*phys));
  CK(vkEnumeratePhysicalDevices(g_inst, &n, phys));
  g_phys = phys[0];
  free(phys);

  uint32_t qn = 0;
  vkGetPhysicalDeviceQueueFamilyProperties(g_phys, &qn, NULL);
  VkQueueFamilyProperties *qp = calloc(qn, sizeof(*qp));
  vkGetPhysicalDeviceQueueFamilyProperties(g_phys, &qn, qp);
  g_qfam = UINT32_MAX;
  for (uint32_t i = 0; i < qn; ++i)
    if (qp[i].queueFlags & VK_QUEUE_COMPUTE_BIT) { g_qfam = i; break; }
  free(qp);
  if (g_qfam == UINT32_MAX) { fprintf(stderr, "no compute queue\n"); return 3; }

  VkPhysicalDeviceFeatures feats;
  vkGetPhysicalDeviceFeatures(g_phys, &feats);
  if (!feats.shaderInt64) {
    fprintf(stderr, "device lacks shaderInt64 (needed for I streams)\n");
    return 4;
  }

  float prio = 1.0f;
  VkDeviceQueueCreateInfo qci = { .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO };
  qci.queueFamilyIndex = g_qfam;
  qci.queueCount = 1;
  qci.pQueuePriorities = &prio;

  VkPhysicalDeviceFeatures enabled = { .shaderInt64 = VK_TRUE };
  VkDeviceCreateInfo dci = { .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO };
  dci.queueCreateInfoCount = 1;
  dci.pQueueCreateInfos = &qci;
  dci.pEnabledFeatures = &enabled;
  CK(vkCreateDevice(g_phys, &dci, NULL, &g_dev));
  vkGetDeviceQueue(g_dev, g_qfam, 0, &g_queue);

  VkCommandPoolCreateInfo cp = { .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO };
  cp.queueFamilyIndex = g_qfam;
  cp.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
  CK(vkCreateCommandPool(g_dev, &cp, NULL, &g_cpool));

  int total_bindings = 0;
  for (int i = 0; i < N_DISPATCH; ++i) total_bindings += kDisp[i].nbuf;
  VkDescriptorPoolSize sz = { .type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                              .descriptorCount = (uint32_t)(total_bindings + 8) };
  VkDescriptorPoolCreateInfo dpi = { .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
  dpi.maxSets = (uint32_t)(N_DISPATCH + 1);
  dpi.poolSizeCount = 1;
  dpi.pPoolSizes = &sz;
  CK(vkCreateDescriptorPool(g_dev, &dpi, NULL, &g_dpool));
  return 0;
}

static int make_buffers(void)
{
  for (int i = 0; i < N_BUF; ++i) {
    g_size[i] = buf_bytes(i);
    if (g_size[i] == 0) g_size[i] = 4;
    VkBufferCreateInfo bi = { .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
    bi.size = g_size[i];
    bi.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    CK(vkCreateBuffer(g_dev, &bi, NULL, &g_buf[i]));
    VkMemoryRequirements mr;
    vkGetBufferMemoryRequirements(g_dev, g_buf[i], &mr);
    /* Prefer host-cached coherent memory: reading results straight out of
       write-combined memory is dramatically slower than the compute. */
    uint32_t mt = mem_type(mr.memoryTypeBits,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT |
        VK_MEMORY_PROPERTY_HOST_CACHED_BIT);
    if (mt == UINT32_MAX)
      mt = mem_type(mr.memoryTypeBits,
          VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    if (mt == UINT32_MAX) { fprintf(stderr, "no host-visible memory\n"); return 5; }
    VkMemoryAllocateInfo ma = { .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO };
    ma.allocationSize = mr.size;
    ma.memoryTypeIndex = mt;
    CK(vkAllocateMemory(g_dev, &ma, NULL, &g_mem[i]));
    CK(vkBindBufferMemory(g_dev, g_buf[i], g_mem[i], 0));
    CK(vkMapMemory(g_dev, g_mem[i], 0, VK_WHOLE_SIZE, 0, &g_map[i]));
    memset(g_map[i], 0, (size_t)g_size[i]);
    if (i >= N_PLANE)
      memcpy(g_map[i], kTablePtr[i - N_PLANE], (size_t)g_size[i]);
  }
  return 0;
}

/* ---- plane I/O: files for one-shot runs, stdin/stdout for serve ---- */

static int plane_file_width(int plane)
{
  int elem = kPlaneElem[plane];
  return (elem == E_I8 || elem == E_U8) ? 1 : (elem == E_I64) ? 8 : 4;
}

/* returns 0 ok, 1 clean EOF before any byte, 2 error */
static int read_plane(FILE *f, int plane)
{
  int elem = kPlaneElem[plane];
  int fw = plane_file_width(plane);
  uint64_t n = kPlaneCount[plane];
  size_t want = (size_t)(n * (uint64_t)fw);
  unsigned char *tmp = malloc(want ? want : 1);
  if (!tmp) return 2;
  size_t got = fread(tmp, 1, want, f);
  if (got == 0 && want > 0) { free(tmp); return 1; }
  if (got != want) { free(tmp); return 2; }

  for (uint64_t j = 0; j < n; ++j) {
    switch (elem) {
      case E_I8:  ((int32_t *)g_map[plane])[j] = (int32_t)((int8_t *)tmp)[j]; break;
      case E_U8:  ((uint32_t *)g_map[plane])[j] = (uint32_t)((uint8_t *)tmp)[j]; break;
      case E_I32: ((int32_t *)g_map[plane])[j] = ((int32_t *)tmp)[j]; break;
      case E_I64: ((int64_t *)g_map[plane])[j] = ((int64_t *)tmp)[j]; break;
      case E_F32: ((float *)g_map[plane])[j] = ((float *)tmp)[j]; break;
    }
  }
  free(tmp);
  return 0;
}

static int write_plane(FILE *f, int plane)
{
  int elem = kPlaneElem[plane];
  int fw = plane_file_width(plane);
  uint64_t n = kPlaneCount[plane];
  size_t want = (size_t)(n * (uint64_t)fw);
  unsigned char *tmp = malloc(want ? want : 1);
  if (!tmp) return 6;
  for (uint64_t j = 0; j < n; ++j) {
    switch (elem) {
      case E_I8:  ((int8_t *)tmp)[j] = (int8_t)((int32_t *)g_map[plane])[j]; break;
      case E_U8:  ((uint8_t *)tmp)[j] = (uint8_t)((uint32_t *)g_map[plane])[j]; break;
      case E_I32: ((int32_t *)tmp)[j] = ((int32_t *)g_map[plane])[j]; break;
      case E_I64: ((int64_t *)tmp)[j] = ((int64_t *)g_map[plane])[j]; break;
      case E_F32: ((float *)tmp)[j] = ((float *)g_map[plane])[j]; break;
    }
  }
  int ok = fwrite(tmp, 1, want, f) == want;
  free(tmp);
  return ok ? 0 : 6;
}

static int load_inputs(void)
{
  for (int i = 0; i < N_IN; ++i) {
    FILE *f = fopen(g_argv_in[i], "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", g_argv_in[i]); return 1; }
    int r = read_plane(f, kInPlane[i]);
    fclose(f);
    if (r) {
      fprintf(stderr, "%s: bad plane %s\n", g_argv_in[i], kPlaneName[kInPlane[i]]);
      return 2;
    }
  }
  return 0;
}

static int store_outputs(void)
{
  for (int i = 0; i < N_OUT; ++i) {
    FILE *f = fopen(g_argv_out[i], "wb");
    if (!f) { fprintf(stderr, "cannot write %s\n", g_argv_out[i]); return 6; }
    int r = write_plane(f, kOutPlane[i]);
    fclose(f);
    if (r) return r;
  }
  return 0;
}

static int run(void);

#if SERVE_MODE
/* Resident loop: raw frames on stdin, raw frames on stdout. One frame is
   the serve stream's planes concatenated in argv order. */
static int serve_loop(void)
{
  for (;;) {
    int eof = 0;
    for (int i = 0; i < N_SERVE_IN; ++i) {
      int r = read_plane(stdin, kServeIn[i]);
      if (r == 1) { eof = 1; break; }
      if (r) { fprintf(stderr, "serve: short input frame\n"); return r; }
    }
    if (eof) break;
    int r = run();
    if (r) return r;
    for (int i = 0; i < N_SERVE_OUT; ++i) {
      r = write_plane(stdout, kServeOut[i]);
      if (r) return r;
    }
    fflush(stdout);
  }
  return 0;
}
#endif

static int build_pipelines(void)
{
  for (int d = 0; d < N_DISPATCH; ++d) {
    const Dispatch *dp = &kDisp[d];

    VkDescriptorSetLayoutBinding binds[16];
    for (int k = 0; k < dp->nbuf; ++k) {
      binds[k].binding = (uint32_t)k;
      binds[k].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
      binds[k].descriptorCount = 1;
      binds[k].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
      binds[k].pImmutableSamplers = NULL;
    }
    VkDescriptorSetLayoutCreateInfo li = { .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
    li.bindingCount = (uint32_t)dp->nbuf;
    li.pBindings = binds;
    CK(vkCreateDescriptorSetLayout(g_dev, &li, NULL, &g_dsl[d]));

    VkPushConstantRange pr = { .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
                               .offset = 0, .size = 16 };
    VkPipelineLayoutCreateInfo pi = { .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
    pi.setLayoutCount = 1;
    pi.pSetLayouts = &g_dsl[d];
    pi.pushConstantRangeCount = 1;
    pi.pPushConstantRanges = &pr;
    CK(vkCreatePipelineLayout(g_dev, &pi, NULL, &g_pl[d]));

    VkShaderModuleCreateInfo sm = { .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO };
    sm.pCode = dp->code;
    sm.codeSize = (size_t)dp->words * 4;
    VkShaderModule mod;
    CK(vkCreateShaderModule(g_dev, &sm, NULL, &mod));

    VkComputePipelineCreateInfo cpi = { .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO };
    cpi.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cpi.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    cpi.stage.module = mod;
    cpi.stage.pName = "main";
    cpi.layout = g_pl[d];
    CK(vkCreateComputePipelines(g_dev, VK_NULL_HANDLE, 1, &cpi, NULL, &g_pipe[d]));
    vkDestroyShaderModule(g_dev, mod, NULL);

    VkDescriptorSetAllocateInfo ai = { .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
    ai.descriptorPool = g_dpool;
    ai.descriptorSetCount = 1;
    ai.pSetLayouts = &g_dsl[d];
    CK(vkAllocateDescriptorSets(g_dev, &ai, &g_ds[d]));

    for (int k = 0; k < dp->nbuf; ++k) {
      VkDescriptorBufferInfo bi = { .buffer = g_buf[dp->buf[k]], .offset = 0,
                                    .range = VK_WHOLE_SIZE };
      VkWriteDescriptorSet w = { .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
      w.dstSet = g_ds[d];
      w.dstBinding = (uint32_t)k;
      w.descriptorCount = 1;
      w.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
      w.pBufferInfo = &bi;
      vkUpdateDescriptorSets(g_dev, 1, &w, 0, NULL);
    }
  }
  return 0;
}

static int run(void)
{
  static VkCommandBuffer cmd = VK_NULL_HANDLE;
  if (cmd == VK_NULL_HANDLE) {
    VkCommandBufferAllocateInfo ai = { .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
    ai.commandPool = g_cpool;
    ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    ai.commandBufferCount = 1;
    CK(vkAllocateCommandBuffers(g_dev, &ai, &cmd));
  } else {
    CK(vkResetCommandBuffer(cmd, 0));
  }

  VkCommandBufferBeginInfo bi = { .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
  CK(vkBeginCommandBuffer(cmd, &bi));

  for (int d = 0; d < N_DISPATCH; ++d) {
    const Dispatch *dp = &kDisp[d];
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, g_pipe[d]);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, g_pl[d],
                            0, 1, &g_ds[d], 0, NULL);
    vkCmdPushConstants(cmd, g_pl[d], VK_SHADER_STAGE_COMPUTE_BIT, 0, 16, dp->p);
    vkCmdDispatch(cmd, dp->g[0], dp->g[1], dp->g[2]);

    VkMemoryBarrier mb = { .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER };
    mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         0, 1, &mb, 0, NULL, 0, NULL);
  }

  CK(vkEndCommandBuffer(cmd));
  VkSubmitInfo si = { .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO };
  si.commandBufferCount = 1;
  si.pCommandBuffers = &cmd;
  CK(vkQueueSubmit(g_queue, 1, &si, VK_NULL_HANDLE));
  CK(vkQueueWaitIdle(g_queue));
  return 0;
}

int main(int argc, char **argv)
{
  if (argc != N_IN + N_OUT + 1) {
    fprintf(stderr, "usage: needs %d file arguments (got %d)\n",
            N_IN + N_OUT, argc - 1);
    return 9;
  }
  for (int i = 0; i < N_IN; ++i) g_argv_in[i] = argv[kInArgv[i]];
  for (int i = 0; i < N_OUT; ++i) g_argv_out[i] = argv[kOutArgv[i]];

  int r;
  double t0 = now_ms(), t1, t2, t3, t4, t5, t6;
  if ((r = init_vk())) return r;
  t1 = now_ms();
  if ((r = make_buffers())) return r;
  t2 = now_ms();
  if ((r = build_pipelines())) return r;
  t3 = now_ms();
  if ((r = load_inputs())) return r;
  t4 = now_ms();
#if SERVE_MODE
  if ((r = serve_loop())) return r;
  t5 = t6 = now_ms();
#else
  if ((r = run())) return r;
  t5 = now_ms();
  if ((r = store_outputs())) return r;
  t6 = now_ms();
#endif
  if (getenv("JOTNAR_VK_TIME"))
    fprintf(stderr,
            "phases ms: init=%.2f buffers=%.2f load=%.2f pipelines=%.2f "
            "dispatch=%.2f store=%.2f total=%.2f\n",
            t1 - t0, t2 - t1, t3 - t2, t4 - t3, t5 - t4, t6 - t5, t6 - t0);
  return 0;
}
"#;
