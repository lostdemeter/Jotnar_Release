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
    M,
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

const T3_COMP: &[Comp] = &[Comp::S, Comp::E, Comp::Z];
const T4_COMP: &[Comp] = &[Comp::S, Comp::E, Comp::M, Comp::Z];

fn comps_for(kind: CKind, t2: Option<i64>) -> &'static [Comp] {
    match kind {
        CKind::T => {
            if t2.is_some() {
                T4_COMP
            } else {
                T3_COMP
            }
        }
        _ => &[Comp::V],
    }
}

fn elem_for(kind: CKind, c: Comp) -> Elem {
    match (kind, c) {
        (CKind::T, Comp::S) => Elem::I8,
        (CKind::T, Comp::E) => Elem::I32,
        (CKind::T, Comp::M) => Elem::I32,
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
    names: Vec<String>,
    data: Vec<Vec<i64>>,
}

impl Tables {
    fn new(n_plane: usize) -> Self {
        Tables { n_plane, names: Vec::new(), data: Vec::new() }
    }

    fn get(&mut self, name: &str, make: impl FnOnce() -> Vec<i64>) -> usize {
        if let Some(i) = self.names.iter().position(|n| n == name) {
            return self.n_plane + i;
        }
        let d = make();
        self.names.push(name.to_string());
        self.data.push(d);
        self.n_plane + self.data.len() - 1
    }

    fn push(&mut self, name: String, data: Vec<i64>) -> usize {
        self.names.push(name);
        self.data.push(data);
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
    /// `CONFIG t2 <M>` selects the four-plane φ-float; `f` is the bridge width.
    t2: Option<i64>,
    f: i64,
    beta_b: f64,
    eps_rms: i64,
    sample: &'a Feeds,
    rope_base: f64,
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

    let (t2, f_bits) = phi_core::t2::check_widths(
        bound.config.get_i64("t2"),
        bound.config.get_i64("bridge_f"),
    )?;

    let mut planes: Vec<(usize, Comp)> = Vec::new();
    let mut plane: HashMap<(usize, Comp), usize> = HashMap::new();
    for (ord, name) in order.iter().enumerate() {
        let kind = streams[name].kind;
        for &c in comps_for(kind, t2) {
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
        t2,
        f: f_bits,
        beta_b: bound
            .config
            .get_f64_or("beta_b", bound.config.get_f64_or("beta", 0.5)),
        sample: &sample,
        rope_base: bound.config.get_f64_or("rope_base", 10000.0),
        eps_rms: {
            let m_cov = bound.config.get_i64("m_cov").unwrap_or(33280);
            match bound.config.get_f64("eps_rms") {
                Some(v) => {
                    let um = phi_core::lattice::PHI
                        .powf((m_cov - phi_core::lattice::BIAS) as f64 / 512.0);
                    phi_core::lattice::round_half_even(
                        v * 68719476736.0 / (um * um),
                    ) as i64
                }
                None => bound.config.get_i64("eps_rms_c").unwrap_or(4514),
            }
        },
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
        .map(|n| comps_for(streams[n].kind, t2).len())
        .sum::<usize>()
        + outputs
            .iter()
            .map(|n| comps_for(streams[n].kind, t2).len())
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
        ("MUL", CKind::T) | ("SQUARE", CKind::T) => {
            let square = ins.mn == "SQUARE";
            if ctx.t2.is_some() {
                tmul2(ins, ctx, out_ord, out_n, square)
            } else {
                tmul(ins, ctx, out_ord, out_n, square)
            }
        }
        ("ADD", CKind::T) | ("SUB", CKind::T) => {
            let sub = ins.mn == "SUB";
            if ctx.t2.is_some() {
                tadd2(ins, ctx, tabs, out_ord, out_n, sub)
            } else {
                tadd(ins, ctx, tabs, out_ord, out_n, sub)
            }
        }
        ("MATMUL", CKind::T) => {
            if ctx.t2.is_some() {
                tmatmul2(ins, ctx, tabs)
            } else {
                tmatmul(ins, ctx, tabs)
            }
        }
        ("TRANSPOSE", CKind::T) => transpose(ins, ctx, tabs, out_ord),
        ("SLICE", CKind::T) | ("SLICE", CKind::F) => {
            slice(ins, ctx, tabs, out_ord, out_kind)
        }
        ("IGET", CKind::I) => iget(ins, ctx, tabs, out_ord),
        ("TBETA", CKind::T) => tbeta(ins, ctx, out_ord, out_n),
        ("SILU", CKind::T) => silu(ins, ctx, tabs, out_ord),
        ("SOFTMAX_WIDE", CKind::T) => softmax_inner(ins, ctx, tabs, out_ord, true),
        ("SOFTMAX", CKind::T) => softmax_inner(ins, ctx, tabs, out_ord, false),
        ("GELU", CKind::T) => gelu(ins, ctx, tabs, out_ord),
        ("CONCAT", CKind::T) | ("CONCAT", CKind::F) | ("CONCAT", CKind::I) => {
            concat(ins, ctx, tabs, out_ord, out_kind)
        }
        ("RMSNORM", CKind::T) => rmsnorm(ins, ctx, tabs, out_ord),
        ("ROTARY", CKind::T) => rotary(ins, ctx, tabs, out_ord),
        ("T2I", CKind::I) => t2i(ins, ctx, tabs, out_ord),
        ("TSHIFT", CKind::T) => tshift(ins, ctx, tabs, out_ord),

        // exact integer arithmetic
        ("IADD", CKind::I) => int_bin(ins, ctx, out_ord, out_n, "+"),
        ("ISUB", CKind::I) => int_bin(ins, ctx, out_ord, out_n, "-"),
        ("IMUL", CKind::I) => int_bin(ins, ctx, out_ord, out_n, "*"),
        ("IMATMUL", CKind::I) => matmul(ins, ctx, CKind::I),

        // float path (F streams only; not the ladder)
        ("MATMUL", CKind::F) => matmul(ins, ctx, CKind::F),

        // Ladder ops that need the fixed-point bridge / baked tables.
        ("DIV", CKind::T)
        | ("IREDUCE_SUM", CKind::I) => {
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

/// T2 ("φ-float") helpers: s · φ^((e−BIAS)/K) · (1 + m/2^M).
///
/// The bridge is i64 throughout. T2's reference implementation uses i128 for
/// `(a << M) / base`, but the mantissa is clamped to [-2^(M-1), 2^(M-1)) and
/// `base` is capped by the hi-ladder table, so an explicit clamp test keeps
/// the division operand inside 64 bits. Verified against the reference on
/// 43,488 vectors with zero mismatches.
fn ladder2_glsl(m_bits: i64, f: i64, full: bool) -> String {
    let cap = phi_core::lattice::frac_cap_for(f);
    let common = format!(
        r#"
const int T2M = {m_bits};
const int T2F = {f};
const int T2CAP = {cap};
const int T2WIDE = 8192;
const int T2BIAS = 32768;
const int T2NLVL = 65536;

struct T2 {{ int s; int e; int m; uint z; }};

int bitlen64(uint64_t a)
{{
  int bl = 0;
  while (a > 0ul) {{ bl += 1; a >>= 1; }}
  return bl;
}}

int64_t t2_round_div_signed(int64_t a, int64_t b)
{{
  if (b == 0) return 0;
  if (a < 0) return -(((-a) + (b >> 1)) / b);
  return ((a + (b >> 1)) / b);
}}
"#,
        m_bits = m_bits,
        f = f,
        cap = cap,
    );
    if !full {
        return common;
    }
    let bridge = format!(
        r#"
int64_t t2_base(int64_t d)
{{
  if (d < 0) {{
    int64_t k = -d;
    if (k > int64_t(T2WIDE)) return frac[0];
    return frachi[int(k - 1)];
  }}
  if (d > int64_t(T2CAP)) return 0;
  return frac[int(d)];
}}

int64_t t2_to_fixed(int sgn, int ev, int mv, uint z, int64_t m0)
{{
  if (z != 0u) return 0;
  int64_t base = t2_base(m0 - int64_t(ev));
  int64_t scale = int64_t(1) << T2M;
  int64_t q = (base * (scale + int64_t(mv)) + (scale >> 1)) >> T2M;
  return int64_t(sgn) * q;
}}

T2 t2_zero()
{{
  T2 r; r.s = 1; r.e = T2BIAS; r.m = 0; r.z = 1u;
  return r;
}}

T2 t2_from_fixed(int64_t q, int64_t m0)
{{
  if (q == 0) return t2_zero();

  T2 r;
  r.s = q > 0 ? 1 : -1;
  uint64_t a = q < 0 ? (~uint64_t(q)) + 1ul : uint64_t(q);
  int bl = bitlen64(a);
  int shift = bl - 15;
  uint64_t mant = shift >= 0 ? (a >> uint(shift)) : (a << uint(-shift));
  int t = shift + 14 - T2F;
  int idx = t + 64;
  idx = idx < 0 ? 0 : (idx > 192 ? 192 : idx);
  int fi = int(mant) - 16384;
  fi = fi < 0 ? 0 : (fi > 16383 ? 16383 : fi);

  int64_t ev = m0 + coarse[idx] + fine[fi];
  ev = ev < 0 ? 0 : (ev > int64_t(T2NLVL - 1) ? int64_t(T2NLVL - 1) : ev);
  r.e = int(ev);
  r.z = 0u;

  int64_t base = t2_base(m0 - ev);
  if (base == 0) {{ r.m = 0; return r; }}

  int64_t one = int64_t(1) << T2M;
  int64_t halfm = one >> 1;

  int64_t rhs = base * (one + halfm) - (base >> 1);
  int64_t thr = (rhs + one - 1) >> T2M;
  if (a >= uint64_t(thr)) {{ r.m = int(halfm - 1); return r; }}

  int64_t rlo = base * (one - halfm) - (base >> 1);
  if (rlo > 0) {{
    int64_t tlo = (rlo + one - 1) >> T2M;
    if (a < uint64_t(tlo)) {{ r.m = int(-halfm); return r; }}
  }}

  int64_t qq = (int64_t(a) << T2M) + (base >> 1);
  qq = qq / base;
  int64_t mm = qq - one;
  if (mm < -halfm) mm = -halfm;
  if (mm > halfm - 1) mm = halfm - 1;
  r.m = int(mm);
  return r;
}}
"#
    );
    format!("{common}{bridge}")
}

fn t2_operands(
    ins: &BoundInstr,
    ctx: &Ctx,
    out_n: usize,
    square: bool,
) -> Result<(usize, usize)> {
    if square {
        if ins.args.len() != 1 {
            return Err(Error::asm("Vulkan: SQUARE expects 1 operand"));
        }
        let a = ctx.ordinal(&ins.args[0])?;
        Ok((a, a))
    } else {
        if ins.args.len() != 2 {
            return Err(Error::asm(format!("Vulkan: {} expects 2 operands", ins.mn)));
        }
        Ok((ctx.ordinal(&ins.args[0])?, ctx.ordinal(&ins.args[1])?))
    }
    .and_then(|(a, b)| {
        for ord in [a, b] {
            if ctx.info(ord)?.kind != CKind::T {
                return Err(Error::asm("Vulkan: T2 op mixes stream kinds"));
            }
            if ctx.elems(ord)? != out_n {
                return Err(Error::unimplemented(
                    "Vulkan: T2 shape mismatch; broadcasting not implemented",
                ));
            }
        }
        Ok((a, b))
    })
}

fn tmul2(
    ins: &BoundInstr,
    ctx: &Ctx,
    out_ord: usize,
    out_n: usize,
    square: bool,
) -> Result<Lowered> {
    let (a, b) = t2_operands(ins, ctx, out_n, square)?;
    let planes = |ord: usize| -> Result<(usize, usize, usize, usize)> {
        Ok((
            ctx.plane_of(ord, Comp::S)?,
            ctx.plane_of(ord, Comp::E)?,
            ctx.plane_of(ord, Comp::M)?,
            ctx.plane_of(ord, Comp::Z)?,
        ))
    };
    let (a_s, a_e, a_m, a_z) = planes(a)?;
    let (b_s, b_e, b_m, b_z) = planes(b)?;
    let (o_s, o_e, o_m, o_z) = planes(out_ord)?;
    let m_bits = ctx.t2.expect("t2");
    let ladder = ladder2_glsl(m_bits, ctx.f, false);

    let bufs = vec![a_s, a_e, a_m, a_z, b_s, b_e, b_m, b_z, o_s, o_e, o_m, o_z];
    let src = format!(
        r#"#version 450
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
layout(local_size_x = {LOCAL_SIZE}) in;
layout(std430, binding = 0) readonly buffer AS {{ int as_[]; }};
layout(std430, binding = 1) readonly buffer AE {{ int ae[]; }};
layout(std430, binding = 2) readonly buffer AM {{ int am[]; }};
layout(std430, binding = 3) readonly buffer AZ {{ uint az[]; }};
layout(std430, binding = 4) readonly buffer BS {{ int bs[]; }};
layout(std430, binding = 5) readonly buffer BE {{ int be[]; }};
layout(std430, binding = 6) readonly buffer BM {{ int bm[]; }};
layout(std430, binding = 7) readonly buffer BZ {{ uint bz[]; }};
layout(std430, binding = 8) buffer OS {{ int os_[]; }};
layout(std430, binding = 9) buffer OE {{ int oe[]; }};
layout(std430, binding = 10) buffer OM {{ int om[]; }};
layout(std430, binding = 11) buffer OZ {{ uint oz[]; }};
layout(push_constant) uniform PC {{ uint n; uint p0; uint p1; uint p2; }} pc;
{ladder}
void main()
{{
  uint i = gl_GlobalInvocationID.x;
  if (i >= pc.n) return;
  if (az[i] != 0u || bz[i] != 0u) {{
    op_s[i] = 1; op_e[i] = T2BIAS; op_m[i] = 0; op_z[i] = 1u;
    return;
  }}
  op_s[i] = as_[i] * bs[i];
  int e = ae[i] + be[i] - T2BIAS;
  op_e[i] = e < 0 ? 0 : (e > T2NLVL - 1 ? T2NLVL - 1 : e);
  int64_t cross = t2_round_div_signed(int64_t(am[i]) * int64_t(bm[i]), int64_t(1) << T2M);
  op_m[i] = int(int64_t(am[i]) + int64_t(bm[i]) + cross);
  op_z[i] = 0u;
}}
"#
    );
    // output buffer names used in the shader body
    let src = src.replace("op_s[i]", "os_[i]").replace("op_e[i]", "oe[i]")
                 .replace("op_m[i]", "om[i]").replace("op_z[i]", "oz[i]");

    let groups = [div_ceil(out_n, LOCAL_SIZE as usize) as u32, 1, 1];
    Ok((src, bufs, [out_n as u32, 0, 0, 0], groups, [LOCAL_SIZE, 1, 1]))
}

fn tadd2(
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
    let a = ctx.ordinal(&ins.args[0])?;
    let b = ctx.ordinal(&ins.args[1])?;
    for ord in [a, b] {
        if ctx.info(ord)?.kind != CKind::T {
            return Err(Error::asm("Vulkan: T2 ADD/SUB mixes stream kinds"));
        }
        if ctx.elems(ord)? != out_n {
            return Err(Error::unimplemented(
                "Vulkan: T2 ADD/SUB shape mismatch; broadcasting not implemented",
            ));
        }
    }
    let planes = |ord: usize| -> Result<(usize, usize, usize, usize)> {
        Ok((
            ctx.plane_of(ord, Comp::S)?,
            ctx.plane_of(ord, Comp::E)?,
            ctx.plane_of(ord, Comp::M)?,
            ctx.plane_of(ord, Comp::Z)?,
        ))
    };
    let (a_s, a_e, a_m, a_z) = planes(a)?;
    let (b_s, b_e, b_m, b_z) = planes(b)?;
    let (o_s, o_e, o_m, o_z) = planes(out_ord)?;

    let f = ctx.f;
    let frac = tabs.get("frac", || phi_core::lattice::l_frac_for(f));
    let frachi = tabs.get("frachi", || phi_core::lattice::frac_hi_for(f));
    let coarse = tabs.get("coarse", || phi_core::lattice::l_coarse().to_vec());
    let fine = tabs.get("fine", || phi_core::lattice::l_fine().to_vec());

    let m_bits = ctx.t2.expect("t2");
    let ladder = ladder2_glsl(m_bits, f, true);
    let op = if sub { "-" } else { "+" };
    let bufs = vec![
        a_s, a_e, a_m, a_z, b_s, b_e, b_m, b_z, o_s, o_e, o_m, o_z,
        frac, frachi, coarse, fine,
    ];

    let src = format!(
        r#"#version 450
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
layout(local_size_x = {LOCAL_SIZE}) in;
layout(std430, binding = 0) readonly buffer AS {{ int as_[]; }};
layout(std430, binding = 1) readonly buffer AE {{ int ae[]; }};
layout(std430, binding = 2) readonly buffer AM {{ int am[]; }};
layout(std430, binding = 3) readonly buffer AZ {{ uint az[]; }};
layout(std430, binding = 4) readonly buffer BS {{ int bs[]; }};
layout(std430, binding = 5) readonly buffer BE {{ int be[]; }};
layout(std430, binding = 6) readonly buffer BM {{ int bm[]; }};
layout(std430, binding = 7) readonly buffer BZ {{ uint bz[]; }};
layout(std430, binding = 8) buffer OS {{ int os_[]; }};
layout(std430, binding = 9) buffer OE {{ int oe[]; }};
layout(std430, binding = 10) buffer OM {{ int om[]; }};
layout(std430, binding = 11) buffer OZ {{ uint oz[]; }};
layout(std430, binding = 12) readonly buffer FRAC {{ int64_t frac[]; }};
layout(std430, binding = 13) readonly buffer FRACHI {{ int64_t frachi[]; }};
layout(std430, binding = 14) readonly buffer COARSE {{ int64_t coarse[]; }};
layout(std430, binding = 15) readonly buffer FINE {{ int64_t fine[]; }};
layout(push_constant) uniform PC {{ uint n; uint p0; uint p1; uint p2; }} pc;
{ladder}
void main()
{{
  uint i = gl_GlobalInvocationID.x;
  if (i >= pc.n) return;

  if (az[i] != 0u && bz[i] != 0u) {{
    os_[i] = 1; oe[i] = T2BIAS; om[i] = 0; oz[i] = 1u;
    return;
  }}
  int64_t m0 = az[i] != 0u ? int64_t(be[i])
              : (bz[i] != 0u ? int64_t(ae[i])
                             : int64_t(ae[i] > be[i] ? ae[i] : be[i]));

  int64_t qa = t2_to_fixed(as_[i], ae[i], am[i], az[i], m0);
  int64_t qb = t2_to_fixed(bs[i], be[i], bm[i], bz[i], m0);
  int64_t acc = qa {op} qb;

  T2 r = t2_from_fixed(acc, m0);
  os_[i] = r.s; oe[i] = r.e; om[i] = r.m; oz[i] = r.z;
}}
"#
    );

    let groups = [div_ceil(out_n, LOCAL_SIZE as usize) as u32, 1, 1];
    Ok((src, bufs, [out_n as u32, 0, 0, 0], groups, [LOCAL_SIZE, 1, 1]))
}

/// T2 MATMUL: block-floating dot products, exactly as `matmul2_mt`.
///
/// Each output scans its terms for `emax`, then accumulates
/// `(base * aa * bb + half) >> 2M` with `base = phi^(emax-ep) * 2^F`.
/// The reference uses i128 here because `base*aa*bb` can reach ~2^73. With
/// `emax` the maximum term exponent, `base` never exceeds `2^F`, so splitting
/// `base` into 12-bit halves keeps every partial product under 2^62 and the
/// result is exact in 64 bits.
fn tmatmul2(ins: &BoundInstr, ctx: &Ctx, tabs: &mut Tables) -> Result<Lowered> {
    if ins.args.len() != 2 {
        return Err(Error::asm(format!("Vulkan: {} expects 2 operands", ins.mn)));
    }
    let m_bits = ctx.t2.expect("t2");
    if m_bits != 24 || ctx.f != 24 {
        return Err(Error::unimplemented(format!(
            "Vulkan: T2 MATMUL is only analysed for M = F = 24 (got M={m_bits}, F={})",
            ctx.f
        )));
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
            return Err(Error::asm("Vulkan: T2 MATMUL operand is not a ladder stream"));
        }
    }
    let ash = ctx.info(a)?.static_shape(&ctx.name(a))?;
    let bsh = ctx.info(b)?.static_shape(&ctx.name(b))?;
    if ash.len() != 2 || bsh.len() != 2 {
        return Err(Error::unimplemented(format!(
            "Vulkan: T2 MATMUL supports rank-2 only (got {:?} x {:?})",
            ash, bsh
        )));
    }
    let (m, k) = (ash[0], ash[1]);
    if bsh[0] != k {
        return Err(Error::asm("Vulkan: T2 MATMUL inner dimensions differ"));
    }
    let n = bsh[1];

    let planes = |ord: usize| -> Result<(usize, usize, usize, usize)> {
        Ok((
            ctx.plane_of(ord, Comp::S)?,
            ctx.plane_of(ord, Comp::E)?,
            ctx.plane_of(ord, Comp::M)?,
            ctx.plane_of(ord, Comp::Z)?,
        ))
    };
    let (a_s, a_e, a_m, a_z) = planes(a)?;
    let (b_s, b_e, b_m, b_z) = planes(b)?;
    let (o_s, o_e, o_m, o_z) = planes(out_ord)?;

    let f = ctx.f;
    let frac = tabs.get("frac", || phi_core::lattice::l_frac_for(f));
    let frachi = tabs.get("frachi", || phi_core::lattice::frac_hi_for(f));
    let coarse = tabs.get("coarse", || phi_core::lattice::l_coarse().to_vec());
    let fine = tabs.get("fine", || phi_core::lattice::l_fine().to_vec());
    let ladder = ladder2_glsl(m_bits, f, true);

    let bufs = vec![
        a_s, a_e, a_m, a_z, b_s, b_e, b_m, b_z, o_s, o_e, o_m, o_z,
        frac, frachi, coarse, fine,
    ];

    let src = format!(
        r#"#version 450
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
layout(local_size_x = 16, local_size_y = 16) in;
layout(std430, binding = 0) readonly buffer AS {{ int as_[]; }};
layout(std430, binding = 1) readonly buffer AE {{ int ae[]; }};
layout(std430, binding = 2) readonly buffer AM {{ int am[]; }};
layout(std430, binding = 3) readonly buffer AZ {{ uint az[]; }};
layout(std430, binding = 4) readonly buffer BS {{ int bs[]; }};
layout(std430, binding = 5) readonly buffer BE {{ int be[]; }};
layout(std430, binding = 6) readonly buffer BM {{ int bm[]; }};
layout(std430, binding = 7) readonly buffer BZ {{ uint bz[]; }};
layout(std430, binding = 8) buffer OS {{ int os_[]; }};
layout(std430, binding = 9) buffer OE {{ int oe[]; }};
layout(std430, binding = 10) buffer OM {{ int om[]; }};
layout(std430, binding = 11) buffer OZ {{ uint oz[]; }};
layout(std430, binding = 12) readonly buffer FRAC {{ int64_t frac[]; }};
layout(std430, binding = 13) readonly buffer FRACHI {{ int64_t frachi[]; }};
layout(std430, binding = 14) readonly buffer COARSE {{ int64_t coarse[]; }};
layout(std430, binding = 15) readonly buffer FINE {{ int64_t fine[]; }};
layout(push_constant) uniform PC {{ uint n; uint p0; uint p1; uint p2; }} pc;
{ladder}
/* exact (base * r + half) >> 2M with base <= 2^24 and r < 2^50 */
uint64_t t2_mul_shift(uint64_t base, uint64_t r)
{{
  uint64_t hi = base >> 12;
  uint64_t lo = base & 4095ul;
  uint64_t A = hi * r;
  uint64_t B = lo * r;
  const int SH = 2 * T2M;
  const int SL = SH - 12;
  uint64_t mask = (uint64_t(1) << uint(SL)) - 1ul;
  uint64_t halfv = uint64_t(1) << uint(SH - 1);
  uint64_t low = ((A & mask) << 12) + B + halfv;
  return (A >> uint(SL)) + (low >> uint(SH));
}}

void main()
{{
  uint M = pc.n;
  uint N = pc.p0;
  uint K = pc.p1;
  uint r = gl_GlobalInvocationID.y;
  uint col = gl_GlobalInvocationID.x;
  if (r >= M || col >= N) return;

  int64_t emax = 0;
  bool any = false;
  for (uint t = 0; t < K; ++t) {{
    uint ai = r * K + t;
    uint bi = t * N + col;
    if (az[ai] != 0u || bz[bi] != 0u) continue;
    int64_t ep = int64_t(ae[ai]) + int64_t(be[bi]) - int64_t(T2BIAS);
    if (!any || ep > emax) {{ emax = ep; any = true; }}
  }}

  uint o = r * N + col;
  if (!any) {{
    os_[o] = 1; oe[o] = T2BIAS; om[o] = 0; oz[o] = 1u;
    return;
  }}

  int64_t scale = int64_t(1) << T2M;
  int64_t acc = int64_t(0);
  for (uint t = 0; t < K; ++t) {{
    uint ai = r * K + t;
    uint bi = t * N + col;
    if (az[ai] != 0u || bz[bi] != 0u) continue;
    int64_t ep = int64_t(ae[ai]) + int64_t(be[bi]) - int64_t(T2BIAS);
    int64_t base = t2_base(emax - ep);
    if (base == 0) continue;
    uint64_t aa = uint64_t(scale + int64_t(am[ai]));
    uint64_t bb = uint64_t(scale + int64_t(bm[bi]));
    uint64_t p = t2_mul_shift(uint64_t(base), aa * bb);
    int sgn = as_[ai] * bs[bi];
    acc += int64_t(sgn) * int64_t(p);
  }}

  T2 res = t2_from_fixed(acc, emax);
  os_[o] = res.s; oe[o] = res.e; om[o] = res.m; oz[o] = res.z;
}}
"#
    );

    let groups = [div_ceil(n, 16) as u32, div_ceil(m, 16) as u32, 1];
    Ok((
        src,
        bufs,
        [m as u32, n as u32, k as u32, 0],
        groups,
        [16, 16, 1],
    ))
}

fn plane_glsl_type(kind: CKind, c: Comp) -> &'static str {
    match c {
        Comp::S | Comp::E | Comp::M => "int",
        Comp::Z => "uint",
        Comp::V => {
            if kind == CKind::I {
                "int64_t"
            } else {
                "float"
            }
        }
    }
}

/// Exact plane move: one baked index table, every plane gathered with it.
fn gather_planes(
    ins: &BoundInstr,
    ctx: &Ctx,
    tabs: &mut Tables,
    out_ord: usize,
    kind: CKind,
    idx: Vec<i64>,
    tag: &str,
) -> Result<Lowered> {
    let in_ord = ctx.ordinal(&ins.args[0])?;
    let comps = comps_for(kind, ctx.t2);
    let out_n = ctx.elems(out_ord)?;
    if idx.len() != out_n {
        return Err(Error::asm(format!(
            "Vulkan: {} index table has {} entries, output has {out_n}",
            ins.mn,
            idx.len()
        )));
    }

    let mut bufs = Vec::new();
    let mut decl = String::new();
    let n = comps.len();
    for (i, &c) in comps.iter().enumerate() {
        let t = plane_glsl_type(kind, c);
        bufs.push(ctx.plane_of(in_ord, c)?);
        decl.push_str(&format!(
            "layout(std430, binding = {i}) readonly buffer I{i} {{ {t} i{i}[]; }};\n"
        ));
    }
    for (i, &c) in comps.iter().enumerate() {
        let t = plane_glsl_type(kind, c);
        bufs.push(ctx.plane_of(out_ord, c)?);
        decl.push_str(&format!(
            "layout(std430, binding = {}) buffer O{i} {{ {t} o{i}[]; }};\n",
            n + i
        ));
    }
    bufs.push(tabs.push(format!("{tag}:{n}:{}", idx.len()), idx));
    decl.push_str(&format!(
        "layout(std430, binding = {}) readonly buffer IDX {{ int64_t ix[]; }};\n",
        2 * n
    ));

    let mut body = String::new();
    for i in 0..n {
        body.push_str(&format!("  o{i}[k] = i{i}[int(ix[k])];\n"));
    }

    let src = format!(
        "#version 450\n#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require\n\
layout(local_size_x = {LOCAL_SIZE}) in;\n{decl}\
layout(push_constant) uniform PC {{ uint n; uint p0; uint p1; uint p2; }} pc;\n\
void main() {{\n  uint k = gl_GlobalInvocationID.x;\n  if (k >= pc.n) return;\n{body}}}\n"
    );

    let groups = [div_ceil(out_n, LOCAL_SIZE as usize) as u32, 1, 1];
    Ok((src, bufs, [out_n as u32, 0, 0, 0], groups, [LOCAL_SIZE, 1, 1]))
}

/// TRANSPOSE: swap the last two dims; exact plane move.
fn transpose(ins: &BoundInstr, ctx: &Ctx, tabs: &mut Tables, out_ord: usize) -> Result<Lowered> {
    if ctx.t2.is_some() {
        return Err(Error::unimplemented(
            "Vulkan: T2 TRANSPOSE is not implemented yet (frozen T is)",
        ));
    }
    let in_ord = ctx.ordinal(&ins.args[0])?;
    let shape = ctx.info(in_ord)?.static_shape(&ctx.name(in_ord))?;
    if shape.len() < 2 {
        return Err(Error::asm("Vulkan: TRANSPOSE needs rank >= 2"));
    }
    let r = shape[shape.len() - 2];
    let c = shape[shape.len() - 1];
    let batch: usize = shape[..shape.len() - 2].iter().product();

    let mut idx = Vec::with_capacity(batch * r * c);
    for b in 0..batch {
        for i in 0..c {
            for j in 0..r {
                idx.push((b * r * c + j * c + i) as i64);
            }
        }
    }
    gather_planes(ins, ctx, tabs, out_ord, CKind::T, idx, "transpose")
}

/// SLICE(x, axis, start, end): end is exclusive, matching the reference.
fn slice(
    ins: &BoundInstr,
    ctx: &Ctx,
    tabs: &mut Tables,
    out_ord: usize,
    kind: CKind,
) -> Result<Lowered> {
    if ctx.t2.is_some() && kind == CKind::T {
        return Err(Error::unimplemented(
            "Vulkan: T2 SLICE is not implemented yet (frozen T is)",
        ));
    }
    let in_ord = ctx.ordinal(&ins.args[0])?;
    let shape = ctx.info(in_ord)?.static_shape(&ctx.name(in_ord))?;
    if ins.args.len() != 4 {
        return Err(Error::asm("Vulkan: SLICE expects axis, start, end"));
    }
    let num = |i: usize, what: &str| -> Result<usize> {
        ins.args[i]
            .parse::<i64>()
            .ok()
            .and_then(|v| if v >= 0 { Some(v as usize) } else { None })
            .ok_or_else(|| Error::asm(format!("Vulkan: SLICE {what} must be a non-negative literal")))
    };
    let axis = num(1, "axis")?;
    let start = num(2, "start")?;
    let end = num(3, "end")?;
    if axis >= shape.len() || start >= end || end > shape[axis] {
        return Err(Error::asm(format!(
            "Vulkan: SLICE axis={axis} start={start} end={end} out of range for {shape:?}"
        )));
    }

    let mut oshape = shape.clone();
    oshape[axis] = end - start;
    let mut istr = vec![1usize; shape.len()];
    for i in (0..shape.len().saturating_sub(1)).rev() {
        istr[i] = istr[i + 1] * shape[i + 1];
    }
    let mut ostr = vec![1usize; oshape.len()];
    for i in (0..oshape.len().saturating_sub(1)).rev() {
        ostr[i] = ostr[i + 1] * oshape[i + 1];
    }

    let out_n: usize = oshape.iter().product();
    let mut idx = Vec::with_capacity(out_n);
    for lin in 0..out_n {
        let mut rem = lin;
        let mut src_lin = 0usize;
        for d in 0..oshape.len() {
            let coord = rem / ostr[d];
            rem %= ostr[d];
            let src = if d == axis { coord + start } else { coord };
            src_lin += src * istr[d];
        }
        idx.push(src_lin as i64);
    }
    gather_planes(ins, ctx, tabs, out_ord, kind, idx, "slice")
}

/// IGET: scalar extraction by literal index (v1 contract).
fn iget(ins: &BoundInstr, ctx: &Ctx, tabs: &mut Tables, out_ord: usize) -> Result<Lowered> {
    let in_ord = ctx.ordinal(&ins.args[0])?;
    let lit: i64 = ins
        .args
        .get(1)
        .and_then(|a| a.parse().ok())
        .ok_or_else(|| Error::asm("Vulkan: IGET needs an integer literal index"))?;
    let n = ctx.elems(in_ord)?;
    if lit < 0 || lit as usize >= n {
        return Err(Error::asm(format!(
            "Vulkan: IGET index {lit} out of range {n}"
        )));
    }
    gather_planes(ins, ctx, tabs, out_ord, CKind::I, vec![lit], "iget")
}

/// TBETA: a constant field of the encoded `beta_b` value.
fn tbeta(ins: &BoundInstr, ctx: &Ctx, out_ord: usize, out_n: usize) -> Result<Lowered> {
    if ctx.t2.is_some() {
        return Err(Error::unimplemented(
            "Vulkan: T2 TBETA is not implemented yet (frozen T is)",
        ));
    }
    let _ = ins;
    let enc = phi_core::lattice::encode(&ndarray::ArrayD::from_elem(
        ndarray::IxDyn(&[1]),
        ctx.beta_b,
    ));
    let s0 = enc.s.iter().next().copied().unwrap_or(1) as i32;
    let e0 = enc.e.iter().next().copied().unwrap_or(BIAS as i32);
    let z0 = enc.z.iter().next().copied().unwrap_or(1) as u32;
    let _ = s0;

    let (o_s, o_e, o_z) = (
        ctx.plane_of(out_ord, Comp::S)?,
        ctx.plane_of(out_ord, Comp::E)?,
        ctx.plane_of(out_ord, Comp::Z)?,
    );
    let src = format!(
        "#version 450\nlayout(local_size_x = {LOCAL_SIZE}) in;\n\
layout(std430, binding = 0) buffer OS {{ int s[]; }};\n\
layout(std430, binding = 1) buffer OE {{ int e[]; }};\n\
layout(std430, binding = 2) buffer OZ {{ uint z[]; }};\n\
layout(push_constant) uniform PC {{ uint n; uint p0; uint p1; uint p2; }} pc;\n\
void main() {{\n  uint k = gl_GlobalInvocationID.x;\n  if (k >= pc.n) return;\n\
  s[k] = {s0}; e[k] = {e0}; z[k] = {z0}u;\n}}\n"
    );
    let groups = [div_ceil(out_n, LOCAL_SIZE as usize) as u32, 1, 1];
    Ok((src, vec![o_s, o_e, o_z], [out_n as u32, 0, 0, 0], groups, [LOCAL_SIZE, 1, 1]))
}

/// TSHIFT: per-row max via the wide bridge, subtract, convert back.
fn tshift(ins: &BoundInstr, ctx: &Ctx, tabs: &mut Tables, out_ord: usize) -> Result<Lowered> {
    if ctx.t2.is_some() {
        return Err(Error::unimplemented(
            "Vulkan: T2 TSHIFT is not implemented yet (frozen T is)",
        ));
    }
    let in_ord = ctx.ordinal(&ins.args[0])?;
    let shape = ctx.info(in_ord)?.static_shape(&ctx.name(in_ord))?;
    let cols = *shape.last().unwrap_or(&1);
    let rows: usize = shape[..shape.len().saturating_sub(1)].iter().product();

    let (a_s, a_e, a_z) = (
        ctx.plane_of(in_ord, Comp::S)?,
        ctx.plane_of(in_ord, Comp::E)?,
        ctx.plane_of(in_ord, Comp::Z)?,
    );
    let (o_s, o_e, o_z) = (
        ctx.plane_of(out_ord, Comp::S)?,
        ctx.plane_of(out_ord, Comp::E)?,
        ctx.plane_of(out_ord, Comp::Z)?,
    );
    let frac = tabs.get("frac", || phi_core::lattice::l_frac().to_vec());
    let frac_hi = tabs.get("frac_hi", || phi_core::lattice::frac_hi_lut().to_vec());
    let coarse = tabs.get("coarse", || phi_core::lattice::l_coarse().to_vec());
    let fine = tabs.get("fine", || phi_core::lattice::l_fine().to_vec());

    let ladder = LADDER_GLSL;
    let src = format!(
        r#"#version 450
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
layout(local_size_x = {LOCAL_SIZE}) in;
layout(std430, binding = 0) readonly buffer AS {{ int as_[]; }};
layout(std430, binding = 1) readonly buffer AE {{ int ae[]; }};
layout(std430, binding = 2) readonly buffer AZ {{ uint az[]; }};
layout(std430, binding = 3) buffer OS {{ int os_[]; }};
layout(std430, binding = 4) buffer OE {{ int oe[]; }};
layout(std430, binding = 5) buffer OZ {{ uint oz[]; }};
layout(std430, binding = 6) readonly buffer FRAC {{ int64_t frac[]; }};
layout(std430, binding = 7) readonly buffer FRACHI {{ int64_t frac_hi[]; }};
layout(std430, binding = 8) readonly buffer COARSE {{ int64_t coarse[]; }};
layout(std430, binding = 9) readonly buffer FINE {{ int64_t fine[]; }};
layout(push_constant) uniform PC {{ uint n; uint p0; uint p1; uint p2; }} pc;
{ladder}
const int WIDE_CAP = 8192;

int64_t to_fixed_wide(int sgn, int ev, uint z, int64_t m0)
{{
  int64_t d = m0 - int64_t(ev);
  int64_t q_lo;
  if (d < int64_t(-WIDE_CAP)) {{
    q_lo = int64_t(sgn) * frac[0];
  }} else {{
    int64_t k = -d;
    if (k < 1) k = 1;
    if (k > int64_t(WIDE_CAP)) k = int64_t(WIDE_CAP);
    q_lo = int64_t(sgn) * frac_hi[int(k - 1)];
  }}
  if (z != 0u) return 0;
  if (d < 0) return q_lo;
  if (d > int64_t(FRAC_CAP)) return 0;
  return int64_t(sgn) * frac[int(d)];
}}

void main()
{{
  uint r = gl_GlobalInvocationID.x;
  if (r >= pc.n) return;
  uint c = pc.p0;
  int64_t m0 = int64_t(pc.p1);

  int64_t mx = int64_t(0);
  bool first = true;
  for (uint j = 0; j < c; ++j) {{
    uint i = r * c + j;
    int64_t q = to_fixed_wide(as_[i], ae[i], az[i], m0);
    if (first || q > mx) {{ mx = q; first = false; }}
  }}

  for (uint j = 0; j < c; ++j) {{
    uint i = r * c + j;
    int64_t q = to_fixed_wide(as_[i], ae[i], az[i], m0);
    Ladder res = from_fixed(q - mx, m0);
    os_[i] = res.s;
    oe[i] = res.e;
    oz[i] = res.z;
  }}
}}
"#
    );

    let groups = [div_ceil(rows, LOCAL_SIZE as usize) as u32, 1, 1];
    Ok((
        src,
        vec![a_s, a_e, a_z, o_s, o_e, o_z, frac, frac_hi, coarse, fine],
        [rows as u32, cols as u32, ctx.m_acc as u32, 0],
        groups,
        [LOCAL_SIZE, 1, 1],
    ))
}

/// SILU = x * sigmoid(x), sigmoid through the frozen sigx/sig tables.
fn silu(ins: &BoundInstr, ctx: &Ctx, tabs: &mut Tables, out_ord: usize) -> Result<Lowered> {
    if ctx.t2.is_some() {
        return Err(Error::unimplemented(
            "Vulkan: T2 SILU is not implemented yet (frozen T is)",
        ));
    }
    let in_ord = ctx.ordinal(&ins.args[0])?;
    let out_n = ctx.elems(out_ord)?;
    if ctx.elems(in_ord)? != out_n {
        return Err(Error::unimplemented("Vulkan: SILU shape mismatch"));
    }

    let (a_s, a_e, a_z) = (
        ctx.plane_of(in_ord, Comp::S)?,
        ctx.plane_of(in_ord, Comp::E)?,
        ctx.plane_of(in_ord, Comp::Z)?,
    );
    let (o_s, o_e, o_z) = (
        ctx.plane_of(out_ord, Comp::S)?,
        ctx.plane_of(out_ord, Comp::E)?,
        ctx.plane_of(out_ord, Comp::Z)?,
    );
    let sigx = tabs.get("sigx", || phi_core::lattice::sigx_lut().to_vec());
    let sig = tabs.get("sig", || phi_core::lattice::sig_lut().to_vec());
    let frac = tabs.get("frac", || phi_core::lattice::l_frac().to_vec());
    let coarse = tabs.get("coarse", || phi_core::lattice::l_coarse().to_vec());
    let fine = tabs.get("fine", || phi_core::lattice::l_fine().to_vec());
    let ladder = LADDER_GLSL;

    let src = format!(
        r#"#version 450
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
layout(local_size_x = {LOCAL_SIZE}) in;
layout(std430, binding = 0) readonly buffer AS {{ int as_[]; }};
layout(std430, binding = 1) readonly buffer AE {{ int ae[]; }};
layout(std430, binding = 2) readonly buffer AZ {{ uint az[]; }};
layout(std430, binding = 3) buffer OS {{ int os_[]; }};
layout(std430, binding = 4) buffer OE {{ int oe[]; }};
layout(std430, binding = 5) buffer OZ {{ uint oz[]; }};
layout(std430, binding = 6) readonly buffer SIGX {{ int64_t sigx[]; }};
layout(std430, binding = 7) readonly buffer SIGT {{ int64_t sigt[]; }};
layout(std430, binding = 8) readonly buffer FRAC {{ int64_t frac[]; }};
layout(std430, binding = 9) readonly buffer COARSE {{ int64_t coarse[]; }};
layout(std430, binding = 10) readonly buffer FINE {{ int64_t fine[]; }};
layout(push_constant) uniform PC {{ uint n; uint p0; uint p1; uint p2; }} pc;
{ladder}
const int64_t SPAN = 262144;

void main()
{{
  uint i = gl_GlobalInvocationID.x;
  if (i >= pc.n) return;

  int e = ae[i] < 0 ? 0 : (ae[i] > 65535 ? 65535 : ae[i]);
  int64_t x14 = (az[i] != 0u) ? int64_t(0) : int64_t(as_[i]) * sigx[e];

  int64_t y14;
  if (x14 < -SPAN) y14 = 0;
  else if (x14 > SPAN) y14 = 16384;
  else y14 = sigt[int(x14 + SPAN)];

  Ladder sg = from_fixed(y14 * 16, BIAS);

  os_[i] = sg.s * as_[i];
  int ee = sg.e + ae[i] - BIAS;
  oe[i] = ee < 0 ? 0 : (ee > 65535 ? 65535 : ee);
  oz[i] = sg.z | az[i];
}}
"#
    );

    let groups = [div_ceil(out_n, LOCAL_SIZE as usize) as u32, 1, 1];
    Ok((
        src,
        vec![a_s, a_e, a_z, o_s, o_e, o_z, sigx, sig, frac, coarse, fine],
        [out_n as u32, 0, 0, 0],
        groups,
        [LOCAL_SIZE, 1, 1],
    ))
}

/// T2I: decode the ladder to an exact integer through the t2i table.
fn t2i(ins: &BoundInstr, ctx: &Ctx, tabs: &mut Tables, out_ord: usize) -> Result<Lowered> {
    let in_ord = ctx.ordinal(&ins.args[0])?;
    let out_n = ctx.elems(out_ord)?;
    if ctx.elems(in_ord)? != out_n {
        return Err(Error::unimplemented("Vulkan: T2I shape mismatch"));
    }
    let (a_s, a_e, a_z) = (
        ctx.plane_of(in_ord, Comp::S)?,
        ctx.plane_of(in_ord, Comp::E)?,
        ctx.plane_of(in_ord, Comp::Z)?,
    );
    let o_p = ctx.plane_of(out_ord, Comp::V)?;
    let lut = tabs.get("t2i", || phi_core::lattice::t2i_lut().to_vec());

    let src = format!(
        r#"#version 450
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
layout(local_size_x = {LOCAL_SIZE}) in;
layout(std430, binding = 0) readonly buffer AS {{ int as_[]; }};
layout(std430, binding = 1) readonly buffer AE {{ int ae[]; }};
layout(std430, binding = 2) readonly buffer AZ {{ uint az[]; }};
layout(std430, binding = 3) readonly buffer LUT {{ int64_t lut[]; }};
layout(std430, binding = 4) buffer O {{ int64_t o[]; }};
layout(push_constant) uniform PC {{ uint n; uint p0; uint p1; uint p2; }} pc;
void main() {{
  uint i = gl_GlobalInvocationID.x;
  if (i >= pc.n) return;
  if (az[i] != 0u) {{ o[i] = int64_t(0); return; }}
  int e = ae[i] < 0 ? 0 : (ae[i] > 65535 ? 65535 : ae[i]);
  int64_t mag = lut[e];
  o[i] = as_[i] < 0 ? -mag : (as_[i] == 0 ? int64_t(0) : mag);
}}
"#
    );

    let groups = [div_ceil(out_n, LOCAL_SIZE as usize) as u32, 1, 1];
    Ok((src, vec![a_s, a_e, a_z, lut, o_p], [out_n as u32, 0, 0, 0], groups, [LOCAL_SIZE, 1, 1]))
}

/// SOFTMAX_WIDE: wide bridge at BIAS, exp table, truncating normalisation.
fn softmax_inner(
    ins: &BoundInstr,
    ctx: &Ctx,
    tabs: &mut Tables,
    out_ord: usize,
    wide: bool,
) -> Result<Lowered> {
    if ctx.t2.is_some() {
        return Err(Error::unimplemented(
            "Vulkan: T2 SOFTMAX_WIDE is not implemented yet (frozen T is)",
        ));
    }
    let in_ord = ctx.ordinal(&ins.args[0])?;
    let shape = ctx.info(in_ord)?.static_shape(&ctx.name(in_ord))?;
    let cols = *shape.last().unwrap_or(&1);
    let rows: usize = shape[..shape.len().saturating_sub(1)].iter().product();

    let (a_s, a_e, a_z) = (
        ctx.plane_of(in_ord, Comp::S)?,
        ctx.plane_of(in_ord, Comp::E)?,
        ctx.plane_of(in_ord, Comp::Z)?,
    );
    let (o_s, o_e, o_z) = (
        ctx.plane_of(out_ord, Comp::S)?,
        ctx.plane_of(out_ord, Comp::E)?,
        ctx.plane_of(out_ord, Comp::Z)?,
    );
    let wide_c = if wide { 1u32 } else { 0u32 };
    let exp = tabs.get("exp", || phi_core::lattice::exp_lut().to_vec());
    let frac = tabs.get("frac", || phi_core::lattice::l_frac().to_vec());
    let frac_hi = tabs.get("frac_hi", || phi_core::lattice::frac_hi_lut().to_vec());
    let coarse = tabs.get("coarse", || phi_core::lattice::l_coarse().to_vec());
    let fine = tabs.get("fine", || phi_core::lattice::l_fine().to_vec());
    let ladder = LADDER_GLSL;

    let src = format!(
        r#"#version 450
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
layout(local_size_x = {LOCAL_SIZE}) in;
layout(std430, binding = 0) readonly buffer AS {{ int as_[]; }};
layout(std430, binding = 1) readonly buffer AE {{ int ae[]; }};
layout(std430, binding = 2) readonly buffer AZ {{ uint az[]; }};
layout(std430, binding = 3) buffer OS {{ int os_[]; }};
layout(std430, binding = 4) buffer OE {{ int oe[]; }};
layout(std430, binding = 5) buffer OZ {{ uint oz[]; }};
layout(std430, binding = 6) readonly buffer EXP {{ int64_t expo[]; }};
const int USE_WIDE = {wide_c};
layout(std430, binding = 7) readonly buffer FRAC {{ int64_t frac[]; }};
layout(std430, binding = 8) readonly buffer FRACHI {{ int64_t frac_hi[]; }};
layout(std430, binding = 9) readonly buffer COARSE {{ int64_t coarse[]; }};
layout(std430, binding = 10) readonly buffer FINE {{ int64_t fine[]; }};
layout(push_constant) uniform PC {{ uint n; uint p0; uint p1; uint p2; }} pc;
{ladder}
const int WIDE_CAP = 8192;

int64_t to_fixed_wide(int sgn, int ev, uint z, int64_t m0)
{{
  int64_t d = m0 - int64_t(ev);
  int64_t q_lo;
  if (d < int64_t(-WIDE_CAP)) {{
    q_lo = int64_t(sgn) * frac[0];
  }} else {{
    int64_t k = -d;
    if (k < 1) k = 1;
    if (k > int64_t(WIDE_CAP)) k = int64_t(WIDE_CAP);
    q_lo = int64_t(sgn) * frac_hi[int(k - 1)];
  }}
  if (z != 0u) return 0;
  if (d < 0) return q_lo;
  if (d > int64_t(FRAC_CAP)) return 0;
  return int64_t(sgn) * frac[int(d)];
}}

int64_t tdiv64(int64_t a, int64_t b)
{{
  if (b == 0) return 0;
  int64_t q = (a < 0 ? -a : a) / (b < 0 ? -b : b);
  return ((a < 0) != (b < 0)) ? -q : q;
}}

int64_t exp_at(int64_t dd)
{{
  if (dd < 0) dd = 0;
  if (dd > 262144) dd = 262144;
  return expo[int(dd)];
}}

void main()
{{
  uint r = gl_GlobalInvocationID.x;
  if (r >= pc.n) return;
  uint c = pc.p0;

  int64_t vmax = int64_t(0);
  bool first = true;
  for (uint j = 0; j < c; ++j) {{
    uint i = r * c + j;
    int64_t q = USE_WIDE != 0 ? to_fixed_wide(as_[i], ae[i], az[i], int64_t(BIAS)) : to_fixed(as_[i], ae[i], az[i], int64_t(BIAS));
    int64_t s14 = tdiv64(q, int64_t(16));
    if (first || s14 > vmax) {{ vmax = s14; first = false; }}
  }}

  int64_t dsum = int64_t(0);
  for (uint j = 0; j < c; ++j) {{
    uint i = r * c + j;
    int64_t q = USE_WIDE != 0 ? to_fixed_wide(as_[i], ae[i], az[i], int64_t(BIAS)) : to_fixed(as_[i], ae[i], az[i], int64_t(BIAS));
    int64_t s14 = tdiv64(q, int64_t(16));
    dsum += exp_at(vmax - s14);
  }}
  int64_t den = (dsum == 0) ? int64_t(1) : dsum;

  for (uint j = 0; j < c; ++j) {{
    uint i = r * c + j;
    int64_t q = USE_WIDE != 0 ? to_fixed_wide(as_[i], ae[i], az[i], int64_t(BIAS)) : to_fixed(as_[i], ae[i], az[i], int64_t(BIAS));
    int64_t s14 = tdiv64(q, int64_t(16));
    int64_t num = exp_at(vmax - s14);
    int64_t c18 = tdiv64(num * (int64_t(1) << 18), den);
    Ladder res = from_fixed(c18, int64_t(BIAS));
    os_[i] = res.s; oe[i] = res.e; oz[i] = res.z;
  }}
}}
"#
    );

    let groups = [div_ceil(rows, LOCAL_SIZE as usize) as u32, 1, 1];
    Ok((
        src,
        vec![a_s, a_e, a_z, o_s, o_e, o_z, exp, frac, frac_hi, coarse, fine],
        [rows as u32, cols as u32, 0, 0],
        groups,
        [LOCAL_SIZE, 1, 1],
    ))
}

/// RMSNORM over the last axis at m_cov, matching `norm_rows`.
fn rmsnorm(ins: &BoundInstr, ctx: &Ctx, tabs: &mut Tables, out_ord: usize) -> Result<Lowered> {
    if ctx.t2.is_some() {
        return Err(Error::unimplemented(
            "Vulkan: T2 RMSNORM is not implemented yet (frozen T is)",
        ));
    }
    if ins.args.len() != 2 {
        return Err(Error::asm("Vulkan: RMSNORM expects x and w"));
    }
    let x_ord = ctx.ordinal(&ins.args[0])?;
    let w_ord = ctx.ordinal(&ins.args[1])?;
    let shape = ctx.info(x_ord)?.static_shape(&ctx.name(x_ord))?;
    let cols = *shape.last().unwrap_or(&1);
    let rows: usize = shape[..shape.len().saturating_sub(1)].iter().product();
    if ctx.elems(w_ord)? != cols {
        return Err(Error::asm("Vulkan: RMSNORM weight length mismatch"));
    }

    let (x_s, x_e, x_z) = (
        ctx.plane_of(x_ord, Comp::S)?,
        ctx.plane_of(x_ord, Comp::E)?,
        ctx.plane_of(x_ord, Comp::Z)?,
    );
    let (w_s, w_e, w_z) = (
        ctx.plane_of(w_ord, Comp::S)?,
        ctx.plane_of(w_ord, Comp::E)?,
        ctx.plane_of(w_ord, Comp::Z)?,
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

    let src = format!(
        r#"#version 450
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
layout(local_size_x = {LOCAL_SIZE}) in;
layout(std430, binding = 0) readonly buffer XS {{ int xs_[]; }};
layout(std430, binding = 1) readonly buffer XE {{ int xe[]; }};
layout(std430, binding = 2) readonly buffer XZ {{ uint xz[]; }};
layout(std430, binding = 3) readonly buffer WS {{ int ws[]; }};
layout(std430, binding = 4) readonly buffer WE {{ int we[]; }};
layout(std430, binding = 5) readonly buffer WZ {{ uint wz[]; }};
layout(std430, binding = 6) buffer OS {{ int os_[]; }};
layout(std430, binding = 7) buffer OE {{ int oe[]; }};
layout(std430, binding = 8) buffer OZ {{ uint oz[]; }};
layout(std430, binding = 9) readonly buffer FRAC {{ int64_t frac[]; }};
layout(std430, binding = 10) readonly buffer COARSE {{ int64_t coarse[]; }};
layout(std430, binding = 11) readonly buffer FINE {{ int64_t fine[]; }};
layout(push_constant) uniform PC {{ uint n; uint p0; uint p1; uint p2; }} pc;
{ladder}
int64_t tdiv64(int64_t a, int64_t b)
{{
  if (b == 0) return 0;
  int64_t q = (a < 0 ? -a : a) / (b < 0 ? -b : b);
  return ((a < 0) != (b < 0)) ? -q : q;
}}

uint64_t isqrt64(uint64_t n)
{{
  if (n < 2ul) return n;
  uint64_t x = n;
  uint64_t y = (x + 1ul) >> 1;
  while (y < x) {{ x = y; y = (x + n / x) >> 1; }}
  return x;
}}

void main()
{{
  uint r = gl_GlobalInvocationID.x;
  if (r >= pc.n) return;
  uint c = pc.p0;
  int64_t m0 = int64_t(pc.p1);
  int64_t eps = int64_t(pc.p2);

  int64_t sum = int64_t(0);
  for (uint j = 0; j < c; ++j) {{
    int64_t q = to_fixed(xs_[r * c + j], xe[r * c + j], xz[r * c + j], m0);
    sum += q * q;
  }}
  int64_t ms = tdiv64(sum, int64_t(c)) + eps;
  int64_t rms = ms > 0 ? int64_t(isqrt64(uint64_t(ms))) : int64_t(0);

  int64_t one18 = int64_t(1) << 18;
  for (uint j = 0; j < c; ++j) {{
    uint i = r * c + j;
    int64_t q = to_fixed(xs_[i], xe[i], xz[i], m0);
    int64_t norm = (rms == 0) ? int64_t(0) : tdiv64(q * one18, rms);
    int64_t wq = to_fixed(ws[j], we[j], wz[j], m0);
    int64_t v = tdiv64(norm * wq, one18);
    Ladder res = from_fixed(v, m0);
    os_[i] = res.s; oe[i] = res.e; oz[i] = res.z;
  }}
}}
"#
    );

    let groups = [div_ceil(rows, LOCAL_SIZE as usize) as u32, 1, 1];
    Ok((
        src,
        vec![x_s, x_e, x_z, w_s, w_e, w_z, o_s, o_e, o_z, frac, coarse, fine],
        [rows as u32, cols as u32, ctx.m_cov as u32, ctx.eps_rms as u32],
        groups,
        [LOCAL_SIZE, 1, 1],
    ))
}

/// ROTARY: pair-wise rotation with cos/sin tables baked from the payload's
/// position stream, exactly as `rope_tables` + `lattice::encode` do it.
fn rotary(ins: &BoundInstr, ctx: &Ctx, tabs: &mut Tables, out_ord: usize) -> Result<Lowered> {
    if ctx.t2.is_some() {
        return Err(Error::unimplemented(
            "Vulkan: T2 ROTARY is not implemented yet (frozen T is)",
        ));
    }
    if ins.args.len() != 2 {
        return Err(Error::asm("Vulkan: ROTARY expects x and pos"));
    }
    let x_ord = ctx.ordinal(&ins.args[0])?;
    let pos_ord = ctx.ordinal(&ins.args[1])?;
    if ctx.info(pos_ord)?.kind != CKind::I {
        return Err(Error::unimplemented("Vulkan: ROTARY needs an I position stream"));
    }
    let xshape = ctx.info(x_ord)?.static_shape(&ctx.name(x_ord))?;
    if xshape.is_empty() || xshape[xshape.len() - 1] % 2 != 0 {
        return Err(Error::asm("Vulkan: ROTARY head dim must be even"));
    }
    let d = xshape[xshape.len() - 1];
    let half = d / 2;
    let rows: usize = xshape[..xshape.len() - 1].iter().product();

    // positions come from the payload; the table is sized by their maximum
    let pos_name = &ins.args[1];
    let pos_vals: Vec<i64> = match ctx.sample.get(pos_name) {
        Some(crate::value::Value::I64(a)) => a.iter().copied().collect(),
        _ => {
            return Err(Error::unimplemented(format!(
                "Vulkan: ROTARY needs real position values for '{pos_name}' (not a stub)"
            )))
        }
    };
    if pos_vals.is_empty() || pos_vals.iter().any(|p| *p < 0) {
        return Err(Error::asm("Vulkan: ROTARY needs non-negative positions"));
    }
    let broadcast = pos_vals.len() == 1;
    if !broadcast && pos_vals.len() != rows {
        return Err(Error::asm(format!(
            "Vulkan: ROTARY pos length {} not broadcastable to {rows} rows",
            pos_vals.len()
        )));
    }
    let pmax = *pos_vals.iter().max().unwrap() as usize + 1;
    if pmax * half == 0 {
        return Err(Error::asm("Vulkan: ROTARY empty tables"));
    }

    let base = ctx.rope_base;
    let mut cos = vec![0f64; pmax * half];
    let mut sin = vec![0f64; pmax * half];
    for p in 0..pmax {
        for i in 0..half {
            let theta = base.powf(-2.0 * i as f64 / d as f64);
            let ang = p as f64 * theta;
            cos[p * half + i] = ang.cos();
            sin[p * half + i] = ang.sin();
        }
    }
    let enc = |v: Vec<f64>| -> Result<(Vec<i64>, Vec<i64>, Vec<i64>)> {
        let arr = ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&[pmax, half]), v)
            .map_err(|e| Error::asm(format!("Vulkan: ROTARY table shape: {e}")))?;
        let t = phi_core::lattice::encode(&arr);
        Ok((
            t.s.iter().map(|&x| x as i64).collect(),
            t.e.iter().map(|&x| x as i64).collect(),
            t.z.iter().map(|&x| x as i64).collect(),
        ))
    };
    let (cs, ce, cz) = enc(cos)?;
    let (ss, se, sz) = enc(sin)?;

    let (x_s, x_e, x_z) = (
        ctx.plane_of(x_ord, Comp::S)?,
        ctx.plane_of(x_ord, Comp::E)?,
        ctx.plane_of(x_ord, Comp::Z)?,
    );
    let (o_s, o_e, o_z) = (
        ctx.plane_of(out_ord, Comp::S)?,
        ctx.plane_of(out_ord, Comp::E)?,
        ctx.plane_of(out_ord, Comp::Z)?,
    );
    let pos_p = ctx.plane_of(pos_ord, Comp::V)?;

    // tables are per-instance, so give them unique names
    let tag = format!("rotary{}x{}", rows, d);
    let cs_p = tabs.push(format!("{tag}:cos:s"), cs);
    let ce_p = tabs.push(format!("{tag}:cos:e"), ce);
    let cz_p = tabs.push(format!("{tag}:cos:z"), cz);
    let ss_p = tabs.push(format!("{tag}:sin:s"), ss);
    let se_p = tabs.push(format!("{tag}:sin:e"), se);
    let sz_p = tabs.push(format!("{tag}:sin:z"), sz);
    let frac = tabs.get("frac", || phi_core::lattice::l_frac().to_vec());
    let coarse = tabs.get("coarse", || phi_core::lattice::l_coarse().to_vec());
    let fine = tabs.get("fine", || phi_core::lattice::l_fine().to_vec());
    let ladder = LADDER_GLSL;

    let pos_expr = if broadcast {
        "int64_t(pos[0])".to_string()
    } else {
        "int64_t(pos[row])".to_string()
    };

    let src = format!(
        r#"#version 450
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
layout(local_size_x = {LOCAL_SIZE}) in;
layout(std430, binding = 0) readonly buffer XS {{ int xs_[]; }};
layout(std430, binding = 1) readonly buffer XE {{ int xe[]; }};
layout(std430, binding = 2) readonly buffer XZ {{ uint xz[]; }};
layout(std430, binding = 3) buffer OS {{ int os_[]; }};
layout(std430, binding = 4) buffer OE {{ int oe[]; }};
layout(std430, binding = 5) buffer OZ {{ uint oz[]; }};
layout(std430, binding = 6) readonly buffer CS {{ int64_t cs[]; }};
layout(std430, binding = 7) readonly buffer CE {{ int64_t ce[]; }};
layout(std430, binding = 8) readonly buffer CZ {{ int64_t cz[]; }};
layout(std430, binding = 9) readonly buffer SS {{ int64_t ss[]; }};
layout(std430, binding = 10) readonly buffer SE {{ int64_t se[]; }};
layout(std430, binding = 11) readonly buffer SZ {{ int64_t sz[]; }};
layout(std430, binding = 12) readonly buffer POS {{ int64_t pos[]; }};
layout(std430, binding = 13) readonly buffer FRAC {{ int64_t frac[]; }};
layout(std430, binding = 14) readonly buffer COARSE {{ int64_t coarse[]; }};
layout(std430, binding = 15) readonly buffer FINE {{ int64_t fine[]; }};
layout(push_constant) uniform PC {{ uint n; uint p0; uint p1; uint p2; }} pc;
{ladder}
struct Lad {{ int s; int e; uint z; }};

Lad lad_mul(Lad a, Lad b)
{{
  Lad r;
  r.s = a.s * b.s;
  int e = a.e + b.e - BIAS;
  r.e = e < 0 ? 0 : (e > NLVL - 1 ? NLVL - 1 : e);
  r.z = a.z | b.z;
  return r;
}}

Lad lad_bin(Lad a, Lad b, int64_t m0, bool sub)
{{
  int64_t qa = to_fixed(a.s, a.e, a.z, m0);
  int64_t qb = to_fixed(b.s, b.e, b.z, m0);
  int64_t q = sub ? (qa - qb) : (qa + qb);
  Ladder r = from_fixed(q, m0);
  Lad o; o.s = r.s; o.e = r.e; o.z = r.z;
  return o;
}}

void main()
{{
  uint k = gl_GlobalInvocationID.x;
  uint total = pc.n * pc.p1;
  if (k >= total) return;
  uint hf = pc.p1;
  uint d = pc.p0;
  int64_t m0 = int64_t(pc.p2);
  uint row = k / hf;
  uint j = k - row * hf;

  int64_t pp = {pos_expr};
  int co = int(pp * int64_t(hf) + int64_t(j));

  uint i0 = row * d + 2u * j;
  uint i1 = i0 + 1u;

  Lad a0; a0.s = xs_[i0]; a0.e = xe[i0]; a0.z = xz[i0];
  Lad a1; a1.s = xs_[i1]; a1.e = xe[i1]; a1.z = xz[i1];
  Lad c;  c.s = int(cs[co]);  c.e = int(ce[co]);  c.z = uint(cz[co]);
  Lad s;  s.s = int(ss[co]);  s.e = int(se[co]);  s.z = uint(sz[co]);

  Lad y0 = lad_bin(lad_mul(a0, c), lad_mul(a1, s), m0, true);
  Lad y1 = lad_bin(lad_mul(a0, s), lad_mul(a1, c), m0, false);

  os_[i0] = y0.s; oe[i0] = y0.e; oz[i0] = y0.z;
  os_[i1] = y1.s; oe[i1] = y1.e; oz[i1] = y1.z;
}}
"#
    );

    let total = rows * half;
    let groups = [div_ceil(total, LOCAL_SIZE as usize) as u32, 1, 1];
    Ok((
        src,
        vec![
            x_s, x_e, x_z, o_s, o_e, o_z, cs_p, ce_p, cz_p, ss_p, se_p, sz_p,
            pos_p, frac, coarse, fine,
        ],
        [rows as u32, d as u32, half as u32, ctx.m_cov as u32],
        groups,
        [LOCAL_SIZE, 1, 1],
    ))
}

/// GELU: sigx table, erf table, multiply and truncating divide, from_fixed.
fn gelu(ins: &BoundInstr, ctx: &Ctx, tabs: &mut Tables, out_ord: usize) -> Result<Lowered> {
    if ctx.t2.is_some() {
        return Err(Error::unimplemented(
            "Vulkan: T2 GELU is not implemented yet (frozen T is)",
        ));
    }
    let in_ord = ctx.ordinal(&ins.args[0])?;
    let out_n = ctx.elems(out_ord)?;
    if ctx.elems(in_ord)? != out_n {
        return Err(Error::unimplemented("Vulkan: GELU shape mismatch"));
    }
    let (x_s, x_e, x_z) = (
        ctx.plane_of(in_ord, Comp::S)?,
        ctx.plane_of(in_ord, Comp::E)?,
        ctx.plane_of(in_ord, Comp::Z)?,
    );
    let (o_s, o_e, o_z) = (
        ctx.plane_of(out_ord, Comp::S)?,
        ctx.plane_of(out_ord, Comp::E)?,
        ctx.plane_of(out_ord, Comp::Z)?,
    );
    let sigx = tabs.get("sigx", || phi_core::lattice::sigx_lut().to_vec());
    let phi = tabs.get("phi", || phi_core::lattice::phi_lut().to_vec());
    let frac = tabs.get("frac", || phi_core::lattice::l_frac().to_vec());
    let coarse = tabs.get("coarse", || phi_core::lattice::l_coarse().to_vec());
    let fine = tabs.get("fine", || phi_core::lattice::l_fine().to_vec());
    let ladder = LADDER_GLSL;

    let src = format!(
        r#"#version 450
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
layout(local_size_x = {LOCAL_SIZE}) in;
layout(std430, binding = 0) readonly buffer XS {{ int xs_[]; }};
layout(std430, binding = 1) readonly buffer XE {{ int xe[]; }};
layout(std430, binding = 2) readonly buffer XZ {{ uint xz[]; }};
layout(std430, binding = 3) buffer OS {{ int os_[]; }};
layout(std430, binding = 4) buffer OE {{ int oe[]; }};
layout(std430, binding = 5) buffer OZ {{ uint oz[]; }};
layout(std430, binding = 6) readonly buffer SIGX {{ int64_t sigx[]; }};
layout(std430, binding = 7) readonly buffer PHI {{ int64_t phi14[]; }};
layout(std430, binding = 8) readonly buffer FRAC {{ int64_t frac[]; }};
layout(std430, binding = 9) readonly buffer COARSE {{ int64_t coarse[]; }};
layout(std430, binding = 10) readonly buffer FINE {{ int64_t fine[]; }};
layout(push_constant) uniform PC {{ uint n; uint p0; uint p1; uint p2; }} pc;
{ladder}
const int64_t SPAN = 262144;

int64_t tdiv64(int64_t a, int64_t b)
{{
  if (b == 0) return 0;
  int64_t q = (a < 0 ? -a : a) / (b < 0 ? -b : b);
  return ((a < 0) != (b < 0)) ? -q : q;
}}

void main()
{{
  uint i = gl_GlobalInvocationID.x;
  if (i >= pc.n) return;
  int e = xe[i] < 0 ? 0 : (xe[i] > 65535 ? 65535 : xe[i]);
  int64_t x14 = (xz[i] != 0u) ? int64_t(0) : int64_t(xs_[i]) * sigx[e];

  int64_t p14;
  if (x14 < -SPAN) p14 = 0;
  else if (x14 > SPAN) p14 = 16384;
  else p14 = phi14[int(x14 + SPAN)];

  int64_t y14 = tdiv64(x14 * p14, int64_t(16384));
  Ladder res = from_fixed(y14 * 16, int64_t(BIAS));
  os_[i] = res.s; oe[i] = res.e; oz[i] = res.z;
}}
"#
    );

    let groups = [div_ceil(out_n, LOCAL_SIZE as usize) as u32, 1, 1];
    Ok((
        src,
        vec![x_s, x_e, x_z, o_s, o_e, o_z, sigx, phi, frac, coarse, fine],
        [out_n as u32, 0, 0, 0],
        groups,
        [LOCAL_SIZE, 1, 1],
    ))
}

/// CONCAT: join two streams along one axis using a selector table.
fn concat(
    ins: &BoundInstr,
    ctx: &Ctx,
    tabs: &mut Tables,
    out_ord: usize,
    kind: CKind,
) -> Result<Lowered> {
    if ins.args.len() != 3 {
        return Err(Error::asm("Vulkan: CONCAT expects a, b, axis"));
    }
    let a_ord = ctx.ordinal(&ins.args[0])?;
    let b_ord = ctx.ordinal(&ins.args[1])?;
    let axis: usize = ins
        .args[2]
        .parse::<i64>()
        .ok()
        .and_then(|v| if v >= 0 { Some(v as usize) } else { None })
        .ok_or_else(|| Error::asm("Vulkan: CONCAT axis must be a non-negative literal"))?;
    let ash = ctx.info(a_ord)?.static_shape(&ctx.name(a_ord))?;
    let bsh = ctx.info(b_ord)?.static_shape(&ctx.name(b_ord))?;
    if ash.len() != bsh.len() || axis >= ash.len() {
        return Err(Error::asm("Vulkan: CONCAT rank/axis mismatch"));
    }
    for d in 0..ash.len() {
        if d != axis && ash[d] != bsh[d] {
            return Err(Error::asm("Vulkan: CONCAT shapes differ off-axis"));
        }
    }
    let mut oshape = ash.clone();
    oshape[axis] = ash[axis] + bsh[axis];

    let strides = |sh: &[usize]| -> Vec<usize> {
        let mut st = vec![1usize; sh.len()];
        for i in (0..sh.len().saturating_sub(1)).rev() {
            st[i] = st[i + 1] * sh[i + 1];
        }
        st
    };
    let ast = strides(&ash);
    let bst = strides(&bsh);
    let ost = strides(&oshape);

    let out_n = ctx.elems(out_ord)?;
    let mut sel = Vec::with_capacity(out_n);
    for lin in 0..out_n {
        let mut rem = lin;
        let mut a_idx = 0usize;
        let mut b_idx = 0usize;
        let mut from_b = false;
        for d in 0..oshape.len() {
            let coord = rem / ost[d];
            rem %= ost[d];
            if d == axis {
                if coord < ash[axis] {
                    a_idx += coord * ast[d];
                } else {
                    from_b = true;
                    b_idx += (coord - ash[axis]) * bst[d];
                }
            } else {
                a_idx += coord * ast[d];
                b_idx += coord * bst[d];
            }
        }
        sel.push(if from_b { -1 - (b_idx as i64) } else { a_idx as i64 });
    }

    let comps = comps_for(kind, ctx.t2);
    let n = comps.len();
    let mut bufs = Vec::new();
    let mut decl = String::new();
    for (i, &c) in comps.iter().enumerate() {
        let t = plane_glsl_type(kind, c);
        bufs.push(ctx.plane_of(a_ord, c)?);
        decl.push_str(&format!(
            "layout(std430, binding = {i}) readonly buffer A{i} {{ {t} a{i}[]; }};\n"
        ));
    }
    for (i, &c) in comps.iter().enumerate() {
        let t = plane_glsl_type(kind, c);
        bufs.push(ctx.plane_of(b_ord, c)?);
        decl.push_str(&format!(
            "layout(std430, binding = {}) readonly buffer B{i} {{ {t} b{i}[]; }};\n",
            n + i
        ));
    }
    for (i, &c) in comps.iter().enumerate() {
        let t = plane_glsl_type(kind, c);
        bufs.push(ctx.plane_of(out_ord, c)?);
        decl.push_str(&format!(
            "layout(std430, binding = {}) buffer O{i} {{ {t} o{i}[]; }};\n",
            2 * n + i
        ));
    }
    bufs.push(tabs.push(format!("concat:{}:{out_n}", ash.len()), sel));
    decl.push_str(&format!(
        "layout(std430, binding = {}) readonly buffer SEL {{ int64_t sel[]; }};\n",
        3 * n
    ));

    let mut body = String::new();
    for i in 0..n {
        body.push_str(&format!(
            "  if (v >= 0) o{i}[k] = a{i}[int(v)]; else o{i}[k] = b{i}[int(-1 - v)];\n"
        ));
    }
    let src = format!(
        "#version 450\n#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require\n\
layout(local_size_x = {LOCAL_SIZE}) in;\n{decl}\
layout(push_constant) uniform PC {{ uint n; uint p0; uint p1; uint p2; }} pc;\n\
void main() {{\n  uint k = gl_GlobalInvocationID.x;\n  if (k >= pc.n) return;\n\
  int64_t v = sel[k];\n{body}}}\n"
    );

    let groups = [div_ceil(out_n, LOCAL_SIZE as usize) as u32, 1, 1];
    Ok((src, bufs, [out_n as u32, 0, 0, 0], groups, [LOCAL_SIZE, 1, 1]))
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
            Comp::M => ".m",
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
        for &comp in comps_for(ctx.info(ord)?.kind, ctx.t2) {
            in_pairs.push((ctx.plane_of(ord, comp)?, argv));
            argv += 1;
        }
    }
    let mut out_pairs: Vec<(usize, usize)> = Vec::new();
    for name in outputs {
        let ord = ctx.ordinal(name)?;
        for &comp in comps_for(ctx.info(ord)?.kind, ctx.t2) {
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
            comps_for(ctx.info(ord)?.kind, ctx.t2)
                .iter()
                .map(|&c| ctx.plane_of(ord, c))
                .collect::<Result<_>>()?
        }
        None => Vec::new(),
    };
    let serve_out_planes: Vec<usize> = match serve_out {
        Some(n) => {
            let ord = ctx.ordinal(n)?;
            comps_for(ctx.info(ord)?.kind, ctx.t2)
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
static VkBuffer g_arena;
static VkDeviceMemory g_arena_mem;
static unsigned char *g_arena_map;
static VkDeviceSize g_off[N_BUF];
static VkDeviceSize g_size[N_BUF];
static VkDeviceSize g_align = 1;
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
  {
    VkPhysicalDeviceProperties pr;
    vkGetPhysicalDeviceProperties(g_phys, &pr);
    g_align = pr.limits.minStorageBufferOffsetAlignment;
    if (g_align == 0) g_align = 1;
  }

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
  VkDeviceSize total = 0;
  for (int i = 0; i < N_BUF; ++i) {
    g_size[i] = buf_bytes(i);
    if (g_size[i] == 0) g_size[i] = 4;
    g_off[i] = (total + g_align - 1) / g_align * g_align;
    total = g_off[i] + g_size[i];
  }

  VkBufferCreateInfo bi = { .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
  bi.size = total;
  bi.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
  bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  CK(vkCreateBuffer(g_dev, &bi, NULL, &g_arena));

  VkMemoryRequirements mr;
  vkGetBufferMemoryRequirements(g_dev, g_arena, &mr);
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
  CK(vkAllocateMemory(g_dev, &ma, NULL, &g_arena_mem));
  CK(vkBindBufferMemory(g_dev, g_arena, g_arena_mem, 0));
  CK(vkMapMemory(g_dev, g_arena_mem, 0, VK_WHOLE_SIZE, 0, (void **)&g_arena_map));
  memset(g_arena_map, 0, (size_t)total);

  for (int i = N_PLANE; i < N_BUF; ++i)
    memcpy(g_arena_map + g_off[i], kTablePtr[i - N_PLANE], (size_t)g_size[i]);
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
  unsigned char *dst = g_arena_map + g_off[plane];

  if (elem == E_I32 || elem == E_I64 || elem == E_F32) {
    size_t got = fread(dst, 1, want, f);
    if (got == 0 && want > 0) return 1;
    return got == want ? 0 : 2;
  }

  unsigned char *tmp = malloc(want ? want : 1);
  if (!tmp) return 2;
  size_t got = fread(tmp, 1, want, f);
  if (got == 0 && want > 0) { free(tmp); return 1; }
  if (got != want) { free(tmp); return 2; }
  for (uint64_t j = 0; j < n; ++j) {
    if (elem == E_I8) ((int32_t *)dst)[j] = (int32_t)((int8_t *)tmp)[j];
    else              ((uint32_t *)dst)[j] = (uint32_t)((uint8_t *)tmp)[j];
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
  const unsigned char *src = g_arena_map + g_off[plane];

  if (elem == E_I32 || elem == E_I64 || elem == E_F32)
    return fwrite(src, 1, want, f) == want ? 0 : 6;

  unsigned char *tmp = malloc(want ? want : 1);
  if (!tmp) return 6;
  for (uint64_t j = 0; j < n; ++j) {
    if (elem == E_I8) ((int8_t *)tmp)[j] = (int8_t)((int32_t *)src)[j];
    else              ((uint8_t *)tmp)[j] = (uint8_t)((uint32_t *)src)[j];
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
      VkDescriptorBufferInfo bi = { .buffer = g_arena,
                                    .offset = g_off[dp->buf[k]],
                                    .range = g_size[dp->buf[k]] };
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
