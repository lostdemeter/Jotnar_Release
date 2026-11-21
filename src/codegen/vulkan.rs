//! Vulkan compute backend (correctness-first).
//!
//! Numeric model, forced by the hardware this targets (Adreno 750 / Turnip):
//!   - F streams are **f32**: `shaderFloat64` is false, so there is no f64
//!     path and no silent promotion.
//!   - I streams are **i64**: `shaderInt64` is true, so integer work stays
//!     exact instead of being silently narrowed to 32 bits.
//!   - T streams are refused (decode/quantize first), same as CUDA.
//!
//! Shape of the emitted program: one compute dispatch per instruction, a
//! global memory barrier between dispatches, one command buffer, one submit,
//! one wait. Buffers are host-visible/coherent, so there are no staging
//! copies. This is a correctness backend; scheduling and memory placement are
//! later milestones.
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
    for (name, s) in &streams {
        if s.kind == CKind::T {
            return Err(Error::unimplemented(format!(
                "Vulkan: T stream {name:?} is not supported (decode/quantize first)"
            )));
        }
    }
    for o in &outputs {
        if !streams.contains_key(o) {
            return Err(Error::asm(format!("Vulkan: output '{o}' is not a stream")));
        }
    }
    let in_names: Vec<String> = bound.inputs.iter().map(|d| d.name.clone()).collect();

    // Buffer table: inputs first, then every instruction result in order.
    let mut order: Vec<String> = Vec::new();
    let mut index: HashMap<String, usize> = HashMap::new();
    let add = |name: &str, order: &mut Vec<String>, index: &mut HashMap<String, usize>| {
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

    let mut dispatches: Vec<Dispatch> = Vec::new();
    let mut shader_srcs: Vec<String> = Vec::new();
    let mut shader_ids: HashMap<String, usize> = HashMap::new();

    for ins in &bound.instrs {
        let (glsl, buffers, params, groups, local) =
            lower(ins, &streams, &index)?;

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

    let source = emit_c(&order, &streams, &dispatches, &spirv, &in_names, &outputs)?;
    let n_argv = in_names.len() + outputs.len();

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

fn elems(s: &StreamInfo, name: &str) -> Result<usize> {
    s.elems()
        .ok_or_else(|| Error::unimplemented(format!("Vulkan: dynamic shape in {name}")))
}

fn operand(
    arg: &str,
    streams: &HashMap<String, StreamInfo>,
    index: &HashMap<String, usize>,
) -> Result<Operand> {
    if let Some(&i) = index.get(arg) {
        if streams.contains_key(arg) {
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

fn stream_name(index: &HashMap<String, usize>, i: usize) -> Option<String> {
    index.iter().find(|(_, v)| **v == i).map(|(k, _)| k.clone())
}

fn lower(ins: &BoundInstr, streams: &HashMap<String, StreamInfo>, index: &HashMap<String, usize>) -> Result<Lowered> {
    let out = ins
        .outs
        .first()
        .ok_or_else(|| Error::asm(format!("Vulkan: {} has no output", ins.mn)))?;
    let out_idx = index[out];
    let out_n = elems(&streams[out], out)?;

    match ins.mn.as_str() {
        "ADD" => binary(ins, streams, index, out_idx, out_n, CKind::F, "+"),
        "SUB" => binary(ins, streams, index, out_idx, out_n, CKind::F, "-"),
        "MUL" => binary(ins, streams, index, out_idx, out_n, CKind::F, "*"),
        "DIV" => binary(ins, streams, index, out_idx, out_n, CKind::F, "/"),
        "IADD" => binary(ins, streams, index, out_idx, out_n, CKind::I, "+"),
        "ISUB" => binary(ins, streams, index, out_idx, out_n, CKind::I, "-"),
        "IMUL" => binary(ins, streams, index, out_idx, out_n, CKind::I, "*"),
        "RELU" => unary(ins, streams, index, out_idx, out_n, "max(a[i], 0.0)"),
        "SIGMOID" => unary(
            ins,
            streams,
            index,
            out_idx,
            out_n,
            "1.0 / (1.0 + exp(-a[i]))",
        ),
        "SQRT" => unary(ins, streams, index, out_idx, out_n, "sqrt(a[i])"),
        "SQUARE" => unary(ins, streams, index, out_idx, out_n, "a[i] * a[i]"),
        "MATMUL" => matmul(ins, streams, index, CKind::F),
        "IMATMUL" => matmul(ins, streams, index, CKind::I),
        other => Err(Error::unimplemented(format!(
            "Vulkan backend: no lowering for {other} yet"
        ))),
    }
}

fn check_operand(
    arg: &str,
    o: &Operand,
    ins: &BoundInstr,
    streams: &HashMap<String, StreamInfo>,
    index: &HashMap<String, usize>,
    out_n: usize,
    kind: CKind,
) -> Result<()> {
    if let Operand::Stream(i) = o {
        if let Some(name) = stream_name(index, *i) {
            let s = &streams[&name];
            let n = elems(s, &name)?;
            if n != out_n {
                return Err(Error::unimplemented(format!(
                    "Vulkan: {} with mismatched element counts ({name}={n}, out={out_n}); \
                     broadcasting is not implemented yet",
                    ins.mn
                )));
            }
            if s.kind != kind {
                return Err(Error::asm(format!(
                    "Vulkan: {} mixes stream kinds at {name}",
                    ins.mn
                )));
            }
        }
    }
    let _ = arg;
    Ok(())
}

fn ty_of(kind: CKind) -> &'static str {
    if kind == CKind::I {
        "int64_t"
    } else {
        "float"
    }
}

fn ext_of(kind: CKind) -> &'static str {
    if kind == CKind::I {
        "#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require\n"
    } else {
        ""
    }
}

fn literal(v: f64, kind: CKind) -> String {
    if kind == CKind::I {
        format!("int64_t({}L)", v as i64)
    } else {
        format!("{v:?}")
    }
}

fn binary(
    ins: &BoundInstr,
    streams: &HashMap<String, StreamInfo>,
    index: &HashMap<String, usize>,
    out_idx: usize,
    out_n: usize,
    kind: CKind,
    op: &str,
) -> Result<Lowered> {
    if ins.args.len() != 2 {
        return Err(Error::asm(format!("Vulkan: {} expects 2 operands", ins.mn)));
    }
    let a = operand(&ins.args[0], streams, index)?;
    let b = operand(&ins.args[1], streams, index)?;
    check_operand(&ins.args[0], &a, ins, streams, index, out_n, kind)?;
    check_operand(&ins.args[1], &b, ins, streams, index, out_n, kind)?;

    let ty = ty_of(kind);
    let mut decl = String::new();
    let mut bufs = Vec::new();
    let mut next = 0usize;

    match &a {
        Operand::Stream(i) => {
            decl.push_str(&format!(
                "layout(std430, binding = {next}) readonly buffer InA {{ {ty} a[]; }};\n"
            ));
            bufs.push(*i);
            next += 1;
        }
        Operand::Lit(_) => {}
    }
    match &b {
        Operand::Stream(i) => {
            decl.push_str(&format!(
                "layout(std430, binding = {next}) readonly buffer InB {{ {ty} b[]; }};\n"
            ));
            bufs.push(*i);
            next += 1;
        }
        Operand::Lit(_) => {}
    }
    decl.push_str(&format!(
        "layout(std430, binding = {next}) buffer Out {{ {ty} c[]; }};\n"
    ));
    bufs.push(out_idx);

    let lhs = match &a {
        Operand::Stream(_) => "a[i]".to_string(),
        Operand::Lit(v) => literal(*v, kind),
    };
    let rhs = match &b {
        Operand::Stream(_) => "b[i]".to_string(),
        Operand::Lit(v) => literal(*v, kind),
    };

    let src = format!(
        "#version 450\n{ext}layout(local_size_x = {LOCAL_SIZE}) in;\n{decl}\
layout(push_constant) uniform PC {{ uint n; uint p0; uint p1; uint p2; }} pc;\n\
void main() {{\n  uint i = gl_GlobalInvocationID.x;\n  if (i >= pc.n) return;\n  c[i] = {lhs} {op} {rhs};\n}}\n",
        ext = ext_of(kind),
        decl = decl,
        lhs = lhs,
        op = op,
        rhs = rhs,
    );

    let groups = [div_ceil(out_n, LOCAL_SIZE as usize) as u32, 1, 1];
    Ok((src, bufs, [out_n as u32, 0, 0, 0], groups, [LOCAL_SIZE, 1, 1]))
}

fn matmul(
    ins: &BoundInstr,
    streams: &HashMap<String, StreamInfo>,
    index: &HashMap<String, usize>,
    kind: CKind,
) -> Result<Lowered> {
    if ins.args.len() != 2 {
        return Err(Error::asm(format!("Vulkan: {} expects 2 operands", ins.mn)));
    }
    let a_name = &ins.args[0];
    let b_name = &ins.args[1];
    let a_idx = *index
        .get(a_name)
        .ok_or_else(|| Error::asm(format!("Vulkan: {} lhs is not a stream", ins.mn)))?;
    let b_idx = *index
        .get(b_name)
        .ok_or_else(|| Error::asm(format!("Vulkan: {} rhs is not a stream", ins.mn)))?;
    let out = ins
        .outs
        .first()
        .ok_or_else(|| Error::asm(format!("Vulkan: {} has no output", ins.mn)))?;
    let out_idx = index[out];

    for n in [a_name, b_name] {
        let k = streams[n].kind;
        if k != kind {
            return Err(Error::asm(format!(
                "Vulkan: {} operand {n} has the wrong kind",
                ins.mn
            )));
        }
    }
    let a_sh = streams[a_name].static_shape(a_name)?;
    let b_sh = streams[b_name].static_shape(b_name)?;
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

    let ty = ty_of(kind);
    let ext = ext_of(kind);
    let decl = format!(
        "layout(std430, binding = 0) readonly buffer A {{ {ty} a[]; }};\n\
layout(std430, binding = 1) readonly buffer B {{ {ty} b[]; }};\n\
layout(std430, binding = 2) buffer C {{ {ty} c[]; }};\n"
    );
    let src = format!(
        "#version 450\n{ext}layout(local_size_x = 16, local_size_y = 16) in;\n{decl}\
layout(push_constant) uniform PC {{ uint n; uint p0; uint p1; uint p2; }} pc;\n\
void main() {{\n\
  uint M = pc.n; uint N = pc.p0; uint K = pc.p1;\n\
  uint r = gl_GlobalInvocationID.y;\n\
  uint col = gl_GlobalInvocationID.x;\n\
  if (r >= M || col >= N) return;\n\
  {ty} acc = {ty}(0);\n\
  for (uint kk = 0; kk < K; ++kk)\n\
    acc = acc + a[r * K + kk] * b[kk * N + col];\n\
  c[r * N + col] = acc;\n\
}}\n"
    );

    let groups = [
        div_ceil(n, 16) as u32,
        div_ceil(m, 16) as u32,
        1,
    ];
    Ok((
        src,
        vec![a_idx, b_idx, out_idx],
        [m as u32, n as u32, k as u32, 0],
        groups,
        [16, 16, 1],
    ))
}

fn unary(
    ins: &BoundInstr,
    streams: &HashMap<String, StreamInfo>,
    index: &HashMap<String, usize>,
    out_idx: usize,
    out_n: usize,
    expr: &str,
) -> Result<Lowered> {
    if ins.args.len() != 1 {
        return Err(Error::asm(format!("Vulkan: {} expects 1 operand", ins.mn)));
    }
    let a = operand(&ins.args[0], streams, index)?;
    check_operand(&ins.args[0], &a, ins, streams, index, out_n, CKind::F)?;

    let ty = "float";
    let mut decl = String::new();
    let mut bufs = Vec::new();
    let mut next = 0usize;
    match &a {
        Operand::Stream(i) => {
            decl.push_str(&format!(
                "layout(std430, binding = {next}) readonly buffer InA {{ {ty} a[]; }};\n"
            ));
            bufs.push(*i);
            next += 1;
        }
        Operand::Lit(_) => {}
    }
    decl.push_str(&format!(
        "layout(std430, binding = {next}) buffer Out {{ {ty} c[]; }};\n"
    ));
    bufs.push(out_idx);

    let src = format!(
        "#version 450\nlayout(local_size_x = {LOCAL_SIZE}) in;\n{decl}\
layout(push_constant) uniform PC {{ uint n; uint p0; uint p1; uint p2; }} pc;\n\
void main() {{\n  uint i = gl_GlobalInvocationID.x;\n  if (i >= pc.n) return;\n  c[i] = {expr};\n}}\n"
    );

    let groups = [div_ceil(out_n, LOCAL_SIZE as usize) as u32, 1, 1];
    Ok((src, bufs, [out_n as u32, 0, 0, 0], groups, [LOCAL_SIZE, 1, 1]))
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
    streams: &HashMap<String, StreamInfo>,
    dispatches: &[Dispatch],
    spirv: &[Vec<u8>],
    inputs: &[String],
    outputs: &[String],
) -> Result<String> {
    let name_of = |i: usize| order[i].as_str();
    let idx = |name: &str| -> Result<usize> {
        order
            .iter()
            .position(|n| n == name)
            .ok_or_else(|| Error::asm(format!("Vulkan: stream '{name}' not in table")))
    };

    let mut c = String::new();
    c.push_str(HOST_PRELUDE);

    c.push_str(&format!("\n#define N_STREAMS {}\n", order.len()));
    c.push_str("static const char *kNames[N_STREAMS] = {\n");
    for n in order {
        c.push_str(&format!("  \"{n}\",\n"));
    }
    c.push_str("};\nstatic const uint64_t kBytes[N_STREAMS] = {\n");
    for n in order {
        let s = &streams[n];
        let width = if s.kind == CKind::I { 8 } else { 4 };
        c.push_str(&format!("  {},\n", elems(s, n)? * width));
    }
    c.push_str("};\nstatic const int kIsI64[N_STREAMS] = {\n");
    for n in order {
        c.push_str(&format!(
            "  {},\n",
            if streams[n].kind == CKind::I { 1 } else { 0 }
        ));
    }
    c.push_str("};\n");

    c.push_str(&format!("\n#define N_SHADER {}\n", spirv.len()));
    c.push_str("static const uint32_t kSpvWords[N_SHADER] = {");
    for bytes in spirv {
        c.push_str(&format!(" {},", bytes.len() / 4));
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

    c.push_str(&format!("\n#define N_IN {}\n", inputs.len()));
    c.push_str("static const int kInStream[N_IN] = {");
    for n in inputs {
        c.push_str(&format!(" {},", idx(n)?));
    }
    c.push_str(" };\nstatic const int kInArgv[N_IN] = {");
    for i in 0..inputs.len() {
        c.push_str(&format!(" {},", i + 1));
    }
    c.push_str(" };\n");

    c.push_str(&format!("\n#define N_OUT {}\n", outputs.len()));
    c.push_str("static const int kOutStream[N_OUT] = {");
    for n in outputs {
        c.push_str(&format!(" {},", idx(n)?));
    }
    c.push_str(" };\nstatic const int kOutArgv[N_OUT] = {");
    for i in 0..outputs.len() {
        c.push_str(&format!(" {},", inputs.len() + i + 1));
    }
    c.push_str(" };\n");

    let _ = name_of;
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
#include <vulkan/vulkan.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CK(x) do { VkResult _r = (x); if (_r != VK_SUCCESS) { \
  fprintf(stderr, "vk error %d at line %d\n", (int)_r, __LINE__); return 7; } } while (0)

typedef struct {
  const uint32_t *code;
  uint32_t words;
  int nbuf;
  int buf[3];
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
static VkBuffer g_buf[N_STREAMS];
static VkDeviceMemory g_mem[N_STREAMS];
static void *g_map[N_STREAMS];
static VkDescriptorSetLayout g_dsl[N_DISPATCH];
static VkPipelineLayout g_pl[N_DISPATCH];
static VkPipeline g_pipe[N_DISPATCH];
static VkDescriptorSet g_ds[N_DISPATCH];

static uint32_t mem_type(uint32_t bits, VkMemoryPropertyFlags want)
{
  VkPhysicalDeviceMemoryProperties mp;
  vkGetPhysicalDeviceMemoryProperties(g_phys, &mp);
  for (uint32_t i = 0; i < mp.memoryTypeCount; ++i)
    if ((bits & (1u << i)) && (mp.memoryTypes[i].propertyFlags & want) == want)
      return i;
  return UINT32_MAX;
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
  for (int i = 0; i < N_STREAMS; ++i) {
    VkDeviceSize size = kBytes[i] ? (VkDeviceSize)kBytes[i] : 4;
    VkBufferCreateInfo bi = { .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
    bi.size = size;
    bi.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    CK(vkCreateBuffer(g_dev, &bi, NULL, &g_buf[i]));
    VkMemoryRequirements mr;
    vkGetBufferMemoryRequirements(g_dev, g_buf[i], &mr);
    uint32_t mt = mem_type(mr.memoryTypeBits,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    if (mt == UINT32_MAX) { fprintf(stderr, "no host-visible memory\n"); return 5; }
    VkMemoryAllocateInfo ma = { .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO };
    ma.allocationSize = mr.size;
    ma.memoryTypeIndex = mt;
    CK(vkAllocateMemory(g_dev, &ma, NULL, &g_mem[i]));
    CK(vkBindBufferMemory(g_dev, g_buf[i], g_mem[i], 0));
    CK(vkMapMemory(g_dev, g_mem[i], 0, VK_WHOLE_SIZE, 0, &g_map[i]));
    memset(g_map[i], 0, (size_t)size);
  }
  return 0;
}

static int load_inputs(int argc, char **argv)
{
  (void)argc;
  for (int i = 0; i < N_IN; ++i) {
    const char *path = argv[kInArgv[i]];
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); return 1; }
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    int s = kInStream[i];
    if ((uint64_t)sz > kBytes[s]) {
      fprintf(stderr, "%s: %ld bytes exceeds stream %s (%llu)\n",
              path, sz, kNames[s], (unsigned long long)kBytes[s]);
      fclose(f);
      return 2;
    }
    if (fread(g_map[s], 1, (size_t)sz, f) != (size_t)sz) { fclose(f); return 2; }
    fclose(f);
  }
  return 0;
}

static int build_pipelines(void)
{
  for (int d = 0; d < N_DISPATCH; ++d) {
    const Dispatch *dp = &kDisp[d];

    VkDescriptorSetLayoutBinding binds[3];
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
  VkCommandBufferAllocateInfo ai = { .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
  ai.commandPool = g_cpool;
  ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  ai.commandBufferCount = 1;
  VkCommandBuffer cmd;
  CK(vkAllocateCommandBuffers(g_dev, &ai, &cmd));

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

static int write_outputs(int argc, char **argv)
{
  (void)argc;
  for (int i = 0; i < N_OUT; ++i) {
    int s = kOutStream[i];
    FILE *f = fopen(argv[kOutArgv[i]], "wb");
    if (!f) { fprintf(stderr, "cannot write %s\n", argv[kOutArgv[i]]); return 6; }
    if (fwrite(g_map[s], 1, (size_t)kBytes[s], f) != (size_t)kBytes[s]) {
      fclose(f);
      return 6;
    }
    fclose(f);
  }
  return 0;
}

int main(int argc, char **argv)
{
  if (argc != N_IN + N_OUT + 1) {
    fprintf(stderr, "usage: needs %d file arguments\n", N_IN + N_OUT);
    return 9;
  }
  int r;
  if ((r = init_vk())) return r;
  if ((r = make_buffers())) return r;
  if ((r = load_inputs(argc, argv))) return r;
  if ((r = build_pipelines())) return r;
  if ((r = run())) return r;
  if ((r = write_outputs(argc, argv))) return r;
  return 0;
}
"#;
