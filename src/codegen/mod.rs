//! Backend drivers. Milestone 1 is the portable C emitter; the CUDA and
//! non-FPU targets plug into the same bound-IR seam.

pub mod c;
pub mod cuda;
pub mod infer;
pub mod nonfpu;
pub mod rust;

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

use crate::asm::{self, AsmOptions};
use crate::error::{Error, Result};
use crate::ir::BoundProgram;
use crate::payload::Feeds;

use c::{pattern, Ctx};
use infer::{infer_shapes, CKind};

const INFLATE_C: &str = include_str!("inflate_c.c");

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Target {
    C,
    NonFpu,
    Cuda,
    Rust,
}

impl Target {
    pub fn parse(s: &str) -> Result<Self> {
        match s {
            "c" => Ok(Target::C),
            "nonfpu" => Ok(Target::NonFpu),
            "cuda" => Ok(Target::Cuda),
            "rust" => Ok(Target::Rust),
            other => Err(Error::unimplemented(format!(
                "no '{other}' backend yet (have: c, nonfpu, cuda, rust)"
            ))),
        }
    }
}

#[derive(Clone, Debug)]
pub struct CompileOptions {
    pub target: Target,
    pub outputs: Vec<String>,
    pub asm: AsmOptions,
    pub use_fp16: bool,
}

impl CompileOptions {
    pub fn c(outputs: Vec<String>, asm: AsmOptions) -> Self {
        CompileOptions {
            target: Target::C,
            outputs,
            asm,
            use_fp16: false,
        }
    }
}

#[derive(Clone, Debug)]
pub struct Artifact {
    pub source: String,
    pub streams: HashMap<String, infer::StreamInfo>,
    pub inputs: Vec<String>,
    pub outputs: Vec<String>,
    pub config: crate::ir::Config,
    pub n_argv: usize,
}

/// Rewrite `name#k.stream` (CALL namespacing) to valid C identifiers.
pub(crate) fn sanitize_cnames(
    bound: &mut BoundProgram,
    outputs: &mut [String],
    sample: &mut Feeds,
) -> Result<()> {
    let mut known: HashSet<String> = bound.inputs.iter().map(|d| d.name.clone()).collect();
    for ins in &bound.instrs {
        known.extend(ins.outs.iter().cloned());
    }
    let mut map: HashMap<String, String> = HashMap::new();
    let mut seen: HashMap<String, String> = HashMap::new();
    for n in &known {
        let c: String = n
            .chars()
            .map(|ch| {
                if ch.is_ascii_alphanumeric() || ch == '_' {
                    ch
                } else {
                    '_'
                }
            })
            .collect();
        let c = if c
            .chars()
            .next()
            .map(|x| x.is_ascii_digit())
            .unwrap_or(false)
        {
            format!("_{c}")
        } else {
            c
        };
        if let Some(prev) = seen.get(&c) {
            if prev != n {
                return Err(Error::asm(format!(
                    "emit: stream names {prev:?} and {n:?} collide as C identifier {c:?}"
                )));
            }
        }
        seen.insert(c.clone(), n.clone());
        map.insert(n.clone(), c);
    }
    let remap = |x: &str| -> String { map.get(x).cloned().unwrap_or_else(|| x.to_string()) };

    for decl in &mut bound.inputs {
        decl.name = remap(&decl.name);
    }
    for ins in &mut bound.instrs {
        for o in &mut ins.outs {
            *o = remap(o);
        }
        for a in &mut ins.args {
            if known.contains(a) {
                *a = remap(a);
            }
        }
    }
    for o in outputs.iter_mut() {
        *o = remap(o);
    }
    let old = std::mem::take(sample);
    let mut next = Feeds::new();
    for (k, v) in old {
        next.insert(remap(&k), v);
    }
    *sample = next;
    Ok(())
}

fn static_dim(s: &infer::StreamInfo, i: usize, where_: &str) -> Result<usize> {
    s.shape
        .get(i)
        .copied()
        .flatten()
        .ok_or_else(|| Error::unimplemented(format!("C backend: dynamic dim {i} in {where_}")))
}

fn count_expr(s: &infer::StreamInfo, where_: &str) -> Result<usize> {
    s.elems()
        .ok_or_else(|| Error::unimplemented(format!("C backend: dynamic shape in {where_}")))
}

fn rows_expr(s: &infer::StreamInfo) -> usize {
    s.rows().unwrap_or(1)
}

fn c_decl_stream(name: &str, s: &infer::StreamInfo, frepr: &str) -> String {
    match s.kind {
        CKind::F => format!("static {frepr} *{name}; static int64_t {name}_N;"),
        CKind::I => format!("static int64_t *{name}; static int64_t {name}_N;"),
        CKind::T => format!(
            "static int8_t *{name}_s; static int32_t *{name}_e; static uint8_t *{name}_z; static int64_t {name}_N;"
        ),
    }
}

fn input_load(name: &str, s: &infer::StreamInfo, idx: usize, frepr: &str) -> Result<String> {
    match s.kind {
        CKind::F => {
            let n = static_dim(s, 0, name)?;
            let _ = frepr;
            Ok(format!(
                "  {{ FILE *_fp = fopen(argv[{idx}], \"rb\"); if (!_fp) return 1;\n    fseek(_fp, 0, SEEK_END); long {name}_B = ftell(_fp); fseek(_fp, 0, SEEK_SET);\n    {name}_N = {n};\n    {name} = malloc({name}_B); if (fread({name}, 1, {name}_B, _fp) != (size_t){name}_B) return 2; fclose(_fp); }}"
            ))
        }
        CKind::I => Ok(format!(
            "  {{ FILE *_fp = fopen(argv[{idx}], \"rb\"); if (!_fp) return 1;\n    fseek(_fp, 0, SEEK_END); {name}_N = ftell(_fp)/sizeof(int64_t); fseek(_fp, 0, SEEK_SET);\n    {name} = malloc({name}_N * sizeof(int64_t)); if (fread({name}, 8, {name}_N, _fp) != (size_t){name}_N) return 2; fclose(_fp); }}"
        )),
        CKind::T => {
            let sh = s.static_shape(name)?;
            let cnt = sh.iter().product::<usize>();
            let mut out = format!(
                "  {name}_N = {}; /* rows */\n",
                sh.first().copied().unwrap_or(1)
            );
            let mut j = idx;
            for (comp, ct) in [("s", "int8_t"), ("e", "int32_t"), ("z", "uint8_t")] {
                out.push_str(&format!(
                    "  {{ FILE *_fp = fopen(argv[{j}], \"rb\"); if (!_fp) return 1;\n    {name}_{comp} = malloc(({cnt}) * sizeof({ct})); if (fread({name}_{comp}, sizeof({ct}), ({cnt}), _fp) != (size_t)({cnt})) return 2; fclose(_fp); }}\n"
                ));
                j += 1;
            }
            Ok(out)
        }
    }
}

fn alloc_stream(name: &str, s: &infer::StreamInfo, frepr: &str) -> Result<String> {
    let cnt = count_expr(s, name)?;
    match s.kind {
        CKind::F => Ok(format!(
            "  {name} = malloc(({cnt}) * sizeof({frepr}));\n  int64_t {name}_N = {};",
            rows_expr(s)
        )),
        CKind::I => Ok(format!(
            "  {name}_N = ({cnt}); {name} = malloc(({cnt}) * sizeof(int64_t));"
        )),
        CKind::T => Ok(format!(
            "  {name}_s = malloc(({cnt}) * sizeof(int8_t));\n  {name}_e = malloc(({cnt}) * sizeof(int32_t));\n  {name}_z = malloc(({cnt}) * sizeof(uint8_t));\n  int64_t {name}_N = {};",
            rows_expr(s)
        )),
    }
}

fn store_stream(name: &str, s: &infer::StreamInfo, idx: usize, frepr: &str) -> Result<String> {
    let cnt = count_expr(s, name)?;
    match s.kind {
        CKind::F => Ok(format!(
            "  {{ FILE *_fp = fopen(argv[{idx}], \"wb\"); if (!_fp) return 3;\n    fwrite({name}, sizeof({frepr}), {cnt}, _fp); fclose(_fp); }}"
        )),
        CKind::I => Ok(format!(
            "  {{ FILE *_fp = fopen(argv[{idx}], \"wb\"); if (!_fp) return 3;\n    fwrite({name}, 8, {cnt}, _fp); fclose(_fp); }}"
        )),
        CKind::T => {
            let mut out = String::new();
            let mut j = idx;
            for comp in ["s", "e", "z"] {
                let ct = match comp {
                    "s" => "int8_t",
                    "e" => "int32_t",
                    _ => "uint8_t",
                };
                out.push_str(&format!(
                    "  {{ FILE *_fp = fopen(argv[{j}], \"wb\"); if (!_fp) return 3;\n    fwrite({name}_{comp}, sizeof({ct}), {cnt}, _fp); fclose(_fp); }}\n"
                ));
                j += 1;
            }
            Ok(out)
        }
    }
}

/// Compile a listing to C source (no C compiler invocation yet).
pub fn compile_program(text: &str, opts: &CompileOptions, sample: Feeds) -> Result<Artifact> {
    if opts.outputs.is_empty() {
        return Err(Error::asm(
            "compile_program needs outputs=[...] (OUT streams)",
        ));
    }
    if opts.target == Target::Cuda {
        return cuda::compile_cuda(text, opts, sample);
    }
    if opts.target == Target::Rust {
        return rust::compile_rust(text, opts, sample);
    }
    let nonfpu = opts.target == Target::NonFpu;
    let mut bound = asm::assemble(text, &opts.asm)?;
    let mut sample = sample;
    let mut outputs = opts.outputs.clone();
    sanitize_cnames(&mut bound, &mut outputs, &mut sample)?;

    let streams = infer_shapes(&bound, &sample)?;
    for o in &outputs {
        if !streams.contains_key(o) {
            return Err(Error::asm(format!(
                "compile_program: output '{o}' is not a stream"
            )));
        }
    }
    let in_names: Vec<String> = bound.inputs.iter().map(|d| d.name.clone()).collect();
    let mns: Vec<String> = bound.instrs.iter().map(|i| i.mn.clone()).collect();
    let frepr = if nonfpu { "uint64_t" } else { "double" };

    // Rotation tables: C gets runtime sincos + THETA; non-FPU gets baked
    // triples indexed by position.
    let mut theta = Vec::new();
    let mut rope: Option<c::RopeCtx> = None;
    if let Some(ins) = bound.instrs.iter().find(|i| i.mn == "ROTARY") {
        if let Some(src) = ins.args.first() {
            if let Some(s) = streams.get(src) {
                if let Some(d) = s.shape.last().copied().flatten() {
                    let base = bound.config.get_f64_or("rope_base", 10000.0);
                    if nonfpu {
                        let pos_name = ins.args.get(1).cloned().unwrap_or_default();
                        let p = match sample.get(&pos_name) {
                            Some(crate::value::Value::I64(a)) => {
                                a.iter().copied().max().unwrap_or(0) as usize + 1
                            }
                            _ => {
                                return Err(Error::asm(
                                    "nonfpu ROTARY needs real position values (not a stub)",
                                ))
                            }
                        };
                        let half = d / 2;
                        let mut cosv = vec![0f64; p * half];
                        let mut sinv = vec![0f64; p * half];
                        for pp in 0..p {
                            for i in 0..half {
                                let ang = pp as f64 * base.powf(-2.0 * i as f64 / d as f64);
                                cosv[pp * half + i] = ang.cos();
                                sinv[pp * half + i] = ang.sin();
                            }
                        }
                        let cos = lattice_from_vec(&[p, half], cosv)?;
                        let sin = lattice_from_vec(&[p, half], sinv)?;
                        rope = Some(c::RopeCtx { p, d, cos, sin });
                    } else {
                        for i in 0..d / 2 {
                            theta.push(base.powf(-2.0 * i as f64 / d as f64));
                        }
                    }
                }
            }
        }
    }

    let mut ctx = Ctx {
        streams: streams.clone(),
        config: bound.config.clone(),
        gathered: HashMap::new(),
        theta: theta.clone(),
        inputs: in_names.clone(),
        outputs: outputs.clone(),
        nonfpu,
        mns: mns.clone(),
        rope,
    };

    let mut parts: Vec<String> = Vec::new();
    if nonfpu {
        parts.push(nonfpu::prologue(&ctx)?);
    } else {
        parts.push(
            "#include <stdint.h>\n#include <stdio.h>\n#include <stdlib.h>\n#include <string.h>\n#include <math.h>\n#include <time.h>\n"
                .to_string(),
        );
        if !theta.is_empty() {
            let vals: Vec<String> = theta.iter().map(|t| format!("{t:?}")).collect();
            parts.push(format!(
                "static const double THETA[] = {{{}}};",
                vals.join(", ")
            ));
        }
        if mns.iter().any(|m| m == "IINFLATE") {
            parts.push(INFLATE_C.to_string());
        }
    }

    let mut decls = Vec::new();
    decls.push("/* inputs: argv files, order below */".to_string());
    for n in &in_names {
        decls.push(c_decl_stream(n, streams.get(n).unwrap(), frepr));
    }
    for o in &outputs {
        if in_names.contains(o) {
            continue;
        }
        let s = streams.get(o).unwrap();
        match s.kind {
            CKind::F => decls.push(format!("static {frepr} *{o};")),
            CKind::I => decls.push(format!("static int64_t *{o}; static int64_t {o}_N;")),
            CKind::T => decls.push(format!(
                "static int8_t *{o}_s; static int32_t *{o}_e; static uint8_t *{o}_z; static int64_t {o}_N;"
            )),
        }
    }

    // intermediates
    let mut inter: Vec<String> = Vec::new();
    for ins in &bound.instrs {
        for x in &ins.outs {
            if !in_names.contains(x) && !outputs.contains(x) && !inter.contains(x) {
                inter.push(x.clone());
            }
        }
    }
    if !inter.is_empty() {
        decls.push("/* intermediates */".to_string());
        for x in &inter {
            let s = streams.get(x).unwrap();
            match s.kind {
                CKind::F => decls.push(format!("static {frepr} *{x}; int64_t {x}_N;")),
                CKind::I => decls.push(format!("static int64_t *{x}; int64_t {x}_N;")),
                CKind::T => decls.push(format!(
                    "static int8_t *{x}_s; static int32_t *{x}_e; static uint8_t *{x}_z; int64_t {x}_N;"
                )),
            }
        }
    }
    parts.push(decls.join("\n"));

    // main
    let mut n_argv = 0usize;
    for n in &in_names {
        n_argv += if streams.get(n).unwrap().kind == CKind::T {
            3
        } else {
            1
        };
    }
    for o in &outputs {
        n_argv += if streams.get(o).unwrap().kind == CKind::T {
            3
        } else {
            1
        };
    }
    let mut main = Vec::new();
    main.push("int main(int argc, char **argv) {".to_string());
    main.push(format!("  if (argc != {}) return 9;", n_argv + 1));
    let mut idx = 1usize;
    for n in &in_names {
        main.push(input_load(n, streams.get(n).unwrap(), idx, frepr)?);
        idx += if streams.get(n).unwrap().kind == CKind::T {
            3
        } else {
            1
        };
    }
    for x in &inter {
        main.push(alloc_stream(x, streams.get(x).unwrap(), frepr)?);
    }
    for o in &outputs {
        if in_names.contains(o) {
            continue;
        }
        main.push(alloc_stream(o, streams.get(o).unwrap(), frepr)?);
    }

    for ins in &bound.instrs {
        match ins.mn.as_str() {
            "__REPEAT__" => {
                let nv = &ins.args[0];
                if streams.contains_key(nv) {
                    main.push(format!("  int64_t _n = {nv}[0];"));
                } else {
                    main.push(format!("  int64_t _n = {nv};"));
                }
                main.push("  for (int64_t _it = 0; _it < _n; ++_it) {".to_string());
            }
            "__END__" => main.push("  }".to_string()),
            "__BREAK__" => {
                let cv = &ins.args[0];
                if streams.contains_key(cv) {
                    main.push(format!("  if ({cv}[0] != 0) break;"));
                } else {
                    main.push(format!("  if ({} != 0) break;", cv));
                }
            }
            _ => {
                let (body, gathered) = pattern(&ins.mn, &ins.outs, &ins.args, &mut ctx)?;
                if let Some((o, w, pref)) = gathered {
                    ctx.gathered.insert(o, (w, pref));
                }
                let indented = body.replace('\n', "\n  ");
                main.push(format!("  /* L{}: {} */\n  {}", ins.line, ins.mn, indented));
            }
        }
    }

    for o in &outputs {
        main.push(store_stream(o, streams.get(o).unwrap(), idx, frepr)?);
        idx += if streams.get(o).unwrap().kind == CKind::T {
            3
        } else {
            1
        };
    }
    main.push("  return 0;\n}".to_string());
    parts.push(main.join("\n"));

    let source = parts.join("\n") + "\n";
    if nonfpu {
        nonfpu::float_trap(&source)?;
    }
    Ok(Artifact {
        source,
        streams,
        inputs: in_names,
        outputs,
        config: bound.config.clone(),
        n_argv,
    })
}

fn lattice_from_vec(shape: &[usize], v: Vec<f64>) -> Result<crate::value::Triple> {
    let arr = crate::value::array_from_vec(shape, v)?;
    Ok(crate::lattice::encode(&arr))
}

/// Compile emitted source to an executable.
pub fn build(source: &str, workdir: &Path, name: &str, target: Target) -> Result<PathBuf> {
    if target == Target::Cuda {
        return cuda::build_cu(source, workdir, name);
    }
    if target == Target::Rust {
        return build_rust(source, workdir, name);
    }
    std::fs::create_dir_all(workdir)?;
    let src = workdir.join(format!("{name}.c"));
    let exe = workdir.join(name);
    std::fs::write(&src, source)?;
    let mut cmd = std::process::Command::new("cc");
    if target == Target::NonFpu {
        cmd.args(["-O2", "-march=native", "-std=c99", "-Wall", "-fwrapv"]);
    } else {
        cmd.args(["-O2", "-march=native", "-std=c99", "-Wall"]);
    }
    let out = cmd
        .arg(&src)
        .arg("-o")
        .arg(&exe)
        .args(if target == Target::NonFpu {
            vec![]
        } else {
            vec!["-lm"]
        })
        .output()?;
    if !out.status.success() {
        return Err(Error::asm(format!(
            "C build failed:\n{}",
            String::from_utf8_lossy(&out.stderr)
        )));
    }
    Ok(exe)
}

fn build_rust(source: &str, workdir: &Path, name: &str) -> Result<PathBuf> {
    std::fs::create_dir_all(workdir)?;
    let src = workdir.join(format!("{name}.rs"));
    let exe = workdir.join(name);
    std::fs::write(&src, source)?;
    let out = std::process::Command::new("rustc")
        .arg(&src)
        .args(["--edition", "2021", "-O", "-o"])
        .arg(&exe)
        .output()?;
    if !out.status.success() {
        return Err(Error::asm(format!(
            "Rust build failed:\n{}",
            String::from_utf8_lossy(&out.stderr)
        )));
    }
    Ok(exe)
}
