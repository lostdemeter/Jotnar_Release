//! Rust code emission (Target::Rust).
//!
//! Emits a self-contained Rust program (std only) implementing the
//! listing's graph with `Vec<f64>`/`Vec<i64>` streams. T (triple)
//! inputs are decoded to f64 at load (`lattice::decode` formula),
//! mirroring the geometric decode in `dav2_reverse/geo_lut.py`.
//! Unsupported ops fail loud — there is no silent fallback, exactly
//! like the C backend.

use std::collections::HashMap;

use super::infer::{infer_shapes, CKind, StreamInfo};
use crate::asm::{self, AsmOptions};
use crate::error::{Error, Result};
use crate::ir::BoundProgram;
use crate::payload::Feeds;
use super::{Artifact, CompileOptions};

const HELPERS: &str = r#"
// ---- generated Rust helpers (std only) ----
fn erf_approx(x: f64) -> f64 {
    // Abramowitz & Stegun 7.1.26, |err| < 1.5e-7 (float parity tests use 1e-6).
    let s = x.signum();
    let x = x.abs();
    let t = 1.0 / (1.0 + 0.3275911 * x);
    let y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * (-x * x).exp();
    s * y
}
fn gelu(x: f64) -> f64 { 0.5 * x * (1.0 + erf_approx(x / std::f64::consts::SQRT_2)) }
fn sigmoid(x: f64) -> f64 { 1.0 / (1.0 + (-x).exp()) }
fn softmax_rows(x: &[f64], rows: usize, cols: usize) -> Vec<f64> {
    let mut o = vec![0f64; x.len()];
    for r in 0..rows {
        let row = &x[r * cols..(r + 1) * cols];
        let mx = row.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let mut sum = 0f64;
        for j in 0..cols { let e = (row[j] - mx).exp(); o[r * cols + j] = e; sum += e; }
        if sum == 0.0 { continue; }
        for j in 0..cols { o[r * cols + j] /= sum; }
    }
    o
}
fn layernorm_rows(x: &[f64], w: &[f64], b: &[f64], rows: usize, cols: usize, eps: f64) -> Vec<f64> {
    let mut o = vec![0f64; x.len()];
    for r in 0..rows {
        let row = &x[r * cols..(r + 1) * cols];
        let mean: f64 = row.iter().sum::<f64>() / cols as f64;
        let var: f64 = row.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / cols as f64;
        let std = (var + eps).sqrt();
        for j in 0..cols {
            o[r * cols + j] = (row[j] - mean) / std * w[j] + b[j];
        }
    }
    o
}
fn matmul(a: &[f64], b: &[f64], m: usize, k: usize, n: usize) -> Vec<f64> {
    let mut o = vec![0f64; m * n];
    for i in 0..m {
        for kk in 0..k {
            let aik = a[i * k + kk];
            for j in 0..n {
                o[i * n + j] += aik * b[kk * n + j];
            }
        }
    }
    o
}
fn phi_decode(s: &[i8], e: &[i32], z: &[u8]) -> Vec<f64> {
    // value = sign * PHI^((exp - 32768)/512), zero masked
    const PHI: f64 = 1.618033988749895;
    let mut out = Vec::with_capacity(s.len());
    for ((&sv, &ev), &zv) in s.iter().zip(e.iter()).zip(z.iter()) {
        if zv != 0 { out.push(0.0); } else { out.push(sv as f64 * PHI.powf((ev as f64 - 32768.0) / 512.0)); }
    }
    out
}
"#;

pub(crate) fn sanitize_rust(name: &str) -> String {
    let mut out: String = name
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() || c == '_' { c } else { '_' })
        .collect();
    if out.chars().next().map(|c| c.is_ascii_digit()).unwrap_or(false) {
        out = format!("_{out}");
    }
    // Rust keywords need raw identifiers
    match out.as_str() {
        "fn" | "let" | "mut" | "type" | "impl" | "struct" | "enum" | "match" | "if" | "else"
        | "for" | "while" | "loop" | "return" | "break" | "continue" | "crate" | "mod" | "pub"
        | "use" | "where" | "as" | "in" | "dyn" | "async" | "await" | "move" | "ref" | "self" | "Self"
        | "super" | "true" | "false" | "unsafe" | "extern" | "static" | "const" | "trait" | "virtual"
        | "yield" | "try" | "box" => format!("r#{{out}}"),
        _ => out,
    }
}

pub fn compile_rust(text: &str, opts: &CompileOptions, sample: Feeds) -> Result<Artifact> {
    if opts.outputs.is_empty() {
        return Err(Error::asm("compile_program needs outputs=[...] (OUT streams)"));
    }
    let mut bound = asm::assemble(text, &opts.asm)?;
    let mut sample = sample;
    let mut outputs = opts.outputs.clone();
    // Rust identifiers: reuse C sanitizer logic by calling the crate
    // helper through a local remap (kept separate so C names never leak).
    let mut streams_pre = infer_shapes(&bound, &sample)?;
    let _ = &mut streams_pre;
    // sanitize inputs/outs/args/sample keys
    let mut known: std::collections::HashSet<String> = bound.inputs.iter().map(|d| d.name.clone()).collect();
    for ins in &bound.instrs {
        known.extend(ins.outs.iter().cloned());
    }
    let mut map: std::collections::HashMap<String, String> = std::collections::HashMap::new();
    let mut seen: std::collections::HashMap<String, String> = std::collections::HashMap::new();
    for n in &known {
        let c = sanitize_rust(n);
        if let Some(prev) = seen.get(&c) {
            if prev != n {
                return Err(Error::asm(format!(
                    "emit-rust: stream names {prev:?} and {n:?} collide as Rust identifier {c:?}"
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
    let old = std::mem::take(&mut sample);
    let mut next = Feeds::new();
    for (k, v) in old {
        next.insert(remap(&k), v);
    }
    sample = next;

    let streams = infer_shapes(&bound, &sample)?;
    for o in &outputs {
        if !streams.contains_key(o) {
            return Err(Error::asm(format!(
                "compile_program: output '{o}' is not a stream"
            )));
        }
    }
    let in_names: Vec<String> = bound.inputs.iter().map(|d| d.name.clone()).collect();

    let mut parts: Vec<String> = Vec::new();
    parts.push("// generated by jotnar emit-rust (Target::Rust)".to_string());
    parts.push("use std::env;".to_string());
    parts.push(HELPERS.to_string());

    // Declarations + load
    let mut decls = Vec::new();
    for n in &in_names {
        let s = streams.get(n).unwrap();
        match s.kind {
            CKind::F => decls.push(format!("    let {n}: Vec<f64> = {{ let raw = std::fs::read(&args[ai]).unwrap(); let mut v = Vec::with_capacity(raw.len()/8); for c in raw.chunks_exact(8) {{ v.push(f64::from_le_bytes([c[0],c[1],c[2],c[3],c[4],c[5],c[6],c[7]])); }} v }};")),
            CKind::I => decls.push(format!("    let {n}: Vec<i64> = {{ let raw = std::fs::read(&args[ai]).unwrap(); let mut v = Vec::with_capacity(raw.len()/8); for c in raw.chunks_exact(8) {{ v.push(i64::from_le_bytes([c[0],c[1],c[2],c[3],c[4],c[5],c[6],c[7]])); }} v }};")),
            CKind::T => {
                // expects 3 consecutive argv files: s, e, z
                decls.push(format!("    let {n}: Vec<f64> = {{ let rs = std::fs::read(&args[ai]).unwrap(); let rg = std::fs::read(&args[ai+1]).unwrap(); let rz = std::fs::read(&args[ai+2]).unwrap(); let s: Vec<i8> = rs.iter().copied().collect(); let e: Vec<i32> = rg.chunks_exact(4).map(|c| i32::from_le_bytes([c[0],c[1],c[2],c[3]])).collect(); let z: Vec<u8> = rz.clone(); phi_decode(&s, &e, &z) }};"))
            }
        }
    }
    // Intermediates/outputs allocated as zero vecs lazily at use sites
    // via pattern calls; allocate outputs here as well.
    for o in &outputs {
        if in_names.contains(o) {
            continue;
        }
        let s = streams.get(o).unwrap();
        let cnt = s.elems().unwrap_or(0);
        match s.kind {
            CKind::F => decls.push(format!("    let mut {o}: Vec<f64> = vec![0.0; {cnt}];")),
            CKind::I => decls.push(format!("    let mut {o}: Vec<i64> = vec![0; {cnt}];")),
            CKind::T => decls.push(format!("    let mut {o}: Vec<f64> = vec![0.0; {cnt}]; // T stored as f64 in rust emitter")),
        }
    }
    parts.push("fn main() {".to_string());
    parts.push("    let args: Vec<String> = env::args().collect();".to_string());
    parts.push("    let mut ai = 1usize;".to_string());
    // emit decls with ai advance
    for n in &in_names {
        let s = streams.get(n).unwrap();
        let need = if s.kind == CKind::T { 3 } else { 1 };
        // inject load body then advance ai (already inline via args[ai] placeholder above)
        // Rebuild properly below.
        let _ = need;
    }
    // Build main body properly
    let mut body: Vec<String> = Vec::new();
    let mut ai = 1usize;
    for n in &in_names {
        let s = streams.get(n).unwrap();
        match s.kind {
            CKind::F => {
                body.push(format!("    let {n}: Vec<f64> = read_f64_file(&args[{ai}]);"));
                ai += 1;
            }
            CKind::I => {
                body.push(format!("    let {n}: Vec<i64> = read_i64_file(&args[{ai}]);"));
                ai += 1;
            }
            CKind::T => {
                body.push(format!("    let {n}: Vec<f64> = phi_decode_file(&args[{ai}], &args[{ai}+1], &args[{ai}+2]);"));
                ai += 3;
            }
        }
    }
    for o in &outputs {
        if in_names.contains(o) {
            continue;
        }
        let s = streams.get(o).unwrap();
        let cnt = s.elems().unwrap_or(0);
        match s.kind {
            CKind::F | CKind::T => {
                body.push(format!("    let mut {o}: Vec<f64> = vec![0.0; {cnt}];"));
            }
            CKind::I => {
                body.push(format!("    let mut {o}: Vec<i64> = vec![0; {cnt}];"));
            }
        }
    }
    // Intermediates
    {
        let mut inter: Vec<String> = Vec::new();
        for ins in &bound.instrs {
            for x in &ins.outs {
                if !in_names.contains(x) && !outputs.contains(x) && !inter.contains(x) {
                    inter.push(x.clone());
                }
            }
        }
        for x in &inter {
            let s = streams.get(x).unwrap();
            let cnt = s.elems().unwrap_or(0);
            match s.kind {
                CKind::F | CKind::T => {
                    body.push(format!("    let mut {x}: Vec<f64> = vec![0.0; {cnt}];"));
                }
                CKind::I => {
                    body.push(format!("    let mut {x}: Vec<i64> = vec![0; {cnt}];"));
                }
            }
        }
    }
    // helpers for file IO (injected once at top of main via parts already; add fns here as free fns)
    // instructions
    for ins in &bound.instrs {
        match ins.mn.as_str() {
            "__REPEAT__" | "__END__" | "__BREAK__" => {
                return Err(Error::unimplemented(format!(
                    "Rust backend {} not yet (fail loud)", ins.mn
                )))
            }
            _ => {
                let stmt = pattern_rust(&ins.mn, &ins.outs, &ins.args, &streams)?;
                body.push(format!("    /* L{}: {} */\n    {}", ins.line, ins.mn, stmt.trim()));
            }
        }
    }
    for o in &outputs {
        let kind = streams.get(o).map(|s| s.kind).unwrap_or(CKind::F);
        match kind {
            CKind::I => body.push(format!("    write_i64_file(&args[{ai}], &{o});")),
            _ => body.push(format!("    write_f64_file(&args[{ai}], &{o});")),
        }
        ai += 1;
    }
    parts.push(body.join("\n"));
    parts.push("}".to_string());

    // inject file-IO helpers before main (they are used but not defined in HELPERS)
    let mut full = String::new();
    full.push_str(&parts[0]);
    full.push('\n');
    full.push_str(&parts[1]);
    full.push('\n');
    full.push_str(HELPERS);
    full.push('\n');
    full.push_str(IO_HELPERS);
    full.push('\n');
    // skip parts[2] (generated decls) — body already contains loads
    for p in parts.iter().skip(3) {
        full.push_str(p);
        full.push('\n');
    }
    // remove duplicated decl artifacts if present (empty)
    let _ = decls;

    Ok(Artifact {
        source: full,
        streams,
        inputs: in_names,
        outputs,
        config: bound.config.clone(),
        n_argv: ai,
    })
}

const IO_HELPERS: &str = r#"
fn read_f64_file(p: &str) -> Vec<f64> {
    let raw = std::fs::read(p).expect("read input");
    raw.chunks_exact(8).map(|c| f64::from_le_bytes([c[0],c[1],c[2],c[3],c[4],c[5],c[6],c[7]])).collect()
}
fn read_i64_file(p: &str) -> Vec<i64> {
    let raw = std::fs::read(p).expect("read input");
    raw.chunks_exact(8).map(|c| i64::from_le_bytes([c[0],c[1],c[2],c[3],c[4],c[5],c[6],c[7]])).collect()
}
fn phi_decode_file(ps: &str, pe: &str, pz: &str) -> Vec<f64> {
    let rs = std::fs::read(ps).expect("read s");
    let rg = std::fs::read(pe).expect("read e");
    let rz = std::fs::read(pz).expect("read z");
    let s: Vec<i8> = rs.iter().map(|&b| b as i8).collect();
    let e: Vec<i32> = rg.chunks_exact(4).map(|c| i32::from_le_bytes([c[0],c[1],c[2],c[3]])).collect();
    phi_decode(&s, &e, &rz)
}
fn write_f64_file(p: &str, v: &[f64]) {
    let mut buf = Vec::with_capacity(v.len()*8);
    for x in v { buf.extend_from_slice(&x.to_le_bytes()); }
    std::fs::write(p, buf).expect("write output");
}
fn write_i64_file(p: &str, v: &[i64]) {
    let mut buf = Vec::with_capacity(v.len()*8);
    for x in v { buf.extend_from_slice(&x.to_le_bytes()); }
    std::fs::write(p, buf).expect("write output");
}
"#;

fn pattern_rust(mn: &str, outs: &[String], args: &[String], streams: &HashMap<String, StreamInfo>) -> Result<String> {
    let o = outs[0].as_str();
    let get = |n: &str| streams.get(n).ok_or_else(|| Error::asm(format!("Rust backend: unknown stream '{n}'")));
    let elems = |n: &str| -> Result<usize> {
        get(n)?.elems().ok_or_else(|| Error::unimplemented(format!("Rust backend: dynamic shape in {n}")))
    };
    match mn {
        "ADD" | "SUB" | "MUL" | "DIV" => {
            let (a, b) = (args[0].as_str(), args[1].as_str());
            let n = elems(a)?;
            let op = match mn { "ADD" => "+", "SUB" => "-", "MUL" => "*", _ => "/" };
            Ok(format!("for i in 0..{n} {{ {o}[i] = {a}[i] {op} {b}[i]; }}"))
        }
        "IADD" | "ISUB" | "IMUL" | "IDIV" => {
            let (a, b) = (args[0].as_str(), args[1].as_str());
            let n = elems(a)?;
            let op = match mn { "IADD" => "+", "ISUB" => "-", "IMUL" => "*", _ => "/" };
            Ok(format!("for i in 0..{n} {{ {o}[i] = {a}[i] {op} {b}[i]; }}"))
        }
        "MATMUL" | "BATCH_MATMUL" => {
            let (a, b) = (args[0].as_str(), args[1].as_str());
            let ash = get(a)?.shape.clone();
            let bsh = get(b)?.shape.clone();
            if ash.len() == 2 {
                let (m, k, n) = (ash[0].unwrap(), ash[1].unwrap(), bsh[1].unwrap());
                Ok(format!("{o} = matmul(&{a}, &{b}, {m}, {k}, {n});"))
            } else {
                Err(Error::unimplemented("Rust backend BATCH_MATMUL (3D) not yet"))
            }
        }
        "GELU" => {
            let a = args[0].as_str();
            let n = elems(a)?;
            Ok(format!("for i in 0..{n} {{ {o}[i] = gelu({a}[i]); }}"))
        }
        "SIGMOID" => {
            let a = args[0].as_str();
            let n = elems(a)?;
            Ok(format!("for i in 0..{n} {{ {o}[i] = sigmoid({a}[i]); }}"))
        }
        "SOFTMAX" | "SOFTMAX_WIDE" => {
            let a = args[0].as_str();
            let sh = get(a)?.shape.clone();
            if sh.len() != 2 {
                return Err(Error::unimplemented("Rust SOFTMAX needs 2D"));
            }
            let (rows, cols) = (sh[0].unwrap(), sh[1].unwrap());
            Ok(format!("{o} = softmax_rows(&{a}, {rows}, {cols});"))
        }
        "LAYERNORM" => {
            // LAYERNORM(x, w, b) — check C pattern arg order.
            let (x, w, b) = (args[0].as_str(), args[1].as_str(), args[2].as_str());
            let sh = get(x)?.shape.clone();
            if sh.len() != 2 {
                return Err(Error::unimplemented("Rust LAYERNORM needs 2D"));
            }
            let (rows, cols) = (sh[0].unwrap(), sh[1].unwrap());
            Ok(format!("{o} = layernorm_rows(&{x}, &{w}, &{b}, {rows}, {cols}, 1e-6);"))
        }
        "CONV" => Err(Error::unimplemented("Rust backend CONV not yet (next slice)")),
        "DECONV" => Err(Error::unimplemented("Rust backend DECONV not yet (next slice)")),
        "INTERP" => Err(Error::unimplemented("Rust backend INTERP not yet (next slice)")),
        "GATHER" => Err(Error::unimplemented("Rust backend GATHER not yet (next slice)")),
        "RESHAPE2" | "RESHAPE3" => {
            let a = args[0].as_str();
            let n = elems(a)?;
            Ok(format!("{o} = {a}[..{n}].to_vec();"))
        }
        "GT" | "CLIP" | "POOLAVG" | "SQRT" | "SQRT" | "SQUARE" => Err(Error::unimplemented(format!("Rust backend {mn} not yet"))),
        other => Err(Error::unimplemented(format!("Rust backend {other} not yet (fail loud)"))),
    }
}
