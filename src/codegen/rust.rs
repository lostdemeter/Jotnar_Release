//! Rust code emission (Target::Rust).
//!
//! Emits a self-contained Rust program (std only) implementing the
//! listing's graph with `Vec<f64>`/`Vec<i64>` streams. Contract:
//! `IN`/ `OUT` payload kinds accepted are exactly the raw binary
//! conventions of the C target — F streams as little-endian `f64`,
//! I streams as little-endian `i64` (payload F32/F64 and I64/U8/Bool
//! all lower to those two, as in `infer_shapes`). **T (triple)
//! streams are refused**: the emitter may change the schedule, not the
//! arithmetic; silently decoding a triple to f64 would change it.
//! Unsupported ops fail loud — there is no silent fallback, exactly
//! like the C backend.

use std::collections::HashMap;

use super::infer::{infer_shapes, CKind, StreamInfo};
use super::{Artifact, CompileOptions};
use crate::asm::{self};
use crate::error::{Error, Result};
use crate::layout::is_literal;
use crate::payload::Feeds;

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
"#;

pub(crate) fn sanitize_rust(name: &str) -> String {
    let mut out: String = name
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '_' {
                c
            } else {
                '_'
            }
        })
        .collect();
    if out
        .chars()
        .next()
        .map(|c| c.is_ascii_digit())
        .unwrap_or(false)
    {
        out = format!("_{out}");
    }
    // Raw identifiers are not allowed for self/Self/super/crate.
    if matches!(out.as_str(), "self" | "Self" | "super" | "crate") {
        return format!("_{out}");
    }
    match out.as_str() {
        "fn" | "let" | "mut" | "type" | "impl" | "struct" | "enum" | "match" | "if" | "else"
        | "for" | "while" | "loop" | "return" | "break" | "continue" | "mod" | "pub" | "use"
        | "where" | "as" | "in" | "dyn" | "async" | "await" | "move" | "ref" | "true" | "false"
        | "unsafe" | "extern" | "static" | "const" | "trait" | "yield" | "try" | "box" => {
            format!("r#{out}")
        }
        _ => out,
    }
}

pub fn compile_rust(text: &str, opts: &CompileOptions, sample: Feeds) -> Result<Artifact> {
    if opts.outputs.is_empty() {
        return Err(Error::asm(
            "compile_program needs outputs=[...] (OUT streams)",
        ));
    }
    // F/I subset: refuse any T input up front (before shape inference
    // re-labels it), so the refusal is typed and not masked.
    for (name, v) in &sample {
        if matches!(v, crate::value::Value::Triple(_)) {
            return Err(Error::unimplemented(format!(
                "Rust backend: T (triple) stream '{name}' not supported (F/I subset only)"
            )));
        }
    }
    let mut bound = asm::assemble(text, &opts.asm)?;
    let mut sample = sample;
    let mut outputs = opts.outputs.clone();
    // Sanitize identifiers (kept separate so C names never leak).
    let mut known: std::collections::HashSet<String> =
        bound.inputs.iter().map(|d| d.name.clone()).collect();
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
    // Contract: Rust is an F/I subset. Refuse any T stream loudly.
    for (name, s) in &streams {
        if s.kind == CKind::T {
            return Err(Error::unimplemented(format!(
                "Rust backend: T (triple) stream '{name}' not supported (F/I subset only)"
            )));
        }
    }
    for o in &outputs {
        if !streams.contains_key(o) {
            return Err(Error::asm(format!(
                "compile_program: output '{o}' is not a stream"
            )));
        }
    }
    let in_names: Vec<String> = bound.inputs.iter().map(|d| d.name.clone()).collect();

    // Load body: reads raw argv files (F as f64, I as i64), mirroring
    // the C backend's file convention.
    let mut parts: Vec<String> = vec![
        "// generated by jotnar emit-rust (Target::Rust)".to_string(),
        "use std::env;".to_string(),
        HELPERS.to_string(),
        "fn main() {".to_string(),
        "    let args: Vec<String> = env::args().collect();".to_string(),
    ];
    // Build main body properly
    let mut body: Vec<String> = Vec::new();
    let mut ai = 1usize;
    for n in &in_names {
        let s = streams.get(n).unwrap();
        match s.kind {
            CKind::F => {
                body.push(format!(
                    "    let {n}: Vec<f64> = read_f64_file(&args[{ai}]);"
                ));
                ai += 1;
            }
            CKind::I => {
                body.push(format!(
                    "    let {n}: Vec<i64> = read_i64_file(&args[{ai}]);"
                ));
                ai += 1;
            }
            CKind::T => {
                return Err(Error::unimplemented(format!(
                    "Rust backend: T (triple) stream '{n}' not supported (F/I subset only)"
                )));
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
            if s.kind == CKind::T {
                return Err(Error::unimplemented(format!(
                    "Rust backend: T (triple) stream '{x}' not supported (F/I subset only)"
                )));
            }
            match s.kind {
                CKind::F => {
                    body.push(format!("    let mut {x}: Vec<f64> = vec![0.0; {cnt}];"));
                }
                CKind::I => {
                    body.push(format!("    let mut {x}: Vec<i64> = vec![0; {cnt}];"));
                }
                CKind::T => unreachable!(),
            }
        }
    }
    // helpers for file IO (injected once at top of main via parts already; add fns here as free fns)
    // instructions
    for ins in &bound.instrs {
        match ins.mn.as_str() {
            "__REPEAT__" | "__END__" | "__BREAK__" => {
                return Err(Error::unimplemented(format!(
                    "Rust backend {} not yet (fail loud)",
                    ins.mn
                )))
            }
            _ => {
                let stmt = pattern_rust(&ins.mn, &ins.outs, &ins.args, &streams)?;
                body.push(format!(
                    "    /* L{}: {} */\n    {}",
                    ins.line,
                    ins.mn,
                    stmt.trim()
                ));
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

fn scalar_or_index(kind: &str, name: &str, idx: &str) -> Result<String> {
    if is_literal(name) {
        if kind == "I" {
            Ok(format!("{name}i64"))
        } else {
            Ok(name.to_string())
        }
    } else {
        // stream handled by caller via indexed expression; this helper is
        // only for the right operand position (kept for clarity).
        Ok(format!("{name}[{idx}]"))
    }
}

fn pattern_rust(
    mn: &str,
    outs: &[String],
    args: &[String],
    streams: &HashMap<String, StreamInfo>,
) -> Result<String> {
    let o = outs[0].as_str();
    let get = |n: &str| {
        streams
            .get(n)
            .ok_or_else(|| Error::asm(format!("Rust backend: unknown stream '{n}'")))
    };
    let elems = |n: &str| -> Result<usize> {
        get(n)?
            .elems()
            .ok_or_else(|| Error::unimplemented(format!("Rust backend: dynamic shape in {n}")))
    };
    match mn {
        "ADD" | "SUB" | "MUL" | "DIV" => {
            let (a, b) = (args[0].as_str(), args[1].as_str());
            let n = if is_literal(a) { elems(b)? } else { elems(a)? };
            let aexpr = if is_literal(a) {
                a.to_string()
            } else {
                format!("{a}[i]")
            };
            let op = match mn {
                "ADD" => "+",
                "SUB" => "-",
                "MUL" => "*",
                _ => "/",
            };
            let bexpr = scalar_or_index("F", b, "i")?;
            Ok(format!(
                "for i in 0..{n} {{ {o}[i] = {aexpr} {op} {bexpr}; }}"
            ))
        }
        "IADD" | "ISUB" | "IMUL" | "IDIV" => {
            let (a, b) = (args[0].as_str(), args[1].as_str());
            let n = if is_literal(a) { elems(b)? } else { elems(a)? };
            let aexpr = if is_literal(a) {
                format!("{a}i64")
            } else {
                format!("{a}[i]")
            };
            let bexpr = scalar_or_index("I", b, "i")?;
            if mn == "IDIV" {
                return Ok(format!(
                    "for i in 0..{n} {{ let bv = {bexpr}; if bv == 0 || ({aexpr} == i64::MIN && bv == -1) {{ std::process::exit(30); }} {o}[i] = {aexpr} / bv; }}"
                ));
            }
            let op = match mn {
                "IADD" => "+",
                "ISUB" => "-",
                _ => "*",
            };
            Ok(format!(
                "for i in 0..{n} {{ {o}[i] = {aexpr} {op} {bexpr}; }}"
            ))
        }
        "MATMUL" | "BATCH_MATMUL" => {
            let (a, b) = (args[0].as_str(), args[1].as_str());
            let ash = get(a)?.shape.clone();
            let bsh = get(b)?.shape.clone();
            if ash.len() == 2 {
                let (m, k, n) = match (ash[0], ash[1], bsh[1]) {
                    (Some(m), Some(k), Some(n)) => (m, k, n),
                    _ => return Err(Error::unimplemented("Rust backend: dynamic dim in MATMUL")),
                };
                Ok(format!("{o} = matmul(&{a}, &{b}, {m}, {k}, {n});"))
            } else {
                Err(Error::unimplemented(
                    "Rust backend BATCH_MATMUL (3D) not yet",
                ))
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
            let (rows, cols) = match (sh[0], sh[1]) {
                (Some(r), Some(c)) => (r, c),
                _ => return Err(Error::unimplemented("Rust backend: dynamic dim in SOFTMAX")),
            };
            Ok(format!("{o} = softmax_rows(&{a}, {rows}, {cols});"))
        }
        "LAYERNORM" => {
            // LAYERNORM(x, w, b) — check C pattern arg order.
            let (x, w, b) = (args[0].as_str(), args[1].as_str(), args[2].as_str());
            let sh = get(x)?.shape.clone();
            if sh.len() != 2 {
                return Err(Error::unimplemented("Rust LAYERNORM needs 2D"));
            }
            let (rows, cols) = match (sh[0], sh[1]) {
                (Some(r), Some(c)) => (r, c),
                _ => {
                    return Err(Error::unimplemented(
                        "Rust backend: dynamic dim in LAYERNORM",
                    ))
                }
            };
            Ok(format!(
                "{o} = layernorm_rows(&{x}, &{w}, &{b}, {rows}, {cols}, 1e-6);"
            ))
        }
        "CONV" => {
            // C parity: same-size, single-channel, edge-replicate,
            // odd kernel. `infer_shapes` has already validated the
            // geometry, so unwrap the static dims here.
            let (x, k) = (args[0].as_str(), args[1].as_str());
            let xsh = get(x)?.shape.clone();
            let ksh = get(k)?.shape.clone();
            let (Some(h), Some(w)) = (xsh[0], xsh[1]) else {
                return Err(Error::unimplemented(
                    "Rust backend: dynamic dim in CONV input",
                ));
            };
            let (Some(kh), Some(kw)) = (ksh[0], ksh[1]) else {
                return Err(Error::unimplemented(
                    "Rust backend: dynamic dim in CONV kernel",
                ));
            };
            let (rh, rw) = (kh / 2, kw / 2);
            Ok(format!(
                "for oy in 0..{h} {{\n\
                 \x20 for ox in 0..{w} {{\n\
                 \x20   let mut acc = 0.0f64;\n\
                 \x20   for ky in 0..{kh} {{\n\
                 \x20     let iy0 = oy as i64 + ky as i64 - {rh};\n\
                 \x20     let iy = if iy0 < 0 {{ 0usize }} else if iy0 >= {h} as i64 {{ {} }} else {{ iy0 as usize }};\n\
                 \x20     for kx in 0..{kw} {{\n\
                 \x20       let ix0 = ox as i64 + kx as i64 - {rw};\n\
                 \x20       let ix = if ix0 < 0 {{ 0usize }} else if ix0 >= {w} as i64 {{ {} }} else {{ ix0 as usize }};\n\
                 \x20       acc += {x}[iy * {w} + ix] * {k}[ky * {kw} + kx];\n\
                 \x20     }}\n\
                 \x20   }}\n\
                 \x20   {o}[oy * {w} + ox] = acc;\n\
                 \x20 }}\n\
                 }}",
                h - 1,
                w - 1,
            ))
        }
        "DECONV" => Err(Error::unimplemented(
            "Rust backend DECONV not yet (next slice)",
        )),
        "INTERP" => Err(Error::unimplemented(
            "Rust backend INTERP not yet (next slice)",
        )),
        "GATHER" => Err(Error::unimplemented(
            "Rust backend GATHER not yet (next slice)",
        )),
        "TRANSPOSE" => {
            let x = args[0].as_str();
            let sh = get(x)?.shape.clone();
            if sh.len() != 2 {
                return Err(Error::unimplemented("Rust TRANSPOSE needs 2D"));
            }
            let (Some(r), Some(c)) = (sh[0], sh[1]) else {
                return Err(Error::unimplemented(
                    "Rust backend: dynamic dim in TRANSPOSE",
                ));
            };
            Ok(format!(
                "for t_i in 0..{r} {{ for t_j in 0..{c} {{ {o}[t_j * {r} + t_i] = {x}[t_i * {c} + t_j]; }} }}"
            ))
        }
        "CONCAT" => {
            // v0.2 contract: 2D, axis 1, same-kind, same rows, baked widths.
            let (a, b) = (args[0].as_str(), args[1].as_str());
            let ash = get(a)?.shape.clone();
            let bsh = get(b)?.shape.clone();
            let (Some(rows), Some(da), Some(db)) = (ash[0], ash[1], bsh[1]) else {
                return Err(Error::unimplemented("Rust backend: dynamic dim in CONCAT"));
            };
            Ok(format!(
                "for n_i in 0..{rows} {{\n\
                 \x20 for n_j in 0..{da} {{ {o}[n_i * {} + n_j] = {a}[n_i * {da} + n_j]; }}\n\
                 \x20 for n_j in 0..{db} {{ {o}[n_i * {} + {da} + n_j] = {b}[n_i * {db} + n_j]; }}\n\
                 }}",
                da + db,
                da + db,
            ))
        }
        "RESHAPE2" | "RESHAPE3" => {
            let a = args[0].as_str();
            let n = elems(a)?;
            Ok(format!("{o} = {a}[..{n}].to_vec();"))
        }
        "GT" | "CLIP" | "POOLAVG" | "SQRT" | "SQUARE" => {
            Err(Error::unimplemented(format!("Rust backend {mn} not yet")))
        }
        other => Err(Error::unimplemented(format!(
            "Rust backend {other} not yet (fail loud)"
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codegen::infer::StreamInfo;
    use std::collections::HashMap;

    #[test]
    fn dynamic_dims_are_typed_errors() {
        let mut streams = HashMap::new();
        streams.insert("A".to_string(), StreamInfo::f(vec![Some(2), None]));
        streams.insert("B".to_string(), StreamInfo::f(vec![None, Some(2)]));
        let r = pattern_rust(
            "MATMUL",
            &["OUT".to_string()],
            &["A".to_string(), "B".to_string()],
            &streams,
        );
        assert!(r.is_err());
        assert!(r.unwrap_err().to_string().contains("dynamic dim"));
    }
}
