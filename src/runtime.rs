//! The reference interpreter: execute a bound program over typed streams.

use std::collections::HashMap;

use crate::asm::{self, AsmOptions};
use crate::error::{Error, Result};
use crate::ir::{BoundInstr, BoundProgram};
use crate::layout::{check_layouts, is_literal, layout_kind, UNKNOWN};
use crate::ops;
use crate::payload::Feeds;
use crate::value::{Arg, Kind};

fn find_end(prog: &[BoundInstr], i: usize) -> Result<usize> {
    let mut depth = 0i32;
    for (j, ins) in prog.iter().enumerate().skip(i) {
        if ins.is_repeat() {
            depth += 1;
        } else if ins.is_end() {
            depth -= 1;
            if depth == 0 {
                return Ok(j);
            }
        }
    }
    Err(Error::asm("REPEAT without END"))
}

fn stream_shape(a: &Arg) -> Option<Vec<usize>> {
    // 0-d arrays are scalar inputs, like Python bare-number payloads.
    match a.value().and_then(|v| v.shape()) {
        Some(s) if s.is_empty() => None,
        other => other,
    }
}

fn check_shapes(mn: &str, args: &[Arg], line: &str) -> Result<()> {
    let at = |msg: String| Error::asm(format!("line {line} ({mn}): {msg}"));
    let same = |vs: &[Arg], what: &str| -> Result<()> {
        let have: Vec<(usize, Vec<usize>)> = vs
            .iter()
            .enumerate()
            .filter_map(|(i, a)| stream_shape(a).map(|s| (i, s)))
            .collect();
        if have
            .iter()
            .map(|(_, s)| s)
            .collect::<std::collections::HashSet<_>>()
            .len()
            > 1
        {
            return Err(at(format!(
                "{what} shape mismatch {}",
                have.iter()
                    .map(|(i, s)| format!("arg{i}={s:?}"))
                    .collect::<Vec<_>>()
                    .join(", ")
            )));
        }
        Ok(())
    };

    match mn {
        "ADD" | "SUB" | "MUL" | "DIV" => same(args, "elementwise")?,
        "IADD" | "ISUB" | "IMUL" | "IMAD" | "IDIV" | "IMOD" => same(args, "integer elementwise")?,
        "SELECT" => {
            if args.len() >= 3 {
                same(&args[1..3], "SELECT branches")?;
                if let (Some(ms), Some(bs)) = (stream_shape(&args[0]), stream_shape(&args[1])) {
                    if ms != bs {
                        return Err(at(format!("mask {ms:?} vs branches {bs:?}")));
                    }
                }
            }
        }
        "MATMUL" | "BATCH_MATMUL" | "IMATMUL" => {
            if let (Some(a), Some(b)) = (stream_shape(&args[0]), stream_shape(&args[1])) {
                if a.len() < 2 || b.len() < 2 || a[a.len() - 1] != b[b.len() - 2] {
                    return Err(at(format!(
                        "inner dims {a:?} vs {b:?} (did you forget TRANSPOSE?)"
                    )));
                }
            }
        }
        "WARP" => {
            if let (Some(f), Some(fl)) = (stream_shape(&args[0]), stream_shape(&args[1])) {
                if fl.len() < 2 || f != fl[..2] {
                    return Err(at(format!("feature {f:?} vs flow {fl:?} spatial mismatch")));
                }
            }
        }
        "ARGMAX" => {
            if let Some(t) = stream_shape(&args[0]) {
                let axis = args
                    .get(1)
                    .map(|a| a.i64("ARGMAX", "axis"))
                    .transpose()?
                    .unwrap_or(-1);
                if !(0..t.len() as i64).contains(&axis) {
                    return Err(at(format!("axis {axis} out of range for shape {t:?}")));
                }
            }
        }
        "GATHER" => {
            if let (Some(w), Some(ids)) = (stream_shape(&args[0]), args[1].value()) {
                if let Ok(ids) = ids.as_i64("GATHER") {
                    if !ids.is_empty() {
                        let mn = ids.iter().copied().min().unwrap_or(0);
                        let mx = ids.iter().copied().max().unwrap_or(0);
                        if mn < 0 || mx >= w[0] as i64 {
                            return Err(at(format!("ids out of range [0,{})", w[0])));
                        }
                    }
                }
            }
        }
        "GATHER3" => {
            if let Some(w) = stream_shape(&args[0]) {
                if w.len() != 3 {
                    return Err(at(format!("table must be 3D (V1,V2,C), got {w:?}")));
                }
                let ia = args[1].value().map(|v| v.as_i64("GATHER3")).transpose()?;
                let ja = args[2].value().map(|v| v.as_i64("GATHER3")).transpose()?;
                if let (Some(ia), Some(ja)) = (ia, ja) {
                    if ia.shape() != ja.shape() || ia.ndim() != 1 {
                        return Err(at(format!(
                            "i/j must be 1D same length, got {:?} vs {:?}",
                            ia.shape(),
                            ja.shape()
                        )));
                    }
                    if !ia.is_empty() {
                        let imin = ia.iter().copied().min().unwrap_or(0);
                        let imax = ia.iter().copied().max().unwrap_or(0);
                        let jmin = ja.iter().copied().min().unwrap_or(0);
                        let jmax = ja.iter().copied().max().unwrap_or(0);
                        if imin < 0 || imax >= w[0] as i64 {
                            return Err(at(format!("i out of range [0,{})", w[0])));
                        }
                        if jmin < 0 || jmax >= w[1] as i64 {
                            return Err(at(format!("j out of range [0,{})", w[1])));
                        }
                    }
                }
            }
        }
        _ => {}
    }
    Ok(())
}

/// Execute a bound program.
pub fn run(bound: &BoundProgram, payload: &Feeds) -> Result<Feeds> {
    let mut feeds: Feeds = Feeds::new();
    let mut layouts: HashMap<String, String> = HashMap::new();
    for decl in &bound.inputs {
        let v = payload
            .get(&decl.name)
            .ok_or_else(|| Error::asm(format!("payload missing streams: [\"{}\"]", decl.name)))?;
        feeds.insert(decl.name.clone(), v.clone());
        layouts.insert(
            decl.name.clone(),
            decl.layout.clone().unwrap_or_else(|| UNKNOWN.to_string()),
        );
    }

    let prog = &bound.instrs;
    let mut i = 0usize;
    let mut loop_stack: Vec<(usize, usize, i64)> = Vec::new();

    while i < prog.len() {
        let ins = &prog[i];
        if ins.is_repeat() {
            let nv = &ins.args[0];
            let n = if let Some(v) = feeds.get(nv) {
                v.first_i64("REPEAT")?
            } else {
                let f = nv.parse::<f64>().map_err(|_| {
                    Error::asm(format!(
                        "line {}: REPEAT count '{nv}' not defined",
                        ins.line
                    ))
                })?;
                f as i64
            };
            if n < 0 {
                return Err(Error::asm(format!(
                    "line {}: REPEAT count negative",
                    ins.line
                )));
            }
            loop_stack.push((i, find_end(prog, i)?, n));
            i += 1;
            continue;
        }
        if ins.is_end() {
            let (start, end, n) = loop_stack
                .pop()
                .ok_or_else(|| Error::asm(format!("line {}: END without REPEAT", ins.line)))?;
            if n > 1 {
                loop_stack.push((start, end, n - 1));
                i = start + 1;
                continue;
            }
            i += 1;
            continue;
        }
        if ins.is_break() {
            let (_, end, _) = *loop_stack
                .last()
                .ok_or_else(|| Error::asm(format!("line {}: BREAK outside REPEAT", ins.line)))?;
            let cv = &ins.args[0];
            let cond = if let Some(v) = feeds.get(cv) {
                v.first_i64("BREAK")?
            } else {
                cv.parse::<f64>().map(|f| f as i64).map_err(|_| {
                    Error::asm(format!(
                        "line {}: BREAK condition '{cv}' not defined",
                        ins.line
                    ))
                })?
            };
            if cond != 0 {
                loop_stack.pop();
                i = end + 1;
                continue;
            }
            i += 1;
            continue;
        }

        let mut vals: Vec<Arg> = Vec::with_capacity(ins.args.len());
        for a in &ins.args {
            if let Some(v) = feeds.get(a) {
                vals.push(Arg::Stream(v.clone()));
            } else if is_literal(a) {
                vals.push(Arg::Literal(a.parse::<f64>().map_err(|_| {
                    Error::asm(format!("line {}: bad literal '{a}'", ins.line))
                })?));
            } else {
                return Err(Error::asm(format!(
                    "line {} ({}): input stream '{a}' not defined",
                    ins.line, ins.mn
                )));
            }
        }

        let out_lays = check_layouts(
            &ins.outs,
            &ins.mn,
            &ins.args,
            &ins.line,
            ins.sig.as_ref(),
            &layouts,
        )?;

        // Declared kind must match the value (triple-ops on floats are a
        // silent-garbage class; gradual typing skips UNKNOWN).
        for (a, arg) in ins.args.iter().zip(vals.iter()) {
            if is_literal(a) {
                continue;
            }
            let Some(lay) = layouts.get(a) else {
                continue;
            };
            if lay == UNKNOWN || layout_kind(lay).is_none() {
                continue;
            }
            if let Some(v) = arg.value() {
                let got = v.kind();
                if got != Kind::Unknown && !crate::layout::kind_matches(lay, got) {
                    return Err(Error::asm(format!(
                        "line {} ({}): stream '{a}' declared {lay} but holds {got}",
                        ins.line, ins.mn
                    )));
                }
            }
        }

        check_shapes(&ins.mn, &vals, &ins.line)?;
        let outs = ops::dispatch(&ins.mn, &vals, &bound.config)?;
        if outs.len() != ins.outs.len() {
            return Err(Error::asm(format!(
                "line {} ({}): runtime arity break: {:?}",
                ins.line, ins.mn, ins.outs
            )));
        }
        for (name, val) in ins.outs.iter().zip(outs) {
            feeds.insert(name.clone(), val);
            let lay = out_lays
                .iter()
                .find(|(n, _)| n == name)
                .map(|(_, l)| l.clone())
                .unwrap_or_else(|| UNKNOWN.to_string());
            layouts.insert(name.clone(), lay);
        }

        i += 1;
    }

    Ok(feeds)
}

/// Parse, assemble, and execute a listing.
pub fn run_text(text: &str, opts: &AsmOptions, payload: &Feeds) -> Result<Feeds> {
    let bound = asm::assemble(text, opts)?;
    run(&bound, payload)
}

/// Run a listing `n` times, threading STATE streams (the `repeat()` path).
///
/// Non-STATE inputs take lists of length `n`; STATE streams take one seed.
/// `grow` names STATE streams that may grow append-only along axis 0.
pub fn repeat(
    text: &str,
    opts: &AsmOptions,
    payload: &[Feeds],
    grow: &[String],
) -> Result<(Vec<Feeds>, Feeds)> {
    let bound = asm::assemble(text, opts)?;
    let n = payload.len();
    if n == 0 {
        return Err(Error::runtime("repeat: empty payload sequence"));
    }
    for g in grow {
        if !bound.state.contains(g) {
            return Err(Error::asm(format!(
                "grow stream '{g}' is not a STATE stream"
            )));
        }
    }
    let in_names: Vec<String> = bound.inputs.iter().map(|d| d.name.clone()).collect();
    let mut histories = Vec::with_capacity(n);
    let mut carry: Feeds = Feeds::new();
    let mut shapes: HashMap<String, Vec<usize>> = HashMap::new();

    for it in 0..n {
        let mut feeds = Feeds::new();
        for name in &in_names {
            if bound.state.contains(name) {
                if it == 0 {
                    let v = payload[0].get(name).ok_or_else(|| {
                        Error::asm(format!("payload missing streams: [\"{name}\"]"))
                    })?;
                    feeds.insert(name.clone(), v.clone());
                } else {
                    let v = carry.get(name).ok_or_else(|| {
                        Error::runtime(format!("iteration {it}: STATE '{name}' missing"))
                    })?;
                    feeds.insert(name.clone(), v.clone());
                }
            } else {
                let v = payload[it]
                    .get(name)
                    .ok_or_else(|| Error::asm(format!("payload missing streams: [\"{name}\"]")))?;
                feeds.insert(name.clone(), v.clone());
            }
        }
        let got = run(&bound, &feeds)?;
        for name in &bound.state {
            let v = got.get(name).ok_or_else(|| {
                Error::asm(format!("iteration {it}: STATE '{name}' not assigned"))
            })?;
            let sh = v.shape();
            if let (Some(prev), Some(cur)) = (shapes.get(name), sh.as_ref()) {
                if prev != cur {
                    if grow.contains(name) {
                        if prev.len() != cur.len() || prev[1..] != cur[1..] || cur[0] < prev[0] {
                            return Err(Error::asm(format!(
                                "iteration {it}: STATE '{name}' changed {prev:?} -> {cur:?} (append-only along axis 0)"
                            )));
                        }
                    } else {
                        return Err(Error::asm(format!(
                            "iteration {it}: STATE '{name}' shape changed {prev:?} -> {cur:?} (dynamic shapes refused)"
                        )));
                    }
                }
            }
            if let Some(cur) = sh {
                shapes.insert(name.clone(), cur);
            }
            carry.insert(name.clone(), v.clone());
        }
        histories.push(got);
    }
    let last = histories.last().cloned().unwrap_or_default();
    Ok((histories, last))
}
