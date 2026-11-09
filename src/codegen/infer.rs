//! Static shape/dtype inference for the C backend.
//!
//! This mirrors `chain/emit_c.py:infer_shapes` for the ops the C backend
//! lowers. Shapes may carry `None` for dims derived from dynamic ids; the
//! current emitter requires fully static geometry and refuses otherwise.

use std::collections::HashMap;

use crate::error::{Error, Result};
use crate::ir::BoundProgram;
use crate::layout::is_literal;
use crate::payload::Feeds;
use crate::value::Value;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CKind {
    F,
    T,
    I,
}

impl CKind {
    pub fn as_str(self) -> &'static str {
        match self {
            CKind::F => "F",
            CKind::T => "T",
            CKind::I => "I",
        }
    }
}

#[derive(Clone, Debug)]
pub struct StreamInfo {
    pub kind: CKind,
    pub shape: Vec<Option<usize>>,
}

impl StreamInfo {
    pub fn f(shape: Vec<Option<usize>>) -> Self {
        StreamInfo {
            kind: CKind::F,
            shape,
        }
    }

    pub fn t(shape: Vec<Option<usize>>) -> Self {
        StreamInfo {
            kind: CKind::T,
            shape,
        }
    }

    pub fn i(shape: Vec<Option<usize>>) -> Self {
        StreamInfo {
            kind: CKind::I,
            shape,
        }
    }

    pub fn static_shape(&self, where_: &str) -> Result<Vec<usize>> {
        let mut out = Vec::with_capacity(self.shape.len());
        for d in &self.shape {
            match d {
                Some(v) => out.push(*v),
                None => {
                    return Err(Error::unimplemented(format!(
                        "C backend: dynamic shape in {where_} (needs static geometry)"
                    )))
                }
            }
        }
        Ok(out)
    }

    pub fn rows(&self) -> Option<usize> {
        if self.shape.is_empty() {
            Some(1)
        } else {
            self.shape[0]
        }
    }

    pub fn elems(&self) -> Option<usize> {
        let mut n = 1usize;
        for d in &self.shape {
            n *= (*d)?;
        }
        Some(n)
    }
}

fn sh(v: &Value) -> Result<StreamInfo> {
    match v {
        Value::Triple(t) => Ok(StreamInfo::t(t.shape().into_iter().map(Some).collect())),
        Value::F32(a) => Ok(StreamInfo::f(a.shape().iter().map(|d| Some(*d)).collect())),
        Value::F64(a) => Ok(StreamInfo::f(a.shape().iter().map(|d| Some(*d)).collect())),
        Value::I64(a) => Ok(StreamInfo::i(a.shape().iter().map(|d| Some(*d)).collect())),
        Value::U8(a) => Ok(StreamInfo::i(a.shape().iter().map(|d| Some(*d)).collect())),
        Value::Bool(a) => Ok(StreamInfo::i(a.shape().iter().map(|d| Some(*d)).collect())),
        Value::None => Err(Error::asm("C backend: sample stream is None")),
    }
}

fn int_lit(a: &str, what: &str) -> Result<i64> {
    if !is_literal(a) {
        return Err(Error::asm(format!(
            "{what}: need integral literal, got {a}"
        )));
    }
    let f: f64 = a
        .parse()
        .map_err(|_| Error::asm(format!("{what}: bad literal {a}")))?;
    if !f.is_finite() || f.fract() != 0.0 {
        return Err(Error::asm(format!(
            "{what}: need integral literal, got {a}"
        )));
    }
    Ok(f as i64)
}

fn same_dims(a: &[Option<usize>], b: &[Option<usize>]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x == y)
}

fn get<'a>(streams: &'a HashMap<String, StreamInfo>, n: &str) -> Result<&'a StreamInfo> {
    streams
        .get(n)
        .ok_or_else(|| Error::asm(format!("C backend: unknown stream '{n}'")))
}

fn dyadic(f: f64) -> bool {
    if f <= 0.0 {
        return false;
    }
    let bits = f.to_bits();
    let exp = ((bits >> 52) & 0x7ff) as i32 - 1022;
    let mant_bits = (bits & !(0x7ffu64 << 52)) | (1022u64 << 52);
    let mant = f64::from_bits(mant_bits);
    let _ = exp;
    (mant - 0.5).abs() < 1e-15
}

/// Infer static stream shapes for the C backend.
pub fn infer_shapes(bound: &BoundProgram, sample: &Feeds) -> Result<HashMap<String, StreamInfo>> {
    let mut streams: HashMap<String, StreamInfo> = HashMap::new();
    for decl in &bound.inputs {
        let v = sample.get(&decl.name).ok_or_else(|| {
            Error::asm(format!(
                "C backend: sample missing IN stream '{}'",
                decl.name
            ))
        })?;
        streams.insert(decl.name.clone(), sh(v)?);
    }

    for ins in &bound.instrs {
        let o = ins.outs.first().cloned().unwrap_or_default();
        let where_ = format!("line {} ({})", ins.line, ins.mn);
        let a = |i: usize| -> &str { ins.args.get(i).map(|s| s.as_str()).unwrap_or("") };
        match ins.mn.as_str() {
            "__REPEAT__" | "__END__" | "__BREAK__" => continue,
            "GATHER" => {
                let t = get(&streams, a(0))?;
                let ids = get(&streams, a(1))?;
                let n = if ids.shape.len() == 1 {
                    ids.shape[0]
                } else if ids.shape.is_empty() {
                    Some(1)
                } else {
                    None
                };
                if ids.kind != CKind::I || t.shape.len() != 2 {
                    return Err(Error::asm(format!(
                        "{where_}: v0.2 needs table (V,C) + int ids"
                    )));
                }
                let c = t.shape[1];
                if ids.shape.len() == 2 {
                    if ids.shape.iter().any(|d| d.is_none()) {
                        return Err(Error::asm(format!(
                            "{where_}: GATHER 2D ids must be static, got {:?}",
                            ids.shape
                        )));
                    }
                    let shape = vec![ids.shape[0], ids.shape[1], c];
                    streams.insert(
                        o,
                        StreamInfo {
                            kind: t.kind,
                            shape,
                        },
                    );
                } else {
                    if n.is_none() && ids.shape.len() > 1 {
                        return Err(Error::asm(format!("{where_}: v0.2 needs (N,)/scalar ids")));
                    }
                    streams.insert(
                        o,
                        StreamInfo {
                            kind: t.kind,
                            shape: vec![n, c],
                        },
                    );
                }
            }
            "GATHER3" => {
                let t = get(&streams, a(0))?;
                let ii = get(&streams, a(1))?;
                let jj = get(&streams, a(2))?;
                if t.shape.len() != 3 || t.shape.iter().any(|d| d.is_none()) {
                    return Err(Error::asm(format!(
                        "{where_}: GATHER3 v0.1 needs baked table (V1,V2,C), got {:?}",
                        t.shape
                    )));
                }
                if !matches!(t.kind, CKind::T | CKind::F) {
                    return Err(Error::asm(format!("{where_}: GATHER3 table must be T/F")));
                }
                if ii.kind != CKind::I
                    || jj.kind != CKind::I
                    || ii.shape.len() != 1
                    || jj.shape.len() != 1
                {
                    return Err(Error::asm(format!("{where_}: GATHER3 needs 1D I i/j")));
                }
                if ii.shape[0].is_none() || jj.shape[0].is_none() {
                    return Err(Error::asm(format!(
                        "{where_}: GATHER3 v0.1 needs baked i/j length"
                    )));
                }
                if ii.shape[0] != jj.shape[0] {
                    return Err(Error::asm(format!(
                        "{where_}: GATHER3 i/j length mismatch {:?} vs {:?}",
                        ii.shape, jj.shape
                    )));
                }
                streams.insert(
                    o,
                    StreamInfo {
                        kind: t.kind,
                        shape: vec![ii.shape[0], t.shape[2]],
                    },
                );
            }
            "SCATTER" => {
                let b = get(&streams, a(0))?;
                let r = get(&streams, a(1))?;
                let idx = get(&streams, a(2))?;
                if b.kind != CKind::F || r.kind != CKind::F || idx.kind != CKind::I {
                    return Err(Error::asm(format!(
                        "{where_}: SCATTER needs F base + F row + I idx"
                    )));
                }
                if b.shape.len() != 2 || !same_dims(&r.shape, &[Some(1), b.shape[1]]) {
                    return Err(Error::asm(format!(
                        "{where_}: SCATTER base {:?} + row {:?} mismatch (row must be (1,C))",
                        b.shape, r.shape
                    )));
                }
                if idx.shape.len() > 1
                    || (idx.shape.len() == 1 && !matches!(idx.shape[0], None | Some(1)))
                {
                    return Err(Error::asm(format!(
                        "{where_}: SCATTER idx must be scalar/1-elem"
                    )));
                }
                streams.insert(o, StreamInfo::f(b.shape.clone()));
            }
            "IADD" | "ISUB" | "IMUL" | "IMAD" | "IDIV" | "IMOD" | "IDIVU" | "IMODU" | "ISAR" => {
                let x = get(&streams, a(0))?;
                if x.kind != CKind::I {
                    return Err(Error::asm(format!(
                        "{where_}: {} needs I-kind first operand",
                        ins.mn
                    )));
                }
                for (i, arg) in ins.args.iter().enumerate().skip(1) {
                    if streams.contains_key(arg) {
                        let s = get(&streams, arg)?;
                        if s.kind != CKind::I || !same_dims(&s.shape, &x.shape) {
                            return Err(Error::asm(format!(
                                "{where_}: {} operand {i} must be I with shape {:?}, got {} {:?}",
                                ins.mn,
                                x.shape,
                                s.kind.as_str(),
                                s.shape
                            )));
                        }
                    } else {
                        int_lit(arg, &format!("{where_} {} operand {i}", ins.mn))?;
                    }
                }
                streams.insert(o, StreamInfo::i(x.shape.clone()));
            }
            "IMATMUL" => {
                let x = get(&streams, a(0))?;
                let y = get(&streams, a(1))?;
                if x.kind != CKind::I || y.kind != CKind::I {
                    return Err(Error::asm(format!(
                        "{where_}: IMATMUL needs I-kind operands"
                    )));
                }
                if x.shape.len() != 2 || y.shape.len() != 2 {
                    return Err(Error::asm(format!(
                        "{where_}: IMATMUL needs 2D I operands, got {:?} x {:?}",
                        x.shape, y.shape
                    )));
                }
                if let (Some(k1), Some(k2)) = (x.shape[1], y.shape[0]) {
                    if k1 != k2 {
                        return Err(Error::asm(format!(
                            "{where_}: IMATMUL inner dims {k1} vs {k2} (left [M,K], right [K,N])"
                        )));
                    }
                }
                let out = vec![x.shape[0], y.shape[1]];
                streams.insert(o, StreamInfo::i(out));
            }
            "IREDUCE_SUM" => {
                let x = get(&streams, a(0))?;
                if x.kind != CKind::I {
                    return Err(Error::asm(format!(
                        "{where_}: IREDUCE_SUM needs an I-kind stream"
                    )));
                }
                streams.insert(o, StreamInfo::i(vec![]));
            }
            "IEQ" | "ILT" | "IMAX" | "SADD" => {
                let x = get(&streams, a(0))?;
                if x.kind != CKind::I {
                    return Err(Error::asm(format!(
                        "{where_}: {} needs I-kind first operand",
                        ins.mn
                    )));
                }
                if let Some(s) = streams.get(a(1)) {
                    if s.kind != CKind::I || !same_dims(&s.shape, &x.shape) {
                        return Err(Error::asm(format!(
                            "{where_}: {} operand 1 must be I with shape {:?}",
                            ins.mn, x.shape
                        )));
                    }
                } else {
                    int_lit(a(1), &format!("{where_} {} operand 1", ins.mn))?;
                }
                streams.insert(o, StreamInfo::i(x.shape.clone()));
            }
            "ISELECT" => {
                let c = get(&streams, a(0))?;
                if c.kind != CKind::I {
                    return Err(Error::asm(format!(
                        "{where_}: ISELECT needs I-kind condition"
                    )));
                }
                for i in 1..=2 {
                    if let Some(s) = streams.get(a(i)) {
                        if s.kind != CKind::I || !same_dims(&s.shape, &c.shape) {
                            return Err(Error::asm(format!(
                                "{where_}: ISELECT operand {i} must be I with shape {:?}",
                                c.shape
                            )));
                        }
                    } else {
                        int_lit(a(i), &format!("{where_} ISELECT operand {i}"))?;
                    }
                }
                streams.insert(o, StreamInfo::i(c.shape.clone()));
            }
            "IXOR" | "IAND" | "IOR" => {
                let x = get(&streams, a(0))?;
                if x.kind != CKind::I {
                    return Err(Error::asm(format!(
                        "{where_}: {} needs I-kind first operand",
                        ins.mn
                    )));
                }
                if let Some(s) = streams.get(a(1)) {
                    if s.kind != CKind::I || !same_dims(&s.shape, &x.shape) {
                        return Err(Error::asm(format!(
                            "{where_}: {} operand must be I with shape {:?}",
                            ins.mn, x.shape
                        )));
                    }
                } else {
                    int_lit(a(1), &format!("{where_} {} operand", ins.mn))?;
                }
                streams.insert(o, StreamInfo::i(x.shape.clone()));
            }
            "ISHL" | "ISHR" => {
                let x = get(&streams, a(0))?;
                if x.kind != CKind::I {
                    return Err(Error::asm(format!("{where_}: {} needs I", ins.mn)));
                }
                let s = int_lit(a(1), &format!("{where_} {} shift", ins.mn))?;
                if !(0..=63).contains(&s) {
                    return Err(Error::asm(format!(
                        "{where_}: {} shift must be 0..63, got {s}",
                        ins.mn
                    )));
                }
                streams.insert(o, StreamInfo::i(x.shape.clone()));
            }
            "F2I" => {
                let x = get(&streams, a(0))?;
                if x.kind != CKind::F {
                    return Err(Error::asm(format!("{where_}: F2I needs F input")));
                }
                streams.insert(o, StreamInfo::i(x.shape.clone()));
            }
            "I2F" => {
                let x = get(&streams, a(0))?;
                if x.kind != CKind::I {
                    return Err(Error::asm(format!("{where_}: I2F needs I input")));
                }
                streams.insert(o, StreamInfo::f(x.shape.clone()));
            }
            "T2I" => {
                let x = get(&streams, a(0))?;
                if x.kind != CKind::T {
                    return Err(Error::asm(format!("{where_}: T2I needs T input")));
                }
                streams.insert(o, StreamInfo::i(x.shape.clone()));
            }
            "IGET" => {
                let x = get(&streams, a(0))?;
                if x.kind != CKind::I {
                    return Err(Error::asm(format!("{where_}: IGET needs I input")));
                }
                streams.insert(o, StreamInfo::i(Vec::new()));
            }
            "ARGMAX" => {
                let x = get(&streams, a(0))?;
                let axis = if ins.args.len() > 1 && !streams.contains_key(a(1)) {
                    int_lit(a(1), &format!("{where_} axis"))?
                } else {
                    -1
                };
                if !matches!(x.kind, CKind::T | CKind::F | CKind::I) {
                    return Err(Error::asm(format!("{where_}: ARGMAX needs T/F/I input")));
                }
                let out_sh = if x.shape.len() == 1 {
                    vec![]
                } else if x.shape.len() == 2 && axis % 2 == 1 {
                    vec![x.shape[0]]
                } else {
                    return Err(Error::asm(format!(
                        "{where_}: v0.2 ARGMAX handles 1D or 2D-axis-1, got {:?} ax {axis}",
                        x.shape
                    )));
                };
                streams.insert(o, StreamInfo::i(out_sh));
            }
            "TOPK" => {
                let x = get(&streams, a(0))?;
                let k = int_lit(a(1), &format!("{where_} k"))?;
                if x.kind != CKind::F || x.shape.len() != 2 {
                    return Err(Error::asm(format!("{where_}: TOPK needs 2D F scores")));
                }
                let w = x.shape[1];
                if w.is_none() || !(1..=w.unwrap() as i64).contains(&k) {
                    return Err(Error::asm(format!(
                        "{where_}: TOPK k={k} out of (1,S={w:?})"
                    )));
                }
                let rows = x.shape[0];
                streams.insert(
                    ins.outs[0].clone(),
                    StreamInfo::f(vec![rows, Some(k as usize)]),
                );
                streams.insert(
                    ins.outs[1].clone(),
                    StreamInfo::i(vec![rows, Some(k as usize)]),
                );
            }
            "RMSNORM" => {
                let x = get(&streams, a(0))?;
                let w = get(&streams, a(1))?;
                if !matches!(x.kind, CKind::F | CKind::T)
                    || x.kind != w.kind
                    || x.shape.is_empty()
                    || w.shape.len() != 1
                    || w.shape[0] != x.shape[x.shape.len() - 1]
                {
                    return Err(Error::unimplemented(format!(
                        "{where_}: RMSNORM needs same-kind x(...,D)+w(D,)"
                    )));
                }
                streams.insert(o, x.clone());
            }
            "LAYERNORM" => {
                let x = get(&streams, a(0))?;
                let w = get(&streams, a(1))?;
                let b = get(&streams, a(2))?;
                let d = *x.shape.last().unwrap_or(&None);
                if !matches!(x.kind, CKind::F | CKind::T)
                    || x.kind != w.kind
                    || x.kind != b.kind
                    || x.shape.is_empty()
                    || w.shape != vec![d]
                    || b.shape != vec![d]
                {
                    return Err(Error::unimplemented(format!(
                        "{where_}: LAYERNORM needs same-kind x(...,D)+w(D,)+b(D,)"
                    )));
                }
                streams.insert(o, x.clone());
            }
            "INTERP" => {
                let x = get(&streams, a(0))?;
                let sy = a(1).parse::<f64>().map_err(|_| {
                    Error::unimplemented(format!("{where_}: INTERP needs literal scales"))
                })?;
                let sx = a(2).parse::<f64>().map_err(|_| {
                    Error::unimplemented(format!("{where_}: INTERP needs literal scales"))
                })?;
                if x.kind != CKind::F || !(x.shape.len() == 2 || x.shape.len() == 3) {
                    return Err(Error::unimplemented(format!(
                        "{where_}: INTERP needs F x(H,W[,C])"
                    )));
                }
                if !(dyadic(sy) && dyadic(sx)) {
                    return Err(Error::asm(format!(
                        "{where_}: INTERP non-dyadic scale ({sy},{sx}) is a LOWERING ERROR"
                    )));
                }
                let h = x.shape[0].ok_or_else(|| {
                    Error::asm(format!("{where_}: INTERP needs static HW (v0.1)"))
                })?;
                let w = x.shape[1].ok_or_else(|| {
                    Error::asm(format!("{where_}: INTERP needs static HW (v0.1)"))
                })?;
                let ho = ((h as f64 * sy).round().max(1.0)) as usize;
                let wo = ((w as f64 * sx).round().max(1.0)) as usize;
                let mut shape = vec![Some(ho), Some(wo)];
                shape.extend_from_slice(&x.shape[2..]);
                streams.insert(o, StreamInfo::f(shape));
            }
            "CONV2D" => {
                let x = get(&streams, a(0))?;
                let w = get(&streams, a(1))?;
                let b = get(&streams, a(2))?;
                let stride = int_lit(a(3), &format!("{where_} stride"))?;
                let pad = int_lit(a(4), &format!("{where_} pad"))?;
                if x.kind != CKind::F
                    || w.kind != CKind::F
                    || b.kind != CKind::F
                    || x.shape.len() != 3
                    || w.shape.len() != 4
                    || b.shape.len() != 1
                {
                    return Err(Error::unimplemented(format!(
                        "{where_}: CONV2D needs F x(H,W,C)+w(Cout,kh,kw,C)+b(Cout)"
                    )));
                }
                let (Some(h), Some(ww), Some(cin)) = (x.shape[0], x.shape[1], x.shape[2]) else {
                    return Err(Error::asm(format!("{where_}: CONV2D needs static HWC")));
                };
                let (Some(cout), Some(kh), Some(kw), Some(wcin)) =
                    (w.shape[0], w.shape[1], w.shape[2], w.shape[3])
                else {
                    return Err(Error::asm(format!("{where_}: CONV2D needs static weights")));
                };
                if wcin != cin || b.shape[0] != Some(cout) {
                    return Err(Error::asm(format!(
                        "{where_}: CONV2D channel/bias mismatch"
                    )));
                }
                if stride <= 0 || pad < 0 {
                    return Err(Error::asm(format!("{where_}: CONV2D bad stride/pad")));
                }
                let ho = (h as i64 + 2 * pad - kh as i64) / stride + 1;
                let wo = (ww as i64 + 2 * pad - kw as i64) / stride + 1;
                if ho <= 0 || wo <= 0 {
                    return Err(Error::asm(format!("{where_}: CONV2D bad output size")));
                }
                streams.insert(
                    o,
                    StreamInfo::f(vec![Some(ho as usize), Some(wo as usize), Some(cout)]),
                );
            }
            "RESIZE2D" => {
                let x = get(&streams, a(0))?;
                let ho = int_lit(a(1), &format!("{where_} ho"))?;
                let wo = int_lit(a(2), &format!("{where_} wo"))?;
                let align = int_lit(a(3), &format!("{where_} align"))?;
                if x.kind != CKind::F || x.shape.len() != 3 || ho <= 0 || wo <= 0 {
                    return Err(Error::unimplemented(format!(
                        "{where_}: RESIZE2D needs F HWC + positive sizes"
                    )));
                }
                if align != 0 && align != 1 {
                    return Err(Error::asm(format!("{where_}: RESIZE2D align must be 0/1")));
                }
                streams.insert(
                    o,
                    StreamInfo::f(vec![Some(ho as usize), Some(wo as usize), x.shape[2]]),
                );
            }
            "CONV" => {
                let x = get(&streams, a(0))?;
                let k = get(&streams, a(1))?;
                if x.kind != CKind::F
                    || k.kind != CKind::F
                    || x.shape.len() != 2
                    || k.shape.len() != 2
                {
                    return Err(Error::unimplemented(format!(
                        "{where_}: CONV needs F x(H,W)+K(kH,kW) (v0.1)"
                    )));
                }
                let (Some(h), Some(w)) = (x.shape[0], x.shape[1]) else {
                    return Err(Error::asm(format!(
                        "{where_}: CONV needs static shapes (v0.1)"
                    )));
                };
                let (Some(kh), Some(kw)) = (k.shape[0], k.shape[1]) else {
                    return Err(Error::asm(format!(
                        "{where_}: CONV needs static shapes (v0.1)"
                    )));
                };
                if kh % 2 == 0 || kw % 2 == 0 || kh > h || kw > w {
                    return Err(Error::asm(format!(
                        "{where_}: CONV needs odd kernel within image, got x({h},{w}) k({kh},{kw})"
                    )));
                }
                streams.insert(o, StreamInfo::f(vec![Some(h), Some(w)]));
            }
            "MATMUL" | "BATCH_MATMUL" => {
                let x = get(&streams, a(0))?;
                let y = get(&streams, a(1))?;
                if !matches!(x.kind, CKind::F | CKind::T) || x.kind != y.kind {
                    return Err(Error::unimplemented(format!(
                        "{where_}: {} needs same-kind inputs",
                        ins.mn
                    )));
                }
                let out_sh = if x.shape.len() == 2 && y.shape.len() == 2 && x.shape[1] == y.shape[0]
                {
                    vec![x.shape[0], y.shape[1]]
                } else if x.shape.len() == 3
                    && y.shape.len() == 3
                    && (x.shape[0] == y.shape[0] || y.shape[0] == Some(1))
                    && x.shape[2] == y.shape[1]
                {
                    vec![x.shape[0], x.shape[1], y.shape[2]]
                } else {
                    return Err(Error::asm(format!(
                        "{where_}: {} shape mismatch {:?}/{:?}",
                        ins.mn, x.shape, y.shape
                    )));
                };
                streams.insert(
                    o,
                    StreamInfo {
                        kind: x.kind,
                        shape: out_sh,
                    },
                );
            }
            "BMMV" => {
                let x = get(&streams, a(0))?;
                let y = get(&streams, a(1))?;
                if x.kind != CKind::F
                    || y.kind != CKind::F
                    || x.shape.len() < 2
                    || y.shape.len() < 2
                {
                    return Err(Error::unimplemented(format!(
                        "{where_}: BMMV needs F bases rank>=2 (v0.1)"
                    )));
                }
                let nums: Vec<i64> = (2..13)
                    .map(|i| int_lit(a(i), &format!("{where_} BMMV")))
                    .collect::<Result<_>>()?;
                let (bb, m, n, k, laa, sa, lab, sb, _lac, _sc, tb) = (
                    nums[0], nums[1], nums[2], nums[3], nums[4], nums[5], nums[6], nums[7],
                    nums[8], nums[9], nums[10],
                );
                if !(tb == 0 || tb == 1) {
                    return Err(Error::asm(format!("{where_}: BMMV TRANSB must be 0/1")));
                }
                let flat = |s: &StreamInfo| -> Option<usize> {
                    let mut n = 1usize;
                    for d in &s.shape {
                        n *= (*d)?;
                    }
                    Some(n)
                };
                if let Some(na) = flat(x) {
                    if (bb - 1) * sa + (m - 1) * laa + k > na as i64 {
                        return Err(Error::asm(format!("{where_}: BMMV A view overruns base")));
                    }
                }
                let rb = if tb != 0 { n } else { k };
                let cb = if tb != 0 { k } else { n };
                if let Some(nb) = flat(y) {
                    if (bb - 1) * sb + (rb - 1) * lab + cb > nb as i64 {
                        return Err(Error::asm(format!("{where_}: BMMV B view overruns base")));
                    }
                }
                streams.insert(
                    o,
                    StreamInfo::f(vec![Some((bb * m) as usize), Some(n as usize)]),
                );
            }
            "ATTN" => {
                let q = get(&streams, a(0))?;
                let k = get(&streams, a(1))?;
                let v = get(&streams, a(2))?;
                let m = get(&streams, a(3))?;
                if q.kind != CKind::F
                    || k.kind != CKind::F
                    || v.kind != CKind::F
                    || m.kind != CKind::I
                {
                    return Err(Error::asm(format!(
                        "{where_}: ATTN needs F qr/Kf/Vf + I mask"
                    )));
                }
                let nh = int_lit(a(4), &format!("{where_} ATTN"))?;
                let nkv = int_lit(a(5), &format!("{where_} ATTN"))?;
                let dh = int_lit(a(6), &format!("{where_} ATTN"))?;
                let s = int_lit(a(7), &format!("{where_} ATTN"))?;
                if nh % nkv != 0
                    || q.shape != vec![Some(1), Some((nh * dh) as usize)]
                    || k.shape != vec![Some(s as usize), Some((nkv * dh) as usize)]
                    || v.shape != vec![Some(s as usize), Some((nkv * dh) as usize)]
                    || m.shape != vec![Some((nh / nkv) as usize), Some(s as usize)]
                {
                    return Err(Error::asm(format!("{where_}: ATTN geometry mismatch")));
                }
                streams.insert(o, StreamInfo::f(vec![Some(1), Some((nh * dh) as usize)]));
            }
            "SLICE" => {
                let x = get(&streams, a(0))?;
                let ax = int_lit(a(1), &format!("{where_} axis"))?;
                let lo = int_lit(a(2), &format!("{where_} lo"))?;
                let hi = int_lit(a(3), &format!("{where_} hi"))?;
                if !matches!(x.kind, CKind::F | CKind::T | CKind::I)
                    || x.shape.len() != 2
                    || !(ax == 0 || ax == 1)
                {
                    return Err(Error::unimplemented(format!(
                        "{where_}: v0.2 SLICE handles 2D ax0/1 F/T/I; use IWINDOW for 1D I"
                    )));
                }
                let mut d = x.shape.clone();
                if let Some(dd) = d[ax as usize] {
                    if !(0 <= lo && lo <= hi && hi <= dd as i64) {
                        return Err(Error::asm(format!(
                            "{where_}: SLICE [{lo},{hi}) of dim {dd}"
                        )));
                    }
                }
                d[ax as usize] = Some((hi - lo) as usize);
                streams.insert(
                    o,
                    StreamInfo {
                        kind: x.kind,
                        shape: d,
                    },
                );
            }
            "IWINDOW" => {
                let x = get(&streams, a(0))?;
                let ln = int_lit(a(2), &format!("{where_} len"))?;
                if x.kind != CKind::I || x.shape.len() != 1 {
                    return Err(Error::unimplemented(format!(
                        "{where_}: IWINDOW handles 1D I"
                    )));
                }
                if let Some(s) = streams.get(a(1)) {
                    if s.kind != CKind::I
                        || s.shape.len() != 1
                        || !matches!(s.shape[0], None | Some(1))
                    {
                        return Err(Error::asm(format!(
                            "{where_}: IWINDOW dynamic start must be a 1-element I"
                        )));
                    }
                } else {
                    let lo = int_lit(a(1), &format!("{where_} start"))?;
                    if let Some(n) = x.shape[0] {
                        if !(0 <= lo && 0 <= ln && lo + ln <= n as i64) {
                            return Err(Error::asm(format!(
                                "{where_}: IWINDOW [{lo},{}) of dim {n}",
                                lo + ln
                            )));
                        }
                    }
                }
                streams.insert(o, StreamInfo::i(vec![Some(ln as usize)]));
            }
            "ICRC32" => {
                let x = get(&streams, a(0))?;
                if x.kind != CKind::I || x.shape.len() != 1 {
                    return Err(Error::unimplemented(format!(
                        "{where_}: ICRC32 handles 1D I"
                    )));
                }
                streams.insert(o, StreamInfo::i(vec![Some(1)]));
            }
            "ICRC32C" => {
                let x = get(&streams, a(0))?;
                if x.kind != CKind::I || x.shape.len() != 1 {
                    return Err(Error::unimplemented(format!(
                        "{where_}: ICRC32C handles 1D I"
                    )));
                }
                streams.insert(o, StreamInfo::i(vec![Some(1)]));
            }
            "IINFLATE" => {
                let x = get(&streams, a(0))?;
                let n = int_lit(a(1), &format!("{where_} out size"))?;
                if x.kind != CKind::I || x.shape.len() != 1 {
                    return Err(Error::unimplemented(format!(
                        "{where_}: IINFLATE handles 1D I"
                    )));
                }
                if n < 0 {
                    return Err(Error::asm(format!("{where_}: IINFLATE negative size {n}")));
                }
                streams.insert(o, StreamInfo::i(vec![Some(n as usize)]));
            }
            "IVARINT" => {
                let x = get(&streams, a(0))?;
                if x.kind != CKind::I || x.shape.len() != 1 {
                    return Err(Error::unimplemented(format!(
                        "{where_}: IVARINT handles 1D I"
                    )));
                }
                if let Some(s) = streams.get(a(1)) {
                    if s.kind != CKind::I
                        || s.shape.len() != 1
                        || !matches!(s.shape[0], None | Some(1))
                    {
                        return Err(Error::asm(format!(
                            "{where_}: IVARINT dynamic start must be a 1-element I"
                        )));
                    }
                }
                streams.insert(ins.outs[0].clone(), StreamInfo::i(vec![Some(1)]));
                streams.insert(ins.outs[1].clone(), StreamInfo::i(vec![Some(1)]));
            }
            "IFIND" => {
                let x = get(&streams, a(0))?;
                let p = get(&streams, a(1))?;
                if x.kind != CKind::I
                    || x.shape.len() != 1
                    || p.kind != CKind::I
                    || p.shape.len() != 1
                {
                    return Err(Error::unimplemented(format!(
                        "{where_}: IFIND needs 1D I X/PAT"
                    )));
                }
                streams.insert(o, StreamInfo::i(vec![Some(1)]));
            }
            "IGATHER1" => {
                let t = get(&streams, a(0))?;
                let ids = get(&streams, a(1))?;
                if t.kind != CKind::I
                    || t.shape.len() != 1
                    || ids.kind != CKind::I
                    || ids.shape.len() != 1
                {
                    return Err(Error::unimplemented(format!(
                        "{where_}: IGATHER1 needs 1D I table/ids"
                    )));
                }
                streams.insert(o, StreamInfo::i(vec![ids.shape[0]]));
            }
            "IASSEMBLE" => {
                let x = get(&streams, a(0))?;
                let lo = int_lit(a(1), &format!("{where_} start"))?;
                let ln = int_lit(a(2), &format!("{where_} len"))?;
                if x.kind != CKind::I || x.shape.len() != 1 {
                    return Err(Error::unimplemented(format!(
                        "{where_}: IASSEMBLE handles 1D I"
                    )));
                }
                if !(1..=8).contains(&ln) {
                    return Err(Error::asm(format!(
                        "{where_}: IASSEMBLE len must be 1..8, got {ln}"
                    )));
                }
                if let Some(n) = x.shape[0] {
                    if !(0 <= lo && lo + ln <= n as i64) {
                        return Err(Error::asm(format!(
                            "{where_}: IASSEMBLE [{lo},{}) of dim {n}",
                            lo + ln
                        )));
                    }
                }
                streams.insert(o, StreamInfo::i(vec![Some(1)]));
            }
            "ISUM" => {
                let x = get(&streams, a(0))?;
                let lo = int_lit(a(1), &format!("{where_} start"))?;
                let ln = int_lit(a(2), &format!("{where_} len"))?;
                if x.kind != CKind::I || x.shape.len() != 1 {
                    return Err(Error::unimplemented(format!("{where_}: ISUM handles 1D I")));
                }
                if let Some(n) = x.shape[0] {
                    if !(0 <= lo && 0 <= ln && lo + ln <= n as i64) {
                        return Err(Error::asm(format!(
                            "{where_}: ISUM [{lo},{}) of dim {n}",
                            lo + ln
                        )));
                    }
                }
                streams.insert(o, StreamInfo::i(vec![Some(1)]));
            }
            "ISCATTER1" => {
                let base = get(&streams, a(0))?;
                let ids = get(&streams, a(1))?;
                let vals = get(&streams, a(2))?;
                if base.kind != CKind::I
                    || base.shape.len() != 1
                    || ids.kind != CKind::I
                    || ids.shape.len() != 1
                    || vals.kind != CKind::I
                    || vals.shape.len() != 1
                {
                    return Err(Error::unimplemented(format!(
                        "{where_}: ISCATTER1 needs 1D I OUT/IDS/VALS"
                    )));
                }
                if ids.shape[0].is_some()
                    && vals.shape[0].is_some()
                    && ids.shape[0] != vals.shape[0]
                {
                    return Err(Error::asm(format!(
                        "{where_}: ISCATTER1 IDS {:?} vs VALS {:?}",
                        ids.shape[0], vals.shape[0]
                    )));
                }
                streams.insert(o, StreamInfo::i(base.shape.clone()));
            }
            "ROTARY" => {
                let x = get(&streams, a(0))?;
                let pos = get(&streams, a(1))?;
                if !matches!(x.kind, CKind::F | CKind::T)
                    || x.shape.len() != 2
                    || x.shape[1].is_none()
                    || x.shape[1].unwrap() % 2 != 0
                    || pos.kind != CKind::I
                    || pos.shape.len() != 1
                {
                    return Err(Error::unimplemented(format!(
                        "{where_}: ROTARY needs F/T (S,Deven) + I pos (S,)"
                    )));
                }
                if x.shape[0].is_some() && pos.shape[0] != x.shape[0] {
                    return Err(Error::asm(format!(
                        "{where_}: ROTARY pos len {:?} vs rows {:?}",
                        pos.shape, x.shape[0]
                    )));
                }
                streams.insert(o, x.clone());
            }
            "ROPE" => {
                let x = get(&streams, a(0))?;
                let pos = get(&streams, a(1))?;
                let nh = int_lit(a(2), &format!("{where_} nh"))?;
                let dh = int_lit(a(3), &format!("{where_} dh"))?;
                if x.kind != CKind::F
                    || x.shape.len() != 2
                    || pos.kind != CKind::I
                    || pos.shape.len() != 1
                {
                    return Err(Error::asm(format!("{where_}: ROPE needs 2D F + 1D I pos")));
                }
                if x.shape[1].is_some() && x.shape[1].unwrap() as i64 != nh * dh {
                    return Err(Error::asm(format!(
                        "{where_}: ROPE width {:?} != nh*dh",
                        x.shape[1]
                    )));
                }
                if dh % 2 != 0 {
                    return Err(Error::asm(format!(
                        "{where_}: ROPE needs even dh, got {dh}"
                    )));
                }
                if x.shape[0].is_some() && pos.shape[0] != x.shape[0] {
                    return Err(Error::asm(format!("{where_}: ROPE pos len vs rows")));
                }
                streams.insert(o, StreamInfo::f(vec![x.shape[0], Some((nh * dh) as usize)]));
            }
            "SWIGLU" => {
                let x = get(&streams, a(0))?;
                let inter = int_lit(a(1), &format!("{where_} inter"))?;
                if x.kind != CKind::F || x.shape.len() != 2 {
                    return Err(Error::asm(format!("{where_}: SWIGLU needs 2D F")));
                }
                if x.shape[1].is_some() && x.shape[1].unwrap() as i64 != 2 * inter {
                    return Err(Error::asm(format!("{where_}: SWIGLU width != 2*I")));
                }
                streams.insert(o, StreamInfo::f(vec![x.shape[0], Some(inter as usize)]));
            }
            "TRANSPOSE" => {
                let x = get(&streams, a(0))?;
                if !matches!(x.kind, CKind::F | CKind::T) || x.shape.len() != 2 {
                    return Err(Error::unimplemented(format!(
                        "{where_}: v0.2 TRANSPOSE handles 2D F/T"
                    )));
                }
                streams.insert(
                    o,
                    StreamInfo {
                        kind: x.kind,
                        shape: vec![x.shape[1], x.shape[0]],
                    },
                );
            }
            "TBETA" | "BETA" => {
                let r = get(&streams, a(0))?;
                let kind = if r.kind == CKind::I { CKind::F } else { r.kind };
                streams.insert(
                    o,
                    StreamInfo {
                        kind,
                        shape: r.shape.clone(),
                    },
                );
            }
            "MUL" | "ADD" | "SUB" | "DIV" => {
                let x = get(&streams, a(0))?;
                let y = get(&streams, a(1))?;
                if !matches!(x.kind, CKind::F | CKind::T)
                    || x.kind != y.kind
                    || !same_dims(&x.shape, &y.shape)
                {
                    return Err(Error::unimplemented(format!(
                        "{where_}: {} needs same-kind same-shape (no broadcast in v0.2)",
                        ins.mn
                    )));
                }
                streams.insert(o, x.clone());
            }
            "GT" => {
                let x = get(&streams, a(0))?;
                let y = get(&streams, a(1))?;
                if x.kind != CKind::T || y.kind != CKind::T || !same_dims(&x.shape, &y.shape) {
                    return Err(Error::unimplemented(format!(
                        "{where_}: GT needs same-shape T triples"
                    )));
                }
                streams.insert(o, StreamInfo::i(x.shape.clone()));
            }
            "SQUARE" => {
                let x = get(&streams, a(0))?;
                if x.kind != CKind::T {
                    return Err(Error::unimplemented(format!("{where_}: SQUARE needs T")));
                }
                streams.insert(o, x.clone());
            }
            "SELECT" => {
                let m = get(&streams, a(0))?;
                let x = get(&streams, a(1))?;
                let y = get(&streams, a(2))?;
                if m.kind != CKind::I
                    || !matches!(x.kind, CKind::F | CKind::T)
                    || x.kind != y.kind
                    || !same_dims(&x.shape, &y.shape)
                {
                    return Err(Error::unimplemented(format!(
                        "{where_}: SELECT needs I mask + same-kind same-shape branches"
                    )));
                }
                for (dm, da) in m.shape.iter().zip(x.shape.iter()) {
                    if dm != da && da.is_some() {
                        return Err(Error::asm(format!(
                            "{where_}: SELECT mask {:?} vs {:?}",
                            m.shape, x.shape
                        )));
                    }
                }
                if m.shape.len() != x.shape.len() {
                    return Err(Error::asm(format!("{where_}: SELECT rank mismatch")));
                }
                streams.insert(o, x.clone());
            }
            "TSHIFT" | "SOFTMAX_WIDE" => {
                let x = get(&streams, a(0))?;
                if !matches!(x.kind, CKind::F | CKind::T) || x.shape.len() != 2 {
                    return Err(Error::unimplemented(format!(
                        "{where_}: {} needs 2D F/T",
                        ins.mn
                    )));
                }
                streams.insert(o, x.clone());
            }
            "SILU" => {
                let x = get(&streams, a(0))?;
                if !matches!(x.kind, CKind::F | CKind::T) {
                    return Err(Error::unimplemented(format!("{where_}: SILU needs F/T")));
                }
                streams.insert(o, x.clone());
            }
            "GELU" | "RELU" => {
                // Shape-preserving elementwise activation. The interpreter
                // is the triple path; the Rust target lowers F.
                let x = get(&streams, a(0))?;
                if !matches!(x.kind, CKind::F | CKind::T) {
                    return Err(Error::unimplemented(format!(
                        "{where_}: {} needs F/T",
                        ins.mn
                    )));
                }
                streams.insert(o, x.clone());
            }
            "SOFTMAX" => {
                let x = get(&streams, a(0))?;
                if !matches!(x.kind, CKind::F | CKind::T) || x.shape.len() != 2 {
                    return Err(Error::unimplemented(format!(
                        "{where_}: SOFTMAX needs 2D F/T"
                    )));
                }
                streams.insert(o, x.clone());
            }
            "CONCAT" => {
                let x = get(&streams, a(0))?;
                let y = get(&streams, a(1))?;
                let ax = if streams.contains_key(a(2)) {
                    None
                } else {
                    Some(int_lit(a(2), &format!("{where_} axis"))?)
                };
                if !matches!(x.kind, CKind::F | CKind::T)
                    || x.kind != y.kind
                    || x.shape.len() != 2
                    || ax != Some(1)
                    || x.shape[0] != y.shape[0]
                    || x.shape[1].is_none()
                    || y.shape[1].is_none()
                {
                    return Err(Error::unimplemented(format!(
                        "{where_}: v0.2 CONCAT handles 2D ax1 same-kind, same rows, baked widths"
                    )));
                }
                streams.insert(
                    o,
                    StreamInfo {
                        kind: x.kind,
                        shape: vec![x.shape[0], Some(x.shape[1].unwrap() + y.shape[1].unwrap())],
                    },
                );
            }
            other => {
                return Err(Error::unimplemented(format!(
                    "C backend: no pattern for {other} ({where_})"
                )))
            }
        }
    }
    Ok(streams)
}
