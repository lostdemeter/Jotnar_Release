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
                        if s.kind != CKind::I
                            || !(same_dims(&s.shape, &x.shape) || s.shape.is_empty())
                        {
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
                    if s.kind != CKind::I || !(same_dims(&s.shape, &x.shape) || s.shape.is_empty())
                    {
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
                        if s.kind != CKind::I
                            || !(same_dims(&s.shape, &c.shape) || s.shape.is_empty())
                        {
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
                    if s.kind != CKind::I || !(same_dims(&s.shape, &x.shape) || s.shape.is_empty())
                    {
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
            "DECONV2D" => {
                let x = get(&streams, a(0))?;
                let w = get(&streams, a(1))?;
                let b = get(&streams, a(2))?;
                let stride = int_lit(a(3), &format!("{where_} stride"))?;
                let pad = int_lit(a(4), &format!("{where_} pad"))?;
                if !matches!(x.kind, CKind::F | CKind::T)
                    || x.kind != w.kind
                    || x.kind != b.kind
                    || x.shape.len() != 3
                    || w.shape.len() != 4
                    || b.shape.len() != 1
                {
                    return Err(Error::unimplemented(format!(
                        "{where_}: DECONV2D needs F/T x(H,W,C)+w(Cout,kh,kw,C)+b(Cout)"
                    )));
                }
                let (Some(h), Some(ww), Some(cin)) = (x.shape[0], x.shape[1], x.shape[2]) else {
                    return Err(Error::asm(format!("{where_}: DECONV2D needs static HWC")));
                };
                let (Some(cout), Some(kh), Some(kw), Some(wcin)) =
                    (w.shape[0], w.shape[1], w.shape[2], w.shape[3])
                else {
                    return Err(Error::asm(format!(
                        "{where_}: DECONV2D needs static weights"
                    )));
                };
                if wcin != cin || b.shape[0] != Some(cout) {
                    return Err(Error::asm(format!(
                        "{where_}: DECONV2D channel/bias mismatch"
                    )));
                }
                if stride <= 0 || pad < 0 {
                    return Err(Error::asm(format!("{where_}: DECONV2D bad stride/pad")));
                }
                let ho = (h as i64 - 1) * stride - 2 * pad + kh as i64;
                let wo = (ww as i64 - 1) * stride - 2 * pad + kw as i64;
                if ho <= 0 || wo <= 0 {
                    return Err(Error::asm(format!("{where_}: DECONV2D bad output size")));
                }
                streams.insert(
                    o,
                    StreamInfo {
                        kind: x.kind,
                        shape: vec![Some(ho as usize), Some(wo as usize), Some(cout)],
                    },
                );
            }
            "CONV2D" | "CONV2D_RELU" => {
                let x = get(&streams, a(0))?;
                let w = get(&streams, a(1))?;
                let b = get(&streams, a(2))?;
                let stride = int_lit(a(3), &format!("{where_} stride"))?;
                let pad = int_lit(a(4), &format!("{where_} pad"))?;
                if !matches!(x.kind, CKind::F | CKind::T)
                    || x.kind != w.kind
                    || x.kind != b.kind
                    || x.shape.len() != 3
                    || w.shape.len() != 4
                    || b.shape.len() != 1
                {
                    return Err(Error::unimplemented(format!(
                        "{where_}: CONV2D needs F/T x(H,W,C)+w(Cout,kh,kw,C)+b(Cout)"
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
                    StreamInfo {
                        kind: x.kind,
                        shape: vec![Some(ho as usize), Some(wo as usize), Some(cout)],
                    },
                );
            }
            "RESIZE2D" => {
                let x = get(&streams, a(0))?;
                let ho = int_lit(a(1), &format!("{where_} ho"))?;
                let wo = int_lit(a(2), &format!("{where_} wo"))?;
                let align = int_lit(a(3), &format!("{where_} align"))?;
                if !matches!(x.kind, CKind::F | CKind::T)
                    || x.shape.len() != 3
                    || ho <= 0
                    || wo <= 0
                {
                    return Err(Error::unimplemented(format!(
                        "{where_}: RESIZE2D needs F/T HWC + positive sizes"
                    )));
                }
                if align != 0 && align != 1 {
                    return Err(Error::asm(format!("{where_}: RESIZE2D align must be 0/1")));
                }
                streams.insert(
                    o,
                    StreamInfo {
                        kind: x.kind,
                        shape: vec![Some(ho as usize), Some(wo as usize), x.shape[2]],
                    },
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
            "MATMUL_T" => {
                let x = get(&streams, a(0))?;
                let y = get(&streams, a(1))?;
                if !matches!(x.kind, CKind::F | CKind::T) || x.kind != y.kind {
                    return Err(Error::unimplemented(format!(
                        "{where_}: MATMUL_T needs same-kind inputs"
                    )));
                }
                if x.shape.len() != 2 || y.shape.len() != 2 {
                    return Err(Error::asm(format!(
                        "{where_}: MATMUL_T needs 2D inputs, got {:?}/{:?}",
                        x.shape, y.shape
                    )));
                }
                if x.shape[1] != y.shape[1] {
                    return Err(Error::asm(format!(
                        "{where_}: MATMUL_T inner dims {:?} vs {:?}",
                        x.shape, y.shape
                    )));
                }
                streams.insert(
                    o,
                    StreamInfo {
                        kind: x.kind,
                        shape: vec![x.shape[0], y.shape[0]],
                    },
                );
            }
            "MATMUL_BIAS" => {
                let x = get(&streams, a(0))?;
                let y = get(&streams, a(1))?;
                let b = get(&streams, a(2))?;
                if x.kind != CKind::F || y.kind != CKind::F || b.kind != CKind::F {
                    return Err(Error::unimplemented(format!(
                        "{where_}: MATMUL_BIAS needs F inputs"
                    )));
                }
                let (Some(m), Some(k), Some(n)) = (
                    x.shape.first().copied().flatten(),
                    x.shape.get(1).copied().flatten(),
                    y.shape.get(1).copied().flatten(),
                ) else {
                    return Err(Error::asm(format!(
                        "{where_}: MATMUL_BIAS needs static 2D shapes"
                    )));
                };
                if x.shape.len() != 2 || y.shape.len() != 2 || y.shape[0] != Some(k) {
                    return Err(Error::asm(format!(
                        "{where_}: MATMUL_BIAS shape mismatch {:?}/{:?}",
                        x.shape, y.shape
                    )));
                }
                if b.shape.last().copied().flatten() != Some(n) || b.elems() != Some(n) {
                    return Err(Error::asm(format!(
                        "{where_}: MATMUL_BIAS bias must have {n} elements"
                    )));
                }
                streams.insert(o, StreamInfo::f(vec![Some(m), Some(n)]));
            }
            "MATMUL_BIAS_RESIDUAL" => {
                let x = get(&streams, a(0))?;
                let y = get(&streams, a(1))?;
                let b = get(&streams, a(2))?;
                let r = get(&streams, a(3))?;
                if x.kind != CKind::F || y.kind != CKind::F || b.kind != CKind::F || r.kind != CKind::F {
                    return Err(Error::unimplemented(format!(
                        "{where_}: MATMUL_BIAS_RESIDUAL needs F inputs"
                    )));
                }
                let (Some(m), Some(k), Some(n)) = (
                    x.shape.first().copied().flatten(),
                    x.shape.get(1).copied().flatten(),
                    y.shape.get(1).copied().flatten(),
                ) else {
                    return Err(Error::asm(format!(
                        "{where_}: MATMUL_BIAS_RESIDUAL needs static 2D shapes"
                    )));
                };
                if x.shape.len() != 2
                    || y.shape.len() != 2
                    || y.shape[0] != Some(k)
                    || r.shape != vec![Some(m), Some(n)]
                    || b.elems() != Some(n)
                {
                    return Err(Error::asm(format!(
                        "{where_}: MATMUL_BIAS_RESIDUAL shape mismatch"
                    )));
                }
                streams.insert(o, StreamInfo::f(vec![Some(m), Some(n)]));
            }
            "MATMUL_BIAS_GELU" => {
                let x = get(&streams, a(0))?;
                let y = get(&streams, a(1))?;
                let b = get(&streams, a(2))?;
                if x.kind != CKind::F || y.kind != CKind::F || b.kind != CKind::F {
                    return Err(Error::unimplemented(format!(
                        "{where_}: MATMUL_BIAS_GELU needs F inputs"
                    )));
                }
                let (Some(m), Some(k), Some(n)) = (
                    x.shape.first().copied().flatten(),
                    x.shape.get(1).copied().flatten(),
                    y.shape.get(1).copied().flatten(),
                ) else {
                    return Err(Error::asm(format!(
                        "{where_}: MATMUL_BIAS_GELU needs static 2D shapes"
                    )));
                };
                if x.shape.len() != 2 || y.shape.len() != 2 || y.shape[0] != Some(k)
                    || b.elems() != Some(n)
                {
                    return Err(Error::asm(format!(
                        "{where_}: MATMUL_BIAS_GELU shape mismatch"
                    )));
                }
                streams.insert(o, StreamInfo::f(vec![Some(m), Some(n)]));
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
            "SELF_ATTN" => {
                let q = get(&streams, a(0))?;
                let k = get(&streams, a(1))?;
                let v = get(&streams, a(2))?;
                if q.kind != CKind::F || k.kind != CKind::F || v.kind != CKind::F {
                    return Err(Error::unimplemented(format!(
                        "{where_}: SELF_ATTN needs F Q/K/V"
                    )));
                }
                let nh = int_lit(a(3), &format!("{where_} SELF_ATTN"))?;
                let dh = int_lit(a(4), &format!("{where_} SELF_ATTN"))?;
                if nh <= 0 || dh <= 0 {
                    return Err(Error::asm(format!(
                        "{where_}: SELF_ATTN needs positive nh/dh"
                    )));
                }
                if !is_literal(a(5)) {
                    return Err(Error::asm(format!(
                        "{where_}: SELF_ATTN scale must be a literal"
                    )));
                }
                let scale: f64 = a(5)
                    .parse()
                    .map_err(|_| Error::asm(format!("{where_}: SELF_ATTN bad scale")))?;
                if !scale.is_finite() {
                    return Err(Error::asm(format!(
                        "{where_}: SELF_ATTN scale must be finite"
                    )));
                }
                if q.shape.len() != 2 || k.shape != q.shape || v.shape != q.shape {
                    return Err(Error::asm(format!(
                        "{where_}: SELF_ATTN needs equal 2D Q/K/V"
                    )));
                }
                if q.shape[1] != Some((nh * dh) as usize) {
                    return Err(Error::asm(format!(
                        "{where_}: SELF_ATTN width {:?} != nh*dh {}",
                        q.shape[1],
                        nh * dh
                    )));
                }
                streams.insert(o, StreamInfo::f(q.shape.clone()));
            }
            "LINEAR_ATTN" => {
                let q = get(&streams, a(0))?;
                let k = get(&streams, a(1))?;
                let v = get(&streams, a(2))?;
                if q.kind != CKind::F || k.kind != CKind::F || v.kind != CKind::F {
                    return Err(Error::unimplemented(format!(
                        "{where_}: LINEAR_ATTN needs F Q/K/V"
                    )));
                }
                let nh = int_lit(a(12), &format!("{where_} LINEAR_ATTN nh"))?;
                let dh = int_lit(a(13), &format!("{where_} LINEAR_ATTN dh"))?;
                if nh <= 0 || dh <= 0 {
                    return Err(Error::asm(format!(
                        "{where_}: LINEAR_ATTN needs positive nh/dh"
                    )));
                }
                if q.shape.len() != 2 || k.shape != q.shape || v.shape != q.shape {
                    return Err(Error::asm(format!(
                        "{where_}: LINEAR_ATTN needs equal 2D Q/K/V"
                    )));
                }
                if q.shape[1] != Some((nh * dh) as usize) {
                    return Err(Error::asm(format!(
                        "{where_}: LINEAR_ATTN width {:?} != nh*dh {}",
                        q.shape[1],
                        nh * dh
                    )));
                }
                let w1 = get(&streams, a(3))?;
                let b1 = get(&streams, a(4))?;
                let w2 = get(&streams, a(5))?;
                let b2 = get(&streams, a(6))?;
                let lnq_w = get(&streams, a(7))?;
                let lnq_b = get(&streams, a(8))?;
                let lnk_w = get(&streams, a(9))?;
                let lnk_b = get(&streams, a(10))?;
                let temp = get(&streams, a(11))?;
                for (name, s) in [
                    ("W1", w1), ("B1", b1), ("W2", w2), ("B2", b2),
                    ("LNQ_W", lnq_w), ("LNQ_B", lnq_b),
                    ("LNK_W", lnk_w), ("LNK_B", lnk_b), ("TEMP", temp),
                ] {
                    if s.kind != CKind::F {
                        return Err(Error::asm(format!(
                            "{where_}: LINEAR_ATTN {name} must be F"
                        )));
                    }
                }
                if w1.shape.len() != 2 || w1.shape[0] != Some(dh as usize) {
                    return Err(Error::asm(format!(
                        "{where_}: LINEAR_ATTN W1 must be (dh,hid)"
                    )));
                }
                let hid = w1.shape[1].ok_or_else(|| {
                    Error::asm(format!("{where_}: LINEAR_ATTN W1 needs baked hid"))
                })?;
                if w2.shape.len() != 2 || w2.shape[0] != Some(hid) {
                    return Err(Error::asm(format!(
                        "{where_}: LINEAR_ATTN W2 must be (hid,m)"
                    )));
                }
                let m = w2.shape[1].ok_or_else(|| {
                    Error::asm(format!("{where_}: LINEAR_ATTN W2 needs baked m"))
                })?;
                if b1.shape != vec![Some(hid)]
                    || b2.shape != vec![Some(m)]
                    || lnq_w.shape != vec![Some(dh as usize)]
                    || lnq_b.shape != vec![Some(dh as usize)]
                    || lnk_w.shape != vec![Some(dh as usize)]
                    || lnk_b.shape != vec![Some(dh as usize)]
                    || temp.shape != vec![Some(nh as usize)]
                {
                    return Err(Error::asm(format!(
                        "{where_}: LINEAR_ATTN bias/norm/temp shapes do not match"
                    )));
                }
                streams.insert(o, StreamInfo::f(q.shape.clone()));
            }
            "LUMA" => {
                let x = get(&streams, a(0))?;
                if x.kind != CKind::F
                    || x.shape.len() != 3
                    || x.shape[2].is_none()
                    || x.shape[2].unwrap() < 3
                {
                    return Err(Error::unimplemented(format!(
                        "{where_}: LUMA needs F HWC with C>=3"
                    )));
                }
                streams.insert(o, StreamInfo::f(vec![x.shape[0], x.shape[1]]));
            }
            "STATIC" => {
                let x = get(&streams, a(0))?;
                if x.kind != CKind::F
                    || x.shape.len() != 3
                    || x.shape[2].is_none()
                    || x.shape[2].unwrap() < 2
                {
                    return Err(Error::unimplemented(format!(
                        "{where_}: STATIC needs F HW2"
                    )));
                }
                streams.insert(o, StreamInfo::i(vec![x.shape[0], x.shape[1]]));
            }
            "GAIN" => {
                let lin = get(&streams, a(0))?;
                let y = get(&streams, a(1))?;
                let e = get(&streams, a(2))?;
                if lin.kind != CKind::F
                    || lin.shape.len() != 3
                    || lin.shape[2].is_none()
                    || lin.shape[2].unwrap() < 3
                    || y.kind != CKind::F
                    || e.kind != CKind::T
                    || y.shape != lin.shape[..2].to_vec()
                    || e.shape != y.shape
                {
                    return Err(Error::unimplemented(format!(
                        "{where_}: GAIN needs F HWC + F HW + T HW"
                    )));
                }
                streams.insert(o, lin.clone());
            }
            "POOLAVG" => {
                let x = get(&streams, a(0))?;
                if x.kind != CKind::T || x.shape.len() != 3 {
                    return Err(Error::unimplemented(format!(
                        "{where_}: POOLAVG needs T HWC"
                    )));
                }
                streams.insert(o, StreamInfo::t(vec![x.shape[2]]));
            }
            "DECONV" => {
                let x = get(&streams, a(0))?;
                let w = get(&streams, a(1))?;
                let stride = int_lit(a(2), &format!("{where_} stride"))?;
                let pad = int_lit(a(3), &format!("{where_} pad"))?;
                if x.kind != CKind::T
                    || w.kind != CKind::T
                    || x.shape.len() != 3
                    || w.shape.len() != 4
                {
                    return Err(Error::unimplemented(format!(
                        "{where_}: DECONV needs T x(H,W,C)+w(Cout,kh,kw,C)"
                    )));
                }
                let (Some(h), Some(img_w), Some(cin)) = (x.shape[0], x.shape[1], x.shape[2]) else {
                    return Err(Error::asm(format!("{where_}: DECONV needs static HWC")));
                };
                let (Some(cout), Some(kh), Some(kw), Some(wcin)) =
                    (w.shape[0], w.shape[1], w.shape[2], w.shape[3])
                else {
                    return Err(Error::asm(format!("{where_}: DECONV needs static weights")));
                };
                if wcin != cin {
                    return Err(Error::asm(format!(
                        "{where_}: DECONV weight C {wcin} != input C {cin}"
                    )));
                }
                if stride <= 0 || pad < 0 {
                    return Err(Error::asm(format!("{where_}: DECONV bad stride/pad")));
                }
                let ho = (h as i64 - 1) * stride - 2 * pad + kh as i64;
                let wo = (img_w as i64 - 1) * stride - 2 * pad + kw as i64;
                if ho <= 0 || wo <= 0 {
                    return Err(Error::asm(format!("{where_}: DECONV bad output size")));
                }
                streams.insert(
                    o,
                    StreamInfo::t(vec![Some(ho as usize), Some(wo as usize), Some(cout)]),
                );
            }
            "PERMUTE3" => {
                let x = get(&streams, a(0))?;
                if x.kind != CKind::T || x.shape.len() != 3 {
                    return Err(Error::unimplemented(format!(
                        "{where_}: PERMUTE3 needs T 3D"
                    )));
                }
                let mut perm: Vec<usize> = (1..=3)
                    .map(|i| int_lit(a(i), &format!("{where_} perm")).map(|v| v as usize))
                    .collect::<Result<_>>()?;
                let mut sorted = perm.clone();
                sorted.sort_unstable();
                if sorted != [0, 1, 2] {
                    return Err(Error::asm(format!(
                        "{where_}: PERMUTE3 needs a permutation of (0,1,2), got {perm:?}"
                    )));
                }
                let shape: Vec<Option<usize>> = perm.iter().map(|&i| x.shape[i]).collect();
                streams.insert(o, StreamInfo::t(shape));
            }
            "RESHAPE2" | "RESHAPE3" => {
                let x = get(&streams, a(0))?;
                let n = x.elems().ok_or_else(|| {
                    Error::unimplemented(format!("{where_}: RESHAPE needs static numel"))
                })?;
                let dims: Vec<usize> = if ins.mn == "RESHAPE2" {
                    vec![
                        int_lit(a(1), &format!("{where_} d0"))? as usize,
                        int_lit(a(2), &format!("{where_} d1"))? as usize,
                    ]
                } else {
                    vec![
                        int_lit(a(1), &format!("{where_} d0"))? as usize,
                        int_lit(a(2), &format!("{where_} d1"))? as usize,
                        int_lit(a(3), &format!("{where_} d2"))? as usize,
                    ]
                };
                if dims.iter().product::<usize>() != n {
                    return Err(Error::asm(format!(
                        "{where_}: reshape {n} elements into {dims:?}"
                    )));
                }
                streams.insert(
                    o,
                    StreamInfo {
                        kind: x.kind,
                        shape: dims.into_iter().map(Some).collect(),
                    },
                );
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
            "ISCATTER1" | "ISCATTER_MIN1" => {
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
                        "{where_}: ISCATTER1/ISCATTER_MIN1 needs 1D I OUT/IDS/VALS"
                    )));
                }
                if ids.shape[0].is_some()
                    && vals.shape[0].is_some()
                    && ids.shape[0] != vals.shape[0]
                {
                    return Err(Error::asm(format!(
                        "{where_}: ISCATTER1/ISCATTER_MIN1 IDS {:?} vs VALS {:?}",
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
                if !matches!(x.kind, CKind::F | CKind::T) || x.kind != y.kind {
                    return Err(Error::unimplemented(format!(
                        "{where_}: CONCAT needs same-kind F/T"
                    )));
                }
                if x.shape.len() == 2 && y.shape.len() == 2 && ax == Some(1) {
                    if x.shape[0] != y.shape[0]
                        || x.shape[1].is_none()
                        || y.shape[1].is_none()
                    {
                        return Err(Error::unimplemented(format!(
                            "{where_}: 2D CONCAT needs same rows and baked widths"
                        )));
                    }
                    streams.insert(
                        o,
                        StreamInfo {
                            kind: x.kind,
                            shape: vec![
                                x.shape[0],
                                Some(x.shape[1].unwrap() + y.shape[1].unwrap()),
                            ],
                        },
                    );
                } else if x.shape.len() == 2 && y.shape.len() == 2 && ax == Some(0) {
                    // Row concat (prepend/append). Used by the DAV2 patch-embed
                    // prefix to place the CLS token before the patch tokens.
                    if x.shape[1].is_none()
                        || y.shape[1].is_none()
                        || x.shape[1] != y.shape[1]
                        || x.shape[0].is_none()
                        || y.shape[0].is_none()
                    {
                        return Err(Error::unimplemented(format!(
                            "{where_}: 2D axis-0 CONCAT needs same baked width"
                        )));
                    }
                    streams.insert(
                        o,
                        StreamInfo {
                            kind: x.kind,
                            shape: vec![
                                Some(x.shape[0].unwrap() + y.shape[0].unwrap()),
                                x.shape[1],
                            ],
                        },
                    );
                } else if x.shape.len() == 3 && y.shape.len() == 3 && ax == Some(2) {
                    if x.shape[0] != y.shape[0] || x.shape[1] != y.shape[1] {
                        return Err(Error::unimplemented(format!(
                            "{where_}: HWC CONCAT needs same H/W"
                        )));
                    }
                    if x.shape[2].is_none() || y.shape[2].is_none() {
                        return Err(Error::unimplemented(format!(
                            "{where_}: HWC CONCAT needs baked channels"
                        )));
                    }
                    streams.insert(
                        o,
                        StreamInfo {
                            kind: x.kind,
                            shape: vec![
                                x.shape[0],
                                x.shape[1],
                                Some(x.shape[2].unwrap() + y.shape[2].unwrap()),
                            ],
                        },
                    );
                } else {
                    return Err(Error::unimplemented(format!(
                        "{where_}: CONCAT handles 2D ax0/ax1 or HWC ax2, same-kind"
                    )));
                }
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::asm::{assemble, AsmOptions};
    use crate::lattice;
    use crate::value::array_from_vec;
    use ndarray::{ArrayD, IxDyn};

    fn f(shape: &[usize], vals: &[f64]) -> Value {
        Value::F64(array_from_vec(shape, vals.to_vec()).unwrap())
    }

    fn t(shape: &[usize], vals: &[f64]) -> Value {
        let a = ArrayD::from_shape_vec(IxDyn(shape), vals.to_vec()).unwrap();
        Value::Triple(lattice::encode(&a))
    }

    fn feed(items: &[(&str, Value)]) -> Feeds {
        let mut feeds = Feeds::new();
        for (name, v) in items {
            feeds.insert((*name).to_string(), v.clone());
        }
        feeds
    }

    fn infer(text: &str, feeds: &Feeds) -> HashMap<String, StreamInfo> {
        let bound = assemble(text, &AsmOptions::default()).unwrap();
        infer_shapes(&bound, feeds).unwrap()
    }

    #[test]
    fn luma_static_and_gain_infer_their_shapes() {
        let s = infer(
            "IN X\nOUT = LUMA(X)\n",
            &feed(&[("X", f(&[2, 3, 4], &vec![0.5; 24]))]),
        );
        assert_eq!(s["OUT"].kind, CKind::F);
        assert_eq!(s["OUT"].shape, vec![Some(2), Some(3)]);

        let s = infer(
            "IN X\nOUT = STATIC(X)\n",
            &feed(&[("X", f(&[2, 3, 2], &vec![0.0; 12]))]),
        );
        assert_eq!(s["OUT"].kind, CKind::I);
        assert_eq!(s["OUT"].shape, vec![Some(2), Some(3)]);

        let s = infer(
            "IN LIN\nIN Y\nIN E\nOUT = GAIN(LIN, Y, E)\n",
            &feed(&[
                ("LIN", f(&[2, 3, 4], &vec![0.5; 24])),
                ("Y", f(&[2, 3], &vec![0.5; 6])),
                ("E", t(&[2, 3], &vec![1.0; 6])),
            ]),
        );
        assert_eq!(s["OUT"].kind, CKind::F);
        assert_eq!(s["OUT"].shape, vec![Some(2), Some(3), Some(4)]);
    }

    #[test]
    fn pool_deconv_and_permute_infer_their_shapes() {
        let s = infer(
            "IN X\nOUT = POOLAVG(X)\n",
            &feed(&[("X", t(&[2, 3, 4], &vec![0.5; 24]))]),
        );
        assert_eq!(s["OUT"].kind, CKind::T);
        assert_eq!(s["OUT"].shape, vec![Some(4)]);

        let s = infer(
            "IN X\nIN W\nOUT = DECONV(X, W, 2, 1)\n",
            &feed(&[
                ("X", t(&[2, 3, 2], &vec![0.5; 12])),
                ("W", t(&[5, 3, 3, 2], &vec![0.25; 90])),
            ]),
        );
        assert_eq!(s["OUT"].kind, CKind::T);
        assert_eq!(s["OUT"].shape, vec![Some(3), Some(5), Some(5)]);

        let s = infer(
            "IN X\nOUT = PERMUTE3(X, 2, 0, 1)\n",
            &feed(&[("X", t(&[2, 3, 4], &vec![0.5; 24]))]),
        );
        assert_eq!(s["OUT"].kind, CKind::T);
        assert_eq!(s["OUT"].shape, vec![Some(4), Some(2), Some(3)]);
    }

    #[test]
    fn scalar_stream_broadcasts_like_int_operand() {
        let mut feeds = feed(&[(
            "X",
            Value::I64(array_from_vec(&[4], vec![1, 2, 3, 4]).unwrap()),
        )]);
        feeds.insert(
            "S".to_string(),
            Value::I64(array_from_vec(&[], vec![-3]).unwrap()),
        );
        let bound = assemble("IN X\nIN S\nOUT = IADD(X, S)\n", &AsmOptions::default()).unwrap();
        let s = infer_shapes(&bound, &feeds).unwrap();
        assert_eq!(s["OUT"].kind, CKind::I);
        assert_eq!(s["OUT"].shape, vec![Some(4)]);
    }

    #[test]
    fn new_rules_refuse_the_wrong_kind_or_permutation() {
        let bound = assemble("IN X\nOUT = LUMA(X)\n", &AsmOptions::default()).unwrap();
        let feeds = feed(&[("X", t(&[2, 3, 4], &vec![0.5; 24]))]);
        assert!(infer_shapes(&bound, &feeds).is_err());

        let bound = assemble("IN X\nOUT = PERMUTE3(X, 0, 0, 1)\n", &AsmOptions::default()).unwrap();
        let feeds = feed(&[("X", t(&[2, 3, 4], &vec![0.5; 24]))]);
        assert!(infer_shapes(&bound, &feeds).is_err());
    }
}
