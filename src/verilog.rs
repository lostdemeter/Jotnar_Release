//! Combinational/sequential Verilog backend for the supported integer subset.
//!
//! Port of `chain/verilog.py` (Jotnar-MAC structural representation). Widths
//! are caller-declared; intermediate widths are inferred. The simulator here
//! evaluates the emitted plan directly and is checked against the reference
//! ISA model for the CPU8 program.

use std::collections::{HashMap, HashSet};

use regex::Regex;

use crate::error::{Error, Result};

type Widths = HashMap<String, (usize, usize)>;

#[derive(Clone, Debug)]
pub struct Memory {
    pub name: String,
    pub depth: usize,
    pub width: usize,
    pub raddr: String,
    pub rout: String,
    pub we: String,
    pub waddr: String,
    pub wdata: String,
    pub init: Vec<u64>,
    pub raddr2: Option<String>,
    pub rout2: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum VArg {
    Ref(String),
    Int(i64),
}

#[derive(Clone, Debug)]
pub struct VOp {
    pub out: String,
    pub mn: String,
    pub args: Vec<VArg>,
}

const BINOPS: [&str; 3] = ["IXOR", "IAND", "IOR"];
const ARITH: [&str; 4] = ["IADD", "ISUB", "IMUL", "IMOD"];
const SHIFTS: [&str; 2] = ["ISHL", "ISHR"];
const KNOWN: [&str; 12] = [
    "IXOR", "IAND", "IOR", "IADD", "ISUB", "IMUL", "IMOD", "ISHL", "ISHR", "ICRC32", "IEQ", "ILT",
];

fn known(mn: &str) -> bool {
    KNOWN.contains(&mn) || mn == "ISELECT"
}

pub fn parse_program(text: &str) -> Result<(Vec<String>, Vec<VOp>)> {
    let mut inputs = Vec::new();
    let mut ops = Vec::new();
    let re = Regex::new(r"^([A-Za-z_]\w*)\s*=\s*([A-Za-z_]\w*)\s*\((.*)\)$").unwrap();
    for raw in text.lines() {
        let line = raw.split('#').next().unwrap_or("").trim();
        if line.is_empty() {
            continue;
        }
        if line.to_ascii_uppercase().starts_with("IN ") {
            let name = line[3..].trim();
            if !Regex::new(r"^[A-Za-z_]\w*$").unwrap().is_match(name) {
                return Err(Error::asm(format!("bad IN name: {line:?}")));
            }
            inputs.push(name.to_string());
            continue;
        }
        let m = re
            .captures(line)
            .ok_or_else(|| Error::asm(format!("Verilog: unsupported line {line:?}")))?;
        let out = m[1].to_string();
        let mn = m[2].to_ascii_uppercase();
        if !known(&mn) {
            return Err(Error::asm(format!("Verilog: {mn}: no Verilog pattern")));
        }
        let mut args = Vec::new();
        for tok in m[3].split(',').map(str::trim) {
            if tok.is_empty() {
                return Err(Error::asm(format!("Verilog: {mn}: empty argument")));
            }
            if let Ok(v) = tok.parse::<i64>() {
                args.push(VArg::Int(v));
            } else if Regex::new(r"^[A-Za-z_]\w*$").unwrap().is_match(tok) {
                args.push(VArg::Ref(tok.to_string()));
            } else {
                return Err(Error::asm(format!(
                    "Verilog: {mn}: unsupported argument {tok:?}"
                )));
            }
        }
        ops.push(VOp { out, mn, args });
    }
    Ok((inputs, ops))
}

fn need(widths: &Widths, name: &str, where_: &str) -> Result<(usize, usize)> {
    let (n, w) = *widths
        .get(name)
        .ok_or_else(|| Error::asm(format!("Verilog: {where_}: width of '{name}' not declared")))?;
    if n == 0 || w == 0 {
        return Err(Error::asm(format!(
            "Verilog: {where_}: bad width for '{name}'"
        )));
    }
    Ok((n, w))
}

fn plan(
    inputs: &[String],
    ops: &[VOp],
    widths: &Widths,
    seed: &HashMap<String, (usize, usize)>,
) -> Result<(HashMap<String, (usize, usize)>, Vec<(VOp, (usize, usize))>)> {
    let mut wire: HashMap<String, (usize, usize)> = seed.clone();
    for n in inputs {
        if let Some(&got) = wire.get(n) {
            if let Some(&decl) = widths.get(n) {
                if decl != got {
                    return Err(Error::asm(format!(
                        "Verilog: {n}: declared width {decl:?} != state width {got:?}"
                    )));
                }
            }
            continue;
        }
        wire.insert(n.clone(), need(widths, n, "input")?);
    }
    let mut out = Vec::new();
    for op in ops {
        if wire.contains_key(&op.out) {
            return Err(Error::asm(format!(
                "Verilog: {}: reassignment is not supported; use a fresh name",
                op.out
            )));
        }
        let refs: Vec<&str> = op
            .args
            .iter()
            .filter_map(|a| match a {
                VArg::Ref(r) => Some(r.as_str()),
                VArg::Int(_) => None,
            })
            .collect();
        for r in &refs {
            if !wire.contains_key(*r) {
                return Err(Error::asm(format!(
                    "Verilog: {}: unknown stream '{r}'",
                    op.mn
                )));
            }
        }
        let shape = if op.mn == "ICRC32" {
            if op.args.len() != 1 || !matches!(op.args[0], VArg::Ref(_)) {
                return Err(Error::asm("Verilog: ICRC32 expected one byte vector"));
            }
            let (sn, sw) = wire[refs[0]];
            if sw != 8 || sn == 0 {
                return Err(Error::asm(
                    "Verilog: ICRC32 input must be non-empty 8-bit lanes",
                ));
            }
            (1, 32)
        } else if ARITH.contains(&op.mn.as_str()) || SHIFTS.contains(&op.mn.as_str()) {
            if refs.is_empty() {
                return Err(Error::asm(format!(
                    "Verilog: {}: needs at least one stream",
                    op.mn
                )));
            }
            let (lanes, w) = wire[refs[0]];
            if lanes != 1 {
                return Err(Error::asm(format!(
                    "Verilog: {}: multi-lane arithmetic/shift would cross lanes",
                    op.mn
                )));
            }
            for r in &refs[1..] {
                if wire[*r] != (1, w) {
                    return Err(Error::asm(format!(
                        "Verilog: {}: lane width mismatch on {r}",
                        op.mn
                    )));
                }
            }
            (1, w)
        } else if op.mn == "IEQ" || op.mn == "ILT" {
            if refs.is_empty() {
                return Err(Error::asm(format!(
                    "Verilog: {}: needs a stream operand",
                    op.mn
                )));
            }
            let (lanes, w) = wire[refs[0]];
            if lanes != 1 {
                return Err(Error::asm(format!("Verilog: {}: single lane only", op.mn)));
            }
            for r in &refs[1..] {
                if wire[*r] != (1, w) {
                    return Err(Error::asm(format!(
                        "Verilog: {}: width mismatch on {r}",
                        op.mn
                    )));
                }
            }
            (1, w)
        } else if op.mn == "ISELECT" {
            if op.args.len() != 3 || !matches!(op.args[0], VArg::Ref(_)) {
                return Err(Error::asm("Verilog: ISELECT expected three stream args"));
            }
            let c = match &op.args[0] {
                VArg::Ref(r) => r,
                _ => unreachable!(),
            };
            let (lanes, w) = wire[c];
            if lanes != 1 {
                return Err(Error::asm("Verilog: ISELECT single lane only"));
            }
            for a in &op.args[1..] {
                if let VArg::Ref(r) = a {
                    if wire[r] != (1, w) {
                        return Err(Error::asm(format!(
                            "Verilog: ISELECT width mismatch on {r}"
                        )));
                    }
                }
            }
            (1, w)
        } else {
            if refs.is_empty() {
                return Err(Error::asm(format!(
                    "Verilog: {}: needs at least one stream",
                    op.mn
                )));
            }
            let shape = wire[refs[0]];
            for r in &refs[1..] {
                if wire[*r] != shape {
                    return Err(Error::asm(format!(
                        "Verilog: {}: lane width mismatch on {r}",
                        op.mn
                    )));
                }
            }
            shape
        };
        if let Some(&decl) = widths.get(&op.out) {
            if decl != shape {
                return Err(Error::asm(format!(
                    "Verilog: {}: declared width {decl:?} != inferred {shape:?}",
                    op.out
                )));
            }
        }
        wire.insert(op.out.clone(), shape);
        out.push((op.clone(), shape));
    }
    Ok((wire, out))
}

fn lane(name: &str, i: usize, w: usize) -> String {
    format!("{name}[{}:{}]", i * w + w - 1, i * w)
}

pub fn emit(
    text: &str,
    widths: &Widths,
    module: &str,
    state: &HashMap<String, (usize, u64, String)>,
    mem: &[Memory],
) -> Result<String> {
    let (inputs, ops) = parse_program(text)?;
    if inputs.is_empty() {
        return Err(Error::asm("Verilog: no IN streams"));
    }
    let outs: Vec<String> = ops.iter().map(|o| o.out.clone()).collect();
    if outs.iter().collect::<HashSet<_>>().len() != outs.len() {
        return Err(Error::asm("Verilog: duplicate output name"));
    }
    for m in mem {
        if m.depth == 0 || m.depth & (m.depth - 1) != 0 {
            return Err(Error::asm(format!(
                "Verilog: memory '{}': depth must be a power of two",
                m.name
            )));
        }
        if m.width == 0 {
            return Err(Error::asm(format!(
                "Verilog: memory '{}': bad width",
                m.name
            )));
        }
        if m.init.len() > m.depth {
            return Err(Error::asm(format!(
                "Verilog: memory '{}': init too long",
                m.name
            )));
        }
        if m.raddr2.is_some() != m.rout2.is_some() {
            return Err(Error::asm(format!(
                "Verilog: memory '{}': raddr2/rout2 must both be set",
                m.name
            )));
        }
    }
    for (n, (w, init, _)) in state {
        if *w == 0 {
            return Err(Error::asm(format!("Verilog: state '{n}': bad width")));
        }
        if *init >= (1u64 << *w) {
            return Err(Error::asm(format!(
                "Verilog: state '{n}': init outside [0, 2^{w})"
            )));
        }
    }
    let mut seed: HashMap<String, (usize, usize)> = state
        .iter()
        .map(|(n, (w, _, _))| (n.clone(), (1, *w)))
        .collect();
    for m in mem {
        seed.insert(m.rout.clone(), (1, m.width));
        if let Some(r) = &m.rout2 {
            seed.insert(r.clone(), (1, m.width));
        }
    }
    let (wire, planned) = plan(&inputs, &ops, widths, &seed)?;
    for (n, (w, _, nxt)) in state {
        if !inputs.contains(n) {
            return Err(Error::asm(format!(
                "Verilog: state '{n}' must be declared IN"
            )));
        }
        if !wire.contains_key(nxt) {
            return Err(Error::asm(format!(
                "Verilog: state '{n}': next wire '{nxt}' not computed"
            )));
        }
        if wire[nxt] != (1, *w) {
            return Err(Error::asm(format!(
                "Verilog: state '{n}': next wire '{nxt}' shape mismatch"
            )));
        }
    }
    let mut routs: HashSet<String> = mem.iter().map(|m| m.rout.clone()).collect();
    for m in mem {
        if let Some(r) = &m.rout2 {
            routs.insert(r.clone());
        }
        for name in std::iter::once(&m.rout).chain(m.rout2.iter()) {
            if !inputs.contains(name) {
                return Err(Error::asm(format!(
                    "Verilog: memory '{}': read output '{name}' must be declared IN",
                    m.name
                )));
            }
            if state.contains_key(name) || outs.contains(name) {
                return Err(Error::asm(format!(
                    "Verilog: memory '{}': read output '{name}' conflicts",
                    m.name
                )));
            }
        }
        let addr_w = (m.depth - 1).ilog2() as usize + 1;
        let checks = [
            ("read address", Some(m.raddr.as_str())),
            ("write enable", Some(m.we.as_str())),
            ("write address", Some(m.waddr.as_str())),
            ("write data", Some(m.wdata.as_str())),
            ("read2 address", m.raddr2.as_deref()),
        ];
        for (label, name) in checks {
            if let Some(name) = name {
                if !wire.contains_key(name) {
                    return Err(Error::asm(format!(
                        "Verilog: memory '{}': {label} '{name}' is not a known stream",
                        m.name
                    )));
                }
            }
        }
        for (label, name) in [
            ("read address", Some(m.raddr.as_str())),
            ("write address", Some(m.waddr.as_str())),
            ("read2 address", m.raddr2.as_deref()),
        ] {
            if let Some(name) = name {
                let (lanes, w) = wire[name];
                if lanes != 1 || w < addr_w {
                    return Err(Error::asm(format!(
                        "Verilog: memory '{}': {label} '{name}' needs one lane of >= {addr_w} bits",
                        m.name
                    )));
                }
            }
        }
        if wire[&m.wdata] != (1, m.width) {
            return Err(Error::asm(format!(
                "Verilog: memory '{}': write data width mismatch",
                m.name
            )));
        }
    }

    let mut lines = vec![format!("module {module} (")];
    let mut ports = Vec::new();
    if !state.is_empty() || !mem.is_empty() {
        ports.push("  input  wire clk".to_string());
        ports.push("  input  wire rst".to_string());
    }
    for n in &inputs {
        if state.contains_key(n) || routs.contains(n) {
            continue;
        }
        let (lanes, w) = wire[n];
        ports.push(format!("  input  wire [{}:0] {n}", lanes * w - 1));
    }
    for o in &outs {
        let (lanes, w) = wire[o];
        ports.push(format!("  output wire [{}:0] {o}", lanes * w - 1));
    }
    lines.push(ports.join(",\n"));
    lines.push(");".to_string());
    let mut body = Vec::new();
    for (n, (w, _, _)) in state {
        body.push(format!("  reg [{}:0] {n};", w - 1));
    }
    for m in mem {
        body.push(format!(
            "  reg [{}:0] {} [0:{}];",
            m.width - 1,
            m.name,
            m.depth - 1
        ));
    }
    if mem.iter().any(|m| !m.init.is_empty()) {
        body.push("  initial begin".to_string());
        for m in mem {
            for (i, v) in m.init.iter().enumerate() {
                body.push(format!(
                    "    {}[{i}] = {}'h{:x};",
                    m.name,
                    m.width,
                    v & ((1u64 << m.width) - 1)
                ));
            }
        }
        body.push("  end".to_string());
    }
    for m in mem {
        let addr_w = (m.depth - 1).ilog2() as usize + 1;
        let idx = |name: &str| {
            if wire[name].1 == addr_w {
                name.to_string()
            } else {
                format!("{name}[{}:0]", addr_w - 1)
            }
        };
        body.push(format!("  wire [{}:0] {};", m.width - 1, m.rout));
        body.push(format!(
            "  assign {} = {}[{}];",
            m.rout,
            m.name,
            idx(&m.raddr)
        ));
        if let (Some(a), Some(r)) = (&m.raddr2, &m.rout2) {
            body.push(format!("  wire [{}:0] {r};", m.width - 1));
            body.push(format!("  assign {r} = {}[{}];", m.name, idx(a)));
        }
    }
    for (op, (lanes, w)) in &planned {
        if op.mn == "ICRC32" {
            let src = match &op.args[0] {
                VArg::Ref(r) => r,
                _ => unreachable!(),
            };
            let (sn, _) = wire[src];
            body.push(format!("  wire [31:0] {}_c0;", op.out));
            body.push(format!("  assign {}_c0 = 32'hFFFFFFFF;", op.out));
            let mut prev = format!("{}_c0", op.out);
            for i in 0..sn {
                let t = format!("{}_x{i}", op.out);
                body.push(format!("  wire [31:0] {t};"));
                body.push(format!("  assign {t} = {prev} ^ {};", lane(src, i, 8)));
                prev = t;
                for j in 0..8 {
                    let nxt = format!("{}_r{i}_{j}", op.out);
                    body.push(format!("  wire [31:0] {nxt};"));
                    body.push(format!(
                        "  assign {nxt} = {prev}[0] ? (({prev} >> 1) ^ 32'hEDB88320) : ({prev} >> 1);"
                    ));
                    prev = nxt;
                }
            }
            body.push(format!("  wire [31:0] {};", op.out));
            body.push(format!("  assign {} = {prev} ^ 32'hFFFFFFFF;", op.out));
            continue;
        }
        if ARITH.contains(&op.mn.as_str()) || SHIFTS.contains(&op.mn.as_str()) {
            let (a, b) = (&op.args[0], &op.args[1]);
            let opstr = match op.mn.as_str() {
                "IADD" => "+",
                "ISUB" => "-",
                "IMUL" => "*",
                "IMOD" => "%",
                "ISHL" => "<<",
                _ => ">>",
            };
            let lhs = match a {
                VArg::Ref(r) => r.clone(),
                VArg::Int(v) => v.to_string(),
            };
            let rhs = match b {
                VArg::Ref(r) => r.clone(),
                VArg::Int(v) if SHIFTS.contains(&op.mn.as_str()) => {
                    if *v < 0 || *v >= *w as i64 {
                        return Err(Error::asm(format!(
                            "Verilog: {}: shift literal {v} outside [0,{w})",
                            op.mn
                        )));
                    }
                    v.to_string()
                }
                VArg::Int(v) => format!("{w}'h{:x}", (*v as u64) & ((1 << w) - 1)),
            };
            body.push(format!("  wire [{}:0] {};", w - 1, op.out));
            body.push(format!("  assign {} = {lhs} {opstr} {rhs};", op.out));
            continue;
        }
        if op.mn == "IEQ" || op.mn == "ILT" {
            let a = match &op.args[0] {
                VArg::Ref(r) => r.clone(),
                VArg::Int(v) => v.to_string(),
            };
            let b = match &op.args[1] {
                VArg::Ref(r) => r.clone(),
                VArg::Int(v) => format!("{w}'h{:x}", (*v as u64) & ((1 << w) - 1)),
            };
            let opstr = if op.mn == "IEQ" { "==" } else { "<" };
            body.push(format!("  wire [{}:0] {};", w - 1, op.out));
            body.push(format!(
                "  assign {} = ({a} {opstr} {b}) ? 1'b1 : 1'b0;",
                op.out
            ));
            continue;
        }
        if op.mn == "ISELECT" {
            let c = match &op.args[0] {
                VArg::Ref(r) => r.clone(),
                _ => unreachable!(),
            };
            let ta = match &op.args[1] {
                VArg::Ref(r) => r.clone(),
                VArg::Int(v) => format!("{w}'h{:x}", (*v as u64) & ((1 << w) - 1)),
            };
            let tb = match &op.args[2] {
                VArg::Ref(r) => r.clone(),
                VArg::Int(v) => format!("{w}'h{:x}", (*v as u64) & ((1 << w) - 1)),
            };
            body.push(format!("  wire [{}:0] {};", w - 1, op.out));
            body.push(format!("  assign {} = {c} ? {ta} : {tb};", op.out));
            continue;
        }
        let a = match &op.args[0] {
            VArg::Ref(r) => r.clone(),
            _ => unreachable!(),
        };
        let b = match &op.args[1] {
            VArg::Ref(r) => r.clone(),
            VArg::Int(v) => format!(
                "{}'h{:x}",
                lanes * w,
                (*v as u64) & ((1u64 << (lanes * w)) - 1)
            ),
        };
        let opstr = match op.mn.as_str() {
            "IXOR" => "^",
            "IAND" => "&",
            _ => "|",
        };
        body.push(format!("  wire [{}:0] {};", lanes * w - 1, op.out));
        body.push(format!("  assign {} = {a} {opstr} {b};", op.out));
    }
    if !state.is_empty() || !mem.is_empty() {
        body.push("  always @(posedge clk) begin".to_string());
        body.push("    if (rst) begin".to_string());
        for (n, (w, init, _)) in state {
            body.push(format!("      {n} <= {w}'h{init:x};"));
        }
        body.push("    end else begin".to_string());
        for (n, (_, _, nxt)) in state {
            body.push(format!("      {n} <= {nxt};"));
        }
        for m in mem {
            let aw = (m.depth - 1).ilog2() as usize + 1;
            let addr = if wire[&m.waddr].1 == aw {
                m.waddr.clone()
            } else {
                format!("{}[{}:0]", m.waddr, aw - 1)
            };
            body.push(format!(
                "      if ({}) {}[{addr}] <= {};",
                m.we, m.name, m.wdata
            ));
        }
        body.push("    end".to_string());
        body.push("  end".to_string());
    }
    lines.extend(body);
    lines.push("endmodule".to_string());
    Ok(lines.join("\n") + "\n")
}

// ---------------------------------------------------------------------------
// Plan-level simulator (same plan as the emitted source)
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct Sim {
    widths: HashMap<String, usize>,
    ports: Vec<String>,
    regs: Vec<String>,
    ops: Vec<(VOp, (usize, usize))>,
    inits: HashMap<String, u64>,
    nexts: HashMap<String, String>,
    mems: HashMap<String, (usize, usize, Vec<u64>)>,
    reads: Vec<(String, String, String)>,
    writes: Vec<(String, String, String, String)>,
}

fn mask(w: usize) -> u128 {
    if w >= 128 {
        u128::MAX
    } else {
        (1u128 << w) - 1
    }
}

fn arg_ref(a: &VArg) -> Result<&str> {
    match a {
        VArg::Ref(r) => Ok(r),
        VArg::Int(_) => Err(Error::asm("Verilog: expected stream reference")),
    }
}

fn arg_int(a: &VArg) -> Result<i64> {
    match a {
        VArg::Int(v) => Ok(*v),
        VArg::Ref(_) => Err(Error::asm("Verilog: expected integer literal")),
    }
}

impl Sim {
    pub fn compile(
        text: &str,
        widths: &Widths,
        state: &HashMap<String, (usize, u64, String)>,
        mem: &[Memory],
    ) -> Result<Self> {
        let (inputs, ops) = parse_program(text)?;
        let mut seed = HashMap::new();
        for (n, (w, _, _)) in state {
            seed.insert(n.clone(), (1, *w));
        }
        for m in mem {
            seed.insert(m.rout.clone(), (1, m.width));
            if let Some(r) = &m.rout2 {
                seed.insert(r.clone(), (1, m.width));
            }
        }
        let (wire, planned) = plan(&inputs, &ops, widths, &seed)?;
        let mut wmap = HashMap::new();
        for (n, (_, w)) in &wire {
            wmap.insert(n.clone(), *w);
        }
        let outs: Vec<String> = ops.iter().map(|o| o.out.clone()).collect();
        let routs: HashSet<String> = mem.iter().map(|m| m.rout.clone()).collect();
        let ports = outs
            .into_iter()
            .filter(|n| !state.contains_key(n) && !routs.contains(n))
            .collect();
        let regs = state.keys().cloned().collect();
        let inits = state.iter().map(|(n, (_, i, _))| (n.clone(), *i)).collect();
        let nexts = state
            .iter()
            .map(|(n, (_, _, x))| (n.clone(), x.clone()))
            .collect();
        let mut mems = HashMap::new();
        let mut reads = Vec::new();
        let mut writes = Vec::new();
        for m in mem {
            let mut init = m.init.clone();
            init.resize(m.depth, 0);
            mems.insert(m.name.clone(), (m.width, m.depth, init));
            reads.push((m.rout.clone(), m.name.clone(), m.raddr.clone()));
            if let Some(r) = &m.rout2 {
                reads.push((r.clone(), m.name.clone(), m.raddr2.clone().unwrap()));
            }
            writes.push((
                m.we.clone(),
                m.name.clone(),
                m.waddr.clone(),
                m.wdata.clone(),
            ));
        }
        Ok(Sim {
            widths: wmap,
            ports,
            regs,
            ops: planned,
            inits,
            nexts,
            mems,
            reads,
            writes,
        })
    }

    fn env_run(
        &self,
        inputs: &HashMap<String, u128>,
        state: &HashMap<String, u128>,
        mem_state: &HashMap<String, Vec<u64>>,
    ) -> Result<HashMap<String, u128>> {
        let mut env = inputs.clone();
        for (k, v) in state {
            env.insert(k.clone(), *v & mask(self.widths[k]));
        }
        let read_mem = |env: &mut HashMap<String, u128>| {
            for (rout, mname, raddr) in &self.reads {
                if env.contains_key(rout) {
                    continue;
                }
                if let Some(addr) = env.get(raddr) {
                    let (_, depth, _) = self.mems[mname];
                    let idx = (*addr as usize) & (depth - 1);
                    let v = mem_state[mname][idx];
                    let w = self.widths[rout];
                    env.insert(rout.clone(), (v as u128) & mask(w));
                }
            }
        };
        read_mem(&mut env);
        for (op, (lanes, w)) in &self.ops {
            let m = mask(*w);
            let val = if op.mn == "ICRC32" {
                let src = arg_ref(&op.args[0])?;
                let sn = *lanes;
                let packed = env[src];
                let mut crc: u32 = 0xFFFF_FFFF;
                for i in 0..sn {
                    let b = ((packed >> (i * 8)) & 0xFF) as u32;
                    crc ^= b;
                    for _ in 0..8 {
                        crc = (crc >> 1) ^ if crc & 1 != 0 { 0xEDB8_8320 } else { 0 };
                    }
                }
                (crc ^ 0xFFFF_FFFF) as u128
            } else if ARITH.contains(&op.mn.as_str()) || SHIFTS.contains(&op.mn.as_str()) {
                let a = env[arg_ref(&op.args[0])?];
                let b = match &op.args[1] {
                    VArg::Ref(r) => env[r],
                    VArg::Int(v) => *v as u128,
                };
                match op.mn.as_str() {
                    "IADD" => a.wrapping_add(b),
                    "ISUB" => a.wrapping_sub(b),
                    "IMUL" => a.wrapping_mul(b),
                    "IMOD" => {
                        if b == 0 {
                            return Err(Error::asm("Verilog: IMOD division by zero"));
                        }
                        a % b
                    }
                    "ISHL" => a.wrapping_shl(b as u32),
                    _ => a.wrapping_shr(b as u32),
                }
            } else if op.mn == "IEQ" || op.mn == "ILT" {
                let a = env[arg_ref(&op.args[0])?];
                let b = match &op.args[1] {
                    VArg::Ref(r) => env[r],
                    VArg::Int(v) => *v as u128,
                };
                if op.mn == "IEQ" {
                    (a == b) as u128
                } else {
                    (a < b) as u128
                }
            } else if op.mn == "ISELECT" {
                let c = env[arg_ref(&op.args[0])?];
                let a = match &op.args[1] {
                    VArg::Ref(r) => env[r],
                    VArg::Int(v) => *v as u128,
                };
                let b = match &op.args[2] {
                    VArg::Ref(r) => env[r],
                    VArg::Int(v) => *v as u128,
                };
                if c != 0 {
                    a
                } else {
                    b
                }
            } else {
                let a = env[arg_ref(&op.args[0])?];
                let b = match &op.args[1] {
                    VArg::Ref(r) => env[r],
                    VArg::Int(v) => *v as u128,
                };
                match op.mn.as_str() {
                    "IXOR" => a ^ b,
                    "IAND" => a & b,
                    _ => a | b,
                }
            };
            env.insert(op.out.clone(), val & m);
            read_mem(&mut env);
        }
        read_mem(&mut env);
        Ok(env)
    }

    pub fn simulate(&self, inputs: &HashMap<String, u128>) -> Result<HashMap<String, u128>> {
        let state: HashMap<String, u128> = self
            .inits
            .iter()
            .map(|(k, v)| (k.clone(), *v as u128))
            .collect();
        let mems: HashMap<String, Vec<u64>> = self
            .mems
            .iter()
            .map(|(n, (_, _, init))| (n.clone(), init.clone()))
            .collect();
        let env = self.env_run(inputs, &state, &mems)?;
        let mut out = HashMap::new();
        for p in self.ports.iter().chain(self.reads.iter().map(|x| &x.0)) {
            if let Some(v) = env.get(p) {
                out.insert(p.clone(), *v);
            }
        }
        Ok(out)
    }

    pub fn simulate_cycles(
        &self,
        inputs: &HashMap<String, u128>,
        cycles: usize,
        reset: usize,
        trace_mem: bool,
    ) -> Result<Vec<HashMap<String, u128>>> {
        if self.regs.is_empty() && self.mems.is_empty() {
            return Err(Error::asm("Verilog: simulate_cycles needs state"));
        }
        let mut state: HashMap<String, u128> = self
            .inits
            .iter()
            .map(|(k, v)| (k.clone(), *v as u128))
            .collect();
        let mut mem_state: HashMap<String, Vec<u64>> = self
            .mems
            .iter()
            .map(|(n, (_, _, init))| (n.clone(), init.clone()))
            .collect();
        let mut trace = Vec::new();
        for c in 0..cycles {
            if c < reset {
                state = self
                    .inits
                    .iter()
                    .map(|(k, v)| (k.clone(), *v as u128))
                    .collect();
            }
            let env = self.env_run(inputs, &state, &mem_state)?;
            let mut entry = HashMap::new();
            for p in &self.ports {
                entry.insert(p.clone(), env[p]);
            }
            for r in &self.regs {
                entry.insert(r.clone(), env[r]);
            }
            for (r, _, _) in &self.reads {
                entry.insert(r.clone(), env[r]);
            }
            if trace_mem {
                for (n, (_, _, _)) in &self.mems {
                    for (i, v) in mem_state[n].iter().enumerate() {
                        entry.insert(format!("mem:{n}:{i}"), *v as u128);
                    }
                }
            }
            trace.push(entry);
            if c >= reset {
                let mut next = HashMap::new();
                for r in &self.regs {
                    next.insert(r.clone(), env[&self.nexts[r]]);
                }
                state = next;
                for (we, mname, waddr, wdata) in &self.writes {
                    if env[we] != 0 {
                        let (_, depth, _) = self.mems[mname];
                        let idx = (*env.get(waddr).unwrap_or(&0) as usize) & (depth - 1);
                        let mw = self.mems[mname].0;
                        let v = env[wdata] as u64 & ((1u64 << mw) - 1);
                        mem_state.get_mut(mname).unwrap()[idx] = v;
                    }
                }
            }
        }
        Ok(trace)
    }
}
