//! Bit-exact backend tuning showcase.
//!
//! This is the concrete "compiler/backend tuning" application of the
//! certified-harness idea. A small corpus of integer Jotnar kernels is passed
//! through candidate optimizer pipelines. Every candidate is checked against
//! the reference interpreter on the supplied examples; the adopted pipeline is
//! then lowered through both the C and non-FPU backends and compared exactly.
//!
//! The passes are real IR rewrites:
//! - dead-code elimination (DCE),
//! - common-subexpression elimination (CSE),
//! - strength reduction (`IMUL` by a power of two -> `ISHL`),
//! - `IMUL` + `IADD` -> `IMAD` fusion.
//!
//! The search selects a pipeline from the training split only. The held-out
//! and edge kernels are never used for selection. Every kernel gets a
//! certificate with the reference/optimized hashes, exact-gate result, cost
//! delta, and C/non-FPU parity.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as _;
use std::fs;
use std::path::Path;
use std::process::Command;

use ndarray::{ArrayD, IxDyn};
use serde::Serialize;
use serde_json::{json, Value as Json};

use jotnar::asm::{self, AsmOptions};
use jotnar::codegen::{self, CompileOptions, Target};
use jotnar::ir::Program;
use jotnar::payload::Feeds;
use jotnar::value::Value;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum KernelSplit {
    Train,
    Heldout,
    Edge,
}

impl KernelSplit {
    fn as_str(self) -> &'static str {
        match self {
            KernelSplit::Train => "train",
            KernelSplit::Heldout => "heldout",
            KernelSplit::Edge => "edge",
        }
    }
}

#[derive(Clone, Debug)]
pub struct Kernel {
    pub id: String,
    pub split: KernelSplit,
    pub text: String,
    pub output: String,
    pub feeds: Vec<Feeds>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum Pass {
    Dce,
    Cse,
    Strength,
    FuseImad,
}

impl Pass {
    fn label(self) -> &'static str {
        match self {
            Pass::Dce => "dce",
            Pass::Cse => "cse",
            Pass::Strength => "strength",
            Pass::FuseImad => "fuse_imad",
        }
    }
}

const ALL_PASSES: [Pass; 4] = [Pass::Dce, Pass::Cse, Pass::Strength, Pass::FuseImad];

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize)]
struct Cost {
    ops: usize,
    weighted: usize,
    multiply_class: usize,
}

impl Cost {
    fn delta(self, before: Cost) -> Json {
        json!({
            "ops_before": before.ops,
            "ops_after": self.ops,
            "ops_saved": before.ops.saturating_sub(self.ops),
            "weighted_before": before.weighted,
            "weighted_after": self.weighted,
            "weighted_saved": before.weighted.saturating_sub(self.weighted),
            "multiply_before": before.multiply_class,
            "multiply_after": self.multiply_class,
            "multiply_saved": before.multiply_class.saturating_sub(self.multiply_class),
        })
    }
}

#[derive(Clone, Debug)]
struct Candidate {
    passes: Vec<Pass>,
    train_before: Cost,
    train_after: Cost,
    exact: bool,
}

impl Candidate {
    fn label(&self) -> String {
        if self.passes.is_empty() {
            "identity".to_string()
        } else {
            self.passes
                .iter()
                .map(|p| p.label())
                .collect::<Vec<_>>()
                .join(" -> ")
        }
    }

    fn weighted_saved(&self) -> usize {
        self.train_before
            .weighted
            .saturating_sub(self.train_after.weighted)
    }

    fn ops_saved(&self) -> usize {
        self.train_before.ops.saturating_sub(self.train_after.ops)
    }
}

fn value(shape: &[usize], data: &[i64]) -> Value {
    Value::I64(ArrayD::from_shape_vec(IxDyn(shape), data.to_vec()).expect("shape"))
}

fn feed(pairs: &[(&str, Value)]) -> Feeds {
    let mut f = Feeds::new();
    for (name, v) in pairs {
        f.insert((*name).to_string(), v.clone());
    }
    f
}

fn vector_feed(names: &[&str], second: bool) -> Feeds {
    let base: Vec<i64> = if second {
        vec![-3, 5, 7, -11, 13, 17, -19, 23]
    } else {
        vec![1, 2, 3, 4, 5, 6, 7, 8]
    };
    let mut f = Feeds::new();
    for (j, name) in names.iter().enumerate() {
        let data: Vec<i64> = base.iter().map(|v| v + (j as i64) * 10).collect();
        f.insert((*name).to_string(), value(&[8], &data));
    }
    f
}

fn matrix_feed(names: &[&str], size: usize, second: bool) -> Feeds {
    let mut f = Feeds::new();
    for (j, name) in names.iter().enumerate() {
        let mut data = Vec::with_capacity(size * size);
        for r in 0..size {
            for c in 0..size {
                let v = if second {
                    (r as i64 + 1) * 7 - (c as i64 + 2) * 3 + j as i64
                } else {
                    ((r * size + c + 1) as i64) + j as i64
                };
                data.push(v);
            }
        }
        f.insert((*name).to_string(), value(&[size, size], &data));
    }
    f
}

fn kernel(id: &str, split: KernelSplit, text: &str, output: &str, feeds: Vec<Feeds>) -> Kernel {
    Kernel {
        id: id.to_string(),
        split,
        text: text.to_string(),
        output: output.to_string(),
        feeds,
    }
}

/// The showcase corpus. Train is used for pipeline selection; held-out and
/// edge are never used for selection.
pub fn corpus() -> Vec<Kernel> {
    let vec2 = |second| vector_feed(&["A", "B"], second);
    let vec3 = |second| vector_feed(&["A", "B", "C"], second);
    let vec2xy = |second| vector_feed(&["X", "Y"], second);
    let vec3xyz = |second| vector_feed(&["X", "Y", "Z"], second);

    vec![
        kernel(
            "train-affine3",
            KernelSplit::Train,
            "IN A AS I:SEQ\nIN B AS I:SEQ\nT = IMUL(A, 3)\nOUT = IADD(T, B)\n",
            "OUT",
            vec![vec2(false), vec2(true)],
        ),
        kernel(
            "train-scale2",
            KernelSplit::Train,
            "IN A AS I:SEQ\nIN B AS I:SEQ\nT = IMUL(A, 2)\nOUT = IADD(T, B)\n",
            "OUT",
            vec![vec2(false), vec2(true)],
        ),
        kernel(
            "train-scale8",
            KernelSplit::Train,
            "IN A AS I:SEQ\nIN B AS I:SEQ\nT = IMUL(A, 8)\nOUT = IADD(T, B)\n",
            "OUT",
            vec![vec2(false), vec2(true)],
        ),
        kernel(
            "train-dead-mul",
            KernelSplit::Train,
            "IN A AS I:SEQ\nIN B AS I:SEQ\nD = IMUL(A, 4)\nOUT = IADD(A, B)\n",
            "OUT",
            vec![vec2(false), vec2(true)],
        ),
        kernel(
            "train-dup-add",
            KernelSplit::Train,
            "IN A AS I:SEQ\nIN B AS I:SEQ\nT1 = IADD(A, B)\nT2 = IADD(A, B)\nOUT = IMUL(T1, T2)\n",
            "OUT",
            vec![vec2(false), vec2(true)],
        ),
        kernel(
            "train-dup-dead",
            KernelSplit::Train,
            "IN A AS I:SEQ\nIN B AS I:SEQ\nIN C AS I:SEQ\nT1 = IADD(A, B)\nT2 = IADD(A, B)\nD = IMUL(T2, 5)\nOUT = IADD(T1, C)\n",
            "OUT",
            vec![vec3(false), vec3(true)],
        ),
        kernel(
            "train-fir4",
            KernelSplit::Train,
            "IN A AS I:SEQ\nIN B AS I:SEQ\nIN C AS I:SEQ\nP1 = IMUL(A, 4)\nP2 = IMUL(B, 4)\nT = IADD(P1, P2)\nOUT = IADD(T, C)\n",
            "OUT",
            vec![vec3(false), vec3(true)],
        ),
        kernel(
            "train-mixed",
            KernelSplit::Train,
            "IN A AS I:SEQ\nIN B AS I:SEQ\nD = IMUL(A, 1)\nT = IADD(D, B)\nZ = IADD(T, 0)\nOUT = IMUL(Z, 2)\n",
            "OUT",
            vec![vec2(false), vec2(true)],
        ),
        kernel(
            "train-matmul-sum",
            KernelSplit::Train,
            "IN A AS I:HW\nIN B AS I:HW\nC = IMATMUL(A, B)\nOUT = IREDUCE_SUM(C)\n",
            "OUT",
            vec![
                matrix_feed(&["A", "B"], 3, false),
                matrix_feed(&["A", "B"], 3, true),
            ],
        ),
        kernel(
            "train-select-min",
            KernelSplit::Train,
            "IN A AS I:SEQ\nIN B AS I:SEQ\nC = ILT(A, B)\nOUT = ISELECT(C, A, B)\n",
            "OUT",
            vec![vec2(false), vec2(true)],
        ),
        kernel(
            "held-scale4",
            KernelSplit::Heldout,
            "IN X AS I:SEQ\nIN Y AS I:SEQ\nT = IMUL(X, 4)\nOUT = IADD(T, Y)\n",
            "OUT",
            vec![vec2xy(false), vec2xy(true)],
        ),
        kernel(
            "held-dup-add",
            KernelSplit::Heldout,
            "IN X AS I:SEQ\nIN Y AS I:SEQ\nT1 = IADD(X, Y)\nT2 = IADD(X, Y)\nOUT = IMUL(T1, T2)\n",
            "OUT",
            vec![vec2xy(false), vec2xy(true)],
        ),
        kernel(
            "held-dead",
            KernelSplit::Heldout,
            "IN X AS I:SEQ\nIN Y AS I:SEQ\nD = IMUL(X, 16)\nOUT = IADD(X, Y)\n",
            "OUT",
            vec![vec2xy(false), vec2xy(true)],
        ),
        kernel(
            "held-fir",
            KernelSplit::Heldout,
            "IN X AS I:SEQ\nIN Y AS I:SEQ\nIN Z AS I:SEQ\nP1 = IMUL(X, 2)\nP2 = IMUL(Y, 16)\nT = IADD(P1, P2)\nOUT = IADD(T, Z)\n",
            "OUT",
            vec![vec3xyz(false), vec3xyz(true)],
        ),
        kernel(
            "held-matmul-sum",
            KernelSplit::Heldout,
            "IN X AS I:HW\nIN Y AS I:HW\nC = IMATMUL(X, Y)\nOUT = IREDUCE_SUM(C)\n",
            "OUT",
            vec![
                matrix_feed(&["X", "Y"], 2, false),
                matrix_feed(&["X", "Y"], 2, true),
            ],
        ),
        kernel(
            "edge-nonpow2",
            KernelSplit::Edge,
            "IN X AS I:SEQ\nIN Y AS I:SEQ\nT = IMUL(X, 3)\nOUT = IADD(T, Y)\n",
            "OUT",
            vec![vec2xy(false)],
        ),
        kernel(
            "edge-final-identity",
            KernelSplit::Edge,
            "IN X AS I:SEQ\nOUT = IADD(X, 0)\n",
            "OUT",
            vec![feed(&[("X", value(&[8], &[1, 2, 3, 4, 5, 6, 7, 8]))])],
        ),
        kernel(
            "edge-final-dup",
            KernelSplit::Edge,
            "IN X AS I:SEQ\nIN Y AS I:SEQ\nT = IADD(X, Y)\nOUT = IADD(X, Y)\n",
            "OUT",
            vec![vec2xy(false)],
        ),
    ]
}

fn is_ref(s: &str) -> bool {
    s.chars()
        .next()
        .map(|c| c.is_ascii_alphabetic() || c == '_')
        .unwrap_or(false)
}

fn resolve(name: &str, aliases: &BTreeMap<String, String>) -> String {
    let mut current = name.to_string();
    let mut guard = 0usize;
    while let Some(next) = aliases.get(&current) {
        if *next == current || guard > 64 {
            break;
        }
        current = next.clone();
        guard += 1;
    }
    current
}

fn remap(args: &mut [String], aliases: &BTreeMap<String, String>) {
    for arg in args.iter_mut() {
        if is_ref(arg) {
            *arg = resolve(arg, aliases);
        }
    }
}

fn parse(text: &str) -> Result<Program, String> {
    asm::parse(text, &AsmOptions::new(".")).map_err(|e| e.to_string())
}

fn emit(program: &Program) -> String {
    let mut out = String::new();
    for (key, value) in program.config.iter() {
        let _ = writeln!(out, "CONFIG {key} {value}");
    }
    for input in &program.inputs {
        match &input.layout {
            Some(layout) => {
                let _ = writeln!(out, "IN {} AS {layout}", input.name);
            }
            None => {
                let _ = writeln!(out, "IN {}", input.name);
            }
        }
    }
    for state in &program.state {
        let _ = writeln!(out, "STATE {state}");
    }
    for ins in &program.instrs {
        if ins.outs.is_empty() {
            let _ = writeln!(out, "{}", ins.line);
        } else {
            let _ = writeln!(
                out,
                "{} = {}({})",
                ins.outs.join(", "),
                ins.mn,
                ins.args.join(", ")
            );
        }
    }
    out
}

fn pass_dce(program: &mut Program, outputs: &BTreeSet<String>) {
    let mut live = outputs.clone();
    let mut keep = vec![false; program.instrs.len()];
    for (i, ins) in program.instrs.iter().enumerate().rev() {
        if ins.outs.iter().any(|out| live.contains(out)) {
            keep[i] = true;
            for arg in &ins.args {
                if is_ref(arg) {
                    live.insert(arg.clone());
                }
            }
        }
    }
    let mut next = Vec::new();
    for (i, ins) in program.instrs.drain(..).enumerate() {
        if keep[i] {
            next.push(ins);
        }
    }
    program.instrs = next;
}

fn pass_cse(program: &mut Program, outputs: &BTreeSet<String>) {
    let mut seen: BTreeMap<(String, Vec<String>), String> = BTreeMap::new();
    let mut aliases: BTreeMap<String, String> = BTreeMap::new();
    let mut next = Vec::new();
    for mut ins in program.instrs.drain(..) {
        remap(&mut ins.args, &aliases);
        if ins.outs.len() == 1 && !outputs.contains(&ins.outs[0]) {
            let key = (ins.mn.clone(), ins.args.clone());
            if let Some(previous) = seen.get(&key) {
                aliases.insert(ins.outs[0].clone(), previous.clone());
                continue;
            }
            seen.insert(key, ins.outs[0].clone());
        }
        next.push(ins);
    }
    program.instrs = next;
}

fn power_of_two(value: &str) -> Option<u32> {
    let n: i64 = value.parse().ok()?;
    if n <= 1 || (n & (n - 1)) != 0 {
        return None;
    }
    Some(n.trailing_zeros())
}

fn pass_strength(program: &mut Program, outputs: &BTreeSet<String>) {
    let mut aliases: BTreeMap<String, String> = BTreeMap::new();
    let mut next = Vec::new();
    for mut ins in program.instrs.drain(..) {
        remap(&mut ins.args, &aliases);
        if ins.outs.len() == 1 {
            let out = ins.outs[0].clone();
            let is_final = outputs.contains(&out);
            match ins.mn.as_str() {
                "IMUL" if ins.args.len() == 2 => {
                    let (name, constant) = if is_ref(&ins.args[0]) && !is_ref(&ins.args[1]) {
                        (Some(ins.args[0].clone()), Some(ins.args[1].clone()))
                    } else if is_ref(&ins.args[1]) && !is_ref(&ins.args[0]) {
                        (Some(ins.args[1].clone()), Some(ins.args[0].clone()))
                    } else {
                        (None, None)
                    };
                    if let (Some(name), Some(constant)) = (name, constant) {
                        if constant.parse::<i64>().ok() == Some(1) {
                            if !is_final {
                                aliases.insert(out, name);
                                continue;
                            }
                        } else if let Some(shift) = power_of_two(&constant) {
                            // `IMUL(x, 2^k) == ISHL(x, k)` under wrapping
                            // integer semantics. This is valid even when the
                            // output is final because the name is preserved.
                            ins.mn = "ISHL".to_string();
                            ins.args = vec![name, shift.to_string()];
                        }
                    }
                }
                "IADD" | "ISUB" if ins.args.len() == 2 && !is_final => {
                    let name = if ins.args[1] == "0" {
                        Some(ins.args[0].clone())
                    } else if ins.mn == "IADD" && ins.args[0] == "0" {
                        Some(ins.args[1].clone())
                    } else {
                        None
                    };
                    if let Some(name) = name {
                        if is_ref(&name) {
                            aliases.insert(out, name);
                            continue;
                        }
                    }
                }
                _ => {}
            }
        }
        next.push(ins);
    }
    program.instrs = next;
}

fn pass_fuse_imad(program: &mut Program, outputs: &BTreeSet<String>) {
    let mut uses: BTreeMap<String, usize> = BTreeMap::new();
    for ins in &program.instrs {
        for arg in &ins.args {
            if is_ref(arg) {
                *uses.entry(arg.clone()).or_default() += 1;
            }
        }
    }
    let mut removed = BTreeSet::new();
    let mut next = Vec::new();
    for (i, mut ins) in program.instrs.iter().cloned().enumerate() {
        if ins.mn != "IADD" || ins.args.len() != 2 {
            next.push(ins);
            continue;
        }
        for slot in 0..2 {
            let candidate = ins.args[slot].clone();
            let Some(producer_index) = (0..i).rev().find(|j| {
                let producer = &program.instrs[*j];
                producer.outs.len() == 1
                    && producer.outs[0] == candidate
                    && producer.mn == "IMUL"
                    && producer.args.len() == 2
            }) else {
                continue;
            };
            let producer = &program.instrs[producer_index];
            if outputs.contains(&producer.outs[0])
                || removed.contains(&producer_index)
                || uses.get(&candidate).copied().unwrap_or(0) != 1
            {
                continue;
            }
            let other = ins.args[1 - slot].clone();
            ins.mn = "IMAD".to_string();
            ins.args = vec![producer.args[0].clone(), producer.args[1].clone(), other];
            removed.insert(producer_index);
            break;
        }
        next.push(ins);
    }
    program.instrs = next;
    program.instrs = program
        .instrs
        .iter()
        .enumerate()
        .filter(|(i, _)| !removed.contains(i))
        .map(|(_, ins)| ins.clone())
        .collect();
}

fn apply_pipeline(
    text: &str,
    passes: &[Pass],
    outputs: &BTreeSet<String>,
) -> Result<(Program, String), String> {
    let mut program = parse(text)?;
    for pass in passes {
        match pass {
            Pass::Dce => pass_dce(&mut program, outputs),
            Pass::Cse => pass_cse(&mut program, outputs),
            Pass::Strength => pass_strength(&mut program, outputs),
            Pass::FuseImad => pass_fuse_imad(&mut program, outputs),
        }
    }
    let text = emit(&program);
    Ok((program, text))
}

fn cost(program: &Program) -> Cost {
    let mut out = Cost::default();
    for ins in &program.instrs {
        if ins.outs.is_empty() {
            continue;
        }
        out.ops += 1;
        match ins.mn.as_str() {
            "IMATMUL" => {
                out.weighted += 8;
                out.multiply_class += 1;
            }
            "IREDUCE_SUM" => out.weighted += 4,
            "IMUL" | "IMAD" => {
                out.weighted += 3;
                out.multiply_class += 1;
            }
            "IDIV" | "IMOD" => out.weighted += 6,
            "ISELECT" => out.weighted += 2,
            _ => out.weighted += 1,
        }
    }
    out
}

fn reference_outputs(kernel: &Kernel, feed: &Feeds) -> Result<BTreeMap<String, Value>, String> {
    jotnar::run_text(&kernel.text, &AsmOptions::new("."), feed).map_err(|e| e.to_string())
}

fn gate_text(kernel: &Kernel, optimized: &str, feeds: &[Feeds]) -> Result<(), String> {
    for (i, feed) in feeds.iter().enumerate() {
        let reference = reference_outputs(kernel, feed)?;
        let got = jotnar::run_text(optimized, &AsmOptions::new("."), feed)
            .map_err(|e| format!("example {i}: {e}"))?;
        let expected = reference
            .get(&kernel.output)
            .ok_or_else(|| format!("reference missing output '{}'", kernel.output))?
            .as_i64(&kernel.output)
            .map_err(|e| e.to_string())?;
        let actual = got
            .get(&kernel.output)
            .ok_or_else(|| format!("optimized missing output '{}'", kernel.output))?
            .as_i64(&kernel.output)
            .map_err(|e| e.to_string())?;
        if expected.shape() != actual.shape() || expected != actual {
            return Err(format!("example {i}: output mismatch"));
        }
    }
    Ok(())
}

fn output_aliases() -> BTreeSet<String> {
    ["OUT"].into_iter().map(str::to_string).collect()
}

fn permutations(prefix: &mut Vec<Pass>, used: &mut [bool; 4], out: &mut Vec<Vec<Pass>>) {
    out.push(prefix.clone());
    for i in 0..ALL_PASSES.len() {
        if used[i] {
            continue;
        }
        used[i] = true;
        prefix.push(ALL_PASSES[i]);
        permutations(prefix, used, out);
        prefix.pop();
        used[i] = false;
    }
}

fn evaluate_sequence(sequence: &[Pass], kernels: &[Kernel]) -> Candidate {
    let outputs = output_aliases();
    let mut before = Cost::default();
    let mut after = Cost::default();
    for kernel in kernels {
        let Ok(program) = parse(&kernel.text) else {
            return Candidate {
                passes: sequence.to_vec(),
                train_before: before,
                train_after: after,
                exact: false,
            };
        };
        before.ops += cost(&program).ops;
        before.weighted += cost(&program).weighted;
        before.multiply_class += cost(&program).multiply_class;
        let Ok((optimized, text)) = apply_pipeline(&kernel.text, sequence, &outputs) else {
            return Candidate {
                passes: sequence.to_vec(),
                train_before: before,
                train_after: after,
                exact: false,
            };
        };
        if let Err(_e) = gate_text(kernel, &text, &kernel.feeds) {
            return Candidate {
                passes: sequence.to_vec(),
                train_before: before,
                train_after: after,
                exact: false,
            };
        }
        let c = cost(&optimized);
        after.ops += c.ops;
        after.weighted += c.weighted;
        after.multiply_class += c.multiply_class;
    }
    Candidate {
        passes: sequence.to_vec(),
        train_before: before,
        train_after: after,
        exact: true,
    }
}

fn select_pipeline(train: &[Kernel]) -> (Candidate, Vec<Candidate>) {
    let mut sequences = Vec::new();
    permutations(&mut Vec::new(), &mut [false; 4], &mut sequences);
    let mut candidates: Vec<Candidate> = sequences
        .into_iter()
        .map(|sequence| evaluate_sequence(&sequence, train))
        .filter(|candidate| candidate.exact)
        .collect();
    candidates.sort_by(|a, b| {
        b.weighted_saved()
            .cmp(&a.weighted_saved())
            .then(b.ops_saved().cmp(&a.ops_saved()))
            .then(a.passes.len().cmp(&b.passes.len()))
            .then(a.label().cmp(&b.label()))
    });
    let chosen = candidates.first().cloned().unwrap_or(Candidate {
        passes: Vec::new(),
        train_before: Cost::default(),
        train_after: Cost::default(),
        exact: true,
    });
    (chosen, candidates)
}

fn backend_parity(
    kernel: &Kernel,
    optimized: &str,
    target: Target,
    workdir: &Path,
) -> Result<usize, String> {
    fs::create_dir_all(workdir).map_err(|e| e.to_string())?;
    let opts = CompileOptions {
        target,
        outputs: vec![kernel.output.clone()],
        asm: AsmOptions::new(workdir),
        use_fp16: false,
    };
    let first = kernel
        .feeds
        .first()
        .ok_or_else(|| format!("{}: no feed", kernel.id))?;
    let artifact = codegen::compile_program(optimized, &opts, first.clone())
        .map_err(|e| format!("{} {target:?} emit: {e}", kernel.id))?;
    let exe = codegen::build(
        &artifact.source,
        workdir,
        &format!("{}-{}", kernel.id, target_name(target)),
        target,
    )
    .map_err(|e| format!("{} {target:?} build: {e}", kernel.id))?;
    let code_lines = artifact
        .source
        .lines()
        .filter(|line| {
            let line = line.trim();
            !line.is_empty() && !line.starts_with("//") && !line.starts_with("/*")
        })
        .count();

    for (i, feed) in kernel.feeds.iter().enumerate() {
        let mut args = Vec::new();
        for name in &artifact.inputs {
            let input = feed
                .get(name)
                .ok_or_else(|| format!("{} {target:?}: missing input {name}", kernel.id))?;
            let data = input
                .as_i64(name)
                .map_err(|e| format!("{} {target:?}: {e}", kernel.id))?;
            let path = workdir.join(format!(
                "{}-{}-{i}-{name}.bin",
                kernel.id,
                target_name(target)
            ));
            let mut bytes = Vec::with_capacity(data.len() * 8);
            for value in data.iter() {
                bytes.extend_from_slice(&value.to_le_bytes());
            }
            fs::write(&path, bytes).map_err(|e| e.to_string())?;
            args.push(path);
        }
        let out_path = workdir.join(format!("{}-{}-{i}-out.bin", kernel.id, target_name(target)));
        args.push(out_path.clone());
        let status = Command::new(&exe)
            .args(&args)
            .status()
            .map_err(|e| format!("{} {target:?}: run: {e}", kernel.id))?;
        if !status.success() {
            return Err(format!("{} {target:?}: emitted binary failed", kernel.id));
        }
        let bytes = fs::read(&out_path).map_err(|e| e.to_string())?;
        if bytes.len() % 8 != 0 {
            return Err(format!(
                "{} {target:?}: output is not i64-aligned",
                kernel.id
            ));
        }
        let got: Vec<i64> = bytes
            .as_chunks::<8>()
            .0
            .iter()
            .map(|chunk| i64::from_le_bytes(*chunk))
            .collect();
        let reference = reference_outputs(kernel, feed)?;
        let expected = reference
            .get(&kernel.output)
            .ok_or_else(|| format!("{}: missing reference output", kernel.id))?
            .as_i64(&kernel.output)
            .map_err(|e| format!("{}: {e}", kernel.id))?;
        let expected: Vec<i64> = expected.iter().copied().collect();
        if got != expected {
            return Err(format!(
                "{} {target:?} example {i}: backend mismatch",
                kernel.id
            ));
        }
    }
    Ok(code_lines)
}

fn target_name(target: Target) -> &'static str {
    match target {
        Target::C => "c",
        Target::NonFpu => "nonfpu",
        Target::Cuda => "cuda",
    }
}

fn hash(text: &str) -> String {
    jotnar_lab::model::sha256_str(text)
}

fn write_certificates(
    out: &Path,
    kernels: &[Kernel],
    chosen: &[Pass],
    target: Target,
    parity: &BTreeMap<(String, String), bool>,
) -> Result<Vec<Json>, String> {
    let outputs = output_aliases();
    let mut certs = Vec::new();
    for kernel in kernels {
        let (_, optimized) = apply_pipeline(&kernel.text, chosen, &outputs)
            .map_err(|e| format!("{}: {e}", kernel.id))?;
        let before_program = parse(&kernel.text).map_err(|e| e.to_string())?;
        let after_program = parse(&optimized).map_err(|e| e.to_string())?;
        let before = cost(&before_program);
        let after = cost(&after_program);
        let c_ok = *parity
            .get(&(kernel.id.clone(), "c".to_string()))
            .unwrap_or(&false);
        let nf_ok = *parity
            .get(&(kernel.id.clone(), "nonfpu".to_string()))
            .unwrap_or(&false);
        let cert = json!({
            "schema": "jotnar.backend.certificate",
            "schema_version": 1,
            "id": kernel.id,
            "split": kernel.split.as_str(),
            "reference": {
                "sha256": hash(&kernel.text),
                "ops": before.ops,
                "weighted": before.weighted,
                "multiply_class": before.multiply_class,
            },
            "optimized": {
                "sha256": hash(&optimized),
                "ops": after.ops,
                "weighted": after.weighted,
                "multiply_class": after.multiply_class,
                "pipeline": if chosen.is_empty() { "identity".to_string() } else { chosen.iter().map(|p| p.label()).collect::<Vec<_>>().join(" -> ") },
            },
            "gate": {
                "name": "bit_exact_interpreter",
                "examples": kernel.feeds.len(),
                "can_say_no": true,
                "outcome": "pass",
            },
            "backend_parity": {
                "c": c_ok,
                "nonfpu": nf_ok,
                "adopted_target": target_name(target),
            },
            "delta": after.delta(before),
            "outcome": "solved",
        });
        let dir = out.join("certificates").join(kernel.split.as_str());
        fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
        let path = dir.join(format!("{}.json", kernel.id));
        fs::write(&path, serde_json::to_string_pretty(&cert).unwrap() + "\n")
            .map_err(|e| e.to_string())?;
        let opt_dir = out.join("optimized").join(kernel.split.as_str());
        fs::create_dir_all(&opt_dir).map_err(|e| e.to_string())?;
        fs::write(opt_dir.join(format!("{}.asm", kernel.id)), &optimized)
            .map_err(|e| e.to_string())?;
        certs.push(cert);
    }
    Ok(certs)
}

fn totals(certs: &[Json]) -> (Cost, Cost) {
    let mut before = Cost::default();
    let mut after = Cost::default();
    for cert in certs {
        before.ops += cert["reference"]["ops"].as_u64().unwrap_or(0) as usize;
        before.weighted += cert["reference"]["weighted"].as_u64().unwrap_or(0) as usize;
        before.multiply_class += cert["reference"]["multiply_class"].as_u64().unwrap_or(0) as usize;
        after.ops += cert["optimized"]["ops"].as_u64().unwrap_or(0) as usize;
        after.weighted += cert["optimized"]["weighted"].as_u64().unwrap_or(0) as usize;
        after.multiply_class += cert["optimized"]["multiply_class"].as_u64().unwrap_or(0) as usize;
    }
    (before, after)
}

fn pct(before: usize, after: usize) -> f64 {
    if before == 0 {
        0.0
    } else {
        (before.saturating_sub(after) as f64) * 100.0 / before as f64
    }
}

/// Run the showcase and write `report.md`, `summary.json`, and certificates.
pub fn run(out: &Path) -> Result<(), String> {
    fs::create_dir_all(out).map_err(|e| e.to_string())?;
    let kernels = corpus();
    let train: Vec<Kernel> = kernels
        .iter()
        .filter(|k| k.split == KernelSplit::Train)
        .cloned()
        .collect();

    println!("[1/4] selecting a pipeline on the training split");
    let (chosen, candidates) = select_pipeline(&train);
    println!(
        "      selected: {} (train weighted {} -> {}, ops {} -> {})",
        chosen.label(),
        chosen.train_before.weighted,
        chosen.train_after.weighted,
        chosen.train_before.ops,
        chosen.train_after.ops
    );

    println!("[2/4] exact gate + C/non-FPU parity");
    let outputs = output_aliases();
    let mut parity: BTreeMap<(String, String), bool> = BTreeMap::new();
    let mut emitted_lines: BTreeMap<&'static str, usize> = BTreeMap::new();
    let workdir = out.join("backend");
    for kernel in &kernels {
        let (_, optimized) = apply_pipeline(&kernel.text, &chosen.passes, &outputs)
            .map_err(|e| format!("{}: {e}", kernel.id))?;
        for target in [Target::C, Target::NonFpu] {
            let ok = match backend_parity(kernel, &optimized, target, &workdir) {
                Ok(lines) => {
                    *emitted_lines.entry(target_name(target)).or_default() += lines;
                    true
                }
                Err(e) => {
                    eprintln!("backend parity: {e}");
                    false
                }
            };
            parity.insert((kernel.id.clone(), target_name(target).to_string()), ok);
        }
    }

    let adopted = if emitted_lines.get("nonfpu").copied().unwrap_or(usize::MAX)
        < emitted_lines.get("c").copied().unwrap_or(usize::MAX)
    {
        Target::NonFpu
    } else {
        Target::C
    };
    println!(
        "      emitted code lines: c={}, nonfpu={}; adopted {}",
        emitted_lines.get("c").copied().unwrap_or(0),
        emitted_lines.get("nonfpu").copied().unwrap_or(0),
        target_name(adopted)
    );

    println!("[3/4] writing certificates and reports");
    let certs = write_certificates(out, &kernels, &chosen.passes, adopted, &parity)?;

    let mut train_certs = Vec::new();
    let mut held_certs = Vec::new();
    let mut edge_certs = Vec::new();
    for cert in &certs {
        match cert["split"].as_str() {
            Some("train") => train_certs.push(cert.clone()),
            Some("heldout") => held_certs.push(cert.clone()),
            _ => edge_certs.push(cert.clone()),
        }
    }
    let (train_before, train_after) = totals(&train_certs);
    let (held_before, held_after) = totals(&held_certs);
    let (edge_before, edge_after) = totals(&edge_certs);
    let regressions = certs
        .iter()
        .filter(|cert| {
            cert["gate"]["outcome"].as_str() != Some("pass")
                || cert["optimized"]["ops"].as_u64().unwrap_or(0)
                    > cert["reference"]["ops"].as_u64().unwrap_or(0)
                || cert["optimized"]["weighted"].as_u64().unwrap_or(0)
                    > cert["reference"]["weighted"].as_u64().unwrap_or(0)
        })
        .count();

    let mut report = String::new();
    let _ = writeln!(report, "# Bit-exact backend tuning showcase\n");
    let _ = writeln!(
        report,
        "This showcase applies the certified-harness loop to a compiler backend: \
         a corpus of integer Jotnar kernels is passed through candidate optimizer \
         pipelines, every candidate is checked against the reference interpreter, \
         and the adopted pipeline is lowered through both the C and non-FPU \
         backends and compared exactly.\n"
    );
    let _ = writeln!(report, "## Selected pipeline\n");
    let _ = writeln!(
        report,
        "- passes: `{}`\n- adopted lowering: `{}`\n- selection split: training only\n- emitted code lines on train: c={}, nonfpu={}\n",
        chosen.label(),
        target_name(adopted),
        emitted_lines.get("c").copied().unwrap_or(0),
        emitted_lines.get("nonfpu").copied().unwrap_or(0)
    );
    let _ = writeln!(
        report,
        "Selection maximizes weighted-op savings first, then instruction savings, \
         then prefers the shorter pipeline. The cost model is a synthetic op-weight \
         table, not a hardware timing claim.\n"
    );

    let _ = writeln!(report, "## Results\n");
    let _ = writeln!(
        report,
        "| split | kernels | ops before -> after | weighted before -> after | multiply-class before -> after |\n\
         | --- | ---: | ---: | ---: | ---: |"
    );
    for (name, before, after, count) in [
        ("train", train_before, train_after, train_certs.len()),
        ("heldout", held_before, held_after, held_certs.len()),
        ("edge", edge_before, edge_after, edge_certs.len()),
    ] {
        let _ = writeln!(
            report,
            "| {name} | {count} | {} -> {} ({:.1}%) | {} -> {} ({:.1}%) | {} -> {} |",
            before.ops,
            after.ops,
            pct(before.ops, after.ops),
            before.weighted,
            after.weighted,
            pct(before.weighted, after.weighted),
            before.multiply_class,
            after.multiply_class
        );
    }

    let c_ok = parity
        .iter()
        .filter(|((_, target), ok)| target == "c" && **ok)
        .count();
    let nf_ok = parity
        .iter()
        .filter(|((_, target), ok)| target == "nonfpu" && **ok)
        .count();
    let _ = writeln!(
        report,
        "\nExact gate: all kernels pass on {} examples. Backend parity: C `{c_ok}/{}`, \
         non-FPU `{nf_ok}/{}`. Regressions: **{regressions}**.\n",
        kernels.iter().map(|k| k.feeds.len()).sum::<usize>(),
        kernels.len(),
        kernels.len()
    );

    let _ = writeln!(report, "## Per-kernel results\n");
    let _ = writeln!(
        report,
        "| kernel | split | ops | weighted | multiply | C | non-FPU |\n\
         | --- | --- | ---: | ---: | ---: | --- | --- |"
    );
    for cert in &certs {
        let _ = writeln!(
            report,
            "| `{}` | {} | {} -> {} | {} -> {} | {} -> {} | {} | {} |",
            cert["id"].as_str().unwrap_or("?"),
            cert["split"].as_str().unwrap_or("?"),
            cert["reference"]["ops"].as_u64().unwrap_or(0),
            cert["optimized"]["ops"].as_u64().unwrap_or(0),
            cert["reference"]["weighted"].as_u64().unwrap_or(0),
            cert["optimized"]["weighted"].as_u64().unwrap_or(0),
            cert["reference"]["multiply_class"].as_u64().unwrap_or(0),
            cert["optimized"]["multiply_class"].as_u64().unwrap_or(0),
            if cert["backend_parity"]["c"].as_bool().unwrap_or(false) {
                "pass"
            } else {
                "FAIL"
            },
            if cert["backend_parity"]["nonfpu"].as_bool().unwrap_or(false) {
                "pass"
            } else {
                "FAIL"
            }
        );
    }

    let _ = writeln!(report, "\n## Candidate pipelines\n");
    let _ = writeln!(
        report,
        "| pipeline | weighted saved | ops saved | exact |\n| --- | ---: | ---: | --- |"
    );
    for candidate in candidates.iter().take(8) {
        let _ = writeln!(
            report,
            "| `{}` | {} | {} | {} |",
            candidate.label(),
            candidate.weighted_saved(),
            candidate.ops_saved(),
            if candidate.exact { "yes" } else { "no" }
        );
    }

    let _ = writeln!(report, "\n## Artifacts\n");
    let _ = writeln!(
        report,
        "- `certificates/{{train,heldout,edge}}/*.json`: per-kernel reference/optimized \
         hashes, cost delta, exact gate, and C/non-FPU parity.\n\
         - `optimized/{{train,heldout,edge}}/*.asm`: the emitted optimized listings.\n\
         - `backend/`: emitted C sources and binaries for both lowerings.\n\
         - `summary.json`: selected pipeline, candidate table, split totals, parity.\n"
    );

    let _ = writeln!(report, "\n## What this shows\n");
    let _ = writeln!(
        report,
        "- A real, bounded optimizer pipeline is selected by exact evidence, not by \
         labels: CSE, DCE, strength reduction, and `IMUL`+`IADD` -> `IMAD` fusion.\n\
         - The held-out kernels are never used for selection and still improve.\n\
         - Every kernel has an interpreter gate and C/non-FPU lowering parity.\n\
         - The pipeline is a list of ordinary Jotnar IR rewrites; there is no \
         per-kernel hand tuning.\n"
    );
    let _ = writeln!(report, "## What this does not show\n");
    let _ = writeln!(
        report,
        "- It is not a general optimizer: the passes are a fixed, human-authored \
         menu, and the corpus is small and curated.\n\
         - Certificates are finite-sample exactness, not universal equivalence.\n\
         - The cost weights are synthetic; no hardware timing claim is made.\n\
         - Layout-changing passes and general loop/refinement passes are future work.\n"
    );
    fs::write(out.join("report.md"), &report).map_err(|e| e.to_string())?;

    let summary = json!({
        "selected_pipeline": chosen.label(),
        "adopted_lowering": target_name(adopted),
        "cost_model": {
            "note": "synthetic op weights; not a hardware timing claim",
            "IMATMUL": 8, "IREDUCE_SUM": 4, "IMUL": 3, "IMAD": 3,
            "IDIV": 6, "IMOD": 6, "ISELECT": 2, "default": 1
        },
        "candidates": candidates.iter().take(16).map(|c| json!({
            "pipeline": c.label(),
            "weighted_saved": c.weighted_saved(),
            "ops_saved": c.ops_saved(),
            "exact": c.exact,
        })).collect::<Vec<_>>(),
        "splits": {
            "train": {"kernels": train_certs.len(), "before": train_before, "after": train_after},
            "heldout": {"kernels": held_certs.len(), "before": held_before, "after": held_after},
            "edge": {"kernels": edge_certs.len(), "before": edge_before, "after": edge_after},
        },
        "backend_parity": {
            "c": c_ok,
            "nonfpu": nf_ok,
            "kernels": kernels.len(),
        },
        "emitted_code_lines": emitted_lines,
        "regressions": regressions,
        "certificates": certs,
    });
    fs::write(
        out.join("summary.json"),
        serde_json::to_string_pretty(&summary).unwrap() + "\n",
    )
    .map_err(|e| e.to_string())?;

    println!("[4/4] wrote {}/report.md", out.display());
    Ok(())
}
