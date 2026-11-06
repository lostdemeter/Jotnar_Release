use std::collections::BTreeMap;
use std::process::ExitCode;

use jotnar::asm::AsmOptions;
use jotnar::payload::Feeds;
use jotnar::value::Value;
use ndarray::ArrayD;

fn one(v: i64) -> Value {
    Value::I64(ArrayD::from_elem(ndarray::IxDyn(&[]), v))
}

fn vec(v: &[i64]) -> Value {
    Value::I64(jotnar::value::array_from_vec(&[v.len()], v.to_vec()).unwrap())
}

fn show(name: &str, v: &Value) -> String {
    match v {
        Value::I64(a) => format!("{name}={:?}", a.iter().copied().collect::<Vec<_>>()),
        other => format!("{name}={other:?}"),
    }
}

fn run(text: &str, payload: Feeds, outs: &[&str]) -> Result<(), String> {
    let opts = AsmOptions::new("showcase/language");
    let got = jotnar::run_text(text, &opts, &payload).map_err(|e| e.to_string())?;
    let mut parts = Vec::new();
    for o in outs {
        parts.push(show(o, got.get(*o).ok_or_else(|| format!("missing {o}"))?));
    }
    println!("{}", parts.join("  "));
    Ok(())
}

fn main() -> ExitCode {
    match tour() {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("{e}");
            ExitCode::FAILURE
        }
    }
}

fn tour() -> Result<(), String> {
    println!("== saxpy: OUT = A*X + Y ==");
    println!("{}", include_str!("../../showcase/language/saxpy.asm"));
    let mut p = BTreeMap::new();
    p.insert("A".to_string(), vec(&[3, 3, 3, 3]));
    p.insert("X".to_string(), vec(&[1, 2, 3, 4]));
    p.insert("Y".to_string(), vec(&[10, 20, 30, 40]));
    run(
        include_str!("../../showcase/language/saxpy.asm"),
        p,
        &["OUT"],
    )?;

    println!("\n== reduce: eight-way tree ==");
    let mut p = BTreeMap::new();
    for i in 0..8 {
        p.insert(format!("A{i}"), one(i + 1));
    }
    run(
        include_str!("../../showcase/language/reduce.asm"),
        p,
        &["OUT"],
    )?;

    println!("\n== matmul2: shared operand graph ==");
    let mut p = BTreeMap::new();
    for (name, v) in [
        ("a00", 1),
        ("a01", 2),
        ("a10", 3),
        ("a11", 4),
        ("b00", 5),
        ("b01", 6),
        ("b10", 7),
        ("b11", 8),
    ] {
        p.insert(name.to_string(), one(v));
    }
    run(
        include_str!("../../showcase/language/matmul2.asm"),
        p,
        &["c00", "c01", "c10", "c11"],
    )?;

    println!("\n== select: exact conditional ==");
    let mut p = BTreeMap::new();
    p.insert("C".to_string(), one(1));
    p.insert("A".to_string(), one(42));
    p.insert("B".to_string(), one(7));
    run(
        include_str!("../../showcase/language/select.asm"),
        p,
        &["OUT"],
    )?;

    Ok(())
}
