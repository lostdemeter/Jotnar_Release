//! `ISCATTER_MIN1` exact min-scatter: interpreter, C, and nonfpu parity.
//!
//! The op is I64-only and order-independent:
//! `OUT[id] = min(OUT[id], val)` for each `(id, val)`. Repeated ids and ties
//! must not depend on the scatter order, and an out-of-range id must fail
//! loudly rather than clamp.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use jotnar::asm::AsmOptions;
use jotnar::codegen::{build, compile_program, Artifact, CompileOptions, Target};
use jotnar::payload::Feeds;
use jotnar::value::Value;
use ndarray::{ArrayD, IxDyn};

const LISTING: &str = "\
IN BASE AS I:SEQ
IN IDS AS I:SEQ
IN VALS AS I:SEQ
OUT = ISCATTER_MIN1(BASE, IDS, VALS)
";

fn opts() -> AsmOptions {
    AsmOptions::new(".")
}

fn value(shape: &[usize], data: &[i64]) -> Value {
    Value::I64(ArrayD::from_shape_vec(IxDyn(shape), data.to_vec()).unwrap())
}

fn feeds() -> Feeds {
    let mut f = Feeds::new();
    f.insert("BASE".into(), value(&[4], &[10, 20, 30, 40]));
    f.insert("IDS".into(), value(&[5], &[0, 1, 0, 2, 3]));
    f.insert("VALS".into(), value(&[5], &[5, 25, 7, 3, 50]));
    f
}

fn i64_array(value: &Value) -> Vec<i64> {
    match value {
        Value::I64(a) => a.iter().copied().collect(),
        other => panic!("expected I64, got {other:?}"),
    }
}

fn compile(target: Target) -> Artifact {
    compile_program(
        LISTING,
        &CompileOptions {
            target,
            outputs: vec!["OUT".to_string()],
            asm: opts(),
            use_fp16: false,
        },
        feeds(),
    )
    .expect("compile ISCATTER_MIN1")
}

fn write_i64(path: &Path, data: &[i64]) {
    let mut bytes = Vec::with_capacity(data.len() * 8);
    for v in data {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    fs::write(path, bytes).unwrap();
}

fn run_artifact(artifact: &Artifact, target: Target, dir: &Path, name: &str) -> Vec<i64> {
    let exe: PathBuf = build(&artifact.source, dir, name, target).expect("emitted build");
    let base_path = dir.join(format!("{name}-base.bin"));
    let ids_path = dir.join(format!("{name}-ids.bin"));
    let vals_path = dir.join(format!("{name}-vals.bin"));
    let out_path = dir.join(format!("{name}-out.bin"));
    write_i64(&base_path, &[10, 20, 30, 40]);
    write_i64(&ids_path, &[0, 1, 0, 2, 3]);
    write_i64(&vals_path, &[5, 25, 7, 3, 50]);
    let status = Command::new(&exe)
        .arg(&base_path)
        .arg(&ids_path)
        .arg(&vals_path)
        .arg(&out_path)
        .status()
        .expect("run emitted binary");
    assert!(status.success(), "{name}: emitted binary failed");
    fs::read(&out_path)
        .unwrap()
        .chunks(8)
        .map(|c| i64::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

fn temp_dir(tag: &str) -> PathBuf {
    let p = std::env::temp_dir().join(format!("jotnar-iscatter-min1-{tag}-{}", std::process::id()));
    let _ = fs::remove_dir_all(&p);
    fs::create_dir_all(&p).unwrap();
    p
}

#[test]
fn min_scatter_is_order_independent_and_exact() {
    let got = jotnar::run_text(LISTING, &opts(), &feeds()).unwrap();
    assert_eq!(i64_array(got.get("OUT").unwrap()), vec![5, 20, 3, 40]);
}

#[test]
fn min_scatter_lowers_to_c_and_nonfpu_and_matches_the_interpreter() {
    let dir = temp_dir("parity");
    let want = vec![5, 20, 3, 40];
    for (target, name) in [(Target::C, "c"), (Target::NonFpu, "nonfpu")] {
        let artifact = compile(target);
        assert!(
            artifact.source.contains("ISCATTER_MIN1"),
            "{name}: missing ISCATTER_MIN1 lowering"
        );
        let got = run_artifact(&artifact, target, &dir, name);
        assert_eq!(got, want, "{name}: emitted result must match interpreter");
    }
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn out_of_range_id_is_loud() {
    let mut f = feeds();
    f.insert("IDS".into(), value(&[1], &[4]));
    f.insert("VALS".into(), value(&[1], &[1]));
    let result = jotnar::run_text(LISTING, &opts(), &f);
    assert!(result.is_err(), "out-of-range id must be a runtime error");
}

#[test]
fn emitted_out_of_range_id_exits_four() {
    let dir = temp_dir("bounds");
    for (target, name) in [(Target::C, "c"), (Target::NonFpu, "nonfpu")] {
        let artifact = compile(target);
        let exe: PathBuf = build(&artifact.source, &dir, name, target).expect("emitted build");
        let base_path = dir.join(format!("{name}-base.bin"));
        let ids_path = dir.join(format!("{name}-ids.bin"));
        let vals_path = dir.join(format!("{name}-vals.bin"));
        let out_path = dir.join(format!("{name}-out.bin"));
        write_i64(&base_path, &[10, 20, 30, 40]);
        write_i64(&ids_path, &[0, 1, 4, 2, 3]);
        write_i64(&vals_path, &[5, 25, 7, 3, 50]);
        let status = Command::new(&exe)
            .arg(&base_path)
            .arg(&ids_path)
            .arg(&vals_path)
            .arg(&out_path)
            .status()
            .expect("run emitted binary");
        assert_eq!(
            status.code(),
            Some(4),
            "{name}: out-of-range id must return the bounds code"
        );
    }
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn rust_target_refuses_iscatter_min1() {
    let err = compile_program(
        LISTING,
        &CompileOptions {
            target: Target::Rust,
            outputs: vec!["OUT".to_string()],
            asm: opts(),
            use_fp16: false,
        },
        feeds(),
    )
    .expect_err("Rust emission must be refused");
    assert!(
        matches!(err, jotnar::error::Error::Unimplemented(_)),
        "expected Unimplemented, got {err:?}"
    );
}
