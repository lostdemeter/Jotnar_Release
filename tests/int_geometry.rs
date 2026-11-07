//! Geometry-first integer primitives: exact semantics and real lowering parity.
//!
//! `IMATMUL` and `IREDUCE_SUM` are shape-defined operations, not per-arity
//! scalar conveniences. This test runs the reference interpreter, compiles the
//! same listing to C and to the integer-only (non-FPU) backend, runs both
//! emitted binaries on raw i64 payloads, and compares the result exactly.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use jotnar::asm::AsmOptions;
use jotnar::codegen::{build, compile_program, Artifact, CompileOptions, Target};
use jotnar::payload::Feeds;
use jotnar::value::Value;
use ndarray::{ArrayD, IxDyn};

const LISTING: &str = "\
IN A AS I:HW
IN B AS I:HW
C = IMATMUL(A, B)
OUT = IREDUCE_SUM(C)
";

fn opts() -> AsmOptions {
    AsmOptions::new(".")
}

fn value(shape: &[usize], data: &[i64]) -> Value {
    Value::I64(ArrayD::from_shape_vec(IxDyn(shape), data.to_vec()).unwrap())
}

fn feeds(pairs: &[(&str, Value)]) -> Feeds {
    let mut f = Feeds::new();
    for (name, v) in pairs {
        f.insert((*name).to_string(), v.clone());
    }
    f
}

fn i64_array(value: &Value) -> Vec<i64> {
    match value {
        Value::I64(a) => a.iter().copied().collect(),
        other => panic!("expected I64, got {other:?}"),
    }
}

fn compile(target: Target) -> Artifact {
    let a = value(&[3, 3], &[1, 2, 3, 4, 5, 6, 7, 8, 9]);
    let b = value(&[3, 3], &[1, 1, 1, 1, 1, 1, 1, 1, 1]);
    let opts = CompileOptions {
        target,
        outputs: vec!["OUT".to_string()],
        asm: opts(),
        use_fp16: false,
    };
    compile_program(LISTING, &opts, feeds(&[("A", a), ("B", b)])).expect("compile geometry ops")
}

fn write_i64(path: &Path, data: &[i64]) {
    let mut bytes = Vec::with_capacity(data.len() * 8);
    for v in data {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    fs::write(path, bytes).unwrap();
}

fn run_artifact(artifact: &Artifact, target: Target, dir: &Path, name: &str) -> i64 {
    let exe: PathBuf = build(&artifact.source, dir, name, target).expect("emitted build");
    let a = [1i64, 2, 3, 4, 5, 6, 7, 8, 9];
    let b = [1i64; 9];
    let a_path = dir.join(format!("{name}-a.bin"));
    let b_path = dir.join(format!("{name}-b.bin"));
    let out_path = dir.join(format!("{name}-out.bin"));
    write_i64(&a_path, &a);
    write_i64(&b_path, &b);
    let status = Command::new(&exe)
        .arg(&a_path)
        .arg(&b_path)
        .arg(&out_path)
        .status()
        .expect("run emitted binary");
    assert!(status.success(), "{name}: emitted binary failed");
    let bytes = fs::read(&out_path).unwrap();
    assert_eq!(bytes.len(), 8, "{name}: expected one i64 output");
    i64::from_le_bytes(bytes[0..8].try_into().unwrap())
}

fn temp_dir(tag: &str) -> PathBuf {
    let p = std::env::temp_dir().join(format!("jotnar-int-geometry-{tag}-{}", std::process::id()));
    let _ = fs::remove_dir_all(&p);
    fs::create_dir_all(&p).unwrap();
    p
}

#[test]
fn imatmul_and_ireduce_sum_are_exact() {
    let a = value(&[3, 3], &[1, 2, 3, 4, 5, 6, 7, 8, 9]);
    let b = value(&[3, 3], &[1; 9]);
    let got = jotnar::run_text(LISTING, &opts(), &feeds(&[("A", a), ("B", b)])).unwrap();
    let out = got.get("OUT").expect("OUT").first_i64("OUT").unwrap();
    // Each row sum is 6, 15, 24; summing the 3x3 product gives 3 * 45 = 135.
    assert_eq!(out, 135);
    assert_eq!(i64_array(got.get("OUT").unwrap()).len(), 1);
}

#[test]
fn ireduce_sum_wraps_like_iadd() {
    let x = value(&[2, 2], &[1, 2, 3, 4]);
    let got = jotnar::run_text(
        "IN X AS I:HW\nOUT = IREDUCE_SUM(X)\n",
        &opts(),
        &feeds(&[("X", x)]),
    )
    .unwrap();
    assert_eq!(got.get("OUT").unwrap().first_i64("OUT").unwrap(), 10);
}

#[test]
fn imatmul_refuses_a_shape_mismatch() {
    let a = value(&[2, 3], &[1, 2, 3, 4, 5, 6]);
    let b = value(&[2, 3], &[1, 2, 3, 4, 5, 6]);
    let result = jotnar::run_text(
        "IN A AS I:HW\nIN B AS I:HW\nOUT = IMATMUL(A, B)\n",
        &opts(),
        &feeds(&[("A", a), ("B", b)]),
    );
    assert!(result.is_err(), "inner dimension mismatch must be loud");
}

#[test]
fn geometry_ops_lower_to_c_and_nonfpu_and_match_the_interpreter() {
    let dir = temp_dir("parity");
    for (target, name) in [(Target::C, "c"), (Target::NonFpu, "nonfpu")] {
        let artifact = compile(target);
        assert!(
            artifact.source.contains("IMATMUL"),
            "{name}: missing IMATMUL lowering"
        );
        assert!(
            artifact.source.contains("IREDUCE_SUM"),
            "{name}: missing IREDUCE_SUM lowering"
        );
        let got = run_artifact(&artifact, target, &dir, name);
        assert_eq!(got, 135, "{name}: emitted result must match interpreter");
    }
}
