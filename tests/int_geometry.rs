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

// --- T2I and IGET boundary gates -----------------------------------------

use jotnar::lattice::{self, BIAS, K, PHI};
use jotnar::value::Triple;

fn t2i_triple(values: &[f64], extra: &[(i8, i32, u8)]) -> Triple {
    let encoded =
        lattice::encode(&ArrayD::from_shape_vec(IxDyn(&[values.len()]), values.to_vec()).unwrap());
    let mut s: Vec<i8> = encoded.s.iter().copied().collect();
    let mut e: Vec<i32> = encoded.e.iter().copied().collect();
    let mut z: Vec<u8> = encoded.z.iter().copied().collect();
    for (sv, ev, zv) in extra {
        s.push(*sv);
        e.push(*ev);
        z.push(*zv);
    }
    Triple::from_vecs(&[s.len()], s, e, z).unwrap()
}

fn t2i_feed(triple: &Triple) -> Feeds {
    let mut f = Feeds::new();
    f.insert("x".to_string(), Value::Triple(triple.clone()));
    f
}

fn write_triple_inputs(dir: &Path, name: &str, triple: &Triple) -> (PathBuf, PathBuf, PathBuf) {
    let s_path = dir.join(format!("{name}-s.bin"));
    let e_path = dir.join(format!("{name}-e.bin"));
    let z_path = dir.join(format!("{name}-z.bin"));
    fs::write(
        &s_path,
        triple.s.iter().map(|v| *v as u8).collect::<Vec<u8>>(),
    )
    .unwrap();
    let mut eb = Vec::new();
    for v in triple.e.iter() {
        eb.extend_from_slice(&v.to_le_bytes());
    }
    fs::write(&e_path, eb).unwrap();
    fs::write(&z_path, triple.z.iter().copied().collect::<Vec<u8>>()).unwrap();
    (s_path, e_path, z_path)
}

fn run_t2i(target: Target, dir: &Path, name: &str, triple: &Triple) -> Result<Vec<i64>, String> {
    let listing = "IN x AS T:*\nOUT = T2I(x)\n";
    let opts = CompileOptions {
        target,
        outputs: vec!["OUT".to_string()],
        asm: opts(),
        use_fp16: false,
    };
    let artifact = compile_program(listing, &opts, t2i_feed(triple))
        .map_err(|e| format!("{name} compile: {e}"))?;
    let exe =
        build(&artifact.source, dir, name, target).map_err(|e| format!("{name} build: {e}"))?;
    let (s_path, e_path, z_path) = write_triple_inputs(dir, name, triple);
    let out_path = dir.join(format!("{name}-out.bin"));
    let status = Command::new(&exe)
        .arg(&s_path)
        .arg(&e_path)
        .arg(&z_path)
        .arg(&out_path)
        .status()
        .map_err(|e| format!("{name} run: {e}"))?;
    if !status.success() {
        return Err(format!("{name}: emitted binary exited {}", status));
    }
    let bytes = fs::read(&out_path).map_err(|e| format!("{name} read: {e}"))?;
    Ok(bytes
        .as_chunks::<8>()
        .0
        .iter()
        .map(|c| i64::from_le_bytes(*c))
        .collect())
}

#[test]
fn t2i_lut_matches_the_reference_for_every_exponent() {
    let lut = lattice::t2i_lut();
    assert_eq!(lut.len(), lattice::NLVL as usize);
    for e in 0..lattice::NLVL {
        let want = PHI.powf((e as f64 - BIAS as f64) / K as f64).round() as i64;
        assert_eq!(lut[e as usize], want, "T2I table mismatch at e={e}");
    }
}

#[test]
fn t2i_boundary_values_match_interpreter_c_and_nonfpu() {
    let values = [
        0.0, 0.4, 0.5, 1.0, 2.0, 10.0, 1000.0, -0.4, -0.5, -1.0, -2.0, -10.0, -1000.0,
    ];
    let triple = t2i_triple(
        &values,
        &[
            (1, 65535, 0),  // largest int64-safe lattice exponent
            (-1, 65535, 0), // negative largest
            (1, 65535, 1),  // zero flag must win over the exponent
        ],
    );
    let listing = "IN x AS T:*\nOUT = T2I(x)\n";
    let got = jotnar::run_text(listing, &opts(), &t2i_feed(&triple)).unwrap();
    let want = i64_array(got.get("OUT").unwrap());
    let dir = temp_dir("t2i-boundary");
    for (target, name) in [(Target::C, "c"), (Target::NonFpu, "nonfpu")] {
        let got = run_t2i(target, &dir, name, &triple).expect("t2i parity");
        assert_eq!(got, want, "{name}: T2I boundary parity");
    }
}

#[test]
fn t2i_out_of_range_exponent_is_loud() {
    let triple = Triple::from_vecs(&[1], vec![1], vec![65536], vec![0]).unwrap();
    let listing = "IN x AS T:*\nOUT = T2I(x)\n";
    let result = jotnar::run_text(listing, &opts(), &t2i_feed(&triple));
    assert!(
        result.is_err(),
        "interpreter must reject an out-of-range exponent"
    );
    let dir = temp_dir("t2i-range");
    for (target, name) in [(Target::C, "c"), (Target::NonFpu, "nonfpu")] {
        let result = run_t2i(target, &dir, name, &triple);
        assert!(
            result.is_err(),
            "{name}: out-of-range exponent must fail loudly"
        );
    }
}

#[test]
fn iget_rejects_out_of_range_and_nonliteral_indices() {
    let x = value(&[4], &[10, 20, 30, 40]);
    let base_feed = feeds(&[("X", x.clone())]);
    for idx in [-1, 4] {
        let listing = format!("IN X AS I:SEQ\nOUT = IGET(X, {idx})\n");
        let result = jotnar::run_text(&listing, &opts(), &base_feed);
        assert!(result.is_err(), "interpreter must reject IGET index {idx}");
        for target in [Target::C, Target::NonFpu] {
            let opts = CompileOptions {
                target,
                outputs: vec!["OUT".to_string()],
                asm: opts(),
                use_fp16: false,
            };
            let result = compile_program(&listing, &opts, base_feed.clone());
            assert!(
                result.is_err(),
                "{target:?}: IGET index {idx} must be rejected at emit time"
            );
        }
    }

    // len-1 is valid and exact.
    let listing = "IN X AS I:SEQ\nOUT = IGET(X, 3)\n";
    let got = jotnar::run_text(listing, &opts(), &base_feed).unwrap();
    assert_eq!(got.get("OUT").unwrap().first_i64("OUT").unwrap(), 40);
    let dir = temp_dir("iget-valid");
    for (target, name) in [(Target::C, "c"), (Target::NonFpu, "nonfpu")] {
        let opts = CompileOptions {
            target,
            outputs: vec!["OUT".to_string()],
            asm: opts(),
            use_fp16: false,
        };
        let artifact = compile_program(listing, &opts, base_feed.clone()).expect("compile IGET");
        let exe = build(&artifact.source, &dir, name, target).expect("build IGET");
        let x_path = dir.join(format!("{name}-x.bin"));
        write_i64(&x_path, &[10, 20, 30, 40]);
        let out_path = dir.join(format!("{name}-out.bin"));
        let status = Command::new(&exe)
            .arg(&x_path)
            .arg(&out_path)
            .status()
            .expect("run IGET");
        assert!(status.success(), "{name}: valid IGET failed");
        let bytes = fs::read(&out_path).unwrap();
        assert_eq!(i64::from_le_bytes(bytes[0..8].try_into().unwrap()), 40);
    }

    // Non-literal scalar index is refused by the v1 literal-only contract.
    let idx = Value::F64(ArrayD::from_elem(IxDyn(&[]), 0.0));
    let nonliteral = feeds(&[("X", x), ("IDX", idx)]);
    let listing = "IN X AS I:SEQ\nIN IDX AS F:SCALAR\nOUT = IGET(X, IDX)\n";
    let err = jotnar::run_text(listing, &opts(), &nonliteral);
    assert!(
        err.as_ref()
            .is_err_and(|e| e.to_string().contains("literal")),
        "non-literal index must fail with the literal-only message: {err:?}"
    );
    for target in [Target::C, Target::NonFpu] {
        let opts = CompileOptions {
            target,
            outputs: vec!["OUT".to_string()],
            asm: opts(),
            use_fp16: false,
        };
        assert!(
            compile_program(listing, &opts, nonliteral.clone()).is_err(),
            "{target:?}: non-literal IGET index must be rejected"
        );
    }
}

#[test]
fn t2i_full_lattice_exponent_domain_matches_c_and_nonfpu() {
    let e: Vec<i32> = (0..lattice::NLVL).map(|v| v as i32).collect();
    let n = e.len();
    let triple = Triple::from_vecs(&[n], vec![1; n], e, vec![0; n]).unwrap();
    let want: Vec<i64> = lattice::t2i_lut().to_vec();
    let dir = temp_dir("t2i-full-domain");
    for (target, name) in [(Target::C, "c"), (Target::NonFpu, "nonfpu")] {
        let got = run_t2i(target, &dir, name, &triple).expect("full-domain t2i");
        assert_eq!(got.len(), want.len(), "{name}: output length");
        for (i, (a, b)) in got.iter().zip(&want).enumerate() {
            assert_eq!(a, b, "{name}: T2I mismatch at exponent {i}");
        }
    }
}
