//! Rust emitter parity: reference interpreter vs emitted Rust binary.
use std::fs;
use std::process::Command;

use jotnar::asm::AsmOptions;
use jotnar::codegen::{build, compile_program, CompileOptions, Target};
use jotnar::payload::Feeds;
use jotnar::value::Value;
use ndarray::{ArrayD, IxDyn};

fn value(v: i64) -> Value {
    Value::I64(ArrayD::from_shape_vec(IxDyn(&[1]), vec![v]).unwrap())
}
fn fvalue(shape: &[usize], xs: &[f64]) -> Value {
    Value::F64(ArrayD::from_shape_vec(IxDyn(shape), xs.to_vec()).unwrap())
}
fn feeds(pairs: &[(&str, Value)]) -> Feeds {
    let mut f = Feeds::new();
    for (k, v) in pairs {
        f.insert((*k).into(), v.clone());
    }
    f
}
fn opts(outputs: &[&str]) -> CompileOptions {
    CompileOptions {
        target: Target::Rust,
        outputs: outputs.iter().map(|s| s.to_string()).collect(),
        asm: AsmOptions::new("."),
        use_fp16: false,
    }
}
fn write_i64(dir: &std::path::Path, name: &str, v: i64) -> String {
    let p = dir.join(name);
    fs::write(&p, v.to_le_bytes()).unwrap();
    p.to_string_lossy().into_owned()
}
fn write_f64(dir: &std::path::Path, name: &str, v: f64) -> String {
    let p = dir.join(name);
    fs::write(&p, v.to_le_bytes()).unwrap();
    p.to_string_lossy().into_owned()
}
fn read_i64(p: &str) -> i64 {
    i64::from_le_bytes(fs::read(p).unwrap()[..8].try_into().unwrap())
}

#[test]
fn rust_emitter_matches_interpreter() {
    let listing = "\
IN a00 AS I
IN a01 AS I
IN b00 AS I
IN b10 AS I
t0 = IMUL(a00, b00)
t1 = IMUL(a01, b10)
OUT = IADD(t0, t1)
";
    let f = feeds(&[
        ("a00", value(2)),
        ("a01", value(3)),
        ("b00", value(11)),
        ("b10", value(17)),
    ]);
    let art = compile_program(listing, &opts(&["OUT"]), f).expect("compile rust");
    assert!(art.source.contains("fn main()"));
    let dir = tempfile::tempdir().unwrap();
    let exe = build(&art.source, dir.path(), "prog_rust", Target::Rust).expect("build rust");
    let a00 = write_i64(dir.path(), "a00.i64", 2);
    let a01 = write_i64(dir.path(), "a01.i64", 3);
    let b00 = write_i64(dir.path(), "b00.i64", 11);
    let b10 = write_i64(dir.path(), "b10.i64", 17);
    let out = dir.path().join("out.i64").to_string_lossy().into_owned();
    let status = Command::new(&exe)
        .args([&a00, &a01, &b00, &b10, &out])
        .status()
        .expect("run");
    assert!(status.success());
    assert_eq!(read_i64(&out), 73);
}

#[test]
fn scalar_operands_right_supported() {
    // Scalar on the right is the upstream contract (C infer + pattern).
    // Scalar on the left has no first operand stream and is refused by
    // infer with a typed error (see scalar_left_refused below) — that is
    // the loud boundary, not invalid emitted source.
    for listing in [
        "IN A AS I\nOUT = IADD(A, 1)\n",
        "IN A AS I\nOUT = ISUB(A, 1)\n",
        "IN A AS I\nOUT = IMUL(A, 2)\n",
        "IN A AS I\nOUT = IDIV(A, 2)\n",
    ] {
        let f = feeds(&[("A", value(6))]);
        let art = compile_program(listing, &opts(&["OUT"]), f).expect("scalar compile");
        // source must not contain literal-index bug
        assert!(!art.source.contains("1[i]"), "literal indexed: {listing}");
        let dir = tempfile::tempdir().unwrap();
        let exe = build(&art.source, dir.path(), "prog", Target::Rust).expect("build");
        let a = write_i64(dir.path(), "a.i64", 6);
        let out = dir.path().join("out.i64").to_string_lossy().into_owned();
        let status = Command::new(&exe).args([&a, &out]).status().expect("run");
        assert!(status.success(), "run failed for {listing}");
        let got = read_i64(&out);
        let want = match listing {
            s if s.contains("IADD") => 7,
            s if s.contains("ISUB(A") => 5,
            s if s.contains("ISUB(1") => -5,
            s if s.contains("IMUL") => 12,
            s if s.contains("IDIV(A") => 3,
            _ => 1,
        };
        assert_eq!(got, want, "for {listing}");
    }
}

#[test]
fn idiv_guards_match_interpreter() {
    let listing = "IN A AS I\nIN B AS I\nOUT = IDIV(A, B)\n";
    for (a, b, ok) in [
        (10i64, 0i64, false),
        (i64::MIN, -1i64, false),
        (20, 4, true),
    ] {
        let f = feeds(&[("A", value(a)), ("B", value(b))]);
        let art = compile_program(listing, &opts(&["OUT"]), f).expect("compile");
        let dir = tempfile::tempdir().unwrap();
        let exe = build(&art.source, dir.path(), "prog", Target::Rust).expect("build");
        let a_f = write_i64(dir.path(), "a.i64", a);
        let b_f = write_i64(dir.path(), "b.i64", b);
        let out = dir.path().join("out.i64").to_string_lossy().into_owned();
        let status = Command::new(&exe)
            .args([&a_f, &b_f, &out])
            .status()
            .expect("run");
        assert_eq!(status.success(), ok, "a={a} b={b}");
        if ok {
            assert_eq!(read_i64(&out), 5);
        }
    }
}

#[test]
fn keyword_stream_names_are_valid_rust() {
    for name in ["fn", "let", "type", "self", "Self", "super", "crate"] {
        let listing = format!("IN {name} AS I\nOUT = IADD({name}, {name})\n");
        let f = feeds(&[(name, value(21))]);
        let art = compile_program(&listing, &opts(&["OUT"]), f).expect("keyword compile");
        assert!(!art.source.contains("r#{out}"), "bad raw idem");
        let dir = tempfile::tempdir().unwrap();
        let exe = build(&art.source, dir.path(), "prog", Target::Rust).expect("build");
        let a = write_i64(dir.path(), "a.i64", 21);
        let out = dir.path().join("out.i64").to_string_lossy().into_owned();
        let status = Command::new(&exe).args([&a, &out]).status().expect("run");
        assert!(status.success(), "keyword {name} failed");
        assert_eq!(read_i64(&out), 42);
    }
}

#[test]
fn t_streams_refused_loudly() {
    // T inputs/outputs must refuse, not silently f64.
    let listing = "IN A AS T:HW\nOUT = IADD(A, A)\n";
    // Build a T feed
    let t = Value::Triple(
        jotnar::value::Triple::from_vecs(&[2, 2], vec![1i8; 4], vec![32768i32; 4], vec![0u8; 4])
            .unwrap(),
    );
    let f = feeds(&[("A", t)]);
    let r = compile_program(listing, &opts(&["OUT"]), f);
    assert!(r.is_err(), "T input must be refused");
    assert!(r.unwrap_err().to_string().contains("T (triple)"));
}

#[test]
fn multi_output_and_float_matmul() {
    // Multi-output: two I outputs
    let listing = "IN A AS I\nIN B AS I\nO1 = IADD(A, B)\nO2 = ISUB(A, B)\n";
    let f = feeds(&[("A", value(10)), ("B", value(3))]);
    let art = compile_program(listing, &opts(&["O1", "O2"]), f).expect("multi");
    let dir = tempfile::tempdir().unwrap();
    let exe = build(&art.source, dir.path(), "prog", Target::Rust).expect("build");
    let a = write_i64(dir.path(), "a.i64", 10);
    let b = write_i64(dir.path(), "b.i64", 3);
    let o1 = dir.path().join("o1.i64").to_string_lossy().into_owned();
    let o2 = dir.path().join("o2.i64").to_string_lossy().into_owned();
    let status = Command::new(&exe)
        .args([&a, &b, &o1, &o2])
        .status()
        .expect("run");
    assert!(status.success());
    assert_eq!(read_i64(&o1), 13);
    assert_eq!(read_i64(&o2), 7);

    // Float MATMUL 2x2
    let listing = "IN A AS F:HW\nIN B AS F:HW\nOUT = MATMUL(A, B)\n";
    let f = feeds(&[
        ("A", fvalue(&[2, 2], &[1.0, 2.0, 3.0, 4.0])),
        ("B", fvalue(&[2, 2], &[5.0, 6.0, 7.0, 8.0])),
    ]);
    let art = compile_program(listing, &opts(&["OUT"]), f).expect("matmul");
    let dir = tempfile::tempdir().unwrap();
    let exe = build(&art.source, dir.path(), "prog", Target::Rust).expect("build matmul");
    // raw f64 files
    let write = |n: &str, v: &[f64]| {
        let p = dir.path().join(n);
        let mut b = Vec::new();
        for x in v {
            b.extend_from_slice(&x.to_le_bytes());
        }
        fs::write(&p, b).unwrap();
        p.to_string_lossy().into_owned()
    };
    let a = write("a.f64", &[1.0, 2.0, 3.0, 4.0]);
    let b = write("b.f64", &[5.0, 6.0, 7.0, 8.0]);
    let out = dir.path().join("out.f64").to_string_lossy().into_owned();
    let status = Command::new(&exe)
        .args([&a, &b, &out])
        .status()
        .expect("run matmul");
    assert!(status.success());
    let raw = fs::read(&out).unwrap();
    let vals: Vec<f64> = raw
        .chunks_exact(8)
        .map(|c| f64::from_le_bytes(c.try_into().unwrap()))
        .collect();
    for (g, w) in vals.iter().zip([19.0, 22.0, 43.0, 50.0]) {
        assert!((g - w).abs() < 1e-9, "matmul {vals:?}");
    }
}

#[test]
fn conv_matches_interpreter() {
    // Edge-replicate 3x3 over a 3x4 image: the Rust lowering must be
    // the C/interpreter contract, not a zero-padded variant.
    let listing = "IN X AS F:HW\nIN K AS F:HW\nOUT = CONV(X, K)\n";
    let xs = [
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
    ];
    let ks = [0.0, 1.0, 0.0, 1.0, -4.0, 1.0, 0.0, 1.0, 0.0];
    let f = feeds(&[("X", fvalue(&[3, 4], &xs)), ("K", fvalue(&[3, 3], &ks))]);
    // The reference interpreter's CONV is the triple path; the C backend
    // is the F-path contract the Rust emitter mirrors, so compare against
    // C rather than the interpreter here.
    let art = compile_program(listing, &opts(&["OUT"]), f.clone()).expect("compile conv rust");
    assert!(
        art.source.contains("for ky in 0..3"),
        "missing CONV lowering"
    );
    let cargs = CompileOptions {
        target: Target::C,
        ..opts(&["OUT"])
    };
    let cart = compile_program(listing, &cargs, f).expect("compile conv c");
    let dir = tempfile::tempdir().unwrap();
    let rexe = build(&art.source, dir.path(), "prog_rust", Target::Rust).expect("build rust");
    let cexe = build(&cart.source, dir.path(), "prog_c", Target::C).expect("build c");
    let write = |n: &str, v: &[f64]| {
        let p = dir.path().join(n);
        let mut b = Vec::new();
        for x in v {
            b.extend_from_slice(&x.to_le_bytes());
        }
        fs::write(&p, b).unwrap();
        p.to_string_lossy().into_owned()
    };
    let x = write("x.f64", &xs);
    let k = write("k.f64", &ks);
    let rout = dir.path().join("rout.f64").to_string_lossy().into_owned();
    let cout = dir.path().join("cout.f64").to_string_lossy().into_owned();
    let status = Command::new(&rexe)
        .args([&x, &k, &rout])
        .status()
        .expect("run rust conv");
    assert!(status.success());
    let status = Command::new(&cexe)
        .args([&x, &k, &cout])
        .status()
        .expect("run c conv");
    assert!(status.success());
    let read = |p: &str| {
        fs::read(p)
            .unwrap()
            .chunks(8)
            .map(|c| f64::from_le_bytes(c.try_into().unwrap()))
            .collect::<Vec<f64>>()
    };
    let got = read(&rout);
    let want = read(&cout);
    assert_eq!(got.len(), want.len());
    for (g, w) in got.iter().zip(want.iter()) {
        assert!((g - w).abs() < 1e-12, "conv {got:?} vs {want:?}");
    }
    // A 3x3 Laplacian over a planar ramp interior is zero; edges bend.
    assert!(got[5].abs() < 1e-9, "interior {got:?}");
}

#[test]
fn transpose_and_concat_match_c() {
    // The transformer block needs the two shape moves: K^T for scores and
    // the per-head column concat. Compare Rust against the C contract.
    let listing = "IN A AS F:HW\nOUT = TRANSPOSE(A)\n";
    let xs = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let f = feeds(&[("A", fvalue(&[2, 3], &xs))]);
    let art = compile_program(listing, &opts(&["OUT"]), f.clone()).expect("compile transpose");
    let cart = compile_program(
        listing,
        &CompileOptions {
            target: Target::C,
            ..opts(&["OUT"])
        },
        f,
    )
    .expect("compile transpose c");
    let dir = tempfile::tempdir().unwrap();
    let rexe = build(&art.source, dir.path(), "tr", Target::Rust).expect("build transpose");
    let cexe = build(&cart.source, dir.path(), "tc", Target::C).expect("build transpose c");
    let write = |n: &str, v: &[f64]| {
        let p = dir.path().join(n);
        let mut b = Vec::new();
        for x in v {
            b.extend_from_slice(&x.to_le_bytes());
        }
        fs::write(&p, b).unwrap();
        p.to_string_lossy().into_owned()
    };
    let read = |p: &str| {
        fs::read(p)
            .unwrap()
            .chunks(8)
            .map(|c| f64::from_le_bytes(c.try_into().unwrap()))
            .collect::<Vec<f64>>()
    };
    let a = write("a.f64", &xs);
    let ro = dir.path().join("ro.f64").to_string_lossy().into_owned();
    let co = dir.path().join("co.f64").to_string_lossy().into_owned();
    assert!(Command::new(&rexe)
        .args([&a, &ro])
        .status()
        .unwrap()
        .success());
    assert!(Command::new(&cexe)
        .args([&a, &co])
        .status()
        .unwrap()
        .success());
    assert_eq!(read(&ro), read(&co));
    assert_eq!(read(&ro), vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);

    // CONCAT along columns: [2,2] ++ [2,1].
    let listing = "IN A AS F:HW\nIN B AS F:HW\nOUT = CONCAT(A, B, 1)\n";
    let f = feeds(&[
        ("A", fvalue(&[2, 2], &[1.0, 2.0, 3.0, 4.0])),
        ("B", fvalue(&[2, 1], &[9.0, 8.0])),
    ]);
    let art = compile_program(listing, &opts(&["OUT"]), f.clone()).expect("compile concat");
    let cart = compile_program(
        listing,
        &CompileOptions {
            target: Target::C,
            ..opts(&["OUT"])
        },
        f,
    )
    .expect("compile concat c");
    let rexe = build(&art.source, dir.path(), "cr", Target::Rust).expect("build concat");
    let cexe = build(&cart.source, dir.path(), "cc", Target::C).expect("build concat c");
    let a = write("ca.f64", &[1.0, 2.0, 3.0, 4.0]);
    let b = write("cb.f64", &[9.0, 8.0]);
    let ro = dir.path().join("cro.f64").to_string_lossy().into_owned();
    let co = dir.path().join("cco.f64").to_string_lossy().into_owned();
    assert!(Command::new(&rexe)
        .args([&a, &b, &ro])
        .status()
        .unwrap()
        .success());
    assert!(Command::new(&cexe)
        .args([&a, &b, &co])
        .status()
        .unwrap()
        .success());
    assert_eq!(read(&ro), read(&co));
    assert_eq!(read(&ro), vec![1.0, 2.0, 9.0, 3.0, 4.0, 8.0]);
}

#[test]
fn gelu_matches_exact_erf() {
    // HF DINOv2 uses exact (erf) GELU; the emitted Rust uses the A&S
    // approximation (|err| < 1.5e-7), so compare at 1e-6.
    let listing = "IN A AS F:HW\nOUT = GELU(A)\n";
    let xs = [-2.0, -0.5, 0.0, 0.5, 2.0];
    let want = [
        -0.04550026389635842,
        -0.15426876936299344,
        0.0,
        0.34573123063700656,
        1.9544997361036416,
    ];
    let f = feeds(&[("A", fvalue(&[1, 5], &xs))]);
    let art = compile_program(listing, &opts(&["OUT"]), f).expect("compile gelu");
    let dir = tempfile::tempdir().unwrap();
    let exe = build(&art.source, dir.path(), "gelu", Target::Rust).expect("build gelu");
    let p = dir.path().join("a.f64");
    let mut b = Vec::new();
    for x in xs {
        b.extend_from_slice(&x.to_le_bytes());
    }
    fs::write(&p, b).unwrap();
    let out = dir.path().join("out.f64").to_string_lossy().into_owned();
    assert!(Command::new(&exe)
        .args([p.to_string_lossy().as_ref(), &out])
        .status()
        .unwrap()
        .success());
    let got: Vec<f64> = fs::read(&out)
        .unwrap()
        .chunks(8)
        .map(|c| f64::from_le_bytes(c.try_into().unwrap()))
        .collect();
    for (g, w) in got.iter().zip(want.iter()) {
        assert!((g - w).abs() < 1e-6, "gelu {got:?} vs {want:?}");
    }
}

#[test]
fn layernorm_eps_comes_from_config() {
    // The C target reads CONFIG eps_ln; the Rust target must do the same
    // instead of hardcoding a value. Compare against C and refuse a
    // listing that omits the key.
    let listing = "\
CONFIG eps_ln 1e-6
IN X AS F:HW
IN W AS F:HW
IN B AS F:HW
OUT = LAYERNORM(X, W, B)
";
    let f = feeds(&[
        ("X", fvalue(&[2, 3], &[1.0, 2.0, 3.0, 10.0, 20.0, 30.0])),
        ("W", fvalue(&[3], &[1.0, 0.5, 2.0])),
        ("B", fvalue(&[3], &[0.1, -0.2, 0.3])),
    ]);
    let art = compile_program(listing, &opts(&["OUT"]), f.clone()).expect("compile ln");
    assert!(art.source.contains("1e-6"), "eps not read from config");
    let cart = compile_program(
        listing,
        &CompileOptions {
            target: Target::C,
            ..opts(&["OUT"])
        },
        f,
    )
    .expect("compile ln c");
    let dir = tempfile::tempdir().unwrap();
    let rexe = build(&art.source, dir.path(), "lnr", Target::Rust).expect("build ln");
    let cexe = build(&cart.source, dir.path(), "lnc", Target::C).expect("build ln c");
    let write = |n: &str, v: &[f64]| {
        let p = dir.path().join(n);
        let mut b = Vec::new();
        for x in v {
            b.extend_from_slice(&x.to_le_bytes());
        }
        fs::write(&p, b).unwrap();
        p.to_string_lossy().into_owned()
    };
    let x = write("x.f64", &[1.0, 2.0, 3.0, 10.0, 20.0, 30.0]);
    let w = write("w.f64", &[1.0, 0.5, 2.0]);
    let b = write("b.f64", &[0.1, -0.2, 0.3]);
    let ro = dir.path().join("ro.f64").to_string_lossy().into_owned();
    let co = dir.path().join("co.f64").to_string_lossy().into_owned();
    assert!(Command::new(&rexe)
        .args([&x, &w, &b, &ro])
        .status()
        .unwrap()
        .success());
    assert!(Command::new(&cexe)
        .args([&x, &w, &b, &co])
        .status()
        .unwrap()
        .success());
    let read = |p: &str| {
        fs::read(p)
            .unwrap()
            .chunks(8)
            .map(|c| f64::from_le_bytes(c.try_into().unwrap()))
            .collect::<Vec<f64>>()
    };
    assert_eq!(read(&ro), read(&co));

    let no_cfg = "IN X AS F:HW\nIN W AS F:HW\nIN B AS F:HW\nOUT = LAYERNORM(X, W, B)\n";
    let f = feeds(&[
        ("X", fvalue(&[2, 3], &[1.0, 2.0, 3.0, 10.0, 20.0, 30.0])),
        ("W", fvalue(&[3], &[1.0, 0.5, 2.0])),
        ("B", fvalue(&[3], &[0.1, -0.2, 0.3])),
    ]);
    let err = compile_program(no_cfg, &opts(&["OUT"]), f)
        .err()
        .expect("missing eps_ln must refuse");
    assert!(err.to_string().contains("eps_ln"), "{err}");
}

#[test]
fn scalar_left_refused_loudly() {
    // Upstream infer requires the first operand to be a stream; a literal
    // first operand (IADD(1, A)) must refuse with a typed error, never
    // emit indexed-literal source.
    let listing = "IN A AS I\nOUT = IADD(1, A)\n";
    let f = feeds(&[("A", value(6))]);
    let r = compile_program(listing, &opts(&["OUT"]), f);
    assert!(r.is_err());
    let err = r.unwrap_err().to_string();
    assert!(
        err.contains("first operand") || err.contains("unknown stream"),
        "got {err}"
    );
}
