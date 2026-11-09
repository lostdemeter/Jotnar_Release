//! Rust emitter parity: reference interpreter vs emitted Rust binary.
use std::fs;
use std::process::Command;

use jotnar::asm::AsmOptions;
use jotnar::codegen::{build, compile_program, CompileOptions, Target};
use jotnar::payload::Feeds;
use jotnar::value::Value;
use ndarray::{ArrayD, IxDyn};

const LISTING: &str = "\
IN a00 AS I
IN a01 AS I
IN b00 AS I
IN b10 AS I
t0 = IMUL(a00, b00)
t1 = IMUL(a01, b10)
OUT = IADD(t0, t1)
";

fn value(v: i64) -> Value {
    Value::I64(ArrayD::from_shape_vec(IxDyn(&[1]), vec![v]).unwrap())
}

#[test]
fn rust_emitter_matches_interpreter() {
    // 2*11 + 3*17 = 73
    let mut feeds = Feeds::new();
    feeds.insert("a00".into(), value(2));
    feeds.insert("a01".into(), value(3));
    feeds.insert("b00".into(), value(11));
    feeds.insert("b10".into(), value(17));
    let opts = CompileOptions {
        target: Target::Rust,
        outputs: vec!["OUT".into()],
        asm: AsmOptions::new("."),
        use_fp16: false,
    };
    let art = compile_program(LISTING, &opts, feeds).expect("compile rust");
    assert!(art.source.contains("fn main()"), "emitted Rust has main");
    let dir = std::env::temp_dir().join("jotnar_rust_emit_test");
    let exe = build(&art.source, &dir, "prog_rust", Target::Rust).expect("build rust");
    // raw i64 payloads
    let write = |name: &str, v: i64| {
        let p = dir.join(name);
        fs::write(&p, v.to_le_bytes()).unwrap();
        p.to_string_lossy().into_owned()
    };
    let a00 = write("a00.i64", 2);
    let a01 = write("a01.i64", 3);
    let b00 = write("b00.i64", 11);
    let b10 = write("b10.i64", 17);
    let out = dir.join("out.i64").to_string_lossy().into_owned();
    let status = Command::new(&exe)
        .args([&a00, &a01, &b00, &b10, &out])
        .status()
        .expect("run rust prog");
    assert!(status.success());
    let got = i64::from_le_bytes(fs::read(&out).unwrap()[..8].try_into().unwrap());
    assert_eq!(got, 73);
}
