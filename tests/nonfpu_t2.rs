//! Non-FPU emission gates for the opt-in T+mantissa (T2) format.
//!
//! The interpreter (via `phi_core::t2`) is the reference. Emitted integer C
//! must match it bit-for-bit on all four planes (s, e, m, z) and pass the
//! float trap.

use std::fs;
use std::path::Path;
use std::process::Command;

use jotnar::asm::AsmOptions;
use jotnar::codegen::{build, compile_program, CompileOptions, Target};
use jotnar::payload::Feeds;
use jotnar::t2;
use jotnar::value::{Triple, Value};
use ndarray::{ArrayD, IxDyn};

const LISTING: &str = "\
CONFIG t2 24
CONFIG bridge_f 24
CONFIG eps_ln 1e-6
IN A
IN B
IN B2
IN X
IN W
IN BB
M = MUL(A, B)
S = ADD(A, B)
D = SUB(A, B)
LN = LAYERNORM(X, W, BB)
G = GELU(X)
SM = SOFTMAX(X)
MM = MATMUL(A, B2)
";

fn arr(shape: &[usize], v: &[f64]) -> Triple {
    t2::encode_array(
        &ArrayD::from_shape_vec(IxDyn(shape), v.to_vec()).unwrap(),
        24,
    )
}

fn feeds() -> Feeds {
    let mut f = Feeds::new();
    f.insert(
        "A".into(),
        Value::Triple(arr(&[2, 3], &[0.5, -1.25, 2.0, -0.75, 3.5, 0.125])),
    );
    f.insert(
        "B".into(),
        Value::Triple(arr(&[2, 3], &[-0.25, 2.5, -1.0, 4.0, 0.5, -2.0])),
    );
    f.insert(
        "B2".into(),
        Value::Triple(arr(&[3, 2], &[0.5, -1.5, 2.0, 0.25, -0.75, 1.25])),
    );
    f.insert(
        "X".into(),
        Value::Triple(arr(&[2, 3], &[0.1, -0.2, 0.3, 1.5, -2.5, 0.5])),
    );
    f.insert("W".into(), Value::Triple(arr(&[3], &[2.0, 0.5, 1.5])));
    f.insert("BB".into(), Value::Triple(arr(&[3], &[0.1, -0.2, 0.3])));
    f
}

fn write_planes(dir: &Path, name: &str, t: &Triple) -> Vec<String> {
    let ps = dir.join(format!("{name}_s.bin"));
    let pe = dir.join(format!("{name}_e.bin"));
    let pm = dir.join(format!("{name}_m.bin"));
    let pz = dir.join(format!("{name}_z.bin"));
    fs::write(&ps, t.s.iter().map(|v| *v as u8).collect::<Vec<u8>>()).unwrap();
    let mut eb = Vec::new();
    for v in t.e.iter() {
        eb.extend_from_slice(&v.to_le_bytes());
    }
    fs::write(&pe, eb).unwrap();
    let mut mb = Vec::new();
    for v in t.m.as_ref().expect("t2").iter() {
        mb.extend_from_slice(&v.to_le_bytes());
    }
    fs::write(&pm, mb).unwrap();
    fs::write(&pz, t.z.iter().copied().collect::<Vec<u8>>()).unwrap();
    vec![
        ps.to_string_lossy().into_owned(),
        pe.to_string_lossy().into_owned(),
        pm.to_string_lossy().into_owned(),
        pz.to_string_lossy().into_owned(),
    ]
}

fn read_planes(dir: &Path, name: &str, shape: &[usize]) -> Triple {
    let s = fs::read(dir.join(format!("{name}_s.bin"))).unwrap();
    let e = fs::read(dir.join(format!("{name}_e.bin"))).unwrap();
    let m = fs::read(dir.join(format!("{name}_m.bin"))).unwrap();
    let z = fs::read(dir.join(format!("{name}_z.bin"))).unwrap();
    let sv: Vec<i8> = s.iter().map(|v| *v as i8).collect();
    let ev: Vec<i32> = e
        .chunks(4)
        .map(|c| i32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    let mv: Vec<i32> = m
        .chunks(4)
        .map(|c| i32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    Triple::from_vecs_opt(shape, sv, ev, z.to_vec(), Some(mv)).unwrap()
}

#[test]
fn nonfpu_t2_matches_interpreter() {
    let f = feeds();
    let expected = jotnar::run_text(LISTING, &AsmOptions::new("."), &f).expect("interpreter");
    let opts = CompileOptions {
        target: Target::NonFpu,
        outputs: vec![
            "M".into(),
            "S".into(),
            "D".into(),
            "LN".into(),
            "G".into(),
            "SM".into(),
            "MM".into(),
        ],
        asm: AsmOptions::new("."),
        use_fp16: false,
    };
    let art = compile_program(LISTING, &opts, f.clone()).expect("compile nonfpu t2");
    let dir = tempfile::tempdir().unwrap();
    let exe = build(&art.source, dir.path(), "nft2", Target::NonFpu).expect("build");

    let mut argv: Vec<String> = Vec::new();
    for name in ["A", "B", "B2", "X", "W", "BB"] {
        let t = match f.get(name).unwrap() {
            Value::Triple(t) => t,
            _ => unreachable!(),
        };
        argv.extend(write_planes(dir.path(), name, t));
    }
    for name in ["M", "S", "D", "LN", "G", "SM", "MM"] {
        for comp in ["s", "e", "m", "z"] {
            argv.push(
                dir.path()
                    .join(format!("{name}_{comp}.bin"))
                    .to_string_lossy()
                    .into_owned(),
            );
        }
    }
    let status = Command::new(&exe).args(&argv).status().expect("run");
    assert!(status.success());

    let shapes: &[(&str, &[usize])] = &[
        ("M", &[2, 3]),
        ("S", &[2, 3]),
        ("D", &[2, 3]),
        ("LN", &[2, 3]),
        ("G", &[2, 3]),
        ("SM", &[2, 3]),
        ("MM", &[2, 2]),
    ];
    for (name, shape) in shapes {
        let got = read_planes(dir.path(), name, shape);
        let want = match expected.get(*name).unwrap() {
            Value::Triple(t) => t,
            _ => unreachable!(),
        };
        assert_eq!(&got.s, &want.s, "{name} sign");
        assert_eq!(&got.e, &want.e, "{name} exponent");
        assert_eq!(
            got.m.as_ref().unwrap(),
            want.m.as_ref().unwrap(),
            "{name} mantissa"
        );
        assert_eq!(&got.z, &want.z, "{name} zero");
    }
}

fn single_feed() -> Feeds {
    let mut f = Feeds::new();
    f.insert(
        "A".into(),
        Value::Triple(arr(&[2, 2], &[0.5, -1.25, 2.0, -0.75])),
    );
    f
}

#[test]
fn t2_requires_nonfpu_target() {
    let listing = "CONFIG t2 24\nIN A AS T:HW\nOUT = TRANSPOSE(A)\n";
    for target in [Target::C, Target::Rust, Target::Cuda] {
        let opts = CompileOptions {
            target,
            outputs: vec!["OUT".into()],
            asm: AsmOptions::new("."),
            use_fp16: false,
        };
        let err = compile_program(listing, &opts, single_feed())
            .expect_err("T2 must not lower on a non-NonFpu target");
        let msg = err.to_string();
        assert!(
            msg.contains("T2 requires --target nonfpu"),
            "target {target:?}: {msg}"
        );
    }
}

#[test]
fn invalid_t2_widths_are_typed_errors() {
    for cfg in ["t2 64", "t2 0", "bridge_f 0", "bridge_f 13", "bridge_f 63"] {
        let listing = format!("CONFIG {cfg}\nIN A AS T:HW\nOUT = MUL(A, A)\n");
        let r = jotnar::run_text(&listing, &AsmOptions::new("."), &single_feed());
        assert!(r.is_err(), "CONFIG {cfg} must refuse");
        let msg = r.unwrap_err().to_string();
        assert!(
            msg.contains("t2") || msg.contains("bridge_f"),
            "CONFIG {cfg}: {msg}"
        );
    }
    // bridge_f below M is raised to the validated T2 bridge width.
    let listing = "CONFIG t2 24\nCONFIG bridge_f 18\nIN A AS T:HW\nOUT = MUL(A, A)\n";
    assert!(jotnar::run_text(listing, &AsmOptions::new("."), &single_feed()).is_ok());
}
