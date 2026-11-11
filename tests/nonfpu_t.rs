//! Non-FPU T gates for LAYERNORM and GELU.
//!
//! These two ops were absent from `Target::NonFpu`; DAV2 layer 0 needs
//! both. The emitted integer C must match the reference interpreter's
//! T path exactly (the bridge/LUTs are baked from the same tables).
use std::fs;
use std::path::Path;
use std::process::Command;

use jotnar::asm::AsmOptions;
use jotnar::codegen::{build, compile_program, CompileOptions, Target};
use jotnar::lattice;
use jotnar::payload::Feeds;
use jotnar::value::{Triple, Value};
use ndarray::{ArrayD, IxDyn};

const LISTING: &str = "\
CONFIG m_cov 36200
CONFIG eps_ln 1e-6
IN X AS T:HW
IN W AS T:SEQ
IN B AS T:SEQ
LN = LAYERNORM(X, W, B)
OUT = GELU(LN)
";

fn encode(shape: &[usize], xs: &[f64]) -> Triple {
    lattice::encode(&ArrayD::from_shape_vec(IxDyn(shape), xs.to_vec()).unwrap())
}

fn feeds() -> Feeds {
    let mut f = Feeds::new();
    f.insert(
        "X".into(),
        Value::Triple(encode(
            &[2, 4],
            &[0.1, -0.2, 0.3, 0.4, -0.5, 0.6, -0.7, 0.8],
        )),
    );
    f.insert(
        "W".into(),
        Value::Triple(encode(&[4], &[2.0, 0.5, 2.0, -1.0])),
    );
    f.insert(
        "B".into(),
        Value::Triple(encode(&[4], &[0.1, -0.2, 0.3, 0.0])),
    );
    f
}

fn write_planes(dir: &Path, name: &str, t: &Triple) -> [String; 3] {
    let ps = dir.join(format!("{name}_s.bin"));
    let pe = dir.join(format!("{name}_e.bin"));
    let pz = dir.join(format!("{name}_z.bin"));
    fs::write(&ps, t.s.iter().map(|v| *v as u8).collect::<Vec<u8>>()).unwrap();
    let mut eb = Vec::new();
    for v in t.e.iter() {
        eb.extend_from_slice(&v.to_le_bytes());
    }
    fs::write(&pe, eb).unwrap();
    fs::write(&pz, t.z.iter().copied().collect::<Vec<u8>>()).unwrap();
    [
        ps.to_string_lossy().into_owned(),
        pe.to_string_lossy().into_owned(),
        pz.to_string_lossy().into_owned(),
    ]
}

fn read_planes(dir: &Path, name: &str) -> Triple {
    let s = fs::read(dir.join(format!("{name}_s.bin"))).unwrap();
    let e = fs::read(dir.join(format!("{name}_e.bin"))).unwrap();
    let z = fs::read(dir.join(format!("{name}_z.bin"))).unwrap();
    let sv: Vec<i8> = s.iter().map(|v| *v as i8).collect();
    let ev: Vec<i32> = e
        .chunks(4)
        .map(|c| i32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    let zv: Vec<u8> = z.to_vec();
    Triple::from_vecs(&[2, 4], sv, ev, zv).unwrap()
}

#[test]
fn nonfpu_layernorm_gelu_matches_interpreter() {
    let f = feeds();
    let expected = jotnar::run_text(LISTING, &AsmOptions::new("."), &f)
        .expect("interpreter")
        .get("OUT")
        .cloned()
        .expect("OUT");
    let want = match expected {
        Value::Triple(t) => t,
        other => panic!("expected triple, got {other:?}"),
    };

    let opts = CompileOptions {
        target: Target::NonFpu,
        outputs: vec!["OUT".into()],
        asm: AsmOptions::new("."),
        use_fp16: false,
    };
    let art = compile_program(LISTING, &opts, f.clone()).expect("compile nonfpu");
    let dir = tempfile::tempdir().unwrap();
    let exe = build(&art.source, dir.path(), "nf", Target::NonFpu).expect("build nonfpu");

    // Inputs in listing order: X, W, B.
    let mut argv: Vec<String> = Vec::new();
    for name in ["X", "W", "B"] {
        let t = match f.get(name).unwrap() {
            Value::Triple(t) => t,
            _ => unreachable!(),
        };
        argv.extend(write_planes(dir.path(), name, t));
    }
    let out_s = dir.path().join("OUT_s.bin");
    let out_e = dir.path().join("OUT_e.bin");
    let out_z = dir.path().join("OUT_z.bin");
    argv.push(out_s.to_string_lossy().into_owned());
    argv.push(out_e.to_string_lossy().into_owned());
    argv.push(out_z.to_string_lossy().into_owned());

    let status = Command::new(&exe).args(&argv).status().expect("run");
    assert!(status.success());
    let got = read_planes(dir.path(), "OUT");
    assert_eq!(got.s, want.s, "sign plane");
    assert_eq!(got.e, want.e, "exponent plane");
    assert_eq!(got.z, want.z, "zero plane");
}
