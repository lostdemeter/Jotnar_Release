use std::collections::BTreeMap;
use std::path::PathBuf;

use jotnar::asm::{self, AsmOptions};
use jotnar::lattice;
use jotnar::value::Value;
use ndarray::ArrayD;

fn opts() -> AsmOptions {
    AsmOptions::new(PathBuf::from("."))
}

#[test]
fn unknown_mnemonic_is_loud() {
    let err = asm::assemble("IN a\nOUT = NOPE(a)\n", &opts()).unwrap_err();
    assert!(err.to_string().contains("unknown mnemonic"), "{err}");
}

#[test]
fn arity_mismatch_is_loud() {
    let err = asm::assemble("IN a\nOUT = ADD(a)\n", &opts()).unwrap_err();
    assert!(err.to_string().contains("wants 2 args"), "{err}");
}

#[test]
fn use_before_def_is_loud() {
    let err = asm::assemble("IN a\nOUT = ADD(a, b)\n", &opts()).unwrap_err();
    assert!(err.to_string().contains("not defined yet"), "{err}");
}

#[test]
fn def_call_namespaces_internal_streams() {
    let text = "\
IN a
DEF twice(x) -> (y)
  T = ADD(x, x)
  y = ADD(T, x)
END
OUT = CALL twice(a)
";
    let bound = asm::assemble(text, &opts()).unwrap();
    let mns: Vec<&str> = bound.instrs.iter().map(|i| i.mn.as_str()).collect();
    assert_eq!(mns, vec!["ADD", "ADD"]);
    assert_eq!(bound.instrs[0].outs, vec!["twice#1.T"]);
    assert_eq!(bound.instrs[1].outs, vec!["OUT"]);
}

#[test]
fn repeat_unrolls_without_break() {
    let text = "IN a\nREPEAT 3\nOUT = ADD(a, a)\nEND\n";
    let bound = asm::assemble(text, &opts()).unwrap();
    assert_eq!(bound.instrs.len(), 3);
}

#[test]
fn repeat_keeps_a_loop_when_break_is_present() {
    let text = "IN a\nREPEAT 3\nOUT = ADD(a, a)\nBREAK OUT\nEND\n";
    let bound = asm::assemble(text, &opts()).unwrap();
    assert!(bound.instrs.iter().any(|i| i.is_repeat()));
    assert!(bound.instrs.iter().any(|i| i.is_break()));
}

#[test]
fn verify_reports_layout_conflict() {
    let text = "IN a AS F:SEQ\nOUT = SQRT(a)\n";
    let report = asm::verify(text, &opts()).unwrap();
    assert_eq!(report.errors.len(), 1);
    assert!(
        report.errors[0].contains("layout mismatch"),
        "{:?}",
        report.errors
    );
}

#[test]
fn verify_reports_coverage_honestly() {
    let text = "IN a\nOUT = ADD(a, a)\n";
    let report = asm::verify(text, &opts()).unwrap();
    assert!(report.errors.is_empty());
    assert_eq!(report.resolved, 0);
    assert_eq!(report.total, 2);
}

#[test]
fn runtime_break_exits_the_loop() {
    let text = "\
IN zero
REPEAT 10
  OUT = IADD(zero, 1)
  BREAK OUT
END
";
    let mut feeds = BTreeMap::new();
    feeds.insert(
        "zero".to_string(),
        Value::I64(ArrayD::from_elem(ndarray::IxDyn(&[1]), 0i64)),
    );
    let got = jotnar::run_text(text, &opts(), &feeds).unwrap();
    match got.get("OUT").unwrap() {
        Value::I64(a) => assert_eq!(*a.iter().next().unwrap(), 1),
        other => panic!("unexpected {other:?}"),
    }
}

#[test]
fn runtime_triples_add_is_exact_on_twins() {
    let text = "IN a AS T:HW\nOUT = ADD(a, a)\n";
    let t = lattice::encode(&ArrayD::from_elem(ndarray::IxDyn(&[2, 2]), 0.25f64));
    let mut feeds = BTreeMap::new();
    feeds.insert("a".to_string(), Value::Triple(t));
    let got = jotnar::run_text(text, &opts(), &feeds).unwrap();
    let out = got.get("OUT").unwrap().triple("OUT").unwrap();
    // 0.25 + 0.25 = 0.5; the lattice bridge is exact for these.
    let dec = lattice::decode(&out.s, &out.e);
    let z = out.z.iter().copied().collect::<Vec<_>>();
    for (v, zz) in dec.iter().zip(z.iter()) {
        assert!((v * (1.0 - *zz as f64) - 0.5).abs() < 1e-3, "{v}");
    }
}

#[test]
fn config_overrides_scales() {
    let text = "CONFIG m_acc 40000\nCONFIG m_cov 40000\nIN a AS T:SEQ\nOUT = ADD(a, a)\n";
    let bound = asm::assemble(text, &opts()).unwrap();
    assert_eq!(bound.config.get_i64("m_acc"), Some(40000));
}

#[test]
fn in_as_applies_to_all_names() {
    let bound = asm::assemble("IN a, b AS T:SEQ\nOUT = ADD(a, b)\n", &opts()).unwrap();
    assert_eq!(bound.inputs.len(), 2);
    assert_eq!(bound.inputs[0].layout.as_deref(), Some("T:SEQ"));
    assert_eq!(bound.inputs[1].layout.as_deref(), Some("T:SEQ"));
}

#[test]
fn duplicate_in_is_loud() {
    let err = asm::assemble("IN a\nIN a\nOUT = ADD(a, a)\n", &opts()).unwrap_err();
    assert!(err.to_string().contains("duplicate IN"), "{err}");
}

#[test]
fn call_arity_is_loud() {
    let text = "IN a\nDEF f(x) -> (y)\ny = ADD(x, x)\nEND\nOUT = CALL f(a, a)\n";
    let err = asm::assemble(text, &opts()).unwrap_err();
    assert!(err.to_string().contains("wants 1 args"), "{err}");
}

#[test]
fn recursive_call_is_loud() {
    let text = "IN a\nDEF f(x) -> (y)\ny = CALL f(x)\nEND\nOUT = CALL f(a)\n";
    let err = asm::assemble(text, &opts()).unwrap_err();
    assert!(err.to_string().contains("recursive CALL"), "{err}");
}

#[test]
fn call_site_layout_contract_is_loud() {
    let text = "IN a AS T:SEQ\nDEF f(x AS F:SEQ) -> (y)\ny = x\nEND\nOUT = CALL f(a)\n";
    let err = asm::assemble(text, &opts()).unwrap_err();
    assert!(err.to_string().contains("violates formal"), "{err}");
}

#[test]
fn state_must_be_assigned() {
    let err = asm::assemble("IN a\nSTATE a\nOUT = ADD(a, a)\n", &opts()).unwrap_err();
    assert!(err.to_string().contains("never assigned as OUT"), "{err}");
}

#[test]
fn duplicate_state_is_loud() {
    let err = asm::assemble("IN a\nSTATE a\nSTATE a\nOUT = a\n", &opts()).unwrap_err();
    assert!(err.to_string().contains("duplicate STATE"), "{err}");
}

#[test]
fn break_outside_repeat_is_loud() {
    let err = asm::assemble("IN a\nBREAK a\nOUT = ADD(a, a)\n", &opts()).unwrap_err();
    assert!(err.to_string().contains("BREAK outside REPEAT"), "{err}");
}

#[test]
fn payload_roundtrip_preserves_values() {
    use jotnar::payload::{feeds_from_json, feeds_to_json};
    let mut feeds = BTreeMap::new();
    feeds.insert(
        "x".to_string(),
        Value::I64(
            ArrayD::from_shape_vec(ndarray::IxDyn(&[2]), vec![1, -9_007_199_254_740_993]).unwrap(),
        ),
    );
    let t = lattice::encode(&ArrayD::from_elem(ndarray::IxDyn(&[2]), 0.3f64));
    feeds.insert("t".to_string(), Value::Triple(t));
    let json = feeds_to_json(&feeds, None).unwrap();
    let back = feeds_from_json(&json).unwrap();
    match back.get("x").unwrap() {
        Value::I64(a) => {
            assert_eq!(
                &a.iter().copied().collect::<Vec<_>>(),
                &[1, -9_007_199_254_740_993]
            )
        }
        other => panic!("{other:?}"),
    }
}

#[test]
fn runtime_repeat_threads_streams() {
    let text = "\
IN x
STATE x
REPEAT 3
x = IADD(x, 1)
END
OUT = IADD(x, 0)
";
    let mut feeds = BTreeMap::new();
    feeds.insert(
        "x".to_string(),
        Value::I64(ArrayD::from_elem(ndarray::IxDyn(&[1]), 0i64)),
    );
    let got = jotnar::run_text(text, &opts(), &feeds).unwrap();
    match got.get("OUT").unwrap() {
        Value::I64(a) => assert_eq!(*a.iter().next().unwrap(), 3),
        other => panic!("unexpected {other:?}"),
    }
}

#[test]
fn extended_integer_family_runs() {
    let text = "\
IN a
IN b
IN s
IN c
SA = SADD(a, b)
MX = IMAX(a, b)
SR = ISAR(a, s)
QU = IDIVU(a, b)
RM = IMODU(a, b)
EQ = IEQ(a, b)
LT = ILT(a, b)
SE = ISELECT(c, a, b)
";
    let a =
        ArrayD::from_shape_vec(ndarray::IxDyn(&[4]), vec![1i64, -1, i64::MAX, i64::MIN]).unwrap();
    let b = ArrayD::from_shape_vec(ndarray::IxDyn(&[4]), vec![2i64, -2, 1, -1]).unwrap();
    let s = ArrayD::from_shape_vec(ndarray::IxDyn(&[4]), vec![0i64, 1, 2, 3]).unwrap();
    let c = ArrayD::from_shape_vec(ndarray::IxDyn(&[4]), vec![0i64, 1, -1, 0]).unwrap();
    let mut feeds = BTreeMap::new();
    for (name, arr) in [("a", a), ("b", b), ("s", s), ("c", c)] {
        feeds.insert(name.to_string(), Value::I64(arr));
    }
    let got = jotnar::run_text(text, &opts(), &feeds).unwrap();
    let i = |name: &str| -> Vec<i64> {
        match got.get(name).unwrap() {
            Value::I64(a) => a.iter().copied().collect(),
            other => panic!("{name}: {other:?}"),
        }
    };
    assert_eq!(i("SA"), vec![3, -3, i64::MAX, i64::MIN]);
    assert_eq!(i("MX"), vec![2, -1, i64::MAX, -1]);
    assert_eq!(i("SR"), vec![1, -1, i64::MAX >> 2, i64::MIN >> 3]);
    assert_eq!(i("QU"), vec![0, 1, i64::MAX, 0]);
    assert_eq!(i("RM"), vec![1, 1, 0, i64::MIN]);
    assert_eq!(i("EQ"), vec![0, 0, 0, 0]);
    assert_eq!(i("LT"), vec![1, 0, 0, 1]);
    assert_eq!(i("SE"), vec![2, -1, i64::MAX, -1]);
}
