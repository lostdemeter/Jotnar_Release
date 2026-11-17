//! MULC / ADDC broadcast gates.

use ndarray::{ArrayD, IxDyn};
use std::path::PathBuf;

use jotnar::asm::{self, AsmOptions};
use jotnar::ir::Config;
use jotnar::lattice;
use jotnar::ops::element::{op_addc, op_mulc};
use jotnar::value::{Arg, Value};

fn triple(shape: &[usize], vals: Vec<f64>) -> Value {
    let a = ArrayD::from_shape_vec(IxDyn(shape), vals).unwrap();
    Value::Triple(lattice::encode(&a))
}

fn args(a: Value, b: Value) -> Vec<Arg> {
    vec![Arg::Stream(a), Arg::Stream(b)]
}

fn decoded(v: &Value) -> Vec<f64> {
    match v {
        Value::Triple(t) => {
            let a = lattice::decode(&t.s, &t.e);
            a.iter().copied().collect()
        }
        _ => panic!("expected triple"),
    }
}

#[test]
fn mulc_per_channel() {
    let x = triple(&[2, 2, 2], vec![1., 2., 3., 4., 5., 6., 7., 8.]);
    let s = triple(&[2], vec![2., 0.5]);
    let out = op_mulc(&args(x, s), &Config::default()).unwrap();
    let got = decoded(&out[0]);
    for (g, e) in got.iter().zip([2., 1., 6., 2., 10., 3., 14., 4.]) {
        assert!((g - e).abs() < 0.02, "{got:?}");
    }
}

#[test]
fn addc_per_channel() {
    let x = triple(&[2, 2, 2], vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]);
    let s = triple(&[2], vec![0.1, 0.2]);
    let out = op_addc(&args(x, s), &Config::default()).unwrap();
    let got = decoded(&out[0]);
    for (g, e) in got.iter().zip([0.2, 0.4, 0.4, 0.6, 0.6, 0.8, 0.8, 1.0]) {
        assert!((g - e).abs() < 0.02, "{got:?}");
    }
}

#[test]
fn addc_spatial_broadcast() {
    let x = triple(&[2, 2, 2], vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]);
    // [H,1,C]: add 0.1 to row 0, 0.2 to row 1
    let row = triple(&[2, 1, 2], vec![0.1, 0.1, 0.2, 0.2]);
    let out = op_addc(&args(x.clone(), row), &Config::default()).unwrap();
    let got = decoded(&out[0]);
    for (g, e) in got.iter().zip([0.2, 0.3, 0.4, 0.5, 0.7, 0.8, 0.9, 1.0]) {
        assert!((g - e).abs() < 0.02, "{got:?}");
    }
    // [1,W,C]: add 0.1 to col 0, 0.2 to col 1
    let col = triple(&[1, 2, 2], vec![0.1, 0.1, 0.2, 0.2]);
    let out = op_addc(&args(x, col), &Config::default()).unwrap();
    let got = decoded(&out[0]);
    for (g, e) in got.iter().zip([0.2, 0.3, 0.5, 0.6, 0.6, 0.7, 0.9, 1.0]) {
        assert!((g - e).abs() < 0.02, "{got:?}");
    }
}

#[test]
fn broadcast_listings_assemble() {
    let opts = AsmOptions::new(PathBuf::from("."));
    for text in [
        "IN X AS F:HWC\nIN S AS F:HW\nOUT = MULC(X, S)\n",
        "IN X AS F:HWC\nIN S AS F:HW\nOUT = ADDC(X, S)\n",
    ] {
        let bound = asm::assemble(text, &opts).unwrap();
        assert_eq!(bound.instrs.len(), 1);
    }
}
