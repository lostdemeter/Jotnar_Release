//! POOLMAX2D / POOLAVG2D reference gates.

use ndarray::{ArrayD, IxDyn};
use std::path::PathBuf;
use jotnar::asm::{self, AsmOptions};
use jotnar::ir::Config;
use jotnar::ops::image::{op_poolavg2d, op_poolmax2d};
use jotnar::value::{Arg, Value};

fn value(shape: &[usize], vals: Vec<f64>) -> Value {
    Value::F64(ArrayD::from_shape_vec(IxDyn(shape), vals).unwrap())
}

fn args(x: Value, k: (f64, f64), s: (f64, f64), p: (f64, f64)) -> Vec<Arg> {
    vec![
        Arg::Stream(x),
        Arg::Literal(k.0),
        Arg::Literal(k.1),
        Arg::Literal(s.0),
        Arg::Literal(s.1),
        Arg::Literal(p.0),
        Arg::Literal(p.1),
    ]
}

fn as_vec(v: &Value) -> Vec<f64> {
    match v {
        Value::F64(a) => a.iter().copied().collect(),
        _ => panic!("expected F64"),
    }
}

#[test]
fn poolmax2d_2x2_stride2() {
    let x = value(&[4, 4, 1], (0..16).map(|v| v as f64).collect());
    let out = op_poolmax2d(&args(x, (2.0, 2.0), (2.0, 2.0), (0.0, 0.0)),
                           &Config::default())
        .unwrap();
    assert_eq!(as_vec(&out[0]), vec![5.0, 7.0, 13.0, 15.0]);
}

#[test]
fn poolavg2d_2x2_stride2() {
    let x = value(&[4, 4, 1], (0..16).map(|v| v as f64).collect());
    let out = op_poolavg2d(&args(x, (2.0, 2.0), (2.0, 2.0), (0.0, 0.0)),
                           &Config::default())
        .unwrap();
    assert_eq!(as_vec(&out[0]), vec![2.5, 4.5, 10.5, 12.5]);
}

#[test]
fn poolaverage2d_strip_rows_and_columns() {
    let x = value(&[4, 4, 1], (0..16).map(|v| v as f64).collect());
    // 1x4 over rows -> mean of each row
    let rows = op_poolavg2d(&args(x.clone(), (1.0, 4.0), (1.0, 1.0), (0.0, 0.0)),
                            &Config::default())
        .unwrap();
    assert_eq!(as_vec(&rows[0]), vec![1.5, 5.5, 9.5, 13.5]);
    // 4x1 over columns -> mean of each column
    let cols = op_poolavg2d(&args(x, (4.0, 1.0), (1.0, 1.0), (0.0, 0.0)),
                            &Config::default())
        .unwrap();
    assert_eq!(as_vec(&cols[0]), vec![6.0, 7.0, 8.0, 9.0]);
}

#[test]
fn pooolmax2d_keeps_channels_independent() {
    // two channels, 2x2 each: channel 0 = 0..4, channel 1 = 100..104
    let x = value(&[2, 2, 2], vec![0., 100., 1., 101., 2., 102., 3., 103.]);
    let out = op_poolmax2d(&args(x, (2.0, 2.0), (2.0, 2.0), (0.0, 0.0)),
                           &Config::default())
        .unwrap();
    assert_eq!(as_vec(&out[0]), vec![3.0, 103.0]);
}

#[test]
fn pool2d_rejects_non_hwc() {
    let x = value(&[4, 4], (0..16).map(|v| v as f64).collect());
    let err = op_poolmax2d(&args(x, (2.0, 2.0), (2.0, 2.0), (0.0, 0.0)),
                           &Config::default())
        .unwrap_err();
    assert!(err.to_string().contains("expected HWC"), "{err}");
}

#[test]
fn pool2d_listing_assembles() {
    let opts = AsmOptions::new(PathBuf::from("."));
    for text in [
        "IN X AS F:HWC\nOUT = POOLMAX2D(X, 2, 2, 2, 2, 0, 0)\n",
        "IN X AS F:HWC\nOUT = POOLAVG2D(X, 1, 4, 1, 1, 0, 0)\n",
    ] {
        let bound = asm::assemble(text, &opts).unwrap();
        assert_eq!(bound.instrs.len(), 1);
        assert!(bound.instrs[0].mn.starts_with("POOL"));
    }
}
