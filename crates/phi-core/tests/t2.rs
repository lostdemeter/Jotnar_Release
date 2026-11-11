//! T+mantissa (T2) format gates.
//!
//! The scalar reference is checked against f64; the array ops are then used
//! as the interpreter oracle for the NonFpu emission gate.

use ndarray::{ArrayD, IxDyn};
use phi_core::t2::{self, Tables, T2};
use phi_core::Triple;

fn lcg(seed: &mut u64) -> f64 {
    *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    ((*seed >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
}

fn rel(a: f64, b: f64) -> f64 {
    (a - b).abs() / b.abs().max(1e-30)
}

#[test]
fn encode_decode_roundtrip() {
    let m = 24;
    let mut seed = 7u64;
    for _ in 0..2000 {
        let v = lcg(&mut seed) * 1e3;
        if v == 0.0 {
            continue;
        }
        let t = t2::encode(v, m);
        let d = t2::decode(t, m);
        assert!(rel(d, v) < 2.0f64.powi(-(m as i32)) * 4.0, "v={v} d={d}");
    }
}

#[test]
fn mul_add_match_f64() {
    let m = 24;
    let tabs = Tables::new(24);
    let mut seed = 11u64;
    let bound = 2.0f64.powi(-(m as i32 - 3));
    for _ in 0..2000 {
        let av = lcg(&mut seed) * 20.0;
        let bv = lcg(&mut seed) * 20.0;
        let a = t2::encode(av, m);
        let b = t2::encode(bv, m);
        let da = t2::decode(a, m);
        let db = t2::decode(b, m);
        let r = t2::mul(a, b, m);
        assert!(rel(t2::decode(r, m), da * db) < bound, "mul {av} {bv}");
        let scale = da.abs().max(db.abs()).max(1.0);
        let abs_bound = bound * scale;
        let s = t2::add(a, b, false, &tabs, m);
        assert!(
            (t2::decode(s, m) - (da + db)).abs() < abs_bound,
            "add {av} {bv}"
        );
        let d = t2::add(a, b, true, &tabs, m);
        assert!(
            (t2::decode(d, m) - (da - db)).abs() < abs_bound,
            "sub {av} {bv}"
        );
    }
}

#[test]
fn fixed_roundtrip_is_canonical() {
    let m = 24;
    let tabs = Tables::new(24);
    let mut seed = 3u64;
    let half = 1i64 << (m - 1);
    for _ in 0..2000 {
        let v = lcg(&mut seed) * 1e4;
        let p = t2::encode(v, m);
        let q = t2::to_fixed(p, 32768, &tabs);
        let t = t2::from_fixed(q, 32768, &tabs, m);
        assert!(t.m >= -(half as i32) && t.m < half as i32);
        let q2 = t2::to_fixed(t, 32768, &tabs);
        assert!(rel(q2 as f64, q as f64) < 2.0f64.powi(-(m as i32)) * 4.0);
    }
}

#[test]
fn matmul_and_layernorm_close_f64() {
    let m = 24;
    let tabs = Tables::new(24);
    let mut seed = 19u64;
    let (rows, k, cols) = (3usize, 8usize, 4usize);
    let av: Vec<f64> = (0..rows * k).map(|_| lcg(&mut seed) * 2.0).collect();
    let bv: Vec<f64> = (0..k * cols).map(|_| lcg(&mut seed) * 2.0).collect();
    let a = t2::encode_array(&ArrayD::from_shape_vec(IxDyn(&[rows, k]), av.clone()).unwrap(), m);
    let b = t2::encode_array(&ArrayD::from_shape_vec(IxDyn(&[k, cols]), bv.clone()).unwrap(), m);
    let got = t2::matmul2(&a, &b, &tabs, m).unwrap();
    let g = t2::decode_array(&got, m).unwrap();
    for i in 0..rows {
        for j in 0..cols {
            let want: f64 = (0..k).map(|t| av[i * k + t] * bv[t * cols + j]).sum();
            assert!(rel(g[IxDyn(&[i, j])], want) < 1e-5);
        }
    }
}

#[test]
fn triple_mantissa_is_optional() {
    let plain = Triple::from_vecs(&[2], vec![1, -1], vec![32768, 32768], vec![0, 0]).unwrap();
    assert!(!plain.is_t2());
    let t2v = Triple::from_vecs_opt(
        &[2],
        vec![1, -1],
        vec![32768, 32768],
        vec![0, 0],
        Some(vec![0, 0]),
    )
    .unwrap();
    assert!(t2v.is_t2());
}
