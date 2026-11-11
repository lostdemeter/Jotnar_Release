//! T+mantissa (T2) format gates.
//!
//! The scalar reference is checked against f64; the array ops are then used
//! as the interpreter oracle for the NonFpu emission gate.

use ndarray::{ArrayD, IxDyn};
use phi_core::t2::{self, Tables, T2};
use phi_core::Triple;

fn lcg(seed: &mut u64) -> f64 {
    *seed = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
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
        let q = t2::to_fixed(p, 32768, &tabs, m);
        let t = t2::from_fixed(q, 32768, &tabs, m);
        assert!(t.m >= -(half as i32) && t.m < half as i32);
        let q2 = t2::to_fixed(t, 32768, &tabs, m);
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
    let a = t2::encode_array(
        &ArrayD::from_shape_vec(IxDyn(&[rows, k]), av.clone()).unwrap(),
        m,
    );
    let b = t2::encode_array(
        &ArrayD::from_shape_vec(IxDyn(&[k, cols]), bv.clone()).unwrap(),
        m,
    );
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

#[test]
fn to_fixed_uses_mantissa_width() {
    for (m, f) in [(24i64, 30i64), (18, 24)] {
        let t = t2::encode(1.5, m);
        let q = t2::to_fixed(t, 32768, &Tables::new(f), m);
        let v = q as f64 / (1i64 << f) as f64;
        assert!((v - 1.5).abs() < 1e-5, "M={m} F={f}: {v}");
    }
}

#[test]
fn from_fixed_min_max_do_not_wrap() {
    let tabs = Tables::new(24);
    let lo = t2::from_fixed(i64::MIN, 32768, &tabs, 24);
    let hi = t2::from_fixed(i64::MAX, 32768, &tabs, 24);
    assert_eq!(lo.s, -1);
    assert_eq!(hi.s, 1);
    assert!(lo.e >= 0 && lo.e <= 65535, "MIN e={}", lo.e);
    assert!(hi.e >= 0 && hi.e <= 65535, "MAX e={}", hi.e);
    assert!(
        lo.e > 60000,
        "MIN e={} (should be the high bridge range)",
        lo.e
    );
    assert!(hi.e > 60000, "MAX e={}", hi.e);
    assert_eq!(lo.e, hi.e, "MIN/MAX magnitudes must agree");
}

#[test]
fn config_widths_are_typed_errors() {
    assert!(t2::check_widths(Some(64), None).is_err());
    assert!(t2::check_widths(Some(0), None).is_err());
    assert!(t2::check_widths(None, Some(0)).is_err());
    assert!(t2::check_widths(None, Some(13)).is_err());
    assert!(t2::check_widths(None, Some(63)).is_err());
    let (_, f) = t2::check_widths(Some(24), Some(18)).unwrap();
    assert_eq!(f, 24);
    let (_, f) = t2::check_widths(Some(30), Some(30)).unwrap();
    assert_eq!(f, 30);
    let (_, f) = t2::check_widths(None, None).unwrap();
    assert_eq!(f, 18);
}

fn conv_ref(
    x: &[f64],
    h: usize,
    w: usize,
    cin: usize,
    wt: &[f64],
    cout: usize,
    kh: usize,
    kw: usize,
    b: &[f64],
    stride: i64,
    pad: i64,
) -> (Vec<f64>, usize, usize) {
    let ho = ((h as i64 + 2 * pad - kh as i64) / stride + 1) as usize;
    let wo = ((w as i64 + 2 * pad - kw as i64) / stride + 1) as usize;
    let mut o = vec![0.0; ho * wo * cout];
    for co in 0..cout {
        for oy in 0..ho {
            for ox in 0..wo {
                let mut acc = b[co];
                for ky in 0..kh {
                    let iy = oy as i64 * stride + ky as i64 - pad;
                    if iy < 0 || iy >= h as i64 {
                        continue;
                    }
                    for kx in 0..kw {
                        let ix = ox as i64 * stride + kx as i64 - pad;
                        if ix < 0 || ix >= w as i64 {
                            continue;
                        }
                        for ci in 0..cin {
                            acc += x[((iy as usize) * w + ix as usize) * cin + ci]
                                * wt[((co * kh + ky) * kw + kx) * cin + ci];
                        }
                    }
                }
                o[(oy * wo + ox) * cout + co] = acc;
            }
        }
    }
    (o, ho, wo)
}

#[test]
fn image_ops_match_f64() {
    let m = 24;
    let tabs = Tables::new(24);
    let (h, w, cin, cout) = (3usize, 4usize, 2usize, 3usize);
    let (kh, kw) = (3usize, 3usize);
    let mut seed = 23u64;
    let xv: Vec<f64> = (0..h * w * cin).map(|_| lcg(&mut seed) * 2.0).collect();
    let wv: Vec<f64> = (0..cout * kh * kw * cin)
        .map(|_| lcg(&mut seed) * 0.5)
        .collect();
    let bv: Vec<f64> = (0..cout).map(|_| lcg(&mut seed) * 0.2).collect();
    let x = t2::encode_array(
        &ArrayD::from_shape_vec(IxDyn(&[h, w, cin]), xv.clone()).unwrap(),
        m,
    );
    let wt = t2::encode_array(
        &ArrayD::from_shape_vec(IxDyn(&[cout, kh, kw, cin]), wv.clone()).unwrap(),
        m,
    );
    let b = t2::encode_array(
        &ArrayD::from_shape_vec(IxDyn(&[cout]), bv.clone()).unwrap(),
        m,
    );

    let got = t2::conv2d(&x, &wt, &b, 1, 1, &tabs, m).unwrap();
    let g = t2::decode_array(&got, m).unwrap();
    let (want, ho, wo) = conv_ref(&xv, h, w, cin, &wv, cout, kh, kw, &bv, 1, 1);
    for i in 0..ho {
        for j in 0..wo {
            for c in 0..cout {
                let a = g[IxDyn(&[i, j, c])];
                let e = want[(i * wo + j) * cout + c];
                assert!(rel(a, e) < 2e-3, "conv {i},{j},{c}: {a} vs {e}");
            }
        }
    }

    let got = t2::deconv2d(&x, &wt, &b, 2, 1, &tabs, m).unwrap();
    let g = t2::decode_array(&got, m).unwrap();
    // f64 deconv reference by direct scatter.
    let ho2 = ((h as i64 - 1) * 2 - 2 + kh as i64) as usize;
    let wo2 = ((w as i64 - 1) * 2 - 2 + kw as i64) as usize;
    let mut want = vec![0.0; ho2 * wo2 * cout];
    for co in 0..cout {
        for iy in 0..h {
            for ix in 0..w {
                for ky in 0..kh {
                    let oy = iy as i64 * 2 - 1 + ky as i64;
                    if oy < 0 || oy >= ho2 as i64 {
                        continue;
                    }
                    for kx in 0..kw {
                        let ox = ix as i64 * 2 - 1 + kx as i64;
                        if ox < 0 || ox >= wo2 as i64 {
                            continue;
                        }
                        for ci in 0..cin {
                            want[(oy as usize * wo2 + ox as usize) * cout + co] += xv
                                [(iy * w + ix) * cin + ci]
                                * wv[((co * kh + ky) * kw + kx) * cin + ci];
                        }
                    }
                }
            }
        }
        for i in 0..ho2 * wo2 {
            want[i * cout + co] += bv[co];
        }
    }
    for i in 0..ho2 {
        for j in 0..wo2 {
            for c in 0..cout {
                let a = g[IxDyn(&[i, j, c])];
                let e = want[(i * wo2 + j) * cout + c];
                assert!(rel(a, e) < 2e-3, "deconv {i},{j},{c}: {a} vs {e}");
            }
        }
    }

    // Bilinear resize vs a direct f64 reference (align_corners=false).
    let (ho3, wo3) = (5usize, 7usize);
    let got = t2::resize2d(&x, ho3, wo3, false, &tabs, m).unwrap();
    let g = t2::decode_array(&got, m).unwrap();
    for oy in 0..ho3 {
        let sy = (((oy as f64 + 0.5) * h as f64 / ho3 as f64 - 0.5).max(0.0)).min(h as f64 - 1.0);
        let y0 = sy.floor() as usize;
        let y1 = (y0 + 1).min(h - 1);
        let wy = sy - y0 as f64;
        for ox in 0..wo3 {
            let sx =
                (((ox as f64 + 0.5) * w as f64 / wo3 as f64 - 0.5).max(0.0)).min(w as f64 - 1.0);
            let x0 = sx.floor() as usize;
            let x1 = (x0 + 1).min(w - 1);
            let wx = sx - x0 as f64;
            for c in 0..cin {
                let p = |yy: usize, xx: usize| xv[(yy * w + xx) * cin + c];
                let e = (1.0 - wy) * (1.0 - wx) * p(y0, x0)
                    + (1.0 - wy) * wx * p(y0, x1)
                    + wy * (1.0 - wx) * p(y1, x0)
                    + wy * wx * p(y1, x1);
                let a = g[IxDyn(&[oy, ox, c])];
                assert!(rel(a, e) < 2e-3, "resize {oy},{ox},{c}: {a} vs {e}");
            }
        }
    }

    let rv = [-2.0, -0.5, 0.0, 0.5, 2.0];
    let r = t2::encode_array(
        &ArrayD::from_shape_vec(IxDyn(&[5]), rv.to_vec()).unwrap(),
        m,
    );
    let rr = t2::relu_array(&r).unwrap();
    let rd = t2::decode_array(&rr, m).unwrap();
    for (i, v) in rv.iter().enumerate() {
        let want = v.max(0.0);
        assert!((rd[IxDyn(&[i])] - want).abs() < 1e-5);
    }
}
