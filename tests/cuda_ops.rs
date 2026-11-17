//! CUDA target op parity: GELU, DECONV2D, RESHAPE3.
//!
//! These are the ops that were missing for the DAV2-S graph. Each test builds a
//! tiny CUDA program with the real emitter and compares against the documented
//! reference maths. They are skipped when no CUDA device/toolkit is present.
use std::fs;
use std::path::Path;
use std::process::Command;

use jotnar::asm::AsmOptions;
use jotnar::codegen::{build, compile_program, CompileOptions, Target};
use jotnar::payload::Feeds;
use jotnar::value::Value;
use ndarray::{ArrayD, IxDyn};

fn cuda_available() -> bool {
    if !Path::new("/dev/nvidia0").exists() {
        return false;
    }
    Command::new("nvcc")
        .arg("--version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

fn opts(outputs: &[&str]) -> CompileOptions {
    CompileOptions {
        target: Target::Cuda,
        outputs: outputs.iter().map(|s| s.to_string()).collect(),
        asm: AsmOptions::new("."),
        use_fp16: false,
    }
}

/// Compile and run `listing`. `inputs` must be in listing `IN` order.
/// Returns one f32 vector per output.
fn run_cuda(
    listing: &str,
    outputs: &[&str],
    inputs: &[(&str, Vec<usize>, Vec<f32>)],
    dir: &Path,
    name: &str,
) -> Vec<Vec<f32>> {
    let mut feeds = Feeds::new();
    for (n, shape, data) in inputs {
        feeds.insert(
            (*n).into(),
            Value::F32(ArrayD::from_shape_vec(IxDyn(shape), data.clone()).unwrap()),
        );
    }
    let art = compile_program(listing, &opts(outputs), feeds).expect("compile cuda");
    let exe = build(&art.source, dir, name, Target::Cuda).expect("build cuda");

    let mut argv = Vec::new();
    for (n, _, data) in inputs {
        let p = dir.join(format!("{n}.in.f32"));
        let mut bytes = Vec::with_capacity(data.len() * 4);
        for x in data {
            bytes.extend_from_slice(&x.to_le_bytes());
        }
        fs::write(&p, bytes).unwrap();
        argv.push(p.to_string_lossy().into_owned());
    }
    for o in &art.outputs {
        argv.push(dir.join(format!("{o}.out.f32")).to_string_lossy().into_owned());
    }

    let out = Command::new(&exe)
        .args(&argv)
        .output()
        .expect("run cuda artifact");
    assert!(
        out.status.success(),
        "cuda artifact failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );

    art.outputs
        .iter()
        .map(|o| {
            let bytes = fs::read(dir.join(format!("{o}.out.f32"))).unwrap();
            bytes
                .chunks(4)
                .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                .collect()
        })
        .collect()
}

fn erf_approx(x: f64) -> f64 {
    let s = x.signum();
    let x = x.abs();
    let t = 1.0 / (1.0 + 0.3275911 * x);
    let y = 1.0
        - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736)
            * t
            + 0.254829592)
            * t
            * (-x * x).exp();
    s * y
}

fn gelu_ref(x: f64) -> f64 {
    0.5 * x * (1.0 + erf_approx(x / std::f64::consts::SQRT_2))
}

#[test]
fn cuda_gelu_matches_rust_approximation() {
    if !cuda_available() {
        eprintln!("skip: no CUDA device/toolkit");
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let xs: Vec<f32> = vec![-6.0, -3.0, -1.0, -0.25, 0.0, 0.5, 2.0, 5.0];
    let listing = "IN A AS F:HW\nOUT = GELU(A)\n";
    let outs = run_cuda(
        listing,
        &["OUT"],
        &[("A", vec![2, 4], xs.clone())],
        dir.path(),
        "gelu",
    );
    for (got, x) in outs[0].iter().zip(xs.iter()) {
        let want = gelu_ref(*x as f64) as f32;
        assert!(
            (got - want).abs() <= 3e-6,
            "GELU({x}) = {got}, want {want}"
        );
    }
}

fn deconv_ref(
    x: &[f32],
    wt: &[f32],
    b: &[f32],
    h: usize,
    w: usize,
    cin: usize,
    cout: usize,
    kh: usize,
    kw: usize,
    ho: usize,
    wo: usize,
    stride: i64,
    pad: i64,
) -> Vec<f32> {
    let mut o = vec![0f32; ho * wo * cout];
    for ch in 0..cout {
        for iy in 0..h {
            for ix in 0..w {
                for ky in 0..kh {
                    let oy = iy as i64 * stride - pad + ky as i64;
                    if oy < 0 || oy >= ho as i64 {
                        continue;
                    }
                    for kx in 0..kw {
                        let ox = ix as i64 * stride - pad + kx as i64;
                        if ox < 0 || ox >= wo as i64 {
                            continue;
                        }
                        for ci in 0..cin {
                            o[(oy as usize * wo + ox as usize) * cout + ch] += x
                                [(iy * w + ix) * cin + ci]
                                * wt[((ch * kh + ky) * kw + kx) * cin + ci];
                        }
                    }
                }
            }
        }
        for i in 0..ho * wo {
            o[i * cout + ch] += b[ch];
        }
    }
    o
}

#[test]
fn cuda_deconv2d_matches_reference() {
    if !cuda_available() {
        eprintln!("skip: no CUDA device/toolkit");
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    // x: 2x2, 2 channels; weight: Cout=2, 2x2, Cin=2; stride 2, pad 0.
    let (h, w, cin) = (2usize, 2usize, 2usize);
    let (cout, kh, kw) = (2usize, 2usize, 2usize);
    let stride = 2i64;
    let pad = 0i64;
    let ho = (h as i64 - 1) * stride - 2 * pad + kh as i64;
    let wo = (w as i64 - 1) * stride - 2 * pad + kw as i64;

    let x: Vec<f32> = (0..h * w * cin).map(|i| (i as f32) * 0.25 - 0.5).collect();
    let wt: Vec<f32> = (0..cout * kh * kw * cin)
        .map(|i| ((i * 7 % 11) as f32) * 0.1 - 0.5)
        .collect();
    let b: Vec<f32> = vec![0.1, -0.2];

    let listing = "\
IN X AS F:HWC
IN W AS F:CONV
IN B AS F:SEQ
OUT = DECONV2D(X, W, B, 2, 0)
";
    let outs = run_cuda(
        listing,
        &["OUT"],
        &[
            ("X", vec![h, w, cin], x.clone()),
            ("W", vec![cout, kh, kw, cin], wt.clone()),
            ("B", vec![cout], b.clone()),
        ],
        dir.path(),
        "deconv",
    );
    let want = deconv_ref(
        &x, &wt, &b, h, w, cin, cout, kh, kw, ho as usize, wo as usize, stride, pad,
    );
    assert_eq!(outs[0].len(), want.len());
    for (i, (got, want)) in outs[0].iter().zip(want.iter()).enumerate() {
        assert!(
            (got - want).abs() <= 1e-5,
            "DECONV2D[{i}] = {got}, want {want}"
        );
    }
}

fn conv_ref(
    x: &[f32],
    wt: &[f32],
    b: &[f32],
    h: usize,
    w: usize,
    cin: usize,
    cout: usize,
    kh: usize,
    kw: usize,
    ho: usize,
    wo: usize,
    stride: i64,
    pad: i64,
) -> Vec<f32> {
    let mut o = vec![0f32; ho * wo * cout];
    for oy in 0..ho {
        for ox in 0..wo {
            for co in 0..cout {
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
                            acc += x[(iy as usize * w + ix as usize) * cin + ci]
                                * wt[((co * kh + ky) * kw + kx) * cin + ci];
                        }
                    }
                }
                o[(oy * wo + ox) * cout + co] = acc;
            }
        }
    }
    o
}

#[test]
fn cuda_conv2d_nchw_matches_reference() {
    if !cuda_available() {
        eprintln!("skip: no CUDA device/toolkit");
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let (h, w, cin) = (4usize, 5usize, 3usize);
    let (cout, kh, kw) = (4usize, 3usize, 3usize);
    let stride = 1i64;
    let pad = 1i64;
    let ho = ((h as i64 + 2 * pad - kh as i64) / stride + 1) as usize;
    let wo = ((w as i64 + 2 * pad - kw as i64) / stride + 1) as usize;

    let x: Vec<f32> = (0..h * w * cin).map(|i| (i as f32) * 0.07 - 0.8).collect();
    let wt: Vec<f32> = (0..cout * kh * kw * cin)
        .map(|i| ((i * 11 % 17) as f32) * 0.05 - 0.4)
        .collect();
    let b: Vec<f32> = vec![0.1, -0.2, 0.3, -0.4];

    let listing = "\
CONFIG cudnn_nchw 1
IN X AS F:HWC
IN W AS F:CONV
IN B AS F:SEQ
OUT = CONV2D(X, W, B, 1, 1)
";
    let outs = run_cuda(
        listing,
        &["OUT"],
        &[
            ("X", vec![h, w, cin], x.clone()),
            ("W", vec![cout, kh, kw, cin], wt.clone()),
            ("B", vec![cout], b.clone()),
        ],
        dir.path(),
        "conv_nchw",
    );
    let want = conv_ref(
        &x, &wt, &b, h, w, cin, cout, kh, kw, ho, wo, stride, pad,
    );
    assert_eq!(outs[0].len(), want.len());
    for (i, (got, want)) in outs[0].iter().zip(want.iter()).enumerate() {
        assert!(
            (got - want).abs() <= 1e-5,
            "CONV2D NCHW[{i}] = {got}, want {want}"
        );
    }
}

#[test]
fn cuda_matmul_t_matches_reference() {
    if !cuda_available() {
        eprintln!("skip: no CUDA device/toolkit");
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let (m, k, n) = (2usize, 3usize, 4usize);
    let a: Vec<f32> = (0..m * k).map(|i| (i as f32) * 0.2 - 0.5).collect();
    let b: Vec<f32> = (0..n * k).map(|i| (i as f32) * 0.1 - 0.3).collect();
    let listing = "IN A AS F:HW\nIN B AS F:HW\nOUT = MATMUL_T(A, B)\n";
    let outs = run_cuda(
        listing,
        &["OUT"],
        &[("A", vec![m, k], a.clone()), ("B", vec![n, k], b.clone())],
        dir.path(),
        "matmul_t",
    );
    for i in 0..m {
        for j in 0..n {
            let mut want = 0f32;
            for t in 0..k {
                want += a[i * k + t] * b[j * k + t];
            }
            let got = outs[0][i * n + j];
            assert!((got - want).abs() <= 1e-5, "MATMUL_T[{i},{j}] = {got}, want {want}");
        }
    }
}

#[test]
fn cuda_matmul_bias_matches_reference() {
    if !cuda_available() {
        eprintln!("skip: no CUDA device/toolkit");
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let (m, k, n) = (2usize, 3usize, 4usize);
    let a: Vec<f32> = (0..m * k).map(|i| (i as f32) * 0.2 - 0.5).collect();
    let b: Vec<f32> = (0..k * n).map(|i| (i as f32) * 0.1 - 0.3).collect();
    let bias: Vec<f32> = vec![0.1, -0.2, 0.3, -0.4];
    let listing = "IN A AS F:HW\nIN B AS F:HW\nIN BIAS AS F:SEQ\nOUT = MATMUL_BIAS(A, B, BIAS)\n";
    let outs = run_cuda(
        listing,
        &["OUT"],
        &[
            ("A", vec![m, k], a.clone()),
            ("B", vec![k, n], b.clone()),
            ("BIAS", vec![n], bias.clone()),
        ],
        dir.path(),
        "matmul_bias",
    );
    for i in 0..m {
        for j in 0..n {
            let mut want = bias[j];
            for t in 0..k {
                want += a[i * k + t] * b[t * n + j];
            }
            let got = outs[0][i * n + j];
            assert!((got - want).abs() <= 1e-5, "MATMUL_BIAS[{i},{j}] = {got}, want {want}");
        }
    }
}

#[test]
fn cuda_matmul_bias_cublaslt_matches_reference() {
    if !cuda_available() {
        eprintln!("skip: no CUDA device/toolkit");
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let (m, k, n) = (2usize, 3usize, 4usize);
    let a: Vec<f32> = (0..m * k).map(|i| (i as f32) * 0.2 - 0.5).collect();
    let b: Vec<f32> = (0..k * n).map(|i| (i as f32) * 0.1 - 0.3).collect();
    let bias: Vec<f32> = vec![0.1, -0.2, 0.3, -0.4];
    let listing = "CONFIG cublaslt 1\nIN A AS F:HW\nIN B AS F:HW\nIN BIAS AS F:SEQ\nOUT = MATMUL_BIAS(A, B, BIAS)\n";
    let outs = run_cuda(
        listing, &["OUT"],
        &[("A", vec![m,k], a.clone()), ("B", vec![k,n], b.clone()), ("BIAS", vec![n], bias.clone())],
        dir.path(), "matmul_bias_lt");
    for i in 0..m {
        for j in 0..n {
            let mut want = bias[j];
            for t in 0..k { want += a[i * k + t] * b[t * n + j]; }
            let got = outs[0][i * n + j];
            assert!((got - want).abs() <= 1e-5, "MATMUL_BIAS_LT[{i},{j}] = {got}, want {want}");
        }
    }
}

#[test]
fn cuda_matmul_bias_gelu_cublaslt_matches_reference() {
    if !cuda_available() {
        eprintln!("skip: no CUDA device/toolkit");
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let (m, k, n) = (2usize, 3usize, 4usize);
    let a: Vec<f32> = (0..m * k).map(|i| (i as f32) * 0.2 - 0.5).collect();
    let b: Vec<f32> = (0..k * n).map(|i| (i as f32) * 0.1 - 0.3).collect();
    let bias: Vec<f32> = vec![0.1, -0.2, 0.3, -0.4];
    let listing = "CONFIG cublaslt 1\nIN A AS F:HW\nIN B AS F:HW\nIN BIAS AS F:SEQ\nOUT = MATMUL_BIAS_GELU(A, B, BIAS)\n";
    let outs = run_cuda(
        listing, &["OUT"],
        &[("A", vec![m,k], a.clone()), ("B", vec![k,n], b.clone()), ("BIAS", vec![n], bias.clone())],
        dir.path(), "matmul_bias_gelu_lt");
    for i in 0..m {
        for j in 0..n {
            let mut want = bias[j];
            for t in 0..k { want += a[i * k + t] * b[t * n + j]; }
            let want = gelu_ref(want as f64) as f32;
            let got = outs[0][i * n + j];
            assert!((got - want).abs() <= 2e-4, "MATMUL_BIAS_GELU_LT[{i},{j}] = {got}, want {want}");
        }
    }
}

#[test]
fn cuda_matmul_bias_residual_matches_reference() {
    if !cuda_available() {
        eprintln!("skip: no CUDA device/toolkit");
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let (m, k, n) = (2usize, 3usize, 4usize);
    let a: Vec<f32> = (0..m * k).map(|i| (i as f32) * 0.2 - 0.5).collect();
    let b: Vec<f32> = (0..k * n).map(|i| (i as f32) * 0.1 - 0.3).collect();
    let bias: Vec<f32> = vec![0.1, -0.2, 0.3, -0.4];
    let r: Vec<f32> = (0..m * n).map(|i| (i as f32) * 0.05 - 0.1).collect();
    let listing = "IN A AS F:HW\nIN B AS F:HW\nIN BIAS AS F:SEQ\nIN R AS F:HW\nOUT = MATMUL_BIAS_RESIDUAL(A, B, BIAS, R)\n";
    let outs = run_cuda(
        listing, &["OUT"],
        &[("A", vec![m,k], a.clone()), ("B", vec![k,n], b.clone()), ("BIAS", vec![n], bias.clone()), ("R", vec![m,n], r.clone())],
        dir.path(), "matmul_bias_residual");
    for i in 0..m {
        for j in 0..n {
            let mut want = bias[j] + r[i * n + j];
            for t in 0..k { want += a[i * k + t] * b[t * n + j]; }
            let got = outs[0][i * n + j];
            assert!((got - want).abs() <= 1e-5, "MATMUL_BIAS_RESIDUAL[{i},{j}] = {got}, want {want}");
        }
    }
}

#[test]
fn cuda_matmul_bias_gelu_matches_reference() {
    if !cuda_available() {
        eprintln!("skip: no CUDA device/toolkit");
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let (m, k, n) = (2usize, 3usize, 4usize);
    let a: Vec<f32> = (0..m * k).map(|i| (i as f32) * 0.2 - 0.5).collect();
    let b: Vec<f32> = (0..k * n).map(|i| (i as f32) * 0.1 - 0.3).collect();
    let bias: Vec<f32> = vec![0.1, -0.2, 0.3, -0.4];
    let listing = "IN A AS F:HW\nIN B AS F:HW\nIN BIAS AS F:SEQ\nOUT = MATMUL_BIAS_GELU(A, B, BIAS)\n";
    let outs = run_cuda(
        listing, &["OUT"],
        &[("A", vec![m,k], a.clone()), ("B", vec![k,n], b.clone()), ("BIAS", vec![n], bias.clone())],
        dir.path(), "matmul_bias_gelu");
    for i in 0..m {
        for j in 0..n {
            let mut want = bias[j];
            for t in 0..k { want += a[i * k + t] * b[t * n + j]; }
            let want = gelu_ref(want as f64) as f32;
            let got = outs[0][i * n + j];
            assert!((got - want).abs() <= 1e-5, "MATMUL_BIAS_GELU[{i},{j}] = {got}, want {want}");
        }
    }
}

#[test]
fn cuda_conv2d_relu_matches_reference() {
    if !cuda_available() {
        eprintln!("skip: no CUDA device/toolkit");
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let (h, w, cin) = (4usize, 5usize, 3usize);
    let (cout, kh, kw) = (4usize, 3usize, 3usize);
    let stride = 1i64;
    let pad = 1i64;
    let ho = ((h as i64 + 2 * pad - kh as i64) / stride + 1) as usize;
    let wo = ((w as i64 + 2 * pad - kw as i64) / stride + 1) as usize;

    let x: Vec<f32> = (0..h * w * cin).map(|i| (i as f32) * 0.07 - 0.8).collect();
    let wt: Vec<f32> = (0..cout * kh * kw * cin)
        .map(|i| ((i * 11 % 17) as f32) * 0.05 - 0.4)
        .collect();
    let b: Vec<f32> = vec![0.1, -0.2, 0.3, -0.4];

    let listing = "\
CONFIG cudnn_nchw 1
IN X AS F:HWC
IN W AS F:CONV
IN B AS F:SEQ
OUT = CONV2D_RELU(X, W, B, 1, 1)
";
    let outs = run_cuda(
        listing,
        &["OUT"],
        &[
            ("X", vec![h, w, cin], x.clone()),
            ("W", vec![cout, kh, kw, cin], wt.clone()),
            ("B", vec![cout], b.clone()),
        ],
        dir.path(),
        "conv_relu",
    );
    let mut want = conv_ref(&x, &wt, &b, h, w, cin, cout, kh, kw, ho, wo, stride, pad);
    for v in &mut want {
        *v = v.max(0.0);
    }
    assert_eq!(outs[0].len(), want.len());
    for (i, (got, want)) in outs[0].iter().zip(want.iter()).enumerate() {
        assert!((got - want).abs() <= 1e-5, "CONV2D_RELU[{i}] = {got}, want {want}");
    }
}

#[test]
fn cuda_reshape3_is_an_exact_copy() {
    if !cuda_available() {
        eprintln!("skip: no CUDA device/toolkit");
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let xs: Vec<f32> = (0..8).map(|i| i as f32 - 3.5).collect();
    let listing = "IN A AS F:HWC\nOUT = RESHAPE3(A, 2, 2, 2)\n";
    let outs = run_cuda(
        listing,
        &["OUT"],
        &[("A", vec![4, 2, 1], xs.clone())],
        dir.path(),
        "reshape3",
    );
    assert_eq!(outs[0], xs, "RESHAPE3 must be bit-exact");
}

#[test]
fn cuda_self_attn_matches_reference() {
    if !cuda_available() {
        eprintln!("skip: no CUDA device/toolkit");
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let (n, nh, dh) = (70usize, 2usize, 64usize);
    let h = nh * dh;
    let make = |seed: usize| -> Vec<f32> {
        (0..n * h)
            .map(|i| (((i * seed * 37 + 11) % 101) as f64 / 50.0 - 1.0) as f32)
            .collect()
    };
    let q = make(3);
    let k = make(5);
    let v = make(7);
    let scale = 0.125f64;
    let listing = format!(
        "IN Q AS F:HW\nIN K AS F:HW\nIN V AS F:HW\nOUT = SELF_ATTN(Q, K, V, {nh}, {dh}, {scale})\n"
    );
    let outs = run_cuda(
        &listing,
        &["OUT"],
        &[
            ("Q", vec![n, h], q.clone()),
            ("K", vec![n, h], k.clone()),
            ("V", vec![n, h], v.clone()),
        ],
        dir.path(),
        "self_attn",
    );

    let mut want = vec![0f32; n * h];
    for head in 0..nh {
        let hb = head * dh;
        for i in 0..n {
            let mut scores = vec![0f64; n];
            let mut mx = f64::NEG_INFINITY;
            for j in 0..n {
                let mut acc = 0.0;
                for d in 0..dh {
                    acc += q[i * h + hb + d] as f64 * k[j * h + hb + d] as f64;
                }
                let s = acc * scale;
                scores[j] = s;
                mx = mx.max(s);
            }
            let mut se = 0.0;
            for s in scores.iter_mut() {
                *s = (*s - mx).exp();
                se += *s;
            }
            for d in 0..dh {
                let mut acc = 0.0;
                for j in 0..n {
                    acc += (scores[j] / se) * v[j * h + hb + d] as f64;
                }
                want[i * h + hb + d] = acc as f32;
            }
        }
    }
    assert_close(&outs[0], &want, 5e-5, "SELF_ATTN");
}

fn assert_close(got: &[f32], want: &[f32], tol: f32, what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: length");
    for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
        let err = (g - w).abs();
        let bound = tol * (1.0 + w.abs());
        assert!(err <= bound, "{what}[{i}] = {g}, want {w} (err {err})");
    }
}

#[test]
fn cuda_softmax_matches_reference_for_wide_rows() {
    if !cuda_available() {
        eprintln!("skip: no CUDA device/toolkit");
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    // d > 256 exercises the block-stride loop in the row-wise kernel.
    let (rows, d) = (4usize, 300usize);
    let xs: Vec<f32> = (0..rows * d)
        .map(|i| ((i * 37 % 101) as f32) * 0.05 - 2.0)
        .collect();
    let listing = "IN A AS F:HW\nOUT = SOFTMAX(A)\n";
    let outs = run_cuda(
        listing,
        &["OUT"],
        &[("A", vec![rows, d], xs.clone())],
        dir.path(),
        "softmax",
    );
    let mut want = vec![0f32; rows * d];
    for r in 0..rows {
        let row = &xs[r * d..(r + 1) * d];
        let mx = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max) as f64;
        let exps: Vec<f64> = row.iter().map(|x| ((*x as f64) - mx).exp()).collect();
        let se: f64 = exps.iter().sum();
        for j in 0..d {
            want[r * d + j] = (exps[j] / se) as f32;
        }
    }
    assert_close(&outs[0], &want, 1e-6, "SOFTMAX");
}

#[test]
fn cuda_layernorm_matches_reference_for_wide_rows() {
    if !cuda_available() {
        eprintln!("skip: no CUDA device/toolkit");
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let (rows, d) = (5usize, 300usize);
    let xs: Vec<f32> = (0..rows * d)
        .map(|i| ((i * 29 % 97) as f32) * 0.04 - 2.0)
        .collect();
    let w: Vec<f32> = (0..d).map(|j| 0.5 + (j % 7) as f32 * 0.1).collect();
    let b: Vec<f32> = (0..d).map(|j| (j % 5) as f32 * 0.02 - 0.04).collect();
    let listing = "CONFIG eps_ln 1e-5\nIN A AS F:HW\nIN W AS F:HW\nIN B AS F:HW\nOUT = LAYERNORM(A, W, B)\n";
    let outs = run_cuda(
        listing,
        &["OUT"],
        &[
            ("A", vec![rows, d], xs.clone()),
            ("W", vec![d], w.clone()),
            ("B", vec![d], b.clone()),
        ],
        dir.path(),
        "layernorm",
    );
    let eps = 1e-5f64;
    let mut want = vec![0f32; rows * d];
    for r in 0..rows {
        let row = &xs[r * d..(r + 1) * d];
        let mean = row.iter().map(|x| *x as f64).sum::<f64>() / d as f64;
        let var = row
            .iter()
            .map(|x| (*x as f64 - mean).powi(2))
            .sum::<f64>()
            / d as f64;
        let rs = 1.0 / (var + eps).sqrt();
        for j in 0..d {
            want[r * d + j] =
                ((row[j] as f64 - mean) * rs * w[j] as f64 + b[j] as f64) as f32;
        }
    }
    assert_close(&outs[0], &want, 1e-4, "LAYERNORM");
}

#[test]
fn cuda_matmul_cublas_path_matches_reference() {
    if !cuda_available() {
        eprintln!("skip: no CUDA device/toolkit");
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    // m*n*k > 4096 selects the cuBLAS path (row-major mapping check).
    let (m, k, n) = (16usize, 16usize, 17usize);
    let a: Vec<f32> = (0..m * k).map(|i| ((i * 13 % 23) as f32) * 0.05 - 0.5).collect();
    let b: Vec<f32> = (0..k * n).map(|i| ((i * 7 % 19) as f32) * 0.05 - 0.45).collect();
    let listing = "IN A AS F:HW\nIN B AS F:HW\nOUT = MATMUL(A, B)\n";
    let outs = run_cuda(
        listing,
        &["OUT"],
        &[
            ("A", vec![m, k], a.clone()),
            ("B", vec![k, n], b.clone()),
        ],
        dir.path(),
        "matmul",
    );
    let mut want = vec![0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let acc: f64 = (0..k)
                .map(|t| a[i * k + t] as f64 * b[t * n + j] as f64)
                .sum();
            want[i * n + j] = acc as f32;
        }
    }
    assert_close(&outs[0], &want, 2e-5, "MATMUL");
}

#[test]
fn cuda_concat_axis0_prepends_rows() {
    if !cuda_available() {
        eprintln!("skip: no CUDA device/toolkit");
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    // Patch-embed seam: [1, 4] CLS prepended to [3, 4] patch tokens.
    let cls: Vec<f32> = vec![10.0, 11.0, 12.0, 13.0];
    let tok: Vec<f32> = (0..12).map(|i| i as f32 * 0.5).collect();
    let listing = "IN CLS AS F:HW\nIN TOK AS F:HW\nOUT = CONCAT(CLS, TOK, 0)\n";
    let outs = run_cuda(
        listing,
        &["OUT"],
        &[
            ("CLS", vec![1, 4], cls.clone()),
            ("TOK", vec![3, 4], tok.clone()),
        ],
        dir.path(),
        "concat0",
    );
    let mut want = cls;
    want.extend_from_slice(&tok);
    assert_eq!(outs[0].len(), want.len());
    assert_close(&outs[0], &want, 0.0, "CONCAT axis0");
}

#[test]
fn cuda_arena_reuses_buffers_and_preserves_output() {
    if !cuda_available() {
        eprintln!("skip: no CUDA device/toolkit");
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let body = "\
IN A AS F:HW
IN B AS F:HW
C = ADD(A, B)
D = GELU(C)
E = ADD(D, A)
OUT = MUL(E, B)
";
    let plain = format!("{body}");
    let arena = format!("CONFIG cuda_arena 1\n{body}");
    let a: Vec<f32> = (0..16).map(|i| (i as f32) * 0.1 - 0.8).collect();
    let b: Vec<f32> = (0..16).map(|i| 1.5 - (i as f32) * 0.07).collect();
    let inputs = [
        ("A", vec![4, 4], a.clone()),
        ("B", vec![4, 4], b.clone()),
    ];

    // The opt-in build must use the pool.
    let mut feeds = Feeds::new();
    for (n, shape, data) in &inputs {
        feeds.insert(
            (*n).into(),
            Value::F32(ArrayD::from_shape_vec(IxDyn(shape), data.clone()).unwrap()),
        );
    }
    let art = compile_program(&arena, &opts(&["OUT"]), feeds).expect("compile arena");
    assert!(art.source.contains("g_cuda_arena"));
    assert!(art.source.contains("cudaMalloc((void **)&g_cuda_arena,"));

    let base = run_cuda(&plain, &["OUT"], &inputs, dir.path(), "arena_base");
    let got = run_cuda(&arena, &["OUT"], &inputs, dir.path(), "arena_on");
    assert_eq!(got[0].len(), base[0].len());
    for (i, (x, y)) in got[0].iter().zip(base[0].iter()).enumerate() {
        assert_eq!(x, y, "arena output differs at {i}");
    }
}

#[test]
fn cuda_linear_attn_matches_reference() {
    if !cuda_available() {
        eprintln!("skip: no CUDA device/toolkit");
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let (n, nh, dh, hid, m) = (5usize, 2usize, 4usize, 6usize, 5usize);
    let h = nh * dh;
    let q: Vec<f32> = (0..n * h).map(|i| ((i * 17 % 23) as f32) * 0.05 - 0.5).collect();
    let k: Vec<f32> = (0..n * h).map(|i| ((i * 11 % 19) as f32) * 0.04 - 0.35).collect();
    let v: Vec<f32> = (0..n * h).map(|i| ((i * 7 % 13) as f32) * 0.08 - 0.45).collect();
    let w1: Vec<f32> = (0..dh * hid).map(|i| ((i * 5 % 17) as f32) * 0.03 - 0.2).collect();
    let b1: Vec<f32> = (0..hid).map(|i| (i as f32) * 0.01 - 0.02).collect();
    let w2: Vec<f32> = (0..hid * m).map(|i| ((i * 3 % 11) as f32) * 0.05 - 0.2).collect();
    let b2: Vec<f32> = (0..m).map(|i| (i as f32) * 0.02 - 0.05).collect();
    let lnq_w: Vec<f32> = (0..dh).map(|i| 0.8 + (i as f32) * 0.05).collect();
    let lnq_b: Vec<f32> = (0..dh).map(|i| (i as f32) * 0.01 - 0.02).collect();
    let lnk_w: Vec<f32> = (0..dh).map(|i| 0.9 + (i as f32) * 0.04).collect();
    let lnk_b: Vec<f32> = (0..dh).map(|i| (i as f32) * -0.01 + 0.01).collect();
    let temp: Vec<f32> = vec![0.3, -0.2];

    let listing = format!(
        "IN Q AS F:HW\nIN K AS F:HW\nIN V AS F:HW\n\
         IN W1 AS F:HW\nIN B1 AS F:HW\nIN W2 AS F:HW\nIN B2 AS F:HW\n\
         IN LNQ_W AS F:HW\nIN LNQ_B AS F:HW\nIN LNK_W AS F:HW\nIN LNK_B AS F:HW\n\
         IN TEMP AS F:HW\nOUT = LINEAR_ATTN(Q, K, V, W1, B1, W2, B2, \
         LNQ_W, LNQ_B, LNK_W, LNK_B, TEMP, {nh}, {dh})\n"
    );
    let outs = run_cuda(
        &listing,
        &["OUT"],
        &[
            ("Q", vec![n, h], q.clone()),
            ("K", vec![n, h], k.clone()),
            ("V", vec![n, h], v.clone()),
            ("W1", vec![dh, hid], w1.clone()),
            ("B1", vec![hid], b1.clone()),
            ("W2", vec![hid, m], w2.clone()),
            ("B2", vec![m], b2.clone()),
            ("LNQ_W", vec![dh], lnq_w.clone()),
            ("LNQ_B", vec![dh], lnq_b.clone()),
            ("LNK_W", vec![dh], lnk_w.clone()),
            ("LNK_B", vec![dh], lnk_b.clone()),
            ("TEMP", vec![nh], temp.clone()),
        ],
        dir.path(),
        "linear_attn",
    );

    let mut want = vec![0f32; n * h];
    for head in 0..nh {
        let mut fq = vec![0f64; n * m];
        let mut fk = vec![0f64; n * m];
        let mut qn = vec![0f64; dh];
        let mut kn = vec![0f64; dh];
        let qt = (temp[head] as f64).exp();
        for i in 0..n {
            let mut meanq = 0f64;
            let mut meank = 0f64;
            for d in 0..dh {
                meanq += q[i * h + head * dh + d] as f64 / dh as f64;
                meank += k[i * h + head * dh + d] as f64 / dh as f64;
            }
            let mut varq = 0f64;
            let mut vark = 0f64;
            for d in 0..dh {
                let a = q[i * h + head * dh + d] as f64 - meanq;
                let b = k[i * h + head * dh + d] as f64 - meank;
                varq += a * a / dh as f64;
                vark += b * b / dh as f64;
            }
            let rsq = 1.0 / (varq + 1e-5).sqrt();
            let rsk = 1.0 / (vark + 1e-5).sqrt();
            for d in 0..dh {
                qn[d] = ((q[i * h + head * dh + d] as f64 - meanq) * rsq
                    * lnq_w[d] as f64
                    + lnq_b[d] as f64)
                    * qt;
                kn[d] = (k[i * h + head * dh + d] as f64 - meank) * rsk
                    * lnk_w[d] as f64
                    + lnk_b[d] as f64;
            }
            for mm in 0..m {
                let mut aq = b2[mm] as f64;
                let mut ak = b2[mm] as f64;
                for j in 0..hid {
                    let mut hq = b1[j] as f64;
                    let mut hk = b1[j] as f64;
                    for d in 0..dh {
                        let w = w1[d * hid + j] as f64;
                        hq += qn[d] * w;
                        hk += kn[d] * w;
                    }
                    aq += gelu_ref(hq) * w2[j * m + mm] as f64;
                    ak += gelu_ref(hk) * w2[j * m + mm] as f64;
                }
                let sp = |x: f64| if x > 20.0 { x } else { x.exp().ln_1p() };
                fq[i * m + mm] = sp(aq) + 1e-3;
                fk[i * m + mm] = sp(ak) + 1e-3;
            }
        }
        let mut s = vec![0f64; m * dh];
        let mut z = vec![0f64; m];
        for i in 0..n {
            for mm in 0..m {
                let f = fk[i * m + mm];
                z[mm] += f;
                for d in 0..dh {
                    s[mm * dh + d] += f * v[i * h + head * dh + d] as f64;
                }
            }
        }
        for i in 0..n {
            let mut den = 0f64;
            for mm in 0..m {
                den += fq[i * m + mm] * z[mm];
            }
            den = den.max(1e-6);
            for d in 0..dh {
                let mut acc = 0f64;
                for mm in 0..m {
                    acc += fq[i * m + mm] * s[mm * dh + d];
                }
                want[i * h + head * dh + d] = (acc / den) as f32;
            }
        }
    }
    assert_close(&outs[0], &want, 2e-4, "LINEAR_ATTN");
}
