use ndarray::{ArrayD, IxDyn};
use phi_core::lattice::{self, BIAS};
use phi_core::numpy_ops;
use phi_core::Triple;

fn triple(shape: &[usize], s: Vec<i8>, e: Vec<i32>, z: Vec<u8>) -> Triple {
    Triple::from_vecs(shape, s, e, z).unwrap()
}

fn enc(v: &[f64], shape: &[usize]) -> Triple {
    lattice::encode(&ArrayD::from_shape_vec(IxDyn(shape), v.to_vec()).unwrap())
}

fn planes(t: &Triple) -> (Vec<i8>, Vec<i32>, Vec<u8>) {
    (
        t.s.iter().copied().collect(),
        t.e.iter().copied().collect(),
        t.z.iter().copied().collect(),
    )
}

#[test]
fn frozen_luts_match_python() {
    assert_eq!(
        &lattice::l_frac()[..6],
        &[262144, 261898, 261652, 261406, 261160, 260915]
    );
    assert_eq!(
        &lattice::l_coarse()[..6],
        &[-47200, -46462, -45725, -44987, -44250, -43512]
    );
    assert_eq!(&lattice::l_fine()[..6], &[0, 0, 0, 0, 0, 0]);
}

#[test]
fn from_fixed_match_python_vectors() {
    for (q, s, e, z) in [
        (0i64, -1i8, 19268i32, 1u8),
        (1, 1, 20005, 0),
        (2, 1, 20743, 0),
        (3, 1, 21174, 0),
        (5, 1, 21717, 0),
        (12345, 1, 30029, 0),
        (1000000, 1, 34704, 0),
        (-5, -1, 21717, 0),
        (-12345, -1, 30029, 0),
        (1 << 40, 1, 49505, 0),
    ] {
        let arr = ArrayD::from_elem(IxDyn(&[1]), q);
        let t = lattice::from_fixed(&arr, 33280).unwrap();
        assert_eq!(
            (t.s[IxDyn(&[0])], t.e[IxDyn(&[0])], t.z[IxDyn(&[0])]),
            (s, e, z),
            "q={q}"
        );
    }
}

#[test]
fn encode_03_matches_python() {
    let t = enc(&[0.3], &[1]);
    assert_eq!(
        (t.s[IxDyn(&[0])], t.e[IxDyn(&[0])], t.z[IxDyn(&[0])]),
        (1, 31487, 0)
    );
}

#[test]
fn tmul_matches_python() {
    let a = enc(&[0.5, -0.25], &[2]);
    let b = enc(&[2.0, 4.0], &[2]);
    let got = numpy_ops::tmul(&a, &b).unwrap();
    assert_eq!(planes(&got), (vec![1, -1], vec![32768, 32768], vec![0, 0]));
}

#[test]
fn activations_match_python() {
    let x = enc(&[-2.0, -0.5, 0.0, 0.5, 2.0], &[5]);
    assert_eq!(
        planes(&numpy_ops::sigmoid_int(&x).unwrap()),
        (
            vec![1, 1, 1, 1, 1],
            vec![30506, 31731, 32031, 32264, 32633],
            vec![0, 0, 0, 0, 0]
        )
    );
    assert_eq!(
        planes(&numpy_ops::silu_int(&x).unwrap()),
        (
            vec![-1, -1, 1, 1, 1],
            vec![31243, 30994, 0, 31527, 33370],
            vec![0, 0, 0, 0, 0]
        )
    );
    assert_eq!(
        planes(&numpy_ops::gelu_erf_int(&x).unwrap()),
        (
            vec![-1, -1, -1, 1, 1],
            vec![29483, 30780, 18756, 31638, 33480],
            vec![0, 0, 1, 0, 0]
        )
    );
}

#[test]
fn matmul_and_rmsnorm_match_python() {
    let a = enc(&[0.1, 0.2, 0.3, 0.4], &[2, 2]);
    let b = enc(&[0.5, 0.6], &[2, 1]);
    let mm = numpy_ops::matmul_int(&a, &b, 31344).unwrap();
    assert_eq!(planes(&mm), (vec![1, 1], vec![30883, 31766], vec![0, 0]));

    let w = enc(&[1.0, 1.0], &[2]);
    let rms = numpy_ops::rmsnorm_int(&a, &w, 35048, 4514).unwrap();
    assert_eq!(
        planes(&rms),
        (
            vec![1, 1, 1, 1],
            vec![32280, 33019, 32593, 32900],
            vec![0, 0, 0, 0]
        )
    );
}

#[test]
fn softmax_and_scan_match_python() {
    let s = enc(&[0.1, 0.9], &[1, 2]);
    let (num, den) = numpy_ops::softmaxN_triples(&s).unwrap();
    assert_eq!(
        num.iter().copied().collect::<Vec<_>>(),
        vec![7537661, 16777216]
    );
    assert_eq!(den.iter().copied().collect::<Vec<_>>(), vec![24314877]);

    let h = ArrayD::from_shape_vec(IxDyn(&[1, 1]), vec![26214i64]).unwrap();
    let ab = ArrayD::from_shape_vec(IxDyn(&[1, 1]), vec![BIAS]).unwrap();
    let bx = ArrayD::from_shape_vec(IxDyn(&[1, 1]), vec![13107i64]).unwrap();
    let out = numpy_ops::scan_step_int(&h, &ab, &bx).unwrap();
    assert_eq!(out.iter().copied().collect::<Vec<_>>(), vec![16383]);
}

#[test]
fn cmp_clip_helpers_match_lattice_order() {
    let zero = triple(&[1], vec![-1], vec![19268], vec![1]);
    let pos = enc(&[0.5], &[1]);
    let neg = enc(&[-0.5], &[1]);
    assert!(!phi_core::ops::cmp_gt(&zero, &zero).unwrap()[IxDyn(&[0])]);
    assert!(phi_core::ops::cmp_gt(&pos, &zero).unwrap()[IxDyn(&[0])]);
    assert!(!phi_core::ops::cmp_gt(&zero, &neg).unwrap()[IxDyn(&[0])]);
    assert!(phi_core::ops::cmp_gt(&pos, &neg).unwrap()[IxDyn(&[0])]);
    let clipped = phi_core::ops::clip_int(&pos, 0.0, 1.0).unwrap();
    assert_eq!(clipped.shape(), vec![1]);
}

#[test]
fn conv_and_deconv_match_python() {
    let x = enc(&[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], &[3, 3, 1]);
    let k = enc(&[1.0; 9], &[1, 3, 3, 1]);
    let conv = numpy_ops::phi_conv(&x, &k, 33280, 1, None).unwrap();
    assert_eq!(conv.shape(), vec![3, 3, 1]);
    assert_eq!(
        planes(&conv),
        (
            vec![1; 9],
            vec![32963, 33557, 33269, 33825, 34368, 34038, 33700, 34216, 33864],
            vec![0; 9]
        )
    );
    let dec = numpy_ops::deconv_int(&x, &k, 33280, 2, 1, None).unwrap();
    assert_eq!(dec.shape(), vec![5, 5, 1]);
    let (s, e, z) = planes(&dec);
    assert!(s.iter().all(|&v| v == 1));
    assert!(z.iter().all(|&v| v == 0));
    assert_eq!(e[0], 30318);
    assert_eq!(e[12], 32031);
}

#[test]
fn interp_and_nearest_match_python() {
    let x = enc(&[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], &[3, 3, 1]);
    let q = lattice::to_fixed(&x, 33280);
    let q3 =
        ndarray::ArrayD::from_shape_vec(IxDyn(&[3, 3, 1]), q.iter().copied().collect()).unwrap();
    let out = numpy_ops::interp_fixed(&q3, 2.0, 2.0).unwrap();
    assert_eq!(
        out.iter().copied().collect::<Vec<_>>(),
        vec![
            16200, 20253, 28361, 36462, 44556, 48604, 28350, 32405, 40516, 48615, 56700, 60743,
            52650, 56709, 64828, 72921, 80989, 85023, 76965, 81024, 89140, 97231, 105296, 109328,
            101297, 105349, 113455, 121545, 129622, 133660, 113463, 117512, 125612, 133703, 141785,
            145826
        ]
    );
    let near = numpy_ops::nearest2(&x).unwrap();
    assert_eq!(near.shape(), vec![6, 6, 1]);
    assert_eq!(
        near.e.iter().copied().collect::<Vec<_>>(),
        vec![
            30318, 30318, 31056, 31056, 31487, 31487, 30318, 30318, 31056, 31056, 31487, 31487,
            31793, 31793, 32031, 32031, 32224, 32224, 31793, 31793, 32031, 32031, 32224, 32224,
            32389, 32389, 32531, 32531, 32656, 32656, 32389, 32389, 32531, 32531, 32656, 32656
        ]
    );
}

#[test]
fn warp_zero_flow_is_identity() {
    let x = enc(&[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], &[3, 3, 1]);
    let q = lattice::to_fixed(&x, 33280);
    let q3 =
        ndarray::ArrayD::from_shape_vec(IxDyn(&[3, 3, 1]), q.iter().copied().collect()).unwrap();
    let flow = ndarray::ArrayD::zeros(IxDyn(&[3, 3, 2]));
    let w = numpy_ops::warp_fixed(&q3, &flow).unwrap();
    assert_eq!(
        w.iter().copied().collect::<Vec<_>>(),
        q.iter().copied().collect::<Vec<_>>()
    );
}

#[test]
fn tiled_scan_matches_python() {
    let ab = ArrayD::from_shape_vec(IxDyn(&[3, 1, 1]), vec![32768i64, 16384, 24576]).unwrap();
    let bx = ArrayD::from_shape_vec(IxDyn(&[3, 1, 1]), vec![100i64, 200, 300]).unwrap();
    let out = numpy_ops::tiled_scan(&ab, &bx, 2).unwrap();
    assert_eq!(out.iter().copied().collect::<Vec<_>>(), vec![100, 206, 319]);
}

#[test]
fn calibration_and_score_match_reference() {
    assert_eq!(phi_core::m_of(1.0, 512), 33280);
    assert_eq!(phi_core::m_of(0.1621028216371266, 512), 31344);
    let a = ArrayD::from_shape_vec(IxDyn(&[4]), vec![0.0, 1.0, 2.0, 3.0]).unwrap();
    let b = ArrayD::from_shape_vec(IxDyn(&[4]), vec![0.0, 1.0, 2.0, 4.0]).unwrap();
    let p = phi_core::psnr(&a, &b, 255.0).unwrap();
    assert!((p - 54.15140352195873).abs() < 1e-9, "{p}");
    assert!(phi_core::score::lattice_ok(&a, &b, 1.0).unwrap());
}
