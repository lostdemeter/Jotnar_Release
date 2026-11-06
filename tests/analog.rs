use std::collections::BTreeMap;

use jotnar::showcase::analog_demo::{
    boundary_ring, errors, make_fields, readout, run, solve_digital, solve_rc, AnalogOptions,
};

#[test]
fn boundary_ring_is_the_ring_outside_the_hole() {
    let mut mask = ndarray::Array3::<bool>::from_elem((1, 9, 9), false);
    for y in 3..6 {
        for x in 3..6 {
            mask[[0, y, x]] = true;
        }
    }
    let ring = boundary_ring(&mask, 1);
    assert!(!ring[[0, 4, 4]], "interior stays interior");
    assert!(ring[[0, 2, 4]], "one pixel outside the hole");
    assert!(!ring[[0, 1, 1]], "far corner");
}

#[test]
fn ideal_rc_reaches_the_digital_laplace_solution() {
    let (target, mask) = make_fields("linear", 16, 24, 1, 0).unwrap();
    let digital = solve_digital(&target, &mask, 1_500, None, 0.5);
    let analog = solve_rc(&target, &mask, 3_000, 0.2, None, None, 0.5);
    let e = errors(&analog, &target, &mask);
    let d = errors(&digital, &target, &mask);
    assert!((e.hole_l1 - d.hole_l1).abs() < 1e-3, "{e:?} vs {d:?}");
}

#[test]
fn dac_resolution_is_observable() {
    let (target, mask) = make_fields("linear", 16, 24, 1, 0).unwrap();
    let u4 = readout(
        &solve_rc(&target, &mask, 3_000, 0.2, Some(4), None, 0.5),
        Some(12),
        0.0,
        0,
    );
    let u8 = readout(
        &solve_rc(&target, &mask, 3_000, 0.2, Some(8), None, 0.5),
        Some(12),
        0.0,
        0,
    );
    let e4 = errors(&u4, &target, &mask);
    let e8 = errors(&u8, &target, &mask);
    assert!(e8.hole_l1 < e4.hole_l1, "{e8:?} !< {e4:?}");
    assert!(e8.hole_l1 < 2e-3, "{e8:?}");
}

#[test]
fn analog_demo_writes_the_rjm_and_gate() {
    let dir = std::env::temp_dir().join(format!("jotnar_analog_test_{}", std::process::id()));
    let opts = AnalogOptions {
        h: 24,
        w: 32,
        steps: 120,
        digital_iters: 120,
        vector_steps: 120,
        out: dir.clone(),
        quick: true,
    };
    run(&opts).unwrap();
    let metrics: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(dir.join("metrics.json")).unwrap()).unwrap();
    let artifact: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(dir.join("rjm_field2d.research.json")).unwrap(),
    )
    .unwrap();
    assert!(metrics["linear"]["digital"]["hole_l1"].as_f64().unwrap() >= 0.0);
    assert!(artifact["gate"]["bit_exact"] == false);
    assert!(artifact["gate"]["accepted"] == true);
    assert!(artifact["capability_profiles"]["live"]["accepted"] == true);
    assert!(artifact["capability_profiles"]["live_no_dac"]["accepted"] == false);
    assert!(artifact["capability_profiles"]["baked_map"]["accepted"] == true);
    assert!(dir.join("panel.png").exists());
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn vector_scales_cells_not_quality() {
    let mut rows = BTreeMap::new();
    for lanes in [1usize, 2, 4] {
        let (t, m) = make_fields("linear", 16, 24, lanes, 0).unwrap();
        let u = readout(
            &solve_rc(&t, &m, 300, 0.2, Some(8), None, 0.5),
            Some(10),
            0.0,
            0,
        );
        rows.insert(lanes, errors(&u, &t, &m).hole_l1);
    }
    assert!(rows[&1] > 0.0);
    assert!((rows[&1] - rows[&4]).abs() < 1e-3);
}
