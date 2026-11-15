//! Migration gate: with the arena enabled, every listing must produce exactly
//! what the interpreter produces, and the arena path must actually be taken.

use jotnar::asm::{assemble, AsmOptions};
use jotnar::payload::{value_to_json, Feeds};
use jotnar::profile::{self, JobScope};
use jotnar::runtime::{arena_stats, run_with_options, ExecOptions};
use jotnar::stream_arena::set_enabled;
use jotnar::value::Value;

#[global_allocator]
static ALLOC: profile::CountingAlloc = profile::CountingAlloc;

fn feeds(items: &[(&str, Vec<i64>, Vec<usize>)]) -> Feeds {
    let mut f = Feeds::new();
    for (name, vals, shape) in items {
        f.insert(
            name.to_string(),
            Value::I64(jotnar::value::array_from_vec(shape, vals.clone()).unwrap()),
        );
    }
    f
}

fn f_feeds(vals: &[f64]) -> Feeds {
    let mut f = Feeds::new();
    f.insert(
        "X".to_string(),
        Value::F64(jotnar::value::array_from_vec(&[vals.len()], vals.to_vec()).unwrap()),
    );
    f
}

fn compare(text: &str, f: &Feeds) -> (u64, u64) {
    let bound = assemble(text, &AsmOptions::default()).unwrap();
    let opts = ExecOptions::default();

    set_enabled(Some(false));
    let interp = run_with_options(&bound, f, &opts).unwrap();

    set_enabled(Some(true));
    let before = arena_stats();
    let arena = run_with_options(&bound, f, &opts).unwrap();
    let after = arena_stats();
    let (hits, fallbacks) = (after.0 - before.0, after.1 - before.1);

    let mut names: Vec<&String> = interp.keys().collect();
    names.sort();
    let mut an: Vec<&String> = arena.keys().collect();
    an.sort();
    assert_eq!(names, an, "stream set differs for {text:?}");
    for name in names {
        assert_eq!(
            value_to_json(&interp[name]).unwrap(),
            value_to_json(&arena[name]).unwrap(),
            "stream {name:?} differs for {text:?}"
        );
    }
    (hits, fallbacks)
}

#[test]
fn arena_and_interpreter_agree() {
    let x: Vec<i64> = (1..=8).collect();
    let cases: Vec<(&str, Feeds)> = vec![
        (
            "IN X\nT = IADD(X, X)\nOUT = IADD(T, X)\n",
            feeds(&[("X", x.clone(), vec![8])]),
        ),
        (
            "IN X\nS = IREDUCE_SUM(X)\nOUT = IADD(S, S)\n",
            feeds(&[("X", x.clone(), vec![8])]),
        ),
        (
            "IN A\nIN B\nOUT = IMATMUL(A, B)\n",
            feeds(&[
                ("A", vec![1, 2, 3, 4], vec![2, 2]),
                ("B", vec![5, 6, 7, 8], vec![2, 2]),
            ]),
        ),
        (
            "IN T\nIN I\nOUT = IGATHER1(T, I)\n",
            feeds(&[("T", vec![10, 20, 30], vec![3]), ("I", vec![2, 0], vec![2])]),
        ),
        (
            "IN X\nV, N = IVARINT(X, 0)\nOUT = IADD(V, N)\n",
            feeds(&[("X", vec![0xAC, 0x02, 0, 0, 0, 0, 0, 0], vec![8])]),
        ),
        (
            "IN X\nOUT = ICRC32(X)\n",
            feeds(&[("X", x.clone(), vec![8])]),
        ),
        (
            "IN X\nR = RELU(X)\nI = F2I(R)\nOUT = I2F(I)\n",
            f_feeds(&[-1.5, 0.0, 2.5, -3.0, 7.75]),
        ),
        ("IN X\nOUT = RESIZE2D(X, 3, 2, 0)\n", {
            let mut f = Feeds::new();
            f.insert(
                "X".to_string(),
                Value::F64(
                    jotnar::value::array_from_vec(
                        &[2, 2, 2],
                        (0..8).map(|i| i as f64 * 0.25).collect(),
                    )
                    .unwrap(),
                ),
            );
            f
        }),
        (
            "IN X\nOUT = RESHAPE2(X, 4, 2)\n",
            feeds(&[("X", (0..8).collect(), vec![2, 4])]),
        ),
        (
            "CONFIG eps_ln 1e-6\nIN X\nIN W\nIN B\nOUT = LAYERNORM(X, W, B)\n",
            {
                let mut f = Feeds::new();
                f.insert(
                    "X".to_string(),
                    Value::F64(
                        jotnar::value::array_from_vec(
                            &[2, 4],
                            vec![-1.5, 0.25, 2.0, -0.75, 3.5, -2.0, 0.0, 1.25],
                        )
                        .unwrap(),
                    ),
                );
                f.insert(
                    "W".to_string(),
                    Value::F64(
                        jotnar::value::array_from_vec(&[4], vec![1.0, 0.5, -1.0, 2.0]).unwrap(),
                    ),
                );
                f.insert(
                    "B".to_string(),
                    Value::F64(
                        jotnar::value::array_from_vec(&[4], vec![0.25, -0.25, 0.0, 1.0]).unwrap(),
                    ),
                );
                f
            },
        ),
        ("IN X\nV, I = TOPK(X, 3)\n", {
            let mut f = Feeds::new();
            f.insert(
                "X".to_string(),
                Value::F64(
                    jotnar::value::array_from_vec(
                        &[2, 5],
                        vec![1.0, 3.0, 3.0, 2.0, 3.0, -1.0, 5.0, 0.0, 5.0, 4.0],
                    )
                    .unwrap(),
                ),
            );
            f
        }),
    ];
    for (text, f) in cases {
        let (hits, fallbacks) = compare(text, &f);
        assert_eq!(hits, 1, "arena path not taken for {text:?}");
        assert_eq!(fallbacks, 0, "unexpected fallback for {text:?}");
    }
    set_enabled(None);
}

#[test]
fn every_soak_fixture_exercises_the_path_it_claims() {
    use std::path::Path;
    // The soak compares hashes; on its own it would also pass if every
    // fixture fell back to the interpreter. Pin the path per fixture.
    let cases: &[(&str, bool)] = &[
        ("arena_soak", true),
        ("arena_matmul", true),
        ("arena_f", true),
        ("arena_img", true),
        ("arena_conv", true),
        ("arena_fallback", false),
        ("alloc_gate", true),
        ("arena_reshape", true),
        ("arena_norm", true),
        ("arena_fmatmul", true),
        ("arena_conv_f", true),
        ("arena_index", true),
    ];
    let dir = Path::new("tests/fixtures");
    for &(base, on_arena) in cases {
        let text = std::fs::read_to_string(dir.join(format!("{base}.asm"))).unwrap();
        let feeds = jotnar::payload::read_payload(&dir.join(format!("{base}.ndjson"))).unwrap();
        let bound = assemble(&text, &AsmOptions::new(dir)).unwrap();
        set_enabled(Some(true));
        let before = arena_stats();
        let run = run_with_options(&bound, &feeds, &ExecOptions::default());
        let after = arena_stats();
        set_enabled(None);
        if let Err(e) = run {
            panic!("{base}: {e}");
        }
        let (hits, falls) = (after.0 - before.0, after.1 - before.1);
        if on_arena {
            assert_eq!(hits, 1, "{base} did not take the arena path");
            assert_eq!(falls, 0, "{base} unexpectedly fell back");
        } else {
            assert_eq!(hits, 0, "{base} should not be arena-plannable");
            assert_eq!(falls, 1, "{base} should report one fallback");
        }
    }
}

#[test]
fn unported_listing_falls_back_to_the_interpreter() {
    // LUMA has no `infer_shapes` rule, so the arena cannot plan it at all;
    // the interpreter runs it fine. That is the fallback path.
    let mut f = Feeds::new();
    f.insert(
        "X".to_string(),
        Value::F64(jotnar::value::array_from_vec(&[2, 2, 3], vec![0.25f64; 12]).unwrap()),
    );
    let (hits, fallbacks) = compare("IN X\nOUT = LUMA(X)\n", &f);
    assert_eq!(hits, 0, "arena should not handle a listing it cannot plan");
    assert_eq!(fallbacks, 1);
    set_enabled(None);
}

/// Measure the migration's effect: same answers, fewer allocations.
#[test]
fn arena_reduces_allocations_end_to_end() {
    let n = 32usize;
    let mut text = String::from("IN X\n");
    text.push_str("t0 = IADD(X, X)\n");
    for i in 1..n {
        text.push_str(&format!("t{i} = IADD(t{}, X)\n", i - 1));
    }
    text.push_str(&format!("OUT = IADD(t{}, X)\n", n - 1));
    let f = feeds(&[("X", vec![1; 1024], vec![1024])]);
    let bound = assemble(&text, &AsmOptions::default()).unwrap();
    let opts = ExecOptions::default();

    set_enabled(Some(false));
    for _ in 0..2 {
        run_with_options(&bound, &f, &opts).unwrap();
    }
    let scope = JobScope::begin();
    let _ = run_with_options(&bound, &f, &opts).unwrap();
    let interp = scope.finish();

    set_enabled(Some(true));
    for _ in 0..2 {
        run_with_options(&bound, &f, &opts).unwrap();
    }
    let scope = JobScope::begin();
    let _ = run_with_options(&bound, &f, &opts).unwrap();
    let arena = scope.finish();
    set_enabled(None);

    println!(
        "end-to-end: interpreter {} allocs / {} bytes, arena {} allocs / {} bytes",
        interp.allocs, interp.bytes, arena.allocs, arena.bytes
    );
    assert!(arena.allocs < interp.allocs, "arena should allocate less");
}

/// The warm path: a node compiles each program's plan once and reuses the
/// stream arena across jobs.
#[test]
fn node_warm_path_avoids_plan_compilation() {
    use jotnar::protocol::{decode_frame, Node, NodeHandle};

    let n = 32usize;
    let mut text = String::from("IN X\n");
    text.push_str("t0 = IADD(X, X)\n");
    for i in 1..n {
        text.push_str(&format!("t{i} = IADD(t{}, X)\n", i - 1));
    }
    text.push_str(&format!("OUT = IADD(t{}, X)\n", n - 1));

    let frame = |kind: &str| {
        let mut payload = Vec::new();
        payload.extend(
            std::iter::repeat(1i64)
                .take(256)
                .flat_map(|v| v.to_le_bytes()),
        );
        jotnar::protocol::request(
            2,
            "exec",
            serde_json::json!({
                "handle": kind,
                "inputs": {"X": {"kind": "I", "dtype": "i64", "shape": [256], "off": 0, "len": 2048}},
                "outputs": ["OUT"]
            }),
            &payload,
        )
        .unwrap()
    };

    let mut node = Node::new(vec!["lattice".into()]);
    let p = jotnar::protocol::request(
        1,
        "program",
        serde_json::json!({"name": "chain", "backend": "lattice"}),
        text.as_bytes(),
    )
    .unwrap();
    node.handle(&p);

    set_enabled(Some(false));
    let f = frame("chain");
    for _ in 0..2 {
        node.handle(&f);
    }
    let scope = JobScope::begin();
    node.handle(&f);
    let interp = scope.finish();

    set_enabled(Some(true));
    for _ in 0..2 {
        node.handle(&f);
    }
    let scope = JobScope::begin();
    let warm = node.handle(&f);
    let arena = scope.finish();
    set_enabled(None);

    let (ah, _) = decode_frame(&warm).unwrap();
    assert_eq!(ah["ok"], true);
    println!(
        "node warm: interpreter {} allocs / {} bytes, arena {} allocs / {} bytes",
        interp.allocs, interp.bytes, arena.allocs, arena.bytes
    );
    assert!(
        arena.allocs < interp.allocs / 2,
        "cached path should be much cheaper"
    );
    assert_eq!(NodeHandle::DEFAULT_QUEUE, 64);
}

/// The size cap must send an oversized working set back to the interpreter,
/// not fail the job.
#[test]
fn size_cap_falls_back_to_interpreter() {
    use jotnar::stream_arena::{set_enabled, set_max_bytes};

    let f = feeds(&[("X", vec![1; 256], vec![256])]);
    let text = "IN X\nT = IADD(X, X)\nOUT = IADD(T, X)\n";
    let bound = assemble(text, &AsmOptions::default()).unwrap();
    let opts = ExecOptions::default();

    let prev_cap = set_max_bytes(Some(1024)); // working set is 6 KiB
    set_enabled(Some(true));
    let before = arena_stats();
    let capped = run_with_options(&bound, &f, &opts).unwrap();
    let after = arena_stats();
    assert_eq!(after.0 - before.0, 0, "oversized plan should not run");
    assert_eq!(after.1 - before.1, 1, "oversized plan should fall back");

    set_enabled(Some(false));
    let interp = run_with_options(&bound, &f, &opts).unwrap();
    for name in interp.keys() {
        assert_eq!(
            value_to_json(&interp[name]).unwrap(),
            value_to_json(&capped[name]).unwrap(),
            "stream {name:?} differs under the cap"
        );
    }
    set_max_bytes(prev_cap);
    set_enabled(None);
}
