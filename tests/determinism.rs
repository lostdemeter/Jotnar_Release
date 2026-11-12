//! Exact-output determinism gates for the threaded runtime.
//!
//! Independent outputs are partitioned across workers; the reference
//! (single-thread) hash must survive every thread count, repeat, and
//! concurrent job. This test is the cheap CI companion to the `jotnar-det`
//! harness, which does the same over external listings.

use jotnar::asm::AsmOptions;
use jotnar::payload::Feeds;
use jotnar::runtime::{self, ExecOptions};
use jotnar::t2;
use jotnar::value::{Triple, Value};
use ndarray::{ArrayD, IxDyn};

const LISTING: &str = "\
CONFIG t2 24
CONFIG bridge_f 24
CONFIG eps_ln 1e-6
IN X
IN ONES
IN W1
IN B1
IN W2
IN B2
IN LN_W
IN LN_B
IN A
IN B
LN = LAYERNORM(X, LN_W, LN_B)
C1 = CONV2D(X, W1, B1, 1, 1)
R1 = RELU(C1)
D1 = DECONV2D(R1, W2, B2, 2, 1)
T1 = TRANSPOSE(A)
M1 = MATMUL(A, B)
S1 = SOFTMAX_WIDE(M1)
G1 = GELU(S1)
K1 = CONCAT(G1, ONES, 1)
OUT = ADD(K1, K1)
";

fn arr(shape: &[usize], values: impl Iterator<Item = f64>) -> Triple {
    let v: Vec<f64> = values.collect();
    t2::encode_array(&ArrayD::from_shape_vec(IxDyn(shape), v).unwrap(), 24)
}

fn feeds() -> Feeds {
    let mut f = Feeds::new();
    // Deterministic pseudo-data in roughly [-2, 2].
    let vals = |n: usize, salt: usize| {
        (0..n).map(move |i| {
            let x = ((i * 37 + salt * 53) % 101) as f64;
            (x - 50.0) / 25.0
        })
    };
    f.insert(
        "X".into(),
        Value::Triple(arr(&[8, 8, 4], vals(8 * 8 * 4, 1))),
    );
    f.insert("ONES".into(), Value::Triple(arr(&[8, 1], vals(8, 2))));
    f.insert(
        "W1".into(),
        Value::Triple(arr(&[6, 3, 3, 4], vals(6 * 3 * 3 * 4, 3))),
    );
    f.insert("B1".into(), Value::Triple(arr(&[6], vals(6, 4))));
    f.insert(
        "W2".into(),
        Value::Triple(arr(&[4, 3, 3, 6], vals(4 * 3 * 3 * 6, 5))),
    );
    f.insert("B2".into(), Value::Triple(arr(&[4], vals(4, 6))));
    f.insert("LN_W".into(), Value::Triple(arr(&[4], vals(4, 7))));
    f.insert("LN_B".into(), Value::Triple(arr(&[4], vals(4, 8))));
    f.insert("A".into(), Value::Triple(arr(&[8, 12], vals(8 * 12, 9))));
    f.insert("B".into(), Value::Triple(arr(&[12, 8], vals(12 * 8, 10))));
    f
}

const OUTS: &[&str] = &["LN", "C1", "R1", "D1", "T1", "M1", "S1", "G1", "K1", "OUT"];

fn hash_feeds(feeds: &Feeds, names: &[&str]) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for name in names {
        for &b in name.as_bytes() {
            h ^= b as u64;
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
        feeds.get(*name).unwrap().hash_into(&mut h);
    }
    h
}

fn run(threads: usize) -> u64 {
    let f = feeds();
    let got = runtime::run_text_with_options(
        LISTING,
        &AsmOptions::new("."),
        &f,
        &ExecOptions::new(threads),
    )
    .expect("run");
    hash_feeds(&got, OUTS)
}

#[test]
fn exact_t2_is_bit_identical_across_threads() {
    let reference = run(1);
    for threads in [2, 3, 4, 8, 16] {
        assert_eq!(
            run(threads),
            reference,
            "T2 output changed at threads={threads}"
        );
    }
}

#[test]
fn exact_t2_is_bit_identical_across_repeats() {
    let reference = run(1);
    for _ in 0..3 {
        assert_eq!(run(1), reference, "serial repeat changed");
        assert_eq!(run(4), reference, "threaded repeat changed");
    }
}

#[test]
fn exact_t2_is_bit_identical_across_concurrent_jobs() {
    let reference = run(1);
    std::thread::scope(|scope| {
        let mut handles = Vec::new();
        for _ in 0..4 {
            handles.push(scope.spawn(|| run(2)));
        }
        for h in handles {
            assert_eq!(h.join().unwrap(), reference, "concurrent job changed");
        }
    });
}
