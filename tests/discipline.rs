//! The allocation discipline, as a gate rather than a comment.
//!
//! These tests install `CountingAlloc` and assert three separable claims:
//!
//! 1. the boundary (frame assembly) allocates nothing;
//! 2. a warm job's allocation count is a fixed point, and a declared ceiling
//!    can reject it;
//! 3. the worker thread reuses the caller's reply buffer, so a warm
//!    `NodeHandle` round trip allocates nothing on the worker.
//!
//! Claim 2 is the important one: a number that is only printed decays. A
//! number that fails a run does not.

use std::process::Command;

use jotnar::asm::{assemble, AsmOptions};
use jotnar::payload::Feeds;
use jotnar::profile::{self, JobScope};
use jotnar::protocol::{decode_frame, request, Node, NodeHandle};
use jotnar::runtime::{run_budgeted, run_with_options, ExecOptions};
use jotnar::value::{array_from_vec, Value};
use serde_json::json;

#[global_allocator]
static ALLOC: profile::CountingAlloc = profile::CountingAlloc;

const LISTING: &str = "IN X\nT = IADD(X, X)\nOUT = IADD(T, X)\n";

fn i64_bytes(v: i64) -> Vec<u8> {
    v.to_le_bytes().to_vec()
}

fn inline_spec(off: usize, len: usize) -> serde_json::Value {
    json!({"kind": "I", "dtype": "i64", "shape": [1], "off": off, "len": len})
}

fn node_with_program() -> Node {
    let mut node = Node::new(vec!["lattice".into()]);
    let p = request(
        1,
        "program",
        json!({"name": "t", "backend": "lattice"}),
        LISTING.as_bytes(),
    )
    .unwrap();
    node.handle(&p);
    node
}

fn exec_frame(alloc_max: Option<u64>) -> Vec<u8> {
    let mut payload = Vec::new();
    payload.extend(i64_bytes(6));
    let mut fields = json!({
        "handle": "t",
        "inputs": {"X": inline_spec(0, 8)},
        "outputs": ["OUT"]
    });
    if let Some(m) = alloc_max {
        fields["alloc_max"] = json!(m);
    }
    request(2, "exec", fields, &payload).unwrap()
}

/// A declared ceiling is enforced at the job boundary, with a typed code.
#[test]
fn node_alloc_budget_is_enforced() {
    let mut node = node_with_program();
    let frame = exec_frame(Some(0));
    let (h, _) = decode_frame(&node.handle(&frame)).unwrap();
    assert_eq!(h["ok"], false, "ceiling 0 must reject: {h}");
    assert_eq!(h["error"]["code"], "alloc-budget");

    // A generous ceiling admits the same job.
    let frame = exec_frame(Some(1_000_000));
    let (h, _) = decode_frame(&node.handle(&frame)).unwrap();
    assert_eq!(h["ok"], true, "generous ceiling must admit: {h}");
}

/// The same enforcement, one level down, so `jotnar-det` can gate the
/// interpreter without a node.
#[test]
fn runtime_budget_rejects_and_admits() {
    let bound = assemble(LISTING, &AsmOptions::default()).unwrap();
    let mut feeds = Feeds::new();
    feeds.insert("X".to_string(), Value::I64(array_from_vec(&[1], vec![6]).unwrap()));
    let opts = ExecOptions::default();

    let scope = JobScope::begin();
    run_with_options(&bound, &feeds, &opts).unwrap();
    let steady = scope.finish().allocs;
    println!("steady interpreter allocations: {steady}");
    assert!(steady > 0, "the interpreter still allocates per op");

    assert!(run_budgeted(&bound, &feeds, &opts, 0).is_err(), "0 must reject");
    run_budgeted(&bound, &feeds, &opts, steady).expect("steady ceiling must admit");
}

/// Warm jobs do not drift. This is the regression half of the gate: adding an
/// allocation anywhere inside the job scope breaks it.
#[test]
fn warm_job_allocation_count_is_a_fixed_point() {
    let mut node = node_with_program();
    let frame = exec_frame(None);
    let mut out = Vec::new();

    for _ in 0..3 {
        node.handle_into(&frame, &mut out);
    }
    let mut counts = Vec::new();
    for _ in 0..8 {
        let scope = JobScope::begin();
        node.handle_into(&frame, &mut out);
        counts.push(scope.finish().allocs);
    }
    println!("warm job counts: {counts:?}");
    let first = counts[0];
    assert!(
        counts.iter().all(|c| *c == first),
        "allocation count drifted: {counts:?}"
    );
    // Fixed point alone catches drift, not a constant increase. The band is
    // the committed strictness: ~107 today, with room for allocator noise but
    // not for a new per-op allocation.
    assert!(first <= 170, "job allocation count grew to {first} (band: <=170)");
}

/// The worker writes into the caller's buffer, so the transport adds *nothing*
/// to the per-reply count: the worker's steady state equals an in-process
/// `handle_into`. The remaining cost is the job itself, not the channel.
#[test]
fn worker_transport_adds_no_allocations() {
    let node = NodeHandle::spawn(0, &["lattice"]);
    let p = request(
        1,
        "program",
        json!({"name": "t", "backend": "lattice"}),
        LISTING.as_bytes(),
    )
    .unwrap();
    node.request(p).unwrap();

    let exec = exec_frame(None);
    let mut out = Vec::new();
    for _ in 0..4 {
        node.request_into(exec.clone(), &mut out).unwrap();
    }

    // In-process reference: warm a local node, then measure one job.
    let mut local = node_with_program();
    let mut o = Vec::new();
    for _ in 0..3 {
        local.handle_into(&exec, &mut o);
    }
    let scope = JobScope::begin();
    local.handle_into(&exec, &mut o);
    let inproc = scope.finish().allocs;

    let base = node.worker_allocs();
    for _ in 0..12 {
        node.request_into(exec.clone(), &mut out).unwrap();
    }
    let reuse = (node.worker_allocs() - base) / 12;

    let base = node.worker_allocs();
    for _ in 0..12 {
        let _ = node.request(exec.clone()).unwrap();
    }
    let fresh = (node.worker_allocs() - base) / 12;

    println!("in-process handle_into: {inproc} allocations/job");
    println!("worker, reused buffer : {reuse} allocations/job");
    println!("worker, fresh Vec     : {fresh} allocations/job");

    // Caller-side cost of the handoff itself (excluding building the frame).
    let frame = exec.clone();
    let scope = JobScope::begin();
    node.request_into(frame, &mut out).unwrap();
    let caller = scope.finish().allocs;
    println!("caller handoff        : {caller} allocations/request");

    assert_eq!(reuse, inproc, "the transport should add no allocations");
    assert!(
        fresh > reuse,
        "a fresh reply Vec must cost more than a reused one ({fresh} vs {reuse})"
    );
    assert_eq!(node.worker_replies(), 30);
}

/// The `jotnar-det` gate must be able to say yes and no. This runs the real
/// binary, so it also proves the self-check wiring.
#[test]
fn det_alloc_gate_says_yes_and_no() {
    let dir = tempfile::tempdir().unwrap();
    let listing = dir.path().join("gate.asm");
    std::fs::write(&listing, LISTING).unwrap();
    let payload = dir.path().join("payload.ndjson");
    std::fs::write(
        &payload,
        "{\"streams\":{\"X\":{\"kind\":\"I64\",\"shape\":[2],\"data\":[3,4]}}}\n",
    )
    .unwrap();

    let bin = env!("CARGO_BIN_EXE_jotnar-det");
    let run = |gate: &str| {
        Command::new(bin)
            .arg(&listing)
            .arg(&payload)
            .args(["--outputs", "OUT", "--alloc-gate", gate])
            .output()
            .unwrap()
    };

    let ok = run("1000000");
    assert!(
        ok.status.success(),
        "generous gate failed: {}",
        String::from_utf8_lossy(&ok.stderr)
    );
    let out = String::from_utf8_lossy(&ok.stdout);
    assert!(out.contains("self-check=OK"), "gate never proved it can reject:\n{out}");

    let no = run("0");
    assert!(!no.status.success(), "ceiling 0 should fail the run");
    assert!(
        String::from_utf8_lossy(&no.stderr).contains("ALLOC FAIL"),
        "expected ALLOC FAIL, got {}",
        String::from_utf8_lossy(&no.stderr)
    );
}

fn chain_ops(n: usize) -> String {
    let mut s = String::from("IN X\n");
    s.push_str("t0 = IADD(X, X)\n");
    for i in 1..n {
        s.push_str(&format!("t{i} = IADD(t{}, X)\n", i - 1));
    }
    s.push_str(&format!("OUT = IADD(t{}, X)\n", n - 1));
    s
}

/// The spike: same listing, same numbers, two storage disciplines.
#[test]
fn stream_arena_reuses_the_graph_working_set() {
    use jotnar::stream_arena::{Plan, StreamArena};

    let n = 32usize;
    let width = 1024usize;
    let text = chain_ops(n);
    let bound = assemble(&text, &AsmOptions::default()).unwrap();
    let mut feeds = Feeds::new();
    feeds.insert(
        "X".to_string(),
        Value::I64(array_from_vec(&[width], vec![1i64; width]).unwrap()),
    );
    let opts = ExecOptions::default();

    // Interpreter: warm, then measure one job.
    for _ in 0..3 {
        run_with_options(&bound, &feeds, &opts).unwrap();
    }
    let scope = JobScope::begin();
    let got = run_with_options(&bound, &feeds, &opts).unwrap();
    let interp = scope.finish();

    // Arena: plan once, reserve once, then run.
    let plan = Plan::compile(&bound, &feeds, &["OUT".to_string()]).unwrap();
    let mut arena = StreamArena::new();
    arena.reserve_words(plan.words());
    let mut abytes = Vec::new();
    for _ in 0..3 {
        plan.run(&mut arena, &feeds, &mut abytes).unwrap();
    }
    let scope = JobScope::begin();
    plan.run(&mut arena, &feeds, &mut abytes).unwrap();
    let arena_w = scope.finish();

    // Same bytes out of both paths.
    let mut ibytes = Vec::new();
    if let Value::I64(a) = &got["OUT"] {
        for v in a.as_slice().unwrap() {
            ibytes.extend_from_slice(&v.to_le_bytes());
        }
    } else {
        panic!("expected an I output");
    }
    assert_eq!(abytes, ibytes, "arena and interpreter must agree");

    println!(
        "interpreter: allocs={} bytes={}",
        interp.allocs, interp.bytes
    );
    println!(
        "arena      : allocs={} bytes={} arena_words={} arena_bytes={}",
        arena_w.allocs,
        arena_w.bytes,
        arena.used_words(),
        arena.used_bytes()
    );
    println!(
        "per mnemonic: interpreter={:.1} arena={:.1}",
        interp.allocs as f64 / n as f64,
        arena_w.allocs as f64 / n as f64
    );

    assert_eq!(arena_w.allocs, 0, "the arena path must not allocate per job");
    assert_eq!(arena.used_words(), plan.words());
    let rate = interp.allocs as f64 / n as f64;
    assert!(rate > 5.0, "interpreter should still churn: {rate}");
    // Committed band: 24.2 today. A new per-op allocation would push this over.
    assert!(rate <= 32.0, "per-mnemonic allocations grew to {rate:.1} (band: <=32)");
}

/// The case that decides whether the arena scales past elementwise ops: a
/// reduction changes the output shape mid-graph (N -> scalar).
#[test]
fn stream_arena_handles_shape_changing_reductions() {
    use jotnar::stream_arena::{Plan, StreamArena};

    let n = 16usize;
    let width = 1024usize;
    let mut text = String::from("IN X\n");
    text.push_str("t0 = IADD(X, X)\n");
    for i in 1..n {
        text.push_str(&format!("t{i} = IADD(t{}, X)\n", i - 1));
    }
    text.push_str(&format!("S = IREDUCE_SUM(t{})\n", n - 1));
    text.push_str("OUT = IADD(S, S)\n");

    let bound = assemble(&text, &AsmOptions::default()).unwrap();
    let mut feeds = Feeds::new();
    feeds.insert(
        "X".to_string(),
        Value::I64(array_from_vec(&[width], vec![1i64; width]).unwrap()),
    );
    let opts = ExecOptions::default();

    for _ in 0..3 {
        run_with_options(&bound, &feeds, &opts).unwrap();
    }
    let scope = JobScope::begin();
    let got = run_with_options(&bound, &feeds, &opts).unwrap();
    let interp = scope.finish();

    let plan = Plan::compile(&bound, &feeds, &["OUT".to_string()]).unwrap();
    let mut arena = StreamArena::new();
    arena.reserve_words(plan.words());
    let mut abytes = Vec::new();
    for _ in 0..3 {
        plan.run(&mut arena, &feeds, &mut abytes).unwrap();
    }
    let scope = JobScope::begin();
    plan.run(&mut arena, &feeds, &mut abytes).unwrap();
    let arena_w = scope.finish();

    let mut ibytes = Vec::new();
    if let Value::I64(a) = &got["OUT"] {
        for v in a.as_slice().unwrap() {
            ibytes.extend_from_slice(&v.to_le_bytes());
        }
    }
    assert_eq!(abytes, ibytes, "arena and interpreter must agree");
    assert_eq!(abytes.len(), 8, "the reduction must produce one scalar");
    println!(
        "reduction chain: interpreter allocs={} arena allocs={} arena_bytes={}",
        interp.allocs,
        arena_w.allocs,
        arena.used_bytes()
    );
    assert_eq!(arena_w.allocs, 0);
    // 17 streams of `width` plus two scalars.
    assert_eq!(arena.used_words(), (n + 1) * width + 2);
}

/// The last structural unknown: an instruction with more than one output.
/// `IVARINT` returns (value, bytes consumed) as two 1-element streams.
#[test]
fn stream_arena_handles_multi_output_ops() {
    use jotnar::stream_arena::{Plan, StreamArena};

    let n = 16usize;
    let width = 1024usize;
    let mut text = String::from("IN X\n");
    text.push_str("t0 = IADD(X, X)\n");
    for i in 1..n {
        text.push_str(&format!("t{i} = IADD(t{}, X)\n", i - 1));
    }
    text.push_str(&format!("V, N = IVARINT(t{}, 0)\n", n - 1));
    text.push_str("OUT = IADD(V, N)\n");

    let bound = assemble(&text, &AsmOptions::default()).unwrap();
    let mut feeds = Feeds::new();
    feeds.insert(
        "X".to_string(),
        Value::I64(array_from_vec(&[width], vec![1i64; width]).unwrap()),
    );
    let opts = ExecOptions::default();

    for _ in 0..3 {
        run_with_options(&bound, &feeds, &opts).unwrap();
    }
    let scope = JobScope::begin();
    let got = run_with_options(&bound, &feeds, &opts).unwrap();
    let interp = scope.finish();

    let plan = Plan::compile(&bound, &feeds, &["OUT".to_string()]).unwrap();
    let mut arena = StreamArena::new();
    arena.reserve_words(plan.words());
    let mut abytes = Vec::new();
    for _ in 0..3 {
        plan.run(&mut arena, &feeds, &mut abytes).unwrap();
    }
    let scope = JobScope::begin();
    plan.run(&mut arena, &feeds, &mut abytes).unwrap();
    let arena_w = scope.finish();

    let mut ibytes = Vec::new();
    if let Value::I64(a) = &got["OUT"] {
        for v in a.as_slice().unwrap() {
            ibytes.extend_from_slice(&v.to_le_bytes());
        }
    }
    assert_eq!(abytes, ibytes, "arena and interpreter must agree");
    println!(
        "multi-output chain: interpreter allocs={} arena allocs={} arena_bytes={}",
        interp.allocs,
        arena_w.allocs,
        arena.used_bytes()
    );
    assert_eq!(arena_w.allocs, 0);
    // X and 16 intermediates, plus V, N, OUT.
    assert_eq!(arena.used_words(), (n + 1) * width + 3);
}

/// A mixed chain from the elementwise integer family, end to end.
#[test]
fn elementwise_family_chain_is_allocation_free() {
    use jotnar::stream_arena::{Plan, StreamArena};

    let width = 1024usize;
    let text = "IN X\n\
                a = IADD(X, X)\n\
                b = IXOR(a, 7)\n\
                c = IMAD(b, a, 3)\n\
                d = ISHL(c, 3)\n\
                e = ISAR(d, 1)\n\
                f = IMAX(e, b)\n\
                OUT = IADD(f, b)\n";
    let bound = assemble(text, &AsmOptions::default()).unwrap();
    let mut feeds = Feeds::new();
    feeds.insert(
        "X".to_string(),
        Value::I64(array_from_vec(&[width], vec![1i64; width]).unwrap()),
    );
    let opts = ExecOptions::default();

    for _ in 0..3 {
        run_with_options(&bound, &feeds, &opts).unwrap();
    }
    let scope = JobScope::begin();
    let got = run_with_options(&bound, &feeds, &opts).unwrap();
    let interp = scope.finish();

    let plan = Plan::compile(&bound, &feeds, &["OUT".to_string()]).unwrap();
    let mut arena = StreamArena::new();
    arena.reserve_words(plan.words());
    let mut abytes = Vec::new();
    for _ in 0..3 {
        plan.run(&mut arena, &feeds, &mut abytes).unwrap();
    }
    let scope = JobScope::begin();
    plan.run(&mut arena, &feeds, &mut abytes).unwrap();
    let arena_w = scope.finish();

    let mut ibytes = Vec::new();
    if let Value::I64(a) = &got["OUT"] {
        for v in a.as_slice().unwrap() {
            ibytes.extend_from_slice(&v.to_le_bytes());
        }
    }
    assert_eq!(abytes, ibytes);
    println!(
        "family chain: interpreter allocs={} arena allocs={} arena_bytes={}",
        interp.allocs,
        arena_w.allocs,
        arena.used_bytes()
    );
    assert_eq!(arena_w.allocs, 0);
    // X + 6 intermediates + OUT.
    assert_eq!(arena.used_words(), 8 * width);
}

/// The first genuinely different kernel shape: a 2-D contraction whose output
/// shape differs from both operands.
#[test]
fn stream_arena_handles_matmul() {
    use jotnar::stream_arena::{Plan, StreamArena};

    let n = 32usize;
    let text = "IN A\nIN B\nC = IMATMUL(A, B)\nOUT = IADD(C, C)\n";
    let bound = assemble(text, &AsmOptions::default()).unwrap();
    let mut feeds = Feeds::new();
    for name in ["A", "B"] {
        feeds.insert(
            name.to_string(),
            Value::I64(array_from_vec(&[n, n], vec![1i64; n * n]).unwrap()),
        );
    }
    let opts = ExecOptions::default();

    for _ in 0..3 {
        run_with_options(&bound, &feeds, &opts).unwrap();
    }
    let scope = JobScope::begin();
    let got = run_with_options(&bound, &feeds, &opts).unwrap();
    let interp = scope.finish();

    let plan = Plan::compile(&bound, &feeds, &["OUT".to_string()]).unwrap();
    let mut arena = StreamArena::new();
    arena.reserve_words(plan.words());
    let mut abytes = Vec::new();
    for _ in 0..3 {
        plan.run(&mut arena, &feeds, &mut abytes).unwrap();
    }
    let scope = JobScope::begin();
    plan.run(&mut arena, &feeds, &mut abytes).unwrap();
    let arena_w = scope.finish();

    let mut ibytes = Vec::new();
    if let Value::I64(a) = &got["OUT"] {
        for v in a.as_slice().unwrap() {
            ibytes.extend_from_slice(&v.to_le_bytes());
        }
    }
    assert_eq!(abytes, ibytes);
    println!(
        "matmul: interpreter allocs={} arena allocs={} arena_bytes={}",
        interp.allocs,
        arena_w.allocs,
        arena.used_bytes()
    );
    assert_eq!(arena_w.allocs, 0);
    // A + B + C + OUT, each n*n.
    assert_eq!(arena.used_words(), 4 * n * n);
}

/// Gather / window / scatter / checksum, end to end.
#[test]
fn index_ops_chain_is_allocation_free() {
    use jotnar::stream_arena::{Plan, StreamArena};

    let n = 64usize;
    let text = "IN T\nIN I\n\
                G = IGATHER1(T, I)\n\
                W = IWINDOW(G, 0, 64)\n\
                S = ISCATTER1(T, I, W)\n\
                C = ICRC32(S)\n\
                OUT = IADD(S, S)\n";
    let bound = assemble(text, &AsmOptions::default()).unwrap();
    let mut feeds = Feeds::new();
    feeds.insert(
        "T".to_string(),
        Value::I64(array_from_vec(&[n], (0..n as i64).collect()).unwrap()),
    );
    feeds.insert(
        "I".to_string(),
        Value::I64(array_from_vec(&[n], (0..n as i64).rev().collect()).unwrap()),
    );
    let opts = ExecOptions::default();

    for _ in 0..3 {
        run_with_options(&bound, &feeds, &opts).unwrap();
    }
    let scope = JobScope::begin();
    let got = run_with_options(&bound, &feeds, &opts).unwrap();
    let interp = scope.finish();

    let plan = Plan::compile(&bound, &feeds, &["OUT".to_string()]).unwrap();
    let mut arena = StreamArena::new();
    arena.reserve_words(plan.words());
    let mut abytes = Vec::new();
    for _ in 0..3 {
        plan.run(&mut arena, &feeds, &mut abytes).unwrap();
    }
    let scope = JobScope::begin();
    plan.run(&mut arena, &feeds, &mut abytes).unwrap();
    let arena_w = scope.finish();

    let mut ibytes = Vec::new();
    if let Value::I64(a) = &got["OUT"] {
        for v in a.as_slice().unwrap() {
            ibytes.extend_from_slice(&v.to_le_bytes());
        }
    }
    assert_eq!(abytes, ibytes);
    println!(
        "index chain: interpreter allocs={} arena allocs={} arena_bytes={}",
        interp.allocs,
        arena_w.allocs,
        arena.used_bytes()
    );
    assert_eq!(arena_w.allocs, 0);
}
