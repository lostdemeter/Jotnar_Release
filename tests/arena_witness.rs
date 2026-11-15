//! The arena / zero-allocation witness, measured with a real allocator hook.
//!
//! This test binary installs `CountingAlloc` so allocation *counts* are real.
//! The library keeps the default allocator; only binaries that opt in pay for
//! the counters, and only while a `JobScope` is armed.

use jotnar::arena::JobArena;
use jotnar::profile::{self, JobScope};
use serde_json::json;

#[global_allocator]
static ALLOC: profile::CountingAlloc = profile::CountingAlloc;

const LISTING: &str = "IN X\nIN S\nT = IMUL(X, S)\nOUT = IADD(T, S)\n";

fn i64_bytes(v: i64) -> Vec<u8> {
    v.to_le_bytes().to_vec()
}

fn inline_spec(off: usize, len: usize) -> serde_json::Value {
    json!({"kind": "I", "dtype": "i64", "shape": [1], "off": off, "len": len})
}

fn node_with_program() -> jotnar::protocol::Node {
    let mut node = jotnar::protocol::Node::new(vec!["lattice".into()]);
    let f = jotnar::protocol::request(
        1,
        "program",
        json!({"name": "tile", "backend": "lattice"}),
        LISTING.as_bytes(),
    )
    .unwrap();
    node.handle(&f);
    node
}

fn exec_request(profile: bool) -> Vec<u8> {
    let mut payload = Vec::new();
    payload.extend(i64_bytes(7));
    payload.extend(i64_bytes(3));
    jotnar::protocol::request(
        2,
        "exec",
        json!({
            "handle": "tile",
            "inputs": {"X": inline_spec(0, 8), "S": inline_spec(8, 8)},
            "outputs": ["OUT"],
            "profile_ops": profile
        }),
        &payload,
    )
    .unwrap()
}

/// The arena's steady state is allocation-free: once the buffer covers the
/// largest job, `alloc` is a pointer bump and nothing touches the allocator.
#[test]
fn arena_steady_state_is_allocation_free() {
    let mut arena = JobArena::with_capacity(4096);
    // Warm: run the same shape several times so capacity stops growing.
    for _ in 0..3 {
        arena.reset();
        for _ in 0..16 {
            let off = arena.alloc(256).unwrap();
            arena.slice_mut(off, 256)[0] = 0xA5;
        }
    }
    let cap = arena.capacity();

    let scope = JobScope::begin();
    arena.reset();
    for _ in 0..16 {
        let off = arena.alloc(256).unwrap();
        arena.slice_mut(off, 256)[0] = 0x5A;
    }
    let w = scope.finish();

    assert_eq!(w.allocs, 0, "steady-state arena allocated: {w:?}");
    assert_eq!(arena.capacity(), cap, "capacity moved despite warm buffer");
    assert_eq!(arena.used(), 4096);
}

/// A cold job grows the arena; a warm job of the same shape does not. This is
/// exactly why the raw count is a *diagnostic*, not a canonical receipt field.
#[test]
fn job_witness_is_cold_then_warm() {
    let mut node = node_with_program();
    let exec = exec_request(true);

    let first = node.handle(&exec);
    let second = node.handle(&exec);

    let (h1, _) = jotnar::protocol::decode_frame(&first).unwrap();
    let (h2, _) = jotnar::protocol::decode_frame(&second).unwrap();
    let w1 = &h1["alloc_witness"];
    let w2 = &h2["alloc_witness"];
    println!("cold job witness: {w1}");
    println!("warm job witness: {w2}");
    println!("arena detail:     {}", h2["alloc_witness"]["arena_detail"]);

    // The first job pays one-time costs (arena growth plus whatever the
    // runtime initialises lazily); after that the count must be a fixed point.
    assert!(
        w2["allocs"].as_u64().unwrap() < w1["allocs"].as_u64().unwrap(),
        "steady state should allocate less than the cold job: {w1} vs {w2}"
    );
    let steady = w2["allocs"].as_u64().unwrap();
    for _ in 0..3 {
        let (h, _) = jotnar::protocol::decode_frame(&node.handle(&exec)).unwrap();
        assert_eq!(
            h["alloc_witness"]["allocs"].as_u64().unwrap(),
            steady,
            "steady-state allocation count must be a fixed point"
        );
    }
    // The boundary itself is clean on the warm path: the interior interpreter
    // allocations are what remain.
    assert_eq!(w2["arena_detail"]["warm"], true);
    assert_eq!(w1["receipt"]["arena"]["used"], w2["receipt"]["arena"]["used"]);
    assert_eq!(w1["receipt"], w2["receipt"], "receipt must stay canonical");
}

/// The canonical receipt carries only history-independent facts. Two identical
/// jobs must produce byte-identical receipts (and frames) with profiling off.
#[test]
fn receipt_excludes_allocation_history() {
    let mut node = node_with_program();
    let exec = exec_request(false);
    let a = node.handle(&exec);
    let b = node.handle(&exec);
    assert_eq!(a, b, "identical jobs must produce identical frames");

    let (h, _) = jotnar::protocol::decode_frame(&a).unwrap();
    assert_eq!(h["receipt"]["arena"]["used"], 8);
    assert!(
        h["receipt"].get("alloc_witness").is_none(),
        "history-dependent counts must not enter the canonical receipt"
    );
}

/// What the arena does not fix: the interpreter still allocates per op. Report
/// the measured interior count so the next step has a target.
#[test]
fn interior_allocations_are_measured() {
    let mut node = node_with_program();
    let exec = exec_request(true);
    let _ = node.handle(&exec); // warm
    let (h, _) = jotnar::protocol::decode_frame(&node.handle(&exec)).unwrap();
    let allocs = h["alloc_witness"]["allocs"].as_u64().unwrap();
    let bytes = h["alloc_witness"]["bytes"].as_u64().unwrap();
    println!("steady-state job: {allocs} allocations, {bytes} bytes requested");
    // Pin the current reality rather than pretend it is zero. Three ops over
    // one element is small; larger listings scale with op count.
    assert!(allocs >= 3, "expected per-op interior allocations, got {allocs}");
}

/// Separate the interpreter's own allocation cost from protocol/JSON framing.
/// Assembled once, run repeatedly: this is the number the destination-passing
/// work (step 3) has to move.
#[test]
fn interpreter_interior_cost_in_isolation() {
    use jotnar::asm::{assemble, AsmOptions};
    use jotnar::runtime::{run_with_options, ExecOptions};
    use jotnar::value::{array_from_vec, Value};

    let bound = assemble(LISTING, &AsmOptions::default()).unwrap();
    let mut feeds = jotnar::payload::Feeds::new();
    feeds.insert("X".to_string(), Value::I64(array_from_vec(&[1], vec![7]).unwrap()));
    feeds.insert("S".to_string(), Value::I64(array_from_vec(&[1], vec![3]).unwrap()));
    let opts = ExecOptions::default();

    for _ in 0..3 {
        run_with_options(&bound, &feeds, &opts).unwrap();
    }
    let scope = JobScope::begin();
    let out = run_with_options(&bound, &feeds, &opts).unwrap();
    let w = scope.finish();
    println!("interpreter only: {w:?}");
    assert_eq!(
        out["OUT"].shape().unwrap(),
        vec![1],
        "measurement must not change the result"
    );
    assert!(
        w.allocs >= 3,
        "expected one or more allocations per op, got {w:?}"
    );
}

/// Frame assembly itself allocates nothing once the buffer has grown: the
/// header is streamed, the length prefix is patched last, the payload appended
/// in place.
#[test]
fn frame_assembly_is_allocation_free() {
    let header = json!({"v": 1u64, "id": 7u64, "type": "pong", "ok": true,
                        "pong": {"impl": "jotnar-node", "v": 1u64}});
    let payload = vec![0xABu8; 4096];
    let mut out = Vec::new();
    for _ in 0..3 {
        jotnar::protocol::encode_frame_into(&mut out, &header, &payload).unwrap();
    }
    let scope = JobScope::begin();
    jotnar::protocol::encode_frame_into(&mut out, &header, &payload).unwrap();
    let w = scope.finish();
    assert_eq!(w.allocs, 0, "frame assembly allocated: {w:?}");
    let (h, p) = jotnar::protocol::decode_frame(&out).unwrap();
    assert_eq!(h["type"], "pong");
    assert_eq!(p.len(), 4096);
}

/// A reused reply buffer removes the per-reply `Vec`. What is left is the
/// header `Value` tree (request parse + reply build), which is the next target.
#[test]
fn handle_into_breakdown() {
    let mut node = node_with_program();
    let exec = exec_request(false);
    let mut out = Vec::new();
    for _ in 0..3 {
        node.handle_into(&exec, &mut out);
    }

    // 1. decode only
    let scope = JobScope::begin();
    let _ = jotnar::protocol::decode_frame(&exec).unwrap();
    let w_decode = scope.finish();

    // 2. full reply into a reused buffer
    let scope = JobScope::begin();
    node.handle_into(&exec, &mut out);
    let w_recv = scope.finish();

    // 3. same reply with a fresh Vec each time (the old `handle`)
    let scope = JobScope::begin();
    let _ = node.handle(&exec);
    let w_fresh = scope.finish();

    println!("decode_frame      : {:?}", w_decode);
    println!("handle_into (reuse): {:?}", w_recv);
    println!("handle (fresh Vec) : {:?}", w_fresh);
    assert!(w_recv.allocs <= w_fresh.allocs, "reuse must not cost more");
    assert_eq!(
        node.handle(&exec),
        { let mut v = Vec::new(); node.handle_into(&exec, &mut v); v },
        "handle and handle_into must produce identical frames"
    );
}

/// The hand-written header must be exactly what `serde_json` would emit for
/// the same tree: keys sorted, no whitespace, identical escapes. This is the
/// guard that lets the writer skip building the tree.
///
/// The diagnostic extension is deliberately excluded: `profile_ops` carries
/// `wall_ms`, and a timing float does not round-trip through parse+serialize
/// byte-for-byte. That is the same reason timing never enters the receipt.
#[test]
fn reply_header_is_canonical_json() {
    let mut node = node_with_program();
    let exec = exec_request(false);
    let f = node.handle(&exec);
    let nl = f.iter().position(|b| *b == b'\n').unwrap();
    let raw = &f[4..nl];
    let parsed: serde_json::Value = serde_json::from_slice(raw).unwrap();
    assert_eq!(
        raw,
        serde_json::to_vec(&parsed).unwrap().as_slice(),
        "header is not canonical serde_json"
    );

    // The diagnostic path must still be valid JSON; it is not canonicalised.
    let exec = exec_request(true);
    let f = node.handle(&exec);
    let nl = f.iter().position(|b| *b == b'\n').unwrap();
    let parsed: serde_json::Value = serde_json::from_slice(&f[4..nl]).unwrap();
    assert!(parsed["alloc_witness"].is_object());
    assert!(parsed["profile_ops"].is_array());
}

fn chain_listing(n: usize) -> String {
    let mut s = String::from("IN X\n");
    s.push_str("t0 = IADD(X, X)\n");
    for i in 1..n {
        s.push_str(&format!("t{i} = IADD(t{}, X)\n", i - 1));
    }
    s.push_str(&format!("OUT = IADD(t{}, X)\n", n - 1));
    s
}

fn run_chain(n: usize) -> (u64, u64, u64, u64, u64, i64) {
    let mut node = jotnar::protocol::Node::new(vec!["lattice".into()]);
    let p = jotnar::protocol::request(
        1, "program",
        json!({"name": "c", "backend": "lattice"}),
        chain_listing(n).as_bytes(),
    ).unwrap();
    node.handle(&p);
    let mut payload = Vec::new();
    payload.extend(std::iter::repeat(3i64).take(1024).flat_map(|v| v.to_le_bytes()));
    let spec = json!({"kind": "I", "dtype": "i64", "shape": [1024], "off": 0, "len": 8192});
    let exec = jotnar::protocol::request(
        2, "exec",
        json!({"handle": "c", "inputs": {"X": spec}, "outputs": ["OUT"], "profile_ops": true}),
        &payload,
    ).unwrap();
    let (h1, _) = jotnar::protocol::decode_frame(&node.handle(&exec)).unwrap();
    let (h2, _) = jotnar::protocol::decode_frame(&node.handle(&exec)).unwrap();
    let w1 = &h1["alloc_witness"];
    let w2 = &h2["alloc_witness"];
    (
        w1["allocs"].as_u64().unwrap(),
        w2["allocs"].as_u64().unwrap(),
        w2["bytes"].as_u64().unwrap(),
        h2["receipt"]["arena"]["used"].as_u64().unwrap(),
        w2["arena_detail"]["capacity"].as_u64().unwrap(),
        w2["live_delta"].as_i64().unwrap(),
    )
}

/// The graph's working set still scales with the number of mnemonics; the
/// boundary arena does not. This is the measured boundary between "reused"
/// and "still allocated".
#[test]
fn interior_scales_with_ops_arena_does_not() {
    let short = run_chain(4);
    let long = run_chain(32);
    println!(
        "4 ops : cold={} warm={} bytes={} arena_used={} cap={} live={}",
        short.0, short.1, short.2, short.3, short.4, short.5
    );
    println!(
        "32 ops: cold={} warm={} bytes={} arena_used={} cap={} live={}",
        long.0, long.1, long.2, long.3, long.4, long.5
    );

    // Arena: the same 8 KiB output image is reused regardless of op count.
    for m in [short, long] {
        assert_eq!(m.3, 8192, "arena should hold exactly one output image");
        assert_eq!(m.4, 8192, "arena capacity should stay at the image size");
    }
    assert!(short.1 < short.0 && long.1 < long.0, "warm < cold");

    // Interior: allocations grow with ops, roughly 10-40 per mnemonic.
    let per_op = (long.1 - short.1) as f64 / 28.0;
    println!("interior allocations per mnemonic: {per_op:.1}");
    assert!(
        (5.0..60.0).contains(&per_op),
        "unexpected per-op allocation rate: {per_op}"
    );
}
