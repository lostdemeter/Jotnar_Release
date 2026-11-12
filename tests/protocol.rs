use std::collections::BTreeMap;
use std::thread;

use serde_json::json;

use jotnar::protocol::{
    decode_frame, decode_stream, encode_frame, encode_stream, exec_outputs, request, Node,
    NodeHandle,
};
use jotnar::t2;
use jotnar::value::Value;
use ndarray::{ArrayD, IxDyn};

fn i64_bytes(v: i64) -> Vec<u8> {
    v.to_le_bytes().to_vec()
}

fn inline_spec(off: usize, len: usize) -> serde_json::Value {
    json!({"kind": "I", "dtype": "i64", "shape": [1], "off": off, "len": len})
}

#[test]
fn node_program_store_and_exec_are_deterministic() {
    let listing = "IN X\nIN S\nT = IMUL(X, S)\nOUT = IADD(T, S)\n";
    let mut node = Node::new(vec!["lattice".into()]);
    let f = request(
        1,
        "program",
        json!({"name": "tile", "backend": "lattice"}),
        listing.as_bytes(),
    )
    .unwrap();
    let (h, _) = decode_frame(&node.handle(&f)).unwrap();
    assert_eq!(h["ok"], true);
    assert_eq!(h["handle"], "tile");

    let mut payload = Vec::new();
    payload.extend(i64_bytes(7));
    payload.extend(i64_bytes(3));
    let exec = request(
        2,
        "exec",
        json!({
            "handle": "tile",
            "inputs": {"X": inline_spec(0, 8), "S": inline_spec(8, 8)},
            "outputs": ["OUT"]
        }),
        &payload,
    )
    .unwrap();
    let r1 = node.handle(&exec);
    let r2 = node.handle(&exec);
    assert_eq!(r1, r2, "same request must produce the same frame bytes");
    let out = exec_outputs(&r1).unwrap();
    assert_eq!(
        i64::from_le_bytes(out["OUT"].as_slice().try_into().unwrap()),
        24
    );
}

#[test]
fn blob_shelf_binds_by_name() {
    let listing = "IN X\nIN S\nOUT = IADD(X, S)\n";
    let mut node = Node::new(vec!["lattice".into()]);
    let p = request(
        1,
        "program",
        json!({"name": "add", "backend": "lattice"}),
        listing.as_bytes(),
    )
    .unwrap();
    node.handle(&p);
    let s = request(
        2,
        "store",
        json!({"name": "S", "stream": {"kind": "I", "dtype": "i64", "shape": [1]}}),
        &i64_bytes(5),
    )
    .unwrap();
    let (h, _) = decode_frame(&node.handle(&s)).unwrap();
    assert_eq!(h["ok"], true);
    let e = request(
        3,
        "exec",
        json!({"handle": "add", "inputs": {"X": {"store": "S"}, "S": {"store": "S"}}, "outputs": ["OUT"]}),
        &[],
    )
    .unwrap();
    let out = exec_outputs(&node.handle(&e)).unwrap();
    assert_eq!(
        i64::from_le_bytes(out["OUT"].as_slice().try_into().unwrap()),
        10
    );
}

#[test]
fn bad_frame_refuses_loud() {
    let mut node = Node::new(vec!["lattice".into()]);
    let r = node.handle(b"not a frame");
    let (h, _) = decode_frame(&r).unwrap();
    assert_eq!(h["ok"], false);
    assert_eq!(h["error"]["code"], "malformed");
}

#[test]
fn async_nodes_do_not_change_bytes() {
    let listing = "IN X\nOUT = IADD(X, X)\n";
    let n1 = NodeHandle::spawn(0, &["lattice"]);
    let n2 = NodeHandle::spawn(1, &["lattice"]);
    for (i, n) in [&n1, &n2].iter().enumerate() {
        let p = request(
            10,
            "program",
            json!({"name": "twice", "backend": "lattice"}),
            listing.as_bytes(),
        )
        .unwrap();
        n.request(p).unwrap();
        let _ = i;
    }
    let mut outs: Vec<Vec<u8>> = Vec::new();
    let nodes = [&n1, &n2];
    thread::scope(|s| {
        let mut hs = Vec::new();
        for (i, n) in nodes.iter().enumerate() {
            hs.push(s.spawn(move || {
                let ex = request(
                    100 + i as u64,
                    "exec",
                    json!({"handle": "twice", "inputs": {"X": inline_spec(0, 8)}, "outputs": ["OUT"]}),
                    &i64_bytes(21 + i as i64),
                )
                .unwrap();
                n.request(ex).unwrap()
            }));
        }
        for h in hs {
            outs.push(h.join().unwrap());
        }
    });
    let a = exec_outputs(&outs[0]).unwrap();
    let b = exec_outputs(&outs[1]).unwrap();
    assert_eq!(
        i64::from_le_bytes(a["OUT"].as_slice().try_into().unwrap()),
        42
    );
    assert_eq!(
        i64::from_le_bytes(b["OUT"].as_slice().try_into().unwrap()),
        44
    );
    // The frame includes distinct ids, but the output payloads are exact.
    let mut m = BTreeMap::new();
    m.insert("a", a["OUT"].clone());
    m.insert("b", b["OUT"].clone());
    assert_ne!(m["a"], m["b"]);
}

#[test]
fn frame_roundtrip() {
    let h = json!({"v": 1, "id": 9, "type": "ping"});
    let f = encode_frame(&h, b"abc").unwrap();
    let (g, p) = decode_frame(&f).unwrap();
    assert_eq!(g, h);
    assert_eq!(p, b"abc");
}

#[test]
fn node_token_rejects_missing_and_bad_auth() {
    let mut node = Node::new(vec!["lattice".into()]).with_token("s3cret");
    // Missing token.
    let f = request(1, "ping", json!({}), &[]).unwrap();
    let (h, _) = decode_frame(&node.handle(&f)).unwrap();
    assert_eq!(h["ok"], false);
    assert_eq!(h["error"]["code"], "unauthorized");
    // Wrong token.
    let f = request(2, "ping", json!({"auth": "wrong"}), &[]).unwrap();
    let (h, _) = decode_frame(&node.handle(&f)).unwrap();
    assert_eq!(h["ok"], false);
    assert_eq!(h["error"]["code"], "unauthorized");
    // Right token.
    let f = request(3, "ping", json!({"auth": "s3cret"}), &[]).unwrap();
    let (h, _) = decode_frame(&node.handle(&f)).unwrap();
    assert_eq!(h["ok"], true);
}

fn t2_triple(shape: &[usize], values: &[f64]) -> jotnar::Triple {
    let a = ArrayD::from_shape_vec(IxDyn(shape), values.to_vec()).unwrap();
    t2::encode_array(&a, 24)
}

#[test]
fn ping_reports_live_registry_count() {
    let mut node = Node::new(vec!["lattice".into()]);
    let f = request(1, "ping", json!({}), &[]).unwrap();
    let (h, _) = decode_frame(&node.handle(&f)).unwrap();
    assert_eq!(h["pong"]["mnemonics"], jotnar::ops::registry().len());
}

#[test]
fn t2_stream_roundtrips_all_four_planes() {
    let t = t2_triple(&[2, 3], &[0.5, -1.25, 2.0, -0.75, 3.5, 0.125]);
    let (spec, bytes) = encode_stream(&Value::Triple(t.clone())).unwrap();
    assert_eq!(spec.kind, "T");
    assert_eq!(spec.dtype, "t2");
    assert_eq!(bytes.len(), 6 * 10);
    let back = match decode_stream(&spec, &bytes).unwrap() {
        Value::Triple(t) => t,
        _ => panic!("not a triple"),
    };
    assert_eq!(back.s, t.s);
    assert_eq!(back.e, t.e);
    assert_eq!(back.z, t.z);
    assert_eq!(back.m, t.m);
}

#[test]
fn node_t2_exec_is_threaded_and_policy_independent() {
    let listing = "\
CONFIG t2 24
CONFIG bridge_f 24
IN A
IN B
OUT = MATMUL(A, B)
";
    let mut node = Node::new(vec!["lattice".into()]);
    let p = request(
        1,
        "program",
        json!({"name": "mul", "backend": "lattice"}),
        listing.as_bytes(),
    )
    .unwrap();
    let (h, _) = decode_frame(&node.handle(&p)).unwrap();
    assert_eq!(h["ok"], true);

    let a = t2_triple(&[2, 3], &[0.5, -1.25, 2.0, -0.75, 3.5, 0.125]);
    let b = t2_triple(&[3, 2], &[0.5, -1.5, 2.0, 0.25, -0.75, 1.25]);
    for (name, t) in [("A", &a), ("B", &b)] {
        let (spec, bytes) = encode_stream(&Value::Triple((*t).clone())).unwrap();
        let s = request(
            2,
            "store",
            json!({"name": name, "stream": spec.json()}),
            &bytes,
        )
        .unwrap();
        let (h, _) = decode_frame(&node.handle(&s)).unwrap();
        assert_eq!(h["ok"], true, "{name}");
    }

    let mkexec = |id: u64, policy: &str| {
        request(
            id,
            "exec",
            json!({
                "handle": "mul",
                "inputs": {"A": {"store": "A"}, "B": {"store": "B"}},
                "outputs": ["OUT"],
                "thread_policy": policy,
                "determinism": "exact-int"
            }),
            &[],
        )
        .unwrap()
    };

    let r = node.handle(&mkexec(3, "reference"));
    let a_reference = node.handle(&mkexec(4, "auto"));
    let (hr, _) = decode_frame(&r).unwrap();
    let (ha, _) = decode_frame(&a_reference).unwrap();
    assert_eq!(hr["ok"], true, "{hr}");
    assert_eq!(ha["ok"], true, "{ha}");
    assert_eq!(hr["receipt"]["threads"], 1);
    assert!(ha["receipt"]["threads"].as_u64().unwrap() >= 1);
    assert_eq!(hr["receipt"]["profile"], "exact-int");
    assert_eq!(
        hr["receipt"]["sched_version"],
        jotnar::runtime::SCHED_VERSION
    );
    assert_eq!(
        hr["receipt"]["partition"]["kind"],
        "contiguous-output-blocks"
    );
    assert_eq!(hr["receipt"]["program_hash"].as_str().unwrap().len(), 16);
    assert_eq!(
        hr["receipt"]["output_hashes"]["OUT"],
        ha["receipt"]["output_hashes"]["OUT"]
    );

    let or = exec_outputs(&r).unwrap();
    let oa = exec_outputs(&a_reference).unwrap();
    assert_eq!(or["OUT"], oa["OUT"], "policy changed exact output bytes");
}

#[test]
fn node_receipt_is_canonical_and_excludes_timing() {
    let listing = "IN X\nOUT = IADD(X, X)\n";
    let mut node = Node::new(vec!["lattice".into()]);
    let p = request(
        1,
        "program",
        json!({"name": "twice", "backend": "lattice"}),
        listing.as_bytes(),
    )
    .unwrap();
    node.handle(&p);
    let e = request(
        2,
        "exec",
        json!({
            "handle": "twice",
            "inputs": {"X": inline_spec(0, 8)},
            "outputs": ["OUT"],
            "threads": 4,
            "profile_ops": true
        }),
        &i64_bytes(21),
    )
    .unwrap();
    let r1 = node.handle(&e);
    let r2 = node.handle(&e);
    let (h1, _) = decode_frame(&r1).unwrap();
    let (h2, _) = decode_frame(&r2).unwrap();
    // Timing lives outside the receipt; verdict bytes do not.
    assert_eq!(h1["receipt"], h2["receipt"]);
    assert_eq!(h1["receipt"]["threads"], 4);
    assert_eq!(h1["receipt"]["kernel"], "lattice-threaded");
    let a = serde_json::to_vec(&h1["receipt"]).unwrap();
    let b = serde_json::to_vec(&h2["receipt"]).unwrap();
    assert_eq!(a, b, "receipt serialization must be canonical");
    let text = String::from_utf8(a).unwrap();
    for forbidden in ["wall", "elapsed", "alloc_bytes", "out_bytes"] {
        assert!(
            !text.contains(forbidden),
            "receipt leaked {forbidden}: {text}"
        );
    }
}
