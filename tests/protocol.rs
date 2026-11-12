use std::collections::BTreeMap;
use std::thread;

use serde_json::json;

use jotnar::protocol::{
    decode_frame, decode_stream, encode_frame, encode_stream, exec_outputs, request, Node,
    NodeHandle,
};
use jotnar::value::Value;

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
    let (eh, _) = decode_frame(&r1).unwrap();
    assert_eq!(
        eh["runtime_version"],
        format!("{}+{}", env!("CARGO_PKG_VERSION"), jotnar::BUILD_GIT_HASH)
    );
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

#[test]
fn ping_reports_live_registry_count_and_build_identity() {
    let mut node = Node::new(vec!["lattice".into()]);
    let f = request(1, "ping", json!({}), &[]).unwrap();
    let (h, _) = decode_frame(&node.handle(&f)).unwrap();
    assert_eq!(h["pong"]["mnemonics"], jotnar::ops::registry().len());
    assert_eq!(
        h["pong"]["runtime_version"],
        format!("{}+{}", env!("CARGO_PKG_VERSION"), jotnar::BUILD_GIT_HASH)
    );
}

#[test]
fn frozen_t_wire_is_interleaved_and_length_checked() {
    let frozen = jotnar::Triple::from_vecs(
        &[2, 3],
        vec![1, -1, 1, -1, 1, 1],
        vec![100, 200, 300, 400, 500, 600],
        vec![0, 0, 0, 0, 0, 0],
    )
    .unwrap();
    let (spec, bytes) = encode_stream(&Value::Triple(frozen.clone())).unwrap();
    assert_eq!(spec.dtype, "i64x3");
    assert_eq!(bytes.len(), 6 * 24);

    for dtype in ["", "i64", "i64x3", "mystery"] {
        let mut s = spec.clone();
        s.dtype = dtype.to_string();
        let back = match decode_stream(&s, &bytes).unwrap() {
            Value::Triple(t) => t,
            _ => panic!("not a triple"),
        };
        assert_eq!(back.s, frozen.s, "dtype {dtype:?} sign");
        assert_eq!(back.e, frozen.e, "dtype {dtype:?} exponent");
        assert_eq!(back.z, frozen.z, "dtype {dtype:?} zero");
        assert!(back.m.is_none(), "frozen T gained a mantissa");
    }

    let mut unknown = spec.clone();
    unknown.dtype = "mystery".into();
    assert!(decode_stream(&unknown, &bytes[..bytes.len() - 1]).is_err());
}
