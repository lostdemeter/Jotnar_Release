use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};

use serde_json::json;

use jotnar::protocol::{decode_frame, encode_stream, exec_outputs, request, Node};
use jotnar::value::Value;

const VEC: &str = "IN X\nIN Y\nOUT = IADD(X, Y)\n";

fn read_frame(stream: &mut TcpStream) -> Vec<u8> {
    let mut hdr = [0u8; 4];
    stream.read_exact(&mut hdr).unwrap();
    let n = u32::from_be_bytes(hdr) as usize;
    let mut body = vec![0u8; n];
    stream.read_exact(&mut body).unwrap();
    let mut full = hdr.to_vec();
    full.extend_from_slice(&body);
    full
}

/// One framed round trip over loopback TCP must preserve bytes exactly,
/// the same contract the in-process channel tests rely on.
#[test]
fn tcp_frames_preserve_bytes_for_large_payload() {
    let n = 4096usize;
    let xs: Vec<i64> = (0..n as i64).map(|i| i).collect();
    let ys: Vec<i64> = (0..n as i64).map(|i| 1000 + i).collect();
    let mut expected = Vec::with_capacity(n * 8);
    for (x, y) in xs.iter().zip(ys.iter()) {
        expected.extend_from_slice(&(x + y).to_le_bytes());
    }

    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap().to_string();
    let server = std::thread::spawn(move || {
        let (mut stream, _) = listener.accept().unwrap();
        let _ = stream.set_nodelay(true);
        let mut node = Node::new(vec!["lattice".into()]);
        loop {
            let frame = {
                let mut hdr = [0u8; 4];
                if stream.read_exact(&mut hdr).is_err() {
                    break;
                }
                let len = u32::from_be_bytes(hdr) as usize;
                let mut body = vec![0u8; len];
                if stream.read_exact(&mut body).is_err() {
                    break;
                }
                let mut f = hdr.to_vec();
                f.extend_from_slice(&body);
                f
            };
            let out = node.handle(&frame);
            if stream.write_all(&out).is_err() {
                break;
            }
        }
    });

    let mut client = TcpStream::connect(&addr).unwrap();
    let _ = client.set_nodelay(true);
    let prog = request(
        1,
        "program",
        json!({"name": "vec", "backend": "lattice"}),
        VEC.as_bytes(),
    )
    .unwrap();
    client.write_all(&prog).unwrap();
    let (h, _) = decode_frame(&read_frame(&mut client)).unwrap();
    assert_eq!(h["ok"], true);

    let xv = Value::I64(jotnar::value::array_from_vec(&[n], xs).unwrap());
    let yv = Value::I64(jotnar::value::array_from_vec(&[n], ys).unwrap());
    let (sx, bx) = encode_stream(&xv).unwrap();
    let (sy, by) = encode_stream(&yv).unwrap();
    let mut payload = Vec::new();
    payload.extend_from_slice(&bx);
    payload.extend_from_slice(&by);
    let fields = json!({"handle": "vec",
        "inputs": {
            "X": {"kind": sx.kind, "dtype": sx.dtype, "shape": sx.shape, "off": 0, "len": bx.len()},
            "Y": {"kind": sy.kind, "dtype": sy.dtype, "shape": sy.shape, "off": bx.len(), "len": by.len()},
        },
        "outputs": ["OUT"]});
    let frame = request(2, "exec", fields, &payload).unwrap();
    client.write_all(&frame).unwrap();
    let resp = read_frame(&mut client);
    let out = exec_outputs(&resp).unwrap();
    assert_eq!(out["OUT"], expected);
    drop(client);
    let _ = server.join();
}
