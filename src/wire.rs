//! A borrowing JSON writer for protocol headers.
//!
//! `serde_json::to_writer` already avoids an intermediate `String`, but the
//! `json!` macro still builds a `Value` tree: one `String` per key and per
//! string value, plus a map node per object. For a steady-state EXEC reply
//! that tree is ~98 allocations.
//!
//! [`JsonOut`] writes the same bytes directly into a borrowed buffer. It is
//! deliberately small: objects, arrays, strings, integers, booleans, nulls, a
//! 16-digit hex string, and an escape hatch that hands an existing `Json`
//! value to `serde_json::to_writer`.
//!
//! Byte-compatibility with `serde_json` is not assumed, it is tested: object
//! keys are emitted by the caller in sorted order (matching `serde_json::Map`,
//! which is a `BTreeMap`), and [`crate::wire::tests`] asserts that re-parsing
//! our output and re-serializing it with `serde_json` yields the same bytes.

use serde_json::Value as Json;

use crate::error::{Error, Result};

/// One frame under construction: `[u32 body_len][header]\n[payload]`.
///
/// The length is patched last, so the header can be streamed without knowing
/// its size up front.
pub struct FrameBuf<'a> {
    out: &'a mut Vec<u8>,
}

impl<'a> FrameBuf<'a> {
    pub fn begin(out: &'a mut Vec<u8>) -> Self {
        out.clear();
        out.extend_from_slice(&[0u8; 4]);
        Self { out }
    }

    /// Borrow the buffer as a streaming JSON writer for the header.
    pub fn json(&mut self) -> JsonOut<'_> {
        JsonOut::new(self.out)
    }

    pub fn end(self, payload: &[u8]) -> Result<()> {
        self.out.push(b'\n');
        self.out.extend_from_slice(payload);
        let body = self.out.len() - 4;
        if body > u32::MAX as usize {
            return Err(Error::runtime(format!(
                "wire: frame body {body} exceeds u32"
            )));
        }
        self.out[..4].copy_from_slice(&(body as u32).to_be_bytes());
        Ok(())
    }
}

const DEPTH: usize = 16;

pub struct JsonOut<'a> {
    out: &'a mut Vec<u8>,
    /// For each open container: true while it is still empty.
    first: [bool; DEPTH],
    depth: usize,
    /// A key was just written; the next value takes no separator.
    after_key: bool,
}

impl<'a> JsonOut<'a> {
    pub fn new(out: &'a mut Vec<u8>) -> Self {
        Self {
            out,
            first: [true; DEPTH],
            depth: 0,
            after_key: false,
        }
    }

    pub fn finish(self) {}

    fn item(&mut self) {
        if self.after_key {
            self.after_key = false;
            return;
        }
        if self.depth == 0 {
            return;
        }
        let top = self.depth - 1;
        if self.first[top] {
            self.first[top] = false;
        } else {
            self.out.push(b',');
        }
    }

    fn push_open(&mut self, c: u8) {
        self.item();
        self.out.push(c);
        debug_assert!(self.depth < DEPTH, "json depth");
        self.first[self.depth] = true;
        self.depth += 1;
    }

    fn push_close(&mut self, c: u8) {
        self.depth -= 1;
        self.out.push(c);
    }

    pub fn begin_object(&mut self) {
        self.push_open(b'{');
    }

    pub fn end_object(&mut self) {
        self.push_close(b'}');
    }

    pub fn begin_array(&mut self) {
        self.push_open(b'[');
    }

    pub fn end_array(&mut self) {
        self.push_close(b']');
    }

    pub fn key(&mut self, k: &str) {
        self.item();
        escape_into(self.out, k);
        self.out.push(b':');
        self.after_key = true;
    }

    pub fn string(&mut self, v: &str) {
        self.item();
        escape_into(self.out, v);
    }

    pub fn uint(&mut self, v: u64) {
        self.item();
        let mut buf = itoa_buf();
        self.out.extend_from_slice(uint_str(&mut buf, v));
    }

    pub fn int(&mut self, v: i64) {
        self.item();
        let mut buf = itoa_buf();
        self.out.extend_from_slice(int_str(&mut buf, v));
    }

    pub fn boolean(&mut self, v: bool) {
        self.item();
        self.out
            .extend_from_slice(if v { b"true" } else { b"false" });
    }

    pub fn null(&mut self) {
        self.item();
        self.out.extend_from_slice(b"null");
    }

    /// A lowercase, zero-padded 16-digit hex string, matching
    /// `format!("{:016x}")`.
    pub fn hex16(&mut self, v: u64) {
        self.item();
        self.out.push(b'"');
        const HEX: &[u8; 16] = b"0123456789abcdef";
        let mut buf = [0u8; 16];
        for (i, b) in buf.iter_mut().enumerate() {
            *b = HEX[((v >> (4 * (15 - i))) & 0xF) as usize];
        }
        self.out.extend_from_slice(&buf);
        self.out.push(b'"');
    }

    pub fn kv_str(&mut self, k: &str, v: &str) {
        self.key(k);
        self.string(v);
    }

    pub fn kv_uint(&mut self, k: &str, v: u64) {
        self.key(k);
        self.uint(v);
    }

    pub fn kv_int(&mut self, k: &str, v: i64) {
        self.key(k);
        self.int(v);
    }

    pub fn kv_bool(&mut self, k: &str, v: bool) {
        self.key(k);
        self.boolean(v);
    }

    pub fn kv_hex16(&mut self, k: &str, v: u64) {
        self.key(k);
        self.hex16(v);
    }

    /// Write an existing `Json` value without building a new tree.
    pub fn value(&mut self, v: &Json) -> Result<()> {
        self.item();
        serde_json::to_writer(&mut *self.out, v)
            .map_err(|e| Error::runtime(format!("wire: json value: {e}")))
    }

    pub fn kv_value(&mut self, k: &str, v: &Json) -> Result<()> {
        self.key(k);
        self.value(v)
    }
}

fn escape_into(out: &mut Vec<u8>, s: &str) {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    out.push(b'"');
    for &b in s.as_bytes() {
        match b {
            b'"' => out.extend_from_slice(b"\\\""),
            b'\\' => out.extend_from_slice(b"\\\\"),
            0x08 => out.extend_from_slice(b"\\b"),
            0x0C => out.extend_from_slice(b"\\f"),
            b'\n' => out.extend_from_slice(b"\\n"),
            b'\r' => out.extend_from_slice(b"\\r"),
            b'\t' => out.extend_from_slice(b"\\t"),
            0x00..=0x1F => {
                out.extend_from_slice(b"\\u00");
                out.push(HEX[(b >> 4) as usize]);
                out.push(HEX[(b & 0xF) as usize]);
            }
            _ => out.push(b),
        }
    }
    out.push(b'"');
}

fn itoa_buf() -> [u8; 20] {
    [0u8; 20]
}

fn uint_str<'b>(buf: &'b mut [u8; 20], mut v: u64) -> &'b [u8] {
    let mut i = buf.len();
    loop {
        i -= 1;
        buf[i] = b'0' + (v % 10) as u8;
        v /= 10;
        if v == 0 {
            break;
        }
    }
    &buf[i..]
}

fn int_str<'b>(buf: &'b mut [u8; 20], v: i64) -> &'b [u8] {
    if v >= 0 {
        return uint_str(buf, v as u64);
    }
    // Write magnitude into the tail, then prepend '-'.
    let mag = (v as i128).unsigned_abs() as u64;
    let mut i = buf.len();
    let mut m = mag;
    loop {
        i -= 1;
        buf[i] = b'0' + (m % 10) as u8;
        m /= 10;
        if m == 0 {
            break;
        }
    }
    i -= 1;
    buf[i] = b'-';
    &buf[i..]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rendered(f: impl FnOnce(&mut JsonOut)) -> String {
        let mut out = Vec::new();
        let mut w = JsonOut::new(&mut out);
        f(&mut w);
        w.finish();
        String::from_utf8(out).unwrap()
    }

    #[test]
    fn matches_serde_json_bytes() {
        let got = rendered(|w| {
            w.begin_object();
            w.kv_uint("a", 7);
            w.kv_str("b", "x\"y\\z\n\t\u{1}");
            w.kv_bool("c", true);
            w.key("d");
            w.begin_array();
            w.uint(1);
            w.int(-2);
            w.hex16(0xDEAD_BEEF);
            w.end_array();
            w.key("e");
            w.begin_object();
            w.kv_int("z", i64::MIN);
            w.end_object();
            w.key("f");
            w.null();
            w.end_object();
        });
        let want = serde_json::to_string(&serde_json::json!({
            "a": 7,
            "b": "x\"y\\z\n\t\u{1}",
            "c": true,
            "d": [1, -2, "00000000deadbeef"],
            "e": {"z": i64::MIN},
            "f": serde_json::Value::Null,
        }))
        .unwrap();
        assert_eq!(got, want);
    }

    #[test]
    fn value_roundtrip_matches_serde() {
        let v = serde_json::json!({"k": [1, 2, {"n": "s"}], "f": 0.5});
        let got = rendered(|w| {
            w.begin_object();
            w.kv_value("v", &v).unwrap();
            w.end_object();
        });
        let want = serde_json::to_string(&serde_json::json!({"v": v})).unwrap();
        assert_eq!(got, want);
    }

    #[test]
    fn escaping_covers_controls() {
        let got = rendered(|w| w.string("\u{0}\u{1f}\u{7f}\u{80}"));
        let want = serde_json::to_string("\u{0}\u{1f}\u{7f}\u{80}").unwrap();
        assert_eq!(got, want);
    }
}
