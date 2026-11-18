//! Framed client for `jotnar-gpu-runner`.

use std::io::{Read, Write};
use std::net::{TcpStream, ToSocketAddrs};
use std::time::Duration;

use jotnar::protocol::{decode_frame, encode_frame, PROTO_V};
use serde_json::{json, Value as Json};

#[derive(Clone, Debug)]
pub struct Runner {
    pub target: String,
    pub addr: String,
    pub token: Option<String>,
}

#[derive(Clone, Debug)]
pub struct RunReply {
    pub header: Json,
    pub outputs: Json,
}

#[derive(Clone, Debug)]
pub struct BatchReply {
    pub header: Json,
    pub frames: Vec<Json>,
}

fn send_frame(stream: &mut TcpStream, header: &Json, payload: &[u8]) -> Result<(), String> {
    let frame = encode_frame(header, payload).map_err(|e| e.to_string())?;
    stream.write_all(&frame).map_err(|e| format!("write: {e}"))
}

fn recv_frame(stream: &mut TcpStream, max_frame: usize) -> Result<(Json, Vec<u8>), String> {
    let mut hdr = [0u8; 4];
    stream
        .read_exact(&mut hdr)
        .map_err(|e| format!("read header: {e}"))?;
    let n = u32::from_be_bytes(hdr) as usize;
    if n > max_frame {
        return Err(format!("response {n} exceeds max frame {max_frame}"));
    }
    let mut body = vec![0u8; n];
    stream
        .read_exact(&mut body)
        .map_err(|e| format!("read body: {e}"))?;
    let mut frame = Vec::with_capacity(4 + n);
    frame.extend_from_slice(&hdr);
    frame.extend_from_slice(&body);
    let (header, payload) = decode_frame(&frame).map_err(|e| e.to_string())?;
    Ok((header, payload.to_vec()))
}

fn decode_values(kind: &str, shape: &Json, slice: &[u8]) -> Result<Json, String> {
    let data: Vec<Json> = match kind {
        "I64" => slice
            .chunks_exact(8)
            .map(|b| {
                Json::String(
                    i64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]])
                        .to_string(),
                )
            })
            .collect(),
        "F32" => slice
            .chunks_exact(4)
            .map(|b| Json::from(f32::from_le_bytes([b[0], b[1], b[2], b[3]]) as f64))
            .collect(),
        "F64" => slice
            .chunks_exact(8)
            .map(|b| {
                Json::from(f64::from_le_bytes([
                    b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7],
                ]))
            })
            .collect(),
        "U8" => slice.iter().map(|b| Json::from(*b)).collect(),
        "Bool" => slice.iter().map(|b| Json::Bool(*b != 0)).collect(),
        other => return Err(format!("unsupported binary kind {other}")),
    };
    Ok(json!({"kind": kind, "shape": shape, "data": data}))
}

fn decode_binary(header: &Json, body: &[u8]) -> Result<Json, String> {
    let descs = header
        .get("outputs")
        .and_then(|x| x.as_array())
        .ok_or("binary response has no outputs")?;
    let mut map = serde_json::Map::new();
    for d in descs {
        let name = d
            .get("name")
            .and_then(|x| x.as_str())
            .ok_or("binary output has no name")?;
        let kind = d
            .get("kind")
            .and_then(|x| x.as_str())
            .ok_or("binary output has no kind")?;
        let shape = d.get("shape").cloned().unwrap_or(Json::Null);
        let off = d.get("off").and_then(|x| x.as_u64()).unwrap_or(0) as usize;
        let len = d.get("bytes").and_then(|x| x.as_u64()).unwrap_or(0) as usize;
        let end = off.checked_add(len).ok_or("binary output range overflow")?;
        if end > body.len() {
            return Err("binary output truncated".to_string());
        }
        map.insert(
            name.to_string(),
            decode_values(kind, &shape, &body[off..end])?,
        );
    }
    Ok(json!({"outputs": map}))
}

fn decode_binary_frames(header: &Json, body: &[u8]) -> Result<Vec<Json>, String> {
    let n = header
        .get("frames")
        .and_then(|x| x.as_u64())
        .ok_or("binary batch response has no frame count")? as usize;
    let descs = header
        .get("outputs")
        .and_then(|x| x.as_array())
        .ok_or("binary batch response has no outputs")?;
    let mut frames: Vec<serde_json::Map<String, Json>> =
        (0..n).map(|_| serde_json::Map::new()).collect();
    for d in descs {
        let frame = d.get("frame").and_then(|x| x.as_u64()).unwrap_or(0) as usize;
        if frame >= frames.len() {
            return Err(format!("binary batch frame {frame} exceeds count {n}"));
        }
        let name = d
            .get("name")
            .and_then(|x| x.as_str())
            .ok_or("binary batch output has no name")?;
        let kind = d
            .get("kind")
            .and_then(|x| x.as_str())
            .ok_or("binary batch output has no kind")?;
        let shape = d.get("shape").cloned().unwrap_or(Json::Null);
        let off = d.get("off").and_then(|x| x.as_u64()).unwrap_or(0) as usize;
        let len = d.get("bytes").and_then(|x| x.as_u64()).unwrap_or(0) as usize;
        let end = off.checked_add(len).ok_or("binary batch range overflow")?;
        if end > body.len() {
            return Err("binary batch output truncated".to_string());
        }
        frames[frame].insert(
            name.to_string(),
            decode_values(kind, &shape, &body[off..end])?,
        );
    }
    Ok(frames
        .into_iter()
        .map(|map| json!({"outputs": map}))
        .collect())
}

impl Runner {
    pub fn connect(&self) -> Result<TcpStream, String> {
        let addrs = self
            .addr
            .to_socket_addrs()
            .map_err(|e| format!("resolve {}: {e}", self.addr))?;
        let mut last_err = None;
        for addr in addrs {
            match TcpStream::connect_timeout(&addr, Duration::from_secs(3)) {
                Ok(stream) => {
                    let _ = stream.set_nodelay(true);
                    let _ = stream.set_read_timeout(Some(Duration::from_secs(600)));
                    let _ = stream.set_write_timeout(Some(Duration::from_secs(600)));
                    return Ok(stream);
                }
                Err(e) => last_err = Some(e),
            }
        }
        let err = last_err
            .map(|e| e.to_string())
            .unwrap_or_else(|| "no socket addresses".to_string());
        Err(format!("connect {}: {err}", self.addr))
    }

    pub fn ping(&self) -> Result<Json, String> {
        let mut stream = self.connect()?;
        let header = self.header("ping", json!({}));
        send_frame(&mut stream, &header, &[])?;
        let (reply, _) = recv_frame(&mut stream, 16 * 1024 * 1024)?;
        if reply.get("ok").and_then(|x| x.as_bool()) != Some(true) {
            return Err(format!("{} rejected ping: {reply}", self.addr));
        }
        Ok(reply)
    }

    pub fn put_payload(&self, payload: &Json) -> Result<String, String> {
        let mut stream = self.connect()?;
        let header = self.header("payload-put", json!({}));
        let body = serde_json::to_vec(payload).map_err(|e| e.to_string())?;
        send_frame(&mut stream, &header, &body)?;
        let (reply, _) = recv_frame(&mut stream, 16 * 1024 * 1024)?;
        if reply.get("ok").and_then(|x| x.as_bool()) != Some(true) {
            return Err(format!("{} payload-put failed: {reply}", self.addr));
        }
        reply
            .get("payload_id")
            .and_then(|x| x.as_str())
            .map(|s| s.to_string())
            .ok_or_else(|| format!("{} payload-put returned no payload_id", self.addr))
    }

    pub fn run(
        &self,
        listing: &str,
        payload: &Json,
        outputs: &[String],
    ) -> Result<RunReply, String> {
        let body = json!({"listing": listing, "payload": payload});
        self.run_body(
            listing,
            body,
            json!({"target": self.target, "outputs": outputs}),
        )
    }

    pub fn run_with_payload_id(
        &self,
        listing: &str,
        payload_id: &str,
        outputs: &[String],
    ) -> Result<RunReply, String> {
        let body = json!({"listing": listing, "payload_id": payload_id});
        self.run_body(
            listing,
            body,
            json!({
                "target": self.target,
                "outputs": outputs,
                "payload_id": payload_id,
                "output_format": "binary",
            }),
        )
    }

    pub fn run_batch(
        &self,
        listing: &str,
        payload_id: Option<&str>,
        payload: Option<&Json>,
        outputs: &[String],
        frames: usize,
    ) -> Result<BatchReply, String> {
        let mut fields = json!({
            "target": self.target,
            "outputs": outputs,
            "frames": frames,
            "output_format": "binary",
        });
        let mut body = json!({"listing": listing});
        match (payload_id, payload) {
            (Some(pid), _) => {
                if let Some(o) = fields.as_object_mut() {
                    o.insert("payload_id".into(), json!(pid));
                }
                if let Some(o) = body.as_object_mut() {
                    o.insert("payload_id".into(), json!(pid));
                }
            }
            (None, Some(payload)) => {
                if let Some(o) = body.as_object_mut() {
                    o.insert("payload".into(), payload.clone());
                }
            }
            (None, None) => return Err("run_batch needs a payload or payload_id".to_string()),
        }
        let mut stream = self.connect()?;
        let header = self.header("run-batch", fields);
        let body = serde_json::to_vec(&body).map_err(|e| e.to_string())?;
        send_frame(&mut stream, &header, &body)?;
        let (reply, body) = recv_frame(&mut stream, 2 * 1024 * 1024 * 1024)?;
        if reply.get("ok").and_then(|x| x.as_bool()) != Some(true) {
            return Err(format!("{} run-batch failed: {reply}", self.addr));
        }
        let frames_out = if reply.get("output_format").and_then(|x| x.as_str()) == Some("binary") {
            decode_binary_frames(&reply, &body)?
        } else {
            let parsed: Json =
                serde_json::from_slice(&body).map_err(|e| format!("bad batch body: {e}"))?;
            parsed
                .get("frames")
                .and_then(|x| x.as_array())
                .cloned()
                .ok_or("batch response has no frames")?
        };
        if frames_out.len() != frames {
            return Err(format!(
                "{} returned {} frames, expected {frames}",
                self.addr,
                frames_out.len()
            ));
        }
        Ok(BatchReply {
            header: reply,
            frames: frames_out,
        })
    }

    fn run_body(&self, _listing: &str, body: Json, fields: Json) -> Result<RunReply, String> {
        let mut stream = self.connect()?;
        let header = self.header("run", fields);
        let body = serde_json::to_vec(&body).map_err(|e| e.to_string())?;
        send_frame(&mut stream, &header, &body)?;
        let (reply, body) = recv_frame(&mut stream, 512 * 1024 * 1024)?;
        if reply.get("ok").and_then(|x| x.as_bool()) != Some(true) {
            return Err(format!("{} run failed: {reply}", self.addr));
        }
        let outputs = if reply.get("output_format").and_then(|x| x.as_str()) == Some("binary") {
            decode_binary(&reply, &body)?
        } else {
            serde_json::from_slice::<Json>(&body).map_err(|e| format!("bad run response: {e}"))?
        };
        Ok(RunReply {
            header: reply,
            outputs,
        })
    }

    fn header(&self, typ: &str, fields: Json) -> Json {
        let mut h = json!({
            "v": PROTO_V,
            "id": 1,
            "type": typ,
        });
        if let Some(o) = h.as_object_mut() {
            if let Some(f) = fields.as_object() {
                for (k, v) in f {
                    o.insert(k.clone(), v.clone());
                }
            }
            if let Some(t) = &self.token {
                o.insert("auth".into(), json!(t));
            }
        }
        h
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn binary_batch_decodes_frames() {
        let mut body = Vec::new();
        body.extend_from_slice(&1.5f32.to_le_bytes());
        body.extend_from_slice(&2.5f32.to_le_bytes());
        body.extend_from_slice(&9i64.to_le_bytes());
        let header = json!({
            "output_format": "binary",
            "frames": 2,
            "outputs": [
                {"frame": 0, "name": "f", "kind": "F32", "shape": [2], "off": 0, "bytes": 8},
                {"frame": 1, "name": "i", "kind": "I64", "shape": [1], "off": 8, "bytes": 8},
            ],
        });
        let frames = decode_binary_frames(&header, &body).unwrap();
        assert_eq!(frames.len(), 2);
        assert_eq!(frames[0]["outputs"]["f"]["data"], json!([1.5, 2.5]));
        assert_eq!(frames[1]["outputs"]["i"]["data"], json!(["9"]));
    }

    #[test]
    fn binary_output_decodes_f32_and_i64() {
        let mut body = Vec::new();
        body.extend_from_slice(&1.5f32.to_le_bytes());
        body.extend_from_slice(&2.5f32.to_le_bytes());
        body.extend_from_slice(&7i64.to_le_bytes());
        let header = json!({
            "output_format": "binary",
            "outputs": [
                {"name": "f", "kind": "F32", "shape": [2], "off": 0, "bytes": 8},
                {"name": "i", "kind": "I64", "shape": [1], "off": 8, "bytes": 8},
            ],
        });
        let out = decode_binary(&header, &body).unwrap();
        let f = &out["outputs"]["f"];
        assert_eq!(f["kind"], "F32");
        assert_eq!(f["data"], json!([1.5, 2.5]));
        let i = &out["outputs"]["i"];
        assert_eq!(i["kind"], "I64");
        assert_eq!(i["data"], json!(["7"]));
    }
}
