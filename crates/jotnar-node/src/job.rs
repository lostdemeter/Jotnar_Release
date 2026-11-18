//! Small framed client for the local job API used by the capability shim.
//!
//! The capability layer never re-implements job execution: it submits to the
//! same `jotnar-noded` endpoint the controller uses. Program and blob names are
//! opaque to the node; the module sees only the capability operation.

use std::io::{Read, Write};
use std::net::{TcpStream, ToSocketAddrs};
use std::time::Duration;

use jotnar::protocol::{decode_frame, request};
use serde_json::{json, Value as Json};

#[derive(Clone, Debug)]
pub struct JobClient {
    pub endpoint: String,
    pub token: Option<String>,
}

fn connect_with_timeout(endpoint: &str, timeout: Duration) -> Result<TcpStream, String> {
    let addrs = endpoint
        .to_socket_addrs()
        .map_err(|e| format!("resolve {endpoint}: {e}"))?;
    let mut last = None;
    for addr in addrs {
        match TcpStream::connect_timeout(&addr, timeout) {
            Ok(stream) => return Ok(stream),
            Err(e) => last = Some(e),
        }
    }
    Err(format!(
        "connect {endpoint}: {}",
        last.map(|e| e.to_string())
            .unwrap_or_else(|| "no addresses".to_string())
    ))
}

impl JobClient {
    pub fn new(endpoint: impl Into<String>, token: Option<String>) -> Self {
        Self {
            endpoint: endpoint.into(),
            token,
        }
    }

    fn call(&self, typ: &str, fields: Json, payload: &[u8]) -> Result<Json, (String, String)> {
        self.call_frame(typ, fields, payload)
            .map(|(header, _)| header)
    }

    fn call_frame(
        &self,
        typ: &str,
        fields: Json,
        payload: &[u8],
    ) -> Result<(Json, Vec<u8>), (String, String)> {
        let mut stream = connect_with_timeout(&self.endpoint, Duration::from_secs(5))
            .map_err(|e| ("job-unavailable".to_string(), e))?;
        let _ = stream.set_nodelay(true);
        let _ = stream.set_read_timeout(Some(Duration::from_secs(60)));
        let _ = stream.set_write_timeout(Some(Duration::from_secs(60)));
        let mut fields = fields;
        if let Some(token) = &self.token {
            if let Some(o) = fields.as_object_mut() {
                o.insert("auth".into(), json!(token));
            }
        }
        let frame = request(1, typ, fields, payload)
            .map_err(|e| ("job-invalid".to_string(), e.to_string()))?;
        stream
            .write_all(&frame)
            .map_err(|e| ("job-unavailable".to_string(), e.to_string()))?;
        let mut hdr = [0u8; 4];
        stream
            .read_exact(&mut hdr)
            .map_err(|e| ("job-unavailable".to_string(), e.to_string()))?;
        let n = u32::from_be_bytes(hdr) as usize;
        if n > 1024 * 1024 * 1024 {
            return Err(("job-invalid".to_string(), "response too large".to_string()));
        }
        let mut body = vec![0u8; n];
        stream
            .read_exact(&mut body)
            .map_err(|e| ("job-unavailable".to_string(), e.to_string()))?;
        let mut full = Vec::with_capacity(4 + n);
        full.extend_from_slice(&hdr);
        full.extend_from_slice(&body);
        let (header, response) =
            decode_frame(&full).map_err(|e| ("job-invalid".to_string(), e.to_string()))?;
        if header.get("ok").and_then(|x| x.as_bool()) != Some(true) {
            let err = header.get("error").cloned().unwrap_or(json!({}));
            return Err((
                err.get("code")
                    .and_then(|x| x.as_str())
                    .unwrap_or("job-error")
                    .to_string(),
                err.get("msg")
                    .and_then(|x| x.as_str())
                    .unwrap_or("job endpoint rejected the request")
                    .to_string(),
            ));
        }
        Ok((header, response.to_vec()))
    }

    pub fn submit(&self, args: &Json) -> Result<Json, (String, String)> {
        self.call("submit", args.clone(), &[])
    }

    pub fn submit_frame(
        &self,
        args: &Json,
        payload: &[u8],
    ) -> Result<(Json, Vec<u8>), (String, String)> {
        self.call_frame("submit", args.clone(), payload)
    }

    pub fn status(&self, job_id: &str) -> Result<Json, (String, String)> {
        self.call("job-status", json!({"job_id": job_id}), &[])
    }

    pub fn cancel(&self, job_id: &str) -> Result<Json, (String, String)> {
        self.call("job-cancel", json!({"job_id": job_id}), &[])
    }

    pub fn result(&self, job_id: &str) -> Result<(Json, Vec<u8>), (String, String)> {
        self.call_frame("job-result", json!({"job_id": job_id}), &[])
    }

    pub fn list(&self) -> Result<Json, (String, String)> {
        self.call("job-list", json!({}), &[])
    }

    pub fn ping(&self) -> Result<Json, (String, String)> {
        self.call("ping", json!({}), &[])
    }
}
