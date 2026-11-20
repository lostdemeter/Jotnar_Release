//! Small framed client for the local job API used by the capability shim.
//!
//! The capability layer never re-implements job execution: it submits to the
//! same `jotnar-noded` endpoint the controller uses. Program and blob names are
//! opaque to the node; the module sees only the capability operation.
//!
//! A client keeps one TCP session and reuses it for submit/status/result. The
//! executors close a session only on EOF or a malformed request, so a
//! controller can avoid a fresh handshake for every poll. Submit is not
//! retried automatically (that could duplicate a job); status/result are
//! retried once on a stale session.

use std::io::{Read, Write};
use std::net::{TcpStream, ToSocketAddrs};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use jotnar::protocol::{decode_frame, request};
use serde_json::{json, Value as Json};

#[derive(Clone, Debug)]
pub struct JobClient {
    pub endpoint: String,
    pub token: Option<String>,
    session: Arc<Mutex<Option<TcpStream>>>,
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
            session: Arc::new(Mutex::new(None)),
        }
    }

    fn call(&self, typ: &str, fields: Json, payload: &[u8]) -> Result<Json, (String, String)> {
        self.call_frame(typ, fields, payload)
            .map(|(header, _)| header)
    }

    fn call_retry(
        &self,
        typ: &str,
        fields: Json,
        payload: &[u8],
    ) -> Result<Json, (String, String)> {
        match self.call(typ, fields.clone(), payload) {
            Ok(value) => Ok(value),
            Err(_) => self.call(typ, fields, payload),
        }
    }

    fn call_frame(
        &self,
        typ: &str,
        fields: Json,
        payload: &[u8],
    ) -> Result<(Json, Vec<u8>), (String, String)> {
        self.call_frame_with(
            typ,
            fields,
            payload,
            Duration::from_secs(60),
            Duration::from_secs(60),
        )
    }

    /// One request with explicit read/write deadlines.
    ///
    /// A persistent session is reused, but the socket timeouts are set before
    /// every exchange. A blackholed peer therefore fails in `timeout`, not in
    /// the 60 s default.
    pub fn call_frame_with(
        &self,
        typ: &str,
        fields: Json,
        payload: &[u8],
        read_timeout: Duration,
        write_timeout: Duration,
    ) -> Result<(Json, Vec<u8>), (String, String)> {
        let outcome = {
            let mut guard = self.session.lock().unwrap_or_else(|e| e.into_inner());
            if guard.is_none() {
                let stream = connect_with_timeout(&self.endpoint, Duration::from_secs(5))
                    .map_err(|e| ("job-unavailable".to_string(), e))?;
                let _ = stream.set_nodelay(true);
                *guard = Some(stream);
            }
            let stream = guard.as_mut().expect("session was just established");
            let _ = stream.set_read_timeout(Some(read_timeout));
            let _ = stream.set_write_timeout(Some(write_timeout));
            exchange(stream, &self.token, typ, fields, payload)
        };
        if outcome.is_err() {
            let mut guard = self.session.lock().unwrap_or_else(|e| e.into_inner());
            *guard = None;
        }
        outcome
    }

    fn call_frame_retry(
        &self,
        typ: &str,
        fields: Json,
        payload: &[u8],
    ) -> Result<(Json, Vec<u8>), (String, String)> {
        match self.call_frame(typ, fields.clone(), payload) {
            Ok(value) => Ok(value),
            Err(_) => self.call_frame(typ, fields, payload),
        }
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

    /// Submit with a bounded response deadline. Used by interactive streams so
    /// a blackholed peer cannot hold a frame for the 60 s default.
    pub fn submit_frame_with_timeout(
        &self,
        args: &Json,
        payload: &[u8],
        timeout: Duration,
    ) -> Result<(Json, Vec<u8>), (String, String)> {
        self.call_frame_with("submit", args.clone(), payload, timeout, timeout)
    }

    pub fn status(&self, job_id: &str) -> Result<Json, (String, String)> {
        self.call_retry("job-status", json!({"job_id": job_id}), &[])
    }

    pub fn status_with_timeout(
        &self,
        job_id: &str,
        timeout: Duration,
    ) -> Result<Json, (String, String)> {
        self.call_frame_with(
            "job-status",
            json!({"job_id": job_id}),
            &[],
            timeout,
            timeout,
        )
        .map(|(header, _)| header)
    }

    pub fn cancel(&self, job_id: &str) -> Result<Json, (String, String)> {
        self.call_retry("job-cancel", json!({"job_id": job_id}), &[])
    }

    pub fn result(&self, job_id: &str) -> Result<(Json, Vec<u8>), (String, String)> {
        self.call_frame_retry("job-result", json!({"job_id": job_id}), &[])
    }

    pub fn result_with_timeout(
        &self,
        job_id: &str,
        timeout: Duration,
    ) -> Result<(Json, Vec<u8>), (String, String)> {
        self.call_frame_with(
            "job-result",
            json!({"job_id": job_id}),
            &[],
            timeout,
            timeout,
        )
    }

    pub fn list(&self) -> Result<Json, (String, String)> {
        self.call_retry("job-list", json!({}), &[])
    }

    pub fn ping(&self) -> Result<Json, (String, String)> {
        self.call_retry("ping", json!({}), &[])
    }

    pub fn ping_with_timeout(&self, timeout: Duration) -> Result<Json, (String, String)> {
        self.call_frame_with("ping", json!({}), &[], timeout, timeout)
            .map(|(header, _)| header)
    }

    /// Live executor view: queue, running jobs, warm workers, recent jobs.
    ///
    /// `health` is a request that reports; it carries no placement policy and
    /// is not part of discovery. Executors that do not implement it return an
    /// error, which the controller records and treats as unknown.
    pub fn health(&self) -> Result<Json, (String, String)> {
        self.call_retry("health", json!({}), &[])
    }
}

fn exchange(
    stream: &mut TcpStream,
    token: &Option<String>,
    typ: &str,
    fields: Json,
    payload: &[u8],
) -> Result<(Json, Vec<u8>), (String, String)> {
    let mut fields = fields;
    if let Some(token) = token {
        if let Some(o) = fields.as_object_mut() {
            o.insert("auth".into(), json!(token));
        }
    }
    let frame =
        request(1, typ, fields, payload).map_err(|e| ("job-invalid".to_string(), e.to_string()))?;
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
