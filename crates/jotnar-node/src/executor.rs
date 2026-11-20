//! Executor contract shared by the CPU node and GPU runner controllers.
//!
//! The wire contract is the node job API:
//! `submit`, `job-status`, `job-result`, and `job-cancel`. The CPU daemon and
//! the GPU runner both implement it. A controller routes a stage to a remote
//! executor by pairing its discovery capability with a [`RemoteExecutor`].
//!
//! This module intentionally contains no placement, model, or workload logic.

use serde_json::Value as Json;

use crate::job::JobClient;

/// One remote executor endpoint that speaks the common job contract.
pub trait Executor {
    /// Discovery record that selected this executor.
    fn capability(&self) -> &Json;

    /// Submit a job header and optional payload bytes.
    fn submit(&self, spec: &Json, payload: &[u8]) -> Result<Json, (String, String)>;

    /// Poll one job.
    fn status(&self, job_id: &str) -> Result<Json, (String, String)>;

    /// Fetch one completed job's result frame and payload bytes.
    fn result(&self, job_id: &str) -> Result<(Json, Vec<u8>), (String, String)>;

    /// Cancel a queued job; running jobs may reject cancellation.
    fn cancel(&self, job_id: &str) -> Result<Json, (String, String)>;

    /// Live executor view. Reporting only: the controller plans, the executor
    /// answers. Executors without a live view return an error.
    fn telemetry(&self) -> Result<Json, (String, String)> {
        Err((
            "unsupported".to_string(),
            "executor has no telemetry".to_string(),
        ))
    }
}

/// A controller-side handle for one advertised CPU or GPU executor.
pub struct RemoteExecutor {
    capability: Json,
    client: JobClient,
}

impl RemoteExecutor {
    pub fn new(capability: Json, endpoint: impl Into<String>, token: Option<String>) -> Self {
        Self {
            capability,
            client: JobClient::new(endpoint, token),
        }
    }

    pub fn endpoint(&self) -> &str {
        &self.client.endpoint
    }
}

impl Executor for RemoteExecutor {
    fn capability(&self) -> &Json {
        &self.capability
    }

    fn submit(&self, spec: &Json, payload: &[u8]) -> Result<Json, (String, String)> {
        self.client
            .submit_frame(spec, payload)
            .map(|(header, _)| header)
    }

    fn status(&self, job_id: &str) -> Result<Json, (String, String)> {
        self.client.status(job_id)
    }

    fn result(&self, job_id: &str) -> Result<(Json, Vec<u8>), (String, String)> {
        self.client.result(job_id)
    }

    fn cancel(&self, job_id: &str) -> Result<Json, (String, String)> {
        self.client.cancel(job_id)
    }

    fn telemetry(&self) -> Result<Json, (String, String)> {
        self.client.health()
    }
}

/// Validate binary result descriptors against their payload.
///
/// A controller should call this before trusting a result header from a remote
/// executor. It checks bounds, frame indices, known kinds, shape products, and
/// element widths.
pub fn validate_binary_result(header: &Json, body: &[u8]) -> Result<(), String> {
    let frames = header.get("frames").and_then(|x| x.as_u64()).unwrap_or(1) as usize;
    if frames == 0 {
        return Err("result has zero frames".to_string());
    }
    let descs = header
        .get("outputs")
        .and_then(|x| x.as_array())
        .ok_or("binary result has no outputs")?;
    for d in descs {
        let frame = d.get("frame").and_then(|x| x.as_u64()).unwrap_or(0) as usize;
        if frame >= frames {
            return Err(format!("descriptor frame {frame} exceeds {frames}"));
        }
        let name = d
            .get("name")
            .and_then(|x| x.as_str())
            .ok_or("descriptor has no name")?;
        let kind = d
            .get("kind")
            .and_then(|x| x.as_str())
            .ok_or_else(|| format!("descriptor {name:?} has no kind"))?;
        let width = match kind {
            "F32" => 4usize,
            "F64" | "I64" => 8,
            "U8" | "Bool" => 1,
            other => return Err(format!("descriptor {name:?} has unknown kind {other:?}")),
        };
        let shape = d
            .get("shape")
            .and_then(|x| x.as_array())
            .ok_or_else(|| format!("descriptor {name:?} has no shape"))?;
        let mut values = 1usize;
        for dim in shape {
            let dim = dim
                .as_u64()
                .ok_or_else(|| format!("descriptor {name:?} has non-integer shape"))?
                as usize;
            values = values
                .checked_mul(dim)
                .ok_or_else(|| format!("descriptor {name:?} shape overflows"))?;
        }
        let expected = values
            .checked_mul(width)
            .ok_or_else(|| format!("descriptor {name:?} byte count overflows"))?;
        let len = d
            .get("bytes")
            .and_then(|x| x.as_u64())
            .ok_or_else(|| format!("descriptor {name:?} has no byte count"))?
            as usize;
        if len != expected {
            return Err(format!(
                "descriptor {name:?} shape needs {expected} bytes, header says {len}"
            ));
        }
        let off = d.get("off").and_then(|x| x.as_u64()).unwrap_or(0) as usize;
        let end = off
            .checked_add(len)
            .ok_or_else(|| format!("descriptor {name:?} range overflows"))?;
        if end > body.len() {
            return Err(format!(
                "descriptor {name:?} range {off}..{end} exceeds body {}",
                body.len()
            ));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn validates_binary_descriptors() {
        let header = json!({
            "frames": 1,
            "outputs": [{"frame": 0, "name": "OUT", "kind": "F32",
                         "shape": [2], "off": 0, "bytes": 8}],
        });
        validate_binary_result(&header, &[0u8; 8]).unwrap();
        let short = json!({
            "frames": 1,
            "outputs": [{"frame": 0, "name": "OUT", "kind": "F32",
                         "shape": [2], "off": 0, "bytes": 4}],
        });
        assert!(validate_binary_result(&short, &[0u8; 8]).is_err());
        let truncated = json!({
            "frames": 1,
            "outputs": [{"frame": 0, "name": "OUT", "kind": "F32",
                         "shape": [2], "off": 4, "bytes": 8}],
        });
        assert!(validate_binary_result(&truncated, &[0u8; 8]).is_err());
        let bad_frame = json!({
            "frames": 1,
            "outputs": [{"frame": 1, "name": "OUT", "kind": "F32",
                         "shape": [2], "off": 0, "bytes": 8}],
        });
        assert!(validate_binary_result(&bad_frame, &[0u8; 8]).is_err());
    }
}
