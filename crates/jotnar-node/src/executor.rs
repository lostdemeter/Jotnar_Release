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
}
