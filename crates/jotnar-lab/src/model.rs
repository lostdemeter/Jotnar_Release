use std::collections::BTreeMap;
use std::path::Path;

use serde::{Deserialize, Serialize};
use serde_json::Value as Json;
use sha2::{Digest, Sha256};

pub const SCHEMA: &str = "jotnar.certificate";
pub const SCHEMA_VERSION: u32 = 1;

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum Verdict {
    Pass,
    Fail,
    Bottom,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Outcome {
    Solved,
    Refused,
    Blocked,
    NotTested,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum Status {
    Current,
    Stale,
    Historical,
    Blocked,
    Superseded,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Artifact {
    pub role: String,
    pub path: String,
    pub sha256: String,
    pub bytes: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Sample {
    pub seed: u64,
    pub size: String,
    pub domain: String,
    #[serde(default)]
    pub forced_corners: Vec<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Space {
    pub label: String,
    pub map: String,
    pub family: String,
    pub primitive: String,
    pub sample: Sample,
    pub reference: String,
    #[serde(default)]
    pub bounds: Vec<String>,
    #[serde(default)]
    pub hashes: BTreeMap<String, String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Proposal {
    pub kind: String,
    pub candidate: String,
    pub provenance: BTreeMap<String, String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Gate {
    pub name: String,
    pub level: String,
    pub reference: String,
    pub frozen_at: String,
    pub gate_sha256: String,
    pub can_say_no: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Witness {
    pub map: String,
    pub input: String,
    pub output: String,
    pub trace: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Proof {
    pub gate: String,
    pub contract: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Derivation {
    pub trace: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Falsifier {
    pub corruption: String,
    pub expected: String,
    pub actual: String,
    pub caught_by: String,
    pub not_caught: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NullCheck {
    pub name: String,
    pub control: String,
    pub expected: String,
    pub actual: String,
    pub effect: String,
    pub status: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Bench {
    pub clock: String,
    pub scope: String,
    pub kind: String,
    pub runs: u32,
    pub p50_ms: Option<f64>,
    pub p99_ms: Option<f64>,
    pub max_ms: Option<f64>,
    pub units: String,
    pub note: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Refusal {
    pub code: String,
    pub message: String,
    #[serde(default)]
    pub missing: Vec<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Provenance {
    pub command: String,
    pub cwd: String,
    pub commit: String,
    pub host: String,
    #[serde(default)]
    pub environment: BTreeMap<String, String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Certificate {
    pub schema: String,
    pub schema_version: u32,
    pub id: String,
    pub status: Status,
    pub outcome: Outcome,
    pub verdict: Verdict,
    pub space: Space,
    pub proposal: Proposal,
    pub gate: Gate,
    pub witness: Witness,
    pub proof: Proof,
    pub derivation: Derivation,
    pub assumptions: Vec<String>,
    pub unknowns: Vec<String>,
    pub proves: Vec<String>,
    pub does_not_prove: Vec<String>,
    pub falsifiers: Vec<Falsifier>,
    pub nulls: Vec<NullCheck>,
    pub bench: Bench,
    pub refusal: Option<Refusal>,
    pub provenance: Provenance,
    pub artifacts: Vec<Artifact>,
    pub supersedes: Option<String>,
    pub superseded_by: Option<String>,
    pub certificate_sha256: Option<String>,
}

impl Certificate {
    /// Canonical JSON: serde_json's default map is key-sorted, and we
    /// zero the certificate hash so sealing is stable.
    pub fn canonical_json(&self) -> String {
        let mut v = serde_json::to_value(self).unwrap();
        if let Some(o) = v.as_object_mut() {
            o.remove("certificate_sha256");
        }
        serde_json::to_string(&v).unwrap()
    }

    pub fn compute_sha256(&self) -> String {
        let mut h = Sha256::new();
        h.update(self.canonical_json().as_bytes());
        format!("{:x}", h.finalize())
    }

    pub fn seal(&mut self) {
        self.certificate_sha256 = Some(self.compute_sha256());
    }

    /// Returns schema/contract violations. Empty means the certificate is
    /// well-formed; it does not mean the claim is true.
    pub fn validate(&self) -> Vec<String> {
        let mut e = Vec::new();
        if self.schema != SCHEMA {
            e.push(format!("schema must be {SCHEMA:?}"));
        }
        if self.schema_version != SCHEMA_VERSION {
            e.push(format!("schema_version must be {SCHEMA_VERSION}"));
        }
        if self.proves.is_empty() {
            e.push("proves must declare at least one claim".into());
        }
        if self.does_not_prove.is_empty() {
            e.push("does_not_prove is required (an empty list is an overclaim)".into());
        }
        if self.assumptions.is_empty() {
            e.push("assumptions must declare at least one assumption".into());
        }
        if self.gate.gate_sha256.is_empty() {
            e.push("gate.gate_sha256 is required".into());
        }
        if !self.gate.can_say_no {
            e.push("gate.can_say_no must be true; a vacuous gate is not a gate".into());
        }
        if self.falsifiers.is_empty() {
            e.push("falsifiers must include at least one row (not_tested is allowed)".into());
        }
        if self.nulls.is_empty() {
            e.push("nulls must include at least one control (not_tested is allowed)".into());
        }
        if self.bench.clock.is_empty() || self.bench.scope.is_empty() || self.bench.kind.is_empty()
        {
            e.push("bench needs clock, scope, and kind labels".into());
        }
        match self.outcome {
            Outcome::Refused if self.refusal.is_none() => {
                e.push("outcome refused requires a refusal object".into())
            }
            Outcome::Solved if self.refusal.is_some() => {
                e.push("outcome solved must not carry a refusal".into())
            }
            _ => {}
        }
        if self.status == Status::Superseded && self.superseded_by.is_none() {
            e.push("status superseded requires superseded_by".into());
        }
        if let Some(h) = &self.certificate_sha256 {
            if h != &self.compute_sha256() {
                e.push("certificate_sha256 does not match canonical body".into());
            }
        }
        e
    }

    pub fn write(&self, path: &Path) -> std::io::Result<()> {
        if let Some(p) = path.parent() {
            std::fs::create_dir_all(p)?;
        }
        let mut s = serde_json::to_string_pretty(self).unwrap();
        s.push('\n');
        std::fs::write(path, s)
    }

    pub fn read(path: &Path) -> Result<Self, String> {
        let text = std::fs::read_to_string(path).map_err(|e| e.to_string())?;
        serde_json::from_str(&text).map_err(|e| e.to_string())
    }
}

pub fn sha256_file(path: &Path) -> Result<(String, u64), String> {
    let bytes = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut h = Sha256::new();
    h.update(&bytes);
    Ok((format!("{:x}", h.finalize()), bytes.len() as u64))
}

pub fn sha256_str(s: &str) -> String {
    let mut h = Sha256::new();
    h.update(s.as_bytes());
    format!("{:x}", h.finalize())
}

pub fn artifact(role: &str, path: &Path) -> Result<Artifact, String> {
    let (sha256, bytes) = sha256_file(path)?;
    Ok(Artifact {
        role: role.to_string(),
        path: path.display().to_string(),
        sha256,
        bytes,
    })
}

pub fn now_iso() -> String {
    // A deterministic-enough UTC timestamp from the system clock. The
    // certificate hash is what matters; wall time is provenance.
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    format!("unix:{secs}")
}

pub fn hostname() -> String {
    std::env::var("HOSTNAME")
        .ok()
        .or_else(|| {
            std::fs::read_to_string("/etc/hostname")
                .ok()
                .map(|s| s.trim().to_string())
        })
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "unknown".into())
}

pub fn json_map(pairs: &[(&str, &str)]) -> BTreeMap<String, String> {
    pairs
        .iter()
        .map(|(k, v)| ((*k).to_string(), (*v).to_string()))
        .collect()
}

pub fn ensure_nonempty_json(v: &Json) -> bool {
    !v.is_null()
}
