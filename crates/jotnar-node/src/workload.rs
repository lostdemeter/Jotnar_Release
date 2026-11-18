//! Workload bundles, stage graphs, placement helpers, and the workload ledger.
//!
//! This is the model-aware layer above the executors. It owns the bundle
//! manifest, the stage dependency graph, the content hashes, and the decision
//! to place a stage on a particular executor. It never interprets a listing or
//! compiles a GPU program; it sends a stage spec to an executor that does.
//!
//! The first native workload is DOOM on the RV32IM core: frame 1 loads the
//! image and WAD, later frames resume from the previous checkpoint, and the
//! controller moves that checkpoint between executors.

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Component, Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};
use serde_json::{json, Value as Json};
use sha2::{Digest, Sha256};

use crate::executor::{validate_binary_result, Executor, RemoteExecutor};

pub const BUNDLE_SCHEMA: &str = "jotnar.workload/1";
pub const LEDGER_SCHEMA: &str = "jotnar.workload.ledger/1";
pub const RECEIPT_SCHEMA: &str = "jotnar.workload.receipt/1";

pub fn sha256_hex(bytes: &[u8]) -> String {
    let mut h = Sha256::new();
    h.update(bytes);
    format!("{:x}", h.finalize())
}

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

pub fn is_safe_relative(name: &str) -> bool {
    if name.is_empty() {
        return false;
    }
    let p = Path::new(name);
    !p.is_absolute() && p.components().all(|c| matches!(c, Component::Normal(_)))
}

pub fn safe_join(root: &Path, rel: &str) -> Result<PathBuf, String> {
    if !is_safe_relative(rel) {
        return Err(format!("unsafe relative path {rel:?}"));
    }
    Ok(root.join(rel))
}

fn write_atomic(path: &Path, bytes: &[u8]) -> Result<(), String> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|e| format!("create {}: {e}", parent.display()))?;
    }
    let tmp = path.with_extension(format!("tmp-{}-{}", std::process::id(), now_ms()));
    fs::write(&tmp, bytes).map_err(|e| format!("write {}: {e}", tmp.display()))?;
    fs::rename(&tmp, path).map_err(|e| format!("rename {}: {e}", path.display()))?;
    Ok(())
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct ArtifactRef {
    pub sha256: String,
    pub bytes: u64,
    pub path: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct StageExecutor {
    #[serde(default = "native_kind")]
    pub kind: String,
}

impl Default for StageExecutor {
    fn default() -> Self {
        Self {
            kind: native_kind(),
        }
    }
}

fn native_kind() -> String {
    "native".to_string()
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct StageInput {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub artifact: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stage: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output: Option<String>,
    pub path: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct StageSpec {
    pub name: String,
    #[serde(default)]
    pub depends_on: Vec<String>,
    #[serde(default)]
    pub executor: StageExecutor,
    pub command: Vec<String>,
    #[serde(default)]
    pub inputs: BTreeMap<String, StageInput>,
    pub outputs: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub timeout_ms: Option<u64>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct Bundle {
    pub schema: String,
    pub name: String,
    pub version: String,
    #[serde(default)]
    pub description: String,
    #[serde(default)]
    pub artifacts: BTreeMap<String, ArtifactRef>,
    pub stages: Vec<StageSpec>,
}

impl Bundle {
    pub fn validate(&self) -> Result<(), String> {
        if self.schema != BUNDLE_SCHEMA {
            return Err(format!(
                "bundle schema {:?} != {BUNDLE_SCHEMA:?}",
                self.schema
            ));
        }
        if self.name.trim().is_empty() {
            return Err("bundle needs a name".to_string());
        }
        if self.stages.is_empty() {
            return Err("bundle needs at least one stage".to_string());
        }
        let mut names = BTreeSet::new();
        for stage in &self.stages {
            if stage.name.trim().is_empty() {
                return Err("stage needs a name".to_string());
            }
            if !names.insert(stage.name.clone()) {
                return Err(format!("duplicate stage {:?}", stage.name));
            }
            if stage.command.is_empty() {
                return Err(format!("stage {:?} has an empty command", stage.name));
            }
            if stage.executor.kind != "native" {
                return Err(format!(
                    "stage {:?} requests unsupported executor kind {:?}",
                    stage.name, stage.executor.kind
                ));
            }
            if stage.outputs.is_empty() {
                return Err(format!("stage {:?} has no outputs", stage.name));
            }
            let mut outs = BTreeSet::new();
            for out in &stage.outputs {
                if !is_safe_relative(out) {
                    return Err(format!("stage {:?} has unsafe output {out:?}", stage.name));
                }
                if !outs.insert(out.clone()) {
                    return Err(format!("stage {:?} repeats output {out:?}", stage.name));
                }
            }
            let mut paths = BTreeSet::new();
            for (key, input) in &stage.inputs {
                if !is_safe_relative(&input.path) {
                    return Err(format!(
                        "stage {:?} input {key:?} has unsafe path {:?}",
                        stage.name, input.path
                    ));
                }
                if !paths.insert(input.path.clone()) {
                    return Err(format!(
                        "stage {:?} stages two inputs at {:?}",
                        stage.name, input.path
                    ));
                }
                match (&input.artifact, &input.stage) {
                    (Some(a), None) => {
                        if !self.artifacts.contains_key(a) {
                            return Err(format!(
                                "stage {:?} input {key:?} names unknown artifact {a:?}",
                                stage.name
                            ));
                        }
                    }
                    (None, Some(s)) => {
                        if input.output.as_deref().unwrap_or("").is_empty() {
                            return Err(format!(
                                "stage {:?} input {key:?} needs output for stage {s:?}",
                                stage.name
                            ));
                        }
                        if s == &stage.name {
                            return Err(format!("stage {:?} reads itself", stage.name));
                        }
                    }
                    _ => {
                        return Err(format!(
                            "stage {:?} input {key:?} needs exactly one of artifact/stage",
                            stage.name
                        ))
                    }
                }
            }
        }
        for stage in &self.stages {
            for dep in &stage.depends_on {
                if !names.contains(dep) {
                    return Err(format!(
                        "stage {:?} depends on unknown stage {dep:?}",
                        stage.name
                    ));
                }
            }
            for (key, input) in &stage.inputs {
                if let Some(src) = &input.stage {
                    if !stage.depends_on.iter().any(|d| d == src) {
                        return Err(format!(
                            "stage {:?} input {key:?} reads {src:?}; add it to depends_on",
                            stage.name
                        ));
                    }
                }
            }
        }
        for artifact in self.artifacts.values() {
            if !is_safe_relative(&artifact.path) {
                return Err(format!("artifact path {:?} is unsafe", artifact.path));
            }
        }
        self.stage_order()?;
        Ok(())
    }

    /// Deterministic topological order: declaration order breaks ties.
    pub fn stage_order(&self) -> Result<Vec<String>, String> {
        let mut remaining: Vec<&StageSpec> = self.stages.iter().collect();
        let mut done = BTreeSet::new();
        let mut order = Vec::new();
        while !remaining.is_empty() {
            let idx = remaining
                .iter()
                .position(|s| s.depends_on.iter().all(|d| done.contains(d)));
            let Some(idx) = idx else {
                return Err("stage graph has a cycle or an unknown dependency".to_string());
            };
            let stage = remaining.remove(idx);
            done.insert(stage.name.clone());
            order.push(stage.name.clone());
        }
        Ok(order)
    }
}

#[derive(Clone, Debug)]
pub struct LoadedBundle {
    pub bundle: Bundle,
    pub bytes: Vec<u8>,
    pub sha256: String,
    pub dir: PathBuf,
}

impl LoadedBundle {
    pub fn load(path: &Path) -> Result<Self, String> {
        let file = if path.is_dir() {
            path.join("bundle.json")
        } else {
            path.to_path_buf()
        };
        let bytes = fs::read(&file).map_err(|e| format!("read {}: {e}", file.display()))?;
        let bundle: Bundle =
            serde_json::from_slice(&bytes).map_err(|e| format!("parse {}: {e}", file.display()))?;
        bundle.validate()?;
        let dir = file
            .parent()
            .map(|p| p.to_path_buf())
            .unwrap_or_else(|| PathBuf::from("."));
        let sha256 = sha256_hex(&bytes);
        Ok(Self {
            bundle,
            bytes,
            sha256,
            dir,
        })
    }

    pub fn write(&self, dir: &Path) -> Result<PathBuf, String> {
        fs::create_dir_all(dir).map_err(|e| format!("create {}: {e}", dir.display()))?;
        let path = dir.join("bundle.json");
        write_atomic(&path, &self.bytes)?;
        Ok(path)
    }

    pub fn artifact_bytes(&self, name: &str) -> Result<(Vec<u8>, ArtifactRef), String> {
        let artifact = self
            .bundle
            .artifacts
            .get(name)
            .ok_or_else(|| format!("unknown artifact {name:?}"))?
            .clone();
        let path = safe_join(&self.dir, &artifact.path)?;
        let bytes =
            fs::read(&path).map_err(|e| format!("read artifact {}: {e}", path.display()))?;
        let got = sha256_hex(&bytes);
        if got != artifact.sha256 {
            return Err(format!(
                "artifact {name:?} hash {got} != {}",
                artifact.sha256
            ));
        }
        if bytes.len() as u64 != artifact.bytes {
            return Err(format!(
                "artifact {name:?} has {} bytes, manifest says {}",
                bytes.len(),
                artifact.bytes
            ));
        }
        Ok((bytes, artifact))
    }
}

fn copy_artifact(src: &Path, bundle_dir: &Path) -> Result<ArtifactRef, String> {
    let bytes = fs::read(src).map_err(|e| format!("read {}: {e}", src.display()))?;
    let sha = sha256_hex(&bytes);
    let rel = format!("blobs/{sha}");
    let dest = safe_join(bundle_dir, &rel)?;
    if let Some(parent) = dest.parent() {
        fs::create_dir_all(parent).map_err(|e| format!("create {}: {e}", parent.display()))?;
    }
    if dest.exists() {
        let existing = fs::read(&dest).map_err(|e| format!("read {}: {e}", dest.display()))?;
        if sha256_hex(&existing) != sha {
            return Err(format!("blob {} has the wrong hash", dest.display()));
        }
    } else {
        fs::copy(src, &dest)
            .map_err(|e| format!("copy {} -> {}: {e}", src.display(), dest.display()))?;
    }
    Ok(ArtifactRef {
        sha256: sha,
        bytes: bytes.len() as u64,
        path: rel,
    })
}

/// Build the two-or-more-stage DOOM bundle.
///
/// `--frames N` in `jotnar-demo-doom` is an absolute target frame, so stage
/// `frameN` resumes `frameN-1` and passes `--frames N`. This is the checkpoint
/// handoff shape the two-node gate uses.
pub fn build_doom_bundle(
    image: &Path,
    wad: &Path,
    frames: u32,
    out_dir: &Path,
    name: &str,
) -> Result<LoadedBundle, String> {
    let frames = frames.max(1);
    fs::create_dir_all(out_dir).map_err(|e| format!("create {}: {e}", out_dir.display()))?;
    let image_ref = copy_artifact(image, out_dir)?;
    let wad_ref = copy_artifact(wad, out_dir)?;
    let mut artifacts = BTreeMap::new();
    artifacts.insert("doom-image".to_string(), image_ref);
    artifacts.insert("doom-wad".to_string(), wad_ref);
    let mut stages = Vec::new();
    for n in 1..=frames {
        let out = format!("frame{n}");
        let mut inputs = BTreeMap::new();
        let mut depends_on = Vec::new();
        let command = if n == 1 {
            inputs.insert(
                "image".to_string(),
                StageInput {
                    artifact: Some("doom-image".to_string()),
                    stage: None,
                    output: None,
                    path: "doom.bin".to_string(),
                },
            );
            inputs.insert(
                "wad".to_string(),
                StageInput {
                    artifact: Some("doom-wad".to_string()),
                    stage: None,
                    output: None,
                    path: "doom1.wad".to_string(),
                },
            );
            vec![
                "{{program}}".to_string(),
                "--image".to_string(),
                "{{input:image}}".to_string(),
                "--wad".to_string(),
                "{{input:wad}}".to_string(),
                "--frames".to_string(),
                "1".to_string(),
                "--out".to_string(),
                out.clone(),
            ]
        } else {
            let prev = format!("frame{}", n - 1);
            depends_on.push(prev.clone());
            for ext in ["mem", "x", "pc", "cyc", "csr"] {
                let file = format!("{prev}.{ext}");
                inputs.insert(
                    ext.to_string(),
                    StageInput {
                        artifact: None,
                        stage: Some(prev.clone()),
                        output: Some(file.clone()),
                        path: file,
                    },
                );
            }
            vec![
                "{{program}}".to_string(),
                "--resume".to_string(),
                prev,
                "--frames".to_string(),
                n.to_string(),
                "--out".to_string(),
                out.clone(),
            ]
        };
        stages.push(StageSpec {
            name: out.clone(),
            depends_on,
            executor: StageExecutor::default(),
            command,
            inputs,
            outputs: ["fb", "png", "mem", "x", "pc", "cyc", "csr"]
                .iter()
                .map(|ext| format!("{out}.{ext}"))
                .collect(),
            timeout_ms: Some(600_000),
        });
    }
    let bytes = serde_json::to_vec_pretty(&Bundle {
        schema: BUNDLE_SCHEMA.to_string(),
        name: name.to_string(),
        version: "1".to_string(),
        description: "DOOM on the native RV32IM core with checkpoint handoff".to_string(),
        artifacts,
        stages,
    })
    .map_err(|e| e.to_string())?;
    let path = out_dir.join("bundle.json");
    write_atomic(&path, &bytes)?;
    LoadedBundle::load(&path)
}

pub fn resolve_command(
    command: &[String],
    program: &str,
    inputs: &BTreeMap<String, String>,
) -> Result<Vec<String>, String> {
    let mut out = Vec::with_capacity(command.len());
    for arg in command {
        let mut s = arg.replace("{{program}}", program);
        for (key, path) in inputs {
            s = s.replace(&format!("{{{{input:{key}}}}}"), path);
        }
        if s.contains("{{") {
            return Err(format!("unresolved command token in {s:?}"));
        }
        out.push(s);
    }
    Ok(out)
}

#[derive(Clone, Debug, Serialize, Deserialize, Default, PartialEq, Eq)]
pub struct FileInfo {
    pub sha256: String,
    pub bytes: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize, Default)]
pub struct StageRecord {
    pub state: String,
    pub attempts: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub executor: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub job_id: Option<String>,
    #[serde(default)]
    pub reused: bool,
    #[serde(default)]
    pub command: Vec<String>,
    #[serde(default)]
    pub inputs: BTreeMap<String, FileInfo>,
    #[serde(default)]
    pub outputs: BTreeMap<String, FileInfo>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub exit_code: Option<i64>,
    #[serde(default)]
    pub stdout: String,
    #[serde(default)]
    pub stderr: String,
    #[serde(default)]
    pub started_ms: u64,
    #[serde(default)]
    pub finished_ms: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Ledger {
    pub schema: String,
    pub bundle_sha256: String,
    pub updated_ms: u64,
    pub stages: BTreeMap<String, StageRecord>,
}

impl Ledger {
    fn new(bundle_sha256: &str) -> Self {
        Self {
            schema: LEDGER_SCHEMA.to_string(),
            bundle_sha256: bundle_sha256.to_string(),
            updated_ms: now_ms(),
            stages: BTreeMap::new(),
        }
    }

    pub fn load(dir: &Path, bundle_sha256: &str, force: bool) -> Result<Self, String> {
        let path = dir.join("ledger.json");
        if !path.exists() || force {
            return Ok(Self::new(bundle_sha256));
        }
        let mut ledger = Self::read(dir)?;
        if ledger.bundle_sha256 != bundle_sha256 {
            return Err(format!(
                "ledger belongs to bundle {}, not {}; use --force to start over",
                ledger.bundle_sha256, bundle_sha256
            ));
        }
        ledger.updated_ms = now_ms();
        Ok(ledger)
    }

    pub fn read(dir: &Path) -> Result<Self, String> {
        let path = dir.join("ledger.json");
        let bytes = fs::read(&path).map_err(|e| format!("read {}: {e}", path.display()))?;
        let ledger: Ledger =
            serde_json::from_slice(&bytes).map_err(|e| format!("parse {}: {e}", path.display()))?;
        if ledger.schema != LEDGER_SCHEMA {
            return Err(format!(
                "ledger schema {:?} != {LEDGER_SCHEMA:?}",
                ledger.schema
            ));
        }
        Ok(ledger)
    }

    pub fn save(&mut self, dir: &Path) -> Result<(), String> {
        self.updated_ms = now_ms();
        fs::create_dir_all(dir).map_err(|e| format!("create {}: {e}", dir.display()))?;
        let bytes = serde_json::to_vec_pretty(self).map_err(|e| e.to_string())?;
        write_atomic(&dir.join("ledger.json"), &bytes)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ExecutorEndpoint {
    pub addr: String,
    #[serde(default)]
    pub capability: Json,
}

#[derive(Clone, Debug)]
pub struct RunConfig {
    pub program: String,
    pub executors: Vec<ExecutorEndpoint>,
    pub placement: BTreeMap<String, String>,
    pub token: Option<String>,
    pub force: bool,
    pub timeout_ms: Option<u64>,
    pub compare: Vec<(String, PathBuf)>,
}

fn stage_dir(ledger_dir: &Path, stage: &str) -> PathBuf {
    ledger_dir.join("stages").join(stage)
}

fn stage_output_path(ledger_dir: &Path, stage: &str, name: &str) -> Result<PathBuf, String> {
    safe_join(&stage_dir(ledger_dir, stage), name)
}

fn file_info(path: &Path) -> Result<FileInfo, String> {
    let bytes = fs::read(path).map_err(|e| format!("read {}: {e}", path.display()))?;
    Ok(FileInfo {
        sha256: sha256_hex(&bytes),
        bytes: bytes.len() as u64,
    })
}

struct PreparedInput {
    key: String,
    path: String,
    bytes: Vec<u8>,
    info: FileInfo,
}

fn prepare_inputs(
    loaded: &LoadedBundle,
    stage: &StageSpec,
    ledger: &Ledger,
    ledger_dir: &Path,
) -> Result<Vec<PreparedInput>, String> {
    let mut out = Vec::new();
    for (key, input) in &stage.inputs {
        let (bytes, info) = if let Some(artifact) = &input.artifact {
            let (bytes, artifact_ref) = loaded.artifact_bytes(artifact)?;
            (
                bytes,
                FileInfo {
                    sha256: artifact_ref.sha256,
                    bytes: artifact_ref.bytes,
                },
            )
        } else {
            let src = input.stage.as_deref().unwrap_or("");
            let output = input.output.as_deref().ok_or("input needs output")?;
            let record = ledger.stages.get(src).ok_or_else(|| {
                format!("stage {:?} needs {:?} but it has not run", stage.name, src)
            })?;
            let expected = record
                .outputs
                .get(output)
                .ok_or_else(|| format!("stage {:?} did not produce {output:?}", src))?;
            let path = stage_output_path(ledger_dir, src, output)?;
            let got = file_info(&path)?;
            if &got != expected {
                return Err(format!(
                    "checkpoint {} has changed since stage {src:?}",
                    path.display()
                ));
            }
            (
                fs::read(&path).map_err(|e| format!("read {}: {e}", path.display()))?,
                got,
            )
        };
        out.push(PreparedInput {
            key: key.clone(),
            path: input.path.clone(),
            bytes,
            info,
        });
    }
    let mut seen = BTreeSet::new();
    for input in &out {
        if !seen.insert(input.path.clone()) {
            return Err(format!(
                "stage {:?} stages two inputs at {:?}",
                stage.name, input.path
            ));
        }
    }
    Ok(out)
}

fn split_result(
    stage: &StageSpec,
    header: &Json,
    body: &[u8],
    ledger_dir: &Path,
) -> Result<BTreeMap<String, FileInfo>, String> {
    let descs = header
        .get("outputs")
        .and_then(|x| x.as_array())
        .ok_or("result has no output descriptors")?;
    let mut seen = BTreeSet::new();
    let mut infos = BTreeMap::new();
    for desc in descs {
        let name = desc
            .get("name")
            .and_then(|x| x.as_str())
            .ok_or("descriptor has no name")?;
        if !stage.outputs.iter().any(|o| o == name) {
            return Err(format!("executor returned unexpected output {name:?}"));
        }
        if !seen.insert(name.to_string()) {
            return Err(format!("executor returned output {name:?} twice"));
        }
        let off = desc.get("off").and_then(|x| x.as_u64()).unwrap_or(0) as usize;
        let len =
            desc.get("bytes")
                .and_then(|x| x.as_u64())
                .ok_or_else(|| format!("output {name:?} has no byte count"))? as usize;
        let end = off
            .checked_add(len)
            .ok_or_else(|| format!("output {name:?} range overflows"))?;
        if end > body.len() {
            return Err(format!("output {name:?} is truncated"));
        }
        let bytes = &body[off..end];
        let sha = sha256_hex(bytes);
        if let Some(want) = desc.get("sha256").and_then(|x| x.as_str()) {
            if sha != want {
                return Err(format!("output {name:?} hash {sha} != {want}"));
            }
        }
        let path = stage_output_path(ledger_dir, &stage.name, name)?;
        write_atomic(&path, bytes)?;
        infos.insert(
            name.to_string(),
            FileInfo {
                sha256: sha,
                bytes: bytes.len() as u64,
            },
        );
    }
    for name in &stage.outputs {
        if !infos.contains_key(name) {
            return Err(format!("executor did not return {name:?}"));
        }
    }
    Ok(infos)
}

fn poll_job(executor: &RemoteExecutor, job_id: &str, deadline_ms: u64) -> Result<Json, String> {
    let start = now_ms();
    loop {
        let status = executor
            .status(job_id)
            .map_err(|(code, msg)| format!("job-status {code}: {msg}"))?;
        let job = status.get("job").cloned().unwrap_or(Json::Null);
        let state = job
            .get("state")
            .and_then(|x| x.as_str())
            .unwrap_or("unknown");
        if matches!(state, "done" | "failed" | "canceled") {
            return Ok(job);
        }
        if now_ms().saturating_sub(start) >= deadline_ms {
            return Err(format!("job {job_id} timed out in state {state}"));
        }
        std::thread::sleep(std::time::Duration::from_millis(25));
    }
}

pub fn run_bundle(
    loaded: &LoadedBundle,
    ledger_dir: &Path,
    cfg: &RunConfig,
) -> Result<Json, String> {
    if cfg.executors.is_empty() && cfg.placement.is_empty() {
        return Err("run needs at least one --executor or --place stage=addr".to_string());
    }
    let program = cfg.program.clone();
    let order = loaded.bundle.stage_order()?;
    if cfg.executors.is_empty() {
        for name in &order {
            if !cfg.placement.contains_key(name) {
                return Err(format!(
                    "stage {name:?} has no placement and the executor pool is empty"
                ));
            }
        }
    }
    let mut ledger = Ledger::load(ledger_dir, &loaded.sha256, cfg.force)?;
    let stages: BTreeMap<&str, &StageSpec> = loaded
        .bundle
        .stages
        .iter()
        .map(|s| (s.name.as_str(), s))
        .collect();
    let mut comparisons = Vec::new();
    let mut comparison_ok = true;

    for (index, name) in order.iter().enumerate() {
        let stage = stages
            .get(name.as_str())
            .ok_or_else(|| format!("missing stage {name:?}"))?;
        let inputs = prepare_inputs(loaded, stage, &ledger, ledger_dir)?;
        let input_infos: BTreeMap<String, FileInfo> = inputs
            .iter()
            .map(|i| (i.key.clone(), i.info.clone()))
            .collect();
        let input_paths: BTreeMap<String, String> = inputs
            .iter()
            .map(|i| (i.key.clone(), i.path.clone()))
            .collect();
        let command = resolve_command(&stage.command, &program, &input_paths)?;
        let prior = ledger.stages.get(name).cloned().unwrap_or_default();
        if !cfg.force
            && prior.state == "done"
            && prior.command == command
            && prior.inputs == input_infos
            && stage.outputs.iter().all(|out| {
                prior
                    .outputs
                    .get(out)
                    .and_then(|want| {
                        stage_output_path(ledger_dir, name, out)
                            .ok()
                            .and_then(|p| file_info(&p).ok())
                            .map(|got| &got == want)
                    })
                    .unwrap_or(false)
            })
        {
            let mut record = prior.clone();
            record.reused = true;
            record.attempts = record.attempts.max(1);
            ledger.stages.insert(name.clone(), record);
            ledger.save(ledger_dir)?;
            continue;
        }
        let addr = if let Some(addr) = cfg.placement.get(name) {
            addr.clone()
        } else {
            cfg.executors[index % cfg.executors.len()].addr.clone()
        };
        let capability = cfg
            .executors
            .iter()
            .find(|e| e.addr == addr)
            .map(|e| e.capability.clone())
            .unwrap_or_else(|| json!({"kind": "native", "addr": addr}));
        let executor = RemoteExecutor::new(capability, addr.clone(), cfg.token.clone());
        let mut body = Vec::new();
        let mut descs = Vec::new();
        for input in &inputs {
            let off = body.len();
            body.extend_from_slice(&input.bytes);
            descs.push(json!({
                "name": input.path,
                "off": off,
                "bytes": input.bytes.len(),
                "sha256": input.info.sha256,
            }));
        }
        let timeout_ms = cfg
            .timeout_ms
            .or(stage.timeout_ms)
            .unwrap_or(600_000)
            .max(1);
        let mut spec = json!({
            "type": "submit",
            "command": command,
            "outputs": stage.outputs,
            "inputs": descs,
            "timeout_ms": timeout_ms,
        });
        if let Some(t) = &cfg.token {
            spec["auth"] = json!(t);
        }
        let submitted = executor
            .submit(&spec, &body)
            .map_err(|(code, msg)| format!("submit {addr}: {code}: {msg}"))?;
        let job_id = submitted
            .get("job_id")
            .and_then(|x| x.as_str())
            .ok_or("submit returned no job_id")?
            .to_string();
        let job = poll_job(&executor, &job_id, timeout_ms.saturating_add(60_000))?;
        let state = job
            .get("state")
            .and_then(|x| x.as_str())
            .unwrap_or("unknown");
        if state != "done" {
            let mut record = prior.clone();
            record.state = state.to_string();
            record.attempts += 1;
            record.executor = Some(addr);
            record.job_id = Some(job_id);
            record.command = command;
            record.inputs = input_infos;
            record.finished_ms = now_ms();
            ledger.stages.insert(name.clone(), record);
            ledger.save(ledger_dir)?;
            return Err(format!("stage {name:?} ended in state {state}: {job}"));
        }
        let (header, result_body) = executor
            .result(&job_id)
            .map_err(|(code, msg)| format!("job-result {addr}: {code}: {msg}"))?;
        if header.get("output_format").and_then(|x| x.as_str()) == Some("files") {
            validate_binary_result(&header, &result_body)
                .map_err(|e| format!("invalid result from {addr}: {e}"))?;
        }
        let outputs = split_result(stage, &header, &result_body, ledger_dir)?;
        let record = StageRecord {
            state: "done".to_string(),
            attempts: prior.attempts + 1,
            executor: Some(addr.clone()),
            job_id: Some(job_id),
            reused: false,
            command,
            inputs: input_infos,
            outputs,
            exit_code: header.get("exit_code").and_then(|x| x.as_i64()),
            stdout: header
                .get("stdout")
                .and_then(|x| x.as_str())
                .unwrap_or("")
                .to_string(),
            stderr: header
                .get("stderr")
                .and_then(|x| x.as_str())
                .unwrap_or("")
                .to_string(),
            started_ms: job
                .get("started_ms")
                .and_then(|x| x.as_u64())
                .unwrap_or_else(now_ms),
            finished_ms: job
                .get("finished_ms")
                .and_then(|x| x.as_u64())
                .unwrap_or_else(now_ms),
        };
        ledger.stages.insert(name.clone(), record);
        ledger.save(ledger_dir)?;
    }

    for (key, reference) in &cfg.compare {
        let (stage_name, output) = key
            .split_once(':')
            .or_else(|| key.split_once('.'))
            .ok_or_else(|| format!("--compare needs STAGE:OUTPUT or STAGE.OUTPUT, got {key:?}"))?;
        let path = stage_output_path(ledger_dir, stage_name, output)?;
        let got = fs::read(&path).map_err(|e| format!("read {}: {e}", path.display()))?;
        let want = fs::read(reference)
            .map_err(|e| format!("read reference {}: {e}", reference.display()))?;
        let matches = got == want;
        comparison_ok &= matches;
        comparisons.push(json!({
            "stage": stage_name,
            "output": output,
            "reference": reference.display().to_string(),
            "bytes": got.len(),
            "sha256": sha256_hex(&got),
            "reference_sha256": sha256_hex(&want),
            "match": matches,
        }));
    }

    let receipt = json!({
        "schema": RECEIPT_SCHEMA,
        "ok": comparison_ok,
        "bundle": {
            "name": loaded.bundle.name,
            "version": loaded.bundle.version,
            "sha256": loaded.sha256,
        },
        "ledger": ledger_dir.display().to_string(),
        "stages": ledger.stages,
        "comparisons": comparisons,
        "executors": cfg.executors.iter().map(|e| &e.addr).collect::<Vec<_>>(),
        "placement": cfg.placement,
        "finished_ms": now_ms(),
    });
    let bytes = serde_json::to_vec_pretty(&receipt).map_err(|e| e.to_string())?;
    write_atomic(&ledger_dir.join("receipt.json"), &bytes)?;
    if !comparison_ok {
        return Err(format!(
            "comparison failed; receipt: {}",
            ledger_dir.join("receipt.json").display()
        ));
    }
    Ok(receipt)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bundle_with(stages: Vec<StageSpec>) -> Bundle {
        Bundle {
            schema: BUNDLE_SCHEMA.to_string(),
            name: "test".to_string(),
            version: "1".to_string(),
            description: String::new(),
            artifacts: BTreeMap::new(),
            stages,
        }
    }

    fn stage(name: &str, deps: &[&str], outputs: &[&str]) -> StageSpec {
        StageSpec {
            name: name.to_string(),
            depends_on: deps.iter().map(|s| s.to_string()).collect(),
            executor: StageExecutor::default(),
            command: vec!["{{program}}".to_string()],
            inputs: BTreeMap::new(),
            outputs: outputs.iter().map(|s| s.to_string()).collect(),
            timeout_ms: None,
        }
    }

    #[test]
    fn topo_order_is_deterministic_and_rejects_cycles() {
        let b = bundle_with(vec![
            stage("b", &["a"], &["b.out"]),
            stage("a", &[], &["a.out"]),
        ]);
        assert_eq!(b.stage_order().unwrap(), vec!["a", "b"]);
        let cyclic = bundle_with(vec![
            stage("a", &["b"], &["a.out"]),
            stage("b", &["a"], &["b.out"]),
        ]);
        assert!(cyclic.stage_order().is_err());
    }

    #[test]
    fn command_tokens_resolve() {
        let mut inputs = BTreeMap::new();
        inputs.insert("image".to_string(), "doom.bin".to_string());
        let command = vec![
            "{{program}}".to_string(),
            "--image".to_string(),
            "{{input:image}}".to_string(),
        ];
        let got = resolve_command(&command, "/bin/doom", &inputs).unwrap();
        assert_eq!(got, vec!["/bin/doom", "--image", "doom.bin"]);
        assert!(resolve_command(&["{{input:missing}}".to_string()], "p", &inputs).is_err());
    }

    #[test]
    fn unsafe_paths_are_rejected() {
        assert!(is_safe_relative("frame1.mem"));
        assert!(is_safe_relative("a/b"));
        assert!(!is_safe_relative("../x"));
        assert!(!is_safe_relative("/tmp/x"));
        assert!(!is_safe_relative("./x"));
    }

    #[test]
    fn doom_stage_two_uses_absolute_frame_target() {
        let root = std::env::temp_dir().join(format!(
            "jotnar-doom-bundle-test-{}-{}",
            std::process::id(),
            now_ms()
        ));
        fs::create_dir_all(&root).unwrap();
        let image = root.join("doom.bin");
        let wad = root.join("doom1.wad");
        fs::write(&image, b"image").unwrap();
        fs::write(&wad, b"wad").unwrap();
        let loaded = build_doom_bundle(&image, &wad, 2, &root.join("bundle"), "doom-test").unwrap();
        assert_eq!(
            loaded.bundle.stage_order().unwrap(),
            vec!["frame1", "frame2"]
        );
        let frame2 = loaded
            .bundle
            .stages
            .iter()
            .find(|s| s.name == "frame2")
            .unwrap();
        assert!(frame2
            .command
            .windows(2)
            .any(|w| w[0] == "--frames" && w[1] == "2"));
        assert!(frame2
            .command
            .windows(2)
            .any(|w| w[0] == "--resume" && w[1] == "frame1"));
        assert_eq!(frame2.inputs.len(), 5);
        assert!(loaded.artifact_bytes("doom-image").is_ok());
        let _ = fs::remove_dir_all(&root);
    }
}
