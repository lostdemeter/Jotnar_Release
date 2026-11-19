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
    /// GPU target constraint: `cuda`, `hip`, or `any`. Native stages leave it
    /// unset.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub target: Option<String>,
}

impl Default for StageExecutor {
    fn default() -> Self {
        Self {
            kind: native_kind(),
            target: None,
        }
    }
}

fn native_kind() -> String {
    "native".to_string()
}

fn one_u32() -> u32 {
    1
}

/// A content reference used by a GPU stage.
///
/// `listing` accepts only an artifact: the controller reads the artifact text
/// and sends it inline in the submit body. `payloads` accept either an artifact
/// (whose bytes are the JSON payload object) or a previous stage output (whose
/// binary body is decoded into one named payload stream).
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct StageRef {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub artifact: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stage: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output: Option<String>,
    /// Payload stream name for a stage-output source. Defaults to `output`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stream: Option<String>,
}

/// GPU-specific stage payload.
///
/// v1 intentionally accepts one frame and one repetition per stage. Multi-frame
/// binary results are keyed by output name in the current result splitter, so
/// multi-frame stages need an output-naming decision before they are enabled.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct GpuStage {
    pub listing: StageRef,
    #[serde(default)]
    pub payloads: Vec<StageRef>,
    #[serde(default = "one_u32")]
    pub frames: u32,
    #[serde(default = "one_u32")]
    pub count: u32,
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
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub command: Vec<String>,
    #[serde(default)]
    pub inputs: BTreeMap<String, StageInput>,
    pub outputs: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub timeout_ms: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gpu: Option<GpuStage>,
}

impl StageSpec {
    pub fn is_gpu(&self) -> bool {
        self.executor.kind == "gpu"
    }

    pub fn target(&self) -> &str {
        self.executor.target.as_deref().unwrap_or("any")
    }
}

fn validate_listing_ref(
    stage: &StageSpec,
    listing: &StageRef,
    artifacts: &BTreeMap<String, ArtifactRef>,
) -> Result<(), String> {
    if listing.stage.is_some() || listing.output.is_some() || listing.stream.is_some() {
        return Err(format!(
            "gpu stage {:?} listing must name an artifact only",
            stage.name
        ));
    }
    let artifact = listing
        .artifact
        .as_deref()
        .ok_or_else(|| format!("gpu stage {:?} listing needs an artifact", stage.name))?;
    if !artifacts.contains_key(artifact) {
        return Err(format!(
            "gpu stage {:?} listing names unknown artifact {artifact:?}",
            stage.name
        ));
    }
    Ok(())
}

fn validate_payload_ref(
    stage: &StageSpec,
    payload: &StageRef,
    index: usize,
    artifacts: &BTreeMap<String, ArtifactRef>,
    stages: &BTreeMap<&str, &StageSpec>,
) -> Result<(), String> {
    match (&payload.artifact, &payload.stage) {
        (Some(artifact), None) => {
            if payload.output.is_some() || payload.stream.is_some() {
                return Err(format!(
                    "gpu stage {:?} payload {index} artifact source must not name output/stream",
                    stage.name
                ));
            }
            if !artifacts.contains_key(artifact) {
                return Err(format!(
                    "gpu stage {:?} payload {index} names unknown artifact {artifact:?}",
                    stage.name
                ));
            }
        }
        (None, Some(src)) => {
            if payload.output.as_deref().unwrap_or("").is_empty() {
                return Err(format!(
                    "gpu stage {:?} payload {index} needs output for stage {src:?}",
                    stage.name
                ));
            }
            if src == &stage.name {
                return Err(format!(
                    "gpu stage {:?} payload {index} reads itself",
                    stage.name
                ));
            }
            let output = payload.output.as_deref().unwrap_or("");
            let source = stages.get(src.as_str()).ok_or_else(|| {
                format!(
                    "gpu stage {:?} payload {index} names unknown stage {src:?}",
                    stage.name
                )
            })?;
            if !source.outputs.iter().any(|o| o == output) {
                return Err(format!(
                    "gpu stage {:?} payload {index} reads {src:?}.{output:?}, which that stage does not produce",
                    stage.name
                ));
            }
        }
        _ => {
            return Err(format!(
                "gpu stage {:?} payload {index} needs exactly one of artifact/stage",
                stage.name
            ));
        }
    }
    Ok(())
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
        }
        let stage_by_name: BTreeMap<&str, &StageSpec> =
            self.stages.iter().map(|s| (s.name.as_str(), s)).collect();
        for stage in &self.stages {
            match stage.executor.kind.as_str() {
                "native" => {
                    if stage.command.is_empty() {
                        return Err(format!("stage {:?} has an empty command", stage.name));
                    }
                    if stage.gpu.is_some() {
                        return Err(format!(
                            "native stage {:?} must not carry a gpu payload",
                            stage.name
                        ));
                    }
                }
                "gpu" => {
                    if !stage.command.is_empty() {
                        return Err(format!(
                            "gpu stage {:?} must not carry a command",
                            stage.name
                        ));
                    }
                    if !stage.inputs.is_empty() {
                        return Err(format!(
                            "gpu stage {:?} uses payloads, not inputs",
                            stage.name
                        ));
                    }
                    if !matches!(stage.target(), "cuda" | "hip" | "any") {
                        return Err(format!(
                            "gpu stage {:?} target {:?} must be cuda, hip, or any",
                            stage.name,
                            stage.target()
                        ));
                    }
                    let gpu = stage
                        .gpu
                        .as_ref()
                        .ok_or_else(|| format!("gpu stage {:?} needs a gpu payload", stage.name))?;
                    if gpu.frames != 1 || gpu.count != 1 {
                        return Err(format!(
                            "gpu stage {:?} v1 supports frames=1 and count=1 (got {}/{})",
                            stage.name, gpu.frames, gpu.count
                        ));
                    }
                    validate_listing_ref(stage, &gpu.listing, &self.artifacts)?;
                    if gpu.payloads.is_empty() {
                        return Err(format!("gpu stage {:?} has no payloads", stage.name));
                    }
                    for (index, payload) in gpu.payloads.iter().enumerate() {
                        validate_payload_ref(
                            stage,
                            payload,
                            index,
                            &self.artifacts,
                            &stage_by_name,
                        )?;
                    }
                }
                other => {
                    return Err(format!(
                        "stage {:?} requests unsupported executor kind {other:?}",
                        stage.name
                    ));
                }
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
            if let Some(gpu) = &stage.gpu {
                for (index, payload) in gpu.payloads.iter().enumerate() {
                    if let Some(src) = &payload.stage {
                        if !stage.depends_on.iter().any(|d| d == src) {
                            return Err(format!(
                                "gpu stage {:?} payload {index} reads {src:?}; add it to depends_on",
                                stage.name
                            ));
                        }
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
            gpu: None,
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

/// Deterministic synthetic payload used by the batch GPU acceptance test.
///
/// Values are small non-negative integers, so doubling them (the test listing)
/// is exact on both CUDA and HIP. `seed` makes each batch item distinct.
pub fn generated_payload(h: usize, w: usize, c: usize, seed: u64) -> Json {
    let mut data = Vec::with_capacity(h.saturating_mul(w).saturating_mul(c));
    for y in 0..h {
        for x in 0..w {
            for ch in 0..c {
                let v = (seed as usize)
                    .wrapping_mul(37)
                    .wrapping_add(y.wrapping_mul(13))
                    .wrapping_add(x.wrapping_mul(7))
                    .wrapping_add(ch.wrapping_mul(3))
                    % 251;
                data.push(v as f32);
            }
        }
    }
    json!({
        "streams": {
            "X": {"kind": "F32", "shape": [h, w, c], "data": data},
        }
    })
}

fn write_artifact_bytes(bundle_dir: &Path, bytes: &[u8]) -> Result<ArtifactRef, String> {
    let sha = sha256_hex(bytes);
    let rel = format!("blobs/{sha}");
    let dest = safe_join(bundle_dir, &rel)?;
    if dest.exists() {
        let existing = fs::read(&dest).map_err(|e| format!("read {}: {e}", dest.display()))?;
        if sha256_hex(&existing) != sha {
            return Err(format!("blob {} has the wrong hash", dest.display()));
        }
    } else {
        write_atomic(&dest, bytes)?;
    }
    Ok(ArtifactRef {
        sha256: sha,
        bytes: bytes.len() as u64,
        path: rel,
    })
}

#[derive(Clone, Debug)]
pub struct GpuBundleSpec {
    pub listing: PathBuf,
    /// A JSON payload artifact reused for every item.
    pub input: Option<PathBuf>,
    /// `(height, width, channels)` for a deterministic generated payload per
    /// item. Exactly one of `input` and `generate` is required.
    pub generate: Option<(usize, usize, usize)>,
    pub items: usize,
    pub out_dir: PathBuf,
    pub name: String,
    pub timeout_ms: u64,
}

/// Build a short batch GPU pipeline:
///
/// ```text
/// item i:  gpu-a-i (cuda) -> gpu-b-i (any) -> OUT
/// ```
///
/// Stage `gpu-a-i` consumes the item payload artifact. Stage `gpu-b-i` consumes
/// stage `gpu-a-i`'s binary `OUT` as payload stream `X`, which exercises the
/// intermediate GPU artifact handoff. The bundle is target-constrained only on
/// stage A so the same manifest can be pinned to one CUDA node as a reference.
pub fn build_gpu_bundle(spec: &GpuBundleSpec) -> Result<LoadedBundle, String> {
    if spec.input.is_some() == spec.generate.is_some() {
        return Err("gpu bundle needs exactly one of input or generate".to_string());
    }
    let items = spec.items.max(1);
    fs::create_dir_all(&spec.out_dir)
        .map_err(|e| format!("create {}: {e}", spec.out_dir.display()))?;
    let listing_ref = copy_artifact(&spec.listing, &spec.out_dir)?;
    let mut artifacts = BTreeMap::new();
    artifacts.insert("listing".to_string(), listing_ref);
    let mut stages = Vec::new();
    for item in 0..items {
        let input_name = format!("input-{item:02}");
        let input_ref = if let Some(path) = &spec.input {
            let bytes = fs::read(path).map_err(|e| format!("read {}: {e}", path.display()))?;
            let value: Json = serde_json::from_slice(&bytes)
                .map_err(|e| format!("payload {} is not JSON: {e}", path.display()))?;
            if value.get("streams").and_then(|x| x.as_object()).is_none() {
                return Err(format!(
                    "payload {} needs a top-level streams object",
                    path.display()
                ));
            }
            write_artifact_bytes(&spec.out_dir, &bytes)?
        } else {
            let (h, w, c) = spec.generate.unwrap();
            let payload = generated_payload(h, w, c, item as u64);
            let bytes = serde_json::to_vec(&payload).map_err(|e| e.to_string())?;
            write_artifact_bytes(&spec.out_dir, &bytes)?
        };
        artifacts.insert(input_name.clone(), input_ref);
        let first = format!("gpu-a-{item:02}");
        let second = format!("gpu-b-{item:02}");
        let listing = StageRef {
            artifact: Some("listing".to_string()),
            stage: None,
            output: None,
            stream: None,
        };
        stages.push(StageSpec {
            name: first.clone(),
            depends_on: Vec::new(),
            executor: StageExecutor {
                kind: "gpu".to_string(),
                target: Some("cuda".to_string()),
            },
            command: Vec::new(),
            inputs: BTreeMap::new(),
            outputs: vec!["OUT".to_string()],
            timeout_ms: Some(spec.timeout_ms),
            gpu: Some(GpuStage {
                listing: listing.clone(),
                payloads: vec![StageRef {
                    artifact: Some(input_name),
                    stage: None,
                    output: None,
                    stream: None,
                }],
                frames: 1,
                count: 1,
            }),
        });
        stages.push(StageSpec {
            name: second,
            depends_on: vec![first.clone()],
            executor: StageExecutor {
                kind: "gpu".to_string(),
                target: Some("any".to_string()),
            },
            command: Vec::new(),
            inputs: BTreeMap::new(),
            outputs: vec!["OUT".to_string()],
            timeout_ms: Some(spec.timeout_ms),
            gpu: Some(GpuStage {
                listing,
                payloads: vec![StageRef {
                    artifact: None,
                    stage: Some(first),
                    output: Some("OUT".to_string()),
                    stream: Some("X".to_string()),
                }],
                frames: 1,
                count: 1,
            }),
        });
    }
    let bundle = Bundle {
        schema: BUNDLE_SCHEMA.to_string(),
        name: spec.name.clone(),
        version: "1".to_string(),
        description: "batch GPU upscale chain with intermediate artifact handoff".to_string(),
        artifacts,
        stages,
    };
    bundle.validate()?;
    let bytes = serde_json::to_vec_pretty(&bundle).map_err(|e| e.to_string())?;
    let path = spec.out_dir.join("bundle.json");
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
    /// Binary descriptor metadata from a GPU result. Native file results leave
    /// these unset.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub kind: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub shape: Option<Vec<u64>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub frame: Option<u64>,
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
    /// Native stage program path on the executor host. GPU-only bundles do not
    /// need one.
    pub program: Option<String>,
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
        kind: None,
        shape: None,
        frame: None,
    })
}

/// Compare an on-disk output against a ledger descriptor. The bytes are
/// re-hashed; the descriptor metadata is copied from the expected record so a
/// GPU output keeps its kind/shape/frame.
fn output_matches(path: &Path, want: &FileInfo) -> bool {
    let Ok(mut got) = file_info(path) else {
        return false;
    };
    got.kind = want.kind.clone();
    got.shape = want.shape.clone();
    got.frame = want.frame.clone();
    &got == want
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
                    ..Default::default()
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
            let mut got = file_info(&path)?;
            got.kind = expected.kind.clone();
            got.shape = expected.shape.clone();
            got.frame = expected.frame.clone();
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
    let frames = header.get("frames").and_then(|x| x.as_u64()).unwrap_or(1);
    if frames != 1 {
        return Err(format!(
            "stage {:?} returned {frames} frames; the ledger keys outputs by name, so v1 accepts one frame per GPU stage",
            stage.name
        ));
    }
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
        let kind = desc
            .get("kind")
            .and_then(|x| x.as_str())
            .map(str::to_string);
        let shape = match desc.get("shape").and_then(|x| x.as_array()) {
            Some(dims) => Some(
                dims.iter()
                    .map(|d| d.as_u64())
                    .collect::<Option<Vec<u64>>>()
                    .ok_or_else(|| format!("output {name:?} has a non-integer shape"))?,
            ),
            None => None,
        };
        if let Some(kind) = &kind {
            if shape.as_ref().map(|s| s.is_empty()).unwrap_or(true) {
                return Err(format!("output {name:?} kind {kind:?} has no shape"));
            }
        }
        let path = stage_output_path(ledger_dir, &stage.name, name)?;
        write_atomic(&path, bytes)?;
        infos.insert(
            name.to_string(),
            FileInfo {
                sha256: sha,
                bytes: bytes.len() as u64,
                kind,
                shape,
                frame: desc.get("frame").and_then(|x| x.as_u64()),
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

struct StagePlan {
    /// Resolved argv for native stages; a synthetic plan descriptor for GPU
    /// stages. The ledger records it so a reused stage has a stable identity.
    command: Vec<String>,
    inputs: BTreeMap<String, FileInfo>,
    body: Vec<u8>,
    header: Json,
    timeout_ms: u64,
}

fn plan_native_stage(
    loaded: &LoadedBundle,
    stage: &StageSpec,
    ledger: &Ledger,
    ledger_dir: &Path,
    program: &str,
    cfg: &RunConfig,
) -> Result<StagePlan, String> {
    let inputs = prepare_inputs(loaded, stage, ledger, ledger_dir)?;
    let input_infos: BTreeMap<String, FileInfo> = inputs
        .iter()
        .map(|i| (i.key.clone(), i.info.clone()))
        .collect();
    let input_paths: BTreeMap<String, String> = inputs
        .iter()
        .map(|i| (i.key.clone(), i.path.clone()))
        .collect();
    let command = resolve_command(&stage.command, program, &input_paths)?;
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
    let header = json!({
        "type": "submit",
        "command": command,
        "outputs": stage.outputs,
        "inputs": descs,
        "timeout_ms": timeout_ms,
    });
    Ok(StagePlan {
        command,
        inputs: input_infos,
        body,
        header,
        timeout_ms,
    })
}

fn decode_binary_stream(info: &FileInfo, bytes: &[u8]) -> Result<Json, String> {
    if info.frame.unwrap_or(0) != 0 {
        return Err(format!(
            "GPU output frame {:?} cannot be used as a v1 payload; only frame 0 is supported",
            info.frame
        ));
    }
    let kind = info
        .kind
        .as_deref()
        .ok_or("GPU output has no kind; cannot build a payload")?;
    let shape = info
        .shape
        .as_ref()
        .ok_or("GPU output has no shape; cannot build a payload")?;
    let count = shape.iter().try_fold(1usize, |acc, &dim| {
        acc.checked_mul(dim as usize)
            .ok_or_else(|| "GPU output shape overflows".to_string())
    })?;
    let width = match kind {
        "F32" => 4usize,
        "F64" | "I64" => 8,
        "U8" | "Bool" => 1,
        other => return Err(format!("GPU output kind {other:?} has no payload encoding")),
    };
    let expected = count
        .checked_mul(width)
        .ok_or_else(|| "GPU output byte count overflows".to_string())?;
    if bytes.len() != expected {
        return Err(format!(
            "GPU output is {} bytes; kind {kind} shape {shape:?} needs {expected}",
            bytes.len()
        ));
    }
    let data: Vec<Json> = match kind {
        "F32" => bytes
            .chunks_exact(4)
            .map(|c| {
                let v = f32::from_le_bytes([c[0], c[1], c[2], c[3]]);
                if !v.is_finite() {
                    return Err("GPU F32 output contains NaN or infinity".to_string());
                }
                Ok(json!(v))
            })
            .collect::<Result<_, _>>()?,
        "F64" => bytes
            .chunks_exact(8)
            .map(|c| {
                let v = f64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]);
                if !v.is_finite() {
                    return Err("GPU F64 output contains NaN or infinity".to_string());
                }
                Ok(json!(v))
            })
            .collect::<Result<_, _>>()?,
        "I64" => bytes
            .chunks_exact(8)
            .map(|c| {
                let v = i64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]);
                Ok(json!(v.to_string()))
            })
            .collect::<Result<_, String>>()?,
        "U8" => bytes.iter().map(|b| json!(b)).collect(),
        "Bool" => bytes.iter().map(|b| json!(*b != 0)).collect(),
        _ => unreachable!(),
    };
    Ok(json!({"kind": kind, "shape": shape, "data": data}))
}

fn plan_gpu_stage(
    loaded: &LoadedBundle,
    stage: &StageSpec,
    ledger: &Ledger,
    ledger_dir: &Path,
    cfg: &RunConfig,
) -> Result<StagePlan, String> {
    let gpu = stage
        .gpu
        .as_ref()
        .ok_or_else(|| format!("gpu stage {:?} has no gpu payload", stage.name))?;
    let listing_artifact = gpu
        .listing
        .artifact
        .as_deref()
        .ok_or_else(|| format!("gpu stage {:?} listing needs an artifact", stage.name))?;
    let (listing_bytes, listing_ref) = loaded.artifact_bytes(listing_artifact)?;
    let listing = String::from_utf8(listing_bytes)
        .map_err(|e| format!("listing artifact {listing_artifact:?} is not UTF-8: {e}"))?;

    let mut streams = serde_json::Map::new();
    let mut inputs = BTreeMap::new();
    inputs.insert(
        "listing".to_string(),
        FileInfo {
            sha256: listing_ref.sha256.clone(),
            bytes: listing_ref.bytes,
            ..Default::default()
        },
    );
    let mut plan = vec![
        "gpu".to_string(),
        stage.target().to_string(),
        format!("listing:{}", listing_ref.sha256),
    ];
    for (index, payload) in gpu.payloads.iter().enumerate() {
        let info = if let Some(artifact) = &payload.artifact {
            let (bytes, artifact_ref) = loaded.artifact_bytes(artifact)?;
            let value: Json = serde_json::from_slice(&bytes)
                .map_err(|e| format!("payload artifact {artifact:?} is not JSON: {e}"))?;
            let object = value
                .as_object()
                .ok_or_else(|| format!("payload artifact {artifact:?} must be a JSON object"))?;
            let source = object
                .get("streams")
                .and_then(|x| x.as_object())
                .ok_or_else(|| format!("payload artifact {artifact:?} needs a streams object"))?;
            for (name, stream) in source {
                if streams.contains_key(name) {
                    return Err(format!(
                        "gpu stage {:?} payload stream {name:?} is supplied twice",
                        stage.name
                    ));
                }
                streams.insert(name.clone(), stream.clone());
            }
            plan.push(format!("payload{index}:artifact:{}", artifact_ref.sha256));
            FileInfo {
                sha256: artifact_ref.sha256,
                bytes: artifact_ref.bytes,
                ..Default::default()
            }
        } else {
            let src = payload.stage.as_deref().ok_or_else(|| {
                format!(
                    "gpu stage {:?} payload {index} needs artifact or stage",
                    stage.name
                )
            })?;
            let output = payload.output.as_deref().ok_or_else(|| {
                format!("gpu stage {:?} payload {index} needs an output", stage.name)
            })?;
            let stream = payload.stream.as_deref().unwrap_or(output);
            let record = ledger.stages.get(src).ok_or_else(|| {
                format!(
                    "gpu stage {:?} needs {src:?}.{output:?} but {src:?} has not run",
                    stage.name
                )
            })?;
            let expected = record.outputs.get(output).ok_or_else(|| {
                format!(
                    "gpu stage {:?} needs {src:?}.{output:?}, which is missing",
                    stage.name
                )
            })?;
            let path = stage_output_path(ledger_dir, src, output)?;
            let mut got = file_info(&path)?;
            got.kind = expected.kind.clone();
            got.shape = expected.shape.clone();
            got.frame = expected.frame.clone();
            if &got != expected {
                return Err(format!(
                    "GPU output {} has changed since stage {src:?}",
                    path.display()
                ));
            }
            let bytes = fs::read(&path).map_err(|e| format!("read {}: {e}", path.display()))?;
            let value = decode_binary_stream(expected, &bytes)?;
            if streams.contains_key(stream) {
                return Err(format!(
                    "gpu stage {:?} payload stream {stream:?} is supplied twice",
                    stage.name
                ));
            }
            streams.insert(stream.to_string(), value);
            plan.push(format!("payload{index}:stage:{}", expected.sha256));
            got
        };
        inputs.insert(format!("payload{index}"), info);
    }
    if streams.is_empty() {
        return Err(format!("gpu stage {:?} has no payload streams", stage.name));
    }
    let body = serde_json::to_vec(&json!({
        "listing": listing,
        "payload": {"streams": streams},
    }))
    .map_err(|e| e.to_string())?;
    let timeout_ms = cfg
        .timeout_ms
        .or(stage.timeout_ms)
        .unwrap_or(600_000)
        .max(1);
    let header = json!({
        "type": "submit",
        "outputs": stage.outputs,
        "frames": gpu.frames,
        "output_format": "binary",
    });
    Ok(StagePlan {
        command: plan,
        inputs,
        body,
        header,
        timeout_ms,
    })
}

fn executor_matches(capability: &Json, stage: &StageSpec) -> bool {
    let kind = capability
        .get("kind")
        .and_then(|x| x.as_str())
        .unwrap_or("");
    if kind != stage.executor.kind {
        return false;
    }
    if !stage.is_gpu() {
        return true;
    }
    let target = capability
        .get("target")
        .and_then(|x| x.as_str())
        .unwrap_or("");
    stage.target() == "any" || target == stage.target()
}

fn select_executor<'a>(
    cfg: &'a RunConfig,
    stage: &StageSpec,
    index: usize,
) -> Result<&'a ExecutorEndpoint, String> {
    if let Some(addr) = cfg.placement.get(&stage.name) {
        let endpoint = cfg
            .executors
            .iter()
            .find(|e| &e.addr == addr)
            .ok_or_else(|| {
                format!(
                    "stage {:?} is placed on {addr}, which is not in the executor list",
                    stage.name
                )
            })?;
        if !executor_matches(&endpoint.capability, stage) {
            return Err(format!(
                "stage {:?} needs executor kind {:?} target {:?}, but {addr} advertises {}",
                stage.name,
                stage.executor.kind,
                stage.target(),
                endpoint.capability
            ));
        }
        return Ok(endpoint);
    }
    let candidates: Vec<&ExecutorEndpoint> = cfg
        .executors
        .iter()
        .filter(|e| executor_matches(&e.capability, stage))
        .collect();
    if candidates.is_empty() {
        return Err(format!(
            "no executor matches stage {:?} (kind={}, target={})",
            stage.name,
            stage.executor.kind,
            stage.target()
        ));
    }
    Ok(candidates[index % candidates.len()])
}

pub fn run_bundle(
    loaded: &LoadedBundle,
    ledger_dir: &Path,
    cfg: &RunConfig,
) -> Result<Json, String> {
    if cfg.executors.is_empty() && cfg.placement.is_empty() {
        return Err("run needs at least one --executor or --place stage=addr".to_string());
    }
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
        let plan = if stage.is_gpu() {
            plan_gpu_stage(loaded, stage, &ledger, ledger_dir, cfg)?
        } else {
            let program = cfg
                .program
                .as_deref()
                .ok_or_else(|| format!("native stage {:?} needs --program FILE", stage.name))?;
            plan_native_stage(loaded, stage, &ledger, ledger_dir, program, cfg)?
        };
        let prior = ledger.stages.get(name).cloned().unwrap_or_default();
        if !cfg.force
            && prior.state == "done"
            && prior.command == plan.command
            && prior.inputs == plan.inputs
            && stage.outputs.iter().all(|out| {
                prior
                    .outputs
                    .get(out)
                    .and_then(|want| {
                        stage_output_path(ledger_dir, name, out)
                            .ok()
                            .map(|p| output_matches(&p, want))
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
        let endpoint = select_executor(cfg, stage, index)?;
        let addr = endpoint.addr.clone();
        let executor =
            RemoteExecutor::new(endpoint.capability.clone(), addr.clone(), cfg.token.clone());
        let mut spec = plan.header.clone();
        if let Some(t) = &cfg.token {
            spec["auth"] = json!(t);
        }
        let submitted = executor
            .submit(&spec, &plan.body)
            .map_err(|(code, msg)| format!("submit {addr}: {code}: {msg}"))?;
        let job_id = submitted
            .get("job_id")
            .and_then(|x| x.as_str())
            .ok_or("submit returned no job_id")?
            .to_string();
        let job = poll_job(&executor, &job_id, plan.timeout_ms.saturating_add(60_000))?;
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
            record.command = plan.command;
            record.inputs = plan.inputs;
            record.finished_ms = now_ms();
            ledger.stages.insert(name.clone(), record);
            ledger.save(ledger_dir)?;
            return Err(format!("stage {name:?} ended in state {state}: {job}"));
        }
        let (header, result_body) = executor
            .result(&job_id)
            .map_err(|(code, msg)| format!("job-result {addr}: {code}: {msg}"))?;
        if matches!(
            header.get("output_format").and_then(|x| x.as_str()),
            Some("binary") | Some("files")
        ) {
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
            command: plan.command,
            inputs: plan.inputs,
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
            gpu: None,
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

    fn gpu_stage(name: &str, deps: &[&str], target: &str, payloads: Vec<StageRef>) -> StageSpec {
        StageSpec {
            name: name.to_string(),
            depends_on: deps.iter().map(|s| s.to_string()).collect(),
            executor: StageExecutor {
                kind: "gpu".to_string(),
                target: Some(target.to_string()),
            },
            command: Vec::new(),
            inputs: BTreeMap::new(),
            outputs: vec!["OUT".to_string()],
            timeout_ms: Some(60_000),
            gpu: Some(GpuStage {
                listing: StageRef {
                    artifact: Some("listing".to_string()),
                    stage: None,
                    output: None,
                    stream: None,
                },
                payloads,
                frames: 1,
                count: 1,
            }),
        }
    }

    fn artifact_map(names: &[&str]) -> BTreeMap<String, ArtifactRef> {
        names
            .iter()
            .map(|name| {
                (
                    name.to_string(),
                    ArtifactRef {
                        sha256: sha256_hex(name.as_bytes()),
                        bytes: name.len() as u64,
                        path: format!("blobs/{}", sha256_hex(name.as_bytes())),
                    },
                )
            })
            .collect()
    }

    #[test]
    fn gpu_stage_accepts_artifact_and_stage_payloads() {
        let bundle = Bundle {
            schema: BUNDLE_SCHEMA.to_string(),
            name: "gpu-chain".to_string(),
            version: "1".to_string(),
            description: String::new(),
            artifacts: artifact_map(&["listing", "input"]),
            stages: vec![
                gpu_stage(
                    "a",
                    &[],
                    "cuda",
                    vec![StageRef {
                        artifact: Some("input".to_string()),
                        stage: None,
                        output: None,
                        stream: None,
                    }],
                ),
                gpu_stage(
                    "b",
                    &["a"],
                    "any",
                    vec![StageRef {
                        artifact: None,
                        stage: Some("a".to_string()),
                        output: Some("OUT".to_string()),
                        stream: Some("X".to_string()),
                    }],
                ),
            ],
        };
        bundle.validate().unwrap();
        assert_eq!(bundle.stage_order().unwrap(), vec!["a", "b"]);
    }

    #[test]
    fn gpu_stage_rejects_bad_target_and_unknown_payloads() {
        let mut bundle = Bundle {
            schema: BUNDLE_SCHEMA.to_string(),
            name: "gpu-chain".to_string(),
            version: "1".to_string(),
            description: String::new(),
            artifacts: artifact_map(&["listing", "input"]),
            stages: vec![gpu_stage(
                "a",
                &[],
                "vulkan",
                vec![StageRef {
                    artifact: Some("input".to_string()),
                    stage: None,
                    output: None,
                    stream: None,
                }],
            )],
        };
        assert!(bundle.validate().is_err());
        bundle.stages[0].executor.target = Some("cuda".to_string());
        bundle.stages[0].gpu.as_mut().unwrap().payloads[0].artifact = Some("missing".to_string());
        assert!(bundle.validate().is_err());
        bundle.stages[0].gpu.as_mut().unwrap().payloads[0].artifact = Some("input".to_string());
        bundle.stages[0]
            .gpu
            .as_mut()
            .unwrap()
            .payloads
            .push(StageRef {
                artifact: None,
                stage: Some("a".to_string()),
                output: Some("OUT".to_string()),
                stream: Some("X".to_string()),
            });
        assert!(bundle.validate().is_err());
    }

    #[test]
    fn gpu_stage_rejects_a_command_and_inputs() {
        let mut bundle = Bundle {
            schema: BUNDLE_SCHEMA.to_string(),
            name: "gpu-chain".to_string(),
            version: "1".to_string(),
            description: String::new(),
            artifacts: artifact_map(&["listing", "input"]),
            stages: vec![gpu_stage(
                "a",
                &[],
                "cuda",
                vec![StageRef {
                    artifact: Some("input".to_string()),
                    stage: None,
                    output: None,
                    stream: None,
                }],
            )],
        };
        bundle.stages[0].command = vec!["{{program}}".to_string()];
        assert!(bundle.validate().is_err());
    }

    #[test]
    fn executor_targets_are_enforced() {
        let stage = gpu_stage("a", &[], "cuda", Vec::new());
        let cuda = json!({"kind": "gpu", "target": "cuda"});
        let hip = json!({"kind": "gpu", "target": "hip"});
        let native = json!({"kind": "native", "target": "process"});
        assert!(executor_matches(&cuda, &stage));
        assert!(!executor_matches(&hip, &stage));
        assert!(!executor_matches(&native, &stage));
        let mut any = stage.clone();
        any.executor.target = Some("any".to_string());
        assert!(executor_matches(&cuda, &any));
        assert!(executor_matches(&hip, &any));

        let cfg = RunConfig {
            program: None,
            executors: vec![
                ExecutorEndpoint {
                    addr: "cuda:1".to_string(),
                    capability: cuda,
                },
                ExecutorEndpoint {
                    addr: "hip:1".to_string(),
                    capability: hip,
                },
            ],
            placement: BTreeMap::from([("a".to_string(), "hip:1".to_string())]),
            token: None,
            force: false,
            timeout_ms: None,
            compare: Vec::new(),
        };
        assert!(select_executor(&cfg, &stage, 0).is_err());
        let auto = RunConfig {
            placement: BTreeMap::new(),
            ..cfg
        };
        assert_eq!(select_executor(&auto, &stage, 0).unwrap().addr, "cuda:1");
    }

    #[test]
    fn binary_f32_output_decodes_to_a_payload_stream() {
        let mut bytes = Vec::new();
        for v in [1.5f32, 2.5, 3.5, 4.5] {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        let info = FileInfo {
            sha256: sha256_hex(&bytes),
            bytes: bytes.len() as u64,
            kind: Some("F32".to_string()),
            shape: Some(vec![2, 2]),
            frame: Some(0),
        };
        let stream = decode_binary_stream(&info, &bytes).unwrap();
        assert_eq!(stream["kind"], "F32");
        assert_eq!(stream["shape"], json!([2, 2]));
        assert_eq!(stream["data"], json!([1.5, 2.5, 3.5, 4.5]));
        let mut short = bytes.clone();
        short.pop();
        assert!(decode_binary_stream(&info, &short).is_err());
        let mut bad = info.clone();
        bad.kind = Some("T".to_string());
        assert!(decode_binary_stream(&bad, &bytes).is_err());
    }

    #[test]
    fn generated_payload_doubles_exactly() {
        let payload = generated_payload(3, 4, 2, 7);
        let stream = &payload["streams"]["X"];
        assert_eq!(stream["shape"], json!([3, 4, 2]));
        let data = stream["data"].as_array().unwrap();
        assert_eq!(data.len(), 24);
        for value in data {
            let v = value.as_f64().unwrap();
            assert_eq!(v.fract(), 0.0);
            assert!(v * 2.0 <= 502.0);
        }
    }

    #[test]
    fn gpu_bundle_builder_names_and_orders_items() {
        let root = std::env::temp_dir().join(format!(
            "jotnar-gpu-bundle-test-{}-{}",
            std::process::id(),
            now_ms()
        ));
        fs::create_dir_all(&root).unwrap();
        let listing = root.join("double.asm");
        fs::write(&listing, "CONFIG warm_frame 1\nIN X\nOUT = ADD(X, X)\n").unwrap();
        let loaded = build_gpu_bundle(&GpuBundleSpec {
            listing,
            input: None,
            generate: Some((2, 3, 1)),
            items: 2,
            out_dir: root.join("bundle"),
            name: "gpu-test".to_string(),
            timeout_ms: 60_000,
        })
        .unwrap();
        assert_eq!(
            loaded.bundle.stage_order().unwrap(),
            vec!["gpu-a-00", "gpu-b-00", "gpu-a-01", "gpu-b-01"]
        );
        let b1 = loaded
            .bundle
            .stages
            .iter()
            .find(|s| s.name == "gpu-b-01")
            .unwrap();
        assert_eq!(b1.executor.target.as_deref(), Some("any"));
        assert_eq!(
            b1.gpu.as_ref().unwrap().payloads[0].stream.as_deref(),
            Some("X")
        );
        assert_eq!(loaded.bundle.artifacts.len(), 3);
        let _ = fs::remove_dir_all(&root);
    }
}
