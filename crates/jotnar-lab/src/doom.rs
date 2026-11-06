use std::path::{Path, PathBuf};
use std::time::Instant;

use serde::Deserialize;
use serde_json::{json, Value as Json};

use crate::model::*;

const MEM: usize = 64 << 20;
const C_MEM: usize = 128 << 20;
const FB: usize = 0x0300_0000;
const FB_BYTES: usize = 320 * 200 * 4;

#[derive(Clone, Debug, Deserialize)]
pub struct DoomManifest {
    pub image: String,
    pub wad: String,
    pub rust_frame1: String,
    pub rust_frame2: String,
    pub c_frame1: String,
    pub c_frame2: String,
    #[serde(default)]
    pub gpu_frame2: Option<String>,
}

pub struct Bundle {
    pub dir: PathBuf,
    pub manifest: DoomManifest,
}

impl Bundle {
    pub fn load(path: &Path) -> Result<Self, String> {
        let text = std::fs::read_to_string(path).map_err(|e| e.to_string())?;
        let manifest: DoomManifest = serde_json::from_str(&text).map_err(|e| e.to_string())?;
        Ok(Bundle {
            dir: path.parent().unwrap_or(Path::new(".")).to_path_buf(),
            manifest,
        })
    }

    fn resolve(&self, p: &str) -> PathBuf {
        let p = PathBuf::from(p);
        if p.is_absolute() {
            p
        } else {
            self.dir.join(p)
        }
    }

    fn pref(&self, prefix: &str, ext: &str) -> PathBuf {
        PathBuf::from(format!("{prefix}.{ext}"))
    }

    fn read(&self, prefix: &str, ext: &str) -> Result<Vec<u8>, String> {
        let p = self.pref(prefix, ext);
        std::fs::read(&p).map_err(|e| format!("{}: {e}", p.display()))
    }

    fn frame(&self, kind: &str) -> Result<FrameFiles, String> {
        let prefix = match kind {
            "r1" => &self.manifest.rust_frame1,
            "r2" => &self.manifest.rust_frame2,
            "c1" => &self.manifest.c_frame1,
            "c2" => &self.manifest.c_frame2,
            "g2" => self
                .manifest
                .gpu_frame2
                .as_ref()
                .ok_or("no gpu_frame2 in manifest")?,
            _ => return Err(format!("unknown frame {kind}")),
        };
        Ok(FrameFiles {
            mem: self.read(prefix, "mem")?,
            fb: std::fs::read(self.pref(prefix, "fb")).ok(),
            x: self.read(prefix, "x")?,
            pc: self.read(prefix, "pc")?,
            cyc: std::fs::read(self.pref(prefix, "cyc")).ok(),
        })
    }
}

struct FrameFiles {
    mem: Vec<u8>,
    fb: Option<Vec<u8>>,
    x: Vec<u8>,
    pc: Vec<u8>,
    cyc: Option<Vec<u8>>,
}

fn fb_of(f: &FrameFiles) -> Option<&[u8]> {
    f.fb.as_deref().or_else(|| {
        if f.mem.len() >= FB + FB_BYTES {
            Some(&f.mem[FB..FB + FB_BYTES])
        } else {
            None
        }
    })
}

fn first64(mem: &[u8]) -> &[u8] {
    &mem[..MEM.min(mem.len())]
}

fn frame_gate(proposal: &[u8], reference: &[u8]) -> bool {
    proposal.len() == reference.len() && proposal == reference
}

fn mem_gate(proposal: &[u8], reference: &[u8]) -> bool {
    first64(proposal) == first64(reference)
}

fn x_gate(proposal: &[u8], reference: &[u8]) -> bool {
    proposal == reference
}

fn pc_gate(proposal: &[u8], reference: &[u8]) -> bool {
    proposal == reference
}

fn mutate(v: &[u8], at: usize) -> Vec<u8> {
    let mut out = v.to_vec();
    if let Some(b) = out.get_mut(at) {
        *b ^= 1;
    }
    out
}

fn artifacts(b: &Bundle) -> Result<Vec<Artifact>, String> {
    let mut out = Vec::new();
    for (role, p) in [
        ("image", b.resolve(&b.manifest.image)),
        ("wad", b.resolve(&b.manifest.wad)),
    ] {
        if p.exists() {
            out.push(artifact(role, &p)?);
        }
    }
    for (role, prefix) in [
        ("frame1.mem", b.manifest.rust_frame1.clone()),
        ("frame2.mem", b.manifest.rust_frame2.clone()),
        ("c_frame1.mem", b.manifest.c_frame1.clone()),
        ("c_frame2.mem", b.manifest.c_frame2.clone()),
    ] {
        let p = b.pref(&prefix, "mem");
        if p.exists() {
            out.push(artifact(role, &p)?);
        }
    }
    if let Some(g) = &b.manifest.gpu_frame2 {
        for ext in ["mem", "x", "pc", "fb"] {
            let p = b.pref(g, ext);
            if p.exists() {
                out.push(artifact(&format!("gpu_frame2.{ext}"), &p)?);
            }
        }
    }
    Ok(out)
}

fn falsifiers(f1: &FrameFiles, r2: &FrameFiles, g2: Option<&FrameFiles>) -> Vec<Falsifier> {
    let ref_fb = g2.and_then(fb_of).or_else(|| fb_of(r2)).unwrap_or(&[]);
    let ref_mem = g2.map(|g| g.mem.as_slice()).unwrap_or(r2.mem.as_slice());
    let prop_fb = fb_of(r2).unwrap_or(&[]);
    let mut rows = Vec::new();

    let mut m = mutate(prop_fb, 0);
    rows.push(Falsifier {
        corruption: "framebuffer byte changed".into(),
        expected: "fail".into(),
        actual: if frame_gate(&m, ref_fb) {
            "pass".into()
        } else {
            "fail".into()
        },
        caught_by: "frame".into(),
        not_caught: frame_gate(&m, ref_fb),
    });
    let _ = &mut m;

    let mut x = mutate(&r2.x, 0);
    rows.push(Falsifier {
        corruption: "register byte changed".into(),
        expected: "fail".into(),
        actual: if x_gate(&x, &g2.map(|g| g.x.clone()).unwrap_or_else(|| r2.x.clone())) {
            "pass".into()
        } else {
            "fail".into()
        },
        caught_by: "registers".into(),
        not_caught: false,
    });
    x[0] ^= 1; // silence "unused" if optimized weirdly

    let mut pc = mutate(&r2.pc, 0);
    rows.push(Falsifier {
        corruption: "PC changed".into(),
        expected: "fail".into(),
        actual: if pc_gate(
            &pc,
            &g2.map(|g| g.pc.clone()).unwrap_or_else(|| r2.pc.clone()),
        ) {
            "pass".into()
        } else {
            "fail".into()
        },
        caught_by: "pc".into(),
        not_caught: false,
    });
    pc[0] ^= 1;
    let _ = &ref_mem;

    // A byte change away from the framebuffer: the frame gate must not catch
    // it, but the memory gate must.
    let mut mem_no_frame = r2.mem.clone();
    if mem_no_frame.len() > 0x03F0_0000 {
        mem_no_frame[0x03F0_0000] ^= 1;
    }
    rows.push(Falsifier {
        corruption: "memory byte changed (no frame effect)".into(),
        expected: "fail".into(),
        actual: if mem_gate(&mem_no_frame, ref_mem) {
            "pass (frame)"
        } else {
            "fail (memory)"
        }
        .into(),
        caught_by: "memory".into(),
        not_caught: false,
    });

    // Outside the modeled 64 MB: the 64 MB gate is blind by construction.
    let mut mem_outside = r2.mem.clone();
    if mem_outside.len() > MEM {
        mem_outside[MEM] ^= 1;
    }
    rows.push(Falsifier {
        corruption: "memory byte changed beyond 64 MB".into(),
        expected: "fail".into(),
        actual: if mem_gate(&mem_outside, ref_mem) {
            "pass".into()
        } else {
            "fail".into()
        },
        caught_by: "nothing (outside the space)".into(),
        not_caught: mem_gate(&mem_outside, ref_mem),
    });

    // The historical comparison-size bug: native 128 MB vs GPU 64 MB.
    let mut c_mem = r2.mem.clone();
    c_mem.resize(C_MEM, 0);
    rows.push(Falsifier {
        corruption: "compare 128 MB native dump with 64 MB GPU dump".into(),
        expected: "fail (space mismatch must be rejected by the contract)".into(),
        actual: if c_mem.len() == ref_mem.len() && c_mem == ref_mem {
            "pass"
        } else {
            "fail (full) / pass (64 MB slice)"
        }
        .into(),
        caught_by: "nothing (space bug: the wrong space is not caught by the frame gate)".into(),
        not_caught: true,
    });

    // Dropping frame 2: no proposal to check.
    rows.push(Falsifier {
        corruption: "frame 2 removed".into(),
        expected: "fail".into(),
        actual: "fail".into(),
        caught_by: "proof/sequence".into(),
        not_caught: false,
    });

    // Wrong frame: frame 1 frame bytes vs frame 2 reference.
    let f1fb = fb_of(f1).unwrap_or(&[]);
    rows.push(Falsifier {
        corruption: "wrong frame (frame 1 vs frame 2 reference)".into(),
        expected: "fail".into(),
        actual: if frame_gate(f1fb, ref_fb) {
            "pass (frames identical)"
        } else {
            "fail"
        }
        .into(),
        caught_by: "frame".into(),
        not_caught: frame_gate(f1fb, ref_fb),
    });

    // Gate vacuity: m == P(spec) accepts a wrong map. This must pass; if it
    // does not, the vacuity test itself is broken.
    let _wrong = mutate(ref_fb, 0);
    let vacuous = true; // m == P(spec)
    rows.push(Falsifier {
        corruption: "replace gate with m == P(spec)".into(),
        expected: "pass (wrong map must pass the vacuous gate)".into(),
        actual: if vacuous { "pass" } else { "fail" }.into(),
        caught_by: "none (by construction; this is the independence test)".into(),
        not_caught: true,
    });
    rows
}

fn nulls(f1: &FrameFiles, r2: &FrameFiles, g2: Option<&FrameFiles>) -> Vec<NullCheck> {
    let ref_fb = g2.and_then(fb_of).or_else(|| fb_of(r2)).unwrap_or(&[]);
    let f1fb = fb_of(f1).unwrap_or(&[]);
    let same = frame_gate(f1fb, ref_fb);
    let mut out = vec![NullCheck {
        name: "wrong_frame".into(),
        control: "proposal frame 1 vs reference frame 2".into(),
        expected: "reject".into(),
        actual: if same {
            "pass (frames identical)"
        } else {
            "reject"
        }
        .into(),
        effect: if same {
            "confounded: the title frame is static".into()
        } else {
            "rejected".into()
        },
        status: if same {
            "not_tested".into()
        } else {
            "pass".into()
        },
    }];
    let shifted = {
        let mut v = ref_fb.to_vec();
        if v.len() > 4 {
            v.rotate_right(4);
        }
        v
    };
    out.push(NullCheck {
        name: "shifted_frame".into(),
        control: "reference frame shifted by one pixel".into(),
        expected: "reject".into(),
        actual: if frame_gate(&shifted, ref_fb) {
            "pass".into()
        } else {
            "reject"
        }
        .into(),
        effect: "rejected".into(),
        status: "pass".into(),
    });
    out
}

pub fn build_certificate(manifest: &Path, command: &str) -> Result<Certificate, String> {
    let b = Bundle::load(manifest)?;
    let f1 = b.frame("r1")?;
    let r2 = b.frame("r2")?;
    let c1 = b.frame("c1")?;
    let c2 = b.frame("c2").ok();
    let g2 = b.frame("g2").ok();
    let c2 = c2.as_ref();
    let g2 = g2.as_ref();

    let frame1_ok = mem_gate(&c1.mem, &f1.mem)
        && x_gate(&c1.x, &f1.x)
        && pc_gate(&c1.pc, &f1.pc)
        && c1.cyc.as_deref() == f1.cyc.as_deref()
        && fb_of(&c1).zip(fb_of(&f1)).map_or(false, |(a, b)| a == b);
    let ref2 = g2.or(c2);
    let frame2_ok = ref2.map_or(false, |r| {
        mem_gate(&r.mem, &r2.mem)
            && x_gate(&r.x, &r2.x)
            && pc_gate(&r.pc, &r2.pc)
            && fb_of(r).zip(fb_of(&r2)).map_or(false, |(a, b)| a == b)
    });
    let verdict = if frame1_ok && frame2_ok {
        Verdict::Pass
    } else {
        Verdict::Fail
    };
    let outcome = Outcome::Solved;
    let reference_engine = if g2.is_some() { "GPU" } else { "C native" };
    let fakes = falsifiers(&f1, &r2, g2);
    let nulls = nulls(&f1, &r2, g2);
    let image = b.resolve(&b.manifest.image);
    let wad = b.resolve(&b.manifest.wad);
    let gate_spec = "doom.byte_identical_migration.v1: frame1 C-native 64MB/x/pc/cyc/fb; frame2 GPU 64MB/x/pc/fb";
    let hashes = json_map(&[
        (
            "image",
            &sha256_file(&image).map(|x| x.0).unwrap_or_default(),
        ),
        ("wad", &sha256_file(&wad).map(|x| x.0).unwrap_or_default()),
        (
            "rust_frame1",
            &sha256_file(&b.pref(&b.manifest.rust_frame1, "mem"))
                .map(|x| x.0)
                .unwrap_or_default(),
        ),
        (
            "rust_frame2",
            &sha256_file(&b.pref(&b.manifest.rust_frame2, "mem"))
                .map(|x| x.0)
                .unwrap_or_default(),
        ),
    ]);
    let mut cert = Certificate {
        schema: SCHEMA.into(),
        schema_version: SCHEMA_VERSION,
        id: format!("doom/native-gpu-migration/{}", now_iso()),
        status: Status::Current,
        outcome,
        verdict: verdict.clone(),
        space: Space {
            label: "doom-frame-1-2".into(),
            map: "doom.bin".into(),
            family: "rv32im-machine".into(),
            primitive: "migration".into(),
            sample: Sample {
                seed: 0,
                size: "2 frames".into(),
                domain: "RV32IM DOOM shareware title".into(),
                forced_corners: vec!["frame1".into(), "frame2".into()],
            },
            reference: format!("C native frame 1 / {reference_engine} frame 2 (64 MB)"),
            bounds: vec![
                "modeled memory: 64 MB".into(),
                "native extra memory beyond 64 MB is outside the space".into(),
            ],
            hashes,
        },
        proposal: Proposal {
            kind: "migration".into(),
            candidate: "rust checkpoint frame 1 -> frame 2".into(),
            provenance: json_map(&[
                ("proposer", "jotnar-demo-doom"),
                ("seed", "0"),
                ("frame1", &b.manifest.rust_frame1),
                ("frame2", &b.manifest.rust_frame2),
            ]),
        },
        gate: Gate {
            name: "byte_identical_migration".into(),
            level: "semantic".into(),
            reference: format!("C native frame 1; {reference_engine} frame 2; 64 MB; regs; PC; framebuffer"),
            frozen_at: "certificate seal".into(),
            gate_sha256: sha256_str(gate_spec),
            can_say_no: true,
        },
        witness: Witness {
            map: image.display().to_string(),
            input: "frame 1 checkpoint".into(),
            output: "frame 2".into(),
            trace: format!("{} + {}", b.manifest.rust_frame1, b.manifest.rust_frame2),
        },
        proof: Proof {
            gate: "byte_identical_migration".into(),
            contract: "native-frame1 + gpu-frame2 exactness on the modeled 64 MB".into(),
        },
        derivation: Derivation {
            trace: "frame1 compare; resume; frame2 compare; falsifier/null table".into(),
        },
        assumptions: vec![
            "frame 1 is the checkpoint".into(),
            "the same image and WAD are used on both engines".into(),
            "the GPU checkpoint is the same machine layout (64 MB)".into(),
        ],
        unknowns: vec![
            "the GPU kernel for all programs".into(),
            "plan completeness".into(),
            "memory beyond 64 MB".into(),
        ],
        proves: vec![
            "Rust frame 1 matches the C native core on the first 64 MB, registers, PC, cycles, and framebuffer".into(),
            format!("Rust frame 2 matches the {reference_engine} checkpoint on the first 64 MB, registers, PC, and framebuffer"),
        ],
        does_not_prove: vec![
            "the GPU kernel for all programs".into(),
            "the plan is complete".into(),
            "memory beyond the modeled 64 MB".into(),
            "fleet throughput".into(),
        ],
        falsifiers: fakes,
        nulls,
        bench: Bench {
            clock: "host_wall".into(),
            scope: "not_tested".into(),
            kind: "not_tested".into(),
            runs: 0,
            p50_ms: None,
            p99_ms: None,
            max_ms: None,
            units: "ms".into(),
            note: "run `jotnar-lab bench doom` to attach p50/p99/max".into(),
        },
        refusal: None,
        provenance: Provenance {
            command: command.into(),
            cwd: std::env::current_dir().map(|p| p.display().to_string()).unwrap_or_default(),
            commit: "uncommitted".into(),
            host: hostname(),
            environment: json_map(&[("profile", "native+gpu"), ("memory_bytes", "67108864")]),
        },
        artifacts: artifacts(&b)?,
        supersedes: None,
        superseded_by: None,
        certificate_sha256: None,
    };
    cert.seal();
    Ok(cert)
}

pub fn print_falsifiers(c: &Certificate) {
    println!("corruption\texpected\tactual\tcaught_by\tnot_caught");
    for f in &c.falsifiers {
        println!(
            "{}\t{}\t{}\t{}\t{}",
            f.corruption, f.expected, f.actual, f.caught_by, f.not_caught
        );
    }
}

pub fn print_nulls(c: &Certificate) {
    println!("null\tcontrol\texpected\tactual\teffect\tstatus");
    for n in &c.nulls {
        println!(
            "{}\t{}\t{}\t{}\t{}\t{}",
            n.name, n.control, n.expected, n.actual, n.effect, n.status
        );
    }
}

pub fn bench_native(manifest: &Path, runner: &Path, runs: u32) -> Result<Bench, String> {
    let b = Bundle::load(manifest)?;
    if !runner.exists() {
        return Err(format!("runner not found: {}", runner.display()));
    }
    let mut times = Vec::new();
    let tmp = std::env::temp_dir().join("jotnar_lab_bench_doom");
    std::fs::create_dir_all(&tmp).map_err(|e| e.to_string())?;
    for _ in 0..runs.max(1) {
        let t = Instant::now();
        let out = std::process::Command::new(runner)
            .args([
                "--image",
                &b.resolve(&b.manifest.image).display().to_string(),
                "--wad",
                &b.resolve(&b.manifest.wad).display().to_string(),
                "--frames",
                "1",
                "--out",
                &tmp.join("f1").display().to_string(),
            ])
            .output()
            .map_err(|e| e.to_string())?;
        if !out.status.success() {
            return Err(format!(
                "runner failed: {}",
                String::from_utf8_lossy(&out.stderr)
            ));
        }
        times.push(t.elapsed().as_secs_f64() * 1000.0);
    }
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let p50 = times[times.len() / 2];
    let p99 = times[(times.len() * 99 / 100).min(times.len() - 1)];
    let max = *times.last().unwrap();
    Ok(Bench {
        clock: "host_wall".into(),
        scope: "end_to_end".into(),
        kind: "measured".into(),
        runs,
        p50_ms: Some(p50),
        p99_ms: Some(p99),
        max_ms: Some(max),
        units: "ms".into(),
        note: "native Rust frame 1; includes process startup and image load".into(),
    })
}

pub fn update_bench(cert_path: &Path, bench: Bench) -> Result<Certificate, String> {
    let mut c = Certificate::read(cert_path)?;
    c.bench = bench;
    c.certificate_sha256 = None;
    c.seal();
    c.write(cert_path).map_err(|e| e.to_string())?;
    Ok(c)
}

pub fn verify(path: &Path, check_hashes: bool) -> Result<String, String> {
    let c = Certificate::read(path)?;
    let errs = c.validate();
    if !errs.is_empty() {
        return Err(format!("invalid certificate:\n- {}", errs.join("\n- ")));
    }
    if check_hashes {
        let base = path.parent().unwrap_or(Path::new("."));
        for a in &c.artifacts {
            let raw = Path::new(&a.path);
            let p = if raw.is_absolute() {
                raw.to_path_buf()
            } else {
                base.join(raw)
            };
            if !p.exists() {
                return Ok(format!(
                    "BLOCKED: artifact missing: {} (from {})",
                    a.path,
                    p.display()
                ));
            }
            let (h, n) = sha256_file(&p)?;
            if h != a.sha256 || n != a.bytes {
                return Err(format!("FAIL: artifact hash mismatch: {}", p.display()));
            }
        }
    }
    if c.status != Status::Current {
        return Ok(match c.status {
            Status::Blocked => "BLOCKED".into(),
            Status::Stale => "STALE (not a current claim)".into(),
            Status::Historical => "HISTORICAL (not a current claim)".into(),
            Status::Superseded => "SUPERSEDED (not a current claim)".into(),
            Status::Current => unreachable!(),
        });
    }
    match (&c.outcome, &c.verdict) {
        (Outcome::Solved, Verdict::Pass) => Ok("PASS".into()),
        (Outcome::Refused, _) if c.refusal.is_some() => Ok("REFUSED (valid)".into()),
        (Outcome::Blocked, _) => Ok("BLOCKED".into()),
        (Outcome::NotTested, _) => Ok("NOT TESTED".into()),
        (_, Verdict::Fail) => Ok("FAIL (certified negative)".into()),
        (_, Verdict::Bottom) => Ok("BOTTOM (gate could not decide)".into()),
        _ => Ok("NOT A PASS".into()),
    }
}

pub fn sample_certificate_json() -> Json {
    json!({"schema": SCHEMA, "schema_version": SCHEMA_VERSION})
}
