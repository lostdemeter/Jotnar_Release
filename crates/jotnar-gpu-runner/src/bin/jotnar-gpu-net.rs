//! Target-aware controller for HIP/CUDA GPU runners.
//!
//! This is a companion to `jotnar-net`, not a replacement. The CPU node stays
//! model-blind; this controller is explicitly target-aware and runs the same
//! listing on selected GPU runners, comparing outputs with a tolerance policy.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use jotnar::value::hash_bytes;
use jotnar_gpu_runner::client::{RunReply, Runner};
use mdns_sd::{ServiceDaemon, ServiceEvent};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value as Json};

fn usage() -> &'static str {
    "jotnar-gpu-net - target-aware controller for HIP/CUDA runners\n\
     \n\
     usage:\n\
       jotnar-gpu-net [--token T] [--runner hip=ADDR] [--runner cuda=ADDR] \\\n\
         [--atol X] [--rtol X] [--discover] health\n\
       jotnar-gpu-net [--token T] [--runner hip=ADDR] [--runner cuda=ADDR] \\\n\
         [--atol X] [--rtol X] [--discover] run <listing.asm> --payload FILE \\\n\
         --outputs a,b [--target hip|cuda|both] [--count N] [--parallel N]\n\
         [--ledger DIR] [--batch-id ID] [--max-attempts N]\n\
     \n\
     --runner may be repeated. --target both runs the same job on one runner\n\
     of each target and compares the outputs. --count runs N copies. --ledger\n\
     defaults to artifacts/gpu-batches; a repeated batch id resumes a batch.\n"
}

struct Config {
    runners: Vec<Runner>,
    atol: f64,
    rtol: f64,
}

struct RunArgs {
    listing: PathBuf,
    payload: PathBuf,
    outputs: Vec<String>,
    target: String,
    count: usize,
    parallel: usize,
    ledger: PathBuf,
    batch_id: String,
    max_attempts: usize,
}

#[derive(Default)]
struct TargetStats {
    target: String,
    addr: String,
    jobs: usize,
    failed: usize,
    compile_ms: u128,
    build_ms: u128,
    run_ms: u128,
    cache_hits: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
struct LedgerSpec {
    target: String,
    listing_hash: String,
    payload_hash: String,
    outputs: Vec<String>,
    count: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct RunRecord {
    target: String,
    addr: String,
    attempts: usize,
    compile_ms: u64,
    build_ms: u64,
    run_ms: u64,
    cache_hit: bool,
    output_bytes: usize,
    output_hash: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct LedgerJob {
    index: usize,
    state: String,
    error: Option<String>,
    #[serde(default)]
    retries: usize,
    runs: Vec<RunRecord>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct Ledger {
    v: u32,
    batch_id: String,
    spec: LedgerSpec,
    created_ms: u64,
    updated_ms: u64,
    complete: bool,
    jobs: Vec<LedgerJob>,
    summary: Option<Json>,
}

fn main() -> ExitCode {
    match real_main() {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("{e}");
            ExitCode::FAILURE
        }
    }
}

fn real_main() -> Result<(), String> {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    let mut runners = Vec::new();
    let mut token = None;
    let mut atol = 1e-4;
    let mut rtol = 1e-4;
    let mut discover = false;
    let mut discover_timeout_ms = 3000u64;
    while let Some(a) = args.first().cloned() {
        match a.as_str() {
            "--runner" => {
                args.remove(0);
                let raw = args.first().cloned().ok_or("--runner needs target=addr")?;
                args.remove(0);
                let (target, addr) = raw
                    .split_once('=')
                    .ok_or_else(|| format!("bad --runner {raw:?}; use hip=ADDR"))?;
                if target != "hip" && target != "cuda" {
                    return Err(format!("runner target must be hip or cuda: {target}"));
                }
                runners.push(Runner {
                    target: target.to_string(),
                    addr: addr.to_string(),
                    token: None,
                });
            }
            "--token" => {
                args.remove(0);
                token = Some(args.first().cloned().ok_or("--token needs a value")?);
                args.remove(0);
            }
            "--atol" => {
                args.remove(0);
                atol = args
                    .first()
                    .cloned()
                    .ok_or("--atol needs a value")?
                    .parse()
                    .map_err(|_| "--atol must be a number")?;
                args.remove(0);
            }
            "--rtol" => {
                args.remove(0);
                rtol = args
                    .first()
                    .cloned()
                    .ok_or("--rtol needs a value")?
                    .parse()
                    .map_err(|_| "--rtol must be a number")?;
                args.remove(0);
            }
            "--discover" => {
                args.remove(0);
                discover = true;
            }
            "--discover-timeout-ms" => {
                args.remove(0);
                discover_timeout_ms = args
                    .first()
                    .cloned()
                    .ok_or("--discover-timeout-ms needs milliseconds")?
                    .parse()
                    .map_err(|_| "--discover-timeout-ms must be an integer")?;
                args.remove(0);
            }
            "-h" | "--help" => return Err(usage().to_string()),
            _ => break,
        }
    }
    if args.is_empty() {
        return Err(usage().to_string());
    }
    if runners.is_empty() && !discover {
        return Err("at least one --runner hip=ADDR|cuda=ADDR is required".to_string());
    }
    if discover {
        let found = discover_gpu(
            Duration::from_millis(discover_timeout_ms.max(1)),
            token.clone(),
        )?;
        let before = runners.len();
        for r in found {
            if !runners
                .iter()
                .any(|x| x.target == r.target && x.addr == r.addr)
            {
                runners.push(r);
            }
        }
        eprintln!("discovery added {} runner(s)", runners.len() - before);
    }
    if runners.is_empty() {
        return Err("no GPU runners found".to_string());
    }
    for r in &mut runners {
        r.token = token.clone();
    }
    let cmd = args.remove(0);
    let cfg = Config {
        runners,
        atol,
        rtol,
    };
    match cmd.as_str() {
        "health" | "ping" => health(&cfg),
        "run" => {
            let run = parse_run(args)?;
            run_jobs(&cfg, run)
        }
        other => Err(format!("unknown command '{other}'\n\n{}", usage())),
    }
}

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

fn default_batch_id() -> String {
    format!("gpu-batch-{}-{}", now_ms() / 1000, std::process::id())
}

fn batch_ledger_path(dir: &Path, batch_id: &str) -> PathBuf {
    let safe: String = batch_id
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '-' || c == '_' {
                c
            } else {
                '_'
            }
        })
        .collect();
    dir.join(format!("{safe}.json"))
}

fn save_ledger(path: &Path, ledger: &Ledger) -> Result<(), String> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|e| format!("create {}: {e}", parent.display()))?;
    }
    let tmp = path.with_extension("json.tmp");
    let bytes = serde_json::to_vec_pretty(ledger).map_err(|e| e.to_string())?;
    std::fs::write(&tmp, bytes).map_err(|e| format!("write {}: {e}", tmp.display()))?;
    std::fs::rename(&tmp, path).map_err(|e| format!("rename {}: {e}", path.display()))
}

fn record_hash(reply: &RunReply) -> (usize, String) {
    let outputs = reply.outputs.get("outputs").cloned().unwrap_or(Json::Null);
    let body = serde_json::to_vec(&outputs).unwrap_or_default();
    (body.len(), format!("{:016x}", hash_bytes(&body)))
}

fn parse_run(args: Vec<String>) -> Result<RunArgs, String> {
    let mut it = args.into_iter();
    let listing = PathBuf::from(it.next().ok_or("run needs <listing.asm>")?);
    let mut payload = None;
    let mut outputs = None;
    let mut target = "both".to_string();
    let mut count = 1usize;
    let mut parallel = 1usize;
    let mut ledger = PathBuf::from("artifacts/gpu-batches");
    let mut batch_id = None;
    let mut max_attempts = 3usize;
    while let Some(a) = it.next() {
        match a.as_str() {
            "--payload" => {
                payload = Some(PathBuf::from(it.next().ok_or("--payload needs a file")?))
            }
            "--outputs" => {
                outputs = Some(
                    it.next()
                        .ok_or("--outputs needs a comma list")?
                        .split(',')
                        .filter(|s| !s.is_empty())
                        .map(|s| s.to_string())
                        .collect::<Vec<_>>(),
                )
            }
            "--target" => target = it.next().ok_or("--target needs hip|cuda|both")?,
            "--count" => {
                count = it
                    .next()
                    .ok_or("--count needs a number")?
                    .parse()
                    .map_err(|_| "--count must be an integer")?
            }
            "--parallel" => {
                parallel = it
                    .next()
                    .ok_or("--parallel needs a number")?
                    .parse()
                    .map_err(|_| "--parallel must be an integer")?
            }
            "--ledger" => ledger = PathBuf::from(it.next().ok_or("--ledger needs a dir")?),
            "--batch-id" => batch_id = Some(it.next().ok_or("--batch-id needs a value")?),
            "--max-attempts" => {
                max_attempts = it
                    .next()
                    .ok_or("--max-attempts needs a number")?
                    .parse()
                    .map_err(|_| "--max-attempts must be an integer")?
            }
            "-h" | "--help" => return Err(usage().to_string()),
            other => return Err(format!("unknown run argument '{other}'")),
        }
    }
    if !matches!(target.as_str(), "hip" | "cuda" | "both") {
        return Err("--target must be hip, cuda, or both".to_string());
    }
    Ok(RunArgs {
        listing,
        payload: payload.ok_or("run needs --payload FILE")?,
        outputs: outputs.ok_or("run needs --outputs a,b")?,
        target,
        count: count.max(1),
        parallel: parallel.max(1),
        ledger,
        batch_id: batch_id.unwrap_or_else(default_batch_id),
        max_attempts: max_attempts.max(1),
    })
}

fn discover_gpu(timeout: Duration, token: Option<String>) -> Result<Vec<Runner>, String> {
    let daemon = ServiceDaemon::new().map_err(|e| e.to_string())?;
    let receiver = daemon
        .browse("_jotnar-gpu._tcp.local.")
        .map_err(|e| e.to_string())?;
    let deadline = Instant::now() + timeout;
    let mut found: BTreeMap<String, Runner> = BTreeMap::new();
    loop {
        let remaining = deadline.saturating_duration_since(Instant::now());
        if remaining.is_zero() {
            break;
        }
        let event = match receiver.recv_timeout(remaining) {
            Ok(event) => event,
            Err(_) => break,
        };
        if let ServiceEvent::ServiceResolved(info) = event {
            let target = info
                .get_property_val_str("target")
                .map(|s| s.to_string())
                .unwrap_or_default();
            if target != "hip" && target != "cuda" {
                continue;
            }
            let port = info.get_port();
            for ip in info.get_addresses_v4().iter() {
                if ip.is_loopback() {
                    continue;
                }
                let addr = format!("{ip}:{port}");
                let key = format!("{target}|{addr}");
                found.entry(key).or_insert(Runner {
                    target: target.clone(),
                    addr,
                    token: token.clone(),
                });
            }
        }
    }
    let _ = daemon.shutdown();
    Ok(found.into_values().collect())
}

fn health(cfg: &Config) -> Result<(), String> {
    let mut out = Vec::new();
    for r in &cfg.runners {
        match r.ping() {
            Ok(p) => out.push(json!({
                "target": r.target,
                "addr": r.addr,
                "healthy": true,
                "arch": p.get("arch").cloned().unwrap_or(Json::Null),
                "toolchain": p.get("toolchain").cloned().unwrap_or(Json::Null),
                "runtime_version": p.get("runtime_version").cloned().unwrap_or(Json::Null),
            })),
            Err(e) => out.push(json!({
                "target": r.target,
                "addr": r.addr,
                "healthy": false,
                "error": e,
            })),
        }
    }
    println!("{}", serde_json::to_string_pretty(&out).unwrap());
    Ok(())
}

fn run_jobs(cfg: &Config, run: RunArgs) -> Result<(), String> {
    let listing = std::fs::read_to_string(&run.listing)
        .map_err(|e| format!("read {}: {e}", run.listing.display()))?;
    let payload_text = std::fs::read_to_string(&run.payload)
        .map_err(|e| format!("read {}: {e}", run.payload.display()))?;
    let payload: Json = serde_json::from_str(&payload_text).map_err(|e| e.to_string())?;

    let spec = LedgerSpec {
        target: run.target.clone(),
        listing_hash: format!("{:016x}", hash_bytes(listing.as_bytes())),
        payload_hash: format!("{:016x}", hash_bytes(payload_text.as_bytes())),
        outputs: run.outputs.clone(),
        count: run.count,
    };
    let path = batch_ledger_path(&run.ledger, &run.batch_id);
    let existed = path.exists();
    let ledger = if existed {
        let text =
            std::fs::read_to_string(&path).map_err(|e| format!("read {}: {e}", path.display()))?;
        let l: Ledger =
            serde_json::from_str(&text).map_err(|e| format!("parse {}: {e}", path.display()))?;
        if l.spec != spec {
            return Err(format!(
                "ledger {} is for a different spec (target/outputs/count/listing/payload)",
                path.display()
            ));
        }
        l
    } else {
        Ledger {
            v: 1,
            batch_id: run.batch_id.clone(),
            spec,
            created_ms: now_ms(),
            updated_ms: now_ms(),
            complete: false,
            jobs: (0..run.count)
                .map(|i| LedgerJob {
                    index: i,
                    state: "pending".to_string(),
                    error: None,
                    retries: 0,
                    runs: Vec::new(),
                })
                .collect(),
            summary: None,
        }
    };

    if ledger.complete {
        if let Some(mut summary) = ledger.summary.clone() {
            if let Some(o) = summary.as_object_mut() {
                o.insert("cached".into(), json!(true));
            }
            println!("{}", serde_json::to_string_pretty(&summary).unwrap());
            return Ok(());
        }
    }
    let resumed = existed && ledger.jobs.iter().any(|j| j.state == "done");

    let hip: Vec<usize> = cfg
        .runners
        .iter()
        .enumerate()
        .filter(|(_, r)| r.target == "hip")
        .map(|(i, _)| i)
        .collect();
    let cuda: Vec<usize> = cfg
        .runners
        .iter()
        .enumerate()
        .filter(|(_, r)| r.target == "cuda")
        .map(|(i, _)| i)
        .collect();
    let groups: Vec<Vec<usize>> = match run.target.as_str() {
        "hip" => vec![hip],
        "cuda" => vec![cuda],
        _ => {
            if hip.is_empty() || cuda.is_empty() {
                return Err("--target both needs one hip and one cuda runner".to_string());
            }
            vec![hip, cuda]
        }
    };
    if groups.iter().any(|g| g.is_empty()) {
        return Err(format!("no runner for target {}", run.target));
    }

    let pending: Vec<usize> = ledger
        .jobs
        .iter()
        .filter(|j| j.state != "done")
        .map(|j| j.index)
        .collect();
    let pending_len = pending.len();
    let workers = run.parallel.min(pending_len.max(1)).max(1);
    let next = AtomicUsize::new(0);
    let ledger_lock = Mutex::new(ledger);
    let started = Instant::now();

    std::thread::scope(|scope| {
        for _ in 0..workers {
            scope.spawn(|| loop {
                let pos = next.fetch_add(1, Ordering::SeqCst);
                if pos >= pending_len {
                    break;
                }
                let job = pending[pos];
                let (state, error, retries, runs) = run_one_job(
                    job,
                    &groups,
                    cfg,
                    &listing,
                    &payload,
                    &run.outputs,
                    run.max_attempts,
                );
                let mut l = ledger_lock.lock().unwrap_or_else(|e| e.into_inner());
                l.jobs[job] = LedgerJob {
                    index: job,
                    state,
                    error,
                    retries,
                    runs,
                };
                l.updated_ms = now_ms();
                if let Err(e) = save_ledger(&path, &l) {
                    eprintln!("ledger save: {e}");
                }
            });
        }
    });

    let mut ledger = ledger_lock.into_inner().unwrap_or_else(|e| e.into_inner());
    let done = ledger.jobs.iter().filter(|j| j.state == "done").count();
    let failed = ledger.jobs.len() - done;
    let identical = failed == 0;
    let mut stats_map: BTreeMap<String, TargetStats> = BTreeMap::new();
    let mut output_bytes = 0usize;
    let mut errors = Vec::new();
    for job in &ledger.jobs {
        if let Some(e) = &job.error {
            errors.push(format!("job {}: {e}", job.index));
        }
        if job.state != "done" {
            continue;
        }
        for r in &job.runs {
            let s = stats_map
                .entry(r.addr.clone())
                .or_insert_with(|| TargetStats {
                    target: r.target.clone(),
                    addr: r.addr.clone(),
                    ..Default::default()
                });
            s.jobs += 1;
            s.compile_ms += r.compile_ms as u128;
            s.build_ms += r.build_ms as u128;
            s.run_ms += r.run_ms as u128;
            s.cache_hits += if r.cache_hit { 1 } else { 0 };
            output_bytes += r.output_bytes;
        }
    }
    let targets: Vec<Json> = stats_map
        .values()
        .map(|s| {
            json!({
                "target": s.target,
                "addr": s.addr,
                "jobs": s.jobs,
                "failed": s.failed,
                "compile_ms": s.compile_ms,
                "build_ms": s.build_ms,
                "run_ms": s.run_ms,
                "cache_hits": s.cache_hits,
            })
        })
        .collect();
    let wall_ms = started.elapsed().as_millis();
    let retries: usize = ledger.jobs.iter().map(|j| j.retries).sum();
    let mut summary = json!({
        "targets": targets,
        "submitted": run.count,
        "done": done,
        "failed": failed,
        "retries": retries,
        "identical_outputs": identical,
        "output_bytes": output_bytes,
        "wall_ms": wall_ms,
        "parallel": workers,
        "jobs_per_s": if wall_ms == 0 { 0.0 } else { run.count as f64 * 1000.0 / wall_ms as f64 },
        "cached": false,
        "resumed": resumed,
        "errors": errors,
    });
    ledger.complete = failed == 0;
    ledger.updated_ms = now_ms();
    if ledger.complete {
        ledger.summary = Some(summary.clone());
    }
    save_ledger(&path, &ledger)?;
    if let Some(o) = summary.as_object_mut() {
        o.insert("cached".into(), json!(false));
        o.insert("resumed".into(), json!(resumed));
    }
    println!("{}", serde_json::to_string_pretty(&summary).unwrap());
    if failed > 0 || !identical {
        return Err("GPU batch failed or outputs differed".to_string());
    }
    Ok(())
}

fn run_one_job(
    job: usize,
    groups: &[Vec<usize>],
    cfg: &Config,
    listing: &str,
    payload: &Json,
    outputs: &[String],
    max_attempts: usize,
) -> (String, Option<String>, usize, Vec<RunRecord>) {
    let mut replies: Vec<RunReply> = Vec::new();
    let mut runs = Vec::new();
    let mut errors = Vec::new();
    let mut retries = 0usize;
    let mut all_ok = true;
    for group in groups {
        let mut got = false;
        let mut last_err = String::new();
        for attempt in 1..=max_attempts {
            let ri = group[(job + attempt - 1) % group.len()];
            let r = &cfg.runners[ri];
            match r.run(listing, payload, outputs) {
                Ok(reply) => {
                    let (bytes, output_hash) = record_hash(&reply);
                    let meta = &reply.header;
                    runs.push(RunRecord {
                        target: r.target.clone(),
                        addr: r.addr.clone(),
                        attempts: attempt,
                        compile_ms: meta.get("compile_ms").and_then(|x| x.as_u64()).unwrap_or(0),
                        build_ms: meta.get("build_ms").and_then(|x| x.as_u64()).unwrap_or(0),
                        run_ms: meta.get("run_ms").and_then(|x| x.as_u64()).unwrap_or(0),
                        cache_hit: meta.get("cache_hit").and_then(|x| x.as_bool()) == Some(true),
                        output_bytes: bytes,
                        output_hash,
                    });
                    replies.push(reply);
                    got = true;
                    break;
                }
                Err(e) => {
                    retries += 1;
                    last_err = e;
                }
            }
        }
        if !got {
            all_ok = false;
            errors.push(format!(
                "{} after {max_attempts} attempt(s): {last_err}",
                cfg.runners[group[0]].target
            ));
        }
    }
    if all_ok && replies.len() > 1 {
        for idx in 1..replies.len() {
            if let Err(e) = compare_replies(&replies[0], &replies[idx], outputs, cfg.atol, cfg.rtol)
            {
                errors.push(format!(
                    "{} vs {}: {e}",
                    cfg.runners[groups[0][0]].target, cfg.runners[groups[idx][0]].target
                ));
            }
        }
    }
    let state = if errors.is_empty() { "done" } else { "failed" };
    let error = if errors.is_empty() {
        None
    } else {
        Some(errors.join("; "))
    };
    (state.to_string(), error, retries, runs)
}

fn compare_replies(
    a: &RunReply,
    b: &RunReply,
    outputs: &[String],
    atol: f64,
    rtol: f64,
) -> Result<(), String> {
    let ao = a
        .outputs
        .get("outputs")
        .and_then(|x| x.as_object())
        .ok_or("first reply has no outputs")?;
    let bo = b
        .outputs
        .get("outputs")
        .and_then(|x| x.as_object())
        .ok_or("second reply has no outputs")?;
    for name in outputs {
        let x = ao
            .get(name)
            .ok_or_else(|| format!("{name}: missing in first"))?;
        let y = bo
            .get(name)
            .ok_or_else(|| format!("{name}: missing in second"))?;
        compare_stream(name, x, y, atol, rtol)?;
    }
    Ok(())
}

fn compare_stream(name: &str, a: &Json, b: &Json, atol: f64, rtol: f64) -> Result<(), String> {
    let ka = a.get("kind").and_then(|x| x.as_str()).unwrap_or("");
    let kb = b.get("kind").and_then(|x| x.as_str()).unwrap_or("");
    if ka != kb {
        return Err(format!("{name}: kind {ka} != {kb}"));
    }
    let sa = a.get("shape").cloned().unwrap_or(Json::Null);
    let sb = b.get("shape").cloned().unwrap_or(Json::Null);
    if sa != sb {
        return Err(format!("{name}: shape {sa} != {sb}"));
    }
    match ka {
        "I64" | "U8" | "Bool" => {
            let av = int_values(a)?;
            let bv = int_values(b)?;
            if av != bv {
                return Err(format!("{name}: integer outputs differ"));
            }
        }
        "F32" | "F64" => {
            let av = float_values(a)?;
            let bv = float_values(b)?;
            if av.len() != bv.len() {
                return Err(format!("{name}: element count differs"));
            }
            for (i, (x, y)) in av.iter().zip(bv.iter()).enumerate() {
                if !(x.is_finite() && y.is_finite()) {
                    if x != y {
                        return Err(format!("{name}: non-finite mismatch at [{i}]"));
                    }
                    continue;
                }
                if (x - y).abs() > atol + rtol * y.abs() {
                    return Err(format!(
                        "{name}: [{i}] {x} vs {y} exceeds atol={atol} rtol={rtol}"
                    ));
                }
            }
        }
        other => return Err(format!("{name}: unsupported kind {other}")),
    }
    Ok(())
}

fn int_values(v: &Json) -> Result<Vec<i64>, String> {
    let data = v
        .get("data")
        .and_then(|x| x.as_array())
        .ok_or("stream has no data")?;
    data.iter()
        .map(|x| {
            if let Some(n) = x.as_i64() {
                Ok(n)
            } else if let Some(s) = x.as_str() {
                s.parse::<i64>()
                    .map_err(|e| format!("bad integer {s}: {e}"))
            } else {
                Err("bad integer value".to_string())
            }
        })
        .collect()
}

fn float_values(v: &Json) -> Result<Vec<f64>, String> {
    let data = v
        .get("data")
        .and_then(|x| x.as_array())
        .ok_or("stream has no data")?;
    data.iter()
        .map(|x| {
            x.as_f64()
                .or_else(|| x.as_str().and_then(|s| s.parse().ok()))
                .ok_or_else(|| "bad float value".to_string())
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compare_stream_gates_integers_and_floats() {
        let a = json!({"kind":"I64","shape":[2],"data":["1","2"]});
        let b = json!({"kind":"I64","shape":[2],"data":["1","2"]});
        compare_stream("OUT", &a, &b, 1e-4, 1e-4).unwrap();

        let c = json!({"kind":"I64","shape":[2],"data":["1","3"]});
        assert!(compare_stream("OUT", &a, &c, 1e-4, 1e-4).is_err());

        let x = json!({"kind":"F32","shape":[2],"data":[1.0,2.0]});
        let y = json!({"kind":"F32","shape":[2],"data":[1.0,2.00005]});
        compare_stream("OUT", &x, &y, 1e-4, 1e-4).unwrap();

        let z = json!({"kind":"F32","shape":[2],"data":[1.0,2.1]});
        assert!(compare_stream("OUT", &x, &z, 1e-4, 1e-4).is_err());
    }
}

#[test]
fn ledger_path_and_roundtrip() {
    let dir = tempfile::tempdir().unwrap();
    let path = batch_ledger_path(dir.path(), "a/b c");
    assert_eq!(path.file_name().unwrap(), "a_b_c.json");
    let ledger = Ledger {
        v: 1,
        batch_id: "a/b c".to_string(),
        spec: LedgerSpec {
            target: "both".to_string(),
            listing_hash: "01".to_string(),
            payload_hash: "02".to_string(),
            outputs: vec!["OUT".to_string()],
            count: 1,
        },
        created_ms: 1,
        updated_ms: 2,
        complete: false,
        jobs: vec![LedgerJob {
            index: 0,
            state: "pending".to_string(),
            error: None,
            retries: 0,
            runs: Vec::new(),
        }],
        summary: None,
    };
    save_ledger(&path, &ledger).unwrap();
    let text = std::fs::read_to_string(&path).unwrap();
    let back: Ledger = serde_json::from_str(&text).unwrap();
    assert_eq!(back.batch_id, "a/b c");
    assert_eq!(back.jobs.len(), 1);
    assert_eq!(back.spec.count, 1);
}
