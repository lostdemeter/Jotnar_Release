//! Target-aware controller for HIP/CUDA GPU runners.
//!
//! This is a companion to `jotnar-net`, not a replacement. The CPU node stays
//! model-blind; this controller is explicitly target-aware and runs the same
//! listing on selected GPU runners, comparing outputs with a tolerance policy.

use std::collections::HashMap;
use std::path::PathBuf;
use std::process::ExitCode;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Mutex;
use std::time::Instant;

use jotnar_gpu_runner::client::{RunReply, Runner};
use serde_json::{json, Value as Json};

fn usage() -> &'static str {
    "jotnar-gpu-net - target-aware controller for HIP/CUDA runners\n\
     \n\
     usage:\n\
       jotnar-gpu-net [--token T] [--runner hip=ADDR] [--runner cuda=ADDR] \\\n\
         [--atol X] [--rtol X] health\n\
       jotnar-gpu-net [--token T] [--runner hip=ADDR] [--runner cuda=ADDR] \\\n\
         [--atol X] [--rtol X] run <listing.asm> --payload FILE \\\n\
         --outputs a,b [--target hip|cuda|both] [--count N] [--parallel N]\n\
     \n\
     --runner may be repeated. --target both runs the same job on one runner\n\
     of each target and compares the outputs. --count runs N copies.\n"
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
            "-h" | "--help" => return Err(usage().to_string()),
            _ => break,
        }
    }
    if args.is_empty() {
        return Err(usage().to_string());
    }
    if runners.is_empty() {
        return Err("at least one --runner hip=ADDR|cuda=ADDR is required".to_string());
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

fn parse_run(args: Vec<String>) -> Result<RunArgs, String> {
    let mut it = args.into_iter();
    let listing = PathBuf::from(it.next().ok_or("run needs <listing.asm>")?);
    let mut payload = None;
    let mut outputs = None;
    let mut target = "both".to_string();
    let mut count = 1usize;
    let mut parallel = 1usize;
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
    })
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

    let mut selected: Vec<usize> = groups.iter().flatten().copied().collect();
    selected.sort_unstable();
    selected.dedup();
    let stat_index: HashMap<usize, usize> = selected
        .iter()
        .enumerate()
        .map(|(si, &ri)| (ri, si))
        .collect();
    let mut initial_stats = Vec::with_capacity(selected.len());
    for &ri in &selected {
        let r = &cfg.runners[ri];
        initial_stats.push(TargetStats {
            target: r.target.clone(),
            addr: r.addr.clone(),
            ..Default::default()
        });
    }

    let workers = run.parallel.min(run.count).max(1);
    let next = AtomicUsize::new(0);
    let done = AtomicUsize::new(0);
    let failed = AtomicUsize::new(0);
    let output_bytes = AtomicUsize::new(0);
    let identical = AtomicBool::new(true);
    let errors = Mutex::new(Vec::<String>::new());
    let stats = Mutex::new(initial_stats);
    let started = Instant::now();

    std::thread::scope(|scope| {
        for _ in 0..workers {
            scope.spawn(|| loop {
                let job = next.fetch_add(1, Ordering::SeqCst);
                if job >= run.count {
                    break;
                }
                let plan: Vec<usize> = groups.iter().map(|g| g[job % g.len()]).collect();
                let mut replies: Vec<RunReply> = Vec::new();
                let mut job_ok = true;
                for &ri in &plan {
                    let r = &cfg.runners[ri];
                    match r.run(&listing, &payload, &run.outputs) {
                        Ok(reply) => {
                            let meta = &reply.header;
                            let si = stat_index[&ri];
                            {
                                let mut s = stats.lock().unwrap_or_else(|e| e.into_inner());
                                s[si].jobs += 1;
                                s[si].compile_ms +=
                                    meta.get("compile_ms").and_then(|x| x.as_u64()).unwrap_or(0)
                                        as u128;
                                s[si].build_ms +=
                                    meta.get("build_ms").and_then(|x| x.as_u64()).unwrap_or(0)
                                        as u128;
                                s[si].run_ms +=
                                    meta.get("run_ms").and_then(|x| x.as_u64()).unwrap_or(0)
                                        as u128;
                                if meta.get("cache_hit").and_then(|x| x.as_bool()) == Some(true) {
                                    s[si].cache_hits += 1;
                                }
                            }
                            output_bytes.fetch_add(
                                serde_json::to_vec(&reply.outputs)
                                    .map(|v| v.len())
                                    .unwrap_or(0),
                                Ordering::Relaxed,
                            );
                            replies.push(reply);
                        }
                        Err(e) => {
                            job_ok = false;
                            let si = stat_index[&ri];
                            {
                                let mut s = stats.lock().unwrap_or_else(|e| e.into_inner());
                                s[si].failed += 1;
                            }
                            errors
                                .lock()
                                .unwrap_or_else(|e| e.into_inner())
                                .push(format!("job {job} {} {}: {e}", r.target, r.addr));
                        }
                    }
                }
                if job_ok && replies.len() == plan.len() && replies.len() > 1 {
                    for idx in 1..replies.len() {
                        if let Err(e) = compare_replies(
                            &replies[0],
                            &replies[idx],
                            &run.outputs,
                            cfg.atol,
                            cfg.rtol,
                        ) {
                            job_ok = false;
                            identical.store(false, Ordering::SeqCst);
                            let a = &cfg.runners[plan[0]];
                            let b = &cfg.runners[plan[idx]];
                            errors
                                .lock()
                                .unwrap_or_else(|e| e.into_inner())
                                .push(format!(
                                    "job {job} {} {} vs {} {}: {e}",
                                    a.target, a.addr, b.target, b.addr
                                ));
                        }
                    }
                }
                if job_ok {
                    done.fetch_add(1, Ordering::SeqCst);
                } else {
                    failed.fetch_add(1, Ordering::SeqCst);
                }
            });
        }
    });

    let wall_ms = started.elapsed().as_millis();
    let stats = stats.into_inner().unwrap_or_else(|e| e.into_inner());
    let targets: Vec<Json> = stats
        .iter()
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
    let done = done.load(Ordering::SeqCst);
    let failed = failed.load(Ordering::SeqCst);
    let identical = identical.load(Ordering::SeqCst);
    let output_bytes = output_bytes.load(Ordering::SeqCst);
    let errors = errors.into_inner().unwrap_or_else(|e| e.into_inner());
    let summary = json!({
        "targets": targets,
        "submitted": run.count,
        "done": done,
        "failed": failed,
        "identical_outputs": identical,
        "output_bytes": output_bytes,
        "wall_ms": wall_ms,
        "parallel": workers,
        "jobs_per_s": if wall_ms == 0 { 0.0 } else { run.count as f64 * 1000.0 / wall_ms as f64 },
        "errors": errors,
    });
    println!("{}", serde_json::to_string_pretty(&summary).unwrap());
    if failed > 0 || !identical {
        return Err("GPU batch failed or outputs differed".to_string());
    }
    Ok(())
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
