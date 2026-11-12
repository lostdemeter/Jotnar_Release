//! Determinism harness for the warm runtime.
//!
//! Assembles a listing once, then replays NDJSON payloads across thread
//! budgets, repeats, and concurrent jobs. Every exact stream is hashed; any
//! mismatch against the single-thread reference is a hard failure.
//!
//! ```sh
//! jotnar-det layer0.asm payload.ndjson --outputs OUT \
//!   --threads 1,2,4,8,16 --repeat 3 --concurrency 1,2,8
//! ```
//!
//! The harness is model-blind: it knows about streams, hashes, and thread
//! budgets only. Use it before accepting any performance change.

use std::process::ExitCode;
use std::time::{Duration, Instant};

use jotnar::asm::{self, AsmOptions};
use jotnar::ir::BoundProgram;
use jotnar::payload::{self, Feeds};
use jotnar::profile::CountingAlloc;
use jotnar::runtime::{self, ExecOptions};
use jotnar::value::Value;
use serde_json::Value as Json;

#[global_allocator]
static ALLOC: CountingAlloc = CountingAlloc;

const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

fn fnv(bytes: &[u8]) -> u64 {
    let mut h = FNV_OFFSET;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(FNV_PRIME);
    }
    h
}

fn hash_feeds(feeds: &Feeds, names: Option<&[String]>) -> u64 {
    let mut h = FNV_OFFSET;
    let mut fold = |name: &str, v: &Value| {
        for &b in name.as_bytes() {
            h ^= b as u64;
            h = h.wrapping_mul(FNV_PRIME);
        }
        h ^= 0xff;
        h = h.wrapping_mul(FNV_PRIME);
        v.hash_into(&mut h);
    };
    match names {
        Some(names) => {
            for name in names {
                if let Some(v) = feeds.get(name) {
                    fold(name, v);
                }
            }
        }
        None => {
            // Feeds is a BTreeMap: iteration order is sorted, never hash order.
            for (name, v) in feeds {
                fold(name, v);
            }
        }
    }
    h
}

fn usage() -> &'static str {
    "jotnar-det - exact-output determinism harness\n\
     \n\
     usage:\n\
       jotnar-det <listing.asm> <payload.ndjson> [options]\n\
     \n\
     options:\n\
       --outputs a,b        streams to hash (default: every resulting stream)\n\
       --threads 1,2,4,8,16 execution thread budgets (1 is always the reference)\n\
       --repeat N           runs per budget (default 1)\n\
       --concurrency 1,2,8  concurrent jobs to test (default 1)\n\
       --job-threads N      thread budget inside concurrent jobs (default 4)\n\
       --stagger-ms N       delay job i by i*N ms to vary completion order\n\
     \n\
     Exit status is 1 on any exact hash mismatch.\n"
}

struct Args {
    listing: String,
    payload: String,
    outputs: Option<Vec<String>>,
    threads: Vec<usize>,
    repeat: usize,
    concurrency: Vec<usize>,
    job_threads: usize,
    stagger_ms: u64,
}

fn parse_list(s: &str, what: &str) -> Result<Vec<usize>, String> {
    let mut out = Vec::new();
    for part in s.split(',') {
        let v = part
            .trim()
            .parse::<usize>()
            .map_err(|e| format!("bad --{what} item '{part}': {e}"))?;
        if v == 0 {
            return Err(format!("--{what} values must be >= 1"));
        }
        out.push(v);
    }
    if out.is_empty() {
        return Err(format!("--{what} needs at least one value"));
    }
    Ok(out)
}

fn parse_args() -> Result<Args, String> {
    let mut args = std::env::args().skip(1);
    let first = args.next().ok_or_else(|| usage().to_string())?;
    if first == "-h" || first == "--help" || first == "help" {
        return Err(usage().to_string());
    }
    let listing = first;
    let payload = args
        .next()
        .ok_or_else(|| "missing payload.ndjson\n\n".to_string() + usage())?;
    let mut out = Args {
        listing,
        payload,
        outputs: None,
        threads: vec![1],
        repeat: 1,
        concurrency: vec![1],
        job_threads: 4,
        stagger_ms: 0,
    };
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--outputs" => {
                let s = args.next().ok_or("--outputs needs a comma list")?;
                out.outputs = Some(s.split(',').map(|s| s.trim().to_string()).collect());
            }
            "--threads" => {
                let s = args.next().ok_or("--threads needs a list")?;
                out.threads = parse_list(&s, "threads")?;
            }
            "--repeat" => {
                let s = args.next().ok_or("--repeat needs a count")?;
                out.repeat = s
                    .parse::<usize>()
                    .map_err(|e| format!("bad --repeat: {e}"))?;
                if out.repeat == 0 {
                    return Err("--repeat must be >= 1".to_string());
                }
            }
            "--concurrency" => {
                let s = args.next().ok_or("--concurrency needs a list")?;
                out.concurrency = parse_list(&s, "concurrency")?;
            }
            "--job-threads" => {
                let s = args.next().ok_or("--job-threads needs a count")?;
                out.job_threads = s
                    .parse::<usize>()
                    .map_err(|e| format!("bad --job-threads: {e}"))?;
                if out.job_threads == 0 {
                    return Err("--job-threads must be >= 1".to_string());
                }
            }
            "--stagger-ms" => {
                let s = args.next().ok_or("--stagger-ms needs a count")?;
                out.stagger_ms = s
                    .parse::<u64>()
                    .map_err(|e| format!("bad --stagger-ms: {e}"))?;
            }
            "-h" | "--help" => return Err(usage().to_string()),
            other => return Err(format!("unknown argument '{other}'\n\n{}", usage())),
        }
    }
    Ok(out)
}

fn load_payloads(path: &str) -> Result<Vec<Feeds>, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("cannot read {path}: {e}"))?;
    let mut out = Vec::new();
    for (i, line) in text.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let json: Json =
            serde_json::from_str(line).map_err(|e| format!("{path}:{}: {e}", i + 1))?;
        out.push(payload::feeds_from_json(&json).map_err(|e| format!("{path}:{}: {e}", i + 1))?);
    }
    if out.is_empty() {
        return Err(format!("{path}: no payloads"));
    }
    Ok(out)
}

struct Run {
    hash: u64,
    wall: Duration,
}

fn run_once(
    bound: &BoundProgram,
    feeds: &Feeds,
    threads: usize,
    names: Option<&[String]>,
) -> Result<Run, String> {
    let start = Instant::now();
    let got = runtime::run_with_options(bound, feeds, &ExecOptions::new(threads))
        .map_err(|e| e.to_string())?;
    let wall = start.elapsed();
    Ok(Run {
        hash: hash_feeds(&got, names),
        wall,
    })
}

fn median(mut v: Vec<Duration>) -> Duration {
    v.sort_unstable();
    v[v.len() / 2]
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
    let args = parse_args()?;
    let text = std::fs::read_to_string(&args.listing)
        .map_err(|e| format!("cannot read {}: {e}", args.listing))?;
    let opts = AsmOptions::new(
        std::path::Path::new(&args.listing)
            .parent()
            .unwrap_or_else(|| std::path::Path::new(".")),
    );
    let bound = asm::assemble(&text, &opts).map_err(|e| e.to_string())?;
    let payloads = load_payloads(&args.payload)?;
    let names = args.outputs.as_deref();

    let program_hash = fnv(text.as_bytes());
    let payload_hashes: Vec<u64> = payloads.iter().map(|p| hash_feeds(p, None)).collect();
    println!(
        "det: program {:016x} ({} instrs), payloads {}, outputs {}",
        program_hash,
        bound.instrs.len(),
        payloads.len(),
        names.map(|n| n.join(",")).unwrap_or_else(|| "<all>".into())
    );
    println!(
        "det: payload hashes {}",
        payload_hashes
            .iter()
            .map(|h| format!("{h:016x}"))
            .collect::<Vec<_>>()
            .join(" ")
    );

    let mut failed = false;

    // Reference: serial order, one run per payload. Every other execution must
    // match these hashes byte-for-byte.
    let mut reference = Vec::with_capacity(payloads.len());
    for (i, p) in payloads.iter().enumerate() {
        let r = run_once(&bound, p, 1, names)?;
        println!(
            "ref payload={i} threads=1 hash={:016x} wall={:.3}s",
            r.hash,
            r.wall.as_secs_f64()
        );
        reference.push(r.hash);
    }

    let mut threads = args.threads.clone();
    if !threads.contains(&1) {
        threads.insert(0, 1);
    }
    for &t in &threads {
        let mut walls = Vec::new();
        let mut ok = true;
        let mut hashes = Vec::new();
        for (i, p) in payloads.iter().enumerate() {
            for r in 0..args.repeat {
                let run = run_once(&bound, p, t, names)?;
                if run.hash != reference[i] {
                    ok = false;
                    eprintln!(
                        "MISMATCH payload={i} threads={t} run={r}: {:016x} != {:016x}",
                        run.hash, reference[i]
                    );
                }
                hashes.push(run.hash);
                walls.push(run.wall);
            }
        }
        let med = median(walls.clone());
        println!(
            "threads={:<3} hash={} wall_median={:.3}s wall_min={:.3}s runs={} {}",
            t,
            if ok { "match" } else { "MISMATCH" },
            med.as_secs_f64(),
            walls
                .iter()
                .min()
                .copied()
                .unwrap_or_default()
                .as_secs_f64(),
            hashes.len(),
            if ok { "OK" } else { "FAIL" }
        );
        failed |= !ok;
    }

    for &c in &args.concurrency {
        if c <= 1 {
            continue;
        }
        let mut results: Vec<(usize, u64, Duration)> = Vec::new();
        std::thread::scope(|scope| {
            let mut handles = Vec::new();
            for job in 0..c {
                let bound = &bound;
                let payloads = &payloads;
                let names = names;
                let jt = args.job_threads;
                let stagger = args.stagger_ms;
                handles.push(scope.spawn(move || {
                    let idx = job % payloads.len();
                    if stagger > 0 {
                        std::thread::sleep(Duration::from_millis(job as u64 * stagger));
                    }
                    let start = Instant::now();
                    let got =
                        runtime::run_with_options(bound, &payloads[idx], &ExecOptions::new(jt))
                            .map_err(|e| e.to_string())?;
                    Ok::<_, String>((idx, hash_feeds(&got, names), start.elapsed()))
                }));
            }
            for h in handles {
                results.push(h.join().map_err(|_| "worker panicked".to_string())??);
            }
            Ok::<_, String>(())
        })?;
        let mut ok = true;
        for (idx, hash, _) in &results {
            if *hash != reference[*idx] {
                ok = false;
                eprintln!(
                    "MISMATCH concurrent={c} payload={idx} job_threads={}: {:016x} != {:016x}",
                    args.job_threads, hash, reference[*idx]
                );
            }
        }
        let slowest = results.iter().map(|r| r.2).max().unwrap_or_default();
        println!(
            "concurrency={c} job_threads={} stagger_ms={} results={} slowest={:.3}s {}",
            args.job_threads,
            args.stagger_ms,
            results.len(),
            slowest.as_secs_f64(),
            if ok { "OK" } else { "FAIL" }
        );
        failed |= !ok;
    }

    if failed {
        println!("verdict: FAIL");
        Err("determinism matrix failed (exact hash mismatch)".to_string())
    } else {
        println!("verdict: PASS");
        Ok(())
    }
}
