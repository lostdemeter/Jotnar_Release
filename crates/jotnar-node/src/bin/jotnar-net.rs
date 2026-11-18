//! Controller/scheduler CLI for the Jotnar job protocol.
//!
//! N3 v0: static node list, `ping` health, least-loaded placement with a
//! deterministic tie-break, idempotent submission, retries, and aggregation.
//! Nodes remain model-blind; this tool owns the workload and placement.
//!
//! ```sh
//! jotnar-net --node 192.168.1.116:9999 --token "$TOKEN" health
//! jotnar-net --node A --node B --token "$TOKEN" run tile \
//!   --program runtime/showcase/language/matmul2.asm \
//!   --payload /tmp/matmul.json --outputs c00,c01,c10,c11 --count 200
//! ```

use std::net::TcpStream;
use std::path::PathBuf;
use std::process::ExitCode;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use serde_json::{json, Value as Json};

use jotnar_node_tools::executor::{validate_binary_result, Executor, RemoteExecutor};

#[path = "../sched.rs"]
mod sched;

use sched::{
    batch_ledger_path, discover, discover_capabilities, discover_gpu, health, run_batch, BatchSpec,
    NodeSpec,
};

fn usage() -> &'static str {
    "jotnar-net - multi-node scheduler for Jotnar jobs\n\
     \n\
     usage:\n\
       jotnar-net [--token T] [--heartbeat-ms N] [--down-after N] \\\n\
         [--ledger DIR] [--node ADDR]... [--nodes FILE] health\n\
       jotnar-net [--token T] [--heartbeat-ms N] [--down-after N] \\\n\
         [--ledger DIR] [--node ADDR]... [--nodes FILE] run <handle> \\\n\
         --payload FILE --outputs a,b [--program FILE] [--count N] \\\n\
         [--threads N] [--timeout-ms N] [--wait-ms N] [--poll-ms N] \\\n\
         [--max-attempts N] [--batch-id ID] [--receipts FILE]\n\
       jotnar-net [--token T] [--discover] [--discover-timeout-ms N] discover\n\
       jotnar-net [--discover-timeout-ms N] capabilities\n\
       jotnar-net [--token T] gpu <listing.asm> [--addr ADDR] [--target hip|cuda] \\\n\
         --payload FILE --outputs a,b [--frames N] [--count N]\n\
     \n\
     --node may be repeated; --nodes reads one 'addr [token]' per line.\n\
     --ledger defaults to artifacts/batches. --discover browses\n\
     _jotnar-node._tcp and adds discovered ads; tokens are trusted-LAN secrets.\n\
     capabilities merges _jotnar-node, _jotnar-gpu, and _jotnar-native ads into\n\
     one capability list; it is discovery only and never bypasses a node token\n\
     check.\n\
     gpu routes one stage to a GPU runner through the common executor\n\
     contract (submit/job-status/job-result).\n"
}

struct Config {
    nodes: Vec<NodeSpec>,
    heartbeat_ms: u64,
    down_after: u32,
    ledger: PathBuf,
    token: Option<String>,
    discover: bool,
    discover_timeout_ms: u64,
}

fn parse_nodes_file(path: &PathBuf) -> Result<Vec<NodeSpec>, String> {
    let text =
        std::fs::read_to_string(path).map_err(|e| format!("read {}: {e}", path.display()))?;
    let mut nodes = Vec::new();
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let mut it = line.split_whitespace();
        let addr = it.next().unwrap().to_string();
        let token = it.next().map(|s| s.to_string());
        nodes.push(NodeSpec { addr, token });
    }
    Ok(nodes)
}

fn parse_global() -> Result<(Config, String, Vec<String>), String> {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    let mut token = None;
    let mut nodes = Vec::new();
    let mut heartbeat_ms = 1000u64;
    let mut down_after = 3u32;
    let mut ledger = PathBuf::from("artifacts/batches");
    let mut discover = false;
    let mut discover_timeout_ms = 3000u64;
    while let Some(a) = args.first().cloned() {
        match a.as_str() {
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
            "--ledger" => {
                args.remove(0);
                ledger = PathBuf::from(args.first().cloned().ok_or("--ledger needs a dir")?);
                args.remove(0);
            }
            "--token" => {
                args.remove(0);
                token = Some(args.first().cloned().ok_or("--token needs a value")?);
                args.remove(0);
            }
            "--heartbeat-ms" => {
                args.remove(0);
                heartbeat_ms = args
                    .first()
                    .cloned()
                    .ok_or("--heartbeat-ms needs milliseconds")?
                    .parse()
                    .map_err(|_| "--heartbeat-ms must be an integer")?;
                args.remove(0);
            }
            "--down-after" => {
                args.remove(0);
                down_after = args
                    .first()
                    .cloned()
                    .ok_or("--down-after needs a count")?
                    .parse()
                    .map_err(|_| "--down-after must be an integer")?;
                args.remove(0);
            }
            "--node" => {
                args.remove(0);
                nodes.push(NodeSpec {
                    addr: args.first().cloned().ok_or("--node needs an addr")?,
                    token: None,
                });
                args.remove(0);
            }
            "--nodes" => {
                args.remove(0);
                let path = PathBuf::from(args.first().cloned().ok_or("--nodes needs a file")?);
                args.remove(0);
                let mut more = parse_nodes_file(&path)?;
                nodes.append(&mut more);
            }
            _ => break,
        }
    }
    if nodes.is_empty() && !discover {
        nodes.push(NodeSpec {
            addr: "127.0.0.1:9999".to_string(),
            token: None,
        });
    }
    for n in &mut nodes {
        if n.token.is_none() {
            n.token = token.clone();
        }
    }
    let cmd = args.first().cloned().unwrap_or_default();
    if args.is_empty() {
        return Err(usage().to_string());
    }
    args.remove(0);
    Ok((
        Config {
            nodes,
            heartbeat_ms,
            down_after,
            ledger,
            token,
            discover,
            discover_timeout_ms,
        },
        cmd,
        args,
    ))
}

struct RunArgs {
    handle: String,
    payload: PathBuf,
    outputs: Vec<String>,
    program: Option<PathBuf>,
    count: usize,
    threads: Option<usize>,
    timeout_ms: Option<u64>,
    wait_ms: u64,
    poll_ms: u64,
    max_attempts: usize,
    batch_id: String,
    receipts: Option<PathBuf>,
}

struct GpuArgs {
    listing: PathBuf,
    addr: Option<String>,
    target: Option<String>,
    payload: PathBuf,
    outputs: Vec<String>,
    frames: usize,
    count: usize,
}

fn parse_gpu(args: Vec<String>) -> Result<GpuArgs, String> {
    let mut it = args.into_iter();
    let listing = PathBuf::from(it.next().ok_or("gpu needs <listing.asm>")?);
    let mut addr = None;
    let mut target = None;
    let mut payload = None;
    let mut outputs = None;
    let mut frames = 1usize;
    let mut count = 1usize;
    while let Some(a) = it.next() {
        match a.as_str() {
            "--addr" => addr = Some(it.next().ok_or("--addr needs an address")?),
            "--target" => target = Some(it.next().ok_or("--target needs hip or cuda")?),
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
            "--frames" => {
                frames = it
                    .next()
                    .ok_or("--frames needs a number")?
                    .parse()
                    .map_err(|_| "--frames must be an integer")?
            }
            "--count" => {
                count = it
                    .next()
                    .ok_or("--count needs a number")?
                    .parse()
                    .map_err(|_| "--count must be an integer")?
            }
            other => return Err(format!("unknown gpu argument '{other}'")),
        }
    }
    if let Some(target) = &target {
        if !matches!(target.as_str(), "hip" | "cuda") {
            return Err("--target must be hip or cuda".to_string());
        }
    }
    Ok(GpuArgs {
        listing,
        addr,
        target,
        payload: payload.ok_or("gpu needs --payload FILE")?,
        outputs: outputs.ok_or("gpu needs --outputs a,b")?,
        frames: frames.max(1),
        count: count.max(1),
    })
}

fn run_gpu(cfg: &Config, args: Vec<String>) -> Result<Json, String> {
    let args = parse_gpu(args)?;
    let listing = std::fs::read_to_string(&args.listing)
        .map_err(|e| format!("read {}: {e}", args.listing.display()))?;
    let payload_text = std::fs::read_to_string(&args.payload)
        .map_err(|e| format!("read {}: {e}", args.payload.display()))?;
    let payload: Json = serde_json::from_str(&payload_text).map_err(|e| e.to_string())?;
    let (addr, capability) = match args.addr.clone() {
        Some(addr) => (
            addr.clone(),
            json!({"kind": "gpu", "addr": addr, "target": args.target}),
        ),
        None => {
            let caps = discover_gpu(Duration::from_millis(cfg.discover_timeout_ms.max(1)))?;
            let cap = caps
                .iter()
                .find(|cap| {
                    args.target
                        .as_deref()
                        .map(|t| cap.get("target").and_then(|x| x.as_str()) == Some(t))
                        .unwrap_or(true)
                })
                .ok_or_else(|| {
                    format!(
                        "no GPU runner for target {}",
                        args.target.as_deref().unwrap_or("any")
                    )
                })?
                .clone();
            let addrs = cap
                .get("addrs")
                .and_then(|x| x.as_array())
                .ok_or("GPU capability has no addrs")?;
            let chosen = addrs
                .iter()
                .filter_map(|a| a.as_str())
                .find(|a| !a.starts_with("127."))
                .or_else(|| addrs.iter().filter_map(|a| a.as_str()).next())
                .ok_or("GPU capability has no usable address")?
                .to_string();
            (chosen, cap)
        }
    };
    let executor = RemoteExecutor::new(capability, addr.clone(), cfg.token.clone());
    let body = json!({"listing": listing, "payload": payload});
    let body = serde_json::to_vec(&body).map_err(|e| e.to_string())?;
    let mut jobs = Vec::new();
    for _ in 0..args.count {
        let spec = json!({
            "type": "submit",
            "outputs": args.outputs,
            "frames": args.frames,
            "output_format": "binary",
        });
        let submitted = executor
            .submit(&spec, &body)
            .map_err(|(code, msg)| format!("submit {code}: {msg}"))?;
        let job_id = submitted
            .get("job_id")
            .and_then(|x| x.as_str())
            .ok_or("submit returned no job_id")?
            .to_string();
        let deadline = std::time::Instant::now() + Duration::from_secs(600);
        let state = loop {
            let status = executor
                .status(&job_id)
                .map_err(|(code, msg)| format!("job-status {code}: {msg}"))?;
            let state = status
                .get("job")
                .and_then(|j| j.get("state"))
                .and_then(|x| x.as_str())
                .unwrap_or("unknown")
                .to_string();
            if matches!(state.as_str(), "done" | "failed" | "canceled") {
                break state;
            }
            if std::time::Instant::now() >= deadline {
                return Err(format!("job {job_id} timed out in state {state}"));
            }
            std::thread::sleep(Duration::from_millis(25));
        };
        if state == "failed" {
            let status = executor
                .status(&job_id)
                .map_err(|(code, msg)| format!("job-status {code}: {msg}"))?;
            return Err(format!("job {job_id} failed: {status}"));
        }
        if state == "canceled" {
            jobs.push(json!({"job_id": job_id, "state": state}));
            continue;
        }
        let (header, bytes) = executor
            .result(&job_id)
            .map_err(|(code, msg)| format!("job-result {code}: {msg}"))?;
        if header.get("output_format").and_then(|x| x.as_str()) == Some("binary") {
            validate_binary_result(&header, &bytes)
                .map_err(|e| format!("invalid result from {addr}: {e}"))?;
        }
        jobs.push(json!({
            "job_id": job_id,
            "state": state,
            "bytes": bytes.len(),
            "output_format": header.get("output_format").cloned().unwrap_or(Json::Null),
            "outputs": header.get("outputs").cloned().unwrap_or(Json::Null),
        }));
    }
    Ok(json!({
        "addr": addr,
        "target": args.target,
        "frames": args.frames,
        "count": args.count,
        "done": jobs.iter().filter(|j| j.get("state").and_then(|x| x.as_str()) == Some("done")).count(),
        "jobs": jobs,
    }))
}

fn parse_run(handle: String, args: Vec<String>) -> Result<RunArgs, String> {
    let mut payload = None;
    let mut outputs = None;
    let mut program = None;
    let mut count = 1usize;
    let mut threads = None;
    let mut timeout_ms = None;
    let mut wait_ms = 600_000u64;
    let mut poll_ms = 50u64;
    let mut max_attempts = 3usize;
    let mut batch_id = None;
    let mut receipts = None;
    let mut it = args.iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--payload" => {
                payload = Some(PathBuf::from(it.next().ok_or("--payload needs a file")?))
            }
            "--outputs" => {
                let s = it.next().ok_or("--outputs needs a comma list")?;
                outputs = Some(s.split(',').map(|s| s.trim().to_string()).collect());
            }
            "--program" => {
                program = Some(PathBuf::from(it.next().ok_or("--program needs a file")?))
            }
            "--count" => {
                count = it
                    .next()
                    .ok_or("--count needs a number")?
                    .parse()
                    .map_err(|_| "--count must be an integer")?;
            }
            "--threads" => {
                threads = Some(
                    it.next()
                        .ok_or("--threads needs a count")?
                        .parse()
                        .map_err(|_| "--threads must be an integer")?,
                );
            }
            "--timeout-ms" => {
                timeout_ms = Some(
                    it.next()
                        .ok_or("--timeout-ms needs milliseconds")?
                        .parse()
                        .map_err(|_| "--timeout-ms must be an integer")?,
                );
            }
            "--wait-ms" => {
                wait_ms = it
                    .next()
                    .ok_or("--wait-ms needs milliseconds")?
                    .parse()
                    .map_err(|_| "--wait-ms must be an integer")?;
            }
            "--poll-ms" => {
                poll_ms = it
                    .next()
                    .ok_or("--poll-ms needs milliseconds")?
                    .parse()
                    .map_err(|_| "--poll-ms must be an integer")?;
            }
            "--max-attempts" => {
                max_attempts = it
                    .next()
                    .ok_or("--max-attempts needs a number")?
                    .parse()
                    .map_err(|_| "--max-attempts must be an integer")?;
            }
            "--batch-id" => {
                batch_id = Some(it.next().ok_or("--batch-id needs a value")?.to_string())
            }
            "--receipts" => {
                receipts = Some(PathBuf::from(it.next().ok_or("--receipts needs a file")?))
            }
            "-h" | "--help" => return Err(usage().to_string()),
            other => return Err(format!("unknown argument '{other}'")),
        }
    }
    Ok(RunArgs {
        handle,
        payload: payload.ok_or("--payload is required")?,
        outputs: outputs.ok_or("--outputs is required")?,
        program,
        count,
        threads,
        timeout_ms,
        wait_ms,
        poll_ms,
        max_attempts,
        batch_id: batch_id.unwrap_or_else(|| {
            let secs = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0);
            format!("batch-{secs}-{}", std::process::id())
        }),
        receipts,
    })
}

fn real_main() -> Result<(), String> {
    let (cfg, cmd, args) = parse_global()?;
    if cmd == "gpu" {
        let out = run_gpu(&cfg, args)?;
        println!("{}", serde_json::to_string_pretty(&out).unwrap());
        return Ok(());
    }
    let mut nodes = cfg.nodes;
    let heartbeat_ms = cfg.heartbeat_ms;
    let down_after = cfg.down_after;
    let ledger_dir = cfg.ledger.clone();
    if cfg.discover && cmd != "discover" {
        match discover(std::time::Duration::from_millis(
            cfg.discover_timeout_ms.max(1),
        )) {
            Ok(adverts) => {
                let mut added = 0usize;
                for advert in adverts {
                    let Some(addrs) = advert.get("addrs").and_then(|a| a.as_array()) else {
                        continue;
                    };
                    let chosen = {
                        let candidates: Vec<&str> =
                            addrs.iter().filter_map(|a| a.as_str()).collect();
                        candidates
                            .iter()
                            .find(|a| !a.starts_with("127.") && TcpStream::connect(a).is_ok())
                            .or_else(|| candidates.iter().find(|a| TcpStream::connect(a).is_ok()))
                            .or_else(|| candidates.iter().find(|a| !a.starts_with("127.")))
                            .or_else(|| candidates.first())
                            .copied()
                    };
                    if let Some(addr) = chosen {
                        if !nodes.iter().any(|n| n.addr == addr) {
                            nodes.push(NodeSpec {
                                addr: addr.to_string(),
                                token: cfg.token.clone(),
                            });
                            added += 1;
                        }
                    }
                }
                eprintln!("discovery added {added} node address(es)");
            }
            Err(e) => eprintln!("discovery unavailable: {e}"),
        }
    }
    if nodes.is_empty() {
        nodes.push(NodeSpec {
            addr: "127.0.0.1:9999".to_string(),
            token: cfg.token.clone(),
        });
    }
    match cmd.as_str() {
        "capabilities" => {
            let out = discover_capabilities(Duration::from_millis(cfg.discover_timeout_ms.max(1)))?;
            println!("{}", serde_json::to_string_pretty(&out).unwrap());
            Ok(())
        }
        "discover" => {
            let out = discover(std::time::Duration::from_millis(
                cfg.discover_timeout_ms.max(1),
            ))?;
            println!("{}", serde_json::to_string_pretty(&out).unwrap());
            Ok(())
        }
        "health" | "ping" => {
            // Point-in-time health: one failed ping means not healthy.
            let out: Vec<Json> = health(&nodes, heartbeat_ms, 1);
            println!("{}", serde_json::to_string_pretty(&out).unwrap());
            Ok(())
        }
        "run" => {
            if args.is_empty() {
                return Err(format!("run needs <handle>\n\n{}", usage()));
            }
            let mut args = args;
            let handle = args.remove(0);
            let run = parse_run(handle, args)?;
            let feeds = jotnar::payload::read_payload(&run.payload).map_err(|e| e.to_string())?;
            let program = match &run.program {
                Some(path) => {
                    let listing =
                        std::fs::read(path).map_err(|e| format!("read {}: {e}", path.display()))?;
                    Some((run.handle.clone(), listing))
                }
                None => None,
            };
            let spec = BatchSpec {
                handle: run.handle,
                feeds,
                outputs: run.outputs,
                count: run.count,
                threads: run.threads,
                timeout_ms: run.timeout_ms,
                wait_ms: run.wait_ms,
                poll_ms: run.poll_ms,
                max_attempts: run.max_attempts,
                batch_id: run.batch_id,
                heartbeat_ms,
                down_after,
            };
            let ledger = batch_ledger_path(&ledger_dir, &spec.batch_id);
            let summary = run_batch(&nodes, &spec, program, Some(ledger))?;
            if let Some(path) = &run.receipts {
                std::fs::write(
                    path,
                    serde_json::to_vec_pretty(&json!(&summary.receipts)).unwrap(),
                )
                .map_err(|e| format!("write {}: {e}", path.display()))?;
            }
            println!("{}", serde_json::to_string_pretty(&summary).unwrap());
            if summary.failed > 0 {
                return Err(format!("{} job(s) failed", summary.failed));
            }
            if !summary.identical_outputs {
                return Err("batch outputs were not byte-identical".to_string());
            }
            Ok(())
        }
        other => Err(format!("unknown command '{other}'\n\n{}", usage())),
    }
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
