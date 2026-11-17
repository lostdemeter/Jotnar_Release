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
use std::time::{SystemTime, UNIX_EPOCH};

use serde_json::{json, Value as Json};

#[path = "../sched.rs"]
mod sched;

use sched::{batch_ledger_path, discover, health, run_batch, BatchSpec, NodeSpec};

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
     \n\
     --node may be repeated; --nodes reads one 'addr [token]' per line.\n\
     --ledger defaults to artifacts/batches. --discover browses\n\
     _jotnar-node._tcp and adds discovered ads; tokens are trusted-LAN secrets.\n"
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
    let mut nodes = cfg.nodes;
    let heartbeat_ms = cfg.heartbeat_ms;
    let down_after = cfg.down_after;
    let ledger_dir = cfg.ledger;
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
