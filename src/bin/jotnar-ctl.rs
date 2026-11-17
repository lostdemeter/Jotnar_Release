//! Framed control client for `jotnar-node`.
//!
//! Talks to a node over TCP using the same frame codec as the server
//! (`u32BE || JSON header || payload`). Drive a listing, blob shelf,
//! and `EXEC` requests from the shell without writing Rust.
//!
//! ```sh
//! cargo run --release --bin jotnar-ctl -- --addr 127.0.0.1:9999 ping
//! cargo run --release --bin jotnar-ctl -- --addr 127.0.0.1:9999 program tile showcase/language/matmul2.asm
//! cargo run --release --bin jotnar-ctl -- --addr 127.0.0.1:9999 exec tile --payload /tmp/p.json --outputs c00
//! ```
//!
//! `exec --payload` takes the same JSON shape as `jotnar run --payload`.
//! Results are printed as JSON on stdout (streams decoded where possible).

use std::io::{Read, Write};
use std::net::TcpStream;
use std::path::PathBuf;
use std::process::ExitCode;
use std::time::{Duration, Instant};

use jotnar::payload::{self, Feeds};
use jotnar::protocol::{decode_frame, decode_stream, encode_stream, StreamSpec};
use serde_json::{json, Value as Json};
use sha2::{Digest, Sha256};

fn usage() -> &'static str {
    "jotnar-ctl - framed client for jotnar-node\n\
     \n\
     usage:\n\
       jotnar-ctl [--addr ADDR] [--token SECRET] ping\n\
       jotnar-ctl [--addr ADDR] [--token SECRET] list\n\
       jotnar-ctl [--addr ADDR] [--token SECRET] program <name> <file.asm> [--backend lattice]\n\
       jotnar-ctl [--addr ADDR] [--token SECRET] store <name> --payload <payload.json> --stream <NAME>\n\
       jotnar-ctl [--addr ADDR] [--token SECRET] exec <handle> --payload <payload.json> --outputs a,b [--threads N] [--policy reference|auto] [--receipt]\n\
       jotnar-ctl [--addr ADDR] [--token SECRET] submit <handle> --payload <payload.json> --outputs a,b [--threads N] [--policy reference|auto] [--timeout-ms N] [--idempotency-key K] [--receipt]\n\
       jotnar-ctl [--addr ADDR] [--token SECRET] job-status <job_id>\n\
       jotnar-ctl [--addr ADDR] [--token SECRET] job-result <job_id> [--verify] [--receipt]\n\
       jotnar-ctl [--addr ADDR] [--token SECRET] job-cancel <job_id>\n\
       jotnar-ctl [--addr ADDR] [--token SECRET] job-wait <job_id> [--wait-ms N] [--poll-ms N] [--verify] [--receipt]\n\
       jotnar-ctl [--addr ADDR] [--token SECRET] job-run <handle> --payload <payload.json> --outputs a,b [--threads N] [--policy reference|auto] [--timeout-ms N] [--idempotency-key K] [--wait-ms N] [--verify] [--receipt]\n\
       jotnar-ctl [--addr ADDR] [--token SECRET] job-batch <handle> --payload <payload.json> --outputs a,b [--count N] [--threads N] [--policy reference|auto] [--timeout-ms N] [--wait-ms N] [--poll-ms N] [--verify] [--receipt]\n\
       jotnar-ctl [--addr ADDR] [--token SECRET] result-read <result_id> --off N --len N --out FILE\n\
       jotnar-ctl [--addr ADDR] [--token SECRET] result-drop <result_id>\n\
       jotnar-ctl [--addr ADDR] [--token SECRET] job-list\n\
       jotnar-ctl [--addr ADDR] [--token SECRET] drop <name>\n\
     \n\
     --addr defaults to 127.0.0.1:9999. --payload is the JSON shape from\n\
     `jotnar run --payload` (`{\"streams\": {...}}`). Large results are fetched\n\
     as verified chunks; --verify prints a summary instead of the streams.\n"
}

fn connect(addr: &str) -> Result<TcpStream, String> {
    let stream = TcpStream::connect(addr).map_err(|e| format!("cannot connect to {addr}: {e}"))?;
    let _ = stream.set_nodelay(true);
    let _ = stream.set_read_timeout(Some(std::time::Duration::from_secs(30)));
    let _ = stream.set_write_timeout(Some(std::time::Duration::from_secs(30)));
    Ok(stream)
}

/// Send one framed request, return `(header, payload)` of the reply.
fn roundtrip(
    stream: &mut TcpStream,
    id: u64,
    typ: &str,
    fields: Json,
    payload: &[u8],
    token: &Option<String>,
) -> Result<(Json, Vec<u8>), String> {
    let mut fields = fields;
    if let Some(t) = token {
        if let Some(o) = fields.as_object_mut() {
            o.insert("auth".into(), json!(t));
        }
    }
    let frame = jotnar::protocol::request(id, typ, fields, payload).map_err(|e| e.to_string())?;
    stream
        .write_all(&frame)
        .map_err(|e| format!("write: {e}"))?;
    let mut hdr = [0u8; 4];
    stream
        .read_exact(&mut hdr)
        .map_err(|e| format!("read header: {e}"))?;
    let n = u32::from_be_bytes(hdr) as usize;
    const MAX_CLIENT_FRAME: usize = 256 * 1024 * 1024;
    if n > MAX_CLIENT_FRAME {
        return Err(format!(
            "response frame too large ({n} > {MAX_CLIENT_FRAME})"
        ));
    }
    let mut body = vec![0u8; n];
    stream
        .read_exact(&mut body)
        .map_err(|e| format!("read body: {e}"))?;
    let mut full = Vec::with_capacity(4 + n);
    full.extend_from_slice(&hdr);
    full.extend_from_slice(&body);
    let (header, payload) = decode_frame(&full).map_err(|e| e.to_string())?;
    if header.get("id").and_then(|x| x.as_u64()) != Some(id) {
        return Err(format!("response id mismatch: expected {id}, got {header}"));
    }
    if header.get("v").and_then(|x| x.as_u64()) != Some(jotnar::protocol::PROTO_V) {
        return Err(format!(
            "response version mismatch: {}",
            header.get("v").cloned().unwrap_or(json!("?"))
        ));
    }
    Ok((header, payload.to_vec()))
}

fn ok_or_show(header: &Json) -> Result<(), String> {
    if header.get("ok").and_then(|x| x.as_bool()) == Some(true) {
        Ok(())
    } else {
        Err(format!(
            "node error: {}",
            header.get("error").cloned().unwrap_or(json!("?"))
        ))
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

#[derive(Clone)]
struct JobArgs {
    handle: String,
    payload: PathBuf,
    outputs: Vec<String>,
    threads: Option<usize>,
    policy: Option<String>,
    receipt: bool,
    timeout_ms: Option<u64>,
    idempotency_key: Option<String>,
    wait_ms: u64,
    poll_ms: u64,
    count: usize,
    verify: bool,
}

enum Plan {
    Ping,
    List,
    Program {
        name: String,
        file: PathBuf,
        backend: String,
    },
    Store {
        name: String,
        payload: PathBuf,
        stream: String,
    },
    Exec {
        handle: String,
        payload: PathBuf,
        outputs: Vec<String>,
        threads: Option<usize>,
        policy: Option<String>,
        receipt: bool,
    },
    Submit(JobArgs),
    JobStatus {
        job_id: String,
    },
    JobResult {
        job_id: String,
        receipt: bool,
        verify: bool,
    },
    JobCancel {
        job_id: String,
    },
    JobWait {
        job_id: String,
        wait_ms: u64,
        poll_ms: u64,
        receipt: bool,
        verify: bool,
    },
    ResultRead {
        result_id: String,
        off: usize,
        len: usize,
        out: PathBuf,
    },
    ResultDrop {
        result_id: String,
    },
    JobRun(JobArgs),
    JobBatch(JobArgs),
    JobList,
    Drop {
        name: String,
    },
}

fn parse_job_args(handle: String, args: Vec<String>) -> Result<JobArgs, String> {
    let mut payload: Option<PathBuf> = None;
    let mut outputs: Option<Vec<String>> = None;
    let mut threads: Option<usize> = None;
    let mut policy: Option<String> = None;
    let mut receipt = false;
    let mut timeout_ms: Option<u64> = None;
    let mut idempotency_key: Option<String> = None;
    let mut wait_ms = 600_000u64;
    let mut poll_ms = 25u64;
    let mut count = 100usize;
    let mut verify = false;
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
            "--threads" => {
                let s = it.next().ok_or("--threads needs a count")?;
                let n = s
                    .parse::<usize>()
                    .map_err(|e| format!("bad --threads: {e}"))?;
                if n == 0 {
                    return Err("--threads must be >= 1".to_string());
                }
                threads = Some(n);
            }
            "--policy" => {
                let s = it.next().ok_or("--policy needs a value")?;
                if !matches!(
                    s.as_str(),
                    "reference" | "serial" | "auto" | "deterministic-auto"
                ) {
                    return Err(format!("bad --policy {s:?}"));
                }
                policy = Some(s.to_string());
            }
            "--timeout-ms" => {
                timeout_ms = Some(
                    it.next()
                        .ok_or("--timeout-ms needs milliseconds")?
                        .parse()
                        .map_err(|_| "--timeout-ms must be an integer")?,
                );
            }
            "--idempotency-key" => {
                idempotency_key = Some(
                    it.next()
                        .ok_or("--idempotency-key needs a value")?
                        .to_string(),
                );
            }
            "--count" => {
                count = it
                    .next()
                    .ok_or("--count needs a number")?
                    .parse()
                    .map_err(|_| "--count must be an integer")?;
                if count == 0 {
                    return Err("--count must be >= 1".to_string());
                }
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
            "--receipt" => receipt = true,
            "--verify" => verify = true,
            "-h" | "--help" => return Err(usage().to_string()),
            other => return Err(format!("unknown argument '{other}'")),
        }
    }
    Ok(JobArgs {
        handle,
        payload: payload.ok_or("--payload is required")?,
        outputs: outputs.ok_or("--outputs is required")?,
        threads,
        policy,
        receipt,
        timeout_ms,
        idempotency_key,
        wait_ms,
        poll_ms,
        count,
        verify,
    })
}

fn one_id(args: Vec<String>, what: &str) -> Result<String, String> {
    if args.len() != 1 {
        return Err(format!("{what} needs exactly one <job_id>\n\n{}", usage()));
    }
    Ok(args[0].clone())
}

fn parse_wait_flags(args: Vec<String>) -> Result<(u64, u64, bool, bool), String> {
    let mut wait_ms = 600_000u64;
    let mut poll_ms = 25u64;
    let mut receipt = false;
    let mut verify = false;
    let mut it = args.iter();
    while let Some(a) = it.next() {
        match a.as_str() {
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
            "--receipt" => receipt = true,
            "--verify" => verify = true,
            "-h" | "--help" => return Err(usage().to_string()),
            other => return Err(format!("unknown argument '{other}'")),
        }
    }
    Ok((wait_ms, poll_ms, receipt, verify))
}

fn parse_plan(cmd: &str, mut args: Vec<String>) -> Result<Plan, String> {
    match cmd {
        "ping" => Ok(Plan::Ping),
        "list" => Ok(Plan::List),
        "program" => {
            if args.len() < 2 {
                return Err(format!("program needs <name> <file.asm>\n\n{}", usage()));
            }
            let name = args.remove(0);
            let file = PathBuf::from(args.remove(0));
            let mut backend = "lattice".to_string();
            while let Some(a) = args.first() {
                let a = a.clone();
                args.remove(0);
                match a.as_str() {
                    "--backend" => {
                        backend = args
                            .first()
                            .cloned()
                            .ok_or("--backend needs a value".to_string())?;
                        args.remove(0);
                    }
                    "-h" | "--help" => return Err(usage().to_string()),
                    other => return Err(format!("unknown argument '{other}'")),
                }
            }
            Ok(Plan::Program {
                name,
                file,
                backend,
            })
        }
        "store" => {
            if args.is_empty() {
                return Err(format!("store needs <name>\n\n{}", usage()));
            }
            let name = args.remove(0);
            let (payload, stream) = parse_payload_args(&args)?;
            Ok(Plan::Store {
                name,
                payload,
                stream,
            })
        }
        "exec" => {
            if args.is_empty() {
                return Err(format!("exec needs <handle>\n\n{}", usage()));
            }
            let handle = args.remove(0);
            let (payload, outputs, threads, policy, receipt) = parse_exec_args(&args)?;
            Ok(Plan::Exec {
                handle,
                payload,
                outputs,
                threads,
                policy,
                receipt,
            })
        }
        "submit" | "job-run" | "job-batch" => {
            if args.is_empty() {
                return Err(format!(
                    "{cmd} needs <handle> --payload <file> --outputs a,b\n\n{}",
                    usage()
                ));
            }
            let handle = args.remove(0);
            let job = parse_job_args(handle, args)?;
            match cmd {
                "submit" => Ok(Plan::Submit(job)),
                "job-run" => Ok(Plan::JobRun(job)),
                _ => Ok(Plan::JobBatch(job)),
            }
        }
        "job-status" => Ok(Plan::JobStatus {
            job_id: one_id(args, "job-status")?,
        }),
        "job-cancel" => Ok(Plan::JobCancel {
            job_id: one_id(args, "job-cancel")?,
        }),
        "job-result" => {
            let mut receipt = false;
            let mut verify = false;
            let mut rest = Vec::new();
            for a in args {
                match a.as_str() {
                    "--receipt" => receipt = true,
                    "--verify" => verify = true,
                    "-h" | "--help" => return Err(usage().to_string()),
                    other => rest.push(other.to_string()),
                }
            }
            Ok(Plan::JobResult {
                job_id: one_id(rest, "job-result")?,
                receipt,
                verify,
            })
        }
        "job-wait" => {
            if args.is_empty() {
                return Err(format!("job-wait needs <job_id>\n\n{}", usage()));
            }
            let mut args = args;
            let job_id = args.remove(0);
            let (wait_ms, poll_ms, receipt, verify) = parse_wait_flags(args)?;
            Ok(Plan::JobWait {
                job_id,
                wait_ms,
                poll_ms,
                receipt,
                verify,
            })
        }
        "result-read" => {
            let mut result_id = None;
            let mut off = 0usize;
            let mut len = None;
            let mut out = None;
            let mut it = args.iter();
            while let Some(a) = it.next() {
                match a.as_str() {
                    "--off" => {
                        off = it
                            .next()
                            .ok_or("--off needs a byte count")?
                            .parse()
                            .map_err(|_| "--off must be an integer")?
                    }
                    "--len" => {
                        len = Some(
                            it.next()
                                .ok_or("--len needs a byte count")?
                                .parse()
                                .map_err(|_| "--len must be an integer")?,
                        )
                    }
                    "--out" => out = Some(PathBuf::from(it.next().ok_or("--out needs a file")?)),
                    "-h" | "--help" => return Err(usage().to_string()),
                    other => {
                        if result_id.is_some() {
                            return Err(format!("unknown argument '{other}'"));
                        }
                        result_id = Some(other.to_string());
                    }
                }
            }
            Ok(Plan::ResultRead {
                result_id: result_id.ok_or("result-read needs <result_id>")?,
                off,
                len: len.ok_or("--len is required")?,
                out: out.ok_or("--out is required")?,
            })
        }
        "result-drop" => Ok(Plan::ResultDrop {
            result_id: one_id(args, "result-drop")?,
        }),
        "job-list" => Ok(Plan::JobList),
        "drop" => {
            if args.is_empty() {
                return Err(format!("drop needs <name>\n\n{}", usage()));
            }
            Ok(Plan::Drop {
                name: args.remove(0),
            })
        }
        "-h" | "--help" | "help" => Err(usage().to_string()),
        _ => Err(format!("unknown command '{cmd}'\n\n{}", usage())),
    }
}

fn real_main() -> Result<(), String> {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    let mut addr = "127.0.0.1:9999".to_string();
    let mut token: Option<String> = None;
    loop {
        match args.first().map(|s| s.as_str()) {
            Some("--addr") => {
                args.remove(0);
                addr = args
                    .first()
                    .cloned()
                    .ok_or("--addr needs a value".to_string())?;
                args.remove(0);
            }
            Some("--token") => {
                args.remove(0);
                token = Some(
                    args.first()
                        .cloned()
                        .ok_or("--token needs a value".to_string())?,
                );
                args.remove(0);
            }
            _ => break,
        }
    }
    if args.is_empty() {
        return Err(usage().to_string());
    }
    let cmd = args.remove(0);
    // Validate all subcommand arguments before connecting so malformed
    // commands fail with usage, not a connection error.
    let plan = parse_plan(&cmd, args)?;
    let mut stream = connect(&addr)?;
    match plan {
        Plan::Ping => {
            let (h, _) = roundtrip(&mut stream, 1, "ping", json!({}), &[], &token)?;
            ok_or_show(&h)?;
            println!("{}", serde_json::to_string_pretty(&h).unwrap());
        }
        Plan::List => {
            let (h, _) = roundtrip(&mut stream, 2, "list", json!({}), &[], &token)?;
            ok_or_show(&h)?;
            println!("{}", serde_json::to_string_pretty(&h).unwrap());
        }
        Plan::Program {
            name,
            file,
            backend,
        } => {
            let listing =
                std::fs::read(&file).map_err(|e| format!("cannot read {}: {e}", file.display()))?;
            let (h, _) = roundtrip(
                &mut stream,
                3,
                "program",
                json!({"name": name, "backend": backend}),
                &listing,
                &token,
            )?;
            ok_or_show(&h)?;
            println!("{}", serde_json::to_string_pretty(&h).unwrap());
        }
        Plan::Store {
            name,
            payload: payload_file,
            stream: stream_name,
        } => {
            let feeds = read_feeds(&payload_file)?;
            let v = feeds
                .get(&stream_name)
                .ok_or_else(|| format!("stream '{stream_name}' not in {}", payload_file.display()))?
                .clone();
            let (spec, bytes) = encode_stream(&v).map_err(|e| e.to_string())?;
            let fields = json!({"name": name, "stream": spec.json()});
            let (h, _) = roundtrip(&mut stream, 4, "store", fields, &bytes, &token)?;
            ok_or_show(&h)?;
            println!("{}", serde_json::to_string_pretty(&h).unwrap());
        }
        Plan::Exec {
            handle,
            payload: payload_file,
            outputs,
            threads,
            policy,
            receipt,
        } => {
            let feeds = read_feeds(&payload_file)?;
            let (fields, body) = exec_body(&handle, &feeds, &outputs, threads, &policy)?;
            let (h, resp) = roundtrip(&mut stream, 5, "exec", fields, &body, &token)?;
            ok_or_show(&h)?;
            print_streams(&h, &resp, receipt)?;
        }
        Plan::Submit(job) => {
            submit_request(&mut stream, &token, &job, 6, true)?;
        }
        Plan::JobRun(job) => {
            let job_id = submit_request(&mut stream, &token, &job, 7, false)?;
            eprintln!(
                "{}",
                serde_json::to_string_pretty(&json!({"job_id": job_id})).unwrap()
            );
            wait_job(
                &mut stream,
                &token,
                &job_id,
                job.wait_ms,
                job.poll_ms,
                job.receipt,
                job.verify,
                8,
            )?;
        }
        Plan::JobBatch(job) => {
            run_job_batch(&mut stream, &token, &job)?;
        }
        Plan::JobStatus { job_id } => {
            let (h, _) = roundtrip(
                &mut stream,
                10,
                "job-status",
                json!({"job_id": job_id}),
                &[],
                &token,
            )?;
            ok_or_show(&h)?;
            println!("{}", serde_json::to_string_pretty(&h).unwrap());
        }
        Plan::JobResult {
            job_id,
            receipt,
            verify,
        } => {
            let (h, resp) = roundtrip(
                &mut stream,
                11,
                "job-result",
                json!({"job_id": job_id}),
                &[],
                &token,
            )?;
            ok_or_show(&h)?;
            finish_job_result(&mut stream, &token, h, resp, 100, receipt, verify)?;
        }
        Plan::JobCancel { job_id } => {
            let (h, _) = roundtrip(
                &mut stream,
                12,
                "job-cancel",
                json!({"job_id": job_id}),
                &[],
                &token,
            )?;
            ok_or_show(&h)?;
            println!("{}", serde_json::to_string_pretty(&h).unwrap());
        }
        Plan::JobWait {
            job_id,
            wait_ms,
            poll_ms,
            receipt,
            verify,
        } => {
            wait_job(
                &mut stream,
                &token,
                &job_id,
                wait_ms,
                poll_ms,
                receipt,
                verify,
                13,
            )?;
        }
        Plan::ResultRead {
            result_id,
            off,
            len,
            out,
        } => {
            let (h, resp) = roundtrip(
                &mut stream,
                20,
                "result-read",
                json!({"result_id": result_id, "off": off, "len": len}),
                &[],
                &token,
            )?;
            ok_or_show(&h)?;
            let hash = sha256_hex(&resp);
            if h["hash"].as_str() != Some(hash.as_str()) {
                return Err(format!(
                    "chunk hash mismatch: server {} client {hash}",
                    h["hash"]
                ));
            }
            std::fs::write(&out, &resp).map_err(|e| format!("write {}: {e}", out.display()))?;
            println!(
                "{}",
                serde_json::to_string_pretty(&json!({
                    "result_id": h["result_id"],
                    "off": h["off"],
                    "len": resp.len(),
                    "total": h["total"],
                    "hash": hash,
                    "eof": h["eof"],
                    "file": out.display().to_string(),
                }))
                .unwrap()
            );
        }
        Plan::ResultDrop { result_id } => {
            let (h, _) = roundtrip(
                &mut stream,
                21,
                "result-drop",
                json!({"result_id": result_id}),
                &[],
                &token,
            )?;
            ok_or_show(&h)?;
            println!("{}", serde_json::to_string_pretty(&h).unwrap());
        }
        Plan::JobList => {
            let (h, _) = roundtrip(&mut stream, 14, "job-list", json!({}), &[], &token)?;
            ok_or_show(&h)?;
            println!("{}", serde_json::to_string_pretty(&h).unwrap());
        }
        Plan::Drop { name } => {
            let (h, _) = roundtrip(&mut stream, 6, "drop", json!({"name": name}), &[], &token)?;
            ok_or_show(&h)?;
            println!("{}", serde_json::to_string_pretty(&h).unwrap());
        }
    }
    Ok(())
}

fn exec_body(
    handle: &str,
    feeds: &Feeds,
    outputs: &[String],
    threads: Option<usize>,
    policy: &Option<String>,
) -> Result<(Json, Vec<u8>), String> {
    // Inline all inputs as segments.
    let mut inputs = serde_json::Map::new();
    let mut body = Vec::new();
    for (name, v) in feeds {
        let (spec, bytes) = encode_stream(v).map_err(|e| e.to_string())?;
        let off = body.len();
        body.extend_from_slice(&bytes);
        let mut sj = spec.json();
        su(sj.as_object_mut(), "off", json!(off));
        su(sj.as_object_mut(), "len", json!(bytes.len()));
        inputs.insert(name.clone(), sj);
    }
    let mut fields = json!({"handle": handle, "inputs": Json::Object(inputs), "outputs": outputs});
    if let Some(o) = fields.as_object_mut() {
        if let Some(t) = threads {
            o.insert("threads".into(), json!(t));
        }
        if let Some(p) = policy {
            o.insert("thread_policy".into(), json!(p));
        }
    }
    Ok((fields, body))
}

fn submit_request(
    stream: &mut TcpStream,
    token: &Option<String>,
    job: &JobArgs,
    id: u64,
    show: bool,
) -> Result<String, String> {
    let feeds = read_feeds(&job.payload)?;
    let (mut fields, body) =
        exec_body(&job.handle, &feeds, &job.outputs, job.threads, &job.policy)?;
    if let Some(o) = fields.as_object_mut() {
        if let Some(ms) = job.timeout_ms {
            o.insert("timeout_ms".into(), json!(ms));
        }
        if let Some(key) = &job.idempotency_key {
            o.insert("idempotency_key".into(), json!(key));
        }
    }
    let (h, _) = roundtrip(stream, id, "submit", fields, &body, token)?;
    ok_or_show(&h)?;
    if show {
        println!("{}", serde_json::to_string_pretty(&h).unwrap());
    }
    h.get("job_id")
        .and_then(|x| x.as_str())
        .map(|s| s.to_string())
        .ok_or_else(|| "submit response missing job_id".to_string())
}

fn sha256_hex(bytes: &[u8]) -> String {
    let mut h = Sha256::new();
    h.update(bytes);
    format!("{:x}", h.finalize())
}

#[derive(Clone, Debug)]
struct StoredMeta {
    id: String,
    bytes: usize,
    chunk_bytes: usize,
    chunks: usize,
    hash: String,
    chunk_hashes: Vec<String>,
}

fn stored_meta(h: &Json) -> Result<Option<StoredMeta>, String> {
    let Some(result) = h.get("result") else {
        return Ok(None);
    };
    if result.get("transport").and_then(|x| x.as_str()) != Some("store") {
        return Ok(None);
    }
    let s = |k: &str| {
        result
            .get(k)
            .and_then(|x| x.as_str())
            .ok_or_else(|| format!("stored result missing {k}"))
    };
    let n = |k: &str| {
        result
            .get(k)
            .and_then(|x| x.as_u64())
            .map(|v| v as usize)
            .ok_or_else(|| format!("stored result missing {k}"))
    };
    let chunk_hashes = result
        .get("chunk_hashes")
        .and_then(|x| x.as_array())
        .ok_or("stored result missing chunk_hashes")?
        .iter()
        .map(|v| {
            v.as_str()
                .map(|s| s.to_string())
                .ok_or_else(|| "chunk_hashes must be strings".to_string())
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(Some(StoredMeta {
        id: s("id")?.to_string(),
        bytes: n("bytes")?,
        chunk_bytes: n("chunk_bytes")?,
        chunks: n("chunks")?,
        hash: s("hash")?.to_string(),
        chunk_hashes,
    }))
}

fn fetch_stored_payload(
    stream: &mut TcpStream,
    token: &Option<String>,
    meta: &StoredMeta,
    mut id: u64,
) -> Result<Vec<u8>, String> {
    let mut payload = Vec::with_capacity(meta.bytes);
    for i in 0..meta.chunks {
        let off = i * meta.chunk_bytes;
        let len = (meta.bytes - off).min(meta.chunk_bytes);
        let (h, resp) = roundtrip(
            stream,
            id,
            "result-read",
            json!({"result_id": meta.id, "off": off, "len": len}),
            &[],
            token,
        )?;
        id += 1;
        ok_or_show(&h)?;
        let got = h["hash"].as_str().unwrap_or("");
        let want = sha256_hex(&resp);
        if got != want {
            return Err(format!(
                "chunk {i} hash mismatch: server {got} client {want}"
            ));
        }
        if let Some(expected) = meta.chunk_hashes.get(i) {
            if expected != &want {
                return Err(format!("chunk {i} does not match manifest"));
            }
        }
        if resp.len() != len {
            return Err(format!("chunk {i} length {} != {len}", resp.len()));
        }
        if h["eof"].as_bool() != Some(i + 1 == meta.chunks) {
            return Err(format!("chunk {i} eof flag mismatch"));
        }
        payload.extend_from_slice(&resp);
    }
    let got = sha256_hex(&payload);
    if got != meta.hash {
        return Err(format!(
            "result hash mismatch: server {} client {got}",
            meta.hash
        ));
    }
    Ok(payload)
}

fn finish_job_result(
    stream: &mut TcpStream,
    token: &Option<String>,
    header: Json,
    inline_payload: Vec<u8>,
    id: u64,
    receipt: bool,
    verify: bool,
) -> Result<(), String> {
    if let Some(meta) = stored_meta(&header)? {
        let payload = fetch_stored_payload(stream, token, &meta, id)?;
        if receipt {
            if let Some(r) = header.get("receipt") {
                eprintln!("{}", serde_json::to_string_pretty(r).unwrap());
            }
        }
        if verify {
            println!(
                "{}",
                serde_json::to_string_pretty(&json!({
                    "result": {
                        "transport": "store",
                        "bytes": meta.bytes,
                        "chunks": meta.chunks,
                        "hash": meta.hash,
                        "verified": true,
                    }
                }))
                .unwrap()
            );
        } else {
            print_streams(&header, &payload, false)?;
        }
        return Ok(());
    }

    if let Some(hash) = header
        .get("result")
        .and_then(|r| r.get("hash"))
        .and_then(|x| x.as_str())
    {
        let got = sha256_hex(&inline_payload);
        if got != hash {
            return Err(format!(
                "inline result hash mismatch: server {hash} client {got}"
            ));
        }
    }
    if verify {
        if receipt {
            if let Some(r) = header.get("receipt") {
                eprintln!("{}", serde_json::to_string_pretty(r).unwrap());
            }
        }
        println!(
            "{}",
            serde_json::to_string_pretty(&json!({
                "result": {
                    "transport": "inline",
                    "bytes": inline_payload.len(),
                    "hash": sha256_hex(&inline_payload),
                    "verified": true,
                }
            }))
            .unwrap()
        );
    } else {
        print_streams(&header, &inline_payload, receipt)?;
    }
    Ok(())
}

fn wait_job(
    stream: &mut TcpStream,
    token: &Option<String>,
    job_id: &str,
    wait_ms: u64,
    poll_ms: u64,
    receipt: bool,
    verify: bool,
    mut id: u64,
) -> Result<(), String> {
    let deadline = Instant::now() + Duration::from_millis(wait_ms);
    let state = loop {
        let (h, _) = roundtrip(
            stream,
            id,
            "job-status",
            json!({"job_id": job_id}),
            &[],
            token,
        )?;
        id += 1;
        ok_or_show(&h)?;
        let state = h["job"]["state"].as_str().unwrap_or("unknown").to_string();
        if matches!(state.as_str(), "done" | "failed" | "canceled" | "timed_out") {
            break state;
        }
        if Instant::now() >= deadline {
            return Err(format!("job {job_id} still {state} after {wait_ms}ms"));
        }
        std::thread::sleep(Duration::from_millis(poll_ms.max(1)));
    };

    let (h, resp) = roundtrip(
        stream,
        id,
        "job-result",
        json!({"job_id": job_id}),
        &[],
        token,
    )?;
    if h.get("ok").and_then(|x| x.as_bool()) == Some(true) {
        finish_job_result(stream, token, h, resp, id, receipt, verify)?;
    } else {
        eprintln!("terminal state: {state}");
        ok_or_show(&h)?;
    }
    Ok(())
}

fn run_job_batch(
    stream: &mut TcpStream,
    token: &Option<String>,
    job: &JobArgs,
) -> Result<(), String> {
    let feeds = read_feeds(&job.payload)?;
    let (base_fields, body) =
        exec_body(&job.handle, &feeds, &job.outputs, job.threads, &job.policy)?;

    let start = Instant::now();
    let mut ids = Vec::with_capacity(job.count);
    let mut id = 100u64;
    for _ in 0..job.count {
        let mut fields = base_fields.clone();
        if let Some(o) = fields.as_object_mut() {
            if let Some(ms) = job.timeout_ms {
                o.insert("timeout_ms".into(), json!(ms));
            }
        }
        let h = loop {
            let (h, _) = roundtrip(stream, id, "submit", fields.clone(), &body, token)?;
            id += 1;
            if h.get("ok").and_then(|x| x.as_bool()) == Some(true) {
                break h;
            }
            let code = h["error"]["code"].as_str().unwrap_or("");
            if code == "queue-full" {
                // Backpressure: wait for a worker to drain the queue.
                std::thread::sleep(Duration::from_millis(job.poll_ms.max(1)));
                continue;
            }
            ok_or_show(&h)?;
        };
        ids.push(
            h.get("job_id")
                .and_then(|x| x.as_str())
                .ok_or("submit response missing job_id")?
                .to_string(),
        );
    }
    let submit_ms = start.elapsed().as_millis() as u64;

    let deadline = Instant::now() + Duration::from_millis(job.wait_ms);
    loop {
        let (h, _) = roundtrip(stream, id, "job-list", json!({}), &[], token)?;
        id += 1;
        ok_or_show(&h)?;
        let jobs = h["jobs"].as_array().cloned().unwrap_or_default();
        let terminal = jobs
            .iter()
            .filter(|j| {
                let known = j["job_id"]
                    .as_str()
                    .map(|s| ids.iter().any(|id| id == s))
                    .unwrap_or(false);
                let state = j["state"].as_str().unwrap_or("");
                known && matches!(state, "done" | "failed" | "canceled" | "timed_out")
            })
            .count();
        if terminal == ids.len() {
            break;
        }
        if Instant::now() >= deadline {
            return Err(format!(
                "batch still has {} active job(s) after {}ms",
                ids.len() - terminal,
                job.wait_ms
            ));
        }
        std::thread::sleep(Duration::from_millis(job.poll_ms.max(1)));
    }

    let mut done = 0usize;
    let mut failed = 0usize;
    let mut output_bytes = 0usize;
    let mut output_hash: Option<String> = None;
    let mut identical = true;
    for job_id in &ids {
        let (h, resp) = roundtrip(
            stream,
            id,
            "job-result",
            json!({"job_id": job_id}),
            &[],
            token,
        )?;
        id += 1;
        if h.get("ok").and_then(|x| x.as_bool()) == Some(true) {
            done += 1;
            if let Some(meta) = stored_meta(&h)? {
                output_bytes += meta.bytes;
                if job.verify {
                    let _ = fetch_stored_payload(stream, token, &meta, id)?;
                    id += meta.chunks as u64;
                }
                let hash = meta.hash.clone();
                match &output_hash {
                    None => output_hash = Some(hash),
                    Some(previous) if previous != &hash => identical = false,
                    Some(_) => {}
                }
            } else {
                output_bytes += resp.len();
                let hash = sha256_hex(&resp);
                if let Some(expected) = h
                    .get("result")
                    .and_then(|r| r.get("hash"))
                    .and_then(|x| x.as_str())
                {
                    if expected != hash {
                        return Err(format!(
                            "inline result hash mismatch for {job_id}: server {expected} client {hash}"
                        ));
                    }
                }
                match &output_hash {
                    None => output_hash = Some(hash),
                    Some(previous) if previous != &hash => identical = false,
                    Some(_) => {}
                }
            }
            if job.receipt {
                if let Some(r) = h.get("receipt") {
                    eprintln!("{}", serde_json::to_string_pretty(r).unwrap());
                }
            }
        } else {
            failed += 1;
            eprintln!(
                "job {job_id} failed: {}",
                h.get("error").cloned().unwrap_or(json!("?"))
            );
        }
    }

    let wall = start.elapsed();
    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "submitted": ids.len(),
            "done": done,
            "failed": failed,
            "submit_ms": submit_ms,
            "wall_ms": wall.as_millis() as u64,
            "jobs_per_s": (ids.len() as f64) / wall.as_secs_f64().max(1e-9),
            "output_bytes": output_bytes,
            "identical_outputs": identical,
            "output_hash": output_hash,
        }))
        .unwrap()
    );
    if failed > 0 {
        return Err(format!("{failed} job(s) failed"));
    }
    if !identical {
        return Err("batch outputs were not byte-identical".to_string());
    }
    Ok(())
}

fn print_streams(h: &Json, resp: &[u8], receipt: bool) -> Result<(), String> {
    if receipt {
        if let Some(r) = h.get("receipt") {
            eprintln!("{}", serde_json::to_string_pretty(r).unwrap());
        }
    }
    // Decode outputs to JSON streams where specs are present.
    let mut out_streams = serde_json::Map::new();
    if let Some(specs) = h.get("outputs").and_then(|x| x.as_object()) {
        for (name, s) in specs {
            let spec = StreamSpec::from_json(s).map_err(|e| e.to_string())?;
            let off = s.get("off").and_then(|x| x.as_u64()).unwrap_or(0) as usize;
            let len = s.get("len").and_then(|x| x.as_u64()).unwrap_or(0) as usize;
            let end = off.checked_add(len).ok_or("exec output offset overflow")?;
            if end > resp.len() {
                return Err(format!("exec output {name} truncated"));
            }
            let v = decode_stream(&spec, &resp[off..end]).map_err(|e| e.to_string())?;
            out_streams.insert(
                name.clone(),
                payload::value_to_json(&v).map_err(|e| e.to_string())?,
            );
        }
    }
    println!(
        "{}",
        serde_json::to_string_pretty(&json!({"streams": Json::Object(out_streams)})).unwrap()
    );
    Ok(())
}

fn su(m: Option<&mut serde_json::Map<String, Json>>, k: &str, v: Json) {
    if let Some(o) = m {
        o.insert(k.to_string(), v);
    }
}

fn read_feeds(path: &std::path::Path) -> Result<Feeds, String> {
    payload::read_payload(path).map_err(|e| e.to_string())
}

fn parse_payload_args(args: &[String]) -> Result<(PathBuf, String), String> {
    let mut file: Option<PathBuf> = None;
    let mut stream_name: Option<String> = None;
    let mut it = args.iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--payload" => file = Some(PathBuf::from(it.next().ok_or("--payload needs a file")?)),
            "--stream" => stream_name = Some(it.next().ok_or("--stream needs a name")?.clone()),
            "-h" | "--help" => return Err(usage().to_string()),
            other => return Err(format!("unknown argument '{other}'")),
        }
    }
    Ok((
        file.ok_or("--payload is required")?,
        stream_name.ok_or("--stream is required")?,
    ))
}

fn parse_exec_args(
    args: &[String],
) -> Result<(PathBuf, Vec<String>, Option<usize>, Option<String>, bool), String> {
    let mut file: Option<PathBuf> = None;
    let mut outputs: Option<Vec<String>> = None;
    let mut threads: Option<usize> = None;
    let mut policy: Option<String> = None;
    let mut receipt = false;
    let mut it = args.iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--payload" => file = Some(PathBuf::from(it.next().ok_or("--payload needs a file")?)),
            "--outputs" => {
                let s = it.next().ok_or("--outputs needs a comma list")?;
                outputs = Some(s.split(',').map(|s| s.trim().to_string()).collect());
            }
            "--threads" => {
                let s = it.next().ok_or("--threads needs a count")?;
                let n = s
                    .parse::<usize>()
                    .map_err(|e| format!("bad --threads: {e}"))?;
                if n == 0 {
                    return Err("--threads must be >= 1".to_string());
                }
                threads = Some(n);
            }
            "--policy" => {
                let s = it.next().ok_or("--policy needs a value")?;
                if !matches!(
                    s.as_str(),
                    "reference" | "serial" | "auto" | "deterministic-auto"
                ) {
                    return Err(format!("bad --policy {s:?}"));
                }
                policy = Some(s.to_string());
            }
            "--receipt" => receipt = true,
            "-h" | "--help" => return Err(usage().to_string()),
            other => return Err(format!("unknown argument '{other}'")),
        }
    }
    Ok((
        file.ok_or("--payload is required")?,
        outputs.ok_or("--outputs is required")?,
        threads,
        policy,
        receipt,
    ))
}
