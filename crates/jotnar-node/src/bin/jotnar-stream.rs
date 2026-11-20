//! Streaming controller CLI.
//!
//! Reads a `jotnar.stream/1` descriptor, runs the frame loop against the GPU
//! executors, and writes a per-frame receipt and summary. The wall is the first
//! customer; DOOM will use the same controller once its source/sink adapters
//! are wired.

use std::path::PathBuf;
use std::process::ExitCode;

use jotnar_node_tools::stream::{run_stream, GpuEndpoint, StreamDescriptor, StreamRunConfig};

fn usage() -> &'static str {
    "jotnar-stream — typed streaming controller above the executors\n\
     \n\
     usage:\n\
       jotnar-stream --descriptor FILE --cuda ADDR --hip ADDR [--token T]\n\
                     [--frames N] [--fleet-dir DIR] [--lease-ms N]\n\
                     [--reference DIR] [--clip DIR] [--out DIR] [--receipt DIR]\n\
                     [--all-cuda] [--timeout-ms N]\n\
     \n\
     The descriptor declares the stream clock, source, tiled GPU stage, sink,\n\
     and receipt policy. `--clip` overrides a frames source; `--out` overrides\n\
     a frames sink. `--reference` compares each composited frame to\n\
     frame_%06d.rgb in that directory."
}

struct Args {
    descriptor: PathBuf,
    cuda: Option<String>,
    hip: Option<String>,
    token: Option<String>,
    frames: Option<u64>,
    fleet_dir: Option<PathBuf>,
    lease_ms: u64,
    reference: Option<PathBuf>,
    clip: Option<PathBuf>,
    out: Option<PathBuf>,
    receipt: Option<PathBuf>,
    all_cuda: bool,
    timeout_ms: u64,
}

fn parse() -> Result<Args, String> {
    let raw: Vec<String> = std::env::args().skip(1).collect();
    let mut it = raw.into_iter();
    let mut descriptor = None;
    let mut cuda = None;
    let mut hip = None;
    let mut token = None;
    let mut frames = None;
    let mut fleet_dir = None;
    let mut lease_ms = 3_000u64;
    let mut reference = None;
    let mut clip = None;
    let mut out = None;
    let mut receipt = None;
    let mut all_cuda = false;
    let mut timeout_ms = 10_000u64;
    while let Some(arg) = it.next() {
        let mut value = |flag: &str| it.next().ok_or_else(|| format!("{flag} needs a value"));
        match arg.as_str() {
            "--descriptor" => descriptor = Some(PathBuf::from(value("--descriptor")?)),
            "--cuda" => cuda = Some(value("--cuda")?),
            "--hip" => hip = Some(value("--hip")?),
            "--token" => token = Some(value("--token")?),
            "--frames" => {
                frames = Some(
                    value("--frames")?
                        .parse()
                        .map_err(|_| "--frames must be an integer")?,
                )
            }
            "--fleet-dir" => fleet_dir = Some(PathBuf::from(value("--fleet-dir")?)),
            "--lease-ms" => {
                lease_ms = value("--lease-ms")?
                    .parse()
                    .map_err(|_| "--lease-ms must be an integer")?
            }
            "--reference" => reference = Some(PathBuf::from(value("--reference")?)),
            "--clip" => clip = Some(PathBuf::from(value("--clip")?)),
            "--out" => out = Some(PathBuf::from(value("--out")?)),
            "--receipt" => receipt = Some(PathBuf::from(value("--receipt")?)),
            "--all-cuda" => all_cuda = true,
            "--timeout-ms" => {
                timeout_ms = value("--timeout-ms")?
                    .parse()
                    .map_err(|_| "--timeout-ms must be an integer")?
            }
            "-h" | "--help" => return Err(usage().to_string()),
            other => return Err(format!("unknown argument {other:?}\n\n{}", usage())),
        }
    }
    Ok(Args {
        descriptor: descriptor.ok_or("need --descriptor FILE")?,
        cuda,
        hip,
        token,
        frames,
        fleet_dir,
        lease_ms: lease_ms.max(1),
        reference,
        clip,
        out,
        receipt,
        all_cuda,
        timeout_ms: timeout_ms.max(1),
    })
}

fn real_main() -> Result<(), String> {
    let args = parse()?;
    let (mut descriptor, sha) = StreamDescriptor::load(&args.descriptor)?;
    if let Some(clip) = args.clip {
        if descriptor.source.kind != "frames" {
            return Err("--clip only applies to a frames source".to_string());
        }
        descriptor.source.dir = Some(clip);
    }
    if let Some(out) = args.out {
        if descriptor.sink.kind != "frames" {
            return Err("--out only applies to a frames sink".to_string());
        }
        descriptor.sink.dir = Some(out);
    }
    if let Some(receipt) = args.receipt {
        descriptor
            .receipt
            .get_or_insert_with(|| jotnar_node_tools::stream::ReceiptSpec {
                kind: "every-frame".to_string(),
                dir: None,
            })
            .dir = Some(receipt);
    }
    if descriptor.stage.targets.is_empty() {
        return Err("stream stage has no targets".to_string());
    }
    let mut endpoints = Vec::new();
    for target in &descriptor.stage.targets {
        match target.as_str() {
            "cuda" => {
                let addr = args
                    .cuda
                    .clone()
                    .ok_or("stream targets cuda but --cuda ADDR is missing")?;
                endpoints.push(GpuEndpoint::new("cuda", &addr, args.token.clone()));
            }
            "hip" => {
                let addr = args
                    .hip
                    .clone()
                    .ok_or("stream targets hip but --hip ADDR is missing")?;
                endpoints.push(GpuEndpoint::new("hip", &addr, args.token.clone()));
            }
            other => return Err(format!("unknown GPU target {other:?}")),
        }
    }
    if endpoints.is_empty() {
        return Err("stream needs at least one GPU endpoint".to_string());
    }
    let cfg = StreamRunConfig {
        frame_limit: args.frames,
        all_first_target: args.all_cuda,
        reference_dir: args.reference,
        fleet_dir: args.fleet_dir,
        lease_ms: args.lease_ms,
        timeout_ms: args.timeout_ms,
    };
    let summary = run_stream(&descriptor, &sha, &endpoints, &cfg)?;
    println!(
        "{}",
        serde_json::to_string_pretty(&summary).map_err(|e| e.to_string())?
    );
    Ok(())
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
