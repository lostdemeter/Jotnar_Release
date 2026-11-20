//! Workload-controller CLI.
//!
//! The controller owns the bundle, stage graph, placement, checkpoint movement,
//! and the receipt. Executors stay model-blind/process-blind: they run one
//! stage each. The first workload is DOOM on the native RV32IM core.
//!
//! ```sh
//! jotnar-workload bundle-doom --image doom.bin --wad doom1.wad --out doom-bundle
//! jotnar-workload run --bundle doom-bundle --ledger doom-run \
//!   --program target/release/jotnar-demo-doom \
//!   --place frame1=127.0.0.1:9998 --place frame2=127.0.0.1:9999 \
//!   --compare frame2.fb=/tmp/reference/frame2.fb
//! ```

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Duration;

use serde_json::{json, Value as Json};

use jotnar_node_tools::workload::{
    build_doom_bundle, build_gpu_bundle, build_stage_chain, run_bundle, ExecutorEndpoint,
    FleetConfig, GpuBundleSpec, Ledger, LoadedBundle, RunConfig, StageChainSpec,
};

#[allow(dead_code)]
#[path = "../sched.rs"]
mod sched;

fn usage() -> &'static str {
    r#"jotnar-workload — model-aware stage controller above the executors

usage:
  jotnar-workload [--token T] [--discover-timeout-ms N] bundle-doom \
    --image FILE --wad FILE --out DIR [--frames N] [--name NAME]
  jotnar-workload bundle-gpu --listing FILE --out DIR \
    (--input PAYLOAD.json | --generate HxWxC) \
    [--items N] [--name NAME] [--timeout-ms N]
  jotnar-workload bundle-stage-chain --listing FILE --descriptor FILE \
    [--descriptor FILE]... --out DIR \
    (--input PAYLOAD.json | --generate HxWxC) [--name NAME] [--timeout-ms N]
  jotnar-workload [--token T] plan --bundle PATH [--place STAGE=ADDR]...
  jotnar-workload [--token T] [--discover-timeout-ms N] run \
    --bundle PATH --ledger DIR [--program FILE] \
    [--executor ADDR]... [--gpu-executor TARGET=ADDR]... \
    [--place STAGE=ADDR]... [--discover-native] [--discover-gpu] \
    [--force] [--timeout-ms N] [--compare STAGE:OUTPUT=FILE]... \
    [--fleet-dir DIR] [--fleet-slots N] [--fleet-max-wait-ms N] \
    [--fleet-lease-ms N]
  jotnar-workload status --ledger DIR

A DOOM run moves frame1's checkpoint to frame2's executor; the second
stage uses --resume and an absolute --frames target.
A GPU run sends listing + JSON payload to a target-aware runner; a stage
output can feed the next stage's payload without a JSON round trip in the
bundle.
--executor is a native executor. --gpu-executor takes cuda=ADDR or
hip=ADDR.
--fleet-dir enables advisory batch admission against live interactive
leases in a shared directory; without it, behavior is best effort."#
}

struct Global {
    token: Option<String>,
    discover_timeout_ms: u64,
}

fn parse_global() -> Result<(Global, String, Vec<String>), String> {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    let mut token = None;
    let mut discover_timeout_ms = 3000u64;
    while let Some(a) = args.first().cloned() {
        match a.as_str() {
            "--token" => {
                args.remove(0);
                token = Some(args.first().cloned().ok_or("--token needs a value")?);
                args.remove(0);
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
            "-h" | "--help" => {
                println!("{}", usage());
                std::process::exit(0);
            }
            _ => break,
        }
    }
    let cmd = args.first().cloned().unwrap_or_default();
    if args.is_empty() {
        return Err(usage().to_string());
    }
    args.remove(0);
    Ok((
        Global {
            token,
            discover_timeout_ms,
        },
        cmd,
        args,
    ))
}

fn take_value(it: &mut impl Iterator<Item = String>, flag: &str) -> Result<String, String> {
    it.next().ok_or_else(|| format!("{flag} needs a value"))
}

fn parse_place(value: &str) -> Result<(String, String), String> {
    let (stage, addr) = value
        .split_once('=')
        .ok_or_else(|| format!("--place needs STAGE=ADDR, got {value:?}"))?;
    if stage.is_empty() || addr.is_empty() {
        return Err(format!("--place needs STAGE=ADDR, got {value:?}"));
    }
    Ok((stage.to_string(), addr.to_string()))
}

fn bundle_doom(args: Vec<String>) -> Result<Json, String> {
    let mut image = None;
    let mut wad = None;
    let mut out = None;
    let mut frames = 2u32;
    let mut name = "doom-rv32im".to_string();
    let mut it = args.into_iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--image" => image = Some(PathBuf::from(take_value(&mut it, "--image")?)),
            "--wad" => wad = Some(PathBuf::from(take_value(&mut it, "--wad")?)),
            "--out" => out = Some(PathBuf::from(take_value(&mut it, "--out")?)),
            "--frames" => {
                frames = take_value(&mut it, "--frames")?
                    .parse()
                    .map_err(|_| "--frames must be an integer")?
            }
            "--name" => name = take_value(&mut it, "--name")?,
            other => return Err(format!("unknown bundle-doom argument {other:?}")),
        }
    }
    let image = image.ok_or("bundle-doom needs --image FILE")?;
    let wad = wad.ok_or("bundle-doom needs --wad FILE")?;
    let out = out.ok_or("bundle-doom needs --out DIR")?;
    let loaded = build_doom_bundle(&image, &wad, frames.max(1), &out, &name)?;
    let path = loaded.write(&out)?;
    Ok(json!({
        "bundle": path.display().to_string(),
        "sha256": loaded.sha256,
        "name": loaded.bundle.name,
        "version": loaded.bundle.version,
        "frames": frames.max(1),
        "stages": loaded.bundle.stages.iter().map(|s| &s.name).collect::<Vec<_>>(),
        "artifacts": loaded.bundle.artifacts,
    }))
}

fn parse_size(value: &str) -> Result<(usize, usize, usize), String> {
    let parts: Vec<&str> = value.split(['x', 'X']).collect();
    if parts.len() != 3 {
        return Err("--generate needs HxWxC".to_string());
    }
    let parse = |s: &str, what: &str| {
        s.parse::<usize>()
            .map_err(|_| format!("--generate needs HxWxC, bad {what}"))
    };
    let (h, w, c) = (
        parse(parts[0], "height")?,
        parse(parts[1], "width")?,
        parse(parts[2], "channels")?,
    );
    if h == 0 || w == 0 || c == 0 {
        return Err("--generate needs positive HxWxC".to_string());
    }
    Ok((h, w, c))
}

fn bundle_gpu(args: Vec<String>) -> Result<Json, String> {
    let mut listing = None;
    let mut input = None;
    let mut generate = None;
    let mut out = None;
    let mut items = 1usize;
    let mut name = "gpu-batch".to_string();
    let mut timeout_ms = 600_000u64;
    let mut it = args.into_iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--listing" => listing = Some(PathBuf::from(take_value(&mut it, "--listing")?)),
            "--input" => input = Some(PathBuf::from(take_value(&mut it, "--input")?)),
            "--generate" => generate = Some(parse_size(&take_value(&mut it, "--generate")?)?),
            "--out" => out = Some(PathBuf::from(take_value(&mut it, "--out")?)),
            "--items" => {
                items = take_value(&mut it, "--items")?
                    .parse()
                    .map_err(|_| "--items must be an integer")?
            }
            "--name" => name = take_value(&mut it, "--name")?,
            "--timeout-ms" => {
                timeout_ms = take_value(&mut it, "--timeout-ms")?
                    .parse()
                    .map_err(|_| "--timeout-ms must be an integer")?
            }
            other => return Err(format!("unknown bundle-gpu argument {other:?}")),
        }
    }
    let loaded = build_gpu_bundle(&GpuBundleSpec {
        listing: listing.ok_or("bundle-gpu needs --listing FILE")?,
        input,
        generate,
        items,
        out_dir: out.ok_or("bundle-gpu needs --out DIR")?,
        name,
        timeout_ms: timeout_ms.max(1),
    })?;
    let path = loaded.dir.join("bundle.json");
    Ok(json!({
        "bundle": path.display().to_string(),
        "sha256": loaded.sha256,
        "name": loaded.bundle.name,
        "version": loaded.bundle.version,
        "items": items,
        "stages": loaded.bundle.stages.iter().map(|s| &s.name).collect::<Vec<_>>(),
        "artifacts": loaded.bundle.artifacts,
    }))
}

fn bundle_stage_chain(args: Vec<String>) -> Result<Json, String> {
    let mut listing = None;
    let mut descriptors = Vec::new();
    let mut input = None;
    let mut generate = None;
    let mut out = None;
    let mut name = "stage-chain".to_string();
    let mut timeout_ms = 600_000u64;
    let mut it = args.into_iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--listing" => listing = Some(PathBuf::from(take_value(&mut it, "--listing")?)),
            "--descriptor" => descriptors.push(PathBuf::from(take_value(&mut it, "--descriptor")?)),
            "--input" => input = Some(PathBuf::from(take_value(&mut it, "--input")?)),
            "--generate" => generate = Some(parse_size(&take_value(&mut it, "--generate")?)?),
            "--out" => out = Some(PathBuf::from(take_value(&mut it, "--out")?)),
            "--name" => name = take_value(&mut it, "--name")?,
            "--timeout-ms" => {
                timeout_ms = take_value(&mut it, "--timeout-ms")?
                    .parse()
                    .map_err(|_| "--timeout-ms must be an integer")?
            }
            other => return Err(format!("unknown bundle-stage-chain argument {other:?}")),
        }
    }
    let loaded = build_stage_chain(&StageChainSpec {
        listing: listing.ok_or("bundle-stage-chain needs --listing FILE")?,
        descriptors,
        input,
        generate,
        out_dir: out.ok_or("bundle-stage-chain needs --out DIR")?,
        name,
        timeout_ms: timeout_ms.max(1),
    })?;
    let path = loaded.dir.join("bundle.json");
    Ok(json!({
        "bundle": path.display().to_string(),
        "sha256": loaded.sha256,
        "name": loaded.bundle.name,
        "version": loaded.bundle.version,
        "stages": loaded.bundle.stages.iter().map(|s| json!({
            "name": s.name,
            "target": s.target(),
            "outputs": s.outputs,
            "descriptor": s.gpu.as_ref().and_then(|g| g.descriptor.as_ref())
                .and_then(|d| d.artifact.clone()),
        })).collect::<Vec<_>>(),
        "artifacts": loaded.bundle.artifacts,
    }))
}

fn plan(args: Vec<String>) -> Result<Json, String> {
    let mut bundle = None;
    let mut placement = BTreeMap::new();
    let mut it = args.into_iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--bundle" => bundle = Some(PathBuf::from(take_value(&mut it, "--bundle")?)),
            "--place" => {
                let (stage, addr) = parse_place(&take_value(&mut it, "--place")?)?;
                placement.insert(stage, addr);
            }
            other => return Err(format!("unknown plan argument {other:?}")),
        }
    }
    let loaded = LoadedBundle::load(&bundle.ok_or("plan needs --bundle PATH")?)?;
    let order = loaded.bundle.stage_order()?;
    let stages: Vec<Json> = loaded
        .bundle
        .stages
        .iter()
        .map(|s| {
            json!({
                "name": s.name,
                "depends_on": s.depends_on,
                "command": s.command,
                "inputs": s.inputs,
                "outputs": s.outputs,
                "executor": s.executor,
                "gpu": s.gpu,
                "placement": placement.get(&s.name),
            })
        })
        .collect();
    Ok(json!({
        "name": loaded.bundle.name,
        "version": loaded.bundle.version,
        "sha256": loaded.sha256,
        "order": order,
        "stages": stages,
        "artifacts": loaded.bundle.artifacts,
    }))
}

fn add_discovered(caps: Vec<Json>, executors: &mut Vec<ExecutorEndpoint>) {
    for cap in caps {
        let addrs = cap
            .get("addrs")
            .and_then(|x| x.as_array())
            .cloned()
            .unwrap_or_default();
        let chosen = addrs
            .iter()
            .filter_map(|a| a.as_str())
            .find(|a| !a.starts_with("127."))
            .or_else(|| addrs.iter().filter_map(|a| a.as_str()).next())
            .map(|s| s.to_string());
        if let Some(addr) = chosen {
            if !executors.iter().any(|e| e.addr == addr) {
                executors.push(ExecutorEndpoint {
                    addr,
                    capability: cap,
                });
            }
        }
    }
}

fn run(global: &Global, args: Vec<String>) -> Result<Json, String> {
    let mut bundle = None;
    let mut ledger = None;
    let mut program = None;
    let mut executors = Vec::new();
    let mut placement = BTreeMap::new();
    let mut discover_native = false;
    let mut discover_gpu = false;
    let mut force = false;
    let mut timeout_ms = None;
    let mut compare = Vec::new();
    let mut fleet_dir = None;
    let mut fleet_slots = 1u32;
    let mut fleet_max_wait_ms = 30_000u64;
    let mut fleet_lease_ms = 120_000u64;
    let mut it = args.into_iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--bundle" => bundle = Some(PathBuf::from(take_value(&mut it, "--bundle")?)),
            "--ledger" => ledger = Some(PathBuf::from(take_value(&mut it, "--ledger")?)),
            "--program" => program = Some(take_value(&mut it, "--program")?),
            "--executor" => {
                let addr = take_value(&mut it, "--executor")?;
                if addr.is_empty() {
                    return Err("--executor needs a non-empty ADDR".to_string());
                }
                executors.push(ExecutorEndpoint {
                    capability: json!({"kind": "native", "target": "process", "addr": addr}),
                    addr,
                });
            }
            "--gpu-executor" => {
                let value = take_value(&mut it, "--gpu-executor")?;
                let (target, addr) = value
                    .split_once('=')
                    .ok_or_else(|| format!("--gpu-executor needs TARGET=ADDR, got {value:?}"))?;
                if !matches!(target, "cuda" | "hip") {
                    return Err(format!(
                        "--gpu-executor target must be cuda or hip, got {target:?}"
                    ));
                }
                if addr.is_empty() {
                    return Err(format!(
                        "--gpu-executor needs a non-empty ADDR, got {value:?}"
                    ));
                }
                executors.push(ExecutorEndpoint {
                    capability: json!({"kind": "gpu", "target": target, "addr": addr}),
                    addr: addr.to_string(),
                });
            }
            "--place" => {
                let (stage, addr) = parse_place(&take_value(&mut it, "--place")?)?;
                placement.insert(stage, addr);
            }
            "--discover-native" => discover_native = true,
            "--discover-gpu" => discover_gpu = true,
            "--force" => force = true,
            "--timeout-ms" => {
                timeout_ms = Some(
                    take_value(&mut it, "--timeout-ms")?
                        .parse()
                        .map_err(|_| "--timeout-ms must be an integer")?,
                )
            }
            "--compare" => {
                let value = take_value(&mut it, "--compare")?;
                let (key, path) = value
                    .split_once('=')
                    .ok_or_else(|| format!("--compare needs STAGE:OUTPUT=FILE, got {value:?}"))?;
                compare.push((key.to_string(), PathBuf::from(path)));
            }
            "--fleet-dir" => fleet_dir = Some(PathBuf::from(take_value(&mut it, "--fleet-dir")?)),
            "--fleet-slots" => {
                fleet_slots = take_value(&mut it, "--fleet-slots")?
                    .parse()
                    .map_err(|_| "--fleet-slots must be an integer")?
            }
            "--fleet-max-wait-ms" => {
                fleet_max_wait_ms = take_value(&mut it, "--fleet-max-wait-ms")?
                    .parse()
                    .map_err(|_| "--fleet-max-wait-ms must be an integer")?
            }
            "--fleet-lease-ms" => {
                fleet_lease_ms = take_value(&mut it, "--fleet-lease-ms")?
                    .parse()
                    .map_err(|_| "--fleet-lease-ms must be an integer")?
            }
            other => return Err(format!("unknown run argument {other:?}")),
        }
    }
    let loaded = LoadedBundle::load(&bundle.ok_or("run needs --bundle PATH")?)?;
    let order = loaded.bundle.stage_order()?;
    for stage in placement.keys() {
        if !order.iter().any(|s| s == stage) {
            return Err(format!("--place names unknown stage {stage:?}"));
        }
    }
    let timeout = Duration::from_millis(global.discover_timeout_ms.max(1));
    if discover_native {
        let before = executors.len();
        add_discovered(sched::discover_native(timeout)?, &mut executors);
        if executors.len() == before {
            return Err("no native executors discovered".to_string());
        }
    }
    if discover_gpu {
        let before = executors.len();
        add_discovered(sched::discover_gpu(timeout)?, &mut executors);
        if executors.len() == before {
            return Err("no GPU executors discovered".to_string());
        }
    }
    let cfg = RunConfig {
        program,
        executors,
        placement,
        token: global.token.clone(),
        force,
        timeout_ms,
        compare,
        fleet: fleet_dir.map(|dir| FleetConfig {
            dir,
            capacity: fleet_slots.max(1),
            max_wait_ms: fleet_max_wait_ms.max(1),
            lease_ms: fleet_lease_ms.max(1),
        }),
    };
    run_bundle(&loaded, &ledger.ok_or("run needs --ledger DIR")?, &cfg)
}

fn status(args: Vec<String>) -> Result<Json, String> {
    let mut ledger = None;
    let mut it = args.into_iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--ledger" => ledger = Some(PathBuf::from(take_value(&mut it, "--ledger")?)),
            other => return Err(format!("unknown status argument {other:?}")),
        }
    }
    let dir = ledger.ok_or("status needs --ledger DIR")?;
    let ledger = Ledger::read(&dir)?;
    let receipt = dir.join("receipt.json");
    let receipt: Json = if receipt.exists() {
        serde_json::from_slice(
            &std::fs::read(&receipt).map_err(|e| format!("read {}: {e}", receipt.display()))?,
        )
        .map_err(|e| format!("parse {}: {e}", receipt.display()))?
    } else {
        Json::Null
    };
    Ok(json!({"ledger": ledger, "receipt": receipt}))
}

fn real_main() -> Result<(), String> {
    let (global, cmd, args) = parse_global()?;
    let out = match cmd.as_str() {
        "bundle-doom" => bundle_doom(args)?,
        "bundle-gpu" => bundle_gpu(args)?,
        "bundle-stage-chain" => bundle_stage_chain(args)?,
        "plan" => plan(args)?,
        "run" => run(&global, args)?,
        "status" => status(args)?,
        other => return Err(format!("unknown command {other:?}\n\n{}", usage())),
    };
    println!("{}", serde_json::to_string_pretty(&out).unwrap());
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
