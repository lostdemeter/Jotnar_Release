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
    build_doom_bundle, run_bundle, ExecutorEndpoint, Ledger, LoadedBundle, RunConfig,
};

#[allow(dead_code)]
#[path = "../sched.rs"]
mod sched;

fn usage() -> &'static str {
    "jotnar-workload — model-aware stage controller above the executors\n\
     \n\
     usage:\n\
       jotnar-workload [--token T] [--discover-timeout-ms N] bundle-doom \\\n\
         --image FILE --wad FILE --out DIR [--frames N] [--name NAME]\n\
       jotnar-workload [--token T] plan --bundle PATH [--place STAGE=ADDR]...\n\
       jotnar-workload [--token T] [--discover-timeout-ms N] run \\\n\
         --bundle PATH --ledger DIR --program FILE \\\n\
         [--executor ADDR]... [--place STAGE=ADDR]... [--discover-native] \\\n\
         [--force] [--timeout-ms N] [--compare STAGE:OUTPUT=FILE]...\n\
       jotnar-workload status --ledger DIR\n\
     \n\
     A DOOM run moves frame1's checkpoint to frame2's executor; the second\n\
     stage uses --resume and an absolute --frames target."
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

fn run(global: &Global, args: Vec<String>) -> Result<Json, String> {
    let mut bundle = None;
    let mut ledger = None;
    let mut program = None;
    let mut executors = Vec::new();
    let mut placement = BTreeMap::new();
    let mut discover_native = false;
    let mut force = false;
    let mut timeout_ms = None;
    let mut compare = Vec::new();
    let mut it = args.into_iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--bundle" => bundle = Some(PathBuf::from(take_value(&mut it, "--bundle")?)),
            "--ledger" => ledger = Some(PathBuf::from(take_value(&mut it, "--ledger")?)),
            "--program" => program = Some(take_value(&mut it, "--program")?),
            "--executor" => {
                let addr = take_value(&mut it, "--executor")?;
                executors.push(ExecutorEndpoint {
                    capability: json!({"kind": "native", "target": "process", "addr": addr}),
                    addr,
                });
            }
            "--place" => {
                let (stage, addr) = parse_place(&take_value(&mut it, "--place")?)?;
                placement.insert(stage, addr);
            }
            "--discover-native" => discover_native = true,
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
    if discover_native {
        let caps =
            sched::discover_native(Duration::from_millis(global.discover_timeout_ms.max(1)))?;
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
        if executors.is_empty() {
            return Err("no native executors discovered".to_string());
        }
    }
    let cfg = RunConfig {
        program: program.ok_or("run needs --program FILE")?,
        executors,
        placement,
        token: global.token.clone(),
        force,
        timeout_ms,
        compare,
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
