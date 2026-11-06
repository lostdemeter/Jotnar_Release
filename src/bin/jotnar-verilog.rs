use std::collections::HashMap;
use std::path::PathBuf;
use std::process::ExitCode;

use serde_json::Value as Json;

use jotnar::verilog::{emit, Memory, Sim};

fn usage() -> &'static str {
    "jotnar-verilog — emit/simulate the structural Jotnar-MAC target\n\
     \n\
     usage: jotnar-verilog <file.asm> --config CONFIG.json [--module NAME]\n\
     \n\
     CONFIG fields (all optional except widths/state/mem as needed):\n\
       widths: {name: [lanes, lane_width]}\n\
       state:  {name: [width, init, next_wire]}\n\
       mem:    [{name, depth, width, raddr, rout, we, waddr, wdata, init, raddr2, rout2}]\n\
       cycles/reset/inputs/trace: simulate the plan after emitting\n"
}

fn parse_widths(v: &Json) -> HashMap<String, (usize, usize)> {
    let mut out = HashMap::new();
    if let Some(o) = v.get("widths").and_then(|x| x.as_object()) {
        for (k, val) in o {
            let a = val.as_array().unwrap();
            out.insert(
                k.clone(),
                (
                    a[0].as_u64().unwrap() as usize,
                    a[1].as_u64().unwrap() as usize,
                ),
            );
        }
    }
    out
}

fn parse_state(v: &Json) -> HashMap<String, (usize, u64, String)> {
    let mut out = HashMap::new();
    if let Some(o) = v.get("state").and_then(|x| x.as_object()) {
        for (k, val) in o {
            let a = val.as_array().unwrap();
            out.insert(
                k.clone(),
                (
                    a[0].as_u64().unwrap() as usize,
                    a[1].as_u64().unwrap(),
                    a[2].as_str().unwrap().to_string(),
                ),
            );
        }
    }
    out
}

fn parse_mem(v: &Json) -> Vec<Memory> {
    let mut out = Vec::new();
    if let Some(a) = v.get("mem").and_then(|x| x.as_array()) {
        for m in a {
            let get = |k: &str| m.get(k).and_then(|x| x.as_str()).map(|s| s.to_string());
            out.push(Memory {
                name: get("name").unwrap(),
                depth: m.get("depth").and_then(|x| x.as_u64()).unwrap() as usize,
                width: m.get("width").and_then(|x| x.as_u64()).unwrap() as usize,
                raddr: get("raddr").unwrap(),
                rout: get("rout").unwrap(),
                we: get("we").unwrap(),
                waddr: get("waddr").unwrap(),
                wdata: get("wdata").unwrap(),
                init: m
                    .get("init")
                    .and_then(|x| x.as_array())
                    .map(|a| a.iter().map(|x| x.as_u64().unwrap_or(0)).collect())
                    .unwrap_or_default(),
                raddr2: get("raddr2"),
                rout2: get("rout2"),
            });
        }
    }
    out
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("{e}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<(), String> {
    let mut args = std::env::args().skip(1);
    let file = args.next().ok_or_else(|| usage().to_string())?;
    if file == "-h" || file == "--help" {
        println!("{}", usage());
        return Ok(());
    }
    let mut config = None;
    let mut module = "top".to_string();
    while let Some(a) = args.next() {
        match a.as_str() {
            "--config" => config = Some(PathBuf::from(args.next().ok_or("--config needs a file")?)),
            "--module" => module = args.next().ok_or("--module needs a name")?,
            other => return Err(format!("unknown argument {other}\n\n{}", usage())),
        }
    }
    let text = std::fs::read_to_string(&file).map_err(|e| format!("read {file}: {e}"))?;
    let cfg: Json = match config {
        Some(p) => serde_json::from_str(&std::fs::read_to_string(&p).map_err(|e| e.to_string())?)
            .map_err(|e| e.to_string())?,
        None => serde_json::json!({}),
    };
    let widths = parse_widths(&cfg);
    let state = parse_state(&cfg);
    let mem = parse_mem(&cfg);
    let verilog = emit(&text, &widths, &module, &state, &mem).map_err(|e| e.to_string())?;
    if cfg.get("cycles").is_none() {
        print!("{verilog}");
        return Ok(());
    }
    let cycles = cfg["cycles"].as_u64().unwrap() as usize;
    let reset = cfg.get("reset").and_then(|x| x.as_u64()).unwrap_or(1) as usize;
    let trace = cfg.get("trace").and_then(|x| x.as_bool()).unwrap_or(false);
    let mut inputs = HashMap::new();
    if let Some(o) = cfg.get("inputs").and_then(|x| x.as_object()) {
        for (k, v) in o {
            inputs.insert(k.clone(), v.as_u64().unwrap_or(0) as u128);
        }
    }
    let sim = Sim::compile(&text, &widths, &state, &mem).map_err(|e| e.to_string())?;
    let out = sim
        .simulate_cycles(&inputs, cycles, reset, trace)
        .map_err(|e| e.to_string())?;
    let json: Vec<Json> = out
        .iter()
        .map(|m| {
            let mut o = serde_json::Map::new();
            for (k, v) in m {
                o.insert(k.clone(), serde_json::json!(v));
            }
            Json::Object(o)
        })
        .collect();
    println!("{}", serde_json::to_string_pretty(&json).unwrap());
    Ok(())
}
