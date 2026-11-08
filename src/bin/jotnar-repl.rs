//! Warm interpreter for Jotnar listings.
//!
//! Assembles one listing once, then processes newline-delimited JSON
//! payloads on stdin. Each line is `{"streams": {...}}` as accepted by
//! `jotnar run --payload`, and produces one line on stdout holding the
//! resulting `{"streams": {...}}`, or `{"error": "..."}` without exiting.
//! This removes the per-command process spawn and reassembly of the
//! one-shot CLI, which is the "compile every time" cost for hosts that
//! drive Jotnar from code.
//!
//! ```sh
//! cargo run --release --bin jotnar-repl -- showcase/language/saxpy.asm --outputs OUT
//! ```
//!
//! Then pipe payloads on stdin. Empty lines are ignored. The loop ends
//! on a line holding `{"cmd":"exit"}` or on EOF. `--pretty` prints
//! indented JSON for humans; `--stats` prints assemble/run timings to
//! stderr.

use std::io::BufRead;
use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Instant;

use jotnar::asm::{self, AsmOptions};
use jotnar::payload::{self, Feeds};
use serde_json::{json, Value as Json};

fn usage() -> &'static str {
    "jotnar-repl - warm Jotnar interpreter (assemble once, run many)\n\
     \n\
     usage:\n\
       jotnar-repl <file.asm> [--stdlib DIR] [--outputs a,b] [--pretty] [--stats]\n\
     \n\
     Reads NDJSON payloads from stdin, one `{\"streams\": {...}}` per line,\n\
     writes one `{\"streams\": {...}}` (or `{\"error\": ...}`) per line on stdout.\n\
     A line holding `{\"cmd\":\"exit\"}` or EOF terminates the process.\n"
}

struct Repl {
    bound: jotnar::ir::BoundProgram,
    outputs: Option<Vec<String>>,
    pretty: bool,
    stats: bool,
}

#[allow(clippy::type_complexity)]
fn parse_args() -> Result<(PathBuf, Option<PathBuf>, Option<Vec<String>>, bool, bool), String> {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() || args[0] == "-h" || args[0] == "--help" || args[0] == "help" {
        return Err(usage().to_string());
    }
    let file = PathBuf::from(args.remove(0));
    let mut stdlib: Option<PathBuf> = None;
    let mut outputs: Option<Vec<String>> = None;
    let mut pretty = false;
    let mut stats = false;
    let mut it = args.into_iter();
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--stdlib" => {
                stdlib = Some(PathBuf::from(
                    it.next().ok_or("--stdlib needs a directory")?,
                ))
            }
            "--outputs" => {
                let s = it.next().ok_or("--outputs needs a comma list")?;
                outputs = Some(s.split(',').map(|s| s.trim().to_string()).collect());
            }
            "--pretty" => pretty = true,
            "--stats" => stats = true,
            "-h" | "--help" => return Err(usage().to_string()),
            other => return Err(format!("unknown argument '{other}'\n\n{}", usage())),
        }
    }
    Ok((file, stdlib, outputs, pretty, stats))
}

fn options_for(path: &std::path::Path, stdlib: Option<PathBuf>) -> AsmOptions {
    let basedir = path
        .parent()
        .unwrap_or(std::path::Path::new("."))
        .to_path_buf();
    let mut opts = AsmOptions::new(basedir);
    opts.stdlib = stdlib;
    opts.origin = Some(path.display().to_string());
    opts
}

fn render(repl: &Repl, v: &Json) -> String {
    if repl.pretty {
        serde_json::to_string_pretty(v).unwrap()
    } else {
        serde_json::to_string(v).unwrap()
    }
}

fn run_line(repl: &Repl, input: &Json) -> Json {
    let start = Instant::now();
    let feeds: Result<Feeds, _> = payload::feeds_from_json(input);
    let feeds = match feeds {
        Ok(f) => f,
        Err(e) => return json!({"error": e.to_string()}),
    };
    match jotnar::runtime::run(&repl.bound, &feeds) {
        Ok(got) => {
            if repl.stats {
                eprintln!("run {:.3}ms", start.elapsed().as_secs_f64() * 1e3);
            }
            match payload::feeds_to_json(&got, repl.outputs.as_deref()) {
                Ok(j) => j,
                Err(e) => json!({"error": e.to_string()}),
            }
        }
        Err(e) => json!({"error": e.to_string()}),
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

fn real_main() -> Result<(), String> {
    let (file, stdlib, outputs, pretty, stats) = parse_args()?;
    let text = std::fs::read_to_string(&file)
        .map_err(|e| format!("cannot read {}: {e}", file.display()))?;
    let opts = options_for(&file, stdlib);
    let t0 = Instant::now();
    let bound = asm::assemble(&text, &opts).map_err(|e| e.to_string())?;
    if stats {
        eprintln!(
            "assembled {} instructions, {} inputs in {:.3}ms ({}); ready",
            bound.instrs.len(),
            bound.inputs.len(),
            t0.elapsed().as_secs_f64() * 1e3,
            file.display()
        );
    }

    let repl = Repl {
        bound,
        outputs,
        pretty,
        stats,
    };

    let stdin = std::io::stdin();
    let mut handle = stdin.lock();
    // Read whole stdin line by line via a buffer; each line is one request.
    let mut buf = String::new();
    loop {
        buf.clear();
        let n = handle
            .read_line(&mut buf)
            .map_err(|e| format!("cannot read stdin: {e}"))?;
        if n == 0 {
            break;
        }
        let line = buf.trim();
        if line.is_empty() {
            continue;
        }
        let input: Json = match serde_json::from_str(line) {
            Ok(v) => v,
            Err(e) => {
                println!(
                    "{}",
                    render(&repl, &json!({"error": format!("bad json: {e}")}))
                );
                continue;
            }
        };
        if input.get("cmd").and_then(|c| c.as_str()) == Some("exit") {
            break;
        }
        let out = run_line(&repl, &input);
        println!("{}", render(&repl, &out));
    }
    Ok(())
}
