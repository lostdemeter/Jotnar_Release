use std::path::{Path, PathBuf};
use std::process::ExitCode;

use jotnar::asm::{self, AsmOptions};
use jotnar::codegen::{self, CompileOptions, Target};
use jotnar::payload;

fn usage() -> &'static str {
    "jotnar — geometric assembly (Rust reference)\n\
     \n\
     usage:\n\
       jotnar check  <file.asm> [--stdlib DIR]\n\
       jotnar verify <file.asm> [--stdlib DIR] [--json]\n\
       jotnar run    <file.asm> --payload FILE [--out FILE] [--outputs a,b] [--stdlib DIR]\n\
       jotnar emit-c <file.asm> --payload FILE --outputs a,b [--out FILE.c] [--stdlib DIR]\n\
       jotnar emit-rust <file.asm> --payload FILE --outputs a,b [--out FILE.rs] [--stdlib DIR] (needs rustc)\n\
       jotnar build  <file.asm> --payload FILE --outputs a,b [--workdir DIR] [--name NAME] [--stdlib DIR]\n\
       jotnar repeat <file.asm> --payload FILE [--grow a,b] [--out FILE] [--stdlib DIR]\n\
       jotnar ops\n"
}

struct Args {
    cmd: String,
    file: PathBuf,
    stdlib: Option<PathBuf>,
    payload: Option<PathBuf>,
    out: Option<PathBuf>,
    outputs: Option<Vec<String>>,
    json: bool,
    workdir: Option<PathBuf>,
    name: Option<String>,
    target: String,
    fp16: bool,
    grow: Option<Vec<String>>,
}

fn parse_args() -> Result<Args, String> {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() {
        return Err(usage().to_string());
    }
    let cmd = args.remove(0);
    if cmd == "-h" || cmd == "--help" || cmd == "help" {
        return Err(usage().to_string());
    }
    if cmd == "ops" {
        return Ok(Args {
            cmd,
            file: PathBuf::new(),
            stdlib: None,
            payload: None,
            out: None,
            outputs: None,
            json: false,
            workdir: None,
            name: None,
            target: "c".to_string(),
            fp16: false,
            grow: None,
        });
    }
    let file = if args.is_empty() {
        return Err(format!("{cmd}: missing listing path\n\n{}", usage()));
    } else {
        PathBuf::from(args.remove(0))
    };
    let mut a = Args {
        cmd,
        file,
        stdlib: None,
        payload: None,
        out: None,
        outputs: None,
        json: false,
        workdir: None,
        name: None,
        target: "c".to_string(),
        fp16: false,
        grow: None,
    };
    let mut it = args.into_iter();
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--stdlib" => {
                a.stdlib = Some(PathBuf::from(
                    it.next().ok_or("--stdlib needs a directory")?,
                ))
            }
            "--payload" | "--in" => {
                a.payload = Some(PathBuf::from(it.next().ok_or("--payload needs a file")?))
            }
            "--out" => a.out = Some(PathBuf::from(it.next().ok_or("--out needs a file")?)),
            "--outputs" => {
                let s = it.next().ok_or("--outputs needs a comma list")?;
                a.outputs = Some(s.split(',').map(|s| s.trim().to_string()).collect());
            }
            "--json" => a.json = true,
            "--workdir" => {
                a.workdir = Some(PathBuf::from(it.next().ok_or("--workdir needs a dir")?))
            }
            "--name" => a.name = Some(it.next().ok_or("--name needs a value")?),
            "--target" => a.target = it.next().ok_or("--target needs a value")?,
            "--fp16" => a.fp16 = true,
            "--grow" => {
                let s = it.next().ok_or("--grow needs a comma list")?;
                a.grow = Some(s.split(',').map(|x| x.trim().to_string()).collect());
            }
            "-h" | "--help" => return Err(usage().to_string()),
            other => return Err(format!("unknown argument '{other}'\n\n{}", usage())),
        }
    }
    Ok(a)
}

fn options_for(path: &Path, stdlib: Option<PathBuf>) -> AsmOptions {
    let basedir = path.parent().unwrap_or(Path::new(".")).to_path_buf();
    let mut opts = AsmOptions::new(basedir);
    opts.stdlib = stdlib;
    opts.origin = Some(path.display().to_string());
    opts
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
    if args.cmd == "ops" {
        let reg = jotnar::ops::registry();
        for name in reg.names() {
            let def = reg.get(name).unwrap();
            println!(
                "{:<14} {:>2} -> {:<2}  {}",
                name,
                def.arity_in,
                def.arity_out,
                if def.run.is_some() {
                    ""
                } else {
                    "(no pattern yet)"
                }
            );
        }
        println!("\n{} mnemonics", reg.len());
        return Ok(());
    }

    let text = std::fs::read_to_string(&args.file)
        .map_err(|e| format!("cannot read {}: {e}", args.file.display()))?;
    let opts = options_for(&args.file, args.stdlib);

    match args.cmd.as_str() {
        "check" => {
            let bound = asm::assemble(&text, &opts).map_err(|e| e.to_string())?;
            println!(
                "ok: {} instructions, {} inputs, {} state stream(s)",
                bound.instrs.len(),
                bound.inputs.len(),
                bound.state.len()
            );
            Ok(())
        }
        "verify" => {
            let report = asm::verify(&text, &opts).map_err(|e| e.to_string())?;
            if args.json {
                let mut defs = serde_json::Map::new();
                for (name, iface) in &report.defs {
                    defs.insert(
                        name.clone(),
                        serde_json::json!({
                            "formals": iface.formals,
                            "returns": iface.returns,
                            "origin": iface.origin,
                        }),
                    );
                }
                println!(
                    "{}",
                    serde_json::json!({
                        "errors": report.errors,
                        "resolved": report.resolved,
                        "total": report.total,
                        "layouts": report.layouts,
                        "defs": defs,
                    })
                );
            } else {
                for e in &report.errors {
                    eprintln!("error: {e}");
                }
                println!(
                    "layouts resolved: {}/{} ({} error(s))",
                    report.resolved,
                    report.total,
                    report.errors.len()
                );
                for (name, iface) in &report.defs {
                    let f: Vec<String> = iface
                        .formals
                        .iter()
                        .map(|(n, l)| match l {
                            Some(l) => format!("{n}:{l}"),
                            None => n.clone(),
                        })
                        .collect();
                    let r: Vec<String> = iface
                        .returns
                        .iter()
                        .map(|(n, l)| match l {
                            Some(l) => format!("{n}:{l}"),
                            None => n.clone(),
                        })
                        .collect();
                    println!(
                        "DEF {name}({}) -> ({})  [{}]",
                        f.join(", "),
                        r.join(", "),
                        iface.origin
                    );
                }
            }
            if report.errors.is_empty() {
                Ok(())
            } else {
                Err(format!("{} layout error(s)", report.errors.len()))
            }
        }
        "emit-c" | "emit-rust" => {
            let cmd = args.cmd.clone();
            let payload_path = args
                .payload
                .ok_or_else(|| format!("{cmd}: --payload is required\n\n{}", usage()))?;
            let outputs = args
                .outputs
                .ok_or_else(|| format!("{cmd}: --outputs is required\n\n{}", usage()))?;
            let payload = payload::read_payload(&payload_path).map_err(|e| e.to_string())?;
            let target = if args.cmd == "emit-rust" {
                Target::Rust
            } else {
                Target::parse(&args.target).map_err(|e| e.to_string())?
            };
            let opts = CompileOptions {
                target,
                outputs,
                asm: opts,
                use_fp16: args.fp16,
            };
            let art = codegen::compile_program(&text, &opts, payload).map_err(|e| e.to_string())?;
            if let Some(out) = args.out {
                std::fs::write(&out, &art.source)
                    .map_err(|e| format!("cannot write {}: {e}", out.display()))?;
            } else {
                print!("{}", art.source);
            }
            Ok(())
        }
        "build" => {
            let payload_path = args
                .payload
                .ok_or_else(|| format!("build: --payload is required\n\n{}", usage()))?;
            let outputs = args
                .outputs
                .ok_or_else(|| format!("build: --outputs is required\n\n{}", usage()))?;
            let payload = payload::read_payload(&payload_path).map_err(|e| e.to_string())?;
            let target = Target::parse(&args.target).map_err(|e| e.to_string())?;
            let opts = CompileOptions {
                target,
                outputs,
                asm: opts,
                use_fp16: args.fp16,
            };
            let art = codegen::compile_program(&text, &opts, payload).map_err(|e| e.to_string())?;
            let workdir = args.workdir.unwrap_or_else(|| PathBuf::from("."));
            let name = args.name.unwrap_or_else(|| "prog".to_string());
            let exe = codegen::build(&art.source, &workdir, &name, opts.target)
                .map_err(|e| e.to_string())?;
            println!("{}", exe.display());
            Ok(())
        }
        "repeat" => {
            let payload_path = args
                .payload
                .ok_or_else(|| format!("repeat: --payload is required\n\n{}", usage()))?;
            let seq = payload::read_sequence(&payload_path).map_err(|e| e.to_string())?;
            let grow = args.grow.unwrap_or_default();
            let (hist, last) =
                jotnar::runtime::repeat(&text, &opts, &seq, &grow).map_err(|e| e.to_string())?;
            let json = payload::feeds_to_json(&last, None).map_err(|e| e.to_string())?;
            let rendered = serde_json::to_string_pretty(&json).unwrap();
            if let Some(out) = args.out {
                std::fs::write(&out, rendered)
                    .map_err(|e| format!("cannot write {}: {e}", out.display()))?;
            } else {
                println!("{rendered}");
            }
            eprintln!("{} iterations", hist.len());
            Ok(())
        }
        "run" => {
            let payload_path = args
                .payload
                .ok_or_else(|| format!("run: --payload is required\n\n{}", usage()))?;
            let payload = payload::read_payload(&payload_path).map_err(|e| e.to_string())?;
            let feeds = jotnar::run_text(&text, &opts, &payload).map_err(|e| e.to_string())?;
            let json = payload::feeds_to_json(&feeds, args.outputs.as_deref())
                .map_err(|e| e.to_string())?;
            let rendered = serde_json::to_string_pretty(&json).unwrap();
            if let Some(out) = args.out {
                std::fs::write(&out, rendered)
                    .map_err(|e| format!("cannot write {}: {e}", out.display()))?;
            } else {
                println!("{rendered}");
            }
            Ok(())
        }
        other => Err(format!("unknown command '{other}'\n\n{}", usage())),
    }
}
