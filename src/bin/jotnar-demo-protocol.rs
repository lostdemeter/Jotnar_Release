use std::process::ExitCode;

use jotnar::showcase::protocol_demo::{run, ProtocolOptions};

fn usage() -> &'static str {
    "jotnar-demo-protocol — ordering/determinism over model-blind nodes\n\
     \n\
     usage: jotnar-demo-protocol [--nodes N] [--jobs N] [--permutations N] [--no-latency]\n"
}

fn main() -> ExitCode {
    let mut opts = ProtocolOptions::default();
    let mut it = std::env::args().skip(1);
    while let Some(a) = it.next() {
        match a.as_str() {
            "--nodes" => opts.nodes = it.next().and_then(|x| x.parse().ok()).unwrap_or(opts.nodes),
            "--jobs" => opts.jobs = it.next().and_then(|x| x.parse().ok()).unwrap_or(opts.jobs),
            "--permutations" => {
                opts.permutations = it
                    .next()
                    .and_then(|x| x.parse().ok())
                    .unwrap_or(opts.permutations)
            }
            "--no-latency" => opts.latency = false,
            "-h" | "--help" => {
                println!("{}", usage());
                return ExitCode::SUCCESS;
            }
            other => {
                eprintln!("unknown argument {other}\n\n{}", usage());
                return ExitCode::FAILURE;
            }
        }
    }
    match run(&opts) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("{e}");
            ExitCode::FAILURE
        }
    }
}
