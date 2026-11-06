use std::path::PathBuf;
use std::process::ExitCode;

use jotnar::showcase::analog_filter::{run, FilterOptions};

fn usage() -> &'static str {
    "jotnar-demo-analog-filter — Laplace/RC vs Perona-Malik bake-off\n\
     \n\
     usage: jotnar-demo-analog-filter [--out DIR] [--quick] [--steps N]\n"
}

fn main() -> ExitCode {
    let mut opts = FilterOptions::default();
    let mut it = std::env::args().skip(1);
    while let Some(a) = it.next() {
        match a.as_str() {
            "--out" => opts.out = PathBuf::from(it.next().unwrap_or_default()),
            "--quick" => opts.quick = true,
            "--steps" => opts.steps = it.next().and_then(|x| x.parse().ok()).unwrap_or(opts.steps),
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
