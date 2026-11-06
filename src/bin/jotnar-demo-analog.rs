use std::path::PathBuf;
use std::process::ExitCode;

use jotnar::showcase::analog_demo::{run, AnalogOptions};

fn usage() -> &'static str {
    "jotnar-demo-analog — Jotnar-SIM: DAC -> RC field -> ADC\n\
     \n\
     usage: jotnar-demo-analog [--out DIR] [--quick]\n"
}

fn main() -> ExitCode {
    let mut opts = AnalogOptions::default();
    let mut it = std::env::args().skip(1);
    while let Some(a) = it.next() {
        match a.as_str() {
            "--out" => opts.out = PathBuf::from(it.next().unwrap_or_default()),
            "--quick" => opts.quick = true,
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
