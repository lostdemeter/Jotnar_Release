use std::path::PathBuf;
use std::process::ExitCode;

use jotnar::showcase::image_demo::{run, ImageDemoOptions};

fn usage() -> &'static str {
    "jotnar-demo-img — one geometric image program on C / CUDA / non-FPU\n\
     \n\
     usage: jotnar-demo-img [--input PNG] [--gallery DIR] [--workdir DIR] [--skip-cuda]\n"
}

fn main() -> ExitCode {
    let mut opts = ImageDemoOptions::default();
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--input" => {
                opts.input = PathBuf::from(it.next().unwrap_or_default());
            }
            "--gallery" => {
                opts.gallery = PathBuf::from(it.next().unwrap_or_default());
            }
            "--workdir" => {
                opts.workdir = PathBuf::from(it.next().unwrap_or_default());
            }
            "--skip-cuda" => opts.skip_cuda = true,
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
