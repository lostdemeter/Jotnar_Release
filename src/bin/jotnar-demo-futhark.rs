use std::path::PathBuf;
use std::process::ExitCode;

use jotnar::asm::AsmOptions;
use jotnar::showcase::futhark::{self, FutharkPayload};

fn usage() -> &'static str {
    "jotnar-demo-futhark — Futhark Lens as one Jotnar inference unit\n\
     \n\
     usage: jotnar-demo-futhark --payload DIR --word WORD\n\
            jotnar-demo-futhark --payload DIR --corpus FILE\n\
            jotnar-demo-futhark --payload DIR --emit FILE\n"
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
    let mut payload = PathBuf::from("showcase/futhark/payload");
    let mut word = None;
    let mut corpus = None;
    let mut emit = None;
    let mut it = std::env::args().skip(1);
    while let Some(a) = it.next() {
        match a.as_str() {
            "--payload" => payload = PathBuf::from(it.next().ok_or("--payload needs a dir")?),
            "--word" => word = Some(it.next().ok_or("--word needs a value")?),
            "--corpus" => corpus = Some(PathBuf::from(it.next().ok_or("--corpus needs a file")?)),
            "--emit" => emit = Some(PathBuf::from(it.next().ok_or("--emit needs a file")?)),
            "-h" | "--help" => {
                println!("{}", usage());
                return Ok(());
            }
            other => return Err(format!("unknown argument {other}\n\n{}", usage())),
        }
    }
    let p = FutharkPayload::load(&payload).map_err(|e| e.to_string())?;
    if let Some(path) = emit {
        let text = jotnar::showcase::futhark::build_program(p.n)
            .and_then(|prog| prog.text())
            .map_err(|e| e.to_string())?;
        std::fs::write(&path, text).map_err(|e| e.to_string())?;
        println!("wrote {}", path.display());
        return Ok(());
    }
    let opts = AsmOptions::new(&payload);
    if let Some(w) = word {
        let out = futhark::run_word(&p, &w, &opts).map_err(|e| e.to_string())?;
        println!("{w}\t{out}");
        return Ok(());
    }
    if let Some(path) = corpus {
        let words = futhark::corpus(&path).map_err(|e| e.to_string())?;
        let outs = futhark::run_words(&p, &words, &opts).map_err(|e| e.to_string())?;
        for (w, out) in words.iter().zip(outs) {
            println!("{w}\t{out}");
        }
        return Ok(());
    }
    Err(usage().to_string())
}
