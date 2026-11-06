use std::path::PathBuf;
use std::process::ExitCode;

use jotnar_lab::{doom, model::Certificate};

fn usage() -> &'static str {
    "jotnar-lab — trust spine: certificates, gates, falsifiers, nulls, benches\n\
     \n\
     usage:\n\
       jotnar-lab certify doom --bundle MANIFEST --out CERT.json\n\
       jotnar-lab falsify doom --bundle MANIFEST\n\
       jotnar-lab null doom --bundle MANIFEST\n\
       jotnar-lab bench doom --bundle MANIFEST --runner PATH [--runs N] [--certificate CERT.json]\n\
       jotnar-lab seal CERT.json\n       jotnar-lab verify CERT.json [--hashes|--no-hashes]\n"
}

fn main() -> ExitCode {
    match run() {
        Ok(code) => code,
        Err(e) => {
            eprintln!("{e}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<ExitCode, String> {
    let mut args = std::env::args().skip(1);
    let cmd = args.next().ok_or_else(|| usage().to_string())?;
    match cmd.as_str() {
        "-h" | "--help" => {
            println!("{}", usage());
            Ok(ExitCode::SUCCESS)
        }
        "certify" => {
            let what = args.next().ok_or("certify needs an exhibit (doom)")?;
            if what != "doom" {
                return Err(format!("unknown exhibit {what:?}"));
            }
            let (bundle, out) = parse_bundle_out(args)?;
            let command = format!(
                "jotnar-lab certify doom --bundle {} --out {}",
                bundle.display(),
                out.display()
            );
            let cert = doom::build_certificate(&bundle, &command)?;
            cert.write(&out).map_err(|e| e.to_string())?;
            println!("certificate: {}", out.display());
            println!(
                "outcome={:?} verdict={:?} status={:?}",
                cert.outcome, cert.verdict, cert.status
            );
            println!(
                "sha256={}",
                cert.certificate_sha256.clone().unwrap_or_default()
            );
            Ok(ExitCode::SUCCESS)
        }
        "falsify" => {
            let what = args.next().ok_or("falsify needs an exhibit (doom)")?;
            if what != "doom" {
                return Err(format!("unknown exhibit {what:?}"));
            }
            let bundle = parse_required(args, "--bundle")?;
            let cert = doom::build_certificate(&bundle, "jotnar-lab falsify doom")?;
            doom::print_falsifiers(&cert);
            Ok(ExitCode::SUCCESS)
        }
        "null" => {
            let what = args.next().ok_or("null needs an exhibit (doom)")?;
            if what != "doom" {
                return Err(format!("unknown exhibit {what:?}"));
            }
            let bundle = parse_required(args, "--bundle")?;
            let cert = doom::build_certificate(&bundle, "jotnar-lab null doom")?;
            doom::print_nulls(&cert);
            Ok(ExitCode::SUCCESS)
        }
        "bench" => {
            let what = args.next().ok_or("bench needs an exhibit (doom)")?;
            if what != "doom" {
                return Err(format!("unknown exhibit {what:?}"));
            }
            let mut bundle = None;
            let mut runner = PathBuf::from("target/release/jotnar-demo-doom");
            let mut runs = 5u32;
            let mut cert_path = None;
            while let Some(a) = args.next() {
                match a.as_str() {
                    "--bundle" => {
                        bundle = Some(PathBuf::from(args.next().ok_or("--bundle needs a file")?))
                    }
                    "--runner" => {
                        runner = PathBuf::from(args.next().ok_or("--runner needs a path")?)
                    }
                    "--runs" => {
                        runs = args
                            .next()
                            .and_then(|x| x.parse().ok())
                            .ok_or("--runs needs a number")?
                    }
                    "--certificate" => {
                        cert_path = Some(PathBuf::from(
                            args.next().ok_or("--certificate needs a file")?,
                        ))
                    }
                    other => return Err(format!("unknown option {other}")),
                }
            }
            let bundle = bundle.ok_or("bench needs --bundle")?;
            let bench = doom::bench_native(&bundle, &runner, runs)?;
            println!("{}", serde_json::to_string_pretty(&bench).unwrap());
            if let Some(p) = cert_path {
                let c = doom::update_bench(&p, bench)?;
                println!("updated {}", p.display());
                println!("sha256={}", c.certificate_sha256.unwrap_or_default());
            }
            Ok(ExitCode::SUCCESS)
        }
        "seal" => {
            let path = PathBuf::from(args.next().ok_or("seal needs a certificate path")?);
            let mut c = Certificate::read(&path)?;
            c.certificate_sha256 = None;
            c.seal();
            c.write(&path).map_err(|e| e.to_string())?;
            println!(
                "sealed {} sha256={}",
                path.display(),
                c.certificate_sha256.unwrap_or_default()
            );
            Ok(ExitCode::SUCCESS)
        }
        "verify" => {
            let path = args.next().ok_or("verify needs a certificate path")?;
            let mut hashes = true;
            for a in args {
                if a == "--hashes" {
                    hashes = true;
                } else if a == "--no-hashes" {
                    hashes = false;
                } else {
                    return Err(format!("unknown option {a}"));
                }
            }
            let r = doom::verify(&PathBuf::from(path), hashes)?;
            println!("{r}");
            if r.starts_with("PASS") || r.starts_with("REFUSED") {
                Ok(ExitCode::SUCCESS)
            } else if r.starts_with("FAIL")
                || r.starts_with("BLOCKED")
                || r.starts_with("NOT TESTED")
                || r.starts_with("STATUS")
                || r.starts_with("STALE")
                || r.starts_with("HISTORICAL")
                || r.starts_with("SUPERSEDED")
                || r.starts_with("BOTTOM")
            {
                Ok(ExitCode::from(2))
            } else {
                Ok(ExitCode::from(1))
            }
        }
        _ => Err(usage().to_string()),
    }
}

fn parse_required(mut args: impl Iterator<Item = String>, flag: &str) -> Result<PathBuf, String> {
    while let Some(a) = args.next() {
        if a == flag {
            return Ok(PathBuf::from(
                args.next().ok_or(format!("{flag} needs a value"))?,
            ));
        }
    }
    Err(format!("missing {flag}"))
}

fn parse_bundle_out(mut args: impl Iterator<Item = String>) -> Result<(PathBuf, PathBuf), String> {
    let mut bundle = None;
    let mut out = None;
    while let Some(a) = args.next() {
        match a.as_str() {
            "--bundle" => bundle = Some(PathBuf::from(args.next().ok_or("--bundle needs a file")?)),
            "--out" => out = Some(PathBuf::from(args.next().ok_or("--out needs a file")?)),
            other => return Err(format!("unknown option {other}")),
        }
    }
    Ok((
        bundle.ok_or("missing --bundle")?,
        out.ok_or("missing --out")?,
    ))
}

#[allow(dead_code)]
fn _keep(_: Certificate) {}
