use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Instant;

use jotnar::rv32::{read_fb_png, save_fb_png, Rv32, FB_ADDR, FB_BYTES, WAD_ADDR};

fn usage() -> &'static str {
    "jotnar-demo-doom — RV32IM DOOM core (native path)\n\
     \n\
     usage: jotnar-demo-doom --image IMAGE --wad WAD [--frames N] [--out PREFIX]\n\
            jotnar-demo-doom --resume PREFIX --frames N [--out PREFIX2]\n"
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
    let mut image = None;
    let mut wad = None;
    let mut frames = 1u32;
    let mut out = PathBuf::from("/tmp/opencode/doom_rust/frame");
    let mut resume = None;
    let mut compare = None;
    let mut it = std::env::args().skip(1);
    while let Some(a) = it.next() {
        match a.as_str() {
            "--image" => image = Some(PathBuf::from(it.next().ok_or("--image needs a path")?)),
            "--wad" => wad = Some(PathBuf::from(it.next().ok_or("--wad needs a path")?)),
            "--frames" => {
                frames = it
                    .next()
                    .and_then(|x| x.parse().ok())
                    .ok_or("--frames needs a number")?
            }
            "--out" => out = PathBuf::from(it.next().ok_or("--out needs a prefix")?),
            "--resume" => resume = Some(PathBuf::from(it.next().ok_or("--resume needs a prefix")?)),
            "--compare" | "--compare-png" | "--compare-fb" => {
                compare = Some(PathBuf::from(it.next().ok_or("--compare needs a path")?))
            }
            "-h" | "--help" => {
                println!("{}", usage());
                return Ok(());
            }
            other => return Err(format!("unknown argument {other}\n\n{}", usage())),
        }
    }
    let mut cpu = Rv32::new(64 << 20);
    let mut base = 0u64;
    if let Some(prefix) = resume {
        cpu.load_checkpoint(&prefix).map_err(|e| e.to_string())?;
        base = cpu.cycles;
    } else {
        let image = image.ok_or("need --image (or --resume)")?;
        let wad = wad.ok_or("need --wad")?;
        let image = std::fs::read(&image).map_err(|e| e.to_string())?;
        let wad = std::fs::read(&wad).map_err(|e| e.to_string())?;
        cpu.load_doom(&image, &wad).map_err(|e| e.to_string())?;
    }
    let t = Instant::now();
    let n = cpu.run_to_frame(frames, 2_000_000_000);
    let dt = t.elapsed().as_secs_f64();
    if let Some(prefix) = &out.parent() {
        std::fs::create_dir_all(prefix).map_err(|e| e.to_string())?;
    }
    std::fs::write(out.with_extension("fb"), cpu.framebuffer()).map_err(|e| e.to_string())?;
    save_fb_png(cpu.framebuffer(), &out.with_extension("png")).map_err(|e| e.to_string())?;
    cpu.save_checkpoint(&out).map_err(|e| e.to_string())?;
    println!(
        "frame {}: {} instructions (+{}), {:.3}s, {:.1}M instr/s, pc=0x{:08x}",
        cpu.frame(),
        cpu.cycles,
        n,
        dt,
        n as f64 / dt.max(1e-9) / 1e6,
        cpu.pc
    );
    println!("wrote {} {{fb,png,mem,x,pc,cyc,csr}}", out.display());
    let _ = (FB_ADDR, FB_BYTES, WAD_ADDR);
    if let Some(path) = compare {
        let a = if path.extension().and_then(|x| x.to_str()) == Some("fb") {
            std::fs::read(&path).map_err(|e| e.to_string())?
        } else {
            read_fb_png(&path).map_err(|e| e.to_string())?
        };
        let b = cpu.framebuffer();
        if a == b {
            println!("framebuffer: BYTE-IDENTICAL to {}", path.display());
        } else {
            let first = a.iter().zip(b).position(|(x, y)| x != y);
            return Err(format!(
                "framebuffer: MISMATCH at byte {:?} (base {base})",
                first
            ));
        }
    }
    Ok(())
}
