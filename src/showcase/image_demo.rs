//! Visual showcase: one geometric listing, three digital targets.
//!
//! Port of `demo_img.py`: builds a 43-op image pipeline, runs the same
//! Jotnar-LAN text through C / CUDA / non-FPU, compares the outputs, and
//! writes the gallery images.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use ndarray::{ArrayD, IxDyn};

use crate::asm::AsmOptions;
use crate::builder::Prog;
use crate::codegen::{self, CompileOptions, Target};
use crate::error::{Error, Result};
use crate::lattice;
use crate::payload::Feeds;
use crate::value::{array_from_vec, Value};

pub const IMG_OUTS: [&str; 7] = ["gray", "inv", "br", "ct", "blur", "us", "strip"];

pub const CONST_NAMES: [&str; 9] = [
    "cr", "cg", "cb", "boost", "csc", "cmid", "amount", "invone", "ninth",
];

#[derive(Clone, Debug)]
pub struct ImageDemoOptions {
    pub input: PathBuf,
    pub gallery: PathBuf,
    pub workdir: PathBuf,
    pub skip_cuda: bool,
}

impl Default for ImageDemoOptions {
    fn default() -> Self {
        ImageDemoOptions {
            input: PathBuf::from("samples/input_example.png"),
            gallery: PathBuf::from("gallery"),
            workdir: std::env::temp_dir().join("jotnar_img_demo"),
            skip_cuda: false,
        }
    }
}

/// Build the demo listing for an HxW image plane.
pub fn build_img_prog(h: usize, w: usize) -> Result<Prog> {
    let mut p = Prog::new("img-demo");
    p.config("m_acc", "41200").config("m_cov", "41200");
    p.inp(["r", "g", "b"])?;
    p.inp(CONST_NAMES)?;
    p.op_names("MUL", &["r", "cr"], Some("t0"))?;
    p.op_names("MUL", &["g", "cg"], Some("t1"))?;
    p.op_names("ADD", &["t0", "t1"], Some("t2"))?;
    p.op_names("MUL", &["b", "cb"], Some("t3"))?;
    p.op_names("ADD", &["t2", "t3"], Some("gray"))?;
    p.op_names("SUB", &["invone", "gray"], Some("inv"))?;
    p.op_names("ADD", &["gray", "boost"], Some("br"))?;
    p.op_names("SUB", &["gray", "cmid"], Some("t4"))?;
    p.op_names("MUL", &["t4", "csc"], Some("t5"))?;
    p.op_names("ADD", &["t5", "cmid"], Some("ct"))?;

    let rows = [(0usize, h - 2), (1, h - 1), (2, h)];
    let cols = [(0usize, w - 2), (1, w - 1), (2, w)];
    let mut cells = Vec::new();
    for (i, (r0, r1)) in rows.iter().enumerate() {
        for (j, (c0, c1)) in cols.iter().enumerate() {
            let rs = p.op(
                "SLICE",
                &[
                    "gray".into(),
                    (0i64).into(),
                    (*r0 as i64).into(),
                    (*r1 as i64).into(),
                ],
                Some(&format!("r{i}{j}")),
            )?;
            let cell = p.op(
                "SLICE",
                &[
                    rs.clone().into(),
                    (1i64).into(),
                    (*c0 as i64).into(),
                    (*c1 as i64).into(),
                ],
                Some(&format!("b{i}{j}")),
            )?;
            cells.push(cell);
        }
    }
    let mut acc = cells[0].clone();
    for (k, c) in cells.iter().skip(1).enumerate() {
        acc = p.op_names("ADD", &[&acc, c], Some(&format!("s{k}")))?;
    }
    p.op_names("MUL", &[&acc, "ninth"], Some("blur"))?;

    let cr0 = p.op(
        "SLICE",
        &[
            "gray".into(),
            (0i64).into(),
            (1i64).into(),
            ((h - 1) as i64).into(),
        ],
        Some("cr0"),
    )?;
    p.op(
        "SLICE",
        &[
            cr0.into(),
            (1i64).into(),
            (1i64).into(),
            ((w - 1) as i64).into(),
        ],
        Some("crop"),
    )?;
    p.op_names("SUB", &["crop", "blur"], Some("detail"))?;
    p.op_names("MUL", &["amount", "detail"], Some("scaled"))?;
    p.op_names("ADD", &["crop", "scaled"], Some("us"))?;
    p.op(
        "CONCAT",
        &["crop".into(), "blur".into(), (1i64).into()],
        Some("strip"),
    )?;
    Ok(p)
}

pub fn const_vals() -> [(f64, usize); 9] {
    [
        (0.299, 0),
        (0.587, 0),
        (0.114, 0),
        (30.0, 0),
        (1.4, 0),
        (128.0, 0),
        (1.5, 1),
        (255.0, 0),
        (1.0 / 9.0, 1),
    ]
}

fn flat_f64(a: &ArrayD<f64>) -> Vec<f64> {
    a.iter().copied().collect()
}

fn flat_value(v: &Value, shape: &[usize]) -> Value {
    let _ = shape;
    v.clone()
}

fn shape_of(s: &[Option<usize>], what: &str) -> Result<Vec<usize>> {
    s.iter()
        .map(|d| d.ok_or_else(|| Error::runtime(format!("image demo: dynamic shape in {what}"))))
        .collect()
}

fn write_inputs(
    dir: &Path,
    art: &codegen::Artifact,
    payload: &Feeds,
    target: Target,
) -> Result<Vec<PathBuf>> {
    let mut files = Vec::new();
    for n in &art.inputs {
        let info = art
            .streams
            .get(n)
            .ok_or_else(|| Error::runtime(format!("image demo: no stream {n}")))?;
        let v = payload
            .get(n)
            .ok_or_else(|| Error::runtime(format!("image demo: payload missing {n}")))?;
        match info.kind {
            crate::codegen::infer::CKind::T => {
                let t = v.triple(n)?;
                let path_s = dir.join(format!("in_{n}_s.bin"));
                std::fs::write(&path_s, t.s.iter().map(|x| *x as u8).collect::<Vec<_>>())?;
                let path_e = dir.join(format!("in_{n}_e.bin"));
                let mut eb = Vec::with_capacity(t.e.len() * 4);
                for x in t.e.iter() {
                    eb.extend_from_slice(&x.to_le_bytes());
                }
                std::fs::write(&path_e, eb)?;
                let path_z = dir.join(format!("in_{n}_z.bin"));
                std::fs::write(&path_z, t.z.iter().copied().collect::<Vec<_>>())?;
                files.push(path_s);
                files.push(path_e);
                files.push(path_z);
            }
            crate::codegen::infer::CKind::F => {
                let a = v.as_f64(n)?;
                let path = dir.join(format!("in_{n}.bin"));
                if target == Target::Cuda {
                    let mut bytes = Vec::with_capacity(a.len() * 4);
                    for x in a.iter() {
                        bytes.extend_from_slice(&(*x as f32).to_le_bytes());
                    }
                    std::fs::write(&path, bytes)?;
                } else {
                    let mut bytes = Vec::with_capacity(a.len() * 8);
                    for x in a.iter() {
                        bytes.extend_from_slice(&x.to_le_bytes());
                    }
                    std::fs::write(&path, bytes)?;
                }
                files.push(path);
            }
            crate::codegen::infer::CKind::I => {
                let a = v.as_i64(n)?;
                let path = dir.join(format!("in_{n}.bin"));
                let mut bytes = Vec::with_capacity(a.len() * 8);
                for x in a.iter() {
                    bytes.extend_from_slice(&x.to_le_bytes());
                }
                std::fs::write(&path, bytes)?;
                files.push(path);
            }
        }
    }
    Ok(files)
}

fn output_paths(dir: &Path, art: &codegen::Artifact) -> Vec<(String, Vec<PathBuf>)> {
    let mut files = Vec::new();
    for o in &art.outputs {
        let info = art.streams.get(o).unwrap();
        if info.kind == crate::codegen::infer::CKind::T {
            files.push((
                o.clone(),
                ["s", "e", "z"]
                    .iter()
                    .map(|c| dir.join(format!("out_{o}_{c}.bin")))
                    .collect(),
            ));
        } else {
            files.push((o.clone(), vec![dir.join(format!("out_{o}.bin"))]));
        }
    }
    files
}

fn read_outputs(
    dir: &Path,
    art: &codegen::Artifact,
    target: Target,
) -> Result<BTreeMap<String, ArrayD<f64>>> {
    let mut out = BTreeMap::new();
    for o in &art.outputs {
        let info = art.streams.get(o).unwrap();
        let shape = shape_of(&info.shape, o)?;
        match info.kind {
            crate::codegen::infer::CKind::T => {
                let mut planes = Vec::new();
                for comp in ["s", "e", "z"] {
                    let bytes = std::fs::read(dir.join(format!("out_{o}_{comp}.bin")))?;
                    match comp {
                        "s" => planes.push(Value::I64(array_from_vec(
                            &shape,
                            bytes.iter().map(|b| *b as i8 as i64).collect(),
                        )?)),
                        "e" => planes.push(Value::I64(array_from_vec(
                            &shape,
                            bytes
                                .chunks_exact(4)
                                .map(|b| i32::from_le_bytes([b[0], b[1], b[2], b[3]]) as i64)
                                .collect(),
                        )?)),
                        _ => planes.push(Value::I64(array_from_vec(
                            &shape,
                            bytes.iter().map(|b| *b as i64).collect(),
                        )?)),
                    }
                }
                let s = planes[0].as_i64("triple")?;
                let e = planes[1].as_i64("triple")?;
                let z = planes[2].as_i64("triple")?;
                let t = crate::value::Triple::new(
                    s.mapv(|x| x as i8),
                    e.mapv(|x| x as i32),
                    z.mapv(|x| x as u8),
                )?;
                let dec = lattice::decode(&t.s, &t.e);
                let zz = t.z.mapv(|v| 1.0 - v as f64);
                out.insert(o.clone(), dec * zz);
            }
            crate::codegen::infer::CKind::F => {
                let bytes = std::fs::read(dir.join(format!("out_{o}.bin")))?;
                let v = if target == Target::Cuda {
                    bytes
                        .chunks_exact(4)
                        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]) as f64)
                        .collect()
                } else {
                    bytes
                        .chunks_exact(8)
                        .map(|b| {
                            f64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]])
                        })
                        .collect()
                };
                out.insert(o.clone(), array_from_vec(&shape, v)?);
            }
            crate::codegen::infer::CKind::I => {
                let bytes = std::fs::read(dir.join(format!("out_{o}.bin")))?;
                let v: Vec<i64> = bytes
                    .chunks_exact(8)
                    .map(|b| i64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]]))
                    .collect();
                out.insert(o.clone(), array_from_vec(&shape, v)?.mapv(|x| x as f64));
            }
        }
    }
    let _ = dir;
    Ok(out)
}

fn run_target(
    text: &str,
    target: Target,
    payload: &Feeds,
    workdir: &Path,
    name: &str,
) -> Result<BTreeMap<String, ArrayD<f64>>> {
    let opts = CompileOptions {
        target,
        outputs: IMG_OUTS.iter().map(|s| s.to_string()).collect(),
        asm: AsmOptions::new(workdir),
        use_fp16: false,
    };
    let art = codegen::compile_program(text, &opts, payload.clone())?;
    let exe = match target {
        Target::Cuda => codegen::cuda::build_cu(&art.source, workdir, name)?,
        _ => codegen::build(&art.source, workdir, name, target)?,
    };
    let in_files = write_inputs(workdir, &art, payload, target)?;
    let outs = output_paths(workdir, &art);
    let mut cmd = std::process::Command::new(exe);
    cmd.args(&in_files);
    for (_, paths) in &outs {
        cmd.args(paths);
    }
    let out = cmd.output()?;
    if !out.status.success() {
        return Err(Error::runtime(format!(
            "image demo {name} run failed: {}",
            String::from_utf8_lossy(&out.stderr)
        )));
    }
    read_outputs(workdir, &art, target)
}

fn save_gray(path: &Path, arr: &ArrayD<f64>) -> Result<()> {
    let shape = arr.shape();
    if shape.len() != 2 {
        return Err(Error::runtime("image demo: output is not 2D"));
    }
    let (h, w) = (shape[0], shape[1]);
    let mut bytes = Vec::with_capacity(h * w);
    for v in arr.iter() {
        bytes.push(v.round().clamp(0.0, 255.0) as u8);
    }
    image::save_buffer(path, &bytes, w as u32, h as u32, image::ColorType::L8)
        .map_err(|e| Error::runtime(format!("save {}: {e}", path.display())))
}

pub fn run(opts: &ImageDemoOptions) -> Result<()> {
    println!("input: {}", opts.input.display());
    let im = image::open(&opts.input)
        .map_err(|e| Error::runtime(format!("cannot open {}: {e}", opts.input.display())))?
        .to_rgb8();
    let (w, h) = (im.width() as usize, im.height() as usize);
    println!("input: {w}x{h} RGB");

    let prog = build_img_prog(h, w)?;
    let text = prog.text()?;
    std::fs::create_dir_all(&opts.workdir)?;
    std::fs::write(opts.workdir.join("img_demo.asm"), &text)?;
    let nops = text
        .lines()
        .filter(|l| l.contains('=') && !l.trim_start().starts_with("CONFIG"))
        .count();
    println!("listing: {nops} ops");

    let mut fpayload = Feeds::new();
    let mut tpayload = Feeds::new();
    let mut r = vec![0f64; h * w];
    let mut g = vec![0f64; h * w];
    let mut b = vec![0f64; h * w];
    for y in 0..h {
        for x in 0..w {
            let p = im.get_pixel(x as u32, y as u32);
            let i = y * w + x;
            r[i] = p[0] as f64;
            g[i] = p[1] as f64;
            b[i] = p[2] as f64;
        }
    }
    for (i, name) in CONST_NAMES.iter().enumerate() {
        let (v, small) = const_vals()[i];
        let shape = if small == 1 { [h - 2, w - 2] } else { [h, w] };
        let a = ArrayD::from_elem(IxDyn(&shape), v);
        tpayload.insert((*name).to_string(), Value::Triple(lattice::encode(&a)));
    }
    for (name, v) in [("r", &r), ("g", &g), ("b", &b)] {
        let a = array_from_vec(&[h, w], v.clone())?;
        tpayload.insert(name.to_string(), Value::Triple(lattice::encode(&a)));
        fpayload.insert(name.to_string(), Value::F64(a));
    }
    for (i, name) in CONST_NAMES.iter().enumerate() {
        let (v, small) = const_vals()[i];
        let shape = if small == 1 { [h - 2, w - 2] } else { [h, w] };
        fpayload.insert(
            (*name).to_string(),
            Value::F64(ArrayD::from_elem(IxDyn(&shape), v)),
        );
    }

    let mut outs: BTreeMap<&str, BTreeMap<String, ArrayD<f64>>> = BTreeMap::new();
    let c = run_target(&text, Target::C, &fpayload, &opts.workdir, "img_c")?;
    println!("c: ok");
    outs.insert("c", c);

    if !opts.skip_cuda {
        match run_target(&text, Target::Cuda, &fpayload, &opts.workdir, "img_cu") {
            Ok(cu) => {
                println!("cuda: ok");
                outs.insert("cuda", cu);
            }
            Err(e) => println!("cuda: SKIP ({e})"),
        }
    }

    let nf = run_target(&text, Target::NonFpu, &tpayload, &opts.workdir, "img_nf")?;
    // The non-FPU emitter runs the float trap inside compile_program.
    println!("nonfpu: ok (trap-clean)");
    outs.insert("nonfpu", nf);

    let c_ref = outs.get("c").unwrap();
    if let Some(cu) = outs.get("cuda") {
        for o in IMG_OUTS {
            let a = c_ref.get(o).unwrap();
            let b = cu.get(o).unwrap();
            let d = a
                .iter()
                .zip(b.iter())
                .map(|(x, y)| (x - y).abs())
                .fold(0.0f64, f64::max);
            println!("{o:6}: cuda-vs-c={d:.2e}");
        }
    }
    if let Some(nf) = outs.get("nonfpu") {
        for o in IMG_OUTS {
            let a = c_ref.get(o).unwrap();
            let b = nf.get(o).unwrap();
            let mse = a
                .iter()
                .zip(b.iter())
                .map(|(x, y)| (x - y) * (x - y))
                .sum::<f64>()
                / a.len().max(1) as f64;
            let psnr = 10.0 * (255.0f64 * 255.0 / mse.max(1e-12)).log10();
            let d = a
                .iter()
                .zip(b.iter())
                .map(|(x, y)| (x - y).abs())
                .fold(0.0f64, f64::max);
            println!("{o:6}: nonfpu-vs-c={d:.3} PSNR={psnr:.1}dB");
        }
    }

    std::fs::create_dir_all(&opts.gallery)?;
    image::save_buffer(
        opts.gallery.join("input.png"),
        &im.as_raw(),
        w as u32,
        h as u32,
        image::ColorType::Rgb8,
    )
    .map_err(|e| Error::runtime(format!("save input: {e}")))?;
    for o in IMG_OUTS {
        save_gray(
            &opts.gallery.join(format!("c_{o}.png")),
            c_ref.get(o).unwrap(),
        )?;
    }
    if let Some(cu) = outs.get("cuda") {
        save_gray(&opts.gallery.join("cu_us.png"), cu.get("us").unwrap())?;
        save_gray(&opts.gallery.join("cu_strip.png"), cu.get("strip").unwrap())?;
    }
    if let Some(nf) = outs.get("nonfpu") {
        save_gray(&opts.gallery.join("nf_us.png"), nf.get("us").unwrap())?;
    }
    println!("gallery: {}", opts.gallery.display());
    let _ = flat_value;
    let _ = flat_f64;
    Ok(())
}
