//! Jotnar-SIM: a small analog boundary-value field.
//!
//! This is the surviving `stereo/analog_field2d.py` experiment, reduced to
//! what belongs in the first tour: a 2D RC field with a DAC on the hole
//! boundary and an ADC at readout. It is a digital twin, not physical
//! analog hardware, and it is deliberately not the same claim as a
//! bit-exact Jotnar-DIG result.
//!
//! The exhibit has four pieces:
//!   * digital Laplace vs. ideal RC;
//!   * DAC / ADC / settling / noise sweeps;
//!   * the same field applied as a lane vector over K patches;
//!   * an RJM-shaped artifact with a tolerance gate and a baked-map
//!     deployment profile (the fast path: no DAC at application time).

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::time::Instant;

use ndarray::{Array3, Array4};
use serde_json::json;

use crate::error::{Error, Result};

#[derive(Clone, Debug)]
pub struct AnalogOptions {
    pub h: usize,
    pub w: usize,
    pub steps: usize,
    pub digital_iters: usize,
    pub vector_steps: usize,
    pub out: PathBuf,
    pub quick: bool,
}

impl Default for AnalogOptions {
    fn default() -> Self {
        AnalogOptions {
            h: 80,
            w: 112,
            steps: 20_000,
            digital_iters: 6_000,
            vector_steps: 12_000,
            out: PathBuf::from("showcase/analog/out"),
            quick: false,
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct Metrics {
    pub hole_l1: f64,
    pub hole_rmse: f64,
    pub hole_max: f64,
    pub all_l1: f64,
}

impl Metrics {
    pub fn json(self) -> serde_json::Value {
        json!({
            "hole_l1": self.hole_l1,
            "hole_rmse": self.hole_rmse,
            "hole_max": self.hole_max,
            "all_l1": self.all_l1,
        })
    }
}

/// Target fields in 0..1. The channel order is [b, g, r] to match the
/// original experiment; the panel reverses it for display.
pub fn make_fields(
    case: &str,
    h: usize,
    w: usize,
    lanes: usize,
    seed: u64,
) -> Result<(Array4<f32>, Array3<bool>)> {
    if case != "linear" && case != "curved" {
        return Err(Error::runtime(format!("analog: unknown case {case}")));
    }
    let mut target = Array4::<f32>::zeros((lanes, h, w, 3));
    let mut mask = Array3::<bool>::from_elem((lanes, h, w), false);
    let mut rng = Lcg::new(seed);
    let offs: Vec<(f32, f32)> = (0..lanes)
        .map(|_| {
            if lanes == 1 {
                (0.0, 0.0)
            } else {
                (
                    rng.uniform(-0.10, 0.10) as f32,
                    rng.uniform(-0.10, 0.10) as f32,
                )
            }
        })
        .collect();
    for k in 0..lanes {
        let (ox, oy) = offs[k];
        for y in 0..h {
            let v = ((y as f32 + 0.5) / h as f32 + oy).clamp(0.0, 1.0);
            for x in 0..w {
                let u = ((x as f32 + 0.5) / w as f32 + ox).clamp(0.0, 1.0);
                let (b, g, r) = if case == "linear" {
                    (
                        0.12 + 0.62 * u + 0.10 * v,
                        0.14 + 0.24 * u + 0.58 * v,
                        0.10 + 0.70 * u + 0.10 * v,
                    )
                } else {
                    let rad = (-(((u - 0.50).powi(2) + (v - 0.45).powi(2)) / 0.070)).exp();
                    (
                        0.10 + 0.45 * u + 0.10 * v + 0.22 * rad,
                        0.12 + 0.20 * u + 0.45 * v + 0.18 * rad,
                        0.08 + 0.52 * u + 0.08 * v + 0.28 * rad,
                    )
                };
                target[[k, y, x, 0]] = b.clamp(0.0, 1.0);
                target[[k, y, x, 1]] = g.clamp(0.0, 1.0);
                target[[k, y, x, 2]] = r.clamp(0.0, 1.0);
            }
        }
        for y in h / 3..2 * h / 3 {
            for x in w / 3..2 * w / 3 {
                mask[[k, y, x]] = true;
            }
        }
    }
    Ok((target, mask))
}

fn quant(v: f32, bits: u32) -> f32 {
    let levels = ((1u32 << bits) - 1) as f32;
    (v.clamp(0.0, 1.0) * levels).round() / levels
}

pub fn boundary_ring(mask: &Array3<bool>, width: usize) -> Array3<bool> {
    let (k, h, w) = mask.dim();
    let mut ring = Array3::<bool>::from_elem((k, h, w), false);
    for lane in 0..k {
        for y in 0..h {
            for x in 0..w {
                if mask[[lane, y, x]] {
                    continue;
                }
                let mut near = false;
                for dy in 0..=(2 * width) {
                    let yy = y as isize + dy as isize - width as isize;
                    if yy < 0 || yy >= h as isize {
                        continue;
                    }
                    for dx in 0..=(2 * width) {
                        let xx = x as isize + dx as isize - width as isize;
                        if xx < 0 || xx >= w as isize {
                            continue;
                        }
                        if mask[[lane, yy as usize, xx as usize]] {
                            near = true;
                            break;
                        }
                    }
                    if near {
                        break;
                    }
                }
                ring[[lane, y, x]] = near;
            }
        }
    }
    ring
}

/// Digital Jacobi/SOR fill. Boundary pixels are fixed; only the hole moves.
pub fn solve_digital(
    target: &Array4<f32>,
    mask: &Array3<bool>,
    iters: usize,
    known: Option<&Array4<f32>>,
    init: f32,
) -> Array4<f32> {
    let (k, h, w, _) = target.dim();
    let mut u = match known {
        Some(v) => v.clone(),
        None => target.clone(),
    };
    for lane in 0..k {
        for y in 0..h {
            for x in 0..w {
                if mask[[lane, y, x]] {
                    for c in 0..3 {
                        u[[lane, y, x, c]] = init;
                    }
                }
            }
        }
    }
    let mut next = u.clone();
    for _ in 0..iters {
        for lane in 0..k {
            for y in 1..h - 1 {
                for x in 1..w - 1 {
                    if !mask[[lane, y, x]] {
                        continue;
                    }
                    for c in 0..3 {
                        let avg = 0.25
                            * (u[[lane, y - 1, x, c]]
                                + u[[lane, y + 1, x, c]]
                                + u[[lane, y, x - 1, c]]
                                + u[[lane, y, x + 1, c]]);
                        next[[lane, y, x, c]] = avg;
                    }
                }
            }
        }
        std::mem::swap(&mut u, &mut next);
    }
    u
}

/// Explicit RC/Jacobi field with optional PCM quantization on the hole ring.
pub fn solve_rc(
    target: &Array4<f32>,
    mask: &Array3<bool>,
    steps: usize,
    dt: f32,
    dac_bits: Option<u32>,
    known: Option<&Array4<f32>>,
    init: f32,
) -> Array4<f32> {
    let (k, h, w, _) = target.dim();
    let mut t = match known {
        Some(v) => v.clone(),
        None => target.clone(),
    };
    if let Some(bits) = dac_bits {
        let ring = boundary_ring(mask, 2);
        for lane in 0..k {
            for y in 0..h {
                for x in 0..w {
                    if ring[[lane, y, x]] {
                        for c in 0..3 {
                            t[[lane, y, x, c]] = quant(t[[lane, y, x, c]], bits);
                        }
                    }
                }
            }
        }
    }
    let mut u = t.clone();
    for lane in 0..k {
        for y in 0..h {
            for x in 0..w {
                if mask[[lane, y, x]] {
                    for c in 0..3 {
                        u[[lane, y, x, c]] = init;
                    }
                }
            }
        }
    }
    let mut next = u.clone();
    for _ in 0..steps {
        for lane in 0..k {
            for y in 1..h - 1 {
                for x in 1..w - 1 {
                    for c in 0..3 {
                        if mask[[lane, y, x]] {
                            let p = u[[lane, y, x, c]];
                            let lap = u[[lane, y - 1, x, c]]
                                + u[[lane, y + 1, x, c]]
                                + u[[lane, y, x - 1, c]]
                                + u[[lane, y, x + 1, c]]
                                - 4.0 * p;
                            next[[lane, y, x, c]] = p + dt * lap;
                        } else {
                            next[[lane, y, x, c]] = t[[lane, y, x, c]];
                        }
                    }
                }
            }
        }
        std::mem::swap(&mut u, &mut next);
    }
    u
}

pub fn readout(u: &Array4<f32>, adc_bits: Option<u32>, noise: f32, seed: u64) -> Array4<f32> {
    let mut out = u.clone();
    if let Some(bits) = adc_bits {
        for v in out.iter_mut() {
            *v = quant(*v, bits);
        }
    }
    if noise > 0.0 {
        let mut rng = Lcg::new(seed);
        for v in out.iter_mut() {
            *v = (*v + rng.normal(0.0, noise as f64) as f32).clamp(0.0, 1.0);
        }
    }
    out
}

pub fn errors(out: &Array4<f32>, target: &Array4<f32>, mask: &Array3<bool>) -> Metrics {
    let (k, h, w, _) = target.dim();
    let mut sum = 0.0;
    let mut sumsq = 0.0;
    let mut max = 0.0f64;
    let mut all_sum = 0.0;
    let mut n = 0usize;
    for lane in 0..k {
        for y in 0..h {
            for x in 0..w {
                let mut e = 0.0f64;
                for c in 0..3 {
                    e += (out[[lane, y, x, c]] - target[[lane, y, x, c]]).abs() as f64;
                }
                e /= 3.0;
                all_sum += e;
                if mask[[lane, y, x]] {
                    sum += e;
                    sumsq += e * e;
                    max = max.max(e);
                    n += 1;
                }
            }
        }
    }
    let n = n.max(1) as f64;
    Metrics {
        hole_l1: sum / n,
        hole_rmse: (sumsq / n).sqrt(),
        hole_max: max,
        all_l1: all_sum / (k * h * w).max(1) as f64,
    }
}

fn glyph(ch: char) -> [u8; 7] {
    match ch.to_ascii_uppercase() {
        'A' => [
            0b01110, 0b10001, 0b10001, 0b11111, 0b10001, 0b10001, 0b10001,
        ],
        'B' => [
            0b11110, 0b10001, 0b10001, 0b11110, 0b10001, 0b10001, 0b11110,
        ],
        'C' => [
            0b01111, 0b10000, 0b10000, 0b10000, 0b10000, 0b10000, 0b01111,
        ],
        'D' => [
            0b11110, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b11110,
        ],
        'E' => [
            0b11111, 0b10000, 0b10000, 0b11110, 0b10000, 0b10000, 0b11111,
        ],
        'F' => [
            0b11111, 0b10000, 0b10000, 0b11110, 0b10000, 0b10000, 0b10000,
        ],
        'G' => [
            0b01111, 0b10000, 0b10000, 0b10111, 0b10001, 0b10001, 0b01111,
        ],
        'H' => [
            0b10001, 0b10001, 0b10001, 0b11111, 0b10001, 0b10001, 0b10001,
        ],
        'I' => [
            0b01110, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b01110,
        ],
        'J' => [
            0b00111, 0b00010, 0b00010, 0b00010, 0b00010, 0b10010, 0b01100,
        ],
        'K' => [
            0b10001, 0b10010, 0b10100, 0b11000, 0b10100, 0b10010, 0b10001,
        ],
        'L' => [
            0b10000, 0b10000, 0b10000, 0b10000, 0b10000, 0b10000, 0b11111,
        ],
        'M' => [
            0b10001, 0b11011, 0b10101, 0b10101, 0b10001, 0b10001, 0b10001,
        ],
        'N' => [
            0b10001, 0b10001, 0b11001, 0b10101, 0b10011, 0b10001, 0b10001,
        ],
        'O' => [
            0b01110, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01110,
        ],
        'P' => [
            0b11110, 0b10001, 0b10001, 0b11110, 0b10000, 0b10000, 0b10000,
        ],
        'Q' => [
            0b01110, 0b10001, 0b10001, 0b10001, 0b10101, 0b10010, 0b01101,
        ],
        'R' => [
            0b11110, 0b10001, 0b10001, 0b11110, 0b10100, 0b10010, 0b10001,
        ],
        'S' => [
            0b01111, 0b10000, 0b10000, 0b01110, 0b00001, 0b00001, 0b11110,
        ],
        'T' => [
            0b11111, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100,
        ],
        'U' => [
            0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01110,
        ],
        'V' => [
            0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01010, 0b00100,
        ],
        'W' => [
            0b10001, 0b10001, 0b10001, 0b10101, 0b10101, 0b10101, 0b01010,
        ],
        'X' => [
            0b10001, 0b10001, 0b01010, 0b00100, 0b01010, 0b10001, 0b10001,
        ],
        'Y' => [
            0b10001, 0b10001, 0b01010, 0b00100, 0b00100, 0b00100, 0b00100,
        ],
        'Z' => [
            0b11111, 0b00001, 0b00010, 0b00100, 0b01000, 0b10000, 0b11111,
        ],
        '0' => [
            0b01110, 0b10001, 0b10011, 0b10101, 0b11001, 0b10001, 0b01110,
        ],
        '1' => [
            0b00100, 0b01100, 0b00100, 0b00100, 0b00100, 0b00100, 0b01110,
        ],
        '2' => [
            0b01110, 0b10001, 0b00001, 0b00010, 0b00100, 0b01000, 0b11111,
        ],
        '3' => [
            0b11110, 0b00001, 0b00001, 0b01110, 0b00001, 0b00001, 0b11110,
        ],
        '4' => [
            0b00010, 0b00110, 0b01010, 0b10010, 0b11111, 0b00010, 0b00010,
        ],
        '5' => [
            0b11111, 0b10000, 0b10000, 0b11110, 0b00001, 0b00001, 0b11110,
        ],
        '6' => [
            0b01110, 0b10000, 0b10000, 0b11110, 0b10001, 0b10001, 0b01110,
        ],
        '7' => [
            0b11111, 0b00001, 0b00010, 0b00100, 0b01000, 0b01000, 0b01000,
        ],
        '8' => [
            0b01110, 0b10001, 0b10001, 0b01110, 0b10001, 0b10001, 0b01110,
        ],
        '9' => [
            0b01110, 0b10001, 0b10001, 0b01111, 0b00001, 0b00001, 0b01110,
        ],
        ' ' => [0; 7],
        '.' => [0, 0, 0, 0, 0, 0b01100, 0b01100],
        '=' => [0, 0, 0, 0b11111, 0, 0b11111, 0],
        '_' => [0, 0, 0, 0, 0, 0, 0b11111],
        '-' => [0, 0, 0, 0b11111, 0, 0, 0],
        '+' => [0, 0b00100, 0b00100, 0b11111, 0b00100, 0b00100, 0],
        '/' => [0, 0b00001, 0b00010, 0b00100, 0b01000, 0b10000, 0],
        _ => [
            0b11111, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b11111,
        ],
    }
}

fn draw_text(buf: &mut [u8], width: usize, x0: usize, y0: usize, text: &str, scale: usize) {
    for (i, ch) in text.chars().enumerate() {
        let g = glyph(ch);
        for row in 0..7 {
            for col in 0..5 {
                if g[row] & (1 << (4 - col)) != 0 {
                    for dy in 0..scale {
                        for dx in 0..scale {
                            let x = x0 + (i * 6 + col) * scale + dx;
                            let y = y0 + row * scale + dy;
                            if x < width {
                                let o = (y * width + x) * 3;
                                if o + 2 < buf.len() {
                                    buf[o] = 245;
                                    buf[o + 1] = 245;
                                    buf[o + 2] = 245;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

pub fn save_panel(
    path: &Path,
    target: &Array4<f32>,
    mask: &Array3<bool>,
    images: &[(&str, &Array4<f32>)],
) -> Result<()> {
    let (_, h, w, _) = target.dim();
    let scale = 2usize;
    let gap = 4usize;
    let label_h = 22usize;
    let out_w = images.len() * w * scale + (images.len() - 1) * gap;
    let out_h = label_h + h * scale;
    let mut buf = vec![0u8; out_w * out_h * 3];
    for (i, (label, im)) in images.iter().enumerate() {
        let x0 = i * (w * scale + gap);
        let label = label.replace('_', " ");
        let tw = label.chars().count() * 6 * scale;
        let tx = x0 + (w * scale).saturating_sub(tw) / 2;
        draw_text(&mut buf, out_w, tx, 4, &label, scale);
        for y in 0..h * scale {
            let sy = y / scale;
            for x in 0..w * scale {
                let sx = x / scale;
                let mut rgb = [
                    (im[[0, sy, sx, 2]] * 255.0).clamp(0.0, 255.0) as u8,
                    (im[[0, sy, sx, 1]] * 255.0).clamp(0.0, 255.0) as u8,
                    (im[[0, sy, sx, 0]] * 255.0).clamp(0.0, 255.0) as u8,
                ];
                if i > 0 && mask[[0, sy, sx]] {
                    let edge = sy == 0
                        || sx == 0
                        || sy + 1 == h
                        || sx + 1 == w
                        || !mask[[0, sy - 1, sx]]
                        || !mask[[0, sy + 1, sx]]
                        || !mask[[0, sy, sx - 1]]
                        || !mask[[0, sy, sx + 1]];
                    if edge {
                        rgb = [220, 30, 30];
                    }
                }
                let o = (((y + label_h) * out_w) + x0 + x) * 3;
                buf[o..o + 3].copy_from_slice(&rgb);
            }
        }
    }
    image::save_buffer(
        path,
        &buf,
        out_w as u32,
        out_h as u32,
        image::ColorType::Rgb8,
    )
    .map_err(|e| Error::runtime(format!("save {}: {e}", path.display())))
}

fn time_it<T>(f: impl FnOnce() -> T) -> (T, f64) {
    let t = Instant::now();
    let v = f();
    (v, t.elapsed().as_secs_f64())
}

pub fn run(opts: &AnalogOptions) -> Result<()> {
    if opts.h < 8 || opts.w < 8 {
        return Err(Error::runtime("analog: grid must be at least 8x8"));
    }
    std::fs::create_dir_all(&opts.out)?;
    let h = opts.h;
    let w = opts.w;
    let steps = if opts.quick {
        opts.steps.min(2_000)
    } else {
        opts.steps
    };
    let digital_iters = if opts.quick {
        opts.digital_iters.min(1_000)
    } else {
        opts.digital_iters
    };
    let vector_steps = if opts.quick {
        opts.vector_steps.min(2_000)
    } else {
        opts.vector_steps
    };

    let (target, mask) = make_fields("linear", h, w, 1, 0)?;
    let (digital, t_dig) = time_it(|| solve_digital(&target, &mask, digital_iters, None, 0.5));
    let (analog, t_rc) = time_it(|| solve_rc(&target, &mask, steps, 0.2, None, None, 0.5));
    let (dac4, t_dac4) = time_it(|| {
        readout(
            &solve_rc(&target, &mask, steps, 0.2, Some(4), None, 0.5),
            Some(12),
            0.0,
            0,
        )
    });
    let (dac8, t_dac8) = time_it(|| {
        readout(
            &solve_rc(&target, &mask, steps, 0.2, Some(8), None, 0.5),
            Some(12),
            0.0,
            0,
        )
    });

    let mut stats = serde_json::Map::new();
    stats.insert("grid".into(), json!([h, w]));
    stats.insert("holes".into(), json!(mask.iter().filter(|x| **x).count()));
    let mut linear = serde_json::Map::new();
    linear.insert("digital".into(), {
        let mut j = digital_m(json!({
            "hole_l1": errors(&digital, &target, &mask).hole_l1,
            "hole_rmse": errors(&digital, &target, &mask).hole_rmse,
            "hole_max": errors(&digital, &target, &mask).hole_max,
            "all_l1": errors(&digital, &target, &mask).all_l1,
        }));
        j["seconds"] = json!(t_dig);
        j
    });
    linear.insert("analog_ideal".into(), {
        let mut j = digital_m(errors(&analog, &target, &mask).json());
        j["seconds"] = json!(t_rc);
        j
    });

    let mut dac_rows = serde_json::Map::new();
    for bits in [2u32, 4, 6, 8] {
        let (u, secs) = if bits == 4 {
            (dac4.clone(), t_dac4)
        } else if bits == 8 {
            (dac8.clone(), t_dac8)
        } else {
            time_it(|| {
                readout(
                    &solve_rc(&target, &mask, steps, 0.2, Some(bits), None, 0.5),
                    Some(12),
                    0.0,
                    0,
                )
            })
        };
        let mut j = digital_m(errors(&u, &target, &mask).json());
        j["seconds"] = json!(secs);
        dac_rows.insert(bits.to_string(), j);
    }
    linear.insert("dac_pcm".into(), serde_json::Value::Object(dac_rows));

    let mut adc_rows = serde_json::Map::new();
    let u8 = solve_rc(&target, &mask, steps, 0.2, Some(8), None, 0.5);
    for bits in [4u32, 6, 8, 10, 12] {
        let uu = readout(&u8, Some(bits), 0.0, 0);
        adc_rows.insert(
            bits.to_string(),
            digital_m(errors(&uu, &target, &mask).json()),
        );
    }
    linear.insert("adc".into(), serde_json::Value::Object(adc_rows));

    let mut noise_rows = serde_json::Map::new();
    for noise in [0.0f32, 0.002, 0.01, 0.03] {
        let uu = readout(&u8, Some(10), noise, 3);
        noise_rows.insert(
            noise.to_string(),
            digital_m(errors(&uu, &target, &mask).json()),
        );
    }
    linear.insert("noise".into(), serde_json::Value::Object(noise_rows));

    let mut settle_rows = serde_json::Map::new();
    for s in [1_000usize, 5_000, 20_000, 50_000] {
        let s = if opts.quick { s.min(steps) } else { s };
        let uu = solve_rc(&target, &mask, s, 0.2, None, None, 0.5);
        settle_rows.insert(s.to_string(), digital_m(errors(&uu, &target, &mask).json()));
    }
    linear.insert(
        "settle_steps".into(),
        serde_json::Value::Object(settle_rows),
    );
    stats.insert("linear".into(), serde_json::Value::Object(linear));

    // Curved field: same solve, but the target is not harmonic.
    let (ctarget, cmask) = make_fields("curved", h, w, 1, 0)?;
    let cdig = solve_digital(&ctarget, &cmask, digital_iters, None, 0.5);
    let cana = solve_rc(&ctarget, &cmask, steps, 0.2, None, None, 0.5);
    stats.insert(
        "curved".into(),
        json!({
            "digital": digital_m(errors(&cdig, &ctarget, &cmask).json()),
            "analog_ideal": digital_m(errors(&cana, &ctarget, &cmask).json()),
        }),
    );

    // Lane vector: the same map over K independent patches.
    let mut vectors = serde_json::Map::new();
    for lanes in [1usize, 2, 4, 8] {
        let (vt, vm) = make_fields("linear", h, w, lanes, 0)?;
        let (vu, secs) = time_it(|| {
            readout(
                &solve_rc(&vt, &vm, vector_steps, 0.2, Some(8), None, 0.5),
                Some(10),
                0.0,
                0,
            )
        });
        let m = errors(&vu, &vt, &vm);
        vectors.insert(
            lanes.to_string(),
            json!({
                "seconds": secs,
                "hole_l1": m.hole_l1,
                "cells": lanes * h * w * 3,
            }),
        );
    }
    stats.insert("vector".into(), serde_json::Value::Object(vectors));

    // Tolerance gate and the two deployment profiles.
    let observed = errors(&readout(&u8, Some(10), 0.0, 0), &target, &mask).hole_l1;
    let budget = 0.001;
    let artifact = json!({
        "kind": "research.rjm.field2d",
        "version": 1,
        "target": "simulated_analog_field2d",
        "language_op": "dirichlet_laplace_fill",
        "capabilities_required": ["dac.pcm", "adc.pcm", "rc_field2d"],
        "capabilities_available": ["dac.pcm", "adc.pcm", "rc_field2d"],
        "capability_profiles": {
            "live": {"accepted": true, "missing": [], "mode": "dac_rc_adc"},
            "live_no_dac": {
                "accepted": false,
                "missing": ["dac.pcm"],
                "reason": "a live field solve still needs boundary programming"
            },
            "baked_map": {
                "accepted": true,
                "missing": [],
                "mode": "apply_refined_map",
                "note": "the DAC/RC solve is offline; the application path is a Jotnar-VEC over a stored map"
            }
        },
        "placement": {
            "grid": [h, w, 3],
            "boundary": "hole_ring",
            "dac": {"kind": "pcm", "bits": 8},
            "adc": {"kind": "pcm", "bits": 10},
            "schedule": {"kind": "rc_jacobi", "dt": 0.2, "steps": steps}
        },
        "gate": {
            "class": "tolerance",
            "metric": "hole_l1",
            "budget": budget,
            "bit_exact": false,
            "provenance": "simulated",
            "observed": observed,
            "accepted": observed <= budget
        },
        "vector": {
            "base": "field2d.digital_laplace",
            "direction": "compose[field2d_rc]",
            "magnitude": {"lanes": 8, "cells": 8 * h * w * 3},
            "gate": "tolerance",
            "endpoint": "field2d.simulated_rc",
            "level": "shape",
            "note": "research vector; not accepted by the bit-exact RJM rule"
        }
    });
    std::fs::write(
        opts.out.join("rjm_field2d.research.json"),
        serde_json::to_string_pretty(&artifact).unwrap() + "\n",
    )?;

    save_panel(
        &opts.out.join("panel.png"),
        &target,
        &mask,
        &[
            ("target", &target),
            ("digital", &digital),
            ("analog", &analog),
            ("dac4", &dac4),
            ("dac8", &dac8),
        ],
    )?;

    stats.insert("gate".into(), artifact["gate"].clone());
    std::fs::write(
        opts.out.join("metrics.json"),
        serde_json::to_string_pretty(&stats).unwrap() + "\n",
    )?;

    let digital_m = errors(&digital, &target, &mask);
    let analog_m = errors(&analog, &target, &mask);
    let d4 = errors(&dac4, &target, &mask);
    let d8 = errors(&dac8, &target, &mask);
    println!(
        "Jotnar-SIM: 2D RC field, {}x{}x3, {} hole pixels",
        h,
        w,
        mask.iter().filter(|x| **x).count()
    );
    println!("  digital Laplace : {:.3e}", digital_m.hole_l1);
    println!(
        "  ideal RC        : {:.3e}   (same steady state)",
        analog_m.hole_l1
    );
    println!("  RC + 4-bit DAC  : {:.3e}", d4.hole_l1);
    println!("  RC + 8-bit DAC  : {:.3e}", d8.hole_l1);
    println!(
        "  tolerance gate  : observed {:.3e} <= {:.3e} -> {}",
        observed,
        budget,
        observed <= budget
    );
    println!("  vector cells    : {}", 8 * h * w * 3);
    println!("  wrote           : {}", opts.out.display());
    Ok(())
}

fn digital_m(v: serde_json::Value) -> serde_json::Value {
    v
}

// Small deterministic RNG so the analog demo has no extra dependency.
struct Lcg(u64);

impl Lcg {
    fn new(seed: u64) -> Self {
        Lcg(seed.wrapping_mul(6364136223846793005).wrapping_add(1))
    }
    fn next_u64(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0
    }
    fn uniform(&mut self, lo: f64, hi: f64) -> f64 {
        let u = (self.next_u64() >> 11) as f64 / ((1u64 << 53) as f64);
        lo + (hi - lo) * u
    }
    fn normal(&mut self, mu: f64, sigma: f64) -> f64 {
        let u1 = self.uniform(f64::EPSILON, 1.0);
        let u2 = self.uniform(0.0, 1.0);
        mu + sigma * (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
}
