//! Analog-filter bake-off: isotropic RC/Laplace vs Perona-Malik.
//!
//! Port of the synthetic edge case in `stereo/analog_filter2d.py`. The
//! real-hole result is kept in the README as a negative limit: a local PDE
//! cannot invent missing geometry.

use std::path::PathBuf;
use std::time::Instant;

use ndarray::{Array3, Array4};
use serde_json::json;

use crate::error::{Error, Result};
use crate::showcase::analog_demo::{save_panel, solve_digital, solve_rc};

#[derive(Clone, Debug)]
pub struct FilterOptions {
    pub h: usize,
    pub w: usize,
    pub steps: usize,
    pub digital_iters: usize,
    pub out: PathBuf,
    pub quick: bool,
}

impl Default for FilterOptions {
    fn default() -> Self {
        FilterOptions {
            h: 80,
            w: 112,
            steps: 15_000,
            digital_iters: 6_000,
            out: PathBuf::from("showcase/analog_filter/out"),
            quick: false,
        }
    }
}

#[derive(Clone, Debug)]
pub struct EdgeMetrics {
    pub hole_l1: f64,
    pub edge_l1: f64,
    pub all_l1: f64,
}

impl EdgeMetrics {
    fn json(&self) -> serde_json::Value {
        json!({"hole_l1": self.hole_l1, "edge_l1": self.edge_l1, "all_l1": self.all_l1})
    }
}

fn edge_fields(h: usize, w: usize) -> (Array4<f32>, Array3<bool>, f32) {
    let mut target = Array4::<f32>::zeros((1, h, w, 3));
    let mut mask = Array3::<bool>::from_elem((1, h, w), false);
    let edge = 0.5f32;
    let left = [0.85f32, 0.22, 0.14];
    let right = [0.16f32, 0.38, 0.82];
    for y in 0..h {
        let _y = (y as f32 + 0.5) / h as f32;
        for x in 0..w {
            let xx = (x as f32 + 0.5) / w as f32;
            let t = 1.0 / (1.0 + (-(xx - edge) / 0.008).exp());
            for c in 0..3 {
                target[[0, y, x, c]] = (left[c] * (1.0 - t) + right[c] * t).clamp(0.0, 1.0);
            }
        }
    }
    for y in h / 3..2 * h / 3 {
        for x in w / 3..2 * w / 3 {
            mask[[0, y, x]] = true;
        }
    }
    (target, mask, edge)
}

fn face_grad(a: &Array4<f32>, y: usize, x: usize, dy: isize, dx: isize) -> f32 {
    let yy = (y as isize + dy) as usize;
    let xx = (x as isize + dx) as usize;
    let mut s = 0.0f32;
    for c in 0..3 {
        let d = a[[0, yy, xx, c]] - a[[0, y, x, c]];
        s += d * d;
    }
    (s / 3.0).sqrt()
}

fn solve_pm(
    target: &Array4<f32>,
    mask: &Array3<bool>,
    steps: usize,
    dt: f32,
    kappa: f32,
    rational: bool,
) -> Array4<f32> {
    let (_, h, w, _) = target.dim();
    let mut u = target.clone();
    let known = target.clone();
    for y in 0..h {
        for x in 0..w {
            if mask[[0, y, x]] {
                for c in 0..3 {
                    u[[0, y, x, c]] = 0.5;
                }
            }
        }
    }
    let mut next = u.clone();
    let k2 = (kappa.max(1e-6)).powi(2);
    for _ in 0..steps {
        for y in 1..h - 1 {
            for x in 1..w - 1 {
                if !mask[[0, y, x]] {
                    for c in 0..3 {
                        next[[0, y, x, c]] = known[[0, y, x, c]];
                    }
                    continue;
                }
                let gxr = face_grad(&u, y, x, 0, 1);
                let gxl = face_grad(&u, y, x, 0, -1);
                let gyd = face_grad(&u, y, x, 1, 0);
                let gyu = face_grad(&u, y, x, -1, 0);
                let cxr = if rational {
                    1.0 / (1.0 + gxr * gxr / k2)
                } else {
                    (-(gxr * gxr) / k2).exp()
                };
                let cxl = if rational {
                    1.0 / (1.0 + gxl * gxl / k2)
                } else {
                    (-(gxl * gxl) / k2).exp()
                };
                let cyd = if rational {
                    1.0 / (1.0 + gyd * gyd / k2)
                } else {
                    (-(gyd * gyd) / k2).exp()
                };
                let cyu = if rational {
                    1.0 / (1.0 + gyu * gyu / k2)
                } else {
                    (-(gyu * gyu) / k2).exp()
                };
                for c in 0..3 {
                    let p = u[[0, y, x, c]];
                    let div = cxr * (u[[0, y, x + 1, c]] - p)
                        + cxl * (u[[0, y, x - 1, c]] - p)
                        + cyd * (u[[0, y + 1, x, c]] - p)
                        + cyu * (u[[0, y - 1, x, c]] - p);
                    next[[0, y, x, c]] = (p + dt * div).clamp(0.0, 1.0);
                }
            }
        }
        std::mem::swap(&mut u, &mut next);
    }
    u
}

fn edge_error(
    out: &Array4<f32>,
    target: &Array4<f32>,
    mask: &Array3<bool>,
    edge: f32,
) -> EdgeMetrics {
    let (_, h, w, _) = target.dim();
    let mut hole = 0.0f64;
    let mut edge_s = 0.0f64;
    let mut edge_n = 0usize;
    let mut all = 0.0f64;
    let mut hole_n = 0usize;
    for y in 0..h {
        let xf = (y as f32 + 0.5) / h as f32;
        let _ = xf;
        for x in 0..w {
            let mut e = 0.0f64;
            for c in 0..3 {
                e += (out[[0, y, x, c]] - target[[0, y, x, c]]).abs() as f64;
            }
            e /= 3.0;
            all += e;
            if mask[[0, y, x]] {
                hole += e;
                hole_n += 1;
                let xx = (x as f32 + 0.5) / w as f32;
                if (xx - edge).abs() <= 0.035 {
                    edge_s += e;
                    edge_n += 1;
                }
            }
        }
    }
    EdgeMetrics {
        hole_l1: hole / hole_n.max(1) as f64,
        edge_l1: edge_s / edge_n.max(1) as f64,
        all_l1: all / (h * w).max(1) as f64,
    }
}

fn with_seconds(m: &EdgeMetrics, secs: f64) -> serde_json::Value {
    let mut o = m.json().as_object().unwrap().clone();
    o.insert("seconds".into(), json!(secs));
    serde_json::Value::Object(o)
}

fn time<T>(f: impl FnOnce() -> T) -> (T, f64) {
    let t = Instant::now();
    let v = f();
    (v, t.elapsed().as_secs_f64())
}

pub fn run(opts: &FilterOptions) -> Result<()> {
    if opts.h < 16 || opts.w < 16 {
        return Err(Error::runtime("analog filter: grid must be at least 16x16"));
    }
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
    std::fs::create_dir_all(&opts.out)?;
    let (target, mask, edge) = edge_fields(opts.h, opts.w);
    let (digital, t_dig) = time(|| solve_digital(&target, &mask, digital_iters, None, 0.5));
    let (rc, t_rc) = time(|| solve_rc(&target, &mask, steps, 0.12, None, None, 0.5));

    let mut results = serde_json::Map::new();
    let laplace = edge_error(&digital, &target, &mask, edge);
    let rcm = edge_error(&rc, &target, &mask, edge);
    results.insert("laplace".into(), with_seconds(&laplace, t_dig));
    results.insert("rc".into(), with_seconds(&rcm, t_rc));

    let mut best: Option<(f64, String, Array4<f32>, EdgeMetrics, f64)> = None;
    for (variant, rational) in [("exp", false), ("rat", true)] {
        for kappa in [0.03f32, 0.08, 0.16] {
            let (pm, secs) = time(|| solve_pm(&target, &mask, steps, 0.12, kappa, rational));
            let m = edge_error(&pm, &target, &mask, edge);
            println!(
                "pm_{variant}_k{kappa}: hole={:.4} edge={:.4} ({:.3}s)",
                m.hole_l1, m.edge_l1, secs
            );
            let mut j = with_seconds(&m, secs);
            if let Some(o) = j.as_object_mut() {
                o.insert("variant".into(), json!(variant));
                o.insert("kappa".into(), json!(kappa));
            }
            results.insert(format!("pm_{variant}_k{kappa}"), j);
            if best.as_ref().map_or(true, |b| m.hole_l1 < b.0) {
                best = Some((m.hole_l1, format!("pm_{variant}_k{kappa}"), pm, m, secs));
            }
        }
    }

    let (best_name, best_pm, best_m, best_secs) = {
        let (_, name, pm, m, secs) = best.ok_or_else(|| Error::runtime("no PM result"))?;
        (name, pm, m, secs)
    };
    save_panel(
        &opts.out.join("edge_panel.png"),
        &target,
        &mask,
        &[
            ("TARGET", &target),
            ("LAPLACE", &digital),
            ("RC", &rc),
            ("BEST PM .08", &best_pm),
        ],
    )?;

    let summary = json!({
        "grid": [opts.h, opts.w],
        "steps": steps,
        "filters": results,
        "best": {"name": best_name, "seconds": best_secs,
                 "hole_l1": best_m.hole_l1, "edge_l1": best_m.edge_l1},
        "real_hole_reference": {
            "note": "the existing real self-occlusion crop is worse for PM; see docs/ANALOG_FILTERS.md",
            "laplace_hole_l1": 0.17287562787532806,
            "best_pm_hole_l1": 0.17767557501792908
        }
    });
    std::fs::write(
        opts.out.join("metrics.json"),
        serde_json::to_string_pretty(&summary).unwrap() + "\n",
    )?;

    println!("analog filter bake-off: {}x{}", opts.h, opts.w);
    println!(
        "  laplace: hole={:.4} edge={:.4} ({:.3}s)",
        laplace.hole_l1, laplace.edge_l1, t_dig
    );
    println!(
        "  rc:      hole={:.4} edge={:.4} ({:.3}s)",
        rcm.hole_l1, rcm.edge_l1, t_rc
    );
    println!(
        "  best:    {} hole={:.4} edge={:.4} ({:.3}s)",
        best_name, best_m.hole_l1, best_m.edge_l1, best_secs
    );
    println!("  wrote {}", opts.out.display());
    Ok(())
}
