# Analog filter: performance and limits

This is the fourth second-leg exhibit. It asks a narrow question:

> Can an analog-friendly filter help the image problem, rather than just
> reproduce digital Laplace?

The candidate is **Perona–Malik anisotropic diffusion** — a resistor network
whose conductance falls with the local voltage gradient. The baseline is
isotropic RC/Laplace. The test is a sharp vertical color edge crossing a
rectangular hole.

```sh
cargo run --release --bin jotnar-demo-analog-filter
```

`--quick` runs a smaller settle; `--steps N` changes the RC/PM step count.

## Result

Synthetic edge, 80×112×3, 999 hole pixels:

| filter | hole L1 | edge-band L1 | simulated time |
|---|---:|---:|---:|
| Laplace | 0.0715 | 0.1181 | ~0.034 s |
| RC | 0.0715 | 0.1181 | ~0.266 s |
| PM exp `k=0.08` | 0.2380 | 0.1801 | ~0.478 s |
| PM rational `k=0.08` | **0.0686** | **0.1128** | ~0.344 s |
| PM rational `k=0.03` | 0.2072 | 0.2011 | ~0.343 s |
| PM rational `k=0.16` | 0.0708 | 0.1171 | ~0.342 s |

The best synthetic gain is about **4% relative** on the hole and **4.5%** on
the edge band. It costs roughly **10×** the Laplace solve time in this
simulated implementation on this host.

The real self-occlusion result is the important limit:

| filter | real-hole L1 |
|---|---:|
| Laplace / RC | 0.1729 |
| best PM rational `k=0.08` | 0.1777 |

PM is **worse** on the real hole. A local PDE cannot invent the missing
sphere side. The filter sees a boundary and a smoothness prior; it does not
know the geometry.

## How to read the performance numbers

These are **simulated wall-clock numbers on this host**, not analog
hardware speed. They separate three claims:

1. **quality** — hole and edge-band L1;
2. **simulated cost** — steps × cells and wall time;
3. **physical estimate** — RC settling, readout, calibration, noise. Not
   measured.

The useful interpretation is the same as the earlier work:

- not the fill itself;
- possibly boundary/seam correction around a surface completion;
- possibly edge-aware depth/JBU filtering;
- possibly temporal integration/denoising.

The positive hybrid remains **digital surface geometry + Laplace seam
correction**: the surface supplies structure, the field supplies low-frequency
continuity. Edge-aware correction is only ~1–2% better than Laplace there.

## Files

- `src/showcase/analog_filter.rs`
- `src/bin/jotnar-demo-analog-filter.rs`
- `tests/analog_filter.rs`
- checked-in `edge_panel.png` and `metrics.json` from the full run

The panel is: target | Laplace | RC | best PM. The red outline is the hole
boundary.
