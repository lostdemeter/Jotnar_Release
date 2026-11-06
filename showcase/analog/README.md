# The analog layer: Jotnar-SIM

This is the fourth first-contact exhibit. It is deliberately **not** another
digital backend. It is the smallest honest version of the analog story:

```text
digital target / hole mask
        |
        v
DAC on the hole boundary  ->  2D RC field  ->  ADC readout
        |
        v
tolerance gate + provenance
```

Run:

```sh
cargo run --release --bin jotnar-demo-analog -- --out showcase/analog/out
```

The run writes `panel.png`, `metrics.json`, and
`rjm_field2d.research.json`. The checked-in copies in this directory are a
reference run. `reference_image_panel.png` is the original real-hole check
from the Python experiment (sphere-plane warp): target | digital Laplace |
ideal RC | RC + 4-bit DAC | RC + 8-bit DAC. It is kept for continuity; the
Rust binary reproduces the synthetic sweep and the same field numbers, not
the Blender warp.

## What the panel shows

Left to right:

1. the target linear field;
2. a digital Laplace/Jacobi fill;
3. the ideal RC field;
4. RC with a 4-bit PCM DAC on the boundary;
5. RC with an 8-bit PCM DAC.

The digital and ideal-RC fields settle to the same harmonic solution. DAC
resolution is directly visible as quantization error. That is the point:
the field is a real boundary-value operator, not a metaphor.

## The numbers

On the reference 80x112x3 grid, 999 hole pixels:

| method | hole L1 |
|---|---:|
| digital Laplace | 1.39e-6 |
| ideal RC | 1.39e-6 |
| RC + 2-bit DAC | 0.0491 |
| RC + 4-bit DAC | 0.00553 |
| RC + 6-bit DAC | 0.000466 |
| RC + 8-bit DAC | 0.000117 |

At 8-bit DAC, ADC resolution and noise are also observable; the full sweeps
are in `metrics.json`. The 8-bit/10-bit operating point passes the tolerance
gate (`0.000266 <= 0.001`).

## Map vs vector, and the DAC question

The live path needs a DAC because it is programming the boundary of a
physical/simulated field. A **live no-DAC** field is correctly refused:

```json
"live_no_dac": {"accepted": false, "missing": ["dac.pcm"]}
```

The fast path is different, and this is the sense in which the DAC can
disappear from the runtime: solve once, store the settled field as a
**Refined Jotnar Map**-shaped artifact with provenance, then apply it as a
**Jotnar-VEC**-shaped lane batch over K patches (the lane-vector cost model,
per the nomenclature). The artifact therefore carries both profiles:

```json
"live":      {"accepted": true, "mode": "dac_rc_adc"}
"baked_map": {"accepted": true, "mode": "apply_refined_map"}
```

That is storage and compute separated: the map is data at rest; the vector
is work in motion. It is not a license to call the live solve DAC-free.

## What this does and does not prove

It proves:

- a 2D RC field reaches the digital Laplace solution in the ideal limit;
- DAC/ADC bits, settling, and noise are observable and sweepable;
- a capability gate can admit a field map and refuse a live path whose
  required primitive is absent;
- one map can be applied as a lane vector across many patches.

It does not prove:

- physical analog speed, energy, or device accuracy;
- that RC replaces surface completion (it does not);
- that the bit-exact RJM rule accepts this artifact. The gate is explicitly
  `"class": "tolerance"` with `"bit_exact": false` and
  `"provenance": "simulated"`.

This is **Jotnar-SIM**, not **Jotnar-ANA**. The honest hybrid target is
digital geometry + analog boundary conditioning + digital readout/gate.
