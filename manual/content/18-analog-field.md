---
title: The Analog Field
number: "18"
kicker: FIELD NOTES / JOTNAR-SIM
lede: DAC → 2D RC field → ADC. A boundary-value operator, not a metaphor.
columns: 2
---

This is the **analog field** exhibit, not the analog filter of the next chapter. It is the smallest honest version of the analog story: encode a boundary with a DAC, let a 2D RC field settle, read it back with an ADC, and gate the result against digital Laplace.

```sh
cargo run --release --bin jotnar-demo-analog -- --out showcase/analog/out
```

The run writes `panel.png`, `metrics.json`, and `rjm_field2d.research.json`. The checked-in panel is a reference run.

::: figure src="assets/img/analog-field-duotone.png" caption="TARGET | DIGITAL | ANALOG | DAC4 | DAC8. The digital and ideal-RC fields settle to the same harmonic solution; DAC resolution appears as quantization error. The duotone is for the zine; the numbers are in the table."
:::

## What the panel shows

1. the target linear field;
2. a digital Laplace/Jacobi fill;
3. the ideal RC field;
4. RC with a 4-bit PCM DAC on the boundary;
5. RC with an 8-bit PCM DAC.

The digital and ideal-RC fields settle to the same harmonic solution. DAC resolution is directly visible as quantization error. That is the point: the field is a real boundary-value operator, not a metaphor.

## The numbers

On the reference 80×112×3 grid, 999 hole pixels:

| Method | Hole L1 |
|---|---:|
| Digital Laplace | 1.39e-6 |
| Ideal RC | 1.39e-6 |
| RC + 2-bit DAC | 0.0491 |
| RC + 4-bit DAC | 0.00553 |
| RC + 6-bit DAC | 0.000466 |
| RC + 8-bit DAC | 0.000117 |

At 8-bit DAC, ADC resolution and noise are also observable. The 8/10-bit operating point passes the tolerance gate: `0.000266 <= 0.001`.

::: note [THE GATE IS TOLERANCE, NOT BIT-EXACT]
The analog field gate is `"class": "tolerance"`, `"bit_exact": false`, `"provenance": "simulated"`. It is not a bit-exact Jotnar-MAP contract. The field is a simulated analog substrate; the gate says so.
:::

## Live path vs. baked map

The live path needs a DAC because it is programming the boundary of a simulated field. A **live no-DAC** field is correctly refused:

```json
"live_no_dac": {"accepted": false, "missing": ["dac.pcm"]}
```

The fast path is different. Solve once, store the settled field as a **Refined Jotnar Map**-shaped artifact with provenance, then apply it as a **Jotnar-VEC**-shaped lane batch over K patches. The artifact carries both profiles:

```json
"live":      {"accepted": true, "mode": "dac_rc_adc"}
"baked_map": {"accepted": true, "mode": "apply_refined_map"}
```

That is storage and compute separated: the map is data at rest; the vector is work in motion. It is not a license to call the live solve DAC-free.

## What it proves

- A 2D RC field reaches the digital Laplace solution in the ideal limit.
- DAC/ADC bits, settling, and noise are observable and sweepable.
- A capability gate can admit a field map and refuse a live path whose required primitive is absent.
- One map can be applied as a lane vector across many patches.

## What it does not prove

- **Physical analog speed, energy, or device accuracy.** This is Jotnar-SIM, not Jotnar-ANA.
- **That RC replaces surface completion.** It does not.
- **That the bit-exact RJM rule accepts this artifact.** The gate is explicitly a tolerance gate.
- **That a simulated wall-clock is hardware time.** It is not.

The honest hybrid target remains **digital geometry + analog boundary conditioning + digital readout/gate**. The field supplies low-frequency continuity; it does not invent the missing geometry.

::: pullquote [ANALOG FIELD RULE]
The field is a boundary-value operator. It is not a completion model, and a simulated second is not a hardware second.
:::

::: end
:::
