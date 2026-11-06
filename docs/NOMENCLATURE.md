# Nomenclature

Status: **normative glossary.** Use these terms in docs, commits, and
reports. When a sentence uses “Jotnar” without a qualifier, it means the
family/stack, not a specific claim.

## 1. Projects

| Name | Definition | Role |
|---|---|---|
| **Lachesis** | The deterministic geometry/witness effort: host emits a plan, the Jotnar board executes it, outputs geometry/provenance/trace/certificate, byte-identical to native. | A **tool**. |
| **Atropos** | The accurate real-video depth program: monocular depth → normals → layers → surface/fill, validated against Blender/BBB GT. | The **goal**. |
| **Jotnar** | The shared substrate/family: a spatial program, a machine/target, and a witness/contract. | The **stack**. |

Do **not** use “Jotnar” as a project name. Do **not** call the depth model,
JBU, normal net, or layer net “Jotnar”; they are Atropos host code.

## 2. The three Jotnar aspects

Jotnar is an execution loop, not a single object:

```text
Jotnar-LAN  plan/program/IR
    -> Jotnar-MAC  machine/target (RV64 board, OS, memory, lanes)
    -> Jotnar-WIT  witness (trace, certificate, provenance, byte-identity)
```

| Aspect | Meaning | Current instance |
|---|---|---|
| **Jotnar-LAN** | Language/IR/op set/plan formats. The program. | JMV1, JSW1, `plan/emit_*`, `solver_exclusive` shape programs |
| **Jotnar-MAC** | Machine/target: ISA, memory, lanes, OS, board. | RV64 board under Jotnar OS; CUDA/native are host executors of the same plan |
| **Jotnar-WIT** | Witness/contract: trace, certificate, provenance, trust boundary. | `trace.bin`, `cert.txt`, “the board is a witness, not a proof” |

Use the triple-letter suffix whenever the distinction matters. Examples:

- “JMV1 is a **Jotnar-LAN/DIG** plan.”
- “The RV64 board is a **Jotnar-MAC/DIG** machine.”
- “`cert.txt` is a **Jotnar-WIT/DIG** certificate.”
- “The board is a witness, not a proof” is a **Jotnar-WIT** rule.

## 3. Data and compute are different axes

The “Refined Jotnar Maps vs Jotnar Vectors” confusion comes from treating
storage and processing as interchangeable. They are not.

| Term | Axis | Meaning |
|---|---|---|
| **Jotnar-FLD** (Field) | State | The spatial values/lattice (depth, normal, ID, provenance, phi-core state). The *what*. |
| **Jotnar-MAP** (Map) | Storage | A **stored** field/artifact: shape, dtype, layout, hash. Data at rest. |
| **Refined Jotnar Map (RJM)** | Storage + provenance | A Jotnar-MAP that passed a refinement stage, carrying an error bound/provenance. Still data, not compute. |
| **Jotnar-VEC** (Vector) | Processing | A costed, data-parallel **batch of work**. Compute in motion. Not a stored field. |
| **Jotnar-LANE** (Lane) | Resource | The execution resource (1/2/4/8 lanes) a Jotnar-VEC runs on. |
| **Jotnar-PLN** (Plan) | Binding | The executable artifact that binds maps (inputs/outputs) to vectors (work). JMV1/JSW1 are plans. |
| **Jotnar-CERT** (Certificate) | Witness | The Jotnar-WIT output of executing a plan. |

Rule: **Fields are state. Maps are stored. Vectors are executed. Plans bind
them. Certificates witness the result.**

The “lane vector” in the analog work is a **cost model** for a Jotnar-VEC,
not the formal Jotnar-VEC itself.

## 4. Digital / simulated / analog / hybrid

Bare “Jotnar code” is ambiguous. Require a flavor:

| Flavor | Meaning | Status |
|---|---|---|
| **Jotnar-DIG** | Digital: exact, bit-identical, C/CUDA/RV64, JMV1/JSW1. | **Exists** |
| **Jotnar-SIM** | Simulated analog: DAC/RC/ADC, 2D field, digital twin. | **Exists as experiments** |
| **Jotnar-ANA** | Physical analog: RC/crossbar/optical substrate, calibration/error bound, no bit-exactness. | **Does not exist** |
| **Jotnar-HYB** | Hybrid: digital control + analog kernel + digital readout/gate. | **Not built; realistic target** |

Examples:

- “The board is Jotnar-MAC/DIG running a Jotnar-LAN/DIG plan and emitting a
  Jotnar-WIT/DIG certificate.”
- “The RC field is Jotnar-SIM, not Jotnar-ANA.”
- “Jotnar-ANA is a hypothetical inferred-stage backend, not a drop-in
  replacement for arbitrary Jotnar programs.”
- “A hybrid analog completion would be Jotnar-HYB.”

## 5. Naming rules

1. **Jotnar** alone = the family/stack. A specific claim names the aspect.
2. **Jotnar-LAN** = program; **Jotnar-MAC** = machine; **Jotnar-WIT** =
   witness.
3. **Jotnar-MAP/FLD** = data; **Jotnar-VEC** = work; **Jotnar-PLN** = binding.
4. **DIG/SIM/ANA/HYB** must be explicit for any digital/analog claim.
5. **Lachesis** = deterministic geometry/witness; **Atropos** = real-video
   depth. Never use “Jotnar” as a project name.
6. **Board** = Jotnar-MAC; **certificate** = Jotnar-WIT; **plan** =
   Jotnar-LAN.
7. The host depth model, JBU, normal net, and layer net are **Atropos**, not
   Jotnar.

## 6. Existing-name mapping

| Existing name | Nomenclature |
|---|---|
| JMV1 | Jotnar-LAN/DIG plan (multi-view controlled scene) |
| JSW1 | Jotnar-LAN/DIG plan (stereo warp) |
| `plan/emit_plan.py` | Jotnar-LAN/DIG emitter |
| `plan/emit_stereo.py` | Jotnar-LAN/DIG emitter (JSW1) |
| `board/render_plan.c` | Jotnar-MAC/DIG executor (JMV1) |
| `board/stereo_warp.c` | Jotnar-MAC/DIG executor (JSW1) |
| `trace.bin`, `cert.txt` | Jotnar-WIT/DIG |
| `oracle.npz` | host-only evaluation target; **not** a Jotnar artifact |
| `plan/stereo_geom_monitor.py` | host-side join of Jotnar-WIT with Blender GT |
| `stereo/analog_*.py` | Jotnar-SIM experiments |
| `test2dtosbs.py` | Atropos host runtime prototype |
| `depth_revisited` | Atropos host GT/evaluation reference |

## 7. Short version

- **Lachesis** measures the deterministic thread; **Atropos** makes the
  accurate cut.
- **Jotnar-LAN → Jotnar-MAC → Jotnar-WIT** is the execution loop.
- **Field → Map → Vector → Plan → Certificate** is the data/compute loop.
- **DIG / SIM / ANA / HYB** says what kind of Jotnar you mean.

## 8. Rust workspace mapping (this repository)

| Nomenclature | Rust location |
|---|---|
| phi-core substrate | `crates/phi-core/` (`lattice`, `triple`, `ops`, `numpy_ops`) |
| Jotnar-LAN/DIG | `src/asm.rs`, `src/ir.rs`, `src/ops/` |
| Jotnar-MAC/DIG | `src/codegen/` (`c` / `nonfpu` / `cuda`) |
| Jotnar-WIT/DIG | `src/runtime.rs`, `tests/`, `scripts/parity_*.py` |
| Jotnar-FLD | `phi_core::Triple` planes / future field modules |
| Jotnar-MAP | payload maps, static `streams` shape maps |
| Jotnar-VEC | a bound program/instruction batch (future explicit type) |
| Jotnar-PLN | a bound program + target selection |
| Jotnar-CERT | differential parity reports and emitted-artifact checks |

The Rust port keeps the Python names for source compatibility while the
documentation and commit language use the normative nomenclature.
