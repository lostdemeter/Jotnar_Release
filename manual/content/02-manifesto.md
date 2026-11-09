---
title: The Jotnar Manifesto
number: "01"
kicker: MANIFESTO / THE STACK
lede: Jotnar is not one object. It is an execution loop: a program, a machine, and a witness.
columns: 2
---

Jotnar is a family of ideas about computation. The first idea is that a program can be a **graph of named streams** rather than a sequence of instructions that mutates a machine. The second is that the graph can be lowered to very different machines without changing its meaning. The third is that the result can be **witnessed** — compared, byte for byte, against an independent oracle.

Those three ideas have names. Use them.

::: figure src="assets/diagrams/loop.svg" caption="The execution loop. The plan is not the machine, and the machine is not the proof."
:::

## The three aspects

| Aspect | What it is | What it is not |
|---|---|---|
| **Jotnar-LAN** | The language, IR, op set, and plan formats. The program. | A particular file extension or a Python module. |
| **Jotnar-MAC** | The machine or target: ISA, memory, lanes, OS, board. | The program itself. |
| **Jotnar-WIT** | The witness: trace, certificate, provenance, byte-identity. | A proof in the mathematical sense. |

When the distinction matters, write the suffix: **Jotnar-LAN/DIG**, **Jotnar-MAC/DIG**, **Jotnar-WIT/DIG**. When it does not, “Jotnar” means the family. Do not use “Jotnar” as the name of a project. Do not call a neural network “Jotnar.” The network is host code. Jotnar is the thing the host hands a plan to.

::: note [THE SHORT VERSION]
**Lachesis** measures the deterministic thread; **Atropos** makes the accurate cut. **Jotnar-LAN → Jotnar-MAC → Jotnar-WIT** is the execution loop. **Field → Map → Vector → Plan → Certificate** is the data/compute loop. **DIG / SIM / ANA / HYB** says what kind of Jotnar you mean.
:::

## Fields are state. Maps are stored.

Storage and processing are different axes. Confusing them produces bad sentences and worse programs.

| Term | Axis | Meaning |
|---|---|---|
| **Jotnar-FLD** (Field) | State | The spatial values/lattice: depth, normals, IDs, provenance, phi-core state. The *what*. |
| **Jotnar-MAP** (Map) | Storage | A stored field/artifact: shape, dtype, layout, hash. Data at rest. |
| **Refined Jotnar Map (RJM)** | Storage + provenance | A map that passed a refinement stage, carrying an error bound and provenance. Still data. |
| **Jotnar-VEC** (Vector) | Processing | A costed, data-parallel batch of work. Compute in motion. |
| **Jotnar-LANE** | Resource | The execution resource a vector runs on. |
| **Jotnar-PLN** (Plan) | Binding | The executable artifact that binds maps (inputs/outputs) to vectors (work). |
| **Jotnar-CERT** | Witness | The output of executing a plan. |

::: pullquote [THE DATA RULE]
Fields are state. Maps are stored. Vectors are executed. Plans bind them. Certificates witness the result.
:::

## Four flavors

Bare “Jotnar code” is ambiguous. Require a flavor.

| Flavor | Meaning | Status |
|---|---|---|
| **Jotnar-DIG** | Digital: exact, bit-identical, C/CUDA/RV64, JMV1/JSW1. | Exists. |
| **Jotnar-SIM** | Simulated analog: DAC/RC/ADC, 2D field, digital twin. | Exists as experiments. |
| **Jotnar-ANA** | Physical analog: RC/crossbar/optical substrate, calibration and error bounds, no bit-exactness. | Does not exist. |
| **Jotnar-HYB** | Hybrid: digital control + analog kernel + digital readout/gate. | Not built; realistic target. |

The RC field in the analog work is **Jotnar-SIM**, not **Jotnar-ANA**. The board is **Jotnar-MAC/DIG** running a **Jotnar-LAN/DIG** plan and emitting a **Jotnar-WIT/DIG** certificate. Say that, or say less.

## Why geometry?

Because geometry is the part of the problem that does not change when the machine does. A depth map is still a depth map on a CPU, a GPU, a piece of FPGA fabric, or a simulated analog field. A layout like `T:HW` is a promise about shape and kind that survives lowering. An exact integer add is exact on every target. A stream is a name for a value that does not care where it lives.

The language is small on purpose. There are 94 mnemonics. The registry is the authority. Every mnemonic has an arity, a layout signature, and a reference implementation. If a target cannot lower a mnemonic, it says so. If a shape rule fails, the runtime says so. If an emitter has no pattern, it returns `NoPattern`, loudly.

::: warning [THE WITNESS RULE]
The board is a witness, not a proof. Byte-identity against an oracle is strong evidence that the lowering preserved the plan. It is not a theorem that the plan computes what you meant. That part is still your job.
:::

## What Jotnar is not

Jotnar is not a project name. It is not the depth model, the JBU, the normal net, or the layer net; those are **Atropos** host code. It is not a file format, a Python package, or a particular GPU kernel. It is the stack: a program, a machine, and a witness. The names are load-bearing. Use them carefully and the rest of the system stays reviewable.

::: end
:::
