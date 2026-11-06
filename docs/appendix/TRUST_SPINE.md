# Trust spine: certificate and verification

Status: implementation appendix. This is not a language feature and not a
claim that every Jotnar project is certified.

The normative summary is [`../LOOP.md`](../LOOP.md). This page records the
concrete schema and the first implementation in `crates/jotnar-lab`.

## Certificate schema

```json
{
  "schema": "jotnar.certificate",
  "schema_version": 1,
  "id": "doom/native-gpu-migration/...",
  "status": "current",
  "outcome": "solved",
  "verdict": "pass",
  "space": {
    "label": "doom-frame-1-2",
    "map": "doom.bin",
    "family": "rv32im-machine",
    "primitive": "migration",
    "sample": {"seed": 0, "size": "2 frames", "domain": "...",
                "forced_corners": ["frame1", "frame2"]},
    "reference": "C native frame 1 / GPU frame 2 (64 MB)",
    "bounds": ["modeled memory: 64 MB"],
    "hashes": {"image": "...", "wad": "..."}
  },
  "proposal": {"kind": "migration", "candidate": "...",
                "provenance": {"proposer": "...", "seed": "0"}},
  "gate": {"name": "byte_identical_migration", "level": "semantic",
            "reference": "...", "frozen_at": "...",
            "gate_sha256": "...", "can_say_no": true},
  "witness": {"map": "...", "input": "...", "output": "...", "trace": "..."},
  "proof": {"gate": "...", "contract": "..."},
  "derivation": {"trace": "..."},
  "assumptions": ["..."],
  "unknowns": ["..."],
  "proves": ["..."],
  "does_not_prove": ["..."],
  "falsifiers": [{"corruption": "...", "expected": "...", "actual": "...",
                   "caught_by": "...", "not_caught": false}],
  "nulls": [{"name": "...", "control": "...", "expected": "...",
              "actual": "...", "effect": "...", "status": "..."}],
  "bench": {"clock": "host_wall", "scope": "end_to_end",
             "kind": "measured", "runs": 3,
             "p50_ms": 0.0, "p99_ms": 0.0, "max_ms": 0.0,
             "units": "ms", "note": "..."},
  "refusal": null,
  "provenance": {"command": "...", "cwd": "...", "commit": "...",
                  "host": "...", "environment": {}},
  "artifacts": [{"role": "image", "path": "...", "sha256": "...",
                  "bytes": 0}],
  "supersedes": null,
  "superseded_by": null,
  "certificate_sha256": "..."
}
```

`certificate_sha256` is computed over canonical JSON without the hash field,
then written. `verify` recomputes it.

## Required honesty fields

`Certificate::validate` rejects a certificate that omits:

- `proves` or `does_not_prove`;
- assumptions;
- a gate hash;
- `can_say_no = true`;
- at least one falsifier row and one null row;
- bench `clock`, `scope`, and `kind` labels.

A certificate with `outcome = refused` must carry a refusal. A certificate
with `status = superseded` must carry `superseded_by`.

## The DOOM certificate

Example: `certificates/doom/native-gpu-migration.json`.

It records:

- frame 1: Rust vs C native on the first 64 MB, registers, PC, cycles, and
  framebuffer;
- frame 2: Rust checkpoint vs the GPU checkpoint on the first 64 MB,
  registers, PC, and framebuffer;
- **does not prove**: the GPU kernel for all programs, plan completeness,
  memory beyond 64 MB, or fleet throughput.

The falsifier table deliberately includes:

- `memory byte changed beyond 64 MB` → `not_caught` (outside the space);
- the historical `128 MB native vs 64 MB GPU` comparison bug → `not_caught`
  (the wrong space is not caught by the frame gate);
- `wrong frame` → confounded because the title frame is static;
- `replace gate with m == P(spec)` → the wrong map passes; that is the
  gate-vacuity test.

## Commands

```sh
# Build the bundle from the native/C/GPU parity run.
python3 scripts/parity_doom_native.py \
  --gpu-work /tmp/opencode/doom_overlay \
  --bundle /tmp/opencode/doom_bundle

# Read the existing artifacts; do not re-run the CPU.
cargo run -p jotnar-lab -- certify doom \
  --bundle /tmp/opencode/doom_bundle/manifest.json \
  --out certificates/doom/native-gpu-migration.json

cargo run -p jotnar-lab -- falsify doom \
  --bundle /tmp/opencode/doom_bundle/manifest.json

cargo run -p jotnar-lab -- null doom \
  --bundle /tmp/opencode/doom_bundle/manifest.json

cargo run -p jotnar-lab -- bench doom \
  --bundle /tmp/opencode/doom_bundle/manifest.json \
  --runner target/release/jotnar-demo-doom --runs 3 \
  --certificate certificates/doom/native-gpu-migration.json

cargo run -p jotnar-lab -- verify certificates/doom/native-gpu-migration.json
```

`verify` checks artifact hashes by default. `--no-hashes` skips the artifact
check, but the schema and certificate hash are still checked.

If artifacts are missing, `verify` prints `BLOCKED` and exits non-zero. That
is intentional: a certificate of a claim about files that are no longer
present is not a pass. A stale or historical certificate is also not a
current pass.

## Refusal and block

The harness accepts these as honest certificates:

```text
outcome = refused    (with a typed refusal)
outcome = blocked
outcome = not_tested
verdict = fail       (with the failure scope recorded)
status  = stale | historical | superseded
```

It does not print `PASS` for any of them. A caller may choose to treat a
block as acceptable for their own purpose, but that is the caller's policy,
not the certificate's claim.

## Mapping to the broader Jotnar projects

The same schema is intended for:

| project | boundary the certificate must state |
|---|---|
| board v2 | plan omission; rows not carried by the plan |
| solver | curated candidate set; sample size |
| field guide | species recurrence; open datapath case |
| Yarnball | bank proposes, exact gate accepts; boundary test |
| Echion | checked ≠ true; Bloom hit is evidence |
| resonance/Hopfield | frozen pre-registration; negative result |
| depth easyloop | worst frame; board blocked; far/rabbit-hole shots |

Those are imported as artifacts and certificates from their own repos; the
language repo does not have to own their runtimes.
