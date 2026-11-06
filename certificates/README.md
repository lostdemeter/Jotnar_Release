# Certificates

A certificate is evidence with a declared scope. It is not a green checkmark.

Two examples are checked in:

- `example/certificate.json` — self-contained schema example; `verify` passes
  without any external artifacts.
- `doom/native-gpu-migration.json` — real DOOM/native/GPU certificate; it
  references local `/tmp` artifacts and reports `BLOCKED` when they are absent. It was generated
from a local run whose artifacts live under `/tmp/opencode`; on a machine
where those files are absent, `verify` reports **BLOCKED**, not PASS. That is
the intended behaviour.

```sh
cargo run -p jotnar-lab -- verify certificates/doom/native-gpu-migration.json
```

If the artifacts are present and hashes match, this prints `PASS`. If they are
missing, it prints `BLOCKED` and exits non-zero. A blocked certificate is a
valid honest artifact, but it is not a pass.

A certificate must state both `proves` and `does_not_prove`. A certificate
without a falsifier table, null control, gate `can_say_no`, or bench labels
is rejected by the schema.

Regenerate the DOOM bundle and certificate with:

```sh
python3 scripts/parity_doom_native.py \
  --gpu-work /tmp/opencode/doom_overlay \
  --bundle /tmp/opencode/doom_bundle

cargo run -p jotnar-lab -- certify doom \
  --bundle /tmp/opencode/doom_bundle/manifest.json \
  --out certificates/doom/native-gpu-migration.json
```

See `docs/LOOP.md` and `docs/appendix/TRUST_SPINE.md`.
