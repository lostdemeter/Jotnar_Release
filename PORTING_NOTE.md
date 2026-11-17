# Port note — non-FPU pools, broadcast arithmetic, and T2 SiLU

Branch: `port/nonfpu-pools-silu`  
Base: upstream `main` at `51d1e87` (the fresh clone in this workspace)  
Source: `repos/Jotnar_Release` local commit `19a6fe2`

This branch is **staged locally for an integrator session**. It has not been
pushed to GitHub.

## What was ported

- `POOLMAX2D` — HWC max pool with `kh,kw,sh,sw,ph,pw`.
- `POOLAVG2D` — HWC average pool with the same geometry; the denominator
  includes padded taps.
- `MULC` — broadcast multiply for `(C)`, `(H,1,C)`, `(1,W,C)`, `(H,W,C)`.
- `ADDC` — broadcast add for the same shapes.
- T2 `SILU` — emitted integer pattern, phi-core reference, interpreter
  dispatch, and test coverage.
- Registry/docs/manual counts updated from upstream `101` to `105`.
- Tests: `tests/pool2d.rs`, `tests/broadcast.rs`, and the `SILU` case in
  `tests/nonfpu_t2.rs`.

## Upstream changes reviewed

The fresh `main` contains the stream arena/flat/fusion work, CUDA generic DAV2
ops, T2 table-cache changes, infer-rule completion, and many arena tests.
None of those files were reverted. The only textual conflict was the registry
count in `tests/registry_docs.rs`; resolved as `105`.

Relevant upstream files touched by this port:

- `src/ops/mod.rs` — four new registrations; no upstream registrations removed.
- `src/ops/image.rs` — pool functions added after `POOLAVG`; upstream image
  changes preserved.
- `src/ops/element.rs` — broadcast helpers added before `ADD`; upstream
  element changes preserved.
- `src/ops/t2.rs` — `SILU` dispatch added; upstream T2 table-cache changes
  preserved.
- `src/codegen/nonfpu.rs` — T2 `SILU` helper/pattern and `NF_SIG` emission;
  upstream non-FPU changes preserved.
- `crates/phi-core/src/t2.rs` — `silu` reference added.
- `tests/registry_docs.rs` — count updated to 105.
- Docs/manual — counts and image/op register entries updated.

## Verification on this branch

```text
cargo build                         PASS
cargo test                          PASS (full suite)
cargo test --test registry_docs     PASS (105)
cargo test --test nonfpu_t2         PASS (7/7)
cargo test --test pool2d            PASS (6/6)
cargo test --test broadcast         PASS (4/4)
```

Run from this clone:

```bash
cd repos/Jotnar_Release_upstream
git checkout port/nonfpu-pools-silu
git log --oneline origin/main..HEAD
git diff --stat origin/main...HEAD
cargo test
```

## Integrator checklist

1. Review the four op signatures against the intended ZipDepth map:
   `POOLMAX2D`/`POOLAVG2D` take six scalar geometry args; `MULC`/`ADDC` are
   HWC broadcast arithmetic.
2. Confirm the T2 `SILU` LUT path matches the intended non-FPU semantics.
3. Confirm `105` is the correct registry count after the upstream additions.
4. Run the full suite on the integrator's toolchain.
5. Merge into the local integration branch and push only after that review.

Do not push this branch directly from the porting session.

## Generated manual artifacts

`manual/out/manual.html` and `manual/out/jotnar-field-manual.pdf` are generated
by the press and still carry the old 101-count strings in this branch. The
source register and all registry-doc tests are updated to 105. Rebuild the
manual artifacts before publishing a release; do not hand-edit the generated
HTML/PDF.
