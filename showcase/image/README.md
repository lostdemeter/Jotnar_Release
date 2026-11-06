# Image pipeline

One 43-op Jotnar listing, three targets:

```sh
cargo run --release --bin jotnar-demo-img -- \
  --input samples/input_example.png --gallery gallery
```

- `input.png` — source
- `c_strip.png` — C target, crop/blur strip
- `cu_us.png` — CUDA target, unsharp result
- `nf_us.png` — non-FPU integer target, unsharp result

The listing is built by `src/showcase/image_demo.rs` and is byte-identical
to the Python generated listing. The C/non-FPU/CUDA outputs are compared in
the same run; see `docs/PORT_BACKLOG.md` and `scripts/parity_emit.py`.
