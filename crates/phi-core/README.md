# phi-core (Rust)

Standalone Rust port of `phi_core`: the phi lattice codec, fixed-point
bridge, frozen LUTs, exact compare/clip/bin helpers, the numpy-style substrate ops (matmul,
RMSNorm/LayerNorm, softmax, SiLU, GELU, scan, gather, average pool, conv,
deconv, warp, interpolate, groupnorm, IP), and the calibration/scoring
helpers (`m_of`, fold_batchnorm, PSNR, lattice bars).

Jotnar-LAN programs consume this crate through `jotnar::lattice` and the
re-exported `Triple`. The crate is intentionally independent of the Jotnar
frontend so the same substrate can serve other geometric programs.

```sh
cargo test -p phi-core
```
