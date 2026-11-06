# Self-contained certificate example

This is the smallest complete certificate in the repo. It is a **schema
example**, not a Jotnar execution claim.

```sh
cargo run -p jotnar-lab -- verify certificates/example/certificate.json
```

Expected:

```text
PASS
```

It proves only that `tiny.txt` matches its recorded SHA-256. It explicitly
does not prove anything about Jotnar programs or targets.

Use it to see the fields and the verification path without needing DOOM
assets or a CUDA overlay. The real DOOM certificate is one directory up.
