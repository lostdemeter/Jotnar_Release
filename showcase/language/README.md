# The first 15 minutes

Four listings that introduce the language without the rest of the stack:

| file | concept |
|---|---|
| `saxpy.asm` | linear elementwise integer streams (`IMUL`, `IADD`) |
| `reduce.asm` | a balanced reduction (fan-in and dataflow) |
| `matmul2.asm` | a shared operand graph with multiple outputs |
| `select.asm` | an exact predicate/select (`ISELECT`) |

Run the whole tour with the interpreter:

```sh
cargo run --release --bin jotnar-demo-tour
```

Or run one listing yourself:

```sh
cat > /tmp/saxpy.json <<'JSON'
{"streams":{
  "A":{"kind":"I64","shape":[4],"data":["3","3","3","3"]},
  "X":{"kind":"I64","shape":[4],"data":["1","2","3","4"]},
  "Y":{"kind":"I64","shape":[4],"data":["10","20","30","40"]}
}}
JSON
cargo run --release --bin jotnar -- run showcase/language/saxpy.asm \
  --payload /tmp/saxpy.json --outputs OUT
```

Expected:

```
[13, 26, 39, 52]
```

The same listings can be lowered to C:

```sh
cargo run --release --bin jotnar -- build showcase/language/saxpy.asm \
  --payload /tmp/saxpy.json --outputs OUT --workdir /tmp/saxpy --name saxpy
```

The point of this first page is the **program text**: streams, shapes,
exact integer arithmetic, and the fail-loud assembler. The image and machine
demos come next.
