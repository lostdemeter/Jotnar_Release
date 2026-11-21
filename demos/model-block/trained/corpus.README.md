# Checkpoint corpus snapshot

This is the exact eight-file corpus used to train the checkpoint in this
directory. It is the documentation tree at `0f016c9`, before the slice-3 edits
and the Vulkan rebase. The archive is `corpus.tar.gz`; `corpus.manifest.json`
records the uncompressed corpus hash, the archive hash, and every per-file
sha256.

```text
archive        corpus.tar.gz        40,464 bytes
archive sha256 f030758a6f9e78eedd0c788480a095e789de1bb24bcff6ec976ae2bd8b5c1829
corpus bytes   110,491
corpus sha256  eb51498b55d742009b093d9a231e9739df49f80bf48da039949b940b126bf0a7
```

Extract and reproduce:

```sh
mkdir -p /tmp/jotnar-corpus
tar -xzf demos/model-block/trained/corpus.tar.gz -C /tmp/jotnar-corpus
python3 scripts/train_tiny_model.py \
  --corpus-root /tmp/jotnar-corpus \
  --steps 2500 --threads 8 --out /tmp/repro
sha256sum /tmp/repro/manifest.json
# 01b128cea5b4d2d12297503f2742186825d235ad4e432a0313ed60c7e9d0a4c4
```

Do not edit the extracted files. They are historical training bytes, not the
live documentation; some of them still contain the paths and text of the
pre-rebase tree. The live docs are elsewhere in the repository.
