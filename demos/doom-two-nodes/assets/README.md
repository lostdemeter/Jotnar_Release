# Assets

The launcher looks for:

```text
doom.bin     RV32IM DOOM image
doom1.wad    DOOM shareware WAD
```

`run.sh` copies them here from the standard external locations if they are
missing:

```text
$JOTNAR_ROOT/programs/doom/build/doom.bin
$HOME/Documents/OpenCode/jotnar-assets/doom1.wad
```

`package.sh` includes both files in the standalone tarball. They are not
committed to git.
