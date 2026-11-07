//! The registry is the authority; the docs must not drift from it.
//!
//! This is a small guard against the exact failure mode that adding an opcode
//! invites: a new mnemonic in `src/ops/mod.rs` with a stale manual count.

use std::collections::BTreeSet;

use jotnar::ops::registry;

fn read(path: &str) -> String {
    std::fs::read_to_string(format!("{}/{path}", env!("CARGO_MANIFEST_DIR")))
        .unwrap_or_else(|e| panic!("read {path}: {e}"))
}

#[test]
fn registry_and_manual_register_agree() {
    let names: BTreeSet<&str> = registry().names().into_iter().collect();
    let tsv = read("manual/data/ops.tsv");
    let manual: BTreeSet<String> = tsv
        .lines()
        .filter(|line| !line.trim().is_empty())
        .map(|line| line.split('\t').next().unwrap_or_default().to_string())
        .collect();

    assert_eq!(
        names.len(),
        manual.len(),
        "registry vs manual register size"
    );
    for name in &names {
        assert!(
            manual.contains(*name),
            "{name} is in the Rust registry but missing from manual/data/ops.tsv"
        );
    }
    for name in &manual {
        assert!(
            names.contains(name.as_str()),
            "{name} is in manual/data/ops.tsv but not in the Rust registry"
        );
    }
}

#[test]
fn mnemonic_count_is_documented_as_ninety() {
    let count = registry().len();
    assert_eq!(count, 90, "registry mnemonic count");
    let readme = read("README.md");
    assert!(
        readme.contains(&format!("{count} mnemonics")),
        "README must state the registry count ({count})"
    );
    let register = read("manual/content/12-register.md");
    assert!(
        register.contains(&format!("ALL {count} MNEMONICS")),
        "manual register kicker must state the registry count ({count})"
    );
}
