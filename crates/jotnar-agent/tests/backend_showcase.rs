use std::fs;

use serde_json::Value;

#[test]
fn backend_showcase_is_green() {
    let dir = std::env::temp_dir().join(format!(
        "jotnar-agent-backend-showcase-{}",
        std::process::id()
    ));
    let _ = fs::remove_dir_all(&dir);
    jotnar_agent::backend_showcase::run(&dir).expect("backend showcase runs");
    let summary: Value =
        serde_json::from_str(&fs::read_to_string(dir.join("summary.json")).unwrap()).unwrap();

    assert_eq!(summary["backend_parity"]["kernels"], 18);
    assert_eq!(summary["backend_parity"]["c"], 18);
    assert_eq!(summary["backend_parity"]["nonfpu"], 18);
    assert_eq!(summary["regressions"], 0);

    for split in ["train", "heldout"] {
        let before = summary["splits"][split]["before"]["weighted"]
            .as_u64()
            .unwrap();
        let after = summary["splits"][split]["after"]["weighted"]
            .as_u64()
            .unwrap();
        assert!(
            after < before,
            "{split}: expected a weighted-op reduction, {before} -> {after}"
        );
    }

    assert!(dir.join("report.md").exists());
    assert!(dir.join("certificates/train/train-scale2.json").exists());
    assert!(dir.join("optimized/train/train-scale2.asm").exists());
    let _ = fs::remove_dir_all(&dir);
}
