use jotnar::showcase::analog_filter::{run, FilterOptions};

#[test]
fn bakeoff_runs_and_writes_metrics() {
    let dir = std::env::temp_dir().join(format!("jotnar_af_{}", std::process::id()));
    let opts = FilterOptions {
        h: 24,
        w: 32,
        steps: 80,
        digital_iters: 80,
        out: dir.clone(),
        quick: true,
    };
    run(&opts).unwrap();
    let m: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(dir.join("metrics.json")).unwrap()).unwrap();
    assert!(m["filters"]["laplace"]["hole_l1"].as_f64().unwrap() > 0.0);
    assert!(m["best"]["name"].as_str().unwrap().starts_with("pm_"));
    assert!(dir.join("edge_panel.png").exists());
    let _ = std::fs::remove_dir_all(&dir);
}
