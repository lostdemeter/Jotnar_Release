use std::collections::BTreeMap;
use std::fs;
use std::net::{TcpListener, TcpStream};
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use jotnar_node_tools::workload::{
    run_bundle, Bundle, ExecutorEndpoint, LoadedBundle, RunConfig, StageExecutor, StageInput,
    StageSpec, BUNDLE_SCHEMA,
};

fn free_port() -> u16 {
    let l = TcpListener::bind("127.0.0.1:0").unwrap();
    l.local_addr().unwrap().port()
}

fn wait_port(port: u16) {
    let deadline = Instant::now() + Duration::from_secs(10);
    while Instant::now() < deadline {
        if TcpStream::connect(("127.0.0.1", port)).is_ok() {
            return;
        }
        thread::sleep(Duration::from_millis(25));
    }
    panic!("native runner on {port} did not start");
}

fn start_runner(port: u16, work: &PathBuf) -> Child {
    fs::create_dir_all(work).unwrap();
    Command::new(env!("CARGO_BIN_EXE_jotnar-native-runner"))
        .args([
            "--bind",
            &format!("127.0.0.1:{port}"),
            "--allow",
            "/bin/sh",
            "--work-root",
            work.to_str().unwrap(),
        ])
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .expect("start native runner")
}

fn temp_root() -> PathBuf {
    let id = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let root = std::env::temp_dir().join(format!(
        "jotnar-native-workload-test-{}-{id}",
        std::process::id()
    ));
    fs::create_dir_all(&root).unwrap();
    root
}

#[test]
fn two_node_stage_handoff_uses_the_network_and_a_ledger() {
    let root = temp_root();
    let bundle_dir = root.join("bundle");
    fs::create_dir_all(&bundle_dir).unwrap();
    let mut inputs = BTreeMap::new();
    inputs.insert(
        "prev".to_string(),
        StageInput {
            artifact: None,
            stage: Some("stage1".to_string()),
            output: Some("out1".to_string()),
            path: "in1".to_string(),
        },
    );
    let bundle = Bundle {
        schema: BUNDLE_SCHEMA.to_string(),
        name: "two-stage-test".to_string(),
        version: "1".to_string(),
        description: String::new(),
        artifacts: BTreeMap::new(),
        stages: vec![
            StageSpec {
                name: "stage1".to_string(),
                depends_on: vec![],
                executor: StageExecutor::default(),
                command: vec![
                    "{{program}}".to_string(),
                    "-c".to_string(),
                    "printf hello > out1".to_string(),
                ],
                inputs: BTreeMap::new(),
                outputs: vec!["out1".to_string()],
                timeout_ms: Some(30_000),
                gpu: None,
            },
            StageSpec {
                name: "stage2".to_string(),
                depends_on: vec!["stage1".to_string()],
                executor: StageExecutor::default(),
                command: vec![
                    "{{program}}".to_string(),
                    "-c".to_string(),
                    "cat in1 > out2".to_string(),
                ],
                inputs,
                outputs: vec!["out2".to_string()],
                timeout_ms: Some(30_000),
                gpu: None,
            },
        ],
    };
    bundle.validate().unwrap();
    fs::write(
        bundle_dir.join("bundle.json"),
        serde_json::to_vec_pretty(&bundle).unwrap(),
    )
    .unwrap();
    let loaded = LoadedBundle::load(&bundle_dir).unwrap();
    let reference = root.join("reference");
    fs::write(&reference, b"hello").unwrap();

    let port_a = free_port();
    let port_b = free_port();
    let mut a = start_runner(port_a, &root.join("runner-a"));
    let mut b = start_runner(port_b, &root.join("runner-b"));
    wait_port(port_a);
    wait_port(port_b);

    let addr_a = format!("127.0.0.1:{port_a}");
    let addr_b = format!("127.0.0.1:{port_b}");
    let mut placement = BTreeMap::new();
    placement.insert("stage1".to_string(), addr_a.clone());
    placement.insert("stage2".to_string(), addr_b.clone());
    let cfg = RunConfig {
        program: Some("/bin/sh".to_string()),
        executors: vec![
            ExecutorEndpoint {
                addr: addr_a,
                capability: serde_json::json!({"kind": "native", "target": "process"}),
            },
            ExecutorEndpoint {
                addr: addr_b.clone(),
                capability: serde_json::json!({"kind": "native", "target": "process"}),
            },
        ],
        placement,
        token: None,
        force: false,
        timeout_ms: Some(30_000),
        compare: vec![("stage2.out2".to_string(), reference)],
        fleet: None,
    };
    let result = run_bundle(&loaded, &root.join("ledger"), &cfg);
    let result = match result {
        Ok(v) => v,
        Err(e) => {
            let _ = a.kill();
            let _ = b.kill();
            panic!("workload failed: {e}");
        }
    };
    let _ = a.kill();
    let _ = b.kill();
    assert_eq!(result["ok"], true);
    assert_eq!(result["comparisons"][0]["match"], true);
    assert_eq!(result["stages"]["stage2"]["executor"], addr_b);
    assert_eq!(
        fs::read(root.join("ledger/stages/stage2/out2")).unwrap(),
        b"hello"
    );
    let _ = fs::remove_dir_all(&root);
}
