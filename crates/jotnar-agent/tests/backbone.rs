use std::fs;

use jotnar::asm::{self, AsmOptions};
use jotnar_agent::{
    all_cases, case_views, evaluate, run_round, run_round_with, split_cases, Backbone, Harness,
    LoopedTaskEncoder, Split, TaskEncoder,
};

fn temp_dir(tag: &str) -> std::path::PathBuf {
    let p = std::env::temp_dir().join(format!(
        "jotnar-agent-backbone-{tag}-{}",
        std::process::id()
    ));
    let _ = fs::remove_dir_all(&p);
    fs::create_dir_all(&p).unwrap();
    p
}

#[test]
fn backbone_listings_assemble() {
    let named = TaskEncoder::prior();
    let backbone = Backbone::mlp(&named.w_enc, &named.b_enc, 32, 1);
    let encoder = LoopedTaskEncoder::prior(3).with_backbone(backbone);
    for loops in [1, 2, 3] {
        let listing = encoder.listings_file();
        asm::assemble(listing, &AsmOptions::new("."))
            .unwrap_or_else(|err| panic!("loops={loops}: {err}"));
    }
}

#[test]
fn named_backbone_reproduces_the_named_selection() {
    let base = Harness::default();
    let all = all_cases();
    let report = run_round(&base, &all, &temp_dir("named"));
    let views = case_views(&all, &report.base);
    let task = jotnar_agent::task_encoder::task_vector(
        views
            .iter()
            .find(|v| v.family == "matrix")
            .expect("matrix case"),
        &base,
    );
    let named = TaskEncoder::prior();
    let backbone = Backbone::named(&named.w_enc, &named.b_enc);
    assert_eq!(
        backbone.latent(&task),
        named.latent(&task).expect("named latent")
    );
    let plain = LoopedTaskEncoder::prior(3);
    let with_backbone = LoopedTaskEncoder::prior(3).with_backbone(backbone);
    assert_eq!(
        plain.scores(&task, 2).expect("plain scores"),
        with_backbone.scores(&task, 2).expect("backbone scores")
    );
}

#[test]
fn mlp_backbone_fits_the_named_latent() {
    let base = Harness::default();
    let mut train = split_cases(Split::Train);
    train.extend(split_cases(Split::Calibration));
    let report = run_round(&base, &train, &temp_dir("fit"));
    let views = case_views(&train, &report.base);
    let tasks: Vec<Vec<i64>> = views
        .iter()
        .map(|view| jotnar_agent::task_encoder::task_vector(view, &base))
        .collect();
    let named = TaskEncoder::prior();
    let targets: Vec<Vec<i64>> = tasks
        .iter()
        .map(|task| named.latent(task).expect("named latent"))
        .collect();
    let mut backbone = Backbone::mlp(&named.w_enc, &named.b_enc, 32, 1);
    let err = backbone.fit(&tasks, &targets, 1);
    assert!(err <= 8, "latent fit error too large: {err}");
}

#[test]
fn mlp_backbone_drives_corpus_adoption() {
    let base = Harness::default();
    let mut train = split_cases(Split::Train);
    train.extend(split_cases(Split::Calibration));
    let heldout = split_cases(Split::Heldout);
    let report = run_round(&base, &train, &temp_dir("fit-adopt"));
    let views = case_views(&train, &report.base);
    let tasks: Vec<Vec<i64>> = views
        .iter()
        .map(|view| jotnar_agent::task_encoder::task_vector(view, &base))
        .collect();
    let named = TaskEncoder::prior();
    let targets: Vec<Vec<i64>> = tasks
        .iter()
        .map(|task| named.latent(task).expect("named latent"))
        .collect();
    let mut backbone = Backbone::mlp(&named.w_enc, &named.b_enc, 32, 1);
    backbone.fit(&tasks, &targets, 1);

    let encoder = LoopedTaskEncoder::prior(3).with_backbone(backbone);
    let report = run_round_with(&base, &train, &temp_dir("adopt"), &encoder);
    assert!(report.adopted_flag);
    assert_eq!(report.adopted.correct(), train.len());
    assert_eq!(report.adopted.wrong(), 0);
    let held = evaluate(&report.adopted_harness, &heldout, false);
    assert_eq!(held.correct(), heldout.len());
    assert_eq!(held.wrong(), 0);
}

#[test]
fn qwen_backbone_listing_assembles_and_runs() {
    let named = TaskEncoder::prior();
    let backbone = Backbone::qwen_named(&named.w_enc, &named.b_enc, 10, 16);
    let encoder = LoopedTaskEncoder::prior(3).with_backbone(backbone);
    asm::assemble(encoder.listings_file(), &AsmOptions::new("."))
        .expect("qwen backbone listing assembles");
    let listing = encoder.listings_file();
    for op in [
        "MATMUL",
        "RMSNORM",
        "ROTARY",
        "SOFTMAX_WIDE",
        "SILU",
        "T2I",
        "IGET",
    ] {
        assert!(listing.contains(op), "qwen listing missing {op}");
    }

    let base = Harness::default();
    let all = all_cases();
    let report = run_round(&base, &all, &temp_dir("qwen-run"));
    let views = case_views(&all, &report.base);
    let task = jotnar_agent::task_encoder::task_vector(&views[0], &base);
    let scores = encoder.scores(&task, 2).expect("qwen scores");
    assert_eq!(scores.len(), jotnar_agent::ACTIONS.len());
}

#[test]
fn qwen_backbone_reproduces_the_named_latent() {
    let named = TaskEncoder::prior();
    let qwen = Backbone::qwen_named(&named.w_enc, &named.b_enc, 10, 16);
    let encoder = LoopedTaskEncoder::prior(3).with_backbone(qwen);
    let base = Harness::default();
    let all = all_cases();
    let report = run_round(&base, &all, &temp_dir("qwen-latent"));
    let views = case_views(&all, &report.base);
    for view in views {
        let task = jotnar_agent::task_encoder::task_vector(&view, &base);
        let expected = named.latent(&task).expect("named latent");
        assert_eq!(
            encoder.latents(&task, 1).expect("qwen listing latent"),
            expected,
            "qwen listing latent mismatch for {}",
            view.case_id
        );
    }
}

#[test]
fn t2i_iget_lower_to_c_and_nonfpu_and_match_the_interpreter() {
    use jotnar::codegen::{build, compile_program, CompileOptions, Target};
    use jotnar::Value;
    use ndarray::{ArrayD, IxDyn};
    use std::fs;
    use std::process::Command;

    let listing = "IN x AS T:*\ni = T2I(x)\no = IGET(i, 0)\n";
    let x = ArrayD::from_shape_vec(IxDyn(&[1, 2]), vec![1.0, 2.0]).unwrap();
    let triple = jotnar::lattice::encode(&x);
    let mut feeds = jotnar::payload::Feeds::new();
    feeds.insert("x".into(), Value::Triple(triple.clone()));

    // Reference interpreter: T2I rounds to nearest, IGET takes element 0.
    let got = jotnar::run_text(listing, &AsmOptions::new("."), &feeds).expect("interpreter");
    let want = match got.get("o").expect("o") {
        Value::I64(a) => *a.iter().next().unwrap(),
        other => panic!("unexpected {other:?}"),
    };
    assert_eq!(want, 1);

    let dir = std::env::temp_dir().join(format!("jotnar-t2i-iget-parity-{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();

    for (target, name) in [(Target::C, "c"), (Target::NonFpu, "nonfpu")] {
        let opts = CompileOptions {
            target,
            outputs: vec!["o".to_string()],
            asm: AsmOptions::new("."),
            use_fp16: false,
        };
        let artifact = compile_program(listing, &opts, feeds.clone()).expect("compile");
        let exe = build(&artifact.source, &dir, name, target).expect("emitted build");

        let s_path = dir.join(format!("{name}-s.bin"));
        let e_path = dir.join(format!("{name}-e.bin"));
        let z_path = dir.join(format!("{name}-z.bin"));
        let out_path = dir.join(format!("{name}-out.bin"));
        let s_bytes: Vec<u8> = triple.s.iter().map(|v| *v as u8).collect();
        fs::write(&s_path, s_bytes).unwrap();
        let mut e_bytes = Vec::new();
        for v in triple.e.iter() {
            e_bytes.extend_from_slice(&v.to_le_bytes());
        }
        fs::write(&e_path, e_bytes).unwrap();
        fs::write(&z_path, triple.z.iter().copied().collect::<Vec<u8>>()).unwrap();

        let status = Command::new(&exe)
            .arg(&s_path)
            .arg(&e_path)
            .arg(&z_path)
            .arg(&out_path)
            .status()
            .expect("run emitted binary");
        assert!(status.success(), "{name}: emitted binary failed");
        let bytes = fs::read(&out_path).unwrap();
        let got = i64::from_le_bytes(bytes[0..8].try_into().unwrap());
        assert_eq!(got, want, "{name}: emitted result must match interpreter");
    }
}

#[test]
fn qwen_forward_matches_named_with_zero_branches() {
    let named = TaskEncoder::prior();
    let qwen = Backbone::qwen_named(&named.w_enc, &named.b_enc, 10, 16);
    let base = Harness::default();
    let all = all_cases();
    let report = run_round(&base, &all, &temp_dir("qwen-forward"));
    let views = case_views(&all, &report.base);
    for view in views {
        let task = jotnar_agent::task_encoder::task_vector(&view, &base);
        assert_eq!(
            qwen.latent(&task),
            named.latent(&task).expect("named latent"),
            "qwen f64 forward mismatch for {}",
            view.case_id
        );
    }
}

#[test]
fn qwen_readout_training_reduces_masked_error() {
    let base = Harness::default();
    let calibration = split_cases(Split::Calibration);
    let report = run_round(&base, &calibration, &temp_dir("qwen-fit"));
    let views = case_views(&calibration, &report.base);
    let named = TaskEncoder::prior();
    let mut samples = Vec::new();
    for view in views {
        let task = jotnar_agent::task_encoder::task_vector(&view, &base);
        let target: Vec<f64> = named
            .latent(&task)
            .expect("named latent")
            .into_iter()
            .map(|v| v as f64)
            .collect();
        let mut masked = task.clone();
        for (i, value) in masked.iter_mut().enumerate() {
            if i % 3 == 0 {
                *value = 0;
            }
        }
        samples.push((masked, target));
    }

    let mut backbone = Backbone::qwen_random(&named.w_enc, &named.b_enc, 10, 16, 1, 0.2);
    let qwen = backbone.qwen.as_mut().expect("qwen weights");
    let before = samples
        .iter()
        .map(|(task, target)| {
            qwen.forward(task)
                .iter()
                .zip(target)
                .map(|(x, y)| (x - y).abs())
                .fold(0.0f64, f64::max)
        })
        .fold(0.0f64, f64::max);
    let after = qwen.fit_readouts(&samples, 1e-3, 3);
    assert!(
        after <= before,
        "readout fit should not increase masked error: {before} -> {after}"
    );
    let nonzero_wo = qwen.wo.iter().flatten().filter(|v| v.abs() > 1e-9).count();
    let nonzero_down = qwen
        .w_down
        .iter()
        .flatten()
        .filter(|v| v.abs() > 1e-9)
        .count();
    assert!(nonzero_wo > 0, "attention readout did not train");
    assert!(nonzero_down > 0, "MLP readout did not train");
}

#[test]
fn qwen_end_to_end_training_stays_below_zero_init() {
    let base = Harness::default();
    let all = all_cases();
    let report = run_round(&base, &all, &temp_dir("qwen-e2e"));
    let views = case_views(&all, &report.base);
    let named = TaskEncoder::prior();
    let samples: Vec<(Vec<i64>, Vec<f64>)> = views
        .iter()
        .map(|view| {
            let task = jotnar_agent::task_encoder::task_vector(view, &base);
            let target = named
                .latent(&task)
                .expect("named latent")
                .into_iter()
                .map(|v| v as f64 * 1.2 + 0.5)
                .collect();
            (task, target)
        })
        .collect();
    let mut backbone = Backbone::qwen_random(&named.w_enc, &named.b_enc, 10, 16, 1, 0.2);
    let qwen = backbone.qwen.as_mut().expect("qwen weights");
    let before = samples
        .iter()
        .map(|(task, target)| {
            qwen.forward(task)
                .iter()
                .zip(target)
                .map(|(x, y)| (x - y).abs())
                .fold(0.0f64, f64::max)
        })
        .fold(0.0f64, f64::max);
    let warm = qwen.fit_readouts(&samples, 1e-3, 3);
    let after = qwen.fit_end_to_end(&samples, 20, 0.001);
    assert!(warm.is_finite());
    assert!(after.is_finite());
    assert!(
        after <= before,
        "end-to-end should stay below the zero-init error: {before} -> {after}"
    );
}

#[test]
fn qwen_backbone_compiles_to_nonfpu() {
    use jotnar::codegen::{compile_program, CompileOptions, Target};
    use jotnar::Value;
    use ndarray::{ArrayD, IxDyn};

    let named = TaskEncoder::prior();
    let qwen = Backbone::qwen_named(&named.w_enc, &named.b_enc, 10, 16);
    let encoder = LoopedTaskEncoder::prior(3).with_backbone(qwen.clone());
    let base = Harness::default();
    let all = all_cases();
    let report = run_round(&base, &all, &temp_dir("qwen-nonfpu"));
    let views = case_views(&all, &report.base);
    let task = jotnar_agent::task_encoder::task_vector(&views[0], &base);

    let mut feeds = jotnar::payload::Feeds::new();
    for (j, value) in task.iter().enumerate() {
        feeds.insert(
            format!("t{j}"),
            Value::I64(ArrayD::from_elem(IxDyn(&[]), *value)),
        );
    }
    for (name, value) in qwen.feeds(&task).expect("feeds") {
        feeds.insert(name, value);
    }
    let outputs: Vec<String> = ["s0", "s1", "s2", "s3", "s4", "s5"]
        .iter()
        .map(|s| s.to_string())
        .collect();
    let opts = CompileOptions {
        target: Target::NonFpu,
        outputs,
        asm: AsmOptions::new("."),
        use_fp16: false,
    };
    let artifact = compile_program(encoder.listings_file(), &opts, feeds.clone())
        .expect("qwen listing should lower to non-FPU");
    assert!(artifact.source.contains("nf_to_fixed"));
    assert!(artifact.source.contains("IGET") || artifact.source.contains("h0_0"));
}

#[test]
fn qwen_backbone_drives_corpus_adoption() {
    let base = Harness::default();
    let mut train = split_cases(Split::Train);
    train.extend(split_cases(Split::Calibration));
    let heldout = split_cases(Split::Heldout);
    let named = TaskEncoder::prior();
    let qwen = Backbone::qwen_named(&named.w_enc, &named.b_enc, 10, 16);
    let encoder = LoopedTaskEncoder::prior(3).with_backbone(qwen);
    let report = run_round_with(&base, &train, &temp_dir("qwen-adopt"), &encoder);
    eprintln!(
        "QWENDBG base={} adopted={} flag={} chosen={:?} cands={}",
        report.base.correct(),
        report.adopted.correct(),
        report.adopted_flag,
        report.chosen_label,
        report.candidates.len()
    );
    assert!(report.adopted_flag);
    assert_eq!(report.adopted.correct(), train.len());
    assert_eq!(report.adopted.wrong(), 0);
    let held = evaluate(&report.adopted_harness, &heldout, false);
    assert_eq!(held.correct(), heldout.len());
    assert_eq!(held.wrong(), 0);
}

#[test]
fn qwen_backbone_weights_roundtrip() {
    let named = TaskEncoder::prior();
    let qwen = Backbone::qwen_named(&named.w_enc, &named.b_enc, 10, 16);
    let path = temp_dir("qwen-weights").join("qwen.json");
    qwen.save(&path).expect("save qwen");
    let back = Backbone::load(&path).expect("load qwen");
    let task = vec![1, 4, 6, 1, 1, 6, 0, -3, 0, 1, 1, 0, 2, 0, 1, 1, 1, 0, 0, 4];
    assert_eq!(qwen.latent(&task), back.latent(&task));
}

#[test]
fn triple_matmul_scale_probe() {
    use jotnar::Value;
    use ndarray::{ArrayD, IxDyn};
    let listing =
        "CONFIG m_acc 32768\nCONFIG m_cov 32768\nIN x AS T:*\nIN w AS T:*\ny = MATMUL(x, w)\n";
    let x = ArrayD::from_shape_vec(IxDyn(&[1, 1]), vec![1.0]).unwrap();
    let w = ArrayD::from_shape_vec(IxDyn(&[1, 1]), vec![1.0]).unwrap();
    let mut feeds = jotnar::payload::Feeds::new();
    feeds.insert("x".into(), Value::Triple(jotnar::lattice::encode(&x)));
    feeds.insert("w".into(), Value::Triple(jotnar::lattice::encode(&w)));
    let got = jotnar::run_text(listing, &AsmOptions::new("."), &feeds).expect("triple matmul");
    if let Some(Value::Triple(t)) = got.get("y") {
        println!("triple 1x1={:?}", jotnar::lattice::decode_triple(t));
    }
}

#[test]
fn backbone_weights_roundtrip() {
    let named = TaskEncoder::prior();
    let backbone = Backbone::mlp(&named.w_enc, &named.b_enc, 24, 1);
    let path = temp_dir("weights").join("backbone.json");
    backbone.save(&path).expect("save backbone");
    let back = Backbone::load(&path).expect("load backbone");
    let task = vec![1, 4, 6, 1, 1, 6, 0, -3, 0, 1, 1, 0, 2, 0, 1, 1, 1, 0, 0, 4];
    assert_eq!(backbone.latent(&task), back.latent(&task));
}
