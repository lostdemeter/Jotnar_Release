//! Warm-frame emitter wiring. These tests do not need a GPU or toolchain.

use jotnar::asm::AsmOptions;
use jotnar::codegen::{compile_program, CompileOptions, Target};
use jotnar::payload::Feeds;
use jotnar::value::Value;
use ndarray::{ArrayD, IxDyn};

fn feeds() -> Feeds {
    let mut f = Feeds::new();
    f.insert(
        "a".to_string(),
        Value::F64(ArrayD::from_shape_vec(IxDyn(&[4]), vec![1.0, 2.0, 3.0, 4.0]).unwrap()),
    );
    f.insert(
        "b".to_string(),
        Value::F64(ArrayD::from_shape_vec(IxDyn(&[4]), vec![5.0, 6.0, 7.0, 8.0]).unwrap()),
    );
    f
}

fn opts(target: Target) -> CompileOptions {
    CompileOptions {
        target,
        outputs: vec!["OUT".to_string()],
        asm: AsmOptions::new("."),
        use_fp16: false,
    }
}

#[test]
fn warm_frame_emits_worker_for_gpu_targets() {
    let listing = "CONFIG warm_frame 1\nIN a\nIN b\nOUT = ADD(a, b)\n";
    for target in [Target::Cuda, Target::Hip] {
        let art = compile_program(listing, &opts(target), feeds()).unwrap();
        assert!(art.warm_frame, "{target:?} did not mark warm_frame");
        assert!(art.source.contains("int warm_main("), "{target:?}");
        assert!(art.source.contains("--warm-frame"), "{target:?}");
    }
}

#[test]
fn warm_frame_is_off_by_default() {
    let listing = "IN a\nIN b\nOUT = ADD(a, b)\n";
    let art = compile_program(listing, &opts(Target::Cuda), feeds()).unwrap();
    assert!(!art.warm_frame);
    assert!(!art.source.contains("warm_main"));
}

#[test]
fn warm_frame_rejects_cuda_profile() {
    let listing = "CONFIG warm_frame 1\nCONFIG profile 1\nIN a\nIN b\nOUT = ADD(a, b)\n";
    let err = compile_program(listing, &opts(Target::Cuda), feeds()).unwrap_err();
    assert!(
        err.to_string()
            .contains("warm_frame does not support CONFIG profile"),
        "{err}"
    );
}
