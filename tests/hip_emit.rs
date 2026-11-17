//! HIP emitter wiring and CUDA->HIP API surface.
//!
//! These checks do not need a GPU or hipcc: they exercise target selection
//! and the generated source. The numerical gate lives in
//! `scripts/parity_hip.py` and runs on a ROCm host.

use jotnar::asm::AsmOptions;
use jotnar::codegen::{compile_program, CompileOptions, Target};
use jotnar::payload::Feeds;
use jotnar::value::Value;
use ndarray::{ArrayD, IxDyn};

fn feeds(pairs: &[(&str, Value)]) -> Feeds {
    let mut f = Feeds::new();
    for (k, v) in pairs {
        f.insert((*k).into(), v.clone());
    }
    f
}

fn opts(outputs: &[&str]) -> CompileOptions {
    CompileOptions {
        target: Target::Hip,
        outputs: outputs.iter().map(|s| s.to_string()).collect(),
        asm: AsmOptions::new("."),
        use_fp16: false,
    }
}

#[test]
fn hip_target_parses() {
    assert_eq!(Target::parse("hip").unwrap(), Target::Hip);
    assert_eq!(Target::parse("cuda").unwrap(), Target::Cuda);
}

#[test]
fn hip_elementwise_uses_hip_apis() {
    let listing = "IN a\nIN b\nC = ADD(a, b)\nOUT = MUL(C, a)\n";
    let a = Value::F64(ArrayD::from_shape_vec(IxDyn(&[4]), vec![1.0, 2.0, 3.0, 4.0]).unwrap());
    let b = Value::F64(ArrayD::from_shape_vec(IxDyn(&[4]), vec![5.0, 6.0, 7.0, 8.0]).unwrap());
    let art = compile_program(listing, &opts(&["OUT"]), feeds(&[("a", a), ("b", b)]))
        .expect("compile hip");
    assert!(art.source.contains("#include <hip/hip_runtime.h>"));
    assert!(art.source.contains("#include <hip/hip_fp16.h>"));
    assert!(art.source.contains("#include <hipblas/hipblas.h>"));
    assert!(art.source.contains("hipMalloc"));
    assert!(art.source.contains("hipMemcpyHostToDevice"));
    assert!(art.source.contains("hipLaunchKernelGGL"));
    assert!(art.source.contains("hipDeviceSynchronize"));
    assert!(art.source.contains("k_add"));
    assert!(art.source.contains("k_mul"));
    assert!(!art.source.contains("cudaMalloc"));
    assert!(!art.source.contains("cudaMemcpy"));
    assert!(!art.source.contains("<<<"));
}

#[test]
fn hip_integer_and_matmul_use_hip_apis() {
    let listing = "IN a AS I\nIN b AS I\nOUT = IADD(a, b)\n";
    let a = Value::I64(ArrayD::from_shape_vec(IxDyn(&[3]), vec![1, 2, 3]).unwrap());
    let b = Value::I64(ArrayD::from_shape_vec(IxDyn(&[3]), vec![4, 5, 6]).unwrap());
    let art = compile_program(listing, &opts(&["OUT"]), feeds(&[("a", a), ("b", b)]))
        .expect("compile hip integer");
    assert!(art.source.contains("hipMalloc"));
    assert!(art.source.contains("k_iadd"));
    assert!(!art.source.contains("cuda"));

    let listing = "IN a\nIN b\nOUT = MATMUL(a, b)\n";
    let a = Value::F64(
        ArrayD::from_shape_vec(IxDyn(&[2, 3]), (1..=6).map(|x| x as f64).collect()).unwrap(),
    );
    let b = Value::F64(
        ArrayD::from_shape_vec(IxDyn(&[3, 2]), (1..=6).map(|x| x as f64).collect()).unwrap(),
    );
    let art = compile_program(listing, &opts(&["OUT"]), feeds(&[("a", a), ("b", b)]))
        .expect("compile hip matmul");
    assert!(art.source.contains("k_matmul"));
}
