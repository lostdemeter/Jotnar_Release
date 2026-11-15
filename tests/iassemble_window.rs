//! IASSEMBLE must truncate, not underflow, when the top bit of an 8-byte
//! window is set. The old `v - 2^64` branch panicked in debug builds.

use jotnar::asm::{assemble, AsmOptions};
use jotnar::payload::Feeds;
use jotnar::runtime::{run_with_options, ExecOptions};
use jotnar::value::{array_from_vec, Value};

#[test]
fn eight_byte_window_with_sign_bit_is_negative_one() {
    let text = "IN X\nOUT = IASSEMBLE(X, 0, 8)\n";
    let bound = assemble(text, &AsmOptions::default()).unwrap();
    let mut f = Feeds::new();
    f.insert(
        "X".to_string(),
        Value::I64(array_from_vec(&[8], vec![0xFFu8 as i64; 8]).unwrap()),
    );
    let out = run_with_options(&bound, &f, &ExecOptions::default()).unwrap();
    match &out["OUT"] {
        Value::I64(a) => assert_eq!(a[0], -1),
        other => panic!("expected I64, got {:?}", other.kind()),
    }
}
