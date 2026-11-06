use jotnar::asm::AsmOptions;
use jotnar::showcase::futhark::{encode_word, run_word, FutharkPayload};

fn hash_word(word: &str) -> (i64, i64) {
    let chars = encode_word(word);
    let mut h1: u64 = 14695981039346656037;
    let mut h2: u64 = 0;
    for c in chars.iter().take(28) {
        h1 ^= *c as u64;
        h1 = h1.wrapping_mul(1099511628211);
        h2 = h2.wrapping_mul(131).wrapping_add(*c as u64).wrapping_add(1);
    }
    h1 ^= word.chars().count() as u64;
    h2 = h2
        .wrapping_mul(131)
        .wrapping_add(word.chars().count() as u64)
        .wrapping_add(1);
    (h1 as i64, h2 as i64)
}

#[test]
fn lookup_path_runs_as_a_jotnar_program() {
    let (h1, h2) = hash_word("the");
    let mut runes = vec![0i64; 4 * 8];
    // DH = 2, AH = 3 in this tiny fixture.
    runes[2 * 8] = 'ᚦ' as i64;
    runes[3 * 8] = 'ᚨ' as i64;
    let payload = FutharkPayload {
        n: 1,
        lex_h1: vec![h1],
        lex_h2: vec![h2],
        lex_off: vec![0],
        lex_len: vec![2],
        ph: vec![2, 3],
        runes,
        fallback: vec![0],
        actions: vec![0],
        max_phoneme_id: 3,
    };
    let opts = AsmOptions::new(".");
    assert_eq!(run_word(&payload, "the", &opts).unwrap(), "ᚦᚨ");
}

#[test]
fn tokenization_keeps_apostrophes() {
    let w = jotnar::showcase::futhark::words("Trump’s policy isn't final.");
    assert_eq!(w, vec!["Trump's", "policy", "isn't", "final"]);
}
