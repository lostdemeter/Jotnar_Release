use jotnar::builder::Prog;

#[test]
fn builder_emits_python_compatible_text() {
    let mut p = Prog::new("t");
    p.config("beta", "0.5");
    p.inp(["a", "b"]).unwrap();
    let t = p.op_names("ADD", &["a", "b"], Some("T")).unwrap();
    assert_eq!(t, "T");
    p.op_names("MUL", &["T", "a"], None).unwrap();
    p.defn("twice", &["x"], |sub| {
        let y = sub.op_names("ADD", &["x", "x"], None)?;
        Ok(vec![y])
    })
    .unwrap();
    let got = p.text().unwrap();
    let want = "CONFIG beta 0.5\nIN a\nIN b\nT = ADD(a, b)\nt1 = MUL(T, a)\nDEF twice(x) -> (t1)\n  t1 = ADD(x, x)\nEND\n";
    assert_eq!(got, want);
}

#[test]
fn image_builder_has_the_expected_shape() {
    let p = jotnar::showcase::image_demo::build_img_prog(288, 352).unwrap();
    let text = p.text().unwrap();
    let ops = text
        .lines()
        .filter(|l| l.contains('=') && !l.trim_start().starts_with("CONFIG"))
        .count();
    assert_eq!(ops, 43);
    assert!(text.starts_with("CONFIG m_acc 41200\nCONFIG m_cov 41200\n"));
    assert!(text.contains("strip = CONCAT(crop, blur, 1)\n"));
}
