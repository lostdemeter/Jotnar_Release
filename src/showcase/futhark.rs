//! Futhark Lens as a Jotnar inference unit.
//!
//! The host only tokenizes/encodes text and renders the rune codepoints.
//! The translation decision is one Jotnar listing:
//!
//!   * a 128-bit fingerprint index over the trained pronunciation lexicon;
//!   * a compiled spelling-rule table for words outside the lexicon;
//!   * a phoneme -> rune table, including multi-rune phonemes.
//!
//! The data tables are ordinary I payload streams, not embedded host logic.
//! This is the word-level unit; text splitting and the final separator are
//! the host boundary.

use std::collections::BTreeMap;
use std::path::Path;

use serde_json::Value as Json;

use crate::builder::{Arg, Prog};
use crate::error::{Error, Result};
use crate::payload::Feeds;
use crate::value::Value;

pub const L: usize = 28;
pub const STATE: usize = 16;
pub const PH_SLOTS: usize = 3 * L;
pub const RUNE_SLOTS: usize = 8 * PH_SLOTS;
const FNV_OFFSET: i64 = -3750763034362895579; // 14695981039346656037 as i64
const FNV_PRIME: i64 = 1099511628211;

#[derive(Clone, Debug)]
pub struct FutharkPayload {
    pub n: usize,
    pub lex_h1: Vec<i64>,
    pub lex_h2: Vec<i64>,
    pub lex_off: Vec<i64>,
    pub lex_len: Vec<i64>,
    pub ph: Vec<i64>,
    pub runes: Vec<i64>,
    pub fallback: Vec<i64>,
    pub actions: Vec<i64>,
    pub max_phoneme_id: usize,
}

fn read_i64(path: &Path) -> Result<Vec<i64>> {
    let b = std::fs::read(path)?;
    if b.len() % 8 != 0 {
        return Err(Error::runtime(format!(
            "{} length {} is not a multiple of 8",
            path.display(),
            b.len()
        )));
    }
    let mut out = Vec::with_capacity(b.len() / 8);
    for c in b.chunks_exact(8) {
        out.push(i64::from_le_bytes([
            c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7],
        ]));
    }
    Ok(out)
}

impl FutharkPayload {
    pub fn load(dir: &Path) -> Result<Self> {
        let manifest: Json =
            serde_json::from_str(&std::fs::read_to_string(dir.join("manifest.json"))?)?;
        let n = manifest["hash_entries"].as_u64().unwrap_or(0) as usize;
        let p = Self {
            n,
            lex_h1: read_i64(&dir.join("lex_h1.bin"))?,
            lex_h2: read_i64(&dir.join("lex_h2.bin"))?,
            lex_off: read_i64(&dir.join("lex_off.bin"))?,
            lex_len: read_i64(&dir.join("lex_len.bin"))?,
            ph: read_i64(&dir.join("ph.bin"))?,
            runes: read_i64(&dir.join("runes.bin"))?,
            fallback: read_i64(&dir.join("fallback.bin"))?,
            actions: read_i64(&dir.join("actions.bin"))?,
            max_phoneme_id: manifest["max_phoneme_id"].as_u64().unwrap_or(0) as usize,
        };
        if p.lex_h1.len() != n
            || p.lex_h2.len() != n
            || p.lex_off.len() != n
            || p.lex_len.len() != n
        {
            return Err(Error::runtime("futhark payload: lexicon arrays disagree"));
        }
        Ok(p)
    }

    fn feeds(&self, chars: [i64; 32], len: i64) -> Result<Feeds> {
        let mut f = Feeds::new();
        let mut put = |name: &str, shape: &[usize], data: Vec<i64>| -> Result<()> {
            f.insert(
                name.to_string(),
                Value::I64(crate::value::array_from_vec(shape, data)?),
            );
            Ok(())
        };
        put("CHARS", &[32], chars.to_vec())?;
        put("LEN", &[1], vec![len])?;
        put("LEX_H1", &[self.n], self.lex_h1.clone())?;
        put("LEX_H2", &[self.n], self.lex_h2.clone())?;
        put("PH_OFF", &[self.n], self.lex_off.clone())?;
        put("PH_LEN", &[self.n], self.lex_len.clone())?;
        put("PH", &[self.ph.len()], self.ph.clone())?;
        put("RUNES", &[self.runes.len()], self.runes.clone())?;
        put("FALLBACK", &[self.fallback.len()], self.fallback.clone())?;
        put("ACTIONS", &[self.actions.len()], self.actions.clone())?;
        put("Z84", &[PH_SLOTS], vec![0; PH_SLOTS])?;
        put("Z672", &[RUNE_SLOTS], vec![0; RUNE_SLOTS])?;
        put("INIT0", &[1], vec![0])?;
        put("INIT1", &[1], vec![FNV_OFFSET])?;
        put("INIT2", &[1], vec![0])?;
        put("N1", &[1], vec![self.n as i64])?;
        Ok(f)
    }
}

fn named(p: &mut Prog, mn: &str, args: &[Arg], out: &str) -> Result<String> {
    p.op(mn, args, Some(out))
}

fn lit(v: i64) -> Arg {
    Arg::Int(v)
}

/// Build the listing. `n` is the lexicon table size; the binary search
/// count is `ceil(log2(n))`.
pub fn build_program(n: usize) -> Result<Prog> {
    let mut p = Prog::new("futhark-word");
    p.inp([
        "CHARS", "LEN", "LEX_H1", "LEX_H2", "PH_OFF", "PH_LEN", "PH", "RUNES", "FALLBACK",
        "ACTIONS", "Z84", "Z672", "INIT0", "INIT1", "INIT2", "N1",
    ])?;

    // 128-bit fingerprint: FNV-1a and a polynomial, over fixed-width chars.
    let mut h1 = "INIT1".to_string();
    let mut h2 = "INIT2".to_string();
    for i in 0..L {
        let c = named(
            &mut p,
            "IWINDOW",
            &[Arg::from("CHARS"), lit(i as i64), lit(1)],
            &format!("c{i}"),
        )?;
        h1 = named(
            &mut p,
            "IXOR",
            &[Arg::from(&h1), Arg::from(&c)],
            &format!("h1_{i}"),
        )?;
        h1 = named(
            &mut p,
            "IMUL",
            &[Arg::from(&h1), lit(FNV_PRIME)],
            &format!("hm_{i}"),
        )?;
        h2 = named(
            &mut p,
            "IMUL",
            &[Arg::from(&h2), lit(131)],
            &format!("h2m_{i}"),
        )?;
        h2 = named(
            &mut p,
            "IADD",
            &[Arg::from(&h2), Arg::from(&c)],
            &format!("h2a_{i}"),
        )?;
        h2 = named(
            &mut p,
            "IADD",
            &[Arg::from(&h2), lit(1)],
            &format!("h2b_{i}"),
        )?;
    }
    h1 = named(
        &mut p,
        "IXOR",
        &[Arg::from(&h1), Arg::from("LEN")],
        "h1_final",
    )?;
    h2 = named(&mut p, "IMUL", &[Arg::from(&h2), lit(131)], "h2f_0")?;
    h2 = named(&mut p, "IADD", &[Arg::from(&h2), Arg::from("LEN")], "h2f_1")?;
    h2 = named(&mut p, "IADD", &[Arg::from(&h2), lit(1)], "h2_final")?;

    // Binary search the sorted h1/h2 table.
    let mut lo = "INIT0".to_string();
    let mut hi = "N1".to_string();
    let steps = usize::BITS - n.max(1).saturating_sub(1).leading_zeros();
    for s in 0..steps {
        let sum = named(
            &mut p,
            "IADD",
            &[Arg::from(&lo), Arg::from(&hi)],
            &format!("sum{s}"),
        )?;
        let mid = named(
            &mut p,
            "IDIV",
            &[Arg::from(&sum), lit(2)],
            &format!("mid{s}"),
        )?;
        let m1 = named(
            &mut p,
            "IGATHER1",
            &[Arg::from("LEX_H1"), Arg::from(&mid)],
            &format!("m1_{s}"),
        )?;
        let m2 = named(
            &mut p,
            "IGATHER1",
            &[Arg::from("LEX_H2"), Arg::from(&mid)],
            &format!("m2_{s}"),
        )?;
        let l1 = named(
            &mut p,
            "ILT",
            &[Arg::from(&m1), Arg::from(&h1)],
            &format!("l1_{s}"),
        )?;
        let e1 = named(
            &mut p,
            "IEQ",
            &[Arg::from(&m1), Arg::from(&h1)],
            &format!("e1_{s}"),
        )?;
        let l2 = named(
            &mut p,
            "ILT",
            &[Arg::from(&m2), Arg::from(&h2)],
            &format!("l2_{s}"),
        )?;
        let el = named(
            &mut p,
            "IAND",
            &[Arg::from(&e1), Arg::from(&l2)],
            &format!("el_{s}"),
        )?;
        let less = named(
            &mut p,
            "IOR",
            &[Arg::from(&l1), Arg::from(&el)],
            &format!("less{s}"),
        )?;
        let mid1 = named(
            &mut p,
            "IADD",
            &[Arg::from(&mid), lit(1)],
            &format!("mid1{s}"),
        )?;
        lo = named(
            &mut p,
            "ISELECT",
            &[Arg::from(&less), Arg::from(&mid1), Arg::from(&lo)],
            &format!("lo{s}"),
        )?;
        hi = named(
            &mut p,
            "ISELECT",
            &[Arg::from(&less), Arg::from(&hi), Arg::from(&mid)],
            &format!("hi{s}"),
        )?;
    }
    let valid = named(&mut p, "ILT", &[Arg::from(&lo), Arg::from("N1")], "valid")?;
    let cidx = named(
        &mut p,
        "ISELECT",
        &[Arg::from(&valid), Arg::from(&lo), lit(0)],
        "cidx",
    )?;
    let g1 = named(
        &mut p,
        "IGATHER1",
        &[Arg::from("LEX_H1"), Arg::from(&cidx)],
        "g1",
    )?;
    let g2 = named(
        &mut p,
        "IGATHER1",
        &[Arg::from("LEX_H2"), Arg::from(&cidx)],
        "g2",
    )?;
    let e1 = named(&mut p, "IEQ", &[Arg::from(&g1), Arg::from(&h1)], "eqh1")?;
    let e2 = named(&mut p, "IEQ", &[Arg::from(&g2), Arg::from(&h2)], "eqh2")?;
    let ee = named(&mut p, "IAND", &[Arg::from(&e1), Arg::from(&e2)], "eqh")?;
    let found = named(
        &mut p,
        "IAND",
        &[Arg::from(&valid), Arg::from(&ee)],
        "found",
    )?;

    // Fallback: a compiled greedy spelling step at every position.
    let mut fb = "Z84".to_string();
    let mut skip = "INIT0".to_string();
    let mut has = "INIT0".to_string();
    for i in 0..L {
        let win = named(
            &mut p,
            "IWINDOW",
            &[Arg::from("CHARS"), lit(i as i64), lit(4)],
            &format!("win{i}"),
        )?;
        let ctx = named(
            &mut p,
            "IASSEMBLE",
            &[Arg::from(&win), lit(0), lit(4)],
            &format!("ctx{i}"),
        )?;
        let c0 = named(
            &mut p,
            "IAND",
            &[Arg::from(&ctx), lit(255)],
            &format!("f0_{i}"),
        )?;
        let s1 = named(
            &mut p,
            "ISAR",
            &[Arg::from(&ctx), lit(8)],
            &format!("s1_{i}"),
        )?;
        let c1 = named(
            &mut p,
            "IAND",
            &[Arg::from(&s1), lit(255)],
            &format!("f1_{i}"),
        )?;
        let s2 = named(
            &mut p,
            "ISAR",
            &[Arg::from(&ctx), lit(16)],
            &format!("s2_{i}"),
        )?;
        let c2 = named(
            &mut p,
            "IAND",
            &[Arg::from(&s2), lit(255)],
            &format!("f2_{i}"),
        )?;
        let s3 = named(
            &mut p,
            "ISAR",
            &[Arg::from(&ctx), lit(24)],
            &format!("s3_{i}"),
        )?;
        let c3 = named(
            &mut p,
            "IAND",
            &[Arg::from(&s3), lit(255)],
            &format!("f3_{i}"),
        )?;
        let t0 = named(
            &mut p,
            "IMAD",
            &[Arg::from(&c1), lit(28), Arg::from(&c0)],
            &format!("t0_{i}"),
        )?;
        let t1 = named(
            &mut p,
            "IMAD",
            &[Arg::from(&c2), lit(784), Arg::from(&t0)],
            &format!("t1_{i}"),
        )?;
        let ctx28 = named(
            &mut p,
            "IMAD",
            &[Arg::from(&c3), lit(21952), Arg::from(&t1)],
            &format!("cx{i}"),
        )?;

        let rem0 = named(
            &mut p,
            "ISUB",
            &[Arg::from("LEN"), lit(i as i64)],
            &format!("rem0_{i}"),
        )?;
        let rem = named(
            &mut p,
            "IMAX",
            &[Arg::from(&rem0), lit(0)],
            &format!("rem{i}"),
        )?;
        let rlt = named(
            &mut p,
            "ILT",
            &[Arg::from(&rem), lit(5)],
            &format!("rlt{i}"),
        )?;
        let remc = named(
            &mut p,
            "ISELECT",
            &[Arg::from(&rlt), Arg::from(&rem), lit(4)],
            &format!("remc{i}"),
        )?;
        let remidx0 = named(
            &mut p,
            "ISUB",
            &[Arg::from(&remc), lit(1)],
            &format!("ri0_{i}"),
        )?;
        let remidx = named(
            &mut p,
            "IMAX",
            &[Arg::from(&remidx0), lit(0)],
            &format!("ri{i}"),
        )?;
        let pz = named(
            &mut p,
            "IEQ",
            &[Arg::from("INIT0"), lit(if i == 0 { 0 } else { 1 })],
            &format!("pz{i}"),
        )?;
        let st0 = named(
            &mut p,
            "IMUL",
            &[Arg::from(&remidx), lit(4)],
            &format!("st0_{i}"),
        )?;
        let st1 = named(
            &mut p,
            "IMUL",
            &[Arg::from(&has), lit(2)],
            &format!("st1_{i}"),
        )?;
        let st2 = named(
            &mut p,
            "IADD",
            &[Arg::from(&st1), Arg::from(&pz)],
            &format!("st2_{i}"),
        )?;
        let state = named(
            &mut p,
            "IADD",
            &[Arg::from(&st0), Arg::from(&st2)],
            &format!("st{i}"),
        )?;
        let key = named(
            &mut p,
            "IMAD",
            &[Arg::from(&ctx28), lit(STATE as i64), Arg::from(&state)],
            &format!("key{i}"),
        )?;
        let widx0 = named(
            &mut p,
            "ISAR",
            &[Arg::from(&key), lit(3)],
            &format!("wi{i}"),
        )?;
        let widx = named(
            &mut p,
            "ISELECT",
            &[Arg::from(&found), lit(0), Arg::from(&widx0)],
            &format!("wig{i}"),
        )?;
        let kw = named(
            &mut p,
            "IGATHER1",
            &[Arg::from("FALLBACK"), Arg::from(&widx)],
            &format!("kw{i}"),
        )?;
        let kb = named(
            &mut p,
            "IAND",
            &[Arg::from(&key), lit(7)],
            &format!("kb{i}"),
        )?;
        let sh = named(&mut p, "IMUL", &[Arg::from(&kb), lit(8)], &format!("sh{i}"))?;
        let kwsh = named(
            &mut p,
            "ISAR",
            &[Arg::from(&kw), Arg::from(&sh)],
            &format!("kwsh{i}"),
        )?;
        let aid = named(
            &mut p,
            "IAND",
            &[Arg::from(&kwsh), lit(255)],
            &format!("aid{i}"),
        )?;
        let act = named(
            &mut p,
            "IGATHER1",
            &[Arg::from("ACTIONS"), Arg::from(&aid)],
            &format!("act{i}"),
        )?;
        let p0 = named(
            &mut p,
            "IAND",
            &[Arg::from(&act), lit(63)],
            &format!("p0_{i}"),
        )?;
        let a6 = named(
            &mut p,
            "ISAR",
            &[Arg::from(&act), lit(6)],
            &format!("a6_{i}"),
        )?;
        let p1 = named(
            &mut p,
            "IAND",
            &[Arg::from(&a6), lit(63)],
            &format!("p1_{i}"),
        )?;
        let a12 = named(
            &mut p,
            "ISAR",
            &[Arg::from(&act), lit(12)],
            &format!("a12_{i}"),
        )?;
        let p2 = named(
            &mut p,
            "IAND",
            &[Arg::from(&a12), lit(63)],
            &format!("p2_{i}"),
        )?;
        let a18 = named(
            &mut p,
            "ISAR",
            &[Arg::from(&act), lit(18)],
            &format!("a18_{i}"),
        )?;
        let ska = named(
            &mut p,
            "IAND",
            &[Arg::from(&a18), lit(3)],
            &format!("ska{i}"),
        )?;
        let active = named(
            &mut p,
            "ILT",
            &[Arg::from("INIT0"), Arg::from(&rem0)],
            &format!("ac{i}"),
        )?;
        let nskip = named(
            &mut p,
            "IEQ",
            &[Arg::from(&skip), Arg::from("INIT0")],
            &format!("ns{i}"),
        )?;
        let ok = named(
            &mut p,
            "IAND",
            &[Arg::from(&active), Arg::from(&nskip)],
            &format!("ok{i}"),
        )?;
        let p0m = named(
            &mut p,
            "ISELECT",
            &[Arg::from(&ok), Arg::from(&p0), lit(0)],
            &format!("p0m{i}"),
        )?;
        let p1m = named(
            &mut p,
            "ISELECT",
            &[Arg::from(&ok), Arg::from(&p1), lit(0)],
            &format!("p1m{i}"),
        )?;
        let p2m = named(
            &mut p,
            "ISELECT",
            &[Arg::from(&ok), Arg::from(&p2), lit(0)],
            &format!("p2m{i}"),
        )?;
        let sk0 = named(
            &mut p,
            "ISUB",
            &[Arg::from(&skip), lit(1)],
            &format!("sk0_{i}"),
        )?;
        let skn = named(
            &mut p,
            "ISELECT",
            &[Arg::from(&nskip), Arg::from(&ska), Arg::from(&sk0)],
            &format!("skn{i}"),
        )?;
        skip = named(
            &mut p,
            "ISELECT",
            &[Arg::from(&active), Arg::from(&skn), lit(0)],
            &format!("skip{i}"),
        )?;
        let o0 = named(
            &mut p,
            "ILT",
            &[Arg::from("INIT0"), Arg::from(&p0m)],
            &format!("o0_{i}"),
        )?;
        let o1 = named(
            &mut p,
            "ILT",
            &[Arg::from("INIT0"), Arg::from(&p1m)],
            &format!("o1_{i}"),
        )?;
        let o2 = named(
            &mut p,
            "ILT",
            &[Arg::from("INIT0"), Arg::from(&p2m)],
            &format!("o2_{i}"),
        )?;
        let oo = named(
            &mut p,
            "IOR",
            &[Arg::from(&o0), Arg::from(&o1)],
            &format!("oo{i}"),
        )?;
        let oo = named(
            &mut p,
            "IOR",
            &[Arg::from(&oo), Arg::from(&o2)],
            &format!("ooo{i}"),
        )?;
        has = named(
            &mut p,
            "IOR",
            &[Arg::from(&has), Arg::from(&oo)],
            &format!("has{i}"),
        )?;
        for (s, val) in [(0usize, &p0m), (1, &p1m), (2, &p2m)] {
            let id = named(
                &mut p,
                "IADD",
                &[Arg::from("INIT0"), lit((3 * i + s) as i64)],
                &format!("fid{i}_{s}"),
            )?;
            fb = named(
                &mut p,
                "ISCATTER1",
                &[Arg::from(&fb), Arg::from(&id), Arg::from(val)],
                &format!("fb{i}_{s}"),
            )?;
        }
    }

    // Lexicon phonemes, selected over the fallback at the same slots.
    let off = named(
        &mut p,
        "IGATHER1",
        &[Arg::from("PH_OFF"), Arg::from(&cidx)],
        "off",
    )?;
    let plen = named(
        &mut p,
        "IGATHER1",
        &[Arg::from("PH_LEN"), Arg::from(&cidx)],
        "plen",
    )?;
    let mut runes = "Z672".to_string();
    for j in 0..PH_SLOTS {
        let plj = named(
            &mut p,
            "ISUB",
            &[Arg::from(&plen), lit(j as i64)],
            &format!("plj{j}"),
        )?;
        let jlt = named(
            &mut p,
            "ILT",
            &[Arg::from("INIT0"), Arg::from(&plj)],
            &format!("jl{j}"),
        )?;
        let pos0 = named(
            &mut p,
            "IADD",
            &[Arg::from(&off), lit(j as i64)],
            &format!("po{j}"),
        )?;
        let pos = named(
            &mut p,
            "ISELECT",
            &[Arg::from(&jlt), Arg::from(&pos0), lit(0)],
            &format!("pp{j}"),
        )?;
        let val0 = named(
            &mut p,
            "IGATHER1",
            &[Arg::from("PH"), Arg::from(&pos)],
            &format!("pv{j}"),
        )?;
        let val = named(
            &mut p,
            "ISELECT",
            &[Arg::from(&jlt), Arg::from(&val0), lit(0)],
            &format!("pvj{j}"),
        )?;
        let lexp = named(
            &mut p,
            "ISELECT",
            &[Arg::from(&found), Arg::from(&val), lit(0)],
            &format!("lx{j}"),
        )?;
        let fbj = named(
            &mut p,
            "IWINDOW",
            &[Arg::from(&fb), lit(j as i64), lit(1)],
            &format!("fbj{j}"),
        )?;
        let ph = named(
            &mut p,
            "ISELECT",
            &[Arg::from(&found), Arg::from(&lexp), Arg::from(&fbj)],
            &format!("ph{j}"),
        )?;
        for s in 0..8 {
            let ridx = named(
                &mut p,
                "IMAD",
                &[Arg::from(&ph), lit(8), lit(s as i64)],
                &format!("ri{j}_{s}"),
            )?;
            let r = named(
                &mut p,
                "IGATHER1",
                &[Arg::from("RUNES"), Arg::from(&ridx)],
                &format!("r{j}_{s}"),
            )?;
            let id = named(
                &mut p,
                "IADD",
                &[Arg::from("INIT0"), lit((8 * j + s) as i64)],
                &format!("oid{j}_{s}"),
            )?;
            runes = named(
                &mut p,
                "ISCATTER1",
                &[Arg::from(&runes), Arg::from(&id), Arg::from(&r)],
                &format!("ro{j}_{s}"),
            )?;
        }
    }
    named(&mut p, "IEQ", &[Arg::from(&runes), lit(0)], "rz")?;
    // The previous IEQ result is reused for the length count.
    let nz = named(&mut p, "ISELECT", &[Arg::from("rz"), lit(0), lit(1)], "nz")?;
    named(
        &mut p,
        "ISUM",
        &[Arg::from(&nz), lit(0), lit(RUNE_SLOTS as i64)],
        "RLEN",
    )?;
    // Rename the scatter result for the output.
    named(&mut p, "IADD", &[Arg::from(&runes), lit(0)], "RUNES_OUT")?;
    Ok(p)
}

fn decode_runes(got: &Feeds) -> Result<String> {
    let runes = got
        .get("RUNES_OUT")
        .ok_or_else(|| Error::runtime("futhark: missing RUNES_OUT"))?
        .as_i64("RUNES_OUT")?;
    let mut out = String::new();
    for cp in runes.iter() {
        if *cp != 0 {
            if let Some(ch) = char::from_u32(*cp as u32) {
                out.push(ch);
            }
        }
    }
    Ok(out)
}

pub fn run_word(
    payload: &FutharkPayload,
    word: &str,
    opts: &crate::asm::AsmOptions,
) -> Result<String> {
    let text = build_program(payload.n)?.text()?;
    let chars = encode_word(word);
    let feeds = payload.feeds(chars, word.chars().count() as i64)?;
    decode_runes(&crate::runtime::run_text(&text, opts, &feeds)?)
}

/// Run many words against one assembled listing (used by the parity gate).
pub fn run_words(
    payload: &FutharkPayload,
    words: &[String],
    opts: &crate::asm::AsmOptions,
) -> Result<Vec<String>> {
    let text = build_program(payload.n)?.text()?;
    let bound = crate::asm::assemble(&text, opts)?;
    let mut out = Vec::with_capacity(words.len());
    for w in words {
        if w.chars().count() > L {
            return Err(Error::runtime(format!(
                "futhark: word {w:?} is longer than the fixed {L}-character unit"
            )));
        }
        let chars = encode_word(w);
        let feeds = payload.feeds(chars, w.chars().count() as i64)?;
        out.push(decode_runes(&crate::runtime::run(&bound, &feeds)?)?);
    }
    Ok(out)
}

pub fn encode_word(word: &str) -> [i64; 32] {
    let mut out = [0i64; 32];
    for (i, ch) in word.chars().take(L).enumerate() {
        out[i] = match ch {
            '\'' | '’' => 27,
            'a'..='z' => 1 + (ch as u8 - b'a') as i64,
            'A'..='Z' => 1 + (ch as u8 - b'A') as i64,
            _ => 0,
        };
    }
    out
}

/// Split text with the same word shape as the extension.
pub fn words(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    let mut prev_digit = false;
    for ch in text.chars() {
        let letter = ch.is_ascii_alphabetic();
        if letter {
            if cur.is_empty() && prev_digit {
                // The extension leaves a word attached to a preceding digit.
                cur.push(ch);
                continue;
            }
            cur.push(ch);
        } else if (ch == '\'' || ch == '’') && !cur.is_empty() {
            cur.push('\'');
        } else {
            if !cur.is_empty() {
                out.push(std::mem::take(&mut cur));
            }
        }
        prev_digit = ch.is_ascii_digit();
    }
    if !cur.is_empty() {
        out.push(cur);
    }
    out
}

/// Translate a word list, preserving original order and joining with the
/// extension's centered-dot separator.
pub fn translate_words(
    payload: &FutharkPayload,
    words: &[String],
    opts: &crate::asm::AsmOptions,
) -> Result<String> {
    let mut parts = Vec::with_capacity(words.len());
    for w in words {
        parts.push(run_word(payload, w, opts)?);
    }
    Ok(parts.join(" · "))
}

/// Load a simple word corpus (one word per line) and return the reference
/// output shape used by `scripts/parity_futhark.py`.
pub fn corpus(path: &Path) -> Result<Vec<String>> {
    let text = std::fs::read_to_string(path)?;
    Ok(text
        .lines()
        .map(|l| l.trim().to_string())
        .filter(|l| !l.is_empty() && !l.starts_with('#'))
        .collect())
}

#[allow(dead_code)]
fn _keep(_: BTreeMap<String, Value>) {}
