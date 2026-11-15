//! Byte-index, checksum, and format-kernel primitives.

use std::collections::HashMap;

use ndarray::ArrayD;

use crate::error::{Error, Result};
use crate::ir::Config;
use crate::value::{Arg, Value};

use super::{arg_i64, arg_i64_array, shape_of};

fn int_stream(args: &[Arg], i: usize, op: &str, name: &str) -> Result<ArrayD<i64>> {
    let a = arg_i64_array(args, i, op)?;
    if a.ndim() != 1 && !matches!(op, "ISCATTER1") {
        // Most byte primitives are explicitly 1D; per-op checks follow.
    }
    let _ = name;
    Ok(a)
}

fn flat<T: Copy>(a: &ArrayD<T>) -> Vec<T> {
    a.iter().copied().collect()
}

fn dynamic_start(args: &[Arg], i: usize, op: &str) -> Result<i64> {
    let a = args
        .get(i)
        .ok_or_else(|| Error::runtime(format!("{op}: missing start")))?;
    match a {
        Arg::Literal(v) => crate::value::int_literal(*v, op, "start"),
        Arg::Stream(v) => {
            let s = v.as_i64(op)?;
            if s.len() != 1 || s.ndim() > 1 {
                return Err(Error::runtime(format!(
                    "{op}: dynamic start must be scalar/1-elem, got {:?}",
                    s.shape()
                )));
            }
            Ok(s.iter().next().copied().unwrap_or(0))
        }
    }
}

pub fn op_iwindow(args: &[Arg], _: &Config) -> Result<Vec<Value>> {
    let x = int_stream(args, 0, "IWINDOW", "X")?;
    if x.ndim() != 1 {
        return Err(Error::runtime(format!(
            "IWINDOW: expected 1D I, got shape {:?}",
            x.shape()
        )));
    }
    let start = dynamic_start(args, 1, "IWINDOW")?;
    let len = arg_i64(args, 2, "IWINDOW", "len")?;
    if start < 0 || len < 0 || start + len > x.len() as i64 {
        return Err(Error::runtime(format!(
            "IWINDOW: [{start},{}) out of range {}",
            start + len,
            x.len()
        )));
    }
    let s = start as usize;
    let l = len as usize;
    Ok(vec![Value::I64(
        x.slice(ndarray::s![s..s + l]).to_owned().into_dyn(),
    )])
}

pub fn op_igather1(args: &[Arg], _: &Config) -> Result<Vec<Value>> {
    let table = int_stream(args, 0, "IGATHER1", "table")?;
    let ids = int_stream(args, 1, "IGATHER1", "ids")?;
    if table.ndim() != 1 {
        return Err(Error::runtime(format!(
            "IGATHER1: table must be 1D I, got {:?}",
            table.shape()
        )));
    }
    if ids.ndim() != 1 {
        return Err(Error::runtime(format!(
            "IGATHER1: ids must be 1D I, got {:?}",
            ids.shape()
        )));
    }
    let mut out = Vec::with_capacity(ids.len());
    let tf = flat(&table);
    for &id in ids.iter() {
        if id < 0 || id as usize >= tf.len() {
            return Err(Error::runtime(format!(
                "IGATHER1: ids out of range [0,{})",
                tf.len()
            )));
        }
        out.push(tf[id as usize]);
    }
    Ok(vec![Value::I64(crate::value::array_from_vec(
        ids.shape(),
        out,
    )?)])
}

pub fn op_iassemble(args: &[Arg], _: &Config) -> Result<Vec<Value>> {
    let x = int_stream(args, 0, "IASSEMBLE", "X")?;
    let start = arg_i64(args, 1, "IASSEMBLE", "start")?;
    let len = arg_i64(args, 2, "IASSEMBLE", "len")?;
    if x.ndim() != 1 {
        return Err(Error::runtime(format!(
            "IASSEMBLE: expected 1D I, got {:?}",
            x.shape()
        )));
    }
    if !(1..=8).contains(&len) {
        return Err(Error::runtime(format!(
            "IASSEMBLE: len must be 1..8, got {len}"
        )));
    }
    if start < 0 || start + len > x.len() as i64 {
        return Err(Error::runtime(format!(
            "IASSEMBLE: [{start},{}) out of range {}",
            start + len,
            x.len()
        )));
    }
    let mut v: u128 = 0;
    let xf = flat(&x);
    for i in 0..len as usize {
        v |= ((xf[start as usize + i] as u8 as u128) & 0xFF) << (8 * i);
    }
    let out = if len == 8 && v >= (1u128 << 63) {
        (v - (1u128 << 64)) as i64
    } else {
        v as i64
    };
    Ok(vec![Value::I64(ArrayD::from_elem(
        ndarray::IxDyn(&[1]),
        out,
    ))])
}

pub fn op_isum(args: &[Arg], _: &Config) -> Result<Vec<Value>> {
    let x = int_stream(args, 0, "ISUM", "X")?;
    let start = arg_i64(args, 1, "ISUM", "start")?;
    let len = arg_i64(args, 2, "ISUM", "len")?;
    if x.ndim() != 1 {
        return Err(Error::runtime(format!(
            "ISUM: expected 1D I, got {:?}",
            x.shape()
        )));
    }
    if start < 0 || len < 0 || start + len > x.len() as i64 {
        return Err(Error::runtime(format!(
            "ISUM: [{start},{}) out of range {}",
            start + len,
            x.len()
        )));
    }
    let mut acc: i128 = 0;
    let xf = flat(&x);
    for &v in &xf[start as usize..(start + len) as usize] {
        acc = acc.wrapping_add(v as i128);
    }
    Ok(vec![Value::I64(ArrayD::from_elem(
        ndarray::IxDyn(&[1]),
        acc as i64,
    ))])
}

pub fn op_iscatter1(args: &[Arg], _: &Config) -> Result<Vec<Value>> {
    let base = int_stream(args, 0, "ISCATTER1", "OUT")?;
    let ids = int_stream(args, 1, "ISCATTER1", "IDS")?;
    let vals = int_stream(args, 2, "ISCATTER1", "VALS")?;
    if base.ndim() != 1 || ids.ndim() != 1 || vals.ndim() != 1 {
        return Err(Error::runtime("ISCATTER1: OUT/IDS/VALS must be 1D I"));
    }
    if ids.len() != vals.len() {
        return Err(Error::runtime(format!(
            "ISCATTER1: IDS {:?} vs VALS {:?}",
            ids.shape(),
            vals.shape()
        )));
    }
    let mut out = flat(&base);
    let idsf = flat(&ids);
    let valsf = flat(&vals);
    for (n, &id) in idsf.iter().enumerate() {
        if id < 0 || id as usize >= out.len() {
            return Err(Error::runtime(format!(
                "ISCATTER1: ids out of range [0,{})",
                out.len()
            )));
        }
        out[id as usize] = valsf[n];
    }
    Ok(vec![Value::I64(crate::value::array_from_vec(
        base.shape(),
        out,
    )?)])
}

pub fn op_icrc32(args: &[Arg], _: &Config) -> Result<Vec<Value>> {
    let x = int_stream(args, 0, "ICRC32", "X")?;
    if x.ndim() != 1 {
        return Err(Error::runtime(format!(
            "ICRC32: expected 1D I, got {:?}",
            x.shape()
        )));
    }
    let mut crc: u32 = 0xFFFF_FFFF;
    for &b in x.iter() {
        crc ^= (b as u8) as u32;
        for _ in 0..8 {
            crc = (crc >> 1) ^ if crc & 1 != 0 { 0xEDB8_8320 } else { 0 };
        }
    }
    Ok(vec![Value::I64(ArrayD::from_elem(
        ndarray::IxDyn(&[1]),
        (crc ^ 0xFFFF_FFFF) as i64,
    ))])
}

pub fn op_icrc32c(args: &[Arg], _: &Config) -> Result<Vec<Value>> {
    let x = int_stream(args, 0, "ICRC32C", "X")?;
    let seed = int_stream(args, 1, "ICRC32C", "SEED")?;
    if x.ndim() != 1 {
        return Err(Error::runtime(format!(
            "ICRC32C: expected 1D I, got {:?}",
            x.shape()
        )));
    }
    if seed.ndim() != 1 || seed.len() != 1 {
        return Err(Error::runtime("ICRC32C: SEED must be a 1-element I"));
    }
    let mut crc: u32 = seed.iter().next().copied().unwrap_or(0) as u32;
    for &b in x.iter() {
        crc ^= (b as u8) as u32;
        for _ in 0..8 {
            crc = (crc >> 1) ^ if crc & 1 != 0 { 0x82F6_3B78 } else { 0 };
        }
    }
    Ok(vec![Value::I64(ArrayD::from_elem(
        ndarray::IxDyn(&[1]),
        crc as i64,
    ))])
}

pub fn op_ifind(args: &[Arg], _: &Config) -> Result<Vec<Value>> {
    let x = int_stream(args, 0, "IFIND", "X")?;
    let pat = int_stream(args, 1, "IFIND", "PAT")?;
    if x.ndim() != 1 || pat.ndim() != 1 {
        return Err(Error::runtime("IFIND: X and PAT must be 1D I"));
    }
    if pat.is_empty() {
        return Err(Error::runtime("IFIND: empty pattern"));
    }
    let m = pat.len();
    let xf = flat(&x);
    let pf = flat(&pat);
    let mut found = -1i64;
    if xf.len() >= m {
        for i in 0..=xf.len() - m {
            if (0..m).all(|j| xf[i + j] == pf[j]) {
                found = i as i64;
                break;
            }
        }
    }
    Ok(vec![Value::I64(ArrayD::from_elem(
        ndarray::IxDyn(&[1]),
        found,
    ))])
}

pub fn op_ivarint(args: &[Arg], _: &Config) -> Result<Vec<Value>> {
    let x = int_stream(args, 0, "IVARINT", "X")?;
    let start = dynamic_start(args, 1, "IVARINT")?;
    if x.ndim() != 1 {
        return Err(Error::runtime(format!(
            "IVARINT: expected 1D I, got {:?}",
            x.shape()
        )));
    }
    if start < 0 || start as usize >= x.len() {
        return Err(Error::runtime(format!(
            "IVARINT: start {start} out of range {}",
            x.len()
        )));
    }
    let mut v: u128 = 0;
    let xf = flat(&x);
    for i in 0..8 {
        let idx = start as usize + i;
        if idx >= xf.len() {
            return Err(Error::runtime("IVARINT: truncated"));
        }
        let b = (xf[idx] as u8) as u128;
        v = (v << 7) | (b & 0x7F);
        if b & 0x80 == 0 {
            return Ok(vec![
                Value::I64(ArrayD::from_elem(ndarray::IxDyn(&[1]), v as i64)),
                Value::I64(ArrayD::from_elem(ndarray::IxDyn(&[1]), (i + 1) as i64)),
            ]);
        }
    }
    let idx = start as usize + 8;
    if idx >= xf.len() {
        return Err(Error::runtime("IVARINT: truncated"));
    }
    v = (v << 8) | ((xf[idx] as u8) as u128);
    Ok(vec![
        Value::I64(ArrayD::from_elem(ndarray::IxDyn(&[1]), v as i64)),
        Value::I64(ArrayD::from_elem(ndarray::IxDyn(&[1]), 9)),
    ])
}

// ---------------------------------------------------------------------------
// Raw DEFLATE (RFC 1951)
// ---------------------------------------------------------------------------

const LEN_BASE: [usize; 29] = [
    3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 23, 27, 31, 35, 43, 51, 59, 67, 83, 99, 115, 131,
    163, 195, 227, 258,
];
const LEN_EXTRA: [u32; 29] = [
    0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0,
];
const DIST_BASE: [usize; 30] = [
    1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33, 49, 65, 97, 129, 193, 257, 385, 513, 769, 1025, 1537,
    2049, 3073, 4097, 6145, 8193, 12289, 16385, 24577,
];
const DIST_EXTRA: [u32; 30] = [
    0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13,
    13,
];
const CL_ORDER: [usize; 19] = [
    16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15,
];

struct BitReader<'a> {
    d: &'a [u8],
    pos: usize,
    buf: u32,
    cnt: u32,
}

impl<'a> BitReader<'a> {
    fn new(d: &'a [u8]) -> Self {
        BitReader {
            d,
            pos: 0,
            buf: 0,
            cnt: 0,
        }
    }

    fn bit(&mut self) -> Result<u32> {
        if self.cnt == 0 {
            if self.pos >= self.d.len() {
                return Err(Error::runtime("IINFLATE: truncated stream"));
            }
            self.buf = self.d[self.pos] as u32;
            self.pos += 1;
            self.cnt = 8;
        }
        let b = self.buf & 1;
        self.buf >>= 1;
        self.cnt -= 1;
        Ok(b)
    }

    fn bits(&mut self, n: u32) -> Result<u32> {
        let mut v = 0u32;
        for i in 0..n {
            v |= self.bit()? << i;
        }
        Ok(v)
    }

    fn align(&mut self) {
        self.cnt = 0;
    }
}

fn huff_table(lengths: &[u8]) -> (HashMap<(u8, u32), usize>, u8) {
    let max_bits = lengths.iter().copied().max().unwrap_or(0);
    let mut count = vec![0u32; max_bits as usize + 1];
    for &ln in lengths {
        if ln != 0 {
            count[ln as usize] += 1;
        }
    }
    let mut code = 0u32;
    let mut next_code = vec![0u32; max_bits as usize + 1];
    for bits in 1..=max_bits as usize {
        code = (code + count[bits - 1]) << 1;
        next_code[bits] = code;
    }
    let mut table = HashMap::new();
    for (sym, &ln) in lengths.iter().enumerate() {
        if ln != 0 {
            table.insert((ln, next_code[ln as usize]), sym);
            next_code[ln as usize] += 1;
        }
    }
    (table, max_bits)
}

fn huff_decode(
    br: &mut BitReader,
    table: &HashMap<(u8, u32), usize>,
    max_bits: u8,
) -> Result<usize> {
    let mut code = 0u32;
    for ln in 1..=max_bits {
        code = (code << 1) | br.bit()?;
        if let Some(&sym) = table.get(&(ln, code)) {
            return Ok(sym);
        }
    }
    Err(Error::runtime("IINFLATE: bad Huffman code"))
}

fn decode_block(
    br: &mut BitReader,
    out: &mut Vec<u8>,
    lit: &HashMap<(u8, u32), usize>,
    lmax: u8,
    dist: &HashMap<(u8, u32), usize>,
    dmax: u8,
    limit: usize,
) -> Result<()> {
    loop {
        let sym = huff_decode(br, lit, lmax)?;
        if sym < 256 {
            out.push(sym as u8);
        } else if sym == 256 {
            return Ok(());
        } else {
            let li = sym - 257;
            if li >= 29 {
                return Err(Error::runtime("IINFLATE: bad length symbol"));
            }
            let length = LEN_BASE[li] + br.bits(LEN_EXTRA[li])? as usize;
            let ds = huff_decode(br, dist, dmax)?;
            if ds >= 30 {
                return Err(Error::runtime("IINFLATE: bad distance symbol"));
            }
            let d = DIST_BASE[ds] + br.bits(DIST_EXTRA[ds])? as usize;
            if d > out.len() {
                return Err(Error::runtime("IINFLATE: distance beyond the output"));
            }
            for _ in 0..length {
                let b = out[out.len() - d];
                out.push(b);
            }
        }
        if out.len() > limit {
            return Err(Error::runtime(format!("IINFLATE: output exceeds {limit}")));
        }
    }
}

fn inflate_raw(data: &[u8], expected: usize) -> Result<Vec<u8>> {
    let mut br = BitReader::new(data);
    let mut out = Vec::new();
    loop {
        let final_block = br.bit()?;
        let btype = br.bits(2)?;
        match btype {
            0 => {
                br.align();
                let ln = br.bits(16)?;
                let nln = br.bits(16)?;
                if (ln ^ 0xFFFF) != nln {
                    return Err(Error::runtime("IINFLATE: stored LEN/NLEN mismatch"));
                }
                for _ in 0..ln {
                    out.push(br.bits(8)? as u8);
                }
            }
            1 => {
                let mut litlens = vec![8u8; 144];
                litlens.extend(std::iter::repeat_n(9u8, 112));
                litlens.extend(std::iter::repeat_n(7u8, 24));
                litlens.extend(std::iter::repeat_n(8u8, 8));
                let (lit, lmax) = huff_table(&litlens);
                let (dist, dmax) = huff_table(&[5u8; 32]);
                decode_block(&mut br, &mut out, &lit, lmax, &dist, dmax, expected)?;
            }
            2 => {
                let hlit = br.bits(5)? as usize + 257;
                let hdist = br.bits(5)? as usize + 1;
                let hclen = br.bits(4)? as usize + 4;
                let mut cl = vec![0u8; 19];
                for i in 0..hclen {
                    cl[CL_ORDER[i]] = br.bits(3)? as u8;
                }
                let (clt, clmax) = huff_table(&cl);
                let mut lengths = Vec::new();
                while lengths.len() < hlit + hdist {
                    let sym = huff_decode(&mut br, &clt, clmax)?;
                    if sym < 16 {
                        lengths.push(sym as u8);
                    } else if sym == 16 {
                        if lengths.is_empty() {
                            return Err(Error::runtime("IINFLATE: repeat with no previous"));
                        }
                        let prev = *lengths.last().unwrap();
                        let n = (br.bits(2)? + 3) as usize;
                        lengths.resize(lengths.len() + n, prev);
                    } else if sym == 17 {
                        let n = (br.bits(3)? + 3) as usize;
                        lengths.resize(lengths.len() + n, 0);
                    } else if sym == 18 {
                        let n = (br.bits(7)? + 11) as usize;
                        lengths.resize(lengths.len() + n, 0);
                    } else {
                        return Err(Error::runtime("IINFLATE: bad code-length symbol"));
                    }
                }
                let (lit, lmax) = huff_table(&lengths[..hlit]);
                let (dist, dmax) = huff_table(&lengths[hlit..]);
                decode_block(&mut br, &mut out, &lit, lmax, &dist, dmax, expected)?;
            }
            _ => return Err(Error::runtime("IINFLATE: bad block type")),
        }
        if out.len() > expected {
            return Err(Error::runtime(format!(
                "IINFLATE: output exceeds {expected}"
            )));
        }
        if final_block != 0 {
            break;
        }
    }
    if out.len() != expected {
        return Err(Error::runtime(format!(
            "IINFLATE: decoded {}, expected {expected}",
            out.len()
        )));
    }
    Ok(out)
}

pub fn op_iinflate(args: &[Arg], _: &Config) -> Result<Vec<Value>> {
    let x = int_stream(args, 0, "IINFLATE", "X")?;
    let n = arg_i64(args, 1, "IINFLATE", "size")?;
    if x.ndim() != 1 {
        return Err(Error::runtime(format!(
            "IINFLATE: expected 1D I, got {:?}",
            x.shape()
        )));
    }
    if n < 0 {
        return Err(Error::runtime(format!("IINFLATE: negative size {n}")));
    }
    let bytes: Vec<u8> = x.iter().map(|&v| v as u8).collect();
    let raw = inflate_raw(&bytes, n as usize)?;
    Ok(vec![Value::I64(crate::value::array_from_vec(
        &[raw.len()],
        raw.into_iter().map(|b| b as i64).collect(),
    )?)])
}

// Keeps the shape helper in use for future 2D byte ops.
#[allow(dead_code)]
fn _shape(args: &[Arg], i: usize) -> Option<Vec<usize>> {
    shape_of(args, i)
}
