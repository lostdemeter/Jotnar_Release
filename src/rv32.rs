//! RV32IM machine and DOOM target.
//!
//! Port of the native/golden RV32 core used by the DOOM demo. The ISA is
//! RV32I + M with the small CSR/Sv32 surface the freestanding runtime
//! exercises; MMIO is the DOOM layout:
//!
//! ```text
//! 0x01FFF000 WAD length
//! 0x02000000 WAD
//! 0x03000000 320x200 RGBA framebuffer
//! 0x03FF0004 clock (cycles / 1000)
//! 0x03FF0008 frame counter
//! 0x03FF0100 key ring
//! 0x10000000 16550 window
//! 0x11004000 CLINT mtimecmp window
//! ```

use std::path::Path;

use crate::error::{Error, Result};

pub const FB_ADDR: u32 = 0x0300_0000;
pub const FB_W: usize = 320;
pub const FB_H: usize = 200;
pub const FB_BYTES: usize = FB_W * FB_H * 4;
pub const FRAME_ADDR: u32 = 0x03FF_0008;
pub const CLOCK_ADDR: u32 = 0x03FF_0004;
pub const WAD_ADDR: u32 = 0x0200_0000;
pub const WAD_LEN_ADDR: u32 = 0x01FF_F000;

const M32: u32 = 0xFFFF_FFFF;

fn u32v(v: u64) -> u32 {
    v as u32
}

fn s32(v: u32) -> i32 {
    v as i32
}

fn sext(v: u32, bits: u32) -> i32 {
    let m = 1u32 << (bits - 1);
    ((v ^ m).wrapping_sub(m)) as i32
}

#[derive(Clone, Debug)]
pub struct Checkpoint {
    pub mem: Vec<u8>,
    pub x: [u32; 32],
    pub pc: u32,
    pub cycles: u64,
    pub csr: Vec<u32>,
    pub priv_: u8,
}

pub struct Rv32 {
    pub mem: Vec<u8>,
    pub x: [u32; 32],
    pub pc: u32,
    pub cycles: u64,
    pub halted: bool,
    pub trap: Option<String>,
    csr: Vec<u32>,
    priv_: u8,
    reservation: Option<u32>,
    mem_size: usize,
}

impl Rv32 {
    pub fn new(mem_size: usize) -> Self {
        Rv32 {
            mem: vec![0; mem_size],
            x: [0; 32],
            pc: 0,
            cycles: 0,
            halted: false,
            trap: None,
            csr: vec![0; 4096],
            priv_: 3,
            reservation: None,
            mem_size,
        }
    }

    pub fn reset(&mut self) {
        self.mem.fill(0);
        self.x = [0; 32];
        self.pc = 0;
        self.cycles = 0;
        self.halted = false;
        self.trap = None;
        self.csr.fill(0);
        self.priv_ = 3;
        self.reservation = None;
    }

    pub fn load_doom(&mut self, image: &[u8], wad: &[u8]) -> Result<()> {
        if image.len() > self.mem_size {
            return Err(Error::runtime("DOOM image exceeds memory"));
        }
        let wad_end = WAD_ADDR as usize + wad.len();
        if wad_end > self.mem_size || WAD_LEN_ADDR as usize + 4 > self.mem_size {
            return Err(Error::runtime("DOOM WAD exceeds memory"));
        }
        self.mem[..image.len()].copy_from_slice(image);
        self.mem[WAD_ADDR as usize..wad_end].copy_from_slice(wad);
        self.write32(WAD_LEN_ADDR, wad.len() as u32);
        Ok(())
    }

    pub fn read8(&self, addr: u32) -> u8 {
        self.mem.get(addr as usize).copied().unwrap_or(0)
    }

    pub fn read16(&self, addr: u32) -> u16 {
        u16::from_le_bytes([self.read8(addr), self.read8(addr.wrapping_add(1))])
    }

    pub fn read32(&self, addr: u32) -> u32 {
        u32::from_le_bytes([
            self.read8(addr),
            self.read8(addr.wrapping_add(1)),
            self.read8(addr.wrapping_add(2)),
            self.read8(addr.wrapping_add(3)),
        ])
    }

    pub fn write8(&mut self, addr: u32, v: u8) {
        if let Some(b) = self.mem.get_mut(addr as usize) {
            *b = v;
        }
    }

    pub fn write16(&mut self, addr: u32, v: u16) {
        self.write8(addr, v as u8);
        self.write8(addr.wrapping_add(1), (v >> 8) as u8);
    }

    pub fn write32(&mut self, addr: u32, v: u32) {
        self.write8(addr, v as u8);
        self.write8(addr.wrapping_add(1), (v >> 8) as u8);
        self.write8(addr.wrapping_add(2), (v >> 16) as u8);
        self.write8(addr.wrapping_add(3), (v >> 24) as u8);
    }

    pub fn frame(&self) -> u32 {
        self.read32(FRAME_ADDR)
    }

    /// Wall-clock milliseconds for the guest (interactive player).
    pub fn set_ms(&mut self, ms: u32) {
        self.write32(CLOCK_ADDR, ms);
    }

    /// Push one key event onto the DOOM key ring.
    pub fn key(&mut self, pressed: bool, key: u8) {
        let ring = 0x03FF_0100u32;
        let tail = self.read32(ring + 4);
        self.write32(
            ring + 8 + ((tail & 63) * 4),
            ((pressed as u32) << 8) | key as u32,
        );
        self.write32(ring + 4, tail.wrapping_add(1));
    }

    /// Run until the frame counter advances by one (or a 20M guard).
    pub fn run_one_frame(&mut self) {
        let target = self.frame().wrapping_add(1);
        let mut guard = 0u64;
        while !self.halted && self.frame() < target && guard < 20_000_000 {
            self.step();
            guard += 1;
        }
    }

    pub fn framebuffer(&self) -> &[u8] {
        &self.mem[FB_ADDR as usize..FB_ADDR as usize + FB_BYTES]
    }

    fn csr_get(&self, a: usize) -> u32 {
        if a == 0xC00 || a == 0xC01 {
            return self.cycles as u32;
        }
        if a == 0xF14 {
            return 0;
        }
        if a == 0x144 {
            return self.csr[0x344] & 0x222;
        }
        self.csr.get(a).copied().unwrap_or(0)
    }

    fn csr_set(&mut self, a: usize, v: u32) {
        if a == 0xC00 || a == 0xC01 || a == 0xF14 {
            return;
        }
        if a == 0x144 {
            self.csr[0x344] = (self.csr[0x344] & !0x222) | (v & 0x222);
            return;
        }
        if let Some(slot) = self.csr.get_mut(a) {
            *slot = v;
        }
    }

    fn exception(&mut self, cause: u32, tval: u32) {
        let pc = self.pc;
        let mut delegated = false;
        if self.priv_ < 3 {
            if cause & 0x8000_0000 != 0 {
                delegated = ((self.csr[0x303] >> (cause & 0x1F)) & 1) != 0;
            } else {
                delegated = ((self.csr[0x302] >> cause) & 1) != 0;
            }
        }
        if delegated {
            self.csr[0x141] = pc;
            self.csr[0x142] = cause;
            self.csr[0x143] = tval;
            let ss = self.csr[0x100];
            self.csr[0x100] =
                (ss & !2 & !(1 << 8)) | ((ss & 2) << 4) | (((self.priv_ as u32) & 1) << 8);
            self.priv_ = 1;
            self.pc = self.csr[0x105] & !3;
        } else {
            self.csr[0x341] = pc;
            self.csr[0x342] = cause;
            self.csr[0x343] = tval;
            let ms = self.csr[0x300];
            self.csr[0x300] =
                (ms & !8 & !(3 << 11)) | ((ms & 8) << 4) | ((self.priv_ as u32) << 11);
            self.priv_ = 3;
            self.pc = self.csr[0x305] & !3;
        }
    }

    fn timer_pending(&self) -> bool {
        if self.csr[0x305] == 0 {
            return false;
        }
        if (self.csr[0x300] & 8) == 0 || (self.csr[0x304] & 0x80) == 0 {
            return false;
        }
        let cmp = self.read32(0x03FF_0020) as u64 | ((self.read32(0x03FF_0024) as u64) << 32);
        self.cycles >= cmp
    }

    fn s_timer_pending(&self) -> bool {
        if self.priv_ >= 3 {
            return false;
        }
        (self.csr[0x303] & 0x20) != 0
            && (self.csr[0x104] & 0x20) != 0
            && (self.csr[0x100] & 0x2) != 0
            && (self.csr[0x344] & 0x20) != 0
    }

    fn translate(&mut self, va: u32, access: u8) -> Option<u32> {
        let satp = self.csr[0x180];
        if self.priv_ == 3 || (satp & 0x8000_0000) == 0 {
            return Some(va);
        }
        let mut ppn = satp & 0x3F_FFFF;
        let cause = if access == 2 {
            12
        } else if access == 0 {
            13
        } else {
            15
        };
        for level in (0..=1).rev() {
            let pte_addr =
                (ppn << 12).wrapping_add((((va >> (12 + 10 * level)) & 0x3FF) << 2) as u32);
            if pte_addr as usize + 4 > self.mem_size {
                self.exception(cause, va);
                return None;
            }
            let pte = self.read32(pte_addr);
            if pte & 1 == 0 {
                self.exception(cause, va);
                return None;
            }
            if pte & 0xE == 0 {
                ppn = (pte >> 10) & 0x3F_FFFF;
                continue;
            }
            if access == 1 && pte & 4 == 0 {
                self.exception(cause, va);
                return None;
            }
            if access == 0 && pte & 2 == 0 {
                self.exception(cause, va);
                return None;
            }
            if access == 2 && pte & 8 == 0 {
                self.exception(cause, va);
                return None;
            }
            if self.priv_ == 0 && pte & 0x10 == 0 {
                self.exception(cause, va);
                return None;
            }
            if level == 1 && ((pte >> 10) & 0x3FF) != 0 {
                self.exception(cause, va);
                return None;
            }
            let pa = ((pte >> 10) << 12) | (va & if level == 1 { 0x3F_FFFF } else { 0xFFF });
            self.write32(pte_addr, pte | 0x40 | if access == 1 { 0x80 } else { 0 });
            return Some(pa);
        }
        self.exception(cause, va);
        None
    }

    fn mmio_load(&mut self, addr: u32, f3: u32, rd: usize) -> bool {
        if addr == 0x1100_BFF8 {
            self.x[rd] = self.cycles as u32;
            return true;
        }
        if addr == 0x1100_4000 {
            self.x[rd] = self.read32(0x03FF_0020);
            return true;
        }
        if addr == 0x1100_4004 {
            self.x[rd] = self.read32(0x03FF_0024);
            return true;
        }
        if addr & !0xF == 0x1000_0000 {
            let off = addr & 0xF;
            self.x[rd] = if off == 5 {
                0x60
            } else if off == 2 {
                1
            } else {
                0
            };
            return true;
        }
        let _ = f3;
        false
    }

    fn mmio_store(&mut self, addr: u32, b: u32) -> bool {
        if addr == 0x1100_4000 {
            self.write32(0x03FF_0020, b);
            return true;
        }
        if addr == 0x1100_4004 {
            self.write32(0x03FF_0024, b);
            return true;
        }
        if addr & !0xF == 0x1000_0000 {
            if addr & 0xF == 0 {
                let ring = 0x03FF_0300u32;
                let tail = self.read32(ring + 4);
                self.write8(ring + 8 + (tail & 0x3FFF), b as u8);
                self.write32(ring + 4, tail.wrapping_add(1));
            }
            return true;
        }
        false
    }

    fn trap(&mut self, msg: String) {
        self.trap = Some(msg);
        self.halted = true;
    }

    pub fn step(&mut self) {
        if self.halted {
            return;
        }
        if self.timer_pending() {
            self.exception(0x8000_0007, 0);
            self.cycles += 1;
            return;
        }
        if self.s_timer_pending() {
            self.exception(0x8000_0005, 0);
            self.cycles += 1;
            return;
        }
        self.cycles += 1;
        let pc = self.pc;
        let Some(pa) = self.translate(pc, 2) else {
            return;
        };
        if pa as usize + 4 > self.mem_size {
            self.exception(1, pa);
            return;
        }
        let insn = self.read32(pa);
        if insn == 0x0010_0073 {
            if self.csr[0x305] != 0 {
                self.exception(3, 0);
                return;
            }
            self.pc = pc.wrapping_add(4);
            self.halted = true;
            return;
        }
        let op = insn & 0x7F;
        let rd = ((insn >> 7) & 31) as usize;
        let f3 = (insn >> 12) & 7;
        let rs1 = ((insn >> 15) & 31) as usize;
        let rs2 = ((insn >> 20) & 31) as usize;
        let a = self.x[rs1];
        let b = self.x[rs2];
        let mut nxt = pc.wrapping_add(4);

        match op {
            0x37 => self.x[rd] = insn & 0xFFFF_F000,
            0x17 => self.x[rd] = pc.wrapping_add(insn & 0xFFFF_F000),
            0x6F => {
                let imm = sext(
                    ((insn >> 31) << 20)
                        | (((insn >> 12) & 0xFF) << 12)
                        | (((insn >> 20) & 1) << 11)
                        | (((insn >> 21) & 0x3FF) << 1),
                    21,
                );
                self.x[rd] = nxt;
                nxt = pc.wrapping_add(imm as u32);
            }
            0x67 => {
                let imm = sext(insn >> 20, 12);
                let target = a.wrapping_add(imm as u32) & !1;
                self.x[rd] = nxt;
                nxt = target;
            }
            0x63 => {
                let imm = sext(
                    ((insn >> 31) << 12)
                        | (((insn >> 7) & 1) << 11)
                        | (((insn >> 25) & 0x3F) << 5)
                        | (((insn >> 8) & 0xF) << 1),
                    13,
                );
                let take = match f3 {
                    0 => a == b,
                    1 => a != b,
                    4 => s32(a) < s32(b),
                    5 => s32(a) >= s32(b),
                    6 => a < b,
                    7 => a >= b,
                    _ => {
                        self.trap(format!("bad branch f3={f3} at 0x{pc:08x}"));
                        false
                    }
                };
                if take {
                    nxt = pc.wrapping_add(imm as u32);
                }
            }
            0x03 => {
                let imm = sext(insn >> 20, 12) as u32;
                let Some(addr) = self.translate(a.wrapping_add(imm), 0) else {
                    return;
                };
                if self.mmio_load(addr, f3, rd) {
                    // handled
                } else if addr as usize + 4 > self.mem_size {
                    self.exception(5, addr);
                    return;
                } else {
                    match f3 {
                        0 => self.x[rd] = (self.read8(addr) as i8 as i32) as u32,
                        1 => self.x[rd] = (self.read16(addr) as i16 as i32) as u32,
                        2 => self.x[rd] = self.read32(addr),
                        4 => self.x[rd] = self.read8(addr) as u32,
                        5 => self.x[rd] = self.read16(addr) as u32,
                        _ => {
                            self.trap(format!("bad load f3={f3} at 0x{pc:08x}"));
                            return;
                        }
                    }
                }
            }
            0x23 => {
                let imm = sext(((insn >> 25) << 5) | ((insn >> 7) & 0x1F), 12) as u32;
                let Some(addr) = self.translate(a.wrapping_add(imm), 1) else {
                    return;
                };
                if self.mmio_store(addr, b) {
                    // handled
                } else if addr as usize + 4 > self.mem_size {
                    self.exception(7, addr);
                    return;
                } else {
                    match f3 {
                        0 => self.write8(addr, b as u8),
                        1 => self.write16(addr, b as u16),
                        2 => self.write32(addr, b),
                        _ => {
                            self.trap(format!("bad store f3={f3} at 0x{pc:08x}"));
                            return;
                        }
                    }
                }
            }
            0x13 => {
                let imm = sext(insn >> 20, 12);
                let sh = (insn >> 20) & 31;
                self.x[rd] = match f3 {
                    0 => a.wrapping_add(imm as u32),
                    2 => (s32(a) < imm) as u32,
                    3 => (a < imm as u32) as u32,
                    4 => a ^ imm as u32,
                    6 => a | imm as u32,
                    7 => a & imm as u32,
                    1 => a << sh,
                    5 => {
                        if insn & 0x4000_0000 != 0 {
                            (s32(a) >> sh) as u32
                        } else {
                            a >> sh
                        }
                    }
                    _ => {
                        self.trap(format!("bad op-imm f3={f3} at 0x{pc:08x}"));
                        return;
                    }
                };
            }
            0x33 => {
                let sh = b & 31;
                let f7 = insn & 0xFE00_0000;
                if f7 == 0 {
                    self.x[rd] = match f3 {
                        0 => a.wrapping_add(b),
                        1 => a << sh,
                        2 => (s32(a) < s32(b)) as u32,
                        3 => (a < b) as u32,
                        4 => a ^ b,
                        5 => a >> sh,
                        6 => a | b,
                        7 => a & b,
                        _ => {
                            self.trap(format!("bad op f3={f3} at 0x{pc:08x}"));
                            return;
                        }
                    };
                } else if f7 == 0x4000_0000 {
                    match f3 {
                        0 => self.x[rd] = a.wrapping_sub(b),
                        5 => self.x[rd] = (s32(a) >> sh) as u32,
                        _ => {
                            self.trap(format!("bad op f7=0x20 f3={f3}"));
                            return;
                        }
                    }
                } else if f7 == 0x0200_0000 {
                    self.x[rd] = self.muldiv(f3, a, b);
                } else {
                    self.trap(format!("bad op f7=0x{:02x} at 0x{pc:08x}", f7 >> 25));
                    return;
                }
            }
            0x2F => {
                let f5 = insn >> 27;
                let Some(a_addr) = self.translate(a, if f5 == 2 { 0 } else { 1 }) else {
                    return;
                };
                match f5 {
                    2 => {
                        self.x[rd] = self.read32(a_addr);
                        self.reservation = Some(a_addr);
                    }
                    3 => {
                        if self.reservation == Some(a_addr) {
                            self.write32(a_addr, b);
                            self.x[rd] = 0;
                        } else {
                            self.x[rd] = 1;
                        }
                        self.reservation = None;
                    }
                    _ => {
                        let old = self.read32(a_addr);
                        let nv = match f5 {
                            1 => b,
                            0 => old.wrapping_add(b),
                            0x0C => old & b,
                            8 => old | b,
                            4 => old ^ b,
                            0x10 => {
                                if s32(old) < s32(b) {
                                    old
                                } else {
                                    b
                                }
                            }
                            0x14 => {
                                if s32(old) > s32(b) {
                                    old
                                } else {
                                    b
                                }
                            }
                            0x18 => {
                                if old < b {
                                    old
                                } else {
                                    b
                                }
                            }
                            0x1C => {
                                if old > b {
                                    old
                                } else {
                                    b
                                }
                            }
                            _ => {
                                self.exception(2, insn);
                                return;
                            }
                        };
                        self.write32(a_addr, nv);
                        self.x[rd] = old;
                    }
                }
            }
            0x0F => {}
            0x73 => {
                if insn == 0x3020_0073 {
                    let ms = self.csr[0x300];
                    let mie = (ms >> 7) & 1;
                    let mpp = (ms >> 11) & 3;
                    self.csr[0x300] = (ms & !8) | (mie << 3) | (1 << 7) | (ms & !(3 << 11));
                    self.priv_ = mpp as u8;
                    self.pc = self.csr[0x341] & !3;
                    return;
                }
                if insn == 0x1020_0073 {
                    let ss = self.csr[0x100];
                    let spie = (ss >> 5) & 1;
                    let spp = (ss >> 8) & 1;
                    self.csr[0x100] = (ss & !2) | (spie << 1) | (1 << 5) | (ss & !(1 << 8));
                    self.priv_ = if spp != 0 { 1 } else { 0 };
                    self.pc = self.csr[0x141] & !3;
                    return;
                }
                if insn & 0xFE00_7FFF == 0x1200_0073 {
                    // sfence.vma
                } else if insn == 0x1050_0073 {
                    // wfi
                } else if insn == 0x0000_0073 {
                    let ec = if self.priv_ == 3 {
                        11
                    } else if self.priv_ == 1 {
                        9
                    } else {
                        8
                    };
                    if self.csr[0x305] != 0 {
                        self.exception(ec, 0);
                        return;
                    }
                    self.pc = nxt;
                    self.halted = true;
                    return;
                } else if matches!(f3, 1 | 2 | 3 | 5 | 6 | 7) {
                    let c = (insn >> 20) as usize;
                    let old = self.csr_get(c);
                    let src = if f3 >= 5 { rs1 as u32 } else { self.x[rs1] };
                    let nv = if f3 == 1 || f3 == 5 {
                        src
                    } else if f3 == 2 || f3 == 6 {
                        old | src
                    } else {
                        old & !src
                    };
                    self.csr_set(c, nv);
                    self.x[rd] = old;
                } else {
                    if self.csr[0x305] != 0 {
                        self.exception(2, insn);
                        return;
                    }
                    self.pc = nxt;
                    self.halted = true;
                    return;
                }
            }
            _ => {
                if self.csr[0x305] != 0 {
                    self.exception(2, insn);
                    return;
                }
                self.pc = nxt;
                self.halted = true;
                return;
            }
        }
        self.x[0] = 0;
        self.pc = nxt;
    }

    fn muldiv(&mut self, f3: u32, a: u32, b: u32) -> u32 {
        let sa = s32(a);
        let sb = s32(b);
        match f3 {
            0 => a.wrapping_mul(b),
            1 => ((sa as i64 * sb as i64) >> 32) as u32,
            2 => ((sa as i64 * b as i64) >> 32) as u32,
            3 => (((a as u64) * (b as u64)) >> 32) as u32,
            4 => {
                if sb == 0 {
                    M32
                } else if sa == i32::MIN && sb == -1 {
                    a
                } else {
                    (sa / sb) as u32
                }
            }
            5 => {
                if b == 0 {
                    M32
                } else {
                    a / b
                }
            }
            6 => {
                if sb == 0 {
                    a
                } else if sa == i32::MIN && sb == -1 {
                    0
                } else {
                    (sa % sb) as u32
                }
            }
            7 => {
                if b == 0 {
                    a
                } else {
                    a % b
                }
            }
            _ => {
                self.trap(format!("bad muldiv f3={f3}"));
                0
            }
        }
    }

    /// Run until the frame counter reaches `target` or `max_instr` steps.
    pub fn run_to_frame(&mut self, target: u32, max_instr: u64) -> u64 {
        let mut n = 0;
        while !self.halted && n < max_instr {
            self.step();
            n += 1;
            let t = (self.cycles / 1000) as u32;
            self.write32(CLOCK_ADDR, t);
            if self.frame() >= target {
                break;
            }
        }
        n
    }

    pub fn checkpoint(&self) -> Checkpoint {
        Checkpoint {
            mem: self.mem.clone(),
            x: self.x,
            pc: self.pc,
            cycles: self.cycles,
            csr: self.csr.clone(),
            priv_: self.priv_,
        }
    }

    pub fn restore(&mut self, cp: &Checkpoint) -> Result<()> {
        if cp.mem.len() != self.mem_size {
            return Err(Error::runtime(format!(
                "checkpoint memory {} != {}",
                cp.mem.len(),
                self.mem_size
            )));
        }
        self.mem.copy_from_slice(&cp.mem);
        self.x = cp.x;
        self.pc = cp.pc;
        self.cycles = cp.cycles;
        self.csr.copy_from_slice(&cp.csr);
        self.priv_ = cp.priv_;
        self.halted = false;
        self.trap = None;
        self.reservation = None;
        Ok(())
    }

    pub fn save_checkpoint(&self, prefix: &Path) -> Result<()> {
        std::fs::write(prefix.with_extension("mem"), &self.mem)?;
        let mut xb = Vec::with_capacity(32 * 8);
        for v in self.x {
            xb.extend_from_slice(&(v as i64).to_le_bytes());
        }
        std::fs::write(prefix.with_extension("x"), xb)?;
        std::fs::write(prefix.with_extension("pc"), (self.pc as i64).to_le_bytes())?;
        std::fs::write(
            prefix.with_extension("cyc"),
            (self.cycles as i64).to_le_bytes(),
        )?;
        let mut cb = Vec::with_capacity(self.csr.len() * 4);
        for v in &self.csr {
            cb.extend_from_slice(&v.to_le_bytes());
        }
        std::fs::write(prefix.with_extension("csr"), cb)?;
        Ok(())
    }

    pub fn load_checkpoint(&mut self, prefix: &Path) -> Result<()> {
        let mem = std::fs::read(prefix.with_extension("mem"))?;
        if mem.len() != self.mem_size {
            return Err(Error::runtime(format!(
                "checkpoint memory {} != {}",
                mem.len(),
                self.mem_size
            )));
        }
        self.mem.copy_from_slice(&mem);
        let xb = std::fs::read(prefix.with_extension("x"))?;
        if xb.len() != 32 * 8 {
            return Err(Error::runtime("bad checkpoint x"));
        }
        for i in 0..32 {
            self.x[i] = i64::from_le_bytes(xb[i * 8..i * 8 + 8].try_into().unwrap()) as u32;
        }
        let pb = std::fs::read(prefix.with_extension("pc"))?;
        self.pc = i64::from_le_bytes(pb[..8].try_into().unwrap()) as u32;
        let cb = std::fs::read(prefix.with_extension("cyc"))?;
        self.cycles = i64::from_le_bytes(cb[..8].try_into().unwrap()) as u64;
        if let Ok(cb) = std::fs::read(prefix.with_extension("csr")) {
            for (i, c) in cb.chunks_exact(4).enumerate().take(self.csr.len()) {
                self.csr[i] = u32::from_le_bytes(c[0..4].try_into().unwrap());
            }
        }
        self.halted = false;
        self.trap = None;
        self.reservation = None;
        Ok(())
    }
}

/// Convert the RGBA word framebuffer to an RGB PNG, matching the demo helper.
pub fn save_fb_png(fb: &[u8], path: &Path) -> Result<()> {
    if fb.len() != FB_BYTES {
        return Err(Error::runtime(format!(
            "framebuffer {} != {FB_BYTES}",
            fb.len()
        )));
    }
    let mut rgb = vec![0u8; FB_W * FB_H * 3];
    for i in 0..FB_W * FB_H {
        let w = u32::from_le_bytes([fb[i * 4], fb[i * 4 + 1], fb[i * 4 + 2], fb[i * 4 + 3]]);
        rgb[i * 3] = ((w >> 16) & 255) as u8;
        rgb[i * 3 + 1] = ((w >> 8) & 255) as u8;
        rgb[i * 3 + 2] = (w & 255) as u8;
    }
    image::save_buffer(path, &rgb, FB_W as u32, FB_H as u32, image::ColorType::Rgb8)
        .map_err(|e| Error::runtime(format!("save {}: {e}", path.display())))
}

pub fn read_fb_png(path: &Path) -> Result<Vec<u8>> {
    let im = image::open(path)
        .map_err(|e| Error::runtime(format!("open {}: {e}", path.display())))?
        .to_rgba8();
    if im.width() as usize != FB_W || im.height() as usize != FB_H {
        return Err(Error::runtime("framebuffer PNG has the wrong size"));
    }
    let mut out = vec![0u8; FB_BYTES];
    for (i, p) in im.pixels().enumerate() {
        let w = ((p[0] as u32) << 16) | ((p[1] as u32) << 8) | (p[2] as u32);
        out[i * 4..i * 4 + 4].copy_from_slice(&w.to_le_bytes());
    }
    Ok(out)
}

#[allow(dead_code)]
fn _m32(v: u64) -> u32 {
    u32v(v)
}
